r"""Dissipative port-Hamiltonian **SDE** on $SO(3)^n$ with GP subnetworks.

The learned counterpart of ``envs/arm_nlink_so3/arm_nlink_physics.py``: identical
port-Hamiltonian structure, but $M^{-1}$, $V$, $D$, $g$ and $\Sigma$ are replaced
by variational random-Fourier-feature GPs on the product manifold.

The system (§6, §10 of `multi-joint-ph-system.md`)
--------------------------------------------------
.. math::
    H(q,p) &= \tfrac12 p^\top M_\theta^{-1}(q)p + V_\theta(q),
    \qquad \xi = M_\theta^{-1}(q)\,p \\
    dq_i &= R_i[\xi_i]_\times\,dt \\
    dp   &= \Big(\underbrace{\hat P\xi}_{\rm ad^*}
              + \underbrace{\mathcal T(H)}_{\rm gravity+Coriolis}
              - D_\theta(q)\xi + g_\theta(q)u\Big)dt
            \;+\;\Sigma_\theta(q)\circ dW_t

Subnetworks
-----------
=============  ==========================  ==================  ================
subnet         parameterisation            quantity            params ($n=2$)
=============  ==========================  ==================  ================
`M_net`        `StructuredMass`            $M^{-1}(q)$         23
`Dw_net`       `StructuredDissipation`     $D(q)$              2
`g_net`        `StructuredInputMap`        $g(q)$              6
`Sigma_net`    `StructuredSigma` / GP      $\Sigma(q)$         6 / 6 336
`V_net`        `GP_NLink` $\to1$           $V(q)$              352
=============  ==========================  ==================  ================

$M^{-1}$, $D$ and $g$ are **physics-structured** (see
``utils/structured_subnets.py``): they learn the few physical parameters in the
closed forms of §4.1/§7/§8 and evaluate the exact algebra, rather than
approximating the functions with GPs. Measured motivation — a supervised fit of
the 23 mass parameters reaches $2\times10^{-5}$ relative error against $0.044$
for the best GP fit and $0.215$–$0.337$ for the GP as trained through the
dynamics; and `g_net` was spending 12 672 weights to relearn $T(q)^\top$, which
is implied by the state, at 0.465–0.565 relative error.

`V_net` is therefore the only learned *function* left in the drift. It stays a
GP because gravity picks out the world $e_z$, so $V$ is genuinely not
rotation-invariant and has no comparably compact closed form to exploit.

`Sigma_net` replaces the single pendulum's scalar `sigma_net`: with $n$ links the
lever geometry is unknown, so the whole diffusion matrix is learned. It stays
$3n\times\mathbf 3$ — **low rank on purpose**, because physically one wind field
drives every link. Note only $\Sigma\Sigma^\top$ is identifiable
($\Sigma\mapsto\Sigma O$ for orthogonal $O$ leaves the law unchanged); that is
harmless, since $\Sigma\Sigma^\top$ is exactly what enters the likelihood.

Structural prior (§17)
----------------------
$M(q)$ and $D_{\rm joint}(q)$ depend on $q$ only through the **relative**
rotations $R_j^\top R_k$ — verified numerically to $6.7\times10^{-16}$ in Phase 1.
The structured subnets get this exactly, by construction, since the closed forms
contain $R_j^\top R_k$ and $T(q)$ literally.

`relative_inputs` is therefore **inert** and kept only so that checkpoints and
CLI scripts predating the structured subnets still load. It used to feed
$\{\mathrm{vec}(R_i^\top R_{i+1})\}$ to the GP `M_net`/`Dw_net`/`g_net` as an
approximate version of the same prior; those subnets no longer exist here.

Public API (mirrors the single-pendulum model)
----------------------------------------------
    drift_p(q, p, u, keys)               -> $\dot p$              (3n,)
    stochastic_increment_p(q, dW, keys)  -> $\Sigma_\theta(q)dW$  (3n,)
    M_inv(q, keys)                       -> $M^{-1}(q)$           (3n,3n)
    Sigma(q, keys)                       -> $\Sigma_\theta(q)$    (3n,3)
    drift(q, omega, u, keys)             -> $\dot\omega$          (3n,)   [for PL]
    kl_loss()                            -> $\sum_i\mathrm{KL}$
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import equinox as eqx

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from utils.gp_model_nlink import GP_NLink, MatrixGP_NLink                # noqa: E402
from utils.lie_integrator_nlink import hat                               # noqa: E402
from utils.structured_subnets import (                                   # noqa: E402
    StructuredMass, StructuredDissipation, StructuredInputMap, StructuredSigma,
)

hat_batch = jax.vmap(hat)


class DissipativeArmHamSDE(eqx.Module):
    r"""Port-Hamiltonian SDE on $SO(3)^n\times\mathbb{R}^{3n}$ with GP subnets."""

    M_net:     StructuredMass
    V_net:     GP_NLink
    Dw_net:    StructuredDissipation
    g_net:     StructuredInputMap
    Sigma_net: MatrixGP_NLink

    log_sigma_R:     jnp.ndarray    # rollout-NLL noise on rotations (geodesic)
    log_sigma_omega: jnp.ndarray    # rollout-NLL noise on rates

    # Frozen at the dataset's `obs_noise_std`; feeds the PL term's noise floor.
    sigma_obs_omega: float = eqx.field(static=True)

    n:               int  = eqx.field(static=True)
    u_dim:           int  = eqx.field(static=True)
    friction:        bool = eqx.field(static=True)
    relative_inputs: bool = eqx.field(static=True)
    sigma_mode:      str  = eqx.field(static=True)   # 'full' | 'structured'
    # Change 5: stop_gradient on Sigma in the ROLLOUT path only. The rollout
    # NLL's gradient through Sigma carries nothing but collapse pressure
    # (model-side diffusion can only add variance to the residual against an
    # independently-noisy target), so with the scale gauge anchored it crushed
    # ||Sigma|| to ~30% of truth and zeroed the link-2 channel entirely. With
    # the path detached, the PL term solely owns the diffusion — which was the
    # stated design intent all along ("L_PL is what gives the diffusion a
    # non-collapsing data-fit signal"). Rollout SAMPLES still use the current
    # Sigma; only its gradient is cut.
    sigma_detach_rollout: bool = eqx.field(static=True)

    def __init__(self, *, key, n: int = 2, u_dim: int = None,
                 hidden_dim: int = 32, friction: bool = True,
                 relative_inputs: bool = False,
                 init_sigma_R: float = 0.1, init_sigma_omega: float = 0.1,
                 init_sigma_obs_omega: float = 0.1,
                 sigma_mode: str = 'full',
                 sigma_detach_rollout: bool = True,
                 variational_core: bool = False, core_prior_std: float = 1.0,
                 gp_core: bool = False, i_epsilon: float = None,
                 anchor_m1: float = None, anchor_trace: float = None,
                 dtype=jnp.float32):
        """`dtype` fixes the precision of **every** parameter explicitly.

        This must not be left to JAX's global default: the environment enables
        `jax_enable_x64` at import time, so an implicit default silently gives
        float64 weights against a float32 batch — which surfaces far away, as a
        primal/tangent dtype mismatch inside the `jax.jvp` in :meth:`drift`.
        Pin it here and the model is self-consistent regardless of the flag.
        """
        if sigma_mode not in ('full', 'structured'):
            raise ValueError("sigma_mode must be 'full' or 'structured', "
                             f"got {sigma_mode!r}")
        self.n = int(n)
        self.u_dim = int(3 * n if u_dim is None else u_dim)
        self.friction = bool(friction)
        # In gp_core mode the invariance is no longer free, so default it ON.
        self.relative_inputs = bool(relative_inputs or gp_core) and self.n > 1
        self.sigma_mode = sigma_mode
        self.sigma_detach_rollout = bool(sigma_detach_rollout)

        d = 3 * self.n
        self.log_sigma_R = jnp.log(jnp.asarray(init_sigma_R, dtype=dtype))
        self.log_sigma_omega = jnp.log(jnp.asarray(init_sigma_omega, dtype=dtype))
        self.sigma_obs_omega = float(init_sigma_obs_omega)

        kM, kV, kD, kg, kS = jax.random.split(key, 5)

        # $M^{-1}$, $D$ and $g$ are physics-structured: they learn the handful of
        # parameters in the closed forms rather than approximating the functions.
        # 23 + 2 + 6 = 31 numbers at n=2, against 27 456 GP weights previously.
        # `gp_core` makes the physical constants functions of q, emitted by the
        # same Matern x periodic GP used elsewhere; `relative_inputs` then
        # matters again, because it is what preserves M(hq) = M(q).
        vkw = dict(variational=variational_core, prior_std=core_prior_std,
                   gp_core=gp_core, hidden_dim=hidden_dim)
        ikw = dict(relative_inputs=self.relative_inputs)
        # `i_epsilon` is a *static* field, so it is taken from the skeleton at
        # deserialise time, not from the file. Runs trained before the floor
        # existed must be rebuilt with 0.0 or their M silently changes --
        # `load_run` passes that explicitly.
        mkw = {} if i_epsilon is None else dict(i_epsilon=i_epsilon)
        # Gauge anchors: `anchor_trace` normalises tr M(q) in function space
        # (the anchor that actually kills the scale gauge); `anchor_m1` is the
        # earlier, insufficient parameter-space pin, kept as harmless. None or
        # <=0 disables either; sentinel normalisation lives in StructuredMass.
        self.M_net = StructuredMass(kM, n=self.n, dtype=dtype,
                                    anchor_m1=anchor_m1,
                                    anchor_trace=anchor_trace,
                                    **vkw, **ikw, **mkw)
        self.Dw_net = StructuredDissipation(kD, n=self.n, dtype=dtype, **vkw, **ikw)
        self.g_net = StructuredInputMap(kg, n=self.n, dtype=dtype, **vkw, **ikw)

        # $V$ is the one genuinely learned *function* left in the drift.
        self.V_net = GP_NLink(kV, n_links=self.n, output_dim=1,
                              n_matern_features=hidden_dim, dtype=dtype)
        # Only the field's *type* changes between modes; the name and position
        # are fixed, so a 'full'-mode checkpoint still round-trips unchanged.
        self.Sigma_net = (
            StructuredSigma(kS, n=self.n, dtype=dtype, **vkw)
            if sigma_mode == 'structured' else
            MatrixGP_NLink(kS, n_links=self.n, shape=(d, 3),
                           n_matern_features=hidden_dim, dtype=dtype))

    # ── Input featurisation ──────────────────────────────────────────

    def _invariant_q(self, q):
        r"""Input for the rotation-invariant subnets.

        With `relative_inputs`, returns
        $(\mathrm{vec}(R_1^\top R_2),\dots,\mathrm{vec}(R_{n-1}^\top R_n))
        \in\mathbb{R}^{9(n-1)}$; otherwise the raw $q$ unchanged.
        """
        if not self.relative_inputs:
            return q
        R = q.reshape(self.n, 3, 3)
        rel = jnp.einsum('iab,iac->ibc', R[:-1], R[1:])       # $R_i^\top R_{i+1}$
        return rel.reshape(9 * (self.n - 1))

    # ── Subnet wrappers ──────────────────────────────────────────────
    # `key=None` -> deterministic posterior-mean call; `key=<PRNGKey>` ->
    # reparameterised weight sample. Holding one key per subnet fixed across a
    # whole rollout is what makes the ELBO's MC estimate over q_psi(w) coherent.

    def _call(self, net, x, key):
        return net(x, key=key, inference_mode=(key is None))

    @staticmethod
    def _k(keys, name):
        return None if keys is None else keys.get(name)

    # ── Structure matrices ───────────────────────────────────────────

    def M_inv(self, q, keys=None):
        r"""$M^{-1}(q)\in\mathbb{R}^{3n\times3n}$, PSD with a unit diagonal floor."""
        return self._call(self.M_net, q, self._k(keys, 'M'))

    def Sigma(self, q, keys=None):
        r"""$\Sigma_\theta(q)\in\mathbb{R}^{3n\times3}$ — the wind lever map."""
        return self._call(self.Sigma_net, q, self._k(keys, 'Sigma'))

    def stochastic_increment_p(self, q, dW, keys=None):
        r"""$\Sigma_\theta(q)\,dW$, with `dW : (3,)` already scaled by $\sqrt h$.

        Used only by the rollout integrator (the PL term calls :meth:`Sigma`
        directly), so the detach below removes exactly one gradient path: the
        rollout NLL's pure $\Sigma\to0$ collapse pressure. Values are
        unchanged — rollouts still diffuse with the current $\Sigma_\theta$.
        """
        Sig = self.Sigma(q, keys=keys)
        if self.sigma_detach_rollout:
            Sig = jax.lax.stop_gradient(Sig)
        return Sig @ dW

    # ── Drift ────────────────────────────────────────────────────────

    def drift_p(self, q, p, u, keys=None):
        r"""Deterministic momentum drift $\dot p\in\mathbb{R}^{3n}$.

        .. math::
            \dot p_i = p_i\times\xi_i
                     + \sum_{k=1}^{3} (R_i)_{k,:}\times\frac{\partial H}{\partial (R_i)_{k,:}}
                     - (D_\theta\xi)_i + (g_\theta u)_i

        The second term is the trivialized gradient $\mathcal T_i(H)$; because it
        is a directional derivative *along the group action*, it is independent
        of how $H$ is extended off $SO(3)^n$ — which is what makes the redundant
        $\mathbb{R}^{9n}$ representation of $q$ safe.

        With $(q,p)$ as the integrated state, $p$ is independent of $q$ in the
        autograd graph, so `jax.grad` through $H$ needs no `stop_gradient`. That
        single gradient produces **both** gravity and the full inter-link
        Coriolis field — no Christoffel symbols anywhere.
        """
        n = self.n

        def H_of_q(q_):
            return (0.5 * jnp.dot(p, self.M_inv(q_, keys=keys) @ p)
                    + self._call(self.V_net, q_, self._k(keys, 'V'))[0])

        dHdq = jax.grad(H_of_q)(q)                               # (9n,)

        xi = self.M_inv(q, keys=keys) @ p                        # (3n,)
        gyroscopic = jnp.cross(p.reshape(n, 3), xi.reshape(n, 3), axis=-1)
        conservative = jnp.sum(
            jnp.cross(q.reshape(n, 3, 3), dHdq.reshape(n, 3, 3), axis=-1), axis=1)

        g_q = self._call(self.g_net, q, self._k(keys, 'g'))
        actuation = (g_q @ u).reshape(n, 3)

        dp = gyroscopic + conservative + actuation
        if self.friction:
            D_q = self._call(self.Dw_net, q, self._k(keys, 'Dw'))
            dp = dp - (D_q @ xi).reshape(n, 3)
        return dp.reshape(3 * n)

    def drift(self, q, omega, u, keys=None):
        r"""$\dot\omega = M^{-1}\dot p + \dot{(M^{-1})}\,p$ — used by the PL term,
        which evaluates the transition density in $\omega$-space.

        Reconstructs $p = M(q)\omega$ (a solve, since `M_net` returns $M^{-1}$),
        calls :meth:`drift_p`, then converts $\dot p\to\dot\omega$ with the JVP
        of $M^{-1}$ along $\dot q$. The `stop_gradient` on $p$ keeps the observed
        $\omega$ from back-propagating a second, spurious path into `M_net`.
        """
        n = self.n
        M_inv_q = self.M_inv(q, keys=keys)
        p = jax.lax.stop_gradient(jnp.linalg.solve(M_inv_q, omega))

        dp = self.drift_p(q, p, u, keys=keys)

        # $\dot q$ in the flat 9n embedding: row k of $\dot R_i$ is $r_{ik}\times\xi_i$.
        xi = (M_inv_q @ p).reshape(n, 3)
        dq = jnp.cross(q.reshape(n, 3, 3),
                       jnp.broadcast_to(xi[:, None, :], (n, 3, 3)),
                       axis=-1).reshape(9 * n)

        _, dM_inv_dt = jax.jvp(lambda q_: self.M_inv(q_, keys=keys), (q,), (dq,))
        return M_inv_q @ dp + dM_inv_dt @ p

    # ── ELBO helper ──────────────────────────────────────────────────

    def kl_loss(self):
        r"""$\sum_i \mathrm{KL}\big(q_\psi(w_i)\Vert\mathcal N(0,I)\big)$ over the five subnets."""
        return (self.M_net.weight_kl_loss()
                + self.V_net.weight_kl_loss()
                + self.Dw_net.weight_kl_loss()
                + self.g_net.weight_kl_loss()
                + self.Sigma_net.weight_kl_loss())


class KeyedArmModel(eqx.Module):
    r"""Binds a fixed set of GP weight-sample keys to a model.

    The integrator calls `drift_p(q, p, u)` positionally, with no notion of
    variational keys. Wrapping the model here threads one coherent $w$ sample
    through the predictor, the corrector and every substep of a rollout —
    required for the ELBO's Monte-Carlo estimate to be a valid draw from
    $q_\psi(w)$ rather than a fresh sample at each evaluation.
    """
    model: DissipativeArmHamSDE
    keys: dict
    inference_mode: bool = eqx.field(static=True)

    def _eff(self):
        return None if self.inference_mode else self.keys

    @property
    def n(self):
        return self.model.n

    def drift_p(self, q, p, u):
        return self.model.drift_p(q, p, u, keys=self._eff())

    def stochastic_increment_p(self, q, dW):
        return self.model.stochastic_increment_p(q, dW, keys=self._eff())

    def M_inv(self, q):
        return self.model.M_inv(q, keys=self._eff())

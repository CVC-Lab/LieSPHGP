r"""Unified port-Hamiltonian network on $SO(3)^n$ — the core of all four variants.

The four models differ along exactly **two** axes:

    ==============  ================  ==============  ===========================
    model           subnets           diffusion       objective
    ==============  ================  ==============  ===========================
    ``ph_gp_sde``   **structured**    $\Sigma_\theta(q)$  NLL + PL + KL
    ``ph_gp_ode``   variational GP    none            NLL + PL + KL
    ``ph_nn_sde``   MLP               $\Sigma_\theta(q)$  NLL + PL
    ``ph_nn_ode``   MLP               none            NLL + PL
    ==============  ================  ==============  ===========================

so they share one implementation parameterised by `subnet_kind`
(``'structured' | 'gp' | 'nn'``) and `stochastic`, both static fields.

``'structured'`` learns the physical constants of the closed forms for
$M^{-1}$, $D$ and $g$ (31 numbers at $n=2$) instead of approximating those
functions; ``'gp'`` and ``'nn'`` learn them as functions of $q$ and remain the
black-box comparators. Keeping the physics in one place is the point:
a bug fixed here is fixed for every variant, whereas four hand-copied networks
would need four fixes (which is exactly how the torque off-by-one survived).

Where the *sub-components* come from  (`gp_core` / `nn_core`)
------------------------------------------------------------
Orthogonal to `subnet_kind`, the structured kind chooses how the constants of
those closed forms are produced:

    =====================  ==================================================
    flag                   $(m,\mathbb{I},\ell,c,d,\gamma,g)$ come from
    =====================  ==================================================
    neither                learned constants (the physically exact model)
    ``gp_core``            $\theta_{\rm base}+\kappa\tanh(\mathrm{GP}(q)/\kappa)$
    ``nn_core``            $\theta_{\rm base}+\kappa\tanh(\mathrm{MLP}(q)/\kappa)$
    =====================  ==================================================

Both cores are *residual and zero at init*, so they are strict supersets of the
constants-only model. This is the axis that separates the two experiments this
file supports:

    * ``subnet_kind='nn'`` — an MLP emits **every matrix entry** of
      $M^{-1},D,g$ and the scalar $V$. No physics formula anywhere.
    * ``subnet_kind='structured', nn_core=True, structured_potential=True`` —
      an MLP emits only the **sub-components** $(m,\mathbb{I},\ell,c,d,\gamma,g)$
      and the closed forms of §4.1/§4.2/§7/§8 assemble the matrices from them.

``structured_potential`` is what extends the second experiment to $V$: without
it, $V$ stays the black-box GP that ``ph_gp_sde`` uses (default, so bit-parity
with that model is preserved).

The dynamics (§6, §10 of `multi-joint-ph-system.md`)
----------------------------------------------------
.. math::
    H(q,p) &= \tfrac12 p^\top M_\theta^{-1}(q)p + V_\theta(q),
    \qquad \xi = M_\theta^{-1}(q)\,p \\
    dq_i &= R_i[\xi_i]_\times\,dt \\
    dp   &= \big(\hat P\xi + \mathcal T(H) - D_\theta(q)\xi + g_\theta(q)u\big)dt
            \;+\;\Sigma_\theta(q)\circ dW_t

**ODE variants keep the same equations with $\Sigma_\theta \equiv 0$.** They are
not a different model class — just the deterministic limit. That also means the
per-increment pseudo-likelihood works unchanged for them: with $\Sigma=0$ the
Woodbury form collapses to $\Sigma_{\rm eff}=2\sigma_{\rm obs}^2 I$, i.e. a
fixed-variance Gaussian on $\Delta\omega$. No branch in the loss code.

.. note::
   ``ph_gp_sde`` deliberately does **not** import this module — it keeps its own
   validated ``ph_gp_sde/network.py``, and re-pointing it mid-experiment would
   risk silently changing published numbers. ``test_variants.py`` asserts that
   ``ArmPortHamiltonian(subnet_kind='structured')`` reproduces it **bit for
   bit**, which is what keeps the duplication honest: if the two ever diverge,
   that test fails.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gp_model_nlink import GP_NLink, PSD_GP_NLink, MatrixGP_NLink   # noqa: E402
from structured_subnets import (                                   # noqa: E402
    StructuredMass, StructuredDissipation, StructuredInputMap, StructuredSigma,
    StructuredPotential, SharedStructuredPotential,
)
from nn_model_nlink import MLP, PSD_NN, MatrixNet_NN                # noqa: E402
from lie_integrator_nlink import hat                                # noqa: E402

hat_batch = jax.vmap(hat)

SUBNET_NAMES = ('M', 'V', 'Dw', 'g', 'Sigma')


def _make_sigma(key, n, d, hidden_dim, sigma_mode, kind, init_gain, dtype,
                dim_abs=None):
    """Diffusion subnet: physics-structured, or a free learned map."""
    if sigma_mode == 'structured':
        return StructuredSigma(key, n=n, dtype=dtype)
    if kind == 'gp':
        return MatrixGP_NLink(key, n_links=n, shape=(d, 3),
                              n_matern_features=hidden_dim, dtype=dtype)
    return MatrixNet_NN(key, dim_abs, hidden_dim, (d, 3),
                        init_gain=init_gain, dtype=dtype)


class ArmPortHamiltonian(eqx.Module):
    r"""Port-Hamiltonian (S)DE on $SO(3)^n\times\mathbb{R}^{3n}$."""

    M_net:     eqx.Module
    V_net:     eqx.Module
    Dw_net:    eqx.Module
    g_net:     eqx.Module
    Sigma_net: Optional[eqx.Module]      # None for the deterministic variants

    log_sigma_R:     jnp.ndarray
    log_sigma_omega: jnp.ndarray

    sigma_obs_omega: float = eqx.field(static=True)
    n:               int   = eqx.field(static=True)
    u_dim:           int   = eqx.field(static=True)
    friction:        bool  = eqx.field(static=True)
    relative_inputs: bool  = eqx.field(static=True)
    stochastic:      bool  = eqx.field(static=True)
    subnet_kind:     str   = eqx.field(static=True)   # 'gp' | 'nn' | 'structured'
    sigma_mode:      str   = eqx.field(static=True)   # 'full' | 'structured'
    # Ported from ph_gp_sde (gp_change.md, Changes 3-5). `anchor_trace` is the
    # function-space scale-gauge anchor: the pH gauge (M,V,D,g,Sigma) ->
    # beta*(all) is flat for EVERY subnet kind, and parameter-space pins do not
    # close it (see gp_change.md §3c). For 'structured' the pin lives inside
    # StructuredMass (tr M == anchor, bit-parity with ph_gp_sde); for
    # 'gp'/'nn', whose nets emit M^-1 directly, M_inv() pins tr M^-1 instead —
    # either pin kills the same one-dimensional gauge; the value is a pure
    # units choice. `sigma_detach_rollout` cuts the rollout NLL's gradient
    # into Sigma (pure collapse pressure); the PL alone owns the diffusion.
    anchor_trace: Optional[float] = eqx.field(static=True)
    sigma_detach_rollout: bool = eqx.field(static=True)
    # V borrows M's (m, ell, c) instead of re-estimating them. See
    # SharedStructuredPotential and `potential()` below.
    share_mass_potential: bool = eqx.field(static=True)

    def __init__(self, *, key, n: int = 2, subnet_kind: str = 'gp',
                 stochastic: bool = True, u_dim: int = None,
                 hidden_dim: int = 32, friction: bool = True,
                 relative_inputs: bool = False, init_gain: float = 0.5,
                 init_sigma_R: float = 0.1, init_sigma_omega: float = 0.1,
                 init_sigma_obs_omega: float = 0.1,
                 sigma_mode: str = 'full',
                 m_epsilon: float = 0.5, d_epsilon: float = 0.5,
                 variational_core: bool = False, core_prior_std: float = 1.0,
                 gp_core: bool = False, nn_core: bool = False,
                 structured_potential: bool = False,
                 share_mass_potential: bool = False,
                 gravity: float = None, i_epsilon: float = None,
                 anchor_m1: float = None, anchor_trace: float = None,
                 sigma_detach_rollout: bool = True,
                 dtype=jnp.float32):
        if subnet_kind not in ('gp', 'nn', 'structured'):
            raise ValueError("subnet_kind must be 'gp', 'nn' or 'structured', "
                             f"got {subnet_kind!r}")
        if sigma_mode not in ('full', 'structured'):
            raise ValueError("sigma_mode must be 'full' or 'structured', "
                             f"got {sigma_mode!r}")
        if gp_core and nn_core:
            raise ValueError('gp_core and nn_core are mutually exclusive: both '
                             'emit the same physical-constant residual.')
        if gravity is not None and gravity > 0 and not share_mass_potential:
            # Only SharedStructuredPotential can drop log_g cleanly. In the
            # unshared StructuredPotential, log_g is one slice of the core net's
            # output vector, so removing it would silently shift every later
            # slice (ell, c) by one -- a wrong-parameter bug with no error.
            raise ValueError('gravity=... (fixed g) requires '
                             'share_mass_potential=True.')
        if share_mass_potential and not structured_potential:
            # Sharing means "V is built from StructuredMass's constants". With a
            # GP/MLP potential there are no constants to borrow, so silently
            # accepting this would produce an unshared model under a flag that
            # says otherwise.
            raise ValueError('share_mass_potential requires '
                             'structured_potential=True (there is nothing to '
                             'share with a black-box V_net).')
        if structured_potential and subnet_kind != 'structured':
            # Allowing it would silently build a *constants-only* potential
            # (gp/nn ignore both core flags) next to black-box M/D/g — a
            # half-configured model that is easy to mistake for the real thing.
            raise ValueError("structured_potential requires "
                             f"subnet_kind='structured', got {subnet_kind!r}")
        self.sigma_mode = sigma_mode

        self.n = int(n)
        self.subnet_kind = subnet_kind
        self.stochastic = bool(stochastic)
        self.u_dim = int(3 * n if u_dim is None else u_dim)
        self.friction = bool(friction)
        # A core net makes the invariance no longer free (the constants become
        # functions of q), so either core defaults relative_inputs ON — exactly
        # as ph_gp_sde does for gp_core. Only the structured kind has core nets;
        # gp/nn ignore both flags entirely.
        self.relative_inputs = bool(
            relative_inputs
            or ((gp_core or nn_core) and subnet_kind == 'structured')
        ) and self.n > 1
        self.anchor_trace = (None if anchor_trace is None or anchor_trace <= 0
                             else float(anchor_trace))
        self.sigma_detach_rollout = bool(sigma_detach_rollout)
        self.share_mass_potential = bool(share_mass_potential)

        d = 3 * self.n
        self.log_sigma_R = jnp.log(jnp.asarray(init_sigma_R, dtype=dtype))
        self.log_sigma_omega = jnp.log(jnp.asarray(init_sigma_omega, dtype=dtype))
        self.sigma_obs_omega = float(init_sigma_obs_omega)

        kM, kV, kD, kg, kS = jax.random.split(key, 5)

        # M, D and g are rotation-invariant (they depend on q only through
        # relative rotations), so with `relative_inputs` they see n-1 of them.
        # V and Sigma are not: gravity picks out e_z and the wind direction is a
        # world-frame object, so both always take absolute attitudes.
        n_inv = (self.n - 1) if self.relative_inputs else self.n
        dim_inv, dim_abs = 9 * n_inv, 9 * self.n

        if subnet_kind == 'structured':
            # Physics-structured M/D/g: the closed forms of §4.1/§7/§8, learning
            # only the physical constants. Built with the *same* key split as
            # the 'gp' branch so `ph_gp_sde.DissipativeArmHamSDE` and
            # `ArmPortHamiltonian(subnet_kind='structured')` are bit-identical
            # -- see test_variants.py::test_matches_ph_gp_sde.
            vkw = dict(variational=variational_core, prior_std=core_prior_std,
                       gp_core=gp_core, nn_core=nn_core, hidden_dim=hidden_dim)
            ikw = dict(relative_inputs=self.relative_inputs)
            mkw = {} if i_epsilon is None else dict(i_epsilon=i_epsilon)
            # Anchors go INSIDE StructuredMass (tr M pinned there), exactly as
            # in ph_gp_sde — the M_inv() wrapper pin below must therefore skip
            # the structured kind or M would be normalised twice.
            self.M_net = StructuredMass(kM, n=self.n, dtype=dtype,
                                        anchor_m1=anchor_m1,
                                        anchor_trace=anchor_trace,
                                        **vkw, **ikw, **mkw)
            self.Dw_net = StructuredDissipation(kD, n=self.n, dtype=dtype, **vkw, **ikw)
            self.g_net = StructuredInputMap(kg, n=self.n, dtype=dtype, **vkw, **ikw)
            # V is the one subnet ph_gp_sde leaves black-box, so the structured
            # gravity form is opt-in: with `structured_potential=False` the key
            # split and every leaf are unchanged, which is what keeps
            # test_variants.py::test_matches_ph_gp_sde bit-exact.
            if share_mass_potential:
                # One learnable scalar: log g. Everything else comes from M_net.
                self.V_net = SharedStructuredPotential(
                    kV, n=self.n, variational=variational_core,
                    prior_std=core_prior_std, gravity=gravity, dtype=dtype)
            elif structured_potential:
                self.V_net = StructuredPotential(kV, n=self.n, dtype=dtype, **vkw)
            else:
                self.V_net = GP_NLink(kV, n_links=self.n, output_dim=1,
                                      n_matern_features=hidden_dim, dtype=dtype)
            self.Sigma_net = (_make_sigma(kS, self.n, d, hidden_dim,
                                          sigma_mode, 'gp', init_gain, dtype)
                              if self.stochastic else None)
        elif subnet_kind == 'gp':
            self.M_net = PSD_GP_NLink(kM, n_links=n_inv, diag_dim=d,
                                      epsilon=m_epsilon,
                                      n_matern_features=hidden_dim, dtype=dtype)
            self.Dw_net = PSD_GP_NLink(kD, n_links=n_inv, diag_dim=d,
                                       epsilon=d_epsilon,
                                       n_matern_features=hidden_dim, dtype=dtype)
            self.g_net = MatrixGP_NLink(kg, n_links=n_inv, shape=(d, self.u_dim),
                                        n_matern_features=hidden_dim, dtype=dtype)
            self.V_net = GP_NLink(kV, n_links=self.n, output_dim=1,
                                  n_matern_features=hidden_dim, dtype=dtype)
            self.Sigma_net = (_make_sigma(kS, self.n, d, hidden_dim,
                                          sigma_mode, 'gp', init_gain, dtype)
                              if self.stochastic else None)
        else:
            self.M_net = PSD_NN(kM, dim_inv, hidden_dim, d, epsilon=m_epsilon,
                                init_gain=init_gain, dtype=dtype)
            self.Dw_net = PSD_NN(kD, dim_inv, hidden_dim, d, epsilon=d_epsilon,
                                 init_gain=init_gain, dtype=dtype)
            self.g_net = MatrixNet_NN(kg, dim_inv, hidden_dim, (d, self.u_dim),
                                      init_gain=init_gain, dtype=dtype)
            self.V_net = MLP(kV, dim_abs, hidden_dim, 1,
                             init_gain=init_gain, dtype=dtype)
            self.Sigma_net = (_make_sigma(kS, self.n, d, hidden_dim,
                                          sigma_mode, 'nn', init_gain, dtype,
                                          dim_abs=dim_abs)
                              if self.stochastic else None)

    # ── input featurisation ──────────────────────────────────────────

    def _invariant_q(self, q):
        r"""$(\mathrm{vec}(R_1^\top R_2),\dots)$ when `relative_inputs`, else $q$.

        Inert for `subnet_kind='structured'`: those closed forms contain
        $R_j^\top R_k$ and $T(q)$ literally, so the invariance is exact rather
        than an input reparameterisation, and they expect the raw $q$.
        """
        if self.subnet_kind == 'structured' or not self.relative_inputs:
            return q
        R = q.reshape(self.n, 3, 3)
        return jnp.einsum('iab,iac->ibc', R[:-1], R[1:]).reshape(9 * (self.n - 1))

    def _call(self, net, x, key):
        """Uniform call across families: NN modules ignore `key`."""
        return net(x, key=key, inference_mode=(key is None))

    @staticmethod
    def _k(keys, name):
        return None if keys is None else keys.get(name)

    # ── structure matrices ───────────────────────────────────────────

    def M_inv(self, q, keys=None):
        Minv = self._call(self.M_net, self._invariant_q(q), self._k(keys, 'M'))
        # Function-space gauge anchor for the black-box kinds: their nets emit
        # M^-1 directly, so pin tr M^-1 pointwise. ('structured' pins tr M
        # inside StructuredMass instead — do not normalise twice.)
        if self.anchor_trace is not None and self.subnet_kind != 'structured':
            Minv = Minv * (self.anchor_trace / jnp.trace(Minv))
        return Minv

    def potential(self, q, keys=None):
        r"""$V_\theta(q)$ as a **scalar** — the single entry point for the potential.

        Every caller (the drift, the training diagnostics, the report) must come
        through here, because with `share_mass_potential` the potential cannot be
        evaluated from `V_net` alone: it needs $(m,\ell,c)$ and the gauge factor
        $\alpha$ out of `M_net`.

        .. important::
           `M_net.physical` is called with the **same** key as the $M^{-1}$ path
           uses (`keys['M']`, not `keys['V']`). Under `variational_core` that key
           selects which sample of $(m,\ell,c)$ is drawn, and $M$ and $V$ must
           see the *same* draw — otherwise they are built from different physical
           constants within one evaluation and the sharing is undone exactly when
           it matters. `keys['V']` still drives $\log g$, which is V's own.
        """
        if not self.share_mass_potential:
            return self._call(self.V_net, q, self._k(keys, 'V'))[0]
        kM = self._k(keys, 'M')
        m, _, ell, c = self.M_net.physical(q, kM, inference_mode=(kM is None))
        alpha = self.M_net.anchor_scale(q, kM, inference_mode=(kM is None))
        kV = self._k(keys, 'V')
        return self.V_net(q, m, ell, c, alpha, key=kV,
                          inference_mode=(kV is None))[0]

    def Sigma(self, q, keys=None):
        r"""$\Sigma_\theta(q)\in\mathbb{R}^{3n\times3}$; **exactly zero** for the
        deterministic variants, which is what makes `pl_loss_nlink` reduce to a
        fixed-variance Gaussian without needing a separate code path."""
        if not self.stochastic:
            return jnp.zeros((3 * self.n, 3), dtype=q.dtype)
        return self._call(self.Sigma_net, q, self._k(keys, 'Sigma'))

    def stochastic_increment_p(self, q, dW, keys=None):
        r"""$\Sigma_\theta(q)\,dW$. Used only by the rollout integrator (the PL
        calls :meth:`Sigma` directly), so the detach removes exactly one
        gradient path: the rollout NLL's pure $\Sigma\to0$ collapse pressure.
        Values are unchanged; a no-op for the deterministic variants."""
        Sig = self.Sigma(q, keys=keys)
        if self.sigma_detach_rollout:
            Sig = jax.lax.stop_gradient(Sig)
        return Sig @ dW

    # ── drift ────────────────────────────────────────────────────────

    def drift_p(self, q, p, u, keys=None):
        r"""$\dot p_i = p_i\times\xi_i + \mathcal T_i(H) - (D\xi)_i + (gu)_i$.

        One `jax.grad` through $H$ produces **both** gravity and the full
        inter-link Coriolis field — no Christoffel symbols. $\mathcal T_i$ is a
        directional derivative along the group action, so it is independent of
        how $H$ is extended off $SO(3)^n$ and the redundant $\mathbb{R}^{9n}$
        representation of $q$ is safe.
        """
        n = self.n

        def H_of_q(q_):
            return (0.5 * jnp.dot(p, self.M_inv(q_, keys=keys) @ p)
                    + self.potential(q_, keys=keys))

        dHdq = jax.grad(H_of_q)(q)

        xi = self.M_inv(q, keys=keys) @ p
        gyroscopic = jnp.cross(p.reshape(n, 3), xi.reshape(n, 3), axis=-1)
        conservative = jnp.sum(
            jnp.cross(q.reshape(n, 3, 3), dHdq.reshape(n, 3, 3), axis=-1), axis=1)

        g_q = self._call(self.g_net, self._invariant_q(q), self._k(keys, 'g'))
        dp = gyroscopic + conservative + (g_q @ u).reshape(n, 3)

        if self.friction:
            D_q = self._call(self.Dw_net, self._invariant_q(q), self._k(keys, 'Dw'))
            dp = dp - (D_q @ xi).reshape(n, 3)
        return dp.reshape(3 * n)

    def drift(self, q, omega, u, keys=None):
        r"""$\dot\omega = M^{-1}\dot p + \dot{(M^{-1})}p$ — used by the PL term."""
        n = self.n
        M_inv_q = self.M_inv(q, keys=keys)
        p = jax.lax.stop_gradient(jnp.linalg.solve(M_inv_q, omega))
        dp = self.drift_p(q, p, u, keys=keys)

        xi = (M_inv_q @ p).reshape(n, 3)
        dq = jnp.cross(q.reshape(n, 3, 3),
                       jnp.broadcast_to(xi[:, None, :], (n, 3, 3)),
                       axis=-1).reshape(9 * n)
        _, dM_inv_dt = jax.jvp(lambda q_: self.M_inv(q_, keys=keys), (q,), (dq,))
        return M_inv_q @ dp + dM_inv_dt @ p

    # ── ELBO helper ──────────────────────────────────────────────────

    def kl_loss(self):
        r"""$\sum_i\mathrm{KL}(q_\psi(w_i)\|\mathcal N(0,I))$.

        Exactly zero for the NN variants — every module there returns zero — so
        the KL term vanishes by construction rather than by being switched off.
        """
        total = (self.M_net.weight_kl_loss() + self.V_net.weight_kl_loss()
                 + self.Dw_net.weight_kl_loss() + self.g_net.weight_kl_loss())
        if self.stochastic:
            total = total + self.Sigma_net.weight_kl_loss()
        return total


class KeyedArmModel(eqx.Module):
    r"""Binds one set of GP weight-sample keys to a model.

    The integrator calls `drift_p(q, p, u)` positionally and knows nothing about
    variational keys; this wrapper threads a single coherent $w$ sample through
    the predictor, the corrector and every substep of a rollout, which is what
    makes the Monte-Carlo estimate a valid draw from $q_\psi(w)$. Harmless for
    the NN variants, whose subnets ignore keys entirely.
    """
    model: ArmPortHamiltonian
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

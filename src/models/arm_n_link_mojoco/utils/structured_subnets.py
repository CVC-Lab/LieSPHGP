r"""Physics-structured replacements for the $M^{-1}$, $D$ and $g$ subnetworks.

Instead of a GP approximating the *function* $q\mapsto M^{-1}(q)$, these modules
learn the handful of **physical parameters** that appear in the closed forms of
§4.1, §7 and §8 of `multi-joint-ph-system.md`, and evaluate the exact algebra.

Parameter counts at $n=2$, against the GP subnets they replace:

===============  =======================  ==============
subnet           GP weights               structured
===============  =======================  ==============
`M_net`          7 392                    23
`Dw_net`         7 392                    2
`g_net`          12 672                   6
`V_net`          3 168                    12
===============  =======================  ==============

The measured payoff on $M^{-1}$: a supervised fit of the 23 parameters reaches
$2\times10^{-5}$ relative error on held-out configurations, versus $0.044$ for
the best GP fit (at $\epsilon=0.5$) and $0.215$–$0.337$ for the GP as actually
trained through the dynamics.

.. important::
   This module is **self-contained** — it must never import
   ``envs.arm_nlink_SO3``. The whole value of
   ``ph_gp_sde/eval_ground_truth_match.py`` is that it compares two
   *independent* implementations of the same algebra; sharing code would make
   that gate vacuous. The kinematics below are therefore rewritten here.

.. note::
   These learn the same **identifiable combinations** the data determines, not
   the true physical constants. $\mathbb{I}_n$ and $m_n[c_n]_\times^\top[c_n]_\times$
   only ever appear summed (the classic *base inertial parameters* degeneracy),
   so the recovered $m_i$, $c_i$ are not the true masses and offsets even when
   $M(q)$ is reproduced to machine precision. Do not report them as identified
   physical quantities.
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

from gp_model_nlink import GP_NLink                                # noqa: E402
from nn_model_nlink import MLP                                     # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# Self-contained kinematics (deliberately not imported from the env)
# ══════════════════════════════════════════════════════════════════════

def _hat(v):
    r"""$[v]_\times$ for a single 3-vector, autodiff-safe."""
    z = jnp.zeros_like(v[0])
    return jnp.stack([
        jnp.stack([z, -v[2], v[1]]),
        jnp.stack([v[2], z, -v[0]]),
        jnp.stack([-v[1], v[0], z]),
    ])


def _strict_suffix_sum(x):
    r"""`out[j] = ` $\sum_{i>j}x_i$ along axis 0 (`out[-1]` is zero).

    The downstream-mass accumulator $\mu_j^{>}=\sum_{i>j}m_i$ of the gravity
    formula. Rewritten here rather than imported for the same reason as
    :func:`joint_rate_map` — this module must stay independent of the env.
    """
    inclusive = jnp.cumsum(x[::-1], axis=0)[::-1]          # $\sum_{i\ge j}$
    zero = jnp.zeros((1,) + x.shape[1:], dtype=x.dtype)
    return jnp.concatenate([inclusive[1:], zero], axis=0)


def joint_rate_map(R, n):
    r"""$T(q)$ with $\Omega = T(q)\,\omega$, shape `(3n, 3n)`.

    Block lower-bidiagonal: $T_{ii}=I_3$ and $T_{i,i-1}=-R_i^\top R_{i-1}$,
    from $\Omega_i = \omega_i - R_i^\top R_{i-1}\omega_{i-1}$ with $\omega_0:=0$.
    $\det T = 1$, so $T$ is always invertible — which is why joint friction
    alone damps every direction.
    """
    d = 3 * n
    T = jnp.eye(d, dtype=R.dtype)
    for i in range(1, n):
        T = T.at[3 * i:3 * i + 3, 3 * (i - 1):3 * i].set(-R[i].T @ R[i - 1])
    return T


# ══════════════════════════════════════════════════════════════════════
# Optional variational treatment of the physical constants
# ══════════════════════════════════════════════════════════════════════
# With `variational=True` each constant $\theta$ carries a mean-field Gaussian
# posterior $q_\psi(\theta)=\mathcal N(\mu,\sigma^2)$ instead of a point
# estimate, sampled by reparameterisation exactly as the GP weights are:
#
#     $\theta = \mu + \sigma\,\epsilon$,   $\epsilon\sim\mathcal N(0,I)$
#
# i.e. **Bayesian system identification** of the physical constants, rather
# than a maximum-likelihood fit. The closed forms are untouched; only where the
# numbers come from changes.
#
# .. note::
#    The prior is $\mathcal N(0,\texttt{prior\_std}^2)$. For these constants that
#    is *weakly informative* rather than wrong — it says "values of order
#    `prior_std` are plausible", and the truths here ($m=1$, $\ell=c=e_z$,
#    $d=0.5$, $\gamma\sim0.5$) all sit within one prior std of zero. It does
#    still shrink toward zero, so a large `prior_std` is the conservative
#    choice if the true scale is unknown.
#
# .. warning::
#    The KL enters the objective weighted by `--beta_max`, which **defaults to
#    0**. At that default the posterior is unregularised and this only injects
#    sampling noise; set `--beta_max > 0` for the variational treatment to
#    actually do anything.

_LOG_2PI = float(jnp.log(2.0 * jnp.pi))

# ── Inertia conditioning floor ───────────────────────────────────────
# $\mathbb{I}_i = L_iL_i^\top + \varepsilon_{\mathbb I} I_3$.
#
# This is a **guarantee**, not a heuristic. The translational term of $M$ is
# PSD, so $M\succeq\mathrm{blkdiag}(\mathbb{I}_i)\succeq\varepsilon_{\mathbb I}I$
# and therefore $\lambda_{\min}(M)\ge\varepsilon_{\mathbb I}$ at *every*
# configuration.
#
# It is needed because $L L^\top$ is only positive **semi**-definite -- it can
# be rank-deficient -- and $M$'s axial-spin block comes from $\mathbb{I}$ alone
# (both $[\ell_i]_\times$ and $[c_i]_\times$ annihilate $e_z$ when the levers lie
# along $e_z$, so the translational term contributes nothing there; see the
# warning in ``uniform_chain_params``). Without the floor, `L_body` initialised
# at `init_scale=0.3` gives $\lambda_{\min}(M)\approx0.019$ against a true
# $0.382$ -- twenty times too small, $\max|M^{-1}|\approx43$ at step 0 -- and
# any configuration where $\mathbb{I}$ degenerates sends $M^{-1}$ to NaN. That
# is survivable with constant parameters (one globally decent $\mathbb{I}$
# suffices) and fatal with `gp_core`, where the condition must hold at every
# $q$ the rollout visits.
#
# 0.05 sits 7.6x below the true $\lambda_{\min}=0.382$, so it cannot distort the
# fit the way $\epsilon=1$ distorted the old `PSD_GP_NLink` (where 33% of the
# true diagonal lay below the floor).
I_EPSILON = 0.05

# ── GP-residual bound (gp_core) ──────────────────────────────────────
# Every `gp_core` residual is squashed before it enters the closed forms:
#
#     theta(q) = theta_base + kappa * tanh(GP(q) / kappa)
#
# so |theta(q) - theta_base| < kappa at every q and for EVERY sampled
# weight draw.
#
# Why this is necessary and not cosmetic: the log-parameterised constants
# are exponentiated -- m = exp(log_m), d = exp(log_d) -- so Gaussian
# weight noise becomes heavy-tailed *multiplicative* noise on the physics.
# With --beta_max > 0 the KL widens the weight posteriors toward N(0,1)
# (dKL/dsigma = sigma - 1/sigma < 0 for sigma < 1), and one tail draw
# |GP(q)| ~ 10 gives m ~ e^10: the stiffness lambda_max(M^{-1}D) crosses
# the explicit-Heun stability limit h*lambda ~ 2, the rollout grows
# geometrically over the substeps and overflows float32 -> NaN. Observed
# twice, both times ~500-1500 steps after beta saturated; weights are
# (correctly) held fixed per trajectory, so a single bad draw poisons a
# whole window, and `clip_by_global_norm` of NaN gradients is NaN, which
# kills the run permanently.
#
# The bound removes the failure *structurally*: m, d stay within
# e^{+-kappa} of their base values, so the drift's growth rate is
# uniformly bounded at every q for every draw and float32 overflow cannot
# occur. It costs nothing representationally -- the residual gp_core
# exists to capture (e.g. the varying-friction modulation rho in [1, 2])
# needs |residual| <= ln 2 ~ 0.69, deep inside the linear regime of
# kappa = 2.5 where kappa*tanh(r/kappa) is within ~3% of the identity.
# The gradient sech^2(r/kappa) in (0, 1] never vanishes, so no dead zones.
# Linear constants (L, ell, c, gains, v) are squashed too as cheap
# insurance: they enter M quadratically and inflate lambda_max the same way.
RESIDUAL_KAPPA = 2.5


def _squash(r, kappa=RESIDUAL_KAPPA):
    r"""$\kappa\tanh(r/\kappa)$ — identity near $0$, hard-bounded by $\pm\kappa$."""
    return kappa * jnp.tanh(r / kappa)


def _init_log_std(shape, dtype, variational, value=-3.0):
    r"""Posterior log-std, or **None** for the point-estimate path.

    Returning `None` rather than a zero array matters: JAX treats `None` as an
    empty pytree node, so with `variational=False` these modules have exactly
    the leaves they had before the variational option existed, and checkpoints
    written by the old code still deserialise.

    Initialised tight ($\sigma=e^{-3}\approx0.05$) so training starts
    near-deterministic and the posterior widens only if the data supports it.
    """
    return None if not variational else jnp.full(shape, value, dtype=dtype)


def _draw(mean, log_std, key, variational, inference_mode):
    r"""$\mu$, or a reparameterised sample $\mu+\sigma\epsilon$."""
    if (not variational) or inference_mode or key is None or log_std is None:
        return mean
    eps = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    return mean + jnp.exp(log_std) * eps


# ══════════════════════════════════════════════════════════════════════
# Optional GP-predicted constants  (`gp_core`)
# ══════════════════════════════════════════════════════════════════════
# With `gp_core=True` the physical parameters stop being constants and become
# **functions of the configuration**, produced by the same Matern x periodic
# variational GP the rest of the model uses:
#
#     GP(q) -> [m, I, ell, c] -> closed form -> M(q)
#
# The formula is untouched; only where its inputs come from changes. This is a
# grey-box: the structure supplies the skeleton, the GP absorbs whatever the
# rigid-body model cannot express.
#
# .. important::
#    Use :class:`GP_NLink`, **never** ``src/utils/JAX/gp_model.GP_Model``.
#    That class does ``rot_part = x[..., :9]``, so on an $9n$-dimensional $q$ it
#    silently reads only link 1 and ignores the rest of the arm -- no error, a
#    model blind to half the system. `GP_NLink` is the verified generalisation
#    (bit-identical at $n=1$, product Matern kernel on $SO(3)^n$ for $n>1$).
#
# .. warning::
#    $M(hq)=M(q)$ under a global rotation $h$ is **exact** while the constants
#    are constants. Once they are $\mathrm{GP}(q)$ it holds only if the GP is
#    fed rotation-invariant inputs -- hence `relative_inputs`, which this mode
#    revives. Feed absolute attitudes and the invariance must be learned from
#    data instead of being free.
#
# The GP already carries `w_mean` / `log_w_covar` / `weight_kl_loss`, so this
# mode is variational by construction and `--beta_max` acts on it directly;
# `--variational_core` is redundant here.
#
# .. important::
#    The GP predicts a **residual on top of base constants**, not the constants
#    outright, and the residual is bounded (see `RESIDUAL_KAPPA` above):
#
#        theta(q) = theta_base + kappa * tanh(GP(q) / kappa)
#
#    This is not cosmetic. `GP_NLink` initialises `w_mean` at scale
#    $1/(D_mD_p)$ so the initial function is $\approx0$ -- and a raw GP output
#    would then give $\mathbb{I}\approx0$, $\ell\approx0$, $c\approx0$, hence
#    $M(q)\approx0$, a **singular** mass matrix whose inverse is NaN on the
#    first step (the same degeneracy `uniform_chain_params` warns about).
#    With the residual form the model *starts* exactly at the constant-only
#    model and departs from it only as the GP learns, so `--gp_core` is a
#    strict superset: if the GP learns nothing, you recover the constants.


def _make_core_gp(key, n, n_out, hidden_dim, relative_inputs, dtype):
    """GP over $SO(3)^n$ emitting `n_out` physical parameters."""
    n_inv = (n - 1) if (relative_inputs and n > 1) else n
    return GP_NLink(key, n_links=n_inv, output_dim=int(n_out),
                    n_matern_features=hidden_dim, dtype=dtype)


def _core_input(q, n, relative_inputs):
    r"""$(\mathrm{vec}(R_1^\top R_2),\dots)$ when invariance is wanted, else $q$."""
    if not (relative_inputs and n > 1):
        return q
    R = q.reshape(n, 3, 3)
    return jnp.einsum('iab,iac->ibc', R[:-1], R[1:]).reshape(9 * (n - 1))


# ══════════════════════════════════════════════════════════════════════
# Optional MLP-predicted constants  (`nn_core`)
# ══════════════════════════════════════════════════════════════════════
# The deterministic twin of `gp_core`: an MLP, not a variational GP, emits the
# physical parameters as a function of the configuration,
#
#     MLP(q) -> [m, I, ell, c] -> closed form -> M(q)
#
# so the black-box/grey-box axis is separated from the GP/NN axis. This is the
# subnet mode of the "NN predicts the sub-components, the formula assembles the
# matrix" experiment: the NN never sees a matrix entry, only the handful of
# constants the closed forms of §4.1/§4.2/§7/§8 take as inputs.
#
# Same residual form and the same bound as `gp_core` (see `RESIDUAL_KAPPA`):
#
#     theta(q) = theta_base + kappa * tanh(MLP(q) / kappa)
#
# .. important::
#    The MLP's **output layer is zero-initialised** (weight and bias), so the
#    residual is *exactly* 0 at step 0 and the model starts at the
#    constant-only structured model. This is not cosmetic — it is the same
#    requirement that forces `gp_core` to be a residual: an orthogonally
#    initialised MLP emits an O(1) random vector, which would give
#    I ~ ell ~ c ~ O(1) garbage per q, and any q where I degenerates sends
#    M^-1 to NaN on the first step. Zero-init makes `--nn_core` a strict
#    superset of the constants-only model: if the MLP learns nothing, you
#    recover the constants exactly.
#
# .. note::
#    Unlike `gp_core` this mode carries **no** weight posterior, so its KL is
#    identically zero and `--beta_max` is inert -- matching every other NN
#    subnet in ``nn_model_nlink.py``.


def _check_cores(gp_core, nn_core):
    """Reject `gp_core and nn_core`; return `bool(nn_core)`.

    Both write the *same* residual slot, so enabling both would silently apply
    only one of them (whichever `_core_residual` checks first) while the other's
    parameters trained on nothing.
    """
    if gp_core and nn_core:
        raise ValueError('gp_core and nn_core are mutually exclusive: both '
                         'emit the same physical-constant residual.')
    return bool(nn_core)


def _make_core_nn(key, n, n_out, hidden_dim, relative_inputs, dtype):
    """MLP over $SO(3)^n$ emitting `n_out` physical parameters, zero at init."""
    n_inv = (n - 1) if (relative_inputs and n > 1) else n
    net = MLP(key, 9 * n_inv, hidden_dim, int(n_out), init_gain=1.0, dtype=dtype)
    # Zero the read-out so MLP(q) == 0 for every q at step 0. See the note above.
    net = eqx.tree_at(lambda m: m.linear3.weight, net,
                      jnp.zeros_like(net.linear3.weight))
    return eqx.tree_at(lambda m: m.linear3.bias, net,
                       jnp.zeros_like(net.linear3.bias))


def _core_residual(gp, nn, q, n, relative_inputs, key, inference_mode):
    r"""Bounded residual $\kappa\tanh(\cdot/\kappa)$ from whichever core is on.

    Returns `None` when neither `gp_core` nor `nn_core` is active, so the
    caller's constant-only path is a plain `if ... is not None` guard rather
    than a duplicated formula per mode. The two cores are mutually exclusive
    (enforced in every ``__init__``), so the order of the checks is immaterial.
    """
    if gp is not None:
        return _squash(gp(_core_input(q, n, relative_inputs), key=key,
                          inference_mode=inference_mode))
    if nn is not None:
        # An MLP has no weight posterior: `key` / `inference_mode` are ignored,
        # exactly as in `nn_model_nlink.MLP.__call__`.
        return _squash(nn(_core_input(q, n, relative_inputs)))
    return None


def _kl(mean, log_std, prior_std):
    r"""$\mathrm{KL}\big(\mathcal N(\mu,\sigma^2)\,\Vert\,\mathcal N(0,s^2)\big)$, summed."""
    var = jnp.exp(2.0 * log_std)
    s2 = prior_std ** 2
    return 0.5 * jnp.sum(
        (var + mean ** 2) / s2 - 1.0 - 2.0 * log_std + jnp.log(s2))


# ══════════════════════════════════════════════════════════════════════
# Subnets
# ══════════════════════════════════════════════════════════════════════

class StructuredMass(eqx.Module):
    r"""$M^{-1}(q)$ from the closed-form mass matrix (§4.1).

    .. math::
        M_{jk}(q)=\mathbb{I}_j\delta_{jk}
                 -\sum_{i\ge\max(j,k)} m_i\,[u_{ij}]_\times R_j^\top R_k\,[u_{ik}]_\times,
        \qquad u_{ij}=\ell_j\ (j<i),\quad u_{ii}=c_i

    All $q$-dependence enters through the relative rotations $R_j^\top R_k$, so
    invariance under a global world rotation is exact by construction rather
    than learned — the inductive bias `--relative_inputs` only approximates.

    Learned: $\log m_i$ ($n$), a Cholesky factor of $\mathbb{I}_i$ ($6n$, keeping
    it PSD), the free joint offsets $\ell_0,\dots,\ell_{n-2}$ ($3(n{-}1)$;
    $\ell_{n-1}$ never enters the dynamics, since the last link has no
    downstream neighbour) and $c_i$ ($3n$). At $n=2$ that is 23 numbers.

    Returns $M^{-1}$ because that is what the model API and the Hamiltonian
    need; the inverse is a $3n\times3n$ solve, cheap at small $n$ and
    differentiable.

    Positive-**semi**-definiteness comes free from $m_i=e^{\log m_i}>0$ and
    $\mathbb{I}_i=L_iL_i^\top$, but *definiteness* does **not** — $LL^\top$ can
    be rank-deficient and every translational term is singular along $e_z$. The
    $\varepsilon_{\mathbb I}$ floor (see `I_EPSILON`) supplies it, guaranteeing
    $\lambda_{\min}(M)\ge\varepsilon_{\mathbb I}$ everywhere.
    """
    log_m:  jnp.ndarray            # (n,)      base value (GP adds a residual)
    L_body: jnp.ndarray            # (n, 3, 3) Cholesky of I_i
    ell0:   jnp.ndarray            # (3,)
    c:      jnp.ndarray            # (n, 3)
    # Posterior log-stds -- None unless `variational`, which keeps the pytree
    # identical to the pre-variational version so old checkpoints still load.
    log_m_ls:  Optional[jnp.ndarray]
    L_body_ls: Optional[jnp.ndarray]
    ell0_ls:   Optional[jnp.ndarray]
    c_ls:      Optional[jnp.ndarray]
    # `gp_core` / `nn_core` mode: a GP or an MLP emits all of the above as a
    # function of q. Mutually exclusive; both None on the constants-only path.
    gp: Optional[GP_NLink]
    nn: Optional[MLP]
    n: int = eqx.field(static=True)
    variational: bool = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)
    gp_core: bool = eqx.field(static=True)
    nn_core: bool = eqx.field(static=True)
    relative_inputs: bool = eqx.field(static=True)
    i_epsilon: float = eqx.field(static=True)
    # Width of the flat `ell0` block: 3*max(n-1, 1). See __init__ for why flat.
    _n_ell: int = eqx.field(static=True)
    # Gauge anchor: m_1 frozen at this value (None = unanchored). See physical().
    # NOTE: proved insufficient on its own -- kept as harmless; see anchor_trace.
    anchor_m1: Optional[float] = eqx.field(static=True)
    # Function-space gauge anchor: tr M(q) normalised to this value (None =
    # off). This is the one that actually kills the scale gauge -- see
    # mass_matrix() for why the parameter-space m_1 anchor could not.
    anchor_trace: Optional[float] = eqx.field(static=True)

    def __init__(self, key, n: int, init_scale: float = 0.3,
                 variational: bool = False, prior_std: float = 1.0,
                 gp_core: bool = False, nn_core: bool = False,
                 relative_inputs: bool = True,
                 hidden_dim: int = 32, i_epsilon: float = I_EPSILON,
                 anchor_m1: Optional[float] = None,
                 anchor_trace: Optional[float] = None,
                 dtype=jnp.float32):
        self.n = int(n)
        self.variational = bool(variational)
        self.prior_std = float(prior_std)
        self.gp_core = bool(gp_core)
        self.nn_core = _check_cores(gp_core, nn_core)
        self.relative_inputs = bool(relative_inputs)
        self.i_epsilon = float(i_epsilon)
        # <=0 is the CLI's "disabled" sentinel (masses are positive), so
        # normalise it to None here and every caller can pass the raw value.
        self.anchor_m1 = (None if anchor_m1 is None or anchor_m1 <= 0
                          else float(anchor_m1))
        self.anchor_trace = (None if anchor_trace is None or anchor_trace <= 0
                             else float(anchor_trace))
        k_gp, key = jax.random.split(key)
        # `ell0` holds the FREE joint offsets, flattened: ell_0 .. ell_{n-2}.
        # ell_{n-1} never enters the dynamics (the last link has no downstream
        # neighbour), so it is not a parameter.
        #
        # Flat, and sized 3*max(n-1,1) rather than (n-1, 3), on purpose: at
        # n <= 2 that is exactly the (3,) leaf this field has always had, so
        # every checkpoint written before n >= 3 was supported still
        # deserialises. Shaping it (n-1, 3) would give (1, 3) at n=2 and break
        # all of them for no gain.
        self._n_ell = 3 * max(n - 1, 1)
        # n (log_m) + 9n (L_body) + 3*(n-1) (ell) + 3n (c)
        n_core = n + 9 * n + self._n_ell + 3 * n
        self.gp = (_make_core_gp(k_gp, n, n_core,
                                 hidden_dim, self.relative_inputs, dtype)
                   if self.gp_core else None)
        # The MLP key is FOLDED from k_gp rather than split off `key`: consuming
        # another draw would shift every base parameter below and break the
        # bit-parity of `ArmPortHamiltonian(subnet_kind='structured')` against
        # ph_gp_sde (test_variants.py::test_matches_ph_gp_sde).
        self.nn = (_make_core_nn(jax.random.fold_in(k_gp, 1), n, n_core,
                                 hidden_dim, self.relative_inputs, dtype)
                   if self.nn_core else None)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.log_m = (init_scale * jax.random.normal(k1, (n,))).astype(dtype)
        self.L_body = (init_scale * jax.random.normal(k2, (n, 3, 3))).astype(dtype)
        self.ell0 = (init_scale
                     * jax.random.normal(k3, (self._n_ell,))).astype(dtype)
        self.c = (init_scale * jax.random.normal(k4, (n, 3))).astype(dtype)
        self.log_m_ls = _init_log_std((n,), dtype, self.variational)
        self.L_body_ls = _init_log_std((n, 3, 3), dtype, self.variational)
        self.ell0_ls = _init_log_std((self._n_ell,), dtype, self.variational)
        self.c_ls = _init_log_std((n, 3), dtype, self.variational)

    def physical(self, q=None, key=None, inference_mode=False):
        r"""$(m, \mathbb{I}, \ell, c)$ with the positivity constraints applied.

        Three sources, in order of precedence:
          * `gp_core` / `nn_core` -- a GP or MLP at `q` emits every parameter;
          * `variational`  -- reparameterised draws from per-constant posteriors;
          * otherwise      -- the stored point estimates.

        The constraints ($m>0$ via `exp`, $\mathbb{I}\succeq0$ via $LL^\top$,
        $\ell_{n-1}\equiv0$) are applied identically in all three.
        """
        n = self.n
        ks = (None, None, None, None) if key is None else tuple(
            jax.random.split(key, 4))
        d = lambda mu, ls, k: _draw(mu, ls, k, self.variational,
                                    inference_mode)
        log_m = d(self.log_m, self.log_m_ls, ks[0])
        L_raw = d(self.L_body, self.L_body_ls, ks[1])
        ell0 = d(self.ell0, self.ell0_ls, ks[2])
        c = d(self.c, self.c_ls, ks[3])

        # Bounded residual on top of the base, from whichever core is active.
        vec = _core_residual(self.gp, self.nn, q, n, self.relative_inputs,
                             key, inference_mode)
        if vec is not None:
            log_m = log_m + vec[:n]
            L_raw = L_raw + vec[n:n + 9 * n].reshape(n, 3, 3)
            o = n + 9 * n
            ell0 = ell0 + vec[o:o + self._n_ell]
            c = c + vec[o + self._n_ell:].reshape(n, 3)

        # ── Gauge anchor ─────────────────────────────────────────────
        # The scale direction (M, V, D, g, Sigma) -> beta*(...) with
        # p -> beta*p is *exactly* unidentifiable: the data determines only
        # ratios, never the absolute mass scale. Left free, the rollout NLL
        # ratchets the gauge (an unanchored run reached beta ~ 21, sending
        # Sigma chasing a 21x target it never caught). So the gauge is fixed
        # by convention: m_1 == anchor_m1, a constant. Overwriting the entry
        # AFTER base + posterior draw + gp_core residual kills, in one place,
        # every path that could re-open the gauge -- including a constant GP
        # offset on log m_1, which freezing only the base would miss. This
        # carries zero information about the data (any positive anchor is
        # gauge-equivalent); it only chooses the units the answer is
        # expressed in.
        if self.anchor_m1 is not None:
            log_m = log_m.at[0].set(
                jnp.log(jnp.asarray(self.anchor_m1, dtype=log_m.dtype)))

        m = jnp.exp(log_m)
        L = jnp.tril(L_raw)
        # $LL^\top\succeq0$ only; the floor makes it $\succ0$, hence M PD.
        I_body = (jnp.einsum('iab,icb->iac', L, L)
                  + self.i_epsilon * jnp.eye(3, dtype=L.dtype)[None])
        # ell_0 .. ell_{n-2} are free; ell_{n-1} is structurally zero. The old
        # code wrote [ell0, 0, ..., 0], which is right at n <= 2 but at n >= 3
        # forced ell_1 .. ell_{n-2} to zero as well -- silently deleting real
        # degrees of freedom (link 2 of a 3-link arm could not have a length).
        n_free = max(n - 1, 0)
        ell_free = ell0.reshape(-1, 3)[:n_free]
        ell = jnp.concatenate(
            [ell_free, jnp.zeros((n - n_free, 3), dtype=ell0.dtype)], axis=0)
        return m, I_body, ell, c

    def mass_matrix_raw(self, q, key=None, inference_mode=False):
        r"""$M(q)$ from the closed form, **before** the gauge anchor.

        Split out from :meth:`mass_matrix` so a shared gravity potential can ask
        for the anchor's scale factor separately (see :meth:`anchor_scale`):
        the anchor is a *gauge* move, and a gauge move must scale $V$, $D$, $g$
        and $\Sigma$ along with $M$ or it stops being one.
        """
        n = self.n
        R = q.reshape(n, 3, 3)
        m, I_body, ell, c = self.physical(q, key, inference_mode)

        rows = []
        for j in range(n):
            cols = []
            for k in range(n):
                B = I_body[j] if j == k else jnp.zeros((3, 3), dtype=q.dtype)
                for i in range(max(j, k), n):
                    u_ij = ell[j] if i > j else c[j]
                    u_ik = ell[k] if i > k else c[k]
                    B = B - m[i] * (_hat(u_ij) @ (R[j].T @ R[k]) @ _hat(u_ik))
                cols.append(B)
            rows.append(cols)
        return jnp.block(rows)

    def trace_raw(self, q=None, key=None, inference_mode=False):
        r"""$\operatorname{tr}M_{\rm raw}(q)$ in closed form, without assembling $M$.

        .. math::
            \operatorname{tr}M = \sum_j\Big[\operatorname{tr}\mathbb I_j
                + 2\sum_{i\ge j} m_i\lVert u_{ij}\rVert^2\Big]

        because the diagonal blocks have $R_j^\top R_j=I$, so
        $\operatorname{tr}\big([u]_\times^2\big)
        =\operatorname{tr}(uu^\top-\lVert u\rVert^2I)=-2\lVert u\rVert^2$.

        Two consequences worth noting: the trace carries **no** $q$-dependence
        while the constants are constants (which is what makes `anchor_trace` a
        pure gauge fix), and it costs $O(n^2)$ scalars instead of assembling a
        $3n\times3n$ matrix — so a shared potential can ask for the anchor scale
        on every drift evaluation without doubling the cost of $M$.
        """
        n = self.n
        m, I_body, ell, c = self.physical(q, key, inference_mode)
        total = jnp.trace(I_body, axis1=1, axis2=2).sum()
        for j in range(n):
            for i in range(j, n):
                u = ell[j] if i > j else c[j]
                total = total + 2.0 * m[i] * jnp.sum(u * u)
        return total

    def anchor_scale(self, q=None, key=None, inference_mode=False):
        r"""The factor $\alpha$ this module multiplies $M$ by, or exactly 1.

        $M=\alpha M_{\rm raw}$ with $\alpha=\texttt{anchor\_trace}/
        \operatorname{tr}M_{\rm raw}$. Anything sharing this module's physical
        constants must apply the same $\alpha$, since $(M,V,D,g,\Sigma)\to
        \beta(\cdot)$ is a gauge only when *all* of them move together.
        """
        if self.anchor_trace is None:
            dt = self.log_m.dtype
            return jnp.ones((), dtype=dt)
        return self.anchor_trace / self.trace_raw(q, key, inference_mode)

    def mass_matrix(self, q, key=None, inference_mode=False):
        r"""$M(q)\in\mathbb{R}^{3n\times3n}$ from the closed form, anchor applied."""
        M = self.mass_matrix_raw(q, key, inference_mode)

        # ── Function-space gauge anchor ──────────────────────────────
        # The scale gauge M -> beta*M has MANY parameter realisations: M is
        # quadratic in the levers, so  u -> sqrt(beta)*u, I -> beta*I  with
        # the masses FIXED scales M by beta exactly. That is how the gauge
        # escaped the m_1 anchor above (measured: beta ran to 23.6 with
        # m_1 pinned at 1). No finite set of frozen parameters closes every
        # such route, so the anchor must live where the gauge acts -- on the
        # function M(q) itself. Normalising the trace pins the scale of M
        # pointwise; a "gauge" move would then have to scale V, D, g, Sigma
        # WITHOUT M, which changes the observable dynamics and is therefore
        # identifiable, not flat. tr M is the natural readout: with constant
        # parameters it is exactly configuration-independent (the diagonal
        # blocks of M carry no q-dependence). Under gp_core this pins the
        # trace at every q -- slightly stronger than pure gauge fixing (it
        # asserts the OVERALL scale of M is constant in q), the same mild,
        # physically-certain assumption class as the m_1 anchor made.
        # Positive-definiteness is preserved (positive scalar multiple).
        if self.anchor_trace is not None:
            M = M * (self.anchor_trace / jnp.trace(M))
        return M

    def __call__(self, q, key=None, inference_mode=False):
        return jnp.linalg.inv(self.mass_matrix(q, key, inference_mode))

    def weight_kl_loss(self):
        if self.gp_core:
            return self.gp.weight_kl_loss()
        if not self.variational:
            return jnp.zeros((), dtype=self.log_m.dtype)
        return (_kl(self.log_m, self.log_m_ls, self.prior_std)
                + _kl(jnp.tril(self.L_body), jnp.tril(self.L_body_ls),
                      self.prior_std)
                + _kl(self.ell0, self.ell0_ls, self.prior_std)
                + _kl(self.c, self.c_ls, self.prior_std))


class StructuredDissipation(eqx.Module):
    r"""$D(q) = T(q)^\top\,\mathrm{blkdiag}(d_i I_3)\,T(q)$ — joint friction (§7).

    Friction opposes the **relative** joint rate $\Omega = T(q)\omega$, which is
    why $T$ appears rather than the identity. Since $\det T = 1$, $D\succ0$
    whenever every $d_i>0$: joint friction alone damps all $3n$ directions.

    Learned: $\log d_i$, i.e. $n$ numbers. Note this is a function of $q$ alone,
    so with `--varying_friction` the $\tfrac12\tanh\lVert\omega_i\rVert$ part of
    the true $\rho_i$ remains unrepresentable — exactly as for the GP it
    replaces. The $q$-dependent height term is absorbed into neither; that is a
    known gap, not a regression.
    """
    log_d: jnp.ndarray                # (n,)  base value
    log_d_ls: Optional[jnp.ndarray]   # (n,) posterior log-std, None if not variational
    gp: Optional[GP_NLink]
    nn: Optional[MLP]
    n: int = eqx.field(static=True)
    variational: bool = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)
    gp_core: bool = eqx.field(static=True)
    nn_core: bool = eqx.field(static=True)
    relative_inputs: bool = eqx.field(static=True)
    i_epsilon: float = eqx.field(static=True)

    def __init__(self, key, n: int, init_scale: float = 0.3,
                 variational: bool = False, prior_std: float = 1.0,
                 gp_core: bool = False, nn_core: bool = False,
                 relative_inputs: bool = True,
                 hidden_dim: int = 32, i_epsilon: float = I_EPSILON,
                 dtype=jnp.float32):
        self.n = int(n)
        self.variational = bool(variational)
        self.prior_std = float(prior_std)
        self.gp_core = bool(gp_core)
        self.nn_core = _check_cores(gp_core, nn_core)
        self.relative_inputs = bool(relative_inputs)
        self.i_epsilon = float(i_epsilon)
        k_gp, key = jax.random.split(key)
        self.gp = (_make_core_gp(k_gp, n, n, hidden_dim,
                                 self.relative_inputs, dtype)
                   if self.gp_core else None)
        # Folded, not split -- see the note in StructuredMass.__init__.
        self.nn = (_make_core_nn(jax.random.fold_in(k_gp, 1), n, n, hidden_dim,
                                 self.relative_inputs, dtype)
                   if self.nn_core else None)
        self.log_d = (init_scale * jax.random.normal(key, (n,))).astype(dtype)
        self.log_d_ls = _init_log_std((n,), dtype, self.variational)

    def physical(self, q=None, key=None, inference_mode=False):
        r"""$d_i>0$ — the per-joint viscous friction coefficients, `(n,)`.

        Same three sources and precedence as :meth:`StructuredMass.physical`.
        Exposed so a diagnostic can report the *constants* rather than
        re-deriving the $\exp(\log d+\kappa\tanh(\cdot))$ chain, which is how a
        readout silently drifts from the model it is meant to read.
        """
        log_d = _draw(self.log_d, self.log_d_ls, key,
                      self.variational, inference_mode)
        # d_i(q) = d_base + squash(core(q)): this is the mode that can represent
        # the *configuration* part of the varying-friction modulation rho_i,
        # which a constant d_i provably cannot. The target residual is <= ln 2,
        # well inside the kappa bound.
        res = _core_residual(self.gp, self.nn, q, self.n, self.relative_inputs,
                             key, inference_mode)
        if res is not None:
            log_d = log_d + res
        return jnp.exp(log_d)

    def __call__(self, q, key=None, inference_mode=False):
        n = self.n
        T = joint_rate_map(q.reshape(n, 3, 3), n)
        coeff = jnp.repeat(self.physical(q, key, inference_mode), 3)   # (3n,)
        return T.T @ (coeff[:, None] * T)

    def weight_kl_loss(self):
        if self.gp_core:
            return self.gp.weight_kl_loss()
        if not self.variational:
            return jnp.zeros((), dtype=self.log_d.dtype)
        return _kl(self.log_d, self.log_d_ls, self.prior_std)


class StructuredInputMap(eqx.Module):
    r"""$g(q) = T(q)^\top\,\mathrm{blkdiag}(\mathrm{diag}(\gamma_i))$ (§8).

    The kinematic factor $T(q)^\top$ is fully determined by the chain topology
    and carries **no** free parameters — $\big(T^\top u\big)_i
    = u_i - R_i^\top R_{i+1}u_{i+1}$, the second term being Newton's third law
    rather than a second control. The only unknown is the per-axis actuator
    gain $\gamma_i$, mirroring `g_diag` in the single-pendulum environment.

    Learned: $\gamma\in\mathbb{R}^{n\times3}$, i.e. $3n$ numbers ($6$ at $n=2$),
    against 12 672 GP weights for a subnet that was reaching only 0.465–0.565
    relative error while relearning kinematics already implied by the state.

    The gain is **not** constrained positive: a sign flip is physically
    meaningful (a motor wired backwards) and the data identifies it.
    """
    gain: jnp.ndarray                 # (n, 3)  base value
    gain_ls: Optional[jnp.ndarray]    # (n,3) posterior log-std, None if not variational
    gp: Optional[GP_NLink]
    nn: Optional[MLP]
    n: int = eqx.field(static=True)
    variational: bool = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)
    gp_core: bool = eqx.field(static=True)
    nn_core: bool = eqx.field(static=True)
    relative_inputs: bool = eqx.field(static=True)
    i_epsilon: float = eqx.field(static=True)

    def __init__(self, key, n: int, init_scale: float = 0.3,
                 variational: bool = False, prior_std: float = 1.0,
                 gp_core: bool = False, nn_core: bool = False,
                 relative_inputs: bool = True,
                 hidden_dim: int = 32, i_epsilon: float = I_EPSILON,
                 dtype=jnp.float32):
        self.n = int(n)
        self.variational = bool(variational)
        self.prior_std = float(prior_std)
        self.gp_core = bool(gp_core)
        self.nn_core = _check_cores(gp_core, nn_core)
        self.relative_inputs = bool(relative_inputs)
        self.i_epsilon = float(i_epsilon)
        k_gp, key = jax.random.split(key)
        self.gp = (_make_core_gp(k_gp, n, 3 * n, hidden_dim,
                                 self.relative_inputs, dtype)
                   if self.gp_core else None)
        # Folded, not split -- see the note in StructuredMass.__init__.
        self.nn = (_make_core_nn(jax.random.fold_in(k_gp, 1), n, 3 * n,
                                 hidden_dim, self.relative_inputs, dtype)
                   if self.nn_core else None)
        self.gain = (1.0 + init_scale
                     * jax.random.normal(key, (n, 3))).astype(dtype)
        self.gain_ls = _init_log_std((n, 3), dtype, self.variational)

    def physical(self, q=None, key=None, inference_mode=False):
        r"""$\gamma\in\mathbb{R}^{n\times3}$ — the per-axis actuator gains.

        Not constrained positive: a sign flip is physically meaningful (a motor
        wired backwards) and the data identifies it.
        """
        gain = _draw(self.gain, self.gain_ls, key,
                     self.variational, inference_mode)
        res = _core_residual(self.gp, self.nn, q, self.n, self.relative_inputs,
                             key, inference_mode)
        if res is not None:
            gain = gain + res.reshape(self.n, 3)
        return gain

    def __call__(self, q, key=None, inference_mode=False):
        n = self.n
        T = joint_rate_map(q.reshape(n, 3, 3), n)
        gain = self.physical(q, key, inference_mode)
        # Gain multiplies on the right: it scales input *channels* (columns).
        return T.T * gain.reshape(-1)[None, :]

    def weight_kl_loss(self):
        if self.gp_core:
            return self.gp.weight_kl_loss()
        if not self.variational:
            return jnp.zeros((), dtype=self.gain.dtype)
        # Prior is centred at 0, but the gain is initialised at 1 -- the KL is
        # therefore non-zero at init by design, and shrinks the gain unless the
        # likelihood pushes back.
        return _kl(self.gain, self.gain_ls, self.prior_std)


class StructuredSigma(eqx.Module):
    r"""Physics-structured diffusion: $\Sigma_j = [v_j]_\times R_j^\top$.

    The true wind lever map is

    .. math::
        \Sigma(q)_j=\Big(A_j^{>}[\ell_j]_\times+a_j[c_j]_\times\Big)R_j^\top
                   =\big[\underbrace{A_j^{>}\ell_j+a_jc_j}_{v_j}\big]_\times R_j^\top

    because $[\ell]_\times$ and $[c]_\times$ are skew and skew matrices are closed
    under addition. The geometry factor therefore carries **no** $q$-dependence:
    all of it is the single explicit $R_j^\top$, and the learnable content is one
    vector per link — $3n$ numbers (6 at $n=2$) against $6\,336$ GP weights,
    while still representing the truth exactly.

    Note $\|\Sigma\|_F^2=2\sum_j\|v_j\|^2$, so $v$ carries the wind amplitude
    too; there is no separate scale.

    .. note::
       Only $\Sigma\Sigma^\top$ is identifiable ($\Sigma\to\Sigma O$ leaves the
       law unchanged), and $O$ acts on the **shared** 3-D wind space — so a
       *global* sign flip of every $v_j$ is free, but flipping one link relative
       to another is a genuine error, not a gauge.
    """
    v: jnp.ndarray                      # (n, 3)  base value
    v_ls: Optional[jnp.ndarray]         # (n,3) posterior log-std, None if not variational
    gp: Optional[GP_NLink]
    nn: Optional[MLP]
    n: int = eqx.field(static=True)
    variational: bool = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)
    gp_core: bool = eqx.field(static=True)
    nn_core: bool = eqx.field(static=True)

    def __init__(self, key, n: int, init_scale: float = 0.2,
                 variational: bool = False, prior_std: float = 1.0,
                 gp_core: bool = False, nn_core: bool = False,
                 hidden_dim: int = 32, dtype=jnp.float32):
        self.n = int(n)
        self.variational = bool(variational)
        self.prior_std = float(prior_std)
        self.gp_core = bool(gp_core)
        self.nn_core = _check_cores(gp_core, nn_core)
        # Absolute attitudes: Sigma_j = [v_j]_x R_j^T is explicitly
        # frame-dependent, so there is no invariance to preserve here.
        k_gp, key = jax.random.split(key)
        self.gp = (_make_core_gp(k_gp, n, 3 * n, hidden_dim, False, dtype)
                   if self.gp_core else None)
        # Folded, not split -- see the note in StructuredMass.__init__.
        self.nn = (_make_core_nn(jax.random.fold_in(k_gp, 1), n, 3 * n,
                                 hidden_dim, False, dtype)
                   if self.nn_core else None)
        # Deliberately not seeded near ground truth: ||Sigma||_F starts around
        # 2*init_scale*sqrt(3) ~ 0.7, matching where the unstructured version
        # starts, so the comparison stays fair.
        self.v = (init_scale * jax.random.normal(key, (n, 3))).astype(dtype)
        self.v_ls = _init_log_std((n, 3), dtype, self.variational)

    def __call__(self, q, key=None, inference_mode=False):
        R = q.reshape(self.n, 3, 3)
        v = _draw(self.v, self.v_ls, key, self.variational, inference_mode)
        # relative_inputs=False: the wind direction is a world-frame object, so
        # the core sees absolute attitudes (matching V_net, not M/D/g).
        res = _core_residual(self.gp, self.nn, q, self.n, False,
                             key, inference_mode)
        if res is not None:
            v = v + res.reshape(self.n, 3)
        return jnp.einsum('jab,jcb->jac',
                          jax.vmap(_hat)(v), R).reshape(3 * self.n, 3)

    def weight_kl_loss(self):
        if self.gp_core:
            return self.gp.weight_kl_loss()
        if not self.variational:
            return jnp.zeros((), dtype=self.v.dtype)
        return _kl(self.v, self.v_ls, self.prior_std)


def gravity_levers(m, ell, c, g):
    r"""$w_j = g\big(\mu_j^{>}\ell_j + m_jc_j\big)$, `(n, 3)`.

    The **one** implementation of the gravity lever, shared by
    :class:`StructuredPotential` (own constants) and
    :class:`SharedStructuredPotential` (constants borrowed from
    :class:`StructuredMass`) — the load-bearing part is $\mu_j^{>}=\sum_{i>j}m_i$,
    and duplicating it is how one copy silently ends up with an *inclusive*
    suffix sum and a potential wrong by exactly one link's worth of gravity.
    """
    mu_downstream = _strict_suffix_sum(m)                          # (n,)
    return g * (mu_downstream[:, None] * ell + m[:, None] * c)


def potential_from_levers(q, n, w):
    r"""$V(q)=\sum_j[R_jw_j]_z$ as a shape-`(1,)` array."""
    R = q.reshape(n, 3, 3)
    return jnp.sum(jnp.einsum('jab,jb->ja', R, w)[:, 2])[None]


class StructuredPotential(eqx.Module):
    r"""Gravitational potential $V(q)$ from the closed form (§4.2).

    .. math::
        V(q) = g\sum_j e_z^\top R_j\big(\mu_j^{>}\ell_j + m_j c_j\big)
             = \sum_j\big[R_j w_j\big]_z,
        \qquad
        \mu_j^{>}=\sum_{i>j}m_i,\quad w_j := g\big(\mu_j^{>}\ell_j+m_jc_j\big)

    Read: link $j$ carries its own COM plus **all the mass hanging off its tip**,
    which is why $\mu_j^{>}$ appears and why $V$ is not a sum of independent
    per-link terms.

    Learned: $\log m_i$ ($n$), $\ell_0,\dots,\ell_{n-2}$ ($3(n{-}1)$;
    $\ell_{n-1}$ never enters — the last link has no downstream neighbour),
    $c_i$ ($3n$) and $\log g$ ($1$). At $n=2$ that is 12 numbers against 1 697
    MLP weights for the black-box `V_net` it replaces.

    .. note::
       $V$ is **not** rotation-invariant — gravity picks out the world $e_z$ —
       so unlike $M$, $D$ and $g(q)$ there is no relative-rotation form and the
       core net is fed **absolute** attitudes (`relative_inputs=False`), exactly
       as :class:`StructuredSigma` is.

    .. note::
       Only the $n$ lever vectors $w_j$ are identifiable: the map
       $(m,\ell,c,g)\mapsto w$ is many-to-one, so $\log m$, $\ell$, $c$, $\log g$
       recovered here are *not* the true physical constants even when $V(q)$ is
       reproduced to machine precision. This is the same base-parameter
       degeneracy documented for :class:`StructuredMass`; the redundant
       parameterisation is kept because it is the one the closed form is written
       in. Do not report these as identified physical quantities.

    .. warning::
       $\log g$ is initialised at $\approx0$ ($g\approx1$), i.e. ~10x below the
       true $9.81$, deliberately: seeding a known constant at its true value
       would make the structured potential start far closer to the truth than
       the MLP baseline it is compared against. The optimiser reaches $9.81$
       easily since the parameter is logarithmic.
    """
    log_m:    jnp.ndarray             # (n,)      base value
    ell_free: jnp.ndarray             # (n-1, 3)  ell_{n-1} == 0, never free
    c:        jnp.ndarray             # (n, 3)
    log_g:    jnp.ndarray             # ()        gravitational acceleration
    # Posterior log-stds -- None unless `variational` (same pytree trick as the
    # other subnets, so the non-variational leaves are exactly the base ones).
    log_m_ls:    Optional[jnp.ndarray]
    ell_free_ls: Optional[jnp.ndarray]
    c_ls:        Optional[jnp.ndarray]
    log_g_ls:    Optional[jnp.ndarray]
    gp: Optional[GP_NLink]
    nn: Optional[MLP]
    n: int = eqx.field(static=True)
    variational: bool = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)
    gp_core: bool = eqx.field(static=True)
    nn_core: bool = eqx.field(static=True)

    def __init__(self, key, n: int, init_scale: float = 0.3,
                 variational: bool = False, prior_std: float = 1.0,
                 gp_core: bool = False, nn_core: bool = False,
                 hidden_dim: int = 32, dtype=jnp.float32):
        self.n = int(n)
        self.variational = bool(variational)
        self.prior_std = float(prior_std)
        self.gp_core = bool(gp_core)
        self.nn_core = _check_cores(gp_core, nn_core)
        k_gp, key = jax.random.split(key)
        # n (log_m) + 3(n-1) (ell_free) + 3n (c) + 1 (log_g)
        n_core = n + 3 * (n - 1) + 3 * n + 1
        self.gp = (_make_core_gp(k_gp, n, n_core, hidden_dim, False, dtype)
                   if self.gp_core else None)
        self.nn = (_make_core_nn(jax.random.fold_in(k_gp, 1), n, n_core,
                                 hidden_dim, False, dtype)
                   if self.nn_core else None)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.log_m = (init_scale * jax.random.normal(k1, (n,))).astype(dtype)
        self.ell_free = (init_scale
                         * jax.random.normal(k2, (n - 1, 3))).astype(dtype)
        self.c = (init_scale * jax.random.normal(k3, (n, 3))).astype(dtype)
        self.log_g = (init_scale * jax.random.normal(k4, ())).astype(dtype)
        self.log_m_ls = _init_log_std((n,), dtype, self.variational)
        self.ell_free_ls = _init_log_std((n - 1, 3), dtype, self.variational)
        self.c_ls = _init_log_std((n, 3), dtype, self.variational)
        self.log_g_ls = _init_log_std((), dtype, self.variational)

    def physical(self, q=None, key=None, inference_mode=False):
        r"""$(m,\ell,c,g)$ with $m,g>0$ and $\ell_{n-1}\equiv0$ imposed.

        Same three sources and the same precedence as
        :meth:`StructuredMass.physical`: core net, then posterior draw, then the
        stored point estimates.
        """
        n = self.n
        ks = (None, None, None, None) if key is None else tuple(
            jax.random.split(key, 4))
        d = lambda mu, ls, k: _draw(mu, ls, k, self.variational, inference_mode)
        log_m = d(self.log_m, self.log_m_ls, ks[0])
        ell_free = d(self.ell_free, self.ell_free_ls, ks[1])
        c = d(self.c, self.c_ls, ks[2])
        log_g = d(self.log_g, self.log_g_ls, ks[3])

        # Bounded residual on top of the base -- absolute attitudes, see above.
        vec = _core_residual(self.gp, self.nn, q, n, False, key, inference_mode)
        if vec is not None:
            log_m = log_m + vec[:n]
            ell_free = ell_free + vec[n:n + 3 * (n - 1)].reshape(n - 1, 3)
            c = c + vec[n + 3 * (n - 1):n + 3 * (n - 1) + 3 * n].reshape(n, 3)
            log_g = log_g + vec[-1]

        ell = jnp.concatenate(
            [ell_free, jnp.zeros((1, 3), dtype=ell_free.dtype)], axis=0)
        return jnp.exp(log_m), ell, c, jnp.exp(log_g)

    def levers(self, q=None, key=None, inference_mode=False):
        r"""$w_j = g(\mu_j^{>}\ell_j + m_jc_j)$, `(n, 3)` — the identifiable part."""
        return gravity_levers(*self.physical(q, key, inference_mode))

    def __call__(self, q, key=None, inference_mode=False):
        r"""$V(q)$ as a **shape-(1,) array**, matching `GP_NLink`/`MLP` output.

        The drift indexes `V_net(q)[0]` (see
        ``ph_network_nlink.ArmPortHamiltonian.potential``), so a bare scalar
        would break the call site.
        """
        return potential_from_levers(q, self.n,
                                     self.levers(q, key, inference_mode))

    def weight_kl_loss(self):
        if self.gp_core:
            return self.gp.weight_kl_loss()
        if not self.variational:
            return jnp.zeros((), dtype=self.log_m.dtype)
        return (_kl(self.log_m, self.log_m_ls, self.prior_std)
                + _kl(self.ell_free, self.ell_free_ls, self.prior_std)
                + _kl(self.c, self.c_ls, self.prior_std)
                + _kl(self.log_g, self.log_g_ls, self.prior_std))


class SharedStructuredPotential(eqx.Module):
    r"""$V(q)$ built from :class:`StructuredMass`'s $(m,\ell,c)$ — only $g$ is learned.

    .. math::
        V(q)=\alpha\sum_j\big[R_j\,w_j\big]_z,\qquad
        w_j = g\big(\mu_j^{>}\ell_j+m_jc_j\big)

    where $(m,\ell,c)$ and $\alpha$ come from the mass module, so **one**
    parameter lives here: $\log g$.

    Why sharing is the physically right thing
    -----------------------------------------
    $m$, $\ell$ and $c$ are the *same physical quantities* in $M$ and in $V$.
    Estimating them twice does not merely waste parameters — it throws away a
    **mutual** identifiability gain, because the two functions constrain
    different things about them:

    * $M$ is **quadratic** in the levers,
      $M_{jk}=\mathbb I_j\delta_{jk}-\sum_i m_i[u_{ij}]_\times R_j^\top R_k[u_{ik}]_\times$,
      so a global flip $u\to-u$ leaves it **exactly invariant**: $M$ can never
      determine the sign of $\ell$ or $c$.
    * $V$ is **linear** in them, so the sign *is* identifiable from $V$.
    * Conversely $V$ only ever sees the combination $w_j$, so it cannot separate
      $m_i$ from $g$ or from $c_i$ — but $M$ can.

    Sharing therefore lets each pin down what the other structurally cannot.
    Measured on the unshared run: the two heads disagreed by 120% relative on
    $w_j$, and did not even agree on the sign of the dominant $z$-component.

    The $\alpha$ factor is not cosmetic
    -----------------------------------
    :class:`StructuredMass` may rescale $M$ by
    $\alpha=\texttt{anchor\_trace}/\operatorname{tr}M_{\rm raw}$. That is a
    **gauge** move, and the port-Hamiltonian gauge is
    $(M,V,D,g,\Sigma)\to\beta(\cdot)$ — all together. Borrowing $(m,\ell,c)$
    while ignoring $\alpha$ would scale $M$ without scaling $V$, which changes
    the observable dynamics rather than the units. With constant parameters
    $\alpha$ is constant and $\log g$ could absorb it; under a core net
    $\alpha=\alpha(q)$ and no constant can.

    .. note::
       There is deliberately **no core net** here. Gravity is a genuine physical
       constant, so a $q$-dependent $g$ is unphysical — and it would reintroduce
       exactly the redundancy sharing exists to remove (a core's output bias is
       perfectly redundant with $\log g$, one more flat ridge in the objective).

    Fixing $g$ (``gravity=9.81``) removes the lever-scale gauge
    ----------------------------------------------------------
    With $g$ **learned**, the lever *magnitude* is exactly unidentifiable. The
    flat direction is

    .. math::
        \ell\to\lambda\ell,\quad c\to\lambda c,\quad
        \mathbb I\to\lambda^2\mathbb I,\quad g\to\lambda g

    — note $g$ scales the **same** way as the levers, not inversely. Then
    $M_{\rm raw}\to\lambda^2M_{\rm raw}$ (both the $\mathbb I$ term and the
    quadratic translational term) and
    $V_{\rm raw}\to\lambda\cdot\lambda\,V_{\rm raw}=\lambda^2V_{\rm raw}$ (one
    factor from $g$, one from the levers), so the *same* anchor factor
    $\alpha\to\alpha/\lambda^2$ cancels **both**: $M$ and $V$ are simultaneously
    invariant while $D$ and $g(q)$ never involved these parameters at all.
    Verified numerically: at $\lambda=2$ the observables move by $0$ and
    $3\times10^{-16}$.

    Pinning $g$ closes it — the same $\lambda=2$ move then changes $V$ by 50%,
    so the lever magnitude becomes identifiable and the recovered $\ell$, $c$ are
    physically meaningful numbers rather than gauge-equivalent ones.

    This costs no generality. $g=9.81$ is *known*, not identified, and the
    overall pH scale remains free through the independent direction
    $m\to\mu m,\ \mathbb I\to\mu\mathbb I$ at fixed levers (also exactly flat;
    closed separately by ``anchor_m1``).

    .. note::
       $\varepsilon_{\mathbb I}$ breaks this gauge *weakly* on its own, since
       $\lambda^2LL^\top+\varepsilon\neq\lambda^2(LL^\top+\varepsilon)$ — at
       $\varepsilon=0.05$, $\lambda=2$ moves $M^{-1}$ by ~5%. That is a shallow
       curvature, not identification: the optimiser can still wander far along
       the valley. Fixing $g$ is what makes the direction sharply identified.

    When fixed, `log_g` is `None` — an empty pytree node — so this module holds
    **zero** learnable parameters and $V$ is determined entirely by the mass
    module's constants plus a physical constant.
    """
    log_g: Optional[jnp.ndarray]        # ()   None when gravity is fixed
    log_g_ls: Optional[jnp.ndarray]    # posterior log-std, None if not variational
    n: int = eqx.field(static=True)
    variational: bool = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)
    # Known value of g, or None to learn log_g. Static, so fixing it removes the
    # leaf entirely rather than merely freezing its gradient.
    gravity_fixed: Optional[float] = eqx.field(static=True)

    def __init__(self, key, n: int, init_scale: float = 0.3,
                 variational: bool = False, prior_std: float = 1.0,
                 gravity: Optional[float] = None, dtype=jnp.float32):
        self.n = int(n)
        self.variational = bool(variational)
        self.prior_std = float(prior_std)
        # <=0 is the CLI's "learn it" sentinel (g is positive), normalised here
        # so every caller can pass the raw flag value.
        self.gravity_fixed = (None if gravity is None or gravity <= 0
                              else float(gravity))
        if self.gravity_fixed is not None:
            self.log_g = None
            self.log_g_ls = None
        else:
            # Initialised at g ~ 1, ~10x below the true 9.81, for the same
            # reason StructuredPotential is: seeding a known constant at its
            # true value would start the structured model far closer to the
            # truth than the MLP baseline it is compared against. The parameter
            # is logarithmic, so the optimiser covers that decade easily.
            self.log_g = (init_scale * jax.random.normal(key, ())).astype(dtype)
            self.log_g_ls = _init_log_std((), dtype, self.variational)

    def gravity(self, key=None, inference_mode=False):
        r"""$g>0$ — the known constant, or $e^{\log g}$.

        The fixed branch returns a **Python float**, deliberately: JAX weak-types
        Python scalars, so `g * levers` keeps the levers' dtype. Wrapping it in
        `jnp.asarray` would make it float64 under `jax_enable_x64` and silently
        promote a float32 model.
        """
        if self.gravity_fixed is not None:
            return self.gravity_fixed
        return jnp.exp(_draw(self.log_g, self.log_g_ls, key,
                             self.variational, inference_mode))

    def levers(self, m, ell, c, key=None, inference_mode=False):
        r"""$w_j$ from **borrowed** $(m,\ell,c)$ and this module's $g$."""
        return gravity_levers(m, ell, c, self.gravity(key, inference_mode))

    def __call__(self, q, m, ell, c, alpha=1.0, key=None, inference_mode=False):
        r"""$V(q)$ as a shape-`(1,)` array.

        Note the signature deliberately does **not** match the
        ``(x, key, inference_mode)`` protocol of the other subnets: this module
        cannot be evaluated without someone else's physical constants, and a
        signature that pretended otherwise would let it be called through
        ``ArmPortHamiltonian._call`` and silently return garbage. It is reached
        only via :meth:`ArmPortHamiltonian.potential`.
        """
        w = self.levers(m, ell, c, key, inference_mode)
        return alpha * potential_from_levers(q, self.n, w)

    def weight_kl_loss(self):
        # A fixed g is not a random variable, so it contributes no KL — and
        # `log_g` is None, so the variational branch below would crash.
        if self.gravity_fixed is not None or not self.variational:
            return jnp.zeros(())
        return _kl(self.log_g, self.log_g_ls, self.prior_std)

r"""Correctness tests for the four $n$-link port-Hamiltonian variants.

Run:  ``python src/models/arm_n_link/test_variants.py``

The load-bearing test is #1: the shared :class:`ArmPortHamiltonian` with
``subnet_kind='gp', stochastic=True`` must reproduce the **existing, validated**
``ph_gp_sde`` network bit-for-bit. If it does, the shared core is not a rewrite
but the same model, and the three new variants inherit its correctness.

    1  unified structured-SDE == ph_gp_sde.DissipativeArmHamSDE   (exact)
    2  ODE variants have Sigma identically zero
    3  ODE rollout == SDE rollout driven with dW = 0
    4  every variant: shapes, symmetry, PSD
    5  NN variants have exactly zero KL
    6  pl_loss with Sigma = 0 reduces to the fixed-variance Gaussian
    7  anchor_trace pins the trace for every subnet kind
    8  the Sigma rollout detach zeroes that gradient path

Tests 9-12 cover the ``nn_core`` / ``structured_potential`` experiment — an MLP
predicting the physical **sub-components** which the closed forms then assemble,
as opposed to ``subnet_kind='nn'`` where the net emits every matrix entry:

    9   nn_core residual is exactly 0 at init (strict superset of constants)
    10  StructuredMass at the true constants == phys.mass_matrix   (exact)
    11  StructuredPotential at the true constants == phys.potential (exact)
    12  nn_core has zero KL and a live gradient into the core MLP
    13  shared V (--share_mass_potential) == phys.potential exactly
    14  shared V_net holds exactly one parameter (log g)
    15  the anchor_trace gauge scales V and M together
    16  gravity=9.81 leaves V_net with zero parameters, still exact
    17  StructuredMass is exact for n=1..4 (free-lever count / ell bug)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update('jax_enable_x64', True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
for _p in (_HERE, os.path.join(_HERE, 'utils'),
           os.path.join(_HERE, 'ph_gp_sde'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ph_network_nlink import ArmPortHamiltonian, KeyedArmModel   # noqa: E402
from lie_integrator_nlink import (                               # noqa: E402
    lie_heun_sde_rollout_nlink, lie_heun_ode_rollout_nlink, exp_so3_batch,
)
from elbo_loss_nlink import pl_loss_nlink                        # noqa: E402
from structured_subnets import (                                 # noqa: E402
    StructuredMass, StructuredPotential,
)
from envs.arm_nlink_so3 import arm_nlink_physics as phys         # noqa: E402

N, HID, DT = 2, 8, jnp.float64
D = 3 * N


def _state(seed=0):
    kR, kw = jax.random.split(jax.random.PRNGKey(seed))
    R = exp_so3_batch(jax.random.normal(kR, (N, 3), dtype=DT))
    return R.reshape(-1), 0.6 * jax.random.normal(kw, (D,), dtype=DT)


def test_matches_ph_gp_sde(tol=0.0):
    """The shared core must be the *same model* as the validated ph_gp_sde."""
    from network import DissipativeArmHamSDE          # ph_gp_sde/network.py

    key = jax.random.PRNGKey(42)
    q, w = _state(1)
    p = jnp.ones(D, dtype=DT)
    u = 0.3 * jnp.ones(D, dtype=DT)
    dW = jnp.array([0.1, -0.2, 0.3], dtype=DT)

    worst = 0.0
    # Parity must hold with anchors active AND in gp_core mode — ph_gp_ode's
    # default is now structured(-minus-diffusion), trained with the same flags
    # as the ph_gp_sde run it is compared against.
    for gp_core in (False, True):
        kw = dict(n=N, hidden_dim=HID, relative_inputs=True,
                  init_sigma_obs_omega=0.1, anchor_m1=1.0, anchor_trace=8.0,
                  gp_core=gp_core, dtype=DT)
        ref = DissipativeArmHamSDE(key=key, **kw)
        new = ArmPortHamiltonian(key=key, subnet_kind='structured',
                                 stochastic=True, **kw)
        diffs = {
            'M_inv':    jnp.max(jnp.abs(ref.M_inv(q) - new.M_inv(q))),
            'Sigma':    jnp.max(jnp.abs(ref.Sigma(q) - new.Sigma(q))),
            'drift_p':  jnp.max(jnp.abs(ref.drift_p(q, p, u)
                                        - new.drift_p(q, p, u))),
            'drift':    jnp.max(jnp.abs(ref.drift(q, w, u)
                                        - new.drift(q, w, u))),
            'dp_stoch': jnp.max(jnp.abs(ref.stochastic_increment_p(q, dW)
                                        - new.stochastic_increment_p(q, dW))),
            'kl':       jnp.abs(ref.kl_loss() - new.kl_loss()),
        }
        worst = max(worst, float(max(diffs.values())))
    return ('unified structured-SDE == ph_gp_sde', worst <= tol,
            f'max diff {worst:.2e}')


def test_ode_sigma_zero():
    """ODE variants must have Sigma exactly zero — that is what makes the
    per-increment loss collapse to a fixed-variance Gaussian with no branch."""
    q, _ = _state(2)
    worst = 0.0
    for kind in ('gp', 'nn'):
        m = ArmPortHamiltonian(key=jax.random.PRNGKey(0), n=N, subnet_kind=kind,
                               stochastic=False, hidden_dim=HID, dtype=DT)
        worst = max(worst, float(jnp.max(jnp.abs(m.Sigma(q)))),
                    float(jnp.max(jnp.abs(m.stochastic_increment_p(
                        q, jnp.ones(3, dtype=DT))))))
    return ('ODE variants: Sigma == 0', worst == 0.0, f'max |Sigma| {worst:.1e}')


def test_ode_equals_sde_zero_noise(tol=1e-12):
    """The two integrators are geometrically identical; the ODE path just never
    touches `stochastic_increment_p`. Driving the SDE with dW=0 must match."""
    m = ArmPortHamiltonian(key=jax.random.PRNGKey(3), n=N, subnet_kind='nn',
                           stochastic=True, hidden_dim=HID, dtype=DT)
    keyed = KeyedArmModel(model=m, keys={}, inference_mode=True)
    q, w = _state(4)
    x0 = jnp.concatenate([q, w])
    T, NS, h = 6, 10, 0.005
    u = 0.2 * jnp.ones((T, D), dtype=DT)

    a = lie_heun_sde_rollout_nlink(keyed, x0, u, h,
                                   jnp.zeros((T, NS, 3), dtype=DT))
    b = lie_heun_ode_rollout_nlink(keyed, x0, u, h, NS, T)
    e = float(jnp.max(jnp.abs(a - b)))
    return ('ODE rollout == SDE rollout @ dW=0', e < tol, f'max diff {e:.2e}')


def test_shapes_all_variants():
    """Shapes, symmetry and positive-definiteness across all four variants."""
    q, w = _state(5)
    p = jnp.ones(D, dtype=DT)
    u = jnp.zeros(D, dtype=DT)
    bad, worst_sym, min_eig = [], 0.0, np.inf
    for kind in ('gp', 'nn'):
        for stoch in (True, False):
            m = ArmPortHamiltonian(key=jax.random.PRNGKey(7), n=N,
                                   subnet_kind=kind, stochastic=stoch,
                                   hidden_dim=HID, dtype=DT)
            tag = f'{kind}_{"sde" if stoch else "ode"}'
            checks = {'M_inv': (m.M_inv(q).shape, (D, D)),
                      'Sigma': (m.Sigma(q).shape, (D, 3)),
                      'drift_p': (m.drift_p(q, p, u).shape, (D,)),
                      'drift': (m.drift(q, w, u).shape, (D,))}
            bad += [f'{tag}.{k}' for k, (g, e) in checks.items() if tuple(g) != e]
            M = m.M_inv(q)
            worst_sym = max(worst_sym, float(jnp.max(jnp.abs(M - M.T))))
            min_eig = min(min_eig, float(jnp.min(jnp.linalg.eigvalsh(M))))
    ok = (not bad) and worst_sym < 1e-10 and min_eig > 0
    return ('all variants: shapes + PSD', ok,
            f'sym {worst_sym:.1e} eig {min_eig:.3f}' if not bad else str(bad[:3]))


def test_nn_kl_zero():
    """NN variants carry no weight posterior, so the KL must be exactly 0 —
    absent by construction rather than switched off by --beta_max."""
    vals = [float(ArmPortHamiltonian(key=jax.random.PRNGKey(0), n=N,
                                     subnet_kind='nn', stochastic=s,
                                     hidden_dim=HID, dtype=DT).kl_loss())
            for s in (True, False)]
    return ('NN variants: KL == 0', all(v == 0.0 for v in vals), f'{vals}')


def test_pl_reduces_to_gaussian(tol=1e-10):
    r"""With $\Sigma=0$ the Woodbury pseudo-likelihood must collapse to

    .. math:: \tfrac12\|r\|^2/s + \tfrac{3n}{2}\log s,\qquad s=2\sigma_{\rm obs}^2

    i.e. a plain fixed-variance Gaussian on $\Delta\omega$. This is what lets the
    ODE variants share the loss code with no special case.
    """
    m = ArmPortHamiltonian(key=jax.random.PRNGKey(11), n=N, subnet_kind='nn',
                           stochastic=False, hidden_dim=HID,
                           init_sigma_obs_omega=0.1, dtype=DT)
    T, B, dt = 3, 4, 0.05
    R = exp_so3_batch(jax.random.normal(jax.random.PRNGKey(12), (T * B * N, 3),
                                        dtype=DT)).reshape(T, B, 9 * N)
    om = 0.3 * jax.random.normal(jax.random.PRNGKey(13), (T, B, D), dtype=DT)
    uu = 0.1 * jnp.ones((T, B, D), dtype=DT)
    batch = jnp.concatenate([R, om, uu], axis=-1)
    keys = {k: jax.random.split(jax.random.PRNGKey(i), B).reshape(B, 1, 2)
            for i, k in enumerate(('M', 'V', 'Dw', 'g', 'Sigma'))}

    got = float(pl_loss_nlink(m, batch, dt, m.sigma_obs_omega, keys, N,
                              inference_mode=True)['pl_loss'])

    s = 2 * 0.1 ** 2
    mu = jax.vmap(jax.vmap(lambda q, o, uu_: m.drift(q, o, uu_)))(
        batch[:-1, :, :9 * N], batch[:-1, :, 9 * N:12 * N],
        batch[1:, :, 12 * N:15 * N])
    r = (batch[1:, :, 9 * N:12 * N] - batch[:-1, :, 9 * N:12 * N]) - mu * dt
    want = float(jnp.mean(0.5 * jnp.sum(r * r, -1) / s + 1.5 * N * jnp.log(s)))
    e = abs(got - want) / max(abs(want), 1.0)
    return ('pl(Sigma=0) == fixed-var Gaussian', e < tol, f'rel {e:.2e}')


def test_anchor_trace_all_kinds(tol=1e-8):
    """The function-space gauge anchor must pin the trace for every kind:
    tr M == anchor for 'structured' (inside StructuredMass), tr M^-1 == anchor
    for 'gp'/'nn' (at the M_inv wrapper). Either closes the same scale gauge;
    parameter-space pins provably do not (gp_change.md §3c)."""
    q, _ = _state(8)
    worst = 0.0
    for kind in ('structured', 'gp', 'nn'):
        m = ArmPortHamiltonian(key=jax.random.PRNGKey(1), n=N,
                               subnet_kind=kind, stochastic=True,
                               hidden_dim=HID, anchor_trace=8.0, dtype=DT)
        Minv = m.M_inv(q)
        tr = (jnp.trace(jnp.linalg.inv(Minv)) if kind == 'structured'
              else jnp.trace(Minv))
        worst = max(worst, float(jnp.abs(tr - 8.0)))
    return ('anchor_trace pins tr for every kind', worst < tol,
            f'max |err| {worst:.1e}')


def test_sigma_detach_grad():
    """The detach must zero the rollout-path gradient into Sigma (and only
    exist as a gradient change — values are compared in test #1)."""
    q, _ = _state(9)
    dW = jnp.array([0.1, -0.2, 0.3], dtype=DT)

    def g_of(detach):
        m = ArmPortHamiltonian(key=jax.random.PRNGKey(2), n=N,
                               subnet_kind='nn', stochastic=True,
                               hidden_dim=HID, sigma_detach_rollout=detach,
                               dtype=DT)
        grads = eqx.filter_grad(
            lambda mm: jnp.sum(mm.stochastic_increment_p(q, dW) ** 2))(m)
        leaves = jax.tree_util.tree_leaves(
            eqx.filter(grads.Sigma_net, eqx.is_array))
        return max(float(jnp.max(jnp.abs(g))) for g in leaves)

    on, off = g_of(True), g_of(False)
    return ('sigma detach cuts rollout grad', on == 0.0 and off > 0.0,
            f'on {on:.1e} off {off:.1e}')


# ══════════════════════════════════════════════════════════════════════
# nn_core / structured_potential — the "NN predicts sub-components" variant
# ══════════════════════════════════════════════════════════════════════

def _structured(nn_core, key=jax.random.PRNGKey(31), **kw):
    return ArmPortHamiltonian(key=key, n=N, subnet_kind='structured',
                              stochastic=True, hidden_dim=HID,
                              relative_inputs=True, nn_core=nn_core,
                              structured_potential=True, anchor_m1=1.0,
                              anchor_trace=4.0 * N, dtype=DT, **kw)


def test_nn_core_zero_at_init(tol=0.0):
    r"""``--nn_core`` must be a **strict superset** of the constants-only model.

    The core MLP's read-out layer is zero-initialised, so
    $\theta(q)=\theta_{\rm base}+\kappa\tanh(0/\kappa)=\theta_{\rm base}$ at every
    $q$ — `nn_core=True` and `nn_core=False` must therefore be *bit-identical*
    at step 0. Two things break if this regresses: the model would start at
    random $(\mathbb I,\ell,c)$ per configuration (an $\mathbb I$ that
    degenerates sends $M^{-1}$ to NaN on the first step), and `--nn_core` would
    stop being a clean ablation of the constants-only model.

    It also pins the key discipline: the MLP key is `fold_in`-ed from the GP key
    rather than split off the chain, so enabling the core cannot shift the base
    parameters (the same requirement that keeps test #1 bit-exact).
    """
    q, w = _state(6)
    p = jnp.ones(D, dtype=DT)
    u = 0.3 * jnp.ones(D, dtype=DT)
    base, core = _structured(False), _structured(True)
    diffs = {
        'M_inv':   jnp.max(jnp.abs(base.M_inv(q) - core.M_inv(q))),
        'V':       jnp.abs(base._call(base.V_net, q, None)[0]
                           - core._call(core.V_net, q, None)[0]),
        'D':       jnp.max(jnp.abs(base._call(base.Dw_net, q, None)
                                   - core._call(core.Dw_net, q, None))),
        'g':       jnp.max(jnp.abs(base._call(base.g_net, q, None)
                                   - core._call(core.g_net, q, None))),
        'Sigma':   jnp.max(jnp.abs(base.Sigma(q) - core.Sigma(q))),
        'drift_p': jnp.max(jnp.abs(base.drift_p(q, p, u)
                                   - core.drift_p(q, p, u))),
        'drift':   jnp.max(jnp.abs(base.drift(q, w, u) - core.drift(q, w, u))),
    }
    worst = float(max(diffs.values()))
    return ('nn_core residual == 0 at init', worst <= tol,
            f'max diff {worst:.2e}')


def _true_params():
    """The environment's defaults at $n=2$: $m=1$, $\\ell=c=e_z$, $g=9.81$."""
    return phys.uniform_chain_params(N, d=0.5, dtype=DT)


def test_structured_mass_exact(tol=1e-13):
    r"""Seeded with the true constants, :class:`StructuredMass` must reproduce
    ``phys.mass_matrix`` to machine precision.

    This is the gate that says the closed form in `structured_subnets.py` *is*
    the environment's algebra — the whole premise of the sub-component
    experiment. The two implementations are deliberately independent (the module
    never imports the env), so agreement is evidence, not a tautology.

    Note `i_epsilon=0` here: the true $\mathbb{I}=\mathrm{diag}(0,0,mL^2)$ is
    rank 1, so the conditioning floor $\mathbb{I}=LL^\top+\varepsilon I$ can
    never represent it exactly. That floor is the accuracy ceiling of the
    structured $M$ in training (default $\varepsilon=0.05$), not a bug — the
    test switches it off to check the *formula*.
    """
    params = _true_params()
    sm = StructuredMass(jax.random.PRNGKey(0), n=N, i_epsilon=0.0,
                        relative_inputs=True, dtype=DT)
    # The true I_i is diagonal, so its Cholesky factor is the elementwise sqrt
    # of the diagonal. `jnp.linalg.cholesky` cannot be used: I_i = diag(0,0,mL^2)
    # is singular and would come back NaN. `physical()` applies jnp.tril, which
    # leaves a diagonal L untouched.
    L_true = jax.vmap(lambda I: jnp.diag(jnp.sqrt(jnp.diagonal(I))))(params.I_body)
    sm = eqx.tree_at(lambda m: (m.log_m, m.L_body, m.ell0, m.c), sm,
                     (jnp.log(params.m), L_true, params.ell[0], params.c))

    worst = 0.0
    for seed in (40, 41, 42):
        R = exp_so3_batch(jax.random.normal(jax.random.PRNGKey(seed),
                                            (N, 3), dtype=DT))
        ref = phys.mass_matrix(params, R)
        got = sm.mass_matrix(R.reshape(-1))
        worst = max(worst, float(jnp.max(jnp.abs(ref - got))
                                 / jnp.max(jnp.abs(ref))))
    return ('StructuredMass == phys.mass_matrix', worst < tol,
            f'max rel {worst:.2e}')


def test_structured_potential_exact(tol=1e-14):
    r"""Seeded with the true constants, :class:`StructuredPotential` must
    reproduce ``phys.potential`` to machine precision:

    .. math::
        V(q)=g\sum_j e_z^\top R_j\big(\mu_j^{>}\ell_j+m_jc_j\big),
        \qquad \mu_j^{>}=\sum_{i>j}m_i

    The load-bearing part is $\mu_j^{>}$ — link $j$ carries every downstream
    mass, so dropping it (or using an inclusive suffix sum) leaves a potential
    that is wrong by exactly one link's worth of gravity and still looks
    plausible. It also checks the shape contract: `drift_p` indexes
    ``V_net(q)[0]``, so the return must be `(1,)`, not a scalar.
    """
    params = _true_params()
    sp = StructuredPotential(jax.random.PRNGKey(0), n=N, dtype=DT)
    sp = eqx.tree_at(lambda m: (m.log_m, m.ell_free, m.c, m.log_g), sp,
                     (jnp.log(params.m), params.ell[:-1], params.c,
                      jnp.log(params.g)))

    worst = 0.0
    for seed in (50, 51, 52):
        R = exp_so3_batch(jax.random.normal(jax.random.PRNGKey(seed),
                                            (N, 3), dtype=DT))
        out = sp(R.reshape(-1))
        assert out.shape == (1,), f'V_net must return (1,), got {out.shape}'
        ref = phys.potential(params, R)
        worst = max(worst, float(jnp.abs(ref - out[0])
                                 / jnp.maximum(jnp.abs(ref), 1.0)))
    return ('StructuredPotential == phys.potential', worst < tol,
            f'max rel {worst:.2e}')


def test_nn_core_kl_and_grad():
    r"""Every `nn_core` subnet must carry **no** KL (an MLP is a point estimate,
    so ``--beta_max`` is inert) while still being *wired in* — a live gradient
    into the core MLP.

    Zero KL alone would also be satisfied by a core that is silently ignored, so
    the gradient half is what actually proves the residual reaches $M^{-1}$. At
    step 0 only the zero-initialised **read-out** layer has a non-zero gradient
    (the earlier layers' path is multiplied by that zero weight); the hidden
    layers become live as soon as it moves off zero.

    The four drift subnets are summed rather than `kl_loss()`: with
    ``sigma_mode='full'`` the diffusion subnet is a `MatrixGP_NLink` whose KL is
    (correctly) non-zero, which would mask the thing being tested. The ODE
    experiment has ``Sigma_net is None`` anyway.
    """
    q, _ = _state(7)
    core = _structured(True)
    kl = float(sum(getattr(core, f'{s}_net').weight_kl_loss()
                   for s in ('M', 'V', 'Dw', 'g')))

    grads = eqx.filter_grad(lambda m: jnp.sum(m.M_inv(q) ** 2))(core)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads.M_net.nn, eqx.is_array))
    live = max(float(jnp.max(jnp.abs(g))) for g in leaves)
    return ('nn_core: KL == 0, core grad live', kl == 0.0 and live > 0.0,
            f'kl {kl:.1e} grad {live:.1e}')


def _seed_mass_truth(model, params, dtype=DT):
    """Overwrite M_net's base constants with ground truth (i_epsilon aware)."""
    ie = model.M_net.i_epsilon
    Ld = jnp.sqrt(jnp.maximum(
        jnp.diagonal(params.I_body, axis1=1, axis2=2) - ie, 0.0))
    return eqx.tree_at(
        lambda t: (t.M_net.log_m, t.M_net.L_body, t.M_net.ell0, t.M_net.c),
        model, (jnp.log(params.m), jax.vmap(jnp.diag)(Ld),
                params.ell[0], params.c))


def test_shared_potential_exact(tol=1e-13):
    r"""With ``share_mass_potential``, seeding **M_net** with the true constants
    and $\log g=\log 9.81$ must reproduce ``phys.potential`` exactly.

    This is the whole claim of the flag: $V$ is computed from $M$'s $(m,\ell,c)$,
    so setting them once has to make *both* right. If V_net secretly kept its own
    copy, this reads the untouched random init and fails immediately.

    Anchor off and $\varepsilon_{\mathbb I}=0$ so the test measures the sharing
    wiring, not the conditioning floor (covered by test #10).
    """
    params = _true_params()
    m = ArmPortHamiltonian(key=jax.random.PRNGKey(5), n=N,
                           subnet_kind='structured', stochastic=False,
                           hidden_dim=HID, structured_potential=True,
                           share_mass_potential=True, i_epsilon=0.0,
                           anchor_m1=None, anchor_trace=None, dtype=DT)
    m = _seed_mass_truth(m, params)
    m = eqx.tree_at(lambda t: t.V_net.log_g, m,
                    jnp.asarray(jnp.log(params.g), dtype=DT))

    worst = 0.0
    for seed in (60, 61, 62):
        R = exp_so3_batch(jax.random.normal(jax.random.PRNGKey(seed),
                                            (N, 3), dtype=DT))
        got = m.potential(R.reshape(-1))
        ref = phys.potential(params, R)
        worst = max(worst, float(jnp.abs(ref - got)
                                 / jnp.maximum(jnp.abs(ref), 1.0)))
    return ('shared V == phys.potential', worst < tol, f'max rel {worst:.2e}')


def test_shared_potential_is_one_param():
    """V_net must hold exactly one learnable scalar (log g) when shared.

    The point of sharing is that (m, ell, c) exist ONCE. If V_net still carried
    them the flag would be cosmetic, so count the leaves rather than trust it.
    """
    def build(shared):
        return ArmPortHamiltonian(
            key=jax.random.PRNGKey(6), n=N, subnet_kind='structured',
            stochastic=False, hidden_dim=HID, nn_core=True,
            structured_potential=True, share_mass_potential=shared, dtype=DT)
    n_un = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(build(False).V_net, eqx.is_array)))
    n_sh = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(build(True).V_net, eqx.is_array)))
    return ('shared V_net holds only log g', n_sh == 1 and n_un > 1,
            f'{n_un} -> {n_sh} params')


def test_structured_mass_general_n(tol=1e-13):
    r"""``StructuredMass`` must be exact for **every** $n$, not just $n=2$.

    Guards the free-lever count. `ell0` holds $\ell_0,\dots,\ell_{n-2}$ flattened
    and only $\ell_{n-1}$ is structurally zero. The old code wrote
    ``[ell0, 0, ..., 0]``, which is right at $n\le2$ but at $n\ge3$ also forced
    $\ell_1,\dots,\ell_{n-2}$ to zero — silently deleting real degrees of freedom
    (link 2 of a 3-link arm could not have a length). Nothing at $n=2$ can catch
    that, which is why this loops over $n$.

    Also pins the leaf width at $3\max(n{-}1,1)$: at $n\le2$ that is the `(3,)`
    leaf the field has always had, so checkpoints written before $n\ge3$ was
    supported still deserialise.
    """
    worst, shapes = 0.0, []
    for n in (1, 2, 3, 4):
        params = phys.uniform_chain_params(n, d=0.5, dtype=DT)
        sm = StructuredMass(jax.random.PRNGKey(0), n=n, i_epsilon=0.0, dtype=DT)
        shapes.append((n, tuple(sm.ell0.shape)))
        Ld = jnp.sqrt(jnp.diagonal(params.I_body, axis1=1, axis2=2))
        ell_flat = (params.ell[:n - 1].reshape(-1) if n > 1
                    else jnp.zeros(3, dtype=DT))
        sm = eqx.tree_at(lambda m: (m.log_m, m.L_body, m.ell0, m.c), sm,
                         (jnp.log(params.m), jax.vmap(jnp.diag)(Ld), ell_flat,
                          params.c))
        for seed in (40, 41):
            R = exp_so3_batch(jax.random.normal(jax.random.PRNGKey(seed),
                                                (n, 3), dtype=DT))
            ref = phys.mass_matrix(params, R)
            got = sm.mass_matrix(R.reshape(-1))
            worst = max(worst, float(jnp.max(jnp.abs(ref - got))
                                     / jnp.max(jnp.abs(ref))))
    want = [(1, (3,)), (2, (3,)), (3, (6,)), (4, (9,))]
    return ('StructuredMass exact for n=1..4', worst < tol and shapes == want,
            f'max rel {worst:.1e}, shapes ok={shapes == want}')


def test_fixed_gravity(tol=1e-13):
    r"""``gravity=9.81`` must remove the leaf entirely *and* stay exact.

    Two claims in one:

    * `V_net` holds **zero** learnable parameters — `log_g` becomes `None`, an
      empty pytree node, so gravity is genuinely a constant of the model rather
      than a parameter whose gradient happens to be masked.
    * seeding only `M_net` with the true constants still reproduces
      `phys.potential` exactly, i.e. the fixed value is actually used.
    """
    params = _true_params()
    m = ArmPortHamiltonian(key=jax.random.PRNGKey(8), n=N,
                           subnet_kind='structured', stochastic=False,
                           hidden_dim=HID, structured_potential=True,
                           share_mass_potential=True, gravity=float(params.g),
                           i_epsilon=0.0, anchor_m1=None, anchor_trace=None,
                           dtype=DT)
    n_v = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(m.V_net, eqx.is_array)))
    m = _seed_mass_truth(m, params)

    worst = 0.0
    for seed in (80, 81, 82):
        R = exp_so3_batch(jax.random.normal(jax.random.PRNGKey(seed),
                                            (N, 3), dtype=DT))
        got, ref = m.potential(R.reshape(-1)), phys.potential(params, R)
        worst = max(worst, float(jnp.abs(ref - got)
                                 / jnp.maximum(jnp.abs(ref), 1.0)))
    return ('fixed g: 0 params in V_net, exact', n_v == 0 and worst < tol,
            f'{n_v} params, max rel {worst:.2e}')


def test_shared_potential_gauge(tol=1e-11):
    r"""The anchor must scale $V$ and $M$ **together**.

    ``anchor_trace`` multiplies $M$ by $\alpha=\texttt{anchor}/\operatorname{tr}
    M_{\rm raw}$. That is a gauge move only if $V$ moves too — the pH gauge is
    $(M,V,D,g,\Sigma)\to\beta(\cdot)$ jointly. Borrowing $(m,\ell,c)$ and
    forgetting $\alpha$ would scale $M$ alone, which changes the observable
    dynamics rather than the units.

    Checks (a) `anchor_scale` agrees with the trace of the assembled raw matrix —
    i.e. the closed-form `trace_raw` shortcut is right — and (b) V picks the
    factor up: $V_{\rm anchored}/V_{\rm raw}=\alpha$.
    """
    params = _true_params()
    kw = dict(key=jax.random.PRNGKey(7), n=N, subnet_kind='structured',
              stochastic=False, hidden_dim=HID, structured_potential=True,
              share_mass_potential=True, nn_core=True, i_epsilon=0.05,
              anchor_m1=None, dtype=DT)
    m_raw = _seed_mass_truth(ArmPortHamiltonian(anchor_trace=None, **kw), params)
    m_anc = _seed_mass_truth(ArmPortHamiltonian(anchor_trace=4.0 * N, **kw), params)

    worst_tr, worst_v = 0.0, 0.0
    for seed in (70, 71, 72):
        q = exp_so3_batch(jax.random.normal(jax.random.PRNGKey(seed),
                                            (N, 3), dtype=DT)).reshape(-1)
        # (a) closed-form trace vs the assembled matrix
        tr_closed = m_anc.M_net.trace_raw(q, None, True)
        tr_direct = jnp.trace(m_anc.M_net.mass_matrix_raw(q, None, True))
        worst_tr = max(worst_tr, float(jnp.abs(tr_closed - tr_direct)
                                       / jnp.abs(tr_direct)))
        # (b) V scales by exactly alpha
        alpha = float(m_anc.M_net.anchor_scale(q, None, True))
        ratio = float(m_anc.potential(q) / m_raw.potential(q))
        worst_v = max(worst_v, abs(ratio - alpha) / max(abs(alpha), 1e-12))
    ok = worst_tr < tol and worst_v < tol
    return ('shared V follows the anchor gauge', ok,
            f'tr {worst_tr:.1e} V/alpha {worst_v:.1e}')


if __name__ == '__main__':
    results = [test_matches_ph_gp_sde(), test_ode_sigma_zero(),
               test_ode_equals_sde_zero_noise(), test_shapes_all_variants(),
               test_nn_kl_zero(), test_pl_reduces_to_gaussian(),
               test_anchor_trace_all_kinds(), test_sigma_detach_grad(),
               test_nn_core_zero_at_init(), test_structured_mass_exact(),
               test_structured_potential_exact(), test_nn_core_kl_and_grad(),
               test_shared_potential_exact(), test_shared_potential_is_one_param(),
               test_shared_potential_gauge(), test_fixed_gravity(),
               test_structured_mass_general_n()]
    w = max(len(r[0]) for r in results) + 2
    print()
    print('=' * (w + 34))
    print(f'{"test":<{w}}{"status":<8}{"measure":>24}')
    print('=' * (w + 34))
    for name, ok, detail in results:
        print(f'{name:<{w}}{"PASS" if ok else "FAIL":<8}{detail:>24}')
    print('=' * (w + 34))
    nf = sum(1 for _, ok, _ in results if not ok)
    print(f'{len(results) - nf}/{len(results)} passed')
    sys.exit(1 if nf else 0)

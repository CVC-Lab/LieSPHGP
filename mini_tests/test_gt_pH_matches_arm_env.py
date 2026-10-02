r"""Verification suite for the $n$-link port-Hamiltonian arm on $SO(3)^n$.

Implements the checks of §20 in `multi-joint-ph-system.md`. Each test isolates
one failure mode, and several verify our closed-form derivations against
**independent** computations (finite-differenced virtual work, direct kinetic
energy) rather than against other parts of the same code.

Run:  ``python mini_tests/test_gt_pH_matches_arm_env.py``

    #   test                            what it proves
    ─────────────────────────────────────────────────────────────────────────
    1   SO(3) preservation              exp-map integrator; no re-projection
    2   M(q) vs direct kinetic energy   the mass matrix / Jacobian assembly
    3   gravity: 3 ways agree           $\mathcal T(V)$, autodiff, virtual work
    4   Sigma(q) vs virtual work        the wind lever map
    5   energy conservation + order     integrator is 2nd order, no drift
    6   J_z conservation                the gravity term, independently of #5
    7   passivity                       $D\succeq0$ construction
    8   n=1 reduction                   EXACT match to windy_pendulum_3d
    9   Ito == Stratonovich             the vanishing-correction claim (§10.1)
    10  power consistency               $g(q)=T(q)^\top$ index/transpose sanity
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.arm_nlink_SO3.arm_nlink_physics import (                       # noqa: E402
    uniform_chain_params, exp_so3_batch, exp_so3,
    mass_matrix, potential, com_positions, translational_jacobians,
    joint_rate_map, input_map, wind_map, analytic_gravity_torque,
    trivialized_grad, dissipation_matrix, total_energy,
    vertical_angular_momentum, so3_defect, momentum_from_omega,
)
from envs.arm_nlink_SO3.windy_arm_nlink_so3 import (                     # noqa: E402
    windy_arm_nlink_so3, pendulum_equivalent_params,
)
from envs.windy_pendulum_3d import windy_pendulum_3d       # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def random_state(n, seed=0, omega_scale=0.7):
    """A random point on $SO(3)^n\\times\\mathbb{R}^{3n}$."""
    key = jax.random.PRNGKey(seed)
    kR, kw = jax.random.split(key)
    R = exp_so3_batch(jax.random.normal(kR, (n, 3), dtype=jnp.float64))
    omega = omega_scale * jax.random.normal(kw, (n, 3), dtype=jnp.float64)
    return R, omega


def generalized_force_by_virtual_work(params, R, forces, eps=1e-6):
    r"""Generalized force from world-frame COM forces, by **finite differences**.

    Virtual work under $R_j \mapsto R_j\exp([\phi_j]_\times)$:

    .. math:: \delta W = \sum_i f_i^\top \delta p_i = \tau^\top\delta\phi

    so central-differencing $\sum_i f_i^\top p_i(q)$ in each $\phi_{jk}$ gives
    $\tau_{jk}$. This touches **only** the forward-kinematics map
    :func:`com_positions` — it uses neither the Jacobians nor the closed forms
    it is being compared against, so it is a genuinely independent reference.
    """
    R_np = np.asarray(R)
    n = R_np.shape[0]
    f_np = np.asarray(forces)
    tau = np.zeros((n, 3))

    def work(R_arr):
        p = np.asarray(com_positions(params, jnp.asarray(R_arr)))
        return float(np.sum(f_np * p))

    for j in range(n):
        for k in range(3):
            phi = np.zeros(3)
            phi[k] = eps
            E_p = np.asarray(exp_so3(jnp.asarray(phi)))
            E_m = np.asarray(exp_so3(jnp.asarray(-phi)))
            Rp, Rm = R_np.copy(), R_np.copy()
            Rp[j] = R_np[j] @ E_p
            Rm[j] = R_np[j] @ E_m
            tau[j, k] = (work(Rp) - work(Rm)) / (2.0 * eps)
    return tau


def _report(results):
    width = max(len(r[0]) for r in results) + 2
    print()
    print('=' * (width + 34))
    print(f'{"test":<{width}}{"status":<8}{"measure":>24}')
    print('=' * (width + 34))
    for name, ok, detail in results:
        print(f'{name:<{width}}{"PASS" if ok else "FAIL":<8}{detail:>24}')
    print('=' * (width + 34))
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f'{len(results) - n_fail}/{len(results)} passed')
    return n_fail


# ══════════════════════════════════════════════════════════════════════
# 1. SO(3) preservation
# ══════════════════════════════════════════════════════════════════════

def test_so3_preservation(n=3, steps=400, tol=1e-10):
    r"""$\lVert R_i^\top R_i - I\rVert_F$ and $|\det R_i - 1|$ must stay at
    machine precision over a long, energetic rollout — the exponential-map
    integrator keeps $R$ on the manifold by construction, so no SVD
    re-projection is ever needed.
    """
    env = windy_arm_nlink_so3(n=n, friction_coeff=0.1, wind_force_std=0.4, seed=1)
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(steps):
        env.step(rng.uniform(-1.0, 1.0, size=(n, 3)))
        orth, det = env.manifold_defect()
        worst = max(worst, orth, det)
    env.close()
    return 'SO(3) preservation', worst < tol, f'max defect {worst:.2e}'


# ══════════════════════════════════════════════════════════════════════
# 2. Mass matrix vs directly computed kinetic energy
# ══════════════════════════════════════════════════════════════════════

def test_mass_matrix(n=3, tol=1e-8):
    r"""Check $\tfrac12\omega^\top M(q)\omega$ against kinetic energy assembled
    from first principles:

    .. math::
        T = \tfrac12\sum_i\big(m_i\lVert\dot p_i\rVert^2
                             + \omega_i^\top\mathbb{I}_i\omega_i\big)

    where $\dot p_i$ is obtained by **central-differencing** the forward
    kinematics along the actual flow $R_i(\varepsilon)=R_i\exp(\varepsilon[\omega_i]_\times)$.
    Independent of :func:`translational_jacobians`, so it validates the whole
    $M(q)$ derivation rather than just re-running it.
    """
    params = uniform_chain_params(n, com_fraction=0.5, inertia_scale=0.8)
    R, omega = random_state(n, seed=3)

    eps = 1e-6
    R_p = R @ exp_so3_batch(omega * eps)
    R_m = R @ exp_so3_batch(-omega * eps)
    pdot = (np.asarray(com_positions(params, R_p))
            - np.asarray(com_positions(params, R_m))) / (2.0 * eps)

    m = np.asarray(params.m)
    I_body = np.asarray(params.I_body)
    w = np.asarray(omega)
    T_direct = 0.5 * (np.sum(m * np.sum(pdot ** 2, axis=-1))
                      + np.einsum('ia,iab,ib->', w, I_body, w))

    w_flat = omega.reshape(-1)
    T_matrix = float(0.5 * w_flat @ mass_matrix(params, R) @ w_flat)

    err = abs(T_direct - T_matrix) / max(abs(T_direct), 1.0)
    return 'M(q) vs direct KE', err < tol, f'rel err {err:.2e}'


# ══════════════════════════════════════════════════════════════════════
# 3. Gravity computed three independent ways
# ══════════════════════════════════════════════════════════════════════

def test_gravity_three_ways(n=3, tol=1e-7):
    r"""The gravity generalized force must agree across:

    (a) the closed form $\mathcal T_j(V) = -g(\mu_j^{>}[\ell_j]_\times + m_j[c_j]_\times)R_j^\top e_z$ (§6.3),
    (b) autodiff, $\mathcal T_j$ applied to $\partial V/\partial q$ — the path the dynamics actually take,
    (c) finite-differenced virtual work with $f_i = -m_i g e_z$.

    (a) vs (b) checks the hand derivation; (c) anchors both to first principles.
    """
    params = uniform_chain_params(n, com_fraction=0.6, inertia_scale=0.9)
    R, _ = random_state(n, seed=5)

    analytic = np.asarray(analytic_gravity_torque(params, R))

    grad_V = jax.grad(
        lambda qf: potential(params, qf.reshape(n, 3, 3))
    )(R.reshape(-1)).reshape(n, 3, 3)
    autodiff = np.asarray(trivialized_grad(R, grad_V))

    forces = -np.asarray(params.m)[:, None] * float(params.g) * np.array([0.0, 0.0, 1.0])
    virtual = generalized_force_by_virtual_work(params, R, forces)

    e_ab = np.max(np.abs(analytic - autodiff))
    e_ac = np.max(np.abs(analytic - virtual))
    worst = max(e_ab, e_ac)
    return 'gravity: 3 ways agree', worst < tol, f'max diff {worst:.2e}'


# ══════════════════════════════════════════════════════════════════════
# 4. Wind lever map
# ══════════════════════════════════════════════════════════════════════

def test_wind_map(n=3, tol=1e-7):
    r"""$\Sigma(q)F$ must equal the generalized force of the world forces
    $f_i = a_i F$, computed by finite-differenced virtual work.

    Validates the closed form
    $\Sigma(q)_j=(A_j^{>}[\ell_j]_\times + a_j[c_j]_\times)R_j^\top$, whose
    index manipulation is the easiest place in the derivation to slip.
    """
    params = uniform_chain_params(n, com_fraction=0.4, inertia_scale=1.0)
    R, _ = random_state(n, seed=7)

    F = np.array([0.7, -1.3, 0.45])
    from_map = np.asarray(wind_map(params, R) @ jnp.asarray(F)).reshape(n, 3)

    forces = np.asarray(params.a)[:, None] * F[None, :]
    virtual = generalized_force_by_virtual_work(params, R, forces)

    err = np.max(np.abs(from_map - virtual))
    return 'Sigma(q) vs virtual work', err < tol, f'max diff {err:.2e}'


# ══════════════════════════════════════════════════════════════════════
# 5. Energy conservation and integrator order
# ══════════════════════════════════════════════════════════════════════

def _conserved_quantity_drift(n, horizon, quantity, seed):
    r"""Max relative drift of a conserved quantity at three substep resolutions.

    Returns `(d10, d20, d40)` for `n_substeps` $\in\{10,20,40\}$, all started
    from an identical initial condition so the numbers are directly comparable.
    `quantity` is a callable `env -> float`.
    """
    R0, omega0 = random_state(n, seed=seed, omega_scale=1.0)
    R0_np, omega0_np = np.asarray(R0), np.asarray(omega0)

    def max_drift(n_sub):
        env = windy_arm_nlink_so3(
            n=n, dt=0.05, n_substeps=n_sub, friction_coeff=0.0,
            air_drag=0.0, wind_force_std=0.0, external_force_std=0.0, seed=0)
        env.reset(seed=0, options={'R_init': R0_np, 'omega_init': omega0_np})
        q0 = quantity(env)
        scale = max(abs(q0), 1e-3)
        worst = 0.0
        for _ in range(int(horizon / env.dt)):
            env.step(np.zeros((n, 3)))
            worst = max(worst, abs(quantity(env) - q0) / scale)
        env.close()
        return worst

    return max_drift(10), max_drift(20), max_drift(40)


def _second_order(d10, d20, d40, coarse_tol):
    r"""Both the magnitude and the **convergence order** must be acceptable.

    Lie–Heun is an explicit second-order scheme, not a symplectic/momentum-
    preserving one, so conserved quantities drift at $O(h^2)$ — no fixed
    absolute tolerance can be met at fixed $h$. The meaningful assertion is that
    halving $h$ cuts the drift by $\approx4\times$: truncation error converges,
    a structural bug does not. The $2.5\times$ threshold leaves room for the
    oscillatory (non-monotone) component of the error.
    """
    order_ok = (d10 / max(d20, 1e-18) > 2.5) and (d20 / max(d40, 1e-18) > 2.5)
    return (d10 < coarse_tol) and order_ok


def test_energy_conservation(n=2, horizon=4.0, tol=5e-3):
    r"""With $D=0$, $u=0$ and no wind, $H$ is exactly conserved by the continuous
    flow, so any deviation is pure truncation error — and must vanish at
    $O(h^2)$ under refinement. See :func:`_second_order`.
    """
    d10, d20, d40 = _conserved_quantity_drift(
        n, horizon, lambda e: e.energy(), seed=11)
    return ('energy conservation + order',
            _second_order(d10, d20, d40, tol),
            f'{d10:.1e}/{d20:.1e}/{d40:.1e}')


# ══════════════════════════════════════════════════════════════════════
# 6. Vertical angular momentum
# ══════════════════════════════════════════════════════════════════════

def test_vertical_momentum(n=2, horizon=4.0, tol=5e-3):
    r"""$V$ is invariant under a global rotation about $e_z$, whose trivialized
    generator is $\delta\phi_i = \alpha R_i^\top e_z$. Noether's theorem then
    conserves

    .. math:: J_z = e_z^\top\sum_i R_i p_i .

    This probes the **gravity** term specifically: an error in $\mathcal T(V)$
    that happened to preserve energy would still break $J_z$.

    As with energy, the integrator conserves $J_z$ only to $O(h^2)$, so the
    assertion is on the convergence order (see :func:`_second_order`).
    """
    d10, d20, d40 = _conserved_quantity_drift(
        n, horizon, lambda e: e.vertical_momentum(), seed=13)
    return ('J_z conservation + order',
            _second_order(d10, d20, d40, tol),
            f'{d10:.1e}/{d20:.1e}/{d40:.1e}')


# ══════════════════════════════════════════════════════════════════════
# 7. Passivity
# ══════════════════════════════════════════════════════════════════════

def test_passivity(n=3, steps=200):
    r"""With $u=0$ and no wind the energy balance reduces to

    .. math:: \frac{dH}{dt} = -\xi^\top D(q,\omega)\,\xi \le 0

    so $H$ must be non-increasing. Also checks $D\succeq0$ spectrally at every
    visited state.
    """
    env = windy_arm_nlink_so3(
        n=n, dt=0.05, friction_coeff=0.4, air_drag=0.1,
        varying_friction=True, wind_force_std=0.0,
        external_force_std=0.0, seed=17)
    env.reset(seed=17)

    worst_increase = 0.0
    worst_eig = np.inf
    H_prev = env.energy()
    for _ in range(steps):
        env.step(np.zeros((n, 3)))
        H_now = env.energy()
        worst_increase = max(worst_increase, H_now - H_prev)
        H_prev = H_now
        D = dissipation_matrix(env.params, jnp.asarray(env.R), jnp.asarray(env.omega))
        worst_eig = min(worst_eig, float(jnp.min(jnp.linalg.eigvalsh(D))))
    env.close()

    ok = (worst_increase < 1e-9) and (worst_eig > -1e-12)
    return ('passivity dH/dt <= 0', ok,
            f'dH+ {worst_increase:.1e} eig {worst_eig:.1e}')


# ══════════════════════════════════════════════════════════════════════
# 8. n = 1 reduction to windy_pendulum_3d   ← the key regression test
# ══════════════════════════════════════════════════════════════════════

def test_n1_reduction(steps=100, varying_friction=False, tol=1e-9):
    r"""At $n=1$ the arm must reproduce ``windy_pendulum_3d`` **exactly**.

    Parameters per §13.1: $c_1=\ell e_z$, $\mathbb{I}_1=\mathrm{diag}(0,0,m\ell^2)$,
    giving $M=m\ell^2 I_3$, $V=mg\ell R_{22}$, $g(q)=I_3$, $D=d_1I_3$ and
    $\Sigma=[c_1]_\times R^\top$.

    The two implementations are genuinely different: ``windy_pendulum_3d`` is
    NumPy Newton–Euler ($\mathbb{I}\dot\omega = \tau - \omega\times\mathbb{I}\omega$)
    carrying $\omega$, while this one is JAX port-Hamiltonian carrying $p$. Both
    are driven by the same random torque sequence from the same initial state,
    with the stochastic wind off so the comparison is deterministic.
    """
    m, l, g, fric = 1.0, 1.0, 9.81, 0.5
    rng = np.random.default_rng(23)

    ref = windy_pendulum_3d(
        g=g, m=m, l=l, dt=0.05, friction_coeff=fric,
        varying_friction=varying_friction, external_force_type='constant',
        external_force_std=0.0, wind_force_std=0.0, g_diag=1.0, seed=23)
    ref.reset(seed=23)
    R0 = ref.R.copy()
    omega0 = ref.omega.copy()

    arm = windy_arm_nlink_so3(
        n=1, dt=0.05, n_substeps=10, wind_force_std=0.0, external_force_std=0.0,
        params=pendulum_equivalent_params(m=m, l=l, g=g, friction_coeff=fric,
                                          varying_friction=varying_friction),
        seed=23)
    arm.reset(seed=23, options={'R_init': R0[None], 'omega_init': omega0[None]})

    u_seq = rng.uniform(-2.0, 2.0, size=(steps, 3))
    worst_R = worst_w = 0.0
    for k in range(steps):
        ref.step(u_seq[k])
        arm.step(u_seq[k].reshape(1, 3))
        worst_R = max(worst_R, np.linalg.norm(ref.R - arm.R[0]))
        worst_w = max(worst_w, np.linalg.norm(ref.omega - arm.omega[0]))
    ref.close()
    arm.close()

    worst = max(worst_R, worst_w)
    label = 'varfric' if varying_friction else 'constfric'
    return (f'n=1 reduction ({label})', worst < tol,
            f'|dR| {worst_R:.1e} |dw| {worst_w:.1e}')


# ══════════════════════════════════════════════════════════════════════
# 9. Ito == Stratonovich
# ══════════════════════════════════════════════════════════════════════

def test_ito_equals_stratonovich(n=3, tol=1e-12):
    r"""Verify the §10.1 claim that the Stratonovich$\to$Itô correction vanishes.

    With $x=(q,p)$ the full diffusion matrix is
    $\Xi(x) = \big(0_{9n\times3};\ \sigma\Sigma(q)\big)$ and the correction is

    .. math:: \tfrac12\sum_{k=1}^{3}\big(\partial_x\Xi_{\cdot k}\big)\Xi_{\cdot k} .

    It is structurally zero because $\Xi$ depends on $q$ alone while its only
    non-zero rows lie in the $p$-block. Here we compute it numerically by
    autodiff rather than trusting the argument — if it were non-zero, the
    Heun (Stratonovich) integrator would be sampling the wrong SDE.
    """
    params = uniform_chain_params(n, com_fraction=0.7, inertia_scale=1.0)
    R, omega = random_state(n, seed=29)
    p = momentum_from_omega(params, R, omega)
    x = jnp.concatenate([R.reshape(-1), p.reshape(-1)])
    sigma = 0.83

    def Xi(x_):
        R_ = x_[:9 * n].reshape(n, 3, 3)
        top = jnp.zeros((9 * n, 3), dtype=x_.dtype)
        return jnp.concatenate([top, sigma * wind_map(params, R_)], axis=0)

    J = jax.jacfwd(Xi)(x)                       # (12n, 3, 12n)
    correction = 0.5 * jnp.einsum('akb,bk->a', J, Xi(x))
    worst = float(jnp.max(jnp.abs(correction)))
    return 'Ito == Stratonovich', worst < tol, f'max corr {worst:.2e}'


# ══════════════════════════════════════════════════════════════════════
# 10. Power consistency of the input map
# ══════════════════════════════════════════════════════════════════════

def test_power_consistency(n=4, g_diag=(1.0, 1.0, 1.0), tol=1e-12):
    r"""$g(q)=T(q)^\top\Gamma$ was *derived* from the power pairing, so

    .. math:: (\Gamma u)^\top\Omega = \big(g(q)u\big)^\top\omega,
              \qquad \Omega = T(q)\omega

    for arbitrary $u,\omega$. The gain sits on the *delivered* torque $\Gamma u$,
    not the commanded $u$: the motor's output is what does work on the joint.
    At $\Gamma=I$ this reduces to $u^\top\Omega=(gu)^\top\omega$.

    Cheap, but it catches any transpose or block-index slip in
    :func:`joint_rate_map` / :func:`input_map`, which would silently break both
    the actuation and the joint friction.
    """
    params = uniform_chain_params(n, d=0.5, varying_friction=False,
                                  g_diag=g_diag)
    R, omega = random_state(n, seed=31)
    u = jax.random.normal(jax.random.PRNGKey(37), (n, 3), dtype=jnp.float64)

    T = joint_rate_map(R)
    Omega = T @ omega.reshape(-1)
    lhs = float(jnp.dot((params.gain * u).reshape(-1), Omega))
    rhs = float(jnp.dot(input_map(params, R) @ u.reshape(-1), omega.reshape(-1)))

    err = abs(lhs - rhs) / max(abs(lhs), 1.0)
    tag = 'unit' if tuple(g_diag) == (1.0, 1.0, 1.0) else 'gain'
    return f'power ({tag}): (Gu).Omega == (gu).w', err < tol, f'rel err {err:.2e}'


def test_unit_gain_is_pure_kinematics(n=3, tol=0.0):
    r"""$\gamma_i=(1,1,1)$ must reproduce the old parameter-free
    $g(q)=T(q)^\top$ **bit for bit**.

    Guards the backward compatibility of every result produced before the
    actuator gain existed: if this drifts, previously-generated datasets are no
    longer reproducible by the current code.
    """
    params = uniform_chain_params(n, d=0.5, varying_friction=False)
    R, _ = random_state(n, seed=53)
    err = float(jnp.max(jnp.abs(input_map(params, R) - joint_rate_map(R).T)))
    return 'unit gain == T(q)^T exactly', err <= tol, f'max diff {err:.1e}'


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = [
        test_so3_preservation(),
        test_mass_matrix(),
        test_gravity_three_ways(),
        test_wind_map(),
        test_energy_conservation(),
        test_vertical_momentum(),
        test_passivity(),
        test_n1_reduction(varying_friction=False),
        test_n1_reduction(varying_friction=True),
        test_ito_equals_stratonovich(),
        test_power_consistency(),
        test_power_consistency(g_diag=(0.5, 0.7, 0.1)),
        test_unit_gain_is_pure_kinematics(),
    ]
    sys.exit(1 if _report(results) else 0)

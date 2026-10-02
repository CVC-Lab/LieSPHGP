r"""Correctness gate for the $n$-link model: **ground-truth subnetworks**.

Replaces every GP subnetwork with the exact analytic physics from
``envs/arm_nlink_so3/arm_nlink_physics.py`` and checks that the model's
port-Hamiltonian algebra and integrator reproduce the environment.

This separates two questions that are otherwise impossible to tell apart when
training goes wrong:

    1. Is ``network.py`` + ``lie_integrator_nlink.py`` mathematically correct?   <- here
    2. Does the GP actually learn the right functions?                          <- training

If these tests pass, any subsequent training failure is a *learning* problem,
not an algebra problem.

Run:  ``python src/models/arm_n_link/ph_gp_sde/eval_ground_truth_match.py``
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..', '..', '..', '..'))
for _p in (PKG_ROOT, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.arm_nlink_so3 import arm_nlink_physics as phys          # noqa: E402
from envs.arm_nlink_so3.windy_arm_nlink_so3 import windy_arm_nlink_so3  # noqa: E402
from utils.lie_integrator_nlink import (                          # noqa: E402
    lie_heun_sde_rollout_nlink, exp_so3_batch,
)


# ══════════════════════════════════════════════════════════════════════
# Ground-truth "model": the analytic physics behind the model's API
# ══════════════════════════════════════════════════════════════════════

class GroundTruthArmModel:
    r"""Exposes exactly the interface ``network.DissipativeArmHamSDE`` exposes,
    but every subnetwork returns the analytic quantity instead of a GP.

    Bridges the two conventions: the physics module works with `R : (n,3,3)` and
    `p : (n,3)`, the model API with flat `q : (9n,)` and `p : (3n,)`.

    .. note::
       The true $D(q,\omega)$ depends on $\omega$ through the friction
       modulation $\rho_i$, whereas the model's `Dw_net(q)` is a function of $q$
       alone. With `varying_friction=False` (our test configuration) $\rho\equiv1$
       and the two agree exactly. With it on, this class deliberately uses the
       $\omega$-dependent truth — the resulting mismatch measures the
       *architectural* limitation, not a bug.
    """

    def __init__(self, params, sigma: float = 0.0):
        self.params = params
        self.n = phys.n_links(params)
        self.sigma = float(sigma)

    # ── shape adapters ──
    def _R(self, q):
        return q.reshape(self.n, 3, 3)

    # ── structure matrices ──
    def M_inv(self, q):
        r"""$M^{-1}(q)$ by solving against the identity — the analytic inverse."""
        return jnp.linalg.inv(phys.mass_matrix(self.params, self._R(q)))

    def Sigma(self, q):
        r"""$\sigma\,\Sigma(q)\in\mathbb{R}^{3n\times3}$."""
        return self.sigma * phys.wind_map(self.params, self._R(q))

    def stochastic_increment_p(self, q, dW):
        return self.Sigma(q) @ dW

    def drift_p(self, q, p, u):
        r"""$\dot p$ via the physics module, reshaped to the model's flat convention."""
        R = self._R(q)
        return phys.drift_p(self.params, R, p.reshape(self.n, 3),
                            u.reshape(self.n, 3),
                            jnp.zeros(3, dtype=q.dtype)).reshape(3 * self.n)

    def drift(self, q, omega, u):
        r"""$\dot\omega = M^{-1}\dot p + \dot{(M^{-1})}p$ — the PL-term path."""
        M_inv_q = self.M_inv(q)
        p = jnp.linalg.solve(M_inv_q, omega)
        dp = self.drift_p(q, p, u)
        xi = (M_inv_q @ p).reshape(self.n, 3)
        dq = jnp.cross(self._R(q),
                       jnp.broadcast_to(xi[:, None, :], (self.n, 3, 3)),
                       axis=-1).reshape(9 * self.n)
        _, dM_inv_dt = jax.jvp(self.M_inv, (q,), (dq,))
        return M_inv_q @ dp + dM_inv_dt @ p


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════

def _random_state(n, seed=0):
    kR, kw = jax.random.split(jax.random.PRNGKey(seed))
    R = exp_so3_batch(jax.random.normal(kR, (n, 3), dtype=jnp.float64))
    omega = 0.6 * jax.random.normal(kw, (n, 3), dtype=jnp.float64)
    return R, omega


def test_drift_matches_physics(n=2, tol=1e-12):
    r"""The model-API `drift_p` must equal the physics `drift_p` exactly.

    Checks only the flat$\leftrightarrow$blocked shape adapters; any transpose
    or reshape slip between $(9n,)/(3n,)$ and $(n,3,3)/(n,3)$ shows up here.
    """
    params = phys.uniform_chain_params(n, d=0.5, varying_friction=False)
    R, omega = _random_state(n, seed=3)
    gt = GroundTruthArmModel(params)

    p_blocked = phys.momentum_from_omega(params, R, omega)
    u_blocked = 0.4 * jnp.ones((n, 3), dtype=jnp.float64)

    ref = phys.drift_p(params, R, p_blocked, u_blocked, jnp.zeros(3))
    got = gt.drift_p(R.reshape(-1), p_blocked.reshape(-1), u_blocked.reshape(-1))

    err = float(jnp.max(jnp.abs(ref.reshape(-1) - got)))
    return 'drift_p vs physics', err < tol, f'max diff {err:.2e}'


def test_rollout_matches_env(n=2, steps=60, tol=1e-9):
    r"""GT model + **model-side integrator** must reproduce the environment.

    The deterministic case ($\sigma=0$) with a random torque sequence. This is
    the end-to-end gate: it exercises `lie_heun_sde_rollout_nlink`, the
    $\omega\leftrightarrow p$ boundary conversions and the whole drift.
    """
    params = phys.uniform_chain_params(n, d=0.5, varying_friction=False)
    rng = np.random.default_rng(11)

    env = windy_arm_nlink_so3(n=n, dt=0.05, n_substeps=10, params=params,
                              wind_force_std=0.0, external_force_std=0.0, seed=11)
    env.reset(seed=11)
    R0, omega0 = env.get_state()

    u_seq = rng.uniform(-1.5, 1.5, size=(steps, 3 * n))
    env_R, env_w = [], []
    for k in range(steps):
        env.step(u_seq[k])
        env_R.append(env.R.copy())
        env_w.append(env.omega.copy())
    env.close()

    gt = GroundTruthArmModel(params, sigma=0.0)
    x0 = jnp.concatenate([jnp.asarray(R0).reshape(-1), jnp.asarray(omega0).reshape(-1)])
    dW = jnp.zeros((steps, 10, 3), dtype=jnp.float64)
    traj = lie_heun_sde_rollout_nlink(gt, x0, jnp.asarray(u_seq), 0.05 / 10, dW)

    got_R = np.asarray(traj[1:, :9 * n]).reshape(steps, n, 3, 3)
    got_w = np.asarray(traj[1:, 9 * n:]).reshape(steps, n, 3)

    eR = float(np.max(np.abs(np.stack(env_R) - got_R)))
    ew = float(np.max(np.abs(np.stack(env_w) - got_w)))
    worst = max(eR, ew)
    return 'rollout vs env (sigma=0)', worst < tol, f'|dR| {eR:.1e} |dw| {ew:.1e}'


def test_stochastic_path(n=2, steps=40, sigma=0.5, tol=1e-11):
    r"""With matched Wiener increments, the model-side and physics-side
    integrators must agree **including** the diffusion.

    The environment draws its own noise internally, so it cannot be compared
    path-by-path; instead both integrators are driven by the identical `dW`
    sequence. Validates `stochastic_increment_p` and the Stratonovich (same-`dW`
    in both Heun stages) convention.
    """
    params = phys.uniform_chain_params(n, d=0.5, varying_friction=False)
    R0, omega0 = _random_state(n, seed=17)
    h = 0.05 / 10

    key = jax.random.PRNGKey(23)
    dW = jax.random.normal(key, (steps, 10, 3), dtype=jnp.float64) * jnp.sqrt(h)
    u = 0.3 * jnp.ones((steps, 3 * n), dtype=jnp.float64)

    # ── model-side ──
    gt = GroundTruthArmModel(params, sigma=sigma)
    x0 = jnp.concatenate([R0.reshape(-1), omega0.reshape(-1)])
    traj = lie_heun_sde_rollout_nlink(gt, x0, u, h, dW)

    # ── physics-side, same dW ──
    R_c, p_c = R0, phys.momentum_from_omega(params, R0, omega0)
    outs = []
    for k in range(steps):
        R_c, p_c = phys.lie_heun_substeps(
            params, R_c, p_c, u[k].reshape(n, 3), h,
            jnp.zeros(3), jnp.asarray(sigma), dW[k])
        outs.append((R_c, phys.omega_from_momentum(params, R_c, p_c)))

    eR = float(np.max(np.abs(np.stack([np.asarray(o[0]) for o in outs])
                             - np.asarray(traj[1:, :9 * n]).reshape(steps, n, 3, 3))))
    ew = float(np.max(np.abs(np.stack([np.asarray(o[1]) for o in outs])
                             - np.asarray(traj[1:, 9 * n:]).reshape(steps, n, 3))))
    worst = max(eR, ew)
    return 'stochastic path (matched dW)', worst < tol, f'|dR| {eR:.1e} |dw| {ew:.1e}'


def test_omega_dot(n=2, tol=1e-7):
    r"""The $\dot\omega$ used by the PL term must match a central difference of
    $\omega(t)=M^{-1}(q(t))\,p(t)$ along the flow.

    Verifies the JVP path $\dot\omega = M^{-1}\dot p + \dot{(M^{-1})}p$, which is
    the one place the model does something the integrator never exercises.
    """
    params = phys.uniform_chain_params(n, d=0.5, varying_friction=False)
    R, omega = _random_state(n, seed=29)
    gt = GroundTruthArmModel(params)

    q = R.reshape(-1)
    u = 0.25 * jnp.ones(3 * n, dtype=jnp.float64)
    p = jnp.linalg.solve(gt.M_inv(q), omega.reshape(-1))
    dp = gt.drift_p(q, p, u)

    xi = (gt.M_inv(q) @ p).reshape(n, 3)
    dq = jnp.cross(R, jnp.broadcast_to(xi[:, None, :], (n, 3, 3)), axis=-1).reshape(-1)

    eps = 1e-6
    w_plus = gt.M_inv(q + eps * dq) @ (p + eps * dp)
    w_minus = gt.M_inv(q - eps * dq) @ (p - eps * dp)
    fd = (w_plus - w_minus) / (2 * eps)

    got = gt.drift(q, omega.reshape(-1), u)
    err = float(jnp.max(jnp.abs(fd - got)))
    return 'omega_dot (JVP vs FD)', err < tol, f'max diff {err:.2e}'


def test_network_shapes(n=2):
    """The learned network must produce every structure matrix at the right shape."""
    sys.path.insert(0, THIS_FILE_DIR)
    from network import DissipativeArmHamSDE      # noqa: E402

    model = DissipativeArmHamSDE(key=jax.random.PRNGKey(0), n=n, hidden_dim=8)
    R, omega = _random_state(n, seed=5)
    q = R.reshape(-1)
    p = jnp.ones(3 * n, dtype=jnp.float64)
    u = jnp.zeros(3 * n, dtype=jnp.float64)

    d = 3 * n
    checks = {
        'M_inv': (model.M_inv(q).shape, (d, d)),
        'Sigma': (model.Sigma(q).shape, (d, 3)),
        'drift_p': (model.drift_p(q, p, u).shape, (d,)),
        'drift': (model.drift(q, omega.reshape(-1), u).shape, (d,)),
        'dp_stoch': (model.stochastic_increment_p(q, jnp.ones(3)).shape, (d,)),
    }
    bad = [k for k, (got, want) in checks.items() if tuple(got) != want]
    # M_inv must also be symmetric PSD with the epsilon floor respected.
    M = model.M_inv(q)
    sym = float(jnp.max(jnp.abs(M - M.T)))
    min_eig = float(jnp.min(jnp.linalg.eigvalsh(M)))
    ok = (not bad) and sym < 1e-10 and min_eig > 0
    return ('network shapes + PSD', ok,
            f'sym {sym:.1e} eig {min_eig:.3f}' if not bad else f'bad: {bad}')


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = [
        test_drift_matches_physics(),
        test_rollout_matches_env(),
        test_stochastic_path(),
        test_omega_dot(),
        test_network_shapes(),
    ]
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
    sys.exit(1 if n_fail else 0)

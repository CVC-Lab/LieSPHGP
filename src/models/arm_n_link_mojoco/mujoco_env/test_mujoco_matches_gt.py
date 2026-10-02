r"""**The gate.** Does MuJoCo agree with our analytic port-Hamiltonian model?

Nothing downstream — no dataset, no training run, no report — means anything
unless this passes. The whole point of generating data in an independent,
widely-trusted simulator is to remove "you evaluated on your own generator" as a
criticism; that argument only works if the two really do describe the same
mechanical system, and the ways they can silently differ (quaternion sign,
velocity frame, relative-vs-absolute coordinates, gear conventions) all produce
plausible-looking but wrong trajectories.

The coordinate change that governs everything
---------------------------------------------
MuJoCo's generalized coordinates are the **relative** joint rates
$\Omega = T(q)\,\omega$; ours are the **absolute** body rates $\omega$. Velocities
transform with $T$, so generalized forces transform with $T^{-\top}$ (power is
invariant: $\tau_{\rm abs}^\top\omega = \tau_{\rm rel}^\top\Omega$) and the mass
matrix transforms congruently:

.. math::
    M_{\rm rel} = T^{-\top} M_{\rm abs} T^{-1}, \qquad
    \tau_{\rm rel} = T^{-\top}\tau_{\rm abs}

Every algebraic test below is that identity applied to one term.

What is tested
--------------
========  =====================================================================
T1        ball joints — $n_v = 3n$, else the DOF counts do not even match
T2        $(R,\omega)\to$ MuJoCo $\to(R,\omega)$ round-trips exactly
T3        $\omega$ agrees three ways: qvel recursion, ``mj_objectVelocity``,
          and finite differences of ``xmat`` (the neutral arbiter)
T4        $M$: ``qM`` $= T^{-\top}M_{\rm abs}T^{-1}$
T5        gravity: ``qfrc_bias``$\big|_{\Omega=0} = -T^{-\top}\mathcal T(V)$
T6        damping: ``qfrc_passive`` $= -T^{-\top}D\omega = -\mathrm{diag}(d)\Omega$
T7        actuation: ``qfrc_actuator`` $= T^{-\top}g(q)u = \Gamma u$
T8        the full flow: Lie–Heun $\to$ MuJoCo as $h\to0$, at 2nd order
========  =====================================================================

T1–T7 are *algebraic* and hold to machine precision (~$10^{-10}$); they isolate
each term, so a failure says exactly which one is wrong. T8 is a **convergence
study, not an equality**: MuJoCo integrates with RK4 at $10^{-3}$ while Lie–Heun
is 2nd order, so the two agree only in the limit. Asserting a fixed tolerance
there would be testing the integrator, not the physics.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import jax                                                        # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                           # noqa: E402
import mujoco                                                     # noqa: E402

import arm_nlink_physics as phys                                  # noqa: E402
from build_mjcf import ArmSpec                                    # noqa: E402
from mujoco_arm_nlink import MujocoArmEnv, random_state           # noqa: E402


TOL = 1e-9
_RESULTS: list[tuple[str, bool, str]] = []


def _check(name: str, err: float, tol: float = TOL, note: str = '') -> bool:
    ok = bool(err <= tol)
    detail = f'err = {err:.3e}  (tol {tol:.0e})' + (f'   {note}' if note else '')
    _RESULTS.append((name, ok, detail))
    print(f'  [{"PASS" if ok else "FAIL"}]  {name:<46s} {detail}')
    return ok


def _jax_params(env: MujocoArmEnv):
    """`ArmParams` (numpy, straight out of mjModel) → JAX pytree."""
    return jax.tree_util.tree_map(jnp.asarray, env.params)


def _T_and_inv(R: np.ndarray):
    r"""$T(q)$ and $T(q)^{-1}$ as numpy, from the analytic joint-rate map."""
    T = np.asarray(phys.joint_rate_map(jnp.asarray(R)))
    return T, np.linalg.inv(T)


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════

def test_ball_joints(env):
    """T1 — the DOF count must be $3n$, i.e. one spherical joint per link."""
    n = env.n
    ok = (env.model.nv == 3 * n) and (env.model.nq == 4 * n)
    _RESULTS.append(('T1 ball joints: nv = 3n, nq = 4n', ok,
                     f'nv={env.model.nv} nq={env.model.nq} n={n}'))
    print(f'  [{"PASS" if ok else "FAIL"}]  '
          f'{"T1 ball joints: nv = 3n, nq = 4n":<46s} '
          f'nv={env.model.nv} nq={env.model.nq} (want {3 * n}, {4 * n})')


def test_state_roundtrip(env, rng):
    """T2 — writing $(R,\\omega)$ and reading it back must be lossless."""
    errs_R, errs_w = [], []
    for _ in range(20):
        R0, w0 = random_state(rng, env.n)
        env.reset(R0, w0)
        errs_R.append(np.abs(env.rotations() - R0).max())
        errs_w.append(np.abs(env.body_rates() - w0).max())
    _check('T2 state round-trip (R)', float(np.max(errs_R)), 1e-12)
    _check('T2 state round-trip (omega)', float(np.max(errs_w)), 1e-12)


def test_omega_conventions(env, rng):
    r"""T3 — $\omega$ three independent ways.

    Finite differences of ``xmat`` define the body rate directly from
    $\dot R_i = R_i[\omega_i]_\times$, so they arbitrate between our qvel
    recursion and MuJoCo's ``mj_objectVelocity``. FD is only first-order
    accurate, hence the looser tolerance — but a *convention* error would be
    $O(1)$, not $O(h)$.
    """
    R0, w0 = random_state(rng, env.n)
    env.reset(R0, w0)
    _check('T3 omega: qvel recursion vs mj_objectVelocity',
           float(np.abs(env.body_rates() - env.body_rates_mujoco()).max()), 1e-7)

    h = 1e-7
    saved = env.model.opt.timestep
    env.model.opt.timestep = h
    Ra = env.rotations().copy()
    env.data.ctrl[:] = 0.0
    mujoco.mj_step(env.model, env.data)
    Rb = env.rotations().copy()
    env.model.opt.timestep = saved

    Rdot = (Rb - Ra) / h
    w_fd = np.stack([_vee(Ra[i].T @ Rdot[i]) for i in range(env.n)])
    env.reset(R0, w0)
    _check('T3 omega: qvel recursion vs finite differences',
           float(np.abs(w_fd - env.body_rates()).max()), 1e-5,
           note='1st-order FD; a frame error would be O(1)')


def _vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def test_mass_matrix(env, rng):
    r"""T4 — $M_{\rm rel} \stackrel{?}{=} T^{-\top}M_{\rm abs}T^{-1}$.

    This is the test people skip and then spend a day debugging: ``qM`` looks
    like a mass matrix, is the right shape, and is simply in different
    coordinates.
    """
    p = _jax_params(env)
    errs, conds = [], []
    for _ in range(10):
        R0, w0 = random_state(rng, env.n)
        env.reset(R0, w0)
        M_abs = np.asarray(phys.mass_matrix(p, jnp.asarray(R0)))
        _, Tinv = _T_and_inv(env.rotations())
        pred = Tinv.T @ M_abs @ Tinv
        got = env.mass_matrix_relative()
        errs.append(np.abs(pred - got).max() / np.abs(got).max())
        conds.append(np.linalg.cond(M_abs))
    _check('T4 mass matrix: qM = T^-T M_abs T^-1', float(np.max(errs)), 1e-10,
           note=f'cond(M_abs) <= {max(conds):.1f}')


def test_gravity(env, rng):
    r"""T5 — gravity, isolated by evaluating at $\Omega=0$.

    MuJoCo's equation of motion is $M\dot\Omega + c = \tau$ with
    ``qfrc_bias`` $=c$ holding Coriolis, centrifugal **and** gravity. At zero
    velocity only gravity survives, and it enters with the opposite sign to a
    generalized force, so $c = -\tau_{\rm grav}$.
    """
    p = _jax_params(env)
    errs = []
    for _ in range(10):
        R0, _ = random_state(rng, env.n)
        env.reset(R0, np.zeros((env.n, 3)))
        env.data.ctrl[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        tau_abs = np.asarray(
            phys.analytic_gravity_torque(p, jnp.asarray(R0))).reshape(-1)
        _, Tinv = _T_and_inv(env.rotations())
        pred = -(Tinv.T @ tau_abs)
        got = np.asarray(env.data.qfrc_bias)
        errs.append(np.abs(pred - got).max() / max(np.abs(got).max(), 1e-12))
    _check('T5 gravity: qfrc_bias = -T^-T tau_V', float(np.max(errs)), 1e-10)


def test_damping(env, rng):
    r"""T6 — joint friction.

    Our $D(q)=T^\top\mathrm{blkdiag}(d_i I_3)T$ acts on the absolute rate, so in
    MuJoCo's relative coordinates the force is

    .. math::
        -T^{-\top} D\,\omega = -T^{-\top}T^\top\mathrm{diag}(d)\,T\omega
                             = -\mathrm{diag}(d)\,\Omega ,

    i.e. exactly MuJoCo's per-DOF ``damping``. Both forms are checked — the
    second confirms MuJoCo's behaviour, the first confirms our $D$ is the right
    lift of it into absolute coordinates.
    """
    p = _jax_params(env)
    errs_direct, errs_D = [], []
    for _ in range(10):
        R0, w0 = random_state(rng, env.n)
        env.reset(R0, w0)
        env.data.ctrl[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        d_full = np.repeat(env.params.d, 3)
        got = np.asarray(env.data.qfrc_passive)
        errs_direct.append(np.abs(-d_full * env.data.qvel - got).max())

        D = np.asarray(phys.dissipation_matrix(
            p, jnp.asarray(R0), jnp.asarray(w0)))
        _, Tinv = _T_and_inv(env.rotations())
        pred = -(Tinv.T @ (D @ np.asarray(w0).reshape(-1)))
        errs_D.append(np.abs(pred - got).max() / max(np.abs(got).max(), 1e-12))
    _check('T6 damping: qfrc_passive = -diag(d) Omega',
           float(np.max(errs_direct)), 1e-10)
    _check('T6 damping: qfrc_passive = -T^-T D omega',
           float(np.max(errs_D)), 1e-10)


def test_actuation(env, rng):
    r"""T7 — the input map.

    Our $g(q)=T(q)^\top\Gamma$ pushes joint torques into absolute coordinates,
    so back in relative coordinates $T^{-\top}g(q)u = \Gamma u$. The reaction
    torque on the parent link — the $-R_i^\top R_{i+1}u_{i+1}$ term, Newton's
    third law, *not* a second control — is contained in $T^\top$ and is applied
    by MuJoCo automatically.
    """
    p = _jax_params(env)
    errs_direct, errs_g = [], []
    Gamma = np.diag(env.params.gain.reshape(-1))
    for _ in range(10):
        R0, w0 = random_state(rng, env.n)
        env.reset(R0, w0)
        u = rng.normal(size=3 * env.n)
        env.data.ctrl[:] = u
        mujoco.mj_forward(env.model, env.data)

        got = np.asarray(env.data.qfrc_actuator)
        errs_direct.append(np.abs(Gamma @ u - got).max())

        gq = np.asarray(phys.input_map(p, jnp.asarray(R0)))
        _, Tinv = _T_and_inv(env.rotations())
        errs_g.append(np.abs(Tinv.T @ (gq @ u) - got).max())
    _check('T7 actuation: qfrc_actuator = Gamma u',
           float(np.max(errs_direct)), 1e-10)
    _check('T7 actuation: qfrc_actuator = T^-T g(q) u',
           float(np.max(errs_g)), 1e-10)


def test_flow_convergence(env, rng, horizon=0.05):
    r"""T8 — the whole vector field, via a convergence study.

    Both integrators approximate the same flow, so their difference is bounded
    by the sum of their truncation errors. MuJoCo runs RK4 at $10^{-3}$ (error
    $\sim10^{-12}$), so refining Lie–Heun should drive the gap down at its own
    2nd order until it hits roundoff. A wrong term in the vector field instead
    shows up as an error that **stops decreasing** — that is the signal to watch,
    not any particular number.
    """
    p = _jax_params(env)
    R0, w0 = random_state(rng, env.n)
    u = rng.normal(size=(env.n, 3))

    env.reset(R0, w0)
    for _ in range(int(round(horizon / env.dt))):
        env.step(u.reshape(-1))
    R_mj, w_mj = env.rotations(), env.body_rates()

    zero3 = jnp.zeros(3)
    print('        substeps        h        |dR|_inf     |dw|_inf     ratio')
    prev, ratios = None, []
    for n_sub in (10, 20, 40, 80, 160, 320):
        h = horizon / n_sub
        R_a, w_a = phys.step_qomega(
            p, jnp.asarray(R0), jnp.asarray(w0), jnp.asarray(u),
            jnp.asarray(h), zero3, jnp.asarray(0.0), jnp.zeros((n_sub, 3)))
        eR = float(np.abs(np.asarray(R_a) - R_mj).max())
        ew = float(np.abs(np.asarray(w_a) - w_mj).max())
        tot = max(eR, ew)
        ratio = prev / tot if prev else float('nan')
        if prev:
            ratios.append(ratio)
        print(f'        {n_sub:>8d}  {h:.5f}   {eR:.3e}    {ew:.3e}   '
              f'{ratio:>6.2f}')
        prev = tot

    # The criterion is the *order*, not any absolute number. If the two vector
    # fields differed by some delta, the error would plateau at ~delta*horizon
    # and the ratio would collapse towards 1 no matter how fine the step. Clean
    # 2nd order sustained all the way to the finest step therefore bounds the
    # discrepancy below the smallest error observed. An absolute threshold, by
    # contrast, would just be measuring how many substeps we chose to run.
    # `last > 3.0` only has to rule out a plateau, where the ratio collapses
    # towards 1; 3.0 still corresponds to order log2(3) = 1.58. Setting it just
    # under the ideal 4 would make the test knife-edge on how deep into the
    # asymptotic regime the refinement happens to reach, which varies with n.
    order = float(np.log2(np.mean(ratios))) if ratios else 0.0
    last = ratios[-1] if ratios else 0.0
    ok = (1.8 < order < 2.2) and last > 3.0
    _RESULTS.append(('T8 flow: Lie-Heun -> MuJoCo as h->0', ok,
                     f'order {order:.2f}, final ratio {last:.2f}, '
                     f'residual {tot:.2e}'))
    print(f'  [{"PASS" if ok else "FAIL"}]  '
          f'{"T8 flow: Lie-Heun -> MuJoCo as h->0":<46s} '
          f'order = {order:.2f} (want ~2), final ratio = {last:.2f} '
          f'(want ~4, i.e. no plateau)')
    print(f'          => vector-field discrepancy bounded by {tot:.2e} '
          f'over t = {horizon}')


# ══════════════════════════════════════════════════════════════════════

def main(n: int = 2, seed: int = 0) -> int:
    spec = ArmSpec(n=n)
    env = MujocoArmEnv(spec)
    rng = np.random.default_rng(seed)

    print(f'\nMuJoCo {mujoco.__version__}  |  n = {n}  |  spec = {spec.tag()}')
    print(f'ArmParams read back from mjModel:')
    print(f'  m    = {env.params.m}')
    print(f'  d    = {env.params.d}      g = {env.params.g}')
    print(f'  ell  = {env.params.ell.reshape(-1)}')
    print(f'  c    = {env.params.c.reshape(-1)}')
    print(f'  I[0] = diag-ish {np.diag(env.params.I_body[0])}')
    print(f'  gain = {env.params.gain.reshape(-1)}')
    print(f'\n─── algebraic identities (machine precision) ───')
    test_ball_joints(env)
    test_state_roundtrip(env, rng)
    test_omega_conventions(env, rng)
    test_mass_matrix(env, rng)
    test_gravity(env, rng)
    test_damping(env, rng)
    test_actuation(env, rng)
    print(f'\n─── flow convergence (not an equality) ───')
    test_flow_convergence(env, rng)

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f'\n{"=" * 72}')
    if n_fail:
        print(f'GATE FAILED — {n_fail}/{len(_RESULTS)} checks failed. '
              f'Do NOT generate data or train until these pass.')
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f'  FAIL  {name}: {detail}')
    else:
        print(f'GATE PASSED — {len(_RESULTS)}/{len(_RESULTS)} checks. '
              f'MuJoCo and the analytic pH model describe the same system.')
    print(f'{"=" * 72}\n')
    return 1 if n_fail else 0


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    sys.exit(main(**vars(ap.parse_args())))

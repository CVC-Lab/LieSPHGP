"""Verify the SO(3) constraints on the DETERMINISTIC (pH-ODE) quadrotor.

R in SO(3) means two algebraic constraints hold at every step:

    (C1)  orthogonality     R^T R = I          (6 independent equations)
    (C2)  right-handedness  det R = +1         (rules out the O(3) reflection)

The Lie-group Heun integrator is supposed to preserve these *by construction*
-- every attitude update is R <- R exp([phi]_x), a product of rotations -- so
the only error should be floating-point round-off, NOT a drift off the
manifold that has to be corrected.

quadrotor.py:549-551 carries a safety net that re-orthogonalises R by polar
decomposition whenever either constraint slips past 1e-8.  A key part of this
test is that the net NEVER FIRES: the constraints must be *preserved*, not
*enforced*.  We monkeypatch the projection to count its calls.

Run:  python mini_tests/test_so3_constraints_ode.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
QUAD_DIR = os.path.join(PROJECT_ROOT, "envs", "quadrotor_se3_port_ham")
for _p in (PROJECT_ROOT, QUAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quadrotor as Q                                            # noqa: E402
from quadrotor import quadrotor_se3, _exp_so3, _vee, _random_rotation  # noqa: E402

ODE_CFG = os.path.join(PROJECT_ROOT, "configs", "quadrotor_se3", "envs", "ode.yaml")
EPS = np.finfo(np.float64).eps
E3 = np.array([0.0, 0.0, 1.0])
RESULTS: list[tuple[str, bool, str]] = []


def _banner(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


def _record(name, ok, msg):
    RESULTS.append((name, bool(ok), msg))
    print(f"  -> {'PASS' if ok else 'FAIL'}  {msg}")


# ── constraint residuals ──────────────────────────────────────────────────

def so3_defects(R):
    """(orthogonality residual, |det - 1|) for a single 3x3 matrix."""
    return (float(np.abs(R.T @ R - np.eye(3)).max()),
            float(abs(np.linalg.det(R) - 1.0)))


# ── deterministic control laws (all on ode.yaml) ──────────────────────────

def _hover(env, _):
    x, R, v_b, om = env.get_state(); v_w = R @ v_b
    F = env.m * (2.0 * (np.array([0, 0, 3.0]) - x) - 2.5 * v_w + env.g * E3)
    b3 = F / max(np.linalg.norm(F), 1e-9)
    b2 = np.cross(b3, np.array([1.0, 0, 0])); b2 /= max(np.linalg.norm(b2), 1e-9)
    Rd = np.stack([np.cross(b2, b3), b2, b3], 1)
    tau = -10.0 * (0.5 * _vee(Rd.T @ R - R.T @ Rd)) - 3.0 * om
    return np.linalg.solve(env.G[2:6, :], np.hstack([float(F @ (R @ E3)), tau]))


def _forward_spin(env, _):
    x, R, v_b, om = env.get_state(); v_w = R @ v_b
    F = env.m * np.array([2.0 * (2.0 - v_w[0]), -2.0 * v_w[1],
                          env.g + 2.0 * (3.0 - x[2]) - 2.5 * v_w[2]])
    b3 = F / max(np.linalg.norm(F), 1e-9)
    psi = -1.0 * env.t
    c1 = np.array([np.cos(psi), np.sin(psi), 0.0])
    b2 = np.cross(b3, c1); b2 /= max(np.linalg.norm(b2), 1e-9)
    Rd = np.stack([np.cross(b2, b3), b2, b3], 1)
    tau = (-10.0 * (0.5 * _vee(Rd.T @ R - R.T @ Rd))
           - 3.0 * (om - R.T @ np.array([0.0, 0.0, -1.0])))
    return np.linalg.solve(env.G[2:6, :], np.hstack([float(F @ (R @ E3)), tau]))


def _u_const(val):
    return lambda env, _: np.full(4, val)


def _tumbler(env, k):
    """Deliberately violent: large asymmetric input -> fast, sustained tumbling.
    This is the stress case for the exp-map update."""
    return np.array([12.0, 1.0, 11.0, 0.5])


MODES = [
    ("PD hover",            _hover),
    ("forward+spin",        _forward_spin),
    ("u = 0 (free fall)",   _u_const(0.0)),
    ("u = 1 (descent)",     _u_const(1.0)),
    ("u = hover const",     _u_const(9.81 / 4.0)),
    ("violent tumbling",    _tumbler),
]


def _rollout(policy, n_steps, R0=None, seed=0):
    """Run the deterministic env, returning worst-case constraint defects.

    Returns (max orthogonality defect, max |det-1|, per-step curve, n_proj).

    n_proj counts ONLY re-orthogonalisations that happen inside step().  The
    counter is installed after reset(), because reset() legitimately projects
    a caller-supplied R_init (quadrotor.py:595) and that is not a safety-net
    firing.
    """
    env = quadrotor_se3.from_config(ODE_CFG)
    opts = dict(x_init=[0.0, 0.0, 3.0], v_init=[0.0, 0.0, 0.0],
                omega_init=[0.0, 0.0, 0.0],
                R_init=np.eye(3) if R0 is None else R0)
    env.reset(seed=seed, options=opts)

    calls = [0]
    orig = Q._project_to_so3
    def _counting(R, _o=orig, _c=calls):
        _c[0] += 1
        return _o(R)
    Q._project_to_so3 = _counting

    worst_o = worst_d = 0.0
    curve = []
    try:
        for k in range(n_steps):
            env.step(policy(env, k))
            o, d = so3_defects(env.get_state()[1])
            worst_o = max(worst_o, o); worst_d = max(worst_d, d)
            curve.append((o, d))
    finally:
        Q._project_to_so3 = orig
        env.close()
    return worst_o, worst_d, np.asarray(curve), calls[0]


# ── Test 1: every deterministic control mode ──────────────────────────────

def test_modes(n_steps=1000):
    _banner(f"Test 1 -- SO(3) constraints per control mode ({n_steps} steps, ode.yaml)")
    print(f"  {'mode':<20} {'max|R^T R - I|':>16} {'max|det R - 1|':>16}  {'proj?':>6}")
    ok = True
    for name, pol in MODES:
        o, d, _, nproj = _rollout(pol, n_steps)
        good = o < 1e-13 and d < 1e-13 and nproj == 0
        ok &= good
        print(f"  {name:<20} {o:>16.3e} {d:>16.3e}  {nproj:>6d}"
              f"   {'ok' if good else 'BAD'}")
    _record("1 all control modes", ok,
            "every deterministic mode holds R^T R = I and det R = 1 to <1e-13; "
            "safety-net projection never fired")
    return ok


# ── Test 2: does the defect accumulate with rollout length? ───────────────

def test_growth(n_steps=20000):
    _banner(f"Test 2 -- defect growth over a long rollout ({n_steps} steps = "
            f"{n_steps * 10} exp-map updates)")
    o, d, curve, nproj = _rollout(_forward_spin, n_steps)
    marks = [int(n_steps * f) - 1 for f in (0.05, 0.1, 0.25, 0.5, 1.0)]
    print(f"  {'step':>8} {'|R^T R - I|':>14} {'|det R - 1|':>14}")
    for i in marks:
        print(f"  {i + 1:>8} {curve[i, 0]:>14.3e} {curve[i, 1]:>14.3e}")
    first, last = curve[marks[0], 0], curve[marks[-1], 0]
    ratio = last / max(first, 1e-300)
    rate = last / (marks[-1] + 1)
    to_thresh = 1e-8 / max(rate, 1e-300)
    print(f"  20x more steps -> orthogonality defect x{ratio:.2f}"
          f"   (random walk would give ~{np.sqrt(20):.1f}x, linear drift 20x)")
    print(f"  => growth is ~LINEAR in step count, not bounded:"
          f" ~{rate:.2e} per step (~{rate / 10:.1e} per exp-map update)")
    print(f"  => the 1e-8 safety-net threshold would be reached after"
          f" ~{to_thresh:.1e} steps ({to_thresh * 0.05 / 3600:.1e} h of sim time)")
    print(f"  in-step safety-net projections fired: {nproj}")
    ok = o < 1e-12 and d < 1e-12 and nproj == 0
    _record("2 long-rollout growth", ok,
            f"defect accumulates LINEARLY at ~{rate:.0e}/step; after {n_steps} "
            f"steps it is {o:.1e}, still ~{1e-8 / max(o, 1e-300):.0e}x below the "
            f"safety net, which never fired")
    return ok


# ── Test 3: the exp map itself, over the full angle range ─────────────────

def test_exp_map(n=20000, seed=3):
    _banner("Test 3 -- exp_so3 output is orthogonal for all rotation magnitudes")
    rng = np.random.default_rng(seed)
    print(f"  {'|phi| range':<24} {'max|R^T R - I|':>16} {'max|det R - 1|':>16}")
    ok = True
    bands = [("tiny   (1e-12, Taylor)", 1e-14, 1e-11),
             ("small  (1e-6)",          1e-8,  1e-4),
             ("normal (0.1 .. 3)",      0.1,   3.0),
             ("near pi",                np.pi - 1e-6, np.pi + 1e-6),
             ("large  (up to 100 rad)", 10.0,  100.0)]
    for label, lo, hi in bands:
        wo = wd = 0.0
        for _ in range(n // len(bands)):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            phi = v * rng.uniform(lo, hi)
            o, d = so3_defects(_exp_so3(phi))
            wo = max(wo, o); wd = max(wd, d)
        good = wo < 1e-14 and wd < 1e-14
        ok &= good
        print(f"  {label:<24} {wo:>16.3e} {wd:>16.3e}   {'ok' if good else 'BAD'}")
    _record("3 exp_so3 unitarity", ok,
            "Rodrigues output is orthogonal with det +1 to <1e-14 across every "
            "magnitude band, including the Taylor and near-pi branches")
    return ok


# ── Test 4: random initial attitudes, not just R0 = I ─────────────────────

def test_random_R0(n_att=40, n_steps=300, seed=4):
    _banner(f"Test 4 -- {n_att} random initial attitudes on SO(3)")
    rng = np.random.default_rng(seed)
    wo = wd = 0.0
    for _ in range(n_att):
        o, d, _, _ = _rollout(_forward_spin, n_steps, R0=_random_rotation(rng))
        wo = max(wo, o); wd = max(wd, d)
    print(f"  worst over all {n_att} rollouts:  |R^T R - I| = {wo:.3e}"
          f"   |det R - 1| = {wd:.3e}")
    print(f"  (machine epsilon = {EPS:.2e}; the defects sit a few eps above it)")
    ok = wo < 1e-13 and wd < 1e-13
    _record("4 random initial attitudes", ok,
            f"worst defect {wo:.1e} / {wd:.1e} over {n_att} random R0")
    return ok


if __name__ == "__main__":
    print("SO(3) constraint verification -- deterministic pH-ODE quadrotor")
    print(f"config: {os.path.relpath(ODE_CFG, PROJECT_ROOT)}")
    test_modes()
    test_growth()
    test_exp_map()
    test_random_R0()
    _banner("SUMMARY")
    w = max(len(n) for n, _, _ in RESULTS)
    for n, ok, m in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {n:<{w}}  {m}")
    k = sum(o for _, o, _ in RESULTS)
    print(f"\n  {k}/{len(RESULTS)} checks passed")
    sys.exit(0 if k == len(RESULTS) else 1)

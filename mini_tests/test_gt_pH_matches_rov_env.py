"""Checks of the BlueROV2 port-Hamiltonian env (envs/rov_se3_port_ham/bluerov2.py).

    python mini_tests/test_gt_pH_matches_rov_env.py

1  Coriolis term conserves energy: nu . J(P) nu = 0
2  restoring force = -grad V (finite differences in z and on SO(3))
3  power balance: dH/dt = nu.tau - nu.D(nu)nu along trajectories (Heun steps, O(h^2) residual)
4  free, undamped, unforced motion conserves H to O(h^2)
5  R stays on SO(3) (orthogonality and det)
6  term-by-term agreement with the Fossen model used as baseline in Marinarium (fossen/BlueROV2_wrench.py, if present)
7  allocation E: rank 6 and the expected surge / heave / yaw signs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.rov_se3_port_ham.bluerov2 import BlueROV2, exp_so3, allocation_matrix, T200  # noqa: E402

RNG = np.random.default_rng(0)
results = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:42s} {detail}")


def random_state():
    return RNG.normal(size=3), exp_so3(RNG.normal(size=3)), RNG.normal(scale=0.5, size=6)


def roll(env, x, R, nu, tau, h, steps):
    H = [env.hamiltonian(x, R, nu)]; work = 0.0; diss = 0.0
    for _ in range(steps):
        x2, R2, nu2 = env.heun_step(x, R, nu, tau, h)
        nu_mid = 0.5 * (nu + nu2)
        work += float(nu_mid @ tau) * h
        diss += 0.5 * (float(nu @ -env.damping_force(nu)) + float(nu2 @ -env.damping_force(nu2))) * h
        x, R, nu = x2, R2, nu2; H.append(env.hamiltonian(x, R, nu))
    return np.asarray(H), work, diss, R


env = BlueROV2()
print("BlueROV2 Heavy pH env: M diag", np.round(env.M, 3), "| W - B =", round(env.p.weight - env.p.buoyancy, 3), "N")

worst = max(abs(float(nu @ env.coriolis_force(nu))) for _, _, nu in (random_state() for _ in range(200)))
check("1 Coriolis power nu.J(P)nu = 0", worst < 1e-12, f"max |power| {worst:.1e}")

errs = []
for _ in range(50):
    x, R, _ = random_state(); e = 1e-6
    fz = -(env.potential(x + e * np.array([0, 0, 1]), R) - env.potential(x - e * np.array([0, 0, 1]), R)) / (2 * e)
    tq = np.array([-(env.potential(x, R @ exp_so3(e * k)) - env.potential(x, R @ exp_so3(-e * k))) / (2 * e) for k in np.eye(3)])
    g = env.restoring(R)
    errs.append(max(abs(fz - float((R @ g[:3])[2])), np.abs(tq - g[3:]).max()))
check("2 restoring = -grad V", max(errs) < 1e-6, f"max error {max(errs):.1e}")

x0, R0, nu0 = random_state(); tau = RNG.normal(scale=5, size=6); residuals = []
for h in (0.01, 0.005):
    H, work, diss, _ = roll(env, x0, R0, nu0, tau, h, int(1.0 / h))
    residuals.append((H[-1] - H[0]) - (work - diss))
    print(f"      h = {h}: dH {H[-1] - H[0]:+.4f}, input work {work:+.4f}, dissipated {diss:.4f}, residual {residuals[-1]:+.2e}")
check("3 power balance dH = work - dissipation", abs(residuals[1]) < 1e-3 * abs(work) and abs(residuals[0] / residuals[1]) > 3,
      f"residual {residuals[0]:.1e} -> {residuals[1]:.1e} (ratio {residuals[0] / residuals[1]:.1f}, O(h^2) gives 4)")

free = BlueROV2(dissipation={"linear": "none", "angular": "none"})
drifts = []
x0, R0, nu0 = random_state()
for h in (0.01, 0.005):
    H, _, _, R_end = roll(free, x0, R0, nu0, np.zeros(6), h, int(5.0 / h))
    drifts.append(np.abs(H - H[0]).max() / max(abs(H[0]), 1.0))
check("4 free motion conserves H (O(h^2))", drifts[1] < 1e-3 and drifts[0] / drifts[1] > 3, f"rel. drift {drifts[0]:.1e} -> {drifts[1]:.1e} (ratio {drifts[0] / drifts[1]:.1f})")
check("5 R stays on SO(3)", np.abs(R_end.T @ R_end - np.eye(3)).max() < 1e-10 and abs(np.linalg.det(R_end) - 1) < 1e-10,
      f"|R^T R - I| {np.abs(R_end.T @ R_end - np.eye(3)).max():.1e}")

fossen = ROOT / "envs/rov_se3_marinarium/marinarium_raw/fossen"
if (fossen / "BlueROV2_wrench.py").exists():
    sys.path.insert(0, str(fossen))
    from BlueROV2_wrench import BlueROV2 as Theirs  # noqa: E402
    them = Theirs()
    diffs = {"M": np.abs(np.diag(them.M) - env.M).max(), "D": 0.0, "g": 0.0, "C": 0.0}
    for _ in range(200):
        _, R, nu = random_state()
        diffs["D"] = max(diffs["D"], np.abs(-them._damping(nu) @ nu - env.damping_force(nu)).max())
        diffs["g"] = max(diffs["g"], np.abs(-them._restoring_from_R(R) - env.restoring(R)).max())
        diffs["C"] = max(diffs["C"], np.abs(-them._coriolis(nu) @ nu - env.coriolis_force(nu)).max())
    print("      max difference to their terms:", {k: f"{v:.1e}" for k, v in diffs.items()})
    check("6 M, D, g agree with their Fossen model", max(diffs["M"], diffs["D"], diffs["g"]) < 1e-9, "")
    check("6b Coriolis agrees with their (corrected) C", diffs["C"] < 1e-9, f"max {diffs['C']:.1e}")
else:
    print("      (6 skipped: marinarium_raw not present)")

E = allocation_matrix()
check("7 allocation rank 6, surge/heave/yaw signs", np.linalg.matrix_rank(E) == 6 and E[0, :4].sum() > 0 and np.abs(E[2, 4:]).sum() > 0,
      f"rank {np.linalg.matrix_rank(E)}, surge from T1-4 {E[0, :4].round(3)}, yaw {E[5, :4].round(3)}")
t200 = T200()
lo, hi = t200.limits(16.0)
check("8 T200 map: limits and inverse", abs(hi - 51.44) < 0.1 and abs(t200.command_from_thrust(t200.thrust_from_command(np.array([0.6]), 16.0), 16.0)[0] - 0.6) < 1e-3,
      f"16 V thrust range [{lo:.1f}, {hi:.1f}] N")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

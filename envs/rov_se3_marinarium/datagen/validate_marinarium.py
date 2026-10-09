"""Checks of the raw Marinarium recordings (npz from bag_to_npz.py): frames, sensors and the thruster input.

    python envs/rov_se3_marinarium/datagen/validate_marinarium.py [--recording manual]

A  pose vs integrated motion-capture twist over 100 ms (body frame), receive and header clocks; robust (median, p95)
B  PX4 gyro vs motion-capture body rate (same axes? bias, scale), glitch windows excluded
F  pose glitches: attitude jumps > 3 deg in one step that the gyro does not see (marker swaps / flips)
C  accelerometer vs gravity: world z DOWN or UP
D  thrust input E @ T200(u, V) vs the force the motion needs under the published model (per axis corr, gain)
E  collinearity of the 8 motor commands (which thruster columns of G the data can identify)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from envs.rov_se3_marinarium.bluerov2 import BlueROV2, T200  # noqa: E402

NPZ = THIS_DIR.parent / "marinarium_raw" / "npz"
corr = lambda A, B: np.array([np.corrcoef(A[:, i], B[:, i])[0, 1] for i in range(A.shape[1])])


def clean_windows(t: np.ndarray, k: int) -> np.ndarray:
    i = np.arange(len(t) - k)
    return i[(t[i + k] - t[i]) < k * np.median(np.diff(t)) * 1.2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", default="manual", help="oct30 | manual | stabilized")
    args = parser.parse_args()
    d = np.load(NPZ / f"{args.recording}.npz")
    t, R = d["mocap_t"], Rotation.from_quat(d["mocap_quaternion_xyzw"]).as_matrix()
    p, v, w = d["mocap_position"], d["mocap_v_body"], d["mocap_omega_body"]
    gaps = np.diff(t)
    print(f"recording {args.recording}: {t[-1]:.0f} s, mocap {1 / np.median(gaps):.0f} Hz, {int(np.sum(gaps > 0.05))} dropouts > 50 ms "
          f"(longest {gaps.max():.2f} s), battery {d['battery_voltage'].min():.2f}-{d['battery_voltage'].max():.2f} V")

    gyro_t, gyro = d["imu_t"], d["imu_gyro"]
    Rr = Rotation.from_quat(d["mocap_quaternion_xyzw"])
    step_deg = np.degrees((Rr[:-1].inv() * Rr[1:]).magnitude())
    gyro_deg = np.degrees(np.linalg.norm(np.stack([np.interp(t[1:], gyro_t, gyro[:, j]) for j in range(3)], 1), axis=1) * gaps)
    glitch = (step_deg > 5 * np.maximum(gyro_deg, 0.2)) & (step_deg > 3)
    print("\nA  pose vs integrated twist over 100 ms (median, p95), dropouts excluded")
    for clock in ("mocap_t", "mocap_t_header"):
        tc = d[clock]; dtc = np.diff(tc); k = 10
        cp = np.vstack([np.zeros(3), np.cumsum(np.einsum("kij,kj->ki", R, v)[:-1] * dtc[:, None], 0)])
        cw = np.vstack([np.zeros(3), np.cumsum(w[:-1] * dtc[:, None], 0)])
        i = clean_windows(tc, k)
        pos = np.linalg.norm((p[i + k] - p[i]) - (cp[i + k] - cp[i]), axis=1)
        rot = Rotation.from_matrix(np.einsum("kji,kjl->kil", R[i], R[i + k])).as_rotvec()
        att = np.degrees(np.linalg.norm(rot - (cw[i + k] - cw[i]), axis=1))
        print(f"   {clock:15s} position {np.median(pos) * 1e3:.2f} mm (p95 {np.percentile(pos, 95) * 1e3:.1f}), attitude {np.median(att):.3f} deg "
              f"(p95 {np.percentile(att, 95):.2f}); motion in 100 ms: {np.median(np.linalg.norm(p[i + k] - p[i], axis=1)) * 1e3:.0f} mm, "
              f"{np.median(np.degrees(np.linalg.norm(rot, axis=1))):.2f} deg")

    k = 10; i = clean_windows(t, k); cb = np.r_[0, np.cumsum(glitch)]; i = i[(cb[i + k] - cb[i]) == 0]; dt = (t[i + k] - t[i])[:, None]
    iw = np.vstack([np.zeros(3), np.cumsum(w[1:] * gaps[:, None], 0)])
    w_mocap = (iw[i + k] - iw[i]) / dt
    cg = np.vstack([np.zeros(3), np.cumsum(gyro[1:] * np.diff(gyro_t)[:, None], 0)])
    g_avg = np.stack([np.interp(t[i + k], gyro_t, cg[:, j]) - np.interp(t[i], gyro_t, cg[:, j]) for j in range(3)], 1) / dt
    res = g_avg - w_mocap
    print("\nB  PX4 gyro vs motion-capture body rate, 100 ms means (identity axis map, glitch windows excluded)")
    print(f"   corr {np.round(corr(g_avg, w_mocap), 3)}  bias {np.round(np.median(res, 0), 4)} rad/s  residual MAD {np.round(np.median(np.abs(res - np.median(res, 0)), 0), 3)}  "
          f"scale {np.round([np.polyfit(w_mocap[:, j], g_avg[:, j], 1)[0] for j in range(3)], 3)}")
    print(f"\nF  pose glitches: {int(glitch.sum())} one-step attitude jumps > 3 deg not seen by the gyro (largest {step_deg[glitch].max() if glitch.any() else 0:.0f} deg)")

    acc = np.stack([np.interp(t, gyro_t, d["imu_accel"][:, j]) for j in range(3)], 1)
    for label, g in (("z DOWN", [0, 0, 9.81]), ("z UP", [0, 0, -9.81])):
        print(f"\nC  world {label}: RMS(accel + R^T g) = {np.sqrt(np.mean((acc + np.einsum('kji,j->ki', R, g)) ** 2)):.2f} m/s^2") if label == "z DOWN" \
            else print(f"   world {label}: RMS(accel + R^T g) = {np.sqrt(np.mean((acc + np.einsum('kji,j->ki', R, g)) ** 2)):.2f} m/s^2")

    env, t200 = BlueROV2(), T200()
    grid = np.arange(t[0] + 0.2, t[-1] - 0.2, 0.1)
    big = gaps > 0.05; gs, ge = t[:-1][big], t[1:][big]
    grid = grid[[not np.any((gs < x + 0.15) & (ge > x - 0.15)) for x in grid]]
    nu = np.c_[v, w]
    nu_at = lambda s: np.stack([np.interp(s, t, nu[:, j]) for j in range(6)], 1)
    nu_g, nu_dot = nu_at(grid), (nu_at(grid + 0.05) - nu_at(grid - 0.05)) / 0.1
    Rg = Slerp(t, Rr)(grid).as_matrix()                                  # slerp (q and -q are the same rotation)
    needed = np.array([env.M * a - env.coriolis_force(n) - env.damping_force(n) - env.restoring(r) for a, n, r in zip(nu_dot, nu_g, Rg)])
    cmd = np.stack([np.interp(grid, d["motor_t"], np.nan_to_num(d["motor_command"][:, j])) for j in range(8)], 1)
    applied = t200.thrust_from_command(cmd, np.interp(grid, d["battery_t"], d["battery_voltage"])) @ env.E.T
    print(f"\nD  E @ T200(u, V) vs M nu_dot - J - D - g (published model), {len(grid)} samples at 10 Hz")
    for j, name in enumerate(("X surge", "Y sway", "Z heave", "K roll", "M pitch", "N yaw")):
        c = np.corrcoef(applied[:, j], needed[:, j])[0, 1]
        print(f"   {name:8s} corr {c:+.2f}   gain needed/applied {np.polyfit(applied[:, j], needed[:, j], 1)[0]:+.2f}")

    U = np.nan_to_num(d["motor_command"])
    C = np.corrcoef(U.T)
    pairs = [(a + 1, b + 1, C[a, b]) for a in range(8) for b in range(a + 1, 8) if abs(C[a, b]) > 0.9]
    print(f"\nE  command collinearity: {np.mean(np.isnan(d['motor_command']).all(1)):.1%} of motor messages have every motor off (coasting)")
    print("   pairs with |corr| > 0.9 (their thruster columns of G cannot be separated):",
          ", ".join(f"u{a}/u{b} {c:+.2f}" for a, b, c in pairs) or "none")
    print(f"   singular values of the command matrix (relative): {np.round(np.linalg.svd(U - U.mean(0), compute_uv=False) / np.linalg.svd(U - U.mean(0), compute_uv=False)[0], 3)}")


if __name__ == "__main__":
    main()

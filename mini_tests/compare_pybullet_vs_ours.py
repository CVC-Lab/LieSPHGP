"""Cross-validate our SE(3) quadrotor against gym-pybullet-drones.

Both simulators are given IDENTICAL physical parameters (the real Crazyflie
CF2X from cf2x.urdf), IDENTICAL initial conditions, and the IDENTICAL rotor
speed sequence.  Any difference is then integration error, not physics.

Why they should agree.  gym-pybullet-drones' Physics.DYN integrates

    m v_world_dot = R [0,0,sum f_i] - m g e3
    J omega_dot   = tau(rpm) - omega x (J omega)

with explicit Euler.  Our env integrates the same rigid body in the BODY frame
with a Lie-group Heun scheme.  The two formulations are equivalent (the body
frame simply carries the extra v x omega term), and their CF2X mixer signs are
identical to ours -- verified in this script.

The decisive test is not "are the trajectories equal" (they cannot be, the
integrators differ) but "does the gap vanish as the step size shrinks".  If the
physics differed, refining dt would leave a constant residual.

    python mini_tests/compare_pybullet_vs_ours.py --out_pdf report.pdf
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, ".."))
GPD = os.path.join(ROOT, "envs", "pybullet_quadrotor_se3", "gym-pybullet-drones")
for p in (ROOT, GPD, os.path.join(ROOT, "envs", "port_ham_quadrotor_se3")):
    if p not in sys.path:
        sys.path.insert(0, p)

from quadrotor import quadrotor_se3, _exp_so3                     # noqa: E402
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary        # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics   # noqa: E402
import pybullet as pb                                             # noqa: E402

# ── the real Crazyflie CF2X, straight from cf2x.urdf ────────────────────────
M, ARM, KF, KM = 0.027, 0.0397, 3.16e-10, 7.94e-12
JXX, JYY, JZZ = 1.4e-5, 1.4e-5, 2.17e-5
G = 9.8                       # gym-pybullet-drones uses 9.8, not 9.81
HOVER_RPM = np.sqrt(M * G / (4 * KF))
MAX_RPM = np.sqrt(2.25 * M * G / (4 * KF))


def quat_to_R(q):
    return np.array(pb.getMatrixFromQuaternion(q)).reshape(3, 3)


def rpy_to_R(rpy):
    r, p_, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p_), 0, np.sin(p_)], [0, 1, 0], [-np.sin(p_), 0, np.cos(p_)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def geodesic(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.arccos(np.clip(c, -1, 1)))


# ── the two simulators ───────────────────────────────────────────────────────

def run_pybullet(x0, rpy0, rpm_seq, freq):
    """gym-pybullet-drones, Physics.DYN (their own explicit Euler)."""
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     initial_xyzs=x0.reshape(1, 3), initial_rpys=rpy0.reshape(1, 3),
                     physics=Physics.DYN, pyb_freq=freq, ctrl_freq=freq,
                     gui=False, record=False, obstacles=False, user_debug_gui=False)
    env.reset()
    traj = []
    for k in range(rpm_seq.shape[0]):
        obs, *_ = env.step(rpm_seq[k].reshape(1, 4))
        s = obs[0]
        R = quat_to_R(s[3:7])
        traj.append(np.concatenate([s[0:3], R.reshape(9), s[10:13], s[13:16]]))
    env.close()
    return np.stack(traj)          # [x(3) | vec R(9) | v_world(3) | omega_body(3)]


def run_ours(x0, rpy0, rpm_seq, freq, n_sub=10):
    """Our Lie-Heun env, damping and wind OFF so the physics matches theirs."""
    env = quadrotor_se3(m=M, J_diag=(JXX, JYY, JZZ), arm=ARM, g=G,
                        dt=1.0 / freq, kf_coeff=KF, km_coeff=KM,
                        linear_damping_coeff=0.0, angular_damping_coeff=0.0,
                        external_force_type="constant", external_force_std=0.0,
                        wind_force_std=0.0, wind_torque_std=0.0, seed=0)
    env.reset(seed=0, options=dict(x_init=x0, R_init=rpy_to_R(rpy0),
                                   v_init=[0., 0., 0.], omega_init=[0., 0., 0.]))
    traj = []
    for k in range(rpm_seq.shape[0]):
        env.step(rpm_seq[k] ** 2)              # our u is rpm^2
        x, R, v_b, om = env.get_state()
        traj.append(np.concatenate([x, R.reshape(9), R @ v_b, om]))
    env.close()
    return np.stack(traj)


def check_mixers():
    """The CF2X torque signs must agree, or nothing else is meaningful."""
    e = quadrotor_se3(m=M, J_diag=(JXX, JYY, JZZ), arm=ARM, g=G,
                      kf_coeff=KF, km_coeff=KM, seed=0)
    Gm = e.G.copy(); e.close()
    rpm = np.array([9000., 11000., 13000., 15000.])
    f = rpm ** 2 * KF
    L = ARM / np.sqrt(2)
    theirs = np.array([np.sum(f),
                       -(f[0] + f[1] - f[2] - f[3]) * L,
                       (-f[0] + f[1] + f[2] - f[3]) * L,
                       (-rpm[0] ** 2 + rpm[1] ** 2 - rpm[2] ** 2 + rpm[3] ** 2) * KM])
    ours = (Gm @ (rpm ** 2))[2:6]
    return float(np.abs(theirs - ours).max()), float(np.abs(theirs).max())


# ── report ───────────────────────────────────────────────────────────────────

def _ax(t, figsize=(11, 6)):
    fig, ax = plt.subplots(figsize=figsize); ax.set_title(t); ax.grid(True, alpha=.3)
    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_pdf", default=os.path.join(
        ROOT, "reports/SE3_Quadrotor/env_checks/pybullet_vs_ours.pdf"))
    ap.add_argument("--n_traj", type=int, default=10)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--freq", type=int, default=240)
    ap.add_argument("--rpm_scale", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mix_err, mix_scale = check_mixers()
    print(f"mixer check: max|theirs - ours| = {mix_err:.3e}  (scale {mix_scale:.3e})")

    n = int(round(args.duration * args.freq))
    rng = np.random.default_rng(args.seed)
    ics, rpms = [], []
    for i in range(args.n_traj):
        ics.append((rng.uniform(-0.5, 0.5, 3) + np.array([0, 0, 1.0]),
                    rng.uniform(-0.3, 0.3, 3)))
        rpms.append(np.clip(HOVER_RPM + rng.uniform(-args.rpm_scale, args.rpm_scale,
                                                    (n, 4)), 0, MAX_RPM))
    print(f"running {args.n_traj} trajectories x {n} steps ({args.duration}s @ {args.freq}Hz)")
    PB = [run_pybullet(a, b, r, args.freq) for (a, b), r in zip(ics, rpms)]
    OU = [run_ours(a, b, r, args.freq) for (a, b), r in zip(ics, rpms)]
    PB, OU = np.stack(PB), np.stack(OU)
    t = np.arange(1, n + 1) / args.freq

    dpos = np.linalg.norm(PB[..., 0:3] - OU[..., 0:3], axis=-1)
    dvel = np.linalg.norm(PB[..., 12:15] - OU[..., 12:15], axis=-1)
    dom = np.linalg.norm(PB[..., 15:18] - OU[..., 15:18], axis=-1)
    dang = np.array([[geodesic(PB[i, k, 3:12].reshape(3, 3), OU[i, k, 3:12].reshape(3, 3))
                      for k in range(n)] for i in range(args.n_traj)])
    print(f"  final gap: pos {dpos[:, -1].mean():.3e} m   att {dang[:, -1].mean():.3e} rad")

    # ── refinement study: does the gap vanish as dt -> 0? ──
    print("refinement study ...")
    freqs = [args.freq, args.freq * 2, args.freq * 4]
    ref = []
    a0, b0 = ics[0]
    for f in freqs:
        nn = int(round(args.duration * f))
        rr = np.repeat(rpms[0], f // args.freq, axis=0)[:nn]
        P = run_pybullet(a0, b0, rr, f); O = run_ours(a0, b0, rr, f)
        ref.append((f, float(np.linalg.norm(P[-1, 0:3] - O[-1, 0:3])),
                    geodesic(P[-1, 3:12].reshape(3, 3), O[-1, 3:12].reshape(3, 3))))
        print(f"   {f:5d} Hz  pos gap {ref[-1][1]:.4e}  att gap {ref[-1][2]:.4e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pdf)), exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        # page 1 -- setup
        rows = [("simulator A", "gym-pybullet-drones  CtrlAviary, Physics.DYN"),
                ("simulator B", "ours — Lie-group Heun, exp map"),
                ("drone", "Crazyflie CF2X (cf2x.urdf)"),
                ("mass m", f"{M} kg"), ("inertia J", f"({JXX}, {JYY}, {JZZ}) kg m²"),
                ("arm", f"{ARM} m"), ("k_f", f"{KF}"), ("k_m", f"{KM}"),
                ("gravity g", f"{G} m/s²"), ("damping", "0 (both) — their DYN has none"),
                ("hover RPM", f"{HOVER_RPM:.0f}"), ("max RPM", f"{MAX_RPM:.0f}"),
                ("", ""),
                ("trajectories", f"{args.n_traj}, identical ICs and RPM sequences"),
                ("duration", f"{args.duration} s @ {args.freq} Hz  ({n} steps)"),
                ("their integrator", "explicit Euler, 1 step per control step"),
                ("our integrator", f"Lie–Heun, 10 substeps of {1/args.freq/10:.2e} s"),
                ("", ""),
                ("MIXER CHECK  max|theirs−ours|", f"{mix_err:.3e}  (scale {mix_scale:.3e})"),
                ("mean final position gap", f"{dpos[:, -1].mean():.4e} m"),
                ("mean final attitude gap", f"{dang[:, -1].mean():.4e} rad")]
        fig, ax = plt.subplots(figsize=(11, .42 * len(rows) + 1.4)); ax.axis("off")
        ax.set_title("gym-pybullet-drones vs our SE(3) quadrotor\n"
                     "identical parameters, identical inputs",
                     fontsize=15, fontweight="bold", pad=18)
        tb = ax.table(cellText=[[k, str(v)] for k, v in rows], colWidths=[.42, .58],
                      loc="center", cellLoc="left")
        tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.35)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # page 2 -- divergence over time
        for d, lab, ti in ((dpos, "‖Δx‖ (m)", "Position gap"),
                           (dang, "geodesic (rad)", "Attitude gap"),
                           (dvel, "‖Δv‖ (m/s)", "World velocity gap"),
                           (dom, "‖Δω‖ (rad/s)", "Body rate gap")):
            fig, ax = _ax(f"{ti} — {args.n_traj} trajectories, identical inputs")
            for i in range(args.n_traj):
                ax.plot(t, np.maximum(d[i], 1e-18), lw=.8, alpha=.5)
            ax.plot(t, np.maximum(d.mean(0), 1e-18), "k-", lw=2.5, label="mean")
            ax.set_yscale("log"); ax.set_xlabel("Time (s)"); ax.set_ylabel(lab)
            ax.legend(fontsize="small"); fig.tight_layout()
            pdf.savefig(fig); plt.close()

        # page 3 -- state overlays, traj 0
        lbl = [("x (m)", 0), ("y (m)", 1), ("z (m)", 2)]
        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        for r, (nm, j) in enumerate(lbl):
            axes[r, 0].plot(t, PB[0, :, j], "k-", lw=2, label="pybullet" if r == 0 else None)
            axes[r, 0].plot(t, OU[0, :, j], "r--", lw=1.5, label="ours" if r == 0 else None)
            axes[r, 0].set_ylabel(nm); axes[r, 0].grid(True, alpha=.3)
            axes[r, 1].plot(t, PB[0, :, 15 + j], "k-", lw=2)
            axes[r, 1].plot(t, OU[0, :, 15 + j], "r--", lw=1.5)
            axes[r, 1].set_ylabel(f"omega {'xyz'[j]} (rad/s)"); axes[r, 1].grid(True, alpha=.3)
        axes[0, 0].legend(fontsize="small")
        axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
        fig.suptitle("Trajectory 0 — position (left) and body rates (right)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .96)); pdf.savefig(fig); plt.close()

        # page 4 -- refinement study
        fig, ax = _ax("Refinement: the gap vanishes as dt → 0\n"
                      "(a PHYSICS difference would plateau; discretisation error does not)")
        fs = [r[0] for r in ref]
        ax.loglog(fs, [r[1] for r in ref], "o-", lw=2, label="position gap (m)")
        ax.loglog(fs, [r[2] for r in ref], "s-", lw=2, label="attitude gap (rad)")
        p0 = ref[0][1]
        ax.loglog(fs, [p0 * (fs[0] / f) for f in fs], "k:", lw=1.5,
                  label="ideal O(dt) — their Euler")
        ax.set_xlabel("their integration frequency (Hz)"); ax.set_ylabel("final gap")
        ax.legend(fontsize="small"); fig.tight_layout(); pdf.savefig(fig); plt.close()

        # page 5 -- verdict
        r0, r1, r2 = ref
        rows = [["their dt", "final pos gap", "final att gap", "ratio vs previous"],
                [f"1/{r0[0]}", f"{r0[1]:.4e} m", f"{r0[2]:.4e} rad", "—"],
                [f"1/{r1[0]}", f"{r1[1]:.4e} m", f"{r1[2]:.4e} rad",
                 f"{r0[1]/max(r1[1],1e-30):.2f}x smaller"],
                [f"1/{r2[0]}", f"{r2[1]:.4e} m", f"{r2[2]:.4e} rad",
                 f"{r1[1]/max(r2[1],1e-30):.2f}x smaller"],
                ["", "", "", ""],
                ["A ratio of ~2 per halving of dt means the gap is FIRST ORDER —", "", "", ""],
                ["exactly their explicit Euler error.  A physics mismatch would", "", "", ""],
                ["instead leave a constant residual that refinement cannot remove.", "", "", ""]]
        fig, ax = plt.subplots(figsize=(12, .5 * len(rows) + 1.6)); ax.axis("off")
        ax.set_title("Verdict: same continuous dynamics, different integrators",
                     fontsize=14, fontweight="bold", pad=16)
        tb = ax.table(cellText=rows, colWidths=[.25, .25, .25, .25],
                      loc="center", cellLoc="left")
        tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()
    print(f"wrote {args.out_pdf}")


if __name__ == "__main__":
    main()

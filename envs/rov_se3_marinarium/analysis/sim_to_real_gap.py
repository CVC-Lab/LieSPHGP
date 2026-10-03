"""Sim-to-real gap of the published BlueROV2 physics on the real Marinarium recordings (paper motivation).

    python envs/rov_se3_marinarium/analysis/sim_to_real_gap.py [--skip-their-fossen]

Writes experiments/rov_se3/analysis/sim_to_real_gap/{report.pdf, results.json, results.md}.

1  Paper protocol (Torroba et al., arXiv:2602.23053, Sec. IV-D, Table 2): rows of their CSV koopman_dataset_50Hz.csv
   (manual recording), chronological 80/20, test N = 9165; 12-D state [x y z phi theta psi u v w p q r] (unwrapped ZYX Euler);
   open loop from every test row with the recorded inputs; endpoint RMSE_H = sqrt(sum ||x_{k+H} - xhat_{k+H|k}||^2 / ((N-H) n))
   for H = 1, 10, 100 (0.02, 0.2, 2 s).
     - their Fossen baseline, their code (fossen/BlueROV2.py: own thrust polynomial + 3rd-order motor lag, explicit Euler)
     - our simulator (envs/rov_se3_port_ham/bluerov2.py): same published parameters, T200 thrust at the measured battery
       voltage, PX4 thruster geometry, Lie-group Heun (10 substeps per sample)
     - persistence (x_{k+H} = x_k): the floor any model must beat
2  Clean protocol: the dropout- and glitch-free test trajectories of datasets/ROV-MARINARIUM-DATASET-PAPER-MANUAL,
   position / attitude / twist error vs horizon (0.1 - 10 s), our simulator vs persistence.
3  Per-axis gap: accel_j = alpha_j thrust_j/M_j - beta_j drag_j/M_j + c_j (+ known Coriolis / restoring), 0.2 s means.
   alpha = beta = 1 if the published model were right.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

THIS_DIR = Path(__file__).resolve().parent
ENV_DIR = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parents[2]
RAW = ENV_DIR / "marinarium_raw"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from envs.rov_se3_port_ham.bluerov2 import BlueROV2, T200  # noqa: E402

CSV = RAW / "rosbags/rosbag2_2025_11_06/rosbag2_2025_11_06-manual/koopman_dataset_50Hz.csv"
NPZ = RAW / "npz/manual.npz"
DATASET = PROJECT_ROOT / "datasets/ROV-MARINARIUM-DATASET-PAPER-MANUAL/PAPER-MANUAL_clean.pkl"
OUT = PROJECT_ROOT / "experiments/rov_se3/analysis/sim_to_real_gap"
PAPER_TABLE2 = {"Koopman (EDMDc-RBF)": [0.0629, 0.0831, 0.1859], "Double Integrator-like (DI)": [0.0784, 0.1088, 0.4683],
                "Fossen (BlueROV2)": [0.0765, 0.2122, 0.5788], "PINc (ResDNN)": [8.7886, 9.1550, 9.1639]}
HORIZONS = (1, 10, 100)
AXES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


# --------------------------------------------------------------------------- vectorised simulator (many starts at once)
def batch_nu_dot(env: BlueROV2, R: np.ndarray, nu: np.ndarray, tau: np.ndarray) -> np.ndarray:
    M = env.M
    v, w = nu[:, :3], nu[:, 3:]
    cor = np.c_[np.cross(M[:3] * v, w), np.cross(M[:3] * v, v) + np.cross(M[3:] * w, w)]
    damp = -(env.D_L + env.D_Q * np.abs(nu)) * nu
    down = R[:, 2, :]                                                     # R^T e3 = third row of R
    rest = np.c_[(env.p.weight - env.p.buoyancy) * down, np.cross(env.r_b, -env.p.buoyancy * down)]
    return (cor + damp + rest + tau) / M


def batch_exp(phi: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(phi).as_matrix()


def batch_heun(env, p, R, nu, tau, h):
    a1 = batch_nu_dot(env, R, nu, tau)
    R1 = R @ batch_exp(nu[:, 3:] * h)
    nu1 = nu + a1 * h
    a2 = batch_nu_dot(env, R1, nu1, tau)
    p_new = p + 0.5 * (np.einsum("kij,kj->ki", R, nu[:, :3]) + np.einsum("kij,kj->ki", R1, nu1[:, :3])) * h
    return p_new, R @ batch_exp(0.5 * (nu[:, 3:] + nu1[:, 3:]) * h), nu + 0.5 * (a1 + a2) * h


def rollout_ours(env, p0, R0, nu0, tau_seq, h, substeps=10):
    """tau_seq (steps, starts, 6); returns lists of (p, R, nu) after every step."""
    p, R, nu, out = p0.copy(), R0.copy(), nu0.copy(), []
    for tau in tau_seq:
        for _ in range(substeps):
            p, R, nu = batch_heun(env, p, R, nu, tau, h / substeps)
        out.append((p.copy(), R.copy(), nu.copy()))
    return out


def euler_continuous(R: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """ZYX Euler (phi, theta, psi) of R on the 2 pi branch closest to `reference` (the unwrapped Euler of the CSV)."""
    zyx = Rotation.from_matrix(R).as_euler("ZYX")[:, ::-1]
    return reference + (zyx - reference + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- 1 paper protocol
def paper_protocol(run_their_fossen: bool) -> dict:
    df = pd.read_csv(CSV).sort_values("t").drop_duplicates(subset="t")
    cols = ["x", "y", "z", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r"]
    X, U, t = df[cols].to_numpy(float), df[[f"u{i}" for i in range(1, 9)]].to_numpy(float), df["t"].to_numpy(float)
    h = float(np.median(np.diff(t)))
    split = int(0.8 * len(X)); Xt, Ut, tt = X[split:], U[split:], t[split:]
    n = Xt.shape[1]
    raw = np.load(NPZ)
    volts = np.interp(tt, raw["battery_t"], raw["battery_voltage"])
    env, t200 = BlueROV2(), T200()
    tau_all = t200.thrust_from_command(np.nan_to_num(Ut), volts) @ env.E.T                 # (N, 6) wrench at each row
    results = {"test_rows": len(Xt), "sample_dt": h}

    def score(pred_end: dict) -> list:
        return [float(np.sqrt(np.sum((pred_end[H] - Xt[H:]) ** 2) / ((len(Xt) - H) * n))) for H in HORIZONS]

    results["persistence"] = score({H: Xt[:-H] for H in HORIZONS})
    # ours: one batch per horizon start set, input of row k drives k -> k+1 (as their evaluator: U_seq = U[k:k+H])
    tic = time.perf_counter()
    starts = np.arange(len(Xt) - 1)
    p0, R0 = Xt[starts, :3], Rotation.from_euler("ZYX", Xt[starts, 3:6][:, ::-1]).as_matrix()
    nu0 = Xt[starts, 6:12]
    steps = max(HORIZONS)
    tau_seq = np.stack([tau_all[np.minimum(starts + s, len(Xt) - 1)] for s in range(steps)])
    traj = rollout_ours(env, p0, R0, nu0, tau_seq, h)
    ends = {}
    for H in HORIZONS:
        valid = starts[: len(Xt) - H]
        p, R, nu = (a[: len(valid)] for a in traj[H - 1])
        ends[H] = np.c_[p, euler_continuous(R, Xt[valid + H, 3:6]), nu]
    results["ours_published"] = score(ends)
    results["ours_seconds"] = time.perf_counter() - tic
    if run_their_fossen:
        sys.path.insert(0, str(RAW))
        from fossen.BlueROV2 import BlueROV2 as Theirs                                     # noqa: E402
        tic = time.perf_counter(); theirs = []
        for H in HORIZONS:                                                                # their evaluator, verbatim logic
            rov, se = Theirs(dt=h), 0.0
            for k in range(len(Xt) - H):
                x = Xt[k].copy()
                for j in range(H):
                    x = x + h * rov.dynamics(x, Ut[k + j], h)
                se += float(np.dot(x - Xt[k + H], x - Xt[k + H]))
            theirs.append(float(np.sqrt(se / ((len(Xt) - H) * n))))
        results["their_fossen"] = theirs
        results["their_fossen_seconds"] = time.perf_counter() - tic
    return results


# --------------------------------------------------------------------------- 2 clean protocol
def clean_protocol() -> dict:
    data = pickle.load(DATASET.open("rb"))
    F, h = data["test_trajectories"], float(data["settings"]["sample_dt"])
    env = BlueROV2()
    horizons_s = (0.1, 0.5, 1.0, 2.0, 5.0)
    starts = [(i, s) for i in range(F.shape[0]) for s in range(0, F.shape[1] - 1, 25)]       # a rollout every 0.5 s
    out = {}
    for H in horizons_s:
        k = int(round(H / h)); sel = [(i, s) for i, s in starts if s + k < F.shape[1]]
        if not sel: continue
        x0 = np.stack([F[i, s] for i, s in sel]); target = np.stack([F[i, s + k] for i, s in sel])
        tau = np.stack([np.stack([env.E @ F[i, s + j + 1, 18:] for i, s in sel]) for j in range(k)])
        p, R, nu = rollout_ours(env, x0[:, :3], x0[:, 3:12].reshape(-1, 3, 3), x0[:, 12:18], tau, h)[-1]
        R_t = target[:, 3:12].reshape(-1, 3, 3)
        att = np.degrees(Rotation.from_matrix(np.einsum("kji,kjl->kil", R, R_t)).magnitude())
        att_still = np.degrees(Rotation.from_matrix(np.einsum("kji,kjl->kil", x0[:, 3:12].reshape(-1, 3, 3), R_t)).magnitude())
        out[str(H)] = {"rollouts": len(sel),
                       "position_m": [float(np.median(np.linalg.norm(p - target[:, :3], axis=1))), float(np.median(np.linalg.norm(x0[:, :3] - target[:, :3], axis=1)))],
                       "attitude_deg": [float(np.median(att)), float(np.median(att_still))],
                       "v_mps": [float(np.median(np.linalg.norm(nu[:, :3] - target[:, 12:15], axis=1))), float(np.median(np.linalg.norm(x0[:, 12:15] - target[:, 12:15], axis=1)))],
                       "omega_radps": [float(np.median(np.linalg.norm(nu[:, 3:] - target[:, 15:18], axis=1))), float(np.median(np.linalg.norm(x0[:, 15:18] - target[:, 15:18], axis=1)))]}
    return out


# --------------------------------------------------------------------------- 3 per-axis gap
def axis_gap() -> dict:
    data = pickle.load(DATASET.open("rb"))
    env, h = BlueROV2(), float(data["settings"]["sample_dt"])
    meas, thr, drag, rest = [], [], [], []
    for f in np.concatenate([data["train_trajectories"], data["test_trajectories"]]):
        for k in range(5, f.shape[0] - 5, 2):
            J = range(k - 5, k + 5)
            meas.append((f[k + 5, 12:18] - f[k - 5, 12:18]) / (10 * h))
            thr.append(np.mean([env.E @ f[j + 1, 18:] / env.M for j in J], 0))
            drag.append(np.mean([-env.damping_force(f[j, 12:18]) / env.M for j in J], 0))
            rest.append(np.mean([(env.coriolis_force(f[j, 12:18]) + env.restoring(f[j, 3:12].reshape(3, 3))) / env.M for j in J], 0))
    m, T, D, C = map(np.array, (meas, thr, drag, rest))
    out = {}
    for j, name in enumerate(AXES):
        X = np.c_[T[:, j], -D[:, j], np.ones(len(m))]; y = m[:, j] - C[:, j]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[name] = {"alpha_thrust": float(coef[0]), "beta_drag": float(coef[1]), "offset": float(coef[2]),
                     "r2_published": float(1 - np.var(m[:, j] - (T[:, j] - D[:, j] + C[:, j])) / np.var(m[:, j])),
                     "r2_refit": float(1 - np.var(y - X @ coef) / np.var(m[:, j]))}
    out["_samples"] = len(m)
    return out


# --------------------------------------------------------------------------- figures
def example_rollouts(pdf, plt) -> None:
    data = pickle.load(DATASET.open("rb"))
    F, h, env = data["test_trajectories"], float(data["settings"]["sample_dt"]), BlueROV2()
    picks = [0, F.shape[0] // 2, F.shape[0] - 1]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for col, i in enumerate(picks):
        f = F[i]; tau = np.stack([[env.E @ f[k + 1, 18:]] for k in range(f.shape[0] - 1)])
        traj = rollout_ours(env, f[:1, :3], f[:1, 3:12].reshape(1, 3, 3), f[:1, 12:18], tau, h)
        p = np.vstack([f[:1, :3], np.stack([s[0][0] for s in traj])]); t = np.arange(len(p)) * h
        ax = axes[0, col]; ax.plot(f[:, 1], f[:, 0], "k", lw=1.5, label="real"); ax.plot(p[:, 1], p[:, 0], "C3--", lw=1.2, label="published physics")
        ax.plot(f[0, 1], f[0, 0], "ko"); ax.set(xlabel="y (m)", ylabel="x (m)", title=f"test trajectory {i}: top view"); ax.axis("equal"); ax.grid(alpha=0.3)
        ax = axes[1, col]; ax.plot(t, f[:, 2], "k", lw=1.5); ax.plot(t, p[:, 2], "C3--", lw=1.2); ax.invert_yaxis()
        ax.set(xlabel="t (s)", ylabel="depth z (m)"); ax.grid(alpha=0.3)
    axes[0, 0].legend(); fig.suptitle("Same start, same recorded thrust, 10 s open loop: real BlueROV2 vs the published physics (our simulator)")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def write_report(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    OUT.mkdir(parents=True, exist_ok=True)
    pp = res["paper_protocol"]
    rows = [("Persistence (x_{k+H} = x_k)", pp["persistence"], "this script")]
    if "their_fossen" in pp: rows.append(("Fossen, their code (reproduced)", pp["their_fossen"], "this script"))
    rows.append(("Published physics, our pH simulator", pp["ours_published"], "this script"))
    rows += [(f"{k} (paper Table 2)", v, "Torroba et al.") for k, v in PAPER_TABLE2.items()]
    md = ["# Sim-to-real gap of the published BlueROV2 physics", "",
          f"Real data: Marinarium `manual` recording, paper protocol (last 20 %, {pp['test_rows']} rows, 50 Hz, 12-D Euler state).", "",
          "## 1. Endpoint RMSE (paper metric; lower is better)", "", "| Model | 1-step | 10-step | 100-step | source |", "|---|---|---|---|---|"]
    md += [f"| {name} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {src} |" for name, v, src in rows]
    md += ["", "## 2. Clean test trajectories (no dropouts / glitches): median error, published physics vs standing still", "",
           "| horizon | position (m) | attitude (deg) | v (m/s) | w (rad/s) | rollouts |", "|---|---|---|---|---|---|"]
    for H, e in res["clean_protocol"].items():
        md.append(f"| {H} s | {e['position_m'][0]:.3f} vs {e['position_m'][1]:.3f} | {e['attitude_deg'][0]:.1f} vs {e['attitude_deg'][1]:.1f} | "
                  f"{e['v_mps'][0]:.3f} vs {e['v_mps'][1]:.3f} | {e['omega_radps'][0]:.3f} vs {e['omega_radps'][1]:.3f} | {e['rollouts']} |")
    md += ["", "## 3. Per-axis gap: accel = alpha thrust/M - beta drag/M + c (alpha = beta = 1 for the published model)", "",
           "| axis | alpha (thrust) | beta (drag) | offset (s^-2) | R^2 published | R^2 refit |", "|---|---|---|---|---|---|"]
    md += [f"| {a} | {g['alpha_thrust']:+.2f} | {g['beta_drag']:+.2f} | {g['offset']:+.3f} | {g['r2_published']:+.2f} | {g['r2_refit']:+.2f} |"
           for a, g in res["axis_gap"].items() if not a.startswith("_")]
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    with PdfPages(OUT / "report.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
        ax.text(0.01, 0.99, "\n".join(md), va="top", family="monospace", fontsize=7.5); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        Hs = [float(k) for k in res["clean_protocol"]]
        for ax, key, lab in ((axes[0], "position_m", "position error (m)"), (axes[1], "attitude_deg", "attitude error (deg)")):
            ax.plot(Hs, [res["clean_protocol"][str(H)][key][0] for H in Hs], "o-C3", label="published physics (open loop)")
            ax.plot(Hs, [res["clean_protocol"][str(H)][key][1] for H in Hs], "s--k", label="standing still")
            ax.set(xscale="log", yscale="log", xlabel="horizon (s)", ylabel=lab); ax.grid(alpha=0.3, which="both"); ax.legend()
        fig.suptitle("Clean real test trajectories: median open-loop error vs horizon"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 4.5)); g = res["axis_gap"]; x = np.arange(6)
        ax.bar(x - 0.2, [g[a]["alpha_thrust"] for a in AXES], 0.4, label="alpha (thrust scale)")
        ax.bar(x + 0.2, [g[a]["beta_drag"] for a in AXES], 0.4, label="beta (drag scale)")
        ax.axhline(1, color="k", lw=1, ls="--", label="published model"); ax.set_xticks(x, AXES); ax.grid(alpha=0.3, axis="y"); ax.legend()
        ax.set_title("Per-axis scales the real data asks for (1 = published model is right)"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        example_rollouts(pdf, plt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-their-fossen", action="store_true", help="skip the slow reproduction of their Fossen baseline (reuses it from results.json if present)")
    args = parser.parse_args()
    res = {"paper_protocol": paper_protocol(not args.skip_their_fossen), "clean_protocol": clean_protocol(), "axis_gap": axis_gap()}
    previous = OUT / "results.json"
    if args.skip_their_fossen and previous.exists():                       # reuse an earlier (slow) reproduction of their row
        old = json.loads(previous.read_text())["paper_protocol"]
        if "their_fossen" in old and old.get("test_rows") == res["paper_protocol"]["test_rows"]:
            res["paper_protocol"]["their_fossen"] = old["their_fossen"]
            res["paper_protocol"]["their_fossen_seconds"] = old.get("their_fossen_seconds")
    write_report(res)
    print((OUT / "results.md").read_text())
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}/report.pdf, results.json, results.md")


if __name__ == "__main__":
    main()

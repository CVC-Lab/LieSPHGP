"""Score WINDSDE-trained SDE models against the KNOWN physics: drift operators, diffusion, accuracy, calibration.

The WINDSDE simulator's truth (settings of the dataset pickle):
    drift      M1^-1 = I/m,  M2^-1 = J^-1,  M1^-1 grad V = g e3,  g = selection (wrench input),
               M1^-1 D_v = c I,  M2^-1 D_w = c I              (linear damping F = -c m v, tau = -c J omega)
    diffusion  M1^-1 Sigma_f(x) = sigma_a(x) I,  sigma_a = linear_sigma (1 + speed_gain |v_b|)
               M2^-1 Sigma_t(x) = sigma_al(x) I, sigma_al = angular_sigma (1 + rate_gain |omega_b|)
Only gauge-invariant products are compared, evaluated on states of the held-out split.

Per run (best saved checkpoint by in-training test loss):
  1. drift products: relative error of thrust gain, torque gain, gravity, both damping products
  2. diffusion in twist units: relative RMS error and Pearson correlation of the learned sigma_v(x), sigma_w(x) with the
     truth, and their mean ratio (1 = right level); a constant diffusion has correlation 0 by construction
  3. accuracy: 50-step open loop on held-out windows (Table 7 layout, drift / mean path)
  4. calibration: evaluate_sde_band.py on the held-out split, with and without the learned observation noise

Usage:  python evaluate_ground_truth.py --dataset WINDSDEHI_CF2P_10s_h0p01_clean.pkl --pattern "*WINDSDEHI*" --output-name NAME
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IDSIA = ROOT / "datasets/QUADROTOR-DATASET-IDSIA"
for p in (ROOT, IDSIA):
    sys.path.insert(0, str(p))

import split_protocol_table as spt  # noqa: E402
from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation  # noqa: E402
from src.models.SE3_Quadrotor.ph_gp_lie_imex_sde_idsia import network as gp_net  # noqa: E402
from src.models.SE3_Quadrotor.ph_gp_lie_imex_sde_idsia.checkpoints import load_checkpoint as gp_load  # noqa: E402
from src.models.SE3_Quadrotor.ph_nn_lie_imex_sde.checkpoints import load_checkpoint as nn_load  # noqa: E402

T = ROOT / "experiments/quadrotor/train_runs"
EVAL_ROOT = ROOT / "experiments/quadrotor/eval_runs"


def label_of(name: str) -> str:
    family = "GP-SDE" if "ph_gp_" in name else "NN-SDE"
    mode = re.search(r"DIFF-([a-z]-[A-Za-z]+|const)", name)
    mode = mode.group(1) if mode else "?"
    return f"{family} {mode}" + (" (sigma on Adam)" if "sigAdam" in name else "")


def best_saved(d: Path) -> int | None:
    s = np.load(d / "training_stats.npz"); st = list(s["step"].astype(int)); t = s["test_total"]
    saved = sorted(int(re.search(r"step_(\d+)", c.name).group(1)) for c in d.glob("checkpoint_step_*.pkl"))
    cand = [(float(t[st.index(c)]), c) for c in saved if c > 0 and c in st and np.isfinite(t[st.index(c)])]
    return min(cand)[1] if cand else None


def load_model(d: Path, step: int):
    """(model with drift + diffusion methods, kind). GP: posterior-mean sample of the SDE package."""
    path = d / f"checkpoint_step_{step:05d}.pkl"
    if "ph_gp_" in d.name:
        payload = gp_load(path)
        cast = lambda tree: jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x, tree)  # noqa: E731
        return gp_net.DissipativeSE3HamODE(cast(payload["params"]), payload["extra"]["gp_setup"]).sample(), "gp"
    module = nn_load(path)["params"]
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x, module), "nn"


def is_momentum_diffusion(model, kind: str) -> bool:
    if kind == "nn":
        return getattr(model, "sigma_net", None) is not None
    return gp_net.diffusion_mode(model.gp_setup) is not None and "sigma" in model.weights


def operators(model, kind: str, states: np.ndarray) -> dict[str, np.ndarray]:
    momentum = is_momentum_diffusion(model, kind)

    def one(s):
        m1 = model.inverse_mass_1(s[:3]); m2 = model.inverse_mass_2(s[3:12]); g = model.control_matrix(s[:12])
        grad_v = jax.grad(model.potential)(s[:12])[:3]
        dv = model.dissipation_v(s[12:15], s[:3]); dw = model.dissipation_w(s[15:18], s[3:12])
        if momentum:
            scale = model.diffusion_scale(s)
            a_v, a_w = m1 @ jnp.diag(scale[:3]), m2 @ jnp.diag(scale[3:])
        else:
            scale = model.process_sigma()
            a_v, a_w = jnp.diag(scale[:3]), jnp.diag(scale[3:])
        # isotropic-equivalent twist diffusion: sqrt(trace(A A^T) / 3)
        sv = jnp.sqrt(jnp.trace(a_v @ a_v.T) / 3.0); sw = jnp.sqrt(jnp.trace(a_w @ a_w.T) / 3.0)
        return m1 @ g[:3, 0], m2 @ g[3:6, 1:4], m1 @ grad_v, m1 @ dv, m2 @ dw, sv, sw

    out = jax.vmap(one)(jnp.asarray(states))
    return dict(zip(("thrust", "torque", "gravity", "damping_v", "damping_w", "sigma_v", "sigma_w"), map(np.asarray, out)))


def relative(learned: np.ndarray, true: np.ndarray) -> float:
    axes = tuple(range(1, learned.ndim))
    return float(np.mean(np.sqrt(np.sum((learned - true) ** 2, axis=axes)) / np.sqrt(np.sum(true ** 2, axis=axes))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="WINDSDEHI_CF2P_10s_h0p01_clean.pkl")
    parser.add_argument("--pattern", default="*WINDSDEHI*")
    parser.add_argument("--samples", type=int, default=2000, help="held-out states for the operator comparison")
    parser.add_argument("--skip-band", action="store_true")
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    data_path = HERE / args.dataset
    data = pickle.load(data_path.open("rb")); s = data["settings"]
    m = float(s["vehicle_parameters"]["mass"]); J = np.diag(np.asarray(s["vehicle_parameters"]["inertia"])); g = float(s["vehicle_parameters"]["gravity_acceleration"])
    c = float(s["linear_damping_coefficient"]); gt = s["diffusion_ground_truth"]
    heldout = np.asarray(data["heldout_trajectories"])
    flat = heldout.reshape(-1, heldout.shape[-1])
    states = flat[np.linspace(0, len(flat) - 1, args.samples).astype(int)]
    speed, rate = np.linalg.norm(states[:, 12:15], axis=1), np.linalg.norm(states[:, 15:18], axis=1)
    truth = {"thrust": np.tile([0.0, 0.0, 1.0 / m], (len(states), 1)), "torque": np.tile(np.diag(1.0 / J), (len(states), 1, 1)),
             "gravity": np.tile([0.0, 0.0, g], (len(states), 1)), "damping_v": np.tile(c * np.eye(3), (len(states), 1, 1)),
             "damping_w": np.tile(c * np.eye(3), (len(states), 1, 1)),
             "sigma_v": gt["linear_sigma"] * (1 + gt["speed_gain"] * speed), "sigma_w": gt["angular_sigma"] * (1 + gt["rate_gain"] * rate)}
    windows = np.stack([f[s0:s0 + 51] for f in heldout for s0 in range(0, f.shape[0] - 51, 10)])

    folder = EVAL_ROOT / args.output_name; (folder / "logs").mkdir(parents=True, exist_ok=True)
    results, curves = {}, {}
    for d in sorted(T.glob(args.pattern)):
        meta = json.loads((d / "metadata.json").read_text())
        step = best_saved(d) if (d / "training_stats.npz").exists() else None
        if meta.get("status") != "completed" or step is None:
            print(f"skip {d.name[:90]} ({meta.get('status')})"); continue
        label = label_of(d.name)
        model, kind = load_model(d, step)
        ops = operators(model, kind, states)
        entry = {"run": d.name, "step": step,
                 "drift_relative_error": {k: relative(ops[k], truth[k]) for k in ("thrust", "torque", "gravity", "damping_v", "damping_w")},
                 "diffusion": {}}
        for k in ("sigma_v", "sigma_w"):
            learned, true = ops[k], truth[k]
            corr = float(np.corrcoef(learned, true)[0, 1]) if np.std(learned) > 1e-12 else 0.0
            entry["diffusion"][k] = {"relative_rms_error": float(np.sqrt(np.mean((learned - true) ** 2)) / np.sqrt(np.mean(true ** 2))),
                                     "correlation_with_truth": corr, "mean_ratio": float(np.mean(learned) / np.mean(true))}
        curves[label] = {"sigma_v": ops["sigma_v"], "sigma_w": ops["sigma_w"]}
        # accuracy: drift (mean) rollout of the same checkpoint through the shared report tooling
        run = evaluation.load_run(d, step); drift_model, *_ = evaluation.build_model(run)
        try:
            entry["accuracy_table7"] = spt.table7_row(windows, evaluation.rollout(drift_model, windows, 0.01))
        except FloatingPointError:
            entry["accuracy_table7"] = None
        if not args.skip_band:
            for tag, extra in (("band_full", []), ("band_noobs", ["--exclude-observation-noise"])):
                log = folder / "logs" / f"{d.name}@{step}_{tag}.log"
                with log.open("w") as fh:
                    subprocess.run([sys.executable, "evaluate_sde_band.py", "--run", str(d), "--selected-step", str(step), "--samples", "32",
                                    "--stride", "10", "--horizons", "1", "10", "50", "--dataset", f"{data_path}@heldout", *extra],
                                   cwd=IDSIA, stdout=fh, stderr=subprocess.STDOUT, check=False)
                produced = sorted(d.glob("sde_band_S32_stride10_*.json"), key=lambda q: q.stat().st_mtime)
                if produced:
                    entry[tag] = json.loads(produced[-1].read_text()); produced[-1].rename(folder / f"{d.name}@{step}_{tag}.json")
        results[label] = entry
        dr = entry["drift_relative_error"]; df = entry["diffusion"]
        print(f"{label:34s} @{step:<5} drift err thrust {dr['thrust']:.3f} torque {dr['torque']:.3f} grav {dr['gravity']:.3f} "
              f"Dv {dr['damping_v']:.3f} Dw {dr['damping_w']:.3f} | sigma_v corr {df['sigma_v']['correlation_with_truth']:+.2f} "
              f"ratio {df['sigma_v']['mean_ratio']:.2f} | sigma_w corr {df['sigma_w']['correlation_with_truth']:+.2f} ratio {df['sigma_w']['mean_ratio']:.2f}", flush=True)

    (folder / "ground_truth_results.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(), "dataset": str(data_path), "truth": {"mass": m, "inertia": np.diag(J).tolist(),
        "gravity": g, "damping": c, "diffusion": gt}, "states": int(len(states)), "windows": int(len(windows)), "results": results},
        indent=2, default=float) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = np.argsort(speed); order_w = np.argsort(rate)
    axes[0].plot(speed[order], truth["sigma_v"][order], color="#0b0b0b", lw=2.5, label="truth")
    axes[1].plot(rate[order_w], truth["sigma_w"][order_w], color="#0b0b0b", lw=2.5, label="truth")
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#52514e", "#9085e9", "#c98500"]
    for i, (label, cur) in enumerate(curves.items()):
        axes[0].scatter(speed, cur["sigma_v"], s=3, alpha=0.35, color=palette[i % len(palette)], label=label)
        axes[1].scatter(rate, cur["sigma_w"], s=3, alpha=0.35, color=palette[i % len(palette)], label=label)
    axes[0].set(xlabel="body speed |v_b| (m/s)", ylabel="velocity diffusion (m/s per sqrt s)", title="learned vs true M1^-1 Sigma_f")
    axes[1].set(xlabel="body rate |omega_b| (rad/s)", ylabel="angular diffusion (rad/s per sqrt s)", title="learned vs true M2^-1 Sigma_tau")
    for ax in axes: ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, markerscale=4, loc="upper left")
    fig.tight_layout(); fig.savefig(folder / "diffusion_vs_truth.png", dpi=140)
    print(f"\nwritten {folder}")


if __name__ == "__main__":
    main()

"""Open-loop pictures on the held-out Melon flights: Naive hold, the IDSIA Physics baseline and our PH-GP-LieIMEX.

Same figures as the simulator open-loop report (``comparision/open_loop.py``): per-segment state tracking grid
(one row per model, GP +-2 sigma band), 3-D trajectory panels, position-error-against-time, and a metrics table.

The two baselines are the benchmark's own (Busetto et al. 2026, ``models/models.py``):
  * Naive     x_{k+1} = x_k
  * Physics   RK4 rigid body with thrust + gravity, torques ignored and omega_dot = 0 (their line 158); thrust
              clipped to [0, 2 m g] as in their code. Attitude is propagated exactly with the constant omega.
Each flight is cut into consecutive segments of the requested horizon, every segment starts from its own measured
state, metrics are over all segments, pictures for a few segments spread over the test set.

Usage:
  python plot_open_loop_baselines.py --run <GP run dir> [--horizon-seconds 0.5 1.0] [--segments 6] [--samples 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.transform import Rotation

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]  # comparision/idsia -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.SE3_Quadrotor.comparision import open_loop as openloop  # noqa: E402
from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation  # noqa: E402
from src.models.SE3_Quadrotor.comparision import report_figures as figures  # noqa: E402

MASS, GRAVITY = 0.045, 9.81          # the benchmark's published constants (models.py, phys_params)
THRUST_TO_WEIGHT = 2.0


def naive_hold(truth: np.ndarray) -> np.ndarray:
    prediction = np.repeat(truth[:, :1], truth.shape[1], axis=1).copy()
    prediction[..., 18:] = truth[..., 18:]
    return prediction


KT_PUBLISHED = 3.72e-8


def thrust_newtons(truth: np.ndarray, input_mode: str) -> np.ndarray:
    """Total thrust T for every sample, from either input layout (convert_idsia.py)."""
    if input_mode == "rotor2":
        return KT_PUBLISHED * truth[..., 18:22].sum(axis=-1) / (4.0 * KT_PUBLISHED / (MASS * GRAVITY))
    return truth[..., 18]


def idsia_physics(truth: np.ndarray, step: float, input_mode: str = "wrench") -> np.ndarray:
    """Their PhysQuadModel in our state layout (x, R flat, v body, omega body, u = wrench or rotor2)."""
    flights, length, _ = truth.shape
    total_thrust = thrust_newtons(truth, input_mode)
    prediction = truth.copy()
    position = truth[:, 0, :3].copy()
    rotation = truth[:, 0, 3:12].reshape(flights, 3, 3).copy()
    omega = truth[:, 0, 15:18].copy()                                   # constant: omega_dot = 0
    velocity = np.einsum("fij,fj->fi", rotation, truth[:, 0, 12:15])    # world frame, as in their model
    gravity = np.array([0.0, 0.0, GRAVITY])
    for k in range(1, length):
        thrust = np.clip(total_thrust[:, k - 1], 0.0, THRUST_TO_WEIGHT * MASS * GRAVITY)   # zero-order hold

        def acceleration(t):
            attitude = rotation @ Rotation.from_rotvec(omega * t).as_matrix()
            return attitude[:, :, 2] * (thrust / MASS)[:, None] - gravity

        a0, a_half, a1 = acceleration(0.0), acceleration(0.5 * step), acceleration(step)
        k1p, k1v = velocity, a0
        k2p, k2v = velocity + 0.5 * step * k1v, a_half
        k3p, k3v = velocity + 0.5 * step * k2v, a_half
        k4p, k4v = velocity + step * k3v, a1
        position = position + step / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        velocity = velocity + step / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        rotation = rotation @ Rotation.from_rotvec(omega * step).as_matrix()
        prediction[:, k, :3] = position
        prediction[:, k, 3:12] = rotation.reshape(flights, 9)
        prediction[:, k, 12:15] = np.einsum("fji,fj->fi", rotation, velocity)
        prediction[:, k, 15:18] = omega
    return prediction


def segment(truth: np.ndarray, keep: int) -> np.ndarray:
    pieces = truth.shape[1] // keep
    return truth[:, : pieces * keep].reshape(-1, keep, truth.shape[2])


def final_errors(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """The benchmark's MAE at the last step of the segment: position, world velocity, geodesic angle, omega."""
    last_t, last_p = truth[:, -1], prediction[:, -1]
    rot_t, rot_p = last_t[:, 3:12].reshape(-1, 3, 3), last_p[:, 3:12].reshape(-1, 3, 3)
    cosine = np.clip((np.einsum("nji,nji->n", rot_t, rot_p) - 1) / 2, -1, 1)
    vel_t = np.einsum("nij,nj->ni", rot_t, last_t[:, 12:15]); vel_p = np.einsum("nij,nj->ni", rot_p, last_p[:, 12:15])
    return {"MAE_p": float(np.linalg.norm(last_p[:, :3] - last_t[:, :3], axis=1).mean()),
            "MAE_v": float(np.linalg.norm(vel_p - vel_t, axis=1).mean()),
            "MAE_R": float(np.arccos(cosine).mean()),
            "MAE_w": float(np.linalg.norm(last_p[:, 15:18] - last_t[:, 15:18], axis=1).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--selected-step", type=int, default=None)
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "datasets/QUADROTOR-DATASET-IDSIA/IDSIA_CF21BL_10s_h0p01_clean.pkl") + "@test")
    parser.add_argument("--horizon-seconds", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--segments", type=int, default=6, help="segments drawn per horizon, spread over the set")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--label", default="PH-GP-LieIMEX (ours)")
    parser.add_argument("--output-name", default=None)
    arguments = parser.parse_args()

    stamp = datetime.now().astimezone().strftime("%d-%m-%H-%M")
    folder = PROJECT_ROOT / "experiments/quadrotor/eval_runs" / (arguments.output_name or f"{stamp}_IDSIA-melon-open-loop-baselines")
    images = folder / "images"; images.mkdir(parents=True, exist_ok=True)

    run = evaluation.load_run(Path(arguments.run), arguments.selected_step)
    model, params, gp_setup = evaluation.build_model(run)
    seed = int(run["config"]["report"]["random_seed"])
    flights = openloop.load_open_loop_set(arguments.dataset, 10**6)
    whole, step = flights["truth"], flights["dt"]
    import pickle
    with Path(str(arguments.dataset).partition("@")[0]).open("rb") as handle:
        input_mode = pickle.load(handle)["settings"].get("input_mode", "wrench")
    print(f"{whole.shape[0]} held-out flights of {(whole.shape[1] - 1) * step:.2f} s, dt = {step}")

    summary = {}
    with PdfPages(folder / "open_loop_baselines.pdf") as pdf:
        for horizon in arguments.horizon_seconds:
            keep = int(round(horizon / step)) + 1
            truth = segment(whole, keep)
            set_name = f"IDSIA-melon_{horizon:g}s"
            print(f"\n{set_name}: {truth.shape[0]} segments of {keep} samples")
            entries = []
            naive = naive_hold(truth); entries.append(("Naive (hold)", naive, None, 0.0))
            physics = idsia_physics(truth, step, input_mode); entries.append(("Physics (IDSIA, dw/dt=0)", physics, None, 0.0))
            ours, compute_ms = openloop.timed_rollout(model, truth, step)
            # a point-estimate model (PH-NN-LieIMEX, PH-NODE) has no posterior to draw from: no band,
            # and the table's uncertainty columns are left empty for it
            samples = (openloop.posterior_rollouts(params, gp_setup, truth, arguments.samples, seed, step)
                       if gp_setup else None)
            entries.append((arguments.label, ours, samples, compute_ms))

            table_entries, grid_entries, error_entries, per_set = [], [], [], {}
            for label, prediction, sample, ms in entries:
                metrics = openloop.open_loop_metrics(truth, prediction, sample, step, ms)
                table_entries.append((label, metrics)); grid_entries.append((label, prediction, sample))
                error_entries.append((label, prediction))
                per_set[label] = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
                per_set[label].update(final_errors(truth, prediction))
                print(f"  {label:28s} pos_rmse={metrics['position_rmse_m']:.4f} m  "
                      + "  ".join(f"{k}@{horizon:g}s={v:.4f}" for k, v in final_errors(truth, prediction).items()))
            pdf.savefig(openloop.open_loop_table(table_entries, set_name, horizon, truth.shape[0], arguments.samples),
                        bbox_inches="tight"); plt.close()
            styles = [figures.COMPARISON_STYLES[i] for i in (3, 1, 0)]
            error_plot = openloop.error_figure(error_entries, truth, step, set_name, styles)
            error_plot.savefig(images / f"{set_name}_error.png", dpi=140, bbox_inches="tight")
            pdf.savefig(error_plot, bbox_inches="tight"); plt.close(error_plot)
            chosen = np.unique(np.linspace(0, truth.shape[0] - 1, arguments.segments).astype(int))
            for index in chosen:
                grid = openloop.states_grid(grid_entries, truth, step, set_name, flight=int(index))
                grid.savefig(images / f"{set_name}_states_segment{index:03d}.png", dpi=140, bbox_inches="tight")
                pdf.savefig(grid, bbox_inches="tight"); plt.close(grid)
                paths = openloop.trajectory_grid(grid_entries, truth, step, set_name, flight=int(index))
                paths.savefig(images / f"{set_name}_trajectory_segment{index:03d}.png", dpi=150, bbox_inches="tight")
                pdf.savefig(paths, bbox_inches="tight"); plt.close(paths)
            summary[set_name] = {"segments": int(truth.shape[0]), "horizon_seconds": horizon, "metrics": per_set,
                                 "drawn_segments": [int(i) for i in chosen]}
    (folder / "open_loop_metrics.json").write_text(json.dumps(
        {"run": str(Path(arguments.run).resolve()), "generated_at": datetime.now().astimezone().isoformat(),
         "sets": summary}, indent=2) + "\n")
    print(f"\nPDF: {folder / 'open_loop_baselines.pdf'}\nimages: {images}")


if __name__ == "__main__":
    main()

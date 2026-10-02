"""One whole 60 s real flight from the training shapes and one from the held-out shape, Naive / IDSIA Physics / ours.

Two views per flight, same figure functions as the open-loop report:
  * pure open loop        one measured x_0 at t = 0, the recorded u(t) for 60 s, no correction at all
  * restarted every 1 s   the benchmark's rolling protocol laid end to end: every 1 s the prediction restarts from
                          the measured state, so the picture shows model quality along the whole flight instead
                          of the inevitable 60 s divergence
Usage: python plot_full_flight_baselines.py --run <GP run dir> [--seconds 60] [--restart-seconds 1.0] [--samples 20]
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
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.transform import Rotation

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.SE3_Quadrotor.comparision import open_loop as openloop  # noqa: E402
from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation  # noqa: E402
from src.models.SE3_Quadrotor.comparision import report_figures as figures  # noqa: E402
from plot_open_loop_baselines import idsia_physics, naive_hold, final_errors  # noqa: E402

KT, KC, ARM = 3.72e-8, 7.74e-12, 0.0353
RAW = PROJECT_ROOT / "other_paper_codes/nanodrone-sysid-benchmark/data"
# (file, seconds skipped at the start). Melon begins with a launch transient (|omega| ~ 7 rad/s at t ~ 0.95 s)
# and is 65 s long, so its 60 s window starts after it; chirp is exactly 60 s and starts calmly.
DEFAULT_FLIGHTS = {"train-chirp-run1": (RAW / "train/chirp_20251017_run1.csv", 0.0),
                   "test-melon-run1": (RAW / "test/melon_20251017_run1.csv", 2.0)}


def load_flight(path: Path, samples: int, skip: int = 0) -> np.ndarray:
    frame = pd.read_csv(path).iloc[skip:skip + samples]
    rotation = Rotation.from_quat(frame[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    squared = frame[[f"m{i}_rads" for i in (1, 2, 3, 4)]].to_numpy(dtype=np.float64) ** 2
    control = np.column_stack([
        KT * squared.sum(axis=1),
        KT * ARM * ((squared[:, 2] + squared[:, 3]) - (squared[:, 0] + squared[:, 1])),
        KT * ARM * ((squared[:, 1] + squared[:, 2]) - (squared[:, 0] + squared[:, 3])),
        KC * ((squared[:, 0] + squared[:, 2]) - (squared[:, 1] + squared[:, 3]))])
    velocity_world = frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    state = np.concatenate([frame[["x", "y", "z"]].to_numpy(dtype=np.float64), rotation.reshape(-1, 9),
                            np.einsum("nji,nj->ni", rotation, velocity_world),
                            frame[["wx", "wy", "wz"]].to_numpy(dtype=np.float64), control], axis=1)
    return state[None]


def segments_of(truth: np.ndarray, keep: int) -> tuple[np.ndarray, list[int]]:
    """Consecutive segments of ``keep`` samples sharing their boundary sample, and their start indices."""
    starts = list(range(0, truth.shape[1] - keep + 1, keep - 1))
    return np.stack([truth[0, s:s + keep] for s in starts]), starts


def stitch(prediction: np.ndarray, truth: np.ndarray, starts: list[int]) -> np.ndarray:
    """Lay per-segment predictions (n, keep, C) back onto the flight's time axis; C may be 18 or 22."""
    out = truth[..., :prediction.shape[-1]].copy()
    keep = prediction.shape[1]
    for piece, s in zip(prediction, starts):
        out[0, s:s + keep] = piece
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--selected-step", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--restart-seconds", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--label", default="PH-GP-LieIMEX (ours)")
    parser.add_argument("--output-name", default=None)
    arguments = parser.parse_args()
    step = 0.01
    stamp = datetime.now().astimezone().strftime("%d-%m-%H-%M")
    folder = PROJECT_ROOT / "experiments/quadrotor/eval_runs" / (arguments.output_name or f"{stamp}_IDSIA-full-flight-baselines")
    images = folder / "images"; images.mkdir(parents=True, exist_ok=True)

    run = evaluation.load_run(Path(arguments.run), arguments.selected_step)
    model, params, gp_setup = evaluation.build_model(run)
    seed = int(run["config"]["report"]["random_seed"])
    keep = int(round(arguments.restart_seconds / step)) + 1
    summary = {}
    with PdfPages(folder / "full_flight_baselines.pdf") as pdf:
        for name, (path, skip_seconds) in DEFAULT_FLIGHTS.items():
            truth = load_flight(path, int(round(arguments.seconds / step)) + 1, int(round(skip_seconds / step)))
            print(f"\n{name}: {truth.shape[1]} samples = {(truth.shape[1] - 1) * step:.0f} s from {path.name}"
                  f" starting at t = {skip_seconds:g} s")
            views = {
                f"{name}_{arguments.seconds:g}s_open-loop": {
                    "Naive (hold)": (lambda t: naive_hold(t), False),
                    "Physics (IDSIA, dw/dt=0)": (lambda t: idsia_physics(t, step), False),
                    arguments.label: (lambda t: openloop.timed_rollout(model, t, step)[0], True)},
            }
            gp_samples = openloop.posterior_rollouts(params, gp_setup, truth, arguments.samples, seed, step)
            entries_by_view = {}
            for view, models in views.items():
                entries = []
                for label, (fn, is_gp) in models.items():
                    entries.append((label, fn(truth), gp_samples if is_gp else None))
                entries_by_view[view] = entries
            # restarted view: the same models on consecutive 1 s segments, stitched back onto the time axis
            view = f"{name}_{arguments.seconds:g}s_restart-every-{arguments.restart_seconds:g}s"
            pieces, starts = segments_of(truth, keep)
            piece_samples = openloop.posterior_rollouts(params, gp_setup, pieces, arguments.samples, seed, step)
            entries_by_view[view] = [
                ("Naive (hold)", stitch(naive_hold(pieces), truth, starts), None),
                ("Physics (IDSIA, dw/dt=0)", stitch(idsia_physics(pieces, step), truth, starts), None),
                (arguments.label, stitch(openloop.timed_rollout(model, pieces, step)[0], truth, starts),
                 np.stack([stitch(piece_samples[i], truth, starts) for i in range(arguments.samples)]))]

            for view, entries in entries_by_view.items():
                per = {}
                for label, prediction, _ in entries:
                    err = np.linalg.norm(prediction[0, :, :3] - truth[0, :, :3], axis=-1)
                    per[label] = {"position_rmse_m": float(np.sqrt(np.mean(err ** 2))),
                                  "position_error_median_m": float(np.median(err)),
                                  "position_error_max_m": float(err.max())}
                    print(f"  {view:45s} {label:28s} pos RMSE {per[label]['position_rmse_m']:.3f} m  median {per[label]['position_error_median_m']:.3f}  max {per[label]['position_error_max_m']:.2f}")
                summary[view] = per
                grid = openloop.states_grid(entries, truth, step, view, flight=0)
                grid.savefig(images / f"{view}_states.png", dpi=140, bbox_inches="tight"); pdf.savefig(grid, bbox_inches="tight"); plt.close(grid)
                paths = openloop.trajectory_grid(entries, truth, step, view, flight=0)
                paths.savefig(images / f"{view}_trajectory.png", dpi=150, bbox_inches="tight"); pdf.savefig(paths, bbox_inches="tight"); plt.close(paths)
                error_plot = openloop.error_figure([(l, p) for l, p, _ in entries], truth, step, view,
                                                   [figures.COMPARISON_STYLES[i] for i in (3, 1, 0)])
                error_plot.savefig(images / f"{view}_error.png", dpi=140, bbox_inches="tight"); pdf.savefig(error_plot, bbox_inches="tight"); plt.close(error_plot)
    (folder / "full_flight_metrics.json").write_text(json.dumps({"run": str(Path(arguments.run).resolve()), "views": summary}, indent=2) + "\n")
    print(f"\nPDF: {folder / 'full_flight_baselines.pdf'}\nimages: {images}")


if __name__ == "__main__":
    main()

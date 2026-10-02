"""Open-loop evaluation of models trained on real flights, scored on held-out trajectory shapes.

This is the real-data counterpart of the open-loop section of ``generate_comparison_report_v2``. The
identification table is deliberately absent: a real vehicle has no analytic operators, so there is nothing to
compare the six gauge-invariant products against. What remains is the honest measure - roll every model
forward from one measured initial state with the recorded wrench and see how long it tracks the measurement.

Writes, into one folder:
    comparison.pdf                        the metrics table, one page per evaluation set
    images/<set>_states_flight<k>.png     one row per model, eight state columns, GP +-2 sigma band
    images/<set>_trajectory_flight<k>.png one 3-D panel per model, prediction versus measurement
    images/<set>_error.png                position error against time
    images/<set>_calibration_<model>.png  GP posterior calibration
    open_loop_metrics.json                every number in the table

Usage:
    python -m src.models.SE3_Quadrotor.comparision.evaluate_real_dataset \
        --run experiments/quadrotor/train_runs/<gp run> \
        --run experiments/quadrotor/train_runs/<other run> \
        --dataset datasets/QUADROTOR-DATASET-NANOBENCH/NANOBENCH_CF2_10s_h0p01_clean.pkl@test
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import open_loop as openloop
from . import report_evaluation as evaluation
from . import report_figures as figures
from ..ph_gp_lie_imex.config import resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
DEFAULT_SAMPLES = 50
MAX_FLIGHT_IMAGES = 10      # a short horizon splits each flight into many segments; do not draw them all


def _label(config: dict[str, Any]) -> str:
    """A short model name for the tables and panels, taken from the model family and solver."""
    name = config["model"]["name"]
    solver = config["model"]["solver"]
    family = {"ph_gp_lie_imex": "PH-GP", "ph_nn_lie_imex": "PH-NN", "ph_node": "PH-NODE"}.get(name, name)
    return f"{family}-{solver.replace('lie-imex', 'LieIMEX').replace('rk4', 'RK4')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True,
                        help="a completed training run directory; repeat for every model in the comparison")
    parser.add_argument("--dataset", action="append", required=True,
                        help="evaluation set as PATH[@split]; repeat for several sets")
    parser.add_argument("--flights", type=int, default=0, help="flights per set (0 = every flight in the split)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="posterior weight samples behind the GP bands and the probabilistic rows")
    parser.add_argument("--selected-step", type=int, default=None)
    parser.add_argument("--horizon-seconds", type=float, default=0.0,
                        help="truncate every flight to this many seconds before scoring; 0 = the whole flight. "
                             "Use it to report at a benchmark's own horizon (the IDSIA protocol is 0.5 s)")
    parser.add_argument("--output-name", type=str, default=None)
    arguments = parser.parse_args()

    stamp = datetime.now().astimezone().strftime("%d-%m-%H-%M")
    folder = EVAL_ROOT / (arguments.output_name or f"{stamp}_real-data-open-loop")
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)

    models = []
    for index, run_dir in enumerate(arguments.run):
        run = evaluation.load_run(Path(run_dir), arguments.selected_step)
        model, params, gp_setup = evaluation.build_model(run)
        config = run["config"]
        colour, style = figures.COMPARISON_STYLES[index % len(figures.COMPARISON_STYLES)]
        models.append({"run": run, "config": config, "model": model, "params": params, "gp_setup": gp_setup,
                       "label": _label(config), "is_gp": config["model"]["name"] == "ph_gp_lie_imex",
                       "colour": colour, "style": style, "directory": str(Path(run_dir).resolve()),
                       "training_dataset": config["data"]["dataset_path"]})
        print(f"loaded {models[-1]['label']:16s} from {Path(run_dir).name}", flush=True)

    table_pages: list[plt.Figure] = []
    summary: dict[str, Any] = {}
    for specification in arguments.dataset:
        flights = (openloop.load_open_loop_set(specification, arguments.flights) if arguments.flights
                   else openloop.load_open_loop_set(specification, 10**6))
        truth, step, set_name = flights["truth"], flights["dt"], flights["name"]
        if arguments.horizon_seconds > 0:
            keep = int(round(arguments.horizon_seconds / step)) + 1
            if keep < truth.shape[1]:
                # Score the same flights over a shorter window by cutting each one into consecutive
                # segments of that length, so every sample still contributes and each segment starts
                # from its own measured state.
                segments = truth.shape[1] // keep
                truth = truth[:, : segments * keep].reshape(-1, keep, truth.shape[2])
                set_name = f"{set_name}_{arguments.horizon_seconds:g}s"
        print(f"\nopen loop on {set_name}: {truth.shape[0]} flights, dt = {step:.4f} s, "
              f"horizon = {(truth.shape[1] - 1) * step:.2f} s", flush=True)
        table_entries, grid_entries, error_entries, per_set = [], [], [], {}
        for entry in models:
            prediction, compute_ms = openloop.timed_rollout(entry["model"], truth, step)
            samples = None
            if entry["is_gp"] and prediction is not None:
                print(f"  {entry['label']}: {arguments.samples} posterior-sample rollouts...", flush=True)
                samples = openloop.posterior_rollouts(entry["params"], entry["gp_setup"], truth, arguments.samples,
                                                      int(entry["config"]["report"]["random_seed"]), step)
            metrics = openloop.open_loop_metrics(truth, prediction, samples, step, compute_ms)
            table_entries.append((entry["label"], metrics))
            grid_entries.append((entry["label"], prediction, samples))
            error_entries.append((entry["label"], prediction))
            per_set[entry["label"]] = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
            print(f"  {entry['label']:16s} " + "  ".join(
                f"{k}={v:.4g}" for k, v in per_set[entry["label"]].items()
                if k in ("position_rmse_m", "valid_prediction_seconds_median", "position_error_final_m")), flush=True)
            if samples is not None:
                figure = openloop.calibration_figure(entry["label"], truth, prediction, samples, step, set_name, metrics)
                figure.savefig(images / f"{set_name}_calibration_{entry['label']}.png", dpi=140, bbox_inches="tight")
                plt.close(figure)
        horizon = (truth.shape[1] - 1) * step
        table_pages.append(openloop.open_loop_table(table_entries, set_name, horizon, truth.shape[0], arguments.samples))
        for flight in range(min(truth.shape[0], MAX_FLIGHT_IMAGES)):
            grid = openloop.states_grid(grid_entries, truth, step, set_name, flight=flight)
            grid.savefig(images / f"{set_name}_states_flight{flight:02d}.png", dpi=140, bbox_inches="tight")
            plt.close(grid)
            paths = openloop.trajectory_grid(grid_entries, truth, step, set_name, flight=flight)
            paths.savefig(images / f"{set_name}_trajectory_flight{flight:02d}.png", dpi=150, bbox_inches="tight")
            plt.close(paths)
        error_plot = openloop.error_figure(error_entries, truth, step, set_name,
                                           [(m["colour"], m["style"]) for m in models])
        error_plot.savefig(images / f"{set_name}_error.png", dpi=140, bbox_inches="tight")
        plt.close(error_plot)
        summary[set_name] = {"dt": step, "flights": int(truth.shape[0]), "source": flights["source"],
                             "horizon_seconds": horizon, "metrics": per_set}

    target = folder / "comparison.pdf"
    with PdfPages(target) as pdf:
        for figure in table_pages:
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    (folder / "open_loop_metrics.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(),
        "models": [{k: entry[k] for k in ("label", "directory", "training_dataset")} for entry in models],
        "evaluation_sets": summary,
    }, indent=2) + "\n")
    print(f"\nPDF: {target} ({len(table_pages)} pages)")
    print(f"images: {images}")


if __name__ == "__main__":
    main()

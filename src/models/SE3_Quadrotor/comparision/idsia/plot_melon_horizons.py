"""Melon open-loop report across long horizons, every model integrated at a chosen sub-step.

Differences from ``plot_open_loop_baselines.py``:

* Flights come from the RAW 60 s melon CSVs, not the 10 s pickle, so horizons beyond 10 s are possible.
* Every learned model is rolled out with ``--substeps`` integrator steps per 0.01 s data interval, i.e. an
  effective step of 0.01/substeps, while the ground truth, the segmentation and every metric stay on the
  recorded 100 Hz grid.  Naive needs no integrator and is unaffected.
* One PDF page set per horizon in ``--horizon-seconds``.

Usage:
    python plot_melon_horizons.py --run LABEL=DIR [--run ...] \
        --horizon-seconds 1 3 5 10 20 30 --substeps 2 --segments 3
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import benchmark_protocol as bp
import plot_open_loop_baselines as bl
import swap_ablation
import oracle_wrench_rollout as oracle

PROJECT_ROOT = bp.PROJECT_ROOT
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
STEP = 0.01                      # the recorded sample interval; horizons are cut on this grid


def jnp_zeros3():
    import jax.numpy as jnp
    return jnp.zeros((3, 3))


def segments_at(flights: list[np.ndarray], horizon_seconds: float, stride_seconds: float) -> np.ndarray:
    """Non-overlapping-ish segments of ``horizon_seconds`` cut every ``stride_seconds`` from each flight."""
    keep = int(round(horizon_seconds / STEP)) + 1
    stride = max(1, int(round(stride_seconds / STEP)))
    pieces = [flight[start:start + keep]
              for flight in flights
              for start in range(0, flight.shape[0] - keep, stride)]
    if not pieces:
        raise SystemExit(f"no segment of {horizon_seconds} s fits in flights of "
                         f"{flights[0].shape[0] * STEP:.1f} s")
    return np.stack(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=DIR, one per learned model")
    parser.add_argument("--oracle", action="store_true",
                        help="add the analytic rigid body driven by the wrench reconstructed from the\nmeasurements (tau = J wdot + w x Jw, T = k_F sum Omega^2): the ceiling given perfect knowledge")
    parser.add_argument("--preserve-damping", action="store_true",
                        help="rescale D_v, D_w so the identifiable product M^-1 D keeps its learned value")
    parser.add_argument("--swap-run", action="append", default=[],
                        help="LABEL=DIR: same model but with ALL FIVE operators replaced by the\nbenchmark's published truth (M1^-1, M2^-1, V, g_f, g_tau). Inference only; D_v and D_w stay learned\nbecause the real vehicle's damping is unmeasured.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "tmp/idsia_raw/data/test")
    parser.add_argument("--pattern", default="melon*.csv")
    parser.add_argument("--horizon-seconds", type=float, nargs="+", default=[1, 3, 5, 10, 20, 30])
    parser.add_argument("--substeps", type=int, default=0,
                        help="integrator steps per 0.01 s data interval; 0 = use each run's own training value")
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--segments", type=int, default=3, help="segments drawn per horizon")
    parser.add_argument("--samples", type=int, default=0, help="GP posterior draws (0 = none)")
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    from src.models.SE3_Quadrotor.comparision import open_loop as openloop
    from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation
    from src.models.SE3_Quadrotor.comparision import report_figures as figures

    loaded = []
    for spec in arguments.swap_run:
        label, _, run_dir = spec.rpartition("=")
        run = evaluation.load_run(Path(run_dir), None)
        base, params, gp_setup = evaluation.build_model(run)
        loaded.append((label, run_dir, swap_ablation.all_true(base, preserve_damping=arguments.preserve_damping),
                       params, gp_setup,
                       bp.input_mode_of_run(run), int((run["config"]["model"].get("integration_substeps") or 1))))
    for spec in arguments.run:
        label, _, run_dir = spec.rpartition("=")   # rpartition: labels may contain "="
        run = evaluation.load_run(Path(run_dir), None)
        model, params, gp_setup = evaluation.build_model(run)
        # each model is rolled out at the step size it was TRAINED at, read from its own config
        own = int((run["config"]["model"].get("integration_substeps") or 1))
        loaded.append((label, run_dir, model, params, gp_setup, bp.input_mode_of_run(run), own))
    modes = {t[5] for t in loaded}
    if len(modes) > 1:
        raise SystemExit(f"runs disagree on input mode: {modes}")
    input_mode = modes.pop() if modes else "wrench"

    flights = bp.load_melon(arguments.source, input_mode, arguments.pattern)
    if not flights:
        raise SystemExit(f"no flights matching {arguments.pattern!r} under {arguments.source}")
    oracle_flights = (oracle.load_with_reconstructed_wrench(arguments.source, arguments.pattern)
                      if arguments.oracle else None)
    oracle_model = oracle.RigidBody(damping_v=jnp_zeros3(), damping_w=jnp_zeros3()) if arguments.oracle else None
    span = flights[0].shape[0] * STEP
    print(f"\n{len(flights)} melon flights of {span:.0f} s, dt = {STEP}, "
          + ("integrator step: each model's own training value" if arguments.substeps == 0
                 else f"integrator step = {STEP / arguments.substeps:g} s ({arguments.substeps} sub-steps/interval)"))

    folder = EVAL_ROOT / arguments.output_name
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    with PdfPages(folder / "melon_horizons.pdf") as pdf:
        for horizon in arguments.horizon_seconds:
            if horizon + STEP > span:
                print(f"  skipping {horizon:g} s: longer than a {span:.0f} s flight")
                continue
            truth = segments_at(flights, horizon, arguments.stride_seconds)
            set_name = f"IDSIA-melon_{horizon:g}s"
            print(f"\n{set_name}: {truth.shape[0]} segments of {truth.shape[1]} samples")

            entries = [("Naive (hold)", bl.naive_hold(truth), None, 0.0),
                       ("Physics (IDSIA, dw/dt=0)", bl.idsia_physics(truth, STEP, input_mode), None, 0.0)]
            for label, _d, model, params, gp_setup, _m, own in loaded:
                sub = own if arguments.substeps == 0 else arguments.substeps
                try:
                    prediction = evaluation.rollout(model, truth, STEP, substeps=sub)
                except FloatingPointError:
                    print(f"  {label:28s} non-finite rollout, dropped from this horizon")
                    continue
                samples = None
                if gp_setup and arguments.samples > 0:
                    samples = openloop.posterior_rollouts(params, gp_setup, truth, arguments.samples, 42, STEP)
                entries.append((label, prediction, samples, float("nan")))

            if oracle_model is not None:
                truth_oracle = segments_at(oracle_flights, horizon, arguments.stride_seconds)
                try:
                    entries.append(("Oracle (true rigid body + true tau)",
                                    evaluation.rollout(oracle_model, truth_oracle, STEP), None, float("nan")))
                except FloatingPointError:
                    print("  Oracle (true rigid body + true tau)  non-finite rollout, dropped from this horizon")

            table_entries, grid_entries, error_entries, per_set = [], [], [], {}
            for label, prediction, sample, ms in entries:
                metrics = openloop.open_loop_metrics(truth, prediction, sample, STEP, ms)
                table_entries.append((label, metrics))
                grid_entries.append((label, prediction, sample))
                error_entries.append((label, prediction))
                per_set[label] = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
                per_set[label].update(bl.final_errors(truth, prediction))
                print(f"  {label:28s} pos_rmse={metrics['position_rmse_m']:.4f} m  "
                      f"VPT={metrics['valid_prediction_seconds_median']:.2f} s")

            pdf.savefig(openloop.open_loop_table(table_entries, set_name, horizon, truth.shape[0],
                                                 arguments.samples), bbox_inches="tight")
            plt.close()
            styles = [figures.COMPARISON_STYLES[i % len(figures.COMPARISON_STYLES)] for i in range(len(entries))]
            error_plot = openloop.error_figure(error_entries, truth, STEP, set_name, styles)
            error_plot.savefig(images / f"{set_name}_error.png", dpi=140, bbox_inches="tight")
            pdf.savefig(error_plot, bbox_inches="tight")
            plt.close(error_plot)

            chosen = np.unique(np.linspace(0, truth.shape[0] - 1, arguments.segments).astype(int))
            for index in chosen:
                grid = openloop.states_grid(grid_entries, truth, STEP, set_name, flight=int(index))
                grid.savefig(images / f"{set_name}_states_segment{index:03d}.png", dpi=140, bbox_inches="tight")
                pdf.savefig(grid, bbox_inches="tight")
                plt.close(grid)
                paths = openloop.trajectory_grid(grid_entries, truth, STEP, set_name, flight=int(index))
                paths.savefig(images / f"{set_name}_trajectory_segment{index:03d}.png", dpi=150,
                              bbox_inches="tight")
                pdf.savefig(paths, bbox_inches="tight")
                plt.close(paths)

            summary[set_name] = {"segments": int(truth.shape[0]), "horizon_seconds": horizon,
                                 "integrator_step": (None if arguments.substeps == 0 else STEP / arguments.substeps),
                                 "metrics": per_set,
                                 "drawn_segments": [int(i) for i in chosen]}

    (folder / "melon_horizons.json").write_text(json.dumps({
        "source": str(arguments.source), "pattern": arguments.pattern, "flights": len(flights),
        "flight_seconds": span, "data_step": STEP, "substeps": arguments.substeps,
        "integrator_step": (None if arguments.substeps == 0 else STEP / arguments.substeps),
        "input_mode": input_mode,
        "runs": {label: {"dir": str(Path(d).resolve()), "substeps": own} for label, d, *_r, own in loaded},
        "generated_at": datetime.now().astimezone().isoformat(), "sets": summary,
    }, indent=2, default=float) + "\n")
    print(f"\nPDF: {folder / 'melon_horizons.pdf'}\nimages: {images}")


if __name__ == "__main__":
    main()

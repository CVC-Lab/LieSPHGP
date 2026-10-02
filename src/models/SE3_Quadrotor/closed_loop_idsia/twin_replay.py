"""Perfect-model twin: can an EXACT model reproduce a trajectory open loop from the recorded inputs?

The "real drone" here is the analytic rigid body (published constants, no damping) flown by the closed-loop test
(``run.py``) along the three recorded melon paths, at 500 Hz with Lie-IMEX steps of 2 ms.  Its rotor speeds are
logged the way the real dataset logs them: averaged onto the 100 Hz grid.

The same model is then replayed OPEN LOOP from those logged rotor speeds, starting from the twin's true state:

  * "same integrator"      Lie-IMEX at 2 ms, exactly as the twin was generated; the only imperfection is that the
                           input is the 100 Hz log instead of the 500 Hz command
  * "explicit midpoint"    the numpy integrator of ``oracle_full_wrench.py`` at 10 ms (the table shown in chat)

The model has no error at all, so any drift is caused purely by replaying logged inputs without feedback.

Usage (from the project root):
    python -m src.models.SE3_Quadrotor.closed_loop_idsia.twin_replay --closed-loop-run DIR --output-name NAME
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ..comparision import open_loop as openloop  # noqa: E402
from ..comparision import report_evaluation as evaluation  # noqa: E402
from ..comparision import report_figures as figures  # noqa: E402
from . import controller as ctl  # noqa: E402
from . import plants as pl  # noqa: E402
from . import reference as rf  # noqa: E402

PROJECT_ROOT = pl.PROJECT_ROOT
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
STEP = 0.01
TWIN = "Analytic rigid body (published constants)"
RECORDED_STYLE = {"color": "k", "ls": ":", "lw": 1.2, "label": "IDSIA melon (recorded, reference only)"}
sys.path.insert(0, str(PROJECT_ROOT / "datasets/QUADROTOR-DATASET-IDSIA"))


def twin_flights(folder: Path) -> np.ndarray:
    """(B, N, 22): the twin's 100 Hz states, and in 18:22 of sample k+1 the wrench of the rotor speeds logged
    over interval k (the layout ``evaluation.rollout`` reads)."""
    arrays = np.load(folder / "closed_loop_arrays.npz")
    states, speeds = arrays[f"{TWIN}|states"], arrays[f"{TWIN}|rotor_speeds"]
    if (arrays[f"{TWIN}|first_dead"] >= 0).any():
        raise SystemExit("the twin itself diverged in this closed-loop run")
    wrench = np.asarray(ctl.encode(speeds**2, "wrench"))
    control = np.concatenate([np.zeros((states.shape[0], 1, 4)), wrench], axis=1)
    return np.concatenate([states, control], axis=2)


def segments_at(flights: np.ndarray, horizon: float, stride_seconds: float) -> np.ndarray:
    keep = int(round(horizon / STEP)) + 1
    stride = int(round(stride_seconds / STEP))
    return np.stack([f[s:s + keep] for f in flights for s in range(0, f.shape[0] - keep, stride)])


def midpoint_replay(truth: np.ndarray) -> np.ndarray | None:
    """The numpy explicit-midpoint rollout of oracle_full_wrench, same analytic model, 10 ms step."""
    from oracle_full_wrench import rollout
    out = []
    for seg in truth:
        control = seg[1:, 18:22]
        force = np.zeros((control.shape[0], 3))
        force[:, 2] = control[:, 0]
        result = rollout(seg[0, :3], seg[0, 3:12].reshape(3, 3), seg[0, 12:15], seg[0, 15:18],
                         force, control[:, 1:])
        if result is None:
            return None
        x, r, v, w = result
        out.append(np.concatenate([x, r.reshape(-1, 9), v, w], axis=1))
    return np.stack(out)


def overlay_recorded(figure, recorded: np.ndarray, tag: str) -> None:
    """Draw the real IDSIA melon recording on an already-built page, for reference only (metrics ignore it)."""
    if tag == "states":
        keys = [key for _name, key in openloop.STATE_COLUMNS]
        columns = openloop._columns_of(recorded[None])
        times = np.arange(recorded.shape[0]) * STEP
        for index, axis in enumerate(figure.axes):
            axis.plot(times, columns[keys[index % len(keys)]][0], **{**RECORDED_STYLE,
                      "label": RECORDED_STYLE["label"] if index == 0 else None})
        handles, labels = figure.axes[0].get_legend_handles_labels()
    else:
        for index, axis in enumerate(figure.axes):
            axis.plot3D(*recorded[:, :3].T, **{**RECORDED_STYLE,
                        "label": RECORDED_STYLE["label"] if index == 0 else None})
        handles, labels = figure.axes[0].get_legend_handles_labels()
    for legend in list(figure.legends):
        legend.remove()
    figure.legend(handles, labels, loc="lower center", ncol=3, prop={"size": 10})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--closed-loop-run", type=Path,
                        default=EVAL_ROOT / "24-09-00-00_IDSIA-closed-loop-melon")
    parser.add_argument("--horizon-seconds", type=float, nargs="+", default=[1, 3, 5, 10, 20, 30])
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--segments", type=int, default=3, help="segments drawn per horizon")
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    flights = twin_flights(arguments.closed_loop_run)
    recorded_flights = np.stack([f["states"][:flights.shape[1]] for f in
                                 rf.load_flights(PROJECT_ROOT / "tmp/idsia_raw/data/test", "melon*.csv")])
    model = pl.analytic_plant().model
    folder = EVAL_ROOT / arguments.output_name
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    summary = {}

    with PdfPages(folder / "twin_replay.pdf") as pdf:
        for horizon in arguments.horizon_seconds:
            truth = segments_at(flights, horizon, arguments.stride_seconds)
            recorded = segments_at(recorded_flights, horizon, arguments.stride_seconds)
            set_name = f"perfect-model-twin_melon_{horizon:g}s"
            print(f"\n{set_name}: {truth.shape[0]} segments")
            entries = []
            try:
                same = evaluation.rollout(model, truth, STEP, substeps=5)
            except FloatingPointError:
                same = None
            entries.append(("Twin, Lie-IMEX 2 ms", same))
            entries.append(("Twin, midpoint 10 ms", midpoint_replay(truth)))

            table, grid, errors, per_set = [], [], [], {}
            for label, prediction in entries:
                metrics = openloop.open_loop_metrics(truth, prediction, None, STEP, float("nan"))
                table.append((label, metrics))
                per_set[label] = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
                if prediction is not None:
                    grid.append((label, prediction, None))
                    errors.append((label, prediction))
                    final = np.linalg.norm(prediction[:, -1, :3] - truth[:, -1, :3], axis=-1)
                    per_set[label]["final_position_error_median_m"] = float(np.median(final))
                    print(f"  {label:48s} pos RMSE {metrics['position_rmse_m']:.4f} m   "
                          f"median final error {np.median(final):.3f} m")
                else:
                    print(f"  {label:48s} non-finite: dropped from the plots at this horizon")

            pdf.savefig(openloop.open_loop_table(table, set_name, horizon, truth.shape[0], 0), bbox_inches="tight")
            plt.close()
            if not grid:
                continue
            styles = [figures.COMPARISON_STYLES[i % len(figures.COMPARISON_STYLES)] for i in range(len(errors))]
            error_plot = openloop.error_figure(errors, truth, STEP, set_name, styles)
            error_plot.savefig(images / f"{set_name}_error.png", dpi=140, bbox_inches="tight")
            pdf.savefig(error_plot, bbox_inches="tight")
            plt.close(error_plot)
            for index in np.unique(np.linspace(0, truth.shape[0] - 1, arguments.segments).astype(int)):
                for maker, tag in ((openloop.states_grid, "states"), (openloop.trajectory_grid, "trajectory")):
                    figure = maker(grid, truth, STEP, set_name, flight=int(index))
                    if figure._suptitle is not None:            # the shared title mentions a GP band; none here
                        figure._suptitle.set_text(
                            f"PERFECT model replayed open loop — {set_name}, segment {index}\n"
                            f"dashed: truth (the same model flown closed loop along melon), solid: replay from the "
                            f"100 Hz-logged rotor speeds, black dotted: real IDSIA melon recording (reference only)")
                    overlay_recorded(figure, recorded[int(index)], tag)
                    figure.savefig(images / f"{set_name}_{tag}_segment{index:03d}.png", dpi=140,
                                   bbox_inches="tight")
                    pdf.savefig(figure, bbox_inches="tight")
                    plt.close(figure)
            summary[set_name] = {"segments": int(truth.shape[0]), "metrics": per_set}

    (folder / "twin_replay.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(),
        "closed_loop_run": str(arguments.closed_loop_run.resolve()),
        "note": "truth = analytic rigid body flown closed loop along the recorded melon paths; replay = the SAME "
                "model open loop from its 100 Hz-logged rotor speeds",
        "sets": summary}, indent=2, default=float) + "\n")
    print(f"\nPDF: {folder / 'twin_replay.pdf'}")


if __name__ == "__main__":
    main()

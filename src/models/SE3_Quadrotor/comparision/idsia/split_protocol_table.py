"""Benchmark-protocol MAE on a chosen split, with Naive and Physics COMPUTED rather than quoted.

``benchmark_protocol.py`` compares against the IDSIA paper's Table 7, whose five baseline rows - Naive,
Physics, Res-MLP, Hybrid, Res-LSTM - are hardcoded numbers measured on the held-out melon flights.  They are
therefore valid only for the test split.  To report a TRAINING-shape table we must recompute whatever we can
ourselves: Naive (hold the initial state) and Physics (the benchmark's own rigid-body model with dw/dt = 0)
have reference implementations in ``plot_open_loop_baselines.py``, so those two are recomputed here on
whichever flights are loaded.  The three learned baselines cannot be: the paper publishes no training-split
numbers and we do not have their weights.

Metric is the benchmark's own: mean Euclidean norm per horizon (``benchmark_protocol.per_horizon``), reported
as the cumulative sum over h = 1..50.

SDE runs (``ph_gp_lie_imex_sde_idsia``, ``ph_nn_lie_imex_sde``) are scored with ``--sde-seeds S`` independent
rollouts ("wind seeds"): seed k is one Brownian draw (and, for the GP, one posterior weight sample) shared by every
window; each seed gets its own table row and the reported row is the mean over the S seeds (std kept in the JSON).
ODE runs are deterministic and rolled out once. ``DIR@STEP`` selects an intermediate checkpoint.

Usage:
    python split_protocol_table.py --run LABEL=DIR[@STEP] [--run ...] [--sde-seeds 5] \
        --source envs/quadrotor_se3_idsia/idsia_raw/data/test --pattern "melon*.csv" [--stride 2] [--horizon 50]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import benchmark_protocol as bp
import plot_open_loop_baselines as bl

PROJECT_ROOT = bp.PROJECT_ROOT
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
KEYS = (("p", "MAE_p [m]"), ("R", "MAE_R [rad]"), ("v", "MAE_v [m/s]"), ("w", "MAE_w [rad/s]"))


HORIZONS = (1, 10, 50)
SDE_PACKAGES = ("ph_gp_lie_imex_sde_idsia", "ph_nn_lie_imex_sde")


def mean_rows(rows: list[dict[str, dict[str, float]]]) -> tuple[dict, dict]:
    """Cell-wise mean and std over the seeds' table rows."""
    mean = {k: {c: float(np.mean([r[k][c] for r in rows])) for c in rows[0][k]} for k in rows[0]}
    std = {k: {c: float(np.std([r[k][c] for r in rows])) for c in rows[0][k]} for k in rows[0]}
    return mean, std


def sde_seed_rows(run_dir: str, step: str, truth: np.ndarray, seeds: int, batch: int) -> list[dict]:
    """One table row per seed: S sample paths of every window, seed k = the k-th Brownian draw / weight sample."""
    import pickle
    from src.models.SE3_Quadrotor.comparision import four_model_comparison as F
    run = F.evaluation.load_run(Path(run_dir), int(step) if step else None)
    with (PROJECT_ROOT / run["config"]["data"]["dataset_path"]).open("rb") as handle:
        settings = pickle.load(handle)["settings"]
    entry = F.Entry("SDE", f"{run_dir}@{step}" if step else run_dir, settings)
    predictions = []
    for begin in range(0, truth.shape[0], batch):
        chunk = np.transpose(truth[begin:begin + batch], (1, 0, 2))              # (T, B, 22), time-major
        paths = entry.sample_paths(chunk, seeds, 0)                              # (S, T, B, 22), same keys every chunk
        predictions.append(np.transpose(paths, (0, 2, 1, 3)))                    # (S, B, T, 22)
        print(f"  sampled {begin + chunk.shape[1]}/{truth.shape[0]} windows x {seeds} seeds", end="\r", flush=True)
    prediction = np.concatenate(predictions, axis=1)
    diverged = [int((~np.isfinite(prediction[k][..., :18]).all(axis=(1, 2))).sum()) for k in range(seeds)]
    if any(diverged):
        print(f"\n  WARNING: non-finite windows per seed {diverged} (kept, not filtered)")
    return [table7_row(truth, prediction[k]) for k in range(seeds)], diverged


def table7_row(truth: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float]]:
    """Their Table 7 layout: MAE at h = 1, 10, 50 plus the cumulative sum over h = 1..H, per metric."""
    per = bp.per_horizon(truth, prediction)          # each entry indexed by h-1
    row = {}
    for key, _label in KEYS:
        series = per[key]
        cells = {f"h{h}": float(series[h - 1]) for h in HORIZONS if h <= series.shape[0]}
        cells["cum"] = float(np.sum(series))
        row[key] = cells
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=DIR[@STEP] for each trained model")
    parser.add_argument("--sde-seeds", type=int, default=5, help="rollouts per window for SDE runs (0 = zero-noise path)")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pattern", default="melon*.csv")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation

    # the input parameterisation is a property of the dataset each model was trained on; every model in one
    # table must share it, so take it from the first run and check the rest agree
    runs = []
    for spec in arguments.run:
        label, _, target = spec.partition("=")
        run_dir, _, checkpoint_step = target.partition("@")
        run = evaluation.load_run(Path(run_dir), int(checkpoint_step) if checkpoint_step else None)
        runs.append((label, run_dir, run, bp.input_mode_of_run(run), checkpoint_step))
    modes = {mode for _l, _d, _r, mode, _s in runs}
    if len(modes) > 1:
        raise SystemExit(f"runs disagree on input mode: {modes}")
    input_mode = modes.pop() if modes else "wrench"

    flights = bp.load_melon(arguments.source, input_mode, arguments.pattern)
    if not flights:
        raise SystemExit(f"no flights matching {arguments.pattern!r} under {arguments.source}")
    truth = bp.rolling_windows(flights, arguments.horizon, arguments.stride)
    step = 0.01
    print(f"\n{arguments.source.name}/{arguments.pattern}: {len(flights)} flights -> "
          f"{truth.shape[0]} windows of {arguments.horizon} steps, stride {arguments.stride}")

    rows: dict[str, dict[str, float]] = {}
    rows["Naive"] = table7_row(truth, bl.naive_hold(truth))
    rows["Physics (IDSIA)"] = table7_row(truth, bl.idsia_physics(truth, step, input_mode))
    seed_rows, seed_std, diverged = {}, {}, {}
    for label, run_dir, run, _mode, checkpoint_step in runs:
        if run["config"]["model"]["name"] in SDE_PACKAGES and arguments.sde_seeds > 0:
            seed_rows[label], diverged[label] = sde_seed_rows(run_dir, checkpoint_step, truth, arguments.sde_seeds, arguments.batch)
            rows[label], seed_std[label] = mean_rows(seed_rows[label])
            continue
        model, _params, _setup = evaluation.build_model(run)
        prediction = bp.batched_rollout(model, truth, step, arguments.batch) \
            if hasattr(bp, "batched_rollout") else evaluation.rollout(model, truth, step)
        rows[label] = table7_row(truth, prediction)

    width = max(len(name) for name in rows) + 2
    columns = [(k, c) for k, _l in KEYS for c in ("h1", "h10", "h50", "cum")]
    header = "".join(f"{k + '.' + c:>11s}" for k, c in columns)
    print(f"\n{'model':{width}s}" + header)
    for name, values in rows.items():
        print(f"{name:{width}s}" + "".join(f"{values[k][c]:>11.4f}" for k, c in columns))

    folder = EVAL_ROOT / arguments.output_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "split_protocol_table.json").write_text(json.dumps({
        "source": str(arguments.source), "pattern": arguments.pattern, "flights": len(flights),
        "windows": int(truth.shape[0]), "horizon": arguments.horizon, "stride": arguments.stride,
        "input_mode": input_mode, "generated_at": datetime.now().astimezone().isoformat(),
        "table7": rows,
        "runs": {label: str(Path(d).resolve()) + (f"@{s}" if s else "") for label, d, _r, _m, s in runs},
        "checkpoints": {label: str(r["checkpoint"]) for label, _d, r, _m, _s in runs},
        "sde_seeds": arguments.sde_seeds,
        "sde_seed_rows": seed_rows, "sde_seed_std": seed_std, "sde_nonfinite_windows_per_seed": diverged,
        "note": "Naive and Physics recomputed on these flights; learned baselines are not available off the "
                "paper's test-split table.",
    }, indent=2) + "\n")
    print(f"\nwritten {folder / 'split_protocol_table.json'}")


if __name__ == "__main__":
    main()

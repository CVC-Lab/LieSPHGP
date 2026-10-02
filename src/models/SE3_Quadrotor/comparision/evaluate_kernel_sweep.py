"""Kernel sensitivity: one row per GP kernel setting, accuracy and calibration side by side.

NeurIPS reviewer 6Ap3 (W4) asked how sensitive the results are to the kernel choice. The variational GP uses a
Matern(nu) x periodic random-feature kernel whose hyperparameters are *fixed*, not learned
(``model_backend: utils-gp-model``), so the question is a genuine sensitivity study rather than a tuning artefact:

    k(x, x') = k_Matern-nu(x_m, x'_m ; l_m) * k_periodic(x_p, x'_p ; l_p),
    k_Matern-nu -> squared exponential as nu -> infinity; nu = 1/2 is the rough (Ornstein-Uhlenbeck) limit.

Each run is scored twice on the same held-out flights: open-loop accuracy and posterior calibration through
``open_loop.open_loop_metrics`` (identical to every other table in results.md), and operator identification
through ``report_evaluation.product_metrics`` on the shared D0 reference flights, so that a kernel that helps
the trajectory but corrupts the physics is visible.

Usage:
    python -m src.models.SE3_Quadrotor.comparision.evaluate_kernel_sweep \
        --run LABEL=DIR [--run LABEL=DIR ...] \
        --dataset datasets/QUADROTOR-DATASET-EVALSET/EVALSET_CF2P_10s_h0p01_clean.pkl@heldout \
        [--horizon-seconds 1.0] [--samples 50] [--flights 10]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import open_loop as openloop
from . import report_evaluation as evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
REFERENCE = PROJECT_ROOT / "datasets/QUADROTOR-EVAL-REFERENCE/D0_CF2P_PID_contact-free_nonlinear-damping-c0p5_3s_seed0.pkl"

ROWS = (("position_rmse_m", "position RMSE (m)", "{:.4f}"),
        ("attitude_rmse_deg", "attitude RMSE (deg)", "{:.2f}"),
        ("valid_prediction_seconds_median", "VPT median (s)", "{:.3f}"),
        ("coverage_95", "coverage at 2 sigma (target 0.954)", "{:.3f}"),
        ("nlpd", "NLPD, position (nats)", "{:.3f}"),
        ("sharpness_m", "sharpness, mean 2 sigma (m)", "{:.4f}"),
        ("sigma_error_spearman", "Spearman rho(sigma, error)", "{:+.3f}"),
        ("operator_error_percent", "operator error, mean of 5 (%)", "{:.1f}"),
        ("operator_error_identified_percent", "operator error, identified 4 (%)", "{:.1f}"),
        ("thrust_gain_percent", "  thrust gain mu g_f (%)", "{:.1f}"),
        ("torque_gain_percent", "  torque gain M2^-1 g_tau (%)", "{:.1f}"),
        ("gravity_percent", "  gravity mu grad V (%)", "{:.1f}"),
        ("damping_v_percent", "  damping mu D_v (%)", "{:.1f}"),
        ("damping_w_percent", "  damping M2^-1 D_w (%)", "{:.0f}"))


def segment(truth: np.ndarray, seconds: float, step: float) -> np.ndarray:
    """Identical to ``evaluate_ensemble.segment``: each flight is cut into non-overlapping windows of
    ``seconds``, so a 10 s flight at 1 s gives nine windows rather than only its first second. The two
    tables then sit on exactly the same held-out windows."""
    if seconds <= 0:
        return truth
    keep = int(round(seconds / step)) + 1
    if keep >= truth.shape[1]:
        return truth
    pieces = truth.shape[1] // keep
    return truth[:, : pieces * keep].reshape(-1, keep, truth.shape[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="LABEL=DIR, repeat once per kernel setting")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--horizon-seconds", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--flights", type=int, default=10, help="held-out flights for the open-loop half")
    parser.add_argument("--reference-flights", type=int, default=10, help="D0 flights for the operator half")
    parser.add_argument("--output-name", default=None)
    arguments = parser.parse_args()

    loaded = openloop.load_open_loop_set(arguments.dataset, arguments.flights)
    truth, step = loaded["truth"], loaded["dt"]
    truth = segment(truth, arguments.horizon_seconds, step)
    horizon = (truth.shape[1] - 1) * step
    print(f"\n{loaded['name']}: {truth.shape[0]} flights x {horizon:.2f} s at dt = {step}")

    # the shared D0 reference flights, for the physics half of the table
    reference = evaluation.load_common_dataset(arguments.reference)
    vehicle = evaluation.vehicle_constants(reference["settings"])
    reference_truth = evaluation.shared_truth(reference, arguments.reference_flights)
    targets = evaluation.product_targets(reference_truth, vehicle)
    product_keys = [k for k, _ in evaluation.PRODUCT_LABELS]

    columns: list[tuple[str, dict]] = []
    for spec in arguments.run:
        label, _, run_dir = spec.rpartition("=")
        if not label:                    # no "=" at all: rpartition puts everything in run_dir
            label, run_dir = Path(spec).name[:18], spec
        if not run_dir:
            raise SystemExit(f"--run wants LABEL=DIR, got {spec!r}")
        run = evaluation.load_run(Path(run_dir), None)
        model, params, gp_setup = evaluation.build_model(run)
        prediction, compute_ms = openloop.timed_rollout(model, truth, step)
        if prediction is None:
            print(f"  {label:18s} non-finite rollout, dropped")
            continue
        samples = openloop.posterior_rollouts(params, gp_setup, truth, arguments.samples,
                                              int(run["config"]["report"]["random_seed"]), step)
        metrics = dict(openloop.open_loop_metrics(truth, prediction, samples, step, compute_ms))
        products = evaluation.gauge_invariant_products(model, reference_truth)
        per_product = evaluation.product_metrics(products, targets, reference_truth)
        errors = {k: float(np.mean(per_product[k]["relative_rms_error_percent"])) for k in product_keys}
        metrics["operator_error_percent"] = float(np.mean(list(errors.values())))
        # the rotational damping M2^-1 D_w is unidentified in every setting (it barely moves a 1 s flight),
        # so its hundreds-of-percent error would swamp a plain mean; the four identified products get their own row
        identified = [v for k, v in errors.items() if k != "damping_w"]
        metrics["operator_error_identified_percent"] = float(np.mean(identified))
        for key, value in errors.items():
            metrics[f"{key}_percent"] = value
        metrics["operator_error_by_product"] = errors
        metrics["run"] = str(Path(run_dir).resolve())
        columns.append((label, metrics))
        print(f"  {label:18s} position RMSE {metrics['position_rmse_m']:.4f}   operators {metrics['operator_error_percent']:.1f}%")

    width = max(len(name) for name, _ in columns) + 2
    print(f"\n{'metric':38s}" + "".join(f"{name:>{width}s}" for name, _ in columns))
    for key, label, fmt in ROWS:
        cells = []
        for _name, m in columns:
            value = m.get(key)
            cells.append(fmt.format(value) if isinstance(value, (int, float)) and np.isfinite(value) else "—")
        print(f"{label:38s}" + "".join(f"{c:>{width}s}" for c in cells))

    # spread across the sweep, which is the number the sensitivity claim rests on
    print()
    for key, label, fmt in ROWS:
        values = [m[key] for _n, m in columns if isinstance(m.get(key), (int, float)) and np.isfinite(m[key])]
        if len(values) > 1:
            lo, hi = min(values), max(values)
            span = f"{fmt.format(lo)} to {fmt.format(hi)}"
            ratio = f"   x{hi / lo:.2f}" if lo > 0 else ""
            print(f"{label:38s} {span}{ratio}")

    folder = EVAL_ROOT / (arguments.output_name or f"{datetime.now().astimezone():%d-%m-%H-%M}_kernel-sensitivity")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel_sensitivity.json").write_text(json.dumps({
        "dataset": arguments.dataset, "flights": int(truth.shape[0]), "horizon_seconds": horizon,
        "reference": str(arguments.reference), "posterior_samples": arguments.samples,
        "columns": {name: {k: v for k, v in m.items() if not isinstance(v, np.ndarray)} for name, m in columns},
    }, indent=2, default=float) + "\n")
    print(f"\nwritten {folder / 'kernel_sensitivity.json'}")


if __name__ == "__main__":
    main()

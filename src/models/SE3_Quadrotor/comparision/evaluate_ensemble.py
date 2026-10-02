"""Deep ensemble versus GP posterior: the same uncertainty metrics for both mechanisms.

NeurIPS reviewer 6Ap3 (W3) asked why a Gaussian process rather than a scalable alternative such as a deep
ensemble. This script answers it on equal terms: N independently seeded point-estimate models (PH-NN-LieIMEX
or PH-NODE) form an ensemble whose member rollouts play exactly the role the GP's posterior weight samples
play, and both are scored with ``open_loop.open_loop_metrics`` - the same coverage, NLPD, sharpness and
Spearman rho, on the same held-out flights, at the same horizon.

    ensemble:  x^(i) = Roll(x0, u; theta_i),  theta_i from seed i      (epistemic across training runs)
    GP:        x^(i) = Roll(x0, u; w^(i)),    w^(i) ~ q(w)             (epistemic across the posterior)

Usage:
    python -m src.models.SE3_Quadrotor.comparision.evaluate_ensemble \
        --member <nn run> [--member ...] --gp <gp run> \
        --dataset datasets/QUADROTOR-DATASET-EVALSET/EVALSET_CF2P_10s_h0p01_clean.pkl@heldout \
        [--horizon-seconds 1.0] [--samples 50]
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
ROWS = (("position_rmse_m", "position RMSE (m)", "{:.4f}"),
        ("attitude_rmse_deg", "attitude RMSE (deg)", "{:.2f}"),
        ("valid_prediction_seconds_median", "VPT median (s)", "{:.3f}"),
        ("posterior_samples", "ensemble / posterior members", "{:.0f}"),
        ("coverage_95", "coverage at 2 sigma (target 0.954)", "{:.3f}"),
        ("nlpd", "NLPD, position (nats)", "{:.3f}"),
        ("sharpness_m", "sharpness, mean 2 sigma (m)", "{:.4f}"),
        ("sigma_error_spearman", "Spearman rho(sigma, error)", "{:+.3f}"),
        ("ensemble_valid_seconds_median", "ensemble VPT median (s)", "{:.3f}"))


def segment(truth: np.ndarray, seconds: float, step: float) -> np.ndarray:
    keep = int(round(seconds / step)) + 1
    if keep >= truth.shape[1]:
        return truth
    pieces = truth.shape[1] // keep
    return truth[:, : pieces * keep].reshape(-1, keep, truth.shape[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member", action="append", required=True, help="a point-estimate run; repeat for each seed")
    parser.add_argument("--gp", action="append", default=[], help="a GP run scored with its own posterior")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon-seconds", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=50, help="posterior draws for the GP column")
    parser.add_argument("--flights", type=int, default=10)
    parser.add_argument("--output-name", default=None)
    arguments = parser.parse_args()

    loaded = openloop.load_open_loop_set(arguments.dataset, arguments.flights)
    truth, step = loaded["truth"], loaded["dt"]
    if arguments.horizon_seconds > 0:
        truth = segment(truth, arguments.horizon_seconds, step)
    horizon = (truth.shape[1] - 1) * step
    print(f"\n{loaded['name']}: {truth.shape[0]} flights x {horizon:.2f} s at dt = {step}")

    columns: list[tuple[str, dict]] = []

    member_rollouts = []
    for run_dir in arguments.member:
        run = evaluation.load_run(Path(run_dir), None)
        model, _params, _setup = evaluation.build_model(run)
        prediction, compute_ms = openloop.timed_rollout(model, truth, step)
        if prediction is None:
            print(f"  member {Path(run_dir).name[:50]}: non-finite rollout, dropped")
            continue
        member_rollouts.append(prediction)
        point = openloop.open_loop_metrics(truth, prediction, None, step, compute_ms)
        print(f"  member {Path(run_dir).name[:56]:56s} position RMSE {point['position_rmse_m']:.4f}")
    if len(member_rollouts) < 2:
        raise SystemExit("an ensemble needs at least two finite members")
    members = np.stack(member_rollouts, axis=0)
    # the ensemble's point prediction is the member mean; its band is the member spread
    ensemble_mean = members.mean(axis=0)
    metrics = openloop.open_loop_metrics(truth, ensemble_mean, members, step, float("nan"))
    columns.append((f"deep ensemble ({members.shape[0]} seeds)", metrics))
    # the single best member, so the ensemble's own gain is visible
    single = openloop.open_loop_metrics(truth, member_rollouts[0], None, step, float("nan"))
    columns.append(("single member (seed 0)", single))

    for run_dir in arguments.gp:
        run = evaluation.load_run(Path(run_dir), None)
        model, params, gp_setup = evaluation.build_model(run)
        prediction, compute_ms = openloop.timed_rollout(model, truth, step)
        samples = openloop.posterior_rollouts(params, gp_setup, truth, arguments.samples,
                                              int(run["config"]["report"]["random_seed"]), step)
        columns.append((f"GP posterior ({samples.shape[0]} draws)",
                        openloop.open_loop_metrics(truth, prediction, samples, step, compute_ms)))
        print(f"  GP     {Path(run_dir).name[:56]:56s} position RMSE {columns[-1][1]['position_rmse_m']:.4f}")

    width = max(len(name) for name, _ in columns) + 2
    print(f"\n{'metric':38s}" + "".join(f"{name:>{width}s}" for name, _ in columns))
    for key, label, fmt in ROWS:
        cells = []
        for _name, m in columns:
            value = m.get(key)
            cells.append(fmt.format(value) if isinstance(value, (int, float)) and np.isfinite(value) else "—")
        print(f"{label:38s}" + "".join(f"{c:>{width}s}" for c in cells))

    folder = EVAL_ROOT / (arguments.output_name or
                          f"{datetime.now().astimezone():%d-%m-%H-%M}_ensemble-vs-gp")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ensemble_vs_gp.json").write_text(json.dumps({
        "dataset": arguments.dataset, "flights": int(truth.shape[0]), "horizon_seconds": horizon,
        "members": [str(Path(m).resolve()) for m in arguments.member],
        "gp_runs": [str(Path(g).resolve()) for g in arguments.gp],
        "columns": {name: {k: v for k, v in m.items() if not isinstance(v, np.ndarray)} for name, m in columns},
    }, indent=2, default=float) + "\n")
    print(f"\nwritten {folder / 'ensemble_vs_gp.json'}")


if __name__ == "__main__":
    main()

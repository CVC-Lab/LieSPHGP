"""Operator identification of trained SE(3) models against the simulator's constants, without the full report.

For every run: the five gauge-invariant products (thrust gain mu g_f, torque gain M2^-1 g_tau, gravity mu grad V,
translational damping mu D_v, rotational damping M2^-1 D_w) evaluated along the D0 reference flights and scored as
relative RMS error against the analytic targets (report_evaluation.product_metrics), plus - for SDE checkpoints -
the learned process-noise scales next to the ground-truth equivalent of the dataset's OU gust,
    sigma_v^GT = (sigma_f / m) sqrt(2 tau),   sigma_f = fraction * m g.

Usage: python identification_table.py --run DIR [--run DIR ...] [--reference <D0 pickle>]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]  # comparision -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation  # noqa: E402

REFERENCE = PROJECT_ROOT / "datasets/QUADROTOR-EVAL-REFERENCE/D0_CF2P_PID_contact-free_nonlinear-damping-c0p5_3s_seed0.pkl"


def gust_equivalent(dataset_path: Path) -> tuple[float | None, dict]:
    if not dataset_path.is_file():
        return None, {}          # e.g. a training set that has since been archived or renamed
    with dataset_path.open("rb") as handle:
        settings = pickle.load(handle)["settings"]
    cfg = settings.get("config", {}); gust = cfg.get("gusts", {})
    env = cfg.get("environment", {})
    if not gust.get("enabled"):
        return None, {}
    mass = float(settings.get("vehicle_parameters", {}).get("mass", env.get("mass", 0.027)))
    fraction, tau = float(gust["sigma_fraction_of_weight"]), float(gust["time_constant_seconds"])
    sigma_f = fraction * mass * 9.81
    return (sigma_f / mass) * np.sqrt(2.0 * tau), {"fraction": fraction, "tau": tau, "mass": mass, "sigma_f_newtons": sigma_f}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--flights", type=int, default=10)
    arguments = parser.parse_args()
    data = evaluation.load_common_dataset(arguments.reference)
    vehicle = evaluation.vehicle_constants(data["settings"])
    truth = evaluation.shared_truth(data, arguments.flights)
    targets = evaluation.product_targets(truth, vehicle)
    keys = [k for k, _ in evaluation.PRODUCT_LABELS]
    print(f"\nreference flights: {truth.shape[0]} x {truth.shape[1]} samples from {arguments.reference.name}")
    print(f"{'run':58s}" + "".join(f"{k:>13s}" for k in keys) + f"{'mean':>8s}   sigma_v (GT)   sigma_w")
    rows = {}
    for run_dir in arguments.run:
        run = evaluation.load_run(Path(run_dir), None)
        model, params, _ = evaluation.build_model(run)
        products = evaluation.gauge_invariant_products(model, truth)
        metrics = evaluation.product_metrics(products, targets, truth)
        errors = [float(np.mean(metrics[k]["relative_rms_error_percent"])) for k in keys]   # per-flight values -> mean
        dataset = evaluation.resolve_project_path(run["config"]["data"]["dataset_path"]) if hasattr(evaluation, "resolve_project_path") else PROJECT_ROOT / run["config"]["data"]["dataset_path"]
        gt_sigma, gust = gust_equivalent(Path(dataset))
        process = params.get("process") if isinstance(params, dict) else None
        sig = None if process is None else np.exp(np.asarray(process["log_sigma"], dtype=float)).tolist()
        name = Path(run_dir).name.split("_", 2)[-1][:58]
        sig_text = "" if sig is None else f"   {sig[0]:.3f} ({gt_sigma:.2f})" + f"   {sig[1]:.3f}" if gt_sigma is not None else (f"   {sig[0]:.3f} (n/a)   {sig[1]:.3f}" if sig else "")
        print(f"{name:58s}" + "".join(f"{e:>12.1f}%" for e in errors) + f"{np.mean(errors):>7.1f}%" + sig_text)
        rows[Path(run_dir).name] = {"relative_rms_error_percent": dict(zip(keys, errors)), "process_sigma": sig, "gust_equivalent_sigma_v": gt_sigma, "gust": gust}
    out = Path(arguments.run[0]).parent / "identification_table_latest.json"
    out.write_text(json.dumps(rows, indent=2) + "\n"); print(f"\nwritten {out}")


if __name__ == "__main__":
    main()

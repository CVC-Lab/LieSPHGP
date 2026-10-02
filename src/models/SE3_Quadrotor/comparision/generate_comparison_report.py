"""Two-model comparison report: PH-GP-LieIMEX and PH-NN-LieIMEX on the same pages.

Same page types and styling as the single-model report; every curve page carries both trained
models plus the simulator's analytic operators, and every table carries one column per model.
The per-model closed-loop tracking plots are images, so they appear once per model.

Usage:
    python -m src.models.SE3_Quadrotor.comparision.generate_comparison_report \
        --run <gp_run_dir> --run <nn_run_dir> [--evaluation-dataset <pkl>] [--output-name name.pdf]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import report_evaluation as evaluation
from . import report_figures as figures_module
from .generate_report import COMMON_EVALUATION_DATASET, REPORT_ROOT, _is_gp, _model_label, _noise_level, _tag
from .report_controller import controller_comparison, run_controller
from ..ph_gp_lie_imex.config import load_config, resolve_project_path

BENCHMARK_REPEATS = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="run directory; give the flag twice, GP first then NN")
    parser.add_argument("--selected-step", type=int, action="append", default=None)
    parser.add_argument("--evaluation-dataset", type=Path, default=None)
    parser.add_argument("--controller-seconds", type=float, default=20.0)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def _prepare(run_dir: Path, selected_step: int | None, data: dict[str, Any], truth: np.ndarray,
             vehicle: dict[str, Any], benchmark_repeats: int, device, controller_seconds: float) -> dict[str, Any]:
    """Everything the report needs from one trained model."""
    run = evaluation.load_run(run_dir, selected_step)
    config = run["config"]
    noise = _noise_level(config)
    label = _model_label(config, noise)
    flat = label.replace("\n", " ")
    model, params, gp_setup = evaluation.build_model(run)
    run["levels"] = evaluation.learned_levels(params)
    is_gp = _is_gp(config)
    # Epistemic band: independent posterior weight samples rolled out on trajectory 0, same x0 and u(t).
    samples, sample_count = None, 0
    if is_gp:
        requested = int(config["report"].get("posterior_samples", 10) or 0)
        if requested > 0:
            print(f"Rolling out {requested} posterior weight samples for {flat}...", flush=True)
            samples, sample_count = evaluation.posterior_sample_rollouts(
                params, gp_setup, truth, requested, int(config["report"]["random_seed"]))
    print(f"Benchmarking {flat}...", flush=True)
    benchmarks = evaluation.benchmark(model, truth, benchmark_repeats, device)
    prediction = benchmarks.pop("prediction")
    subnets = evaluation.query_subnetworks(model, truth)
    targets = evaluation.analytic_targets(truth, vehicle)
    aligned, fits, subnet_metrics = evaluation.align_subnetworks(subnets, targets)
    products = evaluation.gauge_invariant_products(model, truth)
    product_targets = evaluation.product_targets(truth, vehicle)
    selected = run["selected_step"]
    print(f"Running the closed-loop PyBullet controller for {flat}...", flush=True)
    controller = run_controller(
        run, model, run_dir / ("controller" if selected == run["total_steps"] else f"controller_step{selected:05d}"),
        model_label=flat, training_dataset=resolve_project_path(config["data"]["dataset_path"]),
        vehicle=vehicle, duration_seconds=controller_seconds, seed=int(config["report"]["random_seed"]),
        device_text=f"{device.platform}:{device.id}",
        use_dissipation=config["report"].get("controller_use_dissipation"),
    )
    return {
        "run": run, "config": config, "noise": noise, "label": label, "flat": flat, "selected": selected,
        "params": params, "benchmarks": benchmarks, "prediction": prediction,
        "errors": evaluation.rollout_errors(truth, prediction, vehicle),
        "subnets": subnets, "aligned": aligned, "fits": fits, "subnet_metrics": subnet_metrics,
        "products": products, "product_targets": product_targets,
        "product_metrics": evaluation.product_metrics(products, product_targets, truth),
        "targets": targets, "controller": controller, "comparison": controller_comparison(controller),
        "standard_deviations": evaluation.posterior_standard_deviations(params),
        "is_gp": is_gp, "samples": samples, "sample_count": sample_count,
    }


def generate_comparison_report(run_dirs: list[Path], *, selected_steps: list[int | None] | None = None,
                               evaluation_dataset: Path | None = None, controller_seconds: float = 20.0,
                               benchmark_repeats: int = BENCHMARK_REPEATS, output_name: str | None = None) -> Path:
    if len(run_dirs) < 2:
        raise ValueError("Give at least two --run directories")
    selected_steps = list(selected_steps or [None] * len(run_dirs))
    device = jax.devices()[0]
    first_config = load_config(sorted(Path(run_dirs[0]).glob("*.yaml"))[0])
    dataset = Path(evaluation_dataset) if evaluation_dataset is not None else (
        Path(resolve_project_path(first_config["report"]["evaluation_dataset"]))
        if first_config["report"].get("evaluation_dataset") else COMMON_EVALUATION_DATASET)
    data = evaluation.load_common_dataset(dataset)
    vehicle = evaluation.vehicle_constants(data["settings"])
    truth = evaluation.shared_truth(data, int(first_config["report"]["trajectory_count"]))
    time_axis = np.arange(truth.shape[1]) * evaluation.DT
    horizon = float(time_axis[-1])

    models = [_prepare(Path(d), s, data, truth, vehicle, benchmark_repeats, device, controller_seconds)
              for d, s in zip(run_dirs, selected_steps)]
    for entry, (colour, style) in zip(models, figures_module.COMPARISON_STYLES):
        entry["colour"], entry["style"] = colour, style

    gt_label, gt_colour, gt_style = figures_module.GROUND_TRUTH_MODEL_STYLE
    print("Benchmarking the ground-truth-operator Lie-IMEX model...", flush=True)
    gt_benchmarks = evaluation.benchmark(evaluation.ground_truth_model(vehicle), truth, benchmark_repeats, device)
    gt_prediction = gt_benchmarks.pop("prediction")
    gt_errors = evaluation.rollout_errors(truth, gt_prediction, vehicle)

    error_series = [(m["label"], m["colour"], m["style"], m["errors"]) for m in models] + [(gt_label, gt_colour, gt_style, gt_errors)]
    prediction_series = [(m["label"], m["colour"], m["style"], m["prediction"]) for m in models] + [(gt_label, gt_colour, gt_style, gt_prediction)]
    benchmark_series = [(m["label"], m["colour"], m["style"], m["benchmarks"]) for m in models] + [(gt_label, gt_colour, gt_style, gt_benchmarks)]
    product_entries = [(m["label"], m["colour"], m["style"], m["products"]) for m in models]
    raw_entries = [(m["label"], m["colour"], m["style"], m["subnets"], m["fits"]) for m in models]
    fixed_entries = [(m["label"], m["colour"], m["style"], m["aligned"], m["fits"]) for m in models]
    sample_bands = [(m["label"], m["colour"], m["samples"]) for m in models if m["samples"] is not None]
    targets = models[0]["targets"]
    product_targets = models[0]["product_targets"]

    noise_text = ", ".join(f"{m['flat'].split(chr(10))[0]}: {m['noise']:g}" for m in models)
    title = (f"Model comparison on the contact-free {vehicle.get('damping_law', 'nonlinear')}-damping "
             f"Gym-PyBullet CF2P flight\nSame data, same windows, same Lie-IMEX integrator, same evaluation; "
             f"training observation noise {models[0]['noise']:g}\n" + " vs ".join(m["flat"] for m in models))

    pages: list[plt.Figure] = [
        figures_module.summary_figure(error_series, horizon, title, truth=truth),
        figures_module.subnetwork_summary_multi([(m["label"], m["subnet_metrics"]) for m in models], horizon),
        figures_module.product_summary_multi([(m["label"], m["product_metrics"]) for m in models],
                                             evaluation.PRODUCT_LABELS, horizon),
    ]
    for key, name, units in (
        ("thrust_gain", "Thrust gain μ·g_f", "m/s² per N"),
        ("torque_gain", "Torque gain M2⁻¹·g_τ (truth J⁻¹)", "rad/s² per N·m"),
        ("gravity", "Gravity μ·∇V (truth g·e_z)", "m/s²"),
        ("damping_v", "Translational damping μ·Dv", "1/s"),
        ("damping_w", "Rotational damping M2⁻¹·Dω", "1/s"),
    ):
        pages.append(figures_module.product_trajectory_multi(product_entries, product_targets, time_axis, key, name, units))
    pages.append(figures_module.damping_speed_multi(product_entries, product_targets, truth, vehicle))

    def _curve(entry, key, test):
        stats = entry["run"]["stats"]
        if test:
            source = {"loss": stats["test_total"], "position": stats["position"], "attitude": stats["attitude"],
                      "linear_velocity": stats["linear_velocity"], "angular_velocity": stats["angular_velocity"]}
            return stats["step"], source[key]
        stepped = "train_step" in stats and stats["train_step"].size > 0
        steps = stats["train_step"] if stepped else stats["step"]
        source = {"loss": stats["train_loss"] if stepped else stats["train_total"],
                  "position": stats["train_position"] if stepped else stats["position"],
                  "attitude": stats["train_attitude"] if stepped else stats["attitude"],
                  "linear_velocity": stats["train_linear_velocity"] if stepped else stats["linear_velocity"],
                  "angular_velocity": stats["train_angular_velocity"] if stepped else stats["angular_velocity"]}
        return steps, source[key]

    for test, prefix in ((False, "Train"), (True, "Test-window")):
        for key, name, ylabel in (("loss", "total loss", "loss"), ("position", "position loss", "position MSE"),
                                  ("attitude", "attitude geodesic squared", "geodesic squared"),
                                  ("linear_velocity", "linear-velocity loss", "velocity MSE"),
                                  ("angular_velocity", "angular-velocity loss", "angular-velocity MSE")):
            entries = [(m["label"], m["colour"], m["style"], *_curve(m, key, test)) for m in models]
            pages.append(figures_module.training_figure_multi(entries, f"{prefix} {name}", ylabel, test=test))

    for key, name, ylabel in (("position", "Position error vs GT", "squared position error"),
                              ("attitude", "Attitude geodesic error vs GT", "squared geodesic error"),
                              ("velocity", "Linear-velocity error vs GT", "squared velocity error"),
                              ("omega", "Angular-velocity error vs GT", "squared angular-velocity error")):
        pages.append(figures_module.error_figure(error_series, time_axis, key, name, ylabel))
    energy_ensemble, energy_single = figures_module.energy_figures(prediction_series, truth, time_axis, vehicle)
    note = "Learned dissipation matrices; dotted is analytic GT"
    pages.extend([
        energy_ensemble, energy_single,
        figures_module.geometry_figure(error_series, time_axis, "determinant", "SO(3) violation — determinant", "absolute determinant error"),
        figures_module.geometry_figure(error_series, time_axis, "orthogonality", "SO(3) violation — orthogonality", "orthogonality Frobenius error"),
        figures_module.state_ensemble_figure(prediction_series, truth, time_axis, sample_bands=sample_bands),
        figures_module.state_single_figure(prediction_series, truth, time_axis, sample_bands=sample_bands),
        figures_module.phase_figure(prediction_series, truth),
        figures_module.matrix_trajectory_multi(raw_entries, targets, time_axis, "m1", "Translational inverse mass M1^-1(x)", gauge_fixed=False),
        figures_module.matrix_trajectory_multi(fixed_entries, targets, time_axis, "m1", "Translational inverse mass M1^-1(x)", gauge_fixed=True),
        figures_module.matrix_trajectory_multi(raw_entries, targets, time_axis, "m2", "Rotational inverse inertia M2^-1(R)", gauge_fixed=False),
        figures_module.matrix_trajectory_multi(fixed_entries, targets, time_axis, "m2", "Rotational inverse inertia M2^-1(R)", gauge_fixed=True),
        figures_module.matrix_trajectory_multi(raw_entries, targets, time_axis, "dv", "Translational dissipation Dv(v_body)", gauge_fixed=False, note=note),
        figures_module.matrix_trajectory_multi(fixed_entries, targets, time_axis, "dv", "Translational dissipation Dv(v_body)", gauge_fixed=True, note=note),
        figures_module.matrix_trajectory_multi(raw_entries, targets, time_axis, "dw", "Rotational dissipation Domega(omega_body)", gauge_fixed=False, note=note),
        figures_module.matrix_trajectory_multi(fixed_entries, targets, time_axis, "dw", "Rotational dissipation Domega(omega_body)", gauge_fixed=True, note=note),
        figures_module.control_trajectory_multi(raw_entries, targets, time_axis, gauge_fixed=False),
        figures_module.control_trajectory_multi(fixed_entries, targets, time_axis, gauge_fixed=True),
        figures_module.potential_trajectory_multi(raw_entries, targets, time_axis, gauge_fixed=False),
        figures_module.potential_trajectory_multi(fixed_entries, targets, time_axis, gauge_fixed=True),
    ])

    columns = [m["label"] for m in models] + [gt_label]
    def _row(getter, gt_value):
        return [getter(m) for m in models] + [gt_value]
    pages.extend([
        figures_module._table_multi(
            "Fairness and computation",
            ["Model class", "Integrator and dissipation inputs", "Parameters", "Training observation noise", "Training steps", "Training wall time (min)",
             "Rollout median wall time (s)", "Transitions per wall second", "Full NFE per transition",
             "Peak device memory (MiB)", "Uncertainty"],
            columns,
            [_row(lambda m: "GP, random Fourier features + levels" if m["is_gp"] else f"MLP width {m['config']['model'].get('hidden_dim')}", "analytic operators"),
             _row(lambda m: f"{m['config']['model']['solver']}, D inputs "
                            + ("v_b, omega_b" if m["is_gp"] or m["config"]["model"]["name"] != "ph_node" else "x, R"),
                  "lie-imex (reference)"),
             _row(lambda m: f"{m['run']['metadata']['parameter_count']:,}", "0"),
             _row(lambda m: f"{m['noise']:g}", "none"),
             _row(lambda m: str(m["run"]["total_steps"]), "0"),
             _row(lambda m: f"{m['run']['metadata']['elapsed_seconds'] / 60:.2f}", "0"),
             _row(lambda m: f"{m['benchmarks']['median_seconds']:.5f}", f"{gt_benchmarks['median_seconds']:.5f}"),
             _row(lambda m: f"{m['benchmarks']['transitions_per_second']:.1f}", f"{gt_benchmarks['transitions_per_second']:.1f}"),
             _row(lambda m: f"{m['benchmarks']['nfe_per_transition']:.1f}", f"{gt_benchmarks['nfe_per_transition']:.1f}"),
             _row(lambda m: f"{m['benchmarks']['peak_device_memory_bytes'] / 2**20:.1f}", f"{gt_benchmarks['peak_device_memory_bytes'] / 2**20:.1f}"),
             _row(lambda m: f"variational posterior over weights\n{m['sample_count']}-sample ±2σ band on the state pages" if m["is_gp"] else "none (point estimate)", "not applicable")],
            "Every model uses the same dataset, the same window curriculum and the same evaluation protocol. Each is rolled "
            "out in the integrator it was trained with, so PH-NODE is evaluated under RK4 and the others under second-order "
            "Lie-IMEX. The last column is the simulator's analytic operators in Lie-IMEX with the same recorded u(t): the "
            "floor any learned model can reach.",
            scale=(1.0, 1.6), fontsize=8,
        ),
        figures_module.compute_figure(benchmark_series, models[0]["selected"], models[0]["noise"], device.platform.upper()),
        figures_module.controller_table_multi(
            [(m["label"], m["controller"]["metadata"], m["comparison"], m["selected"]) for m in models], models[0]["noise"]),
    ])
    for entry in models:
        if entry["is_gp"]:
            pages.append(figures_module.gp_uncertainty_figure(entry["standard_deviations"]))
    for entry in models:
        for plot_key, plot_name in (("labeled_tracking_plot", "controller tracking plot"),
                                    ("labeled_trajectory_plot", "controller trajectory plot")):
            pages.append(figures_module.image_page(Path(entry["controller"]["metadata"]["plots"][plot_key]),
                                                   f"{entry['flat']} — {plot_name}"))

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%d-%m-%H-%M-%S")
    name = output_name or (f"{stamp}_report_comparison_PH-GP-vs-PH-NN-LieIMEX_"
                           f"obs-noise{_tag(models[0]['noise'])}_{dataset.stem[-24:]}_{len(pages)}-page.pdf")
    output = REPORT_ROOT / name
    with PdfPages(output) as pdf:
        for figure in pages:
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    print(f"PDF: {output}", flush=True)
    print(f"Pages: {len(pages)}", flush=True)
    return output


def main() -> None:
    args = parse_args()
    generate_comparison_report(
        args.run, selected_steps=args.selected_step, evaluation_dataset=args.evaluation_dataset,
        controller_seconds=args.controller_seconds, benchmark_repeats=args.benchmark_repeats,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    sys.exit(main())

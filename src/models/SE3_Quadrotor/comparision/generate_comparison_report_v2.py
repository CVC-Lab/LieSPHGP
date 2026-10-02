"""Spec-driven multi-model comparison report.

Implements the page map in ``comparision_report_format.md``:

  1      gauge-invariant physics identification, relative RMS + paired per-flight margin
  2-6    one page per gauge-invariant product
  7      training curves, every component on one axis
  8      test-window curves, every component on one axis
  9      GP-only objective terms (one page per Bayesian model)
  10     open-loop rollout error, scalar summary
  11     state trajectories, every held-out flight
  12     state trajectory, flight 0
  13-14  SO(3) violation, determinant and orthogonality
  15     physical energy
  16     state error, every block on one axis
  17     variational posterior uncertainty (one page per Bayesian model)
  18+    closed-loop controller plots, two per model

One invocation writes one folder under experiments/quadrotor/eval_runs, named the way a training run
folder is named, holding one PDF per section plus report_metadata.json:

  gauge-invariant-identification.pdf   spec pages 1-6
  training-and-test-losses.pdf         spec pages 7-9
  open-loop-rollout.pdf                spec pages 10-17
  closed-loop-controller.pdf           spec pages 18+

Usage:
    python -m src.models.SE3_Quadrotor.comparision.generate_comparison_report_v2 \
        --run <gp_run> --run <baseline_run> [--run ...] [--evaluation-dataset <pkl>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import report_evaluation as evaluation
from . import report_figures as figures
from . import report_figures_v2 as v2
from .generate_report import PROJECT_ROOT, _is_gp, _model_label, _noise_level, _tag
from . import open_loop as openloop      # 'open_loop' is a local figure list inside generate()
from .report_controller import (VALID_TRACKING_FRACTION, controller_comparison, euler_angles,
                                register_recorded_references, run_controller, save_plots,
                                valid_tracking_seconds, wrap_to_pi)
from ..ph_gp_lie_imex import network
from ..ph_gp_lie_imex.config import load_config, resolve_project_path

PRODUCT_PAGES = (
    # key, page title, the analytic truth, units, and the symbol used to name the y axis
    ("thrust_gain", "Thrust gain  μ·g_f", "(0, 0, 1/m)", "m/s² per N", "μ·g_f"),
    ("torque_gain", "Torque gain  M2⁻¹·g_τ", "J⁻¹, diagonal", "rad/s² per N·m", "M₂⁻¹·g_τ"),
    ("gravity", "Gravity  μ·∇V", "(0, 0, g)", "m/s²", "μ·∇V"),
    ("damping_v", "Translational damping  μ·Dv", "c(1+‖v_b‖)·I", "1/s", "μ·D_v"),
    ("damping_w", "Rotational damping  M2⁻¹·Dω", "c(1+‖ω_b‖)·I", "1/s", "M₂⁻¹·D_ω"),
)
POSTERIOR_SAMPLES = 10
OPEN_LOOP_SAMPLES = 50      # posterior weight draws behind the open-loop bands and calibration rows
BENCHMARK_REPEATS = 5
# One eval run writes one folder, named the way a training run folder is named, holding one PDF per
# section of comparision_report_format.md instead of a single 23-page file.
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
BOOKLETS = (
    ("physics-identification", "Gauge-invariant physics identification (spec pages 1-6, plus the paper view)"),
    ("training-and-test-losses", "Training and test loss curves (spec pages 7-9)"),
    ("open-loop-rollout", "Open-loop rollout on the held-out flights (spec pages 10-17)"),
    ("closed-loop-controller", "Closed-loop controller tracking (spec pages 18+)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--selected-step", type=int, action="append", default=None)
    parser.add_argument("--evaluation-dataset", type=Path, default=None)
    parser.add_argument("--controller-seconds", type=float, default=20.0)
    parser.add_argument("--controller-reference", type=str, action="append", default=None,
                        help="closed-loop reference shape; repeatable; default diamond. One controller booklet per shape.")
    parser.add_argument("--controller-dissipation", choices=("config", "on", "off"), default="config",
                        help="damping feedforward +Dv v_b, +Dw omega_b in the controller: 'config' = each run's own "
                             "report.controller_use_dissipation (may differ between models); 'on'/'off' = the same law for every model")
    parser.add_argument("--controller-recorded-reference", action="append", default=[],
                        help="fly every flight of this evaluation set (PATH[@split]) as a closed-loop reference, "
                             "named <SET>-<split>-flight<k>, in addition to --controller-reference (repeatable)")
    parser.add_argument("--open-loop-dataset", action="append", default=[],
                        help="additional evaluation set for the open-loop comparison (repeatable); the primary "
                             "--evaluation-dataset is always included. HARD-V5-style pickles are read from "
                             "their test_trajectories")
    parser.add_argument("--open-loop-samples", type=int, default=OPEN_LOOP_SAMPLES,
                        help="posterior weight samples behind the open-loop bands and calibration rows")
    parser.add_argument("--closed-loop-subfolders", action="store_true",
                        help="also write closed-loop/<reference>/ with one PDF per model and a per-shape comparison "
                             "table; skipped by default, since closed-loop/images/ and closed-loop/comparison.pdf "
                             "already carry the same content")
    parser.add_argument("--include-ground-truth", action="store_true",
                        help="add PH-GT: the simulator's analytic operators in the same port-Hamiltonian model "
                             "(identification floor and closed-loop ceiling)")
    parser.add_argument("--controller-refly", action="store_true",
                        help="re-fly every closed-loop reference even when a matching flight already exists in the run folder")
    parser.add_argument("--controller-only", action="store_true",
                        help="skip the identification, loss and open-loop booklets; run and report the controller only")
    parser.add_argument("--gust-sigma-fraction", type=float, default=0.0,
                        help="unobserved OU gust on the PLANT during the closed-loop flights, as a fraction of the weight "
                             "(0 = off; the WIND dataset used 0.10)")
    parser.add_argument("--gust-tau", type=float, default=0.5, help="OU gust time constant in seconds")
    parser.add_argument("--gust-seed", type=int, default=0)
    parser.add_argument("--controller-damping-schedule", choices=("none", "process-noise"), default="none",
                        help="process-noise: scale the damping-injection gains of SDE checkpoints by max(1, (sigma/sigma_ref)^2)")
    parser.add_argument("--damping-sigma-reference", type=float, nargs=2, default=(0.05, 0.1), metavar=("SIG_V", "SIG_W"))
    parser.add_argument("--output-name", type=str, default=None,
                        help="name of the eval-run folder; defaults to <DD-MM-HH-MM>_spec-comparison_"
                             "obs-noise<tag>_<dataset stem>")
    return parser.parse_args()


def _model_slug(label: str) -> str:
    """'PH-GP-LieIMEX levels / NLL, obs noise 0.25, priors 1/1' -> 'PH-GP-LieIMEX_priors-1-1'."""
    family = label.replace("\n", " ").split()[0]
    match = re.search(r"priors (\S+)/(\S+)", label)
    return family + (f"_priors-{match[1]}-{match[2]}".replace(".", "p") if match else "")


def _short_label(label: str) -> str:
    """The panel caption: family plus the prior weights, without the loss and noise boilerplate."""
    family = label.replace("\n", " ").split()[0]
    match = re.search(r"priors (\S+)/(\S+)", label)
    return f"{family}  priors {match[1]}/{match[2]}" if match else family


def _load_flight(directory: Path, reference: str, expected_mode: str | None, duration: float):
    """A previous flight of this reference in this folder, if its recorded law and horizon match; else None."""
    meta_path = directory / "controller_metadata.json"
    if not meta_path.exists() or not (directory / "controller_rollout.npz").exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("reference") != reference or abs(float(meta.get("duration_seconds", -1)) - duration) > 1e-9:
        return None
    if expected_mode is not None and meta.get("dissipation_mode") != expected_mode:
        return None
    if meta.get("reference_anchor") != "measured initial position; vehicle starts at the shape's own heading":
        return None                     # flown before the reference was anchored to the start state: re-fly it
    z = np.load(directory / "controller_rollout.npz")
    return {"directory": directory, "metadata": meta,
            "data": {"s_traj": z["s_traj"], "s_plan": z["s_plan"], "requested_rpm": z["requested_rpm"],
                     "applied_physical_control": z["applied_physical_control"],
                     "current_rotation": z["current_rotation"], "tilt_angle": z["tilt_angle"]}}


def _damping_scale(params, schedule: dict[str, Any] | None) -> tuple[tuple[float, float], dict[str, Any]]:
    """Damping-injection multipliers from the learned process noise (SDE checkpoints only).

    From the stochastic-passivity bound of the shaped closed loop, E[dH_d/dt] <= -xi^T R_d xi + tr(Sigma^T H_d'' Sigma)/2,
    the ultimate bound of the state scales with |Sigma|^2 / lambda_min(R_d). Holding that bound at the level a
    reference noise sigma_ref would give therefore asks for lambda_min(R_d) proportional to sigma^2, i.e.
        K_v <- K_v max(1, (sigma_v / sigma_ref,v)^2),   K_omega <- K_omega max(1, (sigma_omega / sigma_ref,omega)^2).
    A model without process noise (ODE, PH-NODE) keeps the nominal gains, so the comparison stays fair.
    """
    if not schedule or schedule.get("kind") != "process-noise":
        return (1.0, 1.0), {"kind": "none"}
    process = params.get("process") if isinstance(params, dict) else None
    if process is None:
        return (1.0, 1.0), {"kind": "process-noise", "applied": False, "reason": "checkpoint has no process noise"}
    sigma = np.exp(np.asarray(process["log_sigma"], dtype=np.float64))
    reference = np.asarray(schedule.get("sigma_reference", (0.05, 0.1)), dtype=np.float64)
    scale = np.maximum(1.0, (sigma / reference) ** 2)
    return (float(scale[0]), float(scale[1])), {"kind": "process-noise", "applied": True, "sigma": sigma.tolist(),
                                                 "sigma_reference": reference.tolist(), "scale": scale.tolist()}


def _gust_suffix(gust: dict[str, Any] | None) -> str:
    if not gust or float(gust.get("sigma_fraction_of_weight", 0.0)) <= 0.0:
        return ""
    return f"_gust{float(gust['sigma_fraction_of_weight']):g}-tau{float(gust['time_constant_seconds']):g}-s{int(gust.get('seed', 0))}"


def _prepare(run_dir, selected_step, truth, vehicle, device, controller_seconds,
             references=("diamond",), controller_only=False, dissipation="config", refly=False,
             gust=None, damping_schedule=None) -> dict[str, Any]:
    run = evaluation.load_run(Path(run_dir), selected_step)
    config = run["config"]
    noise = _noise_level(config)
    label = _model_label(config, noise)
    flat = label.replace("\n", " ")
    model, params, gp_setup = evaluation.build_model(run)
    run["levels"] = evaluation.learned_levels(params)
    is_gp = _is_gp(config)
    benchmarks = prediction = products = targets = None
    samples, state_samples = None, None
    if not controller_only:
        print(f"Rolling out {flat}...", flush=True)
        benchmarks = evaluation.benchmark(model, truth, BENCHMARK_REPEATS, device)
        prediction = benchmarks.pop("prediction")
        products = evaluation.gauge_invariant_products(model, truth)
        targets = evaluation.product_targets(truth, vehicle)
    if is_gp and not controller_only:
        print(f"Sampling {POSTERIOR_SAMPLES} posterior weight sets for the product bands...", flush=True)
        variational = network.DissipativeSE3HamODE(params, gp_setup)
        drawn = [evaluation.gauge_invariant_products(
            variational.sample(jax.random.PRNGKey(int(config["report"]["random_seed"]) + index)), truth)
            for index in range(POSTERIOR_SAMPLES)]
        # (samples, flights, time, ...): stacking rather than concatenating keeps the two axes apart,
        # so every flight can carry its own band instead of sharing flight 0's.
        samples = {key: np.stack([item[key] for item in drawn], axis=0) for key in drawn[0]}
        # Pages 11 and 12 need rolled-out samples, not pointwise ones: same weights, integrated forward.
        state_samples, _finite = evaluation.posterior_sample_rollouts(
            params, gp_setup, truth, POSTERIOR_SAMPLES, int(config["report"]["random_seed"]))
    selected = run["selected_step"]
    step_suffix = "" if selected == run["total_steps"] else f"_step{selected:05d}"
    controllers = {}
    damping_scale, schedule_record = _damping_scale(params, damping_schedule)
    for reference in references:
        # The diamond keeps its historical folder name; every other shape gets its own, so nothing is overwritten.
        folder = f"controller{step_suffix}" if reference == "diamond" else f"controller_{reference}{step_suffix}"
        if dissipation != "config":
            folder += f"_ff-{dissipation}"        # a forced law gets its own folder: the per-config flights are kept
        folder += _gust_suffix(gust)              # flights under a plant disturbance never alias the undisturbed ones
        if schedule_record.get("applied"):
            folder += "_dsched"
        expected_mode = None if dissipation == "config" else dissipation
        cached = None if refly else _load_flight(Path(run_dir) / folder, reference, expected_mode, float(controller_seconds))
        if cached is not None:
            print(f"Reusing the recorded closed-loop flight of {flat} on '{reference}' ({folder})", flush=True)
            controllers[reference] = cached
            continue
        print(f"Running the closed-loop controller for {flat} on '{reference}'...", flush=True)
        controllers[reference] = run_controller(
            run, model, Path(run_dir) / folder,
            model_label=flat, training_dataset=resolve_project_path(config["data"]["dataset_path"]),
            vehicle=vehicle, duration_seconds=controller_seconds, seed=int(config["report"]["random_seed"]),
            device_text=f"{device.platform}:{device.id}",
            use_dissipation=(config["report"].get("controller_use_dissipation") if dissipation == "config"
                             else dissipation == "on"),
            reference=reference, gust=gust, damping_scale=damping_scale, damping_schedule=schedule_record,
        )
    first = controllers[references[0]]
    return {
        "run": run, "config": config, "noise": noise, "label": label, "flat": flat, "is_gp": is_gp,
        "benchmarks": benchmarks, "prediction": prediction,
        "errors": None if controller_only else evaluation.rollout_errors(truth, prediction, vehicle),
        "products": products, "samples": samples, "state_samples": state_samples,
        "product_metrics": None if controller_only else evaluation.product_metrics(products, targets, truth),
        "targets": targets, "controller": first, "comparison": controller_comparison(first),
        "controllers": controllers, "comparisons": {name: controller_comparison(c) for name, c in controllers.items()},
        "standard_deviations": evaluation.posterior_standard_deviations(params),
        "model": model, "params": params, "gp_setup": gp_setup,
    }


GROUND_TRUTH_LABEL = "PH-GT analytic operators\nsimulator constants (no training)"
GROUND_TRUTH_FLIGHTS = EVAL_ROOT / ".ground-truth-flights"


def _prepare_ground_truth(truth, vehicle, device, controller_seconds, references, controller_only,
                          dissipation, refly, seed, dataset, gust=None) -> dict[str, Any]:
    """The same model entry as _prepare, but the six operators are the simulator's own constants.

    M1^-1 = I/m, M2^-1 = J^-1, V = m g z, g = selection matrix, Dv = m c (1+|v|) I, Dw = c (1+|w|) J.
    Nothing here is learned, so there are no training curves, no posterior bands and the relative RMS
    error of every gauge-invariant product is zero by construction: it is the floor of the table and
    the ceiling of the controller.
    """
    model = evaluation.ground_truth_model(vehicle)
    flat = GROUND_TRUTH_LABEL.replace("\n", " ")
    run = {"metadata": {"model_name": "ground_truth", "integrator": "lie-imex", "dataset_sha256": None},
           "directory": "n/a", "checkpoint": Path("ground-truth"), "checkpoint_sha256": "n/a",
           "stats": {}, "selected_step": 0, "total_steps": 0}
    benchmarks = prediction = products = targets = None
    if not controller_only:
        print(f"Rolling out {flat}...", flush=True)
        benchmarks = evaluation.benchmark(model, truth, BENCHMARK_REPEATS, device)
        prediction = benchmarks.pop("prediction")
        products = evaluation.gauge_invariant_products(model, truth)
        targets = evaluation.product_targets(truth, vehicle)
    controllers = {}
    mode = "on" if dissipation == "on" else "off" if dissipation == "off" else "config"
    for reference in references:
        directory = GROUND_TRUTH_FLIGHTS / (f"controller_{reference}_ff-{mode}" + _gust_suffix(gust))
        cached = None if refly else _load_flight(directory, reference,
                                                 None if dissipation == "config" else dissipation,
                                                 float(controller_seconds))
        if cached is not None:
            print(f"Reusing the recorded closed-loop flight of {flat} on '{reference}'", flush=True)
            controllers[reference] = cached
            continue
        print(f"Running the closed-loop controller for {flat} on '{reference}'...", flush=True)
        controllers[reference] = run_controller(
            run, model, directory, model_label=flat, training_dataset=dataset, vehicle=vehicle,
            duration_seconds=controller_seconds, seed=seed, device_text=f"{device.platform}:{device.id}",
            gust=gust,
            use_dissipation=(True if dissipation == "on" else False if dissipation == "off" else True),
            reference=reference, provider_description="the simulator's analytic operators")
    first = controllers[references[0]]
    return {
        "run": run, "config": {"model": {"name": "ground_truth"}}, "noise": 0.0, "label": GROUND_TRUTH_LABEL, "flat": flat, "is_gp": False,
        "synthetic": True, "benchmarks": benchmarks, "prediction": prediction,
        "errors": None if controller_only else evaluation.rollout_errors(truth, prediction, vehicle),
        "products": products, "samples": None, "state_samples": None,
        "product_metrics": None if controller_only else evaluation.product_metrics(products, targets, truth),
        "targets": targets, "controller": first, "comparison": controller_comparison(first),
        "controllers": controllers, "comparisons": {name: controller_comparison(c) for name, c in controllers.items()},
        "standard_deviations": {}, "model": model, "params": None, "gp_setup": None,
    }


def _loss_series(stats, *, test: bool):
    if test:
        steps = stats["step"]
        keys = {"total": "test_total", "position": "position", "attitude": "attitude",
                "linear velocity": "linear_velocity", "angular velocity": "angular_velocity"}
    else:
        stepped = "train_step" in stats and np.asarray(stats["train_step"]).size > 0
        steps = stats["train_step"] if stepped else stats["step"]
        keys = {"total": "train_loss" if stepped else "train_total",
                "position": "train_position" if stepped else "position",
                "attitude": "train_attitude" if stepped else "attitude",
                "linear velocity": "train_linear_velocity" if stepped else "linear_velocity",
                "angular velocity": "train_angular_velocity" if stepped else "angular_velocity"}
    return {name: (steps, stats[key]) for name, key in keys.items() if key in stats}


def _single_state_page(prediction_series, truth, time_axis, state_bands):
    """state_single_figure with the title the spec asks for: flight 0 of N, not 'single trajectory'."""
    figure = figures.state_single_figure(prediction_series, truth, time_axis, sample_bands=state_bands)
    band_text = "" if not state_bands else (
        f" — shaded: ±2σ over {state_bands[0][2].shape[0]} posterior weight-sample rollouts on this flight")
    figure.suptitle(f"State trajectory — flight 0 of {truth.shape[0]} held-out flights{band_text}",
                    fontsize=13, fontweight="bold")
    return figure


def generate(run_dirs, *, selected_steps=None, evaluation_dataset=None,
             controller_seconds=20.0, output_name=None, references=None, controller_only=False,
             dissipation="config", refly=False, include_ground_truth=False, subfolders=False,
             open_loop_datasets=(), open_loop_samples=OPEN_LOOP_SAMPLES, gust=None, damping_schedule=None) -> Path:
    selected_steps = list(selected_steps or [None] * len(run_dirs))
    references = tuple(references or ("diamond",))
    device = jax.devices()[0]
    first_config = load_config(sorted(Path(run_dirs[0]).glob("*.yaml"))[0])
    dataset = Path(evaluation_dataset) if evaluation_dataset is not None else Path(
        resolve_project_path(first_config["report"]["evaluation_dataset"]))
    data = evaluation.load_common_dataset(dataset)
    vehicle = evaluation.vehicle_constants(data["settings"])
    truth = evaluation.shared_truth(data, int(first_config["report"]["trajectory_count"]))
    time_axis = np.arange(truth.shape[1]) * evaluation.DT
    horizon = float(time_axis[-1])
    flight_name = dataset.stem

    models = [_prepare(d, s, truth, vehicle, device, controller_seconds, references, controller_only, dissipation, refly,
                       gust, damping_schedule)
              for d, s in zip(run_dirs, selected_steps)]
    if include_ground_truth:
        models.append(_prepare_ground_truth(
            truth, vehicle, device, controller_seconds, references, controller_only, dissipation, refly,
            int(first_config["report"]["random_seed"]), dataset, gust))
    learned = [m for m in models if not m.get("synthetic")]      # the trained models only: loss curves, GT rollout row
    for entry, (colour, style) in zip(models, figures.COMPARISON_STYLES):
        entry["colour"], entry["style"] = colour, style
    gt_label, gt_colour, gt_style = figures.GROUND_TRUTH_MODEL_STYLE
    grouped: dict[str, list[plt.Figure]] = {}
    descriptions: dict[str, str] = {}
    if controller_only:
        # Only the controller booklets are built; the rest of the report is skipped on purpose.
        gt_prediction = gt_errors = None
    else:
        synthetic = next((m for m in models if m.get("synthetic")), None)
        if synthetic is not None:            # identical operators and integrator: do not roll the same model out twice
            gt_prediction, gt_errors = synthetic["prediction"], synthetic["errors"]
        else:
            print("Rolling out the ground-truth-operator model...", flush=True)
            gt_benchmarks = evaluation.benchmark(evaluation.ground_truth_model(vehicle), truth, BENCHMARK_REPEATS, device)
            gt_prediction = gt_benchmarks.pop("prediction")
            gt_errors = evaluation.rollout_errors(truth, gt_prediction, vehicle)

    if not controller_only:
        styled = [(m["label"], m["colour"], m["style"]) for m in learned]
        styled_all = [(m["label"], m["colour"], m["style"]) for m in models]   # identification also draws the GT entry
        error_series = [(*s, m["errors"]) for s, m in zip(styled, learned)] + [(gt_label, gt_colour, gt_style, gt_errors)]
        prediction_series = [(*s, m["prediction"]) for s, m in zip(styled, learned)] + [(gt_label, gt_colour, gt_style, gt_prediction)]
        product_entries = [(*s, m["products"]) for s, m in zip(styled_all, models)]
        metrics_by_label = {m["label"]: m["product_metrics"] for m in models}
        bands = [(m["label"], m["colour"], m["samples"]) for m in models if m["samples"] is not None]
        state_bands = [(m["label"], m["colour"], m["state_samples"]) for m in models if m["state_samples"] is not None]
        targets = models[0]["targets"]

        identification: list[plt.Figure] = [
            v2.product_table([(m["label"], m["product_metrics"]) for m in models],
                             evaluation.PRODUCT_LABELS, horizon, flight_name),
        ]
        for key, title, truth_text, units, symbol in PRODUCT_PAGES:
            identification.append(v2.product_page(product_entries, targets, time_axis, key, title, truth_text, units,
                                                  symbol, metrics_by_label, bands,
                                                  flight_name=flight_name, horizon=horizon))
        # The two state-dependent products re-plotted against the state they depend on, which is the
        # way they should appear in the paper: time is the wrong axis for a damping law.
        identification.append(v2.damping_law_page(product_entries, targets, truth, bands,
                                                  flight_name=flight_name, horizon=horizon))
        # Closing page: how the number in the page-1 table is actually computed.
        identification.append(v2.metric_definition_page(metrics_by_label, models[0]["label"],
                                                        int(truth.shape[0])))

        common_note = ("Every curve is an MSE-type quantity computed by the same pose_loss_components for every model; "
                       "the negative log-likelihood is never plotted here. The totals are therefore comparable. What is "
                       "asymmetric is whose objective is drawn: the point-estimate baselines minimise exactly this total, "
                       "the GP minimises NLL + beta*KL + penalties, shown on the GP objective page.")
        losses: list[plt.Figure] = [
            v2.loss_page([(m["label"], m["colour"], _loss_series(m["run"]["stats"], test=False)) for m in learned],
                         "Training curves — every loss component on one axis",
                         "Final curriculum stage (K=100), 101-step moving average.  " + common_note,
                         test=False),
            v2.loss_page([(m["label"], m["colour"], _loss_series(m["run"]["stats"], test=True)) for m in learned],
                         "Test-window curves — every loss component on one axis",
                         "Final curriculum stage (K=100), recorded at the checkpoint interval and left unsmoothed.  "
                         + common_note
                         + "  Test observations are clean: the dataset noises the training split only.",
                         test=True),
        ]
        for entry in models:
            if entry["is_gp"]:
                losses.append(v2.gp_objective_page(entry["run"]["stats"], entry["label"], entry["noise"]))

        open_loop: list[plt.Figure] = [
            v2.rollout_summary_table([(m["label"], m["errors"]) for m in learned] + [(gt_label, gt_errors)],
                                     truth, horizon, flight_name),
            figures.state_ensemble_figure(prediction_series, truth, time_axis, sample_bands=state_bands),
            _single_state_page(prediction_series, truth, time_axis, state_bands),
            figures.geometry_figure(error_series, time_axis, "determinant",
                                    "SO(3) violation — |det R − 1|", "absolute determinant error"),
            figures.geometry_figure(error_series, time_axis, "orthogonality",
                                    "SO(3) violation — ‖RᵀR − I‖_F", "orthogonality Frobenius error"),
            v2.energy_page(prediction_series, truth, time_axis, vehicle),
            v2.state_error_page(error_series, time_axis),
        ]
        for entry in models:
            if entry["is_gp"]:
                open_loop.append(figures.gp_uncertainty_figure(entry["standard_deviations"]))

        grouped.update({"physics-identification": identification, "training-and-test-losses": losses,
                        "open-loop-rollout": open_loop})
        descriptions.update({name: description for name, description in BOOKLETS[:3]})

    folder = EVAL_ROOT / (output_name or (
        f"{datetime.now().astimezone().strftime('%d-%m-%H-%M')}_spec-comparison_"
        f"obs-noise{_tag(models[0]['noise'])}_{flight_name}"))
    folder.mkdir(parents=True, exist_ok=True)
    law_text = ("damping feedforward ON for every model" if dissipation == "on" else
                "reference law (no feedforward) for every model" if dissipation == "off" else "each run's own controller law")

    # ---- closed-loop/<reference>/ : one PDF per model for that shape, plus a comparison table ----
    closed_dir = folder / "closed-loop"
    closed_dir.mkdir(parents=True, exist_ok=True)
    closed_files: list[dict[str, Any]] = []
    slugs: list[str] = []
    for entry in models:                                  # one stable file name per model, shared by every shape folder
        slug = _model_slug(entry["label"])
        if slug in slugs:
            slug = f"{slug}_{len(slugs)}"
        slugs.append(slug)
    # Every learned flight is redrawn with the analytic-operator flight of the same shape underneath it, so the
    # ringing at the reference's instantaneous hold can be read as "the command is infeasible", not "the model is wrong".
    truth_entry = next((m for m in models if m.get("synthetic")), None)
    if truth_entry is not None:
        print("Redrawing the closed-loop plots against PH-GT...", flush=True)
        for entry in models:
            if entry is truth_entry:
                continue
            for reference in references:
                flight, baseline_flight = entry["controllers"][reference], truth_entry["controllers"][reference]
                payload = np.load(Path(flight["directory"]) / "controller_rollout.npz")
                baseline = np.load(Path(baseline_flight["directory"]) / "controller_rollout.npz")["s_traj"]
                # Two renderings of the same flight: against the commanded reference, and against the flight the
                # exact-physics controller flew. They answer different questions, so they get their own pages.
                flight["metadata"]["plots"] = save_plots(
                    Path(flight["directory"]), entry["flat"], payload["s_traj"], payload["s_plan"],
                    payload["s_plan_full"], reference)
                flight["metadata"]["plots_vs_ground_truth"] = save_plots(
                    Path(flight["directory"]), entry["flat"], payload["s_traj"], payload["s_plan"],
                    payload["s_plan_full"], reference, baseline=baseline, show_plan=False, suffix="_vs-gt")
                # How far this flight is from the flight the exact-physics controller flew, on the shared horizon.
                count = min(payload["s_traj"].shape[0], baseline.shape[0])
                gap = payload["s_traj"][:count, :3] - baseline[:count, :3]
                velocity_gap = payload["s_traj"][:count, 3:6] - baseline[:count, 3:6]
                comparison = entry["comparisons"][reference]
                comparison["position_rmse_vs_truth_m"] = float(np.sqrt(np.mean(np.sum(gap**2, axis=1))))
                comparison["velocity_rmse_vs_truth_m_per_s"] = float(np.sqrt(np.mean(np.sum(velocity_gap**2, axis=1))))
                truth_rmse = truth_entry["comparisons"][reference]["position_rmse_m"]
                comparison["position_gap_vs_truth_percent"] = float(
                    100.0 * (comparison["position_rmse_m"] - truth_rmse) / truth_rmse) if truth_rmse > 0 else float("nan")
                # Attitude against the flight the exact-physics controller flew: zero floor, chart-free.
                own, truth_rotation = payload["current_rotation"][:count], np.load(
                    Path(baseline_flight["directory"]) / "controller_rollout.npz")["current_rotation"][:count]
                trace = np.einsum("nij,nij->n", own, truth_rotation)
                angle = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
                comparison["attitude_rmse_vs_truth_deg"] = float(np.sqrt(np.mean(angle**2)))
                roll, pitch, _ = euler_angles(own)
                truth_roll, truth_pitch, _ = euler_angles(truth_rotation)
                comparison["roll_rmse_vs_truth_deg"] = float(np.degrees(np.sqrt(np.mean(wrap_to_pi(roll - truth_roll) ** 2))))
                comparison["pitch_rmse_vs_truth_deg"] = float(np.degrees(np.sqrt(np.mean(wrap_to_pi(pitch - truth_pitch) ** 2))))
                # Valid tracking time: how long this flight stays with the exact-physics flight. The threshold is
                # the VPT convention, sqrt(0.025) of the reference's own excursion, so it scales with the shape.
                plan = payload["s_plan"]
                excursion = float(np.sqrt(np.mean(np.sum((plan[:, :3] - plan[0, :3]) ** 2, axis=1))))
                threshold = VALID_TRACKING_FRACTION * excursion
                distance = np.linalg.norm(gap, axis=1)
                times = payload["s_traj"][:count, -1]
                comparison["valid_tracking_seconds_vs_truth"] = valid_tracking_seconds(times, distance, threshold)
                comparison["valid_tracking_threshold_m"] = threshold
                (Path(flight["directory"]) / "controller_metadata.json").write_text(
                    json.dumps(flight["metadata"], indent=2) + "\n")

    comparison_pages: list[plt.Figure] = []
    for reference in references:
        shape_dir = closed_dir / reference
        if subfolders:
            shape_dir.mkdir(parents=True, exist_ok=True)
        for slug, entry in (zip(slugs, models) if subfolders else ()):
            flight = entry["controllers"][reference]
            # Page 1 is the metrics-as-rows table for this model on this shape; then the two comparisons, each as
            # a tracking page and a 3-D trajectory page.
            pages = [v2.controller_table([(entry["label"], entry["comparisons"][reference])], reference,
                                         float(controller_seconds), [flight["metadata"].get("dissipation_mode")])]
            for against, plots in (("PH-GT", flight["metadata"].get("plots_vs_ground_truth")),
                                   ("reference", flight["metadata"]["plots"])):
                if not plots:
                    continue
                for plot_key, plot_name in (("labeled_tracking_plot", "controller tracking"),
                                            ("labeled_trajectory_plot", "controller trajectory")):
                    pages.append(figures.image_page(
                        Path(plots[plot_key]),
                        f"{entry['flat']} — {plot_name}: learned vs {against} — reference: {reference}"))
            target = shape_dir / f"{slug}.pdf"
            with PdfPages(target) as pdf:
                for figure in pages:
                    pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)
            closed_files.append({"file": f"closed-loop/{reference}/{target.name}", "pages": len(pages),
                                 "model": entry["flat"], "reference": reference,
                                 "contents": f"{entry['flat']} on '{reference}': metrics table, then tracking and 3-D "
                                             f"trajectory pages against PH-GT and against the reference"})
        # One image per shape: every model's 3-D flight against the reference, in its own panel.
        grid_entries = []
        for entry in models:
            payload = np.load(Path(entry["controllers"][reference]["directory"]) / "controller_rollout.npz")
            grid_entries.append((_short_label(entry["label"]), payload["s_traj"], payload["s_plan_full"]))
        grid = v2.trajectory_grid(grid_entries, reference, float(controller_seconds), law_text)
        images_dir = closed_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)   # every shape's grid in one place, named by the shape
        grid.savefig(images_dir / f"{reference}_trajectory.png", dpi=150, bbox_inches="tight")
        plt.close(grid)
        closed_files.append({"file": f"closed-loop/images/{reference}_trajectory.png", "pages": 1, "reference": reference,
                             "contents": f"'{reference}': one 3-D panel per model, learned vs reference"})
        tracking = v2.tracking_grid([(label, traj, np.load(Path(entry["controllers"][reference]["directory"])
                                                           / "controller_rollout.npz")["s_plan"])
                                     for (label, traj, _), entry in zip(grid_entries, models)],
                                    reference, float(controller_seconds), law_text)
        tracking.savefig(images_dir / f"{reference}_tracking.png", dpi=140, bbox_inches="tight")
        plt.close(tracking)
        closed_files.append({"file": f"closed-loop/images/{reference}_tracking.png", "pages": 1, "reference": reference,
                             "contents": f"'{reference}': one tracking row per model, the eight states as columns"})

        # the same table figure serves this shape's folder and the top-level booklet, so it is built once
        table = v2.controller_table([(m["label"], m["comparisons"][reference]) for m in models], reference,
                                    float(controller_seconds),
                                    [m["controllers"][reference]["metadata"]["dissipation_mode"] for m in models],
                                    show_truth_rows=False)
        if subfolders:
            with PdfPages(shape_dir / "comparison.pdf") as pdf:
                pdf.savefig(table, bbox_inches="tight")
            closed_files.append({"file": f"closed-loop/{reference}/comparison.pdf", "pages": 1, "reference": reference,
                                 "contents": f"'{reference}': one failure-aware table, every model as a column"})
            print(f"PDF: {shape_dir}  ({len(models)} model PDFs + comparison.pdf)", flush=True)
        comparison_pages.append(table)
    target = closed_dir / "comparison.pdf"
    with PdfPages(target) as pdf:
        for figure in comparison_pages:
            pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)
    closed_files.append({"file": "closed-loop/comparison.pdf", "pages": len(comparison_pages),
                         "contents": "one failure-aware comparison table per reference, every model as a column"})
    print(f"PDF: {target}  ({len(comparison_pages)} pages)", flush=True)

    # ---- open-loop/ : one table page per evaluation set, plus the state grid, error plot and GP calibration ----
    open_files: list[dict[str, Any]] = []
    if not controller_only:
        open_dir = folder / "open-loop"
        open_images = open_dir / "images"
        open_images.mkdir(parents=True, exist_ok=True)
        table_pages: list[plt.Figure] = []
        open_loop_summary: dict[str, Any] = {}
        for set_path in [dataset, *map(Path, open_loop_datasets)]:
            flights = openloop.load_open_loop_set(set_path)
            set_truth, step, set_name = flights["truth"], flights["dt"], flights["name"]
            print(f"Open loop on {set_name}: {set_truth.shape[0]} flights, dt = {step:.4f} s", flush=True)
            table_entries, grid_entries, error_entries, summary = [], [], [], {}
            for entry in models:
                model = (evaluation.ground_truth_model(flights["vehicle"]) if entry.get("synthetic") else entry["model"])
                prediction, compute_ms = openloop.timed_rollout(model, set_truth, step)
                samples = None
                if entry["is_gp"] and prediction is not None:
                    print(f"  {entry['flat']}: {open_loop_samples} posterior-sample rollouts...", flush=True)
                    samples = openloop.posterior_rollouts(entry["params"], entry["gp_setup"], set_truth,
                                                           open_loop_samples, int(entry["config"]["report"]["random_seed"]),
                                                           step)
                metrics = openloop.open_loop_metrics(set_truth, prediction, samples, step, compute_ms)
                open_label = (entry["label"].replace("PH-GT ", "PH-GT-LieIMEX ", 1) if entry.get("synthetic")
                              else entry["label"])          # in open loop the analytic operators ARE integrated
                table_entries.append((open_label, metrics))
                grid_entries.append((_short_label(open_label), prediction, samples))
                error_entries.append((_short_label(open_label), prediction))
                summary[entry["flat"]] = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
                if samples is not None:
                    figure = openloop.calibration_figure(_short_label(entry["label"]), set_truth, prediction, samples,
                                                          step, set_name, metrics)
                    target = open_images / f"{set_name}_calibration_{_model_slug(entry['label'])}.png"
                    figure.savefig(target, dpi=140, bbox_inches="tight"); plt.close(figure)
                    open_files.append({"file": f"open-loop/images/{target.name}", "pages": 1,
                                       "contents": f"{entry['flat']} posterior calibration on {set_name}"})
            horizon_seconds = (set_truth.shape[1] - 1) * step
            table_pages.append(openloop.open_loop_table(table_entries, set_name, horizon_seconds,
                                                         set_truth.shape[0], open_loop_samples))
            for flight in range(set_truth.shape[0]):
                grid = openloop.states_grid(grid_entries, set_truth, step, set_name, flight=flight)
                grid.savefig(open_images / f"{set_name}_states_flight{flight:02d}.png", dpi=140, bbox_inches="tight")
                plt.close(grid)
                open_files.append({"file": f"open-loop/images/{set_name}_states_flight{flight:02d}.png", "pages": 1,
                                   "contents": f"{set_name} flight {flight}: one row per model, eight state columns, GP ±2σ band"})
                paths = openloop.trajectory_grid(grid_entries, set_truth, step, set_name, flight=flight)
                paths.savefig(open_images / f"{set_name}_trajectory_flight{flight:02d}.png", dpi=150, bbox_inches="tight")
                plt.close(paths)
                open_files.append({"file": f"open-loop/images/{set_name}_trajectory_flight{flight:02d}.png", "pages": 1,
                                   "contents": f"{set_name} flight {flight}: one 3-D panel per model, prediction vs truth"})
            error_plot = openloop.error_figure(error_entries, set_truth, step, set_name,
                                                [(m["colour"], m["style"]) for m in models])
            error_plot.savefig(open_images / f"{set_name}_error.png", dpi=140, bbox_inches="tight"); plt.close(error_plot)
            open_files.append({"file": f"open-loop/images/{set_name}_error.png", "pages": 1,
                               "contents": f"{set_name}: position error against time, every model"})
            open_loop_summary[set_name] = {"dt": step, "flights": int(set_truth.shape[0]), "source": flights["source"],
                                           "metrics": summary}
            print(f"images: {open_images}/{set_name}_*.png", flush=True)
        target = open_dir / "comparison.pdf"
        with PdfPages(target) as pdf:
            for figure in table_pages:
                pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)
        open_files.append({"file": "open-loop/comparison.pdf", "pages": len(table_pages),
                           "contents": "one open-loop metrics table per evaluation set; bold = best learned model"})
        (open_dir / "open_loop_metrics.json").write_text(json.dumps(open_loop_summary, indent=2) + "\n")
        print(f"PDF: {target}  ({len(table_pages)} pages)", flush=True)

    booklets: list[dict[str, Any]] = []
    for name, description in descriptions.items():
        target = folder / f"{name}.pdf"
        with PdfPages(target) as pdf:
            for figure in grouped[name]:
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
        booklets.append({"file": target.name, "pages": len(grouped[name]), "contents": description})
        print(f"PDF: {target}  ({len(grouped[name])} pages)", flush=True)

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": Path(__file__).name,
        "specification": "src/models/SE3_Quadrotor/comparision/comparision_report_format.md",
        "evaluation_dataset": str(dataset),
        "plant_gust": gust, "controller_damping_schedule": damping_schedule,
        "held_out_flights": int(truth.shape[0]),
        "horizon_seconds": horizon,
        "controller_seconds": float(controller_seconds),
        "controller_references": list(references),
        "controller_dissipation": dissipation,
        "controller_only": bool(controller_only),
        "posterior_samples": POSTERIOR_SAMPLES,
        "benchmark_repeats": BENCHMARK_REPEATS,
        "models": [{
            "label": m["flat"],
            "run": str(m["run"]["directory"]),
            "model": m["config"]["model"]["name"],
            "observation_noise": m["noise"],
            "selected_step": int(m["run"]["selected_step"]),
            "total_steps": int(m["run"]["total_steps"]),
        } for m in models],
        "booklets": booklets + closed_files + open_files,
        "closed_loop_layout": ("closed-loop/images/<reference>_trajectory.png and _tracking.png plus "
                               "closed-loop/comparison.pdf" + (" and closed-loop/<reference>/<model>.pdf "
                               "(table + tracking + trajectory) with a per-shape comparison.pdf" if subfolders else "")),
    }
    (folder / "report_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Eval run: {folder}", flush=True)
    print(f"Pages: {sum(len(pages) for pages in grouped.values())}", flush=True)
    return folder


def main() -> None:
    args = parse_args()
    for spec in args.controller_recorded_reference:
        registered = register_recorded_references(spec)
        print(f"Recorded references from {spec}: {registered[0]} ... {registered[-1]}", flush=True)
        args.controller_reference = list(args.controller_reference or ["diamond"]) + registered
    generate(args.run, selected_steps=args.selected_step, evaluation_dataset=args.evaluation_dataset,
             controller_seconds=args.controller_seconds, output_name=args.output_name,
             references=args.controller_reference, controller_only=args.controller_only,
             dissipation=args.controller_dissipation, refly=args.controller_refly,
             include_ground_truth=args.include_ground_truth, subfolders=args.closed_loop_subfolders,
             open_loop_datasets=args.open_loop_dataset, open_loop_samples=args.open_loop_samples,
             gust=(None if args.gust_sigma_fraction <= 0 else {"time_constant_seconds": args.gust_tau,
                                                                 "sigma_fraction_of_weight": args.gust_sigma_fraction,
                                                                 "seed": args.gust_seed}),
             damping_schedule=(None if args.controller_damping_schedule == "none" else
                               {"kind": "process-noise", "sigma_reference": tuple(args.damping_sigma_reference)}))


if __name__ == "__main__":
    sys.exit(main())

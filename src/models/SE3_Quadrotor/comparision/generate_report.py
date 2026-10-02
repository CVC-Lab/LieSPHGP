"""54-page single-model report for a completed PH-GP-LieIMEX JAX run.

Page map:
  1 summary table, 2 subnetwork MSE/NMSE table, 3 gauge-invariant products table,
  4-9 product trajectories (thrust gain, torque gain, gravity, mu*Dv, M2^-1*Dw, damping vs speed),
  10-14 train losses, 15-19 test losses, 20-23 error versus ground truth, 24-25 physical energy,
  26-27 SO(3) violation, 28 state ensemble, 29 single trajectory, 30 phase portraits,
  31-42 subnetwork trajectories raw / gauge-fixed (M1, M2, Dv, Dw, g, V),
  43 fairness and computation, 44 compute bars, 45 controller table,
  46 GP posterior uncertainty, 47-48 controller plots, 49-54 audit tables.

Everything is computed in JAX / NumPy from the run folder, the common clean
240 Hz PID benchmark dataset, and a closed-loop PyBullet controller run.
Only one PDF and one JSON sidecar are written per run.
"""

from __future__ import annotations

import argparse
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from ..ph_gp_lie_imex.config import load_config, resolve_project_path
from ..ph_gp_lie_imex.experiment import append_history, save_json, utc_now
from . import report_evaluation as evaluation
from . import report_figures as figures_module
from .report_controller import controller_comparison, run_controller

PROJECT_ROOT = Path(__file__).resolve().parents[4]
COMMON_EVALUATION_DATASET = PROJECT_ROOT / (
    "datasets/data/pybullet_quadrotor/"
    "D0_CF2P_PID_contact-free_nonlinear-damping-c0p5_seed0_260905.pkl"
)
REPORT_ROOT = PROJECT_ROOT / "reports/SE3_Quadrotor"
EXPECTED_PAGES = 54
REFERENCE_SECTION_PAGES = 42  # pages 1-42: summary, tables, products, curves, errors, states, subnetworks
BENCHMARK_REPEATS = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="training config (the copy inside the run folder)")
    parser.add_argument("--run", type=Path, default=None, help="run folder; defaults to the config's parent")
    parser.add_argument("--selected-step", type=int, default=None)
    parser.add_argument("--evaluation-dataset", type=Path, default=None,
                        help="defaults to report.evaluation_dataset from the config, else the common 0.5 s flight")
    parser.add_argument("--controller-seconds", type=float, default=20.0)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    return parser.parse_args()


def _is_gp(config: dict[str, Any]) -> bool:
    return bool(config.get("gp", {}).get("enabled", False))


def _network_family(config: dict[str, Any]) -> str:
    """Point-estimate baselines are named by their integrator: PH-NODE is the published RK4 model."""
    return "PH-NODE-RK4" if config["model"]["name"] == "ph_node" else "PH-NN-LieIMEX"


def _priors_tag(config: dict[str, Any]) -> str:
    """', priors on' when the gravity or actuation penalty was active in training; the columns must be tellable apart."""
    gp = config.get("gp") or {}
    wg, wa = float(gp.get("gravity_penalty_weight") or 0.0), float(gp.get("actuation_penalty_weight") or 0.0)
    return f", priors {wg:g}/{wa:g}" if (wg > 0.0 or wa > 0.0) else ""


def _model_label(config: dict[str, Any], noise: float) -> str:
    if not _is_gp(config):
        width = config["model"].get("hidden_dim")
        return f"{_network_family(config)} MLP width {width}\nMSE, obs noise {noise:g}{_priors_tag(config)}"
    fit = "NLL" if config["gp"]["data_fit"] == "se3-nll" else "MSE"
    return f"PH-GP-LieIMEX levels\n{fit}, obs noise {noise:g}{_priors_tag(config)}"


def _noise_level(config: dict[str, Any]) -> float:
    """Training observation noise: the loader override, else the level recorded in the dataset pickle."""
    override = config["data"]["observation_noise_override"]
    if override is not None:
        return float(override)
    return _dataset_noise(config)["level"]


def _dataset_noise(config: dict[str, Any]) -> dict[str, Any]:
    """Noise metadata stored by the dataset generator (pre-noised pickles), or a clean record."""
    dataset_path = resolve_project_path(config["data"]["dataset_path"])
    settings: Any = {}
    if dataset_path is not None and Path(dataset_path).is_file():
        with Path(dataset_path).open("rb") as handle:
            settings = pickle.load(handle).get("settings", {})
    record = settings.get("observation_noise") if isinstance(settings, dict) else None
    if isinstance(record, dict) and record.get("enabled") and record.get("level") is not None:
        return {"level": float(record["level"]), "source": "dataset", "record": record}
    return {"level": 0.0, "source": "none", "record": None}


def _noise_text(config: dict[str, Any], noise: float) -> str:
    if noise == 0.0:
        return "0 (clean)"
    if config["data"]["observation_noise_override"] is not None:
        return f"{noise:g} absolute; rotations re-projected onto SO(3)"
    return f"{noise:g} absolute, pre-noised in the dataset pickle; rotation noise R*Exp(eta)"


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _audit_pages(run: dict[str, Any], controller: dict[str, Any], noise: float, horizon_states: int) -> list[plt.Figure]:
    config, metadata = run["config"], run["metadata"]
    gp, training, optimizer, data = config["gp"], config["training"], config["optimizer"], config["data"]
    is_gp = _is_gp(config)
    nll = bool(is_gp) and gp["data_fit"] == "se3-nll"
    def _gp_text(key: str) -> str:
        value = gp.get(key)
        return "not applicable" if value is None else f"{float(value):g}"
    metrics = controller["metadata"]["metrics"]
    levels = run["levels"]
    sigma = levels.get("likelihood_sigma", {})
    window_points = int(data["window_points"])
    step = float(metadata["step_size_seconds"])
    table = figures_module.table_page
    column = ["PH-GP-LieIMEX" if is_gp else f"{_network_family(config)} (width {config['model'].get('hidden_dim')})"]
    return [
        table(
            "Data and optimization audit",
            ["Training observation noise", "Clean test observations", "Data-fit term", "NLL enabled",
             "Learned sigma (x, R, v, omega)", "Maximum KL beta", "KL normalizer", "Initial log sigma_w",
             "Learning rate", "Gradient clipping", "Training precision"],
            column,
            [[_noise_text(config, noise)],
             ["yes"],
             ["SE(3) negative log-likelihood, learned per-block sigma" if nll else "unscaled SE(3) trajectory MSE"],
             ["yes" if nll else "no"],
             [", ".join(f"{sigma.get(key, float('nan')):.3g}" for key in ("position", "attitude", "linear_velocity", "angular_velocity")) if sigma else "not applicable"],
             [_gp_text("kl_beta_max")],
             ["all training transitions (minibatch ELBO)" if is_gp else "not applicable (point-estimate network)"],
             [_gp_text("initial_log_standard_deviation")],
             [f"{float(optimizer['learning_rate']):g}"],
             ["off" if optimizer["gradient_clip_norm"] is None else f"{optimizer['gradient_clip_norm']}"],
             [metadata.get("dtype", "jax.float32")]],
            "Noise, when nonzero, is added to the 18 training state channels before training; controls and the "
            "clean test split are unchanged. The deterministic report uses the GP posterior mean.",
        ),
        table(
            "PH-GP-LieIMEX learned-level architecture",
            ["M1 inverse", "M2 inverse", "Potential V", "Dissipation Dv, Domega", "Control map g",
             "Learned levels", "Initial operator values", "Dataset physics read by the model"],
            column,
            [["exp(lambda0 + GP(x)) I3, strictly positive" if is_gp else "PSD head on an MLP(x), strictly positive"],
             ["L L^T, L from six learned levels plus GP(R)" if is_gp else "L L^T, L from an MLP(R)"],
             [("lambda_V^T x + GP(x); no mgz mean; no rotation input" if any(levels.get("V_level", [0, 0, 0])) else "direct GP V(x); no mgz mean; no rotation input")
              if is_gp else "MLP(x, R); no mgz mean"],
             ["L L^T, L from six learned levels plus GP(v_b) / GP(omega_b)" if is_gp else "L L^T, L from MLP(v_b) / MLP(omega_b)"],
             ["Lambda0 (6x4 learned) + reshape(GP(x,R)); no selection matrix" if is_gp else "reshape(MLP(x, R)); no selection matrix"],
             [f"lambda0={levels['M1_log_level']:.3g}; M2 diag {np.round(np.exp(2*np.asarray(levels['M2_level'][:3])),1).tolist()}; g[2,0]={levels['g_level'][2][0]:.3g}; lambda_V={np.round(np.asarray(levels.get('V_level', [0,0,0])),3).tolist()}"
              if is_gp else "none: the MLPs have no explicit constant levels"],
             [f"M1^-1=I, M2^-1={gp['inverse_mass_2_initial_value']:g} I, Dv={gp['dissipation_v_initial_value']:g} I, Dw={gp['dissipation_w_initial_value']:g} I"
              if is_gp else f"MLP initialisation gain {config['model'].get('initialization_gain')}, activation {config['model'].get('activation')}"],
             ["none (mass, inertia, gravity and damping appear only on the report's ground-truth pages)"]],
            "The analytical dotted targets are I/m, J^-1, mc(1+|v|)I, c(1+|omega|)J, m*g*z and the selection matrix; "
            "because every level is free, the data fix only the gauge-invariant products, which the gauge-fixed pages show.",
        ),
        table(
            "Training-window and control audit",
            ["Integrator step h (s)", "State points", "Transitions K", "Window duration (s)", "Window stride",
             "Overlapping transitions", "Control indexing", "Training windows", "Clean test windows"],
            column,
            [[f"{step:g}"], [str(window_points)], [str(window_points - 1)], [f"{step * (window_points - 1):.2f}"],
             [f"{int(data['window_stride'])} source steps"],
             [str(max(0, window_points - 1 - int(data["window_stride"])))],
             ["recorded u[k+1] drives state k to k+1" if data["time_varying_controls"] else "fixed u inside the window"],
             [str(metadata["train_shape"][1])], [str(metadata["test_shape"][1])]],
            "Adjacent windows share only their boundary state; no transition is reused.",
        ),
        table(
            "Checkpoint-selection audit",
            ["Requested report checkpoint", "Final clean test MSE", "Best validation checkpoint", "Best clean test MSE", "Main plots use"],
            column,
            [[f"step {run['selected_step']}"], [f"{run['selected_test']['total']:.8e}"],
             [f"step {run['best_test']['step']}"], [f"{run['best_test']['clean_test_mse']:.8e}"],
             [f"step {run['selected_step']} (best shown separately)"]],
            "No favorable checkpoint is silently substituted for the requested final model.",
        ),
        table(
            "Open-loop report-inference audit",
            ["Evaluation step h (s)", "State points", "Integration transitions", "Horizon (s)", "Initial state",
             "Controls", "State propagation", "GP weights"],
            column,
            [[f"1/240 = {evaluation.DT:.8f}"], [str(horizon_states)], [str(horizon_states - 1)],
             [f"{(horizon_states - 1) * evaluation.DT:.8f}"],
             ["one clean held-out state, repeated 10 times"], ["identical recorded PID wrench u(t)"],
             ["fully autoregressive Lie-IMEX; no state reset"], ["deterministic posterior mean"]],
            f"The h={step:g} setting is used for training. Common report inference uses the clean 240 Hz benchmark grid.",
        ),
        table(
            "Closed-loop PyBullet controller audit",
            ["Duration (s)", "Physics/controller rate", "Transitions / stored states", "Plant integrator", "Model role",
             "Position RMSE (m)", "Motor saturation fraction", "Minimum altitude (m)"],
            column,
            [[f"{controller['metadata']['duration_seconds']:g}"], [f"{controller['metadata']['physics_hz']} Hz"],
             [f"{round(controller['metadata']['duration_seconds'] * controller['metadata']['physics_hz'])} / {controller['metadata']['full_reference_samples']}"],
             ["PyBullet Physics.PYB"], ["subnetworks compute control; PyBullet advances state"],
             [f"{metrics['position_rmse_m']:.6g}"], [f"{metrics['motor_saturation_fraction']:.3%}"],
             [f"{metrics['minimum_altitude_m']:.6g}"]],
            "Open-loop loss and closed-loop control answer different questions; low short-window MSE does not guarantee stable 20 s control.",
        ),
    ]


def generate_report(
    config: dict[str, Any],
    run_dir: Path,
    *,
    selected_step: int | None = None,
    evaluation_dataset: Path = COMMON_EVALUATION_DATASET,
    controller_seconds: float = 20.0,
    benchmark_repeats: int = BENCHMARK_REPEATS,
) -> Path:
    run = evaluation.load_run(run_dir, selected_step)
    noise = _noise_level(run["config"])
    label = _model_label(run["config"], noise)
    flat_label = label.replace("\n", " ")
    selected = run["selected_step"]
    device = jax.devices()[0]
    device_text = f"{device.platform}:{device.id}"

    # Model, data, open-loop evaluation -------------------------------------
    model, params, _gp_setup = evaluation.build_model(run)
    run["levels"] = evaluation.learned_levels(params)
    data = evaluation.load_common_dataset(evaluation_dataset)
    settings = data["settings"]
    vehicle = evaluation.vehicle_constants(settings)
    truth = evaluation.shared_truth(data, int(config["report"]["trajectory_count"]))
    time_axis = np.arange(truth.shape[1]) * evaluation.DT
    horizon = float(time_axis[-1])
    print(f"Benchmarking {flat_label} on {device_text}...", flush=True)
    benchmarks = evaluation.benchmark(model, truth, benchmark_repeats, device)
    prediction = benchmarks.pop("prediction")
    errors = evaluation.rollout_errors(truth, prediction, vehicle)
    # Report-only upper bound: the simulator's analytic operators in the same integrator with the same u(t).
    gt_label, gt_color, gt_linestyle = figures_module.GROUND_TRUTH_MODEL_STYLE
    print("Benchmarking the ground-truth-operator Lie-IMEX model...", flush=True)
    gt_benchmarks = evaluation.benchmark(evaluation.ground_truth_model(vehicle), truth, benchmark_repeats, device)
    gt_prediction = gt_benchmarks.pop("prediction")
    gt_errors = evaluation.rollout_errors(truth, gt_prediction, vehicle)
    learned_style = (label, figures_module.MODEL_COLOR, figures_module.MODEL_LINESTYLE)
    # Epistemic bands: independent posterior weight samples rolled out on trajectory 0 (report.posterior_samples, 0 = off).
    posterior_sample_count = int(config["report"].get("posterior_samples", 10) or 0) if _is_gp(config) else 0
    sample_bands, posterior_sample_summary = [], {"count": 0}
    if posterior_sample_count > 0:
        print(f"Rolling out {posterior_sample_count} posterior weight samples...", flush=True)
        samples, finite_count = evaluation.posterior_sample_rollouts(
            params, _gp_setup, truth, posterior_sample_count, int(config["report"]["random_seed"])
        )
        sample_bands = [(label, figures_module.MODEL_COLOR, samples)]
        position_spread = 2.0 * samples[..., :3].std(0)  # (time, 3): ±2σ half-width per axis
        posterior_sample_summary = {
            "count": posterior_sample_count, "finite": int(finite_count), "seed": int(config["report"]["random_seed"]),
            "band": "mean ± 2 std across weight-sample rollouts, trajectory 0, same x0 and u(t)",
            "position_2sigma_halfwidth_m_final": [float(v) for v in position_spread[-1]],
            "position_2sigma_halfwidth_m_time_mean": [float(v) for v in position_spread.mean(0)],
            "attitude_2sigma_halfwidth_rad_final": [float(v) for v in 2.0 * figures_module.euler_angles(samples)[:, -1].std(0)],
        }
    error_series = [(*learned_style, errors), (gt_label, gt_color, gt_linestyle, gt_errors)]
    prediction_series = [(*learned_style, prediction), (gt_label, gt_color, gt_linestyle, gt_prediction)]
    benchmark_series = [(*learned_style, benchmarks), (gt_label, gt_color, gt_linestyle, gt_benchmarks)]
    subnets = evaluation.query_subnetworks(model, truth)
    targets = evaluation.analytic_targets(truth, vehicle)
    aligned, fits, subnet_metrics = evaluation.align_subnetworks(subnets, targets)
    products = evaluation.gauge_invariant_products(model, truth)
    product_targets = evaluation.product_targets(truth, vehicle)
    product_metrics = evaluation.product_metrics(products, product_targets, truth)
    standard_deviations = evaluation.posterior_standard_deviations(params)

    # Closed-loop controller --------------------------------------------------
    print("Running the closed-loop PyBullet controller...", flush=True)
    controller = run_controller(
        run, model, run_dir / ("controller" if selected == run["total_steps"] else f"controller_step{selected:05d}"),
        model_label=flat_label,
        training_dataset=resolve_project_path(run["config"]["data"]["dataset_path"]),
        vehicle=vehicle, duration_seconds=controller_seconds, seed=int(config["report"]["random_seed"]),
        device_text=device_text,
        # report.controller_use_dissipation: true/false selects the learned damping feedforward; absent = module default.
        use_dissipation=config["report"].get("controller_use_dissipation"),
    )
    comparison = controller_comparison(controller)

    # Figures ----------------------------------------------------------------
    stats = run["stats"]
    step_text = figures_module.checkpoint_step_text(selected)
    summary_title = (
        f"Contact-free {vehicle.get('damping_law', 'nonlinear')}-damping Gym-PyBullet CF2P PID — {step_text}\n"
        f"Training observation noise={noise}; clean evaluation\n"
        f"PH-GP-LieIMEX learned levels, {'SE(3) NLL' if run['config']['gp']['data_fit'] == 'se3-nll' else 'MSE'}, "
        "position-only V(x), level+GP g(x,R); single trained model; no other learned models included"
    )
    pages: list[plt.Figure] = [
        figures_module.summary_figure(error_series, horizon, summary_title, truth=truth),
        figures_module.subnetwork_summary(label, subnet_metrics, horizon),
        figures_module.product_summary(label, product_metrics, evaluation.PRODUCT_LABELS, horizon),
        figures_module.product_trajectory_figure(label, products, product_targets, time_axis, "thrust_gain", "Thrust gain μ·g_f (force column of g scaled by the inverse mass)", "m/s² per N"),
        figures_module.product_trajectory_figure(label, products, product_targets, time_axis, "torque_gain", "Torque gain M2⁻¹·g_τ (truth J⁻¹)", "rad/s² per N·m"),
        figures_module.product_trajectory_figure(label, products, product_targets, time_axis, "gravity", "Gravity μ·∇V (truth g·e_z)", "m/s²"),
        figures_module.product_trajectory_figure(label, products, product_targets, time_axis, "damping_v", "Translational damping μ·Dv (truth c(1+|v|)·I)", "1/s"),
        figures_module.product_trajectory_figure(label, products, product_targets, time_axis, "damping_w", "Rotational damping M2⁻¹·Dω (truth c(1+|ω|)·I)", "1/s"),
        figures_module.damping_speed_figure(label, products, product_targets, truth, vehicle),
    ]
    has_step_stats = "train_step" in stats and stats["train_step"].size > 0
    train_steps = stats["train_step"] if has_step_stats else stats["step"]
    train_source = {
        "loss": stats["train_loss"] if has_step_stats else stats["train_total"],
        "position": stats["train_position"] if has_step_stats else stats["position"],
        "attitude": stats["train_attitude"] if has_step_stats else stats["attitude"],
        "linear_velocity": stats["train_linear_velocity"] if has_step_stats else stats["linear_velocity"],
        "angular_velocity": stats["train_angular_velocity"] if has_step_stats else stats["angular_velocity"],
    }
    for key, title, ylabel in (
        ("loss", "Train total loss", "loss"),
        ("position", "Train position loss", "position MSE"),
        ("attitude", "Train attitude geodesic squared", "geodesic squared"),
        ("linear_velocity", "Train linear-velocity loss", "velocity MSE"),
        ("angular_velocity", "Train angular-velocity loss", "angular-velocity MSE"),
    ):
        pages.append(figures_module.training_figure(label, train_steps, train_source[key], title, ylabel, test=False))
    test_source = {
        "loss": stats["test_total"], "position": stats["position"], "attitude": stats["attitude"],
        "linear_velocity": stats["linear_velocity"], "angular_velocity": stats["angular_velocity"],
    }
    for key, title, ylabel in (
        ("loss", "Test-window total loss", "loss"),
        ("position", "Test-window position loss", "position MSE"),
        ("attitude", "Test-window attitude geodesic squared", "geodesic squared"),
        ("linear_velocity", "Test-window linear-velocity loss", "velocity MSE"),
        ("angular_velocity", "Test-window angular-velocity loss", "angular-velocity MSE"),
    ):
        pages.append(figures_module.training_figure(label, stats["step"], test_source[key], title, ylabel, test=True))
    for key, title, ylabel in (
        ("position", "Position error vs GT", "squared position error"),
        ("attitude", "Attitude geodesic error vs GT", "squared geodesic error"),
        ("velocity", "Linear-velocity error vs GT", "squared velocity error"),
        ("omega", "Angular-velocity error vs GT", "squared angular-velocity error"),
    ):
        pages.append(figures_module.error_figure(error_series, time_axis, key, title, ylabel))
    energy_ensemble, energy_single = figures_module.energy_figures(prediction_series, truth, time_axis, vehicle)
    note = "PH-GP D-on matrices are learned levels plus GP residuals; dotted is analytic GT"
    pages.extend([
        energy_ensemble,
        energy_single,
        figures_module.geometry_figure(error_series, time_axis, "determinant", "SO(3) violation — determinant", "absolute determinant error"),
        figures_module.geometry_figure(error_series, time_axis, "orthogonality", "SO(3) violation — orthogonality", "orthogonality Frobenius error"),
        figures_module.state_ensemble_figure(prediction_series, truth, time_axis, sample_bands=sample_bands),
        figures_module.state_single_figure(prediction_series, truth, time_axis, sample_bands=sample_bands),
        figures_module.phase_figure(prediction_series, truth),
        figures_module.matrix_trajectory_figure(label, subnets, targets, fits, time_axis, "m1", "Translational inverse mass M1^-1(x)", gauge_fixed=False),
        figures_module.matrix_trajectory_figure(label, aligned, targets, fits, time_axis, "m1", "Translational inverse mass M1^-1(x)", gauge_fixed=True),
        figures_module.matrix_trajectory_figure(label, subnets, targets, fits, time_axis, "m2", "Rotational inverse inertia M2^-1(R)", gauge_fixed=False),
        figures_module.matrix_trajectory_figure(label, aligned, targets, fits, time_axis, "m2", "Rotational inverse inertia M2^-1(R)", gauge_fixed=True),
        figures_module.matrix_trajectory_figure(label, subnets, targets, fits, time_axis, "dv", "Translational dissipation Dv(v_body)", gauge_fixed=False, note=note),
        figures_module.matrix_trajectory_figure(label, aligned, targets, fits, time_axis, "dv", "Translational dissipation Dv(v_body)", gauge_fixed=True, note=note),
        figures_module.matrix_trajectory_figure(label, subnets, targets, fits, time_axis, "dw", "Rotational dissipation Domega(omega_body)", gauge_fixed=False, note=note),
        figures_module.matrix_trajectory_figure(label, aligned, targets, fits, time_axis, "dw", "Rotational dissipation Domega(omega_body)", gauge_fixed=True, note=note),
        figures_module.control_trajectory_figure(label, subnets, targets, fits, time_axis, gauge_fixed=False),
        figures_module.control_trajectory_figure(label, aligned, targets, fits, time_axis, gauge_fixed=True),
        figures_module.potential_trajectory_figure(label, subnets, targets, fits, time_axis, gauge_fixed=False),
        figures_module.potential_trajectory_figure(label, aligned, targets, fits, time_axis, gauge_fixed=True),
    ])
    if len(pages) != REFERENCE_SECTION_PAGES:
        raise AssertionError(f"Reference section has {len(pages)} pages, expected {REFERENCE_SECTION_PAGES}")

    metadata = run["metadata"]
    compute_note = (
        "All declared single-model audit checks passed. Single-model report: no cross-model fairness or solver "
        "comparison is claimed. Open-loop evaluation uses the same clean common-evaluation protocol as the reference report."
    )
    pages.extend([
        figures_module.table_page(
            f"{step_text.capitalize()} fairness and computation",
            ["Common-network init hash", "Full init hash", "Training observation noise", "Parameters", "Training steps",
             "Training wall time (min)", "Rollout median wall time (s)", "Transitions per wall second",
             "Full NFE per transition", "Auxiliary D queries per transition",
             "Peak device memory (MiB)" if device.platform == "gpu" else "Peak device memory (MiB; CPU not tracked)"],
            [label, gt_label],
            [["not applicable", "not applicable"], ["not applicable", "not applicable"],
             [figures_module.format_cell(noise if noise > 0 else None), "none (analytic operators)"],
             [f"{metadata['parameter_count']:,}", "0 (analytic)"], [str(run["total_steps"]), "0"],
             [f"{metadata['elapsed_seconds'] / 60:.3f}", "0"],
             [f"{benchmarks['median_seconds']:.5f}", f"{gt_benchmarks['median_seconds']:.5f}"],
             [f"{benchmarks['transitions_per_second']:.1f}", f"{gt_benchmarks['transitions_per_second']:.1f}"],
             [f"{benchmarks['nfe_per_transition']:.1f}", f"{gt_benchmarks['nfe_per_transition']:.1f}"],
             [f"{benchmarks['auxiliary_damping_queries_per_transition']:.1f}", f"{gt_benchmarks['auxiliary_damping_queries_per_transition']:.1f}"],
             [f"{benchmarks['peak_device_memory_bytes'] / 2**20:.1f}", f"{gt_benchmarks['peak_device_memory_bytes'] / 2**20:.1f}"]],
            compute_note + " The second column is the simulator's analytic operators in the same Lie-IMEX integrator with the same recorded u(t): the floor any learned model can reach.",
        ),
        figures_module.compute_figure(benchmark_series, selected, noise, device.platform.upper()),
        figures_module.controller_table(label, controller["metadata"], comparison, selected, noise),
        figures_module.gp_uncertainty_figure(standard_deviations),
        figures_module.image_page(Path(controller["metadata"]["plots"]["labeled_tracking_plot"]), f"{flat_label} — controller tracking plot"),
        figures_module.image_page(Path(controller["metadata"]["plots"]["labeled_trajectory_plot"]), f"{flat_label} — controller trajectory plot"),
    ])
    pages.extend(_audit_pages(run, controller, noise, truth.shape[1]))
    if len(pages) != EXPECTED_PAGES:
        raise AssertionError(f"Generated {len(pages)} pages, expected {EXPECTED_PAGES}")

    # Write --------------------------------------------------------------------
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%d-%m-%H-%M-%S")
    window_points = int(run["config"]["data"]["window_points"])
    step = float(metadata["step_size_seconds"])
    fit_tag = "NLL" if _is_gp(run["config"]) and run["config"]["gp"]["data_fit"] == "se3-nll" else "MSE"
    clip_tag = "noclip" if run["config"]["optimizer"]["gradient_clip_norm"] is None else "clip"
    common = (f"step{selected}-h{_tag(step)}-K{window_points - 1}-T{_tag(step * (window_points - 1))}-"
              f"nonoverlap{int(run['config']['data']['window_stride'])}-clean-common-eval-reference-{EXPECTED_PAGES}-page.pdf")
    if _is_gp(run["config"]):
        output = REPORT_ROOT / (
            f"{stamp}_report_single_PH-GP-LieIMEX-levels_obs-noise{_tag(noise)}_"
            f"{fit_tag}-logstd{_tag(float(run['config']['gp']['initial_log_standard_deviation']))}-"
            f"beta{_tag(float(run['config']['gp']['kl_beta_max']))}-lr{_tag(float(run['config']['optimizer']['learning_rate']))}-"
            f"{clip_tag}_" + common
        )
    else:
        output = REPORT_ROOT / (
            f"{stamp}_report_single_{_network_family(run['config'])}_obs-noise{_tag(noise)}_"
            f"{fit_tag}-width{run['config']['model'].get('hidden_dim')}-"
            f"lr{_tag(float(run['config']['optimizer']['learning_rate']))}-{clip_tag}_" + common
        )
    with PdfPages(output) as pdf:
        for figure in pages:
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
        info = pdf.infodict()
        info["Title"] = f"{step_text.capitalize()} {vehicle.get('damping_law', 'nonlinear')} damping: PH-GP-LieIMEX learned levels; training observation noise={noise}"
        info["Author"] = "Reproducible local JAX report"
    copy_name = Path(config["report"]["output_name"])
    if selected != run["total_steps"]:
        # Reports for intermediate checkpoints must not overwrite the final-checkpoint copy.
        copy_name = copy_name.with_name(f"{copy_name.stem}_step{selected:05d}{copy_name.suffix}")
    copy_target = run_dir / copy_name
    copy_target.write_bytes(output.read_bytes())

    payload = {
        "report": {
            "pdf": str(output),
            "sha256": evaluation.sha256(output),
            "pages": len(pages),
            "run_copy": str(copy_target),
            "reference_structure_pages": REFERENCE_SECTION_PAGES,
            "controller_plot_pages": 2,
            "gp_uncertainty_pages": 1,
            "supplemental_audit_pages": 6,
            "mode": "single-model",
            "selected_optimizer_step": selected,
            "training_observation_noise": noise,
            "evaluation_observations_clean": True,
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": evaluation.sha256(Path(__file__).resolve()),
            "created_utc": utc_now(),
            "device": device_text,
            "python": platform.python_version(),
            "jax": jax.__version__,
            "inference": {
                "posterior": (
                    "deterministic mean line; ±2σ band over posterior weight-sample rollouts on the state pages"
                    if posterior_sample_count > 0 else "deterministic mean"
                ),
                "precision": "jax.float64",
                "open_loop_step_seconds": evaluation.DT,
                "open_loop_transitions": int(truth.shape[1] - 1),
                "open_loop_states": int(truth.shape[1]),
                "open_loop_horizon_seconds": horizon,
                "closed_loop_seconds": controller_seconds,
                "closed_loop_physics_step_seconds": evaluation.DT,
            },
        },
        "evaluation_dataset": {"path": str(evaluation_dataset), "sha256": evaluation.sha256(evaluation_dataset), "settings": settings, "observations": "clean"},
        "posterior_sample_rollouts": posterior_sample_summary,
        "shared_open_loop_evaluation": {
            "num_trajectories": int(truth.shape[0]),
            "horizon_seconds": horizon,
            "identical_initial_state": True,
            "identical_recorded_pid_wrench": True,
            "rollout_error_summary": {
                key: {"mean": float(values.mean()), "final_mean": float(values[:, -1].mean()), "maximum": float(values.max())}
                for key, values in errors.items()
            },
            "rms_error_summary": figures_module.rms_summary(errors, truth),
            "ground_truth_operator_model": {
                "description": "analytic operators of the evaluation dataset inside the same Lie-IMEX integrator with the same recorded u(t)",
                "rollout_error_summary": {
                    key: {"mean": float(values.mean()), "final_mean": float(values[:, -1].mean()), "maximum": float(values.max())}
                    for key, values in gt_errors.items()
                },
                "rms_error_summary": figures_module.rms_summary(gt_errors, truth),
                "benchmark": gt_benchmarks,
            },
            "mass_gauge_fits": fits,
            "subnetwork_metric_summary": {
                metric: {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values))}
                for metric, values in subnet_metrics.items()
            },
            "gauge_invariant_products": {
                key: {
                    "nmse_mean": float(np.mean(product_metrics[key]["nmse"])),
                    "relative_rms_error_percent_mean": float(np.mean(product_metrics[key]["relative_rms_error_percent"])),
                    "model_axis_mean": [float(v) for v in product_metrics[key]["model_axis_mean"]],
                    "truth_axis_mean": [float(v) for v in product_metrics[key]["truth_axis_mean"]],
                    "relative_error_axis": [None if not np.isfinite(v) else float(v) for v in product_metrics[key]["relative_error_axis"]],
                    "offdiagonal_over_diagonal": (None if not np.isfinite(product_metrics[key]["offdiagonal_over_diagonal"])
                                                  else float(product_metrics[key]["offdiagonal_over_diagonal"])),
                }
                for key, _ in evaluation.PRODUCT_LABELS
            } | {"damping_v_speed_bins": product_metrics["damping_v_speed_bins"]},
        },
        "learned_levels": run["levels"],
        "model": metadata,
        "checkpoint": {"path": str(run["checkpoint"]), "sha256": run["checkpoint_sha256"]},
        "best_clean_validation_checkpoint": run["best_test"],
        "compute": {"benchmark_repeats": benchmark_repeats, "benchmarks": benchmarks},
        "failure_aware_controller_comparison": comparison,
        "controller": controller["metadata"],
    }
    save_json(output.with_suffix(".json"), payload)
    save_json(copy_target.with_suffix(".json"), payload)
    print(f"PDF: {output}", flush=True)
    print(f"JSON: {output.with_suffix('.json')}", flush=True)
    print(f"Pages: {len(pages)}", flush=True)
    return output


def generate_reports(config_path: str | Path, *, experiment_dirs: list[Path] | None = None) -> list[Path]:
    """Entry point used by the trainer after a completed run."""
    config = load_config(config_path)
    if int(config["report"]["exact_pages"]) != EXPECTED_PAGES:
        print(f"note: report.exact_pages={config['report']['exact_pages']} in the config; the generator now produces "
              f"{EXPECTED_PAGES} pages (page 3 and pages 4-9 are the gauge-invariant products)", flush=True)
    if experiment_dirs is None:
        listed = config["report"]["experiment_dirs"]
        if not listed:
            raise ValueError("report.experiment_dirs is empty and no experiment_dirs were supplied")
        experiment_dirs = [Path(resolve_project_path(item)) for item in listed]
    # Optional report.evaluation_dataset in the training config selects the clean 240 Hz reference
    # pickle for the open-loop pages (default: the 0.5 s D0 benchmark).
    configured_dataset = resolve_project_path(config["report"].get("evaluation_dataset"))
    evaluation_dataset = Path(configured_dataset) if configured_dataset is not None else COMMON_EVALUATION_DATASET
    outputs = []
    for run_dir in experiment_dirs:
        run_dir = Path(run_dir).resolve()
        output = generate_report(config, run_dir, evaluation_dataset=evaluation_dataset)
        append_history(config, run_dir, action="report", status="completed", details=f"pdf={output.name}")
        outputs.append(output)
    return outputs


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = (args.run or Path(args.config).resolve().parent).resolve()
    evaluation_dataset = args.evaluation_dataset
    if evaluation_dataset is None:
        configured = config.get("report", {}).get("evaluation_dataset")
        evaluation_dataset = Path(resolve_project_path(configured)) if configured else COMMON_EVALUATION_DATASET
    generate_report(
        config, run_dir,
        selected_step=args.selected_step,
        evaluation_dataset=Path(evaluation_dataset).resolve(),
        controller_seconds=args.controller_seconds,
        benchmark_repeats=args.benchmark_repeats,
    )


if __name__ == "__main__":
    sys.exit(main())

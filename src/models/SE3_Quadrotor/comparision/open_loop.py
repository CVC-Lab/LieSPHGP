"""Open-loop comparison: one table per evaluation set, plus the images that go with it.

Every model is handed the true initial state and the recorded wrench u(t) of each held-out flight and
integrates forward with its own integrator; nothing corrects it, so errors compound. Point rows score the
posterior-MEAN weights (the same object the closed-loop controller used); the probabilistic rows score an
ensemble of posterior weight SAMPLES, which only the GP has -- the baselines print a dash there.

Layout written by generate_comparison_report_v2:
    open-loop/comparison.pdf                 one page per evaluation set
    open-loop/images/<set>_states.png        one row per model x eight state columns, flight 0, GP band
    open-loop/images/<set>_error.png         position error against time, every model, log axis
    open-loop/images/<set>_calibration_<model>.png   GP only: reliability, band width, sigma vs error
"""
from __future__ import annotations

import math
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from . import report_evaluation as evaluation
from . import report_figures_v2 as v2
from ..ph_gp_lie_imex import network

VALID_PREDICTION_FRACTION = 0.158      # sqrt(0.025): the VPT convention, as a fraction of the flight's extent
SIGMA_FLOOR_M = 1e-3                   # every sample starts at the same x0, so sigma(0)=0: a 1 mm floor keeps NLPD finite
COVERAGE_LEVELS = ((0.50, 0.674), (0.80, 1.282), (0.90, 1.645), (0.954, 2.0), (0.99, 2.576))
STATE_COLUMNS = (("x (m)", "x"), ("y (m)", "y"), ("z (m)", "z"), ("roll (rad)", "roll"), ("pitch (rad)", "pitch"),
                 ("yaw (rad)", "yaw"), ("‖v‖ (m/s)", "speed"), ("‖ω‖ (rad/s)", "rate"))


# ---------------------------------------------------------------------------
# Evaluation sets
# ---------------------------------------------------------------------------
def load_open_loop_set(path, flights: int = evaluation.REPORT_TRAJECTORIES) -> dict[str, Any]:
    """A set of held-out flights in the (flight, time, 22) layout the rollout expects, with its time step.

    Two on-disk layouts are understood: the common D0 evaluation pickles (control windows, 240 Hz) and the
    HARD-V5/V6-style pickles, whose ``<split>_trajectories`` are whole flights sampled at ``sample_hz``.
    ``path@heldout`` / ``path@test`` selects the split of such a pickle; the default is the held-out
    (generalisation) split when the file has one, else the test split.
    """
    text = str(path)
    path, _, split = text.partition("@")
    path = Path(path)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    settings = data["settings"]
    if "test_trajectories" in data or "heldout_trajectories" in data:
        split = split or ("heldout" if "heldout_trajectories" in data else "test")
        truth = np.asarray(data[f"{split}_trajectories"], dtype=np.float64)[:flights]
        step = 1.0 / float(settings["config"]["environment"]["sample_hz"])
        source = f"{split} split of {path.stem} (whole flights)"
    else:
        common = evaluation.load_common_dataset(path)
        truth = evaluation.shared_truth(common, flights)
        step = evaluation.DT
        source = "common evaluation set (control windows)"
    name = path.stem + (f"_{split}" if split else "")
    return {"name": name, "path": path, "truth": truth, "dt": step,
            "vehicle": evaluation.vehicle_constants(settings), "source": source}


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
def timed_rollout(model, truth: np.ndarray, step: float, repeats: int = 3) -> tuple[np.ndarray | None, float]:
    """Posterior-mean rollout of every flight, and the median wall-clock milliseconds per flight."""
    try:
        evaluation.rollout(model, truth[:, : min(9, truth.shape[1])], step)      # compile and warm up
        timings, prediction = [], None
        for _ in range(repeats):
            started = time.perf_counter()
            prediction = evaluation.rollout(model, truth, step)
            timings.append(time.perf_counter() - started)
        return prediction, 1000.0 * float(np.median(timings)) / truth.shape[0]
    except FloatingPointError:
        return None, float("nan")


def posterior_rollouts(params, gp_setup, truth: np.ndarray, count: int, seed: int, step: float) -> np.ndarray:
    """(samples, flights, time, 18): every flight rolled out under ``count`` independent posterior weight draws."""
    variational = network.DissipativeSE3HamODE(params, gp_setup)
    rollouts = []
    for index in range(int(count)):
        sampled = variational.sample(jax.random.PRNGKey(int(seed) + index))
        try:
            rollouts.append(evaluation.rollout(sampled, truth, step))
        except FloatingPointError:
            continue                              # a diverged draw is dropped, and the count reported
    if not rollouts:
        raise FloatingPointError("Every posterior-sample rollout was non-finite")
    return np.stack(rollouts, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _extent(truth: np.ndarray) -> np.ndarray:
    """RMS distance of each flight from its own start: the scale the VPT threshold is a fraction of."""
    return np.sqrt(np.mean(np.sum((truth[..., :3] - truth[:, :1, :3]) ** 2, axis=-1), axis=1))


def valid_prediction_times(truth: np.ndarray, prediction: np.ndarray, step: float) -> np.ndarray:
    """Per flight: first time the position error exceeds 0.158 x the flight's extent, capped at the horizon."""
    error = np.linalg.norm(prediction[..., :3] - truth[..., :3], axis=-1)
    threshold = VALID_PREDICTION_FRACTION * _extent(truth)
    times = np.arange(truth.shape[1]) * step
    out = np.empty(truth.shape[0])
    for flight in range(truth.shape[0]):
        exceeded = np.flatnonzero(error[flight] > threshold[flight])
        out[flight] = times[exceeded[0]] if exceeded.size else times[-1]
    return out


def growth_rates(truth: np.ndarray, prediction: np.ndarray, step: float, valid: np.ndarray) -> np.ndarray:
    """Per flight: slope of log position error against time over the valid window (1/s); NaN if too short."""
    error = np.linalg.norm(prediction[..., :3] - truth[..., :3], axis=-1)
    times = np.arange(truth.shape[1]) * step
    out = np.full(truth.shape[0], np.nan)
    for flight in range(truth.shape[0]):
        window = (times > 0.05) & (times <= valid[flight]) & (error[flight] > 0)
        if window.sum() >= 10:
            out[flight] = np.polyfit(times[window], np.log(error[flight, window]), 1)[0]
    return out


def open_loop_metrics(truth: np.ndarray, prediction: np.ndarray | None, samples: np.ndarray | None,
                      step: float, compute_ms: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {"compute_ms_per_flight": compute_ms}
    if prediction is None:
        metrics.update({"diverged": True})
        return metrics
    errors = evaluation.rollout_errors(truth, prediction, {"mass": 1.0, "inertia": np.eye(3), "gravity": 0.0})
    valid = valid_prediction_times(truth, prediction, step)
    metrics.update({
        "diverged": False,
        "position_rmse_m": float(np.sqrt(np.mean(errors["position"]))),
        "attitude_rmse_deg": float(np.degrees(np.sqrt(np.mean(errors["attitude"])))),
        "velocity_rmse_m_per_s": float(np.sqrt(np.mean(errors["velocity"]))),
        "omega_rmse_rad_per_s": float(np.sqrt(np.mean(errors["omega"]))),
        "valid_prediction_seconds_median": float(np.median(valid)),
        "valid_prediction_seconds_per_flight": valid.tolist(),
        "growth_rate_per_s_median": float(np.nanmedian(growth_rates(truth, prediction, step, valid))),
        "so3_orthogonality": float(np.mean(errors["orthogonality"])),
    })
    if samples is None or len(samples) < 2:
        return metrics
    # Probabilistic rows: ensemble mean and spread of the position over the posterior draws.
    mu = samples[..., :3].mean(axis=0)                                         # (F, T, 3)
    sigma = np.sqrt(samples[..., :3].var(axis=0, ddof=1) + SIGMA_FLOOR_M ** 2)  # floor: sigma(0) = 0
    residual = truth[..., :3] - mu
    z = residual / sigma
    metrics.update({
        "posterior_samples": int(samples.shape[0]),
        "nlpd": float(np.mean(0.5 * z ** 2 + 0.5 * np.log(2 * np.pi * sigma ** 2))),
        "coverage": {f"{nominal:.3f}": float(np.mean(np.abs(z) <= score)) for nominal, score in COVERAGE_LEVELS},
        "coverage_95": float(np.mean(np.abs(z) <= 2.0)),
        "sharpness_m": float(np.mean(2.0 * sigma)),
        "plugin_vs_predictive_mean_gap_m": float(np.sqrt(np.mean(np.sum((prediction[..., :3] - mu) ** 2, axis=-1)))),
    })
    band = np.linalg.norm(sigma[:, 1:], axis=-1).ravel()
    error = np.linalg.norm(prediction[:, 1:, :3] - truth[:, 1:, :3], axis=-1).ravel()
    metrics["sigma_error_spearman"] = float(spearmanr(band, error).correlation)
    ensemble_valid = np.concatenate([valid_prediction_times(truth, sample, step) for sample in samples])
    metrics["ensemble_valid_prediction_seconds"] = [float(np.percentile(ensemble_valid, q)) for q in (25, 50, 75)]
    return metrics


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def open_loop_table(entries, set_name: str, horizon: float, flights: int, samples: int) -> plt.Figure:
    """entries: (label, metrics dict). Rows = metrics; PH-GT is a column that is never ranked."""
    rows = [
        ("position RMSE (m)", "position_rmse_m", "{:.3f}", "low"),
        ("attitude error, geodesic (deg)", "attitude_rmse_deg", "{:.2f}", "low"),
        ("velocity RMSE (m/s)", "velocity_rmse_m_per_s", "{:.3f}", "low"),
        ("angular-rate RMSE (rad/s)", "omega_rmse_rad_per_s", "{:.3f}", "low"),
        ("valid prediction time, median (s)", "valid_prediction_seconds_median", "{:.2f}", "high"),
        ("error growth rate, median (1/s)", "growth_rate_per_s_median", "{:.2f}", "low"),
        ("SO(3) violation ‖RᵀR−I‖", "so3_orthogonality", "{:.1e}", "low"),
        ("rollout compute (ms per flight)", "compute_ms_per_flight", "{:.1f}", "low"),
        ("NLPD, position (nats)", "nlpd", "{:.2f}", "low"),
        ("coverage at 2σ (target 0.954)", "coverage_95", "{:.3f}", "target"),
        ("sharpness, mean 2σ (m)", "sharpness_m", "{:.3f}", "low"),
        ("Spearman ρ(σ, error)", "sigma_error_spearman", "{:+.2f}", "high"),
        ("ensemble VPT, median [Q1, Q3] (s)", "ensemble_valid_prediction_seconds", None, None),
        ("plug-in vs predictive-mean gap (m)", "plugin_vs_predictive_mean_gap_m", "{:.3f}", None),
    ]
    columns = [label for label, _ in entries]
    truth_column = next((i for i, (label, _) in enumerate(entries) if label.replace("\n", " ").startswith("PH-GT")), None)
    learned = [i for i in range(len(entries)) if i != truth_column]
    cells, labels, highlight = [], [], {}
    for row_index, (name, key, fmt, direction) in enumerate(rows):
        labels.append(name)
        line = []
        for _, m in entries:
            value = m.get(key)
            if m.get("diverged") and key != "compute_ms_per_flight":
                line.append("diverged")
            elif value is None or (isinstance(value, float) and not np.isfinite(value)):
                line.append("—")
            elif key == "ensemble_valid_prediction_seconds":
                line.append(f"{value[1]:.2f} [{value[0]:.2f}, {value[2]:.2f}]")
            else:
                line.append(fmt.format(value))
        cells.append(line)
        if direction is None:
            continue
        values = {i: entries[i][1].get(key) for i in learned}
        values = {i: v for i, v in values.items() if isinstance(v, (int, float)) and np.isfinite(v)}
        if len(values) < 2:
            continue                              # a row only one model can fill has nothing to rank
        if direction == "high":
            best = max(values, key=values.get)
        elif direction == "target":
            best = min(values, key=lambda i: abs(values[i] - 0.954))
        else:
            best = min(values, key=values.get)
        printed = fmt.format(values[best])
        highlight[row_index] = [i for i in values if fmt.format(values[i]) == printed]
    return v2._table(
        f"Open-loop prediction — {set_name}: {flights} held-out flights, {horizon:.0f} s, true x₀ and recorded u(t)\n"
        f"point rows use the posterior-mean weights; probabilistic rows use {samples} posterior weight samples "
        f"(GP only — the baselines are point estimates and have no band)\n"
        "bold = best learned model in that row (PH-GT is the reference, never ranked); "
        f"VPT threshold = {VALID_PREDICTION_FRACTION:.3f} × the flight's own extent",
        labels, columns, cells, "", fontsize=8.6, scale=(1.0, 2.0), highlight=highlight,
    )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def _columns_of(states: np.ndarray) -> dict[str, np.ndarray]:
    euler = evaluation.euler_angles(states)
    return {"x": states[..., 0], "y": states[..., 1], "z": states[..., 2],
            "roll": euler[..., 0], "pitch": euler[..., 1], "yaw": euler[..., 2],
            "speed": np.linalg.norm(states[..., 12:15], axis=-1), "rate": np.linalg.norm(states[..., 15:18], axis=-1)}


def states_grid(entries, truth: np.ndarray, step: float, set_name: str, flight: int = 0) -> plt.Figure:
    """entries: (short label, prediction (F,T,18) or None, samples (S,F,T,18) or None). Flight ``flight`` only."""
    times = np.arange(truth.shape[1]) * step
    truth_columns = _columns_of(truth[flight:flight + 1])
    rows = len(entries)
    figure, axes = plt.subplots(rows, len(STATE_COLUMNS), figsize=(2.45 * len(STATE_COLUMNS), 1.75 * rows), squeeze=False)
    for row, (label, prediction, samples) in enumerate(entries):
        predicted = _columns_of(prediction[flight:flight + 1]) if prediction is not None else None
        sampled = _columns_of(samples[:, flight]) if samples is not None else None
        for column, (name, key) in enumerate(STATE_COLUMNS):
            axis = axes[row][column]
            axis.grid(True, alpha=0.3)
            if sampled is not None:
                mean, spread = sampled[key].mean(axis=0), sampled[key].std(axis=0, ddof=1)
                axis.fill_between(times, mean - 2 * spread, mean + 2 * spread, color="b", alpha=0.18, lw=0,
                                  label="±2σ, posterior samples" if (row == 0 and column == 0) else None)
            if predicted is not None:
                axis.plot(times, predicted[key][0], color="b", lw=1.0,
                          label="prediction" if (row == 0 and column == 0) else None)
            axis.plot(times, truth_columns[key][0], "--", color="chocolate", lw=1.0,
                      label="truth" if (row == 0 and column == 0) else None)
            axis.tick_params(labelsize=6)
            if row == 0:
                axis.set_title(name, fontsize=9, fontweight="bold")
            if row == rows - 1:
                axis.set_xlabel("Time (s)", fontsize=7)
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(label.replace("  ", "\n"), fontsize=7, fontweight="bold", labelpad=2)
    handles, labels = [], []
    for axis in axes[0]:
        h, l = axis.get_legend_handles_labels(); handles += h; labels += l
    figure.legend(handles, labels, loc="lower center", ncol=3, prop={"size": 10})
    figure.suptitle(f"Open-loop prediction — {set_name}, flight {flight} of {truth.shape[0]} — prediction vs truth\n"
                    "true x₀ and recorded u(t); one row per model; the shaded band is the GP's ±2σ over posterior weight samples",
                    fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    return figure


def trajectory_grid(entries, truth: np.ndarray, step: float, set_name: str, flight: int = 0,
                    sample_paths: int = 20) -> plt.Figure:
    """entries: (short label, prediction (F,T,18) or None, samples (S,F,T,18) or None). One 3-D axis per model.

    Each axis is scaled to its own prediction plus the truth, so a model that leaves the box shows a large span
    instead of being clipped; the span is printed in the sub-title. The GP panel also draws a few posterior
    sample paths faintly -- the 3-D counterpart of the band on the state grid.
    """
    count = len(entries)
    columns = min(4, count)
    rows = math.ceil(count / columns)
    figure = plt.figure(figsize=(4.1 * columns, 4.0 * rows))
    true_path = truth[flight, :, :3]
    for index, (label, prediction, samples) in enumerate(entries, start=1):
        axis = figure.add_subplot(rows, columns, index, projection="3d")
        if samples is not None:
            for path in samples[:sample_paths, flight, :, :3]:
                axis.plot3D(path[:, 0], path[:, 1], path[:, 2], color="b", lw=0.5, alpha=0.12)
        if prediction is not None:
            predicted = prediction[flight, :, :3]
            axis.plot3D(predicted[:, 0], predicted[:, 1], predicted[:, 2], color="b", lw=1.4,
                        label="prediction" if index == 1 else None)
        else:
            predicted = true_path
        axis.plot3D(true_path[:, 0], true_path[:, 1], true_path[:, 2], "--", color="chocolate", lw=1.5,
                    label="truth" if index == 1 else None)
        positions = np.concatenate((predicted, true_path), axis=0)
        lower, upper = np.nanmin(positions, axis=0), np.nanmax(positions, axis=0)
        span = max(float(np.max(upper - lower)) * 1.1, 2.0)
        centre = (lower + upper) / 2.0
        axis.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
        axis.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
        axis.set_zlim(centre[2] - span / 2, centre[2] + span / 2)
        axis.view_init(elev=25.0, azim=35.0)
        axis.set_xlabel("x (m)", fontsize=7); axis.set_ylabel("y (m)", fontsize=7); axis.set_zlabel("z (m)", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.set_title(f"{label}\naxis span {span:.1f} m" + ("\n(faint: posterior sample paths)" if samples is not None else ""),
                       fontsize=9, fontweight="bold")
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, prop={"size": 11})
    horizon = (truth.shape[1] - 1) * step
    figure.suptitle(f"Open-loop trajectory — {set_name}, flight {flight} of {truth.shape[0]}, {horizon:.0f} s — prediction vs truth\n"
                    "true x₀ and recorded u(t), no correction; one panel per model, each scaled to its own flight",
                    fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    return figure


def error_figure(entries, truth: np.ndarray, step: float, set_name: str, styles) -> plt.Figure:
    """entries: (short label, prediction or None). Mean-over-flights position error against time, log axis."""
    times = np.arange(truth.shape[1]) * step
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for (label, prediction), (colour, style) in zip(entries, styles):
        if prediction is None:
            continue
        error = np.linalg.norm(prediction[..., :3] - truth[..., :3], axis=-1)
        axis.plot(times, np.maximum(error.mean(axis=0), 1e-6), color=colour, ls=style, lw=1.5, label=label)
    threshold = VALID_PREDICTION_FRACTION * float(np.median(_extent(truth)))
    axis.axhline(threshold, color="0.3", ls=":", lw=1.2, label=f"VPT threshold (median flight): {threshold:.3f} m")
    axis.set_yscale("log"); axis.grid(True, which="both", alpha=0.3)
    axis.set_xlabel("Time (s)"); axis.set_ylabel("position error ‖p̂ − p‖ (m), mean over flights")
    axis.set_title(f"Open-loop position error — {set_name}, {truth.shape[0]} flights, true x₀ and recorded u(t)",
                   fontsize=12, fontweight="bold")
    axis.legend(fontsize=9)
    figure.tight_layout()
    return figure


def calibration_figure(label: str, truth: np.ndarray, prediction: np.ndarray, samples: np.ndarray, step: float,
                       set_name: str, metrics: dict[str, Any]) -> plt.Figure:
    """GP only: reliability diagram, band width against time, and sigma against the realised error."""
    times = np.arange(truth.shape[1]) * step
    mu = samples[..., :3].mean(axis=0)
    sigma = np.sqrt(samples[..., :3].var(axis=0, ddof=1) + SIGMA_FLOOR_M ** 2)
    z = np.abs(truth[..., :3] - mu) / sigma
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    nominal = [level for level, _ in COVERAGE_LEVELS]
    empirical = [float(np.mean(z <= score)) for _, score in COVERAGE_LEVELS]
    axes[0].plot([0, 1], [0, 1], "k:", lw=1); axes[0].plot(nominal, empirical, "o-", color="b")
    for n, e in zip(nominal, empirical):
        axes[0].annotate(f"{e:.2f}", (n, e), textcoords="offset points", xytext=(4, -10), fontsize=8)
    axes[0].set_xlabel("nominal coverage"); axes[0].set_ylabel("empirical coverage (position, all flights)")
    axes[0].set_title("reliability — on the diagonal = calibrated", fontsize=10); axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, (2 * sigma).mean(axis=(0, 2)), color="b", label="mean 2σ over flights and axes")
    axes[1].plot(times, np.linalg.norm(prediction[..., :3] - truth[..., :3], axis=-1).mean(axis=0),
                 color="chocolate", ls="--", label="mean position error")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("m"); axes[1].set_yscale("log"); axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_title("band width against realised error", fontsize=10); axes[1].legend(fontsize=8)
    band = np.linalg.norm(sigma[:, 1:], axis=-1).ravel()
    error = np.linalg.norm(prediction[:, 1:, :3] - truth[:, 1:, :3], axis=-1).ravel()
    axes[2].loglog(band, np.maximum(error, 1e-6), ".", ms=2, alpha=0.25, color="b")
    axes[2].set_xlabel("σ of the position band (m)"); axes[2].set_ylabel("position error (m)")
    axes[2].set_title(f"does σ track the error?  Spearman ρ = {metrics.get('sigma_error_spearman', float('nan')):+.2f}", fontsize=10)
    axes[2].grid(True, which="both", alpha=0.3)
    figure.suptitle(f"Posterior calibration — {label} — {set_name} ({samples.shape[0]} posterior weight samples)\n"
                    f"NLPD {metrics.get('nlpd', float('nan')):.2f}   coverage@2σ {metrics.get('coverage_95', float('nan')):.3f}   "
                    f"sharpness {metrics.get('sharpness_m', float('nan')):.3f} m",
                    fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure

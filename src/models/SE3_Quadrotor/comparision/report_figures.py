"""Matplotlib pages of the 47-page single-model PH-GP-LieIMEX report.

The page layouts, titles, table rows and notes follow the established
reference report exactly; only the model label and the audited facts change
with the run.  Every function returns one matplotlib figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report_evaluation import attitude_error_squared, euler_angles, physical_energy

plt.rcParams["figure.max_open_warning"] = 60

MODEL_COLOR = "#CC79A7"
MODEL_LINESTYLE = ":"
GT_STYLE = ("GT", "#000000", "-")
GROUND_TRUTH_MODEL_STYLE = ("GT operators + Lie-IMEX\nh=1/240, recorded u", "#0072B2", "--")
# A "series" is a list of (label, color, linestyle, payload) tuples, one per model.


def checkpoint_step_text(selected_step: int) -> str:
    return f"optimizer step {selected_step}"


def format_cell(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"{float(value):.4e}"
    return str(value)


def ms(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2e}±{values.std():.1e}"


def smooth(values: np.ndarray, window: int = 101) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return values
    width = min(window, values.size if values.size % 2 else values.size - 1)
    width = max(width, 3)
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


# ---------------------------------------------------------------------------
# Generic pages
# ---------------------------------------------------------------------------
def table_page(title: str, rows: list[str], columns: list[str], cells, note: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11.3, 7.6))
    axis.axis("off")
    axis.set_title(title, fontsize=13, fontweight="bold", pad=15)
    table = axis.table(
        cellText=cells,
        rowLabels=rows,
        colLabels=columns,
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)
    for column in range(len(columns)):
        table[0, column].set_text_props(weight="bold")
    axis.text(0.02, 0.04, note, transform=axis.transAxes, fontsize=9, wrap=True)
    figure.tight_layout()
    return figure


def image_page(path: Path, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11.3, 7.6))
    axis.imshow(plt.imread(path))
    axis.axis("off")
    axis.set_title(title, fontsize=13, fontweight="bold", pad=8)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Pages 1-2: summary tables
# ---------------------------------------------------------------------------
def ground_truth_excursions(truth: np.ndarray) -> dict[str, np.ndarray]:
    """Squared distance of the ground truth from its own initial state, per step."""
    position = np.sum((truth[..., :3] - truth[:, :1, :3]) ** 2, axis=-1)
    velocity = np.sum((truth[..., 12:15] - truth[:, :1, 12:15]) ** 2, axis=-1)
    omega = np.sum((truth[..., 15:18] - truth[:, :1, 15:18]) ** 2, axis=-1)
    initial = np.repeat(truth[:, :1], truth.shape[1], axis=1)
    attitude = attitude_error_squared(initial, truth)
    return {"position": position, "attitude": attitude, "velocity": velocity, "omega": omega}


def rms_summary(errors: dict, truth: np.ndarray) -> dict[str, dict[str, float]]:
    """RMS errors in physical units and relative to the ground truth's own excursion."""
    excursions = ground_truth_excursions(truth)
    summary = {}
    for key in ("position", "attitude", "velocity", "omega"):
        rms = np.sqrt(errors[key].mean(axis=1))
        reference = np.sqrt(np.maximum(excursions[key].mean(axis=1), 1e-30))
        summary[key] = {
            "rms_mean": float(rms.mean()),
            "rms_final": float(np.sqrt(errors[key][:, -1]).mean()),
            "ground_truth_excursion_rms": float(reference.mean()),
            "relative": float((rms / reference).mean()),
        }
    return summary


def summary_figure(series: list, horizon: float, title: str, truth: np.ndarray | None = None) -> plt.Figure:
    """``series``: list of (label, color, linestyle, errors) for every model column."""
    rows = [
        ("‖Δx‖² mean", "position", "mean"),
        ("‖Δx‖² final", "position", "final"),
        ("geo² mean", "attitude", "mean"),
        ("geo² final", "attitude", "final"),
        ("‖Δv‖² mean", "velocity", "mean"),
        ("‖Δv‖² final", "velocity", "final"),
        ("‖Δω‖² mean", "omega", "mean"),
        ("‖Δω‖² final", "omega", "final"),
        ("|ΔE| mean", "energy", "mean"),
        ("max|det(R)−1|", "determinant", "max"),
        ("max‖RᵀR−I‖", "orthogonality", "max"),
    ]
    labels = [item[0] for item in series]
    cells = []
    for _label, key, reduction in rows:
        values = []
        for _name, _color, _ls, errors in series:
            array = errors[key]
            if reduction == "mean":
                per_trajectory = array.mean(axis=1)
            elif reduction == "final":
                per_trajectory = array[:, -1]
            else:
                per_trajectory = array.max(axis=1)
            values.append(ms(per_trajectory))
        cells.append(values)
    row_labels = [row[0] for row in rows]
    if truth is not None:
        excursions = ground_truth_excursions(truth)
        for name, key, unit in (
            ("‖Δx‖", "position", "m"),
            ("geo", "attitude", "rad"),
            ("‖Δv‖", "velocity", "m/s"),
            ("‖Δω‖", "omega", "rad/s"),
        ):
            reference = np.sqrt(np.maximum(excursions[key].mean(axis=1), 1e-30))
            rms_row, final_row, relative_row = [], [], []
            for _name, _color, _ls, errors in series:
                rms = np.sqrt(errors[key].mean(axis=1))
                rms_row.append(ms(rms))
                final_row.append(ms(np.sqrt(errors[key][:, -1])))
                relative_row.append(ms(rms / reference))
            row_labels += [f"{name} RMS ({unit})", f"{name} final ({unit})", f"{name} RMS / GT excursion"]
            cells += [rms_row, final_row, relative_row]
    figure, axis = plt.subplots(figsize=(11.3, 7.6))
    axis.axis("off")
    axis.set_title(title, fontsize=11.5, fontweight="bold", pad=12)
    table = axis.table(
        cellText=cells,
        rowLabels=row_labels,
        colLabels=labels,
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0 if truth is not None else 8.5)
    table.scale(1.0, 1.15 if truth is not None else 1.55)
    for column in range(len(labels)):
        table[0, column].set_text_props(weight="bold")
    if truth is not None:
        axis.text(
            0.5, 0.01,
            "Squared rows are the reference-report metrics (SI units squared). RMS rows are their square roots in physical "
            "units; the relative rows divide the RMS error by the RMS excursion of the ground truth from its initial state "
            "over the same horizon, so 1.0 means the error equals the true motion.",
            transform=axis.transAxes, ha="center", va="bottom", fontsize=7.5, wrap=True,
        )
    figure.tight_layout()
    return figure


def subnetwork_summary(label: str, metrics: dict[str, np.ndarray], horizon: float) -> plt.Figure:
    """Page 2: one row per subnetwork, absolute MSE next to the unit-free NMSE and relative RMS error."""
    columns = ("MSE — raw", "MSE — gauge-fixed", "NMSE — raw", "NMSE — gauge-fixed", "relative RMS error (%) — gauge-fixed")
    headers = ("MSE raw", "MSE gauge-fixed", "NMSE raw", "NMSE gauge-fixed", "rel. RMS error (%)\ngauge-fixed = 100·√NMSE")
    names = []
    for key in metrics:
        name = key.split(" MSE — ")[0] if " MSE — " in key else None
        if name and name not in names:
            names.append(name)
    cells = []
    for name in names:
        row = []
        for column in columns:
            values = metrics.get(f"{name} {column}")
            if values is None:
                row.append("–")
            elif column.startswith("relative"):
                values = np.asarray(values, dtype=float); row.append(f"{values.mean():.2f}±{values.std():.2f}")
            else:
                row.append(ms(values))
        cells.append(row)
    figure, axis = plt.subplots(figsize=(11.3, 7.6))
    axis.axis("off")
    axis.set_title(
        f"Subnetwork error against analytical ground truth — horizon {horizon:.3f}s\n"
        f"{label}: absolute MSE (SI units squared) and unit-free NMSE; "
        "raw and gauge-fixed rows use the per-time subnetwork values shown on pages 31–42; page 3 has the gauge-invariant products",
        fontsize=10.5,
        fontweight="bold",
        pad=12,
    )
    table = axis.table(
        cellText=cells,
        rowLabels=names,
        colLabels=headers,
        colWidths=[0.17] * len(headers),
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.6)
    for column_index in range(len(headers)):
        table[0, column_index].set_text_props(weight="bold")
        table[0, column_index].set_height(table[0, column_index].get_height() * 1.6)
    axis.text(
        0.5,
        0.06,
        "MSE = mean of (prediction − analytical GT)² over all time steps and all scalar subnetwork outputs. "
        "NMSE = MSE / mean of GT² over the same horizon, so it is unit-free: 0 = perfect, 1 = as bad as predicting zero, "
        "and √NMSE is the relative RMS error (last column, in %). "
        "Raw NMSE near 1 for M1⁻¹, M2⁻¹, Dv, Dω and g reflects the port-Hamiltonian gauge freedom (only the products "
        "M⁻¹g, M⁻¹D and M⁻¹∇V are identifiable), which the gauge-fixed rows remove with one scale per mass block. "
        "Table entries are mean±SD over the 10 report trajectories.",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Page 3 and pages 4-9: gauge-invariant products against the analytical ground truth
# ---------------------------------------------------------------------------
def product_summary(label: str, metrics: dict, product_labels, horizon: float) -> plt.Figure:
    figure, (axis_top, axis_bottom) = plt.subplots(2, 1, figsize=(13.5, 9.2), gridspec_kw={"height_ratios": [5, 3.2]})
    for axis in (axis_top, axis_bottom):
        axis.axis("off")
    headers = ("model\n(x, y, z)", "truth\n(x, y, z)", "rel. error\nper axis (%)", "off-diag /\ndiag (%)", "NMSE", "rel. RMS\nerror (%)")
    rows, cells = [], []
    for key, name in product_labels:
        m = metrics[key]
        fmt = lambda v: ", ".join("–" if not np.isfinite(x) else (f"{x:.0f}" if abs(x) >= 1000 else f"{x:.3g}") for x in v)
        rel = ", ".join("–" if not np.isfinite(x) else f"{100*x:+.1f}" for x in m["relative_error_axis"])
        off = "–" if not np.isfinite(m["offdiagonal_over_diagonal"]) else f"{100*m['offdiagonal_over_diagonal']:.1f}"
        rows.append(name)
        cells.append([fmt(m["model_axis_mean"]), fmt(m["truth_axis_mean"]), rel, off,
                      ms(m["nmse"]), f"{np.mean(m['relative_rms_error_percent']):.1f}±{np.std(m['relative_rms_error_percent']):.1f}"])
    axis_top.set_title(
        f"Gauge-invariant products against analytical ground truth — horizon {horizon:.3f}s\n{label}\n"
        "These products are what the trajectories identify; no scale fit is applied (compare the gauge-fixed rows of page 2)",
        fontsize=10.5, fontweight="bold", pad=10,
    )
    table = axis_top.table(cellText=cells, rowLabels=rows, colLabels=headers, colWidths=[0.16, 0.15, 0.13, 0.1, 0.13, 0.1],
                           cellLoc="center", rowLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(7.2); table.scale(1.0, 1.7)
    for column_index in range(len(headers)):
        table[0, column_index].set_text_props(weight="bold")
        table[0, column_index].set_height(table[0, column_index].get_height() * 1.7)
    bins = metrics["damping_v_speed_bins"]
    bin_rows = [f"|v| in [{b['low']:.2f}, {b['high']:.2f}) m/s" for b in bins]
    bin_cells = [[str(b["count"]), f"{b['model']:.3g}", f"{b['truth']:.3g}",
                  "–" if not np.isfinite(b["model"]) else f"{b['model']/b['truth']:.2f}"] for b in bins]
    axis_bottom.set_title("Translational damping μ·Dv by speed (mean diagonal over report samples; truth c(1+|v|))",
                          fontsize=10, fontweight="bold", pad=6)
    table2 = axis_bottom.table(cellText=bin_cells, rowLabels=bin_rows, colLabels=("samples", "model μ·Dv", "truth μ·Dv", "ratio model/truth"),
                               colWidths=[0.12, 0.14, 0.14, 0.16], cellLoc="center", rowLoc="left", loc="center")
    table2.auto_set_font_size(False); table2.set_fontsize(7.5); table2.scale(1.0, 1.5)
    for column_index in range(4):
        table2[0, column_index].set_text_props(weight="bold")
    axis_bottom.text(0.5, -0.02,
        "Per-axis columns show the mean of the diagonal (3×3 products) or of the vector components over the horizon; "
        "rel. error per axis = model/truth − 1. NMSE and rel. RMS error are entry-wise over the whole product "
        "(mean±SD over the 10 report trajectories). A ratio above 1 in the speed table is damping learned too high.",
        transform=axis_bottom.transAxes, ha="center", va="top", fontsize=7.5, wrap=True)
    figure.tight_layout()
    return figure


def product_trajectory_figure(label, products, targets, time_axis, key, title, units) -> plt.Figure:
    pred, target = products[key][0], targets[key][0]
    if pred.ndim == 3:
        figure, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True)
        for index, axis in enumerate(axes.flat):
            row, column = divmod(index, 3)
            axis.plot(time_axis, target[:, row, column], "k:", lw=1.6, label="GT")
            axis.plot(time_axis, pred[:, row, column], color=MODEL_COLOR, ls=MODEL_LINESTYLE, lw=1.2, label=label)
            axis.set_title(f"({row},{column})"); axis.grid(True, alpha=0.3)
            if row == 2:
                axis.set_xlabel("Time (s)")
        axes[0, 0].legend(fontsize="x-small")
    else:
        figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for index, axis in enumerate(axes):
            axis.plot(time_axis, target[:, index], "k:", lw=1.6, label="GT")
            axis.plot(time_axis, pred[:, index], color=MODEL_COLOR, ls=MODEL_LINESTYLE, lw=1.2, label=label)
            axis.set_title(("x", "y", "z")[index] + " component"); axis.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)"); axes[0].legend(fontsize="x-small")
    figure.suptitle(f"{title} along the shared-PID trajectory — gauge-invariant, no scale fit\nunits: {units}",
                    fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def damping_speed_figure(label, products, targets, truth, vehicle) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    c = vehicle["damping"]
    for axis, key, norm_slice, name in ((axes[0], "damping_v", slice(12, 15), "|v_body| (m/s)"),
                                        (axes[1], "damping_w", slice(15, 18), "|ω_body| (rad/s)")):
        magnitude = np.linalg.norm(truth[..., norm_slice], axis=-1).reshape(-1)
        diag = np.stack([products[key][..., i, i] for i in range(3)], -1).reshape(-1, 3)
        for i, colour in enumerate(("#D55E00", "#009E73", "#CC79A7")):
            axis.scatter(magnitude, diag[:, i], s=4, alpha=0.35, color=colour, label=f"{label} diag[{i}]")
        grid = np.linspace(0, max(magnitude.max(), 1e-3), 200)
        linear_law = vehicle.get("damping_law", "nonlinear") == "linear"
        axis.plot(grid, np.full_like(grid, c) if linear_law else c * (1.0 + grid), "k:", lw=2.0,
                  label="GT  c (linear law)" if linear_law else "GT  c(1+|·|)")
        axis.set_xlabel(name); axis.set_ylabel("product diagonal (1/s)"); axis.grid(True, alpha=0.3); axis.legend(fontsize="x-small")
    axes[0].set_title("translational damping μ·Dv vs speed"); axes[1].set_title("rotational damping M2⁻¹·Dω vs rate")
    figure.suptitle("Learned damping products against the true speed-dependent law (report-trajectory samples)", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


# ---------------------------------------------------------------------------
# Pages 3-12: training curves
# ---------------------------------------------------------------------------
def training_figure(label: str, steps: np.ndarray, values: np.ndarray, title: str, ylabel: str, *, test: bool) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.set_title(title)
    axis.set_xlabel("step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.set_yscale("log")
    values = np.asarray(values, dtype=float)
    steps = np.asarray(steps, dtype=int)[: len(values)]
    if not test:
        values = smooth(values)
    axis.plot(steps, np.maximum(values, 1e-14), color=MODEL_COLOR, ls=MODEL_LINESTYLE, lw=1.4, label=label)
    axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Pages 13-23: open-loop comparison with ground truth
# ---------------------------------------------------------------------------
def error_figure(series: list, time_axis: np.ndarray, key: str, title: str, ylabel: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 6))
    for label, color, linestyle, errors in series:
        axis.plot(time_axis, np.maximum(errors[key].mean(axis=0), 1e-14), color=color, ls=linestyle, lw=1.6, label=label)
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def energy_figures(series: list, truth: np.ndarray, time_axis: np.ndarray, vehicle: dict) -> tuple[plt.Figure, plt.Figure]:
    """``series``: list of (label, color, linestyle, prediction)."""
    gt_energy = physical_energy(truth, vehicle)
    energies = [(label, color, ls, physical_energy(prediction, vehicle)) for label, color, ls, prediction in series]
    figure_1, axis_1 = plt.subplots(figsize=(11, 6))
    for index in range(gt_energy.shape[0]):
        axis_1.plot(time_axis, gt_energy[index], "k-", lw=1.1, alpha=0.55, label="GT" if index == 0 else None)
    for label, color, linestyle, predicted in energies:
        for index in range(predicted.shape[0]):
            axis_1.plot(time_axis, predicted[index], color=color, ls=linestyle, lw=1.1, alpha=0.55,
                        label=label if index == 0 else None)
    axis_1.set_title(f"Physical energy — {gt_energy.shape[0]} distinct held-out flights")
    axis_1.set_xlabel("Time (s)")
    axis_1.set_ylabel("Energy (J)")
    axis_1.grid(True, alpha=0.3)
    axis_1.legend(fontsize="small")
    figure_1.tight_layout()

    figure_2, axis_2 = plt.subplots(figsize=(11, 6))
    axis_2.plot(time_axis, gt_energy[0], "k-", lw=2, label="GT")
    for label, color, linestyle, predicted in energies:
        axis_2.plot(time_axis, predicted[0], color=color, ls=linestyle, lw=1.6, label=label)
    axis_2.set_title("Physical energy — flight 0 of the held-out set")
    axis_2.set_xlabel("Time (s)")
    axis_2.set_ylabel("Energy (J)")
    axis_2.grid(True, alpha=0.3)
    axis_2.legend(fontsize="small")
    figure_2.tight_layout()
    return figure_1, figure_2


def geometry_figure(series: list, time_axis: np.ndarray, key: str, title: str, ylabel: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(time_axis, np.full_like(time_axis, 1e-16), "k--", lw=1.5, label="GT reference")
    for label, color, linestyle, errors in series:
        axis.plot(time_axis, np.maximum(errors[key].mean(0), 1e-16), color=color, ls=linestyle, lw=1.5, label=label)
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def _state_values(states: np.ndarray):
    return (
        (states[..., :3], ("x (m)", "y (m)", "z (m)")),
        (euler_angles(states), ("roll (rad)", "pitch (rad)", "yaw (rad)")),
        (states[..., 12:15], ("v_x (m/s)", "v_y (m/s)", "v_z (m/s)")),
        (states[..., 15:18], ("omega_x", "omega_y", "omega_z")),
    )


def _draw_sample_bands(axis, time_axis: np.ndarray, sample_bands: list | None, row: int, column: int) -> None:
    """Shade mean ± 2 std across posterior-sample rollouts; ``sample_bands``: [(label, color, samples (S, T, 18))]."""
    for label, color, samples in sample_bands or []:
        values = _state_values(samples)[row][0][..., column]
        mean, std = values.mean(0), values.std(0)
        axis.fill_between(time_axis, mean - 2 * std, mean + 2 * std, color=color, alpha=0.2, lw=0,
                          label=f"±2σ, {samples.shape[0]} posterior samples" if (row, column) == (0, 0) else None)


def state_ensemble_figure(series: list, truth: np.ndarray, time_axis: np.ndarray, sample_bands: list | None = None) -> plt.Figure:
    """``series``: list of (label, color, linestyle, prediction); ``sample_bands``: [(label, color, samples (S, T, 18))]."""
    figure, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
    truth_blocks = _state_values(truth)
    prediction_blocks = [(label, color, ls, _state_values(prediction)) for label, color, ls, prediction in series]
    for row, (gt_values, labels) in enumerate(truth_blocks):
        for column in range(3):
            axis = axes[row, column]
            # The flights are physically distinct, so their mean is not a trajectory anything flew.
            # Draw every flight as a bundle: colour and style encode the model, never the flight.
            gt_curves = gt_values[..., column]
            for index in range(gt_curves.shape[0]):
                axis.plot(time_axis, gt_curves[index], "k-", lw=1.1, alpha=0.55,
                          label="GT" if index == 0 else None)
            _draw_sample_bands(axis, time_axis, sample_bands, row, column)
            for label, color, linestyle, blocks in prediction_blocks:
                values = blocks[row][0][..., column]
                for index in range(values.shape[0]):
                    axis.plot(time_axis, values[index], color=color, ls=linestyle, lw=1.0,
                              alpha=0.55, label=label if index == 0 else None)
            axis.set_ylabel(labels[column])
            axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small", ncol=2)
    band_text = "" if not sample_bands else f"; shaded: ±2σ over {sample_bands[0][2].shape[0]} posterior weight-sample rollouts, flight 0"
    figure.suptitle(f"State trajectories — {truth.shape[0]} distinct held-out flights, one colour per model{band_text}",
                    fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def state_single_figure(series: list, truth: np.ndarray, time_axis: np.ndarray, sample_bands: list | None = None) -> plt.Figure:
    figure, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
    truth_blocks = _state_values(truth[:1])
    prediction_blocks = [(label, color, ls, _state_values(prediction[:1])) for label, color, ls, prediction in series]
    for row, (gt_values, labels) in enumerate(truth_blocks):
        for column in range(3):
            axis = axes[row, column]
            axis.plot(time_axis, gt_values[0, :, column], "k-", lw=2, label="GT")
            _draw_sample_bands(axis, time_axis, sample_bands, row, column)
            for label, color, linestyle, blocks in prediction_blocks:
                values = blocks[row][0]
                axis.plot(time_axis, values[0, :, column], color=color, ls=linestyle, lw=1.4, label=label)
            axis.set_ylabel(labels[column])
            axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small", ncol=2)
    band_text = "" if not sample_bands else f" — line: posterior mean; shaded: ±2σ over {sample_bands[0][2].shape[0]} posterior samples"
    figure.suptitle(f"Single trajectory (traj 0) state{band_text}", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def phase_figure(series: list, truth: np.ndarray) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    gt_euler = euler_angles(truth[:1])[0]
    predictions = [(label, color, ls, prediction, euler_angles(prediction[:1])[0]) for label, color, ls, prediction in series]
    for column in range(3):
        axes[0, column].plot(truth[0, :, column], truth[0, :, 12 + column], "k-", lw=2, label="GT")
        axes[1, column].plot(gt_euler[:, column], truth[0, :, 15 + column], "k-", lw=2, label="GT")
        for label, color, linestyle, prediction, pred_euler in predictions:
            axes[0, column].plot(prediction[0, :, column], prediction[0, :, 12 + column], color=color, ls=linestyle, lw=1.4, label=label)
            axes[1, column].plot(pred_euler[:, column], prediction[0, :, 15 + column], color=color, ls=linestyle, lw=1.4, label=label)
        axes[0, column].set_xlabel(f"position {column}")
        axes[0, column].set_ylabel(f"velocity {column}")
        axes[1, column].set_xlabel(("roll", "pitch", "yaw")[column])
        axes[1, column].set_ylabel(f"omega {column}")
        axes[0, column].grid(True, alpha=0.3)
        axes[1, column].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize="x-small")
    figure.suptitle("Phase portraits — single trajectory (traj 0)", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


# ---------------------------------------------------------------------------
# Pages 24-35: subnetwork trajectories
# ---------------------------------------------------------------------------
def gauge_scale_text(fits: dict) -> str:
    return f"GP Lie: beta_v={fits['beta_v']:.3g}, beta_omega={fits['beta_w']:.3g}"


def matrix_trajectory_figure(label, values, targets, fits, time_axis, key, title, *, gauge_fixed, note=None) -> plt.Figure:
    figure, axes = plt.subplots(3, 3, figsize=(14, 14), sharex=True)
    target = targets[key][0]
    for index, axis in enumerate(axes.flat):
        row, column = divmod(index, 3)
        axis.plot(time_axis, target[:, row, column], "k:", lw=1.6, label="GT")
        axis.plot(time_axis, values[key][0, :, row, column], color=MODEL_COLOR, ls=MODEL_LINESTYLE, lw=1.2, label=label)
        axis.set_title(f"({row},{column})")
        axis.grid(True, alpha=0.3)
        if row == 2:
            axis.set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small")
    mode = "mass-gauge fixed" if gauge_fixed else "RAW — no gauge/scale correction"
    details = gauge_scale_text(fits) if gauge_fixed else "network outputs exactly as saved"
    if note:
        details += "\n" + note
    figure.suptitle(f"{title} along shared-PID trajectory — {mode}\n{details}", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def control_trajectory_figure(label, values, targets, fits, time_axis, *, gauge_fixed) -> plt.Figure:
    figure, axes = plt.subplots(6, 4, figsize=(16, 16), sharex=True)
    target = targets["control"][0]
    for row in range(6):
        for column in range(4):
            axis = axes[row, column]
            axis.plot(time_axis, target[:, row, column], "k:", lw=1.6, label="GT")
            axis.plot(time_axis, values["control"][0, :, row, column], color=MODEL_COLOR, ls=MODEL_LINESTYLE, lw=1.1, label=label)
            axis.set_title(f"g[{row},{column}]", fontsize=9)
            axis.grid(True, alpha=0.3)
            if row == 5:
                axis.set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="xx-small")
    mode = "mass-gauge fixed: ĝ=S⁻¹g" if gauge_fixed else "RAW — no gauge/scale correction"
    details = gauge_scale_text(fits) if gauge_fixed else "network outputs exactly as saved"
    figure.suptitle(f"Control map trajectory g(q(t)) — {mode}\n{details}", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def potential_trajectory_figure(label, values, targets, fits, time_axis, *, gauge_fixed) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(time_axis, targets["potential"][0], "k:", lw=1.8, label="GT")
    axis.plot(time_axis, values["potential"][0], color=MODEL_COLOR, ls=MODEL_LINESTYLE, lw=1.4, label=label)
    mode = "mass-gauge fixed: V̂=V/βv+c" if gauge_fixed else "RAW — no gauge/scale correction"
    details = gauge_scale_text(fits) if gauge_fixed else "network outputs exactly as saved"
    axis.set_title(f"Potential V(q(t)) along shared-PID trajectory — {mode}\n{details}")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Potential energy (J)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Pages 36-39: computation, controller table, uncertainty
# ---------------------------------------------------------------------------
def compute_figure(series: list, selected_step: int, training_noise: float, device_text: str) -> plt.Figure:
    """``series``: list of (label, color, linestyle, benchmarks)."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    labels = [item[0] for item in series]
    colors = [item[1] for item in series]
    axes[0].bar(labels, [item[3]["median_seconds"] for item in series], color=colors)
    axes[0].set_ylabel("Wall time (s)")
    axes[0].set_title("Median open-loop rollout time")
    axes[1].bar(labels, [item[3]["nfe_per_transition"] for item in series], color=colors)
    axes[1].set_ylabel("Full vector-field evaluations")
    axes[1].set_title("NFE per physical transition")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle(
        f"Same JAX framework, batch, horizon and {device_text}; {checkpoint_step_text(selected_step)}\n"
        f"Training observation noise={training_noise}; clean evaluation",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def controller_table(label: str, controller: dict, comparison: dict, selected_step: int, training_noise: float) -> plt.Figure:
    rows = [
        "shared_horizon_seconds",
        "shared_position_rmse_m",
        "shared_position_final_m",
        "shared_velocity_rmse_m_per_s",
        "completed_duration_seconds",
        "completion_fraction",
        "controller_failure",
        "failure_time_seconds",
        "maximum_contact_points",
        "maximum_requested_rpm",
        "motor_saturation_fraction",
        "controller_time_median_ms",
    ]
    full = controller["metrics"]
    cells = []
    for row in rows:
        if row == "shared_horizon_seconds":
            value = comparison["shared_horizon_seconds"]
        elif row.startswith("shared_"):
            value = comparison[row.removeprefix("shared_")]
        elif row in comparison:
            value = comparison[row]
        elif row == "failure_time_seconds":
            value = comparison["failure"]["time_seconds"] if comparison["failure"] else None
        else:
            value = full[row]
        cells.append([format_cell(value)])
    return table_page(
        "Failure-aware controller results on contact-free PyBullet",
        rows,
        [label],
        cells,
        f"Checkpoint protocol: {checkpoint_step_text(selected_step)}. "
        f"Training observation noise={training_noise}. Direct tracking metrics "
        "use the common measured horizon; failed curves are truncated, never padded.",
    )


def gp_uncertainty_figure(standard_deviations: dict[str, np.ndarray]) -> plt.Figure:
    if not standard_deviations:
        # Point-estimate network: the page is kept so the report length matches the GP reports.
        figure, axis = plt.subplots(figsize=(11, 7.5))
        axis.axis("off")
        axis.text(0.5, 0.5, "Point-estimate network: no weight posterior.\n\nThis page shows the variational posterior "
                            "weight uncertainty for GP models;\nan MLP has deterministic weights, so there is no\n"
                            "epistemic uncertainty to display and the state-trajectory\npages carry no posterior bands.",
                  ha="center", va="center", fontsize=13)
        figure.suptitle("Weight-space uncertainty - not applicable", fontweight="bold")
        figure.tight_layout()
        return figure
    figure, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for axis, name in zip(axes.ravel(), ("M1", "M2", "Dv", "Dw", "V", "g")):
        values = standard_deviations[name]
        axis.hist(values, bins=35, color=MODEL_COLOR, alpha=0.85)
        axis.axvline(np.median(values), color="black", ls="--", lw=1)
        axis.set_title(f"{name}: median={np.median(values):.3g}")
        axis.set_xlabel("posterior weight standard deviation")
        axis.set_ylabel("count")
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "PH-GP-LieIMEX variational posterior uncertainty\n"
        "Weight-space uncertainty; deterministic report rollouts use posterior means",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


# ---------------------------------------------------------------------------
# Multi-model variants: identical layout, one curve or column per trained model.
# Entries are (label, colour, linestyle, payload) unless stated otherwise.
# ---------------------------------------------------------------------------
# Up to six models: the same three hues twice. Columns 1-3 solid, columns 4-6 the original dashed styles, so in a
# priors-on / priors-off report colour encodes the model family and line style the prior.
COMPARISON_STYLES = (("#CC79A7", "-"), ("#009E73", "-"), ("#D55E00", "-"),
                     ("#CC79A7", ":"), ("#009E73", "-."), ("#D55E00", (0, (3, 1, 1, 1))),
                     ("#0072B2", "-"))   # 7th: the analytic ground-truth entry


def training_figure_multi(entries, title: str, ylabel: str, *, test: bool) -> plt.Figure:
    """entries: (label, colour, linestyle, steps, values)."""
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.set_title(title); axis.set_xlabel("step"); axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3); axis.set_yscale("log")
    for label, colour, style, steps, values in entries:
        values = np.asarray(values, dtype=float)
        steps = np.asarray(steps, dtype=int)[: len(values)]
        if not test:
            values = smooth(values)
        axis.plot(steps, np.maximum(values, 1e-14), color=colour, ls=style, lw=1.4, label=label)
    axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def _table_multi(title: str, rows, columns, cells, note: str, scale=(1.0, 1.6), fontsize=7.2, widths=None) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(13.5, 8.0))
    axis.axis("off")
    axis.set_title(title, fontsize=10.5, fontweight="bold", pad=12)
    table = axis.table(cellText=cells, rowLabels=rows, colLabels=columns, colWidths=widths,
                       cellLoc="center", rowLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(fontsize); table.scale(*scale)
    for index in range(len(columns)):
        table[0, index].set_text_props(weight="bold")
    axis.text(0.5, 0.02, note, transform=axis.transAxes, ha="center", va="bottom", fontsize=7.5, wrap=True)
    figure.tight_layout()
    return figure


def subnetwork_summary_multi(entries, horizon: float) -> plt.Figure:
    """entries: (label, metrics dict). One column per model, rows are the per-subnetwork errors."""
    names, columns = [], []
    for key in entries[0][1]:
        name = key.split(" MSE — ")[0] if " MSE — " in key else None
        if name and name not in names:
            names.append(name)
    rows = [f"{n} — {c}" for n in names for c in ("NMSE gauge-fixed", "rel. RMS error (%)")]
    cells = []
    for name in names:
        for column in ("NMSE — gauge-fixed", "relative RMS error (%) — gauge-fixed"):
            row = []
            for label, metrics in entries:
                values = metrics.get(f"{name} {column}")
                row.append("–" if values is None else
                           (f"{np.mean(values):.2f}" if column.startswith("relative") else ms(values)))
            cells.append(row)
    columns = [label.replace("\n", " ") for label, _ in entries]
    return _table_multi(
        f"Subnetwork error against analytical ground truth — horizon {horizon:.3f}s\nmass-gauge-fixed, unit-free",
        rows, columns, cells,
        "NMSE = MSE / mean of GT² over the same horizon (0 = perfect, 1 = as bad as predicting zero); "
        "the second row of each block is 100·√NMSE. Raw NMSE near 1 would reflect the port-Hamiltonian gauge, "
        "which the gauge fix removes with one scale per mass block.",
    )


def product_summary_multi(entries, product_labels, horizon: float) -> plt.Figure:
    """entries: (label, metrics dict) from evaluation.product_metrics."""
    columns = [label.replace("\n", " ") for label, _ in entries]
    rows, cells = [], []
    for key, name in product_labels:
        rows.append(f"{name} — per axis")
        cells.append([", ".join("–" if not np.isfinite(v) else f"{v:.4g}" for v in m[key]["model_axis_mean"]) for _, m in entries])
        rows.append(f"{name} — rel. RMS error (%)")
        cells.append([f"{np.mean(m[key]['relative_rms_error_percent']):.1f} ± "
                      f"{np.std(m[key]['relative_rms_error_percent']):.1f}" for _, m in entries])
        # Paired per-flight margin against the first column.  Flight-to-flight variation is large and
        # common to both models, so it cancels in the difference; a gap that survives pairing is real
        # even when the two independent spreads overlap.
        rows.append(f"{name} — paired margin vs column 1")
        reference = np.asarray(entries[0][1][key]["relative_rms_error_percent"], dtype=float)
        paired = ["reference"]
        for _, other_metrics in entries[1:]:
            other = np.asarray(other_metrics[key]["relative_rms_error_percent"], dtype=float)
            delta = other - reference  # positive: column 1 is the better model on that flight
            wins = int(np.sum(delta > 0.0))
            paired.append(f"{delta.mean():+.1f} ± {delta.std():.1f} pp, col1 better {wins}/{delta.size}")
        cells.append(paired)
    truth_axis = entries[0][1][product_labels[0][0]]["truth_axis_mean"]
    rows.append("translational damping μ·Dv by speed (ratio model/truth)")
    cells.append([", ".join("–" if not b["count"] else f"{b['model']/b['truth']:.1f}" for b in m["damping_v_speed_bins"]) for _, m in entries])
    return _table_multi(
        f"Gauge-invariant products against analytical ground truth — horizon {horizon:.3f}s\n"
        "no scale fit is applied; these products are what the trajectories identify",
        rows, columns, cells,
        "Per-axis rows show the mean diagonal (3×3 products) or vector components over the horizon. "
        "Relative RMS error is 100·√NMSE: 0 is exact, 100 is as bad as predicting zero, above 100 is worse "
        "than no model. Its ± is the spread across the distinct held-out flights. The paired row scores both "
        "models on the same flight and differences the results, which cancels the flight-to-flight variation "
        "common to both; a margin whose sign holds on most flights is a real gap even when the ± ranges overlap. "
        "Truth: thrust gain (0, 0, 1/m), torque gain J⁻¹, gravity (0, 0, g), and damping ratio 1 in every speed bin. "
        f"Speed bins are {', '.join(f'[{b['low']:.2f},{b['high']:.2f})' for b in entries[0][1]['damping_v_speed_bins'])} m/s.",
    )


def product_trajectory_multi(entries, targets, time_axis, key, title, units) -> plt.Figure:
    """entries: (label, colour, linestyle, products dict)."""
    target = targets[key]
    flights = target.shape[0]
    if target.ndim == 4:
        figure, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True)
        for index, axis in enumerate(axes.flat):
            row, column = divmod(index, 3)
            for flight in range(flights):
                axis.plot(time_axis, target[flight, :, row, column], "k:", lw=1.3, alpha=0.6,
                          label="GT" if flight == 0 else None)
            for label, colour, style, products in entries:
                for flight in range(flights):
                    axis.plot(time_axis, products[key][flight, :, row, column], color=colour, ls=style,
                              lw=1.0, alpha=0.55, label=label if flight == 0 else None)
            axis.set_title(f"({row},{column})"); axis.grid(True, alpha=0.3)
            if row == 2:
                axis.set_xlabel("Time (s)")
        axes[0, 0].legend(fontsize="xx-small")
    else:
        figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for index, axis in enumerate(axes):
            for flight in range(flights):
                axis.plot(time_axis, target[flight, :, index], "k:", lw=1.3, alpha=0.6,
                          label="GT" if flight == 0 else None)
            for label, colour, style, products in entries:
                for flight in range(flights):
                    axis.plot(time_axis, products[key][flight, :, index], color=colour, ls=style,
                              lw=1.0, alpha=0.55, label=label if flight == 0 else None)
            axis.set_title(("x", "y", "z")[index] + " component"); axis.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)"); axes[0].legend(fontsize="xx-small")
    figure.suptitle(f"{title} — gauge-invariant, no scale fit\n"
                    f"{flights} distinct held-out flights; colour and style encode the model, never the flight\n"
                    f"units: {units}",
                    fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def damping_speed_multi(entries, targets, truth, vehicle) -> plt.Figure:
    """entries: (label, colour, linestyle, products dict)."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    c = vehicle["damping"]
    linear_law = vehicle.get("damping_law", "nonlinear") == "linear"
    for axis, key, slice_, name in ((axes[0], "damping_v", slice(12, 15), "|v_body| (m/s)"),
                                    (axes[1], "damping_w", slice(15, 18), "|ω_body| (rad/s)")):
        magnitude = np.linalg.norm(truth[..., slice_], axis=-1).reshape(-1)
        for label, colour, _style, products in entries:
            diag = np.stack([products[key][..., i, i] for i in range(3)], -1).reshape(-1, 3).mean(-1)
            axis.scatter(magnitude, diag, s=4, alpha=0.35, color=colour, label=label)
        grid = np.linspace(0, max(magnitude.max(), 1e-3), 200)
        axis.plot(grid, np.full_like(grid, c) if linear_law else c * (1.0 + grid), "k:", lw=2.0,
                  label="GT  c (linear law)" if linear_law else "GT  c(1+|·|)")
        axis.set_xlabel(name); axis.set_ylabel("product diagonal, mean of diag (1/s)")
        axis.grid(True, alpha=0.3); axis.legend(fontsize="x-small")
    axes[0].set_title("translational damping μ·Dv vs speed"); axes[1].set_title("rotational damping M2⁻¹·Dω vs rate")
    figure.suptitle("Learned damping products against the true speed-dependent law", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def matrix_trajectory_multi(entries, targets, time_axis, key, title, *, gauge_fixed, note=None) -> plt.Figure:
    """entries: (label, colour, linestyle, values dict, fits dict)."""
    figure, axes = plt.subplots(3, 3, figsize=(14, 14), sharex=True)
    target = targets[key][0]
    for index, axis in enumerate(axes.flat):
        row, column = divmod(index, 3)
        axis.plot(time_axis, target[:, row, column], "k:", lw=1.6, label="GT")
        for label, colour, style, values, _fits in entries:
            axis.plot(time_axis, values[key][0, :, row, column], color=colour, ls=style, lw=1.2, label=label)
        axis.set_title(f"({row},{column})"); axis.grid(True, alpha=0.3)
        if row == 2:
            axis.set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="xx-small")
    mode = "mass-gauge fixed" if gauge_fixed else "RAW — no gauge/scale correction"
    details = " | ".join(f"{label.splitlines()[0]}: {gauge_scale_text(fits)}" for label, _c, _s, _v, fits in entries) if gauge_fixed \
        else "network outputs exactly as saved"
    if note:
        details += "\n" + note
    figure.suptitle(f"{title} along shared-PID trajectory — {mode}\n{details}", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def control_trajectory_multi(entries, targets, time_axis, *, gauge_fixed) -> plt.Figure:
    figure, axes = plt.subplots(6, 4, figsize=(16, 16), sharex=True)
    target = targets["control"][0]
    for row in range(6):
        for column in range(4):
            axis = axes[row, column]
            axis.plot(time_axis, target[:, row, column], "k:", lw=1.6, label="GT")
            for label, colour, style, values, _fits in entries:
                axis.plot(time_axis, values["control"][0, :, row, column], color=colour, ls=style, lw=1.1, label=label)
            axis.set_title(f"g[{row},{column}]", fontsize=9); axis.grid(True, alpha=0.3)
            if row == 5:
                axis.set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="xx-small")
    mode = "mass-gauge fixed: ĝ=S⁻¹g" if gauge_fixed else "RAW — no gauge/scale correction"
    details = " | ".join(f"{label.splitlines()[0]}: {gauge_scale_text(fits)}" for label, _c, _s, _v, fits in entries) if gauge_fixed \
        else "network outputs exactly as saved"
    figure.suptitle(f"Control map trajectory g(q(t)) — {mode}\n{details}", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def potential_trajectory_multi(entries, targets, time_axis, *, gauge_fixed) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(time_axis, targets["potential"][0], "k:", lw=1.8, label="GT")
    for label, colour, style, values, _fits in entries:
        axis.plot(time_axis, values["potential"][0], color=colour, ls=style, lw=1.4, label=label)
    mode = "mass-gauge fixed: V̂=V/βv+c" if gauge_fixed else "RAW — no gauge/scale correction"
    details = " | ".join(f"{label.splitlines()[0]}: {gauge_scale_text(fits)}" for label, _c, _s, _v, fits in entries) if gauge_fixed \
        else "network outputs exactly as saved"
    axis.set_title(f"Potential V(q(t)) along shared-PID trajectory — {mode}\n{details}")
    axis.set_xlabel("Time (s)"); axis.set_ylabel("Potential energy (J)")
    axis.grid(True, alpha=0.3); axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def controller_table_multi(entries, training_noise: float) -> plt.Figure:
    """entries: (label, controller metadata dict, comparison dict, selected_step)."""
    rows = ["shared_horizon_seconds", "shared_position_rmse_m", "shared_position_final_m",
            "shared_velocity_rmse_m_per_s", "completed_duration_seconds", "completion_fraction",
            "controller_failure", "failure_time_seconds", "maximum_contact_points",
            "maximum_requested_rpm", "motor_saturation_fraction", "controller_time_median_ms"]
    cells = []
    for row in rows:
        line = []
        for _label, controller, comparison, _step in entries:
            full = controller["metrics"]
            if row == "shared_horizon_seconds":
                value = comparison["shared_horizon_seconds"]
            elif row.startswith("shared_"):
                value = comparison[row.removeprefix("shared_")]
            elif row in comparison:
                value = comparison[row]
            elif row == "failure_time_seconds":
                value = comparison["failure"]["time_seconds"] if comparison["failure"] else None
            else:
                value = full[row]
            line.append(format_cell(value))
        cells.append(line)
    return _table_multi(
        "Failure-aware controller results on contact-free PyBullet",
        rows, [label.replace("\n", " ") for label, _c, _cmp, _s in entries], cells,
        f"Training observation noise={training_noise}. Direct tracking metrics use the common measured horizon; "
        "failed curves are truncated, never padded. Both models fly the same 20 s diamond with the same reference law.",
        scale=(1.0, 1.5), fontsize=8,
    )

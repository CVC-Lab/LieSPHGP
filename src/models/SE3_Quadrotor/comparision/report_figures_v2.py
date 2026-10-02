"""Figures for the spec-driven comparison report.

Implements comparision_report_format.md.  Conventions, which differ per page on purpose:

* pages 2-6, 11, 12  colour AND line style encode the model; every held-out flight is drawn.
* pages 7, 8, 16     colour encodes the model, line style encodes the loss or error block.
* page 9             one model only, so colour encodes the objective term.
"""
from __future__ import annotations

from typing import Any

import math

import matplotlib.pyplot as plt

import numpy as np

from .report_figures import (
    attitude_error_squared,
    euler_angles,
    format_cell,
    ground_truth_excursions,
    physical_energy,
    smooth,
)

# Five distinguishable styles: the practical limit before they stop separating in print.
COMPONENT_STYLES = (
    ("total", "-"),
    ("position", "--"),
    ("attitude", "-."),
    ("linear velocity", (0, (3, 1, 1, 1))),
    ("angular velocity", ":"),
)
ERROR_STYLES = (
    ("position", "-"),
    ("attitude", "--"),
    ("linear velocity", "-."),
    ("angular velocity", ":"),
)
TERM_COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")


def _table(title: str, rows, columns, cells, note: str, *, fontsize=7.6, scale=(1.0, 1.6),
           highlight=None) -> plt.Figure:
    """``highlight``: {row index: [column indices]} whose cells are set in bold — the best model(s) in that row."""
    # Wide tables (four or more model columns) get a wider page and a smaller header font, and the column
    # labels are re-wrapped to three lines, so six model names no longer spill across each other.
    wide = len(columns) >= 4
    if wide:
        def rewrap(label: str) -> str:
            first, _, second = label.partition("\n")
            return f"{first}\n{second.replace(', ', chr(10), 1)}" if second else first
        columns = [rewrap(c) for c in columns]
    # The page grows with the table: at sixteen or more rows a fixed 8.5 in page puts the title on the header
    # row and the caption on the last rows.
    header_rows = 1.7
    note_lines = (note.count("\n") + 1) if note else 0
    height = max(9.0, 2.6 + 0.46 * (len(rows) + header_rows) + 0.16 * note_lines)
    figure, axis = plt.subplots(figsize=(19 if wide else 14, height))
    axis.axis("off")
    axis.set_title(title, fontsize=10.5, fontweight="bold", pad=16)
    table = axis.table(cellText=cells, rowLabels=rows, colLabels=columns, cellLoc="center",
                       rowLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(min(fontsize, 7.2) if wide else fontsize)
    table.scale(scale[0], 1.0)
    # Explicit cell heights rather than a scale factor: the table then fills the axes whatever the row count.
    row_height = 0.92 / (len(rows) + header_rows)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_height(row_height * (header_rows if row_index == 0 else 1.0))
    for index in range(len(columns)):
        table[0, index].set_text_props(weight="bold", fontsize=6.6 if wide else None)
    for row_index, column_indices in (highlight or {}).items():
        for column_index in column_indices:
            table[row_index + 1, column_index].set_text_props(weight="bold")   # +1: row 0 holds the column labels
    if note:
        # Explicit line breaks, not wrap=True: wrapping is done against the whole figure width, which
        # pushes the caption out to both page edges.
        figure.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=8.0)
    figure.tight_layout(rect=(0.02, 0.02 + 0.018 * note_lines, 0.98, 0.98))
    return figure


# --- page 1 ---------------------------------------------------------------------
def product_table(entries, product_labels, horizon: float, flight_name: str) -> plt.Figure:
    """entries: (label, product_metrics).  One relative-RMS row and one paired row per product."""
    columns = [label for label, _ in entries]   # keep the newline: long names collide on one line
    # The paired row is named after the reference model, not after its column position.
    reference_name = entries[0][0].replace("\n", " ").split()[0]
    rows, cells = [], []
    for key, name in product_labels:
        short = name.split("(")[0].strip()
        rows.append(f"{short} — rel. RMS error (%)")
        cells.append([f"{np.mean(m[key]['relative_rms_error_percent']):.1f} ± "
                      f"{np.std(m[key]['relative_rms_error_percent']):.1f}" for _, m in entries])
        rows.append(f"{short} — vs {reference_name}")
        reference = np.asarray(entries[0][1][key]["relative_rms_error_percent"], dtype=float)
        paired = ["—"]
        for _, other in entries[1:]:
            delta = np.asarray(other[key]["relative_rms_error_percent"], dtype=float) - reference
            paired.append(f"{delta.mean():+.1f} ± {delta.std():.1f} pp")
        cells.append(paired)
    return _table(
        f"Gauge-invariant physics identification — relative RMS error (%)\n"
        f"{flight_name}, horizon {horizon:.2f} s, {reference.size} distinct held-out flights; no scale fit",
        rows, columns, cells,
        "0 = exact.   100 = as bad as predicting zero.   Above 100 = worse than no model.\n"
        f"± = spread across the {reference.size} held-out flights.\n"
        f"“vs {reference_name}” = the same-flight difference, in pp (percentage points); positive = that "
        "column is worse.",
        fontsize=8.2, scale=(1.0, 1.9),
    )


# --- pages 2-6 ------------------------------------------------------------------
def product_page(entries, targets, time_axis, key, title, truth_text, units, symbol,
                 metrics, bands=None, flight_name: str = "", horizon: float = 0.0) -> plt.Figure:
    """entries: (label, colour, style, products).  bands: [(label, colour, samples dict)], every flight.

    ``samples[key]`` has shape (posterior samples, flights, time, ...): each flight gets its own
    +-2 sigma band, so the band is a property of the flight it is drawn on, not of flight 0 alone.
    """
    target = targets[key]
    flights = target.shape[0]
    matrix = target.ndim == 4
    if matrix:
        figure, axes = plt.subplots(3, 3, figsize=(15, 13), sharex=True)
        panels = [(divmod(i, 3), axis) for i, axis in enumerate(axes.flat)]
    else:
        figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        panels = [((i, None), axis) for i, axis in enumerate(axes)]

    diagonal_mean = float(np.mean([np.abs(target[..., i, i]).mean() for i in range(3)])) if matrix else 0.0
    for panel_index, ((row, column), axis) in enumerate(panels):
        first_panel = panel_index == 0
        def pick(array):
            return array[..., row, column] if matrix else array[..., row]
        for flight in range(flights):
            axis.plot(time_axis, pick(target)[flight], "k:", lw=1.3, alpha=0.65,
                      label="ground truth" if flight == 0 else None)
        for band_label, band_colour, samples in bands or []:
            values = pick(samples[key])          # (samples, flights, time)
            mean, std = values.mean(0), values.std(0)
            # One band per flight, faint enough that ten of them overlaid stay readable.
            for flight in range(mean.shape[0]):
                axis.fill_between(time_axis, mean[flight] - 2 * std[flight], mean[flight] + 2 * std[flight],
                                  color=band_colour, alpha=0.11, lw=0,
                                  label=f"{band_label.splitlines()[0]} ±2σ, per flight"
                                  if (first_panel and flight == 0) else None)
        for label, colour, style, products in entries:
            for flight in range(flights):
                axis.plot(time_axis, pick(products[key])[flight], color=colour, ls=style, lw=1.0, alpha=0.6,
                          label=label.replace("\n", " ") if flight == 0 else None)
        if matrix:
            axis.set_title(f"({row},{column})", fontsize=9)
            if row != column and diagonal_mean > 0:
                peak = max(float(np.abs(pick(products[key])).max()) for _, _, _, products in entries)
                axis.text(0.02, 0.94, f"peak {100 * peak / diagonal_mean:.1f}% of diagonal",
                          transform=axis.transAxes, fontsize=7, va="top", color="#555555")
        else:
            axis.set_title(("x", "y", "z")[row] + " component", fontsize=10)
        axis.grid(True, alpha=0.3)
        # Name the quantity on the y axis the way the x axis names time.  No entry index here:
        # the panel title above the plot already says which entry this is.
        axis.set_ylabel(f"{symbol}   ({units})", fontsize=8)
    for axis in (axes.flat if matrix else axes):
        axis.set_xlabel("Time (s)", fontsize=8)

    summary = "   |   ".join(
        f"{label.splitlines()[0]}: {np.mean(metrics[label][key]['relative_rms_error_percent']):.1f}%"
        for label, _c, _s, _p in entries)
    figure.text(0.015, 0.012, f"relative RMS error — {summary}", fontsize=8.5, va="bottom")
    handles, labels = (axes.flat[0] if matrix else axes[0]).get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.905),
                  fontsize=7.5, ncol=1, framealpha=0.9)
    figure.suptitle(f"{title}   —   truth: {truth_text}   |   units: {units}\n"
                    f"{flight_name}, horizon {horizon:.2f} s, {flights} distinct held-out flights\n"
                    "colour and style encode the model, never the flight",
                    fontsize=11.5, fontweight="bold")
    figure.tight_layout(rect=(0, 0.03, 1, 0.90))
    return figure


# --- pages 7 and 8: the paper view of the same five products ---------------------
def _binned(x: np.ndarray, values: np.ndarray, edges: np.ndarray):
    """Mean and standard deviation of ``values`` inside each bin of ``x``; NaN where a bin is empty."""
    index = np.clip(np.digitize(x, edges) - 1, 0, edges.size - 2)
    mean = np.full(edges.size - 1, np.nan)
    deviation = np.full(edges.size - 1, np.nan)
    count = np.zeros(edges.size - 1, dtype=int)
    for b in range(edges.size - 1):
        mask = index == b
        count[b] = int(mask.sum())
        if count[b]:
            mean[b], deviation[b] = values[mask].mean(), values[mask].std()
    return mean, deviation, count


def _diagonal_mean(array: np.ndarray) -> np.ndarray:
    """Average of the three diagonal entries of a (..., 3, 3) field, flattened."""
    return np.stack([array[..., i, i] for i in range(3)], axis=-1).mean(-1).reshape(-1)


def damping_law_page(entries, targets, truth, bands=None, bin_count: int = 18,
                     flight_name: str = "", horizon: float = 0.0) -> plt.Figure:
    """The two state-dependent products against the state they depend on, pooled over every flight.

    Damping is a function of the body velocity and of the body rate, so time is the wrong x axis:
    two flights at the same instant sit at different speeds and therefore have different *true*
    damping.  Pooling every flight at a given speed is the legitimate pooling, and the shaded
    spread is then the model's own anisotropy and error, never the diversity of the test set.
    """
    speed = np.linalg.norm(truth[..., 12:15], axis=-1).reshape(-1)
    rate = np.linalg.norm(truth[..., 15:18], axis=-1).reshape(-1)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    panels = (
        ("damping_v", speed, "‖v_b‖   (m/s)", "μ·D_v  diagonal mean   (1/s)",
         "Translational damping against body speed"),
        ("damping_w", rate, "‖ω_b‖   (rad/s)", "M₂⁻¹·D_ω  diagonal mean   (1/s)",
         "Rotational damping against body rate"),
    )
    for axis, (key, x, xlabel, ylabel, title) in zip(axes, panels):
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bin_count + 1)))
        centres = 0.5 * (edges[:-1] + edges[1:])
        truth_mean, _dev, count = _binned(x, targets[key][..., 0, 0].reshape(-1), edges)
        axis.plot(centres, truth_mean, "k:", lw=2.2, label="ground truth  c(1+‖·‖)")
        for label, colour, style, products in entries:
            mean, deviation, _n = _binned(x, _diagonal_mean(products[key]), edges)
            axis.plot(centres, mean, color=colour, ls=style, lw=1.9, label=label.replace("\n", " "))
            axis.fill_between(centres, mean - deviation, mean + deviation, color=colour, alpha=0.15, lw=0)
        for band_label, band_colour, samples in bands or []:
            # Epistemic spread: bin each posterior sample separately, then take the spread across samples.
            per_sample = np.stack([_binned(x, _diagonal_mean(samples[key][s]), edges)[0]
                                   for s in range(samples[key].shape[0])], axis=0)
            centre, width = per_sample.mean(0), per_sample.std(0)
            axis.plot(centres, centre + 2 * width, color=band_colour, ls=(0, (1, 1)), lw=1.2,
                      label=f"{band_label.splitlines()[0]} ±2σ posterior")
            axis.plot(centres, centre - 2 * width, color=band_colour, ls=(0, (1, 1)), lw=1.2)
        axis.set_xlabel(xlabel, fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.set_title(title, fontsize=11)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        counts = axis.twinx()
        counts.step(centres, count, where="mid", color="#999999", lw=0.9, alpha=0.7)
        counts.set_ylabel("samples per bin", fontsize=7.5, color="#777777")
        counts.tick_params(labelsize=7, colors="#777777")
        counts.set_ylim(0, count.max() * 4.0)      # keep the histogram low, out of the curves
    figure.suptitle(
        "Damping laws against the state they depend on — the paper view of pages 5 and 6\n"
        f"{flight_name}, horizon {horizon:.2f} s, {truth.shape[0]} distinct held-out flights pooled, "
        f"{bin_count} equal-count bins",
        fontsize=12, fontweight="bold")
    figure.text(0.015, 0.012,
                "Shaded: ±1σ of the model inside the bin — its anisotropy plus its error, since the truth is "
                "isotropic and identical for every sample in a bin.\n"
                "Dotted: ±2σ across posterior weight samples, the epistemic band, plotted separately so the two "
                "spreads are never read as one.\n"
                "Grey step: how many samples fall in each bin — most flight time is spent slow, which is where "
                "the models are worst.",
                fontsize=8.5, va="bottom")
    figure.tight_layout(rect=(0, 0.09, 1, 0.88))
    return figure


# --- closing page of the identification booklet ----------------------------------
def metric_definition_page(metrics, reference_label: str, flight_count: int) -> plt.Figure:
    """How the relative RMS error on page 1 is computed, with a worked example from this report."""
    thrust = metrics[reference_label]["thrust_gain"]
    truth_z = float(np.asarray(thrust["truth_axis_mean"])[2])
    model_z = float(np.asarray(thrust["model_axis_mean"])[2])
    reported = float(np.mean(thrust["relative_rms_error_percent"]))
    hand = 100.0 * abs(model_z - truth_z) / abs(truth_z)
    short = reference_label.replace("\n", " ").split()[0]

    figure = plt.figure(figsize=(14, 9.5))
    figure.suptitle("How the relative RMS error is computed", fontsize=15, fontweight="bold", y=0.965)

    blocks = [
        (0.905, "1.  What is compared", 12, "bold"),
        (0.868, "At every time step of every held-out flight, the learned product $P$ is compared with the "
                "analytic truth $P^{\\star}$ built from the", 10.5, "normal"),
        (0.840, "simulator's own constants.  Nothing is rescaled first: these five products are gauge-free, so "
                "no fitted factor stands between them.", 10.5, "normal"),

        (0.783, "2.  The formula, one number per flight", 12, "bold"),
        (0.685, r"$\mathrm{relRMS}_k \;=\; 100 \times \sqrt{\dfrac{\dfrac{1}{NC}\sum_{t}\sum_{c}"
                r"\left(P_{tc}-P^{\star}_{tc}\right)^{2}}{\dfrac{1}{NC}\sum_{t}\sum_{c}"
                r"\left(P^{\star}_{tc}\right)^{2}}}$", 17, "normal"),
        (0.590, "$k$ is the flight, $t$ runs over its $N$ time steps and $c$ over the $C$ components of the "
                "product — 3 for a vector, 9 for a matrix.", 10.5, "normal"),

        (0.535, "3.  The same thing in plain words", 12, "bold"),
        (0.498, "•  Subtract truth from model, square it, average over every instant and every component.  "
                "That is the mean squared error on top.", 10.5, "normal"),
        (0.468, "•  Do the same to the truth on its own.  That is how big the quantity is in the first place — "
                "the bottom.", 10.5, "normal"),
        (0.438, "•  Divide.  The units cancel, so the answer is a pure number that can be compared across "
                "products of wildly different size.", 10.5, "normal"),
        (0.408, "•  Take the square root to return to the original units, and multiply by 100 to read it as a "
                "percentage.", 10.5, "normal"),

        (0.350, "4.  How to read the number", 12, "bold"),
        (0.313, "0 = exact.   100 = the error is as large as the quantity itself, so the model is no better than "
                "predicting zero.   Above 100 = worse than", 10.5, "normal"),
        (0.283, "having no model at all.  Lower is always better.", 10.5, "normal"),

        (0.225, "5.  Worked example — thrust gain, %s" % short, 12, "bold"),
        (0.188, "Only the $z$ entry of $\\mu g_f$ is non-zero, so the $\\frac{1}{NC}$ cancels top and bottom and "
                "the formula collapses to $100\\,|P_z-P^{\\star}_z|/|P^{\\star}_z|$:", 10.5, "normal"),
        (0.148, "truth $%.1f$,   learned $%.1f$,   difference $%.1f$   $\\Rightarrow$   "
                "$100 \\times %.1f/%.1f = %.1f\\%%$,   against $%.1f\\%%$ in the table."
                % (truth_z, model_z, model_z - truth_z, abs(model_z - truth_z), abs(truth_z), hand, reported),
         11.5, "normal"),
        (0.112, "The small gap is the part the single averaged value hides: the learned gain drifts over the "
                "flight, and the table counts that drift as error too.", 10.5, "normal"),

        (0.060, "The table reports the mean of these %d per-flight numbers plus or minus their spread.  "
                "The “vs %s” row subtracts the two models’" % (flight_count, short), 10.5, "normal"),
        (0.030, "scores on the same flight before averaging, which removes the difficulty of the flight itself "
                 "and leaves only the gap between the models.", 10.5, "normal"),
    ]
    for y, text, size, weight in blocks:
        figure.text(0.055, y, text, fontsize=size, fontweight=weight, va="center", ha="left")
    return figure


# --- closed-loop controller booklet, page 1 ---------------------------------------
def controller_table(entries, reference: str, duration: float, dissipation_modes=None,
                     show_truth_rows: bool = True) -> plt.Figure:
    """entries: (label, controller_comparison dict). Failure-aware: a model that crashed is scored on what it flew."""
    columns = [label for label, _ in entries]
    modes = list(dissipation_modes or [])
    law = ("damping feedforward +Dv v_b, +Dw omega_b ON for every model" if modes and all(m == "on" for m in modes)
           else "reference law, no damping feedforward, for every model" if modes and all(m == "off" for m in modes)
           else "damping feedforward: " + ", ".join(modes) + "  (NOT the same law for every column)")
    # The vs-PH-GT rows only exist when the analytic-operator model was flown alongside (--include-ground-truth).
    against_truth = show_truth_rows and any("position_rmse_vs_truth_m" in c for _, c in entries)
    valid_tracking = any("valid_tracking_seconds_vs_truth" in c for _, c in entries)
    # ("label", key, format, direction): "low"/"high" = best is the smallest/largest, "gt" = closest to the
    # PH-GT column (a thrust far below hover is as wrong as one far above), None = not ranked.
    rows = [
        ("position RMSE (m)", "position_rmse_m", "{:.3f}", "low"),
        ("position max error (m)", "position_max_m", "{:.3f}", "low"),
        ("position final error (m)", "position_final_m", "{:.3f}", "low"),
        ("velocity RMSE (m/s)", "velocity_rmse_m_per_s", "{:.3f}", "low"),
        ("roll RMSE vs flat reference (deg)", "roll_rmse_deg", "{:.1f}", "gt"),
        ("pitch RMSE vs flat reference (deg)", "pitch_rmse_deg", "{:.1f}", "gt"),
        ("yaw RMSE (deg)", "yaw_rmse_deg", "{:.2f}", "low"),
        *((("valid tracking time vs PH-GT (s)", "valid_tracking_seconds_vs_truth", "{:.2f}", "high"),) if valid_tracking else ()),
        ("RMS thrust / hover", "thrust_rms_hover", "{:.3f}", "gt"),
        ("control chattering (1/s)", "control_chatter_per_s", "{:.2f}", "low"),
        ("max tilt (deg)", "max_tilt_deg", "{:.1f}", "low"),
        ("motor saturation fraction", "shared_motor_saturation_fraction", "{:.3f}", "low"),
        ("completed / requested (s)", None, None, None),
        ("completion fraction", "completion_fraction", "{:.2f}", "high"),
        ("controller failure", None, None, None),
        ("controller time p95 (ms)", "controller_time_p95_ms", "{:.2f}", "low"),
        *((("position RMSE vs PH-GT (m)", "position_rmse_vs_truth_m", "{:.3f}", "low"),
           ("velocity RMSE vs PH-GT (m/s)", "velocity_rmse_vs_truth_m_per_s", "{:.3f}", "low"),
           ("attitude error vs PH-GT (deg)", "attitude_rmse_vs_truth_deg", "{:.2f}", "low"),
           ("roll RMSE vs PH-GT (deg)", "roll_rmse_vs_truth_deg", "{:.2f}", "low"),
           ("pitch RMSE vs PH-GT (deg)", "pitch_rmse_vs_truth_deg", "{:.2f}", "low"),
           ("gap vs PH-GT (%)", "position_gap_vs_truth_percent", "{:+.0f} %", "low")) if against_truth else ()),
    ]
    # PH-GT is the reference, not a competitor: it is never the "best" cell.
    truth_column = next((index for index, (label, _) in enumerate(entries)
                         if label.replace("\n", " ").startswith("PH-GT")), None)
    learned = [index for index in range(len(entries)) if index != truth_column]
    highlight: dict[int, list[int]] = {}
    labels, cells = [], []
    for row_index, (name, key, fmt, direction) in enumerate(rows):
        labels.append(name)
        line = []
        for _, c in entries:
            if name.startswith("completed"):
                line.append(f"{c['completed_duration_seconds']:.1f} / {c['requested_duration_seconds']:.0f}")
            elif name == "controller failure":
                f = c.get("failure")
                line.append("none" if not f else f"{f.get('kind', 'failure')} at {f.get('time_seconds', float('nan')):.1f} s")
            elif key not in c:
                line.append("—")               # PH-GT itself: it is the baseline, not a distance from it
            else:
                line.append(fmt.format(c[key]))
        cells.append(line)
        if direction is None or len(learned) < 2:
            continue
        values = {index: entries[index][1][key] for index in learned if key in entries[index][1]}
        values = {index: value for index, value in values.items() if np.isfinite(value)}
        if not values:
            continue
        if direction == "high":
            best = max(values, key=values.get)
        elif direction == "gt" and truth_column is not None and key in entries[truth_column][1]:
            target = entries[truth_column][1][key]
            best = min(values, key=lambda index: abs(values[index] - target))
        else:
            best = min(values, key=values.get)
        # Ties share the honour: three columns printing 0.000 should not have one of them singled out.
        printed = fmt.format(values[best])
        highlight[row_index] = [index for index in values if fmt.format(values[index]) == printed]
    return _table(
        f"Closed-loop tracking — reference '{reference}', {duration:.0f} s, same controller and gains for every model\n"
        f"energy-based controller on the learned operators; {law}; plant: contact-free PyBullet CF2P\n"
        "bold = best learned model in that row (PH-GT is the reference, never ranked); roll/pitch references are "
        "flatness-derived with the true m, g, c, so PH-GT is their attainable floor",
        labels, columns, cells, "", fontsize=8.6, scale=(1.0, 2.0), highlight=highlight,
    )


# --- closed-loop, per-model booklet page 1: this model over every reference ---------
def controller_model_table(model_label: str, comparisons: dict, duration: float, law: str,
                           reference: str | None = None) -> plt.Figure:
    """comparisons: {reference name: controller_comparison dict}. One row per reference."""
    against_truth = any("position_rmse_vs_truth_m" in c for c in comparisons.values())
    rows, cells = [], []
    for name, c in comparisons.items():          # not 'reference': that is the parameter used in the title
        rows.append(name)
        f = c.get("failure")
        extra = ([f"{c['position_rmse_vs_truth_m']:.3f}", f"{c['position_gap_vs_truth_percent']:+.0f} %",
                  f"{c['velocity_rmse_vs_truth_m_per_s']:.3f}"] if against_truth else [])
        cells.append([f"{c['position_rmse_m']:.3f}", *extra[:2], f"{c['position_final_m']:.3f}", f"{c['position_max_m']:.2f}",
                      f"{c['velocity_rmse_m_per_s']:.3f}", *extra[2:],
                      f"{c['completed_duration_seconds']:.1f} / {c['requested_duration_seconds']:.0f}",
                      f"{c['shared_motor_saturation_fraction']:.3f}",
                      "none" if not f else f"{f.get('kind', 'failure')} at {f.get('time_seconds', float('nan')):.1f} s"])
    columns = ["position RMSE\nvs reference (m)",
               *(["position RMSE\nvs PH-GT (m)", "gap vs\nPH-GT (%)"] if against_truth else []),
               "position\nfinal (m)", "position\nmax (m)", "velocity RMSE\nvs reference (m/s)",
               *(["velocity RMSE\nvs PH-GT (m/s)"] if against_truth else []),
               "completed /\nrequested (s)", "motor\nsaturation", "controller\nfailure"]
    header = (f"Closed-loop tracking — {model_label} — reference '{reference}'" if reference else
              f"Closed-loop tracking — {model_label}")
    return _table(
        f"{header}\n{duration:.0f} s per reference; {law}; plant: contact-free PyBullet CF2P",
        rows, columns, cells,
        "Each row is one reference shape flown by this model; the pages that follow show its tracking and 3-D trajectory per shape.\n"

        "Errors are against the reference over the horizon actually completed; a failure or a completion below the request means the flight ended early.",
        fontsize=8.4, scale=(1.0, 1.9),
    )


# --- one image per shape: every model's 3-D flight, side by side --------------------
def trajectory_grid(entries, reference: str, duration: float, law: str) -> plt.Figure:
    """entries: (short label, s_traj, full_plan). One independent 3-D axis per model, learned vs reference.

    Each axis is scaled to its own flight plus the reference, so a model that leaves the box shows a large span
    instead of being clipped; the span is printed in the sub-title so the scales can be compared at a glance.
    """
    count = len(entries)
    columns = min(4, count)
    rows = math.ceil(count / columns)
    figure = plt.figure(figsize=(4.1 * columns, 4.0 * rows))
    for index, (label, trajectory, plan) in enumerate(entries, start=1):
        axis = figure.add_subplot(rows, columns, index, projection="3d")
        axis.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="b", lw=1.4,
                    label="learned" if index == 1 else None)
        axis.plot3D(plan[:, 0], plan[:, 1], plan[:, 2], "--", color="chocolate", lw=1.5,
                    label="reference" if index == 1 else None)
        positions = np.concatenate((trajectory[:, :3], plan[:, :3]), axis=0)
        lower, upper = np.nanmin(positions, axis=0), np.nanmax(positions, axis=0)
        span = max(float(np.max(upper - lower)) * 1.1, 2.0)
        centre = (lower + upper) / 2.0
        axis.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
        axis.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
        axis.set_zlim(centre[2] - span / 2, centre[2] + span / 2)
        axis.view_init(elev=25.0, azim=35.0)
        axis.set_xlabel("x (m)", fontsize=7); axis.set_ylabel("y (m)", fontsize=7); axis.set_zlabel("z (m)", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.set_title(f"{label}\naxis span {span:.1f} m", fontsize=9, fontweight="bold")
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, prop={"size": 11})
    figure.suptitle(f"Closed-loop trajectory — reference '{reference}', {duration:.0f} s — learned vs reference\n"
                    f"{law}; plant: contact-free PyBullet CF2P; same reference, gains and start point in every panel",
                    fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    return figure


# --- one image per shape: every model's tracking, one row each ----------------------
def tracking_grid(entries, reference: str, duration: float, law: str) -> plt.Figure:
    """entries: (short label, s_traj, s_plan). One row per model, the eight tracked states as columns.

    Every cell is learned (blue) against the commanded reference (chocolate dashed); the omega column holds the
    three body rates in their own colours with the analytic yaw-rate command, as on the per-model pages.
    """
    states = (("x (m)", 0), ("y (m)", 1), ("z (m)", 2), ("yaw (rad)", 9),
              ("vx (m/s)", 3), ("vy (m/s)", 4), ("vz (m/s)", 5), (r"$\omega$ (rad/s)", None))
    rows = len(entries)
    figure, axes = plt.subplots(rows, len(states), figsize=(2.45 * len(states), 1.75 * rows), squeeze=False)
    for row, (label, trajectory, plan) in enumerate(entries):
        time = trajectory[:, -1]
        for column, (name, index) in enumerate(states):
            axis = axes[row][column]
            axis.grid(True, alpha=0.3)
            if index is None:                         # the three body rates, and the commanded yaw rate
                for state_column, colour in ((10, "#1f77b4"), (11, "#2ca02c"), (12, "#9467bd")):
                    axis.plot(time, trajectory[:, state_column], color=colour, lw=0.8)
            else:
                axis.plot(time, trajectory[:, index], color="b", lw=1.1,
                          label="learned" if (row == 0 and column == 0) else None)
                axis.plot(plan[:, -1], plan[:, index], "--", color="chocolate", lw=1.1,
                          label="reference" if (row == 0 and column == 0) else None)
            axis.tick_params(labelsize=6)
            if row == 0:
                axis.set_title(name, fontsize=9, fontweight="bold")
            if row == rows - 1:
                axis.set_xlabel("Time (s)", fontsize=7)
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(label.replace("  ", "\n"), fontsize=7, fontweight="bold", labelpad=2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, prop={"size": 11})
    figure.suptitle(f"Closed-loop tracking — reference '{reference}', {duration:.0f} s — learned vs reference\n"
                    f"{law}; one row per model; the omega column shows the three body rates "
                    r"($\omega_x$ blue, $\omega_y$ green, $\omega_z$ purple)",
                    fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    return figure


# --- closed-loop sweep summary ---------------------------------------------------
def controller_sweep_page(results, law: str) -> plt.Figure:
    """results: {reference name: [(model label, colour, comparison dict), ...]}.

    One row per sweep family (name prefix before the level), three panels: position RMSE, max error, saturation
    against the level. Every level is drawn, so a model's breaking point is read off the curve, not chosen.
    """
    import re
    families: dict[str, dict[float, list]] = {}
    for name, entries in results.items():
        m = re.match(r"^(spiral_k|lissajous_yaw|stop_v)(\d+p?\d*)$", name)
        if not m:
            continue
        families.setdefault(m.group(1), {})[float(m.group(2).replace("p", "."))] = entries
    if not families:
        figure, axis = plt.subplots(figsize=(11, 7)); axis.axis("off")
        axis.text(0.5, 0.5, "no sweep families in this run", ha="center", va="center"); return figure
    xlabels = {"spiral_k": "turn-rate scale k  (peak speed 1.63 k m/s)", "lissajous_yaw": "yaw amplitude A (rad), period 3 s",
               "stop_v": "cruise speed v0 (m/s) at the stop command"}
    figure, axes = plt.subplots(len(families), 3, figsize=(15, 4.2 * len(families)), squeeze=False)
    for row, (family, levels) in enumerate(sorted(families.items())):
        xs = sorted(levels)
        labels = [e[0] for e in levels[xs[0]]]
        for col, (key, title) in enumerate((("position_rmse_m", "position RMSE (m)"), ("position_max_m", "max position error (m)"),
                                            ("shared_motor_saturation_fraction", "motor saturation fraction"))):
            axis = axes[row, col]
            for index, label in enumerate(labels):
                ys = [levels[x][index][2][key] for x in xs]
                colour = levels[xs[0]][index][1]
                axis.plot(xs, ys, marker="o", color=colour, lw=1.8, label=label.replace("\n", " ") if col == 0 else None)
            axis.set_xlabel(xlabels[family], fontsize=9); axis.set_title(f"{family}: {title}", fontsize=10); axis.grid(True, alpha=0.3)
            if key != "shared_motor_saturation_fraction":
                axis.set_yscale("log")
        axes[row, 0].legend(fontsize=8, loc="upper left")
    figure.suptitle(f"Closed-loop difficulty sweeps — every level reported; {law}", fontsize=12, fontweight="bold")
    figure.text(0.015, 0.008, "Each family ramps one parameter along a mechanism: sustained speed (translational damping in the feedforward), "
                "yaw rate (rotational damping), and a stop-at-speed transient (force scale under saturation).", fontsize=8.5)
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    return figure


# --- pages 9 and 10 -------------------------------------------------------------
def loss_page(entries, title: str, note: str, *, test: bool) -> plt.Figure:
    """entries: (label, colour, {component: (steps, values)})."""
    figure, axis = plt.subplots(figsize=(13, 8))
    for label, colour, series in entries:
        for component, style in COMPONENT_STYLES:
            if component not in series:
                continue
            steps, values = series[component]
            values = np.asarray(values, dtype=float)
            steps = np.asarray(steps, dtype=float)[: values.size]
            if not test and values.size > 200:
                values = smooth(values)  # 101-step moving average; stated in the page caption
            axis.plot(steps, np.maximum(values, 1e-14), color=colour, ls=style, lw=1.1)
    axis.set_yscale("log"); axis.grid(True, alpha=0.3, which="both")
    axis.set_xlabel("optimizer step, final curriculum stage"); axis.set_ylabel("loss (log scale)")
    axis.set_title(title, fontsize=12, fontweight="bold")
    model_handles = [plt.Line2D([], [], color=colour, lw=2, label=label.replace("\n", " "))
                     for label, colour, _ in entries]
    style_handles = [plt.Line2D([], [], color="#444444", ls=style, lw=1.4, label=component)
                     for component, style in COMPONENT_STYLES]
    first = axis.legend(handles=model_handles, title="colour: model", loc="upper right", fontsize=8, title_fontsize=8)
    axis.add_artist(first)
    axis.legend(handles=style_handles, title="line style: component", loc="lower left", fontsize=8, title_fontsize=8)
    figure.text(0.5, 0.012, note, ha="center", fontsize=8, wrap=True)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    return figure


# --- page 9 ---------------------------------------------------------------------
def gp_objective_page(stats: dict[str, np.ndarray], label: str, noise: float) -> plt.Figure:
    """Four panels of GP-only objective terms; panels whose series are absent say so."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    eval_step = np.asarray(stats.get("step", []), dtype=float)
    train_step = np.asarray(stats.get("train_step", []), dtype=float)

    def series(name):
        value = stats.get(name)
        return None if value is None or np.asarray(value).size == 0 else np.asarray(value, dtype=float)

    axis = axes[0, 0]
    terms = [("objective", eval_step, series("objective")),
             ("NLL (train)", train_step, series("train_nll_total")),
             ("NLL (held out)", eval_step, series("test_nll_total")),
             ("beta*KL/N", train_step, series("train_kl_contribution")),
             ("latent-state KL", train_step, series("train_latent_kl")),
             ("gravity penalty", train_step, series("train_gravity_penalty")),
             ("actuation penalty", train_step, series("train_actuation_penalty")),
             ("continuity", train_step, series("train_continuity"))]
    drawn = 0
    for (name, steps, values), colour in zip(terms, TERM_COLOURS * 2):
        if values is None or steps.size == 0:
            continue
        inactive = bool(np.allclose(values, 0.0))
        axis.plot(steps[: values.size], values, color=colour, lw=1.3, alpha=0.5 if inactive else 1.0,
                  label=f"{name} (inactive, weight 0)" if inactive else name)
        drawn += 1
    axis.set_yscale("symlog", linthresh=1e-3)
    axis.set_title("A. objective and its additive terms", fontsize=10, fontweight="bold")
    axis.set_xlabel("step"); axis.set_ylabel("value (symlog)"); axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, ncol=2)

    axis = axes[0, 1]
    weight_kl = series("train_weight_kl")
    kl_steps = train_step if weight_kl is not None else eval_step
    weight_kl = weight_kl if weight_kl is not None else series("kl")
    if weight_kl is not None and kl_steps.size:
        axis.plot(kl_steps[: weight_kl.size], np.maximum(weight_kl, 1e-14), color=TERM_COLOURS[0], lw=1.4,
                  label="weight-space KL")
        axis.set_yscale("log")
    beta = series("train_kl_beta")
    if beta is not None:
        twin = axis.twinx()
        twin.plot(train_step[: beta.size], beta, color=TERM_COLOURS[1], lw=1.4, ls="--", label="beta")
        twin.set_ylabel("beta (annealing coefficient)", color=TERM_COLOURS[1])
        twin.legend(loc="lower right", fontsize=8)
    else:
        axis.text(0.5, 0.06, "beta schedule not recorded in this run", transform=axis.transAxes,
                  ha="center", fontsize=8, color="#888888")
    axis.set_title("B. KL and its annealing schedule", fontsize=10, fontweight="bold")
    axis.set_xlabel("step"); axis.set_ylabel("KL (log)"); axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right", fontsize=8)

    axis = axes[1, 0]
    groups = sorted(name for name in stats if name.startswith("prodigy_d_"))
    for name, colour in zip(groups, TERM_COLOURS * 2):
        values = np.asarray(stats[name], dtype=float)
        axis.plot(eval_step[: values.size], np.maximum(values, 1e-14), color=colour, lw=1.3,
                  label=name.removeprefix("prodigy_d_"))
    axis.set_yscale("log")
    axis.set_title("C. Prodigy adaptive step size per level group", fontsize=10, fontweight="bold")
    axis.set_xlabel("step"); axis.set_ylabel("step size (log)"); axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    axis.text(0.02, 0.03, "a group pinned at its initial value never moved: that level is unidentifiable",
              transform=axis.transAxes, fontsize=7, color="#555555")

    axis = axes[1, 1]
    sigma_names = [("position", "train_sigma_position"), ("attitude", "train_sigma_attitude"),
                   ("linear velocity", "train_sigma_linear_velocity"),
                   ("angular velocity", "train_sigma_angular_velocity")]
    present = [(name, series(key)) for name, key in sigma_names if series(key) is not None]
    if present:
        for (name, values), colour in zip(present, TERM_COLOURS):
            axis.plot(train_step[: values.size], values, color=colour, lw=1.3, label=name)
        axis.axhline(noise, color="black", ls=":", lw=1.5, label=f"true noise {noise:g} (never shown to the model)")
        axis.set_ylabel("learned sigma"); axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "Learned observation sigma not recorded in this run.\n\nThe trainer now saves "
                            "train_sigma_* and test_sigma_*;\nthis model predates that change, so only the final\n"
                            "values survive, in the checkpoint.",
                  transform=axis.transAxes, ha="center", va="center", fontsize=10, color="#555555")
    axis.set_title("D. learned observation sigma (noise-blind training)", fontsize=10, fontweight="bold")
    axis.set_xlabel("step"); axis.grid(True, alpha=0.3)

    figure.suptitle(f"{label.replace(chr(10), ' ')} — objective terms\n"
                    "Method-specific page: the common MSE yardstick of pages 7 and 8 is not what this model minimises",
                    fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


# --- page 10 --------------------------------------------------------------------
def rollout_summary_table(entries, truth: np.ndarray, horizon: float, flight_name: str) -> plt.Figure:
    """entries: (label, errors dict).  Seven rows: three physical, four unit-free."""
    excursions = ground_truth_excursions(truth)
    columns = [label for label, _ in entries]

    def rms(errors, key):
        return np.sqrt(np.asarray(errors[key]).mean(axis=1))

    def relative(errors, key):
        reference = np.sqrt(np.maximum(np.asarray(excursions[key]).mean(axis=1), 1e-30))
        return rms(errors, key) / reference

    def cell(values, digits=3):
        return f"{np.mean(values):.{digits}g} ± {np.std(values):.2g}"

    rows = ["position RMS (m)", "position final (m)", "attitude RMS (rad)",
            "position RMS / GT excursion", "attitude RMS / GT excursion",
            "linear velocity RMS / GT excursion", "angular velocity RMS / GT excursion"]
    cells = [
        [cell(rms(e, "position")) for _, e in entries],
        [cell(np.sqrt(np.asarray(e["position"])[:, -1])) for _, e in entries],
        [cell(rms(e, "attitude")) for _, e in entries],
        [cell(relative(e, "position")) for _, e in entries],
        [cell(relative(e, "attitude")) for _, e in entries],
        [cell(relative(e, "velocity")) for _, e in entries],
        [cell(relative(e, "omega")) for _, e in entries],
    ]
    return _table(
        f"Open-loop rollout error — {flight_name}, horizon {horizon:.2f} s\n"
        f"{truth.shape[0]} distinct held-out flights; mean ± spread across flights",
        rows, columns, cells,
        "The first three rows are in physical units so the error can be pictured. The last four divide the RMS "
        "error by the RMS excursion of the ground truth from its own initial state over the same horizon, so 1.0 "
        "means the error equals the true motion. The last column is the simulator's own analytic operators in the "
        "same integrator driven by the same recorded u(t): the achievable floor, not a competitor. "
        "This page measures forecasting; page 1 measures identification, and the two can disagree.",
        fontsize=8.6, scale=(1.0, 2.0),
    )


# --- page 15 --------------------------------------------------------------------
def energy_page(series, truth: np.ndarray, time_axis: np.ndarray, vehicle: dict) -> plt.Figure:
    gt_energy = physical_energy(truth, vehicle)
    energies = [(label, colour, style, physical_energy(prediction, vehicle))
                for label, colour, style, prediction in series]
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for flight in range(gt_energy.shape[0]):
        axes[0].plot(time_axis, gt_energy[flight], "k-", lw=1.1, alpha=0.55,
                     label="ground truth" if flight == 0 else None)
    for label, colour, style, predicted in energies:
        for flight in range(predicted.shape[0]):
            axes[0].plot(time_axis, predicted[flight], color=colour, ls=style, lw=1.0, alpha=0.55,
                         label=label.replace("\n", " ") if flight == 0 else None)
    axes[0].set_title(f"all {gt_energy.shape[0]} held-out flights", fontsize=10)
    axes[1].plot(time_axis, gt_energy[0], "k-", lw=2, label="ground truth")
    for label, colour, style, predicted in energies:
        axes[1].plot(time_axis, predicted[0], color=colour, ls=style, lw=1.6, label=label.replace("\n", " "))
    axes[1].set_title("flight 0", fontsize=10)
    for axis in axes:
        axis.set_ylabel("Energy (J)"); axis.grid(True, alpha=0.3); axis.legend(fontsize=8)
    axes[1].set_xlabel("Time (s)")
    figure.suptitle("Physical energy along the held-out flights", fontsize=12, fontweight="bold")
    figure.text(0.5, 0.012, "Energy is a derived diagnostic, not an identifiable quantity: it is built from the "
                            "learned inverse mass and potential and therefore depends on the port-Hamiltonian gauge. "
                            "A model can track energy well while getting the physics wrong, and the reverse.",
                ha="center", fontsize=8, wrap=True)
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    return figure


# --- page 16 --------------------------------------------------------------------
def state_error_page(series, time_axis: np.ndarray) -> plt.Figure:
    """entries: (label, colour, style, errors).  Colour = model, line style = state block."""
    figure, axis = plt.subplots(figsize=(13, 8))
    keys = {"position": "position", "attitude": "attitude",
            "linear velocity": "velocity", "angular velocity": "omega"}
    for label, colour, _style, errors in series:
        for block, style in ERROR_STYLES:
            values = np.asarray(errors[keys[block]]).mean(axis=0)
            axis.plot(time_axis, np.maximum(values, 1e-16), color=colour, ls=style, lw=1.2)
    axis.set_yscale("log"); axis.grid(True, alpha=0.3, which="both")
    axis.set_xlabel("Time (s)"); axis.set_ylabel("squared error, mean over held-out flights (log scale)")
    axis.set_title("State error against ground truth — every block on one axis", fontsize=12, fontweight="bold")
    model_handles = [plt.Line2D([], [], color=colour, lw=2, label=label.replace("\n", " "))
                     for label, colour, _s, _e in series]
    style_handles = [plt.Line2D([], [], color="#444444", ls=style, lw=1.4, label=block)
                     for block, style in ERROR_STYLES]
    first = axis.legend(handles=model_handles, title="colour: model", loc="upper left", fontsize=8, title_fontsize=8)
    axis.add_artist(first)
    axis.legend(handles=style_handles, title="line style: state block", loc="lower right", fontsize=8, title_fontsize=8)
    figure.text(0.5, 0.012, "Units differ per block: m², rad², m²/s², rad²/s². The log axis is what makes them "
                            "comparable on one plot. Squared position error grows without bound while attitude "
                            "error saturates near π². This is the quantitative companion to pages 11 and 12.",
                ha="center", fontsize=8, wrap=True)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    return figure

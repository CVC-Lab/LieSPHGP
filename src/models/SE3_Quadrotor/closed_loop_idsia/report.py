"""PDF of the closed-loop melon test: summary table, error over time, and per-plant tracking pages."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from .reference import SAMPLE_STEP  # noqa: E402

RECORDED = "#0b0b0b"                 # the recording is ink, never a series hue
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#52514e"
DETAIL_SECONDS = 10.0
ROTOR_SECONDS = 3.0

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": RECORDED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.3, "legend.frameon": False})


def _euler(states: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(Rotation.from_matrix(states[:, 3:12].reshape(-1, 3, 3)).as_euler("xyz"), axis=0))


def _alive_slice(arr: dict, b: int, n: int) -> int:
    dead = int(arr["first_dead"][b])
    return n if dead < 0 else min(n, dead)


def summary_page(pdf, results: dict, samples: int) -> None:
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.04, 0.95, "Closed-loop melon test — benchmark controller flying each plant", fontsize=14,
             weight="bold", color=RECORDED)
    fig.text(0.04, 0.915, "Mellinger geometric controller + integral action at 500 Hz, benchmark mixer, recorded path "
             "as reference, start from the recorded state at t = 0, no resets.  RMSE against the RECORDED flight; "
             "* = a flight diverged inside the window.", fontsize=8, color=MUTED, wrap=True)
    full = f"0-{samples * SAMPLE_STEP:g}s"
    columns = ["plant", "input", "survival (s)\nrun1/2/3", "pos RMSE\n0–5 s (m)", "pos RMSE\n0–10 s (m)",
               "pos RMSE\n0–30 s (m)", "pos RMSE\nfull (m)", "att RMSE\n0–10 s (°)", "rotor RMSE\n0–10 s (rad/s)",
               "saturated\n0–10 s"]
    rows = []
    for label, r in results.items():
        def cell(window, key, fmt="{:.3f}"):
            v = r[window][key]
            return ("—" if v != v else fmt.format(v)) + ("" if r[window]["all_flights_survived"] else "*")
        rows.append([label, r["input_mode"], "/".join(f"{s:.1f}" for s in r["survival_seconds"]),
                     cell("0-5s", "position_rmse_m"), cell("0-10s", "position_rmse_m"),
                     cell("0-30s", "position_rmse_m"), cell(full, "position_rmse_m"),
                     cell("0-10s", "attitude_rmse_deg", "{:.1f}"), cell("0-10s", "rotor_speed_rmse_rads", "{:.0f}"),
                     cell("0-10s", "saturated_fraction", "{:.2f}")])
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.7])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="upper center", cellLoc="center",
                     colWidths=[0.25, 0.06, 0.1, 0.07, 0.07, 0.07, 0.07, 0.07, 0.09, 0.07])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.2)
    for (row, col), c in table.get_celld().items():
        c.set_edgecolor("#e4e3df")
        if row == 0:
            c.set_text_props(weight="bold", color=RECORDED)
        if col == 0 and row > 0:
            c.set_text_props(ha="left")
    fig.text(0.04, 0.08, "Rotor RMSE: RMS over the four rotors of (simulated − recorded) speed; hover is ≈1722 rad/s. "
             "A plant that responds like the real drone should need the recorded rotor speeds to fly the recorded path.",
             fontsize=8, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def error_page(pdf, arrays: dict, samples: int, flight: int, name: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.7, 8.3), sharex=True)
    t = np.arange(samples) * SAMPLE_STEP
    for i, (label, arr) in enumerate(arrays.items()):
        n = _alive_slice(arr, flight, samples)
        color = SERIES[i % len(SERIES)]
        axes[0].semilogy(t[1:n], np.maximum(arr["position"][flight, 1:n], 1e-4), color=color, label=label)
        axes[1].plot(t[1:n], arr["attitude"][flight, 1:n], color=color)
        axes[2].plot(t[1:n], arr["speed"][flight, :n - 1], color=color)
        if n < samples:
            for ax in axes:
                ax.axvline(t[n - 1], color=color, lw=0.8, ls=":")
    axes[0].set_ylabel("position error (m, log)")
    axes[1].set_ylabel("attitude error (°)")
    axes[2].set_ylabel("rotor-speed error (rad/s)")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(loc="lower right", ncol=2, fontsize=7)
    fig.suptitle(f"Error against the recording over the whole flight — {name}  (dotted line: divergence)",
                 x=0.04, ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def plant_page(pdf, label: str, color: str, arr: dict, flight: dict, b: int, samples: int, seconds: float) -> None:
    n_all = min(samples, int(round(seconds / SAMPLE_STEP)) + 1)
    n = _alive_slice(arr, b, n_all)
    t = np.arange(n_all) * SAMPLE_STEP
    truth, sim = flight["states"][:n_all], arr["states"][b, :n]
    fig = plt.figure(figsize=(11.7, 8.3))
    grid = fig.add_gridspec(3, 4)
    ax3 = fig.add_subplot(grid[0:2, 0], projection="3d")
    ax3.plot(*truth[:, :3].T, color=RECORDED, label="recorded")
    ax3.plot(*sim[:, :3].T, color=color, label="closed loop")
    ax3.set_title("path", fontsize=9)
    ax3.legend(fontsize=7, loc="upper left")
    for k, axis in enumerate("xyz"):
        ax = fig.add_subplot(grid[0, k + 1])
        ax.plot(t, truth[:, k], color=RECORDED)
        ax.plot(t[:n], sim[:, k], color=color)
        ax.set_title(f"{axis} (m)", fontsize=9)
    e_true, e_sim = _euler(truth), _euler(sim)
    for k, name in enumerate(("roll", "pitch", "yaw")):
        ax = fig.add_subplot(grid[1, k + 1])
        ax.plot(t, e_true[:, k], color=RECORDED)
        ax.plot(t[:n], e_sim[:, k], color=color)
        ax.set_title(f"{name} (°)", fontsize=9)
    n_rotor = min(n, int(round(ROTOR_SECONDS / SAMPLE_STEP)) + 1)
    for k in range(4):
        ax = fig.add_subplot(grid[2, k])
        ax.plot(t[1:n_rotor], flight["rotor_speeds"][1:n_rotor, k], color=RECORDED)
        ax.plot(t[1:n_rotor], arr["rotor_speeds"][b, :n_rotor - 1, k], color=color)
        ax.set_title(f"rotor {k + 1} speed (rad/s), first {ROTOR_SECONDS:g} s", fontsize=9)
        ax.set_xlabel("time (s)")
    dead = "" if n == n_all else f"   — DIVERGED at t = {t[n - 1]:.2f} s"
    fig.suptitle(f"{label}  |  {flight['name']}, first {seconds:g} s  (black: recorded, colour: closed loop){dead}",
                 x=0.04, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(path: Path, flights: list[dict], samples: int, arrays: dict, results: dict) -> None:
    with PdfPages(path) as pdf:
        summary_page(pdf, results, samples)
        for b, flight in enumerate(flights):
            error_page(pdf, arrays, samples, b, flight["name"])
        for i, (label, arr) in enumerate(arrays.items()):
            color = SERIES[i % len(SERIES)]
            plant_page(pdf, label, color, arr, flights[0], 0, samples, DETAIL_SECONDS)
            plant_page(pdf, label, color, arr, flights[0], 0, samples, samples * SAMPLE_STEP)

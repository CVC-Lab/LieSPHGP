"""Paper figure of the pendulum closed-loop swing-up (6 Oct 2026): one image, 2 x 3 panels
(roll, pitch, yaw / omega_x, omega_y, omega_z, all in rad and rad/s), every panel showing the same flight of
Analytical (true physics), Lie-PH-GP-SDE, Lie-PH-NN-SDE and PH-NODE.

    python src/models/3D_SO3_Windy_Pendulum/comparison/closed_loop_figure.py \
        --report-dir experiments/pendulum_so3/campaign_04-10-2026/reports/DampRate-Wind_noise0p5

Reads the flights cached by closed_loop.py --task upright (closed-loop-upright-flights.npz: states (N+1, F, 12) =
[vec(R) row-major, omega_body], flight f = start f // seeds, wind seed f % seeds, control at 100 Hz); nothing is flown
again. Without --start/--seed the flight is chosen automatically among the starts at least --min-start-angle from
upright (a real swing-up) as the one with the largest contrast
    min(mean angle to upright of Lie-PH-NN-SDE, PH-NODE) - mean angle to upright of Lie-PH-GP-SDE,   over 5-10 s. Target: upright, R_d = I (roll = pitch = yaw = 0, mod 2 pi) and omega_d = 0.
Writes <report-dir>/closed-loop-upright-figure.png and .pdf (that one flight), and
<report-dir>/closed-loop-upright-figure-bands.png and .pdf: the same start flown with every wind seed, line = mean over
the seeds, dark / light band = +-1 / +-2 standard deviations over the seeds (the spread the plant's wind causes under
each model's controller; the controller uses the learned drift only, so this is not the model's predictive band).
The true physics is drawn as its dashed mean with dotted +-2 sigma edges, on top of the learned models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

CONTROL_DT = 0.01                                     # closed_loop.py CONTROL_DT (100 Hz)
STYLE = {   # label in the npz -> (legend name, colour, line style, width); colours as in report.py
    "Analytical": ("Analytical (true physics)", "tab:green", (0, (4, 3)), 1.5),
    "Lie-PH-GP-SDE": ("Lie-PH-GP-SDE", "tab:orange", "-", 2.6),
    "Lie-PH-NN-SDE": ("Lie-PH-NN-SDE", "tab:cyan", "-", 1.4),
    "PH-NODE": ("PH-NODE", "tab:pink", "-", 1.4),
}
PANELS = (("roll (rad)", "euler", 0), ("pitch (rad)", "euler", 1), ("yaw (rad)", "euler", 2),
          (r"$\omega_x$ (rad/s)", "omega", 0), (r"$\omega_y$ (rad/s)", "omega", 1), (r"$\omega_z$ (rad/s)", "omega", 2))


def project(rotation_flat: np.ndarray) -> np.ndarray:
    """Nearest rotation matrices (SVD) of (T, 9) row-major entries; NaN rows stay NaN."""
    matrices = rotation_flat.reshape(-1, 3, 3)
    out = np.full_like(matrices, np.nan)
    ok = np.isfinite(matrices).all(axis=(1, 2))
    u, _, vt = np.linalg.svd(matrices[ok])
    fix = np.ones((ok.sum(), 3))
    fix[:, 2] = np.sign(np.linalg.det(u @ vt))
    out[ok] = (u * fix[:, None, :]) @ vt
    return out


def euler_xyz(rotation_flat: np.ndarray) -> np.ndarray:
    """(T, 3) Euler xyz angles in rad, unwrapped along time (so a swing through +-pi stays continuous)."""
    matrices = project(rotation_flat)
    angles = np.full((len(matrices), 3), np.nan)
    ok = np.isfinite(matrices).all(axis=(1, 2))
    angles[ok] = Rotation.from_matrix(matrices[ok]).as_euler("xyz")
    return np.where(np.isfinite(angles), np.unwrap(np.nan_to_num(angles), axis=0), np.nan)


def angle_to_upright(rotation_flat: np.ndarray) -> np.ndarray:
    """(T, F) geodesic angle (rad) between R and the upright target I."""
    trace = rotation_flat[..., 0] + rotation_flat[..., 4] + rotation_flat[..., 8]
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def choose_flight(states: dict, seeds: int, min_start_angle: float) -> tuple[int, int]:
    """Among swing-up starts, the flight where Lie-PH-GP-SDE ends closest to upright relative to the better of the
    other two learned models (mean angle to upright over the second half of the window)."""
    half = states["Analytical"].shape[0] // 2
    late = {label: np.nanmean(angle_to_upright(states[label][half:, :, :9]), axis=0) for label in states}
    contrast = np.minimum(late["Lie-PH-NN-SDE"], late["PH-NODE"]) - late["Lie-PH-GP-SDE"]
    contrast[angle_to_upright(states["Analytical"][0, :, :9]) < min_start_angle] = -np.inf
    flight = int(np.nanargmax(contrast))
    return flight // seeds, flight % seeds


LAYOUTS = {   # name: (rows, cols, figure size in inches, legend columns, font size, line-width factor, file suffix)
    "2x3": (2, 3, (13, 6.2), 4, 10, 1.0, ""),
    "6x1": (6, 1, (3.5, 9.2), 2, 7.5, 0.7, "-6x1"),          # one paper column: roll, pitch, yaw, omega_x, omega_y, omega_z
}


def new_figure(layout: str):
    """(figure, the 6 axes in PANELS order, bottom-row axes, legend columns, font size, line-width factor)."""
    rows, cols, size, ncol, font, scale, _ = LAYOUTS[layout]
    plt.rcParams.update({"font.size": font, "axes.labelsize": font, "xtick.labelsize": font - 1, "ytick.labelsize": font - 1})
    figure, axes = plt.subplots(rows, cols, figsize=size, sharex=True)
    axes = np.asarray(axes).reshape(rows, cols)
    return figure, list(axes.ravel()), list(axes[-1]), ncol, font, scale


def band_figure(states: dict, start: int, seeds: int, t: np.ndarray, out: Path, dpi: int, layout: str = "2x3") -> dict:
    """Seed mean +- 1 / 2 std of every panel for one start; returns the angle-to-upright stats per model."""
    flights = range(start * seeds, (start + 1) * seeds)
    stacks = {label: {"euler": np.stack([euler_xyz(s[:, f, :9]) for f in flights]),        # (seeds, T, 3)
                      "omega": np.stack([s[:, f, 9:12] for f in flights])} for label, s in states.items()}
    figure, panels, bottom, ncol, font, scale = new_figure(layout)
    for axis, (name, block, column) in zip(panels, PANELS):
        for label in ("PH-NODE", "Lie-PH-NN-SDE", "Lie-PH-GP-SDE", "Analytical"):
            legend, colour, style, width = STYLE[label]
            width *= scale
            values = stacks[label][block][..., column]
            mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=1)
            if label == "Analytical":                       # outline only, so the GP-SDE band underneath stays visible
                for edge in (mean - 2 * std, mean + 2 * std):
                    axis.plot(t, edge, color=colour, ls=":", lw=0.9 * scale)
            else:
                axis.fill_between(t, mean - 2 * std, mean + 2 * std, color=colour, alpha=0.12, lw=0)
                axis.fill_between(t, mean - std, mean + std, color=colour, alpha=0.30, lw=0)
            axis.plot(t, mean, color=colour, ls=style, lw=width, label=legend)
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
    for axis in bottom:
        axis.set_xlabel("t (s)")
    handles, names = panels[0].get_legend_handles_labels()
    order = [names.index(STYLE[label][0]) for label in ("Lie-PH-GP-SDE", "Lie-PH-NN-SDE", "PH-NODE", "Analytical")]
    figure.legend([handles[i] for i in order], [names[i] for i in order], loc="upper center", ncol=ncol, frameon=False,
                  fontsize=font, bbox_to_anchor=(0.5, 1.0))
    note = (f"line = mean over {seeds} wind seeds; dark / light band = $\\pm 1\\sigma$ / $\\pm 2\\sigma$ over the seeds "
            "(true physics: dotted $\\pm 2\\sigma$ edges)")
    if layout == "6x1":
        figure.text(0.5, 0.945, f"mean over {seeds} wind seeds, bands $\\pm 1\\sigma$ / $\\pm 2\\sigma$", ha="center", fontsize=font - 1)
        figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.3)
    else:
        figure.text(0.5, 0.935, note, ha="center", fontsize=9)
        figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(out.with_suffix(".png"), dpi=dpi)
    figure.savefig(out.with_suffix(".pdf"))
    plt.close(figure)
    final = {label: angle_to_upright(s[-1, list(flights), :9]) for label, s in states.items()}
    return {label: (float(np.mean(v)), float(np.std(v, ddof=1))) for label, v in final.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, help="start (trajectory) index; default: automatic choice")
    parser.add_argument("--seed", type=int, help="wind seed index; default: automatic choice")
    parser.add_argument("--seeds", type=int, default=5, help="wind seeds per start in the cache (5 for wind data)")
    parser.add_argument("--min-start-angle", type=float, default=1.5, help="rad; automatic choice only")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--layout", default="2x3", choices=tuple(LAYOUTS),
                        help="2x3 (default, full width) or 6x1 (one paper column; files get the suffix -6x1)")
    args = parser.parse_args()

    cache = np.load(args.report_dir / "closed-loop-upright-flights.npz")
    states = {label: cache[f"{label}/states"].astype(np.float64) for label in STYLE}
    if args.start is None or args.seed is None:
        start, seed = choose_flight(states, args.seeds, args.min_start_angle)
    else:
        start, seed = args.start, args.seed
    flight = start * args.seeds + seed
    rmse = {label: float(np.sqrt(np.nanmean(angle_to_upright(s[1:, flight, :9]) ** 2))) for label, s in states.items()}
    final = {label: float(angle_to_upright(s[-1, flight, :9])) for label, s in states.items()}
    t = np.arange(states["Analytical"].shape[0]) * CONTROL_DT

    series = {label: {"euler": euler_xyz(s[:, flight, :9]), "omega": s[:, flight, 9:12]} for label, s in states.items()}
    figure, panels, bottom, ncol, font, scale = new_figure(args.layout)
    for axis, (name, block, column) in zip(panels, PANELS):
        for label in ("PH-NODE", "Lie-PH-NN-SDE", "Lie-PH-GP-SDE", "Analytical"):     # true physics dashed on top
            legend, colour, style, width = STYLE[label]
            axis.plot(t, series[label][block][:, column], color=colour, ls=style, lw=width * scale, label=legend)
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
    for axis in bottom:
        axis.set_xlabel("t (s)")
    handles, names = panels[0].get_legend_handles_labels()
    order = [names.index(STYLE[label][0]) for label in ("Lie-PH-GP-SDE", "Lie-PH-NN-SDE", "PH-NODE", "Analytical")]
    figure.legend([handles[i] for i in order], [names[i] for i in order], loc="upper center", ncol=ncol, frameon=False,
                  fontsize=font, bbox_to_anchor=(0.5, 1.0))
    suffix = LAYOUTS[args.layout][6]
    figure.tight_layout(rect=(0, 0, 1, 0.955 if args.layout == "6x1" else 0.94), h_pad=0.3 if args.layout == "6x1" else None)
    out = args.report_dir / f"closed-loop-upright-figure{suffix}"
    figure.savefig(out.with_suffix(".png"), dpi=args.dpi)
    figure.savefig(out.with_suffix(".pdf"))
    start_angle = float(angle_to_upright(states["Analytical"][0, flight, :9]))
    print(f"flight: start {start}, wind seed {seed}; start angle to upright {start_angle:.2f} rad")
    print("angle-to-upright RMSE over 0-10 s (rad): " + ", ".join(f"{label} {value:.3f}" for label, value in rmse.items()))
    print("angle to upright at 10 s (rad): " + ", ".join(f"{label} {value:.3f}" for label, value in final.items()))
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")
    plt.close(figure)
    bands = args.report_dir / f"closed-loop-upright-figure-bands{suffix}"
    final = band_figure(states, start, args.seeds, t, bands, args.dpi, args.layout)
    print(f"bands: start {start}, all {args.seeds} wind seeds; angle to upright at 10 s, mean +- std (rad): "
          + ", ".join(f"{label} {m:.3f} +- {s:.3f}" for label, (m, s) in final.items()))
    print(f"wrote {bands.with_suffix('.png')} and {bands.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

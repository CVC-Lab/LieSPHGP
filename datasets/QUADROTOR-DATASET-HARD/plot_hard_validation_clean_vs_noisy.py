"""One PDF: the 10 HARD validation flights, clean versus every noisy variant, in the closed-loop report style.

Per flight two pages: (1) one 3-D panel per absolute-noise level (0.05, 0.1, 0.25, 0.5), clean (black dashed) vs ONE noisy realisation (colour);
(2) a state grid, one row per noise variant x 8 columns (x, y, z, yaw, v_b x/y/z, |omega_b|), noisy thin over clean.
The validation flights are clean in every pickle; their noisy copies are the ``test_trajectories_noisy`` arrays.

Usage: python plot_hard_validation_clean_vs_noisy.py [--out PATH]
"""
import argparse, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
TAG = "HARD_CF2P_10s_h0p01"
VARIANTS = [("train-obs-noise-absolute0p05", "absolute σ = 0.05"), ("train-obs-noise-absolute0p1", "absolute σ = 0.1"),
            ("train-obs-noise-absolute0p25", "absolute σ = 0.25 (reported models)"), ("train-obs-noise-absolute0p5", "absolute σ = 0.5")]
COLOURS = ["#009E73", "#56B4E9", "#D55E00", "#CC79A7"]
COLUMNS = [("x (m)", lambda f: f[:, 0]), ("y (m)", lambda f: f[:, 1]), ("z (m)", lambda f: f[:, 2]),
           ("yaw (rad)", lambda f: np.unwrap(np.arctan2(f[:, 6], f[:, 3]))),
           ("v_b,x (m/s)", lambda f: f[:, 12]), ("v_b,y (m/s)", lambda f: f[:, 13]), ("v_b,z (m/s)", lambda f: f[:, 14]),
           ("|ω_b| (rad/s)", lambda f: np.linalg.norm(f[:, 15:18], axis=1))]


def load(name):
    with (HERE / f"{TAG}_{name}.pkl").open("rb") as h:
        return pickle.load(h)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / f"{TAG}_validation_clean_vs_noisy.pdf")
    args = parser.parse_args()
    clean = load("clean")
    truth = np.asarray(clean["test_trajectories"], dtype=np.float64)
    audits = clean["settings"]["test_flight_audits"]
    dt = float(clean["settings"]["sample_dt"])
    noisy = {key: np.asarray(load(key)["test_trajectories_noisy"], dtype=np.float64) for key, _ in VARIANTS}
    t = np.arange(truth.shape[1]) * dt
    with PdfPages(args.out) as pdf:
        for k in range(truth.shape[0]):
            f = truth[k]
            segments = " > ".join(s["name"] for s in audits[k]["segments"])
            head = f"HARD validation flight {k:02d} (seed {audits[k]['seed']}): {segments}"
            # page 1: 3-D panels
            fig = plt.figure(figsize=(18, 4.6))
            for j, ((key, label), colour) in enumerate(zip(VARIANTS, COLOURS)):
                ax = fig.add_subplot(1, len(VARIANTS), j + 1, projection="3d")
                n = noisy[key][k]
                ax.plot(n[:, 0], n[:, 1], n[:, 2], color=colour, lw=0.5, alpha=0.7, label="noisy")
                ax.plot(f[:, 0], f[:, 1], f[:, 2], "k--", lw=1.1, label="clean")
                ax.scatter(*f[0, :3], c="k", s=20)
                rms = float(np.sqrt(np.mean(np.sum((n[:, :3] - f[:, :3]) ** 2, axis=1))))
                ax.set_title(f"{label}\nposition noise RMS {rms:.3f} m", fontsize=8)
                ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7); ax.set_zlabel("z (m)", fontsize=7)
                ax.tick_params(labelsize=6)
                if j == 0: ax.legend(fontsize=7, loc="upper left")
            fig.suptitle(f"{head}\n3-D trajectory: clean (black dashed) versus each noisy variant", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.9)); pdf.savefig(fig); plt.close(fig)
            # page 2: state grid
            fig, axes = plt.subplots(len(VARIANTS), len(COLUMNS), figsize=(18, 2.1 * len(VARIANTS)), sharex=True)
            for i, ((key, label), colour) in enumerate(zip(VARIANTS, COLOURS)):
                n = noisy[key][k]
                for j, (name, fn) in enumerate(COLUMNS):
                    ax = axes[i, j]
                    ax.plot(t, fn(n), color=colour, lw=0.5, alpha=0.8)
                    ax.plot(t, fn(f), "k", lw=0.9)
                    for s in audits[k]["segments"]:
                        ax.axvspan(s["start"], s["start"] + s["duration"], color="grey", alpha=0.06 if (audits[k]["segments"].index(s) % 2) else 0.0)
                    ax.grid(alpha=0.3); ax.tick_params(labelsize=6)
                    if i == 0: ax.set_title(name, fontsize=8)
                    if j == 0: ax.set_ylabel(label.replace(" (", "\n("), fontsize=7)
                    if i == len(VARIANTS) - 1: ax.set_xlabel("t (s)", fontsize=7)
            fig.suptitle(f"{head}\nstates over time: noisy (colour, thin) over clean (black); shaded bands = alternate segments", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.94)); pdf.savefig(fig); plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

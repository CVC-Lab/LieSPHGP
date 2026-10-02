"""Analysis PDF for the converted NanoBench set: the shape-disjoint split and its state coverage.

Usage: python plot_nanobench_dataset.py [--dataset <pkl>] [--out <pdf>]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages

THIS_DIR = Path(__file__).resolve().parents[3] / "datasets/QUADROTOR-DATASET-NANOBENCH"  # data folder, not this script's folder


def family_lists(settings, split):
    """One family label per stored 10 s flight, expanded from the per-source-file audits."""
    labels = []
    for audit in settings[f"{split}_flight_audits"]:
        labels.extend([audit["family"]] * int(audit["kept_chunks"]))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=THIS_DIR / "NANOBENCH_CF2_10s_h0p01_clean.pkl")
    parser.add_argument("--out", type=Path, default=THIS_DIR / "NANOBENCH_CF2_10s_h0p01_dataset_analysis.pdf")
    arguments = parser.parse_args()
    with arguments.dataset.open("rb") as handle:
        data = pickle.load(handle)
    settings = data["settings"]
    train, test = data["train_trajectories"], data["test_trajectories"]
    step = settings["sample_dt"]
    times = np.arange(train.shape[1]) * step
    families = {"train": family_lists(settings, "train"), "test": family_lists(settings, "test")}
    palette = {name: colour for name, colour in zip(
        sorted(set(families["train"]) | set(families["test"])), plt.cm.tab20.colors)}

    with PdfPages(arguments.out) as pdf:
        # 1 summary
        figure, axis = plt.subplots(figsize=(11, 8.5))
        axis.axis("off")
        source = settings["source"]
        lines = [
            f"Dataset {settings['dataset_name']}  (real flights)",
            f"source: {source['name']} ({source['url']}, arXiv {source['arxiv']}, {source['license']})",
            f"platform: {source['platform']};  flying mass {settings['vehicle_parameters']['mass'] * 1000:.2f} g",
            f"sampling {1 / step:.0f} Hz, flights of {(train.shape[1] - 1) * step:.2f} s",
            "",
            f"TRAIN {train.shape[0]} flights ({train.shape[0] * (train.shape[1] - 1) * step:.0f} s) from "
            f"{len(set(families['train']))} shape families:",
            "    " + ", ".join(sorted(set(families["train"]))),
            f"TEST  {test.shape[0]} flights ({test.shape[0] * (test.shape[1] - 1) * step:.0f} s) from "
            f"{len(set(families['test']))} shape families, none of them in training:",
            "    " + ", ".join(sorted(set(families["test"]))),
            f"    shared between the splits: {settings['splits']['families_shared_between_train_and_test'] or 'none'}",
            "",
            "state: Vicon only (position, attitude, world velocity and rate rotated into the body frame)",
            f"input: {settings['control_layout']}",
            f"    {settings['motor_model']['relation']}, V_nom = {settings['motor_model']['v_nominal']} V",
            f"    validation: {settings['motor_model']['validation']}",
            "",
            "coverage (p50 / p90 / max):",
        ]
        for split, values in (("train", settings["audits_train"]), ("test", settings["audits_test"])):
            lines += [
                f"  {split:5s} speed {np.round(values['speed_percentiles_50_90_max'], 2).tolist()} m/s, "
                f"rate {np.round(values['angular_rate_percentiles_50_90_max'], 2).tolist()} rad/s, "
                f"tilt {np.round(values['tilt_deg_percentiles_50_90_max'], 1).tolist()} deg, "
                f"T/mg mean {values['thrust_over_weight_mean_cv'][0]:.3f} (CV {values['thrust_over_weight_mean_cv'][1]:.3f})"]
        lines += ["", "observation noise: " + settings["observation_noise"]["note"]]
        axis.text(0.01, 0.99, "\n".join(lines), va="top", family="monospace", fontsize=8)
        pdf.savefig(figure)
        plt.close(figure)

        # 2-3 the two splits in space
        for split, flights in (("TRAIN (8 shape families)", train), ("TEST (4 held-out shape families)", test)):
            key = "train" if split.startswith("TRAIN") else "test"
            figure = plt.figure(figsize=(11, 8.5))
            axis = figure.add_subplot(projection="3d")
            seen = set()
            for flight, family in zip(flights, families[key]):
                axis.plot(flight[:, 0], flight[:, 1], flight[:, 2], lw=0.7, alpha=0.8,
                          color=palette[family], label=family if family not in seen else None)
                seen.add(family)
            axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                     title=f"{split}: {flights.shape[0]} flights of "
                           f"{(flights.shape[1] - 1) * step:.0f} s, coloured by trajectory family")
            axis.legend(fontsize=8, loc="upper left")
            pdf.savefig(figure)
            plt.close(figure)

        # 4 coverage, train versus test
        figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        panels = (
            (lambda a: np.linalg.norm(a[..., 12:15], axis=-1).ravel(), "speed |v_b| (m/s)"),
            (lambda a: np.linalg.norm(a[..., 15:18], axis=-1).ravel(), "angular rate |omega_b| (rad/s)"),
            (lambda a: np.degrees(np.arccos(np.clip(a[..., 11].ravel(), -1, 1))), "tilt (deg)"),
            (lambda a: a[..., 18].ravel() / (settings["vehicle_parameters"]["mass"]
                                             * settings["vehicle_parameters"]["gravity_acceleration"]),
             "collective thrust T / (m g)"),
        )
        for axis, (function, label) in zip(axes.ravel(), panels):
            for name, flights, colour in (("train", train, "#0072B2"), ("test (unseen shapes)", test, "#D55E00")):
                axis.hist(function(flights), 60, density=True, histtype="step", lw=1.5, color=colour, label=name)
            axis.set_title(label, fontsize=10)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
        figure.suptitle("state coverage by split: the envelopes overlap, the shapes do not")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(figure)
        plt.close(figure)

        # 5 one example flight per family
        for key in ("train", "test"):
            flights = train if key == "train" else test
            shown = {}
            for index, family in enumerate(families[key]):
                shown.setdefault(family, index)
            count = len(shown)
            figure, axes = plt.subplots(count, 3, figsize=(11, 1.9 * count), squeeze=False)
            for row, (family, index) in enumerate(sorted(shown.items())):
                flight = flights[index]
                axes[row][0].plot(times, flight[:, :3], lw=0.8)
                axes[row][0].set_ylabel(f"{family}\nx (m)", fontsize=7)
                axes[row][1].plot(times, flight[:, 12:15], lw=0.8)
                axes[row][1].set_ylabel("v_b (m/s)", fontsize=7)
                axes[row][2].plot(times, flight[:, 18], lw=0.8, color="k")
                axes[row][2].set_ylabel("T (N)", fontsize=7)
                for axis in axes[row]:
                    axis.grid(alpha=0.3)
                    axis.tick_params(labelsize=6)
            for axis in axes[-1]:
                axis.set_xlabel("t (s)", fontsize=7)
            figure.suptitle(f"{key}: one example 10 s flight per trajectory family", fontsize=11)
            figure.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(figure)
            plt.close(figure)
    print("wrote", arguments.out)


if __name__ == "__main__":
    main()

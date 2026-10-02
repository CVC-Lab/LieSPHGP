#!/usr/bin/env python3
"""Plot the noisy-training and clean/noisy-test quadrotor trajectories."""

from __future__ import annotations

import argparse
import pickle
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / (
    "datasets/data/pybullet_quadrotor/"
    "D0_CF2P_PIDplusEXC100Hz_PYB1000Hz_contact-free_nonlinear-damping-c0p5_"
    "162flights_1sec_h0p01_T0p05_nonoverlap5_"
    "train-obs-noise-absolute0p1_260907.pkl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "reports/SE3_Quadrotor/dataset_obs0p1_trajectories/"
    "quadrotor_obs0p1_train_test_trajectories.pdf"
)

STATE_GROUPS = (
    ("World position $p_w$", slice(0, 3), ("$p_x$", "$p_y$", "$p_z$"), "m"),
    ("Body linear velocity $v_b$", slice(12, 15), ("$v_x$", "$v_y$", "$v_z$"), "m/s"),
    (
        "Body angular velocity $\\omega_b$",
        slice(15, 18),
        ("$\\omega_x$", "$\\omega_y$", "$\\omega_z$"),
        "rad/s",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_pickle(path: Path) -> dict[str, Any]:
    with path.resolve().open("rb") as handle:
        return pickle.load(handle)


def rotations(data: np.ndarray) -> np.ndarray:
    return np.asarray(data[..., 3:12], dtype=np.float64).reshape(data.shape[:-1] + (3, 3))


def euler_zyx_degrees(data: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to roll, pitch, yaw using the ZYX convention."""
    rotation = rotations(data)
    roll = np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = np.arcsin(np.clip(-rotation[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
    return np.rad2deg(np.stack((roll, pitch, yaw), axis=-1))


def rotation_error_degrees(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    relative = np.matmul(np.swapaxes(rotations(clean), -1, -2), rotations(noisy))
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def validate_dataset(noisy: dict[str, Any], clean: dict[str, Any]) -> None:
    train = np.asarray(noisy["train_trajectories"])
    test = np.asarray(noisy["test_trajectories"])
    test_noisy = np.asarray(noisy["test_trajectories_noisy"])
    clean_train = np.asarray(clean["train_trajectories"])
    expected_train = (144, 101, 22)
    expected_test = (18, 101, 22)
    if train.shape != expected_train or clean_train.shape != expected_train:
        raise ValueError(f"Expected train shape {expected_train}; got {train.shape}")
    if test.shape != expected_test or test_noisy.shape != expected_test:
        raise ValueError(f"Expected test shape {expected_test}; got {test.shape}")
    noise = noisy["settings"]["observation_noise"]
    if not np.isclose(noise["level"], 0.1):
        raise ValueError(f"Expected observation noise 0.1; got {noise['level']}")
    np.testing.assert_array_equal(test, clean["test_trajectories"])
    np.testing.assert_array_equal(train[..., 18:22], clean_train[..., 18:22])
    np.testing.assert_array_equal(test_noisy[..., 18:22], test[..., 18:22])


def save_page(pdf: PdfPages, figure: plt.Figure) -> None:
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def add_cover_page(
    pdf: PdfPages,
    dataset_path: Path,
    noisy: dict[str, Any],
    train: np.ndarray,
    test: np.ndarray,
) -> None:
    settings = noisy["settings"]
    noise = settings["observation_noise"]
    composition = settings["training_composition"]
    dt = float(settings["sample_dt_seconds"])
    duration = (train.shape[1] - 1) * dt
    dataset_display = textwrap.fill(str(dataset_path.resolve()), width=105)
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.suptitle("Quadrotor dataset trajectories: observation noise $\\sigma_{obs}=0.1$", fontsize=20)
    text = (
        "Dataset inspected\n"
        f"{dataset_display}\n\n"
        "What is plotted\n"
        f"• Training: {train.shape[0]} trajectories × {train.shape[1]} points "
        f"({duration:.2f} s at $h={dt:.2f}$ s)\n"
        f"• Test: {test.shape[0]} clean trajectories and their separately saved noisy views\n"
        f"• Training composition: {composition['original_pid_flights']} PID + "
        f"{composition['pid_plus_excitation_flights']} PID-plus-excitation flights\n"
        "• State: $[p_w,\\,\\mathrm{vec}(R),\\,v_b,\\,\\omega_b]\\in\\mathbb{R}^{18}$\n"
        "• Control: $u=[T,\\tau_x,\\tau_y,\\tau_z]\\in\\mathbb{R}^{4}$\n\n"
        "Noise model\n"
        "• Euclidean channels: $\\tilde y_j=y_j+\\sigma_{obs}\\epsilon_j$, "
        "$\\epsilon_j\\sim\\mathcal{N}(0,1)$\n"
        "• Rotation: $\\tilde R=R\\operatorname{Exp}(\\widehat{\\eta})$, "
        "$\\eta\\sim\\mathcal{N}(0,\\sigma_{obs}^{2}I)$\n"
        f"• Stored noise level: $\\sigma_{{obs}}={noise['level']}$\n"
        "• Controls and time values are unchanged by the observation noise\n\n"
        "Line styles used throughout\n"
        "• Gray: clean trajectory\n"
        "• Blue: noisy PID training trajectory\n"
        "• Orange: noisy PID-plus-excitation training trajectory\n"
        "• Red: noisy test view"
    )
    figure.text(0.07, 0.86, text, va="top", fontsize=11.5, linespacing=1.45)
    pdf.savefig(figure)
    plt.close(figure)


def set_equal_3d_limits(axes: list[Any], blocks: tuple[np.ndarray, ...]) -> None:
    position = np.concatenate([block[..., :3].reshape(-1, 3) for block in blocks], axis=0)
    low = np.nanpercentile(position, 0.25, axis=0)
    high = np.nanpercentile(position, 99.75, axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 0.1)
    for axis in axes:
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_xlabel("$p_x$ [m]")
        axis.set_ylabel("$p_y$ [m]")
        axis.set_zlabel("$p_z$ [m]")


def add_3d_overview_page(
    pdf: PdfPages,
    clean_train: np.ndarray,
    noisy_train: np.ndarray,
    clean_test: np.ndarray,
    noisy_test: np.ndarray,
) -> None:
    figure = plt.figure(figsize=(16, 10))
    figure.suptitle("All train and test position trajectories in 3D", fontsize=18)
    axes = [figure.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    groups = (
        (clean_train[:72], "Clean source: 72 PID training flights", "0.45"),
        (noisy_train[:72], "Noisy training: 72 PID flights", "tab:blue"),
        (noisy_train[72:], "Noisy training: 72 PID-plus-excitation flights", "tab:orange"),
        (clean_test, "Test: 18 clean (gray) and noisy (red) views", "0.40"),
    )
    for axis, (data, title, color) in zip(axes, groups):
        for trajectory in data:
            axis.plot(*trajectory[:, :3].T, color=color, alpha=0.45, linewidth=0.8)
        axis.set_title(title)
    for trajectory in noisy_test:
        axes[3].plot(*trajectory[:, :3].T, color="tab:red", alpha=0.55, linewidth=0.8)
    set_equal_3d_limits(axes, (clean_train, noisy_train, clean_test, noisy_test))
    save_page(pdf, figure)


def add_state_time_page(
    pdf: PdfPages,
    time: np.ndarray,
    clean_train: np.ndarray,
    noisy_train: np.ndarray,
    clean_test: np.ndarray,
    noisy_test: np.ndarray,
    title: str,
    channel_slice: slice,
    labels: tuple[str, str, str],
    units: str,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    figure.suptitle(f"{title}: every trajectory", fontsize=18)
    clean_values = clean_train[..., channel_slice]
    train_values = noisy_train[..., channel_slice]
    test_values = clean_test[..., channel_slice]
    test_noisy_values = noisy_test[..., channel_slice]
    for component in range(3):
        left, right = axes[component]
        left.plot(time, clean_values[:, :, component].T, color="0.72", alpha=0.18, linewidth=0.55)
        left.plot(time, train_values[:72, :, component].T, color="tab:blue", alpha=0.25, linewidth=0.55)
        left.plot(time, train_values[72:, :, component].T, color="tab:orange", alpha=0.25, linewidth=0.55)
        right.plot(time, test_values[:, :, component].T, color="0.35", alpha=0.55, linewidth=0.8)
        right.plot(time, test_noisy_values[:, :, component].T, color="tab:red", alpha=0.38, linewidth=0.65)
        left.set_ylabel(f"{labels[component]} [{units}]")
        right.set_ylabel(f"{labels[component]} [{units}]")
        left.grid(alpha=0.25)
        right.grid(alpha=0.25)
    axes[0, 0].set_title("Training: clean source + noisy PID/PID-plus-excitation")
    axes[0, 1].set_title("Test: clean + separately saved noisy view")
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    save_page(pdf, figure)


def add_euler_page(
    pdf: PdfPages,
    time: np.ndarray,
    clean_train: np.ndarray,
    noisy_train: np.ndarray,
    clean_test: np.ndarray,
    noisy_test: np.ndarray,
) -> None:
    clean_train_euler = euler_zyx_degrees(clean_train)
    noisy_train_euler = euler_zyx_degrees(noisy_train)
    clean_test_euler = euler_zyx_degrees(clean_test)
    noisy_test_euler = euler_zyx_degrees(noisy_test)
    labels = ("roll $\\phi$", "pitch $\\theta$", "yaw $\\psi$")
    figure, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    figure.suptitle("Orientation trajectories: ZYX Euler-angle view", fontsize=18)
    for component in range(3):
        left, right = axes[component]
        left.plot(time, clean_train_euler[:, :, component].T, color="0.72", alpha=0.18, linewidth=0.55)
        left.plot(time, noisy_train_euler[:72, :, component].T, color="tab:blue", alpha=0.25, linewidth=0.55)
        left.plot(time, noisy_train_euler[72:, :, component].T, color="tab:orange", alpha=0.25, linewidth=0.55)
        right.plot(time, clean_test_euler[:, :, component].T, color="0.35", alpha=0.55, linewidth=0.8)
        right.plot(time, noisy_test_euler[:, :, component].T, color="tab:red", alpha=0.38, linewidth=0.65)
        left.set_ylabel(f"{labels[component]} [deg]")
        right.set_ylabel(f"{labels[component]} [deg]")
        left.grid(alpha=0.25)
        right.grid(alpha=0.25)
    axes[0, 0].set_title("Training")
    axes[0, 1].set_title("Test")
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    save_page(pdf, figure)


def add_control_page(
    pdf: PdfPages,
    time: np.ndarray,
    noisy_train: np.ndarray,
    clean_test: np.ndarray,
) -> None:
    labels = ("total thrust $T$", "$\\tau_x$", "$\\tau_y$", "$\\tau_z$")
    figure, axes = plt.subplots(4, 2, figsize=(16, 11), sharex=True)
    figure.suptitle("Recorded wrench controls (controls are not noised)", fontsize=18)
    for component in range(4):
        left, right = axes[component]
        left.plot(time, noisy_train[:72, :, 18 + component].T, color="tab:blue", alpha=0.27, linewidth=0.6)
        left.plot(time, noisy_train[72:, :, 18 + component].T, color="tab:orange", alpha=0.27, linewidth=0.6)
        right.plot(time, clean_test[:, :, 18 + component].T, color="0.3", alpha=0.6, linewidth=0.8)
        left.set_ylabel(labels[component])
        right.set_ylabel(labels[component])
        left.grid(alpha=0.25)
        right.grid(alpha=0.25)
    axes[0, 0].set_title("Training controls")
    axes[0, 1].set_title("Test controls")
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    save_page(pdf, figure)


def add_train_gallery_pages(
    pdf: PdfPages, clean_train: np.ndarray, noisy_train: np.ndarray
) -> None:
    per_page = 24
    for start in range(0, noisy_train.shape[0], per_page):
        stop = min(start + per_page, noisy_train.shape[0])
        figure = plt.figure(figsize=(16, 11))
        figure.suptitle(
            f"Training trajectory gallery: flights {start + 1}–{stop} "
            "(clean gray, noisy colored)",
            fontsize=16,
        )
        axes = []
        for local, index in enumerate(range(start, stop)):
            axis = figure.add_subplot(4, 6, local + 1, projection="3d")
            axes.append(axis)
            clean_position = clean_train[index, :, :3]
            noisy_position = noisy_train[index, :, :3]
            color = "tab:blue" if index < 72 else "tab:orange"
            axis.plot(*clean_position.T, color="0.55", linewidth=0.8, alpha=0.8)
            axis.plot(*noisy_position.T, color=color, linewidth=0.7, alpha=0.72)
            axis.scatter(*noisy_position[0], color="green", s=5)
            axis.set_title(f"train {index + 1}", fontsize=8)
            axis.tick_params(labelsize=5, pad=0)
            axis.set_xlabel("x", fontsize=6, labelpad=-4)
            axis.set_ylabel("y", fontsize=6, labelpad=-4)
            axis.set_zlabel("z", fontsize=6, labelpad=-4)
        save_page(pdf, figure)


def add_test_gallery_page(
    pdf: PdfPages, clean_test: np.ndarray, noisy_test: np.ndarray
) -> None:
    figure = plt.figure(figsize=(16, 10))
    figure.suptitle("Test trajectory gallery: clean gray, noisy red", fontsize=17)
    for index in range(clean_test.shape[0]):
        axis = figure.add_subplot(3, 6, index + 1, projection="3d")
        clean_position = clean_test[index, :, :3]
        noisy_position = noisy_test[index, :, :3]
        axis.plot(*clean_position.T, color="0.35", linewidth=1.0)
        axis.plot(*noisy_position.T, color="tab:red", linewidth=0.8, alpha=0.7)
        axis.scatter(*noisy_position[0], color="green", s=7)
        axis.set_title(f"test {index + 1}", fontsize=9)
        axis.tick_params(labelsize=5, pad=0)
        axis.set_xlabel("x", fontsize=6, labelpad=-4)
        axis.set_ylabel("y", fontsize=6, labelpad=-4)
        axis.set_zlabel("z", fontsize=6, labelpad=-4)
    save_page(pdf, figure)


def add_noise_diagnostics_page(
    pdf: PdfPages,
    clean_train: np.ndarray,
    noisy_train: np.ndarray,
    clean_test: np.ndarray,
    noisy_test: np.ndarray,
) -> None:
    channel_groups = (
        (slice(0, 3), "$p_w$ error [m]"),
        (slice(12, 15), "$v_b$ error [m/s]"),
        (slice(15, 18), "$\\omega_b$ error [rad/s]"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle("Observation-noise diagnostics", fontsize=18)
    for axis, (channel_slice, label) in zip(axes.flat[:3], channel_groups):
        train_delta = (noisy_train[..., channel_slice] - clean_train[..., channel_slice]).ravel()
        test_delta = (noisy_test[..., channel_slice] - clean_test[..., channel_slice]).ravel()
        axis.hist(train_delta, bins=80, density=True, color="tab:blue", alpha=0.55, label="train")
        axis.hist(test_delta, bins=80, density=True, color="tab:red", alpha=0.45, label="test")
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(
            f"{label}\ntrain std={np.std(train_delta):.4f}, test std={np.std(test_delta):.4f}"
        )
        axis.set_xlabel("noisy − clean")
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
        axis.legend()
    train_rotation_error = rotation_error_degrees(clean_train, noisy_train).ravel()
    test_rotation_error = rotation_error_degrees(clean_test, noisy_test).ravel()
    axis = axes.flat[3]
    axis.hist(train_rotation_error, bins=80, density=True, color="tab:blue", alpha=0.55, label="train")
    axis.hist(test_rotation_error, bins=80, density=True, color="tab:red", alpha=0.45, label="test")
    axis.set_title(
        "Rotation geodesic error\n"
        f"train mean={np.mean(train_rotation_error):.2f}°, "
        f"test mean={np.mean(test_rotation_error):.2f}°"
    )
    axis.set_xlabel("$d_{SO(3)}(R,\\tilde R)$ [deg]")
    axis.set_ylabel("density")
    axis.grid(alpha=0.2)
    axis.legend()
    save_page(pdf, figure)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()
    noisy = load_pickle(dataset_path)
    clean_path = Path(noisy["settings"]["clean_source_dataset_path"]).resolve()
    clean = load_pickle(clean_path)
    validate_dataset(noisy, clean)

    noisy_train = np.asarray(noisy["train_trajectories"], dtype=np.float64)
    clean_train = np.asarray(clean["train_trajectories"], dtype=np.float64)
    clean_test = np.asarray(noisy["test_trajectories"], dtype=np.float64)
    noisy_test = np.asarray(noisy["test_trajectories_noisy"], dtype=np.float64)
    dt = float(noisy["settings"]["sample_dt_seconds"])
    time = np.arange(noisy_train.shape[1], dtype=np.float64) * dt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        add_cover_page(pdf, dataset_path, noisy, noisy_train, clean_test)
        add_3d_overview_page(pdf, clean_train, noisy_train, clean_test, noisy_test)
        for title, channel_slice, labels, units in STATE_GROUPS:
            add_state_time_page(
                pdf,
                time,
                clean_train,
                noisy_train,
                clean_test,
                noisy_test,
                title,
                channel_slice,
                labels,
                units,
            )
        add_euler_page(pdf, time, clean_train, noisy_train, clean_test, noisy_test)
        add_control_page(pdf, time, noisy_train, clean_test)
        add_train_gallery_pages(pdf, clean_train, noisy_train)
        add_test_gallery_page(pdf, clean_test, noisy_test)
        add_noise_diagnostics_page(pdf, clean_train, noisy_train, clean_test, noisy_test)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

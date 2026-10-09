"""Is the IDSIA wrench right?  Four consistency checks of the converted input against the measured motion.

The input comes from MEASURED rotor speeds rather than a commanded PWM, so there is no thrust map to calibrate and
no command-to-delivery gap.  Needs the wrench-input dataset (convert_idsia.py default, input_mode "wrench").

  A  independent sensor: reconstructed T against m * a_z,body from the onboard accelerometer.
  B  parameter-free translational fit  dv_w/dt + g e_3 = (1/m) T R e_3, which returns 1/m.
  D  rotational consistency, integral form, with a trim torque and a lag scan.
  C  forward integration of the analytic rigid body from x0 driven by u alone, against a constant-velocity
     reference from the same x0.

Usage: python validate_idsia_input_reconstruction.py [--split test] [--raw envs/quadrotor_se3_idsia/idsia_raw/data]
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]  # envs/quadrotor_se3_idsia/datagen -> project root
DATASET_DIR = PROJECT_ROOT / "datasets/QUADROTOR-DATASET-IDSIA"

KT, KC, ARM = 3.72e-8, 7.74e-12, 0.0353
MASS, GRAVITY = 0.045, 9.81
PUBLISHED_INERTIA = np.array([2.3951e-5, 2.3951e-5, 3.2347e-6])
MOTOR_COLUMNS = [f"m{i}_rads" for i in (1, 2, 3, 4)]


def test_a(raw_dir: Path) -> None:
    ratios = []
    for name in sorted(glob.glob(str(raw_dir / "**/*.csv"), recursive=True)):
        frame = pd.read_csv(name)
        squared = frame[MOTOR_COLUMNS].to_numpy(dtype=float) ** 2
        thrust = KT * squared.sum(axis=1)
        specific = MASS * frame["az_body"].to_numpy()
        ratios.append(thrust.mean() / specific.mean())
    ratios = np.asarray(ratios)
    print(f"A  reconstructed T versus m * a_z,body, {len(ratios)} flights")
    print(f"     mean ratio {ratios.mean():.4f} +- {ratios.std():.4f}\n")


def test_b(flights: np.ndarray, step: float) -> None:
    rotation = flights[..., 3:12].reshape(flights.shape[0], flights.shape[1], 3, 3)
    velocity_world = np.einsum("ftij,ftj->fti", rotation, flights[..., 12:15])
    acceleration = np.gradient(velocity_world, step, axis=1)
    inner = 2
    left = (acceleration + np.array([0.0, 0.0, GRAVITY]))[:, inner:-inner].reshape(-1, 3)
    right = (flights[..., 18, None] * rotation[..., :, 2])[:, inner:-inner].reshape(-1, 3)
    gain = float(right.ravel() @ left.ravel() / (right.ravel() @ right.ravel()))
    residual = left - gain * right
    r_squared = 1.0 - (residual ** 2).sum() / ((left - left.mean(0)) ** 2).sum()
    print("B  parameter-free translational fit   dv_w/dt + g e_3 = (1/m) T R e_3")
    print(f"     implied mass {1 / gain * 1000:.2f} g   (published {MASS * 1000:.1f} g, "
          f"error {100 * (1 / gain - MASS) / MASS:+.1f} %)   R^2 {r_squared:.3f}\n")


def test_d(flights: np.ndarray, step: float, window: int = 10, lags=(0, 1, 2, 3, 5, 8)) -> None:
    omega_all, torque_all = flights[..., 15:18], flights[..., 19:22]
    print("D  rotational consistency (integral form, trim term included)")
    print("     " + "lag (ms)".ljust(10) + "R2 roll".rjust(9) + "R2 pitch".rjust(10) + "R2 yaw".rjust(9)
          + "Jxx".rjust(11) + "Jyy".rjust(11) + "Jzz".rjust(11))
    for lag in lags:
        omega = omega_all[:, lag:] if lag else omega_all
        torque = torque_all[:, : torque_all.shape[1] - lag] if lag else torque_all
        count = min(omega.shape[1], torque.shape[1])
        omega, torque = omega[:, :count], torque[:, :count]
        delta = omega[:, window:] - omega[:, :-window]
        torque_sum = np.cumsum(torque, axis=1) * step
        omega_sum = np.cumsum(omega, axis=1) * step
        integral_torque = torque_sum[:, window:] - torque_sum[:, :-window]
        integral_omega = omega_sum[:, window:] - omega_sum[:, :-window]
        inertia, r_squared, trim = np.zeros(3), np.zeros(3), np.zeros(3)
        for axis in range(3):
            design = np.column_stack([delta[..., axis].ravel(), integral_omega[..., axis].ravel(),
                                      np.full(delta[..., axis].size, window * step)])
            target = integral_torque[..., axis].ravel()
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            inertia[axis], trim[axis] = coefficients[0], coefficients[2]
            r_squared[axis] = 1.0 - ((target - design @ coefficients) ** 2).sum() / (
                (target - target.mean()) ** 2).sum()
        print(f"     {lag * step * 1000:<10.0f}{r_squared[0]:>9.3f}{r_squared[1]:>10.3f}{r_squared[2]:>9.3f}"
              f"{inertia[0]:>11.2e}{inertia[1]:>11.2e}{inertia[2]:>11.2e}")
    print(f"     published inertia {np.array2string(PUBLISHED_INERTIA, formatter={'float_kind': lambda v: f'{v:.2e}'})}")
    print(f"     fitted trim torque {np.array2string(trim, formatter={'float_kind': lambda v: f'{v:+.2e}'})} N m, "
          f"median |tau| {np.median(np.abs(torque_all)):.2e} N m\n")


def _hat(w: np.ndarray) -> np.ndarray:
    zero = np.zeros(w.shape[:-1])
    return np.stack([np.stack([zero, -w[..., 2], w[..., 1]], -1), np.stack([w[..., 2], zero, -w[..., 0]], -1),
                     np.stack([-w[..., 1], w[..., 0], zero], -1)], -2)


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi, axis=-1)[..., None, None]
    small = theta < 1e-8
    safe = np.where(small, 1.0, theta)
    skew = _hat(phi)
    a = np.where(small, 1.0, np.sin(safe) / safe)
    b = np.where(small, 0.5, (1.0 - np.cos(safe)) / safe ** 2)
    return np.eye(3) + a * skew + b * (skew @ skew)


def _rigid_body_rollout(flights: np.ndarray, step: float, damping: float, substeps: int = 4) -> np.ndarray:
    """Analytic rigid body with the published constants, damping -c m (1 + |v|) v and -c J (1 + |w|) w, input
    u = [T, tau] (body frame), integrated with Lie-group Heun; every flight from its first measured state."""
    inertia, inverse_inertia = np.diag(PUBLISHED_INERTIA), np.diag(1.0 / PUBLISHED_INERTIA)
    e3 = np.array([0.0, 0.0, 1.0])

    def rates(rotation, v, w, u):
        thrust, torque = u[:, :1], u[:, 1:4]
        dv = (-np.cross(w, v) - GRAVITY * np.einsum("fji,j->fi", rotation, e3) + thrust * e3 / MASS
              - damping * (1.0 + np.linalg.norm(v, axis=-1, keepdims=True)) * v)
        jw = w @ inertia
        dw = ((np.cross(jw, w) + torque) @ inverse_inertia
              - damping * (1.0 + np.linalg.norm(w, axis=-1, keepdims=True)) * w)
        return np.einsum("fij,fj->fi", rotation, v), dv, dw

    x, rotation = flights[:, 0, :3].copy(), flights[:, 0, 3:12].reshape(-1, 3, 3).copy()
    v, w = flights[:, 0, 12:15].copy(), flights[:, 0, 15:18].copy()
    path = np.empty(flights.shape[:2] + (3,))
    path[:, 0] = x
    h = step / substeps
    with np.errstate(all="ignore"):
        for k in range(1, flights.shape[1]):
            u = flights[:, k, 18:22]                                  # row k holds the input of interval k-1 -> k
            for _ in range(substeps):
                dx1, dv1, dw1 = rates(rotation, v, w, u)
                rotation_p = rotation @ _exp_so3(h * w)
                dx2, dv2, dw2 = rates(rotation_p, v + h * dv1, w + h * dw1, u)
                x = x + 0.5 * h * (dx1 + dx2)
                rotation = rotation @ _exp_so3(0.5 * h * (w + w + h * dw1))
                v, w = v + 0.5 * h * (dv1 + dv2), w + 0.5 * h * (dw1 + dw2)
            path[:, k] = x
    return path


def _valid_prediction_times(truth: np.ndarray, prediction: np.ndarray, step: float) -> np.ndarray:
    """Per flight: first time the position error exceeds 0.158 x the flight's RMS extent (the VPT convention)."""
    error = np.linalg.norm(prediction - truth[..., :3], axis=-1)
    extent = np.sqrt(np.mean(np.sum((truth[..., :3] - truth[:, :1, :3]) ** 2, axis=-1), axis=1))
    out = np.empty(truth.shape[0])
    for flight in range(truth.shape[0]):
        exceeded = np.flatnonzero(~(error[flight] <= 0.158 * extent[flight]))
        out[flight] = exceeded[0] * step if exceeded.size else (truth.shape[1] - 1) * step
    return out


def test_c(flights: np.ndarray, step: float, damping_values=(0.0, 0.25, 0.5), horizons=(0.1, 0.5, 1.0, 3.0)) -> None:
    truth = np.asarray(flights, dtype=np.float64)
    print("C  forward integration of the analytic rigid body, driven only by x0 and the reconstructed u")
    print(f"     {truth.shape[0]} flights, published mass {MASS * 1000:.1f} g and inertia")
    times = np.arange(truth.shape[1]) * step
    start_velocity = np.einsum("fij,fj->fi", truth[:, 0, 3:12].reshape(-1, 3, 3), truth[:, 0, 12:15])
    constant = truth[:, :1, :3] + start_velocity[:, None, :] * times[None, :, None]
    print("     " + "damping c".ljust(14) + "".join(f"{h:>10.1f} s" for h in horizons) + "     VPT median")
    for damping in damping_values:
        prediction = _rigid_body_rollout(truth, step, damping)
        if not np.all(np.isfinite(prediction[:, int(round(horizons[0] / step))])):
            print(f"     {damping:<14.2f}  diverged")
            continue
        errors = [float(np.sqrt(np.mean(np.sum((prediction[:, int(round(h / step))]
                                                - truth[:, int(round(h / step)), :3]) ** 2, axis=-1))))
                  for h in horizons]
        valid = _valid_prediction_times(truth, prediction, step)
        print(f"     {damping:<14.2f}" + "".join(f"{e:>12.3f}" for e in errors) + f"     {np.median(valid):.2f} s")
    reference = [float(np.sqrt(np.mean(np.sum((constant[:, int(round(h / step))]
                                               - truth[:, int(round(h / step)), :3]) ** 2, axis=-1))))
                 for h in horizons]
    print("     " + "constant v".ljust(14) + "".join(f"{e:>12.3f}" for e in reference) + "     (reference)")
    print("     position RMSE in metres\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR / "IDSIA_CF21BL_10s_h0p01_clean.pkl")
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "envs/quadrotor_se3_idsia/idsia_raw/data")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    arguments = parser.parse_args()
    with arguments.dataset.open("rb") as handle:
        payload = pickle.load(handle)
    input_mode = payload["settings"].get("input_mode", "wrench")
    if input_mode != "wrench":
        raise SystemExit(f"{arguments.dataset.name} has input_mode {input_mode!r}; these checks read columns 18:22 as "
                         "the wrench [T, tau_x, tau_y, tau_z], so use the wrench dataset")
    flights = payload[f"{arguments.split}_trajectories"]
    step = payload["settings"]["sample_dt"]
    print(f"\nValidating the IDSIA wrench on the {arguments.split} split: {flights.shape[0]} flights of "
          f"{(flights.shape[1] - 1) * step:.2f} s\n")
    if arguments.raw.is_dir():
        test_a(arguments.raw)
    test_b(flights, step)
    test_d(payload["train_trajectories"], step)
    test_c(flights, step)


if __name__ == "__main__":
    main()

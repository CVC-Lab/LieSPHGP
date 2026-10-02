"""Is the reconstructed wrench u right?  Three independent checks on the converted NanoBench flights.

The dataset records motor PWM, not force, so u = [T, tau] is reconstructed (see convert_nanobench.py).
Hover calibration alone only fixes the scalar thrust scale, so this script tests the reconstruction against
quantities it was never fitted to.

  A  independent sensor.  The IMU measures specific force; for a quadrotor its body-z component is T/m plus a
     small drag term.  Compares the reconstructed T against m * g * a_imu,z over many flights.

  B  parameter-free translational dynamics.  With no unknown constants at all,
         dv_w/dt + g e_3 = (T/m) R e_3 + drag/m,
     so a least-squares fit of the measured left side on T * R e_3 returns 1/m.  The implied mass is compared
     with the vehicle's published flying mass, 40.85 g.  This tests the thrust magnitude AND its direction,
     because a wrong attitude convention or a wrong mixer would destroy the fit.

  D  rotational consistency.  Integral form of J dw = tau - w x Jw - D w with a constant trim torque,
     scanned over a lag to cover the motor time constant, fitted on the training split.

  C  forward integration, the decisive test.  The analytic rigid body (the same GroundTruthSE3HamODE the
     simulator reports use as PH-GT) is driven from one measured initial state by the reconstructed u alone and
     compared with the measured flight.  Nothing about the trajectory enters except x0 and u.  A constant-
     velocity extrapolation from the same x0 is the reference: if the reconstruction carried no information,
     the physics rollout could not beat it.

Usage: python validate_input_reconstruction.py [--dataset <pkl>] [--raw <nanobench csv dir>]
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]

PWM2RPM_SCALE, PWM2RPM_CONST, KF, V_NOMINAL = 0.2685, 4070.3, 3.16e-10, 3.8
MASS, GRAVITY = 0.04085, 9.81
NOMINAL_INERTIA = np.array([2.3951e-5, 2.3951e-5, 3.2347e-5])


def test_a(raw_dir: Path, limit: int = 60) -> None:
    ratios, correlations = [], []
    for name in sorted(glob.glob(str(raw_dir / "*.csv")))[:limit]:
        frame = pd.read_csv(name)
        pwm = frame[[f"motor_motor_m{i}" for i in (1, 2, 3, 4)]].to_numpy(dtype=float)
        volts = frame["pwr_pm_vbat"].to_numpy()
        flying = (frame["pz"].to_numpy() > 0.35) & (pwm.sum(1) > 1000) & (pwm.max(1) < 65535)
        if flying.sum() < 500:
            continue
        thrust = (KF * (PWM2RPM_SCALE * pwm * (volts / V_NOMINAL)[:, None] + PWM2RPM_CONST) ** 2).sum(1)
        specific_force = MASS * GRAVITY * frame[["imu_acc_x", "imu_acc_y", "imu_acc_z"]].to_numpy()[:, 2]
        ratios.append(thrust[flying].mean() / specific_force[flying].mean())
        correlations.append(np.corrcoef(thrust[flying], specific_force[flying])[0, 1])
    ratios, correlations = np.asarray(ratios), np.asarray(correlations)
    print(f"A  reconstructed T versus the IMU specific force, {len(ratios)} flights")
    print(f"     mean ratio {ratios.mean():.3f} +- {ratios.std():.3f}   (1.000 = perfect)")
    print(f"     per-flight correlation: median {np.median(correlations):.3f}, "
          f"range {correlations.min():.3f}-{correlations.max():.3f}")
    print(f"     the IMU stream is radio-limited and interpolated, which caps the correlation, not the mean\n")


def test_b(flights: np.ndarray, step: float) -> None:
    rotation = flights[..., 3:12].reshape(flights.shape[0], flights.shape[1], 3, 3)
    velocity_world = np.einsum("ftij,ftj->fti", rotation, flights[..., 12:15])
    acceleration = np.gradient(velocity_world, step, axis=1)
    body_z = rotation[..., :, 2]                                  # third column of R = body z in world
    thrust = flights[..., 18]
    inner = 2                                                     # drop the finite-difference edges
    left = (acceleration + np.array([0.0, 0.0, GRAVITY]))[:, inner:-inner].reshape(-1, 3)
    right = (thrust[..., None] * body_z)[:, inner:-inner].reshape(-1, 3)
    gain = float(right.ravel() @ left.ravel() / (right.ravel() @ right.ravel()))
    residual = left - gain * right
    r_squared = 1.0 - (residual ** 2).sum() / ((left - left.mean(0)) ** 2).sum()
    per_axis = [1.0 - (residual[:, k] ** 2).sum() / ((left[:, k] - left[:, k].mean()) ** 2).sum() for k in range(3)]
    print("B  parameter-free translational fit   dv_w/dt + g e_3  =  (1/m) T R e_3")
    print(f"     implied mass 1/gain = {1 / gain * 1000:.2f} g   (published flying mass {MASS * 1000:.2f} g, "
          f"error {100 * (1 / gain - MASS) / MASS:+.1f} %)")
    print(f"     R^2 overall {r_squared:.3f}   per axis x/y/z {np.round(per_axis, 3).tolist()}")
    print("     the residual is the unmodelled drag plus Vicon differentiation noise\n")


def test_d(flights: np.ndarray, step: float, window: int = 10, lags=(0, 1, 2, 3, 4, 6, 8, 10)) -> None:
    """Rotational consistency in integral form, so omega is never differentiated twice.

        J (omega(t+W) - omega(t)) = int_t^{t+W} tau dt  -  D int omega dt  -  tau_trim * W

    A constant trim torque per axis is included because real motors are not matched, so the mixer's zero is
    not the vehicle's zero. The scan over lags covers the motor time constant, which the commanded PWM does
    not contain.
    """
    omega_all, torque_all = flights[..., 15:18], flights[..., 19:22]
    print("D  rotational consistency of the reconstructed torque (integral form, trim term included)")
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
    print(f"     nominal CF2 inertia {np.array2string(NOMINAL_INERTIA, formatter={'float_kind': lambda v: f'{v:.2e}'})}")
    print(f"     fitted trim torque {np.array2string(trim, formatter={'float_kind': lambda v: f'{v:+.2e}'})} N m, "
          f"against a median |tau| of {np.median(np.abs(torque_all)):.2e} N m\n")


def test_c(flights: np.ndarray, step: float, damping_values=(0.0, 0.25, 0.5, 1.0), horizons=(0.5, 1.0, 3.0)) -> None:
    import jax
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation
    from src.models.SE3_Quadrotor.comparision import open_loop as openloop

    truth = np.asarray(flights, dtype=np.float64)
    print("C  forward integration of the analytic rigid body, driven only by x0 and the reconstructed u")
    print(f"     {truth.shape[0]} flights, mass {MASS * 1000:.2f} g (published), inertia = nominal CF2 "
          f"(NOT measured for this airframe)")
    # Reference: constant-velocity extrapolation from the same initial state.
    times = np.arange(truth.shape[1]) * step
    start_velocity = np.einsum("fij,fj->fi", truth[:, 0, 3:12].reshape(-1, 3, 3), truth[:, 0, 12:15])
    constant = truth[:, :1, :3] + start_velocity[:, None, :] * times[None, :, None]
    header = "     " + "damping c".ljust(12) + "".join(f"{h:>10.1f} s" for h in horizons) + "      VPT median"
    print(header)
    for damping in damping_values:
        model = evaluation.ground_truth_model(
            {"mass": MASS, "inertia": np.diag(NOMINAL_INERTIA), "gravity": GRAVITY, "damping": damping})
        prediction, _ = openloop.timed_rollout(model, truth, step, repeats=1)
        if prediction is None:
            print(f"     {damping:<12.2f}  diverged")
            continue
        errors = [float(np.sqrt(np.mean(np.sum((prediction[:, int(round(h / step)), :3]
                                                - truth[:, int(round(h / step)), :3]) ** 2, axis=-1))))
                  for h in horizons]
        valid = openloop.valid_prediction_times(truth, prediction, step)
        print(f"     {damping:<12.2f}" + "".join(f"{e:>12.3f}" for e in errors) + f"      {np.median(valid):.2f} s")
    reference = [float(np.sqrt(np.mean(np.sum((constant[:, int(round(h / step))]
                                               - truth[:, int(round(h / step)), :3]) ** 2, axis=-1))))
                 for h in horizons]
    print("     " + "constant v".ljust(12) + "".join(f"{e:>12.3f}" for e in reference) + "      (reference)")
    print("     numbers are position RMSE in metres at that horizon\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=THIS_DIR / "NANOBENCH_CF2_10s_h0p01_clean.pkl")
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "tmp/nanobench_raw/datasets/dataset")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    arguments = parser.parse_args()
    with arguments.dataset.open("rb") as handle:
        payload = pickle.load(handle)
    flights = payload[f"{arguments.split}_trajectories"]
    step = payload["settings"]["sample_dt"]
    print(f"\nValidating the reconstructed wrench on the {arguments.split} split: "
          f"{flights.shape[0]} flights of {(flights.shape[1] - 1) * step:.2f} s\n")
    if arguments.raw.is_dir():
        test_a(arguments.raw)
    else:
        print(f"A  skipped: raw CSVs not found at {arguments.raw}\n")
    test_b(flights, step)
    test_d(payload['train_trajectories'], step)
    test_c(flights, step)


if __name__ == "__main__":
    main()

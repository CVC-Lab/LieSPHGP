"""Replay the real Crazyflie flights inside PyBullet and compare with the measurement.

An independent implementation check of the reconstructed input. Instead of our analytic port-Hamiltonian
vector field, the same motor commands are pushed through gym-pybullet-drones: PyBullet applies its own CF2X
rotor forces and torques (f_i = k_f rpm_i^2, tau from its own mixer) and integrates the rigid body itself, with
the vehicle mass and inertia overridden to the real values. If the two implementations agree, a mismatch with
the measurement is a property of the data, not of our code.

Per 100 Hz sample the motor speeds are
    rpm_i = 0.2685 * PWM_i * (V_bat / 3.8) + 4070.3
and the environment advances ten 1 kHz physics steps with them held constant, exactly as in the dataset
generators. The drone starts from the measured position, attitude, linear velocity and body rate.

Three variants isolate where the reconstruction fails:
    full        everything integrated by PyBullet from the initial state
    attitude    the measured attitude and body rate are written back every sample, so only the
                translational channel is being tested
    reference   constant-velocity extrapolation from the same initial state

Usage: python pybullet_roundtrip.py [--flights 25] [--seconds 3.0] [--inertia fitted|nominal]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]  # envs/quadrotor_se3/datagen/real -> project root
DATASET_DIR = PROJECT_ROOT / "datasets/QUADROTOR-DATASET-NANOBENCH"
for path in (PROJECT_ROOT, PROJECT_ROOT / "third_party/gym-pybullet-drones"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pybullet as pb  # noqa: E402
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from src.models.SE3_Quadrotor.comparision.report_controller import (  # noqa: E402
    configure_contact_free_dynamics, remove_ground_plane,
)

PWM2RPM_SCALE, PWM2RPM_CONST, V_NOMINAL = 0.2685, 4070.3, 3.8
MASS = 0.04085
NOMINAL_INERTIA = np.array([2.3951e-5, 2.3951e-5, 3.2347e-5])
FITTED_INERTIA = np.array([1.56e-5, 1.26e-5, 3.03e-6])     # from validate_input_reconstruction.py, test D
SAMPLE_HZ, PHYSICS_HZ = 100, 1000


def rebuild_motor_speeds(control: np.ndarray) -> np.ndarray:
    """The stored wrench is a linear map of the four motor thrusts; invert it to recover the rotor speeds.

    T = sum f_i, tau_x = a(-f1-f2+f3+f4), tau_y = a(-f1+f2+f3-f4), tau_z = kappa(-f1+f2-f3+f4)
    """
    from src.models.SE3_Quadrotor.comparision.report_controller import make_environment
    environment = make_environment()
    arm, ratio, kf = float(environment.L) / np.sqrt(2.0), float(environment.KM / environment.KF), float(environment.KF)
    environment.close()
    mixer = np.array([[1.0, 1.0, 1.0, 1.0],
                      [-arm, -arm, arm, arm],
                      [-arm, arm, arm, -arm],
                      [-ratio, ratio, -ratio, ratio]])
    thrusts = np.linalg.solve(mixer, control.reshape(-1, 4).T).T.reshape(control.shape)
    return np.sqrt(np.maximum(thrusts, 0.0) / kf)


def replay(flight: np.ndarray, rpm: np.ndarray, inertia: np.ndarray, steps: int, force_attitude: bool) -> np.ndarray:
    rotation = flight[:, 3:12].reshape(-1, 3, 3)
    quaternion = Rotation.from_matrix(rotation).as_quat()
    velocity_world = np.einsum("tij,tj->ti", rotation, flight[:, 12:15])
    omega_world = np.einsum("tij,tj->ti", rotation, flight[:, 15:18])
    environment = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=flight[0, :3].reshape(1, 3), initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB, pyb_freq=PHYSICS_HZ, ctrl_freq=SAMPLE_HZ,
        gui=False, record=False, obstacles=False, user_debug_gui=False)
    positions = np.zeros((steps + 1, 3))
    try:
        environment.reset()
        remove_ground_plane(environment)
        configure_contact_free_dynamics(environment)
        body = int(environment.DRONE_IDS[0])
        client = int(environment.CLIENT)
        pb.changeDynamics(body, -1, mass=MASS, localInertiaDiagonal=inertia.tolist(), physicsClientId=client)
        pb.resetBasePositionAndOrientation(body, flight[0, :3].tolist(), quaternion[0].tolist(), physicsClientId=client)
        pb.resetBaseVelocity(body, linearVelocity=velocity_world[0].tolist(),
                             angularVelocity=omega_world[0].tolist(), physicsClientId=client)
        environment._updateAndStoreKinematicInformation()
        positions[0] = flight[0, :3]
        for index in range(steps):
            if force_attitude and index > 0:
                state = environment._getDroneStateVector(0)
                pb.resetBasePositionAndOrientation(body, state[:3].tolist(), quaternion[index].tolist(),
                                                   physicsClientId=client)
                pb.resetBaseVelocity(body, linearVelocity=state[10:13].tolist(),
                                     angularVelocity=omega_world[index].tolist(), physicsClientId=client)
                environment._updateAndStoreKinematicInformation()
            observation, *_ = environment.step(rpm[index].reshape(1, 4))
            positions[index + 1] = observation[0][:3]
            if not np.all(np.isfinite(positions[index + 1])) or np.linalg.norm(positions[index + 1]) > 1e4:
                positions[index + 2:] = np.nan
                break
    finally:
        environment.close()
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR / "NANOBENCH_CF2_10s_h0p01_clean.pkl")
    parser.add_argument("--flights", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--inertia", choices=("fitted", "nominal"), default="fitted")
    arguments = parser.parse_args()
    with arguments.dataset.open("rb") as handle:
        payload = pickle.load(handle)
    flights = payload["test_trajectories"][: arguments.flights]
    step = payload["settings"]["sample_dt"]
    steps = int(round(arguments.seconds / step))
    inertia = FITTED_INERTIA if arguments.inertia == "fitted" else NOMINAL_INERTIA
    rpm = rebuild_motor_speeds(flights[..., 18:22])
    print(f"\nPyBullet replay of {flights.shape[0]} held-out flights, {arguments.seconds:.1f} s each")
    print(f"  mass {MASS * 1000:.2f} g (published), inertia {arguments.inertia} "
          f"{np.array2string(inertia, formatter={'float_kind': lambda v: f'{v:.2e}'})}")
    print(f"  motor speeds rebuilt from the stored wrench: {rpm.min():.0f}-{rpm.max():.0f} rpm\n")

    results = {}
    for name, force in (("full (PyBullet integrates attitude)", False), ("measured attitude written back", True)):
        errors = []
        for index, flight in enumerate(flights):
            predicted = replay(flight, rpm[index], inertia, steps, force)
            errors.append(np.linalg.norm(predicted - flight[: steps + 1, :3], axis=1))
        results[name] = np.asarray(errors)
        print(f"  {name}: done", flush=True)
    times = np.arange(flights.shape[1]) * step
    start_velocity = np.einsum("fij,fj->fi", flights[:, 0, 3:12].reshape(-1, 3, 3), flights[:, 0, 12:15])
    constant = flights[:, :1, :3] + start_velocity[:, None, :] * times[None, : steps + 1, None]
    results["constant velocity (reference)"] = np.linalg.norm(constant - flights[:, : steps + 1, :3], axis=2)

    horizons = [h for h in (0.1, 0.5, 1.0, 3.0) if h <= arguments.seconds]
    print("\n  " + "variant".ljust(38) + "".join(f"{h:>9.1f} s" for h in horizons) + "   finite")
    for name, error in results.items():
        cells = "".join(f"{np.sqrt(np.nanmean(error[:, int(round(h / step))] ** 2)):>11.3f}" for h in horizons)
        finite = 100.0 * np.isfinite(error[:, -1]).mean()
        print("  " + name.ljust(38) + cells + f"{finite:>8.0f} %")
    print("\n  position RMSE in metres over the held-out flights\n")


if __name__ == "__main__":
    main()

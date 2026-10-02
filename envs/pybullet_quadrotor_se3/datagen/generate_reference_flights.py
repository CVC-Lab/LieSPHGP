"""Clean 240 Hz PID reference flights for the open-loop report pages, D0 layout, any duration.

Reproduces the protocol of the common evaluation dataset
``D0_CF2P_PID_contact-free_nonlinear-damping-c0p5_seed0_260905.pkl`` (18 flights,
seed-0 initial/target conditions, CF2P, ``Physics.PYB`` at 240 Hz, ground plane removed,
contact friction zeroed, PyBullet built-in nonlinear damping c=0.5, DSLPIDControl at 48 Hz
with half-scale gains held for 5 physics steps, one initial zero-RPM step, wrench
u = A (KF rpm^2)) but with a configurable flight duration.  Every control window is stored
in the ``test_x`` split so ``report_evaluation.shared_truth`` can use the whole flight.

Usage:
    python envs/pybullet_quadrotor_se3/datagen/generate_reference_flights.py --duration-seconds 3.0
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # envs/pybullet_quadrotor_se3/datagen -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.SE3_Quadrotor.comparision.report_controller import (  # noqa: E402
    PYBULLET_DRONES_DIR,
    configure_contact_free_dynamics,
    contact_point_count,
    remove_ground_plane,
    apply_linear_damping,
)
from src.models.SE3_Quadrotor.comparision.report_evaluation import DAMPING_COEFFICIENT, HOLD_STEPS, PHYSICS_HZ, PID_HZ

import pybullet as pb  # noqa: E402
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl  # noqa: E402
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "datasets/QUADROTOR-EVAL-REFERENCE"
STATE_CONTROL_DIM = 22
PID_ATTRIBUTES = ("P_COEFF_FOR", "I_COEFF_FOR", "D_COEFF_FOR", "P_COEFF_TOR", "I_COEFF_TOR", "D_COEFF_TOR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--trajectory", choices=("hold", "varying"), default="hold",
                        help="hold = fly to a fixed target then hover (D0 protocol); varying = a figure-eight with "
                             "vertical and yaw motion for the whole flight, so the drone never settles")
    parser.add_argument("--damping-law", choices=("nonlinear", "linear"), default="nonlinear",
                        help="nonlinear = PyBullet built-in c(1+|v|); linear = built-in off, external -c m v / -c J omega")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--varying-parameters", type=str, default=None,
                        help="comma-separated key=value overrides for the varying trajectory, e.g. "
                             "'radius=1.6,period=7.0,height=0.7,height_period=5.5,yaw=0.9,yaw_period=9.5'. "
                             "Unlisted keys keep their default, so the released flights stay reproducible.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


# --- D0 protocol -----------------------------------------------------------------
def target_offsets() -> np.ndarray:
    values_x = np.arange(-1.0, 2.0, 1.0)
    values_y = np.arange(-1.0, 2.0, 1.0)
    values_z = 0.75 * np.arange(0.5, 1.5, 0.5)
    grid_x, grid_y, grid_z = np.meshgrid(values_x, values_y, values_z)
    result = np.zeros((grid_x.size, 3))
    result[:, 0], result[:, 1], result[:, 2] = grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)
    return result


def conditions(seed: int) -> dict[str, np.ndarray]:
    """Seed-0 conditions of the D0 generator, translated one metre above the plane."""
    rng = np.random.default_rng(seed)
    base_xy = 0.4 * np.array([[-1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]], dtype=np.float64)
    initial_xy = base_xy + rng.normal(0.0, 0.2, size=(2, 4))
    offsets = target_offsets()
    n_flights = len(offsets)
    initial_xyz = np.zeros((n_flights, 3)); initial_rpy = np.zeros((n_flights, 3))
    target_xyz = np.zeros((n_flights, 3)); target_rpy = np.zeros((n_flights, 3))
    for index in range(n_flights):
        initial_xyz[index] = [initial_xy[0, index % 4], initial_xy[1, index % 4], 0.0]
        initial_rpy[index, 2] = (index % 3) * 30.0 * np.pi / 180.0
        target_xyz[index] = initial_xyz[index] + offsets[index]
        target_rpy[index, 2] = (index % 3) * 15.0 * np.pi / 180.0
    initial_xyz[:, 2] += 1.0
    target_xyz[:, 2] += 1.0
    return {"initial_xyz": initial_xyz, "initial_rpy": initial_rpy, "target_offset": offsets,
            "target_xyz": target_xyz, "target_rpy": target_rpy}


def half_scale_pid(controller: DSLPIDControl) -> dict[str, list[float]]:
    for attribute in PID_ATTRIBUTES:
        setattr(controller, attribute, 0.5 * np.asarray(getattr(controller, attribute)))
    return {attribute: np.asarray(getattr(controller, attribute), dtype=np.float64).tolist() for attribute in PID_ATTRIBUTES}


def conversion_matrix(env: CtrlAviary) -> np.ndarray:
    ratio = env.KM / env.KF
    return np.array([[1.0, 1.0, 1.0, 1.0], [0.0, env.L, 0.0, -env.L], [-env.L, 0.0, env.L, 0.0],
                     [-ratio, ratio, -ratio, ratio]], dtype=np.float64)


def pack_sample(state: np.ndarray, wrench: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pb.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
    velocity_body = rotation.T @ np.asarray(state[10:13])
    omega_body = rotation.T @ np.asarray(state[13:16])
    return np.concatenate([state[:3], rotation.reshape(9), velocity_body, omega_body, wrench])


# Varying reference: incommensurate periods, so the state never repeats and never settles.
VARYING = {"radius": 1.2, "period": 5.0, "height": 0.4, "height_period": 4.0, "yaw": 0.6, "yaw_period": 6.0}


def varying_target(time_seconds: float, centre_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Figure-eight in xy, sinusoidal height and yaw, centred on the flight's own target point."""
    omega = 2.0 * np.pi / VARYING["period"]
    a = VARYING["radius"]
    offset = np.array([
        a * np.sin(omega * time_seconds),
        0.5 * a * np.sin(2.0 * omega * time_seconds),
        VARYING["height"] * np.sin(2.0 * np.pi / VARYING["height_period"] * time_seconds),
    ])
    yaw = VARYING["yaw"] * np.sin(2.0 * np.pi / VARYING["yaw_period"] * time_seconds)
    return centre_xyz + offset, np.array([0.0, 0.0, yaw])


def make_environment(initial_xyz: np.ndarray, initial_rpy: np.ndarray) -> CtrlAviary:
    return CtrlAviary(drone_model=DroneModel.CF2P, num_drones=1, initial_xyzs=initial_xyz.reshape(1, 3),
                      initial_rpys=initial_rpy.reshape(1, 3), physics=Physics.PYB, pyb_freq=PHYSICS_HZ,
                      ctrl_freq=PHYSICS_HZ, gui=False, record=False, obstacles=False, user_debug_gui=False)


def generate_flight(initial_xyz, initial_rpy, target_xyz, target_rpy, *, control_windows: int, reset_seed: int,
                    damping_law: str = "nonlinear", trajectory: str = "hold"):
    env = make_environment(initial_xyz, initial_rpy)
    controller = DSLPIDControl(drone_model=DroneModel.CF2P)
    gains = half_scale_pid(controller)
    mixer = conversion_matrix(env)
    collected = np.zeros((HOLD_STEPS, control_windows, STATE_CONTROL_DIM), dtype=np.float64)
    minimum_rpm, maximum_rpm, maximum_contacts, minimum_altitude = np.inf, -np.inf, 0, np.inf
    try:
        observation, _ = env.reset(seed=reset_seed)
        no_ground_audit = remove_ground_plane(env)
        dynamics_audit = configure_contact_free_dynamics(env, damping_law)
        dynamics_audit["no_ground"] = no_ground_audit
        observation, *_ = env.step(np.zeros((1, 4), dtype=np.float64))  # one initial zero-RPM step, as in D0
        state = observation[0]
        maximum_contacts = max(maximum_contacts, contact_point_count(env))
        minimum_altitude = min(minimum_altitude, float(state[2]))
        for window in range(control_windows):
            if trajectory == "varying":
                window_xyz, window_rpy = varying_target(window / PID_HZ, target_xyz)
            else:
                window_xyz, window_rpy = target_xyz, target_rpy
            rpm, _, _ = controller.computeControlFromState(control_timestep=1.0 / PID_HZ, state=state,
                                                           target_pos=window_xyz, target_rpy=window_rpy)
            rpm = np.asarray(rpm, dtype=np.float64)
            minimum_rpm, maximum_rpm = min(minimum_rpm, float(rpm.min())), max(maximum_rpm, float(rpm.max()))
            wrench = mixer @ (rpm**2 * env.KF)
            for phase in range(HOLD_STEPS):
                if damping_law == "linear":
                    apply_linear_damping(env, state)
                observation, *_ = env.step(rpm.reshape(1, 4))
                state = observation[0]
                contacts = contact_point_count(env)
                maximum_contacts = max(maximum_contacts, contacts)
                minimum_altitude = min(minimum_altitude, float(state[2]))
                if contacts:
                    raise RuntimeError(f"Ground contact in contact-free flight: count={contacts}")
                collected[phase, window] = pack_sample(state, wrench)
        parameters = {"mass": float(env.M), "arm_length": float(env.L), "inertia": np.asarray(env.J, dtype=np.float64).tolist(),
                      "kf": float(env.KF), "km": float(env.KM), "gravity_acceleration": float(env.G),
                      "hover_rpm": float(env.HOVER_RPM), "maximum_rpm": float(env.MAX_RPM)}
    finally:
        env.close()
    return collected, {"pid_gains": gains, "vehicle_parameters": parameters, "minimum_pid_rpm": minimum_rpm,
                       "maximum_pid_rpm": maximum_rpm, "maximum_contact_points": maximum_contacts,
                       "minimum_altitude_m": minimum_altitude, "dynamics_audit": dynamics_audit}


def git_revision(path: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        return {"path": str(path), "commit": commit}
    except Exception as error:  # noqa: BLE001
        return {"path": str(path), "error": str(error)}


def generate_dataset(duration_seconds: float, seed: int, damping_law: str = "nonlinear", trajectory: str = "hold") -> dict:
    control_windows = int(round(duration_seconds * PID_HZ))
    if abs(control_windows / PID_HZ - duration_seconds) > 1e-9:
        raise ValueError(f"duration {duration_seconds} s is not a multiple of the {1 / PID_HZ:.6f} s PID period")
    selected = conditions(seed)
    flights = len(selected["initial_xyz"])
    complete = np.zeros((flights, HOLD_STEPS, control_windows, STATE_CONTROL_DIM), dtype=np.float64)
    audits = []
    for index in range(flights):
        print(f"Generating reference flight {index + 1:02d}/{flights:02d} ({duration_seconds:g} s)", flush=True)
        complete[index], info = generate_flight(selected["initial_xyz"][index], selected["initial_rpy"][index],
                                                selected["target_xyz"][index], selected["target_rpy"][index],
                                                control_windows=control_windows, reset_seed=seed + index,
                                                damping_law=damping_law, trajectory=trajectory)
        audits.append(info)
    zero_contacts = all(info["maximum_contact_points"] == 0 for info in audits)
    if not zero_contacts:
        raise AssertionError("At least one contact event occurred")
    settings = {
        "dataset_name": f"D0-CF2P-PID-contact-free-{damping_law}-damping-c0p5-{duration_seconds:g}s-{trajectory}",
        "trajectory": trajectory,
        "trajectory_parameters": (dict(VARYING) if trajectory == "varying" else
                                  {"note": "fixed target, the drone settles and hovers for the rest of the flight"}),
        "format_version": 4,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "current local Gym-PyBullet-Drones CtrlAviary",
        "generator": str(Path(__file__).resolve()),
        "released_d0_pickle_used": False,
        "seed": seed,
        "drone_model": "CF2P",
        "physics": "Physics.PYB",
        "physics_hz": PHYSICS_HZ,
        "pid_hz": PID_HZ,
        "control_hold_steps": HOLD_STEPS,
        "duration_seconds": duration_seconds,
        "control_windows": control_windows,
        "train_windows": 0,
        "test_windows": control_windows,
        "split_semantics": "every control window is in test_x; x is empty (reference flights for the open-loop report only)",
        "initial_zero_rpm_steps": 1,
        "obstacles": False,
        "ground_plane_removed": True,
        "collision_environment": "no ground plane and no obstacle bodies",
        "contact_free": True,
        "zero_contact_events": zero_contacts,
        "contact_friction_disabled": True,
        "minimum_altitude_m": min(info["minimum_altitude_m"] for info in audits),
        "damping_law": damping_law,
        "linear_damping_coefficient": DAMPING_COEFFICIENT,
        "angular_damping_coefficient": DAMPING_COEFFICIENT,
        "damping_equation": ("a=-c*(1+norm(v))*v and omega_dot damping=-c*(1+norm(omega))*omega" if damping_law == "nonlinear"
                             else "a=-c*v and tau=-c*J*omega"),
        "pybullet_builtin_damping": damping_law == "nonlinear",
        "custom_external_linear_damping": damping_law == "linear",
        "recording_semantics": "step first, then record state and current PID wrench",
        "state_layout": "x_w(3), vec(R)(9), v_b(3), omega_b(3), wrench(4)",
        "control_layout": "total thrust, body tau_x, body tau_y, body tau_z",
        "control_generation": "DSLPIDControl RPM converted by u=A*(KF*RPM^2)",
        "pid_gains": audits[0]["pid_gains"],
        "vehicle_parameters": audits[0]["vehicle_parameters"],
        "minimum_pid_rpm": min(info["minimum_pid_rpm"] for info in audits),
        "maximum_pid_rpm": max(info["maximum_pid_rpm"] for info in audits),
        "initial_xyz": selected["initial_xyz"].tolist(),
        "initial_rpy": selected["initial_rpy"].tolist(),
        "target_offset": selected["target_offset"].tolist(),
        "target_xyz": selected["target_xyz"].tolist(),
        "target_rpy": selected["target_rpy"].tolist(),
        "flight_audits": audits,
        "gym_pybullet_drones_repository": git_revision(PYBULLET_DRONES_DIR),
        "pybullet_api_version": int(pb.getAPIVersion()),
        "numpy_version": np.__version__,
        "python_version": sys.version,
    }
    return {"x": complete[:, :, :0, :], "test_x": complete, "t": np.arange(HOLD_STEPS, dtype=np.float64) / PHYSICS_HZ,
            "settings": settings}


def apply_varying_overrides(text: str | None) -> None:
    """Override entries of ``VARYING`` in place; the settings record the values actually used."""
    if not text:
        return
    for item in text.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in VARYING:
            raise ValueError(f"Unknown varying parameter {key!r}; expected one of {sorted(VARYING)}")
        VARYING[key] = float(value)


def main() -> None:
    args = parse_args()
    apply_varying_overrides(args.varying_parameters)
    tag = "" if args.trajectory == "hold" else f"_{args.trajectory}"
    output = args.output or DATASET_ROOT / (
        f"D0_CF2P_PID_contact-free_{args.damping_law}-damping-c0p5_{args.duration_seconds:g}s{tag}_seed{args.seed}.pkl")
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to overwrite")
    data = generate_dataset(args.duration_seconds, args.seed, args.damping_law, args.trajectory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    states = data["test_x"].shape
    print(f"Wrote {output}\n  test_x shape {states} = {states[0]} flights x {states[1] * states[2]} states at 1/{PHYSICS_HZ} s"
          f" ({(states[1] * states[2] - 1) / PHYSICS_HZ:.4f} s horizon)\n  sha256 {digest}", flush=True)


if __name__ == "__main__":
    main()

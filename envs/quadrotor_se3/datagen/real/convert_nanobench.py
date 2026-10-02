"""NanoBench (real Crazyflie 2.1 flights) -> the 22-channel layout used by the SE(3) quadrotor models.

Source: Ullah & Baca, "NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor System Identification,
Control, and State Estimation", arXiv:2603.09908, https://github.com/syediu/nanobench (BSD 3-Clause).
107 flights of a Crazyflie 2.1 under Vicon, resampled to a uniform 100 Hz time base, flying mass 40.85 g
(stock 27 g plus markers and the wireless-charging deck).

Output layout (identical to QUADROTOR-DATASET-HARD, so the trainers and report tooling need no change):

    s = [ x_w(3), vec(R)(9), v_b(3), omega_b(3), u(4) ]   with u = [T, tau_x, tau_y, tau_z]

STATE (all from Vicon, one sensor and one clock):
    x_w      px, py, pz                                     (m, world)
    R        from the Vicon quaternion (qx,qy,qz,qw)         (body -> world, stored row-major)
    v_b      R^T [vx,vy,vz]                                  (the CSV velocity is world frame)
    omega_b  R^T [wx_vicon,wy_vicon,wz_vicon]                (the CSV rate is world frame)
  The onboard gyro is NOT used: the firmware telemetry is radio-bandwidth limited and interpolated, and it
  agrees with d(R)/dt only at corr 0.80 versus 0.97 for the Vicon rate (yaw-rate amplitude 0.63x).
  Nothing is smoothed; the Vicon rate carries ~0.10 rad/s of differentiation jitter, which the NLL's learned
  per-block sigma is meant to absorb.

INPUT (per-motor PWM -> wrench), two steps:
  1. Battery-compensated Foerster motor model.  The stock relation rpm = S*PWM + C (Foerster 2015; the same
     constants the CF2 URDF and gym-pybullet-drones use) holds at one supply voltage only: the motor sees
     duty x V_bat, and the firmware raises PWM as the cell drains, so PWM alone over-states thrust by ~20 %
     at the end of a discharge.  Using

         rpm_i = S * PWM_i * (V_bat / V_nom) + C,     f_i = k_f * rpm_i^2,

     with V_nom = 3.8 V removes the drift: over the two battery-drain hovers the implied hover thrust is
     0.96-1.00 x m g (nominal model: 1.18-1.23) and its coefficient of variation falls from 0.039-0.052 to
     0.009-0.010, with the first-decile / last-decile ratio going from 0.86-0.90 to 0.98-0.99.
  2. CF2X mixer (the Crazyflie 2.1 is an X frame), the gym-pybullet-drones convention:

         T = sum f_i,   tau_x = a(-f1-f2+f3+f4),   tau_y = a(-f1+f2+f3-f4),   tau_z = (k_m/k_f)(-f1+f2-f3+f4)

     with a = L/sqrt(2).  The roll and pitch sign patterns are confirmed from the data itself: regressing the
     Vicon angular acceleration on the four motor thrusts returns the patterns (-,-,+,+) and (-,+,+,-).
     Any residual constant error in this map is absorbed by the learned control level Lambda_0, since only the
     product M^-1 g is identifiable.

TRIMMING: the longest contiguous airborne run (pz > 0.35 m and the motors commanded above idle), minus a
0.5 s margin at each end, cut into non-overlapping 1001-sample (10.01 s) flights.  A chunk is dropped if more
than 2 % of its samples have a motor at the PWM ceiling (the commanded value is then not what was delivered).

SPLIT: by trajectory family, train and test share no shape.
    train   multisine excitation, circle, figure-eight, oval, linear ramp, staircase climb,
            random waypoints, battery-drain hover
    test    helix, star, trefoil knot, lissajous          (up to --test-flights-per-family flights each)

Usage:  python convert_nanobench.py [--source tmp/nanobench_raw/datasets/dataset] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]  # envs/quadrotor_se3/datagen/real -> project root
DATASET_DIR = PROJECT_ROOT / "datasets/QUADROTOR-DATASET-NANOBENCH"

# ----------------------------------------------------------------- motor model
PWM2RPM_SCALE = 0.2685        # Foerster 2015 / CF2 URDF, also gym-pybullet-drones DSLPIDControl
PWM2RPM_CONST = 4070.3
KF = 3.16e-10                 # N / rpm^2
KM = 7.94e-12                 # N m / rpm^2
ARM_LENGTH = 0.0397           # m
V_NOMINAL = 3.8               # V, the battery compensation reference (see the module docstring)
PWM_CEILING = 65535

# ------------------------------------------------------------------- selection
TRAIN_FAMILIES = ("multisine_sysid", "circle", "figure8", "oval", "linear_ramp",
                  "staircase_climb", "random_waypoints", "battery_drain")
TEST_FAMILIES = ("helix", "star", "trefoil", "lissajous")

FLIGHT_POINTS = 1001          # 10.01 s at 100 Hz, the same flight length as QUADROTOR-DATASET-HARD
SAMPLE_HZ = 100
ALTITUDE_FLOOR = 0.35         # m, above the ground-effect region
EDGE_TRIM_SECONDS = 0.5
MAX_SATURATED_FRACTION = 0.02

MOTOR_COLUMNS = [f"motor_motor_m{i}" for i in (1, 2, 3, 4)]
MASS_KG = 0.04085             # measured instrumented flying mass (NanoBench README)
GRAVITY = 9.81
NOMINAL_INERTIA = (2.3951e-5, 2.3951e-5, 3.2347e-5)   # stock CF2 URDF; NOT measured for this airframe


def family_of(path: Path) -> str:
    """Trajectory family from the file name, e.g. B9_trefoil_fast_rep3.csv -> trefoil."""
    stem = path.stem
    # Speed and controller suffixes (possibly several, e.g. B5_helix_fast_mell_rep1) are not part of the shape.
    match = re.match(r"[A-Z]\d+[a-z]?_(.+?)(?:_(?:slow|medium|fast|mellinger|mell|pid))*_rep\d+$", stem)
    if match is None:
        raise ValueError(f"cannot parse the trajectory family from {path.name}")
    return match.group(1)


def motor_thrusts(pwm: np.ndarray, battery_volts: np.ndarray) -> np.ndarray:
    """Per-motor thrust in newtons from the commanded PWM and the measured cell voltage."""
    rpm = PWM2RPM_SCALE * pwm * (battery_volts / V_NOMINAL)[:, None] + PWM2RPM_CONST
    return KF * rpm**2


def wrench_from_thrusts(thrusts: np.ndarray) -> np.ndarray:
    """CF2X mixer: [T, tau_x, tau_y, tau_z] from the four motor thrusts."""
    arm = ARM_LENGTH / np.sqrt(2.0)
    ratio = KM / KF
    f1, f2, f3, f4 = thrusts.T
    return np.column_stack([
        f1 + f2 + f3 + f4,
        arm * (-f1 - f2 + f3 + f4),
        arm * (-f1 + f2 + f3 - f4),
        ratio * (-f1 + f2 - f3 + f4),
    ])


def longest_airborne_run(altitude: np.ndarray, pwm: np.ndarray) -> tuple[int, int]:
    """Start and stop index (exclusive) of the longest contiguous airborne stretch."""
    flying = (altitude > ALTITUDE_FLOOR) & (pwm.sum(axis=1) > 1000)
    indices = np.flatnonzero(flying)
    if indices.size == 0:
        return 0, 0
    pieces = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    longest = max(pieces, key=len)
    margin = int(round(EDGE_TRIM_SECONDS * SAMPLE_HZ))
    return int(longest[0]) + margin, int(longest[-1]) + 1 - margin


def convert_flight(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (chunks, audit) where chunks has shape (n, FLIGHT_POINTS, 22)."""
    frame = pd.read_csv(path)
    pwm = frame[MOTOR_COLUMNS].to_numpy(dtype=np.float64)
    altitude = frame["pz"].to_numpy(dtype=np.float64)
    start, stop = longest_airborne_run(altitude, pwm)
    audit: dict[str, Any] = {
        "file": path.name, "family": family_of(path), "rows": int(len(frame)),
        "airborne_start": start, "airborne_stop": stop,
        "airborne_seconds": round(max(stop - start, 0) / SAMPLE_HZ, 2),
        "battery_volts": [round(float(frame["pwr_pm_vbat"].iloc[0]), 3),
                          round(float(frame["pwr_pm_vbat"].iloc[-1]), 3)],
    }
    if stop - start < FLIGHT_POINTS:
        audit["kept_chunks"], audit["rejected_saturated"] = 0, 0
        return np.zeros((0, FLIGHT_POINTS, 22)), audit

    window = slice(start, stop)
    rotation = Rotation.from_quat(frame[["qx", "qy", "qz", "qw"]].to_numpy()[window]).as_matrix()
    position = frame[["px", "py", "pz"]].to_numpy(dtype=np.float64)[window]
    velocity_world = frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)[window]
    omega_world = frame[["wx_vicon", "wy_vicon", "wz_vicon"]].to_numpy(dtype=np.float64)[window]
    velocity_body = np.einsum("nji,nj->ni", rotation, velocity_world)
    omega_body = np.einsum("nji,nj->ni", rotation, omega_world)
    control = wrench_from_thrusts(motor_thrusts(pwm[window], frame["pwr_pm_vbat"].to_numpy()[window]))
    states = np.concatenate([position, rotation.reshape(-1, 9), velocity_body, omega_body, control], axis=1)
    saturated = (pwm[window] >= PWM_CEILING).any(axis=1)

    chunks, rejected = [], 0
    for begin in range(0, states.shape[0] - FLIGHT_POINTS + 1, FLIGHT_POINTS):
        piece = slice(begin, begin + FLIGHT_POINTS)
        if saturated[piece].mean() > MAX_SATURATED_FRACTION:
            rejected += 1
            continue
        chunks.append(states[piece])
    audit["kept_chunks"], audit["rejected_saturated"] = len(chunks), rejected
    return (np.stack(chunks) if chunks else np.zeros((0, FLIGHT_POINTS, 22))), audit


def sliding_windows(flights: np.ndarray, points: int, stride: int) -> np.ndarray:
    """(flight, time, 22) -> (points, windows, 22), the stored short-window tensor."""
    if flights.shape[0] == 0:
        return np.zeros((points, 0, 22))
    pieces = [flight[start:start + points] for flight in flights
              for start in range(0, flights.shape[1] - points + 1, stride)]
    return np.transpose(np.stack(pieces), (1, 0, 2))


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def coverage(flights: np.ndarray) -> dict[str, Any]:
    if flights.shape[0] == 0:
        return {}
    flat = flights.reshape(-1, 22)
    speed = np.linalg.norm(flat[:, 12:15], axis=1)
    rate = np.linalg.norm(flat[:, 15:18], axis=1)
    tilt = np.degrees(np.arccos(np.clip(flat[:, 11], -1.0, 1.0)))
    thrust = flat[:, 18] / (MASS_KG * GRAVITY)
    rotations = flat[:, 3:12].reshape(-1, 3, 3)
    orthogonality = np.linalg.norm(
        np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3), axis=(1, 2)).max()
    percentiles = [50, 90, 100]
    return {
        "speed_percentiles_50_90_max": [float(np.percentile(speed, q)) for q in percentiles],
        "angular_rate_percentiles_50_90_max": [float(np.percentile(rate, q)) for q in percentiles],
        "tilt_deg_percentiles_50_90_max": [float(np.percentile(tilt, q)) for q in percentiles],
        "thrust_over_weight_mean_cv": [float(thrust.mean()), float(thrust.std() / thrust.mean())],
        "thrust_over_weight_below_half": float((thrust < 0.5).mean()),
        "thrust_over_weight_within_10_percent_of_hover": float((np.abs(thrust - 1.0) < 0.1).mean()),
        "position_min": flat[:, :3].min(axis=0).tolist(),
        "position_max": flat[:, :3].max(axis=0).tolist(),
        "max_orthogonality_frobenius": float(orthogonality),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "tmp/nanobench_raw/datasets/dataset")
    parser.add_argument("--output", type=Path, default=DATASET_DIR)
    parser.add_argument("--test-flights-per-family", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    files = sorted(source.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV flights under {source}")

    shared = sorted(set(TRAIN_FAMILIES) & set(TEST_FAMILIES))
    if shared:
        raise ValueError(f"train and test libraries must share no trajectory family; shared: {shared}")

    by_family: dict[str, list[Path]] = {}
    for path in files:
        by_family.setdefault(family_of(path), []).append(path)
    unknown = sorted(set(by_family) - set(TRAIN_FAMILIES) - set(TEST_FAMILIES))
    if unknown:
        raise ValueError(f"unassigned trajectory families: {unknown}")

    selected = {"train": [], "test": []}
    for family in sorted(by_family):
        chosen = by_family[family]
        if family in TEST_FAMILIES:
            chosen = chosen[: int(arguments.test_flights_per_family)]
        selected["train" if family in TRAIN_FAMILIES else "test"].extend(chosen)

    split_flights, split_audits = {}, {}
    for split, paths in selected.items():
        pieces, audits = [], []
        for path in paths:
            chunks, audit = convert_flight(path)
            audits.append(audit)
            if chunks.shape[0]:
                pieces.append(chunks)
            print(f"[{split}] {path.name:44s} family={audit['family']:16s} "
                  f"airborne={audit['airborne_seconds']:6.2f}s chunks={audit['kept_chunks']} "
                  f"rejected={audit['rejected_saturated']}", flush=True)
        if not pieces:
            raise RuntimeError(f"the {split} split produced no usable flights")
        split_flights[split] = np.concatenate(pieces)
        split_audits[split] = audits

    train, test = split_flights["train"], split_flights["test"]
    step = 1.0 / SAMPLE_HZ
    settings = {
        "dataset_name": "NANOBENCH_CF2_10s_h0p01",
        "source": {
            "name": "NanoBench", "arxiv": "2603.09908", "url": "https://github.com/syediu/nanobench",
            "license": "BSD 3-Clause", "platform": "Crazyflie 2.1 under Vicon motion capture",
            "flights_used": {split: [audit["file"] for audit in audits] for split, audits in split_audits.items()},
        },
        "vehicle_parameters": {
            "mass": MASS_KG, "inertia": np.diag(NOMINAL_INERTIA).tolist(),
            "gravity_acceleration": GRAVITY, "arm": ARM_LENGTH, "kf": KF, "km": KM,
            "mass_note": "measured instrumented flying mass reported by the dataset authors (27 g airframe "
                         "plus markers and charging deck)",
            "inertia_note": "stock CF2 URDF value, NOT measured for this airframe; present only so the "
                            "loaders have a diagonal inertia, and unused by the GP forward model",
        },
        # No physical damping law is known for the real vehicle. The value is a placeholder that the GP
        # setup requires; it never enters the GP vector field (only dissipation_enabled does).
        "linear_damping_coefficient": 0.5,
        "angular_damping_coefficient": 0.5,
        "damping_law": "unknown (real vehicle)",
        "damping_equation": "not applicable: the true dissipation of the real Crazyflie is not known",
        "state_layout": "x_w(3), vec(R)(9), v_b(3), omega_b(3), wrench(4)",
        "control_layout": "wrench [T, tau_x, tau_y, tau_z] from battery-compensated per-motor PWM via the CF2X mixer",
        "input_mode": "wrench",
        "true_control_map": "selection matrix S (up to the constant CF2X mixer calibration)",
        "motor_model": {
            "relation": "rpm = S*PWM*(V_bat/V_nom) + C ; f = k_f rpm^2",
            "pwm2rpm_scale": PWM2RPM_SCALE, "pwm2rpm_const": PWM2RPM_CONST,
            "v_nominal": V_NOMINAL, "kf": KF, "km": KM, "arm_length": ARM_LENGTH,
            "validation": "battery-drain hovers give mean thrust 0.96-1.00 x m g with CV 0.009-0.010",
        },
        "sample_dt": step, "physics_hz": SAMPLE_HZ, "flight_seconds": (FLIGHT_POINTS - 1) * step,
        "config": {"environment": {"sample_hz": SAMPLE_HZ, "drone_model": "CF2X", "physics": "real"}},
        "splits": {
            "train": {"families": list(TRAIN_FAMILIES), "key": "train_trajectories",
                      "flights": int(train.shape[0])},
            "test": {"families": list(TEST_FAMILIES), "key": "test_trajectories",
                     "flights": int(test.shape[0]),
                     "note": "held-out trajectory shapes: no family here occurs in training"},
            "families_shared_between_train_and_test": shared,
        },
        "train_flight_audits": split_audits["train"], "test_flight_audits": split_audits["test"],
        "audits_train": coverage(train), "audits_test": coverage(test),
        "clean_training_state_sha256": array_sha256(train[..., :18]),
        "clean_test_state_sha256": array_sha256(test[..., :18]),
        "observation_noise": {
            "enabled": False,
            "note": "real measurements: no synthetic noise is added. The Vicon rate carries roughly "
                    "0.10 rad/s of differentiation jitter and the motor telemetry is interpolated from a "
                    "radio-limited stream; the learned per-block sigma absorbs both.",
        },
    }

    payload = {
        "x": sliding_windows(train, 6, 5), "test_x": sliding_windows(test, 6, 5),
        "t": np.arange(6) * step,
        "train_trajectories": train, "test_trajectories": test,
        "settings": settings,
    }
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{settings['dataset_name']}_clean.pkl"
    if target.exists() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force")
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output_dir / f"{settings['dataset_name']}_audits.json").write_text(
        json.dumps(settings, indent=1, default=float))

    print(f"\nwrote {target}")
    print(f"  train_trajectories {train.shape}  from families {list(TRAIN_FAMILIES)}")
    print(f"  test_trajectories  {test.shape}  from families {list(TEST_FAMILIES)}")
    print(f"  train seconds {train.shape[0] * (FLIGHT_POINTS - 1) * step:.0f}  "
          f"test seconds {test.shape[0] * (FLIGHT_POINTS - 1) * step:.0f}")
    for split, values in (("train", settings["audits_train"]), ("test", settings["audits_test"])):
        print(f"  {split}: speed p50/p90/max {np.round(values['speed_percentiles_50_90_max'], 2)}  "
              f"rate {np.round(values['angular_rate_percentiles_50_90_max'], 2)}  "
              f"tilt {np.round(values['tilt_deg_percentiles_50_90_max'], 1)}  "
              f"T/mg mean {values['thrust_over_weight_mean_cv'][0]:.3f}")


if __name__ == "__main__":
    main()

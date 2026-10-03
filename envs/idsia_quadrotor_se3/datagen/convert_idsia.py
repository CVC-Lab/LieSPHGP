"""IDSIA nano-drone system-identification benchmark -> the 22-channel layout used by the SE(3) models.

Source: Busetto et al., "Nonlinear System Identification for a Nano-drone Benchmark", Control Engineering
Practice; https://github.com/idsia-robotics/nanodrone-sysid-benchmark.  Real flights of a Crazyflie 2.1
**Brushless** in a motion-capture arena at 100 Hz.

Why this set rather than NanoBench: its input is the **measured** motor angular velocity from the ESC
telemetry, not a commanded PWM.  There is therefore no PWM-to-thrust map to calibrate, no battery term, and no
gap between the command and what the motors actually did.  On NanoBench the rotational channel was
unidentifiable (yaw R^2 = 0.000, best fit needing a 20-30 ms motor lag); here the same test gives roll/pitch/yaw
R^2 = 0.369/0.342/0.417 at **zero** lag.

The benchmark also publishes its physical constants, which NanoBench does not (models/models.py):
    m = 0.045 kg, g = 9.81, J = diag(2.3951e-5, 2.3951e-5, 3.2347e-6) kg m^2, thrust-to-weight 2.0,
    K_t = 3.72e-8 N/(rad/s)^2, K_c = 7.74e-12 N m/(rad/s)^2, arm = 0.0353 m
None of them is given to our models; they are recorded in the settings for reference and used only by the
validation script.

STATE (motion capture, one clock):
    x_w      x, y, z                          (m, world)
    R        from the quaternion (qx,qy,qz,qw)  (body -> world, stored row-major)
    v_b      R^T [vx,vy,vz]                   (the CSV velocity is world frame)
    omega_b  [wx,wy,wz]                       (already body frame, as their own physics model uses it)

INPUT, their published mixer applied to the measured rotor speeds Omega_i (rad/s):
    T     = K_t sum Omega_i^2
    tau_x = K_t a [(Omega_3^2 + Omega_4^2) - (Omega_1^2 + Omega_2^2)]
    tau_y = K_t a [(Omega_2^2 + Omega_3^2) - (Omega_1^2 + Omega_4^2)]
    tau_z = K_c [(Omega_1^2 + Omega_3^2) - (Omega_2^2 + Omega_4^2)]

SPLIT: the benchmark's own protocol, which is already shape-disjoint.
    train   chirp, random, square      (12 flights)
    test    melon                      (3 flights, never seen in training)

INPUT VARIANT ``--input rotor2``: instead of the pre-mixed wrench, the four measured rotor speeds squared,
    u_i = Omega_i^2 / Omega_hover^2,   Omega_hover^2 = m g / (4 K_t)   (so u_i ~ 1 at hover)
so that the learned control map g(x, R) in R^{6x4} learns the mixer itself (per-motor thrust, the yaw
coefficient, any cross-coupling) instead of only rescaling four channels fixed by the published constants.
The two variants differ only in columns 18:22; the state columns are byte-identical.  The rotor2 file is named
``IDSIA_CF21BL_10s_h0p01_rotor2_clean.pkl`` and its settings carry ``input_mode = "rotor2"``.

Usage:  python convert_idsia.py [--source tmp/idsia_raw/data] [--input wrench|rotor2] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]  # envs/idsia_quadrotor_se3/datagen -> project root
DATASET_DIR = PROJECT_ROOT / "datasets/QUADROTOR-DATASET-IDSIA"

KT, KC, ARM = 3.72e-8, 7.74e-12, 0.0353
MASS, GRAVITY = 0.045, 9.81
INERTIA = (2.3951e-5, 2.3951e-5, 3.2347e-6)
THRUST_TO_WEIGHT = 2.0

TRAIN_FAMILIES = ("chirp", "random", "square")
TEST_FAMILIES = ("melon",)
FLIGHT_POINTS = 1001                 # 10.01 s at 100 Hz, the same flight length as the other datasets
SAMPLE_HZ = 100
MOTOR_COLUMNS = [f"m{i}_rads" for i in (1, 2, 3, 4)]


def family_of(path: Path) -> str:
    match = re.match(r"([a-z]+)_\d+_run\d+$", path.stem)
    if match is None:
        raise ValueError(f"cannot parse the trajectory family from {path.name}")
    return match.group(1)


def wrench_from_rotor_speeds(speeds: np.ndarray) -> np.ndarray:
    """The benchmark's own motor_to_phys map, kept in physical units instead of their normalised ones."""
    squared = speeds**2
    return np.column_stack([
        KT * squared.sum(axis=1),
        KT * ARM * ((squared[:, 2] + squared[:, 3]) - (squared[:, 0] + squared[:, 1])),
        KT * ARM * ((squared[:, 1] + squared[:, 2]) - (squared[:, 0] + squared[:, 3])),
        KC * ((squared[:, 0] + squared[:, 2]) - (squared[:, 1] + squared[:, 3])),
    ])


ROTOR2_SCALE = 4.0 * KT / (MASS * GRAVITY)     # 1 / Omega_hover^2: u_i = 1 at hover


def rotor2_from_rotor_speeds(speeds: np.ndarray) -> np.ndarray:
    """Per-motor squared speed in hover units; the map to force and torque is left to the learned g."""
    return speeds**2 * ROTOR2_SCALE


def controls_from_rotor_speeds(speeds: np.ndarray, input_mode: str) -> np.ndarray:
    if input_mode == "wrench":
        return wrench_from_rotor_speeds(speeds)
    if input_mode == "rotor2":
        return rotor2_from_rotor_speeds(speeds)
    raise ValueError(f"unknown input mode {input_mode!r}")


def convert_flight(path: Path, input_mode: str = "wrench") -> tuple[np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(path)
    rotation = Rotation.from_quat(frame[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    position = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
    velocity_world = frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    omega_body = frame[["wx", "wy", "wz"]].to_numpy(dtype=np.float64)
    velocity_body = np.einsum("nji,nj->ni", rotation, velocity_world)
    speeds = frame[MOTOR_COLUMNS].to_numpy(dtype=np.float64)
    control = controls_from_rotor_speeds(speeds, input_mode)
    wrench = wrench_from_rotor_speeds(speeds)          # audit only
    states = np.concatenate([position, rotation.reshape(-1, 9), velocity_body, omega_body, control], axis=1)
    chunks = [states[start:start + FLIGHT_POINTS]
              for start in range(0, states.shape[0] - FLIGHT_POINTS + 1, FLIGHT_POINTS)]
    audit = {
        "file": path.name, "family": family_of(path), "rows": int(len(frame)),
        "seconds": round(len(frame) / SAMPLE_HZ, 2), "kept_chunks": len(chunks),
        "altitude_range": [round(float(position[:, 2].min()), 3), round(float(position[:, 2].max()), 3)],
        "thrust_over_weight_mean": round(float(wrench[:, 0].mean() / (MASS * GRAVITY)), 3),
    }
    return (np.stack(chunks) if chunks else np.zeros((0, FLIGHT_POINTS, 22))), audit


def sliding_windows(flights: np.ndarray, points: int, stride: int) -> np.ndarray:
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


def coverage(flights: np.ndarray, input_mode: str = "wrench") -> dict[str, Any]:
    if flights.shape[0] == 0:
        return {}
    flat = flights.reshape(-1, 22)
    speed = np.linalg.norm(flat[:, 12:15], axis=1)
    rate = np.linalg.norm(flat[:, 15:18], axis=1)
    tilt = np.degrees(np.arccos(np.clip(flat[:, 11], -1.0, 1.0)))
    total_thrust = flat[:, 18] if input_mode == "wrench" else KT * flat[:, 18:22].sum(axis=1) / ROTOR2_SCALE
    thrust = total_thrust / (MASS * GRAVITY)
    rotations = flat[:, 3:12].reshape(-1, 3, 3)
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
        "max_orthogonality_frobenius": float(np.linalg.norm(
            np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3), axis=(1, 2)).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "tmp/idsia_raw/data")
    parser.add_argument("--output", type=Path, default=DATASET_DIR)
    parser.add_argument("--input", choices=("wrench", "rotor2"), default="wrench")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    input_mode = arguments.input

    shared = sorted(set(TRAIN_FAMILIES) & set(TEST_FAMILIES))
    if shared:
        raise ValueError(f"train and test libraries must share no trajectory family; shared: {shared}")

    files = sorted(arguments.source.rglob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV flights under {arguments.source}")
    unknown = sorted({family_of(path) for path in files} - set(TRAIN_FAMILIES) - set(TEST_FAMILIES))
    if unknown:
        raise ValueError(f"unassigned trajectory families: {unknown}")

    split_flights, split_audits = {}, {}
    for split, families in (("train", TRAIN_FAMILIES), ("test", TEST_FAMILIES)):
        pieces, audits = [], []
        for path in [p for p in files if family_of(p) in families]:
            chunks, audit = convert_flight(path, input_mode)
            audits.append(audit)
            if chunks.shape[0]:
                pieces.append(chunks)
            print(f"[{split}] {path.name:32s} family={audit['family']:8s} {audit['seconds']:6.2f}s "
                  f"chunks={audit['kept_chunks']} T/mg={audit['thrust_over_weight_mean']:.3f}", flush=True)
        split_flights[split] = np.concatenate(pieces)
        split_audits[split] = audits

    train, test = split_flights["train"], split_flights["test"]
    step = 1.0 / SAMPLE_HZ
    settings = {
        "dataset_name": "IDSIA_CF21BL_10s_h0p01" + ("" if input_mode == "wrench" else f"_{input_mode}"),
        "source": {
            "name": "IDSIA nano-drone system identification benchmark", "arxiv": "2512.14450",
            "url": "https://github.com/idsia-robotics/nanodrone-sysid-benchmark",
            "paper": "Busetto et al., Nonlinear System Identification for a Nano-drone Benchmark, "
                     "Control Engineering Practice",
            "platform": "Crazyflie 2.1 Brushless under motion capture",
            "input_note": "MEASURED motor angular velocity from ESC telemetry, not a commanded PWM",
            "flights_used": {split: [a["file"] for a in audits] for split, audits in split_audits.items()},
        },
        "vehicle_parameters": {
            "mass": MASS, "inertia": np.diag(INERTIA).tolist(), "gravity_acceleration": GRAVITY,
            "arm": ARM, "kf": KT, "km": KC, "thrust_to_weight": THRUST_TO_WEIGHT,
            "note": "published by the benchmark (models/models.py); recorded for reference only and never "
                    "supplied to the GP model",
        },
        "linear_damping_coefficient": 0.5,
        "angular_damping_coefficient": 0.5,
        "damping_law": "unknown (real vehicle)",
        "damping_equation": "not applicable: the true dissipation of the real vehicle is not known",
        "state_layout": "x_w(3), vec(R)(9), v_b(3), omega_b(3), " + (
            "wrench(4)" if input_mode == "wrench" else "rotor2(4)"),
        "control_layout": ("wrench [T, tau_x, tau_y, tau_z] from the MEASURED rotor speeds via the "
                           "benchmark's own mixer" if input_mode == "wrench" else
                           "u_i = Omega_i^2 / Omega_hover^2 for the four MEASURED rotor speeds; the mixer is learned by g"),
        "input_mode": input_mode,
        "rotor2_scale": ROTOR2_SCALE if input_mode == "rotor2" else None,
        "true_control_map": "selection matrix S" if input_mode == "wrench" else
                            "unknown: mixer from Omega^2 to force/torque, to be learned",
        "motor_model": {
            "relation": "T = K_t sum(Omega^2); tau from the benchmark's mixer with arm a and K_c",
            "kt": KT, "kc": KC, "arm": ARM,
            "validation": "parameter-free translational fit gives 44.78 g against the published 45.0 g; "
                          "rotational consistency R^2 = 0.369/0.342/0.417 at zero lag",
        },
        "sample_dt": step, "physics_hz": SAMPLE_HZ, "flight_seconds": (FLIGHT_POINTS - 1) * step,
        "config": {"environment": {"sample_hz": SAMPLE_HZ, "drone_model": "CF21BL", "physics": "real"}},
        "splits": {
            "train": {"families": list(TRAIN_FAMILIES), "key": "train_trajectories", "flights": int(train.shape[0])},
            "test": {"families": list(TEST_FAMILIES), "key": "test_trajectories", "flights": int(test.shape[0]),
                     "note": "the benchmark's own held-out trajectory; its shape never occurs in training"},
            "families_shared_between_train_and_test": shared,
        },
        "train_flight_audits": split_audits["train"], "test_flight_audits": split_audits["test"],
        "audits_train": coverage(train, input_mode), "audits_test": coverage(test, input_mode),
        "clean_training_state_sha256": array_sha256(train[..., :18]),
        "clean_test_state_sha256": array_sha256(test[..., :18]),
        "observation_noise": {"enabled": False, "note": "real measurements; no synthetic noise is added"},
    }
    payload = {
        "x": sliding_windows(train, 6, 5), "test_x": sliding_windows(test, 6, 5),
        "t": np.arange(6) * step,
        "train_trajectories": train, "test_trajectories": test, "settings": settings,
    }
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{settings['dataset_name']}_clean.pkl"
    if target.exists() and not arguments.force:
        raise FileExistsError(f"{target} exists; pass --force")
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output_dir / f"{settings['dataset_name']}_audits.json").write_text(json.dumps(settings, indent=1, default=float))

    print(f"\nwrote {target}")
    print(f"  train_trajectories {train.shape}  families {list(TRAIN_FAMILIES)}")
    print(f"  test_trajectories  {test.shape}  families {list(TEST_FAMILIES)}")
    print(f"  train {train.shape[0] * (FLIGHT_POINTS - 1) * step:.0f} s, "
          f"test {test.shape[0] * (FLIGHT_POINTS - 1) * step:.0f} s")
    for split, values in (("train", settings["audits_train"]), ("test", settings["audits_test"])):
        print(f"  {split}: speed p50/p90/max {np.round(values['speed_percentiles_50_90_max'], 2)}  "
              f"rate {np.round(values['angular_rate_percentiles_50_90_max'], 2)}  "
              f"tilt {np.round(values['tilt_deg_percentiles_50_90_max'], 1)}  "
              f"T/mg {values['thrust_over_weight_mean_cv'][0]:.3f}")


if __name__ == "__main__":
    main()

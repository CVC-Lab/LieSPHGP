"""The recorded melon flights and the reference the controller tracks.

The commanded melon setpoints are not stored in the benchmark CSVs (only the flown state and the measured rotor
speeds are), so the reference is the flown path itself: world position as recorded, world velocity as recorded,
feedforward acceleration and yaw rate from lightly smoothed finite differences (the same construction as
``comparision/report_controller.py:make_recorded_reference``).  Everything is resampled from the 100 Hz data grid
to the 500 Hz controller grid by linear interpolation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

SAMPLE_STEP = 0.01
MOTOR_COLUMNS = [f"m{i}_rads" for i in (1, 2, 3, 4)]


def load_flights(source: Path, pattern: str) -> list[dict]:
    """Per flight: name, the 18-wide model state [x_w, vec(R), v_b, omega_b] and the recorded rotor speeds."""
    flights = []
    for path in sorted(Path(source).glob(pattern)):
        frame = pd.read_csv(path)
        rotation = Rotation.from_quat(frame[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
        velocity_world = frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
        states = np.concatenate([
            frame[["x", "y", "z"]].to_numpy(dtype=np.float64), rotation.reshape(-1, 9),
            np.einsum("nji,nj->ni", rotation, velocity_world),
            frame[["wx", "wy", "wz"]].to_numpy(dtype=np.float64)], axis=1)
        flights.append({"name": path.stem, "states": states, "velocity_world": velocity_world,
                        "rotor_speeds": frame[MOTOR_COLUMNS].to_numpy(dtype=np.float64)})
    if not flights:
        raise FileNotFoundError(f"no flights match {pattern} under {source}")
    return flights


def reference_signals(flight: dict, smoothing_seconds: float = 0.05) -> dict[str, np.ndarray]:
    """Position, velocity, feedforward acceleration, yaw and yaw rate on the 100 Hz grid."""
    rotation = flight["states"][:, 3:12].reshape(-1, 3, 3)
    width = max(int(round(smoothing_seconds / SAMPLE_STEP)), 1)
    yaw = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    return {
        "position": flight["states"][:, :3],
        "velocity": flight["velocity_world"],
        "acceleration": uniform_filter1d(np.gradient(flight["velocity_world"], SAMPLE_STEP, axis=0),
                                         width, axis=0, mode="nearest"),
        "yaw": yaw,
        "yaw_rate": uniform_filter1d(np.gradient(yaw, SAMPLE_STEP), width, mode="nearest"),
    }


def at_controller_rate(signals: dict[str, np.ndarray], ticks: int, samples: int) -> dict[str, np.ndarray]:
    """(samples-1, ticks, ...) arrays: tick j of interval k sits at time (k + j/ticks) * SAMPLE_STEP."""
    alpha = (np.arange(ticks) / ticks)[None, :]
    out = {}
    for key, value in signals.items():
        a, b = value[:samples - 1], value[1:samples]
        if value.ndim == 1:
            out[key] = (1.0 - alpha) * a[:, None] + alpha * b[:, None]
        else:
            out[key] = (1.0 - alpha[..., None]) * a[:, None, :] + alpha[..., None] * b[:, None, :]
    return out

"""Load the noisy quadrotor flights and cut them into windows.

Only train_trajectories (noisy in an obs-noise pickle), test_trajectories_noisy and t are read. The clean
test_trajectories / test_x and the settings (mass, inertia, damping, wind, control map) are never opened here.

Row k of a flight = [x_w (3), vec(R) (9, row-major), v_b (3), omega_b (3), u (4) = wrench [T, tau]], where the u in
row k+1 is the control applied during k -> k+1 (the generator's convention, same as the pendulum).

Control scaling ``rms``: every control channel is divided by its root mean square over the TRAINING flights (a data
statistic, like feature normalisation). Thrust (~0.3 N) and torques (~1e-3 N m) then share one scale, so the learned
control map is O(1-10) instead of O(1e4). The factors are stored in the checkpoint; evaluate.py converts back.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

STATE_DIMENSION = 22
CONTROL = slice(18, 22)


def cut_windows(flights: np.ndarray, window_points: int, stride: int) -> np.ndarray:
    """(flights, time, 22) -> time-major (window_points, windows, 22)."""
    values = np.asarray(flights, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != STATE_DIMENSION:
        raise ValueError(f"expected (flights, time, 22), got {values.shape}")
    starts = range(0, values.shape[1] - window_points + 1, stride)
    windows = np.stack([values[:, s:s + window_points] for s in starts], axis=1)
    return windows.reshape(-1, window_points, STATE_DIMENSION).transpose(1, 0, 2)


def control_scale(train_flights: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(4, dtype=np.float32)
    return np.sqrt(np.mean(np.square(np.asarray(train_flights)[..., CONTROL]), axis=(0, 1))).astype(np.float32)


def apply_control_scale(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scaled = np.array(values, dtype=np.float32, copy=True)
    scaled[..., CONTROL] = scaled[..., CONTROL] / scale
    return scaled


def load_noisy_windows(path: Path, window_points: int, stride: int, control_scaling: str, train_flights: int | None = None):
    """(train windows, test windows, observation interval dt in s, control scale (4,)), noisy, controls scaled.

    ``train_flights`` (data.train_flights, optional): use only the first that many training flights (the control
    scale is then taken over those flights too); None = every training flight of the pickle."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if "test_trajectories_noisy" not in payload:
        raise ValueError("the dataset has no test_trajectories_noisy; train on a train-obs-noise pickle, not the clean one")
    times = np.asarray(payload["t"], dtype=np.float64).reshape(-1)
    dt = float(np.median(np.diff(times)))
    flights = np.asarray(payload["train_trajectories"])
    if train_flights is not None:
        if not 0 < int(train_flights) <= flights.shape[0]:
            raise ValueError(f"data.train_flights = {train_flights}, the pickle has {flights.shape[0]} training flights")
        flights = flights[:int(train_flights)]
    scale = control_scale(flights, control_scaling)
    train = cut_windows(apply_control_scale(flights, scale), window_points, stride)
    test = cut_windows(apply_control_scale(payload["test_trajectories_noisy"], scale), window_points, stride)
    return train, test, dt, scale

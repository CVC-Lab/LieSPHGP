"""Load the BlueROV2 trajectories and cut them into windows.

Only the two arrays named in the config (data.train_key, data.test_key) and t are read; the pickle's settings
(published vehicle values, thruster geometry, ground truth of the simulator) are never opened here.
    real Marinarium data   train_trajectories / test_trajectories     (the measurements themselves)
    simulator obs-noise    train_trajectories / test_trajectories_noisy

Row k of a trajectory = [x_w (3), vec(R) (9, row-major), v_b (3), omega_b (3), u (n_u)], where the u in row k+1 is the
input applied during k -> k+1 (both BlueROV2 converters: u of row k = mean over (t_{k-1}, t_k]).

Control scaling ``rms``: every input channel is divided by its root mean square over the TRAINING trajectories (a data
statistic, like feature normalisation). The factors are stored in the checkpoint.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

POSE_TWIST = 18
CONTROL = slice(18, None)


def cut_windows(flights: np.ndarray, window_points: int, stride: int) -> np.ndarray:
    """(trajectories, time, 18 + n_u) -> time-major (window_points, windows, 18 + n_u)."""
    values = np.asarray(flights, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] <= POSE_TWIST:
        raise ValueError(f"expected (trajectories, time, 18 + n_u), got {values.shape}")
    starts = range(0, values.shape[1] - window_points + 1, stride)
    windows = np.stack([values[:, s:s + window_points] for s in starts], axis=1)
    return windows.reshape(-1, window_points, values.shape[-1]).transpose(1, 0, 2)


def control_scale(train_flights: np.ndarray, mode: str) -> np.ndarray:
    count = np.asarray(train_flights).shape[-1] - POSE_TWIST
    if mode == "none":
        return np.ones(count, dtype=np.float32)
    return np.sqrt(np.mean(np.square(np.asarray(train_flights)[..., CONTROL]), axis=(0, 1))).astype(np.float32)


def apply_control_scale(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scaled = np.array(values, dtype=np.float32, copy=True)
    scaled[..., CONTROL] = scaled[..., CONTROL] / scale
    return scaled


def load_noisy_windows(path: Path, window_points: int, stride: int, control_scaling: str,
                       train_key: str = "train_trajectories", test_key: str = "test_trajectories_noisy"):
    """(train windows, test windows, observation interval dt in s, control scale (n_u,)), controls scaled."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    for key in (train_key, test_key):
        if key not in payload:
            raise ValueError(f"the dataset has no {key!r}")
    times = np.asarray(payload["t"], dtype=np.float64).reshape(-1)
    dt = float(np.median(np.diff(times)))
    scale = control_scale(payload[train_key], control_scaling)
    train = cut_windows(apply_control_scale(payload[train_key], scale), window_points, stride)
    test = cut_windows(apply_control_scale(payload[test_key], scale), window_points, stride)
    return train, test, dt, scale

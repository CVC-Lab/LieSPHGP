"""Load the noisy pendulum windows. Only x (noisy train), test_x_noisy (noisy test) and t are read.

The pickle's clean test_x and its settings (the simulator's mass, friction, wind, ...) are never opened here:
training sees what a real experiment would see. evaluate.py is the only place that reads them.

Layout of PENDULUM-DATASET-<name>/<name>_obs-noise<s>.pkl: x, test_x_noisy (batches, T, N, 15) with
row k = [vec(R_k) (9, row-major), omega_k (3, body), u (3)], where the u in row k+1 is the control applied
during k -> k+1 (row 0 repeats the first control).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

STATE_DIMENSION = 15


def _time_major(array: np.ndarray) -> np.ndarray:
    """(batches, T, N, 15) -> (T, batches * N, 15): one window per trajectory."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != STATE_DIMENSION:
        raise ValueError(f"expected (batches, T, N, 15), got {values.shape}")
    batches, steps, samples, _ = values.shape
    return values.transpose(1, 0, 2, 3).reshape(steps, batches * samples, STATE_DIMENSION)


def cut_windows(trajectories: np.ndarray, window_points: int | None, stride: int | None) -> np.ndarray:
    """Time-major (T, N, 15) trajectories -> (window_points, windows, 15): windows starting every ``stride`` samples
    inside each trajectory. ``window_points`` None keeps the whole trajectory as one window."""
    if window_points is None or window_points >= trajectories.shape[0]:
        return trajectories
    starts = range(0, trajectories.shape[0] - window_points + 1, int(stride))
    return np.concatenate([trajectories[s:s + window_points] for s in starts], axis=1)


def load_noisy_windows(path: Path, window_points: int | None = None,
                       stride: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """(train (W, N, 15), test (W, M, 15), observation interval dt in s), both noisy."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if "test_x_noisy" not in payload:
        raise ValueError("the dataset has no test_x_noisy; train on an obs-noise<s> pickle, not the clean one")
    times = np.asarray(payload["t"], dtype=np.float64).reshape(-1)
    dt = float(np.median(np.diff(times)))
    return (cut_windows(_time_major(payload["x"]), window_points, stride),
            cut_windows(_time_major(payload["test_x_noisy"]), window_points, stride), dt)

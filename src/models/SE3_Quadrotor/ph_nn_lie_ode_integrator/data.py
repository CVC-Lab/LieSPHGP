"""Pure NumPy loading and window preparation for quadrotor JAX training."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


STATE_DIMENSION = 22


def _time_major_windows(array: Any, window_points: int, stride: int) -> np.ndarray:
    values = np.asarray(array)
    if values.shape[-1] != STATE_DIMENSION:
        raise ValueError(f"Expected final state dimension 22, got {values.shape}")
    if values.ndim == 3:
        if values.shape[0] != window_points:
            if values.shape[0] < window_points:
                raise ValueError(f"Dataset has only {values.shape[0]} time points")
            chunks = [values[start : start + window_points] for start in range(
                0, values.shape[0] - window_points + 1, stride
            )]
            return np.concatenate(chunks, axis=1)
        return values
    if values.ndim != 4:
        raise ValueError(f"Expected a 3D or 4D state tensor, got {values.shape}")

    # Common stored layout: (trajectory/group, time, sample, state).
    groups, time_count, samples, state_dim = values.shape
    chunks = []
    for start in range(0, time_count - window_points + 1, stride):
        window = values[:, start : start + window_points]
        chunks.append(window.transpose(1, 0, 2, 3).reshape(window_points, groups * samples, state_dim))
    if not chunks:
        raise ValueError(
            f"No windows: time_count={time_count}, window_points={window_points}, stride={stride}"
        )
    return np.concatenate(chunks, axis=1)


def load_dataset(
    path: Path,
    *,
    window_points: int,
    window_stride: int,
    max_train_windows: int | None,
    max_test_windows: int | None,
    observation_noise_override: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not {"x", "test_x", "t"} <= payload.keys():
        raise ValueError("Dataset must be a mapping containing x, test_x, and t")

    train = _time_major_windows(payload["x"], window_points, window_stride)
    test = _time_major_windows(payload["test_x"], window_points, window_stride)
    if max_train_windows is not None:
        train = train[:, : int(max_train_windows)]
    if max_test_windows is not None:
        test = test[:, : int(max_test_windows)]
    if observation_noise_override is not None:
        sigma = float(observation_noise_override)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma, size=train[..., :18].shape)
        train = train.copy()
        train[..., :18] += noise

    raw_times = np.asarray(payload["t"], dtype=np.float64).reshape(-1)
    if raw_times.size < 2:
        raise ValueError("Dataset t must contain at least two points")
    dt = float(np.median(np.diff(raw_times)))
    times = np.arange(window_points, dtype=np.float32) * np.float32(dt)
    return (
        np.asarray(train, dtype=np.float32),
        np.asarray(test, dtype=np.float32),
        times,
        dict(payload.get("settings", {})),
    )

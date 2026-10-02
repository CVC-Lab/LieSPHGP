"""Framework-neutral serialization of JAX parameter pytrees."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np


def _to_host_tree(value: Any) -> Any:
    def convert(leaf: Any) -> Any:
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            return np.asarray(jax.device_get(leaf))
        return leaf

    return jax.tree_util.tree_map(convert, value)


def save_checkpoint(path: Path, *, params: Any, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "framework": "JAX",
        "params": _to_host_tree(params),
        "extra": _to_host_tree(extra or {}),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("framework") != "JAX":
        raise ValueError(f"Not a canonical JAX checkpoint: {path}")
    payload["params"] = jax.tree_util.tree_map(lambda value: jax.numpy.asarray(value), payload["params"])
    return payload

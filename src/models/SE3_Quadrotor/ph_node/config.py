"""Strict YAML configuration for the quadrotor PH-NODE trainer: lie_ph's required keys and prior-knowledge check with
its own schema, plus model.integrator rk4, model.family nn, model.wind false and training.loss rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lie_ph.config import REQUIRED_PATHS, _check_forbidden, _lookup, resolve_project_path  # noqa: F401

SCHEMA = "ph_node/quadrotor/v1"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("The config root must be a YAML mapping")
    for required in REQUIRED_PATHS:
        _lookup(config, required)
    if config["schema"] != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    _check_forbidden(config)
    model = config["model"]
    if str(model.get("integrator")) != "rk4" or str(model.get("family")) != "nn" or bool(model.get("wind", True)):
        raise ValueError("PH-NODE needs model.integrator rk4, model.family nn and model.wind false")
    if str(config["training"].get("loss")) != "rollout":
        raise ValueError("PH-NODE needs training.loss rollout")
    if config["data"]["control_scaling"] not in ("rms", "none"):
        raise ValueError("data.control_scaling must be rms or none")
    for dotted in ("data.window_points", "data.window_stride", "data.substeps", "training.batch_size",
                   "training.eval_every", "training.checkpoint_every", "optimizer.learning_rate"):
        if float(_lookup(config, dotted)) <= 0:
            raise ValueError(f"{dotted} must be positive")
    if not isinstance(config["experiment"]["notes"], list):
        raise TypeError("experiment.notes must be a YAML list")
    config["_config_path"] = str(config_path)
    return config

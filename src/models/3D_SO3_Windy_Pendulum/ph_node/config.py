"""Strict YAML configuration for the PH-NODE trainer: lie_ph's required keys and prior-knowledge check, plus

    model.integrator          rk4 (required)
    training.loss             rollout (PH-NODE) | rollout_reference (PH-NODE-ref)
    model.nn_architecture     xavier | duong
    optimizer.schedule        cosine (default) | constant;   optimizer.weight_decay (default 0)

model.pretrain_inverse_mass_identity is the one allowed key with a forbidden word: Duong & Atanasov's pretraining of
the mass network to the identity (a neutral start, the same M^-1 = I the GP levels start from; no fitted constant).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lie_ph.config import FORBIDDEN_KEY_WORDS, REQUIRED_PATHS, _lookup, resolve_project_path  # noqa: F401

SCHEMA = "ph_node/pendulum/v1"
ALLOWED_KEYS = ("model.pretrain_inverse_mass_identity",)


def _check_forbidden(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if f"{path}{key}" in ALLOWED_KEYS:
                continue
            lowered = str(key).lower()
            for word in FORBIDDEN_KEY_WORDS:
                if word in lowered:
                    raise ValueError(f"Config key {path}{key!r} looks like physical prior knowledge ({word!r}); refused")
            _check_forbidden(value, f"{path}{key}.")


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
    if str(config["model"].get("integrator")) != "rk4":
        raise ValueError("model.integrator must be rk4")
    if str(config["training"].get("loss")) not in ("rollout", "rollout_reference"):
        raise ValueError("training.loss must be rollout or rollout_reference")
    if bool(config["model"].get("pretrain_inverse_mass_identity", False)) and config["model"].get("nn_architecture") != "duong":
        raise ValueError("model.pretrain_inverse_mass_identity is only defined for nn_architecture duong (PH-NODE-ref)")
    for dotted in ("data.window_points", "data.window_stride", "data.substeps", "training.batch_size", "training.eval_every",
                   "training.checkpoint_every", "optimizer.learning_rate"):
        if float(_lookup(config, dotted)) <= 0:
            raise ValueError(f"{dotted} must be positive")
    if not isinstance(config["experiment"]["notes"], list):
        raise TypeError("experiment.notes must be a YAML list")
    config["_config_path"] = str(config_path)
    return config

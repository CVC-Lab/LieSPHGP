"""Strict YAML configuration for the SO(3) pendulum Lie-PH trainer (lie_ph).

Every value the trainer uses must be in the file. Keys that would hand the model physical knowledge (mass, inertia,
gravity, friction, a pretraining target, ...) are refused anywhere in the file, so a config cannot inject a prior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "lie_ph/pendulum/v1"

REQUIRED_PATHS = (
    "schema", "created_at",
    "data.dataset_path", "data.window_points", "data.window_stride", "data.substeps",
    "model.feature_count", "model.matern_smoothness", "model.matern_length_scale",
    "model.initial_log_std", "model.initial_M_inverse", "model.initial_D", "model.initial_control_gp_mean_scale",
    "model.initial_diffusion", "model.initial_observation_sigma",
    "training.seed", "training.total_steps", "training.batch_size", "training.eval_every", "training.eval_windows",
    "training.checkpoint_every", "training.kl_beta_max", "training.kl_anneal_steps",
    "optimizer.learning_rate", "optimizer.final_learning_fraction", "optimizer.level_learning_rate",
    "optimizer.beta1", "optimizer.beta2", "optimizer.epsilon", "optimizer.gradient_clip_norm",
    "runtime.gpu", "runtime.enable_x64",
    "experiment.root", "experiment.name_suffix", "experiment.notes",
)

# Substrings that must not appear in any key: they would mean physical knowledge enters training.
FORBIDDEN_KEY_WORDS = ("mass", "inertia", "gravity", "length_rod", "friction", "damping_coefficient", "pretrain",
                       "physics", "known", "prior", "true", "ground_truth", "penalty")


def _lookup(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Required config argument is missing: {dotted}")
        value = value[part]
    return value


def _check_forbidden(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            for word in FORBIDDEN_KEY_WORDS:
                if word in lowered:
                    raise ValueError(f"Config key {path}{key!r} looks like physical prior knowledge ({word!r}); refused")
            _check_forbidden(value, f"{path}{key}.")


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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
    if str(config["model"].get("integrator", "lie_imex")) != "lie_imex":
        raise ValueError("model.integrator must be lie_imex (the only integrator of lie_ph)")
    for dotted in ("data.window_points", "data.window_stride", "data.substeps", "training.batch_size", "training.eval_every", "training.checkpoint_every",
                   "model.feature_count", "optimizer.learning_rate", "model.initial_M_inverse", "model.initial_D",
                   "model.initial_diffusion", "model.initial_observation_sigma"):
        if float(_lookup(config, dotted)) <= 0:
            raise ValueError(f"{dotted} must be positive")
    if not isinstance(config["experiment"]["notes"], list):
        raise TypeError("experiment.notes must be a YAML list")
    config["_config_path"] = str(config_path)
    return config

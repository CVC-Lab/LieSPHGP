"""Strict YAML configuration loading for every quadrotor training run."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_NAME_PATTERN = re.compile(
    r"^(?P<stamp>\d{2}-\d{2}-\d{4}-\d{2}-\d{2})_[a-z0-9_\-]+\.ya?ml$"
)
MODEL_NAMES = {
    "ph_node",
    "ph_nn_lie_ode_integrator",
    "ph_nn_lie_imex",
    "ph_gp_lie_imex",
}

# Every value consumed by training is required to appear in the YAML file.
REQUIRED_PATHS = (
    "schema_version",
    "created_at",
    "model.name",
    "model.family",
    "model.solver",
    "model.hidden_dim",
    "model.activation",
    "model.initialization_gain",
    "model.dissipation_enabled",
    "model.mass_epsilon",
    "model.mass_factor_epsilon",
    "model.optional_subnetwork",
    "data.dataset_path",
    "data.window_points",
    "data.window_stride",
    "data.time_varying_controls",
    "data.max_train_windows",
    "data.max_test_windows",
    "data.observation_noise_override",
    "training.seed",
    "training.total_steps",
    "training.batch_size",
    "training.eval_every",
    "training.checkpoint_every",
    "training.loss",
    "training.loss_scale",
    "training.mass_pretrain_steps",
    "training.mass_pretrain_learning_rate",
    "training.mass_pretrain_samples",
    "training.initial_checkpoint",
    "training.resume_checkpoint",
    "optimizer.name",
    "optimizer.learning_rate",
    "optimizer.beta1",
    "optimizer.beta2",
    "optimizer.epsilon",
    "optimizer.weight_decay",
    "optimizer.gradient_clip_norm",
    "gp.enabled",
    "gp.kernel",
    "gp.matern_smoothness",
    "gp.model_backend",
    "gp.matern_length_scale",
    "gp.period",
    "gp.periodic_length_scale",
    "gp.periodic_harmonics",
    "gp.periodic_dimension",
    "gp.mass_1_feature_count",
    "gp.mass_2_feature_budget",
    "gp.dissipation_v_feature_count",
    "gp.dissipation_w_feature_count",
    "gp.potential_position_feature_count",
    "gp.potential_rotation_feature_budget",
    "gp.potential_include_rotation",
    "gp.control_position_feature_count",
    "gp.control_rotation_feature_budget",
    "gp.initial_log_standard_deviation",
    "gp.kl_beta_max",
    "gp.kl_anneal_steps",
    "gp.data_fit",
    "gp.initial_observation_sigma",
    "gp.fixed_observation_sigma",
    "gp.nll_squared_error_scale",
    "gp.direct_control_map",
    "gp.optional_mean_subnetwork",
    "runtime.device",
    "runtime.device_index",
    "runtime.require_gpu",
    "runtime.jax_enable_x64",
    "runtime.matmul_precision",
    "runtime.enable_compilation_cache",
    "runtime.compilation_cache_dir",
    "experiment.root",
    "experiment.name_suffix",
    "experiment.history_file",
    "experiment.changes",
    "report.enabled",
    "report.auto_generate",
    "report.generator",
    "report.exact_pages",
    "report.output_name",
    "report.title",
    "report.trajectory_count",
    "report.random_seed",
    "report.dpi",
    "report.experiment_dirs",
)


def _lookup(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Required config argument is missing: {dotted_path}")
        value = value[part]
    return value


def resolve_project_path(value: str | Path | None) -> Path | None:
    """Resolve config paths relative to the project root, never the caller CWD."""
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_config(config: dict[str, Any], path: Path) -> None:
    for required in REQUIRED_PATHS:
        _lookup(config, required)

    match = CONFIG_NAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "Config filename must include date and time as "
            "DD-MM-YYYY-HH-MM_<model>.yaml"
        )
    datetime.strptime(match.group("stamp"), "%d-%m-%Y-%H-%M")

    model_name = config["model"]["name"]
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model.name {model_name!r}; expected one of {sorted(MODEL_NAMES)}")
    expected_family = "gp" if model_name == "ph_gp_lie_imex" else "nn"
    if config["model"]["family"] != expected_family:
        raise ValueError(f"{model_name} requires model.family={expected_family!r}")
    if bool(config["gp"]["enabled"]) != (expected_family == "gp"):
        raise ValueError("gp.enabled must be true only for ph_gp_lie_imex")

    solver_by_model = {
        "ph_node": "rk4",
        "ph_nn_lie_ode_integrator": "lie-heun",
        "ph_nn_lie_imex": "lie-imex",
        "ph_gp_lie_imex": "lie-imex",
    }
    if config["model"]["solver"] != solver_by_model[model_name]:
        raise ValueError(
            f"{model_name} requires model.solver={solver_by_model[model_name]!r}"
        )
    if model_name == "ph_gp_lie_imex" and not config["model"]["dissipation_enabled"]:
        raise ValueError("The canonical PH-GP-LieIMEX V2 model requires dissipation_enabled=true")
    if expected_family == "nn" and config["model"]["activation"] != "tanh":
        raise ValueError("The canonical architecture currently supports activation=tanh")

    for optional_path in (
        "model.optional_subnetwork",
        "gp.optional_mean_subnetwork",
    ):
        optional_value = _lookup(config, optional_path)
        if optional_value is not None:
            raise NotImplementedError(
                f"{optional_path}={optional_value!r} was requested, but no optional "
                "feature with that name is implemented"
            )

    positive = (
        "data.window_points",
        "data.window_stride",
        "training.batch_size",
        "training.eval_every",
        "training.checkpoint_every",
        "optimizer.learning_rate",
        "optimizer.epsilon",
        "report.exact_pages",
        "report.trajectory_count",
        "report.dpi",
    )
    for dotted_path in positive:
        if float(_lookup(config, dotted_path)) <= 0:
            raise ValueError(f"{dotted_path} must be positive")
    if int(config["training"]["total_steps"]) < 0:
        raise ValueError("training.total_steps must be non-negative")
    if int(config["report"]["exact_pages"]) != 47:
        raise ValueError("report.exact_pages must be exactly 47")
    if config["training"]["loss"] not in {"trajectory-mse", "se3-nll"}:
        raise ValueError("training.loss must be trajectory-mse or se3-nll")
    if expected_family == "nn" and config["training"]["loss"] != "trajectory-mse":
        raise ValueError("The canonical NN trainers currently require trajectory-mse")
    if expected_family == "gp" and config["training"]["loss"] != config["gp"]["data_fit"]:
        raise ValueError("training.loss and gp.data_fit must match for the GP model")
    if config["gp"]["kernel"] not in {None, "matern"}:
        raise ValueError("gp.kernel must be null for NN models or matern for the GP model")
    backend = config["gp"]["model_backend"]
    if backend not in {None, "utils-gp-model"}:
        raise ValueError("gp.model_backend must be null or utils-gp-model")
    exact_arguments = (
        "gp.matern_length_scale",
        "gp.period",
        "gp.periodic_length_scale",
        "gp.periodic_harmonics",
        "gp.periodic_dimension",
    )
    exact_values = [_lookup(config, name) for name in exact_arguments]
    if backend is None and any(value is not None for value in exact_values):
        raise ValueError("Exact utility GP arguments require gp.model_backend")
    if backend is not None:
        if expected_family != "gp":
            raise ValueError("gp.model_backend is available only for the GP model")
        if any(value is None for value in exact_values):
            raise ValueError("The utils-gp-model backend requires all exact GP arguments")
        if any(float(value) <= 0.0 for value in exact_values[:-1]):
            raise ValueError("Exact utility GP scales and harmonics must be positive")
        if not 0 <= int(config["gp"]["periodic_dimension"]) < 9:
            raise ValueError("gp.periodic_dimension must be between 0 and 8")
    if config["report"]["generator"] != (
        "src.models.SE3_Quadrotor.comparision.generate_report"
    ):
        raise ValueError("report.generator must name the single canonical report module")
    if not isinstance(config["experiment"]["changes"], list):
        raise TypeError("experiment.changes must be a YAML list")


def load_config(path: str | Path, expected_model: str | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("The config root must be a YAML mapping")
    validate_config(payload, config_path)
    if expected_model is not None and payload["model"]["name"] != expected_model:
        raise ValueError(
            f"Entry point {expected_model!r} cannot train config model "
            f"{payload['model']['name']!r}"
        )
    payload["_config_path"] = str(config_path)
    return payload

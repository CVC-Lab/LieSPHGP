"""Independent pure-JAX trainer for PH-GP-LieIMEX."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import platform
import socket
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import MODEL_NAME
from . import network
from .checkpoints import load_checkpoint, save_checkpoint
from .config import load_config, resolve_project_path
from .data import load_dataset
from .experiment import append_history, create_experiment, save_json, utc_now
from .integrator import rollout, rollout_control_sequence, rollout_control_sequence_bounded
from .losses import (
    initialize_likelihood_parameters,
    per_window_se3_nll,
    likelihood_vector,
    pose_loss_components,
    se3_elbo_nll,
)


LOSS_NAMES = ("total", "position", "linear_velocity", "angular_velocity", "attitude")


def parse_config_argument() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args().config


def _log(run_dir: Path, message: str) -> None:
    timestamped = f"{utc_now()} | {message}"
    print(timestamped, flush=True)
    with (run_dir / "training.log").open("a", encoding="utf-8") as handle:
        handle.write(timestamped + "\n")


def _configure_jax(config: dict[str, Any]) -> jax.Device:
    runtime = config["runtime"]
    jax.config.update("jax_enable_x64", bool(runtime["jax_enable_x64"]))
    jax.config.update("jax_default_matmul_precision", runtime["matmul_precision"])
    if runtime["enable_compilation_cache"]:
        cache = resolve_project_path(runtime["compilation_cache_dir"])
        if cache is None:
            raise ValueError("runtime.compilation_cache_dir cannot be null when cache is enabled")
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))

    requested = runtime["device"]
    index = int(runtime["device_index"])
    if requested == "auto":
        devices = jax.devices()
    elif requested in {"cpu", "gpu"}:
        devices = jax.devices(requested)
        if not devices:
            raise RuntimeError(f"No JAX {requested} device is available")
    else:
        raise ValueError("runtime.device must be auto, cpu, or gpu")
    if not 0 <= index < len(devices):
        raise ValueError(
            f"runtime.device_index={index} is outside the available device "
            f"range [0, {len(devices) - 1}] for runtime.device={requested!r}"
        )
    device = devices[index]
    if runtime["require_gpu"] and device.platform != "gpu":
        raise RuntimeError(f"The config requires a GPU, but JAX selected {device}")
    return device


OPTIMIZER_NAMES = ("adam", "prodigy-levels-per-subnetwork")
LEVEL_KEYS = ("level", "log_level")
ADAM_LABEL = "adam"


def _level_labels(params: Any) -> Any:
    """Label the level scalars of every subnetwork with their own group; everything else with Adam."""
    def label(path, _leaf):
        keys = [getattr(part, "key", None) for part in path]
        if len(keys) >= 2 and keys[-1] in LEVEL_KEYS:
            return f"{keys[0]}_level"
        return ADAM_LABEL

    return jax.tree_util.tree_map_with_path(label, params)


def _level_groups(params: Any) -> tuple[str, ...]:
    labels = jax.tree_util.tree_leaves(_level_labels(params))
    return tuple(sorted({name for name in labels if name != ADAM_LABEL}))


def _optimizer(config: dict[str, Any], params: Any) -> optax.GradientTransformation:
    settings = config["optimizer"]
    if settings["name"] not in OPTIMIZER_NAMES:
        raise ValueError(f"optimizer.name must be one of {OPTIMIZER_NAMES}")
    transforms: list[optax.GradientTransformation] = []
    clip = settings["gradient_clip_norm"]
    if clip is not None:
        transforms.append(optax.clip_by_global_norm(float(clip)))
    if float(settings["weight_decay"]) != 0.0:
        transforms.append(optax.add_decayed_weights(float(settings["weight_decay"])))
    if settings["name"] == "adam":
        transforms.append(
            optax.adam(
                learning_rate=float(settings["learning_rate"]),
                b1=float(settings["beta1"]),
                b2=float(settings["beta2"]),
                eps=float(settings["epsilon"]),
            )
        )
    else:
        # Distance-adaptive Prodigy on the level scalars only, one independent instance per
        # subnetwork (each estimates its own distance-to-solution d_t); the GP weight means,
        # log-stds and likelihood sigmas keep plain Adam at optimizer.learning_rate.
        adam = optax.adam(
            learning_rate=float(settings["learning_rate"]),
            b1=float(settings["beta1"]),
            b2=float(settings["beta2"]),
            eps=float(settings["epsilon"]),
        )
        prodigy = {
            name: optax.contrib.prodigy(
                learning_rate=float(settings["level_learning_rate"]),
                betas=(float(settings["beta1"]), float(settings["beta2"])),
                eps=float(settings["epsilon"]),
                estim_lr0=1.0e-3,  # levels are O(1-40) scalars; 1e-6 wastes ~700 steps of warm-up
                safeguard_warmup=True,
            )
            for name in _level_groups(params)
        }
        transforms.append(optax.multi_transform({ADAM_LABEL: adam, **prodigy}, _level_labels))
    return optax.chain(*transforms)


def _prodigy_distances(optimizer_state: Any) -> dict[str, float]:
    """Current distance estimate d_t of every Prodigy group, empty for Adam."""
    found: dict[str, float] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(optimizer_state):
        keys = [getattr(part, "key", getattr(part, "name", None)) for part in path]
        if keys and keys[-1] == "estim_lr":
            group = next((str(key) for key in keys if isinstance(key, str) and key not in ("inner_states", "inner_state", "estim_lr")), "all")
            found[group] = float(np.asarray(jax.device_get(leaf)).reshape(-1)[0])
    return found


def _initialize_gp(
    config: dict[str, Any], key: jax.Array, train_states: jax.Array, settings: dict[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:

    gp = config["gp"]
    key_setup, key_params = jax.random.split(key)
    normalization = network.feature_normalization(np.asarray(jax.device_get(train_states)))
    physical = network.physics_from_settings(settings)
    gp_setup = network.build_gp_setup(
        key_setup,
        normalization=normalization,
        physics=physical,
        mass_1_feature_count=int(gp["mass_1_feature_count"]),
        mass_2_feature_budget=int(gp["mass_2_feature_budget"]),
        dissipation_v_feature_count=int(gp["dissipation_v_feature_count"]),
        dissipation_w_feature_count=int(gp["dissipation_w_feature_count"]),
        potential_position_feature_count=int(gp["potential_position_feature_count"]),
        potential_rotation_feature_budget=int(gp["potential_rotation_feature_budget"]),
        potential_include_rotation=bool(gp["potential_include_rotation"]),
        control_position_feature_count=int(gp["control_position_feature_count"]),
        control_rotation_feature_budget=int(gp["control_rotation_feature_budget"]),
        direct_control_map=bool(gp["direct_control_map"]),
        matern_smoothness=float(gp["matern_smoothness"]),
        mass_eigenvalue_floor=float(config["model"]["mass_epsilon"]),
        dissipation_enabled=bool(config["model"]["dissipation_enabled"]),
        gp_model_backend=gp["model_backend"],
        matern_length_scale=gp["matern_length_scale"],
        period=gp["period"],
        periodic_length_scale=gp["periodic_length_scale"],
        periodic_harmonics=gp["periodic_harmonics"],
        periodic_dimension=gp["periodic_dimension"],
    )
    variational = network.initialize_variational_parameters(
        key_params,
        gp_setup,
        initial_log_standard_deviation=float(gp["initial_log_standard_deviation"]),
        inverse_mass_2_initial_value=float(gp["inverse_mass_2_initial_value"]),
        dissipation_v_initial_value=float(gp["dissipation_v_initial_value"]),
        dissipation_w_initial_value=float(gp["dissipation_w_initial_value"]),
    )
    if gp["data_fit"] == "se3-nll":
        sigma = gp["fixed_observation_sigma"]
        variational["likelihood"] = initialize_likelihood_parameters(
            float(gp["initial_observation_sigma"] if sigma is None else sigma)
        )

    # Reuse the audited GP mass-only pretraining equations without any bridge.
    steps = int(config["training"]["mass_pretrain_steps"])
    if steps == 0:
        return variational, gp_setup, {"steps": 0, "initial_loss": None, "final_loss": None}
    flat = train_states.reshape(-1, train_states.shape[-1])
    count = min(int(config["training"]["mass_pretrain_samples"]), int(flat.shape[0]))
    indices = jnp.linspace(0, flat.shape[0] - 1, count).astype(jnp.int32)
    positions, rotations = flat[indices, :3], flat[indices, 3:12]
    target_1, target_2 = network.physical_inverse_mass_targets(gp_setup)
    optimizer = optax.adam(float(config["training"]["mass_pretrain_learning_rate"]))
    state = optimizer.init(variational)

    def objective(candidate: Any) -> jax.Array:
        model = network.DissipativeSE3HamODE(candidate, gp_setup).sample()
        mass_1 = jax.vmap(model.inverse_mass_1)(positions)
        mass_2 = jax.vmap(model.inverse_mass_2)(rotations)
        return jnp.mean(jnp.square(mass_1 - target_1)) + jnp.mean(jnp.square(mass_2 - target_2))

    @jax.jit
    def step(candidate: Any, optimizer_state: Any) -> tuple[Any, Any, jax.Array]:
        loss, gradients = jax.value_and_grad(objective)(candidate)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, candidate)
        return optax.apply_updates(candidate, updates), optimizer_state, loss

    initial = float(jax.device_get(objective(variational)))
    loss = jnp.asarray(initial)
    for _ in range(steps):
        variational, state, loss = step(variational, state)
    return variational, gp_setup, {
        "steps": steps,
        "learning_rate": float(config["training"]["mass_pretrain_learning_rate"]),
        "sample_count": count,
        "initial_loss": initial,
        "final_loss": float(jax.device_get(loss)),
    }


def _load_requested_checkpoint(
    config: dict[str, Any], variational: Any
) -> tuple[Any, int, Any | None]:
    training = config["training"]
    selected = training["resume_checkpoint"] or training["initial_checkpoint"]
    if selected is None:
        return variational, 0, None
    checkpoint_path = resolve_project_path(selected)
    assert checkpoint_path is not None
    payload = load_checkpoint(checkpoint_path)
    is_resume = training["resume_checkpoint"] is not None
    start_step = int(payload.get("extra", {}).get("step", 0)) if is_resume else 0
    stored_optimizer = payload.get("extra", {}).get("optimizer_state") if is_resume else None
    return payload["params"], start_step, stored_optimizer




def _gp_components(
    config: dict[str, Any],
    variational: Any,
    gp_setup: Any,
    states: jax.Array,
    key: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

    gp = config["gp"]
    model = network.DissipativeSE3HamODE(variational, gp_setup).sample(key)
    h = states[0, 0, -1] * 0.0 + config["_step_size"]
    if config["data"]["time_varying_controls"]:
        prediction = rollout_control_sequence(
            model, states[0], states[1:, :, 18:22], h
        )
    else:
        prediction = rollout(
            model, states[0], h, steps=int(states.shape[0] - 1)
        )
    metrics = pose_loss_components(states[1:], prediction[1:])
    nll = jnp.zeros((13,), dtype=metrics.dtype)
    if gp["data_fit"] == "se3-nll":
        likelihood = variational["likelihood"]
        if gp["fixed_observation_sigma"] is not None:
            likelihood = jax.tree_util.tree_map(jax.lax.stop_gradient, likelihood)
        nll = likelihood_vector(
            se3_elbo_nll(
                states[1:],
                prediction[1:],
                likelihood,
                squared_error_scale=float(gp["nll_squared_error_scale"]),
            )
        )
    kl = network.DissipativeSE3HamODE(variational, gp_setup).kl_loss()
    return metrics, nll, kl, prediction


def _host_metrics(values: jax.Array) -> dict[str, float]:
    host = np.asarray(jax.device_get(values), dtype=np.float64)
    return {name: float(value) for name, value in zip(LOSS_NAMES, host)}


def _tree_is_finite(tree: Any) -> bool:
    return all(
        bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(value)))))
        for value in jax.tree_util.tree_leaves(tree)
    )


def train_from_config(config_path: str | Path, *, expected_model: str) -> Path:
    config = load_config(config_path, expected_model=expected_model)
    device = _configure_jax(config)
    run_dir = create_experiment(config)
    append_history(config, run_dir, action="training", status="started")
    started = time.time()
    metadata: dict[str, Any] = {
        "status": "running",
        "model_name": expected_model,
        "framework": "JAX",
        "dtype": "jax.float64" if config["runtime"]["jax_enable_x64"] else "jax.float32",
        "solver": config["model"]["solver"],
        "config_path": config["_config_path"],
        "config_copy": str(run_dir / Path(config["_config_path"]).name),
        "started_utc": utc_now(),
        "device": str(device),
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "changes": config["experiment"]["changes"],
        "observation_noise_override": config["data"]["observation_noise_override"],
    }
    stats: dict[str, list[float]] = {
        "step": [], "train_total": [], "test_total": [], "position": [],
        "linear_velocity": [], "angular_velocity": [], "attitude": [],
        "gradient_norm": [], "step_seconds": [], "objective": [], "kl": [],
        # Per-optimizer-step training losses (sampled weights) for the report curves.
        "train_step": [], "train_loss": [], "train_position": [], "train_linear_velocity": [],
        "train_angular_velocity": [], "train_attitude": [],
        "long_horizon_seconds": [], "long_horizon_loss": [], "long_horizon_masked_fraction": [],
        "test_long_position_rms": [], "test_long_position_relative": [], "test_long_masked_fraction": [],
    }
    try:
        dataset_path = resolve_project_path(config["data"]["dataset_path"])
        assert dataset_path is not None
        train_numpy, test_numpy, times, dataset_settings, train_flights_numpy, test_flights_numpy = load_dataset(
            dataset_path,
            window_points=int(config["data"]["window_points"]),
            window_stride=int(config["data"]["window_stride"]),
            max_train_windows=config["data"]["max_train_windows"],
            max_test_windows=config["data"]["max_test_windows"],
            observation_noise_override=config["data"]["observation_noise_override"],
            seed=int(config["training"]["seed"]),
        )
        step_size = float(times[1] - times[0])
        config["_step_size"] = step_size
        train_states = jax.device_put(jnp.asarray(train_numpy), device)
        test_states = jax.device_put(jnp.asarray(test_numpy), device)
        train_flights = jax.device_put(jnp.asarray(train_flights_numpy), device)
        test_flights = jax.device_put(jnp.asarray(test_flights_numpy), device)
        seed = int(config["training"]["seed"])
        key = jax.random.PRNGKey(seed)
        key, initialization_key = jax.random.split(key)
        variational, gp_setup, mass_audit = _initialize_gp(
            config, initialization_key, train_states, dataset_settings
        )
        variational, start_step, stored_optimizer_state = _load_requested_checkpoint(config, variational)
        optimizer = _optimizer(config, variational)
        optimizer_state = (
            optimizer.init(variational)
            if stored_optimizer_state is None
            else jax.tree_util.tree_map(jnp.asarray, stored_optimizer_state)
        )
        training = config["training"]
        batch_size = min(int(training["batch_size"]), int(train_states.shape[1]))
        scale = float(training["loss_scale"])
        # Minibatch ELBO: the KL is spread over every training transition, not the batch.
        number_of_observations = float(train_states.shape[1] * (train_states.shape[0] - 1))

        # ---- Multi-horizon term: curriculum of long rollouts on a few full flights ----
        # schedule: list of [start_step, horizon_seconds]; before the first start the term is off.
        long_schedule = sorted(
            (int(start), float(seconds)) for start, seconds in training["long_horizon_schedule"]
        )
        long_flights = min(int(training["long_horizon_flights"]), int(train_flights.shape[1]))
        long_weight = float(training["long_horizon_weight"])
        truncation_steps = int(training["long_horizon_truncation_steps"])
        twist_bound = float(training["long_horizon_twist_bound"])
        # Optional: start each long training rollout at a random time inside the flight instead of t=0,
        # so hover, cruise and recovery segments all enter the long-horizon term.
        random_offsets = bool(training.get("long_horizon_random_offsets", False))
        flight_length = int(train_flights.shape[0])

        def long_horizon_steps(step: int) -> tuple[int, int]:
            """(number of transitions, start step of the active stage); (0, 0) while the term is off."""
            active = [(start, seconds) for start, seconds in long_schedule if step >= start]
            if not active:
                return 0, 0
            start, seconds = active[-1]
            return min(int(round(seconds / step_size)), flight_length - 1), start

        def long_horizon_term(candidate, flights, random_key):
            """Finite-masked mean NLL of bounded long rollouts; returns (term, masked_fraction)."""
            model = network.DissipativeSE3HamODE(candidate, gp_setup).sample(random_key)
            h = flights[0, 0, -1] * 0.0 + step_size
            prediction, finite = rollout_control_sequence_bounded(
                model, flights[0], flights[1:, :, 18:22], h,
                twist_bound=twist_bound, truncation_steps=truncation_steps,
            )
            # The long term uses the short-window likelihood scales without moving them.
            likelihood = jax.tree_util.tree_map(jax.lax.stop_gradient, candidate["likelihood"])
            per_window = per_window_se3_nll(
                flights[1:], prediction[1:], likelihood,
                squared_error_scale=float(config["gp"]["nll_squared_error_scale"]),
            )
            finite = jax.lax.stop_gradient(finite & jnp.isfinite(per_window))
            safe = jnp.where(finite, per_window, 0.0)
            count = jnp.maximum(1.0, jnp.sum(finite))
            return jnp.sum(safe) / count, 1.0 - jnp.mean(finite.astype(jnp.float32))

        def objective(candidate, batch, random_key, step_value, flights, long_start, long_on):
            metrics, nll, kl, _ = _gp_components(
                config, candidate, gp_setup, batch, random_key
            )
            data_fit = nll[0] if config["gp"]["data_fit"] == "se3-nll" else metrics[0]
            progress = jnp.minimum(
                1.0, step_value / max(1, int(config["gp"]["kl_anneal_steps"]))
            )
            beta = float(config["gp"]["kl_beta_max"]) * progress
            total = scale * data_fit + beta * kl / number_of_observations
            long_value = jnp.asarray(0.0, dtype=total.dtype)
            masked = jnp.asarray(0.0, dtype=total.dtype)
            if long_on:
                key_long = jax.random.fold_in(random_key, 1)
                long_value, masked = long_horizon_term(candidate, flights, key_long)
                ramp = jnp.minimum(1.0, (step_value - long_start) / 1000.0)
                total = total + long_weight * ramp * long_value
            return total, (metrics, nll, kl, long_value, masked)

        @partial(jax.jit, static_argnames=("long_on",))
        def training_step(candidate: Any, state: Any, batch: jax.Array, random_key: jax.Array, step_value: jax.Array, flights: jax.Array, long_start: jax.Array, long_on: bool):
            (value, auxiliaries), gradients = jax.value_and_grad(
                objective, has_aux=True
            )(candidate, batch, random_key, step_value, flights, long_start, long_on)
            updates, state = optimizer.update(gradients, state, candidate)
            return optax.apply_updates(candidate, updates), state, value, auxiliaries, optax.global_norm(gradients)

        # ---- 1 s open-loop metric on the clean test flights (posterior mean, no gradient) ----
        # One or several clean-test rollout horizons (scalar or list in the config); the first one keeps
        # the historical stats keys, the others are stored as test_long{tag}_* (e.g. test_long3s_position_rms).
        test_seconds_setting = training["long_horizon_test_seconds"]
        test_seconds = [
            float(value) for value in (
                test_seconds_setting if isinstance(test_seconds_setting, (list, tuple)) else [test_seconds_setting]
            )
        ]
        test_horizons = [min(int(round(seconds / step_size)), flight_length - 1) for seconds in test_seconds]

        def _horizon_tag(seconds: float) -> str:
            return f"{seconds:g}s".replace(".", "p")

        def _make_evaluate_long(test_horizon: int):
            @jax.jit
            def evaluate_long(candidate):
                model = network.DissipativeSE3HamODE(candidate, gp_setup).sample()
                flights = test_flights[: test_horizon + 1]
                h = flights[0, 0, -1] * 0.0 + step_size
                prediction, finite = rollout_control_sequence_bounded(
                    model, flights[0], flights[1:, :, 18:22], h, twist_bound=twist_bound, truncation_steps=truncation_steps,
                )
                error = jnp.sum((prediction[1:, :, :3] - flights[1:, :, :3]) ** 2, axis=-1)
                excursion = jnp.sum((flights[1:, :, :3] - flights[:1, :, :3]) ** 2, axis=-1)
                rms = jnp.sqrt(jnp.mean(error))
                relative = rms / jnp.sqrt(jnp.maximum(jnp.mean(excursion), 1e-30))
                return rms, relative, 1.0 - jnp.mean(finite.astype(jnp.float32))
            return evaluate_long

        evaluate_long_functions = [_make_evaluate_long(horizon) for horizon in test_horizons]
        evaluate_long = evaluate_long_functions[0]

        @jax.jit
        def evaluate(candidate, states):
            return _gp_components(config, candidate, gp_setup, states, None)

        metadata.update(
            {
                "dataset_path": str(dataset_path),
                "dataset_settings_summary": {
                    key_name: dataset_settings.get(key_name)
                    for key_name in ("dataset_name", "state_layout", "control_layout", "damping_law")
                },
                "train_shape": list(train_numpy.shape),
                "test_shape": list(test_numpy.shape),
                "step_size_seconds": step_size,
                "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "total_steps": int(training["total_steps"]),
                "batch_size": batch_size,
                "eval_every": int(training["eval_every"]),
                "checkpoint_every": int(training["checkpoint_every"]),
                "integrator": config["model"]["solver"],
                "data_fit": config["gp"]["data_fit"],
                "learning_rate": float(config["optimizer"]["learning_rate"]),
                "gradient_clip_global_norm": config["optimizer"]["gradient_clip_norm"],
                "kl_beta_max": float(config["gp"]["kl_beta_max"]),
                "kl_anneal_steps": int(config["gp"]["kl_anneal_steps"]),
                "posterior_initial_log_standard_deviation": float(config["gp"]["initial_log_standard_deviation"]),
                "training_window_points": int(config["data"]["window_points"]),
                "window_stride_physics_steps": int(config["data"]["window_stride"]),
                "time_varying_controls": bool(config["data"]["time_varying_controls"]),
                "architecture_variant": "learned-levels-matern-gp-v3",
                "subnetwork_inputs": {
                    "M1": "x in R3; scalar output mu(x) times I3",
                    "M2": "vec(R) in R9 with SO(3) geodesic features",
                    "V": "x in R3" if not config["gp"]["potential_include_rotation"] else "(x,R) in SE(3)",
                    "g": "(x,R) in SE(3)",
                    "Dv": "v_b in R3",
                    "Dw": "omega_b in R3",
                },
                "operator_parameterization": {
                    "M1_inverse": "exp(lambda0 + GP(x)) I3",
                    "M2_inverse": "L L^T with L from six learned levels plus GP(R)",
                    "Dv": "L L^T with L from six learned levels plus GP(v_b)",
                    "Dw": "L L^T with L from six learned levels plus GP(omega_b)",
                    "V": "GP(x)",
                    "g": "Lambda0 (6x4 learned) + reshape(GP(x,R))",
                },
                "initial_operator_values": {
                    "M1_inverse": 1.0,
                    "M2_inverse": float(config["gp"]["inverse_mass_2_initial_value"]),
                    "Dv": float(config["gp"]["dissipation_v_initial_value"]),
                    "Dw": float(config["gp"]["dissipation_w_initial_value"]),
                },
                "parameter_count": sum(int(value.size) for value in jax.tree_util.tree_leaves(variational)),
                "mass_pretraining": mass_audit,
            }
        )
        save_json(run_dir / "metadata.json", metadata)
        save_checkpoint(
            run_dir / f"checkpoint_step_{start_step:05d}.pkl",
            params=variational,
            extra={
                "step": start_step,
                "gp_setup": gp_setup,
                "optimizer_state": optimizer_state,
            },
        )
        _log(run_dir, f"model={expected_model} device={device} train_shape={train_numpy.shape}")

        rng = np.random.default_rng(seed)
        total_steps = int(training["total_steps"])
        final_test: dict[str, float] | None = None
        for step in range(start_step, total_steps + 1):
            step_seconds = 0.0
            train_values = None
            objective_value = None
            gradient_norm = None
            kl_value = jnp.asarray(0.0)
            if step > start_step:
                indices = rng.choice(int(train_states.shape[1]), size=batch_size, replace=False)
                batch = train_states[:, indices]
                key, sample_key = jax.random.split(key)
                long_steps, long_start = long_horizon_steps(step)
                long_on = long_steps > 0
                flight_indices = rng.choice(int(train_flights.shape[1]), size=long_flights, replace=False)
                if long_on and random_offsets:
                    offsets = rng.integers(0, flight_length - long_steps, size=long_flights)
                    time_index = offsets[None, :] + np.arange(long_steps + 1)[:, None]  # (long_steps+1, long_flights)
                    flights = train_flights[jnp.asarray(time_index), jnp.asarray(flight_indices)[None, :]]
                else:
                    flights = train_flights[: long_steps + 1, flight_indices] if long_on else train_flights[:1, flight_indices]
                tick = time.perf_counter()
                variational, optimizer_state, objective_value, auxiliaries, gradient_norm = training_step(
                    variational, optimizer_state, batch, sample_key, jnp.asarray(step, dtype=jnp.float32),
                    flights, jnp.asarray(long_start, dtype=jnp.float32), long_on,
                )
                jax.device_get(objective_value)
                if not np.isfinite(float(jax.device_get(objective_value))) or not np.isfinite(
                    float(jax.device_get(gradient_norm))
                ):
                    raise FloatingPointError(
                        f"Non-finite optimizer value at step {step}: "
                        f"objective={objective_value}, gradient_norm={gradient_norm}"
                    )
                step_seconds = time.perf_counter() - tick
                train_values, _nll, kl_value, long_value, long_masked = auxiliaries
                step_host = _host_metrics(train_values)
                stats["train_step"].append(float(step))
                stats["train_loss"].append(step_host["total"])
                for name in LOSS_NAMES[1:]:
                    stats[f"train_{name}"].append(step_host[name])

            should_evaluate = step == start_step or step % int(training["eval_every"]) == 0 or step == total_steps
            if should_evaluate:
                test_values, _test_nll, test_kl, _prediction = evaluate(variational, test_states)
                jax.device_get(test_values)
                if train_values is None:
                    train_values, _train_nll, kl_value, _ = evaluate(variational, train_states[:, :batch_size])
                train_host = _host_metrics(train_values)
                final_test = _host_metrics(test_values)
                if not all(np.isfinite(value) for value in (*train_host.values(), *final_test.values())):
                    raise FloatingPointError(f"Non-finite evaluation metric at step {step}")
                stats["step"].append(float(step))
                stats["train_total"].append(train_host["total"])
                stats["test_total"].append(final_test["total"])
                for name in LOSS_NAMES[1:]:
                    stats[name].append(final_test[name])
                stats["gradient_norm"].append(
                    0.0 if gradient_norm is None else float(jax.device_get(gradient_norm))
                )
                stats["step_seconds"].append(step_seconds)
                stats["objective"].append(
                    train_host["total"] if objective_value is None else float(jax.device_get(objective_value))
                )
                stats["kl"].append(float(jax.device_get(test_kl)))
                long_steps_now, _ = long_horizon_steps(step)
                stats["long_horizon_seconds"].append(long_steps_now * step_size)
                stats["long_horizon_loss"].append(0.0 if train_values is None or long_steps_now == 0 else float(jax.device_get(long_value)))
                stats["long_horizon_masked_fraction"].append(0.0 if train_values is None or long_steps_now == 0 else float(jax.device_get(long_masked)))
                long_rms, long_relative, long_test_masked = (float(jax.device_get(v)) for v in evaluate_long(variational))
                stats["test_long_position_rms"].append(long_rms)
                stats["test_long_position_relative"].append(long_relative)
                stats["test_long_masked_fraction"].append(long_test_masked)
                extra_long_log = ""
                for seconds, evaluate_extra in zip(test_seconds[1:], evaluate_long_functions[1:]):
                    rms_extra, relative_extra, masked_extra = (float(jax.device_get(v)) for v in evaluate_extra(variational))
                    tag = _horizon_tag(seconds)
                    stats.setdefault(f"test_long{tag}_position_rms", []).append(rms_extra)
                    stats.setdefault(f"test_long{tag}_position_relative", []).append(relative_extra)
                    stats.setdefault(f"test_long{tag}_masked_fraction", []).append(masked_extra)
                    extra_long_log += f" test{tag}[pos_rms={rms_extra:.3e} rel={relative_extra:.2f} masked={masked_extra:.2f}]"
                distances = _prodigy_distances(optimizer_state)
                for group, value in distances.items():
                    stats.setdefault(f"prodigy_d_{group}", []).append(value)
                np.savez_compressed(run_dir / "training_stats.npz", **{key_name: np.asarray(value) for key_name, value in stats.items()})
                _log(
                    run_dir,
                    f"step={step} train={train_host['total']:.6e} test={final_test['total']:.6e} "
                    f"position={final_test['position']:.3e} attitude={final_test['attitude']:.3e} "
                    f"long[h={long_steps_now * step_size:.2f}s loss={stats['long_horizon_loss'][-1]:.3e} masked={stats['long_horizon_masked_fraction'][-1]:.2f}] "
                    f"test{_horizon_tag(test_seconds[0])}[pos_rms={long_rms:.3e} rel={long_relative:.2f} masked={long_test_masked:.2f}]"
                    + extra_long_log
                    + ("" if not distances else " prodigy_d=" + ",".join(f"{k}:{v:.3g}" for k, v in distances.items())),
                )

            if step > start_step and (
                step % int(training["checkpoint_every"]) == 0 or step == total_steps
            ):
                save_checkpoint(
                    run_dir / f"checkpoint_step_{step:05d}.pkl",
                    params=variational,
                    extra={
                        "step": step,
                        "gp_setup": gp_setup,
                        "optimizer_state": optimizer_state,
                    },
                )

        if final_test is None:
            raise RuntimeError("Training completed without an evaluation")
        report_count = min(int(config["report"]["trajectory_count"]), int(test_states.shape[1]))
        report_rng = np.random.default_rng(int(config["report"]["random_seed"]))
        report_indices = np.sort(
            report_rng.choice(int(test_states.shape[1]), size=report_count, replace=False)
        )
        report_states = test_states[:, report_indices]
        final_metrics, _final_nll, _final_kl, prediction = evaluate(variational, report_states)
        jax.device_get(prediction)
        if not _tree_is_finite(variational) or not bool(
            np.asarray(jax.device_get(jnp.all(jnp.isfinite(prediction))))
        ):
            raise FloatingPointError("Final parameters or report prediction are non-finite")
        np.savez_compressed(
            run_dir / "report_data.npz",
            time=np.asarray(times),
            reference=np.asarray(jax.device_get(report_states)),
            prediction=np.asarray(jax.device_get(prediction)),
        )
        save_checkpoint(
            run_dir / "checkpoint_final.pkl",
            params=variational,
            extra={
                "step": total_steps,
                "gp_setup": gp_setup,
                "optimizer_state": optimizer_state,
            },
        )
        metadata.update(
            {
                "status": "completed",
                "completed_utc": utc_now(),
                "elapsed_seconds": time.time() - started,
                "final_test": final_test,
                "optimizer_name": config["optimizer"]["name"],
                "final_prodigy_distances": _prodigy_distances(optimizer_state),
                "final_checkpoint": "checkpoint_final.pkl",
                "final_loss_vector": np.asarray(jax.device_get(final_metrics)).tolist(),
            }
        )
        save_json(run_dir / "metadata.json", metadata)
        append_history(
            config,
            run_dir,
            action="training",
            status="completed",
            details="checkpoint=checkpoint_final.pkl",
        )

        if config["report"]["enabled"] and config["report"]["auto_generate"]:
            from ..comparision.generate_report import generate_reports

            generate_reports(
                run_dir / Path(config["_config_path"]).name,
                experiment_dirs=[run_dir],
            )
        return run_dir
    except BaseException as error:
        metadata.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "elapsed_seconds": time.time() - started,
                "error": repr(error),
            }
        )
        save_json(run_dir / "metadata.json", metadata)
        append_history(config, run_dir, action="training", status="failed", details=repr(error))
        raise


if __name__ == "__main__":
    train_from_config(parse_config_argument(), expected_model=MODEL_NAME)

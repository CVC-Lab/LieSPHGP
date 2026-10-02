"""Independent pure-JAX trainer for one neural quadrotor model."""

from __future__ import annotations

import math

import argparse
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
from .checkpoints import load_checkpoint, save_checkpoint
from .config import load_config, resolve_project_path
from .data import load_dataset
from .experiment import append_history, create_experiment, save_json, utc_now
from .integrator import lie_imex_sde_step, rollout, rollout_control_sequence, rollout_control_sequence_sde
from .losses import (
    LIKELIHOOD_NAMES,
    likelihood_vector,
    observation_dictionary,
    pose_loss_components,
    project_rotation,
    se3_moment_nll,
    se3_pseudo_likelihood,
    se3_transition_nll,
    transition_diffusion,
)
from .network import (
    initialize_parameters as initialize_nn_parameters,
    inverse_mass_1 as nn_inverse_mass_1,
    inverse_mass_2 as nn_inverse_mass_2,
    load_legacy_parameters,
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


OPTIMIZER_NAMES = ("adam", "prodigy-per-subnetwork")
SUBNETWORKS = ("M1_net", "M2_net", "Dv_net", "Dw_net", "V_net", "g_net", "sigma_net")
NOISE_LEAVES = ("process_log_sigma", "log_sigma_observation")
ADAM_LABEL, FROZEN_LABEL = "adam", "frozen"


def _subnetwork_labels(model: Any) -> Any:
    """Each array leaf labelled with its subnetwork; the SDE noise scales go to Adam; the dissipation switch
    is frozen."""
    def label(path, _leaf):
        head = path[0] if path else None
        name = getattr(head, "name", getattr(head, "key", None))
        if name in SUBNETWORKS:
            return name
        return ADAM_LABEL if name in NOISE_LEAVES else FROZEN_LABEL

    return jax.tree_util.tree_map_with_path(label, model)


def _optimizer(config: dict[str, Any], model: Any = None) -> optax.GradientTransformation:
    settings = config["optimizer"]
    if settings["name"] not in OPTIMIZER_NAMES:
        raise ValueError(f"optimizer.name must be one of {OPTIMIZER_NAMES}")
    transforms: list[optax.GradientTransformation] = []
    clip = settings["gradient_clip_norm"]
    if clip is not None:
        transforms.append(optax.clip_by_global_norm(float(clip)))
    if float(settings["weight_decay"]) != 0.0:
        transforms.append(optax.add_decayed_weights(float(settings["weight_decay"])))
    learning_rate = float(settings["learning_rate"])
    if settings.get("schedule") == "cosine":
        # optimizer.schedule = cosine: decay to optimizer.final_learning_fraction (default 0.1) over training.total_steps
        learning_rate = optax.cosine_decay_schedule(
            learning_rate, int(config["training"]["total_steps"]), alpha=float(settings.get("final_learning_fraction", 0.1)))
    adam = optax.adam(
        learning_rate=learning_rate,
        b1=float(settings["beta1"]),
        b2=float(settings["beta2"]),
        eps=float(settings["epsilon"]),
    )
    if settings["name"] == "adam":
        transforms.append(adam)
        return optax.chain(*transforms)
    # One independent Prodigy per subnetwork (sigma_net included when present), each with its own distance
    # estimate d_t; the SDE noise scales stay on Adam at optimizer.learning_rate.
    present = set(jax.tree_util.tree_leaves(_subnetwork_labels(model)))
    groups: dict[str, optax.GradientTransformation] = {ADAM_LABEL: adam, FROZEN_LABEL: optax.set_to_zero()}
    for name in SUBNETWORKS:
        if name in present:
            groups[name] = optax.contrib.prodigy(
                learning_rate=float(settings.get("prodigy_learning_rate", 1.0)),
                betas=(float(settings["beta1"]), float(settings["beta2"])),
                eps=float(settings["epsilon"]),
                estim_lr_coef=float(settings.get("prodigy_d_coef", 1.0)),
                safeguard_warmup=True,
            )
    transforms.append(optax.multi_transform(groups, _subnetwork_labels))
    return optax.chain(*transforms)


def _prodigy_distances(optimizer_state: Any) -> dict[str, float]:
    """Current distance estimate d_t of every Prodigy group (empty for Adam)."""
    found: dict[str, float] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(optimizer_state):
        keys = [getattr(part, "key", getattr(part, "name", None)) for part in path]
        if keys and keys[-1] == "estim_lr":
            group = next((str(k) for k in keys if isinstance(k, str) and k in SUBNETWORKS), "all")
            found[group] = float(np.asarray(jax.device_get(leaf)).reshape(-1)[0])
    return found


def _physics(settings: dict[str, Any]) -> tuple[float, np.ndarray]:
    vehicle = settings.get("vehicle_parameters", {})
    mass = float(vehicle.get("mass", settings.get("m", 0.027)))
    inertia = vehicle.get("inertia", settings.get("J_diag", (2.3951e-5, 2.3951e-5, 3.2347e-5)))
    inertia_array = np.asarray(inertia, dtype=np.float32)
    if inertia_array.shape == (3, 3):
        inertia_array = np.diag(inertia_array)
    if inertia_array.shape != (3,):
        raise ValueError(f"Expected a diagonal three-vector inertia, got {inertia_array.shape}")
    return mass, inertia_array


def _pretrain_nn_mass(
    model: Any,
    train_states: jax.Array,
    settings: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    training = config["training"]
    steps = int(training["mass_pretrain_steps"])
    if steps == 0:
        return model, {"steps": 0, "initial_loss": None, "final_loss": None}
    flat = train_states.reshape(-1, train_states.shape[-1])
    count = min(int(training["mass_pretrain_samples"]), int(flat.shape[0]))
    indices = jnp.linspace(0, flat.shape[0] - 1, count).astype(jnp.int32)
    positions, rotations = flat[indices, :3], flat[indices, 3:12]
    mass, inertia = _physics(settings)
    target_1 = jnp.eye(3, dtype=jnp.float32) / mass
    target_2 = jnp.diag(1.0 / jnp.asarray(inertia))
    optimizer = optax.adam(float(training["mass_pretrain_learning_rate"]))
    optimizer_state = optimizer.init(model)

    def objective(candidate: Any) -> jax.Array:
        matrices_1 = jax.vmap(lambda value: nn_inverse_mass_1(candidate, value))(positions)
        matrices_2 = jax.vmap(lambda value: nn_inverse_mass_2(candidate, value))(rotations)
        return jnp.mean(jnp.square(matrices_1 - target_1)) + jnp.mean(
            jnp.square(matrices_2 - target_2)
        )

    @jax.jit
    def step(candidate: Any, state: Any) -> tuple[Any, Any, jax.Array]:
        loss, gradients = jax.value_and_grad(objective)(candidate)
        updates, state = optimizer.update(gradients, state, candidate)
        return optax.apply_updates(candidate, updates), state, loss

    initial = float(jax.device_get(objective(model)))
    loss = jnp.asarray(initial)
    for _ in range(steps):
        model, optimizer_state, loss = step(model, optimizer_state)
    return model, {
        "steps": steps,
        "learning_rate": float(training["mass_pretrain_learning_rate"]),
        "sample_count": count,
        "initial_loss": initial,
        "final_loss": float(jax.device_get(loss)),
    }


def _initialize_nn(
    config: dict[str, Any], key: jax.Array, train_states: jax.Array, settings: dict[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:
    model_config = config["model"]
    sde = config.get("sde") or {}
    model = initialize_nn_parameters(
        key,
        hidden_dim=int(model_config["hidden_dim"]),
        initialization_gain=float(model_config["initialization_gain"]),
        mass_epsilon=float(model_config["mass_epsilon"]),
        mass_factor_epsilon=float(model_config["mass_factor_epsilon"]),
        dissipation_enabled=bool(model_config["dissipation_enabled"]),
        initial_process_sigma=(float(sde.get("initial_process_sigma_linear", 0.05)),
                               float(sde.get("initial_process_sigma_angular", 0.05))),
        initial_observation_sigma=float(sde.get("initial_observation_sigma", 0.3)),
        diffusion_input=str(sde.get("diffusion_input", "constant")),
    )
    model, audit = _pretrain_nn_mass(model, train_states, settings, config)
    return model, None, audit



def _load_requested_checkpoint(
    config: dict[str, Any], model: Any
) -> tuple[Any, int, Any | None]:
    training = config["training"]
    selected = training["resume_checkpoint"] or training["initial_checkpoint"]
    if selected is None:
        return model, 0, None
    checkpoint_path = resolve_project_path(selected)
    assert checkpoint_path is not None
    payload = load_checkpoint(checkpoint_path)
    is_resume = training["resume_checkpoint"] is not None
    start_step = int(payload.get("extra", {}).get("step", 0)) if is_resume else 0
    stored_optimizer = payload.get("extra", {}).get("optimizer_state") if is_resume else None
    restored = payload["params"]
    if isinstance(restored, dict):
        return load_legacy_parameters(model, restored), start_step, None
    return restored, start_step, stored_optimizer


def _trajectory(config: dict[str, Any], model: Any, states: jax.Array):
    step_size = states[0, 0, -1] * 0.0 + config["_step_size"]
    if config["data"]["time_varying_controls"]:
        prediction = rollout_control_sequence(
            model, states[0], states[1:, :, 18:22], step_size
        )
    else:
        prediction = rollout(
            model, states[0], step_size, steps=int(states.shape[0] - 1)
        )
    return pose_loss_components(states[1:], prediction[1:]), prediction


def _observation_log_sigma(config: dict[str, Any], model: Any) -> jax.Array:
    """(4,) log observation sigma: the learned one, or the constant sde.fixed_observation_sigma (the learned leaf
    then receives no gradient), so the observation noise cannot absorb the process noise."""
    fixed = (config.get("sde") or {}).get("fixed_observation_sigma")
    if fixed is None:
        return model.log_sigma_observation
    return jnp.full((4,), math.log(float(fixed)), dtype=model.log_sigma_observation.dtype)


def _sde_trajectory(config: dict[str, Any], model: Any, states: jax.Array, key: jax.Array | None):
    """S stochastic Lie-IMEX rollouts, the moment-matched NLL, and the (projected) sample-mean path.

    ``sde.enabled = false`` keeps the same likelihood with a single noise-free path, which is the matched
    control for the ODE/SDE comparison (same objective, no process noise).  A no-gradient preview screens
    windows whose path diverges: they are held inert inside the differentiable rollout, because the zero
    cotangent of a masked window still traverses an exploded path and 0 * inf = NaN reaches the weights.
    """
    sde, training = config["sde"], config["training"]
    if not config["data"]["time_varying_controls"]:
        raise ValueError("the SDE variant needs data.time_varying_controls = true")
    enabled = bool(sde.get("enabled", True))
    sample_count = int(sde["samples"]) if enabled else 1
    step_size = states[0, 0, -1] * 0.0 + config["_step_size"]
    controls = states[1:, :, 18:22]
    keys = jax.random.split(jax.random.PRNGKey(0) if key is None else key, sample_count)
    scale = 1.0 if enabled else 0.0

    def noise_of(path_key):
        return scale * jax.random.normal(path_key, controls.shape[:2] + (6,), dtype=states.dtype)

    limit = training.get("diverged_window_position_limit")
    frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, model)
    preview = jax.vmap(lambda k: rollout_control_sequence_sde(
        frozen, jax.lax.stop_gradient(states[0]), controls, step_size, noise_of(k)))(keys)
    healthy = jnp.all(jnp.isfinite(preview), axis=(0, 1, 3))
    if limit is not None:
        deviation = jnp.max(jnp.linalg.norm(jnp.nan_to_num(preview[..., :3] - states[None, ..., :3], nan=jnp.inf),
                                            axis=-1), axis=(0, 1))
        healthy = healthy & (deviation <= float(limit))
    twist = jnp.nan_to_num(preview[..., 12:18], nan=jnp.inf)
    healthy = healthy & (jnp.max(jnp.linalg.norm(twist[..., 3:6], axis=-1), axis=(0, 1)) <= float(sde.get("screen_omega_limit", 100.0)))
    healthy = healthy & (jnp.max(jnp.linalg.norm(twist[..., :3], axis=-1), axis=(0, 1)) <= float(sde.get("screen_velocity_limit", 50.0)))
    healthy = jax.lax.stop_gradient(healthy)

    samples = jax.vmap(lambda k: rollout_control_sequence_sde(
        model, states[0], controls, step_size, noise_of(k), healthy))(keys)
    mean_state = jnp.mean(samples, axis=0)
    mean_path = mean_state.at[..., 3:12].set(
        project_rotation(mean_state[..., 3:12]).reshape(*mean_state.shape[:-1], 9))
    metrics = pose_loss_components(states[1:], mean_path[1:])
    samples = jnp.where(healthy[None, None, :, None], samples, jnp.nan)
    horizon_seconds = jnp.arange(1, states.shape[0], dtype=states.dtype) * step_size
    values = se3_moment_nll(
        states[1:], samples[:, 1:], observation_dictionary(_observation_log_sigma(config, model)), model.process_log_sigma,
        decay=float(training.get("loss_horizon_weight") or 0.0),
        position_limit=None if limit is None else float(limit),
        horizon_seconds=horizon_seconds,
    )
    return metrics, likelihood_vector(values), mean_path


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
    }
    stats: dict[str, list[float]] = {
        "step": [], "train_total": [], "test_total": [], "position": [],
        "linear_velocity": [], "angular_velocity": [], "attitude": [],
        "gradient_norm": [], "step_seconds": [], "objective": [], "kl": [],
    }
    try:
        dataset_path = resolve_project_path(config["data"]["dataset_path"])
        assert dataset_path is not None
        train_numpy, test_numpy, times, dataset_settings = load_dataset(
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
        seed = int(config["training"]["seed"])
        key = jax.random.PRNGKey(seed)
        key, initialization_key = jax.random.split(key)
        model, banks, mass_audit = _initialize_nn(
            config, initialization_key, train_states, dataset_settings
        )
        model, start_step, stored_optimizer_state = _load_requested_checkpoint(config, model)
        optimizer = _optimizer(config, model)
        optimizer_state = (
            optimizer.init(model)
            if stored_optimizer_state is None
            else jax.tree_util.tree_map(jnp.asarray, stored_optimizer_state)
        )
        training = config["training"]
        batch_size = min(int(training["batch_size"]), int(train_states.shape[1]))
        scale = float(training["loss_scale"])

        # ---- Optional physics priors, ported verbatim from ph_gp_lie_imex/train.py so the baselines can be trained
        # with the same penalties as the GP.  Both are written in gauge-invariant products and call only the six
        # operator methods this module shares with the GP; weight 0 / absent keys leave the objective untouched.
        gp_block = config.get("gp") or {}
        gravity_weight = float(gp_block.get("gravity_penalty_weight") or 0.0)
        gravity_value = float(gp_block.get("known_gravity") or 0.0)
        gravity_points = int(gp_block.get("gravity_penalty_points") or 1024)
        actuation_weight = float(gp_block.get("actuation_penalty_weight") or 0.0)
        actuation_reference = float(gravity_value if gravity_value > 0.0 else 9.81)
        if gravity_weight > 0.0 and gravity_value <= 0.0:
            raise ValueError("gp.gravity_penalty_weight needs gp.known_gravity > 0")

        def gravity_penalty(candidate, batch):
            # L_g = E_x |mu(x) grad V(x) - g e3|^2 / g^2 on a stride of the batch states.
            poses = batch[:, :, :12].reshape(-1, 12)
            poses = poses[:: max(1, poses.shape[0] // gravity_points)]
            mu = jax.vmap(lambda pose: candidate.inverse_mass_1(pose[:3])[0, 0])(poses)
            grad_v = jax.vmap(lambda pose: jax.grad(candidate.potential)(pose)[:3])(poses)
            target = jnp.array([0.0, 0.0, gravity_value], dtype=poses.dtype)
            return jnp.mean(jnp.sum(jnp.square(mu[:, None] * grad_v - target), axis=-1)) / (gravity_value**2)

        def actuation_penalty(candidate, batch):
            # L_a = E_x [ |mu g_xy e1 u_h|^2 + |M2^-1 g_tau e1 u_h|^2 ] / g^2: the thrust command produces no lateral
            # body force and no torque.  Direction only; the thrust gain and the torque gains stay learned.
            poses = batch[:, :, :12].reshape(-1, 12)
            poses = poses[:: max(1, poses.shape[0] // gravity_points)]
            thrust_magnitude = jnp.mean(jnp.abs(batch[:, :, 18]))

            def leak(pose):
                g_matrix = candidate.control_matrix(pose)
                mu = candidate.inverse_mass_1(pose[:3])[0, 0]
                lateral = mu * g_matrix[:2, 0] * thrust_magnitude
                torque = (candidate.inverse_mass_2(pose[3:12]) @ g_matrix[3:6, 0]) * thrust_magnitude
                return jnp.sum(jnp.square(lateral)) + jnp.sum(jnp.square(torque))

            return jnp.mean(jax.vmap(leak)(poses)) / (actuation_reference**2)

        pl_weight = float(training.get("pl_weight") or 0.0)
        sde_enabled_package = True        # this package always scores with the moment-matched likelihood
        # training.objective = transition: the exact one-step transition NLL under the model's own Lie-IMEX-SDE step
        # (se3_transition_nll) is the loss; the moment-matched multi-step NLL enters only with training.moment_weight.
        objective_mode = str(training.get("objective") or "moment")
        if objective_mode not in ("moment", "transition"):
            raise ValueError("training.objective must be moment or transition")
        moment_weight = float(training.get("moment_weight", 1.0 if objective_mode == "moment" else 0.0))
        step_size_array = jnp.asarray(config["_step_size"])

        # training.validate_transition (default: on in transition mode): score checkpoints by the held-out (clean
        # test split) transition NLL with a tiny validation sigma_obs, whatever the training objective and its sigma_obs.
        validate_flag = bool(training.get("validate_transition", objective_mode == "transition"))
        validation_log_sigma = math.log(float(training.get("validation_observation_sigma") or 1e-5))

        def transition_loss(candidate, states):
            log_sigma = _observation_log_sigma(config, candidate)
            return se3_transition_nll(candidate, states, step_size_array, log_sigma[2], log_sigma[3], lie_imex_sde_step)

        def objective(candidate, batch, random_key, step_value):
            del step_value
            if objective_mode == "transition":
                transition = transition_loss(candidate, batch)
                total = transition
                metrics = jnp.zeros((len(LOSS_NAMES),), dtype=total.dtype)
                nll = jnp.zeros((len(LIKELIHOOD_NAMES),), dtype=total.dtype)
                if moment_weight > 0.0:
                    metrics, nll, _ = _sde_trajectory(config, candidate, batch, random_key)
                    total = total + moment_weight * scale * nll[0]
                return total, (metrics, nll, transition, jnp.zeros((2,), dtype=total.dtype))
            metrics, nll, _ = _sde_trajectory(config, candidate, batch, random_key)
            total = scale * nll[0]
            pl = jnp.asarray(0.0, dtype=total.dtype)
            if pl_weight > 0.0:
                log_sigma = _observation_log_sigma(config, candidate)
                pl = se3_pseudo_likelihood(candidate, batch, config["_step_size"], log_sigma[2], log_sigma[3])
                total = total + pl_weight * pl
            penalty = jnp.asarray(0.0, dtype=total.dtype)
            actuation = jnp.asarray(0.0, dtype=total.dtype)
            if gravity_weight > 0.0:
                penalty = gravity_penalty(candidate, batch)
                total = total + gravity_weight * penalty
            if actuation_weight > 0.0:
                actuation = actuation_penalty(candidate, batch)
                total = total + actuation_weight * actuation
            return total, (metrics, nll, pl, jnp.stack([penalty, actuation]))

        @jax.jit
        def training_step(candidate: Any, state: Any, batch: jax.Array, random_key: jax.Array, step_value: jax.Array):
            (value, auxiliaries), gradients = jax.value_and_grad(
                objective, has_aux=True
            )(candidate, batch, random_key, step_value)
            updates, state = optimizer.update(gradients, state, candidate)
            return optax.apply_updates(candidate, updates), state, value, auxiliaries, optax.global_norm(gradients)

        @jax.jit
        def evaluate(candidate, states):
            metrics, nll, prediction = _sde_trajectory(config, candidate, states, None)
            return metrics, nll, jnp.asarray(0.0), prediction

        @jax.jit
        def validate_transition(candidate, states):
            """Held-out transition NLL (the checkpoint criterion in transition mode) and the learned twist diffusion."""
            log_sigma = (_observation_log_sigma(config, candidate) if objective_mode == "transition"
                         else jnp.full((4,), validation_log_sigma))
            nll = se3_transition_nll(candidate, states, step_size_array, log_sigma[2], log_sigma[3], lie_imex_sde_step)
            return nll, transition_diffusion(candidate, states[:, :512], step_size_array, lie_imex_sde_step)

        best_validation = math.inf

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
                "sde": config.get("sde"),
                "parameter_count": sum(int(value.size) for value in jax.tree_util.tree_leaves(model)),
                "mass_pretraining": mass_audit,
            }
        )
        save_json(run_dir / "metadata.json", metadata)
        save_checkpoint(
            run_dir / f"checkpoint_step_{start_step:05d}.pkl",
            params=model,
            extra={
                "step": start_step,
                "banks": banks,
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
                tick = time.perf_counter()
                model, optimizer_state, objective_value, auxiliaries, gradient_norm = training_step(
                    model, optimizer_state, batch, sample_key, jnp.asarray(step, dtype=jnp.float32)
                )
                jax.block_until_ready(objective_value)
                if not np.isfinite(float(jax.device_get(objective_value))) or not np.isfinite(
                    float(jax.device_get(gradient_norm))
                ):
                    raise FloatingPointError(
                        f"Non-finite optimizer value at step {step}: "
                        f"objective={objective_value}, gradient_norm={gradient_norm}"
                    )
                step_seconds = time.perf_counter() - tick
                train_values, _nll, kl_value, penalty_values = auxiliaries
                penalty_host = np.asarray(jax.device_get(penalty_values), dtype=np.float64)
                stats.setdefault("train_step", []).append(float(step))
                stats.setdefault("train_gravity_penalty", []).append(float(penalty_host[0]))
                stats.setdefault("train_actuation_penalty", []).append(float(penalty_host[1]))
                stats.setdefault("train_transition_nll" if objective_mode == "transition" else "train_pseudo_likelihood",
                                 []).append(float(jax.device_get(kl_value)))

            should_evaluate = step == start_step or step % int(training["eval_every"]) == 0 or step == total_steps
            if should_evaluate:
                test_values, _test_nll, test_kl, _prediction = evaluate(model, test_states)
                jax.block_until_ready(test_values)
                if train_values is None:
                    train_values, _train_nll, kl_value, _ = evaluate(model, train_states[:, :batch_size])
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
                for index, name in enumerate(LIKELIHOOD_NAMES):
                    stats.setdefault(f"test_{name}", []).append(float(np.asarray(jax.device_get(_test_nll))[index]))
                distances = _prodigy_distances(optimizer_state)
                for group, value in distances.items():
                    stats.setdefault(f"prodigy_d_{group}", []).append(value)
                validation_text = ""
                if validate_flag:
                    validation, diffusion = validate_transition(model, test_states)
                    validation = float(jax.device_get(validation))
                    diffusion = np.asarray(jax.device_get(diffusion), dtype=np.float64)
                    stats.setdefault("test_transition_nll", []).append(validation)
                    stats.setdefault("learned_diffusion", []).append(diffusion.tolist())
                    validation_text = (f" tnll={validation:.5f} diff=[" + " ".join(f"{value:.4f}" for value in diffusion) + "]")
                    if np.isfinite(validation) and validation < best_validation:
                        best_validation = validation
                        save_checkpoint(run_dir / "checkpoint_best.pkl", params=model,
                                        extra={"step": step, "banks": banks, "optimizer_state": optimizer_state,
                                               "test_transition_nll": validation})
                        metadata["best_step"], metadata["best_test_transition_nll"] = step, validation
                        validation_text += " *best*"
                np.savez_compressed(run_dir / "training_stats.npz", **{key_name: np.asarray(value) for key_name, value in stats.items()})
                _log(
                    run_dir,
                    f"step={step} train={train_host['total']:.6e} test={final_test['total']:.6e} "
                    f"position={final_test['position']:.3e} attitude={final_test['attitude']:.3e}"
                    + (f" gravity_penalty={stats['train_gravity_penalty'][-1]:.3e} actuation_penalty={stats['train_actuation_penalty'][-1]:.3e}"
                       if (gravity_weight > 0.0 or actuation_weight > 0.0) and stats.get("train_gravity_penalty") else "")
                    + " sde[" + " ".join(
                        f"{short}={float(np.asarray(jax.device_get(_test_nll))[LIKELIHOOD_NAMES.index(name)]):.3g}"
                        for short, name in (("nll", "nll_total"), ("sig_v", "process_sigma_linear"),
                                            ("sig_w", "process_sigma_angular"), ("obs_p", "sigma_position"),
                                            ("obs_v", "sigma_linear_velocity"),
                                            ("cov_p", "coverage_position_2sigma"), ("cov_R", "coverage_attitude_2sigma"),
                                            ("cov_v", "coverage_linear_velocity_2sigma"),
                                            ("cov_w", "coverage_angular_velocity_2sigma"))) + "]"
                    + validation_text
                    + ("" if not distances else " prodigy_d=" + ",".join(f"{k}:{v:.3g}" for k, v in distances.items())),
                )

            if step > start_step and (
                step % int(training["checkpoint_every"]) == 0 or step == total_steps
            ):
                save_checkpoint(
                    run_dir / f"checkpoint_step_{step:05d}.pkl",
                    params=model,
                    extra={
                        "step": step,
                        "banks": banks,
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
        final_metrics, _final_nll, _final_kl, prediction = evaluate(model, report_states)
        jax.block_until_ready(prediction)
        if not _tree_is_finite(model) or not bool(
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
            params=model,
            extra={
                "step": total_steps,
                "banks": banks,
                "optimizer_state": optimizer_state,
            },
        )
        metadata.update(
            {
                "status": "completed",
                "completed_utc": utc_now(),
                "elapsed_seconds": time.time() - started,
                "final_test": final_test,
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

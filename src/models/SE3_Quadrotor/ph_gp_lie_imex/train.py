"""Independent pure-JAX trainer for PH-GP-LieIMEX."""

from __future__ import annotations

import argparse
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
from .data import load_dataset, load_test_flights
from .experiment import append_history, create_experiment, save_json, utc_now
from .integrator import exp_so3, rollout, rollout_control_sequence
from .losses import (
    LIKELIHOOD_NAMES,
    initialize_likelihood_parameters,
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
FROZEN_LABEL = "frozen"


def _frozen_level_groups(config: dict[str, Any]) -> tuple[str, ...]:
    """Level groups that receive no gradient and must not get a Prodigy instance (0/0 distance = NaN)."""
    frozen: list[str] = []
    if bool(config.get("gp", {}).get("structured_potential", False)):
        frozen.append("V_level")  # V(x) = g z / mu(x): the linear V level is unused
    return tuple(frozen)


def _level_labels(params: Any, frozen: tuple[str, ...] = ()) -> Any:
    """Label the level scalars of every subnetwork with their own group; everything else with Adam."""
    def label(path, _leaf):
        keys = [getattr(part, "key", None) for part in path]
        if len(keys) >= 2 and keys[-1] in LEVEL_KEYS:
            group = f"{keys[0]}_level"
            return FROZEN_LABEL if group in frozen else group
        return ADAM_LABEL

    return jax.tree_util.tree_map_with_path(label, params)


def _level_groups(params: Any, frozen: tuple[str, ...] = ()) -> tuple[str, ...]:
    labels = jax.tree_util.tree_leaves(_level_labels(params, frozen))
    return tuple(sorted({name for name in labels if name not in (ADAM_LABEL, FROZEN_LABEL)}))


def _optimizer(config: dict[str, Any], params: Any) -> optax.GradientTransformation:
    settings = config["optimizer"]
    frozen = _frozen_level_groups(config)
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
            for name in _level_groups(params, frozen)
        }
        groups = {ADAM_LABEL: adam, **prodigy}
        if frozen:
            groups[FROZEN_LABEL] = optax.set_to_zero()
        transforms.append(optax.multi_transform(groups, lambda tree: _level_labels(tree, frozen)))
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
        potential_periodic_harmonics=gp.get("potential_periodic_harmonics"),
        # Hard level prior V_z = g exp(-lambda0) z: on when known_gravity is set unless
        # gp.gravity_level_prior is false (e.g. when only the soft gravity penalty is wanted).
        known_gravity=(
            None if gp.get("known_gravity") is None or not bool(gp.get("gravity_level_prior", True))
            else float(gp["known_gravity"])
        ),
        structured_potential=bool(gp.get("structured_potential", False)),
        dissipation_pose_input=bool(gp.get("dissipation_pose_input", False)),
    )
    if bool(gp.get("structured_potential", False)):
        if gp.get("known_gravity") is None:
            raise ValueError("gp.structured_potential needs gp.known_gravity (g in m/s^2)")
        # The structured potential uses known_gravity directly; the level prior is meaningless here.
        gp_setup["_physics"]["known_gravity"] = jnp.asarray(float(gp["known_gravity"]), dtype=jnp.float32)
        gp_setup["_physics"]["known_gravity_enabled"] = jnp.asarray(0.0, dtype=jnp.float32)
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
        return variational, 0, None, None
    checkpoint_path = resolve_project_path(selected)
    assert checkpoint_path is not None
    payload = load_checkpoint(checkpoint_path)
    is_resume = training["resume_checkpoint"] is not None
    start_step = int(payload.get("extra", {}).get("step", 0)) if is_resume else 0
    stored_optimizer = payload.get("extra", {}).get("optimizer_state") if is_resume else None
    # Per-window latent initial states only make sense for the same window set, i.e. on resume.
    stored_latent = payload.get("extra", {}).get("latent_initial_state") if is_resume else None
    return payload["params"], start_step, stored_optimizer, stored_latent




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


def _record_likelihood(stats: dict[str, list[float]], prefix: str, values: jax.Array) -> None:
    """Unpack the 13-entry likelihood vector into one series per term.

    Records, under ``<prefix>_<name>`` for every name in ``LIKELIHOOD_NAMES``: the total NLL,
    the four per-block NLL terms, the four squared-error norms they are built from, and the
    four learned observation sigma.  One device transfer per call.  When the data fit is not
    ``se3-nll`` the vector is zeros and the series are recorded as zeros, so the schema of
    ``training_stats.npz`` does not depend on the configuration.
    """
    host = np.asarray(jax.device_get(values), dtype=np.float64)
    for name, value in zip(LIKELIHOOD_NAMES, host):
        stats.setdefault(f"{prefix}_{name}", []).append(float(value))


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
        # Follow JAX's default float width: the data is stored as float32, but with
        # runtime.jax_enable_x64 the model's own arrays are float64, and jax.jvp requires the
        # primal and tangent dtypes to match exactly.
        float_dtype = jnp.zeros(()).dtype
        train_states = jax.device_put(jnp.asarray(train_numpy, dtype=float_dtype), device)
        test_states = jax.device_put(jnp.asarray(test_numpy, dtype=float_dtype), device)
        seed = int(config["training"]["seed"])
        key = jax.random.PRNGKey(seed)
        key, initialization_key = jax.random.split(key)
        variational, gp_setup, mass_audit = _initialize_gp(
            config, initialization_key, train_states, dataset_settings
        )
        variational, start_step, stored_optimizer_state, stored_latent = _load_requested_checkpoint(config, variational)
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

        # ---- Optional long-horizon validation: open-loop position RMS on the clean test flights ----
        # training.long_horizon_test_seconds: scalar or list of horizons (s); absent/null = off.
        # Logged as test_long{tag}_position_rms / _position_relative / _masked_fraction (tag e.g. 1s, 3s).
        long_test_setting = training.get("long_horizon_test_seconds")
        long_test_seconds: list[float] = []
        if long_test_setting is not None:
            long_test_seconds = [
                float(value) for value in (
                    long_test_setting if isinstance(long_test_setting, (list, tuple)) else [long_test_setting]
                )
            ]
        evaluate_long_functions = []
        if long_test_seconds:
            test_flights = jax.device_put(jnp.asarray(load_test_flights(dataset_path), dtype=float_dtype), device)
            flight_length = int(test_flights.shape[0])

            def _make_evaluate_long(test_horizon: int):
                @jax.jit
                def evaluate_long(candidate):
                    model = network.DissipativeSE3HamODE(candidate, gp_setup).sample()
                    flights = test_flights[: test_horizon + 1]
                    h = flights[0, 0, -1] * 0.0 + step_size
                    prediction = rollout_control_sequence(model, flights[0], flights[1:, :, 18:22], h)
                    finite = jnp.all(jnp.isfinite(prediction[1:, :, :18]), axis=-1)  # (time, flights)
                    error = jnp.sum((prediction[1:, :, :3] - flights[1:, :, :3]) ** 2, axis=-1)
                    excursion = jnp.sum((flights[1:, :, :3] - flights[:1, :, :3]) ** 2, axis=-1)
                    safe_error = jnp.where(finite, error, 0.0)
                    count = jnp.maximum(1.0, jnp.sum(finite))
                    rms = jnp.sqrt(jnp.sum(safe_error) / count)
                    relative = rms / jnp.sqrt(jnp.maximum(jnp.mean(excursion), 1e-30))
                    return rms, relative, 1.0 - jnp.mean(finite.astype(jnp.zeros(()).dtype))
                return evaluate_long

            evaluate_long_functions = [
                _make_evaluate_long(min(int(round(seconds / step_size)), flight_length - 1))
                for seconds in long_test_seconds
            ]

        def _horizon_tag(seconds: float) -> str:
            return f"{seconds:g}s".replace(".", "p")

        # ---- Optional known-gravity penalty on the identifiable product mu(x) grad V(x) ----
        # L_g = w / g^2 * mean_x || mu(x) grad V(x) - g e_3 ||^2 over (a subsample of) the batch states,
        # with the full GP-including mu and V and the same sampled weights as the data fit.  Gauge-invariant;
        # the true model satisfies it exactly.  gp.gravity_penalty_weight = 0 / absent turns it off.
        gravity_weight = float(config["gp"].get("gravity_penalty_weight") or 0.0)
        gravity_value = float(config["gp"].get("known_gravity") or 0.0)
        gravity_points = int(config["gp"].get("gravity_penalty_points") or 1024)
        if gravity_weight > 0.0 and gravity_value <= 0.0:
            raise ValueError("gp.gravity_penalty_weight needs gp.known_gravity > 0")

        def gravity_penalty(model, batch):
            poses = batch[:, :, :12].reshape(-1, 12)
            stride = max(1, poses.shape[0] // gravity_points)
            poses = poses[::stride]
            mu = jax.vmap(lambda pose: model.inverse_mass_1(pose[:3])[0, 0])(poses)
            grad_v = jax.vmap(lambda pose: jax.grad(model.potential)(pose)[:3])(poses)
            target = jnp.array([0.0, 0.0, gravity_value], dtype=poses.dtype)
            residual = mu[:, None] * grad_v - target
            return jnp.mean(jnp.sum(jnp.square(residual), axis=-1)) / (gravity_value**2)

        # ---- Optional actuation-direction prior on the learned input map g(x) ----
        # gp.actuation_penalty_weight (default 0 = off): the thrust command (input channel 0) acts along body z only,
        # i.e. it produces no lateral body force and no torque. Penalty = E_x[ |mu g_xy(x) e1 u_h|^2 + |M2^-1 g_tau(x) e1 u_h|^2 ] / g^2
        # with u_h the mean thrust magnitude of the batch, so both terms are accelerations per hover thrust in units of g,
        # like the gravity penalty. Only the direction is imposed; the thrust gain mu g_fz and all torque gains stay learned.
        actuation_weight = float(config["gp"].get("actuation_penalty_weight") or 0.0)
        actuation_reference = float(gravity_value if gravity_value > 0.0 else 9.81)

        def actuation_penalty(model, batch):
            poses = batch[:, :, :12].reshape(-1, 12)
            stride = max(1, poses.shape[0] // gravity_points)
            poses = poses[::stride]
            thrust_magnitude = jnp.mean(jnp.abs(batch[:, :, 18]))

            def leak(pose):
                g_matrix = model.control_matrix(pose)
                mu = model.inverse_mass_1(pose[:3])[0, 0]
                lateral = mu * g_matrix[:2, 0] * thrust_magnitude
                torque = (model.inverse_mass_2(pose[3:12]) @ g_matrix[3:6, 0]) * thrust_magnitude
                return jnp.sum(jnp.square(lateral)) + jnp.sum(jnp.square(torque))

            return jnp.mean(jax.vmap(leak)(poses)) / (actuation_reference**2)

        # ---- Optional latent initial state per training window (errors-in-variables treatment) ----
        # training.latent_initial_state: true enables a variational q(x0_i) = N(x0_obs + m_i, s_i^2) over the 12
        # tangent coordinates (position, SO(3) tangent applied as R exp(hat(phi)), v_b, omega_b) of every window's
        # first state, with prior N(x0_obs, sigma^2); the ELBO gains -KL(q || p) per window (per transition).
        # Off (default) keeps the observed first state as the exact initial condition.
        latent_enabled = bool(training.get("latent_initial_state", False))
        # training.latent_initial_state_sigma: a number = fixed prior width (assumes the observation noise is known);
        # "learned" = the prior on the initial-state correction IS the observation model x0_obs = x0 + eps with the
        # learned per-block likelihood sigma (position, attitude, linear velocity, angular velocity), including the
        # log-sigma term, i.e. the missing -log p(x0_obs | x0) of the window likelihood. No noise level is assumed.
        latent_sigma_setting = training.get("latent_initial_state_sigma")
        latent_sigma_learned = isinstance(latent_sigma_setting, str) and latent_sigma_setting.lower() == "learned"
        latent_sigma = float(config["gp"]["initial_observation_sigma"] if latent_sigma_learned or not latent_sigma_setting
                             else latent_sigma_setting)
        latent_learning_rate = float(training.get("latent_initial_state_learning_rate") or 1.0e-2)
        latent_kl_weight = float(training.get("latent_initial_state_kl_weight", 1.0))
        # Floor on the posterior width s_i (0 = none): keeps q(x0_i) from collapsing onto a point the model can
        # co-adapt to (damping absorbing the initial-state noise); s_i = max(exp(log_std), floor).
        latent_std_floor = float(training.get("latent_initial_state_std_floor") or 0.0)
        if latent_std_floor >= latent_sigma:
            raise ValueError("latent_initial_state_std_floor must be below latent_initial_state_sigma")
        # training.latent_initial_state_sample (default true): true = reparameterised ELBO, the rollout starts from a
        # sample x0_obs + m_i + s_i*eps and the full Gaussian KL enters; false = MAP initial state, the rollout starts
        # from the mean x0_obs + m_i only (no injected perturbation) and the KL reduces to the Gaussian prior penalty
        # sum(m_i^2) / (2 sigma^2) on the correction (log_std is then unused).
        latent_sample = bool(training.get("latent_initial_state_sample", True))
        # ---- Optional multiple-shooting continuity between overlapping windows ----
        # training.continuity_weight (default 0 = off): the batch is drawn as adjacent window pairs (i, i+1), whose
        # starts are `window_stride` samples apart, and the corrected initial state of window i+1 must agree with
        # window i's own rollout `window_stride` steps in: lambda * mean_pairs sum_b |x0^(i+1) (-) Phi_s(x0^(i))|_b^2 / (2 sigma_b^2)
        # (SE(3) tangent difference, learned per-block sigma). Gradients only ever pass through one short rollout, but
        # a dynamics bias now accumulates along the chain of windows, so it becomes visible to the loss.
        continuity_weight = float(training.get("continuity_weight") or 0.0)
        continuity_stride = int(config["data"]["window_stride"])
        pair_starts = np.zeros((0,), dtype=np.int64)
        if continuity_weight > 0.0:
            if not latent_enabled:
                raise ValueError("training.continuity_weight requires training.latent_initial_state=true")
            if continuity_stride > int(train_states.shape[0]) - 1:
                raise ValueError("window_stride must not exceed the window length for the continuity penalty")
            host_states = np.asarray(train_states)
            adjacent = np.all(host_states[continuity_stride, :-1, :18] == host_states[0, 1:, :18], axis=-1)
            pair_starts = np.nonzero(adjacent)[0]
            if pair_starts.size < int(training["batch_size"]) // 2:
                raise ValueError("not enough adjacent window pairs for the continuity penalty")
        window_count = int(train_states.shape[1])
        window_transitions = float(train_states.shape[0] - 1)
        latent_shape = (window_count if latent_enabled else 1, 12)
        latent = {
            "mean": jnp.zeros(latent_shape, dtype=train_states.dtype),
            "log_std": jnp.full(latent_shape, jnp.log(latent_sigma), dtype=train_states.dtype),
        }
        if latent_enabled and stored_latent is not None:
            latent = jax.tree_util.tree_map(jnp.asarray, stored_latent)
        latent_optimizer = optax.adam(latent_learning_rate)
        latent_state = latent_optimizer.init(latent)

        def apply_initial_correction(batch, delta):
            x0 = batch[0]
            position = x0[:, :3] + delta[:, :3]
            rotation = x0[:, 3:12].reshape(-1, 3, 3) @ exp_so3(delta[:, 3:6])
            velocity = x0[:, 12:15] + delta[:, 6:9]
            omega = x0[:, 15:18] + delta[:, 9:12]
            corrected = jnp.concatenate([position, rotation.reshape(-1, 9), velocity, omega, x0[:, 18:]], axis=1)
            return batch.at[0].set(corrected)

        def latent_prior_sigma(candidate):
            """(12,) prior width per tangent coordinate: constant, or the learned per-block likelihood sigma."""
            if not latent_sigma_learned:
                return jnp.full((12,), latent_sigma, dtype=train_states.dtype)
            likelihood = candidate["likelihood"]
            if config["gp"]["fixed_observation_sigma"] is not None:
                likelihood = jax.tree_util.tree_map(jax.lax.stop_gradient, likelihood)
            blocks = ("log_sigma_position", "log_sigma_attitude", "log_sigma_linear_velocity", "log_sigma_angular_velocity")
            return jnp.repeat(jnp.stack([jnp.exp(likelihood[name]) for name in blocks]), 3)

        def latent_terms(latent_params, indices, random_key, prior_sigma):
            mean = latent_params["mean"][indices]
            std = jnp.maximum(jnp.exp(latent_params["log_std"][indices]), latent_std_floor)
            if not latent_sample:
                # -log N(x0_obs; x0_obs + m_i, sigma^2) up to constants: quadratic term plus log sigma (so a learned
                # sigma is estimated from the corrections as well as from the rollout residuals).
                prior_penalty = jnp.sum(mean**2 / (2.0 * prior_sigma**2) + jnp.log(prior_sigma), axis=1)
                return mean, jnp.mean(prior_penalty)
            noise = jax.random.normal(jax.random.fold_in(random_key, 7), mean.shape, dtype=mean.dtype)
            delta = mean + std * noise
            kl_per_window = jnp.sum(
                jnp.log(prior_sigma / std) + (std**2 + mean**2) / (2.0 * prior_sigma**2) - 0.5, axis=1
            )
            return delta, jnp.mean(kl_per_window)

        def continuity_term(candidate, batch, prediction):
            """Mean over pairs of the sigma-normalised SE(3) defect between window i+1's start and window i's rollout."""
            half = batch.shape[1] // 2
            reached = prediction[continuity_stride, :half]
            start = batch[0, half:]
            sigma = latent_prior_sigma(candidate)               # (12,): position, attitude, v, omega blocks
            position = jnp.sum(jnp.square(start[:, :3] - reached[:, :3]), axis=1) / (2.0 * sigma[0] ** 2)
            relative = jnp.einsum("nji,njk->nik", reached[:, 3:12].reshape(-1, 3, 3), start[:, 3:12].reshape(-1, 3, 3))
            cosine = jnp.clip((jnp.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
            attitude = jnp.square(jnp.arccos(cosine)) / (2.0 * sigma[3] ** 2)
            velocity = jnp.sum(jnp.square(start[:, 12:15] - reached[:, 12:15]), axis=1) / (2.0 * sigma[6] ** 2)
            omega = jnp.sum(jnp.square(start[:, 15:18] - reached[:, 15:18]), axis=1) / (2.0 * sigma[9] ** 2)
            return jnp.mean(position + attitude + velocity + omega)

        def objective(candidate, latent_params, batch, indices, random_key, step_value):
            latent_kl = jnp.asarray(0.0, dtype=batch.dtype)
            if latent_enabled:
                delta, latent_kl = latent_terms(latent_params, indices, random_key, latent_prior_sigma(candidate))
                batch = apply_initial_correction(batch, delta)
            metrics, nll, kl, prediction = _gp_components(
                config, candidate, gp_setup, batch, random_key
            )
            data_fit = nll[0] if config["gp"]["data_fit"] == "se3-nll" else metrics[0]
            progress = jnp.minimum(
                1.0, step_value / max(1, int(config["gp"]["kl_anneal_steps"]))
            )
            beta = float(config["gp"]["kl_beta_max"]) * progress
            total = scale * data_fit + beta * kl / number_of_observations
            if latent_enabled:
                # The data fit is a per-transition mean, so the per-window KL is spread over the window's transitions.
                total = total + latent_kl_weight * latent_kl / window_transitions
            continuity = jnp.asarray(0.0, dtype=total.dtype)
            if continuity_weight > 0.0:
                continuity = continuity_term(candidate, batch, prediction)
                total = total + continuity_weight * continuity
            penalty = jnp.asarray(0.0, dtype=total.dtype)
            actuation = jnp.asarray(0.0, dtype=total.dtype)
            if gravity_weight > 0.0 or actuation_weight > 0.0:
                # Same key as inside _gp_components, so the penalties see the same weight sample.
                sampled = network.DissipativeSE3HamODE(candidate, gp_setup).sample(random_key)
                if gravity_weight > 0.0:
                    penalty = gravity_penalty(sampled, batch)
                    total = total + gravity_weight * penalty
                if actuation_weight > 0.0:
                    actuation = actuation_penalty(sampled, batch)
                    total = total + actuation_weight * actuation
            return total, (metrics, nll, kl, penalty, latent_kl, actuation, continuity)

        @jax.jit
        def training_step(candidate: Any, state: Any, latent_params: Any, latent_opt_state: Any, batch: jax.Array,
                          indices: jax.Array, random_key: jax.Array, step_value: jax.Array):
            (value, auxiliaries), (gradients, latent_gradients) = jax.value_and_grad(
                objective, argnums=(0, 1), has_aux=True
            )(candidate, latent_params, batch, indices, random_key, step_value)
            updates, state = optimizer.update(gradients, state, candidate)
            if latent_enabled:
                latent_updates, latent_opt_state = latent_optimizer.update(latent_gradients, latent_opt_state, latent_params)
                latent_params = optax.apply_updates(latent_params, latent_updates)
            return (optax.apply_updates(candidate, updates), state, latent_params, latent_opt_state,
                    value, auxiliaries, optax.global_norm(gradients))

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
                "latent_initial_state": {
                    "enabled": bool(config["training"].get("latent_initial_state", False)),
                    "prior_sigma": (
                        "learned per-block likelihood sigma (observation model applied to x0, with log-sigma term)"
                        if str(config["training"].get("latent_initial_state_sigma")).lower() == "learned"
                        else float(config["training"].get("latent_initial_state_sigma") or config["gp"]["initial_observation_sigma"])
                    ),
                    "learning_rate": float(config["training"].get("latent_initial_state_learning_rate") or 1.0e-2),
                    "kl_weight": float(config["training"].get("latent_initial_state_kl_weight", 1.0)),
                    "std_floor": float(config["training"].get("latent_initial_state_std_floor") or 0.0),
                    "sample": bool(config["training"].get("latent_initial_state_sample", True)),
                    "description": (
                        "variational q(x0_i)=N(x0_obs+m_i, s_i^2) per training window on the 12 SE(3) tangent coordinates, KL to N(x0_obs, sigma^2) in the ELBO"
                        if bool(config["training"].get("latent_initial_state_sample", True))
                        else "MAP initial state x0_obs+m_i per training window on the 12 SE(3) tangent coordinates (no sampling), Gaussian prior penalty |m_i|^2/(2 sigma^2)"
                    ),
                },
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
                    "V": (
                        f"structured g z / mu(x), mu(x) = exp(lambda0 + GP_M1(x)) learned, g={float(config['gp']['known_gravity']):g} (V GP unused)"
                        if bool(config["gp"].get("structured_potential", False)) else
                        f"g exp(-lambda0) z + lambda_V,xy^T x_xy + GP(x) (known-gravity level prior g={float(config['gp']['known_gravity']):g})"
                        if config["gp"].get("known_gravity") is not None and bool(config["gp"].get("gravity_level_prior", True))
                        else "lambda_V^T x + GP(x) (learned linear level, zero init)"
                    ) + (
                        f"; soft penalty w={float(config['gp']['gravity_penalty_weight']):g} on ||mu grad V - g e3||^2/g^2"
                        if float(config["gp"].get("gravity_penalty_weight") or 0.0) > 0.0 else ""
                    ),
                    "dissipation_inputs": (
                        "Dv(x, v_b) = exp(s_v(x)) * SPD(level + f_Dv(v_b)); Dw(R, omega_b) = SPD(level + f_Dw(R, omega_b))"
                        if bool(config["gp"].get("dissipation_pose_input", False)) else "Dv(v_b), Dw(omega_b)"
                    ),
                    "continuity": (
                        f"multiple-shooting continuity between adjacent overlapping windows (stride {int(config['data']['window_stride'])}): "
                        f"weight {float(config['training'].get('continuity_weight') or 0.0)} on the sigma-normalised SE(3) defect "
                        "between window i+1's corrected start and window i's rollout; batches are adjacent pairs"
                        if float(config["training"].get("continuity_weight") or 0.0) > 0.0 else "off"
                    ),
                    "actuation_prior": (
                        f"thrust command acts along body z only: penalty E_x[|mu g_xy e1 u_h|^2 + |M2^-1 g_tau e1 u_h|^2]/g^2, weight {float(config['gp'].get('actuation_penalty_weight') or 0.0)}"
                        if float(config["gp"].get("actuation_penalty_weight") or 0.0) > 0.0 else "off"
                    ),
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
                "latent_initial_state": latent if latent_enabled else None,
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
                if continuity_weight > 0.0:
                    first = rng.choice(pair_starts, size=batch_size // 2, replace=False)
                    indices = np.concatenate([first, first + 1])
                else:
                    indices = rng.choice(int(train_states.shape[1]), size=batch_size, replace=False)
                batch = train_states[:, indices]
                key, sample_key = jax.random.split(key)
                tick = time.perf_counter()
                variational, optimizer_state, latent, latent_state, objective_value, auxiliaries, gradient_norm = training_step(
                    variational, optimizer_state, latent, latent_state, batch, jnp.asarray(indices, dtype=jnp.int32),
                    sample_key, jnp.asarray(step, dtype=jnp.zeros(()).dtype),
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
                train_values, _nll, kl_value, gravity_value_step, latent_kl_step, actuation_value_step, continuity_step = auxiliaries
                step_host = _host_metrics(train_values)
                stats["train_step"].append(float(step))
                stats["train_loss"].append(step_host["total"])
                stats.setdefault("train_gravity_penalty", []).append(float(jax.device_get(gravity_value_step)))
                stats.setdefault("train_actuation_penalty", []).append(float(jax.device_get(actuation_value_step)))
                stats.setdefault("train_continuity", []).append(float(jax.device_get(continuity_step)))
                stats.setdefault("train_latent_kl", []).append(float(jax.device_get(latent_kl_step)))
                # GP objective terms: bare NLL, its per-block parts, the learned sigma, and the
                # weight-space KL with the annealing coefficient that scales it into the total.
                _record_likelihood(stats, "train", _nll)
                weight_kl_step = float(jax.device_get(kl_value))
                beta_step = float(config["gp"]["kl_beta_max"]) * min(
                    1.0, step / max(1, int(config["gp"]["kl_anneal_steps"]))
                )
                stats.setdefault("train_weight_kl", []).append(weight_kl_step)
                stats.setdefault("train_kl_beta", []).append(beta_step)
                stats.setdefault("train_kl_contribution", []).append(
                    beta_step * weight_kl_step / float(number_of_observations)
                )
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
                _record_likelihood(stats, "test", _test_nll)
                long_log = ""
                for seconds, evaluate_long in zip(long_test_seconds, evaluate_long_functions):
                    rms_long, relative_long, masked_long = (float(jax.device_get(v)) for v in evaluate_long(variational))
                    tag = _horizon_tag(seconds)
                    stats.setdefault(f"test_long{tag}_position_rms", []).append(rms_long)
                    stats.setdefault(f"test_long{tag}_position_relative", []).append(relative_long)
                    stats.setdefault(f"test_long{tag}_masked_fraction", []).append(masked_long)
                    long_log += f" test{tag}[pos_rms={rms_long:.3e} rel={relative_long:.2f} masked={masked_long:.2f}]"
                distances = _prodigy_distances(optimizer_state)
                for group, value in distances.items():
                    stats.setdefault(f"prodigy_d_{group}", []).append(value)
                np.savez_compressed(run_dir / "training_stats.npz", **{key_name: np.asarray(value) for key_name, value in stats.items()})
                _log(
                    run_dir,
                    f"step={step} train={train_host['total']:.6e} test={final_test['total']:.6e} "
                    f"position={final_test['position']:.3e} attitude={final_test['attitude']:.3e}"
                    + long_log
                    + (f" gravity_penalty={stats['train_gravity_penalty'][-1]:.3e}" if gravity_weight > 0.0 and stats.get("train_gravity_penalty") else "")
                    + (f" actuation_penalty={stats['train_actuation_penalty'][-1]:.3e}" if actuation_weight > 0.0 and stats.get("train_actuation_penalty") else "")
                    + (f" continuity={stats['train_continuity'][-1]:.3e}" if continuity_weight > 0.0 and stats.get("train_continuity") else "")
                    + (
                        f" latent_x0[kl={stats['train_latent_kl'][-1]:.3e} rms_mean={float(jnp.sqrt(jnp.mean(latent['mean'] ** 2))):.3e}"
                        f" v_rms={float(jnp.sqrt(jnp.mean(latent['mean'][:, 6:9] ** 2))):.3e} std={float(jnp.mean(jnp.maximum(jnp.exp(latent['log_std']), latent_std_floor))):.3e}]"
                        if latent_enabled and stats.get("train_latent_kl") else ""
                    )
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
                "latent_initial_state": latent if latent_enabled else None,
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
                "latent_initial_state": latent if latent_enabled else None,
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

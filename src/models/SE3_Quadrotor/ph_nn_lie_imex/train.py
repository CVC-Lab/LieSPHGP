"""Independent pure-JAX trainer for one neural quadrotor model."""

from __future__ import annotations

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
from .integrator import rollout, rollout_control_sequence
from .losses import pose_loss_components, pose_loss_components_masked, window_health
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
SUBNETWORKS = ("M1_net", "M2_net", "Dv_net", "Dw_net", "V_net", "g_net")
FROZEN_LABEL = "frozen"


def _subnetwork_labels(model: Any) -> Any:
    """Every array leaf labelled with the subnetwork it belongs to; anything outside the six subnetworks
    (the dissipation_enabled switch) is frozen."""
    def label(path, _leaf):
        head = path[0] if path else None
        name = getattr(head, "name", getattr(head, "key", None))
        return name if name in SUBNETWORKS else FROZEN_LABEL

    return jax.tree_util.tree_map_with_path(label, model)


def _optimizer(config: dict[str, Any]) -> optax.GradientTransformation:
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
        # One independent Prodigy per subnetwork: each estimates its own distance-to-solution d_t, so the six
        # operators get their own effective step size instead of sharing one Adam learning rate.
        groups = {
            name: optax.contrib.prodigy(
                learning_rate=float(settings["learning_rate"]),
                betas=(float(settings["beta1"]), float(settings["beta2"])),
                eps=float(settings["epsilon"]),
                estim_lr_coef=float(settings.get("prodigy_d_coef", 1.0)),   # scales the d_t estimate itself
                safeguard_warmup=True,
            )
            for name in SUBNETWORKS
        }
        groups[FROZEN_LABEL] = optax.set_to_zero()
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
    model = initialize_nn_parameters(
        key,
        hidden_dim=int(model_config["hidden_dim"]),
        initialization_gain=float(model_config["initialization_gain"]),
        mass_epsilon=float(model_config["mass_epsilon"]),
        mass_factor_epsilon=float(model_config["mass_factor_epsilon"]),
        dissipation_enabled=bool(model_config["dissipation_enabled"]),
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


def _trajectory(config: dict[str, Any], model: Any, states: jax.Array, mask_diverged: bool = False):
    step_size = states[0, 0, -1] * 0.0 + config["_step_size"]
    # model.integration_substeps: Lie-IMEX steps per DATA interval; the loss still compares only at the data
    # times, so only the integrator's effective step changes (h -> h / substeps).
    substeps = int(config["model"].get("integration_substeps", 1) or 1)
    if config["data"]["time_varying_controls"]:
        prediction = rollout_control_sequence(
            model, states[0], states[1:, :, 18:22], step_size, substeps
        )
    else:
        prediction = rollout(
            model, states[0], step_size, steps=int(states.shape[0] - 1)
        )
    if mask_diverged:
        healthy = window_health(prediction[1:])
        metrics, masked_fraction = pose_loss_components_masked(states[1:], prediction[1:], healthy)
        return metrics, prediction, masked_fraction
    return pose_loss_components(states[1:], prediction[1:]), prediction, jnp.asarray(0.0)


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
        # Follow JAX's default float width: the data is stored as float32, but with runtime.jax_enable_x64
        # the model's own arrays are float64 and the scan carry dtypes must match exactly.
        # Same treatment as ph_gp_lie_imex_idsia.train.
        float_dtype = jnp.zeros(()).dtype
        train_states = jax.device_put(jnp.asarray(train_numpy, dtype=float_dtype), device)
        test_states = jax.device_put(jnp.asarray(test_numpy, dtype=float_dtype), device)
        seed = int(config["training"]["seed"])
        key = jax.random.PRNGKey(seed)
        key, initialization_key = jax.random.split(key)
        model, banks, mass_audit = _initialize_nn(
            config, initialization_key, train_states, dataset_settings
        )
        model, start_step, stored_optimizer_state = _load_requested_checkpoint(config, model)
        optimizer = _optimizer(config)
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

        def objective(candidate, batch, random_key, step_value):
            del random_key, step_value
            metrics, _, _ = _trajectory(config, candidate, batch)   # objective never masks: training loss is unchanged
            total = scale * metrics[0]
            penalty = jnp.asarray(0.0, dtype=total.dtype)
            actuation = jnp.asarray(0.0, dtype=total.dtype)
            if gravity_weight > 0.0:
                penalty = gravity_penalty(candidate, batch)
                total = total + gravity_weight * penalty
            if actuation_weight > 0.0:
                actuation = actuation_penalty(candidate, batch)
                total = total + actuation_weight * actuation
            return total, (metrics, jnp.zeros((13,)), jnp.asarray(0.0), jnp.stack([penalty, actuation]))

        @jax.jit
        def training_step(candidate: Any, state: Any, batch: jax.Array, random_key: jax.Array, step_value: jax.Array):
            (value, auxiliaries), gradients = jax.value_and_grad(
                objective, has_aux=True
            )(candidate, batch, random_key, step_value)
            updates, state = optimizer.update(gradients, state, candidate)
            return optax.apply_updates(candidate, updates), state, value, auxiliaries, optax.global_norm(gradients)

        # training.mask_diverged_eval_windows (optional, default false): average the EVALUATION metric over
        # windows whose rollout stayed finite instead of over all of them.  A single diverged window makes the
        # plain mean NaN and the guard below aborts at step 0 - which is what stopped this model training on
        # the real IDSIA flights.  The training objective is untouched; only the reported metric is masked,
        # and the excluded fraction is logged as test_masked_fraction.
        mask_eval = bool(training.get("mask_diverged_eval_windows", False))

        @jax.jit
        def evaluate(candidate, states):
            metrics, prediction, masked_fraction = _trajectory(config, candidate, states, mask_eval)
            return metrics, jnp.zeros((13,)), jnp.asarray(0.0), prediction, masked_fraction

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

            should_evaluate = step == start_step or step % int(training["eval_every"]) == 0 or step == total_steps
            if should_evaluate:
                test_values, _test_nll, test_kl, _prediction, test_masked = evaluate(model, test_states)
                jax.block_until_ready(test_values)
                if train_values is None:
                    train_values, _train_nll, kl_value, _, _ = evaluate(model, train_states[:, :batch_size])
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
                stats.setdefault("test_masked_fraction", []).append(float(jax.device_get(test_masked)))
                distances = _prodigy_distances(optimizer_state)
                for group, value in distances.items():
                    stats.setdefault(f"prodigy_d_{group}", []).append(value)
                np.savez_compressed(run_dir / "training_stats.npz", **{key_name: np.asarray(value) for key_name, value in stats.items()})
                _log(
                    run_dir,
                    f"step={step} train={train_host['total']:.6e} test={final_test['total']:.6e} "
                    f"position={final_test['position']:.3e} attitude={final_test['attitude']:.3e}"
                    + (f" gravity_penalty={stats['train_gravity_penalty'][-1]:.3e} actuation_penalty={stats['train_actuation_penalty'][-1]:.3e}"
                       if (gravity_weight > 0.0 or actuation_weight > 0.0) and stats.get("train_gravity_penalty") else "")
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
        final_metrics, _final_nll, _final_kl, prediction, _final_masked = evaluate(model, report_states)
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

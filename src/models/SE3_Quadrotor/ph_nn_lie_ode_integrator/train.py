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
from .losses import pose_loss_components
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


def _optimizer(config: dict[str, Any]) -> optax.GradientTransformation:
    settings = config["optimizer"]
    if settings["name"] != "adam":
        raise ValueError("optimizer.name currently must be adam")
    transforms: list[optax.GradientTransformation] = []
    clip = settings["gradient_clip_norm"]
    if clip is not None:
        transforms.append(optax.clip_by_global_norm(float(clip)))
    if float(settings["weight_decay"]) != 0.0:
        transforms.append(optax.add_decayed_weights(float(settings["weight_decay"])))
    transforms.append(
        optax.adam(
            learning_rate=float(settings["learning_rate"]),
            b1=float(settings["beta1"]),
            b2=float(settings["beta2"]),
            eps=float(settings["epsilon"]),
        )
    )
    return optax.chain(*transforms)


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
        optimizer = _optimizer(config)
        optimizer_state = (
            optimizer.init(model)
            if stored_optimizer_state is None
            else jax.tree_util.tree_map(jnp.asarray, stored_optimizer_state)
        )
        training = config["training"]
        batch_size = min(int(training["batch_size"]), int(train_states.shape[1]))
        scale = float(training["loss_scale"])

        def objective(candidate, batch, random_key, step_value):
            del random_key, step_value
            metrics, _ = _trajectory(config, candidate, batch)
            return scale * metrics[0], (metrics, jnp.zeros((13,)), jnp.asarray(0.0))

        @jax.jit
        def training_step(candidate: Any, state: Any, batch: jax.Array, random_key: jax.Array, step_value: jax.Array):
            (value, auxiliaries), gradients = jax.value_and_grad(
                objective, has_aux=True
            )(candidate, batch, random_key, step_value)
            updates, state = optimizer.update(gradients, state, candidate)
            return optax.apply_updates(candidate, updates), state, value, auxiliaries, optax.global_norm(gradients)

        @jax.jit
        def evaluate(candidate, states):
            metrics, prediction = _trajectory(config, candidate, states)
            return metrics, jnp.zeros((13,)), jnp.asarray(0.0), prediction

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
                train_values, _nll, kl_value = auxiliaries

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
                np.savez_compressed(run_dir / "training_stats.npz", **{key_name: np.asarray(value) for key_name, value in stats.items()})
                _log(
                    run_dir,
                    f"step={step} train={train_host['total']:.6e} test={final_test['total']:.6e} "
                    f"position={final_test['position']:.3e} attitude={final_test['attitude']:.3e}",
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

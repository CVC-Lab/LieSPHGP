"""Train an SO(3) pendulum PH-NODE baseline (RK4 + trajectory loss, the prior work's recipe).

    python src/models/3D_SO3_Windy_Pendulum/ph_node/train.py --config src/models/3D_SO3_Windy_Pendulum/configs/<experiment>/<PH-NODE file>.yaml

Per step, on a batch of noisy windows: roll the deterministic model out with RK4 from each window's first sample and
minimise the trajectory loss (losses.py: rollout for PH-NODE, rollout_reference for PH-NODE-ref). Adam with global-norm
clipping (optimizer.gradient_clip_norm; null = none), cosine decay over training.total_steps or a constant rate
(optimizer.schedule), optional weight decay (optimizer.weight_decay). model.pretrain_inverse_mass_identity: the
reference's M^-1 -> I pretraining before training (PH-NODE-ref only).

Fixed budget: the result is the model after training.total_steps (checkpoint_final.pkl). The loss of the final model on
the NOISY test split is logged as a monitor only. Clean data is never touched during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=None, help="override training.total_steps (smoke tests)")
    parser.add_argument("--root", type=Path, default=None, help="override experiment.root (smoke tests)")
    return parser.parse_args()


ARGS = _parse() if __name__ == "__main__" else None
if ARGS is not None:   # the GPU must be chosen before jax is imported
    _runtime = yaml.safe_load(ARGS.config.read_text())["runtime"]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_runtime["gpu"]))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from lie_ph.data import load_noisy_windows  # noqa: E402
from lie_ph.losses import ERROR_DIMENSION  # noqa: E402
from lie_ph.network import build_gp_setup  # noqa: E402
from lie_ph.train import ADAM, _labels, prodigy_distances, to_host  # noqa: E402
from ph_node.config import load_config, resolve_project_path  # noqa: E402
from ph_node.losses import rollout_loss, rollout_loss_reference  # noqa: E402
from ph_node.network import initialize_node_parameters, model_from_params  # noqa: E402

LOSSES = {"rollout": rollout_loss, "rollout_reference": rollout_loss_reference}


def make_optimizer(config: dict, params) -> optax.GradientTransformation:
    opt, total = config["optimizer"], int(config["training"]["total_steps"])
    betas = (float(opt["beta1"]), float(opt["beta2"]))
    # optimizer.schedule (optional, default cosine): constant = no decay (PH-NODE-ref, the reference's torch Adam).
    schedule = str(opt.get("schedule", "cosine"))
    if schedule not in ("cosine", "constant"):
        raise ValueError("optimizer.schedule must be cosine or constant")
    rate = float(opt["learning_rate"])
    learning_rate = (optax.constant_schedule(rate) if schedule == "constant"
                     else optax.cosine_decay_schedule(rate, max(total, 1), alpha=float(opt["final_learning_fraction"])))
    # optimizer.weight_decay (optional, default 0): L2 term added to the gradient before Adam, torch.optim.Adam's
    # weight_decay (the reference uses 1e-4).
    decay = float(opt.get("weight_decay", 0.0))
    adam = optax.adam(learning_rate, b1=betas[0], b2=betas[1], eps=float(opt["epsilon"]))
    groups = {ADAM: optax.chain(optax.add_decayed_weights(decay), adam) if decay > 0 else adam}
    clip_norm = opt.get("gradient_clip_norm")
    clip = optax.clip_by_global_norm(float(clip_norm)) if clip_norm else optax.identity()
    return optax.chain(clip, optax.multi_transform(groups, _labels(params)))


def uniform_rotations(count: int, seed: int) -> np.ndarray:
    """(count, 9) row-major rotations, uniform on SO(3): the quaternion recipe of Duong & Atanasov's pretrain()
    (http://planning.cs.uiuc.edu/node198.html) with their quaternion order (x, y, z, w)."""
    u1, u2, u3 = np.random.default_rng(seed).uniform(size=(3, count))
    x, y = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    z, w = np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)
    rotation = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)
    return rotation.astype(np.float32)


def pretrain_inverse_mass(params, setup, model_cfg: dict, seed: int, log, count: int = 250_000,
                          tolerance: float = 1.0e-6, max_steps: int = 20_000):
    """model.pretrain_inverse_mass_identity: Duong & Atanasov's SE3HamNODE.pretrain() for the SO(3) mass network.
    Full-batch Adam (lr 1e-3, no weight decay) on mean((M^-1_theta(R) - I)^2) over ``count`` uniform rotations until
    the mean is below ``tolerance`` (their loop has no cap; here the run aborts after ``max_steps``). Only the M
    network changes; D, V and G keep their initial weights."""
    rotations = jnp.asarray(uniform_rotations(count, seed))
    identity = jnp.eye(3, dtype=jnp.float32)

    def loss(m_weights):
        model = model_from_params({**params, "M": m_weights}, setup, model_cfg)
        return jnp.mean(jnp.square(jax.vmap(model.inverse_mass)(rotations) - identity))

    optimizer = optax.adam(1.0e-3)
    m_weights, state = params["M"], optimizer.init(params["M"])

    @jax.jit
    def step(m_weights, state):
        value, grads = jax.value_and_grad(loss)(m_weights)
        updates, state = optimizer.update(grads, state, m_weights)
        return optax.apply_updates(m_weights, updates), state, value

    value = float(loss(m_weights))
    log(f"pretraining M^-1 to I on {count} uniform rotations (Adam 1e-3, full batch, until mean sq. error < {tolerance:g}): start {value:.3e}")
    steps = 0
    while value >= tolerance:
        if steps >= max_steps:
            raise RuntimeError(f"M^-1 pretraining did not reach {tolerance:g} in {max_steps} steps (error {value:.3e})")
        m_weights, state, value = step(m_weights, state)
        value, steps = float(value), steps + 1
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite M^-1 pretraining loss at step {steps}")
        if steps % 500 == 0:
            log(f"  pretrain step {steps}: {value:.3e}")
    value = float(loss(m_weights))
    log(f"pretraining done in {steps} steps: mean sq. error {value:.3e}")
    return {**params, "M": m_weights}


def save_checkpoint(path: Path, params, config: dict, step: int) -> None:
    payload = {"framework": "JAX", "package": "ph_node/pendulum", "params": to_host(params),
               "config": {k: v for k, v in config.items() if not k.startswith("_")}, "step": step}
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_run_dir(config: dict, root_override: Path | None) -> Path:
    root = root_override or resolve_project_path(config["experiment"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%d-%m-%H-%M")
    run_dir = root / f"{stamp}_ph_node_{config['experiment']['name_suffix']}"
    counter = 2
    while run_dir.exists():
        run_dir = root / f"{stamp}_ph_node_{config['experiment']['name_suffix']}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True)
    shutil.copy2(config["_config_path"], run_dir / Path(config["_config_path"]).name)
    return run_dir


def train(config_path: Path, total_steps_override: int | None = None, root_override: Path | None = None) -> Path:
    config = load_config(config_path)
    if total_steps_override is not None:
        config["training"]["total_steps"] = int(total_steps_override)
    jax.config.update("jax_enable_x64", bool(config["runtime"]["enable_x64"]))
    training, model_cfg = config["training"], config["model"]
    dataset_path = resolve_project_path(config["data"]["dataset_path"])
    substeps = int(config["data"]["substeps"])
    train_np, test_np, interval = load_noisy_windows(dataset_path, int(config["data"]["window_points"]),
                                                     int(config["data"]["window_stride"]))
    run_dir = create_run_dir(config, root_override)
    log_path = run_dir / "train_log.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    seed = int(training["seed"])
    key_setup, key_params, key_train = jax.random.split(jax.random.PRNGKey(seed), 3)
    loss_kind = str(training["loss"])
    trajectory_loss = LOSSES[loss_kind]
    setup = build_gp_setup(model_cfg, key_setup)          # unused by the MLP subnetworks; kept for the shared loaders
    params = initialize_node_parameters(model_cfg, setup, key_params)
    pretrain_log: list[str] = []
    if bool(model_cfg.get("pretrain_inverse_mass_identity", False)):
        params = pretrain_inverse_mass(params, setup, model_cfg, seed, pretrain_log.append)
    optimizer = make_optimizer(config, params)
    opt_state = optimizer.init(params)

    train_windows = jnp.asarray(train_np)
    test_windows = jnp.asarray(test_np)
    total_steps, batch_size = int(training["total_steps"]), int(training["batch_size"])
    window_count = int(train_windows.shape[1])
    observation_count = float(window_count * (train_windows.shape[0] - 1))
    eval_count = test_windows.shape[1] if training["eval_windows"] is None else min(int(training["eval_windows"]),
                                                                                     int(test_windows.shape[1]))
    log(f"run {run_dir}")
    log(f"dataset {dataset_path} (sha256 {hashlib.sha256(dataset_path.read_bytes()).hexdigest()[:16]})")
    log(f"train windows {tuple(train_windows.shape)}  test windows {tuple(test_windows.shape)} (noisy)  "
        f"interval {interval:g} s  substeps {substeps}  eval windows {eval_count}")
    for line in pretrain_log:
        log(line)
    log(f"PH-NODE architecture {model_cfg.get('nn_architecture', 'xavier')}  integrator rk4  loss {loss_kind}  "
        f"parameters {sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))}")

    def objective(candidate, batch, key, beta):
        model = model_from_params(candidate, setup, model_cfg)
        nll = trajectory_loss(model, batch, interval, substeps)
        kl = jnp.zeros((), jnp.float32)                    # no weight posterior; the same objective form as lie_ph
        return nll + beta * kl / (observation_count * ERROR_DIMENSION), (nll, kl)

    @jax.jit
    def train_step(candidate, state, batch, key, beta):
        (loss, (nll, kl)), grads = jax.value_and_grad(objective, has_aux=True)(candidate, batch, key, beta)
        norm = optax.global_norm(grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(norm)
        updates, new_state = optimizer.update(grads, state, candidate)
        # Every step is applied; a non-finite loss or gradient aborts the run (never skipped).
        return optax.apply_updates(candidate, updates), new_state, loss, nll, kl, norm, finite

    @jax.jit
    def mean_model_loss(candidate, windows):
        return trajectory_loss(model_from_params(candidate, setup, model_cfg), windows, interval, substeps)

    def evaluate(candidate, windows, count, chunk=128):
        values = []
        for start in range(0, count, chunk):
            part = windows[:, start:min(start + chunk, count)]
            values.append(float(mean_model_loss(candidate, part)) * part.shape[1])
        return sum(values) / count

    stats: dict[str, list] = {name: [] for name in (
        "step", "loss", "train_ekf_nll", "kl", "beta", "gradient_norm", "step_seconds", "eval_step", "test_ekf_nll",
        "train_subset_ekf_nll", "sigma_obs", "diffusion", "prodigy_d")}
    stats["monitor_loss"] = [loss_kind]     # test_ekf_nll / train_subset_ekf_nll hold this trajectory loss
    rng = np.random.default_rng(seed)
    started = time.time()
    for step in range(1, total_steps + 1):
        indices = np.sort(rng.choice(window_count, size=min(batch_size, window_count), replace=False))
        beta = float(training["kl_beta_max"]) * min(1.0, step / max(1, int(training["kl_anneal_steps"])))
        tic = time.time()
        params, opt_state, loss, nll, kl, norm, finite = train_step(
            params, opt_state, train_windows[:, indices], jax.random.fold_in(key_train, step), jnp.asarray(beta))
        loss = float(loss)
        seconds = time.time() - tic
        if not bool(finite):
            log(f"step {step}: non-finite loss/gradient ({loss}, {float(norm)}) - run aborted")
            raise FloatingPointError(f"non-finite loss/gradient at step {step}")
        for name, value in (("step", step), ("loss", loss), ("train_ekf_nll", float(nll)), ("kl", float(kl)),
                            ("beta", beta), ("gradient_norm", float(norm)), ("step_seconds", seconds)):
            stats[name].append(value)
        if step % 50 == 0 or step == 1:
            log(f"step {step:6d} loss {loss:+.5f} {loss_kind} {float(nll):+.5f} |g| {float(norm):.3e} {seconds:.2f}s/step")
        if step % int(training["eval_every"]) == 0 or step == total_steps:
            test_value = evaluate(params, test_windows, eval_count)
            train_value = evaluate(params, train_windows, min(eval_count, window_count))
            stats["eval_step"].append(step)
            stats["test_ekf_nll"].append(test_value)
            stats["train_subset_ekf_nll"].append(train_value)
            stats["sigma_obs"].append([])
            stats["diffusion"].append([])
            stats["prodigy_d"].append(prodigy_distances(opt_state))
            log(f"MONITOR step {step}: test {loss_kind} loss (noisy, final model) {test_value:+.5f}  train subset {train_value:+.5f}")
            (run_dir / "metrics.json").write_text(json.dumps(stats, indent=1))
        if step % int(training["checkpoint_every"]) == 0:
            save_checkpoint(run_dir / f"checkpoint_step{step:06d}.pkl", params, config, step)
    save_checkpoint(run_dir / "checkpoint_final.pkl", params, config, total_steps)
    (run_dir / "metrics.json").write_text(json.dumps(stats, indent=1))
    log(f"done in {(time.time() - started) / 60:.1f} min; final test {loss_kind} (monitor) {stats['test_ekf_nll'][-1]:+.5f}")
    return run_dir


if __name__ == "__main__":
    train(ARGS.config, ARGS.total_steps, ARGS.root)

"""Train an SO(3) pendulum Lie-PH model (Lie-PH-GP-SDE/ODE, Lie-PH-NN-SDE/ODE).

    python src/models/3D_SO3_Windy_Pendulum/lie_ph/train.py --config src/models/3D_SO3_Windy_Pendulum/configs/<experiment>/<Lie-PH file>.yaml

Objective per step, on a batch of noisy windows and ONE reparameterised GP weight sample w ~ q(w):

    L = EKF-NLL(windows | w, sigma_obs, Sigma) / 6  +  beta * KL(q || p) / (N_obs * 6)

beta rises linearly to training.kl_beta_max over training.kl_anneal_steps; N_obs = predicted observations in the
training windows. Optimiser: Prodigy (one instance per subnetwork) on the level scalars, Adam on everything else,
both with the same cosine decay over training.total_steps, global-norm clipping. training.loss rollout (Lie-PH-NN-ODE)
replaces the EKF NLL by the trajectory loss of losses.rollout_loss.

Fixed budget: the result is the model after training.total_steps (checkpoint_final.pkl). The EKF NLL of the
posterior-mean model on the NOISY test split is logged as a monitor only; it never selects a checkpoint. Clean data
is never touched during training.
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

from lie_ph.config import load_config, resolve_project_path  # noqa: E402
from lie_ph.data import load_noisy_windows  # noqa: E402
from lie_ph.losses import ERROR_DIMENSION, ekf_nll, rollout_loss  # noqa: E402
from lie_ph.network import GP_SPECS, build_gp_setup, initialize_parameters, kl_divergence, model_from_params  # noqa: E402

ADAM = "adam"
GP_HYPER = "gp_hyper"                  # log_amplitude / log_length of every GP (model.learn_gp_hyperparameters)


def _labels(params, split_hyper: bool = False):
    def label(path, _leaf):
        keys = [getattr(part, "key", None) for part in path]
        if split_hyper and keys[-1] in ("log_amplitude", "log_length"):
            return GP_HYPER
        return f"{keys[0]}_level" if keys[-1] == "level" else ADAM
    return jax.tree_util.tree_map_with_path(label, params)


def make_optimizer(config: dict, params) -> optax.GradientTransformation:
    opt, total = config["optimizer"], int(config["training"]["total_steps"])
    betas = (float(opt["beta1"]), float(opt["beta2"]))

    def cosine(rate: float):
        return optax.cosine_decay_schedule(rate, max(total, 1), alpha=float(opt["final_learning_fraction"]))

    groups = {ADAM: optax.adam(cosine(float(opt["learning_rate"])), b1=betas[0], b2=betas[1], eps=float(opt["epsilon"]))}
    # optimizer.hyper_learning_rate (optional): its own Adam rate for the GP amplitudes and length-scales, same cosine
    # decay; absent = they share optimizer.learning_rate with the GP weights (the behaviour before 5 Oct 2026).
    # optimizer.hyper_optimizer: prodigy (optional): one Prodigy instance for those 8 scalars instead of Adam, with the
    # levels' settings (base rate optimizer.hyper_learning_rate if given, else optimizer.level_learning_rate).
    hyper_prodigy = str(opt.get("hyper_optimizer", "adam")) == "prodigy"
    if str(opt.get("hyper_optimizer", "adam")) not in ("adam", "prodigy"):
        raise ValueError("optimizer.hyper_optimizer must be adam or prodigy")
    split_hyper = "hyper_learning_rate" in opt or hyper_prodigy
    labels = _labels(params, split_hyper)
    present = set(jax.tree_util.tree_leaves(labels))
    if GP_HYPER in present and hyper_prodigy:
        groups[GP_HYPER] = optax.contrib.prodigy(
            learning_rate=cosine(float(opt.get("hyper_learning_rate", opt["level_learning_rate"]))), betas=betas,
            eps=float(opt["epsilon"]), estim_lr0=float(opt.get("hyper_initial_distance", 1.0e-3)),   # optimizer.hyper_initial_distance: Prodigy d0 of tau, ell
            safeguard_warmup=True)
    elif GP_HYPER in present:
        groups[GP_HYPER] = optax.adam(cosine(float(opt["hyper_learning_rate"])), b1=betas[0], b2=betas[1], eps=float(opt["epsilon"]))
    for name in GP_SPECS:
        if f"{name}_level" not in present:      # the NN family has no level scalars
            continue
        # Prodigy's distance estimate d is not decayed by itself: without the schedule the levels keep taking
        # full-size steps to the last iteration (measured: d frozen after ~1000 steps of the 10k runs).
        # optimizer.level_initial_distance (optional, default 1e-3): Prodigy's start distance d0 for the levels. With
        # 1e-3 the M level's d stays ~0.002-0.008 and the level cannot travel far enough in 5000 steps (5 Oct 2026).
        groups[f"{name}_level"] = optax.contrib.prodigy(learning_rate=cosine(float(opt["level_learning_rate"])),
                                                        betas=betas, eps=float(opt["epsilon"]),
                                                        estim_lr0=float(opt.get("level_initial_distance", 1.0e-3)),
                                                        safeguard_warmup=True)
    # optimizer.gradient_clip_norm: null (or 0) = no clipping (6 Oct 2026 test); otherwise global-norm clip before the groups.
    clip_norm = opt.get("gradient_clip_norm")
    clip = optax.clip_by_global_norm(float(clip_norm)) if clip_norm else optax.identity()
    return optax.chain(clip,
                       optax.multi_transform(groups, labels))


def prodigy_distances(opt_state) -> dict[str, float]:
    found = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(opt_state):
        keys = [getattr(part, "key", getattr(part, "name", None)) for part in path]
        if keys and keys[-1] == "estim_lr":
            group = next((str(k) for k in keys if isinstance(k, str) and (k.endswith("_level") or k == GP_HYPER)), "?")
            found[group] = float(np.asarray(leaf).reshape(-1)[0])
    return found


def to_host(tree):
    return jax.tree_util.tree_map(lambda v: np.asarray(jax.device_get(v)), tree)


def save_checkpoint(path: Path, params, config: dict, step: int, extra: dict) -> None:
    payload = {"framework": "JAX", "package": "lie_ph/pendulum", "params": to_host(params),
               "config": {k: v for k, v in config.items() if not k.startswith("_")}, "step": step, **extra}
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_run_dir(config: dict, root_override: Path | None) -> Path:
    root = root_override or resolve_project_path(config["experiment"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%d-%m-%H-%M")
    run_dir = root / f"{stamp}_lie_ph_{config['experiment']['name_suffix']}"
    counter = 2
    while run_dir.exists():
        run_dir = root / f"{stamp}_lie_ph_{config['experiment']['name_suffix']}_{counter:02d}"
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
    loss_kind = str(training.get("loss", "ekf"))
    if loss_kind not in ("ekf", "rollout"):
        raise ValueError("training.loss must be ekf or rollout")
    setup = build_gp_setup(model_cfg, key_setup)
    params = initialize_parameters(model_cfg, setup, key_params, with_likelihood=(loss_kind == "ekf"))
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
    log(f"model family {model_cfg.get('family', 'gp')}  wind {model_cfg.get('wind', True)}  integrator lie_imex  "
        f"loss {loss_kind}  "
        f"parameters {sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))}")

    def data_fit(model, candidate, batch):
        if loss_kind == "rollout":
            return rollout_loss(model, batch, interval, substeps)
        return ekf_nll(model, batch, candidate["likelihood"]["log_sigma"], interval, substeps)

    def objective(candidate, batch, key, beta):
        model = model_from_params(candidate, setup, key, model_cfg)
        nll = data_fit(model, candidate, batch)
        kl = kl_divergence(candidate)                      # zero for the NN family
        return nll + beta * kl / (observation_count * ERROR_DIMENSION), (nll, kl)

    @jax.jit
    def train_step(candidate, state, batch, key, beta):
        (loss, (nll, kl)), grads = jax.value_and_grad(objective, has_aux=True)(candidate, batch, key, beta)
        norm = optax.global_norm(grads)
        finite = jnp.isfinite(loss) & jnp.isfinite(norm)
        updates, new_state = optimizer.update(grads, state, candidate)
        # Every step is applied. A non-finite loss or gradient is NOT skipped: the loop below aborts the run
        # (user decision 6 Oct 2026 - a model that only trains by dropping bad steps is not an honest result).
        return optax.apply_updates(candidate, updates), new_state, loss, nll, kl, norm, finite

    @jax.jit
    def mean_model_nll(candidate, windows):
        return data_fit(model_from_params(candidate, setup, None, model_cfg), candidate, windows)

    def evaluate(candidate, windows, count, chunk=128):
        values = []
        for start in range(0, count, chunk):
            part = windows[:, start:min(start + chunk, count)]
            values.append(float(mean_model_nll(candidate, part)) * part.shape[1])
        return sum(values) / count

    stats: dict[str, list] = {name: [] for name in (
        "step", "loss", "train_ekf_nll", "kl", "beta", "gradient_norm", "step_seconds", "eval_step", "test_ekf_nll",
        "train_subset_ekf_nll", "sigma_obs", "diffusion", "prodigy_d")}
    stats["monitor_loss"] = [loss_kind]     # test_ekf_nll / train_subset_ekf_nll hold this loss (EKF NLL or rollout loss)
    rng = np.random.default_rng(seed)
    kl_scale = 1.0 / (observation_count * ERROR_DIMENSION)
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
            sigma_obs = np.exp(np.asarray(params["likelihood"]["log_sigma"])) if "likelihood" in params else np.zeros(0)
            diffusion = np.exp(np.asarray(params["process"]["log_sigma"])) if "process" in params else np.zeros(0)
            log(f"step {step:6d} loss {loss:+.5f} {loss_kind} {float(nll):+.5f} kl-term {beta * float(kl) * kl_scale:.5f} "
                f"(raw kl {float(kl):.3e}, beta {beta:.3f}) "
                f"|g| {float(norm):.3e} sigma_obs {np.round(sigma_obs, 4).tolist()} "
                f"Sigma {np.round(diffusion, 4).tolist()} {seconds:.2f}s/step")
        if step % int(training["eval_every"]) == 0 or step == total_steps:
            test_value = evaluate(params, test_windows, eval_count)
            train_value = evaluate(params, train_windows, min(eval_count, window_count))
            distances = prodigy_distances(opt_state)
            stats["eval_step"].append(step)
            stats["test_ekf_nll"].append(test_value)
            stats["train_subset_ekf_nll"].append(train_value)
            stats["sigma_obs"].append(np.exp(np.asarray(params["likelihood"]["log_sigma"])).tolist() if "likelihood" in params else [])
            stats["diffusion"].append(np.exp(np.asarray(params["process"]["log_sigma"])).tolist() if "process" in params else [])
            stats["prodigy_d"].append(distances)
            log(f"MONITOR step {step}: test {'EKF-NLL' if loss_kind == 'ekf' else 'rollout loss'} (noisy, mean model) {test_value:+.5f}  train subset {train_value:+.5f}"
                f"  prodigy d {({k: round(v, 5) for k, v in distances.items()})}")
            hyper = {name: [round(float(np.exp(params[name]["log_amplitude"])), 4), round(float(np.exp(params[name]["log_length"])), 4)]
                     for name in GP_SPECS if isinstance(params.get(name), dict) and "log_amplitude" in params[name]}
            if hyper:                                        # learned GP hyperparameters: [amplitude tau, length-scale factor ell]
                stats.setdefault("gp_hyperparameters", []).append(hyper)
                log(f"GP hyperparameters step {step} [tau, ell]: {hyper}")
            (run_dir / "metrics.json").write_text(json.dumps(stats, indent=1))
        if step % int(training["checkpoint_every"]) == 0:
            save_checkpoint(run_dir / f"checkpoint_step{step:06d}.pkl", params, config, step, {})
    save_checkpoint(run_dir / "checkpoint_final.pkl", params, config, total_steps, {})
    (run_dir / "metrics.json").write_text(json.dumps(stats, indent=1))
    log(f"done in {(time.time() - started) / 60:.1f} min; final test {loss_kind} (monitor) {stats['test_ekf_nll'][-1]:+.5f}")
    return run_dir


if __name__ == "__main__":
    train(ARGS.config, ARGS.total_steps, ARGS.root)

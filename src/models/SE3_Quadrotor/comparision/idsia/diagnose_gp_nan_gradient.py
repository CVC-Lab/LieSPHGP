"""Where does a ph_gp_lie_imex_idsia training run's gradient first go NaN?  Read-only: no training code is changed.

The run is replayed from one of its checkpoints with the SAME batch sequence and random keys the original loop drew
(np.random.default_rng(seed).choice once per step, jax.random.split(key) once per step), the same objective
(``_gp_components`` + KL + latent initial-state penalty, as in ``train_from_config``) and the same optimizer state.
One difference from the original: the checkpoint stores the latent initial-state parameters but not their Adam
moments, so those restart -- the replay can drift slightly from the original run.

At the first step whose gradient is non-finite it reports:
  1. which parameter leaves have non-finite gradient entries (NaN vs inf, how many)
  2. which windows of that batch produce a non-finite gradient on their own, with their forward rollout statistics
  3. the first primitive that produces a NaN, located with jax_debug_nans on the worst window

Usage:
    python diagnose_gp_nan_gradient.py --run DIR --from-step 2000 [--max-steps 3000] --output-name NAME
"""
from __future__ import annotations

import argparse
import glob
import json
import traceback
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import benchmark_protocol as bp

from src.models.SE3_Quadrotor.ph_gp_lie_imex_idsia import MODEL_NAME
from src.models.SE3_Quadrotor.ph_gp_lie_imex_idsia import train as T
from src.models.SE3_Quadrotor.ph_gp_lie_imex_idsia.checkpoints import load_checkpoint
from src.models.SE3_Quadrotor.ph_gp_lie_imex_idsia.config import load_config, resolve_project_path
from src.models.SE3_Quadrotor.ph_gp_lie_imex_idsia.data import load_dataset
from src.models.SE3_Quadrotor.ph_gp_lie_imex_idsia.integrator import exp_so3

EVAL_ROOT = bp.PROJECT_ROOT / "experiments/quadrotor/eval_runs"
BLOCKS = ("log_sigma_position", "log_sigma_attitude", "log_sigma_linear_velocity", "log_sigma_angular_velocity")


def leaf_report(tree) -> list[dict]:
    out = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        a = np.asarray(jax.device_get(leaf))
        name = "/".join(str(getattr(k, "key", k)) for k in path)
        nan, inf = int(np.isnan(a).sum()), int(np.isinf(a).sum())
        finite = a[np.isfinite(a)]
        out.append({"parameter": name, "shape": list(a.shape), "nan": nan, "inf": inf, "size": int(a.size),
                    "max_abs_finite": float(np.abs(finite).max()) if finite.size else None})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--from-step", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    config = load_config(sorted(glob.glob(str(arguments.run / "*.yaml")))[0], expected_model=MODEL_NAME)
    device = T._configure_jax(config)
    training, gp = config["training"], config["gp"]
    dataset_path = resolve_project_path(config["data"]["dataset_path"])
    train_numpy, _test, times, _settings = load_dataset(
        dataset_path, window_points=int(config["data"]["window_points"]),
        window_stride=int(config["data"]["window_stride"]), max_train_windows=config["data"]["max_train_windows"],
        max_test_windows=config["data"]["max_test_windows"],
        observation_noise_override=config["data"]["observation_noise_override"], seed=int(training["seed"]))
    config["_step_size"] = float(times[1] - times[0])
    dtype = jnp.zeros(()).dtype
    train_states = jax.device_put(jnp.asarray(train_numpy, dtype=dtype), device)

    payload = load_checkpoint(arguments.run / f"checkpoint_step_{arguments.from_step:05d}.pkl")
    extra = payload["extra"]
    assert int(extra["step"]) == arguments.from_step
    candidate = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    gp_setup = jax.tree_util.tree_map(jnp.asarray, extra["gp_setup"])
    optimizer = T._optimizer(config, candidate)
    opt_state = jax.tree_util.tree_map(jnp.asarray, extra["optimizer_state"])
    latent = jax.tree_util.tree_map(jnp.asarray, extra["latent_initial_state"])
    latent_optimizer = optax.adam(float(training.get("latent_initial_state_learning_rate") or 1e-2))
    latent_state = latent_optimizer.init(latent)            # not stored in the checkpoint: restarts (see docstring)

    seed = int(training["seed"])
    batch_size = min(int(training["batch_size"]), int(train_states.shape[1]))
    n_obs = float(train_states.shape[1] * (train_states.shape[0] - 1))
    transitions = float(train_states.shape[0] - 1)
    latent_kl_weight = float(training.get("latent_initial_state_kl_weight", 1.0))
    assert bool(training.get("latent_initial_state")) and not bool(training.get("latent_initial_state_sample", True))
    assert float(training.get("continuity_weight") or 0.0) == 0.0
    assert float(gp.get("gravity_penalty_weight") or 0.0) == 0.0 and float(gp.get("actuation_penalty_weight") or 0.0) == 0.0

    def objective(cand, lat, batch, indices, key, step_value):
        mean = lat["mean"][indices]
        sigma = jnp.repeat(jnp.stack([jnp.exp(cand["likelihood"][b]) for b in BLOCKS]), 3)
        latent_kl = jnp.mean(jnp.sum(mean**2 / (2.0 * sigma**2) + jnp.log(sigma), axis=1))
        x0 = batch[0]
        corrected = jnp.concatenate([x0[:, :3] + mean[:, :3],
                                     (x0[:, 3:12].reshape(-1, 3, 3) @ exp_so3(mean[:, 3:6])).reshape(-1, 9),
                                     x0[:, 12:15] + mean[:, 6:9], x0[:, 15:18] + mean[:, 9:12], x0[:, 18:]], axis=1)
        batch = batch.at[0].set(corrected)
        metrics, nll, kl, prediction = T._gp_components(config, cand, gp_setup, batch, key)
        data_fit = nll[0] if gp["data_fit"] == "se3-nll" else metrics[0]
        beta = float(gp["kl_beta_max"]) * jnp.minimum(1.0, step_value / max(1, int(gp["kl_anneal_steps"])))
        total = float(training["loss_scale"]) * data_fit + beta * kl / n_obs + latent_kl_weight * latent_kl / transitions
        return total, prediction

    grad_fn = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
    grad_jit = jax.jit(grad_fn)

    @jax.jit
    def apply(cand, state, lat, lstate, grads, lgrads):
        updates, state = optimizer.update(grads, state, cand)
        lup, lstate = latent_optimizer.update(lgrads, lstate, lat)
        return optax.apply_updates(cand, updates), state, optax.apply_updates(lat, lup), lstate

    # ---- reproduce the original random streams up to the checkpoint -------------------------------------
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, _init = jax.random.split(key)
    for _ in range(arguments.from_step):
        rng.choice(int(train_states.shape[1]), size=batch_size, replace=False)
        key, _ = jax.random.split(key)

    failure = None
    for step in range(arguments.from_step + 1, arguments.max_steps + 1):
        indices = rng.choice(int(train_states.shape[1]), size=batch_size, replace=False)
        key, sample_key = jax.random.split(key)
        batch = train_states[:, indices]
        idx = jnp.asarray(indices, dtype=jnp.int32)
        step_value = jnp.asarray(step, dtype=dtype)
        (value, _pred), (grads, lgrads) = grad_jit(candidate, latent, batch, idx, sample_key, step_value)
        gnorm = float(optax.global_norm(grads))
        if step % 50 == 0:
            print(f"  replay step {step}: objective {float(value):+.4f}  |grad| {gnorm:.3g}", flush=True)
        if not (np.isfinite(float(value)) and np.isfinite(gnorm)):
            failure = {"step": step, "objective": float(value), "gradient_norm": gnorm}
            break
        candidate, opt_state, latent, latent_state = apply(candidate, opt_state, latent, latent_state, grads, lgrads)

    folder = EVAL_ROOT / arguments.output_name
    folder.mkdir(parents=True, exist_ok=True)
    report = {"generated_at": datetime.now().astimezone().isoformat(), "run": str(arguments.run.resolve()),
              "from_step": arguments.from_step, "failure": failure}
    if failure is None:
        print(f"\nno non-finite gradient between steps {arguments.from_step + 1} and {arguments.max_steps}")
        (folder / "nan_gradient_diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
        return

    print(f"\nFIRST NON-FINITE GRADIENT at replay step {failure['step']}: objective {failure['objective']}, "
          f"|grad| {failure['gradient_norm']}")
    # ---- 1. which parameters ----------------------------------------------------------------------------
    leaves = leaf_report(grads)
    bad = [l for l in leaves if l["nan"] or l["inf"]]
    print(f"\n1. parameters with non-finite gradient ({len(bad)} of {len(leaves)} leaves):")
    for l in sorted(bad, key=lambda r: -(r["nan"] + r["inf"]) / r["size"]):
        print(f"   {l['parameter']:<40} shape {str(l['shape']):<10} NaN {l['nan']:>5}/{l['size']:<5} inf {l['inf']}")
    good = [l["parameter"] for l in leaves if not (l["nan"] or l["inf"])]
    print(f"   finite gradients: {', '.join(good) if good else 'none'}")
    latent_bad = leaf_report(lgrads)
    print(f"   latent initial-state gradients non-finite: "
          f"{[l['parameter'] for l in latent_bad if l['nan'] or l['inf']] or 'none'}")
    report["parameters"] = leaves
    report["latent_parameters"] = latent_bad

    # ---- 2. which windows ---------------------------------------------------------------------------
    single = jax.jit(grad_fn)
    offenders = []
    for i, window in enumerate(indices):
        (v1, pred), (g1, _l1) = single(candidate, latent, batch[:, i:i + 1], idx[i:i + 1], sample_key, step_value)
        if not np.isfinite(float(optax.global_norm(g1))):
            p = np.asarray(pred)[:, 0]
            offenders.append({"batch_position": i, "window": int(window), "objective": float(v1),
                              "prediction_finite": bool(np.isfinite(p[:, :18]).all()),
                              "max_speed": float(np.nanmax(np.linalg.norm(p[:, 12:15], axis=-1))),
                              "max_omega": float(np.nanmax(np.linalg.norm(p[:, 15:18], axis=-1))),
                              "max_position_error": float(np.nanmax(np.linalg.norm(
                                  p[:, :3] - np.asarray(batch[:, i, :3]), axis=-1)))})
    print(f"\n2. windows whose OWN gradient is non-finite: {len(offenders)} of {len(indices)}")
    for o in offenders[:10]:
        print(f"   window {o['window']:>5}: objective {o['objective']:+.3f}  forward finite {o['prediction_finite']}  "
              f"max|v| {o['max_speed']:.3g}  max|w| {o['max_omega']:.3g}  max pos err {o['max_position_error']:.3g}")
    report["windows"] = offenders

    # ---- 3. which primitive -------------------------------------------------------------------------
    if offenders:
        i = offenders[0]["batch_position"]
        jax.config.update("jax_debug_nans", True)
        try:
            grad_fn(candidate, latent, batch[:, i:i + 1], idx[i:i + 1], sample_key, step_value)
            print("\n3. jax_debug_nans did not trigger on the un-jitted single window")
            report["first_nan_primitive"] = None
        except FloatingPointError as error:
            frames = [f for f in traceback.extract_tb(error.__traceback__) if "SE3_Quadrotor" in f.filename]
            lines = [f"{Path(f.filename).name}:{f.lineno} in {f.name}: {f.line}" for f in frames]
            print(f"\n3. first NaN-producing primitive: {str(error).splitlines()[0]}")
            for line in lines[-8:]:
                print(f"   {line}")
            report["first_nan_primitive"] = {"message": str(error).splitlines()[0], "frames": lines}
        finally:
            jax.config.update("jax_debug_nans", False)

    (folder / "nan_gradient_diagnosis.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwritten {folder / 'nan_gradient_diagnosis.json'}")


if __name__ == "__main__":
    main()

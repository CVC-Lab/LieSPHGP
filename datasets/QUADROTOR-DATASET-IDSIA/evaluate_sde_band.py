"""Predictive-band calibration of a PH-GP model on the Melon protocol windows: ODE (weight samples only) or SDE
(weight samples + process noise), scored with the same rolling H = 50 windows as benchmark_protocol.py.

For every start t and horizon h the predictive law is moment-matched from S sample paths,
    N(mu_h, diag(var_h) + sigma_obs^2 I)      (attitude in the tangent space of the projected mean rotation)
and we report, per block: coverage of the truth by the 2-sigma band, the mean predicted sigma against the rms
error (a calibration ratio; 1 = calibrated), Spearman rho between predicted sigma and error over starts, and the
MAE of the sample-mean path so it can be set next to the deterministic protocol numbers.

Usage: python evaluate_sde_band.py --run <run dir> [--samples 32] [--stride 10] [--horizons 10 25 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation  # noqa: E402
from src.models.SE3_Quadrotor.ph_gp_lie_imex_sde_idsia import network as sde_network  # noqa: E402
from src.models.SE3_Quadrotor.ph_gp_lie_imex_sde_idsia.integrator import rollout_control_sequence_sde  # noqa: E402
from src.models.SE3_Quadrotor.ph_gp_lie_imex_sde_idsia.losses import log_so3, project_rotation  # noqa: E402
import benchmark_protocol as protocol  # noqa: E402
from src.models.SE3_Quadrotor.ph_nn_lie_imex_sde.integrator import rollout_control_sequence_sde as nn_rollout_sde  # noqa: E402

BLOCKS = {"p": slice(0, 3), "v": slice(12, 15), "w": slice(15, 18)}


def sample_paths(params, gp_setup, windows: np.ndarray, samples: int, seed: int, step: float) -> np.ndarray:
    """(S, T, B, 22) stochastic rollouts; process sigma is zero for ODE checkpoints (weights-only band)."""
    states = jnp.asarray(np.transpose(windows, (1, 0, 2)))            # time-major
    controls = states[1:, :, 18:22]
    h = jnp.asarray(step, dtype=states.dtype)
    keys = jax.random.split(jax.random.PRNGKey(seed), samples)

    def one(key):
        weight_key, noise_key = jax.random.split(key)
        model = sde_network.DissipativeSE3HamODE(params, gp_setup).sample(weight_key)
        noise = jax.random.normal(noise_key, controls.shape[:2] + (6,), dtype=states.dtype)
        return rollout_control_sequence_sde(model, states[0], controls, h, noise)

    return np.asarray(jax.vmap(one)(keys))


def sample_paths_nn(model, windows: np.ndarray, samples: int, seed: int, step: float) -> np.ndarray:
    """(S, T, B, 22) rollouts of a PH-NN-SDE point estimate: the band is process noise (constant or the
    state-dependent diffusion subnetwork) plus the observation noise; there is no weight uncertainty."""
    dtype = jax.tree_util.tree_leaves(model)[0].dtype
    states = jnp.asarray(np.transpose(windows, (1, 0, 2)), dtype=dtype)
    controls = states[1:, :, 18:22]
    h = jnp.asarray(step, dtype=dtype)
    keys = jax.random.split(jax.random.PRNGKey(seed), samples)

    def one(key):
        noise = jax.random.normal(key, controls.shape[:2] + (6,), dtype=dtype)
        return nn_rollout_sde(model, states[0], controls, h, noise)

    return np.asarray(jax.vmap(one)(keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--selected-step", type=int, default=None)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "tmp/idsia_raw/data/test")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-observation-noise", action="store_true",
                        help="predictive band from the sample spread only (state uncertainty); use on CLEAN test flights when "
                             "the model was trained on noisy observations, otherwise sigma_obs swamps the band")
    parser.add_argument("--dataset", default=None,
                        help="PATH[@split] of a HARD-layout pickle (whole flights, 22 columns); overrides the Melon CSV source")
    arguments = parser.parse_args()

    run = evaluation.load_run(Path(arguments.run), arguments.selected_step)
    _, params, gp_setup = evaluation.build_model(run)
    is_gp = bool(gp_setup)            # PH-NN checkpoints return the module itself and an empty setup
    if arguments.dataset:
        import pickle
        path, _, split = str(arguments.dataset).partition("@")
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        split = split or ("heldout" if "heldout_trajectories" in data else "test")
        flights = list(np.asarray(data[f"{split}_trajectories"], dtype=np.float64))
        input_mode = f"pickle:{Path(path).stem}@{split}"
    else:
        input_mode = protocol.input_mode_of_run(run)
        flights = protocol.load_melon(arguments.source, input_mode)
    windows = protocol.rolling_windows(flights, arguments.horizon, arguments.stride)
    if is_gp:
        sigma_obs = {name: float(np.exp(np.asarray(params["likelihood"][f"log_sigma_{name}"])))
                     for name in ("position", "attitude", "linear_velocity", "angular_velocity")}
        process = params.get("process")
        process_sigma = None if process is None else np.exp(np.asarray(process["log_sigma"])).tolist()
        if "sigma" in params:
            process_sigma = f"state-dependent diffusion GP ({sde_network.diffusion_mode(gp_setup)})"
    else:
        sigma_obs = dict(zip(("position", "attitude", "linear_velocity", "angular_velocity"),
                             np.exp(np.asarray(params.log_sigma_observation)).tolist()))
        process_sigma = (np.exp(np.asarray(params.process_log_sigma)).tolist() if params.sigma_net is None
                         else f"state-dependent diffusion subnetwork ({params.diffusion_input})")
    print(f"\n{Path(arguments.run).name}\n  input={input_mode}  windows={windows.shape[0]} (stride {arguments.stride})  "
          f"S={arguments.samples}  sigma_obs={ {k: round(v, 4) for k, v in sigma_obs.items()} }  process_sigma={process_sigma}")

    mean_paths, var_paths, truths, diverged = [], [], [], []
    for begin in range(0, windows.shape[0], arguments.batch):
        chunk = windows[begin:begin + arguments.batch]
        paths = (sample_paths(params, gp_setup, chunk, arguments.samples, arguments.seed + begin, 0.01) if is_gp
                 else sample_paths_nn(params, chunk, arguments.samples, arguments.seed + begin, 0.01))  # (S, T, B, 22)
        # a window with any non-finite or runaway path (> 10 m from the truth) is counted as diverged and
        # left out of the moment statistics; its share is reported separately below
        ok = (np.isfinite(paths).all(axis=(0, 1, 3))
              & (np.nan_to_num(np.linalg.norm(paths[..., :3] - np.transpose(chunk, (1, 0, 2))[None, ..., :3], axis=-1), nan=np.inf).max(axis=(0, 1)) <= 10.0)
              & (np.nan_to_num(np.linalg.norm(paths[..., 15:18], axis=-1), nan=np.inf).max(axis=(0, 1)) <= 100.0)
              & (np.nan_to_num(np.linalg.norm(paths[..., 12:15], axis=-1), nan=np.inf).max(axis=(0, 1)) <= 50.0))
        diverged.append(~ok)
        paths, chunk = paths[:, :, ok], chunk[ok]
        if chunk.shape[0] == 0:
            continue
        mean = paths.mean(axis=0)
        rot_mean = np.asarray(project_rotation(jnp.asarray(mean[..., 3:12])))                          # (T, B, 3, 3)
        S = paths.shape[0]
        var = {k: paths[..., s].var(axis=0, ddof=1) for k, s in BLOCKS.items()}
        tangents = np.asarray(log_so3(jnp.asarray(np.swapaxes(rot_mean, -1, -2)[None] @ paths[..., 3:12].reshape(*paths.shape[:-1], 3, 3))))
        var["R"] = (tangents ** 2).sum(axis=0) / (S - 1)
        truth = np.transpose(chunk, (1, 0, 2))
        resid = {k: truth[..., s] - mean[..., s] for k, s in BLOCKS.items()}
        resid["R"] = np.asarray(log_so3(jnp.asarray(np.swapaxes(rot_mean, -1, -2) @ truth[..., 3:12].reshape(*truth.shape[:-1], 3, 3))))
        mean_paths.append((resid, var)); print(f"  rolled {begin + chunk.shape[0]}/{windows.shape[0]}", end="\r", flush=True)
    print(" " * 40, end="\r")
    diverged = np.concatenate(diverged)
    print(f"  diverged windows (any of the {arguments.samples} paths non-finite, > 10 m off, |omega| > 100 or |v| > 50): {int(diverged.sum())} / {diverged.size} "
          f"({100 * diverged.mean():.2f} %); statistics below are over the remaining windows")
    obs_var = {"p": sigma_obs["position"] ** 2, "R": sigma_obs["attitude"] ** 2,
               "v": sigma_obs["linear_velocity"] ** 2, "w": sigma_obs["angular_velocity"] ** 2}
    if arguments.exclude_observation_noise:
        obs_var = {k: 0.0 for k in obs_var}
        print("  observation-noise term EXCLUDED from the band (sample spread only)")
    likelihood = params["likelihood"] if is_gp else {}
    gamma = {k: float(np.exp(np.asarray(likelihood[f"log_gamma_{n}"]))) if f"log_gamma_{n}" in likelihood else 0.0
             for k, n in (("p", "position"), ("R", "attitude"), ("v", "linear_velocity"), ("w", "angular_velocity"))}
    if any(gamma.values()):
        print(f"  horizon-growing observation variance gamma (units^2/s): { {k: round(v, 5) for k, v in gamma.items()} }")
    names = {"p": "position", "R": "attitude", "v": "velocity", "w": "angular rate"}
    result = {}
    print(f"\n  {'block':13s}{'h':>4}{'coverage@2s':>13}{'mean sigma':>12}{'rms error':>11}{'sigma/err':>11}{'spearman':>10}{'MAE(mean path)':>16}")
    for k in ("p", "R", "v", "w"):
        for h in arguments.horizons:
            res = np.concatenate([r[k][h] for r, _ in mean_paths], axis=0)          # (N, 3)
            var = np.concatenate([v[k][h] for _, v in mean_paths], axis=0) + obs_var[k] + gamma[k] * h * 0.01
            inside = np.all(np.abs(res) <= 2.0 * np.sqrt(var), axis=-1).mean()
            sigma = np.sqrt(var.sum(-1)); err = np.linalg.norm(res, axis=-1)
            rho = spearmanr(sigma, err).correlation
            ratio = np.sqrt(np.mean(var.sum(-1))) / np.sqrt(np.mean(err ** 2))
            result[f"{k}_h{h}"] = {"coverage_2sigma": float(inside), "sigma_over_rms_error": float(ratio), "spearman": float(rho), "mae_mean_path": float(err.mean())}
            print(f"  {names[k]:13s}{h:>4}{inside:>13.3f}{np.sqrt(np.mean(var.sum(-1))):>12.4f}{np.sqrt(np.mean(err**2)):>11.4f}{ratio:>11.2f}{rho:>10.2f}{err.mean():>16.4f}")
    suffix = (f"_{Path(str(arguments.dataset).partition('@')[0]).stem}" if arguments.dataset else "") + ("_noobs" if arguments.exclude_observation_noise else "")
    out = Path(arguments.run) / f"sde_band_S{arguments.samples}_stride{arguments.stride}{suffix}.json"
    out.write_text(json.dumps({"sigma_obs": sigma_obs, "process_sigma": process_sigma, "samples": arguments.samples,
                               "diverged_window_fraction": float(diverged.mean()),
                               "stride": arguments.stride, "results": result}, indent=2) + "\n")
    print(f"\n  written {out}")


if __name__ == "__main__":
    main()

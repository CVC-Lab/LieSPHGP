"""Predictive calibration on the Melon protocol windows (paper Table 2), for every model that has a predictive law.

Each window (start t, H = 50 steps, the benchmark protocol) is rolled out S times from the true state with the recorded
inputs: GP-ODE = posterior weight samples; GP-SDE = weight samples x Brownian paths; NN-SDE = Brownian paths. Per state
block (position, attitude as the rotation vector about the projected mean rotation, body velocity, body rate) and per axis the
predictive law is N(mu, var + sigma_obs^2) (sigma_obs: learned for the GP models, the fixed value for NN-SDE), and over
all windows and all horizons h = 1..50:
    coverage@k = fraction of axis values with |y - mu| <= k sigma          (targets 0.683 / 0.954)
    NLPD       = mean of 0.5 log(2 pi sigma^2) + (y - mu)^2 / (2 sigma^2)  (nats per axis)
    sharpness  = mean sigma of the block (norm over its 3 axes)
    Spearman   = rank correlation of the block's sigma norm and its error norm over (window, h)
Windows where any path is non-finite or runs away (> 10 m, |omega| > 100, |v| > 50) are counted and left out.

Usage: python idsia_calibration_table.py --run LABEL=DIR[@STEP] [...] [--samples 32] [--stride 10] --output-name NAME
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

import benchmark_protocol as bp
import evaluate_sde_band as band
from src.models.SE3_Quadrotor.comparision import four_model_comparison as F
from src.models.SE3_Quadrotor.ph_gp_lie_imex_sde_idsia.losses import log_so3, project_rotation

BLOCKS = {"p": slice(0, 3), "v": slice(12, 15), "w": slice(15, 18)}
OBS_NAMES = {"p": "position", "R": "attitude", "v": "linear_velocity", "w": "angular_velocity"}


def sampler(run_dir: str, step: str):
    """(paths(chunk) -> (S, T, B, 22), sigma_obs per block, description) for one run."""
    run = F.evaluation.load_run(Path(run_dir), int(step) if step else None)
    name = run["config"]["model"]["name"]
    if name in F.KIND_BY_PACKAGE and F.KIND_BY_PACKAGE[name] in ("gp_sde", "nn_sde"):
        with (bp.PROJECT_ROOT / run["config"]["data"]["dataset_path"]).open("rb") as handle:
            settings = pickle.load(handle)["settings"]
        entry = F.Entry(name, f"{run_dir}@{step}" if step else run_dir, settings)
        if entry.kind == "gp_sde":
            likelihood = entry.params["likelihood"]
            sigma = {k: float(np.exp(np.asarray(likelihood[f"log_sigma_{n}"]).ravel()[0])) for k, n in OBS_NAMES.items()}
        else:
            fixed = float(run["config"]["sde"]["fixed_observation_sigma"])
            sigma = {k: fixed for k in OBS_NAMES}
        return (lambda chunk, samples, seed: entry.sample_paths(np.transpose(chunk, (1, 0, 2)), samples, seed)), sigma, entry.describe()
    _, params, gp_setup = F.evaluation.build_model(run)
    if not gp_setup:
        raise SystemExit(f"{run_dir}: a point-estimate ODE has no predictive law")
    sigma = {k: float(np.exp(np.asarray(params["likelihood"][f"log_sigma_{n}"]).ravel()[0])) for k, n in OBS_NAMES.items()}
    return (lambda chunk, samples, seed: band.sample_paths(params, gp_setup, chunk, samples, seed, 0.01)), sigma, \
        f"{Path(run_dir).name} @ step {run['selected_step']} (posterior weight samples)"


def score(paths_fn, sigma_obs: dict, windows: np.ndarray, samples: int, batch: int) -> dict:
    residuals = {k: [] for k in OBS_NAMES}
    variances = {k: [] for k in OBS_NAMES}
    diverged = 0
    for begin in range(0, windows.shape[0], batch):
        chunk = windows[begin:begin + batch]
        paths = np.asarray(paths_fn(chunk, samples, 0), dtype=np.float64)[:, 1:]      # (S, 50, B, 22), h = 1..50
        truth = np.transpose(chunk, (1, 0, 2))[1:]
        with np.errstate(invalid="ignore", over="ignore"):
            ok = (np.isfinite(paths[..., :18]).all(axis=(0, 1, 3))
                  & (np.nan_to_num(np.linalg.norm(paths[..., :3] - truth[None, ..., :3], axis=-1), nan=np.inf).max(axis=(0, 1)) <= 10.0)
                  & (np.nan_to_num(np.linalg.norm(paths[..., 15:18], axis=-1), nan=np.inf).max(axis=(0, 1)) <= 100.0)
                  & (np.nan_to_num(np.linalg.norm(paths[..., 12:15], axis=-1), nan=np.inf).max(axis=(0, 1)) <= 50.0))
        diverged += int((~ok).sum())
        paths, truth = paths[:, :, ok], truth[:, ok]
        if truth.shape[1] == 0:
            continue
        mean = paths.mean(axis=0)
        for k, cols in BLOCKS.items():
            residuals[k].append((truth[..., cols] - mean[..., cols]).reshape(-1, 3))
            variances[k].append(paths[..., cols].var(axis=0, ddof=1).reshape(-1, 3))
        rot_mean = np.asarray(project_rotation(jnp.asarray(mean[..., 3:12])))
        rot_t = np.swapaxes(rot_mean, -1, -2)
        tangents = np.asarray(log_so3(jnp.asarray(rot_t[None] @ paths[..., 3:12].reshape(*paths.shape[:-1], 3, 3))))
        residuals["R"].append(np.asarray(log_so3(jnp.asarray(rot_t @ truth[..., 3:12].reshape(*truth.shape[:-1], 3, 3)))).reshape(-1, 3))
        variances["R"].append(((tangents - tangents.mean(axis=0)) ** 2).sum(axis=0).reshape(-1, 3) / (paths.shape[0] - 1))
        print(f"  {begin + chunk.shape[0]}/{windows.shape[0]} windows", end="\r", flush=True)
    out = {"diverged_windows": diverged, "windows": int(windows.shape[0])}
    for k in OBS_NAMES:
        res = np.concatenate(residuals[k]); var = np.concatenate(variances[k]) + sigma_obs[k] ** 2
        sigma = np.sqrt(var); z = np.abs(res) / sigma
        out[k] = {"coverage_1sigma": float(np.mean(z <= 1.0)), "coverage_2sigma": float(np.mean(z <= 2.0)),
                  "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * var) + res ** 2 / (2 * var))),
                  "sharpness": float(np.mean(np.linalg.norm(sigma, axis=-1))),
                  "spearman": float(spearmanr(np.linalg.norm(sigma, axis=-1), np.linalg.norm(res, axis=-1)).correlation),
                  "rms_error": float(np.sqrt(np.mean(np.sum(res ** 2, axis=-1))))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="LABEL=DIR[@STEP]")
    parser.add_argument("--source", type=Path, default=bp.PROJECT_ROOT / "envs/quadrotor_se3_idsia/idsia_raw/data/test")
    parser.add_argument("--pattern", default="melon*.csv")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()
    results = {}
    for spec in arguments.run:
        label, _, target = spec.partition("=")
        run_dir, _, step = target.partition("@")
        paths_fn, sigma_obs, description = sampler(run_dir, step)
        mode = bp.input_mode_of_run(F.evaluation.load_run(Path(run_dir), int(step) if step else None))
        windows = bp.rolling_windows(bp.load_melon(arguments.source, mode, arguments.pattern), arguments.horizon, arguments.stride)
        print(f"[calibration] {label}: {windows.shape[0]} windows, S = {arguments.samples}, sigma_obs = {sigma_obs}", flush=True)
        results[label] = {"source": description, "sigma_obs": sigma_obs, **score(paths_fn, sigma_obs, windows, arguments.samples, arguments.batch)}
        r = results[label]
        print(f"  diverged {r['diverged_windows']}/{r['windows']}; " + "; ".join(
            f"{k}: cov1 {r[k]['coverage_1sigma']:.3f} cov2 {r[k]['coverage_2sigma']:.3f} nlpd {r[k]['nlpd']:.2f} "
            f"sharp {r[k]['sharpness']:.4f} rho {r[k]['spearman']:.2f}" for k in OBS_NAMES), flush=True)
    folder = bp.PROJECT_ROOT / "experiments/quadrotor/eval_runs" / arguments.output_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "calibration_table.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(), "source": str(arguments.source), "pattern": arguments.pattern,
        "samples": arguments.samples, "stride": arguments.stride, "horizon": arguments.horizon, "results": results}, indent=2) + "\n")
    print(f"written {folder / 'calibration_table.json'}")


if __name__ == "__main__":
    main()

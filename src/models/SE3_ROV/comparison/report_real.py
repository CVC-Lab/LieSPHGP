"""Open-loop report on the REAL BlueROV2 data (KTH Marinarium), same layout as the pendulum / quadrotor campaign reports.

    python src/models/SE3_ROV/comparison/report_real.py [--gp-run <run dir>] [--phnode-run <run dir>]
                                                          [--phnode-checkpoint final | stepNNNNNN] [--samples 32]

Written to experiments/rov_se3/real_marinarium_05-10-2026/reports/open-loop-horizons-comparison.pdf (+ summary json).

Evaluation pieces: the 15 test trajectories of 10 s of datasets/ROV-MARINARIUM-DATASET-PAPER-MANUAL-COMMANDS (the last 20 %
of the manual recording, the paper's test split; cut between motion-capture dropouts and pose glitches; never used in
training). Training used the first 80 % only (the -5s dataset).

Protocol: every model starts from the MEASURED state at t = 0 and runs open loop over 10 s (500 intervals of 0.02 s)
with the recorded commands, no feedback; scored against the measured pieces over 0-1, 0-3, 0-5, 0-10 s. Real data has
no clean state, no recorded wind and no true Hamiltonian: there is no Analytical / -wind-same model and no energy row.

Models
    Lie-PH-GP-SDE   this package, 32 sample paths (posterior weight sample x Brownian path of the learned diffusion)
    PH-NODE         the latest finished *_PH-NODE-ORIG_REAL-MARINARIUM run (the original PH-NODE real-flight recipe:
                    9-step RK4 sequences, full batch, Adam 1e-3, float64, L1 on G / D_v / D_omega), one deterministic path;
                    '-' if no such run exists (the 2 s-window recipe produced NaN gradients from step 5)
    Koopman         EDMDc-RBF of the Marinarium paper (their code, Koopman/koopmanEDMDc.py, K = 500, gamma = 3,
                    lambda = 0.1), refitted on OUR 5 s training pieces with fit_multi (no cross-piece transitions):
                    state [x, y, z, phi, theta, psi, u, v, w, p, q, r] (ZYX Euler), the raw commands. No priors.
    Published physics  the BlueROV2 model (envs/rov_se3_marinarium/bluerov2.py) with the published von Benzon 2022 parameters,
                    the T200 thrust at the measured battery voltage and the PX4 thruster geometry: a PRIOR-BASED
                    reference, not ranked.
    Persistence     x(t) = x(0): the floor any model must beat, not ranked.

An extra page gives the paper's Table 2 protocol (CSV rows of the test split, open loop from every row, endpoint RMSE
over the 12-D Euler state at H = 1, 10, 100).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
MARINARIUM_RAW = PROJECT_ROOT / "envs" / "rov_se3_marinarium" / "marinarium_raw"
for _path in (PROJECT_ROOT, THIS_DIR.parent, PROJECT_ROOT / "envs" / "rov_se3_marinarium" / "analysis", MARINARIUM_RAW):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from Koopman.koopmanEDMDc import KoopmanEDMDc  # noqa: E402  (the Marinarium paper's code)
from envs.rov_se3_marinarium.bluerov2 import BlueROV2  # noqa: E402
from comparison.models import build_model as model_from_params, load_run, products, rollout_function  # noqa: E402
from sim_to_real_gap import CSV, PAPER_TABLE2, euler_continuous, rollout_ours  # noqa: E402

EXPERIMENT = PROJECT_ROOT / "experiments" / "rov_se3" / "real_marinarium_05-10-2026"
EVAL_COMMANDS = PROJECT_ROOT / "datasets/ROV-MARINARIUM-DATASET-PAPER-MANUAL-COMMANDS/PAPER-MANUAL-COMMANDS_clean.pkl"
EVAL_THRUST = PROJECT_ROOT / "datasets/ROV-MARINARIUM-DATASET-PAPER-MANUAL/PAPER-MANUAL_clean.pkl"
TRAIN_5S = PROJECT_ROOT / "datasets/ROV-MARINARIUM-DATASET-PAPER-MANUAL-COMMANDS-5s/PAPER-MANUAL-COMMANDS-5s_clean.pkl"
GAP_RESULTS = PROJECT_ROOT / "experiments/rov_se3/analysis/sim_to_real_gap/results.json"
KOOPMAN = {"n_rbfs": 500, "gamma": 3.0, "ridge": 0.1}

HORIZON_MARKS = (1.0, 3.0, 5.0, 10.0)
PAPER_HORIZONS = (1, 10, 100)
PAGE = (11.69, 8.27)
COLORS = {"Lie-PH-GP-SDE": "tab:orange", "PH-NODE": "tab:pink", "Koopman EDMDc-RBF": "tab:blue",
          "Published physics": "tab:green", "Persistence": "0.35"}
BAND_TEXT = {"sample_gp_sde": "32 paths: posterior weight sample x Brownian path of the learned diffusion",
             "point": "none: one deterministic path", "failed": "not trained (rollout loss NaN gradients) -> '-' everywhere",
             "point_phnode": "none: one deterministic path (original PH-NODE recipe, RK4)",
             "point_published": "none: deterministic, published parameters (prior-based reference)",
             "point_persistence": "none: x(t) = x(0)"}
BLOCKS = ("p", "R", "v", "w")
# Valid prediction time: first time the error of the prediction exceeds the threshold (= the horizon if it never does).
# Per block: 0.1 m (the "interval of valid prediction" convention cited by the Marinarium paper), 0.1 rad, 0.1 m/s,
# 0.1 rad/s. Normalised (Pathak et al. 2018): e_N(t) = sqrt(1/4 sum_b (e_b(t) / sigma_b)^2) > 0.4, sigma_b = the RMS
# spread of the measured data in block b (deviation from each piece's own mean), so every block weighs the same.
VPT_THRESHOLDS = {"p": 0.1, "R": 0.1, "v": 0.1, "w": 0.1}
VPT_NORMALISED = 0.4
SIGMA_TEXT: dict[str, str] = {}          # filled from the data by report(), printed under the tables
BLOCK_NAMES = {"p": "position error |p - p_meas| (m)", "R": "attitude error (rad)", "v": "body velocity error (m/s)",
               "w": "body angular-velocity error (rad/s)"}
UQ_NAMES = {"p": ("Position", "m"), "R": ("Attitude", "rad"), "v": ("Velocity", "m/s"), "w": ("Angular-velocity", "rad/s")}
TABLE_ROWS = tuple(row for block in BLOCKS for row in (
    (block, "mean", f"{UQ_NAMES[block][0]} mean error ({UQ_NAMES[block][1]})"),
    (block, "final", f"{UQ_NAMES[block][0]} final error ({UQ_NAMES[block][1]})"))) + tuple(
    (block, "vpt", f"VPT {UQ_NAMES[block][0].lower()} (s), error > {VPT_THRESHOLDS[block]:g} {UQ_NAMES[block][1]}")
    for block in BLOCKS) + (
    ("N", "vpt", f"VPT normalised (s), e_N > {VPT_NORMALISED:g}"),
    ("det", "max", "Determinant max error"), ("orth", "max", "Orthogonality max error")) + tuple(
    row for block in BLOCKS for row in (
        (block, "crps", f"{UQ_NAMES[block][0]} CRPS ({UQ_NAMES[block][1]})"),
        (block, "cover1", f"{UQ_NAMES[block][0]} coverage mean +- 1 sigma"),
        (block, "cover2", f"{UQ_NAMES[block][0]} coverage mean +- 2 sigma"),
        (block, "spread", f"{UQ_NAMES[block][0]} spread-skill RMSE / sigma")))
ROW_TARGETS = {"cover1": 0.683, "cover2": 0.954, "spread": 1.0}
HIGHER_IS_BETTER = {"vpt"}
# Per-piece tables (user, 7 Oct 2026): for a model with sample paths, every error / VPT / SO(3) cell is mean +- std over
# the first TABLE_PATHS sample paths (the ones drawn on the 3-D pages), each path scored against the measured piece alone.
TABLE_PATHS = 8
PATH_ROWS = ("mean", "final", "vpt", "max")
SO3_REFERENCE_ONLY = {"det", "orth"}          # Koopman (Euler angles) and Persistence are exact by construction: not ranked


@dataclass
class Entry:
    label: str
    kind: str                        # gp_sde | koopman | published | persistence | phnode
    band: str
    group: str                       # learned (ranked) | reference
    failed: bool = False
    run_dir: Path | None = None
    result: dict = field(default_factory=dict)
    checkpoint: str = "final"        # final | stepNNNNNN (PH-NODE: a run that aborted before its final step)


def entries(phnode_run: Path | None = None, phnode_checkpoint: str = "final") -> list[Entry]:
    band = "point_phnode" if phnode_checkpoint == "final" else "point_phnode_partial"
    phnode = (Entry("PH-NODE", "phnode", "failed", "learned", failed=True) if phnode_run is None
              else Entry("PH-NODE", "phnode", band, "learned", run_dir=phnode_run, checkpoint=phnode_checkpoint))
    return [Entry("Lie-PH-GP-SDE", "gp_sde", "sample_gp_sde", "learned"),
            phnode,
            Entry("Koopman EDMDc-RBF", "koopman", "point", "learned"),
            Entry("Published physics", "published", "point_published", "reference"),
            Entry("Persistence", "persistence", "point_persistence", "reference")]


# ---------------------------------------------------------------------------------------------------------------
# geometry and metrics (numpy, float64), as the campaign reports
# ---------------------------------------------------------------------------------------------------------------
def project_matrices(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    out = np.full(matrices.shape, np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        ok = np.all(np.isfinite(matrices), axis=(-2, -1)) & (np.max(np.abs(matrices), axis=(-2, -1)) < 1e100)
    if np.any(ok):
        u, _, vt = np.linalg.svd(matrices[ok])
        d = np.sign(np.linalg.det(u @ vt))
        u[..., :, -1] *= d[..., None]
        out[ok] = u @ vt
    return out


def project(rotation_flat: np.ndarray) -> np.ndarray:
    return project_matrices(rotation_flat.reshape(*rotation_flat.shape[:-1], 3, 3)).reshape(rotation_flat.shape)


def ensemble_mean(paths: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(np.isfinite(paths), paths, np.nan), axis=0)
    mean[..., 3:12] = project(np.nan_to_num(mean[..., 3:12]) + 1e-12 * np.eye(3).reshape(9))
    return mean


def geodesic_rad(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    a = reference.reshape(*reference.shape[:-1], 3, 3)
    b = prediction.reshape(*prediction.shape[:-1], 3, 3)
    with np.errstate(invalid="ignore"):
        cosine = np.clip((np.trace(np.swapaxes(a, -1, -2) @ b, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def tangent_rad(rotation_flat: np.ndarray, reference_flat: np.ndarray) -> np.ndarray:
    shape = rotation_flat.shape[:-1]
    a = np.broadcast_to(reference_flat, rotation_flat.shape).reshape(-1, 3, 3)
    b = rotation_flat.reshape(-1, 3, 3)
    with np.errstate(invalid="ignore", over="ignore"):
        relative = project_matrices(np.swapaxes(a, -1, -2) @ b)
    vectors = np.full((len(relative), 3), np.nan)
    ok = np.all(np.isfinite(relative), axis=(-2, -1))
    if np.any(ok):
        vectors[ok] = Rotation.from_matrix(relative[ok]).as_rotvec()
    return vectors.reshape(*shape, 3)


def euler_rad(rotation_flat: np.ndarray) -> np.ndarray:
    shape = rotation_flat.shape[:-1]
    matrices = project(rotation_flat.reshape(-1, 9)).reshape(-1, 3, 3)
    angles = np.full((len(matrices), 3), np.nan)
    ok = np.all(np.isfinite(matrices), axis=(-2, -1))
    if np.any(ok):
        angles[ok] = Rotation.from_matrix(matrices[ok]).as_euler("xyz")
    angles = angles.reshape(*shape, 3)
    return np.where(np.isfinite(angles), np.unwrap(np.nan_to_num(angles), axis=-2), np.nan)


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    count = samples.shape[0]
    x = np.sort(samples, axis=0)
    first = np.mean(np.abs(x - truth[None]), axis=0)
    weights = (2.0 * np.arange(1, count + 1) - count - 1).reshape((count,) + (1,) * (samples.ndim - 1))
    return first - np.sum(weights * x, axis=0) / (count * count)


def error_series(measured, centre) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore", over="ignore"):
        out = {"p": np.linalg.norm(centre[..., :3] - measured[..., :3], axis=-1),
               "R": geodesic_rad(measured[..., 3:12], centre[..., 3:12]),
               "v": np.linalg.norm(centre[..., 12:15] - measured[..., 12:15], axis=-1),
               "w": np.linalg.norm(centre[..., 15:18] - measured[..., 15:18], axis=-1)}
    bad = ~np.isfinite(centre[..., :18]).all(axis=-1)
    return {k: np.where(bad, np.inf, np.nan_to_num(v, nan=np.inf)) for k, v in out.items()}


def normalised_error(errors: dict[str, np.ndarray], sigma: dict[str, float]) -> np.ndarray:
    """e_N(t) = sqrt(1/4 sum_b (e_b(t) / sigma_b)^2) over position, attitude, velocity, angular velocity."""
    with np.errstate(invalid="ignore", over="ignore"):
        return np.sqrt(np.mean(np.stack([(errors[b] / sigma[b]) ** 2 for b in BLOCKS]), axis=0))


def valid_prediction_time(series: np.ndarray, threshold: float, count: int, interval: float) -> tuple[np.ndarray, np.ndarray]:
    """(per-trajectory VPT in s, censored flag): the first t_k in (0, t_count] with error > threshold; t_count if none."""
    exceeded = ~(series[1:count + 1] <= threshold)                      # non-finite errors count as exceeded
    crossed = np.any(exceeded, axis=0)
    first = np.argmax(exceeded, axis=0) + 1
    return np.where(crossed, first, count) * interval, ~crossed


def so3_violations(states: np.ndarray) -> dict[str, np.ndarray]:
    rotation = np.asarray(states[..., 3:12], dtype=np.float64).reshape(*states.shape[:-1], 3, 3)
    with np.errstate(invalid="ignore", over="ignore"):
        determinant = np.abs(np.linalg.det(np.nan_to_num(rotation, nan=1e6)) - 1.0)
        gram = np.swapaxes(rotation, -1, -2) @ rotation - np.eye(3)
        orthogonality = np.nan_to_num(np.sqrt(np.sum(gram * gram, axis=(-2, -1))), nan=np.inf)
    return {"det": np.where(np.isfinite(rotation).all(axis=(-2, -1)), determinant, np.inf), "orth": orthogonality}


def calibration(measured, paths, centre) -> dict:
    out = {}
    for block in BLOCKS:
        if block == "R":
            y = tangent_rad(measured[..., 3:12], centre[..., 3:12])
            x = None if paths is None else tangent_rad(paths[..., 3:12], np.broadcast_to(centre[..., 3:12], paths[..., 3:12].shape))
            c = np.zeros_like(y)
        else:
            cols = {"p": slice(0, 3), "v": slice(12, 15), "w": slice(15, 18)}[block]
            y, x, c = measured[..., cols], None if paths is None else paths[..., cols], centre[..., cols]
        if x is None:
            out[block] = {"crps": np.nan_to_num(np.mean(np.abs(c - y), axis=-1), nan=np.inf)}
            continue
        with np.errstate(invalid="ignore"):
            mu, sigma = np.nanmean(x, axis=0), np.nanstd(x, axis=0, ddof=1)
            z = np.abs(y - mu) / np.maximum(sigma, 1e-12)
            out[block] = {"crps": np.mean(crps_ensemble(np.nan_to_num(x, nan=1e6, posinf=1e6, neginf=-1e6), y), axis=-1),
                          "cover1": np.mean(z <= 1.0, axis=-1), "cover2": np.mean(z <= 2.0, axis=-1),
                          "sq_err": np.mean((y - mu) ** 2, axis=-1), "var": np.mean(sigma ** 2, axis=-1)}
    return out


def table_values(result: dict, marks, interval: float) -> dict:
    values = {}
    for mark in marks:
        count = int(round(mark / interval))
        window = slice(1, count + 1)
        row = {}
        for block, series in result["errors"].items():
            row[f"{block}_mean"] = series[window].mean(axis=0)
            row[f"{block}_final"] = series[count]
        for block in BLOCKS:
            row[f"{block}_vpt"], row[f"{block}_vpt_censored"] = valid_prediction_time(
                result["errors"][block], VPT_THRESHOLDS[block], count, interval)
        row["N_vpt"], row["N_vpt_censored"] = valid_prediction_time(result["normalised"], VPT_NORMALISED, count, interval)
        for key, series in result["so3"].items():
            worst = series[..., window, :].max(axis=-2)
            row[f"{key}_max"] = worst.mean(axis=0) if worst.ndim == 2 else worst
        for block, stats in result["calib"].items():
            row[f"{block}_crps"] = stats["crps"][window].mean(axis=0)
            if "cover1" in stats:
                row[f"{block}_cover1"] = stats["cover1"][window].mean(axis=0)
                row[f"{block}_cover2"] = stats["cover2"][window].mean(axis=0)
                row[f"{block}_spread"] = np.sqrt(stats["sq_err"][window].mean(axis=0)
                                                 / np.maximum(stats["var"][window].mean(axis=0), 1e-24))
            else:
                row[f"{block}_cover1"] = row[f"{block}_cover2"] = row[f"{block}_spread"] = None
        if "path_errors" in result:                # (paths, pieces) arrays, key suffix _paths (per-piece tables only)
            own = result["path_errors"]
            for block in own[0]:
                row[f"{block}_mean_paths"] = np.stack([e[block][window].mean(axis=0) for e in own])
                row[f"{block}_final_paths"] = np.stack([e[block][count] for e in own])
            for block, series, threshold in [(b, [e[b] for e in own], VPT_THRESHOLDS[b]) for b in BLOCKS] + [
                    ("N", result["path_normalised"], VPT_NORMALISED)]:
                pairs = [valid_prediction_time(x, threshold, count, interval) for x in series]
                row[f"{block}_vpt_paths"] = np.stack([vpt for vpt, _ in pairs])
                row[f"{block}_vpt_censored_paths"] = np.stack([flag for _, flag in pairs])
            for key, series in result["so3"].items():
                row[f"{key}_max_paths"] = series[:len(own), window, :].max(axis=-2)
        values[f"0-{mark:g}s"] = row
    return values


def plottable(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(values) & (np.abs(values) < 1e12), values, np.nan)


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
def to_euler_state(states: np.ndarray) -> np.ndarray:
    """(T, 18+) SE(3) rows -> (T, 12) [x, y, z, phi, theta, psi, u, v, w, p, q, r], ZYX Euler, yaw unwrapped."""
    zyx = Rotation.from_matrix(states[:, 3:12].reshape(-1, 3, 3)).as_euler("ZYX")[:, ::-1]
    zyx[:, 2] = np.unwrap(zyx[:, 2])
    return np.c_[states[:, :3], zyx, states[:, 12:18]]


def from_euler_state(euler: np.ndarray) -> np.ndarray:
    """(T, 12) Koopman state -> (T, 18) SE(3) rows."""
    rotation = Rotation.from_euler("ZYX", euler[:, 3:6][:, ::-1]).as_matrix().reshape(-1, 9)
    return np.c_[euler[:, :3], rotation, euler[:, 6:12]]


class Data:
    def __init__(self):
        with EVAL_COMMANDS.open("rb") as handle:
            commands = pickle.load(handle)
        with EVAL_THRUST.open("rb") as handle:
            thrust = pickle.load(handle)
        self.pieces = np.asarray(commands["test_trajectories"], np.float64)                 # (F, T, 26) measured
        self.thrust = np.asarray(thrust["test_trajectories"], np.float64)[..., 18:]          # (F, T, 8) T200 thrust (N)
        if not np.array_equal(self.pieces[..., :18], np.asarray(thrust["test_trajectories"], np.float64)[..., :18]):
            raise ValueError("the commands and thrust evaluation pieces differ")
        self.measured = np.swapaxes(self.pieces, 0, 1)                                       # (T, F, 26)
        self.interval = float(commands["settings"]["sample_dt"])
        self.t = np.arange(self.measured.shape[0]) * self.interval
        with TRAIN_5S.open("rb") as handle:
            train = pickle.load(handle)
        self.train_pieces = np.asarray(train["train_trajectories"], np.float64)
        self.train_test_count = int(np.asarray(train["test_trajectories"]).shape[0])
        # RMS spread of the measured data per block (deviation from each piece's own mean): the VPT normalisation
        centred = lambda block: self.pieces[..., block] - self.pieces[..., block].mean(axis=1, keepdims=True)
        mean_rotation = project(self.pieces[..., 3:12].mean(axis=1))                         # chordal mean per piece
        angle = geodesic_rad(np.broadcast_to(mean_rotation[:, None], self.pieces[..., 3:12].shape), self.pieces[..., 3:12])
        self.sigma = {"p": float(np.sqrt(np.mean(np.sum(centred(slice(0, 3)) ** 2, -1)))),
                      "R": float(np.sqrt(np.mean(angle ** 2))),
                      "v": float(np.sqrt(np.mean(np.sum(centred(slice(12, 15)) ** 2, -1)))),
                      "w": float(np.sqrt(np.mean(np.sum(centred(slice(15, 18)) ** 2, -1))))}


# ---------------------------------------------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------------------------------------------
def fit_koopman(data: Data) -> KoopmanEDMDc:
    model = KoopmanEDMDc(state_dim=12, input_dim=8, **KOOPMAN)
    states = [to_euler_state(piece) for piece in data.train_pieces]
    inputs = [np.r_[piece[1:, 18:], np.zeros((1, 8))] for piece in data.train_pieces]   # u of row k+1 drives k -> k+1
    model.fit_multi(states, inputs)
    return model


def latest_gp_run() -> Path:
    runs = sorted((EXPERIMENT / "runs").glob("*_Lie-PH-GP-SDE_REAL-MARINARIUM*"))
    finished = [r for r in runs if (r / "checkpoint_final.pkl").exists()]
    if not finished:
        raise FileNotFoundError("no finished Lie-PH-GP-SDE run")
    return finished[-1]


def latest_phnode_run(checkpoint: str = "final") -> Path | None:
    runs = sorted((EXPERIMENT / "runs").glob("*_PH-NODE-ORIG_REAL-MARINARIUM*"))
    finished = [r for r in runs if (r / f"checkpoint_{checkpoint}.pkl").exists()]
    return finished[-1] if finished else None


def predict(data: Data, items: list[Entry], gp_run: Path, samples: int, seed: int) -> dict:
    measured = data.measured
    steps = measured.shape[0] - 1
    context = {}
    for entry in items:
        if entry.failed:
            continue
        paths, centre = None, None
        if entry.kind == "gp_sde":
            config, setup, params, payload = load_run(gp_run, "final")
            scale = np.asarray(payload["control_scale"], np.float64)
            context["gp"] = (config, setup, params, scale)
            entry.run_dir = gp_run
            start = jnp.asarray(measured[0], jnp.float32)
            controls = jnp.asarray(np.swapaxes(measured[1:, :, 18:] / scale, 0, 1), jnp.float32)      # (F, T-1, 8)
            substeps = int(config["data"]["substeps"])

            @jax.jit
            def sampled(key):
                weight_key, noise_key = jax.random.split(key)
                model = model_from_params(params, setup, weight_key, config["model"])
                noise = jax.random.normal(noise_key, (start.shape[0], steps, substeps, 6), jnp.float32)
                return jax.vmap(lambda s, u, e: rollout_function(config)(model, s, u, data.interval, e))(start, controls, noise)

            raw = np.stack([np.asarray(sampled(jax.random.PRNGKey(seed + i))) for i in range(samples)])
            paths = np.swapaxes(raw, 1, 2).astype(np.float64)                                # (S, T, F, 26)
            centre = ensemble_mean(paths)
        elif entry.kind == "phnode":
            config, setup, params, payload = load_run(entry.run_dir, entry.checkpoint)
            context["phnode_checkpoint"] = entry.checkpoint
            scale = np.asarray(payload["control_scale"], np.float64)
            context["phnode"] = (config, setup, params, scale)
            model = model_from_params(params, setup, None, config["model"])
            start = jnp.asarray(measured[0], jnp.float32)
            controls = jnp.asarray(np.swapaxes(measured[1:, :, 18:] / scale, 0, 1), jnp.float32)      # (F, T-1, 8)
            noise = jnp.zeros((steps, int(config["data"]["substeps"]), 6), jnp.float32)
            raw = jax.jit(jax.vmap(lambda s, u: rollout_function(config)(model, s, u, data.interval, noise)))(start, controls)
            centre = np.swapaxes(np.asarray(raw), 0, 1).astype(np.float64)                     # (T, F, 26)
        elif entry.kind == "koopman":
            koopman = fit_koopman(data)
            context["koopman"] = koopman
            out = []
            for piece in data.pieces:
                euler = koopman.simulate(to_euler_state(piece)[0], piece[1:, 18:])
                out.append(np.c_[from_euler_state(euler), piece[:, 18:]])
            centre = np.swapaxes(np.stack(out), 0, 1)
        elif entry.kind == "published":
            env = BlueROV2()
            tau = np.einsum("ij,ftj->tfi", env.E, data.thrust[:, 1:])                         # (T-1, F, 6)
            first = measured[0]
            trajectory = rollout_ours(env, first[:, :3], first[:, 3:12].reshape(-1, 3, 3), first[:, 12:18], tau, data.interval)
            states = [first[:, :18]] + [np.c_[p, R.reshape(-1, 9), nu] for p, R, nu in trajectory]
            centre = np.concatenate([np.stack(states), measured[..., 18:]], axis=-1)
        elif entry.kind == "persistence":
            centre = np.repeat(measured[:1], measured.shape[0], axis=0)
        errors = error_series(measured, centre)
        entry.result = {"centre": centre, "paths": paths, "errors": errors, "normalised": normalised_error(errors, data.sigma),
                        "so3": so3_violations(paths if paths is not None else centre),
                        "calib": calibration(measured, paths, centre)}
        if paths is not None:                      # each of the first TABLE_PATHS paths scored on its own (per-piece tables)
            own = [error_series(measured, path) for path in paths[:TABLE_PATHS]]
            entry.result["path_errors"] = own
            entry.result["path_normalised"] = [normalised_error(e, data.sigma) for e in own]
        print(f"[report] {entry.label:20s} done", flush=True)
    return context


def paper_protocol(context: dict) -> dict:
    """Endpoint RMSE over the 12-D Euler state at H = 1, 10, 100 on the paper's CSV test rows (Table 2 protocol)."""
    df = pd.read_csv(CSV).sort_values("t").drop_duplicates(subset="t")
    X = df[["x", "y", "z", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r"]].to_numpy(float)
    U = np.nan_to_num(df[[f"u{i}" for i in range(1, 9)]].to_numpy(float))
    h = float(np.median(np.diff(df["t"].to_numpy(float))))
    split = int(0.8 * len(X))
    Xt, Ut = X[split:], U[split:]
    rows = {"Persistence": [float(np.sqrt(np.sum((Xt[:-H] - Xt[H:]) ** 2) / ((len(Xt) - H) * 12))) for H in PAPER_HORIZONS]}
    # the paper's Koopman, reproduced with their code and their split (fit on the first 80 % of the CSV rows)
    theirs = KoopmanEDMDc(state_dim=12, input_dim=8, **KOOPMAN)
    theirs.fit(X[:split], U[:split])
    rows["Koopman EDMDc-RBF (their code, their CSV split)"] = [float(theirs.multistep_rmse(X[split - 1:], U[split - 1:], H=H))
                                                               for H in PAPER_HORIZONS]
    gap = json.loads(GAP_RESULTS.read_text())["paper_protocol"]
    rows["Published physics (our simulator)"] = gap["ours_published"]
    rows["Fossen, their code (reproduced)"] = gap["their_fossen"]
    phnode_label = "PH-NODE (original recipe" + ("" if context.get("phnode_checkpoint", "final") == "final"
                                                  else f", checkpoint {context['phnode_checkpoint']}") + ")"
    for name, label in (("gp", "Lie-PH-GP-SDE (posterior-mean model, ours)"), ("phnode", phnode_label)):
        if name not in context:
            rows[label] = None
            continue
        config, setup, params, scale = context[name]
        model = model_from_params(params, setup, None, config["model"])
        starts = np.arange(len(Xt) - 1)
        rotation = Rotation.from_euler("ZYX", Xt[starts, 3:6][:, ::-1]).as_matrix().reshape(-1, 9)
        initial = np.c_[Xt[starts, :3], rotation, Xt[starts, 6:12], Ut[starts] / scale].astype(np.float32)
        steps = max(PAPER_HORIZONS)
        controls = np.stack([Ut[np.minimum(starts + s, len(Xt) - 1)] / scale for s in range(steps)], axis=1).astype(np.float32)
        noise = jnp.zeros((steps, int(config["data"]["substeps"]), 6), jnp.float32)
        roll = jax.jit(jax.vmap(lambda x0, u, advance=rollout_function(config): advance(model, x0, u, h, noise)))
        trajectory = np.concatenate([np.asarray(roll(jnp.asarray(initial[i:i + 1024]), jnp.asarray(controls[i:i + 1024])))
                                     for i in range(0, len(starts), 1024)], axis=0).astype(np.float64)      # (N, steps+1, 26)
        values = []
        for H in PAPER_HORIZONS:
            valid = starts[: len(Xt) - H]
            end = trajectory[: len(valid), H]
            ok = np.isfinite(end[:, :18]).all(axis=1)
            euler = np.full((len(valid), 3), 1e6)
            if np.any(ok):
                euler[ok] = euler_continuous(project_matrices(end[ok, 3:12].reshape(-1, 3, 3)), Xt[valid[ok] + H, 3:6])
            predicted = np.c_[end[:, :3], euler, end[:, 12:18]]
            error = np.where(np.isfinite(predicted), predicted - Xt[valid + H], 1e6)
            values.append(float(np.sqrt(np.sum(error ** 2) / ((len(Xt) - H) * 12))))
        rows[label] = values
    return {"rows": rows, "paper": PAPER_TABLE2, "test_rows": len(Xt), "sample_dt": h}


# ---------------------------------------------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------------------------------------------
def styled_table(axis, cells, header, widths, fontsize, scale, best=(), failed_cols=()):
    table = axis.table(cellText=cells, colLabels=header, colWidths=widths, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, scale)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d0d8e8")
            cell.set_height(cell.get_height() * 1.6)
        elif c == 0:
            cell.set_facecolor("#f2f2f2")
        elif c in failed_cols:
            cell.set_facecolor("#f6e0e0")
    for r, c in best:
        table[r, c].set_facecolor("#c8ecc8")
        table[r, c].set_text_props(weight="bold")
    return table


def published_products(states: np.ndarray) -> dict:
    env = BlueROV2()
    m_inv = env.M_inv
    nu = states[:, 12:18]
    damping = (env.D_L + env.D_Q * np.abs(nu)) * m_inv
    down = states[:, 3:12].reshape(-1, 3, 3)[:, 2, :]
    restoring = m_inv * np.c_[(env.p.weight - env.p.buoyancy) * down, np.cross(env.r_b, -env.p.buoyancy * down)]
    return {"damping_linear": np.mean(damping[:, :3], 0), "damping_angular": np.mean(damping[:, 3:], 0),
            "restoring_rms": np.sqrt(np.mean(restoring ** 2, 0)), "anisotropy": m_inv[:3] / np.mean(m_inv[:3])}


def physics_page(data: Data, items: list[Entry], context: dict) -> plt.Figure:
    figure = plt.figure(figsize=PAGE)
    figure.text(0.03, 0.955, "Physics Identification", fontsize=17, weight="bold")
    figure.text(0.33, 0.957, "BlueROV2 Heavy, REAL data (KTH Marinarium, manual recording)", fontsize=12)
    spec = ["dataset     training: ROV-MARINARIUM-DATASET-PAPER-MANUAL-COMMANDS-5s (first 80 % of the manual recording, paper's "
            "chronological split)",
            f"            evaluation: ROV-MARINARIUM-DATASET-PAPER-MANUAL-COMMANDS, {data.pieces.shape[0]} test pieces of "
            f"{data.t[-1]:g} s (last 20 %)",
            "vehicle     BlueROV2 Heavy, 8 T200 thrusters; world z down, body forward-right-down; Qualisys motion capture (pose, "
            "body twist), 50 Hz",
            "control     the 8 raw PX4 motor commands in [-1, 1] (no thrust map, no thruster geometry); models see u / training RMS",
            "noise       real measurement noise (unknown): the EKF learns sigma_obs per block; motion-capture dropouts and pose "
            "glitches are cut out",
            f"sampling    dt = {data.interval:g} s; training pieces 5 s",
            f"training    {data.train_pieces.shape[0]} pieces of 5 s, 2 s windows, 25 % overlap, batch 16, 5000 steps, final model; "
            "float32; no priors, no pretraining",
            f"evaluation  open loop from the MEASURED state at t = 0 for {data.t[-1]:g} s with the recorded commands"]
    figure.text(0.03, 0.925, "\n".join(spec), fontsize=7.2, va="top", family="monospace")
    header = ["model", "M^-1 G row norm\n(surge, sway, heave,\nroll, pitch, yaw)", "M1^-1 D_v diag\n(mean, x / y / z)",
              "M2^-1 D_w diag\n(mean, x / y / z)", "restoring accel. rms\n(x, y, z | roll, pitch, yaw)",
              "M1^-1 anisotropy\n(eig / mean)", "Sigma\n(lin / ang mean)", "sigma_obs\n(p / R / v / w)",
              "test EKF\nNLL / d", "NIS / d"]
    triple = lambda v: " / ".join(f"{x:.3g}" for x in v)
    states = data.pieces.reshape(-1, data.pieces.shape[-1])[::5]
    published = published_products(states)
    rows = [["Published nominal\n(von Benzon 2022;\nNOT ground truth)", "- (thrust units)",
             triple(published["damping_linear"]), triple(published["damping_angular"]),
             triple(published["restoring_rms"][:3]) + "\n" + triple(published["restoring_rms"][3:]),
             triple(published["anisotropy"]), "-", "-", "-", "-"]]
    failed_rows = []
    for entry in items:
        if entry.group != "learned":
            continue
        context_key = {"gp_sde": "gp", "phnode": "phnode"}.get(entry.kind)
        if context_key in context:
            config, setup, params, scale = context[context_key]
            model = model_from_params(params, setup, None, config["model"])
            learned = products(model, (states / np.r_[np.ones(18), scale]).astype(np.float32), scale)
            evaluation_path = entry.run_dir / "evaluation_final.json"
            evaluation = json.loads(evaluation_path.read_text())["learned"] if evaluation_path.exists() else {}
            norms = np.mean(np.linalg.norm(learned["control"], axis=-1), 0)
            sigma, obs = evaluation.get("diffusion"), evaluation.get("sigma_obs")
            rows.append([entry.label, triple(norms[:3]) + "\n" + triple(norms[3:]),
                         triple(np.diagonal(np.mean(learned["damping_linear"], 0))),
                         triple(np.diagonal(np.mean(learned["damping_angular"], 0))),
                         triple(np.sqrt(np.mean(learned["restoring"] ** 2, 0))[:3]) + "\n"
                         + triple(np.sqrt(np.mean(learned["restoring"] ** 2, 0))[3:]),
                         triple(np.mean(learned["anisotropy"], 0)),
                         "-" if not sigma else f"{np.mean(sigma[:3]):.3g} /\n{np.mean(sigma[3:]):.3g}",
                         "-" if not obs else f"{obs[0]:.3g} / {obs[1]:.3g}\n{obs[2]:.3g} / {obs[3]:.3g}",
                         f"{evaluation['test_ekf_nll']:.4f}" if "test_ekf_nll" in evaluation else "-",
                         f"{evaluation['test_nis_per_dimension']:.3f}" if "test_nis_per_dimension" in evaluation else "-"])
        elif entry.kind == "koopman":
            rows.append([f"{entry.label}\n(no physical operators)"] + ["-"] * (len(header) - 1))
        else:
            failed_rows.append(len(rows))
            rows.append([f"{entry.label}\n(not trained)"] + ["-"] * (len(header) - 1))
    axis = figure.add_axes([0.005, 0.18, 0.99, 0.52])
    axis.axis("off")
    widths = [0.12, 0.12, 0.1, 0.1, 0.14, 0.09, 0.08, 0.1, 0.07, 0.06]
    table = styled_table(axis, rows, header, widths, 6.4, 4.2)
    for c in range(len(header)):
        table[1, c].set_facecolor("#e8e8e8")
    for r in failed_rows:
        for c in range(1, len(header)):
            table[r + 1, c].set_facecolor("#f6e0e0")
    notes = ("Real data: there is NO ground truth, so nothing is ranked here. The first row evaluates the published von Benzon 2022 "
             "model on the same states for orientation only: the sim-to-real test (envs/rov_se3_marinarium/analysis/sim_to_real_gap.py) "
             "showed that the real vehicle needs 0.25-0.6x its thrust and drag in surge / sway / heave / yaw. Only gauge-invariant products "
             "are identifiable ((M^-1, V, D, G) -> (c M^-1, V / c, D / c, G / c)). Values: posterior-mean model, mean over the states of the "
             "evaluation pieces (every 5th sample). M^-1 G row norm = acceleration per unit raw command (|row| over the 8 commands); "
             "M^-1 D_v, M^-1 D_w = effective damping rates (1/s) at the measured twist (published: (D_L + D_Q |nu|) / M per axis); "
             "restoring = -M^-1 grad V (gravity and buoyancy, generalised); anisotropy = eigenvalues of M1^-1 over their mean (published "
             "from the added mass, 1 = isotropic). Sigma = learned twist diffusion, sigma_obs = learned observation noise. EKF NLL / d "
             "and NIS / d: evaluate.py on the 5 s test pieces (2 s windows).")
    figure.text(0.03, 0.02, "\n".join(textwrap.wrap(notes, 205)), fontsize=6.4)
    return figure


def protocol_page(data: Data, items: list[Entry], samples: int) -> plt.Figure:
    lines = ["dataset   evaluation: ROV-MARINARIUM-DATASET-PAPER-MANUAL-COMMANDS (test split = last 20 % of the manual recording)",
             f"pieces    {data.pieces.shape[0]} measured pieces of {data.t[-1]:g} s ({data.measured.shape[0]} samples at {data.interval:g} s), "
             "cut between dropouts and glitches; never used in training",
             f"protocol  ONE nonstop open-loop rollout per piece from the MEASURED state at t = 0 to {data.t[-1]:g} s with the",
             "          recorded commands (scaled by the training RMS for Lie-PH-GP-SDE), no feedback. Scored against the measured",
             "          piece over 0-1, 0-3, 0-5, 0-10 s. Nothing is filtered: a diverged rollout keeps its (large or infinite) error.",
             "reference the data are measurements (mm-level motion-capture noise; the angular rates are noisier), not a clean state.",
             "          No true Hamiltonian and no recorded wind exist, so there is no energy row and no -wind-same / Analytical model.",
             f"line/band {samples} sample paths: line = their mean (mean rotation projected onto SO(3)), dark = +- 1 sigma, light = +- 2 sigma.",
             "VPT       valid prediction time = first t at which the prediction's error exceeds a threshold (= H if it never does",
             "          within the horizon, shown '>= H'): per block 0.1 m, 0.1 rad, 0.1 m/s, 0.1 rad/s; normalised e_N(t) =",
             "          sqrt(1/4 sum_b (e_b(t) / sigma_b)^2) > 0.4 (Pathak et al. 2018), sigma_b = RMS spread of the measured data:",
             "          " + ", ".join(f"{b} {SIGMA_TEXT[b]}" for b in BLOCKS) + ". All angles in rad.",
             "", "models"]
    for e in items:
        lines.append(f"  {e.label:20s} band: {BAND_TEXT[e.band]}")
    lines += ["", "  Koopman EDMDc-RBF  = the Marinarium paper's model with their code and hyperparameters (K = 500, gamma = 3,",
              "                       lambda = 0.1), refitted on OUR 5 s training pieces (fit_multi, no cross-piece transitions);",
              "                       state = [x, y, z, phi, theta, psi (ZYX, yaw unwrapped per piece), u, v, w, p, q, r], raw commands.",
              "  Published physics  = envs/rov_se3_marinarium/bluerov2.py with the published parameters, T200 thrust at the measured voltage",
              "                       (inputs of ROV-MARINARIUM-DATASET-PAPER-MANUAL, same pieces), Lie-group Heun, 10 substeps.",
              "", "how to read",
              "  Green bold = best per row among the learned models (Lie-PH-GP-SDE, PH-NODE if trained, Koopman).",
              "  Published physics (prior-based) and Persistence are references and are not ranked. Coverage / spread-skill rows",
              "  exist only for the model with a band (Lie-PH-GP-SDE)."]
    figure = plt.figure(figsize=PAGE)
    figure.text(0.04, 0.95, "Open-loop prediction on the real test pieces: protocol and models", fontsize=15, weight="bold")
    figure.text(0.04, 0.90, "\n".join(lines), fontsize=8.2, va="top", family="monospace")
    return figure


def paper_page(protocol: dict) -> plt.Figure:
    figure = plt.figure(figsize=PAGE)
    figure.text(0.03, 0.955, "Marinarium paper protocol (Torroba et al. 2026, Table 2)", fontsize=15, weight="bold")
    lines = [f"CSV rows of the manual recording (koopman_dataset_50Hz.csv), chronological 80/20, test rows N = {protocol['test_rows']}, "
             f"dt = {protocol['sample_dt']:.3g} s;",
             "open loop from EVERY test row with the recorded commands (row k drives k -> k+1, as their evaluator), endpoint RMSE",
             "RMSE_H = sqrt( sum_k ||x_{k+H} - x_hat_{k+H|k}||^2 / ((N - H) 12) ) over [x, y, z, phi, theta, psi, u, v, w, p, q, r], "
             "H = 1, 10, 100 (0.02, 0.2, 2 s).",
             "Note: the CSV interpolates across motion-capture dropouts (our training pieces exclude them); at H = 1, 98 % of the "
             "persistence error is p, q, r noise."]
    figure.text(0.03, 0.92, "\n".join(lines), fontsize=7.5, va="top", family="monospace")
    cells, kinds = [], []
    for name, values in protocol["paper"].items():
        cells.append([f"{name} (paper, as printed)"] + [f"{v:.4f}" for v in values])
        kinds.append("paper")
    for name, values in protocol["rows"].items():
        cells.append([name] + (["-"] * 3 if values is None else [f"{v:.4f}" for v in values]))
        kinds.append("ours")
    axis = figure.add_axes([0.08, 0.15, 0.84, 0.62])
    axis.axis("off")
    table = styled_table(axis, cells, ["model", "H = 1", "H = 10", "H = 100"], [0.52, 0.16, 0.16, 0.16], 8.5, 1.9)
    learned = [i for i, (row, kind) in enumerate(zip(cells, kinds)) if kind == "ours" and row[1] != "-" and (
        row[0].startswith("Lie-PH") or row[0].startswith("Koopman") or row[0].startswith("PH-NODE"))]
    for c in (1, 2, 3):
        if learned:
            best = min(learned, key=lambda i: float(cells[i][c]))
            table[best + 1, c].set_facecolor("#c8ecc8")
            table[best + 1, c].set_text_props(weight="bold")
    for i, row in enumerate(cells):
        if row[1] == "-":
            for c in range(1, 4):
                table[i + 1, c].set_facecolor("#f6e0e0")
    figure.text(0.03, 0.05, "Green bold = best among the prior-free learned models computed here (Koopman with their code, Lie-PH-GP-SDE, PH-NODE).\n"
                "Rows marked 'paper' are copied from the paper; the others are computed here on the same rows. Published physics and Fossen "
                "use prior parameters.\nTheir Koopman code with the stated K = 500, gamma = 3, lambda = 0.1 on their split does not reproduce the "
                "printed Koopman row.", fontsize=7)
    return figure


def fmt(value, block: str, censored=None) -> str:
    if value is None:
        return "-"
    if censored is not None:                                     # VPT: censored = never crossed within the horizon
        if isinstance(value, tuple):
            text = f"{value[0]:.3g} +- {value[1]:.2g}"
            return text + (f" ({int(censored)} >= H)" if censored else "")
        return (">= " if censored else "") + f"{value:.3g}"
    if isinstance(value, tuple):
        mean, std = value
        return f"{mean:.2e} +- {std:.1e}" if block in ("det", "orth") else f"{mean:.4g} +- {std:.2g}"
    return f"{value:.2e}" if block in ("det", "orth") else f"{value:.4g}"


def table_page(items, tables, key, title, subtitle, piece: int | None) -> plt.Figure:
    figure = plt.figure(figsize=PAGE)
    figure.text(0.03, 0.96, title, fontsize=14, weight="bold")
    figure.text(0.03, 0.935, subtitle, fontsize=7.2)
    header = ["metric"] + [e.label.replace("Lie-PH-", "Lie-PH-\n").replace("Koopman ", "Koopman\n") for e in items]
    cells, best = [], []
    for r, (block, kind, label) in enumerate(TABLE_ROWS):
        raw, censor = [], []
        for e in items:
            v = None if e.failed else tables[e.label][key].get(f"{block}_{kind}")
            c = None if (e.failed or kind != "vpt") else tables[e.label][key][f"{block}_vpt_censored"]
            spread = None if (e.failed or piece is None) else tables[e.label][key].get(f"{block}_{kind}_paths")
            if spread is not None:                 # per piece, sample paths: mean +- std over the TABLE_PATHS paths
                v = (float(np.mean(spread[:, piece])), float(np.std(spread[:, piece])))
                if c is not None:
                    c = int(np.sum(tables[e.label][key][f"{block}_vpt_censored_paths"][:, piece]))
            elif v is not None:
                v = (float(np.mean(v)), float(np.std(v))) if piece is None else float(v[piece])
                if c is not None:
                    c = int(np.sum(c)) if piece is None else bool(c[piece])
            raw.append(v)
            censor.append(c)
        target = ROW_TARGETS.get(kind)
        value = lambda i: raw[i][0] if isinstance(raw[i], tuple) else raw[i]
        score = lambda i: (abs(value(i) - target) if target is not None
                           else -value(i) if kind in HIGHER_IS_BETTER else value(i))
        ranked = [i for i, e in enumerate(items) if e.group == "learned" and raw[i] is not None and np.isfinite(score(i))
                  and not (block in SO3_REFERENCE_ONLY and e.kind == "koopman")]
        if len(ranked) > 1:
            best.append((r + 1, 1 + min(ranked, key=score)))
        cells.append([label] + [fmt(v, block, c) for v, c in zip(raw, censor)])
    axis = figure.add_axes([0.01, 0.05, 0.98, 0.87])
    axis.axis("off")
    widths = [0.25] + [0.75 / len(items)] * len(items)
    styled_table(axis, cells, header, widths, 6.4, 1.06, best, failed_cols=[1 + i for i, e in enumerate(items) if e.failed])
    figure.text(0.03, 0.012, table_footnote(piece is not None), fontsize=5.6)
    return figure


def table_footnote(per_piece: bool = False) -> str:
    text = ("Green bold = best per row among the learned models (Lie-PH-GP-SDE, PH-NODE, Koopman); Published physics and Persistence "
                "are references, not ranked. Best = lowest, except coverage (closest to 0.683 / 0.954) and spread-skill (closest to 1). CRPS "
                "of a single path = its absolute error; '-' = no samples, or not trained (red column).\nMean error = time average over "
                "(0, H]; final = at H; SO(3) rows = max over (0, H] on the RAW rollout. Uncertainty rows: time average over (0, H]. "
                "Errors are against the MEASURED states; angles in rad.\nVPT = first time the error exceeds the threshold (higher = "
                "better); '>= H' = never within the horizon (H counted). Normalised: e_N = sqrt(1/4 sum_b (e_b / sigma_b)^2), "
                "sigma_b = RMS spread of the measured data: " + ", ".join(f"{b} {SIGMA_TEXT.get(b, '')}" for b in BLOCKS) + ". "
                "SO(3) rows rank only the Lie-group models: Koopman predicts Euler angles, which are orthonormal by construction "
                "once converted to R.")
    if per_piece:
        text += (f" THIS PIECE: for Lie-PH-GP-SDE every error, VPT and SO(3) cell = mean +- std over the {TABLE_PATHS} sample paths "
                 "drawn on the 3-D page, each path scored against the measured piece on its own (so the mean is the error of a typical "
                 "path, not of the mean prediction as in Table A; VPT '(k >= H)' = k of those paths never crossed). Point models: one "
                 "path, no std. CRPS / coverage / spread-skill use all sample paths.")
    return "\n".join(textwrap.wrap(text, 250))


def error_page(t, items, marks) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=PAGE)
    axes = axes.ravel()
    for axis, key in zip(axes, BLOCKS):
        for e in items:
            if e.failed:
                continue
            series = e.result["errors"][key]
            reference = e.group == "reference"
            axis.plot(t[1:], plottable(np.median(series, axis=1))[1:], color=COLORS[e.label], lw=1.5 if reference else 2,
                      ls="--" if reference else "-", label=e.label)
            if not reference:
                axis.fill_between(t[1:], plottable(np.quantile(series, 0.25, axis=1))[1:],
                                  plottable(np.quantile(series, 0.75, axis=1))[1:], color=COLORS[e.label], alpha=0.10, lw=0)
        for mark in marks[:-1]:
            axis.axvline(mark, color="0.5", lw=0.8, ls="--")
        axis.set_yscale("log")
        axis.set_title(BLOCK_NAMES[key], fontsize=8.5)
        axis.set_xlabel("t (s)")
        axis.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    figure.suptitle("Open-loop error vs time on the real test pieces: median over pieces, shade = inter-quartile, dashed = references",
                    fontsize=10)
    figure.tight_layout()
    return figure


def bar_page(items, tables, marks) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=PAGE)
    axes = axes.ravel()
    alive = [e for e in items if not e.failed]
    width = 0.8 / max(len(alive), 1)
    for axis, key in zip(axes, BLOCKS):
        for k, e in enumerate(alive):
            values = plottable([np.median(tables[e.label][f"0-{m:g}s"][f"{key}_mean"]) for m in marks])
            axis.bar(np.arange(len(marks)) + (k - (len(alive) - 1) / 2) * width, values, width, color=COLORS[e.label],
                     label=e.label, edgecolor="k", lw=0.3, hatch="//" if e.group == "reference" else None)
        axis.set_xticks(range(len(marks)))
        axis.set_xticklabels([f"0-{m:g} s" for m in marks])
        axis.set_yscale("log")
        axis.set_title("time-averaged " + BLOCK_NAMES[key], fontsize=8)
        axis.grid(alpha=0.3, axis="y", which="both")
    axes[0].legend(fontsize=7)
    figure.suptitle("Time-averaged error over " + " / ".join(f"0-{m:g}" for m in marks) + " s (median over test pieces; hatched = references)",
                    fontsize=10)
    figure.tight_layout()
    return figure


def calibration_page(t, items, marks) -> plt.Figure:
    figure, axes = plt.subplots(4, 4, figsize=(15.0, PAGE[1]), sharex=True)
    names = {"p": "position", "R": "attitude (tangent)", "v": "body velocity", "w": "angular velocity"}
    for col, key in enumerate(BLOCKS):
        for e in items:
            if e.failed:
                continue
            stats = e.result["calib"][key]
            style = dict(color=COLORS[e.label], lw=1.4, ls="--" if e.group == "reference" else "-", label=e.label)
            if "cover1" in stats:
                axes[0, col].plot(t, stats["cover1"].mean(axis=1), **style)
                axes[1, col].plot(t, stats["cover2"].mean(axis=1), **style)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.sqrt(stats["sq_err"].mean(axis=1) / np.maximum(stats["var"].mean(axis=1), 1e-24))
                axes[2, col].plot(t[1:], plottable(ratio[1:]), **style)
            axes[3, col].plot(t[1:], plottable(stats["crps"].mean(axis=1)[1:]), **style)
        axes[0, col].axhline(0.683, color="k", lw=0.8, ls=":")
        axes[1, col].axhline(0.954, color="k", lw=0.8, ls=":")
        axes[2, col].axhline(1.0, color="k", lw=0.8, ls=":")
        axes[0, col].set_ylim(0, 1.02)
        axes[1, col].set_ylim(0, 1.02)
        axes[2, col].set_yscale("log")
        axes[3, col].set_yscale("log")
        for row, what in enumerate(("coverage +- 1 sigma", "coverage +- 2 sigma", "spread-skill RMSE / sigma", "CRPS")):
            axes[row, col].set_title(f"{names[key]}: {what}", fontsize=7.5)
            axes[row, col].grid(alpha=0.3, which="both")
            for mark in marks[:-1]:
                axes[row, col].axvline(mark, color="0.5", lw=0.6, ls="--")
        axes[3, col].set_xlabel("t (s)")
    axes[3, 0].legend(fontsize=5.5)
    figure.suptitle("Calibration of the sample paths vs time (mean over test pieces and axes). Dotted = ideal 0.683, 0.954, 1. "
                    "Spread-skill > 1 = over-confident. CRPS of a single path = its absolute error.", fontsize=9)
    figure.tight_layout()
    return figure


def path3d_page(data, items, index) -> plt.Figure:
    columns = 3
    figure = plt.figure(figsize=(13.5, PAGE[1]))
    piece = data.pieces[index]
    lo, hi = piece[:, :3].min(axis=0) - 0.5, piece[:, :3].max(axis=0) + 0.5
    for k, e in enumerate(items):
        axis = figure.add_subplot(math.ceil(len(items) / columns), columns, k + 1, projection="3d")
        axis.plot(*piece[:, :3].T, color="k", lw=1.2)
        if e.failed:
            axis.set_title(f"{e.label}\nnot trained", fontsize=8, color="0.4")
        else:
            if e.result["paths"] is not None:
                for path in e.result["paths"][:TABLE_PATHS, :, index]:
                    axis.plot(*plottable(path[:, :3]).T, color=COLORS[e.label], lw=0.4, alpha=0.35)
            centre = e.result["centre"][:, index]
            axis.plot(*plottable(centre[:, :3]).T, color=COLORS[e.label], lw=1.5)
            axis.scatter(*piece[0, :3], color="k", s=8)
            axis.set_title(e.label, fontsize=8, color=COLORS[e.label], weight="bold")
        axis.set_xlim(lo[0], hi[0])
        axis.set_ylim(lo[1], hi[1])
        axis.set_zlim(hi[2], lo[2])                                # z down: depth increases downwards
        axis.tick_params(labelsize=5)
    figure.suptitle(f"Test piece {index}: 3-D path, z (depth) axis pointing down (black = measured, bold = prediction from the measured "
                    "x_0, thin = 8 sample paths)", fontsize=10)
    figure.tight_layout()
    return figure


def block_page(data, items, index, block, marks) -> plt.Figure:
    names = {"p": (["x (m)", "y (m)", "z (m, down)"], "Position (world)"),
             "euler": (["roll (rad)", "pitch (rad)", "yaw (rad)"], "Attitude (Euler xyz of R, unwrapped)"),
             "v": (["v_x (m/s)", "v_y (m/s)", "v_z (m/s)"], "Body linear velocity"),
             "w": (["w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"], "Body angular velocity")}
    labels, title = names[block]

    def values(states):
        if block == "euler":
            return euler_rad(states[..., 3:12])
        return states[..., {"p": slice(0, 3), "v": slice(12, 15), "w": slice(15, 18)}[block]]

    figure, axes = plt.subplots(3, len(items), figsize=(max(PAGE[0], 2.4 * len(items)), PAGE[1]), sharex=True, squeeze=False)
    t = data.t
    truth = values(data.pieces[index])
    for col, e in enumerate(items):
        for row in range(3):
            axis = axes[row, col]
            axis.plot(t, truth[:, row], color="k", lw=0.9)
            low, high = np.nanmin(truth[:, row]), np.nanmax(truth[:, row])
            reach = 3.0 * max(high - low, 1e-3)
            extent = [low, high]
            if e.failed:
                if row == 1:
                    axis.text(0.5, 0.5, "not trained", transform=axis.transAxes, ha="center", color="0.4")
            else:
                paths = e.result["paths"]
                if paths is not None:
                    sample_values = values(paths[:, :, index])[..., row]
                    if block == "euler":
                        sample_values = sample_values - 2.0 * np.pi * np.round((sample_values[:, :1] - truth[0, row]) / (2.0 * np.pi))
                    with np.errstate(invalid="ignore"):
                        mu, sd = np.nanmean(sample_values, axis=0), np.nanstd(sample_values, axis=0, ddof=1)
                    axis.fill_between(t, plottable(mu - 2 * sd), plottable(mu + 2 * sd), color=COLORS[e.label], alpha=0.15, lw=0)
                    axis.fill_between(t, plottable(mu - sd), plottable(mu + sd), color=COLORS[e.label], alpha=0.30, lw=0)
                    line = mu
                else:
                    line = values(e.result["centre"][:, index])[:, row]
                    if block == "euler":
                        line = line - 2.0 * np.pi * np.round((line[0] - truth[0, row]) / (2.0 * np.pi))
                axis.plot(t, plottable(line), color=COLORS[e.label], lw=1.2)
                finite = line[np.isfinite(line)]
                if finite.size:
                    extent = [min(extent[0], finite.min()), max(extent[1], finite.max())]
            clipped = extent[0] < low - reach or extent[1] > high + reach
            extent = [max(extent[0], low - reach), min(extent[1], high + reach)]
            pad = 0.05 * max(extent[1] - extent[0], 1e-3)
            axis.set_ylim(extent[0] - pad, extent[1] + pad)
            if clipped:
                axis.text(0.98, 0.97, "clipped", transform=axis.transAxes, ha="right", va="top", fontsize=6, color="0.3")
            for mark in marks[:-1]:
                axis.axvline(mark, color="0.5", lw=0.5, ls="--")
            axis.grid(alpha=0.25)
            axis.tick_params(labelsize=6)
            if col == 0:
                axis.set_ylabel(labels[row], fontsize=7)
            if row == 0:
                axis.set_title(e.label.replace("Lie-PH-", "Lie-PH-\n"), color=COLORS[e.label], weight="bold", fontsize=7)
            if row == 2:
                axis.set_xlabel("t (s)", fontsize=7)
    figure.suptitle(f"Test piece {index}: {title}. Black = measured; line = mean of the sample paths (single path if none), "
                    "dark = +- 1 sigma, light = +- 2 sigma;\ndashed = 1 / 3 / 5 s; 'clipped' = y-axis capped at the data range +- 3x its width",
                    fontsize=9)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------
def abort_note(run: Path) -> str:
    """'aborted at step N (non-finite loss/gradient)' from a run's train_log.txt, or '' if it did not abort."""
    log = run / "train_log.txt"
    found = re.findall(r"step (\d+): non-finite loss/gradient", log.read_text()) if log.exists() else []
    return f"run aborted at step {found[-1]} (non-finite loss/gradient)" if found else ""


def report(gp_run: Path, samples: int, seed: int, output: Path, phnode_run: Path | None = None,
           phnode_checkpoint: str = "final") -> Path:
    data = Data()
    SIGMA_TEXT.update({"p": f"{data.sigma['p']:.3g} m", "R": f"{data.sigma['R']:.3g} rad", "v": f"{data.sigma['v']:.3g} m/s",
                       "w": f"{data.sigma['w']:.3g} rad/s"})
    items = entries(phnode_run, phnode_checkpoint)
    if phnode_run is not None and phnode_checkpoint != "final":
        BAND_TEXT["point_phnode_partial"] = (f"none: one path (original recipe, RK4); {abort_note(phnode_run) or 'run not finished'}"
                                             f" -> checkpoint {phnode_checkpoint}")
    print(f"[report] {data.pieces.shape[0]} test pieces of {data.t[-1]:g} s; GP-SDE run {gp_run}; PH-NODE run {phnode_run}", flush=True)
    context = predict(data, items, gp_run, samples, seed)
    protocol = paper_protocol(context)
    print("[report] paper protocol done", flush=True)
    t = data.t
    marks = [m for m in HORIZON_MARKS if m <= t[-1] + 1e-9]
    tables = {e.label: table_values(e.result, marks, data.interval) for e in items if not e.failed}
    count = data.pieces.shape[0]
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "open-loop-horizons-comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(physics_page(data, items, context)); plt.close("all")
        pdf.savefig(protocol_page(data, items, samples)); plt.close("all")
        pdf.savefig(paper_page(protocol)); plt.close("all")
        for mark in marks:
            key = f"0-{mark:g}s"
            pdf.savefig(table_page(items, tables, key, f"Table A - open-loop prediction error on the real test pieces, 0-{mark:g} s",
                                   f"ROV-MARINARIUM-DATASET-PAPER-MANUAL-COMMANDS: mean +- std over {count} test pieces, every model from the "
                                   "measured state at t = 0. Prediction = mean of the sample paths (single path for point models).", None))
            plt.close("all")
        pdf.savefig(error_page(t, items, marks)); plt.close("all")
        pdf.savefig(bar_page(items, tables, marks)); plt.close("all")
        pdf.savefig(calibration_page(t, items, marks)); plt.close("all")
        for index in range(count):
            for mark in marks:
                key = f"0-{mark:g}s"
                pdf.savefig(table_page(items, tables, key, f"Test piece {index} - open-loop prediction error, 0-{mark:g} s",
                                       f"ROV-MARINARIUM-DATASET-PAPER-MANUAL-COMMANDS, test piece {index} alone.", index)); plt.close("all")
            pdf.savefig(path3d_page(data, items, index)); plt.close("all")
            for block in ("p", "euler", "v", "w"):
                pdf.savefig(block_page(data, items, index, block, marks)); plt.close("all")
    summary = {"generated_at": datetime.now().astimezone().isoformat(), "evaluation": str(EVAL_COMMANDS), "pieces": count,
               "seconds": float(t[-1]), "samples": samples, "gp_run": str(gp_run),
               "phnode_run": None if phnode_run is None else str(phnode_run), "phnode_checkpoint": phnode_checkpoint,
               "paper_protocol": protocol,
               "models": {e.label: {"failed": e.failed, "band": BAND_TEXT[e.band], "group": e.group} for e in items},
               "table": {e.label: None if e.failed else {key: {row: None if v is None else
                                                                {"mean": float(np.mean(v)), "std": float(np.std(v)),
                                                                 "per_piece": np.asarray(v, float).tolist()}
                                                                for row, v in rows.items()}
                                                          for key, rows in tables[e.label].items()} for e in items}}
    (output / "open-loop-horizons-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[report] wrote {pdf_path}", flush=True)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gp-run", type=Path, default=None)
    parser.add_argument("--phnode-run", type=Path, default=None, help="default: the latest finished PH-NODE-ORIG run")
    parser.add_argument("--phnode-checkpoint", default="final",
                        help="final (default) or stepNNNNNN: e.g. step002000 for a run that aborted before its final step "
                             "(disclosed in the report, 7 Oct 2026 user decision for the real-data PH-NODE)")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "reports")
    args = parser.parse_args()
    report(args.gp_run or latest_gp_run(), args.samples, args.seed, args.output,
           args.phnode_run or latest_phnode_run(args.phnode_checkpoint), args.phnode_checkpoint)


if __name__ == "__main__":
    main()

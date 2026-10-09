"""Open-loop horizons report of the pendulum campaign (4 Oct 2026): one PDF per dataset (setting x observation noise).

    python src/models/3D_SO3_Windy_Pendulum/comparison/report.py --setting DampConst-Wind --noise 0.25
    python src/models/3D_SO3_Windy_Pendulum/comparison/report.py --all          # the 24 reports
    python src/models/3D_SO3_Windy_Pendulum/comparison/report.py --all --settings DampRate-Wind

Layout (the same as src/models/SE3_Quadrotor/comparison/report.py):
    page 1    Physics identification: dataset specs, gauge-invariant products of every trained model vs the truth
    page 2    protocol and how to read
    pages 3-6 Table A for 0-1, 0-3, 0-5, 0-10 s: mean +- std over the 10 evaluation trajectories
    pages 7-9 error vs time, time-averaged error per horizon, calibration vs time
    then per trajectory: Table A for the four horizons (that trajectory alone), 3-D bob path, bob position, attitude
    (Euler xyz), angular velocity, energy.

Protocol (agreed 4 Oct 2026; ground-truth start since 4 Oct 16:30, user's choice). Evaluation set = the dataset's
long_test block: 10 trajectories (separate seeds), the same random controls U(-7, 7) per interval for every model.
    start     t = 0, every model from the CLEAN (ground-truth) state; no filter, no start uncertainty.
    open loop 10 s (200 intervals, samples 0..200) with the recorded controls and no feedback.
    energy    the TRUE Hamiltonian H(R, omega) = 1/2 omega^T J omega + m g l (R e_z) . e_z on predicted and true states
              (the learned H is defined only up to the gauge c); error |E[H(x_hat_t)] - H(x_t)|, E over sample paths.
    -wind-same       ONE path, posterior-mean model, the learned diffusion replaced by the simulator's RECORDED wind
                     increment of every interval (long_test_wind).
    new wind         32 sample paths: GP weight sample x Brownian path (GP-SDE), Brownian path (NN-SDE), GP weight
                     sample only (GP-ODE, epistemic), the true diffusion (Analytical-SDE-wind-different).
    point models     Lie-PH-NN-ODE, PH-NODE, Analytical-ODE (the true drift with the wind switched off): one path.
Scores are against the CLEAN trajectory. A model whose training failed is a '-' column; a model that was not trained
on the dataset (SDE models on wind-free data) is left out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent, PROJECT_ROOT / "envs" / "pendulum_so3"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from comparison.campaign import BASE_CONFIG, MODELS, OPTIONAL_MODELS, RESULT_DIR, SETTINGS, dataset_name, noise_tag, run_name  # noqa: E402
from comparison.models import load_run, model_factory, rollout_function  # noqa: E402
from lie_ph.evaluate import analytical_model, analytical_params, load_truth, make_env, operator_errors  # noqa: E402
from lie_ph.integrator import lie_imex_increment_step, rollout  # noqa: E402
from lie_ph.network import build_gp_setup  # noqa: E402

START_INDEX = 0                       # open loop from the ground-truth state at t = 0 (no filter)
OPEN_LOOP_STEPS = 200                 # 10 s at dt = 0.05 s
END_INDEX = START_INDEX + OPEN_LOOP_STEPS + 1
HORIZON_MARKS = (1.0, 3.0, 5.0, 10.0)
NOISE_LEVELS = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75)
COLORS = {"Lie-PH-GP-SDE": "tab:orange", "Lie-PH-GP-SDE-wind-same": "tab:brown", "Lie-PH-NN-SDE": "tab:cyan",
          "Lie-PH-NN-SDE-wind-same": "tab:olive", "Lie-PH-GP-ODE": "tab:red", "Lie-PH-NN-ODE": "tab:blue",
          "PH-NODE": "tab:pink", "PH-NODE-ref": "tab:brown", "PH-NODE-ref-pretrain": "tab:brown", "Analytical-SDE-wind-same": "tab:green", "Analytical-SDE-wind-different": "tab:purple",
          "Analytical-ODE": "0.35", "NeuralSDE": "goldenrod", "NeuralSDE-wind-same": "darkkhaki"}
BAND_TEXT = {"sample_gp_sde": "32 paths: posterior weight sample x Brownian path of the learned diffusion",
             "sample_nn_sde": "32 paths: Brownian paths of the learned diffusion",
             "sample_neural_sde": "32 paths: Brownian paths of the learned diffusion (unstructured neural SDE, Heun, 10 substeps)",
             "sample_gp_ode": "32 paths: posterior weight samples (epistemic only, no wind)",
             "same": "none: ONE path, posterior-mean model driven by the RECORDED wind of the data",
             "point": "none: one deterministic path",
             "sample_truth": "32 paths: Brownian paths of the TRUE diffusion, independent of the data's wind",
             "point_truth": "none: the true drift with the wind switched off"}
BLOCK_NAMES = {"b": "bob position error |p - p_true| (m)", "R": "attitude error (rad)",
               "H": "energy error |H(x_hat) - H(x)| (J, true H)",
               "w": "angular-velocity error |omega - omega_true| (rad/s)"}
UQ_NAMES = {"b": ("Bob position", "m"), "R": ("Attitude", "rad"), "w": ("Angular-velocity", "rad/s"), "H": ("Energy", "J")}
BLOCKS = ("b", "R", "w", "H")
TABLE_ROWS = (("b", "mean", "Bob position mean error (m)"), ("b", "final", "Bob position final error (m)"),
              ("R", "mean", "Attitude mean error (rad)"), ("R", "final", "Attitude final error (rad)"),
              ("w", "mean", "Angular-velocity mean error (rad/s)"), ("w", "final", "Angular-velocity final error (rad/s)"),
              ("H", "mean", "Energy mean error (J)"), ("H", "final", "Energy final error (J)"),
              ("det", "max", "Determinant max error"), ("orth", "max", "Orthogonality max error")) + tuple(
    row for block in BLOCKS for row in (
        (block, "crps", f"{UQ_NAMES[block][0]} CRPS ({UQ_NAMES[block][1]})"),
        (block, "cover1", f"{UQ_NAMES[block][0]} coverage mean +- 1 sigma"),
        (block, "cover2", f"{UQ_NAMES[block][0]} coverage mean +- 2 sigma"),
        (block, "spread", f"{UQ_NAMES[block][0]} spread-skill RMSE / sigma")))
ROW_TARGETS = {"cover1": 0.683, "cover2": 0.954, "spread": 1.0}
PAGE = (11.69, 8.27)


# ---------------------------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Entry:
    label: str
    source: str                      # campaign model name, or "analytical"
    mode: str                        # sample | same | point
    band: str                        # key of BAND_TEXT
    group: str                       # different | same | reference (ranking group)
    failed: bool = False
    note: str = ""
    run_dir: Path | None = None
    result: dict = field(default_factory=dict)


def entries_for(setting: str) -> list[Entry]:
    wind = SETTINGS[setting][1]
    out = []
    for model in sorted(MODELS, key=lambda m: not MODELS[m][1]):          # SDE models (and their -wind-same) first
        family, model_wind, _, _ = MODELS[model]
        if model_wind and not wind:
            continue                                  # not trained on this dataset: left out entirely
        if model_wind:
            out.append(Entry(model, model, "sample", f"sample_{family}_sde", "different"))
            out.append(Entry(f"{model}-wind-same", model, "same", "same", "same"))
        elif family == "gp":
            out.append(Entry(model, model, "sample", "sample_gp_ode", "different"))
        else:
            out.append(Entry(model, model, "point", "point", "different"))
    if wind:
        out.append(Entry("Analytical-SDE-wind-same", "analytical", "same", "same", "reference"))
        out.append(Entry("Analytical-SDE-wind-different", "analytical", "sample", "sample_truth", "reference"))
    out.append(Entry("Analytical-ODE", "analytical", "point", "point_truth", "reference"))
    return out


def campaign_records(result_dir: Path) -> dict[str, dict]:
    """Last record per run name over every results_*.jsonl of a campaign (queue, GPU and reused files)."""
    records = {}
    for path in sorted(result_dir.glob("results_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["name"]] = record
    return records


def wind_rollout(model, x0, controls, increments, interval: float, substeps: int):
    """(T, 15) path driven by given omega increments per interval (T-1, 3), split evenly over the substeps."""
    step = jnp.asarray(interval / substeps, x0.dtype)

    def body(state, inputs):
        control, increment = inputs
        state = state.at[12:15].set(control)
        for _ in range(substeps):
            state = lie_imex_increment_step(model, state, step, increment / substeps)
        return state, state

    _, path = jax.lax.scan(body, x0, (controls, increments))
    return jnp.concatenate([x0[None], path], axis=0)


# ---------------------------------------------------------------------------------------------------------------
# geometry and metrics (numpy, float64)
# ---------------------------------------------------------------------------------------------------------------
def project_matrices(matrices: np.ndarray) -> np.ndarray:
    """Nearest rotation (SVD); a diverged (non-finite or overflowing) matrix stays NaN."""
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
    """Mean over the sample axis 0 of (S, ..., 15) paths; the mean rotation is projected onto SO(3)."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(np.isfinite(paths), paths, np.nan), axis=0)
    mean[..., :9] = project(np.nan_to_num(mean[..., :9]) + 1e-12 * np.eye(3).reshape(9))
    return mean


def bob(states: np.ndarray, length: float) -> np.ndarray:
    """Bob position l R e_z (third column of R) of (..., 15) states."""
    rotation = states[..., :9].reshape(*states.shape[:-1], 3, 3)
    return length * rotation[..., :, 2]


def geodesic_deg(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    a = reference.reshape(*reference.shape[:-1], 3, 3)
    b = prediction.reshape(*prediction.shape[:-1], 3, 3)
    with np.errstate(invalid="ignore"):
        cosine = np.clip((np.trace(np.swapaxes(a, -1, -2) @ b, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def tangent_deg(rotation_flat: np.ndarray, reference_flat: np.ndarray) -> np.ndarray:
    """Rotation vector Log(R_ref^T R) in degrees, (..., 3); NaN where either is not finite."""
    shape = rotation_flat.shape[:-1]
    a = np.broadcast_to(reference_flat, rotation_flat.shape).reshape(-1, 3, 3)
    b = rotation_flat.reshape(-1, 3, 3)
    with np.errstate(invalid="ignore", over="ignore"):
        relative = project_matrices(np.swapaxes(a, -1, -2) @ b)
    vectors = np.full((len(relative), 3), np.nan)
    ok = np.all(np.isfinite(relative), axis=(-2, -1))
    if np.any(ok):
        vectors[ok] = np.degrees(Rotation.from_matrix(relative[ok]).as_rotvec())
    return vectors.reshape(*shape, 3)


def euler_deg(rotation_flat: np.ndarray) -> np.ndarray:
    """Euler xyz (deg) of the projected rotations, unwrapped along the time axis (-2 of the leading shape)."""
    shape = rotation_flat.shape[:-1]
    matrices = project(rotation_flat.reshape(-1, 9)).reshape(-1, 3, 3)
    angles = np.full((len(matrices), 3), np.nan)
    ok = np.all(np.isfinite(matrices), axis=(-2, -1))
    if np.any(ok):
        angles[ok] = Rotation.from_matrix(matrices[ok]).as_euler("xyz")
    angles = angles.reshape(*shape, 3)
    filled = np.nan_to_num(angles)
    return np.where(np.isfinite(angles), np.degrees(np.unwrap(filled, axis=-2)), np.nan)


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Ensemble CRPS per element, E|X - y| - 0.5 E|X - X'| over axis 0."""
    count = samples.shape[0]
    x = np.sort(samples, axis=0)
    first = np.mean(np.abs(x - truth[None]), axis=0)
    weights = (2.0 * np.arange(1, count + 1) - count - 1).reshape((count,) + (1,) * (samples.ndim - 1))
    return first - 0.5 * 2.0 * np.sum(weights * x, axis=0) / (count * count)


def true_energy(states: np.ndarray, truth: dict) -> np.ndarray:
    """TRUE Hamiltonian of (..., 15) states, H = 1/2 J |omega|^2 + m g l R[2, 2] (J = m l^2; the potential of the
    analytical model, evaluate.analytical_params). NaN / inf where the state is not finite."""
    with np.errstate(invalid="ignore", over="ignore"):
        omega = states[..., 9:12]
        return 0.5 * truth["inertia"] * np.sum(omega * omega, axis=-1) + truth["m"] * truth["g"] * truth["l"] * states[..., 8]


def predicted_energy(paths: np.ndarray | None, single: np.ndarray | None, truth: dict) -> np.ndarray:
    """(T, B) energy prediction: mean of H over the sample paths, or H of the single path."""
    if paths is None:
        return true_energy(single, truth)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(np.isfinite(paths).all(axis=-1), true_energy(paths, truth), np.nan), axis=0)


def error_series(truth: np.ndarray, centre: np.ndarray, length: float, energy: np.ndarray,
                 true_h: np.ndarray) -> dict[str, np.ndarray]:
    """(T, B) error of the prediction per block; a non-finite state counts as an infinite error."""
    with np.errstate(invalid="ignore", over="ignore"):
        out = {"b": np.linalg.norm(bob(centre, length) - bob(truth, length), axis=-1),
               "R": np.radians(geodesic_deg(truth[..., :9], centre[..., :9])),     # rad (user, 5 Oct 2026)
               "w": np.linalg.norm(centre[..., 9:12] - truth[..., 9:12], axis=-1),
               "H": np.abs(energy - true_h)}
    bad = ~np.isfinite(centre[..., :12]).all(axis=-1) | ~np.isfinite(energy)
    return {k: np.where(bad, np.inf, np.nan_to_num(v, nan=np.inf)) for k, v in out.items()}


def so3_violations(states: np.ndarray) -> dict[str, np.ndarray]:
    """|det R - 1| and ||R^T R - I||_F of the RAW rollout, (..., T, B)."""
    rotation = np.asarray(states[..., :9], dtype=np.float64).reshape(*states.shape[:-1], 3, 3)
    with np.errstate(invalid="ignore", over="ignore"):
        determinant = np.abs(np.linalg.det(np.nan_to_num(rotation, nan=1e6)) - 1.0)
        gram = np.swapaxes(rotation, -1, -2) @ rotation - np.eye(3)
        orthogonality = np.nan_to_num(np.sqrt(np.sum(gram * gram, axis=(-2, -1))), nan=np.inf)
    return {"det": np.where(np.isfinite(rotation).all(axis=(-2, -1)), determinant, np.inf), "orth": orthogonality}


def calibration(truth: np.ndarray, paths: np.ndarray | None, centre: np.ndarray, length: float, energy: np.ndarray,
                true_h: np.ndarray, truth_constants: dict) -> dict:
    """Per block, (T, B) arrays averaged over the block's axes: CRPS, coverage of mean +- 1 / 2 sigma, squared error
    of the sample mean and sample variance. Attitude in the tangent space at the projected mean, rad; energy = the
    true H of every path (1 axis). Without samples only the CRPS (= absolute error per axis) is defined."""
    out = {}
    for block in BLOCKS:
        if block == "b":
            y, x, c = bob(truth, length), None if paths is None else bob(paths, length), bob(centre, length)
        elif block == "R":
            y = np.radians(tangent_deg(truth[..., :9], centre[..., :9]))       # rad (user, 5 Oct 2026)
            x = None if paths is None else np.radians(
                tangent_deg(paths[..., :9], np.broadcast_to(centre[..., :9], paths[..., :9].shape)))
            c = np.zeros_like(y)
        elif block == "H":
            y, c = true_h[..., None], energy[..., None]
            x = None if paths is None else true_energy(paths, truth_constants)[..., None]
        else:
            y, x, c = truth[..., 9:12], None if paths is None else paths[..., 9:12], centre[..., 9:12]
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


def table_values(result: dict, marks, interval: float) -> dict[str, dict[str, np.ndarray | None]]:
    """Per horizon and row, the (B,) per-trajectory value: time average over (0, H] ("mean"), value at H ("final"),
    max SO(3) violation over (0, H] (averaged over sample paths), and the uncertainty rows (time averages)."""
    values = {}
    for mark in marks:
        count = int(round(mark / interval))
        window = slice(1, count + 1)
        row = {}
        for block, series in result["errors"].items():
            row[f"{block}_mean"] = series[window].mean(axis=0)
            row[f"{block}_final"] = series[count]
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
        values[f"0-{mark:g}s"] = row
    return values


def plottable(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(values) & (np.abs(values) < 1e12), values, np.nan)


# ---------------------------------------------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------------------------------------------
class Dataset:
    def __init__(self, setting: str, level: float):
        name = dataset_name(setting)
        self.setting, self.level, self.name = setting, level, name
        self.path = PROJECT_ROOT / "datasets" / f"PENDULUM-DATASET-{name}" / f"{name}_{noise_tag(level)}.pkl"
        import pickle
        with self.path.open("rb") as handle:
            payload = pickle.load(handle)
        if "long_test_x" not in payload:
            raise KeyError(f"{self.path} has no long_test block; regenerate the dataset")
        self.settings = payload["settings"]
        # (5, T, 2, 15) batches -> time-major (T, B, 15), B = 10
        merge = lambda a: np.concatenate(list(np.asarray(a)), axis=1)
        self.clean = merge(payload["long_test_x"]).astype(np.float32)
        self.noisy = merge(payload["long_test_x_noisy"]).astype(np.float32)
        self.wind = merge(payload["long_test_wind"]).astype(np.float32)
        self.t = np.asarray(payload["long_t"], dtype=np.float64)
        self.interval = float(self.t[1] - self.t[0])
        self.train_shape, self.test_shape = payload["x"].shape, payload["test_x"].shape
        self.truth = load_truth(self.path)
        self.env = make_env(self.truth)


NEURAL_SDE_DIR = Path(__file__).resolve().parents[1] / "neural_sde"


def load_neural_sde(record: dict):
    """The unstructured NeuralSDE (src/models/3D_SO3_Windy_Pendulum/neural_sde) from the .eqx checkpoint of its own trainer."""
    import importlib.util
    import equinox as eqx
    spec = importlib.util.spec_from_file_location("neural_sde_network", NEURAL_SDE_DIR / "network.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                     # dataclasses (equinox Modules) look their module up while executing
    spec.loader.exec_module(module)
    template = module.NeuralSO3SDE(key=jax.random.PRNGKey(0), u_dim=3, hidden_dim=int(record.get("hidden_dim", 500)))
    return eqx.tree_deserialise_leaves(record["checkpoint"], template)


def neural_rollouts(model, start, controls, increments, interval: float, substeps: int, samples: int, seed: int, mode: str):
    """NeuralSDE paths in the report layout. sample: (S, T, B, 15), Brownian increments var = h per substep (its own law);
    same: (T, B, 15), drift-only Heun substeps + the data's recorded omega increment of each interval, split evenly over
    the substeps (the -wind-same replay of the other SDE models)."""
    h = interval / substeps
    x0 = start[:, :12]
    steps = controls.shape[1]

    def pad(path, u):                                   # (B, T, 12) -> (B, T, 15) with the recorded controls (row k+1 = u_k)
        u_rows = jnp.concatenate([u[:, :1], u], axis=1)
        return jnp.concatenate([path, u_rows], axis=-1)

    if mode == "same":
        def one(x, u, inc):
            def outer(state, inputs):
                u_k, inc_k = inputs
                def inner(s, _):
                    s = model.step(s, u_k, h, jnp.zeros(3, s.dtype))
                    return s.at[9:12].add(inc_k / substeps), None
                state, _ = jax.lax.scan(inner, state, None, length=substeps)
                return state, state
            _, path = jax.lax.scan(outer, x, (u, inc))
            return jnp.concatenate([x[None], path], axis=0)
        path = jax.jit(jax.vmap(one))(x0, controls, increments)
        return np.swapaxes(np.asarray(pad(path, controls)), 0, 1).astype(np.float64)

    @jax.jit
    def sampled(key):
        dW = jnp.sqrt(h) * jax.random.normal(key, (x0.shape[0], steps, substeps, 3), jnp.float32)
        path = jax.vmap(lambda x, u, w: model.rollout(x, u, h, w))(x0, controls, dW)
        return pad(path, controls)

    raw = np.stack([np.asarray(sampled(jax.random.PRNGKey(seed + i))) for i in range(samples)])     # (S, B, T, 15)
    return np.swapaxes(raw, 1, 2).astype(np.float64)


def predict_entries(data: Dataset, entries: list[Entry], records: dict, samples: int, seed: int) -> dict:
    """Fill entry.result for every entry; returns the per-source context (models, params, start states)."""
    truth_t = data.clean[START_INDEX:END_INDEX]                                    # (T, B, 15), T = 201
    controls = jnp.asarray(np.swapaxes(data.clean[START_INDEX + 1:END_INDEX, :, 12:15], 0, 1))   # (B, T-1, 3)
    increments = jnp.asarray(np.swapaxes(data.wind[START_INDEX + 1:END_INDEX], 0, 1))            # (B, T-1, 3)
    start = jnp.asarray(data.clean[START_INDEX])                                                 # (B, 15) ground truth
    steps = truth_t.shape[0] - 1
    length = data.truth["l"]
    sources: dict[str, dict] = {}

    def source_context(source: str) -> dict | None:
        if source in sources:
            return sources[source]
        if source == "analytical":
            # the model class of the campaign's base config (its GP features are irrelevant: all weights are zero)
            config = yaml.safe_load(BASE_CONFIG.read_text())
            setup = build_gp_setup(config["model"], jax.random.PRNGKey(0))
            params = analytical_params(data.truth, setup, config["model"])
            factory = lambda p, key: analytical_model(p, setup, data.truth, None)      # true operators: no weight noise
            context = {"params": params, "factory": factory, "config": config, "substeps": int(config["data"]["substeps"]),
                       "run_dir": None, "rollout": rollout}
        elif MODELS.get(source, ("",))[0] == "neural":
            record = records.get(run_name(source, data.setting, data.level))
            if record is None or record.get("status") != "done" or not Path(record.get("checkpoint", "")).exists():
                sources[source] = None
                return None
            context = {"neural": load_neural_sde(record), "run_dir": Path(record["run_dir"]),
                       "substeps": int(record.get("n_substeps", 10)), "params": {}, "config": {}}
            sources[source] = context
            return context
        else:
            record = records.get(run_name(source, data.setting, data.level))
            run_dir = Path(record["run_dir"]) if record and record.get("run_dir") else None
            if record is None or record.get("status") != "done" or not (run_dir / "checkpoint_final.pkl").exists():
                sources[source] = None
                return None
            config, setup, params, _ = load_run(run_dir, "final")
            leaves = jax.tree_util.tree_leaves(params)
            if not all(bool(jnp.all(jnp.isfinite(v))) for v in leaves):
                sources[source] = None
                return None
            context = {"params": params, "factory": model_factory(config, setup), "config": config,
                       "substeps": int(config["data"]["substeps"]), "run_dir": run_dir, "rollout": rollout_function(config)}
        context["mean_model"] = context["factory"](context["params"], None)
        sources[source] = context
        return context

    for entry in entries:
        context = source_context(entry.source)
        if context is None:
            entry.failed = True
            entry.note = "training failed (non-finite)"
            continue
        entry.run_dir = context["run_dir"]
        entry.note = "ground-truth state at t = 0"
        paths, mean = None, None
        if "neural" in context:                              # unstructured NeuralSDE (its own model class and integrator)
            out = neural_rollouts(context["neural"], start, controls, increments, data.interval, context["substeps"],
                                  samples, seed, entry.mode)
            if entry.mode == "same":
                mean = np.swapaxes(out, 0, 1)                # (B, T, 15), the layout of the other point paths
            else:
                paths = out
            substeps, model = context["substeps"], None
        else:
            substeps, model = context["substeps"], context["mean_model"]
        zero = jnp.zeros((steps, substeps, 3), jnp.float32)
        if "neural" in context:
            pass
        elif entry.mode == "point":
            advance = context["rollout"]
            mean = np.asarray(jax.jit(jax.vmap(lambda x0, u: advance(model, x0, u, data.interval, zero)))(start, controls))
        elif entry.mode == "same":
            mean = np.asarray(jax.jit(jax.vmap(lambda x0, u, i: wind_rollout(model, x0, u, i, data.interval, substeps)))(
                start, controls, increments))
        else:
            params, factory, advance = context["params"], context["factory"], context["rollout"]

            @jax.jit
            def sampled(key):
                weight_key, noise_key = jax.random.split(key)
                sample_model = factory(params, weight_key)
                noise = jax.random.normal(noise_key, (start.shape[0], steps, substeps, 3), jnp.float32)
                return jax.vmap(lambda s, u, e: advance(sample_model, s, u, data.interval, e))(start, controls, noise)

            raw = np.stack([np.asarray(sampled(jax.random.PRNGKey(seed + i))) for i in range(samples)])   # (S, B, T, 15)
            paths = np.swapaxes(raw, 1, 2).astype(np.float64)                                      # (S, T, B, 15)
            mean = None
        raw_mean = None if mean is None else np.swapaxes(mean, 0, 1).astype(np.float64)            # (T, B, 15)
        centre = ensemble_mean(paths) if paths is not None else raw_mean
        truth64 = truth_t.astype(np.float64)
        true_h = true_energy(truth64, data.truth)
        energy = predicted_energy(paths, raw_mean, data.truth)
        entry.result = {"centre": centre, "paths": paths, "energy": energy,
                        "errors": error_series(truth64, centre, length, energy, true_h),
                        "so3": so3_violations(paths if paths is not None else raw_mean),
                        "calib": calibration(truth64, paths, centre, length, energy, true_h, data.truth)}
        print(f"[report] {entry.label:31s} start: {entry.note}", flush=True)
    return sources


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


def law_text(truth: dict) -> str:
    c = truth["friction"][0]
    if truth["friction_law"] == "rate_dependent":
        return f"rate-dependent  tau_D = -c (1 + |omega|) omega,  c = {c:g}  (the quadrotor's law)"
    if truth["friction_law"] == "varying":
        return f"varying  tau_D = -c (1 + 0.5 h + 0.5 tanh|omega|) omega,  h = (1 - R_33) / 2,  c = {c:g}"
    return f"constant  tau_D = -c omega,  c = {c:g}"


def physics_page(data: Dataset, entries: list[Entry], sources: dict, names: dict | None = None,
                 training_note: str | None = None) -> plt.Figure:
    """``names`` (optional): {campaign model: name printed in the table}; ``training_note`` replaces the default
    training line of the spec block (closed_loop.py --label / --models)."""
    names = names or {}
    truth, s = data.truth, data.settings
    figure = plt.figure(figsize=PAGE)
    figure.text(0.03, 0.955, "Physics Identification", fontsize=17, weight="bold")
    figure.text(0.33, 0.957, f"{data.setting}, observation noise {data.level:g}", fontsize=12)
    n_train = data.train_shape[0] * data.train_shape[2]
    n_test = data.test_shape[0] * data.test_shape[2]
    spec = [
        f"dataset     {data.path.name}  (in datasets/PENDULUM-DATASET-<same name>/)",
        f"plant       3-D pendulum on SO(3): m = {truth['m']:g} kg, l = {truth['l']:g} m, g = {truth['g']:g} m/s^2, J = m l^2 I = {truth['inertia']:g} I;"
        f"  bob at l R e_z",
        f"damping     {law_text(truth)}",
        f"wind        " + (f"constant diffusion, force std {truth['wind']:g} at the bob -> Sigma = (0.5, 0.5, 0) rad s^-1.5 per body axis"
                           if truth["wind"] > 0 else "none (ODE data)"),
        f"control     u ~ U(-{s['random_u_scale']:g}, {s['random_u_scale']:g})^3 per interval, held; G = diag({', '.join(f'{v:g}' for v in truth['g_diag'])}) (torque = G u)",
        f"noise       y = x (+) eps:  R Exp(a), omega + b,  a, b ~ N(0, {data.level:g}^2 I)" + ("  (clean data)" if data.level == 0 else ""),
        f"sampling    dt = {data.interval:g} s; simulator {s['integrator']} with {s['substeps']} substeps per interval",
        training_note or (f"training    {n_train} trajectories x {data.train_shape[1]} samples ({(data.train_shape[1] - 1) * data.interval:g} s), noisy;"
                          f" 2 s windows, 25 % overlap; 5000 steps, final model; float32; no priors, no pretraining"),
        f"evaluation  {data.clean.shape[1]} separate trajectories x {data.clean.shape[0]} samples ({data.t[-1]:g} s, long_test, own seeds):"
        f" open loop from the ground-truth state at t = 0 for {OPEN_LOOP_STEPS * data.interval:g} s",
    ]
    figure.text(0.03, 0.925, "\n".join(spec), fontsize=7.4, va="top", family="monospace")

    # rows: ground truth, then one row per trained model (the -wind-same versions are the same trained model)
    states = data.clean.reshape(-1, 15)
    header = ["model", "M^-1 g diag\n(xx / yy / zz)", "M^-1 g\nrel. err", "gravity accel.\nrel. err",
              "M^-1 D diag (mean)\n(xx / yy / zz)", "M^-1 D\nrel. err", "Sigma\n(x / y / z)", "sigma_obs\n(R / omega)",
              "test EKF\nNLL / d", "test NIS / d\n(1 = consistent)"]
    true_damping = None
    rows, failed_rows = [], []
    scores: list[dict[int, float]] = []          # per row: column -> score (lower = closer to the truth)
    seen = []
    for entry in entries:
        if entry.source == "analytical" or entry.source in seen:
            continue
        seen.append(entry.source)
        context = sources.get(entry.source)
        if context is None:
            failed_rows.append(len(rows) + 1)
            rows.append([f"{names.get(entry.source, entry.source)}\n(training failed)"] + ["-"] * (len(header) - 1))
            scores.append({})
            continue
        if "neural" in context:                              # unstructured: no M^-1, D, V, g to compare
            rows.append([f"{names.get(entry.source, entry.source)}\n(no physical operators)"] + ["-"] * (len(header) - 1))
            scores.append({})
            continue
        ops = operator_errors(context["mean_model"], states, truth, data.env)
        true_damping = ops["damping_M_inv_D"]["true"]
        evaluation = json.loads((context["run_dir"] / "evaluation_final.json").read_text()).get("learned", {})
        params = context["params"]
        sigma = "-" if "process" not in params else " / ".join(f"{v:.3g}" for v in np.exp(np.asarray(params["process"]["log_sigma"])))
        obs = "-" if "likelihood" not in params else " / ".join(f"{v:.3g}" for v in np.exp(np.asarray(params["likelihood"]["log_sigma"])))
        score = {1: ops["control_M_inv_g"]["relative_error"], 2: ops["control_M_inv_g"]["relative_error"],
                 3: ops["gravity_acceleration"]["relative_error"],
                 4: ops["damping_M_inv_D"]["relative_error"], 5: ops["damping_M_inv_D"]["relative_error"]}
        if "process" in params:
            score[6] = float(np.linalg.norm(np.exp(np.asarray(params["process"]["log_sigma"])) - truth["diffusion"]))
        if "likelihood" in params:
            score[7] = float(np.linalg.norm(np.exp(np.asarray(params["likelihood"]["log_sigma"])) - data.level))
        if "test_ekf_nll_noisy" in evaluation:
            score[8] = float(evaluation["test_ekf_nll_noisy"])
            score[9] = abs(float(evaluation["test_nis_per_dimension"]) - 1.0)
        scores.append({c: v for c, v in score.items() if np.isfinite(v)})
        rows.append([names.get(entry.source, entry.source) + ("\n(+ -wind-same)" if MODELS[entry.source][1] else ""),
                     " / ".join(f"{v:.3g}" for v in np.diag(ops["control_M_inv_g"]["mean_learned"])),
                     f"{ops['control_M_inv_g']['relative_error'] * 100:.1f} %",
                     f"{ops['gravity_acceleration']['relative_error'] * 100:.1f} %",
                     " / ".join(f"{v:.3g}" for v in np.diag(ops["damping_M_inv_D"]["mean_learned"])),
                     f"{ops['damping_M_inv_D']['relative_error'] * 100:.1f} %", sigma, obs,
                     f"{evaluation['test_ekf_nll_noisy']:.4f}" if "test_ekf_nll_noisy" in evaluation else "-",
                     f"{evaluation['test_nis_per_dimension']:.3f}" if "test_nis_per_dimension" in evaluation else "-"])
    if true_damping is None:
        true_damping = np.diag(truth["friction"] / truth["inertia"]).tolist()
    reference = {}
    for context in sources.values():
        if context and context.get("run_dir") is not None:
            reference = json.loads((context["run_dir"] / "evaluation_final.json").read_text()).get("analytical", {})
            break
    truth_row = ["Ground truth", " / ".join(f"{v:.3g}" for v in truth["g_diag"] / truth["inertia"]), "0", "0",
                 " / ".join(f"{v:.3g}" for v in np.diag(true_damping)), "0",
                 " / ".join(f"{v:.3g}" for v in truth["diffusion"]) if truth["wind"] > 0 else "0 (no wind)",
                 f"{data.level:g} / {data.level:g}",
                 f"{reference['test_ekf_nll_noisy']:.4f}" if "test_ekf_nll_noisy" in reference and data.level > 0 else "-",
                 f"{reference['test_nis_per_dimension']:.3f}" if "test_nis_per_dimension" in reference and data.level > 0 else "-"]
    cells = [truth_row] + rows
    axis = figure.add_axes([0.01, 0.10, 0.98, 0.63])
    axis.axis("off")
    widths = [0.13, 0.12, 0.065, 0.075, 0.12, 0.065, 0.12, 0.09, 0.07, 0.085]
    table = styled_table(axis, cells, header, widths, 7.0, 2.8, failed_cols=())
    for c in range(len(header)):
        table[1, c].set_facecolor("#e8e8e8")
    for r in failed_rows:
        for c in range(1, len(header)):
            table[r + 1, c].set_facecolor("#f6e0e0")
    for c in range(1, len(header)):                   # best learned model per column: green bold (needs 2+ candidates)
        candidates = [(score[c], r) for r, score in enumerate(scores) if c in score]
        if len(candidates) > 1:
            cell = table[min(candidates)[1] + 2, c]   # + header row + ground-truth row
            cell.set_facecolor("#c8ecc8")
            cell.set_text_props(weight="bold")
    notes = ("Only gauge-invariant products are identifiable ((M^-1, V, D, g) -> (c M^-1, V/c, D/c, g/c) leaves the dynamics unchanged). "
             "Values: posterior-mean model, averaged over the clean states of the evaluation set; rel. err = mean over states of "
             "||O_hat - O||_F / mean ||O||_F (for rate-dependent damping O = c (1 + |omega|) / J state by state). Gravity accel. = M^-1 "
             "sum_i r_i x dV/dr_i at omega = 0. Sigma = learned diffusion (rad s^-1.5), sigma_obs = learned observation noise ('-' = the model "
             "has none: ODE models have no Sigma, rollout-loss models no likelihood). EKF NLL / d and NIS / d: evaluate.py on the 25 noisy "
             "5 s test trajectories (2 s windows); the ground-truth row is the analytical model in the same filter. Green bold = best learned "
             "model per column: lowest rel. err (also marks its diag values), Sigma and sigma_obs closest to the truth (Euclidean), lowest "
             "NLL, NIS closest to 1.")
    figure.text(0.03, 0.02, "\n".join(textwrap.wrap(notes, 200)), fontsize=6.6)
    return figure


def protocol_page(data: Dataset, entries: list[Entry], samples: int) -> plt.Figure:
    lines = [f"dataset   {data.path.name}   evaluation: long_test, {data.clean.shape[1]} trajectories",
             f"protocol  ONE nonstop open-loop rollout per trajectory from t = 0 to {OPEN_LOOP_STEPS * data.interval:g} s with the recorded",
             "          controls, no feedback. Scored against the CLEAN trajectory over 0-1, 0-3, 0-5, 0-10 s. Nothing is",
             "          filtered: a diverged rollout keeps its (large or infinite) error.",
             "start     every model starts from the GROUND-TRUTH (clean) state at t = 0: no filter, no start uncertainty.",
             "energy    the TRUE Hamiltonian H = 1/2 omega^T J omega + m g l (R e_z).e_z (J = m l^2 I) evaluated on the predicted and",
             "          the true state (the learned H is only defined up to the gauge c, so it is not compared directly);",
             "          energy error = |E[H(x_hat_t)] - H(x_t)| with E over the sample paths. H is not conserved here (damping,",
             "          control and wind change it), so this is an energy-trajectory error, not a conservation test.",
             f"line/band {samples} sample paths: line = their mean (mean rotation projected onto SO(3)), dark = +- 1 sigma, light = +- 2 sigma.",
             "", "models"]
    for e in entries:
        status = "TRAINING FAILED -> '-' everywhere" if e.failed else f"start: {e.note}"
        lines.append(f"  {e.label:30s} {status}")
        lines.append(f"  {'':30s} band: {BAND_TEXT[e.band]}")
    lines += ["", "how to read",
              "  Analytical = the model class with every subnetwork set to the simulator's value (one Lie-IMEX step per sample).",
              "  -wind-same models are fed the data's own recorded wind increments: the Analytical-SDE-wind-same path differs from the",
              "  data only by the 1-step integrator (10 substeps in the simulator). With a new wind draw (-wind-different) even the true",
              "  model drifts away: its error is the irreducible open-loop error of the wind, and its band is the spread a perfect SDE",
              "  must show. Green bold = best per row among learned new-wind models, and separately among learned -wind-same models."]
    figure = plt.figure(figsize=PAGE)
    figure.text(0.04, 0.95, "Open-loop horizons: protocol and models", fontsize=16, weight="bold")
    figure.text(0.04, 0.90, "\n".join(lines), fontsize=8.2, va="top", family="monospace")
    return figure


def fmt(value, block: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, tuple):
        mean, std = value
        return f"{mean:.2e} +- {std:.1e}" if block in ("det", "orth") else f"{mean:.4g} +- {std:.2g}"
    return f"{value:.2e}" if block in ("det", "orth") else f"{value:.4g}"


def table_page(entries: list[Entry], tables: dict, key: str, title: str, subtitle: str, trajectory: int | None) -> plt.Figure:
    """Rows = metrics, columns = models. ``trajectory`` None = mean +- std over trajectories, else that trajectory."""
    figure = plt.figure(figsize=PAGE)
    figure.text(0.03, 0.955, title, fontsize=14, weight="bold")
    figure.text(0.03, 0.925, subtitle, fontsize=7.2)
    header = ["metric"] + [e.label.replace("-wind-", "-\nwind-").replace("Lie-PH-", "Lie-PH-\n") for e in entries]
    cells, best = [], []
    for r, (block, kind, label) in enumerate(TABLE_ROWS):
        raw = []
        for e in entries:
            v = None if e.failed else tables[e.label][key].get(f"{block}_{kind}")
            if v is not None:
                v = (float(np.mean(v)), float(np.std(v))) if trajectory is None else float(v[trajectory])
            raw.append(v)
        target = ROW_TARGETS.get(kind)
        score = lambda i: (abs((raw[i][0] if isinstance(raw[i], tuple) else raw[i]) - target) if target is not None
                           else (raw[i][0] if isinstance(raw[i], tuple) else raw[i]))
        for group in ("different", "same"):
            ranked = [i for i, e in enumerate(entries) if e.group == group and raw[i] is not None and np.isfinite(score(i))]
            if len(ranked) > 1:
                best.append((r + 1, 1 + min(ranked, key=score)))
        cells.append([label] + [fmt(v, block) for v in raw])
    axis = figure.add_axes([0.01, 0.06, 0.98, 0.84])
    axis.axis("off")
    widths = [0.17] + [0.83 / len(entries)] * len(entries)
    styled_table(axis, cells, header, widths, 6.0 if len(entries) > 6 else 7.0, 1.3, best,
                 failed_cols=[1 + i for i, e in enumerate(entries) if e.failed])
    figure.text(0.03, 0.015, "Green bold = best per row among the learned models with a NEW wind draw, and separately among the learned "
                "models fed the RECORDED wind (-wind-same); Analytical columns are references, not ranked. Best = lowest, except coverage "
                "(closest to 0.683 / 0.954)\nand spread-skill (closest to 1). CRPS of a single path = its absolute error; '-' = no samples, "
                "or training failed (red column). Mean error = time average over (0, H]; final = at H; SO(3) rows = max over (0, H] on the RAW "
                "rollout. Uncertainty rows: time average over (0, H].", fontsize=6.0)
    return figure


def error_page(t, entries, marks) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=PAGE)
    axes = axes.ravel()
    for axis, key in zip(axes, BLOCKS):
        for e in entries:
            if e.failed:
                continue
            series = e.result["errors"][key]
            analytical = e.group == "reference"
            axis.plot(t[1:], plottable(np.median(series, axis=1))[1:], color=COLORS[e.label], lw=1.5 if analytical else 2,
                      ls="--" if analytical else "-", label=e.label)
            if not analytical:
                axis.fill_between(t[1:], plottable(np.quantile(series, 0.25, axis=1))[1:],
                                  plottable(np.quantile(series, 0.75, axis=1))[1:], color=COLORS[e.label], alpha=0.10, lw=0)
        for mark in marks[:-1]:
            axis.axvline(mark, color="0.5", lw=0.8, ls="--")
        axis.set_yscale("log")
        axis.set_title(BLOCK_NAMES[key], fontsize=9)
        axis.set_xlabel("t since the open-loop start (s)")
        axis.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=6.5)
    figure.suptitle("Open-loop error of the prediction (mean of the sample paths) vs time: median over trajectories, shade = "
                    "inter-quartile, dashed = analytical", fontsize=10)
    figure.tight_layout()
    return figure


def bar_page(entries, tables, marks) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=PAGE)
    axes = axes.ravel()
    alive = [e for e in entries if not e.failed]
    width = 0.8 / max(len(alive), 1)
    for axis, key in zip(axes, BLOCKS):
        for k, e in enumerate(alive):
            values = plottable([np.median(tables[e.label][f"0-{m:g}s"][f"{key}_mean"]) for m in marks])
            axis.bar(np.arange(len(marks)) + (k - (len(alive) - 1) / 2) * width, values, width, color=COLORS[e.label],
                     label=e.label, edgecolor="k", lw=0.3, hatch="//" if e.group == "reference" else None)
        axis.set_xticks(range(len(marks)))
        axis.set_xticklabels([f"0-{m:g} s" for m in marks])
        axis.set_yscale("log")
        axis.set_title("time-averaged " + BLOCK_NAMES[key], fontsize=8.5)
        axis.grid(alpha=0.3, axis="y", which="both")
    axes[0].legend(fontsize=6.5)
    figure.suptitle("Time-averaged error over 0-1 / 0-3 / 0-5 / 0-10 s (median over trajectories; hatched = analytical)", fontsize=10)
    figure.tight_layout()
    return figure


def calibration_page(t, entries, marks) -> plt.Figure:
    figure, axes = plt.subplots(4, 4, figsize=(15.0, PAGE[1]), sharex=True)
    names = {"b": "bob position", "R": "attitude (tangent, rad)", "w": "angular velocity", "H": "energy (true H)"}
    for col, key in enumerate(BLOCKS):
        for e in entries:
            if e.failed:
                continue
            stats = e.result["calib"][key]
            style = dict(color=COLORS[e.label], lw=1.5, ls="--" if e.group == "reference" else "-", label=e.label)
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
        for row, what in enumerate(("coverage of mean +- 1 sigma", "coverage of mean +- 2 sigma", "spread-skill RMSE / sigma", "CRPS")):
            axes[row, col].set_title(f"{names[key]}: {what}", fontsize=8)
            axes[row, col].grid(alpha=0.3, which="both")
            for mark in marks[:-1]:
                axes[row, col].axvline(mark, color="0.5", lw=0.6, ls="--")
        axes[3, col].set_xlabel("t since the open-loop start (s)")
    axes[3, 0].legend(fontsize=5.5)
    figure.suptitle("Calibration of the sample paths vs time (mean over trajectories and axes). Dotted = ideal 0.683, 0.954, 1. "
                    "Spread-skill > 1 = over-confident.\nCRPS of a single path = its absolute error.", fontsize=9)
    figure.tight_layout()
    return figure


def path3d_page(data: Dataset, entries, index) -> plt.Figure:
    length = data.truth["l"]
    columns = 5 if len(entries) > 6 else 3
    figure = plt.figure(figsize=(16.5 if columns == 5 else PAGE[0], PAGE[1]))
    truth_bob = bob(data.clean[START_INDEX:END_INDEX, index].astype(np.float64), length)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
    for k, e in enumerate(entries):
        axis = figure.add_subplot(math.ceil(len(entries) / columns), columns, k + 1, projection="3d")
        axis.plot_wireframe(length * np.cos(u) * np.sin(v), length * np.sin(u) * np.sin(v), length * np.cos(v),
                            color="0.85", lw=0.3)
        axis.plot(*truth_bob.T, color="k", lw=1.0)
        if e.failed:
            axis.set_title(f"{e.label}\ntraining failed", fontsize=8, color="0.4")
        else:
            if e.result["paths"] is not None:
                for path in e.result["paths"][:8, :, index]:
                    axis.plot(*plottable(bob(path, length)).T, color=COLORS[e.label], lw=0.4, alpha=0.35)
            centre = bob(e.result["centre"][:, index], length)
            axis.plot(*plottable(centre).T, color=COLORS[e.label], lw=1.5)
            axis.scatter(*centre[0], color="k", s=8)
            axis.set_title(e.label, fontsize=8, color=COLORS[e.label], weight="bold")
        lim = 1.1 * length
        axis.set_xlim(-lim, lim)
        axis.set_ylim(-lim, lim)
        axis.set_zlim(-lim, lim)
        axis.tick_params(labelsize=5)
    figure.suptitle(f"Evaluation trajectory {index}: bob path l R e_z on the sphere (black = data 0-10 s, bold = prediction "
                    "from the ground-truth state at t = 0, thin = 8 sample paths, dot = start)", fontsize=10)
    figure.tight_layout()
    return figure


def block_page(data: Dataset, entries, index, block, marks) -> plt.Figure:
    length = data.truth["l"]
    names = {"b": (["bob x (m)", "bob y (m)", "bob z (m)"], "Bob position l R e_z"),
             "euler": (["roll (rad)", "pitch (rad)", "yaw (rad)"], "Attitude (Euler xyz of R, unwrapped)"),
             "w": (["omega_x (rad/s)", "omega_y (rad/s)", "omega_z (rad/s)"], "Body angular velocity")}
    labels, title = names[block]

    def values(states):
        if block == "b":
            return bob(states, length)
        if block == "euler":
            return np.radians(euler_deg(states[..., :9]))                      # rad (user, 6 Oct 2026)
        return states[..., 9:12]

    figure, axes = plt.subplots(3, len(entries), figsize=(max(PAGE[0], 2.5 * len(entries)), PAGE[1]), sharex=True, squeeze=False)
    t_all = data.t[:END_INDEX]
    t_open = data.t[START_INDEX:END_INDEX]
    truth = values(data.clean[:END_INDEX, index].astype(np.float64))
    for col, e in enumerate(entries):
        for row in range(3):
            axis = axes[row, col]
            axis.plot(t_all, truth[:, row], color="k", lw=0.9)
            low, high = np.nanmin(truth[:, row]), np.nanmax(truth[:, row])
            reach = 3.0 * max(high - low, 1e-3)
            extent = [low, high]
            if e.failed:
                if row == 1:
                    axis.text(0.5, 0.5, "training failed", transform=axis.transAxes, ha="center", color="0.4")
            else:
                paths = e.result["paths"]
                if paths is not None:
                    sample_values = values(paths[:, :, index])[..., row]
                    if block == "euler":                                  # put each path on the truth's 2 pi branch
                        sample_values = sample_values - 2 * np.pi * np.round((sample_values[:, :1] - truth[START_INDEX, row]) / (2 * np.pi))
                    with np.errstate(invalid="ignore"):
                        mu, sd = np.nanmean(sample_values, axis=0), np.nanstd(sample_values, axis=0, ddof=1)
                    axis.fill_between(t_open, plottable(mu - 2 * sd), plottable(mu + 2 * sd), color=COLORS[e.label], alpha=0.15, lw=0)
                    axis.fill_between(t_open, plottable(mu - sd), plottable(mu + sd), color=COLORS[e.label], alpha=0.30, lw=0)
                    line = mu
                    for bound in (mu - 2 * sd, mu + 2 * sd):
                        finite = bound[np.isfinite(bound)]
                        if finite.size:
                            extent = [min(extent[0], finite.min()), max(extent[1], finite.max())]
                else:
                    line = values(e.result["centre"][:, index])[:, row]
                    if block == "euler":
                        line = line - 360.0 * np.round((line[0] - truth[START_INDEX, row]) / 360.0)
                axis.plot(t_open, plottable(line), color=COLORS[e.label], lw=1.2)
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
                axis.axvline(t_all[START_INDEX] + mark, color="0.5", lw=0.5, ls="--")
            axis.grid(alpha=0.25)
            axis.tick_params(labelsize=6)
            if col == 0:
                axis.set_ylabel(labels[row], fontsize=7)
            if row == 0:
                axis.set_title(e.label.replace("-wind-", "-\nwind-"), color=COLORS[e.label], weight="bold", fontsize=7)
            if row == 2:
                axis.set_xlabel("t (s)", fontsize=7)
    figure.suptitle(f"Evaluation trajectory {index}: {title}. Black = clean data, every model starts from its ground-truth state at "
                    "t = 0, line = mean of the sample paths\n(single path if none), dark = +- 1 sigma, light = +- 2 sigma, "
                    "dashed = 1 / 3 / 5 s after the start; 'clipped' = y-axis capped at the data range +- 3x its width", fontsize=9)
    figure.tight_layout()
    return figure


def energy_page(data: Dataset, entries, index, marks) -> plt.Figure:
    """True Hamiltonian H(t) of the data (black) and of every model's prediction (line = mean of H over the sample
    paths, dark / light = +- 1 / 2 sigma of H over the paths), one panel per model."""
    columns = 5 if len(entries) > 6 else max(len(entries), 2)
    rows = math.ceil(len(entries) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16.5 if columns == 5 else PAGE[0], PAGE[1]), sharex=True,
                                sharey=True, squeeze=False)
    t = data.t[START_INDEX:END_INDEX]
    truth = true_energy(data.clean[START_INDEX:END_INDEX, index].astype(np.float64), data.truth)
    low, high = np.nanmin(truth), np.nanmax(truth)
    extent = [low, high]                     # shared y-range: data and every prediction, capped at data +- 1x its width
    for e in entries:
        if not e.failed:
            finite = e.result["energy"][:, index][np.isfinite(e.result["energy"][:, index])]
            if finite.size:
                extent = [min(extent[0], finite.min()), max(extent[1], finite.max())]
    reach = 1.0 * max(high - low, 1e-3)
    extent = [max(extent[0], low - reach), min(extent[1], high + reach)]
    for k, axis in enumerate(axes.ravel()):
        if k >= len(entries):
            axis.axis("off")
            continue
        e = entries[k]
        axis.plot(t, truth, color="k", lw=0.9)
        if e.failed:
            axis.text(0.5, 0.5, "training failed", transform=axis.transAxes, ha="center", color="0.4")
        else:
            paths = e.result["paths"]
            if paths is not None:
                values = true_energy(paths[:, :, index], data.truth)
                with np.errstate(invalid="ignore"):
                    sd = np.nanstd(np.where(np.isfinite(values), values, np.nan), axis=0, ddof=1)
                mu = e.result["energy"][:, index]
                axis.fill_between(t, plottable(mu - 2 * sd), plottable(mu + 2 * sd), color=COLORS[e.label], alpha=0.15, lw=0)
                axis.fill_between(t, plottable(mu - sd), plottable(mu + sd), color=COLORS[e.label], alpha=0.30, lw=0)
            axis.plot(t, plottable(e.result["energy"][:, index]), color=COLORS[e.label], lw=1.2)
        for mark in marks[:-1]:
            axis.axvline(t[0] + mark, color="0.5", lw=0.5, ls="--")
        axis.set_title(e.label, color=COLORS[e.label], weight="bold", fontsize=8)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=6)
        if k % columns == 0:
            axis.set_ylabel("H (J)", fontsize=7)
        if k >= len(entries) - columns:
            axis.set_xlabel("t (s)", fontsize=7)
    pad = 0.08 * max(extent[1] - extent[0], 1e-3)
    axes[0, 0].set_ylim(extent[0] - pad, extent[1] + pad)
    figure.suptitle(f"Evaluation trajectory {index}: true Hamiltonian H = 1/2 omega^T J omega + m g l (R e_z).e_z along the data "
                    "(black) and the prediction (line = mean of H over the sample paths,\ndark / light = +- 1 / 2 sigma); shared y-axis "
                    "covers the data and the predictions, capped at the data range +- 1x its width", fontsize=9)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------
def report(setting: str, level: float, samples: int, seed: int, records: dict, output_root: Path) -> Path:
    data = Dataset(setting, level)
    entries = [e for e in entries_for(setting)                       # optional models only where a run is recorded
               if e.source not in OPTIONAL_MODELS or run_name(e.source, setting, level) in records]
    print(f"[report] {setting} noise {level:g}: {data.path.name}", flush=True)
    sources = predict_entries(data, entries, records, samples, seed)
    t = data.t[START_INDEX:END_INDEX] - data.t[START_INDEX]
    marks = [m for m in HORIZON_MARKS if m <= t[-1] + 1e-9]
    tables = {e.label: table_values(e.result, marks, data.interval) for e in entries if not e.failed}
    count = data.clean.shape[1]
    out_dir = output_root / f"{setting}_noise{level:g}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "open-loop-horizons-comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        training_note = None
        if any(e.source == "NeuralSDE" and not e.failed for e in entries):
            training_note = textwrap.fill(
                f"training    {data.train_shape[0] * data.train_shape[2]} trajectories x {data.train_shape[1]} samples, noisy. "
                "Port-Hamiltonian models: 2 s windows, 25 % overlap; 5000 steps, final model; float32; no priors, no pretraining. "
                "NeuralSDE: its own recipe (unstructured drift + diffusion MLPs 500 x 2, Stratonovich Heun 10 substeps, path MSE + "
                "geodesic loss on 5-point windows, full batch, AdamW 1e-3, clip 1), 5000 steps, final model, float32",
                width=150, subsequent_indent=" " * 12)
        pdf.savefig(physics_page(data, entries, sources, training_note=training_note)); plt.close("all")
        pdf.savefig(protocol_page(data, entries, samples)); plt.close("all")
        for mark in marks:
            key = f"0-{mark:g}s"
            pdf.savefig(table_page(entries, tables, key, f"Table A - open-loop prediction error, 0-{mark:g} s after the start",
                                   f"{data.path.name}: mean +- std over {count} evaluation trajectories. Prediction = mean of the "
                                   "sample paths (single path for -wind-same and point models).", None)); plt.close("all")
        pdf.savefig(error_page(t, entries, marks)); plt.close("all")
        pdf.savefig(bar_page(entries, tables, marks)); plt.close("all")
        pdf.savefig(calibration_page(t, entries, marks)); plt.close("all")
        for index in range(count):
            for mark in marks:
                key = f"0-{mark:g}s"
                pdf.savefig(table_page(entries, tables, key, f"Trajectory {index} - open-loop prediction error, 0-{mark:g} s after the start",
                                       f"{data.path.name}, evaluation trajectory {index} alone. Prediction = mean of the sample paths "
                                       "(single path for -wind-same and point models).", index)); plt.close("all")
            pdf.savefig(path3d_page(data, entries, index)); plt.close("all")
            for block in ("b", "euler", "w"):
                pdf.savefig(block_page(data, entries, index, block, marks)); plt.close("all")
            pdf.savefig(energy_page(data, entries, index, marks)); plt.close("all")
    summary = {"generated_at": datetime.now().astimezone().isoformat(), "dataset": str(data.path), "setting": setting,
               "noise": level, "trajectories": count, "open_loop_seconds": float(t[-1]), "samples": samples,
               "models": {e.label: {"failed": e.failed, "start": e.note, "band": BAND_TEXT[e.band],
                                    "run_dir": str(e.run_dir) if e.run_dir else None} for e in entries},
               "table": {e.label: None if e.failed else {key: {row: None if v is None else
                                                                {"mean": float(np.mean(v)), "std": float(np.std(v)),
                                                                 "per_trajectory": [float(x) for x in v]}
                                                                for row, v in rows.items()}
                                                          for key, rows in tables[e.label].items()} for e in entries}}
    (out_dir / "open-loop-horizons-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[report] wrote {pdf_path}", flush=True)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=tuple(SETTINGS))
    parser.add_argument("--noise", type=float)
    parser.add_argument("--all", action="store_true", help="every setting and noise level")
    parser.add_argument("--settings", nargs="+", choices=tuple(SETTINGS), help="with --all: only these settings")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="default: <campaign folder>/reports")
    args = parser.parse_args()
    records = campaign_records(RESULT_DIR)
    output = args.output or RESULT_DIR / "reports"
    if args.all:
        jobs = [(s, n) for s in (args.settings or SETTINGS) for n in NOISE_LEVELS]
    else:
        if args.setting is None or args.noise is None:
            parser.error("--setting and --noise, or --all")
        jobs = [(args.setting, args.noise)]
    for setting, level in jobs:
        report(setting, level, args.samples, args.seed, records, output)


if __name__ == "__main__":
    main()

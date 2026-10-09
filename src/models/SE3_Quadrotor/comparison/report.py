"""Open-loop report of the quadrotor campaign (4-5 Oct 2026): one PDF per dataset, on the 20 HELD-OUT flights.

    python src/models/SE3_Quadrotor/comparison/report.py --setting DampRate-Wind

Written to experiments/quadrotor/campaign_06-10-2026/reports/<setting>_noise0p25/open-loop-horizons-comparison.pdf.
Same layout as the pendulum report (src/models/3D_SO3_Windy_Pendulum/comparison/report.py):
    page 1    Physics Identification: dataset specs, gauge-invariant products of every trained model vs the truth
              (from each run's evaluation_final.json), best learned value per column in green bold
    page 2    protocol and models
    pages 3-5 Table A for 0-1, 0-3, 0-5 s: mean +- std over the 20 held-out flights
    then      error vs time, time-averaged error per horizon, calibration vs time
    then per flight: Table A for the three horizons (that flight alone), 3-D path, position, attitude (Euler xyz),
              body velocity, body angular velocity, energy

Evaluation flights (5 Oct 2026, user: reports to 10 s): the 20 flights of the dataset's EVAL10s companion
(QUADROTOR-DATASET-<name>-EVAL10s: same plant, wind, PID and eval library as the held-out split, 10 s, never used in
training), with the wind recorded as interval means (gust_force_mean / gust_torque_mean).
Protocol: every model starts from the GROUND-TRUTH state at t = 0 and runs open loop over 10 s (500 intervals of
0.02 s) with the recorded wrench (scaled by the training RMS, as in training), no feedback; scored against the clean
flight over 0-1, 0-3, 0-5, 0-10 s. Sample paths (32): GP weight sample x Brownian path (GP-SDE), Brownian path
(NN-SDE), weight samples only (GP-ODE), the true diffusion (Analytical-SDE-wind-different). -wind-same: ONE path,
posterior-mean model, the learned diffusion replaced by the data's own wind, twist increment per interval
[F_mean / m, J^-1 tau_mean] dt (m and J only convert the recorded force to a twist increment; evaluation only).
Point models: NN-ODE, PH-NODE, Analytical-ODE (true drift, wind off). Energy = the TRUE Hamiltonian
H = 1/2 m |v|^2 + 1/2 w^T J w + m g z on the predicted and the true state; error |E[H(x_hat)] - H(x)|.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent):
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

from comparison.campaign import BASE_CONFIG, CAMPAIGN_MODELS, MODELS, NOISE, RESULT_DIR, SETTINGS, TRAINING_NOTE, dataset_path, run_name  # noqa: E402
from comparison.models import build_model, load_run, rollout_function  # noqa: E402
from lie_ph.data import control_scale  # noqa: E402
from lie_ph.evaluate import analytical_model, analytical_params, load_truth  # noqa: E402
from lie_ph.integrator import lie_imex_increment_step, rollout  # noqa: E402
from lie_ph.network import build_gp_setup  # noqa: E402

HORIZON_MARKS = (1.0, 3.0, 5.0, 10.0)
PAGE = (11.69, 8.27)
COLORS = {"Lie-PH-GP-SDE": "tab:orange", "Lie-PH-NN-SDE": "tab:cyan", "Lie-PH-GP-ODE": "tab:red",
          "Lie-PH-NN-ODE": "tab:blue", "PH-NODE": "tab:pink", "Lie-PH-GP-SDE-wind-same": "tab:brown",
          "Lie-PH-NN-SDE-wind-same": "tab:olive", "Analytical-SDE-wind-same": "tab:green",
          "Analytical-SDE-wind-different": "tab:purple", "Analytical-ODE": "0.35"}
BAND_TEXT = {"sample_gp_sde": "32 paths: posterior weight sample x Brownian path of the learned diffusion",
             "sample_nn_sde": "32 paths: Brownian paths of the learned diffusion",
             "sample_gp_ode": "32 paths: posterior weight samples (epistemic only, no wind)",
             "point": "none: one deterministic path",
             "same": "none: ONE path, posterior-mean model driven by the data's own (recorded, interval-mean) wind",
             "sample_truth": "32 paths: Brownian paths of the TRUE diffusion (a new wind draw)",
             "point_truth": "none: the true drift with the wind switched off"}
BLOCKS = ("p", "R", "v", "w", "H")
BLOCK_NAMES = {"p": "position error |p - p_true| (m)", "R": "attitude error (rad)", "v": "body velocity error (m/s)",
               "w": "body angular-velocity error (rad/s)", "H": "energy error |H(x_hat) - H(x)| (J, true H)"}
UQ_NAMES = {"p": ("Position", "m"), "R": ("Attitude", "rad"), "v": ("Velocity", "m/s"), "w": ("Angular-velocity", "rad/s"),
            "H": ("Energy", "J")}
TABLE_ROWS = tuple(row for block in BLOCKS for row in (
    (block, "mean", f"{UQ_NAMES[block][0]} mean error ({UQ_NAMES[block][1]})"),
    (block, "final", f"{UQ_NAMES[block][0]} final error ({UQ_NAMES[block][1]})"))) + (
    ("det", "max", "Determinant max error"), ("orth", "max", "Orthogonality max error")) + tuple(
    row for block in BLOCKS for row in (
        (block, "crps", f"{UQ_NAMES[block][0]} CRPS ({UQ_NAMES[block][1]})"),
        (block, "cover1", f"{UQ_NAMES[block][0]} coverage mean +- 1 sigma"),
        (block, "cover2", f"{UQ_NAMES[block][0]} coverage mean +- 2 sigma"),
        (block, "spread", f"{UQ_NAMES[block][0]} spread-skill RMSE / sigma")))
ROW_TARGETS = {"cover1": 0.683, "cover2": 0.954, "spread": 1.0}


@dataclass
class Entry:
    label: str
    source: str                      # campaign model name or "analytical"
    mode: str                        # sample | point
    band: str
    group: str                       # different | same | reference (ranking group)
    failed: bool = False
    run_dir: Path | None = None
    result: dict = field(default_factory=dict)


def entries_for(setting: str, models: tuple | None = None) -> list[Entry]:
    """``models`` (optional): only these campaign models (+ the analytical references)."""
    out = []
    for model in sorted(MODELS, key=lambda m: not MODELS[m][1]):          # SDE models first
        if models is not None and model not in models:
            continue
        family, wind, _, _ = MODELS[model]
        if wind:
            out.append(Entry(model, model, "sample", f"sample_{family}_sde", "different"))
            out.append(Entry(f"{model}-wind-same", model, "same", "same", "same"))
        elif family == "gp":
            out.append(Entry(model, model, "sample", "sample_gp_ode", "different"))
        else:
            out.append(Entry(model, model, "point", "point", "different"))
    out.append(Entry("Analytical-SDE-wind-same", "analytical", "same", "same", "reference"))
    out.append(Entry("Analytical-SDE-wind-different", "analytical", "sample", "sample_truth", "reference"))
    out.append(Entry("Analytical-ODE", "analytical", "point", "point_truth", "reference"))
    return out


def campaign_records(result_dir: Path = RESULT_DIR) -> dict[str, dict]:
    records = {}
    for path in sorted(Path(result_dir).glob("results_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["name"]] = record
    return records


# ---------------------------------------------------------------------------------------------------------------
# geometry and metrics (numpy, float64)
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


def geodesic_deg(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    a = reference.reshape(*reference.shape[:-1], 3, 3)
    b = prediction.reshape(*prediction.shape[:-1], 3, 3)
    with np.errstate(invalid="ignore"):
        cosine = np.clip((np.trace(np.swapaxes(a, -1, -2) @ b, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def tangent_deg(rotation_flat: np.ndarray, reference_flat: np.ndarray) -> np.ndarray:
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
    shape = rotation_flat.shape[:-1]
    matrices = project(rotation_flat.reshape(-1, 9)).reshape(-1, 3, 3)
    angles = np.full((len(matrices), 3), np.nan)
    ok = np.all(np.isfinite(matrices), axis=(-2, -1))
    if np.any(ok):
        angles[ok] = Rotation.from_matrix(matrices[ok]).as_euler("xyz")
    angles = angles.reshape(*shape, 3)
    return np.where(np.isfinite(angles), np.degrees(np.unwrap(np.nan_to_num(angles), axis=-2)), np.nan)


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    count = samples.shape[0]
    x = np.sort(samples, axis=0)
    first = np.mean(np.abs(x - truth[None]), axis=0)
    weights = (2.0 * np.arange(1, count + 1) - count - 1).reshape((count,) + (1,) * (samples.ndim - 1))
    return first - np.sum(weights * x, axis=0) / (count * count)


def true_energy(states: np.ndarray, truth: dict) -> np.ndarray:
    """TRUE Hamiltonian of (..., 22) states: 1/2 m |v_b|^2 + 1/2 w^T J w + m g z."""
    with np.errstate(invalid="ignore", over="ignore"):
        v, w = states[..., 12:15], states[..., 15:18]
        return (0.5 * truth["mass"] * np.sum(v * v, axis=-1) + 0.5 * np.sum(truth["inertia"] * w * w, axis=-1)
                + truth["mass"] * truth["gravity"] * states[..., 2])


def predicted_energy(paths, single, truth) -> np.ndarray:
    if paths is None:
        return true_energy(single, truth)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(np.isfinite(paths).all(axis=-1), true_energy(paths, truth), np.nan), axis=0)


def error_series(truth_states, centre, energy, true_h) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore", over="ignore"):
        out = {"p": np.linalg.norm(centre[..., :3] - truth_states[..., :3], axis=-1),
               "R": np.radians(geodesic_deg(truth_states[..., 3:12], centre[..., 3:12])),      # rad (user, 6 Oct 2026)
               "v": np.linalg.norm(centre[..., 12:15] - truth_states[..., 12:15], axis=-1),
               "w": np.linalg.norm(centre[..., 15:18] - truth_states[..., 15:18], axis=-1),
               "H": np.abs(energy - true_h)}
    bad = ~np.isfinite(centre[..., :18]).all(axis=-1) | ~np.isfinite(energy)
    return {k: np.where(bad, np.inf, np.nan_to_num(v, nan=np.inf)) for k, v in out.items()}


def so3_violations(states: np.ndarray) -> dict[str, np.ndarray]:
    rotation = np.asarray(states[..., 3:12], dtype=np.float64).reshape(*states.shape[:-1], 3, 3)
    with np.errstate(invalid="ignore", over="ignore"):
        determinant = np.abs(np.linalg.det(np.nan_to_num(rotation, nan=1e6)) - 1.0)
        gram = np.swapaxes(rotation, -1, -2) @ rotation - np.eye(3)
        orthogonality = np.nan_to_num(np.sqrt(np.sum(gram * gram, axis=(-2, -1))), nan=np.inf)
    return {"det": np.where(np.isfinite(rotation).all(axis=(-2, -1)), determinant, np.inf), "orth": orthogonality}


def calibration(truth_states, paths, centre, energy, true_h, truth) -> dict:
    out = {}
    for block in BLOCKS:
        if block == "R":
            y = np.radians(tangent_deg(truth_states[..., 3:12], centre[..., 3:12]))           # rad
            x = None if paths is None else np.radians(tangent_deg(paths[..., 3:12], np.broadcast_to(centre[..., 3:12], paths[..., 3:12].shape)))
            c = np.zeros_like(y)
        elif block == "H":
            y, c = true_h[..., None], energy[..., None]
            x = None if paths is None else true_energy(paths, truth)[..., None]
        else:
            cols = {"p": slice(0, 3), "v": slice(12, 15), "w": slice(15, 18)}[block]
            y, x, c = truth_states[..., cols], None if paths is None else paths[..., cols], centre[..., cols]
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
# data and predictions
# ---------------------------------------------------------------------------------------------------------------
def eval_path(setting: str) -> Path:
    """The 10 s evaluation companion of a training dataset (trajectory_set eval, interval-mean wind recorded)."""
    name = SETTINGS[setting][0] + "-EVAL10s"
    return PROJECT_ROOT / "datasets" / f"QUADROTOR-DATASET-{name}" / f"{name}_CF2P_10s_h0p02_clean.pkl"


class Dataset:
    def __init__(self, setting: str):
        self.setting = setting
        self.path = PROJECT_ROOT / dataset_path(setting)  # the training (noisy) pickle: truth, scale
        with self.path.open("rb") as handle:
            payload = pickle.load(handle)
        self.settings = payload["settings"]
        self.scale = control_scale(payload["train_trajectories"], "rms")             # the training RMS (as train.py)
        self.train_count = int(payload["train_trajectories"].shape[0])
        self.test_count = int(payload["test_trajectories"].shape[0])
        self.truth = load_truth(self.path)
        self.eval_path = eval_path(setting)
        with self.eval_path.open("rb") as handle:
            evaluation = pickle.load(handle)
        self.flights = np.asarray(evaluation["heldout_trajectories"], np.float32)    # (F, T, 22), clean, 10 s
        self.clean = np.swapaxes(self.flights, 0, 1)                                  # time-major (T, F, 22)
        self.interval = float(evaluation["settings"]["sample_dt"])
        self.t = np.arange(self.clean.shape[0]) * self.interval
        # recorded wind as the twist increment of every interval (row k+1 = interval k -> k+1), body frame
        force = np.asarray(evaluation["heldout_gust_force_mean"], np.float64)
        torque = np.asarray(evaluation["heldout_gust_torque_mean"], np.float64)
        increment = np.concatenate([force / self.truth["mass"], torque / self.truth["inertia"]], axis=-1) * self.interval
        self.wind = np.swapaxes(increment, 0, 1).astype(np.float32)                  # (T, F, 6)


def wind_rollout(model, x0, controls, increments, interval: float, substeps: int):
    """(T, 22) path driven by given twist increments per interval (T-1, 6), split evenly over the substeps."""
    step = jnp.asarray(interval / substeps, x0.dtype)

    def body(state, inputs):
        control, increment = inputs
        state = state.at[18:22].set(control)
        for _ in range(substeps):
            state = lie_imex_increment_step(model, state, step, increment / substeps)
        return state, state

    _, path = jax.lax.scan(body, x0, (controls, increments))
    return jnp.concatenate([x0[None], path], axis=0)


def predict_entries(data: Dataset, entries: list[Entry], records: dict, samples: int, seed: int) -> dict:
    truth_t = data.clean.astype(np.float64)                                    # (T, F, 22)
    controls = jnp.asarray(np.swapaxes(data.clean[1:, :, 18:22] / data.scale, 0, 1))        # (F, T-1, 4), scaled
    start = jnp.asarray(data.clean[0])
    increments = jnp.asarray(np.swapaxes(data.wind[1:], 0, 1))                         # (F, T-1, 6)
    steps = truth_t.shape[0] - 1
    sources: dict[str, dict | None] = {}

    def source_context(source: str):
        if source in sources:
            return sources[source]
        if source == "analytical":
            config = yaml.safe_load(BASE_CONFIG.read_text())
            setup = build_gp_setup(config["model"], jax.random.PRNGKey(0))
            params = analytical_params(data.truth, setup, config["model"], data.scale)
            context = {"params": params, "factory": lambda p, key: analytical_model(p, setup, data.truth),
                       "substeps": int(config["data"]["substeps"]), "run_dir": None, "rollout": rollout}
        else:
            record = records.get(run_name(source, data.setting))
            run_dir = Path(record["run_dir"]) if record and record.get("run_dir") else None
            if record is None or record.get("status") != "done" or not (run_dir / "checkpoint_final.pkl").exists():
                sources[source] = None
                return None
            config, setup, params, payload = load_run(run_dir, "final")
            if not all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(params)):
                sources[source] = None
                return None
            if not np.allclose(np.asarray(payload["control_scale"]), data.scale, rtol=1e-4):
                raise ValueError(f"{source}: the run's control scale differs from the dataset's training RMS")
            context = {"params": params, "factory": lambda p, key, s=setup, c=config: build_model(p, s, key, c["model"]),
                       "substeps": int(config["data"]["substeps"]), "run_dir": run_dir, "rollout": rollout_function(config)}
        context["mean_model"] = context["factory"](context["params"], None)
        sources[source] = context
        return context

    for entry in entries:
        context = source_context(entry.source)
        if context is None:
            entry.failed = True
            continue
        entry.run_dir = context["run_dir"]
        substeps, model = context["substeps"], context["mean_model"]
        paths, raw_mean = None, None
        if entry.mode == "point":
            zero = jnp.zeros((steps, substeps, 6), jnp.float32)
            advance = context["rollout"]
            mean = jax.jit(jax.vmap(lambda x0, u: advance(model, x0, u, data.interval, zero)))(start, controls)
            raw_mean = np.swapaxes(np.asarray(mean), 0, 1).astype(np.float64)
        elif entry.mode == "same":
            mean = jax.jit(jax.vmap(lambda x0, u, i: wind_rollout(model, x0, u, i, data.interval, substeps)))(
                start, controls, increments)
            raw_mean = np.swapaxes(np.asarray(mean), 0, 1).astype(np.float64)
        else:
            params, factory, advance = context["params"], context["factory"], context["rollout"]

            @jax.jit
            def sampled(key):
                weight_key, noise_key = jax.random.split(key)
                sample_model = factory(params, weight_key)
                noise = jax.random.normal(noise_key, (start.shape[0], steps, substeps, 6), jnp.float32)
                return jax.vmap(lambda s, u, e: advance(sample_model, s, u, data.interval, e))(start, controls, noise)

            raw = np.stack([np.asarray(sampled(jax.random.PRNGKey(seed + i))) for i in range(samples)])
            paths = np.swapaxes(raw, 1, 2).astype(np.float64)                  # (S, T, F, 22)
        centre = ensemble_mean(paths) if paths is not None else raw_mean
        true_h = true_energy(truth_t, data.truth)
        energy = predicted_energy(paths, raw_mean, data.truth)
        entry.result = {"centre": centre, "paths": paths, "energy": energy,
                        "errors": error_series(truth_t, centre, energy, true_h),
                        "so3": so3_violations(paths if paths is not None else raw_mean),
                        "calib": calibration(truth_t, paths, centre, energy, true_h, data.truth)}
        print(f"[report] {entry.label:16s} done", flush=True)
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


def physics_page(data: Dataset, entries: list[Entry]) -> plt.Figure:
    truth, s = data.truth, data.settings
    vehicle = s["vehicle_parameters"]
    figure = plt.figure(figsize=PAGE)
    figure.text(0.03, 0.955, "Physics Identification", fontsize=17, weight="bold")
    figure.text(0.33, 0.957, f"quadrotor, {data.setting}, observation noise {NOISE:g}", fontsize=12)
    law = ("rate-dependent  F = -c m (1+|v_b|) v_b,  tau = -c J (1+|w_b|) w_b" if truth["damping_law"] == "nonlinear"
           else "constant  F = -c m v_b,  tau = -c J w_b")
    spec = [f"dataset     {data.path.name}",
            f"vehicle     Crazyflie 2.x (CF2P) in PyBullet {s['physics_hz']} Hz: m = {vehicle['mass']:g} kg, "
            f"J = diag({', '.join(f'{v:.3g}' for v in truth['inertia'])}) kg m^2, g = {vehicle['gravity_acceleration']:g}",
            f"damping     {law},  c = {truth['c_linear']:g} (linear and angular)",
            f"wind        constant diffusion on the body twist: Sigma = 0.5 per axis (m s^-1.5 / rad s^-1.5), new gust every 0.01 s",
            "control     recorded wrench u = [T, tau] of a PID tracking random segment chains (the PID reads 0.01-noise measurements);"
            " models see u / training RMS",
            f"noise       observations x + a, R Exp(b), v + c, w + d with a, b, c, d ~ N(0, {NOISE:g}^2 I)",
            f"sampling    dt = {data.interval:g} s ({1 / data.interval:g} Hz); training flights 5 s",
            textwrap.fill("training    " + TRAINING_NOTE.format(count=data.train_count), width=150, subsequent_indent=" " * 12),
            f"evaluation  {data.clean.shape[1]} held-out flights of {data.t[-1]:g} s ({data.eval_path.name}; eval library, never seen in"
            f" training), open loop from the ground-truth state for {data.t[-1]:g} s"]
    figure.text(0.03, 0.925, "\n".join(spec), fontsize=7.2, va="top", family="monospace")
    header = ["model", "thrust gain\n1/m (1/kg)", "torque map diag\n(xx, yy, zz)", "torque map\nrel. err",
              "gravity accel.\nrel. err", "M1^-1 D_v diag\n(mean, x / y / z)", "M1^-1 D_v\nrel. err",
              "M2^-1 D_w diag\n(mean, x / y / z)", "M2^-1 D_w\nrel. err", "Sigma\n(lin / ang mean)",
              "sigma_obs\n(p / R / v / w)", "test EKF\nNLL / d", "NIS / d"]
    diag3 = lambda m: " / ".join(f"{m[i][i]:.3g}" for i in range(3))
    rows, scores, failed_rows, reference = [], [], [], None
    seen = []
    for entry in entries:
        if entry.source == "analytical" or entry.source in seen:
            continue
        seen.append(entry.source)
        if entry.failed or entry.run_dir is None:
            failed_rows.append(len(rows) + 1)
            rows.append([f"{entry.source}\n(training failed)"] + ["-"] * (len(header) - 1))
            scores.append({})
            continue
        evaluation = json.loads((entry.run_dir / "evaluation_final.json").read_text())
        reference = reference or evaluation
        learned, ops = evaluation["learned"], evaluation["learned"]["operators"]
        sigma = learned.get("diffusion")
        obs = learned.get("sigma_obs")
        score = {1: ops["thrust_gain"]["relative_error"], 2: ops["torque_map"]["relative_error"],
                 3: ops["torque_map"]["relative_error"], 4: ops["gravity_acceleration"]["relative_error"],
                 5: ops["damping_linear_M1inv_Dv"]["relative_error"], 6: ops["damping_linear_M1inv_Dv"]["relative_error"],
                 7: ops["damping_angular_M2inv_Dw"]["relative_error"], 8: ops["damping_angular_M2inv_Dw"]["relative_error"]}
        if sigma:
            score[9] = float(np.linalg.norm(np.asarray(sigma) - truth["diffusion"]))
        if obs:
            score[10] = float(np.linalg.norm(np.asarray(obs) - NOISE))
        if "test_ekf_nll_noisy" in learned:
            score[11] = float(learned["test_ekf_nll_noisy"])
            score[12] = abs(float(learned["test_nis_per_dimension"]) - 1.0)
        scores.append({c: v for c, v in score.items() if np.isfinite(v)})
        rows.append([entry.source, f"{ops['thrust_gain']['mean_learned']:.3g} ({ops['thrust_gain']['relative_error'] * 100:.0f} %)",
                     "\n".join(f"{v:.3g}" for v in ops["torque_map"]["mean_learned_diag"]),
                     f"{ops['torque_map']['relative_error'] * 100:.1f} %",
                     f"{ops['gravity_acceleration']['relative_error'] * 100:.1f} %",
                     diag3(ops["damping_linear_M1inv_Dv"]["mean_learned"]),
                     f"{ops['damping_linear_M1inv_Dv']['relative_error'] * 100:.1f} %",
                     diag3(ops["damping_angular_M2inv_Dw"]["mean_learned"]),
                     f"{ops['damping_angular_M2inv_Dw']['relative_error'] * 100:.1f} %",
                     "-" if not sigma else f"{np.mean(sigma[:3]):.3g} / {np.mean(sigma[3:]):.3g}",
                     "-" if not obs else " / ".join(f"{v:.3g}" for v in obs),
                     f"{learned['test_ekf_nll_noisy']:.4f}" if "test_ekf_nll_noisy" in learned else "-",
                     f"{learned['test_nis_per_dimension']:.3f}" if "test_nis_per_dimension" in learned else "-"])
    true_ops = (reference or {}).get("learned", {}).get("operators", {})
    analytical = (reference or {}).get("analytical", {})
    truth_row = ["Ground truth", f"{1 / truth['mass']:.3g}", "\n".join(f"{v:.3g}" for v in 1.0 / truth["inertia"]), "0", "0",
                 diag3(true_ops["damping_linear_M1inv_Dv"]["true"]) if true_ops else "-", "0",
                 diag3(true_ops["damping_angular_M2inv_Dw"]["true"]) if true_ops else "-", "0",
                 f"{np.mean(truth['diffusion'][:3]):.3g} / {np.mean(truth['diffusion'][3:]):.3g}", f"{NOISE:g} (all)",
                 f"{analytical['test_ekf_nll_noisy']:.4f}" if "test_ekf_nll_noisy" in analytical else "-",
                 f"{analytical['test_nis_per_dimension']:.3f}" if "test_nis_per_dimension" in analytical else "-"]
    axis = figure.add_axes([0.005, 0.12, 0.99, 0.58])
    axis.axis("off")
    widths = [0.1, 0.075, 0.1, 0.06, 0.065, 0.095, 0.06, 0.095, 0.06, 0.07, 0.1, 0.06, 0.05]
    table = styled_table(axis, [truth_row] + rows, header, widths, 6.3, 3.4)
    for c in range(len(header)):
        table[1, c].set_facecolor("#e8e8e8")
    for r in failed_rows:
        for c in range(1, len(header)):
            table[r + 1, c].set_facecolor("#f6e0e0")
    for c in range(1, len(header)):
        candidates = [(score[c], r) for r, score in enumerate(scores) if c in score]
        if len(candidates) > 1:
            cell = table[min(candidates)[1] + 2, c]
            cell.set_facecolor("#c8ecc8")
            cell.set_text_props(weight="bold")
    notes = ("Only gauge-invariant products are identifiable. Values: posterior-mean model, mean over the clean TEST flights' states "
             "(evaluate.py); controls converted back to SI. Thrust gain = M1^-1 g_F (z, thrust); torque map = M2^-1 g_tau (1 / (kg m^2)); "
             "gravity accel. = -M1^-1 R^T grad V; damping rel. err = mean ||O_hat - O|| / mean ||O|| with the true O per state "
             "(c (1 + |.|) for rate-dependent damping). Sigma = learned twist diffusion, sigma_obs = learned observation noise "
             "('-' = the model has none). EKF NLL / d and NIS / d on the noisy test windows; the ground-truth row is the analytical "
             "model in the same filter. Green bold = best learned model per column (lowest rel. err; Sigma and sigma_obs closest "
             "to the truth; lowest NLL; NIS closest to 1).")
    figure.text(0.03, 0.02, "\n".join(textwrap.wrap(notes, 200)), fontsize=6.4)
    return figure


def protocol_page(data: Dataset, entries: list[Entry], samples: int) -> plt.Figure:
    lines = [f"dataset   {data.path.name} (training), evaluation flights {data.eval_path.name}",
             f"flights   {data.clean.shape[1]} held-out flights of {data.t[-1]:g} s ({data.clean.shape[0]} samples at {data.interval:g} s), "
             "eval library, never used in training",
             f"protocol  ONE nonstop open-loop rollout per flight from the GROUND-TRUTH state at t = 0 to {data.t[-1]:g} s with the",
             "          recorded wrench (scaled by the training RMS, as in training), no feedback. Scored against the clean flight",
             "          over 0-1, 0-3, 0-5, 0-10 s. Nothing is filtered: a diverged rollout keeps its (large or infinite) error.",
             "energy    the TRUE Hamiltonian H = 1/2 m |v_b|^2 + 1/2 w^T J w + m g z on the predicted and the true state; error",
             "          |E[H(x_hat)] - H(x)| with E over the sample paths (H is not conserved: thrust, damping and wind change it).",
             f"line/band {samples} sample paths: line = their mean (mean rotation projected onto SO(3)), dark = +- 1 sigma, light = +- 2 sigma.",
             "wind-same the data's own wind, recorded as interval means (two 0.01 s gusts per 0.02 s sample), replayed as the twist",
             "          increment [F_mean / m, J^-1 tau_mean] dt: the true model replays one interval to 0.008 per sqrt(s) (wind 0.5).",
             "", "models"]
    for e in entries:
        lines.append(f"  {e.label:16s} {'TRAINING FAILED -> - everywhere' if e.failed else 'band: ' + BAND_TEXT[e.band]}")
    lines += ["", "how to read",
              "  Analytical = the model class with every subnetwork set to the simulator's value (one Lie-IMEX step per sample).",
              "  -wind-same models are fed the data's own wind: Analytical-SDE-wind-same differs from the data only by the 1-step",
              "  integrator. With a new wind draw (-wind-different) even the true model drifts away: its error is the irreducible",
              "  open-loop error of the wind, and its band is the spread a perfect SDE must show. Green bold = best per row among",
              "  the learned new-wind models, and separately among the learned -wind-same models."]
    figure = plt.figure(figsize=PAGE)
    figure.text(0.04, 0.95, "Open-loop prediction on the held-out flights: protocol and models", fontsize=15, weight="bold")
    figure.text(0.04, 0.90, "\n".join(lines), fontsize=8.2, va="top", family="monospace")
    return figure


def fmt(value, block: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, tuple):
        mean, std = value
        return f"{mean:.2e} +- {std:.1e}" if block in ("det", "orth") else f"{mean:.4g} +- {std:.2g}"
    return f"{value:.2e}" if block in ("det", "orth") else f"{value:.4g}"


def table_page(entries, tables, key, title, subtitle, flight: int | None) -> plt.Figure:
    figure = plt.figure(figsize=PAGE)
    figure.text(0.03, 0.96, title, fontsize=14, weight="bold")
    figure.text(0.03, 0.935, subtitle, fontsize=7.2)
    header = ["metric"] + [e.label.replace("Lie-PH-", "Lie-PH-\n").replace("Analytical-", "Analytical-\n") for e in entries]
    cells, best = [], []
    for r, (block, kind, label) in enumerate(TABLE_ROWS):
        raw = []
        for e in entries:
            v = None if e.failed else tables[e.label][key].get(f"{block}_{kind}")
            if v is not None:
                v = (float(np.mean(v)), float(np.std(v))) if flight is None else float(v[flight])
            raw.append(v)
        target = ROW_TARGETS.get(kind)
        value = lambda i: raw[i][0] if isinstance(raw[i], tuple) else raw[i]
        score = lambda i: abs(value(i) - target) if target is not None else value(i)
        for group in ("different", "same"):
            ranked = [i for i, e in enumerate(entries) if e.group == group and raw[i] is not None and np.isfinite(score(i))]
            if len(ranked) > 1:
                best.append((r + 1, 1 + min(ranked, key=score)))
        cells.append([label] + [fmt(v, block) for v in raw])
    axis = figure.add_axes([0.01, 0.05, 0.98, 0.87])
    axis.axis("off")
    widths = [0.19] + [0.81 / len(entries)] * len(entries)
    styled_table(axis, cells, header, widths, 6.0, 1.08, best, failed_cols=[1 + i for i, e in enumerate(entries) if e.failed])
    figure.text(0.03, 0.012, "Green bold = best per row among the learned new-wind models, and separately among the learned -wind-same models; Analytical columns are references, not ranked. Best = lowest, except "
                "coverage (closest to 0.683 / 0.954) and spread-skill (closest to 1). CRPS of a single path = its absolute error; '-' = no "
                "samples or training failed (red column).\nMean error = time average over (0, H]; final = at H; SO(3) rows = max over (0, H] "
                "on the RAW rollout. Uncertainty rows: time average over (0, H].", fontsize=5.8)
    return figure


def error_page(t, entries, marks) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=PAGE)
    axes = axes.ravel()
    for axis, key in zip(axes, BLOCKS):
        for e in entries:
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
    axes[0].legend(fontsize=6.5)
    axes[-1].axis("off")
    figure.suptitle("Open-loop error vs time on the held-out flights: median over flights, shade = inter-quartile, dashed = analytical",
                    fontsize=10)
    figure.tight_layout()
    return figure


def bar_page(entries, tables, marks) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=PAGE)
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
        axis.set_title("time-averaged " + BLOCK_NAMES[key], fontsize=8)
        axis.grid(alpha=0.3, axis="y", which="both")
    axes[0].legend(fontsize=6.5)
    axes[-1].axis("off")
    figure.suptitle("Time-averaged error over 0-1 / 0-3 / 0-5 s (median over held-out flights; hatched = analytical)", fontsize=10)
    figure.tight_layout()
    return figure


def calibration_page(t, entries, marks) -> plt.Figure:
    figure, axes = plt.subplots(4, 5, figsize=(16.5, PAGE[1]), sharex=True)
    names = {"p": "position", "R": "attitude (tangent)", "v": "body velocity", "w": "angular velocity", "H": "energy (true H)"}
    for col, key in enumerate(BLOCKS):
        for e in entries:
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
    figure.suptitle("Calibration of the sample paths vs time (mean over held-out flights and axes). Dotted = ideal 0.683, 0.954, 1. "
                    "Spread-skill > 1 = over-confident. CRPS of a single path = its absolute error.", fontsize=9)
    figure.tight_layout()
    return figure


def path3d_page(data, entries, index) -> plt.Figure:
    columns = 4
    figure = plt.figure(figsize=(14.5, PAGE[1]))
    flight = data.flights[index].astype(np.float64)
    lo, hi = flight[:, :3].min(axis=0) - 0.5, flight[:, :3].max(axis=0) + 0.5
    for k, e in enumerate(entries):
        axis = figure.add_subplot(math.ceil(len(entries) / columns), columns, k + 1, projection="3d")
        axis.plot(*flight[:, :3].T, color="k", lw=1.2)
        if e.failed:
            axis.set_title(f"{e.label}\ntraining failed", fontsize=8, color="0.4")
        else:
            if e.result["paths"] is not None:
                for path in e.result["paths"][:8, :, index]:
                    axis.plot(*plottable(path[:, :3]).T, color=COLORS[e.label], lw=0.4, alpha=0.35)
            centre = e.result["centre"][:, index]
            axis.plot(*plottable(centre[:, :3]).T, color=COLORS[e.label], lw=1.5)
            axis.scatter(*flight[0, :3], color="k", s=8)
            axis.set_title(e.label, fontsize=8, color=COLORS[e.label], weight="bold")
        axis.set_xlim(lo[0], hi[0])
        axis.set_ylim(lo[1], hi[1])
        axis.set_zlim(lo[2], hi[2])
        axis.tick_params(labelsize=5)
    figure.suptitle(f"Held-out flight {index}: 3-D path (black = data, bold = prediction from the true x_0, thin = 8 sample paths)",
                    fontsize=10)
    figure.tight_layout()
    return figure


def block_page(data, entries, index, block, marks) -> plt.Figure:
    names = {"p": (["x (m)", "y (m)", "z (m)"], "Position (world)"),
             "euler": (["roll (rad)", "pitch (rad)", "yaw (rad)"], "Attitude (Euler xyz of R, unwrapped)"),
             "v": (["v_x (m/s)", "v_y (m/s)", "v_z (m/s)"], "Body linear velocity"),
             "w": (["w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"], "Body angular velocity")}
    labels, title = names[block]

    def values(states):
        if block == "euler":
            return np.radians(euler_deg(states[..., 3:12]))                        # rad
        return states[..., {"p": slice(0, 3), "v": slice(12, 15), "w": slice(15, 18)}[block]]

    figure, axes = plt.subplots(3, len(entries), figsize=(max(PAGE[0], 2.4 * len(entries)), PAGE[1]), sharex=True, squeeze=False)
    t = data.t
    truth = values(data.flights[index].astype(np.float64))
    for col, e in enumerate(entries):
        for row in range(3):
            axis = axes[row, col]
            axis.plot(t, truth[:, row], color="k", lw=0.9)
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
                    if block == "euler":
                        sample_values = sample_values - 2 * np.pi * np.round((sample_values[:, :1] - truth[0, row]) / (2 * np.pi))
                    with np.errstate(invalid="ignore"):
                        mu, sd = np.nanmean(sample_values, axis=0), np.nanstd(sample_values, axis=0, ddof=1)
                    axis.fill_between(t, plottable(mu - 2 * sd), plottable(mu + 2 * sd), color=COLORS[e.label], alpha=0.15, lw=0)
                    axis.fill_between(t, plottable(mu - sd), plottable(mu + sd), color=COLORS[e.label], alpha=0.30, lw=0)
                    line = mu
                else:
                    line = values(e.result["centre"][:, index])[:, row]
                    if block == "euler":
                        line = line - 2 * np.pi * np.round((line[0] - truth[0, row]) / (2 * np.pi))
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
    figure.suptitle(f"Held-out flight {index}: {title}. Black = clean data; line = mean of the sample paths (single path if none), "
                    "dark = +- 1 sigma, light = +- 2 sigma;\ndashed = 1 / 3 s; 'clipped' = y-axis capped at the data range +- 3x its width",
                    fontsize=9)
    figure.tight_layout()
    return figure


def energy_page(data, entries, index, marks) -> plt.Figure:
    columns = 4
    rows = math.ceil(len(entries) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(14.5, PAGE[1]), sharex=True, sharey=True, squeeze=False)
    t = data.t
    truth = true_energy(data.flights[index].astype(np.float64), data.truth)
    low, high = np.nanmin(truth), np.nanmax(truth)
    extent = [low, high]
    for e in entries:
        if not e.failed:
            finite = e.result["energy"][:, index][np.isfinite(e.result["energy"][:, index])]
            if finite.size:
                extent = [min(extent[0], finite.min()), max(extent[1], finite.max())]
    reach = 1.0 * max(high - low, 1e-4)
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
            if e.result["paths"] is not None:
                values = true_energy(e.result["paths"][:, :, index], data.truth)
                with np.errstate(invalid="ignore"):
                    sd = np.nanstd(np.where(np.isfinite(values), values, np.nan), axis=0, ddof=1)
                mu = e.result["energy"][:, index]
                axis.fill_between(t, plottable(mu - 2 * sd), plottable(mu + 2 * sd), color=COLORS[e.label], alpha=0.15, lw=0)
                axis.fill_between(t, plottable(mu - sd), plottable(mu + sd), color=COLORS[e.label], alpha=0.30, lw=0)
            axis.plot(t, plottable(e.result["energy"][:, index]), color=COLORS[e.label], lw=1.2)
        for mark in marks[:-1]:
            axis.axvline(mark, color="0.5", lw=0.5, ls="--")
        axis.set_title(e.label, color=COLORS[e.label], weight="bold", fontsize=8)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=6)
        if k % columns == 0:
            axis.set_ylabel("H (J)", fontsize=7)
    pad = 0.08 * max(extent[1] - extent[0], 1e-4)
    axes[0, 0].set_ylim(extent[0] - pad, extent[1] + pad)
    figure.suptitle(f"Held-out flight {index}: true Hamiltonian H = 1/2 m |v|^2 + 1/2 w^T J w + m g z of the data (black) and the "
                    "prediction (line = mean of H over the sample paths, dark / light = +- 1 / 2 sigma)", fontsize=9)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------
def report(setting: str, samples: int, seed: int, output_root: Path | None = None,
           models: tuple | None = CAMPAIGN_MODELS) -> Path:
    data = Dataset(setting)
    entries = entries_for(setting, models)
    records = campaign_records(RESULT_DIR)
    output_root = output_root or RESULT_DIR / "reports"
    print(f"[report] {setting}: {data.path.name}, {data.clean.shape[1]} held-out flights", flush=True)
    predict_entries(data, entries, records, samples, seed)
    t = data.t
    marks = [m for m in HORIZON_MARKS if m <= t[-1] + 1e-9]
    tables = {e.label: table_values(e.result, marks, data.interval) for e in entries if not e.failed}
    count = data.clean.shape[1]
    out_dir = output_root / f"{setting}_noise{NOISE:g}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "open-loop-horizons-comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(physics_page(data, entries)); plt.close("all")
        pdf.savefig(protocol_page(data, entries, samples)); plt.close("all")
        for mark in marks:
            key = f"0-{mark:g}s"
            pdf.savefig(table_page(entries, tables, key, f"Table A - open-loop prediction error on the held-out flights, 0-{mark:g} s",
                                   f"{data.path.name}: mean +- std over {count} held-out flights, every model from the ground-truth "
                                   "state at t = 0. Prediction = mean of the sample paths (single path for point models).", None))
            plt.close("all")
        pdf.savefig(error_page(t, entries, marks)); plt.close("all")
        pdf.savefig(bar_page(entries, tables, marks)); plt.close("all")
        pdf.savefig(calibration_page(t, entries, marks)); plt.close("all")
        for index in range(count):
            for mark in marks:
                key = f"0-{mark:g}s"
                pdf.savefig(table_page(entries, tables, key, f"Held-out flight {index} - open-loop prediction error, 0-{mark:g} s",
                                       f"{data.path.name}, held-out flight {index} alone.", index)); plt.close("all")
            pdf.savefig(path3d_page(data, entries, index)); plt.close("all")
            for block in ("p", "euler", "v", "w"):
                pdf.savefig(block_page(data, entries, index, block, marks)); plt.close("all")
            pdf.savefig(energy_page(data, entries, index, marks)); plt.close("all")
    summary = {"generated_at": datetime.now().astimezone().isoformat(), "dataset": str(data.path), "setting": setting,
               "flights": count, "samples": samples,
               "models": {e.label: {"failed": e.failed, "band": BAND_TEXT[e.band], "run_dir": str(e.run_dir) if e.run_dir else None}
                          for e in entries},
               "table": {e.label: None if e.failed else {key: {row: None if v is None else
                                                                {"mean": float(np.mean(v)), "std": float(np.std(v)),
                                                                 "per_flight": [float(x) for x in v]}
                                                                for row, v in rows.items()}
                                                          for key, rows in tables[e.label].items()} for e in entries}}
    (out_dir / "open-loop-horizons-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[report] wrote {pdf_path}", flush=True)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", nargs="+", required=True, choices=tuple(SETTINGS))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="default: <campaign folder>/reports")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(CAMPAIGN_MODELS),
                        help="learned models to show (default: the campaign's Lie-PH-GP-SDE and PH-NODE)")
    args = parser.parse_args()
    for setting in args.setting:
        report(setting, args.samples, args.seed, args.output, tuple(args.models))


if __name__ == "__main__":
    main()

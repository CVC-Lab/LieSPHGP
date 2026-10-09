"""Closed-loop tracking report of the quadrotor campaign (5 Oct 2026): every model flies the first 10 held-out flights
of the dataset's 10 s evaluation companion (report.eval_path: QUADROTOR-DATASET-<name>-EVAL10s, eval library, never
used in training; the user asked for 10 s reports)
as references, 5 wind seeds each, in the dataset's own PyBullet simulator.

    JAX_PLATFORMS=cpu python src/models/SE3_Quadrotor/comparison/closed_loop.py --setting DampRate-Wind

Written to experiments/quadrotor/campaign_06-10-2026/reports/<setting>_noise0p25/closed-loop-comparison.pdf
(+ closed-loop-summary.json, closed-loop-flights.npz cache).

Plant = the generator's own simulator (envs/quadrotor_se3_pybullet/datagen/generate_dataset.py): CF2P in PyBullet at
the dataset's physics rate, contact-free, its damping law (constant or rate-dependent), its wind (new gust every
0.01 s, seed-dependent), the motor mixer and the rpm limit; the vehicle starts in the held-out flight's recorded x_0.

Controller (the quadrotor booklet's energy-based geometric tracking law, written with the model's gauge-invariant
quantities only), at CONTROL_HZ:
    f = model d(v_b, w)/dt at u = 0 (gravity, damping, Coriolis),  B = blockdiag(M1^-1, M2^-1) g   (6 x 4, per scaled u)
    position loop    a_w = a_d - K_p (p - p_d) - K_v (v_w - v_d),   thrust acceleration a_T = R (R^T a_w - w x v_b - f_v)
                     (tilt-limited to 0.698 rad = 40 deg), b3d = a_T / |a_T|, R_d = [b2d x b3d, b2d, b3d] with the yaw reference
    attitude loop    e_R = 1/2 vee(R_d^T R - R^T R_d), e_w = w - R^T R_d w_d (w_d from R_d by finite difference)
                     dw_cmd = -K_R e_R - K_w e_w
    allocation       rows (v_z, w_x, w_y, w_z) of B u = [ (R^T a_T)_z ; dw_cmd - f_w ],  u = lstsq(...) * training RMS
The gains are the booklet's (K_p = (10, 10, 50), K_v = 3, K_R = 250, K_w = 20, in acceleration units here), the same
for every model. Only the drift is used, so no -wind-same models. A flight fails when its state or control becomes
non-finite or the vehicle is more than 10 m from the reference (then it is frozen).
Reference = the held-out flight: p_d(t), v_d = R v_b, a_d = smoothed finite difference, yaw = atan2(R_10, R_00)
unwrapped, linearly interpolated to the control rate.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import sys
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent, PROJECT_ROOT / "envs" / "quadrotor_se3_pybullet" / "datagen"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as pb  # noqa: E402
import yaml  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import generate_dataset as gd  # noqa: E402
from comparison import report as rp  # noqa: E402
from comparison.campaign import BASE_CONFIG, CAMPAIGN_MODELS, MODELS, NOISE, RESULT_DIR, SETTINGS, run_name  # noqa: E402
from comparison.models import build_model, load_run  # noqa: E402
from lie_ph.evaluate import analytical_model, analytical_params  # noqa: E402
from lie_ph.integrator import log_so3  # noqa: E402
from lie_ph.network import build_gp_setup  # noqa: E402

FLIGHTS = 10
SEEDS = 5
SEED_BASE = 90000
CONTROL_HZ = 200
K_P = np.asarray([10.0, 10.0, 50.0])
K_V = np.asarray([3.0, 3.0, 3.0])
K_R, K_W = 250.0, 20.0
MAX_TILT = math.radians(40.0)
ESCAPE = 10.0                       # m from the reference: the flight is frozen and counted as failed
VALID_FRACTION = 0.158              # the booklet's valid-tracking threshold (x RMS reference excursion)
HORIZON_MARKS = (1.0, 3.0, 5.0, 10.0)
ANALYTICAL = "Analytical"
LEARNED = ("Lie-PH-GP-SDE", "Lie-PH-NN-SDE", "Lie-PH-GP-ODE", "Lie-PH-NN-ODE", "PH-NODE")
COLORS = {**rp.COLORS, ANALYTICAL: "tab:green"}
REF_COLOR = "chocolate"
ROWS = (("p_rmse", "Position RMSE (m)", "low"), ("p_max", "Position max error (m)", "low"),
        ("p_final", "Position final error (m)", "low"), ("v_rmse", "Velocity RMSE (m/s, world)", "low"),
        ("w_rmse", "Angular-velocity RMSE (rad/s, body, vs reference flight)", "gt"), ("w_rms", "RMS body rate |w| (rad/s)", "gt"),
        ("p_crps", f"Position CRPS over the {SEEDS} wind seeds (m)", "low"),
        ("w_crps", f"Angular-velocity CRPS over the {SEEDS} wind seeds (rad/s)", "gt"),
        ("yaw_rmse", "Yaw RMSE (rad)", "low"), ("att_cmd", "Attitude RMSE vs commanded R_d (rad)", "low"),
        ("H_rmse", "Energy error RMSE |H - H_ref| (J)", "low"),
        ("valid", "Valid tracking time vs Analytical (s)", "high"), ("thrust_rms", "RMS thrust / hover", "gt"),
        ("chatter", "Control chattering (1/s)", "low"), ("tilt_max", "Max tilt (rad)", "low"),
        ("sat", "Motor saturation fraction", "low"), ("completion", "Completion fraction", "high"),
        ("fail", "Failures (failed / flown)", "none"))


# ---------------------------------------------------------------------------------------------------------------
# models and controller
# ---------------------------------------------------------------------------------------------------------------
def load_sources(data: rp.Dataset, records: dict, learned: tuple = LEARNED) -> dict:
    sources = {}
    for model in learned:
        record = records.get(run_name(model, data.setting))
        run_dir = Path(record["run_dir"]) if record and record.get("run_dir") else None
        if record is None or record.get("status") != "done" or not (run_dir / "checkpoint_final.pkl").exists():
            sources[model] = None
            continue
        config, setup, params, _ = load_run(run_dir, "final")
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(params)):
            sources[model] = None
            continue
        sources[model] = {"model": build_model(params, setup, None, config["model"]), "run_dir": run_dir}
    config = yaml.safe_load(BASE_CONFIG.read_text())
    setup = build_gp_setup(config["model"], jax.random.PRNGKey(0))
    params = analytical_params(data.truth, setup, config["model"], data.scale)
    sources["analytical"] = {"model": analytical_model(params, setup, data.truth), "run_dir": None}
    return sources


def make_controller(model, scale: np.ndarray):
    """Batched controller: (F, 22) states, references (F, 3) p_d, v_d, a_d, (F,) yaw, (F, 3, 3) previous R_d ->
    (SI wrench (F, 4), R_d (F, 3, 3), requested tilt (F,))."""
    scale = jnp.asarray(scale, jnp.float32)
    k_p, k_v = jnp.asarray(K_P, jnp.float32), jnp.asarray(K_V, jnp.float32)
    dt = 1.0 / CONTROL_HZ

    def one(state, p_d, v_d, a_d, yaw, rd_prev):
        rotation = state[3:12].reshape(3, 3)
        v_b, omega = state[12:15], state[15:18]
        drift = model.vector_field(jnp.concatenate([state[:18], jnp.zeros(4, state.dtype)]))[12:18]
        gain = jnp.concatenate([model.inverse_mass_1(state[:3]) @ model.control_matrix(state[3:12], state[:3])[:3],
                                model.inverse_mass_2(state[3:12]) @ model.control_matrix(state[3:12], state[:3])[3:]])
        v_w = rotation @ v_b
        a_w = a_d - k_p * (state[:3] - p_d) - k_v * (v_w - v_d)
        thrust_body = rotation.T @ a_w - jnp.cross(omega, v_b) - drift[:3]          # needed thrust acceleration (body)
        thrust_world = rotation @ thrust_body
        tilt = jnp.arctan2(jnp.linalg.norm(thrust_world[:2]), jnp.maximum(thrust_world[2], 1e-6))
        horizontal = jnp.linalg.norm(thrust_world[:2])
        limit = jnp.tan(MAX_TILT) * jnp.maximum(thrust_world[2], 1e-3)
        factor = jnp.where(horizontal > limit, limit / jnp.maximum(horizontal, 1e-9), 1.0)
        thrust_world = jnp.concatenate([thrust_world[:2] * factor, jnp.maximum(thrust_world[2:], 1e-3)])
        b3 = thrust_world / jnp.linalg.norm(thrust_world)
        heading = jnp.stack([jnp.cos(yaw), jnp.sin(yaw), 0.0])
        b2 = jnp.cross(b3, heading)
        b2 = b2 / jnp.maximum(jnp.linalg.norm(b2), 1e-9)
        b1 = jnp.cross(b2, b3)
        rd = jnp.stack([b1, b2, b3], axis=1)
        wd = log_so3(rd_prev.T @ rd) / dt
        error = rd.T @ rotation
        e_r = 0.5 * jnp.stack([error[2, 1] - error[1, 2], error[0, 2] - error[2, 0], error[1, 0] - error[0, 1]])
        e_w = omega - rotation.T @ rd @ wd
        accel = -K_R * e_r - K_W * e_w
        rows = jnp.asarray([2, 3, 4, 5])
        target = jnp.concatenate([(rotation.T @ thrust_world)[2:3], accel - drift[3:]])
        u = jnp.linalg.lstsq(gain[rows], target, rcond=1e-6)[0] * scale
        return u.at[0].set(jnp.maximum(u[0], 0.0)), rd, tilt

    return jax.jit(jax.vmap(one))


# ---------------------------------------------------------------------------------------------------------------
# references and the simulator
# ---------------------------------------------------------------------------------------------------------------
def references(data: rp.Dataset) -> dict:
    flights = data.flights[:FLIGHTS].astype(np.float64)                       # (F, T, 22)
    sample_dt = data.interval
    rotation = flights[..., 3:12].reshape(*flights.shape[:2], 3, 3)
    position = flights[..., :3]
    velocity = np.einsum("ftij,ftj->fti", rotation, flights[..., 12:15])
    accel = uniform_filter1d(np.gradient(velocity, sample_dt, axis=1), 5, axis=1, mode="nearest")
    yaw = np.unwrap(np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0]), axis=1)
    steps = int(round((flights.shape[1] - 1) * sample_dt * CONTROL_HZ))
    t = np.arange(steps + 1) / CONTROL_HZ
    sample_t = np.arange(flights.shape[1]) * sample_dt
    interp = lambda values: np.stack([np.stack([np.interp(t, sample_t, values[f, :, k]) for k in range(values.shape[-1])], -1)
                                      for f in range(values.shape[0])])
    return {"t": t, "p": interp(position), "v": interp(velocity), "a": interp(accel),
            "yaw": interp(yaw[..., None])[..., 0], "x0": flights[:, 0], "sample_t": sample_t, "flights": flights}


class Plant:
    """One PyBullet vehicle with the dataset's physics, wind and actuators (generate_dataset.py's pieces)."""

    def __init__(self, cfg: dict, x0: np.ndarray, seed: int):
        self.cfg = cfg
        rotation = x0[3:12].reshape(3, 3)
        rpy = Rotation.from_matrix(rotation).as_euler("xyz")
        self.env = gd.make_env(cfg, x0[:3].copy(), rpy)
        self.rng = np.random.default_rng(seed)
        self.env.reset(seed=int(seed))
        gd.configure_plant(self.env, cfg)
        self.mixer, self.mixer_inverse = gd.motor_mixer(self.env)
        self.kf, self.rpm_max = float(self.env.KF), float(self.env.MAX_RPM)
        self.mass, grav = float(self.env.M), float(self.env.G)
        self.hover = self.mass * grav
        self.inertia = np.diag([float(self.env.J[0, 0]), float(self.env.J[1, 1]), float(self.env.J[2, 2])])
        self.dt = 1.0 / int(cfg["plant"]["physics_hz"])
        self.wind = gd.Wind(cfg, self.mass, grav, self.inertia, self.dt)
        pb.resetBaseVelocity(int(self.env.DRONE_IDS[0]), linearVelocity=(rotation @ x0[12:15]).tolist(),
                             angularVelocity=(rotation @ x0[15:18]).tolist(), physicsClientId=int(self.env.CLIENT))
        self.env._updateAndStoreKinematicInformation()
        self.state = self.env._getDroneStateVector(0)
        self.step_count = 0
        self.rpm = np.zeros(4)
        self.saturated = 0

    def command(self, wrench: np.ndarray) -> np.ndarray:
        """Motor rpm for a wrench [T, tau] (clipped at the rpm limit); returns the wrench actually applied."""
        force = self.mixer_inverse @ wrench
        rpm = np.sqrt(np.maximum(force, 0.0) / self.kf)
        self.saturated += int(np.any(rpm > self.rpm_max))
        self.rpm = np.clip(rpm, 0.0, self.rpm_max)
        return self.mixer @ (self.kf * self.rpm ** 2)

    def advance(self, physics_steps: int) -> None:
        for _ in range(physics_steps):
            self.step_count += 1
            self.wind.step(self.rng, self.step_count, self.state, 1.0)
            self.wind.apply(self.env)
            gd.apply_external_dissipation(self.env, self.state, self.cfg, self.mass, self.inertia)
            observation, *_ = self.env.step(self.rpm.reshape(1, 4))
            self.state = observation[0]

    def close(self) -> None:
        self.env.close()


def fly(data: rp.Dataset, controller, ref: dict, cfg: dict) -> dict:
    """All flights of one model: flight f = (reference f // SEEDS, seed f % SEEDS), in lockstep. Records at the
    control rate: states (N+1, F, 22), applied wrench (N, F, 4), commanded R_d (N, F, 9), tilt (N, F)."""
    pairs = [(b, k) for b in range(FLIGHTS) for k in range(SEEDS)]
    plants = [Plant(cfg, ref["x0"][b], SEED_BASE + 100 * b + k) for b, k in pairs]
    pick = np.asarray([b for b, _ in pairs])
    steps = len(ref["t"]) - 1
    per_control = int(cfg["plant"]["physics_hz"]) // CONTROL_HZ
    states = np.full((steps + 1, len(pairs), 22), np.nan)
    applied = np.full((steps, len(pairs), 4), np.nan)
    commanded = np.full((steps, len(pairs), 9), np.nan)
    tilt = np.full((steps, len(pairs)), np.nan)
    failed_at = np.full(len(pairs), steps)
    alive = np.ones(len(pairs), bool)
    rd_prev = np.stack([ref["x0"][b][3:12].reshape(3, 3) for b in pick])
    try:
        for k in range(steps + 1):
            for f, plant in enumerate(plants):
                if alive[f]:
                    row = gd.pack_sample(plant.state, np.zeros(4))
                    states[k, f, :row.shape[0]] = row
                    far = np.linalg.norm(row[:3] - ref["p"][pick[f], k]) > ESCAPE
                    if not np.all(np.isfinite(row)) or far:
                        alive[f], failed_at[f] = False, k
                        states[k, f] = np.nan
            if k == steps or not alive.any():
                break
            inputs = (jnp.asarray(np.nan_to_num(states[k]), jnp.float32), jnp.asarray(ref["p"][pick, k], jnp.float32),
                      jnp.asarray(ref["v"][pick, k], jnp.float32), jnp.asarray(ref["a"][pick, k], jnp.float32),
                      jnp.asarray(ref["yaw"][pick, k], jnp.float32))
            if k == 0:                                   # w_d at the start: no finite difference against the recorded R
                rd_prev = np.asarray(controller(*inputs, jnp.asarray(rd_prev, jnp.float32))[1], np.float64)
            u, rd, tilt_k = controller(*inputs, jnp.asarray(rd_prev, jnp.float32))
            u, rd, tilt_k = np.asarray(u, np.float64), np.asarray(rd, np.float64), np.asarray(tilt_k, np.float64)
            rd_prev = rd
            for f, plant in enumerate(plants):
                if not alive[f]:
                    continue
                if not np.all(np.isfinite(u[f])):
                    alive[f], failed_at[f] = False, k
                    continue
                applied[k, f] = plant.command(u[f])
                commanded[k, f], tilt[k, f] = rd[f].reshape(9), tilt_k[f]
                plant.advance(per_control)
    finally:
        saturation = np.asarray([p.saturated for p in plants], float)
        for plant in plants:
            plant.close()
    return {"states": states, "applied": applied, "commanded": commanded, "tilt": tilt, "failed_at": failed_at,
            "saturated_steps": saturation, "hover": plants[0].hover if plants else 1.0}


# ---------------------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------------------
def reference_state_series(ref: dict) -> np.ndarray:
    """(N+1, F, 22) reference flight state at the control times, repeated for every wind seed of its reference."""
    pick = np.asarray([b for b in range(FLIGHTS) for _ in range(SEEDS)])
    reference_states = np.stack([np.concatenate([np.interp(ref["t"], ref["sample_t"], ref["flights"][b, :, k])[:, None]
                                                 for k in range(22)], axis=1) for b in range(FLIGHTS)], axis=1)
    return reference_states[:, pick]


def seed_crps(values: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """CRPS of the wind-seed ensemble of each reference (7 Oct 2026): values, truth (n, F, 3), F = FLIGHTS x SEEDS
    (reference-major). Per time and axis CRPS = E|X - y| - 1/2 E|X - X'| over the SEEDS flights of a reference, then the
    mean over axes and time; returned per flight (F,), the same value for every seed of a reference. A non-finite member
    (a failed flight) makes that reference's CRPS infinite."""
    n = values.shape[0]
    x = values.reshape(n, FLIGHTS, SEEDS, 3)
    y = truth.reshape(n, FLIGHTS, SEEDS, 3)[:, :, 0]
    with np.errstate(invalid="ignore", over="ignore"):
        first = np.mean(np.abs(x - y[:, :, None]), axis=2)
        spread = np.mean(np.abs(x[:, :, :, None] - x[:, :, None, :]), axis=(2, 3))
        crps = np.mean(first - 0.5 * spread, axis=(0, 2))                          # (FLIGHTS,)
    bad = ~np.isfinite(x).all(axis=(0, 2, 3))
    return np.repeat(np.where(bad | ~np.isfinite(crps), np.inf, crps), SEEDS)


def error_series(data: rp.Dataset, ref: dict, flight: dict) -> dict:
    pick = np.asarray([b for b in range(FLIGHTS) for _ in range(SEEDS)])
    states = flight["states"]
    rotation = states[..., 3:12].reshape(*states.shape[:2], 3, 3)
    with np.errstate(invalid="ignore"):
        velocity = np.einsum("tfij,tfj->tfi", rotation, states[..., 12:15])
        yaw = np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
        yaw_error = np.abs((yaw - ref["yaw"][pick].T + np.pi) % (2 * np.pi) - np.pi)           # rad (user, 6 Oct 2026)
        reference_states = np.stack([np.concatenate([np.interp(ref["t"], ref["sample_t"], ref["flights"][b, :, k])[:, None]
                                                     for k in range(22)], axis=1) for b in range(FLIGHTS)], axis=1)
        reference_states = reference_states[:, pick]
        h_error = np.abs(rp.true_energy(states, data.truth) - rp.true_energy(reference_states, data.truth))
        w_error = np.linalg.norm(states[..., 15:18] - reference_states[..., 15:18], axis=-1)        # rad/s, body (7 Oct 2026)
        out = {"p": np.linalg.norm(states[..., :3] - ref["p"][pick].swapaxes(0, 1), axis=-1),
               "v": np.linalg.norm(velocity - ref["v"][pick].swapaxes(0, 1), axis=-1), "w": w_error, "yaw": yaw_error, "H": h_error}
    return {k: np.nan_to_num(v, nan=np.inf) for k, v in out.items()}


def flight_metrics(data, ref, flight, analytical, mark: float) -> dict:
    n = int(round(mark * CONTROL_HZ))
    errors = error_series(data, ref, flight)
    window = slice(1, n + 1)
    out = {}
    with np.errstate(over="ignore", invalid="ignore"):
        out["p_rmse"] = np.sqrt(np.mean(errors["p"][window] ** 2, axis=0))
        out["p_max"] = np.max(errors["p"][window], axis=0)
        out["p_final"] = errors["p"][n]
        out["v_rmse"] = np.sqrt(np.mean(errors["v"][window] ** 2, axis=0))
        out["w_rmse"] = np.sqrt(np.mean(errors["w"][window] ** 2, axis=0))
        out["w_rms"] = np.nan_to_num(np.sqrt(np.nanmean(np.sum(flight["states"][window, :, 15:18] ** 2, axis=-1), axis=0)), nan=np.inf)
        reference_states = reference_state_series(ref)[window]
        out["p_crps"] = seed_crps(flight["states"][window, :, :3], reference_states[..., :3])
        out["w_crps"] = seed_crps(flight["states"][window, :, 15:18], reference_states[..., 15:18])
        out["yaw_rmse"] = np.sqrt(np.mean(errors["yaw"][window] ** 2, axis=0))
        out["H_rmse"] = np.sqrt(np.mean(errors["H"][window] ** 2, axis=0))
        flown = flight["states"][1:n + 1, :, 3:12]
        attitude = np.radians(rp.geodesic_deg(flight["commanded"][:n], flown))                # rad
        out["att_cmd"] = np.nan_to_num(np.sqrt(np.nanmean(attitude ** 2, axis=0)), nan=np.inf)
        thrust = flight["applied"][:n, :, 0] / flight["hover"]
        out["thrust_rms"] = np.nan_to_num(np.sqrt(np.nanmean(thrust ** 2, axis=0)), nan=np.inf)
        arm = 0.0397
        normalised = np.concatenate([flight["applied"][:n, :, :1] / flight["hover"],
                                     flight["applied"][:n, :, 1:] / (flight["hover"] * arm)], axis=-1)
        out["chatter"] = np.nan_to_num(np.sqrt(np.nanmean(np.sum(np.diff(normalised, axis=0) ** 2, axis=-1), axis=0)) * CONTROL_HZ,
                                       nan=np.inf)
        out["tilt_max"] = np.nan_to_num(np.nanmax(flight["tilt"][:n], axis=0), nan=np.inf)     # rad
    out["sat"] = flight["saturated_steps"] / max(len(ref["t"]) - 1, 1)
    never = flight["applied"].shape[0]
    failed = (flight["failed_at"] < never) & (flight["failed_at"] <= n)
    out["fail"] = failed.astype(float)
    out["completion"] = np.minimum(flight["failed_at"], n) / n
    if analytical is not None:
        pick = np.asarray([b for b in range(FLIGHTS) for _ in range(SEEDS)])
        excursion = np.sqrt(np.mean(np.sum((ref["p"] - ref["p"][:, :1]) ** 2, axis=-1), axis=1))[pick]
        with np.errstate(invalid="ignore"):
            gap = np.nan_to_num(np.linalg.norm(flight["states"][:, :, :3] - analytical["states"][:, :, :3], axis=-1),
                                nan=np.inf)[: n + 1]
        lost = gap > VALID_FRACTION * excursion[None]
        out["valid"] = np.where(lost.any(axis=0), lost.argmax(axis=0), n) / CONTROL_HZ
    return out


def table_cells(metrics: dict, labels, failed, flight_index: int | None) -> dict:
    out = {}
    for key, _, _ in ROWS:
        out[key] = {}
        for label in labels:
            values = None if label in failed else metrics[label].get(key)
            if values is None:
                out[key][label] = None
                continue
            per = np.asarray(values, float).reshape(FLIGHTS, SEEDS)
            if key == "fail":
                chosen = per if flight_index is None else per[flight_index]
                out[key][label] = (float(chosen.sum()), float(chosen.size))
            elif flight_index is None:
                seed_mean = per.mean(axis=1)
                out[key][label] = (float(np.mean(seed_mean)), float(np.std(seed_mean)))
            else:
                out[key][label] = (float(np.mean(per[flight_index])), float(np.std(per[flight_index])))
    return out


# ---------------------------------------------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------------------------------------------
def setup_page(data, labels, failed) -> plt.Figure:
    truth = data.truth
    law = "rate-dependent -c m (1+|v_b|) v_b, -c J (1+|w_b|) w_b" if truth["damping_law"] == "nonlinear" else "constant -c m v_b, -c J w_b"
    lines = [f"dataset    {data.path.name}",
             f"references the first {FLIGHTS} held-out flights of {data.eval_path.name} ({data.t[-1]:g} s, eval library, never used in",
             "           training): p_d(t), v_d = R v_b, a_d = smoothed finite difference, yaw from R;",
             f"           linear interpolation to {CONTROL_HZ} Hz. Every flight starts in the recorded x_0 of its reference.",
             f"plant      the dataset's own PyBullet simulator (CF2P, {data.settings['physics_hz']} Hz, contact-free), damping {law},",
             "           c = 0.5; wind = constant diffusion 0.5 on the body twist, new gust every 0.01 s; motor mixer, rpm limit.",
             f"seeds      {SEEDS} wind seeds per reference (seed = {SEED_BASE} + 100 reference + k), the SAME for every model.",
             "controller the quadrotor booklet's energy-based geometric tracking law with the model's gauge-invariant quantities,",
             f"           at {CONTROL_HZ} Hz, the same gains for every model:",
             "             f = model d(v_b, w)/dt at u = 0,  B = blockdiag(M1^-1, M2^-1) g",
             "             a_w = a_d - K_p (p - p_d) - K_v (v_w - v_d),   a_T = R (R^T a_w - w x v_b - f_v),  tilt <= 0.698 rad",
             "             R_d from b3d = a_T / |a_T| and the yaw reference;  e_R = 1/2 vee(R_d^T R - R^T R_d),  e_w = w - R^T R_d w_d",
             "             rows (v_z, w_x, w_y, w_z) of B u = [ (R^T a_T)_z ; -K_R e_R - K_w e_w - f_w ]   (u then x training RMS)",
             f"             K_p = {tuple(K_P)}, K_v = {tuple(K_V)}, K_R = {K_R:g}, K_w = {K_W:g}  (acceleration units)",
             "           Only the drift is used (no diffusion), so there are no -wind-same models.",
             f"failure    non-finite state or control, or more than {ESCAPE:g} m from the reference (the flight is frozen).",
             "metrics    over (0, H] for H = 1, 3, 5, 10 s at the control rate; tables = per reference the mean over seeds, then mean +- std",
             "           over references (per-flight pages: mean +- std over seeds). Valid tracking time = first t where the vehicle",
             f"           leaves the Analytical flight (same seed) by more than {VALID_FRACTION:g} x the RMS reference excursion.",
             "           Energy = TRUE H = 1/2 m |v|^2 + 1/2 w^T J w + m g z of the flown and the reference state.", "", "models"]
    for label in labels:
        lines.append(f"  {label:16s} {'TRAINING FAILED -> -' if label in failed else ('true operators (ceiling, not ranked)' if label == ANALYTICAL else 'posterior-mean drift')}")
    figure = plt.figure(figsize=rp.PAGE)
    figure.text(0.04, 0.95, "Closed-loop tracking of the held-out flights: setup", fontsize=15, weight="bold")
    figure.text(0.04, 0.90, "\n".join(lines), fontsize=7.6, va="top", family="monospace")
    return figure


def fmt(key, value) -> str:
    if value is None:
        return "-"
    if key == "fail":
        return f"{value[0]:.0f} / {value[1]:.0f}"
    return f"{value[0]:.4g} +- {value[1]:.2g}"


def table_page(cells, labels, failed, title, subtitle) -> plt.Figure:
    figure = plt.figure(figsize=rp.PAGE)
    figure.text(0.03, 0.955, title, fontsize=14, weight="bold")
    figure.text(0.03, 0.925, subtitle, fontsize=7.5)
    header = ["metric"] + [label.replace("Lie-PH-", "Lie-PH-\n") for label in labels]
    rows, best = [], []
    learned = [i for i, label in enumerate(labels) if label != ANALYTICAL]
    for r, (key, name, rule) in enumerate(ROWS):
        values = [cells[key][label] for label in labels]
        rows.append([name] + [fmt(key, v) for v in values])
        if rule == "none":
            continue
        reference = cells[key].get(ANALYTICAL)
        candidates = []
        for i in learned:
            if values[i] is None or not np.isfinite(values[i][0]):
                continue
            if rule == "low":
                score = values[i][0]
            elif rule == "high":
                score = -values[i][0]
            elif reference is None:
                continue
            else:
                score = abs(values[i][0] - reference[0])
            candidates.append((round(float(score), 9), i))
        if len(candidates) > 1:
            top = min(candidates)[0]
            best += [(r + 1, 1 + i) for score, i in candidates if score == top]
    axis = figure.add_axes([0.01, 0.07, 0.98, 0.83])
    axis.axis("off")
    widths = [0.24] + [0.76 / len(labels)] * len(labels)
    rp.styled_table(axis, rows, header, widths, 7.2, 1.6, best, failed_cols=[1 + i for i, l in enumerate(labels) if l in failed])
    figure.text(0.03, 0.012, "\n".join(textwrap.wrap(
        "Green bold = best LEARNED model per row (lowest; highest for valid tracking time and completion; RMS thrust, angular-velocity "
        "RMSE, angular-velocity CRPS and RMS body rate closest to Analytical, because even the true model does not follow the reference "
        "flight's body rates; ties share the highlight). CRPS = continuous ranked probability score of the ensemble of the "
        f"{SEEDS} wind-seed flights of a reference against that reference flight, per axis, averaged over axes and time "
        "(a failed flight makes it infinite). Analytical = true operators in the same controller (ceiling, not ranked). "
        "'-' = training failed (red column) or not defined. Failures are not ranked.", 230)), fontsize=6.3)
    return figure


def error_page(t, errors, labels, failed) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=rp.PAGE)
    names = {"p": "position error (m)", "v": "velocity error (m/s, world)", "w": "angular-velocity error (rad/s, body, vs reference flight)",
             "yaw": "yaw error (rad)", "H": "energy error |H - H_ref| (J)"}
    axes[1, 2].axis("off")
    for axis, key in zip(axes.ravel(), ("p", "v", "w", "yaw", "H")):
        for label in labels:
            if label in failed:
                continue
            series = errors[label][key].reshape(len(t), FLIGHTS, SEEDS).mean(axis=2)
            style = dict(color=COLORS[label], lw=1.5 if label == ANALYTICAL else 1.8, ls="--" if label == ANALYTICAL else "-")
            axis.plot(t[1:], rp.plottable(np.median(series, axis=1))[1:], label=label, **style)
            if label != ANALYTICAL:
                axis.fill_between(t[1:], rp.plottable(np.quantile(series, 0.25, axis=1))[1:],
                                  rp.plottable(np.quantile(series, 0.75, axis=1))[1:], color=COLORS[label], alpha=0.10, lw=0)
        for mark in HORIZON_MARKS[:-1]:
            axis.axvline(mark, color="0.5", lw=0.8, ls="--")
        axis.set_yscale("log")
        axis.set_title(names[key], fontsize=9)
        axis.set_xlabel("t (s)")
        axis.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Closed-loop tracking error vs time: median over references of the seed mean, shade = inter-quartile, "
                    "dashed = Analytical", fontsize=10)
    figure.tight_layout()
    return figure


def flight_values(flights, label, index):
    return np.swapaxes(flights[label]["states"][:, index * SEEDS:(index + 1) * SEEDS], 0, 1)        # (S, N+1, 22)


def seed_band(values):
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def path3d_page(ref, flights, labels, failed, index) -> plt.Figure:
    figure = plt.figure(figsize=rp.PAGE)
    reference = ref["p"][index]
    lo, hi = reference.min(axis=0) - 0.5, reference.max(axis=0) + 0.5
    for k, label in enumerate(labels):
        axis = figure.add_subplot(2, 3, k + 1, projection="3d")
        axis.plot(*reference.T, color=REF_COLOR, lw=1.0, ls="--")
        if label in failed:
            axis.set_title(f"{label}\ntraining failed", fontsize=8, color="0.4")
        else:
            states = flight_values(flights, label, index)
            for path in states:
                axis.plot(*rp.plottable(path[:, :3]).T, color=COLORS[label], lw=0.4, alpha=0.4)
            mean, _ = seed_band(states[..., :3])
            axis.plot(*rp.plottable(mean).T, color=COLORS[label], lw=1.5)
            axis.set_title(label, fontsize=8, color=COLORS[label], weight="bold")
        axis.set_xlim(lo[0], hi[0])
        axis.set_ylim(lo[1], hi[1])
        axis.set_zlim(lo[2], hi[2])
        axis.tick_params(labelsize=5)
    figure.suptitle(f"Held-out reference {index}: 3-D path (dashed = reference, thin = {SEEDS} wind seeds, bold = seed mean)", fontsize=10)
    figure.tight_layout()
    return figure


def component_page(ref, flights, labels, failed, index, block, hover) -> plt.Figure:
    names = {"p": (["x (m)", "y (m)", "z (m)"], "Position"), "euler": (["roll (rad)", "pitch (rad)", "yaw (rad)"], "Attitude (Euler xyz)"),
             "v": (["v_x (m/s)", "v_y (m/s)", "v_z (m/s)"], "World velocity"), "w": (["w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"], "Body angular velocity"),
             "u": (["T / hover", "tau_x (mN m)", "tau_y (mN m)"], "Applied wrench (thrust / hover, torques)")}
    ylabels, title = names[block]
    t = ref["t"]
    reference_state = np.stack([np.interp(t, ref["sample_t"], ref["flights"][index, :, k]) for k in range(22)], axis=1)

    def values(states):
        if block == "p":
            return states[..., :3]
        if block == "euler":
            return np.radians(rp.euler_deg(states[..., 3:12]))
        if block == "v":
            rotation = states[..., 3:12].reshape(*states.shape[:-1], 3, 3)
            return np.einsum("...ij,...j->...i", rotation, states[..., 12:15])
        return states[..., 15:18]

    reference = None if block == "u" else values(reference_state)
    figure, axes = plt.subplots(3, len(labels), figsize=(max(rp.PAGE[0], 2.6 * len(labels)), rp.PAGE[1]), sharex=True, squeeze=False)
    for col, label in enumerate(labels):
        for row in range(3):
            axis = axes[row, col]
            if reference is not None:
                axis.plot(t, reference[:, row], color=REF_COLOR, lw=1.0, ls="--")
            if label in failed:
                if row == 1:
                    axis.text(0.5, 0.5, "training failed", transform=axis.transAxes, ha="center", color="0.4")
            else:
                if block == "u":
                    wrench = np.swapaxes(flights[label]["applied"][:, index * SEEDS:(index + 1) * SEEDS], 0, 1)
                    data_rows = wrench[..., 0] / hover if row == 0 else wrench[..., row] * 1e3
                    tt = t[:-1]
                else:
                    data_rows = values(flight_values(flights, label, index))[..., row]
                    if block == "euler":
                        data_rows = data_rows - 2 * np.pi * np.round((data_rows[:, :1] - reference[0, row]) / (2 * np.pi))
                    tt = t
                mean, std = seed_band(data_rows)
                axis.fill_between(tt, rp.plottable(mean - std), rp.plottable(mean + std), color=COLORS[label], alpha=0.3, lw=0)
                axis.plot(tt, rp.plottable(mean), color=COLORS[label], lw=1.1)
            if reference is not None:
                low, high = np.nanmin(reference[:, row]), np.nanmax(reference[:, row])
                reach = 1.0 * max(high - low, 0.05)
                current = axis.get_ylim()
                axis.set_ylim(max(current[0], low - reach), min(current[1], high + reach))
            for mark in HORIZON_MARKS[:-1]:
                axis.axvline(mark, color="0.5", lw=0.5, ls="--")
            axis.grid(alpha=0.25)
            axis.tick_params(labelsize=6)
            if col == 0:
                axis.set_ylabel(ylabels[row], fontsize=7)
            if row == 0:
                axis.set_title(label.replace("Lie-PH-", "Lie-PH-\n"), color=COLORS[label], weight="bold", fontsize=7)
            if row == 2:
                axis.set_xlabel("t (s)", fontsize=7)
    note = "no reference" if block == "u" else "dashed = reference; y-axis capped at the reference range +- 1x its width"
    figure.suptitle(f"Held-out reference {index}: {title}. Line = mean over {SEEDS} wind seeds, band = +- 1 std; {note}", fontsize=9)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------
def closed_loop_report(setting: str, refly: bool = False, models: tuple | None = CAMPAIGN_MODELS) -> Path:
    """``models``: only these learned models (None = all of LEARNED); the Analytical ceiling is always flown."""
    data = rp.Dataset(setting)
    records = rp.campaign_records(RESULT_DIR)
    learned = tuple(m for m in LEARNED if models is None or m in models)
    sources = load_sources(data, records, learned)
    labels = list(learned) + [ANALYTICAL]
    failed = {m for m in learned if sources[m] is None}
    cfg = yaml.safe_load(next(data.path.parent.glob("*_config_used.yaml")).read_text())
    ref = references(data)
    out_dir = RESULT_DIR / "reports" / f"{setting}_noise{NOISE:g}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "closed-loop-flights.npz"
    stored = dict(np.load(cache, allow_pickle=True)) if cache.exists() and not refly else {}
    keys = ("states", "applied", "commanded", "tilt", "failed_at", "saturated_steps", "hover")
    flights = {}
    print(f"[closed-loop] {setting}: {FLIGHTS} references x {SEEDS} seeds, {len(ref['t']) - 1} control steps at {CONTROL_HZ} Hz", flush=True)
    for label in [ANALYTICAL] + [l for l in learned if l not in failed]:
        if f"{label}/states" in stored:
            flights[label] = {key: stored[f"{label}/{key}"] for key in keys}
            flights[label]["hover"] = float(flights[label]["hover"])
            print(f"[closed-loop] {label:16s} loaded from {cache.name}", flush=True)
            continue
        tic = datetime.now()
        source = sources["analytical" if label == ANALYTICAL else label]
        flights[label] = fly(data, make_controller(source["model"], data.scale), ref, cfg)
        lost = int(np.sum(flights[label]["failed_at"] < len(ref["t"]) - 1))
        print(f"[closed-loop] {label:16s} flown in {(datetime.now() - tic).total_seconds():.0f} s, failures {lost}", flush=True)
        np.savez_compressed(cache, **{f"{l}/{key}": np.asarray(v[key]) for l, v in flights.items() for key in keys})
    metrics = {f"0-{m:g}s": {label: flight_metrics(data, ref, flights[label], None if label == ANALYTICAL else flights[ANALYTICAL], m)
                             for label in flights} for m in HORIZON_MARKS}
    summary = {key: table_cells(metrics[key], labels, failed, None) for key in metrics}
    errors = {label: error_series(data, ref, flights[label]) for label in flights}
    hover = flights[ANALYTICAL]["hover"]
    entries = rp.entries_for(setting, learned)
    for entry in entries:
        if entry.source != "analytical":
            entry.failed = sources[entry.source] is None
            entry.run_dir = None if entry.failed else sources[entry.source]["run_dir"]
    pdf_path = out_dir / "closed-loop-comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(rp.physics_page(data, entries)); plt.close("all")
        pdf.savefig(setup_page(data, labels, failed)); plt.close("all")
        for key in metrics:
            pdf.savefig(table_page(summary[key], labels, failed, f"Table C - closed-loop tracking of the held-out flights, {key.replace('s', ' s')}",
                                   f"{setting}: per reference the mean over {SEEDS} wind seeds, then mean +- std over {FLIGHTS} references."))
            plt.close("all")
        pdf.savefig(error_page(ref["t"], errors, labels, failed)); plt.close("all")
        for index in range(FLIGHTS):
            for key in metrics:
                pdf.savefig(table_page(table_cells(metrics[key], labels, failed, index), labels, failed,
                                       f"Held-out reference {index} - closed-loop tracking, {key.replace('s', ' s')}",
                                       f"{setting}, held-out reference {index}: mean +- std over {SEEDS} wind seeds.")); plt.close("all")
            pdf.savefig(path3d_page(ref, flights, labels, failed, index)); plt.close("all")
            for block in ("p", "euler", "v", "w", "u"):
                pdf.savefig(component_page(ref, flights, labels, failed, index, block, hover)); plt.close("all")
    record = {"generated_at": datetime.now().astimezone().isoformat(), "dataset": str(data.path), "setting": setting,
              "references": FLIGHTS, "seeds": SEEDS, "control_hz": CONTROL_HZ,
              "gains": {"K_p": K_P.tolist(), "K_v": K_V.tolist(), "K_R": K_R, "K_w": K_W, "max_tilt_rad": MAX_TILT},
              "models": {label: ("failed" if label in failed else str(sources["analytical" if label == ANALYTICAL else label]["run_dir"]))
                         for label in labels}, "table": summary}
    (out_dir / "closed-loop-summary.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(f"[closed-loop] wrote {pdf_path}", flush=True)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", nargs="+", required=True, choices=tuple(SETTINGS))
    parser.add_argument("--refly", action="store_true")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(CAMPAIGN_MODELS),
                        help="learned models to fly (default: the campaign's Lie-PH-GP-SDE and PH-NODE)")
    args = parser.parse_args()
    for setting in args.setting:
        closed_loop_report(setting, args.refly, tuple(args.models))


if __name__ == "__main__":
    main()

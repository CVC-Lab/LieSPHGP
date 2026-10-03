"""One BlueROV2 Heavy dataset generator (our port-Hamiltonian simulator), driven by one config file (config.yaml here).

    python envs/rov_se3_port_ham/datagen/generate_dataset.py --config envs/rov_se3_port_ham/datagen/config.yaml

Writes datasets/ROV-DATASET-<name>/<name>_BLUEROV2_<T>s_h<h>_<variant>.pkl (+ audits.json, config_used.yaml,
generation.log, dataset_analysis.pdf). Same switches and file layout as the PyBullet quadrotor generator.

Plant (envs/rov_se3_port_ham/bluerov2.py, Lie-group Heun at physics_hz, world z DOWN):
    p_dot = R v,  R_dot = R [w]x,  M nu_dot = J(M nu) nu - D(nu) nu - grad V + E thrust + wind
Excitation: a pose PD controller (zero-order hold at sample_hz) tracks random manoeuvre sequences; thrust = pinv(E) wrench,
clipped to the T200 range at the configured battery voltage. `coast` turns every motor off (free decay), `pulse` adds
an open-loop wrench pulse through the motors. Every external effect except the wind passes through the thrusters, so
the recorded u is the full actuation.

Switches
  trajectory_set      hard | eval | hard+eval          (hard library: train + test; eval library: heldout, always clean)
  dissipation.<ch>    none | linear | quadratic | linear_quadratic   (published D_L, D_Q of von Benzon 2022)
  diffusion.<ch>      none | constant | rate_dependent | ou           (wind on the body, force M a, torque M_w alpha)
  actuator.input      thrusters (8 forces, G = E) | wrench (6, G = I) | commands (8 PX4 commands, T200 map, nonlinear)
  excitation          library (PD + manoeuvres) | random_thrusters (held random thrust + weak PD keeping the tank)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]  # envs/rov_se3_port_ham/datagen -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.rov_se3_port_ham.bluerov2 import BlueROV2, BlueROV2Params, T200, THRUSTERS, exp_so3  # noqa: E402

STATE_DIM = 18
EUCLIDEAN = np.asarray([0, 1, 2, 12, 13, 14, 15, 16, 17])
TRAJECTORY_SETS = ("hard", "eval", "hard+eval")
DISSIPATION_TYPES = ("none", "linear", "quadratic", "linear_quadratic")
DIFFUSION_TYPES = ("none", "constant", "rate_dependent", "ou")
INPUTS = ("thrusters", "wrench", "commands")
SPLIT_KEY = {"train": "train_trajectories", "test": "test_trajectories", "eval": "heldout_trajectories"}
SPLIT_WINDOWS = {"train": "x", "test": "test_x", "eval": "heldout_x"}
SPLIT_PREFIX = {"train": "", "test": "test_", "eval": "heldout_"}
SPLIT_STEM = {"train": "train", "test": "test", "eval": "heldout"}
LOG_LINES: list[str] = []


def log(message: str) -> None:
    LOG_LINES.append(message)
    print(message, flush=True)


# --------------------------------------------------------------------------- config
def validate(cfg: dict) -> None:
    if cfg["trajectory_set"] not in TRAJECTORY_SETS:
        raise ValueError(f"trajectory_set must be one of {TRAJECTORY_SETS}")
    for ch in ("linear", "angular"):
        if cfg["dissipation"][ch]["type"] not in DISSIPATION_TYPES:
            raise ValueError(f"dissipation.{ch}.type must be one of {DISSIPATION_TYPES}")
        if cfg["diffusion"][ch]["type"] not in DIFFUSION_TYPES:
            raise ValueError(f"diffusion.{ch}.type must be one of {DIFFUSION_TYPES}")
    if cfg["actuator"]["input"] not in INPUTS:
        raise ValueError(f"actuator.input must be one of {INPUTS}")
    if cfg["excitation"]["type"] not in ("library", "random_thrusters"):
        raise ValueError("excitation.type must be library or random_thrusters")
    if int(cfg["plant"]["physics_hz"]) % int(cfg["plant"]["sample_hz"]):
        raise ValueError("plant.sample_hz must divide plant.physics_hz")
    shared = sorted(set(cfg["libraries"]["hard"]) & set(cfg["libraries"]["eval"]))
    if shared:
        raise ValueError(f"the eval library must share no segment with the hard library; shared: {shared}")


def split_plan(cfg: dict) -> list[tuple[str, list[int], str]]:
    s = cfg["seeds"]
    hard = [("train", s["train"], "hard"), ("test", s["test"], "hard")]
    return {"hard": hard, "eval": [("eval", s["eval"], "eval")], "hard+eval": hard + [("eval", s["eval"], "eval")]}[cfg["trajectory_set"]]


def dataset_name(cfg: dict) -> str:
    return f"{cfg['name']}_BLUEROV2_{int(cfg['flight']['duration_seconds'])}s_h{1.0 / cfg['plant']['sample_hz']:g}".replace(".", "p")


def make_vehicle(cfg: dict) -> BlueROV2:
    params = BlueROV2Params(**{k: v for k, v in (cfg["plant"].get("parameters") or {}).items()})
    return BlueROV2(params, {ch: cfg["dissipation"][ch]["type"] for ch in ("linear", "angular")})


# --------------------------------------------------------------------------- wind (the diffusion term)
class Wind:
    """Body-frame disturbance force M_v a and torque M_w alpha, one type per channel.
    constant / rate_dependent: white, redrawn every hold_seconds: a = sigma (1 + gain |.|) xi / sqrt(hold)
    ou: a <- a exp(-dt/T) + sigma sqrt(1 - exp(-2 dt/T)) xi every physics step (sigma = stationary std)."""

    def __init__(self, cfg: dict, M: np.ndarray, dt: float):
        self.cfg, self.M, self.dt = cfg["diffusion"], M, dt
        self.type = {ch: self.cfg[ch]["type"] for ch in ("linear", "angular")}
        self.hold_steps = max(1, int(round(float(self.cfg.get("hold_seconds", dt)) / dt)))
        self.accel = {"linear": np.zeros(3), "angular": np.zeros(3)}

    def step(self, rng: np.random.Generator, step: int, nu: np.ndarray, mult: float) -> None:
        for ch, sl in (("linear", slice(0, 3)), ("angular", slice(3, 6))):
            kind, p = self.type[ch], self.cfg[ch]
            if kind == "none":
                continue
            xi = rng.normal(size=3)
            if kind == "ou":
                decay = np.exp(-self.dt / float(p["time_constant_seconds"]))
                self.accel[ch] = self.accel[ch] * decay + mult * float(p["sigma"]) * np.sqrt(1 - decay**2) * xi
            elif (step - 1) % self.hold_steps == 0:
                gain = float(p.get("gain", 0.0)) if kind == "rate_dependent" else 0.0
                sigma = mult * float(p["sigma"]) * (1.0 + gain * float(np.linalg.norm(nu[sl])))
                self.accel[ch] = sigma * xi / np.sqrt(self.hold_steps * self.dt)

    def force(self) -> np.ndarray:
        return np.r_[self.M[:3] * self.accel["linear"], self.M[3:] * self.accel["angular"]]


# --------------------------------------------------------------------------- initial state, manoeuvres
def uniform_box(rng: np.random.Generator, box: dict) -> np.ndarray:
    return np.array([rng.uniform(*box[k]) for k in ("x", "y", "z")])


def yaw_rotation(psi: float) -> np.ndarray:
    return exp_so3(np.array([0.0, 0.0, psi]))


def random_initial_state(rng: np.random.Generator, cfg: dict) -> dict[str, np.ndarray]:
    s = cfg["initial_state"]
    tilt = np.radians(s["max_tilt_deg"]) * rng.uniform(0, 1)
    axis = rng.normal(size=2); axis /= np.linalg.norm(axis)
    R = yaw_rotation(rng.uniform(-np.pi, np.pi)) @ exp_so3(np.array([tilt * axis[0], tilt * axis[1], 0.0]))
    v = rng.normal(size=3); v *= rng.uniform(0, s["max_speed"]) / np.linalg.norm(v)
    w = rng.normal(size=3); w *= rng.uniform(0, s["max_angular_rate"]) / np.linalg.norm(w)
    return {"position": uniform_box(rng, s["position_box"]), "R": R, "nu": np.r_[v, w]}


def plan_manoeuvres(rng: np.random.Generator, cfg: dict, library: dict) -> list[dict[str, Any]]:
    """Random segment sequence covering the flight; parameters drawn here, targets resolved at run time."""
    names, weights = list(library), np.array([library[n]["weight"] for n in library], float)
    total, t, segments = float(cfg["flight"]["duration_seconds"]), 0.0, []
    while t < total:
        name = names[rng.choice(len(names), p=weights / weights.sum())]
        p = library[name]
        seg = {"name": name, "start": t, "duration": float(rng.uniform(*p["duration"]))}
        if name == "waypoint_hop":
            seg["target"] = uniform_box(rng, cfg["envelope"]["position_box"]).tolist(); seg["max_distance"] = float(p["max_distance"])
        if name in ("surge_run", "diagonal_dash", "lawnmower"):
            seg["speed"] = float(rng.uniform(*p["speed"])); seg["heading"] = float(rng.uniform(-np.pi, np.pi))
            if name == "lawnmower":
                seg["leg_seconds"] = float(rng.uniform(*p["leg_seconds"])); seg["spacing"] = float(rng.uniform(*p["spacing"]))
        if name in ("yaw_turn", "yaw_sweep"):
            seg["rate"] = float(rng.choice([-1, 1]) * rng.uniform(*p["rate"]))
            if name == "yaw_sweep":
                seg["period"] = float(rng.uniform(*p["period"]))
        if name in ("depth_step", "depth_bob"):
            seg["height"] = float(rng.choice([-1, 1]) * rng.uniform(*p["height"]))
            if name == "depth_bob":
                seg["period"] = float(rng.uniform(*p["period"]))
        if name in ("circle", "helix", "figure_eight"):
            seg["radius"], seg["speed"] = float(rng.uniform(*p["radius"])), float(rng.uniform(*p["speed"]))
            seg["direction"] = float(rng.choice([-1, 1]))
            if name == "helix":
                seg["climb_rate"] = float(rng.choice([-1, 1]) * rng.uniform(*p["climb_rate"]))
        if name in ("attitude_hold", "tilt_sweep"):
            seg["tilt"] = float(np.radians(rng.uniform(*p["tilt_deg"])))
            seg["tilt_axis"] = float(rng.uniform(-np.pi, np.pi))
            if name == "tilt_sweep":
                seg["period"] = float(rng.uniform(*p["period"]))
        if name == "pulse":
            axis = int(rng.integers(6)); seg["axis"] = axis
            seg["pulse_seconds"] = float(rng.uniform(*p["pulse_seconds"]))
            seg["fraction"] = float(rng.choice([-1, 1]) * rng.uniform(*p["fraction"]))
        seg["duration"] = min(seg["duration"], total - t)
        segments.append(seg); t += seg["duration"]
    return segments


def segment_target(seg: dict, t_local: float, anchor: np.ndarray, anchor_yaw: float, box: dict) -> tuple[np.ndarray, float, float, float]:
    """Target position (world, z down), yaw, roll and pitch for the controller at local time t inside a segment."""
    name, xyz, yaw, roll, pitch = seg["name"], anchor.copy(), anchor_yaw, 0.0, 0.0
    if name == "waypoint_hop":
        step = np.asarray(seg["target"]) - anchor
        xyz = anchor + step * min(1.0, seg["max_distance"] / max(np.linalg.norm(step), 1e-9))
    elif name in ("surge_run", "diagonal_dash"):
        heading = seg["heading"] if name == "diagonal_dash" else anchor_yaw
        xyz = anchor + seg["speed"] * t_local * np.array([np.cos(heading), np.sin(heading), 0.0])
        if name == "surge_run":
            yaw = anchor_yaw
    elif name == "lawnmower":
        leg = int(t_local // seg["leg_seconds"]); tau = t_local - leg * seg["leg_seconds"]
        d = np.array([np.cos(seg["heading"]), np.sin(seg["heading"]), 0.0]); n = np.array([-d[1], d[0], 0.0])
        along = seg["speed"] * (tau if leg % 2 == 0 else seg["leg_seconds"] - tau)
        xyz = anchor + along * d + leg * seg["spacing"] * n
    elif name == "yaw_turn":
        yaw = anchor_yaw + seg["rate"] * t_local
    elif name == "yaw_sweep":
        yaw = anchor_yaw + seg["rate"] * seg["period"] / (2 * np.pi) * np.sin(2 * np.pi * t_local / seg["period"])
    elif name == "depth_step":
        xyz = anchor + np.array([0.0, 0.0, seg["height"]])
    elif name == "depth_bob":
        xyz = anchor + np.array([0.0, 0.0, seg["height"] if int(t_local // (seg["period"] / 2)) % 2 == 0 else 0.0])
    elif name in ("circle", "helix", "figure_eight"):
        a, om = seg["radius"], seg["direction"] * seg["speed"] / seg["radius"]
        if name == "figure_eight":
            xyz = anchor + a * np.array([np.sin(om * t_local), np.sin(2 * om * t_local) / 2, 0.0])
        else:
            xyz = anchor + np.array([a * (np.cos(om * t_local) - 1.0), a * np.sin(om * t_local),
                                     seg.get("climb_rate", 0.0) * t_local])
            yaw = anchor_yaw + om * t_local
    elif name in ("attitude_hold", "tilt_sweep"):                 # hold position with the body tilted
        phase = 1.0 if name == "attitude_hold" else np.sin(2 * np.pi * t_local / seg["period"])
        roll, pitch = seg["tilt"] * phase * np.cos(seg["tilt_axis"]), seg["tilt"] * phase * np.sin(seg["tilt_axis"])
    # station_keep, coast, pulse: hold the anchor
    xyz = np.array([np.clip(xyz[i], *box[k]) for i, k in enumerate("xyz")])
    return xyz, yaw, roll, pitch


def pd_wrench(vehicle: BlueROV2, cfg: dict, position, R, nu, target, yaw, roll: float = 0.0, pitch: float = 0.0,
              scale: float = 1.0, v_ref_world: np.ndarray | None = None) -> np.ndarray:
    """Body wrench of a pose tracking controller: PD on the pose error and on (v_ref - v) (acceleration-level gains times
    the diagonal inertia), plus feedforward of the drag at v_ref and of the hydrostatics. R_d = Rz(yaw) Ry(pitch) Rx(roll)."""
    c = cfg["controller"]
    R_d = yaw_rotation(yaw) @ exp_so3(np.array([0.0, pitch, 0.0])) @ exp_so3(np.array([roll, 0.0, 0.0]))
    e_p = R.T @ (target - position)
    v_ref = np.zeros(3) if v_ref_world is None else R.T @ v_ref_world
    E_R = R_d.T @ R - R.T @ R_d
    e_R = 0.5 * np.array([E_R[2, 1], E_R[0, 2], E_R[1, 0]])
    acc_lin = scale * (np.asarray(c["kp_position"]) * e_p + np.asarray(c["kd_velocity"]) * (v_ref - nu[:3]))
    acc_ang = scale * (-np.asarray(c["kp_attitude"]) * e_R - np.asarray(c["kd_rate"]) * nu[3:])
    drag_ff = -vehicle.damping_force(np.r_[v_ref, 0.0, 0.0, 0.0]) if c.get("drag_feedforward", True) else np.zeros(6)
    return np.r_[vehicle.M[:3] * acc_lin, vehicle.M[3:] * acc_ang] + scale * drag_ff - vehicle.restoring(R)


# --------------------------------------------------------------------------- one flight
def generate_flight(cfg: dict, seed: int, index: int, attempt: int, library: dict) -> tuple[np.ndarray, dict, dict]:
    rng = np.random.default_rng(int(cfg["seeds"]["flight_seed_base"]) + 1000 * seed + index + 100_000 * attempt)
    init = random_initial_state(rng, cfg)
    segments = plan_manoeuvres(rng, cfg, library)
    vehicle, t200 = make_vehicle(cfg), T200()
    E, E_pinv = vehicle.E, np.linalg.pinv(vehicle.E)
    pl, act, box = cfg["plant"], cfg["actuator"], cfg["envelope"]["position_box"]
    physics_hz, sample_hz = int(pl["physics_hz"]), int(pl["sample_hz"])
    sub, dt = physics_hz // sample_hz, 1.0 / physics_hz
    n = int(round(cfg["flight"]["duration_seconds"] * sample_hz)) + 1
    volts = float(act["voltage"])
    t_min, t_max = t200.limits(volts)
    t_min, t_max = float(act["thrust_fraction"]) * t_min, float(act["thrust_fraction"]) * t_max
    n_u = 6 if act["input"] == "wrench" else 8
    random_mode = cfg["excitation"]["type"] == "random_thrusters"
    wind = Wind(cfg, vehicle.M, dt)
    position, R, nu = init["position"].copy(), init["R"].copy(), init["nu"].copy()

    rows = np.zeros((n, STATE_DIM + n_u)); thrusts = np.zeros((n, 8)); gust = np.zeros((n, 6))
    rows[0, :STATE_DIM] = np.r_[position, R.reshape(9), nu]
    seg_i, anchor, anchor_yaw = 0, position.copy(), float(np.arctan2(R[1, 0], R[0, 0]))
    thrust_cmd, thrust, saturated, held = np.zeros(8), np.zeros(8), 0, np.zeros(8)
    acc_u, acc_thrust, acc_gust = np.zeros(n_u), np.zeros(8), np.zeros(6)
    min_depth, max_depth, max_tilt = position[2], position[2], 0.0
    for step in range(1, (n - 1) * sub + 1):
        t = step * dt
        seg = segments[seg_i]
        if t >= seg["start"] + seg["duration"] and seg_i + 1 < len(segments):
            seg_i += 1; seg = segments[seg_i]; anchor, anchor_yaw = position.copy(), float(np.arctan2(R[1, 0], R[0, 0]))
        if (step - 1) % sub == 0:                                            # controller at sample_hz (zero-order hold)
            local = t - seg["start"]
            if random_mode:
                ex = cfg["excitation"]["random_thrusters"]
                if (step - 1) % max(1, int(round(float(ex["hold_seconds"]) * physics_hz))) == 0:
                    held = float(ex["scale"]) * rng.uniform(t_min, t_max, 8)        # new random thrust per motor
                centre = np.array([np.mean(box[k]) for k in "xyz"])
                wrench = pd_wrench(vehicle, cfg, position, R, nu, centre, anchor_yaw, scale=float(ex["pd_scale"]))
                thrust_cmd = E_pinv @ wrench + held
            elif seg["name"] == "coast":
                thrust_cmd = np.zeros(8)
            else:
                target, yaw, roll, pitch = segment_target(seg, local, anchor, anchor_yaw, box)
                ahead, *_ = segment_target(seg, local + 0.05, anchor, anchor_yaw, box)   # reference velocity of the target
                wrench = pd_wrench(vehicle, cfg, position, R, nu, target, yaw, roll, pitch, v_ref_world=(ahead - target) / 0.05)
                if seg["name"] == "pulse" and local < seg["pulse_seconds"]:
                    peak = np.abs(E).sum(1) * max(abs(t_min), t_max)          # largest wrench the thrusters can make per axis
                    wrench = wrench.copy(); wrench[seg["axis"]] += seg["fraction"] * peak[seg["axis"]]
                thrust_cmd = E_pinv @ wrench
            saturated += int(np.any((thrust_cmd > t_max) | (thrust_cmd < t_min)))
            thrust_cmd = np.clip(thrust_cmd, t_min, t_max)
            if act["input"] == "commands":                                   # commands go through the T200 map (dead band)
                command = t200.command_from_thrust(thrust_cmd, volts)
                thrust_cmd = t200.thrust_from_command(command, volts)
        lag = float(act["motor_lag_seconds"])
        thrust = thrust_cmd if lag <= 0 else thrust + (dt / lag) * (thrust_cmd - thrust)
        wind.step(rng, step, nu, seg.get("gust_multiplier", 1.0))
        f_wind = wind.force()
        position, R, nu = vehicle.heun_step(position, R, nu, E @ thrust, dt, noise_impulse=f_wind * dt)
        u_now = (E @ thrust if act["input"] == "wrench"
                 else t200.command_from_thrust(thrust, volts) if act["input"] == "commands" else thrust)
        acc_u += u_now; acc_thrust += thrust; acc_gust += f_wind
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        min_depth, max_depth, max_tilt = min(min_depth, position[2]), max(max_depth, position[2]), max(max_tilt, tilt)
        if step % sub == 0:                                                 # record: state at t_k, mean input over (t_{k-1}, t_k]
            k = step // sub
            rows[k] = np.r_[position, R.reshape(9), nu, acc_u / sub]
            thrusts[k], gust[k] = acc_thrust / sub, acc_gust / sub
            acc_u, acc_thrust, acc_gust = np.zeros(n_u), np.zeros(8), np.zeros(6)
    saturation = saturated / (n - 1)
    outside = float(np.max([np.maximum(box[k][0] - rows[:, i], rows[:, i] - box[k][1]).max() for i, k in enumerate("xyz")]))
    g = cfg["gates"]
    violation = None
    if not np.all(np.isfinite(rows)): violation = "non-finite state"
    elif max_tilt > g["max_tilt_deg"]: violation = f"tilt {max_tilt:.1f} deg"
    elif min_depth < g["min_depth"]: violation = f"depth {min_depth:.2f} m (surface)"
    elif max_depth > g["max_depth"]: violation = f"depth {max_depth:.2f} m (floor)"
    elif outside > g["envelope_margin"]: violation = f"left envelope by {outside:.2f} m"
    elif saturation > g["max_saturation_fraction"]: violation = f"saturation {saturation:.3f}"
    audit = {"seed": seed, "index": index, "attempt": attempt, "segments": segments, "saturation_fraction": saturation,
             "min_depth": float(min_depth), "max_depth": float(max_depth), "max_tilt_deg": max_tilt,
             "max_envelope_excursion": outside, "violation": violation}
    return rows, {"thrusts": thrusts, "gust": gust}, audit


def generate_split(cfg: dict, seeds: list[int], label: str, library: dict):
    flights, extras, audits_ = [], {"thrusts": [], "gust": []}, []
    for seed in seeds:
        for index in range(int(cfg["seeds"]["flights_per_seed"])):
            for attempt in range(int(cfg["gates"]["max_attempts"])):
                tic = time.perf_counter()
                rows, extra, audit = generate_flight(cfg, seed, index, attempt, library)
                log(f"[{label}] seed={seed} flight={index} attempt={attempt} {time.perf_counter() - tic:.1f}s "
                    f"segments={[s['name'] for s in audit['segments']]} saturation={audit['saturation_fraction']:.3f} violation={audit['violation']}")
                if audit["violation"] is None:
                    break
            else:
                raise RuntimeError(f"{label} seed={seed} flight={index}: no feasible flight in {cfg['gates']['max_attempts']} attempts")
            flights.append(rows); audits_.append(audit)
            for key in extras: extras[key].append(extra[key])
    return np.stack(flights), {k: np.stack(v) for k, v in extras.items()}, audits_


# --------------------------------------------------------------------------- observation noise (as the PyBullet generator)
def exp_so3_batch(v: np.ndarray) -> np.ndarray:
    theta2 = np.sum(v * v, -1); theta = np.sqrt(theta2); small = theta2 < 1e-12
    a = np.where(small, 1 - theta2 / 6, np.sin(theta) / np.where(small, 1, theta))
    b = np.where(small, 0.5 - theta2 / 24, (1 - np.cos(theta)) / np.where(small, 1, theta2))
    K = np.zeros(v.shape[:-1] + (3, 3)); x, y, z = np.moveaxis(v, -1, 0)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0], K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -z, y, z, -x, -y, x
    return np.eye(3) + a[..., None, None] * K + b[..., None, None] * (K @ K)


def perturb_rotations(flights: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    R = flights[..., 3:12].reshape(flights.shape[:-1] + (3, 3))
    return (R @ exp_so3_batch(rng.normal(0, sigma, size=flights.shape[:-1] + (3,)))).reshape(flights.shape[:-1] + (9,))


def absolute_noise(flights: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng, noisy = np.random.default_rng(seed), flights.copy()
    noisy[..., EUCLIDEAN] += rng.normal(0, sigma, size=flights[..., EUCLIDEAN].shape)
    noisy[..., 3:12] = perturb_rotations(flights, sigma, rng)
    return noisy


def sensor_noise(flights: np.ndarray, cfg: dict, seed: int, h: float) -> np.ndarray:
    s, rng, noisy = cfg["observation_noise"]["sensor"], np.random.default_rng(seed), flights.copy()
    noisy[..., :3] += rng.normal(0, float(s["position_sigma"]), size=flights[..., :3].shape)
    noisy[..., 3:12] = perturb_rotations(flights, np.radians(float(s["attitude_sigma_deg"])), rng)
    if s["velocity_from"] == "savitzky_golay":
        from scipy.signal import savgol_filter
        v_world = savgol_filter(noisy[..., :3], int(s["savitzky_golay_window"]), 2, deriv=1, delta=h, axis=1)
    else:
        v_world = np.gradient(noisy[..., :3], h, axis=1)
    R = noisy[..., 3:12].reshape(flights.shape[:-1] + (3, 3))
    noisy[..., 12:15] = np.einsum("fkji,fkj->fki", R, v_world)
    bias = np.cumsum(rng.normal(0, float(s["gyro_bias_random_walk"]) * np.sqrt(h), size=flights[..., 15:18].shape), axis=1)
    noisy[..., 15:18] += bias + rng.normal(0, float(s["gyro_white_sigma"]), size=flights[..., 15:18].shape)
    return noisy


# --------------------------------------------------------------------------- audits, settings, files
def sliding_windows(flights: np.ndarray, points: int, stride: int) -> np.ndarray:
    return np.transpose(np.stack([f[s:s + points] for f in flights for s in range(0, f.shape[0] - points + 1, stride)]), (1, 0, 2))


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a); d = hashlib.sha256()
    d.update(str(a.dtype).encode()); d.update(np.asarray(a.shape, dtype=np.int64).tobytes()); d.update(a.tobytes())
    return d.hexdigest()


def audits(flights: np.ndarray, cfg: dict, h: float) -> dict[str, Any]:
    flat = flights.reshape(-1, flights.shape[-1])
    tilt = np.degrees(np.arccos(np.clip(flat[:, 11], -1, 1)))
    pct = lambda x: [float(np.percentile(x, q)) for q in (50, 90, 100)]
    horizon = {}
    for T in cfg["audits"]["horizons_seconds"]:
        k = int(round(T / h)); d = np.linalg.norm(flights[:, k:, :3] - flights[:, :-k, :3], axis=-1)
        horizon[str(T)] = {"median_position_change_m": float(np.median(d)),
                           "ratio_to_sigma": {str(s): float(np.median(d) / s) for s in cfg["observation_noise"]["absolute_levels"]}}
    R = flat[:, 3:12].reshape(-1, 3, 3)
    return {"horizon_signal": horizon, "input_mean": flat[:, 18:].mean(0).tolist(), "input_std": flat[:, 18:].std(0).tolist(),
            "input_correlation": np.corrcoef(flat[:, 18:].T).tolist(),
            "coverage": {"position_min": flat[:, :3].min(0).tolist(), "position_max": flat[:, :3].max(0).tolist(),
                         "tilt_deg_50_90_max": pct(tilt), "speed_50_90_max": pct(np.linalg.norm(flat[:, 12:15], axis=1)),
                         "angular_rate_50_90_max": pct(np.linalg.norm(flat[:, 15:18], axis=1))},
            "rotation_validity": {"max_orthogonality": float(np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3)).max())}}


def ground_truth(cfg: dict, vehicle: BlueROV2) -> dict[str, Any]:
    p, act = vehicle.p, cfg["actuator"]
    G = {"thrusters": vehicle.E.tolist(), "wrench": np.eye(6).tolist(), "commands": "nonlinear: E @ T200(u, V)"}[act["input"]]
    diff = {ch: {k: v for k, v in cfg["diffusion"][ch].items()} for ch in ("linear", "angular")}
    return {
        "frames": "world z DOWN (NED-like), body forward-right-down", "hamiltonian": "H = 1/2 nu^T M nu + V,  V = (B - W) z + B e3^T R r_b",
        "M_diag": vehicle.M.tolist(), "M_inverse_diag": (1.0 / vehicle.M).tolist(),
        "mass": p.mass, "inertia": list(p.inertia), "added_mass": list(p.added_mass),
        "weight": p.weight, "buoyancy": p.buoyancy, "center_of_buoyancy": list(p.center_of_buoyancy),
        "J": "rigid-body (Kirchhoff) form from M: [P_v x w ; P_v x v + P_w x w]",
        "damping": {"D_linear_diag": vehicle.D_L.tolist(), "D_quadratic_diag": vehicle.D_Q.tolist(),
                    "law": "D(nu) nu = (D_L + D_Q |nu|) nu (per axis)", "types": {ch: cfg["dissipation"][ch]["type"] for ch in ("linear", "angular")}},
        "G": G, "allocation_matrix": vehicle.E.tolist(),
        "diffusion": {**diff, "hold_seconds": cfg["diffusion"]["hold_seconds"],
                      "identifiable_product": "M^-1 Sigma(x) = diag(sigma_lin(x) I3, sigma_ang(x) I3) for constant / rate_dependent "
                                              "(sigma(x) = sigma (1 + gain |v| or |w|)); OU is coloured (no closed-form Sigma(x))"},
    }


def build(cfg: dict) -> dict[str, dict]:
    pl, out = cfg["plant"], cfg["output"]
    h = 1.0 / pl["sample_hz"]
    splits = {}
    for label, seeds, library in split_plan(cfg):
        flights, extra, flight_audits = generate_split(cfg, seeds, label, cfg["libraries"][library])
        splits[label] = {"flights": flights, "extra": extra, "audits": flight_audits, "seeds": seeds, "library": library}
    vehicle = make_vehicle(cfg)
    n_u = 6 if cfg["actuator"]["input"] == "wrench" else 8
    first = next(iter(splits.values()))
    settings = {
        "dataset_name": dataset_name(cfg), "config": copy.deepcopy(cfg), "git_hash": git_hash(), "trajectory_set": cfg["trajectory_set"],
        "system": "BlueROV2 Heavy, port-Hamiltonian SE(3) simulator (envs/rov_se3_port_ham)",
        "ground_truth": ground_truth(cfg, vehicle),
        "vehicle_parameters": {"mass": vehicle.p.mass, "inertia": np.diag(vehicle.p.inertia).tolist(), "gravity_acceleration": vehicle.p.gravity},
        "state_layout": "p_w(3), vec(R)(9) body->world row-major, v_b(3), omega_b(3), u(n_u)",
        "control_layout": {"thrusters": "T1..T8 thruster forces (N), PX4 order", "wrench": "[Fx, Fy, Fz, Mx, My, Mz] (N, N m) body",
                           "commands": "u1..u8 PX4 commands in [-1, 1] (T200 map at actuator.voltage)"}[cfg["actuator"]["input"]],
        "control_dim": n_u, "input_mode": cfg["actuator"]["input"],
        "input_timing": "u in row k = mean input over (t_{k-1}, t_k]; drives row k-1 -> row k",
        "thrusters_px4_order": [{"axis": list(a), "position": list(q)} for a, q in THRUSTERS],
        "sample_dt": h, "physics_hz": pl["physics_hz"], "flight_seconds": cfg["flight"]["duration_seconds"],
        "splits": {label: {"seeds": s["seeds"], "library": s["library"], "segments": list(cfg["libraries"][s["library"]]),
                           "key": SPLIT_KEY[label], "noised_in_variants": label != "eval"} for label, s in splits.items()},
        "observation_noise": {"enabled": False, "test_split_clean": True, "heldout_split_clean": True},
    }
    for label, s in splits.items():
        settings[f"{SPLIT_STEM[label]}_flight_audits"] = s["audits"]
        settings[f"audits_{SPLIT_STEM[label]}"] = audits(s["flights"], cfg, h)
        settings[f"clean_{SPLIT_STEM[label]}_state_sha256"] = array_sha256(s["flights"][..., :STATE_DIM])
    clean = {"t": np.arange(out["window_points"]) * h}
    for label, s in splits.items():
        pre = SPLIT_PREFIX[label]
        clean[SPLIT_KEY[label]] = s["flights"]
        clean[SPLIT_WINDOWS[label]] = sliding_windows(s["flights"], out["window_points"], out["window_stride"])
        clean[f"{pre}thrusts"], clean[f"{pre}gust_wrench"] = s["extra"]["thrusts"], s["extra"]["gust"]
    clean["settings"] = settings
    variants = {"clean": clean}
    if "train" in splits:
        n, train, test = cfg["observation_noise"], splits["train"]["flights"], splits["test"]["flights"]
        for level in n["absolute_levels"]:
            variants[f"train-obs-noise-absolute{level:g}".replace(".", "p")] = noisy_variant(
                clean, absolute_noise(train, level, n["train_noise_seed"]), absolute_noise(test, level, n["test_noise_seed"]),
                {"scheme": "absolute", "level": level}, cfg)
        if n["sensor"]["enabled"]:
            variants["train-sensor-noise"] = noisy_variant(clean, sensor_noise(train, cfg, n["train_noise_seed"], h),
                                                           sensor_noise(test, cfg, n["test_noise_seed"], h), {"scheme": "sensor", **n["sensor"]}, cfg)
    return variants


def noisy_variant(clean: dict, noisy_train: np.ndarray, noisy_test: np.ndarray, record: dict, cfg: dict) -> dict:
    out = cfg["output"]
    s = copy.deepcopy(clean["settings"])
    s["observation_noise"] = {"enabled": True, "test_split_clean": True, "heldout_split_clean": True, "controls_unchanged_by_noise": True,
                              "train_noise_seed": cfg["observation_noise"]["train_noise_seed"], "test_noise_seed": cfg["observation_noise"]["test_noise_seed"],
                              "noisy_training_state_sha256": array_sha256(noisy_train[..., :STATE_DIM]),
                              "train_noise_std_per_channel": (noisy_train[..., :STATE_DIM] - clean["train_trajectories"][..., :STATE_DIM]).std((0, 1)).tolist(), **record}
    s["dataset_name"] = clean["settings"]["dataset_name"] + "-" + record["scheme"]
    return {**clean, "x": sliding_windows(noisy_train, out["window_points"], out["window_stride"]), "train_trajectories": noisy_train,
            "test_x_noisy": sliding_windows(noisy_test, out["window_points"], out["window_stride"]), "test_trajectories_noisy": noisy_test, "settings": s}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_pdf(variants: dict[str, dict], cfg: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    clean, s = variants["clean"], variants["clean"]["settings"]
    h, box = s["sample_dt"], cfg["envelope"]["position_box"]
    parts = [(label, clean[SPLIT_KEY[label]], s[f"{SPLIT_STEM[label]}_flight_audits"]) for label in s["splits"]]
    main_label, main, main_audits = parts[0]
    t = np.arange(main.shape[1]) * h
    names = list(cfg["libraries"]["hard"]) + list(cfg["libraries"]["eval"])
    colours = {n: c for n, c in zip(names, plt.cm.tab20.colors)}
    gt = s["ground_truth"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
        lines = [f"Dataset {s['dataset_name']}   trajectory_set = {cfg['trajectory_set']}   h = {h} s   git {s['git_hash'][:10]}"]
        lines += [f"  {lab:5s}: {arr.shape[0]} flights x {s['flight_seconds']} s, library {s['splits'][lab]['library']} "
                  f"({', '.join(s['splits'][lab]['segments'])}), key {s['splits'][lab]['key']}" for lab, arr, _ in parts]
        lines += [f"M diag {np.round(gt['M_diag'], 3).tolist()}   W - B = {gt['weight'] - gt['buoyancy']:.2f} N   r_b = {gt['center_of_buoyancy']}",
                  f"damping types {gt['damping']['types']}  D_L {np.round(gt['damping']['D_linear_diag'], 2).tolist()}  D_Q {np.round(gt['damping']['D_quadratic_diag'], 2).tolist()}",
                  f"diffusion linear {cfg['diffusion']['linear']['type']}, angular {cfg['diffusion']['angular']['type']};  input {s['input_mode']} (n_u = {s['control_dim']})",
                  f"rejected attempts {sum(a['attempt'] for _, _, aud in parts for a in aud)};  mean saturation {np.mean([a['saturation_fraction'] for a in main_audits]):.3f}",
                  f"noise variants: {', '.join(k for k in variants if k != 'clean') or 'none'}", "", "config:"] + yaml.safe_dump(cfg, sort_keys=False).splitlines()
        ax.text(0.01, 0.99, "\n".join(lines[:80]), va="top", family="monospace", fontsize=5.8); pdf.savefig(fig); plt.close(fig)
        for lab, arr, _ in parts:
            fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_subplot(projection="3d")
            for f in arr: ax.plot(f[:, 0], f[:, 1], f[:, 2], lw=0.6, alpha=0.7)
            ax.set(xlim=box["x"], ylim=box["y"], zlim=box["z"][::-1], xlabel="x (m)", ylabel="y (m)", zlabel="depth z (m, down)",
                   title=f"{lab.upper()} flights in the tank (z down)"); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5)); axes = axes.ravel()
        for ax, (fn, lab) in zip(axes, ((lambda a: np.linalg.norm(a[..., 12:15], axis=-1).ravel(), "speed |v_b| (m/s)"),
                                        (lambda a: np.linalg.norm(a[..., 15:18], axis=-1).ravel(), "angular rate |w_b| (rad/s)"),
                                        (lambda a: a[..., 2].ravel(), "depth z (m)"),
                                        (lambda a: np.degrees(np.arccos(np.clip(a[..., 11].ravel(), -1, 1))), "tilt (deg)"))):
            for label, arr, _ in parts: ax.hist(fn(arr), 50, density=True, histtype="step", lw=1.4, label=label)
            ax.set_title(lab); ax.grid(alpha=0.3); ax.legend()
        fig.suptitle("state coverage by split"); pdf.savefig(fig); plt.close(fig)
        examples = []
        for label, arr, aud in parts:
            count = int(cfg["output"]["pdf_example_flights"]) if label == main_label else 2 if label == "eval" else 0
            examples += [(arr[i], aud[i], f"{label.upper()} flight {i}") for i in range(min(count, arr.shape[0]))]
        for f, aud, title in examples:
            fig, axes = plt.subplots(5, 1, figsize=(11, 8.5), sharex=True)
            yaw = np.degrees(np.unwrap(np.arctan2(f[:, 6], f[:, 3])))
            for ax, y, lab in zip(axes, (f[:, :3], yaw, f[:, 12:15], f[:, 15:18], f[:, 18:]),
                                  ("p (m)", "yaw (deg)", "v_b (m/s)", "w_b (rad/s)", f"u ({s['input_mode']})")):
                ax.plot(t, y, lw=0.8); ax.set_ylabel(lab); ax.grid(alpha=0.3)
                for seg in aud["segments"]: ax.axvspan(seg["start"], seg["start"] + seg["duration"], color=colours.get(seg["name"], "grey"), alpha=0.08)
            axes[0].set_title(f"{title}: " + " > ".join(seg["name"] for seg in aud["segments"])); axes[-1].set_xlabel("t (s)")
            pdf.savefig(fig); plt.close(fig)
        a_main = s[f"audits_{SPLIT_STEM[main_label]}"]
        fig, ax = plt.subplots(figsize=(11, 8.5)); Ts = [float(k) for k in a_main["horizon_signal"]]
        ax.loglog(Ts, [a_main["horizon_signal"][str(T)]["median_position_change_m"] for T in Ts], "o-", label="median position change over T")
        for lvl in cfg["observation_noise"]["absolute_levels"]: ax.axhline(lvl, ls="--", lw=0.8, label=f"absolute noise {lvl}")
        ax.set(xlabel="horizon T (s)", ylabel="m", title="horizon signal versus position noise"); ax.grid(alpha=0.3, which="both"); ax.legend()
        pdf.savefig(fig); plt.close(fig)
        gw = clean[f"{SPLIT_PREFIX[main_label]}gust_wrench"][0]
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        axes[0].plot(t, gw[:, :3]); axes[0].set(title=f"wind force, flight 0 (N), type {cfg['diffusion']['linear']['type']}", ylabel="N")
        axes[1].plot(t, gw[:, 3:]); axes[1].set(title=f"wind torque, flight 0 (N m), type {cfg['diffusion']['angular']['type']}", ylabel="N m", xlabel="t (s)")
        [a.grid(alpha=0.3) for a in axes]; pdf.savefig(fig); plt.close(fig)
    log(f"wrote {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=THIS_DIR / "config.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "datasets",
                        help="parent folder; the dataset goes to <output-root>/ROV-DATASET-<name>/")
    parser.add_argument("--force", action="store_true", help="overwrite existing files of the same dataset")
    parser.add_argument("--no-save", action="store_true", help="generate and log, write nothing")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate(cfg)
    name = dataset_name(cfg)
    out_dir = (args.output_root / f"ROV-DATASET-{cfg['name']}").resolve()
    if not args.no_save and not args.force and list(out_dir.glob(f"{name}_*")):
        raise FileExistsError(f"{out_dir} already holds {name}_* files; pass --force to overwrite or change `name` in the config")
    log(f"dataset {name} -> {out_dir}  (trajectory_set {cfg['trajectory_set']}, input {cfg['actuator']['input']})")
    tic = time.perf_counter()
    variants = build(cfg)
    log(f"generation finished in {(time.perf_counter() - tic) / 60:.1f} min; dataset {name}")
    if args.no_save:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, payload in variants.items():
        with (out_dir / f"{name}_{variant}.pkl").open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"wrote {name}_{variant}.pkl  " + str({k: payload[k].shape for k in SPLIT_KEY.values() if k in payload}))
    (out_dir / f"{name}_audits.json").write_text(json.dumps({k: v for k, v in variants["clean"]["settings"].items() if k != "config"}, indent=1, default=float))
    (out_dir / f"{name}_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if cfg["output"]["write_pdf"]:
        write_pdf(variants, cfg, out_dir / f"{name}_dataset_analysis.pdf")
    (out_dir / f"{name}_generation.log").write_text("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    main()

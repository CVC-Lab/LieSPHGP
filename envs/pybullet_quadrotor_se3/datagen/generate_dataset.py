"""One PyBullet quadrotor dataset generator, driven by one config file (config.yaml next to this script).

    python envs/pybullet_quadrotor_se3/datagen/generate_dataset.py --config envs/pybullet_quadrotor_se3/datagen/config.yaml

Writes datasets/QUADROTOR-DATASET-<name>/<name>_<drone>_<T>s_h<h>_<variant>.pkl (+ audits.json, config_used.yaml,
generation.log, dataset_analysis.pdf). It replaces the five copied generators (HARD, EVALSET, WIND, WIND25, WINDSDE); every
one of those datasets is rebuilt bit-for-bit by setting the config switches below (see the table at the top of config.yaml).

Plant (gym-pybullet-drones CtrlAviary, one stepSimulation per physics step dt = 1/physics_hz):
    m dv_b = (-m w_b x v_b - m g R^T e3 + T e3 + F_diss + F_kick) dt + F_wind dt
    J dw_b = (-w_b x J w_b + tau + tau_kick + tau_diss) dt + tau_wind dt
driven by the DSL PID (gain_scale) tracking random manoeuvre sequences, recorded at sample_hz.

Switches
  trajectory_set   hard      train (seeds.train) + test (seeds.test) from the hard library; noise variants of train/test
                   eval      eval (seeds.eval) from the eval library, stored as heldout_trajectories, always clean
                   hard+eval train + test from the hard library, heldout from the eval library (clean)
  dissipation.<linear|angular>.law
                   none              0
                   constant          F = -c m v_b             tau = -c J w_b                 (external force)
                   speed_dependent   F = -c m (1+|v_b|) v_b   tau = -c J (1+|w_b|) w_b       (external force; angular alias rate_dependent)
                   pybullet_builtin  Bullet's own body damping (changeDynamics linearDamping / angularDamping = c)
  diffusion.<linear|angular>.law   (the wind; body-frame force m*a and torque J*alpha)
                   none              0
                   constant          dv_b += sigma dW                                     (white, redrawn every hold_seconds)
                   speed_dependent   dv_b += sigma (1 + gain |v_b|) dW                    (angular alias rate_dependent, |w_b|)
                   ou                Ornstein-Uhlenbeck: a <- a e^{-dt/T} + s sqrt(1 - e^{-2dt/T}) xi every physics step
                                     (linear: s = sigma_fraction_of_weight * g, angular: s = sigma in rad/s^2)
  recording.kick_torque
                   interval_mean     recorded torque = motors + kick averaged over the sample interval (correct)
                   last_step         recorded torque = motors + kick of the last physics step (HARD / EVALSET / WIND legacy)
  recording.kick_alignment
                   none              a kick starts at its segment's start time, usually mid-sample (all existing datasets)
                   sample            a kick starts on the next sample boundary and lasts whole samples, so it is constant inside
                                     every recorded interval and the stored u is exact (both kick_torque rules then agree)
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
PROJECT_ROOT = THIS_DIR.parents[2]  # envs/pybullet_quadrotor_se3/datagen -> project root
for path in (PROJECT_ROOT, PROJECT_ROOT / "envs/pybullet_quadrotor_se3/gym-pybullet-drones"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pybullet as pb  # noqa: E402
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl  # noqa: E402
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from src.models.SE3_Quadrotor.comparision.report_controller import (  # noqa: E402
    configure_contact_free_dynamics, motor_mixer, remove_ground_plane,
)

STATE_DIM, CONTROL_DIM = 18, 4          # x(3) vec(R)(9) v_b(3) omega_b(3) | wrench(4)
EUCLIDEAN = np.asarray([0, 1, 2, 12, 13, 14, 15, 16, 17])
TRAJECTORY_SETS = ("hard", "eval", "hard+eval")
DISSIPATION_LAWS = ("none", "constant", "speed_dependent", "rate_dependent", "pybullet_builtin")
DIFFUSION_LAWS = ("none", "constant", "speed_dependent", "rate_dependent", "ou")
KICK_SEGMENTS = ("aggressive_recovery", "yaw_kick", "coast_yaw")
LOG_LINES: list[str] = []

# Split label -> key prefix in the pickle. "test" keeps the legacy test_* keys, the eval split the heldout_* keys,
# so every model loader and report selector (@test, @heldout) works unchanged.
SPLIT_PREFIX = {"train": "", "test": "test_", "eval": "heldout_"}
SPLIT_TRAJECTORY_KEY = {"train": "train_trajectories", "test": "test_trajectories", "eval": "heldout_trajectories"}
SPLIT_WINDOW_KEY = {"train": "x", "test": "test_x", "eval": "heldout_x"}
SPLIT_SHA_KEY = {"train": "clean_training_state_sha256", "test": "clean_test_state_sha256", "eval": "clean_heldout_state_sha256"}
SPLIT_STEM = {"train": "train", "test": "test", "eval": "heldout"}          # settings keys <stem>_flight_audits, audits_<stem>


def log(message: str) -> None:
    LOG_LINES.append(message)
    print(message, flush=True)


# --------------------------------------------------------------------------- config
def law(cfg: dict, block: str, channel: str) -> str:
    """Normalised law of dissipation/diffusion channel; rate_dependent is the angular spelling of speed_dependent."""
    name = str(cfg[block][channel]["law"])
    return "speed_dependent" if name == "rate_dependent" else name


def validate(cfg: dict) -> None:
    if cfg["trajectory_set"] not in TRAJECTORY_SETS:
        raise ValueError(f"trajectory_set must be one of {TRAJECTORY_SETS}, got {cfg['trajectory_set']!r}")
    for channel in ("linear", "angular"):
        if cfg["dissipation"][channel]["law"] not in DISSIPATION_LAWS:
            raise ValueError(f"dissipation.{channel}.law must be one of {DISSIPATION_LAWS}")
        if cfg["diffusion"][channel]["law"] not in DIFFUSION_LAWS:
            raise ValueError(f"diffusion.{channel}.law must be one of {DIFFUSION_LAWS}")
    if cfg["recording"]["kick_torque"] not in ("interval_mean", "last_step"):
        raise ValueError("recording.kick_torque must be interval_mean or last_step")
    if cfg["recording"].get("kick_alignment", "none") not in ("none", "sample"):
        raise ValueError("recording.kick_alignment must be none or sample")
    shared = sorted(set(cfg["libraries"]["hard"]) & set(cfg["libraries"]["eval"]))
    if shared:
        raise ValueError(f"the eval library must share no segment with the hard library; shared: {shared}")


def split_plan(cfg: dict) -> list[tuple[str, list[int], str]]:
    """(split label, seeds, library name) in generation order."""
    s = cfg["seeds"]
    hard = [("train", s["train"], "hard"), ("test", s["test"], "hard")]
    return {"hard": hard, "eval": [("eval", s["eval"], "eval")], "hard+eval": hard + [("eval", s["eval"], "eval")]}[cfg["trajectory_set"]]


def dataset_name(cfg: dict) -> str:
    return f"{cfg['name']}_{cfg['plant']['drone_model']}_{int(cfg['flight']['duration_seconds'])}s_h{1.0 / cfg['plant']['sample_hz']:g}".replace(".", "p")


# --------------------------------------------------------------------------- Step 0: environment
def make_env(cfg: dict, xyz: np.ndarray, rpy: np.ndarray) -> CtrlAviary:
    e = cfg["plant"]
    return CtrlAviary(
        drone_model=DroneModel[e["drone_model"]], num_drones=1,
        initial_xyzs=xyz.reshape(1, 3), initial_rpys=rpy.reshape(1, 3),
        physics=Physics[e["physics"]], pyb_freq=int(e["physics_hz"]), ctrl_freq=int(e["physics_hz"]),
        gui=False, record=False, obstacles=False, user_debug_gui=False,
    )


def configure_plant(env: CtrlAviary, cfg: dict) -> dict[str, Any]:
    """Contact-free world, then Bullet's built-in damping on the channels whose law is pybullet_builtin (0 elsewhere)."""
    audit = {"contact_free": bool(cfg["plant"]["contact_free"])}
    if cfg["plant"]["contact_free"]:
        audit["no_ground"] = remove_ground_plane(env)
        audit["dynamics"] = configure_contact_free_dynamics(env)
    builtin = {ch: float(cfg["dissipation"][ch]["c"]) if law(cfg, "dissipation", ch) == "pybullet_builtin" else 0.0 for ch in ("linear", "angular")}
    pb.changeDynamics(int(env.DRONE_IDS[0]), -1, linearDamping=builtin["linear"], angularDamping=builtin["angular"], physicsClientId=int(env.CLIENT))
    audit["builtin_damping"] = builtin
    audit["dissipation_laws"] = {ch: law(cfg, "dissipation", ch) for ch in ("linear", "angular")}
    return audit


def apply_external_dissipation(env: CtrlAviary, state: np.ndarray, cfg: dict, mass: float, inertia: np.ndarray) -> None:
    """External damping for one physics step (body frame): constant -c m v_b / -c J w_b, speed_dependent x (1 + |.|)."""
    lin, ang = law(cfg, "dissipation", "linear"), law(cfg, "dissipation", "angular")
    external = ("constant", "speed_dependent")
    if lin not in external and ang not in external:
        return
    rotation = np.asarray(pb.getMatrixFromQuaternion(state[3:7]), dtype=np.float64).reshape(3, 3)
    if lin in external:
        c, v_body = float(cfg["dissipation"]["linear"]["c"]), rotation.T @ np.asarray(state[10:13], dtype=np.float64)
        force_body = -c * mass * v_body if lin == "constant" else -c * mass * (1.0 + float(np.linalg.norm(v_body))) * v_body
        pb.applyExternalForce(int(env.DRONE_IDS[0]), -1, force_body.tolist(), [0.0, 0.0, 0.0], pb.LINK_FRAME, physicsClientId=int(env.CLIENT))
    if ang in external:
        c, w_body = float(cfg["dissipation"]["angular"]["c"]), rotation.T @ np.asarray(state[13:16], dtype=np.float64)
        torque_body = -c * (inertia @ w_body) if ang == "constant" else -c * (1.0 + float(np.linalg.norm(w_body))) * (inertia @ w_body)
        pb.applyExternalTorque(int(env.DRONE_IDS[0]), -1, torque_body.tolist(), pb.LINK_FRAME, physicsClientId=int(env.CLIENT))


def pack_sample(state: np.ndarray, wrench: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pb.getMatrixFromQuaternion(state[3:7]), dtype=np.float64).reshape(3, 3)
    return np.concatenate((state[:3], rotation.reshape(9), rotation.T @ state[10:13], rotation.T @ state[13:16], wrench))


def tilt_deg(state: np.ndarray) -> float:
    rotation = np.asarray(pb.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))


# --------------------------------------------------------------------------- Step 2: random states
def uniform_box(rng: np.random.Generator, box: dict) -> np.ndarray:
    return np.array([rng.uniform(*box[k]) for k in ("x", "y", "z")])


def random_initial_state(rng: np.random.Generator, cfg: dict) -> dict[str, np.ndarray]:
    s = cfg["initial_state"]
    tilt = np.radians(s["max_tilt_deg"]) * rng.uniform(0, 1)
    axis = rng.normal(size=2); axis /= np.linalg.norm(axis)
    rpy = np.array([tilt * axis[0], tilt * axis[1], rng.uniform(-np.pi, np.pi)])
    v = rng.normal(size=3); v *= rng.uniform(0, s["max_speed"]) / np.linalg.norm(v)
    w = rng.normal(size=3); w *= rng.uniform(0, s["max_angular_rate"]) / np.linalg.norm(w)
    return {"xyz": uniform_box(rng, s["position_box"]), "rpy": rpy, "velocity_world": v, "omega_world": w}


# --------------------------------------------------------------------------- Step 3: manoeuvres
def plan_manoeuvres(rng: np.random.Generator, cfg: dict, library: dict) -> list[dict[str, Any]]:
    """A random sequence of segments from ``library`` covering the flight duration. Targets are resolved at run time."""
    lib = library
    names, weights = list(lib), np.array([lib[n]["weight"] for n in lib], dtype=float)
    total, segments = float(cfg["flight"]["duration_seconds"]), []
    t = 0.0
    while t < total:
        name = names[rng.choice(len(names), p=weights / weights.sum())]
        p = lib[name]
        seg = {"name": name, "start": t, "duration": float(rng.uniform(*p["duration"]))}
        if name in ("figure_eight", "circle"):
            seg["radius"], seg["speed"] = float(rng.uniform(*p["radius"])), float(rng.uniform(*p["speed"]))
        if name == "vertical_step":
            seg["height"] = float(rng.choice([-1, 1]) * rng.uniform(*p["height"]))
        if name == "yaw_turn":
            seg["rate"] = float(rng.choice([-1, 1]) * rng.uniform(*p["rate"]))
        if name == "aggressive_recovery":
            seg["rate"], seg["kick_seconds"] = float(rng.uniform(*p["rate"])), float(p["kick_seconds"])
            axis = rng.normal(size=3); seg["kick_axis"] = (axis / np.linalg.norm(axis)).tolist()
        if name == "yaw_kick":
            # Yaw-rate excitation: an impulsive torque about the body z axis (J_zz * rate over kick_seconds), then
            # PID recovery; gives yaw rates of several rad/s and their free-ish decay, which identifies yaw damping.
            seg["rate"], seg["kick_seconds"] = float(rng.uniform(*p["rate"])), float(p["kick_seconds"])
            seg["kick_axis"] = [0.0, 0.0, float(rng.choice([-1.0, 1.0]))]
        if name == "disturbed_hover":
            seg["gust_multiplier"] = float(p["gust_multiplier"])
        if name in ("coast", "coast_yaw"):
            # Free decay: position loop off, hover thrust, roll/pitch held level by a PD, yaw torque zero. A coast only
            # carries damping information if it starts with speed, so a short circle at coast_entry_speed is inserted
            # first unless the previous segment already moves the vehicle.
            moving = ("figure_eight", "circle", "waypoint_hop", "vertical_step", "dash", "slalom", "helix")
            if not segments or segments[-1]["name"] not in moving:
                entry = {"name": "circle", "start": t, "duration": float(p.get("entry_seconds", 2.0)),
                         "radius": float(p.get("entry_radius", 1.0)), "speed": float(rng.uniform(*p["entry_speed"]))}
                entry["duration"] = min(entry["duration"], total - t)
                segments.append(entry); t += entry["duration"]
                seg["start"] = t
            seg["pd_gains"] = [float(p.get("kp", 1.5e-3)), float(p.get("kd", 2.7e-4))]
            if name == "coast_yaw":
                seg["rate"], seg["kick_seconds"] = float(rng.uniform(*p["rate"])), float(p["kick_seconds"])
                seg["kick_axis"] = [0.0, 0.0, float(rng.choice([-1.0, 1.0]))]
        if name == "waypoint_hop":
            seg["target"] = uniform_box(rng, cfg["envelope"]["position_box"]).tolist(); seg["max_distance"] = float(p["max_distance"])
        if name in ("dash", "slalom"):
            # Straight (dash) or serpentine (slalom) run at a sustained speed. The heading is drawn here and blended
            # toward the box centre at run time, because the anchor is only known then and the run needs room.
            seg["speed"] = float(rng.uniform(*p["speed"]))
            heading = rng.normal(size=2); seg["heading"] = (heading / np.linalg.norm(heading)).tolist()
            if name == "slalom":
                seg["amplitude"], seg["period"] = float(rng.uniform(*p["amplitude"])), float(rng.uniform(*p["period"]))
        if name == "thrust_pulse":
            # Collective thrust at thrust_fraction of weight for cut_seconds, then at (2 - thrust_fraction) of weight for
            # the same time (attitude held level by the coast PD throughout), then PID recovery to the anchor. The two
            # halves cancel in vertical impulse, so the pulse needs no altitude margin beyond min_altitude.
            seg["cut_seconds"], seg["thrust_fraction"] = float(rng.uniform(*p["cut_seconds"])), float(rng.uniform(*p["thrust_fraction"]))
            seg["min_altitude"], seg["pd_gains"] = float(p["min_altitude"]), [float(p.get("kp", 1.5e-3)), float(p.get("kd", 2.7e-4))]
        if name == "helix":
            seg["radius"], seg["speed"] = float(rng.uniform(*p["radius"])), float(rng.uniform(*p["speed"]))
            seg["climb_rate"] = float(rng.choice([-1, 1]) * rng.uniform(*p["climb_rate"]))
        if name == "chirp":
            # Lateral position chirp plus a yaw chirp, frequency swept f0 -> f1 across the segment: inertia dominates
            # the response at high frequency and damping at low, which separates M2^-1 g_tau from M2^-1 D_omega.
            seg["amplitude"], seg["yaw_amplitude"] = float(rng.uniform(*p["amplitude"])), float(rng.uniform(*p["yaw_amplitude"]))
            seg["f0"], seg["f1"] = float(p["f0"]), float(p["f1"])
            heading = rng.normal(size=2); seg["heading"] = (heading / np.linalg.norm(heading)).tolist()
        if name == "bounce":
            seg["height"], seg["period"] = float(rng.uniform(*p["height"])), float(rng.uniform(*p["period"]))
        if name == "tumble":
            # Two torque kicks about different random axes, the second landing mid-recovery: compound rotation about
            # several axes at once. Every hard-library kick is a single impulse.
            seg["rate"], seg["kick_seconds"], seg["gap_seconds"] = float(rng.uniform(*p["rate"])), float(p["kick_seconds"]), float(p["gap_seconds"])
            a1, a2 = rng.normal(size=3), rng.normal(size=3)
            seg["kick_axis"], seg["kick_axis_2"] = (a1 / np.linalg.norm(a1)).tolist(), (a2 / np.linalg.norm(a2)).tolist()
        seg["duration"] = min(seg["duration"], total - t)
        segments.append(seg); t += seg["duration"]
    return segments


def segment_target(seg: dict, t_local: float, anchor_xyz: np.ndarray, anchor_yaw: float, box: dict) -> tuple[np.ndarray, np.ndarray]:
    """Target position and (roll, pitch, yaw) for the PID at local time t inside a segment."""
    name, xyz, yaw = seg["name"], anchor_xyz.copy(), anchor_yaw
    if name == "waypoint_hop":                                   # hop toward the drawn point, at most max_distance away
        step = np.asarray(seg["target"]) - anchor_xyz
        xyz = anchor_xyz + step * min(1.0, seg["max_distance"] / max(np.linalg.norm(step), 1e-9))
    elif name in ("figure_eight", "circle"):
        a, omega = seg["radius"], seg["speed"] / seg["radius"]
        if name == "figure_eight":
            xyz = anchor_xyz + a * np.array([np.sin(omega * t_local), np.sin(2 * omega * t_local), 0.0])
        else:
            xyz = anchor_xyz + a * np.array([np.cos(omega * t_local) - 1.0, np.sin(omega * t_local), 0.0])
    elif name == "vertical_step":
        xyz = anchor_xyz + np.array([0.0, 0.0, seg["height"]])
    elif name == "yaw_turn":
        yaw = anchor_yaw + seg["rate"] * t_local
    elif name in ("dash", "slalom", "chirp"):
        # Heading blended half-and-half toward the box centre so a long run does not leave the envelope at once.
        centre = np.array([(box["x"][0] + box["x"][1]) / 2, (box["y"][0] + box["y"][1]) / 2])
        to_centre = centre - anchor_xyz[:2]; to_centre /= max(np.linalg.norm(to_centre), 1e-9)
        heading = 0.5 * np.asarray(seg["heading"]) + 0.5 * to_centre; heading /= max(np.linalg.norm(heading), 1e-9)
        perpendicular = np.array([-heading[1], heading[0]])
        if name == "dash":
            offset = seg["speed"] * t_local * heading
        elif name == "slalom":
            offset = seg["speed"] * t_local * heading + seg["amplitude"] * np.sin(2 * np.pi * t_local / seg["period"]) * perpendicular
        else:                                                   # chirp: linear frequency sweep f0 -> f1 over the segment
            phase = 2 * np.pi * (seg["f0"] * t_local + 0.5 * (seg["f1"] - seg["f0"]) * t_local**2 / max(seg["duration"], 1e-9))
            offset = seg["amplitude"] * np.sin(phase) * heading
            yaw = anchor_yaw + seg["yaw_amplitude"] * np.sin(phase)
        xyz = anchor_xyz + np.array([offset[0], offset[1], 0.0])
    elif name == "helix":                                        # a circle that climbs (or descends) at a steady rate
        a, omega = seg["radius"], seg["speed"] / seg["radius"]
        xyz = anchor_xyz + np.array([a * (np.cos(omega * t_local) - 1.0), a * np.sin(omega * t_local), seg["climb_rate"] * t_local])
    elif name == "bounce":                                       # square wave in height: every edge is a thrust transient
        up = int(t_local // (seg["period"] / 2)) % 2 == 0
        xyz = anchor_xyz + np.array([0.0, 0.0, seg["height"] if up else 0.0])
    # aggressive_recovery, yaw_kick, disturbed_hover, thrust_pulse, tumble: hold the anchor (the kick / pulse does the work)
    xyz = np.array([np.clip(xyz[i], *box[k]) for i, k in enumerate("xyz")])
    return xyz, np.array([0.0, 0.0, np.arctan2(np.sin(yaw), np.cos(yaw))])


def coast_rpm(state: np.ndarray, pd_gains: list, hover_force: float, mixer_inverse: np.ndarray, kf: float) -> np.ndarray:
    """Free-decay control: thrust = weight, roll/pitch PD to level (torque = -kp*angle - kd*rate), zero yaw torque."""
    rotation = np.asarray(pb.getMatrixFromQuaternion(state[3:7]), dtype=np.float64).reshape(3, 3)
    omega_body = rotation.T @ np.asarray(state[13:16], dtype=np.float64)
    roll, pitch = float(state[7]), float(state[8])
    kp, kd = pd_gains
    wrench = np.array([hover_force, -kp * roll - kd * omega_body[0], -kp * pitch - kd * omega_body[1], 0.0])
    forces = np.maximum(mixer_inverse @ wrench, 0.0)
    return np.sqrt(forces / kf)


def sample_aligned_kick(seg: dict, step: int, sub: int, sample_hz: int) -> tuple[list | None, bool]:
    """Kick axis active in physics step ``step`` when kicks are snapped to the sample grid (recording.kick_alignment: sample).

    A kick starts at the first sample boundary at or after its nominal time and lasts round(kick_seconds * sample_hz) whole
    samples, so it is constant inside every recorded interval and the stored u is exact under any recording rule.
    Physics step s acts on ((s-1) dt, s dt]; sample k covers steps (k-1)*sub+1 .. k*sub. Returns (axis or None, onset).
    """
    if seg["name"] in KICK_SEGMENTS:
        pulses = [(0, seg["kick_axis"])]
    elif seg["name"] == "tumble":                                   # second kick a whole number of samples after the first
        pulses = [(0, seg["kick_axis"]), (int(round(seg["gap_seconds"] * sample_hz)), seg["kick_axis_2"])]
    else:
        return None, False
    boundary = int(np.ceil(round(seg["start"] * sample_hz, 9)))     # first sample boundary at or after the segment start
    length = int(round(seg["kick_seconds"] * sample_hz)) * sub      # physics steps in the kick
    for offset, axis in pulses:
        first = (boundary + offset) * sub + 1
        if first <= step < first + length:
            return axis, step == first
    return None, False


# --------------------------------------------------------------------------- Step 4: wind (the diffusion term)
class Wind:
    """Body-frame wind force m*a and torque J*alpha, one law per channel (see the module docstring).

    Random draws per physics step, in this order: linear channel, then angular channel. White laws draw at the start of
    every hold (hold_seconds) and keep the value; OU draws every physics step. sigma(x) is evaluated at the draw.
    """

    def __init__(self, cfg: dict, mass: float, grav: float, inertia: np.ndarray, dt: float):
        self.cfg, self.mass, self.inertia, self.dt = cfg["diffusion"], mass, inertia, dt
        self.law = {ch: law(cfg, "diffusion", ch) for ch in ("linear", "angular")}
        self.hold_steps = max(1, int(round(float(self.cfg.get("hold_seconds", dt)) / dt)))
        self.force, self.torque = np.zeros(3), np.zeros(3)
        lin, ang = self.cfg["linear"], self.cfg["angular"]
        if self.law["linear"] == "ou":
            self.sigma_force, self.tau_force = float(lin["sigma_fraction_of_weight"]) * mass * grav, float(lin["time_constant_seconds"])
        if self.law["angular"] == "ou":
            self.sigma_alpha, self.tau_alpha, self.alpha = float(ang["sigma"]), float(ang["time_constant_seconds"]), np.zeros(3)

    @property
    def active(self) -> dict[str, bool]:
        return {ch: self.law[ch] != "none" for ch in ("linear", "angular")}

    def _white_scale(self, channel: str, mult: float, magnitude: float) -> float:
        p = self.cfg[channel]
        gain = float(p.get("gain", 0.0)) if self.law[channel] == "speed_dependent" else 0.0
        return mult * float(p["sigma"]) * (1.0 + gain * magnitude)

    def step(self, rng: np.random.Generator, step: int, state: np.ndarray, mult: float) -> None:
        white = [ch for ch in ("linear", "angular") if self.law[ch] in ("constant", "speed_dependent")]
        if white:
            rotation = np.asarray(pb.getMatrixFromQuaternion(state[3:7]), dtype=np.float64).reshape(3, 3)
            speed = float(np.linalg.norm(rotation.T @ np.asarray(state[10:13], dtype=np.float64)))
            rate = float(np.linalg.norm(rotation.T @ np.asarray(state[13:16], dtype=np.float64)))
        new_hold = (step - 1) % self.hold_steps == 0
        hold = self.hold_steps * self.dt
        if self.law["linear"] == "ou":
            self.force = self.force * np.exp(-self.dt / self.tau_force) + mult * self.sigma_force * np.sqrt(1 - np.exp(-2 * self.dt / self.tau_force)) * rng.normal(size=3)
        elif "linear" in white and new_hold:
            self.force = self.mass * self._white_scale("linear", mult, speed) * rng.normal(size=3) / np.sqrt(hold)
        if self.law["angular"] == "ou":
            self.alpha = self.alpha * np.exp(-self.dt / self.tau_alpha) + mult * self.sigma_alpha * np.sqrt(1 - np.exp(-2 * self.dt / self.tau_alpha)) * rng.normal(size=3)
            self.torque = self.inertia @ self.alpha
        elif "angular" in white and new_hold:
            self.torque = self.inertia @ (self._white_scale("angular", mult, rate) * rng.normal(size=3)) / np.sqrt(hold)

    def apply(self, env: CtrlAviary) -> None:
        if self.active["linear"]:
            pb.applyExternalForce(int(env.DRONE_IDS[0]), -1, self.force.tolist(), [0, 0, 0], pb.LINK_FRAME, physicsClientId=int(env.CLIENT))
        if self.active["angular"]:
            pb.applyExternalTorque(int(env.DRONE_IDS[0]), -1, self.torque.tolist(), pb.LINK_FRAME, physicsClientId=int(env.CLIENT))


# --------------------------------------------------------------------------- Steps 0b-5: one flight
def generate_flight(cfg: dict, seed: int, index: int, attempt: int, library: dict) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(int(cfg["seeds"]["flight_seed_base"]) + 1000 * seed + index + 100_000 * attempt)
    init, segments = random_initial_state(rng, cfg), plan_manoeuvres(rng, cfg, library)
    e, act, box = cfg["plant"], cfg["actuator"], cfg["envelope"]["position_box"]
    physics_hz, sample_hz = int(e["physics_hz"]), int(e["sample_hz"])
    sub, dt = physics_hz // sample_hz, 1.0 / physics_hz
    n_samples = int(round(cfg["flight"]["duration_seconds"] * sample_hz)) + 1
    last_step_kick = cfg["recording"]["kick_torque"] == "last_step"
    aligned_kicks = cfg["recording"].get("kick_alignment", "none") == "sample"
    env = make_env(cfg, init["xyz"], init["rpy"])
    try:
        observation, _ = env.reset(seed=int(rng.integers(1 << 31)))
        plant = configure_plant(env, cfg)
        controller = DSLPIDControl(drone_model=DroneModel[e["drone_model"]])
        for name in ("P_COEFF_FOR", "I_COEFF_FOR", "D_COEFF_FOR", "P_COEFF_TOR", "I_COEFF_TOR", "D_COEFF_TOR"):
            setattr(controller, name, float(cfg["controller"]["gain_scale"]) * np.asarray(getattr(controller, name)))
        mixer, mixer_inverse = motor_mixer(env)
        kf, rpm_max, mass, grav = float(env.KF), float(env.MAX_RPM), float(env.M), float(env.G)
        rpm_cap = min(rpm_max, controller.PWM2RPM_SCALE * controller.MAX_PWM + controller.PWM2RPM_CONST)   # the PID's own PWM ceiling
        gain = 1.0 + float(act["motor_gain_spread"]) * rng.uniform(-1, 1, size=4)      # per-motor spread (motor mode realism)
        inertia = np.diag([float(env.J[0, 0]), float(env.J[1, 1]), float(env.J[2, 2])])
        wind = Wind(cfg, mass, grav, inertia, dt)

        pb.resetBaseVelocity(int(env.DRONE_IDS[0]), linearVelocity=init["velocity_world"], angularVelocity=init["omega_world"], physicsClientId=int(env.CLIENT))
        env._updateAndStoreKinematicInformation()
        state = env._getDroneStateVector(0)

        trajectory = np.empty((n_samples, STATE_DIM + CONTROL_DIM)); commands = np.empty((n_samples, 4)); commanded = np.empty((n_samples, 4))
        gusts, gust_torques = np.zeros((n_samples, 3)), np.zeros((n_samples, 3))
        trajectory[0], commands[0], commanded[0] = pack_sample(state, np.zeros(4)), 0.0, 0.0
        rpm_applied, rpm_cmd = np.full(4, float(env.HOVER_RPM)), np.full(4, float(env.HOVER_RPM))
        seg_i, saturated, kicks = 0, 0, []
        kick_torque, kick_sum = np.zeros(3), np.zeros(3)                                    # kick in this step / summed over the sample
        anchor_xyz, anchor_yaw = state[:3].copy(), float(state[9])
        min_alt, max_tilt = float(state[2]), tilt_deg(state)

        for step in range(1, (n_samples - 1) * sub + 1):
            t = step * dt
            seg = segments[seg_i]
            if t >= seg["start"] + seg["duration"] and seg_i + 1 < len(segments):
                seg_i += 1; seg = segments[seg_i]; anchor_xyz, anchor_yaw = state[:3].copy(), float(state[9])
            if (step - 1) % sub == 0:                                                     # PID at sample_hz
                target_xyz, target_rpy = segment_target(seg, t - seg["start"], anchor_xyz, anchor_yaw, box)
                if seg["name"] in ("coast", "coast_yaw"):
                    rpm_cmd = coast_rpm(state, seg["pd_gains"], mass * grav, mixer_inverse, kf)
                elif seg["name"] == "thrust_pulse" and t - seg["start"] < 2 * seg["cut_seconds"] and anchor_xyz[2] >= seg["min_altitude"]:
                    # Cut, then the compensating boost, with the coast PD holding attitude level; the PID resumes after.
                    fraction = seg["thrust_fraction"] if t - seg["start"] < seg["cut_seconds"] else 2.0 - seg["thrust_fraction"]
                    rpm_cmd = coast_rpm(state, seg["pd_gains"], fraction * mass * grav, mixer_inverse, kf)
                else:
                    rpm_cmd, _, _ = controller.computeControlFromState(control_timestep=1.0 / sample_hz, state=state, target_pos=target_xyz, target_rpy=target_rpy)
                rpm_cmd = np.clip(np.asarray(rpm_cmd, dtype=np.float64), 0.0, rpm_max)
            tau_m = float(act["motor_lag_seconds"])                                       # motor lag
            rpm_applied = rpm_cmd if tau_m <= 0 else rpm_applied + (dt / tau_m) * (rpm_cmd - rpm_applied)
            wind.step(rng, step, state, seg.get("gust_multiplier", 1.0))                   # Step 4: wind (diffusion)
            wind.apply(env)
            kick_torque = np.zeros(3)
            kick_axis, local = None, t - seg["start"]
            if aligned_kicks:                                                               # kicks on whole samples only
                kick_axis, kick_onset = sample_aligned_kick(seg, step, sub, sample_hz)
            elif seg["name"] in KICK_SEGMENTS and local < seg["kick_seconds"]:            # torque kick (a shove)
                kick_axis, kick_onset = seg["kick_axis"], local < dt * 1.5
            elif seg["name"] == "tumble":                                                   # two kicks, the second mid-recovery
                if local < seg["kick_seconds"]:
                    kick_axis, kick_onset = seg["kick_axis"], local < dt * 1.5
                elif seg["gap_seconds"] <= local < seg["gap_seconds"] + seg["kick_seconds"]:
                    kick_axis, kick_onset = seg["kick_axis_2"], local - seg["gap_seconds"] < dt * 1.5
            if kick_axis is not None:
                kick_torque = inertia @ np.asarray(kick_axis) * seg["rate"] / seg["kick_seconds"]   # impulse J*rate over kick_seconds
                pb.applyExternalTorque(int(env.DRONE_IDS[0]), -1, kick_torque.tolist(), pb.LINK_FRAME, physicsClientId=int(env.CLIENT))
                if kick_onset:
                    kicks.append({"time": round(t, 3), "torque": kick_torque.tolist()})
            kick_sum += kick_torque
            apply_external_dissipation(env, state, cfg, mass, inertia)
            observation, *_ = env.step((rpm_applied * np.sqrt(gain)).reshape(1, 4))     # PyBullet applies k_f (rpm sqrt(gain))^2
            state = observation[0]
            saturated += int(np.any(rpm_cmd >= rpm_cap - 1.0))                             # a motor at the ceiling
            min_alt, max_tilt = min(min_alt, float(state[2])), max(max_tilt, tilt_deg(state))
            if step % sub == 0:                                                           # record at sample_hz
                k = step // sub
                wrench = mixer @ (kf * gain * rpm_applied**2)
                # recorded torque = motors + external kick. interval_mean averages the kick over the sample interval: a
                # 0.1 s kick can switch on/off mid-interval, and the end-of-interval value (last_step, the legacy record)
                # left those samples' torque wrong (0.28 % of samples; 25 Sep 2026).
                wrench[1:] += kick_torque if last_step_kick else kick_sum / sub
                kick_sum = np.zeros(3)
                trajectory[k], commands[k], commanded[k] = pack_sample(state, wrench), (rpm_applied / rpm_max) ** 2, (rpm_cmd / rpm_max) ** 2
                gusts[k], gust_torques[k] = wind.force, wind.torque
    finally:
        env.close()
    saturation = saturated / ((n_samples - 1) * sub)
    outside = float(np.max([np.maximum(box[k][0] - trajectory[:, i], trajectory[:, i] - box[k][1]).max() for i, k in enumerate("xyz")]))
    g = cfg["gates"]
    violation = None
    if max_tilt > g["max_tilt_deg"]: violation = f"tilt {max_tilt:.1f} deg"
    elif min_alt < g["min_altitude"]: violation = f"altitude {min_alt:.2f} m"
    elif outside > g["envelope_margin"]: violation = f"left envelope by {outside:.2f} m"
    elif saturation > g["max_saturation_fraction"]: violation = f"saturation {saturation:.3f}"
    audit = {"seed": seed, "index": index, "attempt": attempt, "initial": {k: np.asarray(v).tolist() for k, v in init.items()},
             "segments": segments, "kicks": kicks, "saturation_fraction": saturation, "min_altitude": min_alt, "max_tilt_deg": max_tilt,
             "max_envelope_excursion": outside, "motor_gain": gain.tolist(), "violation": violation, "rpm_cap": rpm_cap,
             "plant": {"mass": mass, "inertia_diagonal": np.diag(inertia).tolist(), "kf": kf, "km": float(env.KM), "arm": float(env.L),
                       "gravity_acceleration": grav, "hover_rpm": float(env.HOVER_RPM), "maximum_rpm": rpm_max, **plant}}
    return trajectory, {"commands": commands, "commanded": commanded, "gusts": gusts, "gust_torques": gust_torques}, audit


def generate_split(cfg: dict, seeds: list[int], label: str, library: dict) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    flights, extras, audits_ = [], {"commands": [], "commanded": [], "gusts": [], "gust_torques": []}, []
    for seed in seeds:
        for index in range(int(cfg["seeds"]["flights_per_seed"])):
            for attempt in range(int(cfg["gates"]["max_attempts"])):
                tic = time.perf_counter()
                trajectory, extra, audit = generate_flight(cfg, seed, index, attempt, library)
                log(f"[{label}] seed={seed} flight={index} attempt={attempt} {time.perf_counter() - tic:.1f}s "
                    f"segments={[s['name'] for s in audit['segments']]} saturation={audit['saturation_fraction']:.3f} violation={audit['violation']}")
                if audit["violation"] is None:
                    break
            else:
                raise RuntimeError(f"{label} seed={seed} flight={index}: no feasible flight in {cfg['gates']['max_attempts']} attempts")
            flights.append(trajectory); audits_.append(audit)
            for key in extras: extras[key].append(extra[key])
    return np.stack(flights), {k: np.stack(v) for k, v in extras.items()}, audits_


# --------------------------------------------------------------------------- Step 7: observation-noise variants
def hat(v: np.ndarray) -> np.ndarray:
    out = np.zeros(v.shape[:-1] + (3, 3)); x, y, z = np.moveaxis(v, -1, 0)
    out[..., 0, 1], out[..., 0, 2], out[..., 1, 0], out[..., 1, 2], out[..., 2, 0], out[..., 2, 1] = -z, y, z, -x, -y, x
    return out


def exp_so3(v: np.ndarray) -> np.ndarray:
    theta2 = np.sum(v * v, -1); theta = np.sqrt(theta2); small = theta2 < 1e-12
    a = np.where(small, 1 - theta2 / 6, np.sin(theta) / np.where(small, 1, theta))
    b = np.where(small, 0.5 - theta2 / 24, (1 - np.cos(theta)) / np.where(small, 1, theta2))
    K = hat(v)
    return np.eye(3) + a[..., None, None] * K + b[..., None, None] * (K @ K)


def perturb_rotations(flights: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    R = flights[..., 3:12].reshape(flights.shape[:-1] + (3, 3))
    eta = rng.normal(0, sigma, size=flights.shape[:-1] + (3,))
    return (R @ exp_so3(eta)).reshape(flights.shape[:-1] + (9,))


def absolute_noise(flights: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Additive Gaussian on x, v_b, omega_b; R * Exp(eta) on rotations; controls untouched."""
    rng, noisy = np.random.default_rng(seed), flights.copy()
    noisy[..., EUCLIDEAN] += rng.normal(0, sigma, size=flights[..., EUCLIDEAN].shape)
    noisy[..., 3:12] = perturb_rotations(flights, sigma, rng)
    return noisy


def sensor_noise(flights: np.ndarray, cfg: dict, seed: int, h: float) -> np.ndarray:
    """Motion capture + IMU: mm pose noise, velocity by differencing noisy positions, gyro white noise + bias walk."""
    s, rng, noisy = cfg["observation_noise"]["sensor"], np.random.default_rng(seed), flights.copy()
    noisy[..., :3] += rng.normal(0, float(s["position_sigma"]), size=flights[..., :3].shape)
    noisy[..., 3:12] = perturb_rotations(flights, np.radians(float(s["attitude_sigma_deg"])), rng)
    x = noisy[..., :3]
    if s["velocity_from"] == "savitzky_golay":
        from scipy.signal import savgol_filter
        v_world = savgol_filter(x, int(s["savitzky_golay_window"]), 2, deriv=1, delta=h, axis=1)
    else:
        v_world = np.gradient(x, h, axis=1)                                 # central difference (one-sided at the ends)
    R = noisy[..., 3:12].reshape(flights.shape[:-1] + (3, 3))
    noisy[..., 12:15] = np.einsum("fkji,fkj->fki", R, v_world)               # R^T v_world with the noisy rotation
    bias = np.cumsum(rng.normal(0, float(s["gyro_bias_random_walk"]) * np.sqrt(h), size=flights[..., 15:18].shape), axis=1)
    noisy[..., 15:18] += bias + rng.normal(0, float(s["gyro_white_sigma"]), size=flights[..., 15:18].shape)
    return noisy


# --------------------------------------------------------------------------- Step 8: audits
def sliding_windows(flights: np.ndarray, points: int, stride: int) -> np.ndarray:
    n = flights.shape[1]
    return np.transpose(np.stack([f[s:s + points] for f in flights for s in range(0, n - points + 1, stride)]), (1, 0, 2))


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a); d = hashlib.sha256()
    d.update(str(a.dtype).encode()); d.update(np.asarray(a.shape, dtype=np.int64).tobytes()); d.update(a.tobytes())
    return d.hexdigest()


def rotation_validity(flights: np.ndarray) -> dict[str, float]:
    R = flights[..., 3:12].reshape(-1, 3, 3)
    return {"max_orthogonality_frobenius": float(np.linalg.norm(np.swapaxes(R, 1, 2) @ R - np.eye(3), axis=(1, 2)).max()),
            "max_abs_det_minus_one": float(np.abs(np.linalg.det(R) - 1).max())}


def audits(flights: np.ndarray, cfg: dict, h: float) -> dict[str, Any]:
    flat = flights.reshape(-1, flights.shape[-1])
    u = flat[:, 18:22]
    horizon = {}
    for T in cfg["audits"]["horizons_seconds"]:
        k = int(round(T / h)); d = np.linalg.norm(flights[:, k:, :3] - flights[:, :-k, :3], axis=-1)
        horizon[str(T)] = {"median_position_change_m": float(np.median(d)), "ratio_to_sigma": {str(s): float(np.median(d) / s) for s in cfg["observation_noise"]["absolute_levels"]}}
    spectrum = np.mean([np.abs(np.fft.rfft(f[:, 18:22] - f[:, 18:22].mean(0), axis=0)) ** 2 for f in flights], axis=0)
    tilt = np.degrees(np.arccos(np.clip(flat[:, 11], -1, 1)))
    box = cfg["envelope"]["position_box"]
    return {
        "horizon_signal": horizon,
        "input_correlation": np.corrcoef(u.T).tolist(), "input_std": u.std(0).tolist(), "input_mean": u.mean(0).tolist(),
        "input_psd_frequencies_hz": np.fft.rfftfreq(flights.shape[1], h).tolist(), "input_psd": spectrum.tolist(),
        "coverage": {"position_min": flat[:, :3].min(0).tolist(), "position_max": flat[:, :3].max(0).tolist(),
                     "tilt_deg_percentiles_50_90_max": [float(np.percentile(tilt, 50)), float(np.percentile(tilt, 90)), float(tilt.max())],
                     "speed_percentiles_50_90_max": [float(np.percentile(np.linalg.norm(flat[:, 12:15], axis=1), q)) for q in (50, 90, 100)],
                     "angular_rate_percentiles_50_90_max": [float(np.percentile(np.linalg.norm(flat[:, 15:18], axis=1), q)) for q in (50, 90, 100)],
                     "envelope_fraction_visited": float(np.mean([(np.histogram(flat[:, i], 10, box[k])[0] > 0).mean() for i, k in enumerate("xyz")]))},
        "rotation_validity": rotation_validity(flights),
    }


# --------------------------------------------------------------------------- Step 9: settings and files
def legacy_damping_law(cfg: dict) -> str:
    """The single-word law the model loaders and reports read: linear = -c v, nonlinear = -c (1 + |v|) v."""
    laws = {law(cfg, "dissipation", ch) for ch in ("linear", "angular")}
    if laws <= {"constant", "none"}:
        return "linear"
    if laws <= {"pybullet_builtin", "speed_dependent"}:
        return "nonlinear"
    return "mixed"


def dissipation_record(cfg: dict) -> dict[str, Any]:
    out = {}
    for ch, (v, J) in (("linear", ("v_b", "m")), ("angular", ("w_b", "J"))):
        name = law(cfg, "dissipation", ch); c = float(cfg["dissipation"][ch]["c"]) if name != "none" else 0.0
        out[ch] = {"law": name, "c": c, "equation": {
            "none": "0", "constant": f"-c {J} {v}", "speed_dependent": f"-c {J} (1+|{v}|) {v}",
            "pybullet_builtin": f"Bullet body damping with coefficient c (documented as -c {J} (1+|{v}|) {v})"}[name]}
    return out


def diffusion_record(cfg: dict) -> dict[str, Any]:
    """Ground-truth wind. When both channels are white (none/constant/speed_dependent) the legacy white_state_dependent
    block is filled, which evaluate_windsde_ground_truth.py reads; M^-1 Sigma(x) = diag(sigma_a(x) I3, sigma_alpha(x) I3)."""
    d = cfg["diffusion"]
    laws = {ch: law(cfg, "diffusion", ch) for ch in ("linear", "angular")}
    record = {"linear": {"law": laws["linear"], **{k: v for k, v in d["linear"].items() if k != "law"}},
              "angular": {"law": laws["angular"], **{k: v for k, v in d["angular"].items() if k != "law"}},
              "hold_seconds": d.get("hold_seconds")}
    if set(laws.values()) == {"none"}:
        return {"model": "none", **record}
    if "ou" not in laws.values():
        def sg(ch):
            if laws[ch] == "none":
                return 0.0, 0.0
            return float(d[ch]["sigma"]), float(d[ch].get("gain", 0.0)) if laws[ch] == "speed_dependent" else 0.0
        (ls, lg), (as_, ag) = sg("linear"), sg("angular")
        return {"model": "white_state_dependent",
                "equation": "dp_v += m*sigma_a(x) dW, dp_w += J*sigma_alpha(x) dW; sigma_a = linear_sigma*(1+speed_gain*|v_b|), "
                            "sigma_alpha = angular_sigma*(1+rate_gain*|omega_b|)",
                "identifiable_product": "M^-1 Sigma(x) = diag(sigma_a(x) I_3, sigma_alpha(x) I_3)  (twist units per sqrt(s))",
                "linear_sigma": ls, "speed_gain": lg, "angular_sigma": as_, "rate_gain": ag, **record}
    return {"model": "ou" if set(laws.values()) <= {"ou", "none"} else "mixed",
            "note": "OU wind is coloured noise: the state alone is not Markov, so there is no closed-form Sigma(x)", **record}


def build(cfg: dict) -> dict[str, dict]:
    e, out = cfg["plant"], cfg["output"]
    h = 1.0 / e["sample_hz"]
    splits = {}
    for label, seeds, library in split_plan(cfg):
        flights, extra, flight_audits = generate_split(cfg, seeds, label, cfg["libraries"][library])
        splits[label] = {"flights": flights, "extra": extra, "audits": flight_audits, "seeds": seeds, "library": library}
    first = next(iter(splits.values()))
    plant = first["audits"][0]["plant"]
    t = np.arange(first["flights"].shape[1]) * h
    diss = dissipation_record(cfg)
    settings = {
        "dataset_name": dataset_name(cfg), "config": copy.deepcopy(cfg), "git_hash": git_hash(), "trajectory_set": cfg["trajectory_set"],
        "vehicle_parameters": {"mass": plant["mass"], "inertia": np.diag(plant["inertia_diagonal"]).tolist(),
                               "gravity_acceleration": plant["gravity_acceleration"], "arm": plant["arm"],
                               "kf": plant["kf"], "km": plant["km"], "maximum_rpm": plant["maximum_rpm"]},
        "dissipation": diss, "damping_law": legacy_damping_law(cfg),
        "linear_damping_coefficient": diss["linear"]["c"], "angular_damping_coefficient": diss["angular"]["c"],
        "damping_equation": f"linear: {diss['linear']['equation']}; angular: {diss['angular']['equation']}",
        "pybullet_builtin_damping": any(diss[ch]["law"] == "pybullet_builtin" for ch in diss),
        "custom_external_linear_damping": any(diss[ch]["law"] in ("constant", "speed_dependent") for ch in diss),
        "diffusion_ground_truth": diffusion_record(cfg), "kick_torque_recording": cfg["recording"]["kick_torque"],
        "kick_alignment": cfg["recording"].get("kick_alignment", "none"),
        "state_layout": "x_w(3), vec(R)(9), v_b(3), omega_b(3), wrench(4)",
        "control_layout": "wrench [T, tau_x, tau_y, tau_z]: motor wrench after rpm clipping plus the external kick torque of recovery segments",
        "input_mode": cfg["actuator"]["input_mode"], "true_control_map": "selection matrix S" if cfg["actuator"]["input_mode"] == "wrench" else "S @ M_mix @ kf @ rpm_max^2",
        "motor_command_layout": "(rpm_i / rpm_max)^2 applied; 'motor_commands_commanded' before the motor lag",
        "sample_dt": h, "physics_hz": e["physics_hz"], "flight_seconds": cfg["flight"]["duration_seconds"],
        "splits": {label: {"seeds": s["seeds"], "library": s["library"], "segments": list(cfg["libraries"][s["library"]]),
                           "key": SPLIT_TRAJECTORY_KEY[label], "noised_in_variants": label != "eval"} for label, s in splits.items()},
        "observation_noise": {"enabled": False, "test_split_clean": True, "heldout_split_clean": True},
    }
    for label, s in splits.items():
        stem = SPLIT_STEM[label]
        settings[f"{stem}_flight_audits"] = s["audits"]
        settings[f"audits_{stem}"] = audits(s["flights"], cfg, h)
        settings[SPLIT_SHA_KEY[label]] = array_sha256(s["flights"][..., :STATE_DIM])
    clean = {"t": t[:out["window_points"]]}
    for label, s in splits.items():
        p = SPLIT_PREFIX[label]
        clean[SPLIT_TRAJECTORY_KEY[label]] = s["flights"]
        clean[SPLIT_WINDOW_KEY[label]] = sliding_windows(s["flights"], out["window_points"], out["window_stride"])
        clean[f"{p}motor_commands"], clean[f"{p}motor_commands_commanded"] = s["extra"]["commands"], s["extra"]["commanded"]
        clean[f"{p}gust_force"], clean[f"{p}gust_torque"] = s["extra"]["gusts"], s["extra"]["gust_torques"]
    clean["settings"] = settings
    variants = {"clean": clean}
    if "train" in splits:                                    # observation noise only on train/test; eval stays clean
        n, train, test = cfg["observation_noise"], splits["train"]["flights"], splits["test"]["flights"]
        for level in n["absolute_levels"]:
            variants[f"train-obs-noise-absolute{level:g}".replace(".", "p")] = _noisy_variant(clean, absolute_noise(train, level, n["train_noise_seed"]), absolute_noise(test, level, n["test_noise_seed"]),
                {"scheme": "absolute", "level": level, "equation": "y~=y+sigma*eps on x,v_b,omega_b; R~=R*Exp(eta), eta~N(0,sigma^2 I)"}, cfg)
        if n["sensor"]["enabled"]:
            variants["train-sensor-noise"] = _noisy_variant(clean, sensor_noise(train, cfg, n["train_noise_seed"], h), sensor_noise(test, cfg, n["test_noise_seed"], h),
                {"scheme": "sensor", **n["sensor"], "equation": "x~=x+sigma_x eps; R~=R Exp(eta); v_b=R~^T d/dt x~; omega~=omega+bias_walk+sigma_g eps"}, cfg)
    return variants


def _noisy_variant(clean: dict, noisy_train: np.ndarray, noisy_test: np.ndarray, record: dict, cfg: dict) -> dict:
    out = cfg["output"]
    s = copy.deepcopy(clean["settings"])
    s["observation_noise"] = {"enabled": True, "test_split_clean": True, "heldout_split_clean": True, "noisy_test_split_saved": True, "controls_unchanged_by_noise": True,
                              "noise_added_before_windowing": True, "train_noise_seed": cfg["observation_noise"]["train_noise_seed"], "test_noise_seed": cfg["observation_noise"]["test_noise_seed"],
                              "noisy_training_state_sha256": array_sha256(noisy_train[..., :STATE_DIM]), "rotation_validity": rotation_validity(noisy_train),
                              "train_noise_std_per_channel": (noisy_train[..., :STATE_DIM] - clean["train_trajectories"][..., :STATE_DIM]).std((0, 1)).tolist(), **record}
    s["dataset_name"] = clean["settings"]["dataset_name"] + "-" + record["scheme"]
    return {**clean, "x": sliding_windows(noisy_train, out["window_points"], out["window_stride"]), "train_trajectories": noisy_train,
            "test_x_noisy": sliding_windows(noisy_test, out["window_points"], out["window_stride"]), "test_trajectories_noisy": noisy_test, "settings": s}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def save(path: Path, payload: dict, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    shapes = {k: payload[k].shape for k in SPLIT_TRAJECTORY_KEY.values() if k in payload}
    log(f"wrote {path.name}  {shapes}")


# --------------------------------------------------------------------------- Step 10: PDF
def write_pdf(variants: dict[str, dict], cfg: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    clean, s = variants["clean"], variants["clean"]["settings"]
    h = s["sample_dt"]
    parts = [(label, clean[SPLIT_TRAJECTORY_KEY[label]], s[f"{SPLIT_STEM[label]}_flight_audits"]) for label in s["splits"]]
    main_label, main, main_audits = parts[0]
    a_main = s[f"audits_{SPLIT_STEM[main_label]}"]
    t = np.arange(main.shape[1]) * h
    box = cfg["envelope"]["position_box"]
    colours = {n: c for n, c in zip(list(cfg["libraries"]["hard"]) + list(cfg["libraries"]["eval"]), plt.cm.tab20.colors)}
    with PdfPages(path) as pdf:
        # 1 summary + config
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
        rej = sum(a["attempt"] for _, _, aud in parts for a in aud)
        lines = [f"Dataset {s['dataset_name']}   trajectory_set = {cfg['trajectory_set']}   h = {h} s   git {s['git_hash'][:10]}"]
        lines += [f"  {label:5s}: {arr.shape[0]} flights x {s['flight_seconds']} s, library {s['splits'][label]['library']} "
                  f"({', '.join(s['splits'][label]['segments'])}), key {s['splits'][label]['key']}" for label, arr, _ in parts]
        lines += [f"dissipation: linear {s['dissipation']['linear']['equation']} (c={s['dissipation']['linear']['c']}), "
                  f"angular {s['dissipation']['angular']['equation']} (c={s['dissipation']['angular']['c']})",
                  f"diffusion: linear {s['diffusion_ground_truth']['linear']['law']}, angular {s['diffusion_ground_truth']['angular']['law']} "
                  f"(model {s['diffusion_ground_truth']['model']}); kick torque recorded as {s['kick_torque_recording']}, kicks aligned to {s['kick_alignment']}",
                  f"rejected attempts {rej}; mean saturation {np.mean([a['saturation_fraction'] for a in main_audits]):.4f}",
                  f"noise variants: {', '.join(k for k in variants if k != 'clean') or 'none'}", "", "config:"] + yaml.safe_dump(cfg, sort_keys=False).splitlines()
        ax.text(0.01, 0.99, "\n".join(lines[:75]), va="top", family="monospace", fontsize=6.0); pdf.savefig(fig); plt.close(fig)
        # 2 3-D flights per split
        for label, arr, _ in parts:
            fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_subplot(projection="3d")
            for f in arr: ax.plot(f[:, 0], f[:, 1], f[:, 2], lw=0.6, alpha=0.7)
            ax.set(xlim=box["x"], ylim=box["y"], zlim=box["z"], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=f"{label.upper()} flights over the envelope box")
            pdf.savefig(fig); plt.close(fig)
        # 3 state coverage by split
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5)); axes = axes.ravel()
        for ax, (fn, lab, lim) in zip(axes, ((lambda a: np.linalg.norm(a[..., 12:15], axis=-1).ravel(), "speed |v_b| (m/s)", cfg["envelope"]["max_speed"]),
                                             (lambda a: np.linalg.norm(a[..., 15:18], axis=-1).ravel(), "angular rate |omega_b| (rad/s)", cfg["envelope"]["max_angular_rate"]),
                                             (lambda a: a[..., 18].ravel() / (s["vehicle_parameters"]["mass"] * s["vehicle_parameters"]["gravity_acceleration"]), "collective thrust T / (m g)", None),
                                             (lambda a: np.degrees(np.arccos(np.clip(a[..., 11].ravel(), -1, 1))), "tilt (deg)", cfg["envelope"]["max_tilt_deg"]))):
            for label, arr, _ in parts: ax.hist(fn(arr), 50, density=True, histtype="step", lw=1.4, label=label)
            if lim is not None: ax.axvline(lim, c="r", lw=0.8)
            ax.set_title(lab); ax.grid(alpha=0.3); ax.legend()
        fig.suptitle("state coverage by split (densities)"); pdf.savefig(fig); plt.close(fig)
        # 4 example flights
        examples = []
        for label, arr, aud in parts:
            count = int(cfg["output"]["pdf_example_flights"]) if label == main_label else 2 if label == "eval" else 0
            examples += [(arr[i], aud[i], f"{label.upper()} flight {i}") for i in range(min(count, arr.shape[0]))]
        for f, aud, title in examples:
            rpy = Rotation.from_matrix(f[:, 3:12].reshape(-1, 3, 3)).as_euler("xyz")
            fig, axes = plt.subplots(5, 1, figsize=(11, 8.5), sharex=True)
            for ax, y, lab in zip(axes, (f[:, :3], rpy, f[:, 12:15], f[:, 15:18], f[:, 18:22]), ("x (m)", "roll pitch yaw (rad)", "v_b (m/s)", "omega_b (rad/s)", "wrench (N, N m)")):
                ax.plot(t, y, lw=0.8); ax.set_ylabel(lab); ax.grid(alpha=0.3)
                for seg in aud["segments"]: ax.axvspan(seg["start"], seg["start"] + seg["duration"], color=colours.get(seg["name"], "grey"), alpha=0.08)
            axes[0].set_title(f"{title}: " + " > ".join(seg["name"] for seg in aud["segments"])); axes[-1].set_xlabel("t (s)"); pdf.savefig(fig); plt.close(fig)
        # 5 coverage of the main split
        flat = main.reshape(-1, 22); fig, axes = plt.subplots(2, 3, figsize=(11, 8.5)); axes = axes.ravel()
        for ax, (i, k) in zip(axes[:3], enumerate("xyz")): ax.hist(flat[:, i], 40); ax.axvline(box[k][0], c="r"); ax.axvline(box[k][1], c="r"); ax.set_title(f"{k} (m)")
        axes[3].hist(np.degrees(np.arccos(np.clip(flat[:, 11], -1, 1))), 40); axes[3].axvline(cfg["envelope"]["max_tilt_deg"], c="r"); axes[3].set_title("tilt (deg)")
        axes[4].hist(np.linalg.norm(flat[:, 12:15], axis=1), 40); axes[4].axvline(cfg["envelope"]["max_speed"], c="r"); axes[4].set_title("speed (m/s)")
        axes[5].hist(np.linalg.norm(flat[:, 15:18], axis=1), 40); axes[5].axvline(cfg["envelope"]["max_angular_rate"], c="r"); axes[5].set_title("angular rate (rad/s)")
        fig.suptitle(f"coverage of the {main_label} flights (red: envelope limits)"); pdf.savefig(fig); plt.close(fig)
        # 6 inputs
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), constrained_layout=True); axes = axes.ravel()
        for j, lab in enumerate(("T (N)", "tau_x (N m)", "tau_y (N m)", "tau_z (N m)")): axes[j].hist(flat[:, 18 + j], 40); axes[j].set_title(lab)
        im = axes[4].imshow(np.asarray(a_main["input_correlation"]), vmin=-1, vmax=1, cmap="coolwarm"); axes[4].set_title("input correlation (measured, not engineered)"); fig.colorbar(im, ax=axes[4])
        freqs, psd = np.asarray(a_main["input_psd_frequencies_hz"]), np.asarray(a_main["input_psd"])
        for j, lab in enumerate(("T", "tau_x", "tau_y", "tau_z")): axes[5].loglog(freqs[1:], psd[1:, j] / psd[1:, j].max(), lw=0.8, label=lab)
        axes[5].set(title="input power spectrum (normalised)", xlabel="Hz"); axes[5].legend(); pdf.savefig(fig); plt.close(fig)
        # 7 horizon signal vs noise
        fig, ax = plt.subplots(figsize=(11, 8.5)); Ts = [float(k) for k in a_main["horizon_signal"]]
        ax.loglog(Ts, [a_main["horizon_signal"][str(T)]["median_position_change_m"] for T in Ts], "o-", label="median position change over horizon T")
        for lvl in cfg["observation_noise"]["absolute_levels"]: ax.axhline(lvl, ls="--", lw=0.8, label=f"absolute noise {lvl}")
        ax.set(xlabel="horizon T (s)", ylabel="m", title="horizon signal versus position noise"); ax.grid(alpha=0.3, which="both"); ax.legend(); pdf.savefig(fig); plt.close(fig)
        # 8 clean vs noisy overlays
        for name, var in variants.items():
            if name == "clean": continue
            f0, fn = clean["train_trajectories"][0], var["train_trajectories"][0]
            fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
            for ax, sl, lab in zip(axes, (slice(0, 3), slice(12, 15), slice(15, 18), slice(3, 12)), ("x (m)", "v_b (m/s)", "omega_b (rad/s)", "vec(R)")):
                ax.plot(t, fn[:, sl], lw=0.5, alpha=0.6); ax.set_prop_cycle(None); ax.plot(t, f0[:, sl], lw=1.2); ax.set_ylabel(lab); ax.grid(alpha=0.3)
            axes[0].set_title(f"{name}: noisy (thin) versus clean (thick), flight 0; per-channel std {np.round(var['settings']['observation_noise']['train_noise_std_per_channel'][:3], 4).tolist()} ..."); pdf.savefig(fig); plt.close(fig)
        # 9 wind force and torque
        p = SPLIT_PREFIX[main_label]
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5))
        g, gt = clean[f"{p}gust_force"][0], clean[f"{p}gust_torque"][0]
        axes[0].plot(t, g); axes[0].set(title=f"wind force, flight 0 (N), law {s['diffusion_ground_truth']['linear']['law']}", ylabel="N"); axes[0].grid(alpha=0.3)
        axes[1].plot(t, gt); axes[1].set(title=f"wind torque, flight 0 (N m), law {s['diffusion_ground_truth']['angular']['law']}", ylabel="N m"); axes[1].grid(alpha=0.3)
        for series, lab in ((g[:, 0], "force x"), (gt[:, 0], "torque x")):
            centred = series - series.mean()
            if np.any(centred):
                ac = np.correlate(centred, centred, "full")[len(centred) - 1:]; ac /= ac[0]
                axes[2].plot(t[:300], ac[:300], label=f"{lab} autocorrelation")
        axes[2].set(title="wind autocorrelation (OU: exp(-t/T); white: delta at the hold)", xlabel="lag (s)"); axes[2].grid(alpha=0.3)
        if axes[2].lines: axes[2].legend()
        pdf.savefig(fig); plt.close(fig)
        # 10 actuator model
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        axes[0].plot(t, main[0][:, 18:22]); axes[0].set(title=f"actuator: input mode {s['input_mode']}, motor gain spread {cfg['actuator']['motor_gain_spread']}, lag {cfg['actuator']['motor_lag_seconds']} s", ylabel="wrench u_f")
        axes[1].plot(t, clean[f"{p}motor_commands"][0]); axes[1].set(ylabel="(rpm_i/rpm_max)^2", xlabel="t (s)"); [ax.grid(alpha=0.3) for ax in axes]; pdf.savefig(fig); plt.close(fig)
        # 11 validity + reproducibility
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
        rows = [f"{label} clean state sha256 {s[SPLIT_SHA_KEY[label]][:16]}..." for label, _, _ in parts]
        rows += [f"clean rotation validity ({main_label}) {a_main['rotation_validity']}"]
        rows += [f"{n}: noisy sha256 {v['settings']['observation_noise']['noisy_training_state_sha256'][:16]}..., rotation validity {v['settings']['observation_noise']['rotation_validity']}" for n, v in variants.items() if n != "clean"]
        rows += ["", "rejected flight attempts:"] + [f"  {label} {a['seed']}/{a['index']}: {a['attempt']} rejections" for label, _, aud in parts for a in aud if a["attempt"] > 0]
        ax.text(0.01, 0.99, "\n".join(rows), va="top", family="monospace", fontsize=8); pdf.savefig(fig); plt.close(fig)
    log(f"wrote {path.name}")


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=THIS_DIR / "config.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "datasets",
                        help="parent folder; the dataset goes to <output-root>/QUADROTOR-DATASET-<name>/")
    parser.add_argument("--force", action="store_true", help="overwrite existing files of the same dataset")
    parser.add_argument("--no-save", action="store_true", help="generate and log, write nothing")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate(cfg)
    name = dataset_name(cfg)
    out_dir = (args.output_root / f"QUADROTOR-DATASET-{cfg['name']}").resolve()
    if not args.no_save and not args.force and list(out_dir.glob(f"{name}_*")):
        raise FileExistsError(f"{out_dir} already holds {name}_* files; pass --force to overwrite or change `name` in the config")
    log(f"dataset {name} -> {out_dir}  (trajectory_set {cfg['trajectory_set']})")
    tic = time.perf_counter()
    variants = build(cfg)
    log(f"generation finished in {(time.perf_counter() - tic) / 60:.1f} min; dataset {name}")
    if args.no_save:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, payload in variants.items():
        save(out_dir / f"{name}_{variant}.pkl", payload, args.force)
    (out_dir / f"{name}_audits.json").write_text(json.dumps({k: v for k, v in variants["clean"]["settings"].items() if k != "config"}, indent=1, default=float))
    (out_dir / f"{name}_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if cfg["output"]["write_pdf"]:
        write_pdf(variants, cfg, out_dir / f"{name}_dataset_analysis.pdf")
    (out_dir / f"{name}_generation.log").write_text("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    main()

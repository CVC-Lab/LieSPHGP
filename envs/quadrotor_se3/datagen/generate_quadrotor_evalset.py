"""QUADROTOR-DATASET-EVALSET generator: the unseen-shape evaluation set for models trained on QUADROTOR-DATASET-HARD.

One split only ("eval", stored under the key ``heldout_trajectories`` so the report loaders' ``@heldout`` selector
works unchanged), always clean (no observation-noise variants). Same plant, controller, envelope and flight length as
QUADROTOR-DATASET-HARD; the manoeuvre library shares NO segment name with the HARD training library, which is read from
the HARD config on disk and asserted at build time.

Library (all sequences of 3-5 random segments over 10 s, like the training flights):
  dash            straight run at a sustained random speed 0.5-3.2 m/s (steady thrust/drag balance per speed)
  thrust_pulse    collective thrust 25-45 % of weight for 0.35-0.5 s, then the compensating boost (thrust/gravity split)
  disturbed_hover hover hold at the anchor for 1-2 s (gusts are disabled, so this is a plain hover: T = m g)
  slalom          straight run with a lateral sine
  helix           circle that climbs or descends at a steady rate
  chirp           lateral + yaw sinusoid whose frequency sweeps 0.2 -> 1.5 Hz
  bounce          square wave in height (every edge a thrust transient)
  tumble          two torque kicks about different axes, the second mid-recovery

Procedure (unchanged from the HARD generator, envs/quadrotor_se3/datagen/generate_quadrotor_hard_v2.py lineage):
  0   Gym-PyBullet-Drones CF2P, Physics.PYB at 1000 Hz, contact-free, built-in nonlinear damping c = 0.5
  0b  wrench input recorded: u = [T, tau_x, tau_y, tau_z] from the applied (clipped) rpm plus any external kick torque
  1-3 10 s flights sampled at 100 Hz, random initial state, DSL PID (gain scale 0.5) tracking the segment targets
  5-6 gates (tilt, altitude, envelope, saturation), seeds, single split
  8   audits stored in settings
  9   pickle + audits.json + generation.log + config copy, 10 analysis PDF

Usage:  python generate_quadrotor_evalset.py --config evalset_config.yaml [--force] [--no-save]
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
PROJECT_ROOT = THIS_DIR.parents[2]  # envs/quadrotor_se3/datagen -> project root
for path in (PROJECT_ROOT, PROJECT_ROOT / "third_party/gym-pybullet-drones"):
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
LOG_LINES: list[str] = []


def log(message: str) -> None:
    LOG_LINES.append(message)
    print(message, flush=True)


# --------------------------------------------------------------------------- Step 0: environment
def make_env(cfg: dict, xyz: np.ndarray, rpy: np.ndarray) -> CtrlAviary:
    e = cfg["environment"]
    env = CtrlAviary(
        drone_model=DroneModel[e["drone_model"]], num_drones=1,
        initial_xyzs=xyz.reshape(1, 3), initial_rpys=rpy.reshape(1, 3),
        physics=Physics[e["physics"]], pyb_freq=int(e["physics_hz"]), ctrl_freq=int(e["physics_hz"]),
        gui=False, record=False, obstacles=False, user_debug_gui=False,
    )
    return env


def configure_plant(env: CtrlAviary, cfg: dict) -> dict[str, Any]:
    e = cfg["environment"]
    audit = {"contact_free": bool(e["contact_free"])}
    if e["contact_free"]:
        audit["no_ground"] = remove_ground_plane(env)
        audit["dynamics"] = configure_contact_free_dynamics(env)
    c = float(e["builtin_damping_coefficient"])
    law = str(e.get("damping_law", "nonlinear"))
    if law == "nonlinear":       # PyBullet built-in: a = -c(1+|v|)v, omega_dot = -c(1+|omega|)omega
        pb.changeDynamics(int(env.DRONE_IDS[0]), -1, linearDamping=c, angularDamping=c, physicsClientId=int(env.CLIENT))
    elif law == "linear":        # built-in off; external force -c m v and torque -c J omega applied every physics step
        pb.changeDynamics(int(env.DRONE_IDS[0]), -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=int(env.CLIENT))
    else:
        raise ValueError(f"unknown damping_law {law!r}")
    audit["builtin_damping_coefficient"] = c if law == "nonlinear" else 0.0
    audit["damping_law"] = law
    audit["custom_external_linear_damping"] = law == "linear"
    return audit


def apply_linear_damping(env: CtrlAviary, state: np.ndarray, c: float, mass: float, inertia: np.ndarray) -> None:
    """External linear damping for one physics step: F_b = -c m R^T v_w, tau_b = -c J R^T omega_w (link frame)."""
    rotation = np.asarray(pb.getMatrixFromQuaternion(state[3:7]), dtype=np.float64).reshape(3, 3)
    force_body = -c * mass * (rotation.T @ np.asarray(state[10:13], dtype=np.float64))
    torque_body = -c * (inertia @ (rotation.T @ np.asarray(state[13:16], dtype=np.float64)))
    pb.applyExternalForce(int(env.DRONE_IDS[0]), -1, force_body.tolist(), [0.0, 0.0, 0.0], pb.LINK_FRAME, physicsClientId=int(env.CLIENT))
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
        # ---- V6 training segments ----
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
        # ---- V6 heldout segments ----
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
            # several axes at once. Every training kick is a single impulse.
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


# --------------------------------------------------------------------------- Steps 0b-5: one flight
def generate_flight(cfg: dict, seed: int, index: int, attempt: int, library: dict) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(int(cfg["splits"]["flight_seed_base"]) + 1000 * seed + index + 100_000 * attempt)
    init, segments = random_initial_state(rng, cfg), plan_manoeuvres(rng, cfg, library)
    e, act, gust_cfg, box = cfg["environment"], cfg["actuator"], cfg["gusts"], cfg["envelope"]["position_box"]
    physics_hz, sample_hz = int(e["physics_hz"]), int(e["sample_hz"])
    sub, dt = physics_hz // sample_hz, 1.0 / physics_hz
    n_samples = int(round(cfg["flight"]["duration_seconds"] * sample_hz)) + 1
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

        pb.resetBaseVelocity(int(env.DRONE_IDS[0]), linearVelocity=init["velocity_world"], angularVelocity=init["omega_world"], physicsClientId=int(env.CLIENT))
        env._updateAndStoreKinematicInformation()
        state = env._getDroneStateVector(0)

        trajectory = np.empty((n_samples, STATE_DIM + CONTROL_DIM)); commands = np.empty((n_samples, 4)); commanded = np.empty((n_samples, 4)); gusts = np.empty((n_samples, 3))
        trajectory[0], commands[0], commanded[0], gusts[0] = pack_sample(state, np.zeros(4)), 0.0, 0.0, 0.0
        rpm_applied, rpm_cmd, gust = np.full(4, float(env.HOVER_RPM)), np.full(4, float(env.HOVER_RPM)), np.zeros(3)
        sigma_w, tau_w = float(gust_cfg["sigma_fraction_of_weight"]) * mass * grav, float(gust_cfg["time_constant_seconds"])
        seg_i, seg_start_state, saturated, kicks = 0, state.copy(), 0, []
        kick_torque = np.zeros(3)                                                            # external torque active in the current step
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
            if gust_cfg["enabled"]:                                                       # Step 4: OU gust
                mult = seg.get("gust_multiplier", 1.0)
                gust = gust * np.exp(-dt / tau_w) + mult * sigma_w * np.sqrt(1 - np.exp(-2 * dt / tau_w)) * rng.normal(size=3)
                pb.applyExternalForce(int(env.DRONE_IDS[0]), -1, gust.tolist(), [0, 0, 0], pb.LINK_FRAME, physicsClientId=int(env.CLIENT))
            kick_torque = np.zeros(3)
            kick_axis, local = None, t - seg["start"]
            if seg["name"] in ("aggressive_recovery", "yaw_kick", "coast_yaw") and local < seg["kick_seconds"]:   # torque kick (a shove)
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
            if cfg["environment"].get("damping_law", "nonlinear") == "linear":
                apply_linear_damping(env, state, float(cfg["environment"]["builtin_damping_coefficient"]), mass, inertia)
            observation, *_ = env.step((rpm_applied * np.sqrt(gain)).reshape(1, 4))     # PyBullet applies k_f (rpm sqrt(gain))^2
            state = observation[0]
            saturated += int(np.any(rpm_cmd >= rpm_cap - 1.0))                             # a motor at the ceiling
            min_alt, max_tilt = min(min_alt, float(state[2])), max(max_tilt, tilt_deg(state))
            if step % sub == 0:                                                           # record at sample_hz
                k = step // sub
                wrench = mixer @ (kf * gain * rpm_applied**2)
                wrench[1:] += kick_torque                                                     # recorded torque = motors + external kick
                trajectory[k], commands[k], commanded[k], gusts[k] = pack_sample(state, wrench), (rpm_applied / rpm_max) ** 2, (rpm_cmd / rpm_max) ** 2, gust
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
    return trajectory, {"commands": commands, "commanded": commanded, "gusts": gusts}, audit


def generate_split(cfg: dict, seeds: list[int], label: str, library: dict) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    flights, extras, audits = [], {"commands": [], "commanded": [], "gusts": []}, []
    for seed in seeds:
        for index in range(int(cfg["splits"]["flights_per_seed"])):
            for attempt in range(int(cfg["gates"]["max_attempts"])):
                tic = time.perf_counter()
                trajectory, extra, audit = generate_flight(cfg, seed, index, attempt, library)
                log(f"[{label}] seed={seed} flight={index} attempt={attempt} {time.perf_counter() - tic:.1f}s "
                    f"segments={[s['name'] for s in audit['segments']]} saturation={audit['saturation_fraction']:.3f} violation={audit['violation']}")
                if audit["violation"] is None:
                    break
            else:
                raise RuntimeError(f"{label} seed={seed} flight={index}: no feasible flight in {cfg['gates']['max_attempts']} attempts")
            flights.append(trajectory); audits.append(audit)
            for key in extras: extras[key].append(extra[key])
    return np.stack(flights), {k: np.stack(v) for k, v in extras.items()}, audits


# --------------------------------------------------------------------------- Step 7: noise variants
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
    """v1 scheme: additive Gaussian on x, v_b, omega_b; R * Exp(eta) on rotations; controls untouched."""
    rng, noisy = np.random.default_rng(seed), flights.copy()
    noisy[..., EUCLIDEAN] += rng.normal(0, sigma, size=flights[..., EUCLIDEAN].shape)
    noisy[..., 3:12] = perturb_rotations(flights, sigma, rng)
    return noisy


def sensor_noise(flights: np.ndarray, cfg: dict, seed: int, h: float) -> np.ndarray:
    """Motion capture + IMU: mm pose noise, velocity by differencing noisy positions, gyro white noise + bias walk."""
    s, rng, noisy = cfg["noise"]["sensor"], np.random.default_rng(seed), flights.copy()
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
        horizon[str(T)] = {"median_position_change_m": float(np.median(d)), "ratio_to_sigma": {str(s): float(np.median(d) / s) for s in cfg["noise"]["absolute_levels"]}}
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


# --------------------------------------------------------------------------- Step 9: files
def dataset_tag(cfg: dict, base_cfg: dict) -> str:
    """Output tag; changed values relative to the shipped defaults are appended so variants never overwrite."""
    diff = [f"{k}-{v}" for k, v in _flatten(cfg).items() if _flatten(base_cfg).get(k) != v and not k.startswith("output.")]
    tag = f"{cfg['output']['tag']}_{cfg['environment']['drone_model']}_{int(cfg['flight']['duration_seconds'])}s_h{1.0 / cfg['environment']['sample_hz']:g}".replace(".", "p")
    return tag + ("_" + "_".join(diff).replace("/", "").replace(" ", "")[:80] if diff else "")


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        out.update(_flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": str(v)})
    return out


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
    log(f"wrote {path.name}  heldout_trajectories={payload['heldout_trajectories'].shape}")


HARD_CONFIG = PROJECT_ROOT / "datasets/QUADROTOR-DATASET-HARD/HARD_CF2P_10s_h0p01_config_used.yaml"


def training_library_names(cfg: dict) -> list[str]:
    """Segment names of the training set this evalset must not share (read from the HARD config on disk)."""
    path = PROJECT_ROOT / cfg["manoeuvres"].get("training_config", str(HARD_CONFIG.relative_to(PROJECT_ROOT)))
    return list(yaml.safe_load(path.read_text())["manoeuvres"]["library"])


def build(cfg: dict, base_cfg: dict) -> dict[str, dict]:
    e, out = cfg["environment"], cfg["output"]
    h = 1.0 / e["sample_hz"]
    library = cfg["manoeuvres"]["library"]
    train_names = training_library_names(cfg)
    shared = sorted(set(library) & set(train_names))
    if shared:
        raise ValueError(f"the evalset library must share no segment with the HARD training library; shared: {shared}")
    flights, extra, flight_audits = generate_split(cfg, cfg["splits"]["eval_seeds"], "eval", library)
    t = np.arange(flights.shape[1]) * h
    tag = dataset_tag(cfg, base_cfg)
    plant = flight_audits[0]["plant"]
    settings = {
        "dataset_name": tag, "config": copy.deepcopy(cfg), "git_hash": git_hash(),
        "vehicle_parameters": {"mass": plant["mass"], "inertia": np.diag(plant["inertia_diagonal"]).tolist(),
                               "gravity_acceleration": plant["gravity_acceleration"], "arm": plant["arm"],
                               "kf": plant["kf"], "km": plant["km"], "maximum_rpm": plant["maximum_rpm"]},
        "linear_damping_coefficient": e["builtin_damping_coefficient"], "damping_law": str(e.get("damping_law", "nonlinear")),
        "angular_damping_coefficient": e["builtin_damping_coefficient"],
        "damping_equation": ("a=-c*(1+norm(v))*v and omega_dot damping=-c*(1+norm(omega))*omega" if e.get("damping_law", "nonlinear") == "nonlinear"
                             else "a=-c*v and tau=-c*J*omega"),
        "pybullet_builtin_damping": e.get("damping_law", "nonlinear") == "nonlinear",
        "custom_external_linear_damping": e.get("damping_law", "nonlinear") == "linear",
        "state_layout": "x_w(3), vec(R)(9), v_b(3), omega_b(3), wrench(4)", "control_layout": "wrench [T, tau_x, tau_y, tau_z]: motor wrench after rpm clipping plus the external kick torque of recovery segments",
        "input_mode": cfg["actuator"]["input_mode"], "true_control_map": "selection matrix S" if cfg["actuator"]["input_mode"] == "wrench" else "S @ M_mix @ kf @ rpm_max^2",
        "motor_command_layout": "(rpm_i / rpm_max)^2 applied; 'motor_commands_commanded' before the motor lag",
        "sample_dt": h, "physics_hz": e["physics_hz"], "flight_seconds": cfg["flight"]["duration_seconds"],
        "splits": {"eval": {"seeds": cfg["splits"]["eval_seeds"], "library": list(library), "key": "heldout_trajectories",
                            "note": "the only split: unseen-shape evaluation flights for models trained on QUADROTOR-DATASET-HARD; always clean"},
                   "training_library_checked_against": train_names, "segment_names_shared_with_training": shared},
        "heldout_flight_audits": flight_audits, "audits_heldout": audits(flights, cfg, h),
        "clean_heldout_state_sha256": array_sha256(flights[..., :STATE_DIM]),
        "observation_noise": {"enabled": False, "heldout_split_clean": True},
    }
    clean = {"t": t[:out["window_points"]],
             "heldout_trajectories": flights, "heldout_x": sliding_windows(flights, out["window_points"], out["window_stride"]),
             "heldout_motor_commands": extra["commands"], "heldout_motor_commands_commanded": extra["commanded"], "heldout_gust_force": extra["gusts"],
             "settings": settings}
    return {"clean": clean}


def _noisy_variant(clean: dict, noisy_train: np.ndarray, noisy_test: np.ndarray, record: dict, cfg: dict) -> dict:
    out = cfg["output"]
    s = copy.deepcopy(clean["settings"])
    s["observation_noise"] = {"enabled": True, "test_split_clean": True, "heldout_split_clean": True, "noisy_test_split_saved": True, "controls_unchanged_by_noise": True,
                              "noise_added_before_windowing": True, "train_noise_seed": cfg["noise"]["train_noise_seed"], "test_noise_seed": cfg["noise"]["test_noise_seed"],
                              "noisy_training_state_sha256": array_sha256(noisy_train[..., :STATE_DIM]), "rotation_validity": rotation_validity(noisy_train),
                              "train_noise_std_per_channel": (noisy_train[..., :STATE_DIM] - clean["train_trajectories"][..., :STATE_DIM]).std((0, 1)).tolist(), **record}
    s["dataset_name"] = clean["settings"]["dataset_name"] + "-" + record["scheme"]
    return {**clean, "x": sliding_windows(noisy_train, out["window_points"], out["window_stride"]), "train_trajectories": noisy_train,
            "test_x_noisy": sliding_windows(noisy_test, out["window_points"], out["window_stride"]), "test_trajectories_noisy": noisy_test, "settings": s}


# --------------------------------------------------------------------------- Step 10: PDF
def write_pdf(variants: dict[str, dict], cfg: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    clean, s = variants["clean"], variants["clean"]["settings"]
    flights, h = clean["heldout_trajectories"], s["sample_dt"]
    t = np.arange(flights.shape[1]) * h
    a, box = s["audits_heldout"], cfg["envelope"]["position_box"]
    names = list(cfg["manoeuvres"]["library"])
    colours = {n: c for n, c in zip(names, plt.cm.tab20.colors)}
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
        rej = sum(x["attempt"] for x in s["heldout_flight_audits"])
        lines = [f"Dataset {s['dataset_name']}",
                 f"eval flights {flights.shape[0]} x {s['flight_seconds']} s (key heldout_trajectories, clean), h = {h} s, git {s['git_hash'][:10]}",
                 f"evalset library : {', '.join(names)}",
                 f"HARD training library (checked, none shared): {', '.join(s['splits']['training_library_checked_against'])}",
                 f"input mode {s['input_mode']}; true control map: {s['true_control_map']}",
                 f"rejected attempts {rej}; mean saturation {np.mean([x['saturation_fraction'] for x in s['heldout_flight_audits']]):.4f}",
                 "", "config:"] + yaml.safe_dump(cfg, sort_keys=False).splitlines()
        ax.text(0.01, 0.99, "\n".join(lines[:70]), va="top", family="monospace", fontsize=6.5); pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_subplot(projection="3d")
        for f in flights: ax.plot(f[:, 0], f[:, 1], f[:, 2], lw=0.6, alpha=0.7)
        ax.set(xlim=box["x"], ylim=box["y"], zlim=box["z"], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="EVALSET flights over the envelope box"); pdf.savefig(fig); plt.close(fig)
        for i in range(min(int(cfg["output"]["pdf_example_flights"]), flights.shape[0])):
            f, aud = flights[i], s["heldout_flight_audits"][i]
            rpy = Rotation.from_matrix(f[:, 3:12].reshape(-1, 3, 3)).as_euler("xyz")
            fig, axes = plt.subplots(5, 1, figsize=(11, 8.5), sharex=True)
            for ax, y, lab in zip(axes, (f[:, :3], rpy, f[:, 12:15], f[:, 15:18], f[:, 18:22]), ("x (m)", "roll pitch yaw (rad)", "v_b (m/s)", "omega_b (rad/s)", "wrench (N, N m)")):
                ax.plot(t, y, lw=0.8); ax.set_ylabel(lab); ax.grid(alpha=0.3)
                for seg in aud["segments"]: ax.axvspan(seg["start"], seg["start"] + seg["duration"], color=colours[seg["name"]], alpha=0.08)
            axes[0].set_title(f"EVAL flight {i}: " + " > ".join(seg["name"] for seg in aud["segments"])); axes[-1].set_xlabel("t (s)"); pdf.savefig(fig); plt.close(fig)
        flat = flights.reshape(-1, 22); fig, axes = plt.subplots(2, 3, figsize=(11, 8.5)); axes = axes.ravel()
        for ax, (i, k) in zip(axes[:3], enumerate("xyz")): ax.hist(flat[:, i], 40); ax.axvline(box[k][0], c="r"); ax.axvline(box[k][1], c="r"); ax.set_title(f"{k} (m)")
        axes[3].hist(np.degrees(np.arccos(np.clip(flat[:, 11], -1, 1))), 40); axes[3].axvline(cfg["envelope"]["max_tilt_deg"], c="r"); axes[3].set_title("tilt (deg)")
        axes[4].hist(np.linalg.norm(flat[:, 12:15], axis=1), 40); axes[4].axvline(cfg["envelope"]["max_speed"], c="r"); axes[4].set_title("speed (m/s)")
        axes[5].hist(np.linalg.norm(flat[:, 15:18], axis=1), 40); axes[5].axvline(cfg["envelope"]["max_angular_rate"], c="r"); axes[5].set_title("angular rate (rad/s)")
        fig.suptitle("coverage of the eval flights (red: envelope limits)"); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), constrained_layout=True); axes = axes.ravel()
        for j, lab in enumerate(("T (N)", "tau_x (N m)", "tau_y (N m)", "tau_z (N m)")): axes[j].hist(flat[:, 18 + j], 40); axes[j].set_title(lab)
        im = axes[4].imshow(np.asarray(a["input_correlation"]), vmin=-1, vmax=1, cmap="coolwarm"); axes[4].set_title("input correlation"); fig.colorbar(im, ax=axes[4])
        freqs, psd = np.asarray(a["input_psd_frequencies_hz"]), np.asarray(a["input_psd"])
        for j, lab in enumerate(("T", "tau_x", "tau_y", "tau_z")): axes[5].loglog(freqs[1:], psd[1:, j] / psd[1:, j].max(), lw=0.8, label=lab)
        axes[5].set(title="input power spectrum (normalised)", xlabel="Hz"); axes[5].legend(); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
        rows = [f"clean eval state sha256 {s['clean_heldout_state_sha256'][:16]}...", f"rotation validity {a['rotation_validity']}", "",
                "segment counts:"] + [f"  {n}: {sum(seg['name'] == n for x in s['heldout_flight_audits'] for seg in x['segments'])}" for n in names]
        rows += ["", "rejected flight attempts:"] + [f"  {x['seed']}/{x['index']}: {x['attempt']} rejections" for x in s["heldout_flight_audits"] if x["attempt"] > 0]
        ax.text(0.01, 0.99, "\n".join(rows), va="top", family="monospace", fontsize=8); pdf.savefig(fig); plt.close(fig)
    log(f"wrote {path.name}")


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=THIS_DIR / "configs" / "evalset_config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    base_cfg = yaml.safe_load((THIS_DIR / "configs" / "evalset_config.yaml").read_text())
    out_dir = (PROJECT_ROOT / cfg["output"]["directory"]).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    variants = build(cfg, base_cfg)
    tag = variants["clean"]["settings"]["dataset_name"]
    log(f"generation finished in {(time.perf_counter() - tic) / 60:.1f} min; tag {tag}")
    if args.no_save:
        return
    for name, payload in variants.items():
        save(out_dir / f"{tag}_{name}.pkl", payload, args.force)
    (out_dir / f"{tag}_audits.json").write_text(json.dumps({k: v for k, v in variants["clean"]["settings"].items() if k != "config"}, indent=1, default=float))
    (out_dir / f"{tag}_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if cfg["output"]["write_pdf"]:
        write_pdf(variants, cfg, out_dir / f"{tag}_dataset_analysis.pdf")
    (out_dir / f"{tag}_generation.log").write_text("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    main()

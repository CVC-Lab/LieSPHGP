"""Closed-loop PyBullet control with the learned JAX port-Hamiltonian model.

The plant is gym-pybullet-drones ``CtrlAviary`` (CF2P, ``Physics.PYB``, 240 Hz)
with the ground plane removed and the contact-free nonlinear-damping
configuration of the training simulator.  The controller is the energy-based
SE(3) law of the reference report, evaluated with the six learned operators of
the JAX model (gradient of the learned potential by ``jax.grad``).  The
learned wrench is allocated to rotor speeds through the plant mixer, clipped
at the physical maximum RPM, and every step is recorded.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from .report_evaluation import DAMPING_COEFFICIENT, DT, PHYSICS_HZ, sha256

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PYBULLET_DRONES_DIR = PROJECT_ROOT / "envs/SE3_quadrotor/gym-pybullet-drones"
if str(PYBULLET_DRONES_DIR) not in sys.path:
    sys.path.insert(0, str(PYBULLET_DRONES_DIR))

import pybullet as pb  # noqa: E402
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402


# Gains of the released energy-based controller (ControllerParams): K_p=2*[5,5,25],
# K_v=1.2*[2.5,2.5,2.5], K_R=[250,250,250], K_w=[20,20,20], maximum tilt 40 degrees.
GAINS = {
    "Kp": np.array([10.0, 10.0, 50.0]),
    "Kv": np.array([3.0, 3.0, 3.0]),
    "KR": np.array([250.0, 250.0, 250.0]),
    "Komega": np.array([20.0, 20.0, 20.0]),
}
MAXIMUM_TILT = 40.0 * np.pi / 180.0
INITIAL_XYZ = np.array([0.0, 0.0, 1.0])
INITIAL_RPY = np.array([0.0, 0.0, 0.0])      # start aligned with the shapes: a non-zero initial heading made
                                            # the yaw row score the slew off that heading, not yaw tracking


class DesiredState:
    def __init__(self, pos, vel, acc, yaw=0.0, yawdot=0.0):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)
        self.acc = np.asarray(acc, dtype=np.float64)
        self.yaw = float(yaw)
        self.yawdot = float(yawdot)


def diamond(t: float) -> DesiredState:
    """The released diamond reference: 15 s of piecewise-linear segments, then hold."""
    T = 15.0
    r2 = np.sqrt(2.0)
    if t < 0:
        pos, vel = np.zeros(3), np.zeros(3)
    elif t < T / 4:
        pos = np.array([0, r2, r2]) * t / (T / 4)
        vel = np.array([0, r2, r2]) / (T / 4)
    elif t < T / 2:
        pos = np.array([0, r2, r2]) * (2 - 4 * t / T) + np.array([0, 0, 2 * r2]) * (4 * t / T - 1)
        vel = np.array([0, r2, r2]) * (-4 / T) + np.array([0, 0, 2 * r2]) * (4 / T)
    elif t < 3 * T / 4:
        pos = np.array([0, 0, 2 * r2]) * (3 - 4 * t / T) + np.array([0, -r2, r2]) * (4 * t / T - 2)
        vel = np.array([0, 0, 2 * r2]) * (-4 / T) + np.array([0, -r2, r2]) * (4 / T)
    elif t < T:
        pos = np.array([0, -r2, r2]) * (4 - 4 * t / T) + np.array([1, 0, 0.5]) * (4 * t / T - 3)
        vel = np.array([0, -r2, r2]) * (-4 / T) + np.array([1, 0, 0.5]) * (4 / T)
    else:
        pos, vel = np.array([1.0, 0.0, 0.5]), np.zeros(3)
    return DesiredState(pos, vel, np.zeros(3), 0.0, 0.0)


# ---------------------------------------------------------------------------
# Additional closed-loop references. None of these shapes occur in the HARD-V5 training/test library
# (waypoint_hop, figure_eight, circle, vertical_step, yaw_turn, kicks, coasts), and the diamond itself
# is a chain of straight hops, so it is not one of the five. Each is analytic: position, velocity and
# acceleration are exact; every shape is shifted so it starts at the vehicle's start point; it is active
# for ACTIVE_SECONDS and then holds its final point. Yaw is zero throughout, as for the diamond.
# ---------------------------------------------------------------------------
ACTIVE_SECONDS = 15.0


def _shaped(pos_vel_acc, t: float, active: float = ACTIVE_SECONDS) -> DesiredState:
    """Shift so pos(0) = 0, hold after ``active`` seconds, clamp t < 0 to the start.

    ``pos_vel_acc(t)`` returns (pos, vel, acc) or (pos, vel, acc, yaw, yawdot); yaw is held with the position.
    """
    origin = pos_vel_acc(0.0)[0]
    def unpack(values):
        pos, vel, acc = values[:3]
        yaw, yawdot = (values[3], values[4]) if len(values) > 3 else (0.0, 0.0)
        return pos, vel, acc, yaw, yawdot
    if t <= 0.0:
        _, _, _, yaw0, _ = unpack(pos_vel_acc(0.0))
        return DesiredState(np.zeros(3), np.zeros(3), np.zeros(3), yaw0, 0.0)
    if t >= active:
        pos, _, _, yaw, _ = unpack(pos_vel_acc(active))
        return DesiredState(pos - origin, np.zeros(3), np.zeros(3), yaw, 0.0)
    pos, vel, acc, yaw, yawdot = unpack(pos_vel_acc(t))
    return DesiredState(pos - origin, vel, acc, yaw, yawdot)


def lissajous_3d(t: float) -> DesiredState:
    """3-D Lissajous with incommensurate axis frequencies: non-repeating, every axis excited at once."""
    def f(s):
        w = 2 * np.pi / 10.0
        A, k, phi = np.array([1.2, 1.2, 0.6]), np.array([1.0, 1.5, 0.75]), np.array([0.0, np.pi / 3, 0.0])
        arg = k * w * s + phi
        return A * np.sin(arg), A * k * w * np.cos(arg), -A * (k * w) ** 2 * np.sin(arg)
    return _shaped(f, t)


def trefoil(t: float) -> DesiredState:
    """Trefoil knot: sustained curvature that reverses in all three axes; a knot, not a circle."""
    def f(s):
        a, th_dot = 0.4, 2 * np.pi / ACTIVE_SECONDS
        th = th_dot * s
        pos = np.array([a * (np.sin(th) + 2 * np.sin(2 * th)), a * (np.cos(th) - 2 * np.cos(2 * th)), 0.5 * np.sin(3 * th)])
        vel = np.array([a * (np.cos(th) + 4 * np.cos(2 * th)), a * (-np.sin(th) + 4 * np.sin(2 * th)), 1.5 * np.cos(3 * th)]) * th_dot
        acc = np.array([-a * (np.sin(th) + 8 * np.sin(2 * th)), a * (-np.cos(th) + 8 * np.cos(2 * th)), -4.5 * np.sin(3 * th)]) * th_dot**2
        return pos, vel, acc
    return _shaped(f, t)


def _polar(r, r_dot, r_ddot, phi, phi_dot):
    c, s = np.cos(phi), np.sin(phi)
    pos = np.array([r * c, r * s])
    vel = np.array([r_dot * c - r * phi_dot * s, r_dot * s + r * phi_dot * c])
    acc = np.array([r_ddot * c - 2 * r_dot * phi_dot * s - r * phi_dot**2 * c,
                    r_ddot * s + 2 * r_dot * phi_dot * c - r * phi_dot**2 * s])
    return pos, vel, acc


def spiral_out_in(t: float) -> DesiredState:
    """Planar spiral whose radius grows to 1.3 m and returns to zero at a fixed turn rate: speed ramps 0 -> ~1.6 m/s -> 0."""
    def f(s):
        R, T, Om = 1.3, ACTIVE_SECONDS, 2 * np.pi / 5.0
        r, r_dot, r_ddot = R * np.sin(np.pi * s / T), R * (np.pi / T) * np.cos(np.pi * s / T), -R * (np.pi / T) ** 2 * np.sin(np.pi * s / T)
        p, v, a = _polar(r, r_dot, r_ddot, Om * s, Om)
        return np.array([p[0], p[1], 0.0]), np.array([v[0], v[1], 0.0]), np.array([a[0], a[1], 0.0])
    return _shaped(f, t)


def chirp_line(t: float) -> DesiredState:
    """Back-and-forth along x with the frequency swept 0.1 -> 0.6 Hz: a bandwidth test, inertia against damping."""
    def f(s):
        A, f0, f1, T = 0.4, 0.1, 0.6, ACTIVE_SECONDS
        c = (f1 - f0) / (2 * T)
        psi, psi_dot, psi_ddot = 2 * np.pi * (f0 * s + c * s**2), 2 * np.pi * (f0 + 2 * c * s), 4 * np.pi * c
        return (np.array([A * np.sin(psi), 0.0, 0.0]), np.array([A * np.cos(psi) * psi_dot, 0.0, 0.0]),
                np.array([-A * np.sin(psi) * psi_dot**2 + A * np.cos(psi) * psi_ddot, 0.0, 0.0]))
    return _shaped(f, t)


def rose_3(t: float) -> DesiredState:
    """Three-petal rose r = R cos(3 phi) with a gentle altitude bob: sharp petal reversals with no straight segment."""
    def f(s):
        R, Om, wz = 1.2, 2 * np.pi / ACTIVE_SECONDS, 2 * np.pi / 7.5
        phi = Om * s
        r, r_dot, r_ddot = R * np.cos(3 * phi), -3 * R * Om * np.sin(3 * phi), -9 * R * Om**2 * np.cos(3 * phi)
        p, v, a = _polar(r, r_dot, r_ddot, phi, Om)
        return (np.array([p[0], p[1], 0.4 * np.sin(wz * s)]), np.array([v[0], v[1], 0.4 * wz * np.cos(wz * s)]),
                np.array([a[0], a[1], -0.4 * wz**2 * np.sin(wz * s)]))
    return _shaped(f, t)


# ---------------------------------------------------------------------------
# Difficulty sweeps. Each family ramps ONE parameter along a mechanism the baselines identify worst, and every level
# is reported, so the levels where a model fails come with the levels where it does not.
#   spiral_k<k>       spiral_out_in with the turn rate scaled by k: peak speed 1.63*k m/s. Sustained speed puts the
#                     translational-damping error into the controller's feedforward (+Dv v_b) in proportion to v.
#   lissajous_yaw<A>  lissajous_3d plus a yaw reference A*sin(2*pi*t/3): yaw rates up to 2*pi*A/3 rad/s. Rotational
#                     damping is the worst-identified product of every baseline; the yaw loop feeds +Dw omega_b forward.
#   stop_v<v0>        cosine ramp to v0 over 1 s, cruise to ~2.5 m of travel, then an instantaneous hold command:
#                     a stop-at-speed transient; the braking demand grows with v0 and saturates the motors first for
#                     the model whose force scale is most wrong.
# ---------------------------------------------------------------------------
def make_spiral(k: float):
    def reference(t: float) -> DesiredState:
        def f(s):
            R, T, Om = 1.3, ACTIVE_SECONDS, k * 2 * np.pi / 5.0
            r, r_dot, r_ddot = R * np.sin(np.pi * s / T), R * (np.pi / T) * np.cos(np.pi * s / T), -R * (np.pi / T) ** 2 * np.sin(np.pi * s / T)
            p, v, a = _polar(r, r_dot, r_ddot, Om * s, Om)
            return np.array([p[0], p[1], 0.0]), np.array([v[0], v[1], 0.0]), np.array([a[0], a[1], 0.0])
        return _shaped(f, t)
    reference.__name__ = f"spiral_k{k:g}"
    return reference


def make_lissajous_yaw(amplitude: float, period: float = 3.0):
    def reference(t: float) -> DesiredState:
        def f(s):
            w = 2 * np.pi / 10.0
            A, kk, phi = np.array([1.2, 1.2, 0.6]), np.array([1.0, 1.5, 0.75]), np.array([0.0, np.pi / 3, 0.0])
            arg = kk * w * s + phi
            wy = 2 * np.pi / period
            return (A * np.sin(arg), A * kk * w * np.cos(arg), -A * (kk * w) ** 2 * np.sin(arg),
                    amplitude * np.sin(wy * s), amplitude * wy * np.cos(wy * s))
        return _shaped(f, t)
    reference.__name__ = f"lissajous_yaw{amplitude:g}"
    return reference


def make_stop(v0: float, travel: float = 2.5, ramp: float = 1.0):
    """Straight run along +x: cosine speed ramp 0 -> v0 over ``ramp`` s, cruise, instantaneous hold at ~``travel`` m."""
    t_run = ramp + max(travel - 0.5 * v0 * ramp, 0.0) / v0            # ramp covers v0*ramp/2 m
    def reference(t: float) -> DesiredState:
        def f(s):
            if s < ramp:
                speed, accel = 0.5 * v0 * (1 - np.cos(np.pi * s / ramp)), 0.5 * v0 * (np.pi / ramp) * np.sin(np.pi * s / ramp)
                dist = 0.5 * v0 * (s - (ramp / np.pi) * np.sin(np.pi * s / ramp))
            else:
                speed, accel, dist = v0, 0.0, 0.5 * v0 * ramp + v0 * (s - ramp)
            return np.array([dist, 0.0, 0.0]), np.array([speed, 0.0, 0.0]), np.array([accel, 0.0, 0.0])
        return _shaped(f, t, active=t_run)
    reference.__name__ = f"stop_v{v0:g}"
    return reference


REFERENCES = {
    "diamond": diamond,
    "lissajous_3d": lissajous_3d,
    "trefoil": trefoil,
    "spiral_out_in": spiral_out_in,
    "chirp_line": chirp_line,
    "rose_3": rose_3,
}
for _k in (1.5, 2.0, 2.5):
    REFERENCES[f"spiral_k{_k:g}".replace(".", "p")] = make_spiral(_k)
for _a in (0.5, 1.0, 1.5):
    REFERENCES[f"lissajous_yaw{_a:g}".replace(".", "p")] = make_lissajous_yaw(_a)
for _v in (1.0, 1.5, 2.0, 2.5):
    REFERENCES[f"stop_v{_v:g}".replace(".", "p")] = make_stop(_v)


# ---------------------------------------------------------------------------
# Recorded flights as closed-loop references: the open-loop evaluation shapes flown under the controller
# ---------------------------------------------------------------------------
def make_recorded_reference(states: np.ndarray, step: float, smoothing_seconds: float = 0.05, relative_yaw: bool = True):
    """A reference built from a recorded flight in the (time, 22) layout: position, velocity, yaw as flown.

    The feedforward acceleration and yaw rate come from finite differences of the recorded velocity and
    heading, lightly smoothed (a boxcar of ``smoothing_seconds``) so the controller is not fed sensor-rate
    jitter. The shape is active for the flight's duration and then holds its final point, exactly like the
    analytic references; the position is re-anchored to the vehicle's own start by run_controller.
    """
    from scipy.ndimage import uniform_filter1d
    states = np.asarray(states, dtype=np.float64)
    times = np.arange(states.shape[0]) * step
    rotation = states[:, 3:12].reshape(-1, 3, 3)
    position = states[:, :3]
    velocity = np.einsum("nij,nj->ni", rotation, states[:, 12:15])          # body velocity -> world
    yaw = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    if relative_yaw:
        yaw = yaw - yaw[0]          # heading relative to the flight's own start: the vehicle starts at yaw 0
    # relative_yaw=False: the absolute recorded heading, for a plant started at the flight's recorded x_0
    width = max(int(round(smoothing_seconds / step)), 1)
    acceleration = uniform_filter1d(np.gradient(velocity, step, axis=0), width, axis=0, mode="nearest")
    yaw_rate = uniform_filter1d(np.gradient(yaw, step), width, mode="nearest")
    active = float(times[-1])

    def f(s: float):
        s = float(np.clip(s, times[0], times[-1]))
        return (np.array([np.interp(s, times, position[:, i]) for i in range(3)]),
                np.array([np.interp(s, times, velocity[:, i]) for i in range(3)]),
                np.array([np.interp(s, times, acceleration[:, i]) for i in range(3)]),
                float(np.interp(s, times, yaw)), float(np.interp(s, times, yaw_rate)))

    def reference(t: float) -> DesiredState:
        return _shaped(f, t, active)

    return reference


def register_recorded_references(spec: str, flights: int = 10) -> list[str]:
    """Register every flight of an evaluation set (``path[@split]``) as ``<SET>-<split>-flight<k>``; returns the names."""
    from .open_loop import load_open_loop_set
    loaded = load_open_loop_set(spec, flights)
    family = Path(str(spec).partition("@")[0]).stem.split("_")[0]           # e.g. HARDV6
    split = loaded["source"].split(" split")[0]                              # e.g. heldout / test / common
    names = []
    for index, flight in enumerate(loaded["truth"]):
        name = f"{family}-{split}-flight{index:02d}"
        REFERENCES[name] = make_recorded_reference(flight, loaded["dt"])
        names.append(name)
    return names


def hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(matrix: np.ndarray) -> np.ndarray:
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


class JaxProvider:
    """Query the six learned operators of the posterior-mean JAX model."""

    def __init__(self, model):
        self.model = model

        def query(pose, velocity, omega):
            inverse_mass_v = model.inverse_mass_1(pose[:3])
            inverse_mass_w = model.inverse_mass_2(pose[3:12])
            potential, potential_gradient = jax.value_and_grad(model.potential)(pose)
            return (
                inverse_mass_v,
                inverse_mass_w,
                potential,
                potential_gradient,
                model.control_matrix(pose),
                model.dissipation_v(velocity, pose[:3]),
                model.dissipation_w(omega, pose[3:12]),
            )

        self._query = jax.jit(query)

    def query(self, position, rotation, velocity_body, omega_body) -> dict[str, Any]:
        pose = jnp.asarray(np.concatenate((position, rotation.reshape(9))), dtype=jnp.float64)
        outputs = self._query(pose, jnp.asarray(velocity_body, jnp.float64), jnp.asarray(omega_body, jnp.float64))
        inverse_mass_v, inverse_mass_w, potential, gradient, control_map, dissipation_v, dissipation_w = (
            np.asarray(value, dtype=np.float64) for value in outputs
        )
        return {
            "inverse_mass_v": inverse_mass_v,
            "inverse_mass_w": inverse_mass_w,
            "mass_v": np.linalg.inv(inverse_mass_v),
            "mass_w": np.linalg.inv(inverse_mass_w),
            "potential": float(potential),
            "potential_gradient": gradient,
            "control_map": control_map,
            "dissipation_v": dissipation_v,
            "dissipation_w": dissipation_w,
        }


SWAP_PRODUCTS = ("force_scale", "torque_gain", "gravity", "damping_v", "damping_w")


class HybridProvider:
    """Base model's operators with ONE gauge-invariant product taken from a donor model.

    The two models sit in different gauges (mu -> beta_v mu, M2^-1 -> beta_w M2^-1, ...), so a raw operator cannot be
    moved between them. The base keeps its own M1^-1 and M2^-1 (the gauge carriers) and the donor's product is
    re-expressed in the base gauge:
        force_scale  mu g_f          g_f^hyb  = (mu_B / mu_A) g_f,B                 (force rows of the control map)
        torque_gain  M2^-1 g_tau     g_tau^hyb = M2_A M2_B^-1 g_tau,B               (torque rows of the control map)
        gravity      mu grad V       grad V^hyb = (mu_B / mu_A) grad V_B             (and the potential value)
        damping_v    mu Dv           Dv^hyb   = (mu_B / mu_A) Dv_B
        damping_w    M2^-1 Dw        Dw^hyb   = M2_A M2_B^-1 Dw_B
    so that, e.g., mu_A Dv^hyb == mu_B Dv_B at every queried state. The controller's command depends on the five
    products only (the gain terms cancel M1 after allocation), so this is exactly "the base flying with the donor's
    product" and nothing else.
    """

    def __init__(self, base: JaxProvider, donor: JaxProvider, product: str):
        if product not in SWAP_PRODUCTS:
            raise ValueError(f"unknown product {product!r}; choose from {SWAP_PRODUCTS}")
        self.base, self.donor, self.product = base, donor, product

    def query(self, position, rotation, velocity_body, omega_body) -> dict[str, Any]:
        a = self.base.query(position, rotation, velocity_body, omega_body)
        b = self.donor.query(position, rotation, velocity_body, omega_body)
        mu_a, mu_b = float(a["inverse_mass_v"][0, 0]), float(b["inverse_mass_v"][0, 0])
        rescale_w = a["mass_w"] @ b["inverse_mass_w"]                     # M2_A M2_B^-1
        out = dict(a)
        if self.product == "force_scale":
            g = a["control_map"].copy(); g[:3, :] = (mu_b / mu_a) * b["control_map"][:3, :]; out["control_map"] = g
        elif self.product == "torque_gain":
            g = a["control_map"].copy(); g[3:, :] = rescale_w @ b["control_map"][3:, :]; out["control_map"] = g
        elif self.product == "gravity":
            out["potential_gradient"] = (mu_b / mu_a) * b["potential_gradient"]; out["potential"] = (mu_b / mu_a) * b["potential"]
        elif self.product == "damping_v":
            out["dissipation_v"] = (mu_b / mu_a) * b["dissipation_v"]
        elif self.product == "damping_w":
            out["dissipation_w"] = rescale_w @ b["dissipation_w"]
        return out


# The reference paper's law has no dissipation feedforward. Adding +Dv v_b / +Dw omega_b from the
# learned operators makes the closed loop depend on the damping estimate, which is the least
# identifiable operator (a 6x too-large Dv crashed an otherwise good model), so it is off by default.
USE_DISSIPATION_FEEDFORWARD = False


class EnergyController:
    """The reference energy-based SE(3) law (optional dissipation compensation)."""

    def __init__(self, provider: JaxProvider, use_dissipation: bool = USE_DISSIPATION_FEEDFORWARD,
                 damping_scale: tuple[float, float] = (1.0, 1.0)):
        self.provider = provider
        self.use_dissipation = bool(use_dissipation)
        self.KP = GAINS["Kp"].copy()
        self.KV = GAINS["Kv"].copy() * float(damping_scale[0])
        self.KR = GAINS["KR"].copy()
        self.KW = GAINS["Komega"].copy() * float(damping_scale[1])
        self.damping_scale = (float(damping_scale[0]), float(damping_scale[1]))
        self.maximum_tilt = MAXIMUM_TILT

    def compute(self, state: np.ndarray, desired: DesiredState) -> dict[str, Any]:
        position = state[:3]
        rotation = Rotation.from_quat(state[3:7]).as_matrix()
        rotation_t = rotation.T
        velocity_world = state[10:13]
        velocity_body = rotation_t @ velocity_world
        omega_body = rotation_t @ state[13:16]
        values = self.provider.query(position, rotation, velocity_body, omega_body)
        mass_v, mass_w = values["mass_v"], values["mass_w"]
        dissipation_v, dissipation_w = values["dissipation_v"], values["dissipation_w"]
        potential_gradient = values["potential_gradient"]
        control_map = values["control_map"]
        momentum_v = mass_v @ velocity_body
        momentum_w = mass_w @ omega_body

        dissipation_gain = 1.0 if self.use_dissipation else 0.0
        force_dissipation = dissipation_gain * (dissipation_v @ velocity_body)
        force_body = (
            rotation_t @ potential_gradient[:3]
            - mass_v @ (rotation_t @ (self.KP * (position - desired.pos)))
            - mass_v @ (self.KV * (velocity_body - rotation_t @ desired.vel))
            + mass_v @ (rotation_t @ desired.acc - hat(omega_body) @ (rotation_t @ desired.vel))
            - np.cross(momentum_v, omega_body)
            + force_dissipation
        )
        force_world = rotation @ force_body
        force_norm = np.linalg.norm(force_world)
        tilt_angle = math.acos(float(np.clip(force_world[2] / force_norm, -1.0, 1.0)))
        tilt_scale = 1.0
        if tilt_angle > self.maximum_tilt:
            horizontal_norm = np.linalg.norm(force_world[:2])
            horizontal_maximum = force_world[2] * math.tan(self.maximum_tilt)
            tilt_scale = horizontal_maximum / horizontal_norm
            force_world[:2] *= tilt_scale
        force_body = rotation_t @ force_world

        desired_b2_reference = np.array([-math.sin(desired.yaw), math.cos(desired.yaw), 0.0])
        desired_b3 = force_world / np.linalg.norm(force_world)
        desired_b1 = np.cross(desired_b2_reference, desired_b3)
        desired_b1 /= np.linalg.norm(desired_b1)
        desired_b2 = np.cross(desired_b3, desired_b1)
        desired_rotation = np.column_stack((desired_b1, desired_b2, desired_b3))

        force_body_derivative = tilt_scale * mass_v @ (-self.KP * (velocity_body - rotation_t @ desired.vel))
        force_world_derivative = rotation @ force_body_derivative
        desired_b3_derivative = np.cross(
            np.cross(desired_b3, force_world_derivative / np.linalg.norm(force_world)), desired_b3
        )
        desired_b2_reference_derivative = np.array(
            [-math.cos(desired.yaw) * desired.yawdot, -math.sin(desired.yaw) * desired.yawdot, 0.0]
        )
        projection_argument = (
            np.cross(desired_b2_reference_derivative, desired_b3)
            + np.cross(desired_b2_reference, desired_b3_derivative)
        ) / np.linalg.norm(np.cross(desired_b2_reference, desired_b3))
        desired_b1_derivative = np.cross(desired_b1, np.cross(projection_argument, desired_b1))
        desired_b2_derivative = np.cross(desired_b3_derivative, desired_b1) + np.cross(desired_b3, desired_b1_derivative)
        desired_rotation_derivative = np.column_stack(
            (desired_b1_derivative, desired_b2_derivative, desired_b3_derivative)
        )
        desired_omega = vee(desired_rotation.T @ desired_rotation_derivative)

        rotation_potential = sum(
            np.cross(rotation[row], potential_gradient[3 + 3 * row : 6 + 3 * row]) for row in range(3)
        )
        attitude_error = 0.5 * self.KR * vee(desired_rotation.T @ rotation - rotation_t @ desired_rotation)
        angular_velocity_error = self.KW * (omega_body - rotation_t @ (desired_rotation @ desired_omega))
        torque_dissipation = dissipation_gain * (dissipation_w @ omega_body)
        torque_body = (
            mass_w @ (-attitude_error - angular_velocity_error)
            - np.cross(momentum_w, omega_body)
            - np.cross(momentum_v, velocity_body)
            + rotation_potential
            + torque_dissipation
        )
        generalized_wrench = np.concatenate((force_body, torque_body))
        # Gauge-invariant allocation: u = pinv(M^-1 g) (M^-1 w). Only the identifiable products
        # M1^-1 g_f, M2^-1 g_w, M1^-1 grad V, M^-1 D enter, so the least-squares fit does not depend on
        # the free scale of the learned operators (the paper's pinv(g) w does).
        inverse_mass = np.zeros((6, 6))
        inverse_mass[:3, :3] = values["inverse_mass_v"]
        inverse_mass[3:, 3:] = values["inverse_mass_w"]
        allocation_matrix = inverse_mass @ control_map
        control = np.linalg.pinv(allocation_matrix, rcond=1e-12) @ (inverse_mass @ generalized_wrench)
        control[0] = max(0.0, control[0])
        return {
            **values,
            "current_rotation": rotation,
            "velocity_world": velocity_world,
            "velocity_body": velocity_body,
            "omega_body": omega_body,
            "desired_rotation": desired_rotation,
            "desired_omega": desired_omega,
            "dissipation_force_body": force_dissipation,
            "dissipation_torque_body": torque_dissipation,
            "desired_generalized_wrench": generalized_wrench,
            "physical_control": control,
            "tilt_angle": tilt_angle,
            "tilt_scale": tilt_scale,
            "control_map_condition_number": float(np.linalg.cond(control_map)),
            "allocation": "u = pinv(M^-1 g) (M^-1 w) (gauge-invariant; paper uses pinv(g) w)",
        }


# ---------------------------------------------------------------------------
# Plant
# ---------------------------------------------------------------------------
def make_environment() -> CtrlAviary:
    return CtrlAviary(
        drone_model=DroneModel.CF2P,
        num_drones=1,
        initial_xyzs=INITIAL_XYZ.reshape(1, 3),
        initial_rpys=INITIAL_RPY.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=PHYSICS_HZ,
        ctrl_freq=PHYSICS_HZ,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )


def contact_point_count(env: CtrlAviary) -> int:
    return len(pb.getContactPoints(bodyA=int(env.DRONE_IDS[0]), physicsClientId=int(env.CLIENT)))


def remove_ground_plane(env: CtrlAviary) -> dict[str, Any]:
    client = int(env.CLIENT)
    drone_ids = {int(body_id) for body_id in np.asarray(env.DRONE_IDS).reshape(-1)}
    before = [int(pb.getBodyUniqueId(index, physicsClientId=client)) for index in range(pb.getNumBodies(physicsClientId=client))]
    removed = []
    for body_id in before:
        if body_id in drone_ids:
            continue
        name = pb.getBodyInfo(body_id, physicsClientId=client)[1].decode("utf-8", errors="replace")
        removed.append({"body_id": body_id, "body_name": name})
        pb.removeBody(body_id, physicsClientId=client)
    env.PLANE_ID = -1
    after = [int(pb.getBodyUniqueId(index, physicsClientId=client)) for index in range(pb.getNumBodies(physicsClientId=client))]
    if set(after) != drone_ids:
        raise AssertionError(f"No-ground audit failed: remaining={after}, drones={drone_ids}")
    contacts = contact_point_count(env)
    if contacts != 0:
        raise AssertionError(f"No-ground audit failed: {contacts} contact points remain")
    return {
        "ground_plane_removed": any("plane" in item["body_name"].lower() for item in removed),
        "removed_non_drone_bodies": removed,
        "body_ids_before": before,
        "body_ids_after": after,
        "remaining_bodies_are_only_drones": True,
        "contact_points_after_removal": contacts,
        "collision_environment": "no ground plane and no obstacle bodies",
    }


def apply_linear_damping(env: CtrlAviary, state: np.ndarray, c: float = DAMPING_COEFFICIENT) -> None:
    """External linear damping for one physics step: F_b = -c m R^T v_w, tau_b = -c J R^T omega_w (link frame)."""
    rotation = np.asarray(pb.getMatrixFromQuaternion(state[3:7]), dtype=np.float64).reshape(3, 3)
    force_body = -c * float(env.M) * (rotation.T @ np.asarray(state[10:13], dtype=np.float64))
    torque_body = -c * (np.asarray(env.J, dtype=np.float64) @ (rotation.T @ np.asarray(state[13:16], dtype=np.float64)))
    client = int(env.CLIENT)
    pb.applyExternalForce(int(env.DRONE_IDS[0]), -1, force_body.tolist(), [0.0, 0.0, 0.0], pb.LINK_FRAME, physicsClientId=client)
    pb.applyExternalTorque(int(env.DRONE_IDS[0]), -1, torque_body.tolist(), pb.LINK_FRAME, physicsClientId=client)


def configure_contact_free_dynamics(env: CtrlAviary, damping_law: str = "nonlinear") -> dict[str, Any]:
    """Zero every contact friction; built-in nonlinear damping (default) or built-in damping off for the linear law."""
    if damping_law not in ("nonlinear", "linear"):
        raise ValueError(f"unknown damping_law {damping_law!r}")
    builtin = DAMPING_COEFFICIENT if damping_law == "nonlinear" else 0.0
    client = int(env.CLIENT)
    records = []
    for body_index in range(pb.getNumBodies(physicsClientId=client)):
        body_id = int(pb.getBodyUniqueId(body_index, physicsClientId=client))
        name = pb.getBodyInfo(body_id, physicsClientId=client)[1].decode("utf-8", errors="replace")
        for link_index in range(-1, pb.getNumJoints(body_id, physicsClientId=client)):
            pb.changeDynamics(
                body_id, link_index,
                lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0,
                linearDamping=0.0, angularDamping=0.0, physicsClientId=client,
            )
            info = pb.getDynamicsInfo(body_id, link_index, physicsClientId=client)
            records.append({
                "body_id": body_id, "body_name": name, "link_index": link_index,
                "collision_shape_count": len(pb.getCollisionShapeData(body_id, link_index, physicsClientId=client)),
                "lateral_friction": float(info[1]), "rolling_friction": float(info[6]), "spinning_friction": float(info[7]),
            })
    pb.changeDynamics(
        int(env.DRONE_IDS[0]), -1,
        linearDamping=builtin, angularDamping=builtin, physicsClientId=client,
    )
    if any(
        record[key] != 0.0
        for record in records if record["collision_shape_count"] > 0
        for key in ("lateral_friction", "rolling_friction", "spinning_friction")
    ):
        raise AssertionError("Contact-friction zero audit failed")
    return {
        "damping_law": damping_law,
        "linear_damping_coefficient": DAMPING_COEFFICIENT,
        "angular_damping_coefficient": DAMPING_COEFFICIENT,
        "pybullet_builtin_damping": damping_law == "nonlinear",
        "custom_external_linear_damping": damping_law == "linear",
        "contact_friction_disabled": True,
        "body_links": records,
    }


def motor_mixer(env: CtrlAviary) -> tuple[np.ndarray, np.ndarray]:
    ratio = env.KM / env.KF
    matrix = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [0.0, env.L, 0.0, -env.L],
        [-env.L, 0.0, env.L, 0.0],
        [-ratio, ratio, -ratio, ratio],
    ])
    return matrix, np.linalg.inv(matrix)


def trajectory_row(state: np.ndarray, omega_body: np.ndarray, t: float) -> np.ndarray:
    row = np.zeros(14)
    row[:3] = state[:3]
    row[3:6] = state[10:13]
    row[9] = state[9]
    row[10:13] = omega_body
    row[-1] = t
    return row


def desired_row(desired: DesiredState, t: float) -> np.ndarray:
    row = np.zeros(11)
    row[:3] = desired.pos
    row[3:6] = desired.vel
    row[6:9] = desired.acc
    row[9] = desired.yaw
    row[-1] = t
    return row


def tracking_metrics(s_traj: np.ndarray, s_plan: np.ndarray, controller_times: np.ndarray) -> dict[str, Any]:
    count = min(len(s_traj), len(s_plan))
    position_error = np.linalg.norm(s_traj[:count, :3] - s_plan[:count, :3], axis=1)
    velocity_error = np.linalg.norm(s_traj[:count, 3:6] - s_plan[:count, 3:6], axis=1)
    yaw_error = np.angle(np.exp(1j * (s_traj[:count, 9] - s_plan[:count, 9])))
    return {
        "samples": int(count),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_final_m": float(position_error[-1]),
        "position_max_m": float(np.max(position_error)),
        "velocity_rmse_m_per_s": float(np.sqrt(np.mean(velocity_error**2))),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error**2))),
        "controller_time_median_ms": float(np.median(controller_times) * 1000.0) if controller_times.size else float("nan"),
        "controller_time_p95_ms": float(np.percentile(controller_times, 95) * 1000.0) if controller_times.size else float("nan"),
        "controller_deadline_miss_fraction": float(np.mean(controller_times > DT)) if controller_times.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Plots (same content as the released plot_states1D / quadplot_update pages)
# ---------------------------------------------------------------------------
def save_plots(output_dir: Path, model_label: str, s_traj: np.ndarray, s_plan: np.ndarray, full_plan: np.ndarray,
               reference: str | None = None, baseline: np.ndarray | None = None,
               baseline_label: str = "PH-GT", show_plan: bool = True,
               suffix: str = "") -> dict[str, str]:
    """``baseline``: another recorded s_traj on the same reference, drawn as a second curve.

    ``show_plan`` draws the commanded reference; turning it off leaves learned-versus-baseline alone, which is
    the page that answers "how far am I from the exact-physics controller". ``suffix`` is appended to the file
    names so the two versions of the same flight live side by side.

    The reference holds instantaneously at ACTIVE_SECONDS, which is not a feasible command, so every controller
    rings there. Drawing the analytic-operator flight on the same axes separates 'the model is wrong' from
    'the command is infeasible': the two curves ring together.
    """
    png_dir = output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    traj_color, plan_color, plan_style = "b", "chocolate", "dashed"
    base_color, base_style = "0.15", "dashdot"

    figure = plt.figure(figsize=(10, 7.5))
    axes = {
        "px": plt.subplot(421), "py": plt.subplot(423), "pz": plt.subplot(425), "yaw": plt.subplot(427),
        "vx": plt.subplot(422), "vy": plt.subplot(424), "vz": plt.subplot(426), "w": plt.subplot(428),
    }
    for axis in axes.values():
        axis.grid(True)
    for key, column, ylabel in (("px", 0, "x (m)"), ("py", 1, "y (m)"), ("pz", 2, "z (m)"),
                                ("vx", 3, "x (m/s)"), ("vy", 4, "y (m/s)"), ("vz", 5, "z (m/s)"), ("yaw", 9, "yaw (rad)")):
        axis = axes[key]
        axis.plot(s_traj[:, -1], s_traj[:, column], color=traj_color, label="learned" if key == "px" else None)
        if show_plan:
            axis.plot(s_plan[:, -1], s_plan[:, column], color=plan_color, linestyle=plan_style,
                      label="reference" if key == "px" else None)
        if baseline is not None:
            axis.plot(baseline[:, -1], baseline[:, column], color=base_color, linestyle=base_style, lw=0.9,
                      label=baseline_label if key == "px" else None)
        axis.set_ylabel(ylabel)
    # The three body rates share one axis, so they need three colours: one line at -2 rad/s next to two near
    # zero is omega_z tracking the commanded yaw rate, not a glitch.
    for column, colour, name in ((10, "#1f77b4", r"$\omega_x$"), (11, "#2ca02c", r"$\omega_y$"), (12, "#9467bd", r"$\omega_z$")):
        axes["w"].plot(s_traj[:, -1], s_traj[:, column], color=colour, lw=1.0, label=name)
        if baseline is not None:      # same colour, dotted: each component pairs with its ground-truth counterpart
            axes["w"].plot(baseline[:, -1], baseline[:, column], color=colour, lw=0.9, ls=":", alpha=0.85,
                           label=f"{name} GT")
    # The dashed line is what the yaw loop is actually given: d(psi_ref)/dt, analytic, and zero for the shapes
    # whose yaw reference is constant.
    if show_plan:
        yaw_rate = (np.array([REFERENCES[reference](float(time)).yawdot for time in s_traj[:, -1]])
                    if reference in REFERENCES else np.zeros(s_traj.shape[0]))
        axes["w"].plot(s_traj[:, -1], yaw_rate, color=plan_color, linestyle=plan_style, lw=1.2,
                       label=r"$\dot\psi_{\mathrm{ref}}$")
    axes["w"].set_ylabel(r"$\omega$ (rad/s)")
    axes["w"].legend(loc="upper right", fontsize=6 if baseline is not None else 7,
                     ncol=(4 if baseline is None else 6 if not show_plan else 7), framealpha=0.85)
    axes["px"].set_title("Position/Yaw")
    axes["vx"].set_title("Velocity")
    axes["yaw"].set_xlabel("Time (s)")
    # The vehicle starts at yaw 30 degrees while the shape starts at its own psi_ref(0): the first tenths of a
    # second of the yaw panel are that slew, and the start is identical for every model.
    axes["yaw"].text(0.99, 0.04, r"start: $p_{\mathrm{ref}}(0)=p(0)$ and $\psi(0)=\psi_{\mathrm{ref}}(0)=0$",
                     transform=axes["yaw"].transAxes, ha="right", va="bottom", fontsize=6.5, color="0.35",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5})
    axes["w"].set_xlabel("Time (s)")
    if baseline is not None:
        # The reference stops instantaneously at ACTIVE_SECONDS; every controller rings there, the exact-physics
        # one included, so the honest comparison on this page is the model against the dash-dot baseline.
        # Short lines, left column: the figure legend sits centred on the same band, so a long line runs under it.
        figure.text(0.01, -0.03,
                    f"dash-dot: {baseline_label}\nsame controller and gains,\nanalytic operators"
                    + ("" if show_plan else "\nreference not drawn here")
                    + f"\nplan holds at {ACTIVE_SECONDS:.0f} s:\nboth ring there",
                    ha="left", va="top", fontsize=7, color="0.3", linespacing=1.4)
    lines, labels = axes["px"].get_legend_handles_labels()
    figure.legend(lines, labels, loc="lower center", ncol=2, prop={"size": 12})
    plt.subplots_adjust(left=0.1, right=0.98, top=0.93, wspace=0.3)
    figure.savefig(png_dir / f"tracking_results{suffix}.png", bbox_inches="tight", pad_inches=0.1)
    against = baseline_label if (baseline is not None and not show_plan) else "reference"
    figure.suptitle(f"{model_label} — plot_states1D — learned vs {against}", fontweight="bold")
    figure.savefig(png_dir / f"tracking_results_labeled{suffix}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    trajectory = plt.figure()
    axis = trajectory.add_subplot(projection="3d")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.plot3D(s_traj[:, 0], s_traj[:, 1], s_traj[:, 2], color=traj_color, label="learned")
    if show_plan:
        axis.plot3D(full_plan[:, 0], full_plan[:, 1], full_plan[:, 2], "--", color=plan_color, label="reference")
    if baseline is not None:
        axis.plot3D(baseline[:, 0], baseline[:, 1], baseline[:, 2], color=base_color, ls=base_style, lw=0.9,
                    label=baseline_label)
    axis.legend(loc="upper center", ncol=2)
    scale_reference = full_plan[:, :3] if show_plan else baseline[:, :3]   # what the axes are sized to
    positions = np.concatenate((s_traj[:, :3], scale_reference), axis=0)
    lower, upper = np.nanmin(positions, axis=0), np.nanmax(positions, axis=0)
    span = max(float(np.max(upper - lower)) * 1.1, 2.0)
    center = (lower + upper) / 2.0
    axis.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
    axis.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
    axis.set_zlim(center[2] - span / 2.0, center[2] + span / 2.0)
    axis.view_init(elev=25.0, azim=35)
    zoom = trajectory.add_axes([0.66, 0.13, 0.28, 0.32], projection="3d")
    if baseline is not None:
        zoom.plot3D(baseline[:, 0], baseline[:, 1], baseline[:, 2], color=base_color, ls=base_style, linewidth=0.8)
    zoom.plot3D(s_traj[:, 0], s_traj[:, 1], s_traj[:, 2], color=traj_color, linewidth=1.0, label="learned")
    if show_plan:
        zoom.plot3D(full_plan[:, 0], full_plan[:, 1], full_plan[:, 2], "--", color=plan_color, linewidth=1.2, label="reference")
    plan_lower, plan_upper = np.nanmin(scale_reference, axis=0), np.nanmax(scale_reference, axis=0)
    plan_span = max(float(np.max(plan_upper - plan_lower)) * 1.15, 2.0)
    plan_center = (plan_lower + plan_upper) / 2.0
    zoom.set_xlim(plan_center[0] - plan_span / 2.0, plan_center[0] + plan_span / 2.0)
    zoom.set_ylim(plan_center[1] - plan_span / 2.0, plan_center[1] + plan_span / 2.0)
    zoom.set_zlim(plan_center[2] - plan_span / 2.0, plan_center[2] + plan_span / 2.0)
    zoom.view_init(elev=25.0, azim=35.0)
    zoom.set_title("Reference-scale zoom", fontsize=8)
    zoom.tick_params(labelsize=6)
    trajectory.savefig(png_dir / f"traj{suffix}.png", dpi=160, bbox_inches="tight", pad_inches=0.1)
    trajectory.suptitle(f"{model_label} — quadplot_update — learned vs {against}", fontweight="bold")
    trajectory.savefig(png_dir / f"trajectory_labeled{suffix}.png", dpi=160, bbox_inches="tight")
    plt.close(trajectory)
    return {
        "released_tracking_plot": str((png_dir / f"tracking_results{suffix}.png").resolve()),
        "labeled_tracking_plot": str((png_dir / f"tracking_results_labeled{suffix}.png").resolve()),
        "released_trajectory_plot": str((png_dir / f"traj{suffix}.png").resolve()),
        "labeled_trajectory_plot": str((png_dir / f"trajectory_labeled{suffix}.png").resolve()),
    }


# ---------------------------------------------------------------------------
# Closed-loop run
# ---------------------------------------------------------------------------
def run_controller(
    run: dict[str, Any],
    model,
    output_dir: Path,
    *,
    model_label: str,
    training_dataset: Path,
    vehicle: dict[str, Any],
    duration_seconds: float = 20.0,
    seed: int = 0,
    device_text: str = "cpu",
    use_dissipation: bool | None = None,
    reference: str = "diamond",
    provider=None,
    provider_description: str = "the model's own six operators",
    gust: dict[str, Any] | None = None,
    damping_scale: tuple[float, float] = (1.0, 1.0),
    damping_schedule: dict[str, Any] | None = None,
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``gust``: optional unobserved Ornstein-Uhlenbeck body-frame force on the PLANT (the same disturbance the WIND
    dataset was generated with): {"time_constant_seconds", "sigma_fraction_of_weight", "seed"}. Updated at every
    physics step with g <- e^{-dt/tau} g + sigma sqrt(1 - e^{-2dt/tau}) xi and applied with applyExternalForce
    in the link frame; the controller never sees it. ``damping_scale`` multiplies the damping-injection gains
    (Kv, Komega); ``damping_schedule`` only documents where the scale came from. ``initial_state``: optional
    {"position", "rotation" (3x3), "velocity_body", "omega_body"} the plant is reset to before t = 0 (a recorded
    flight's x_0), instead of hovering at INITIAL_XYZ / INITIAL_RPY."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if reference not in REFERENCES:
        raise ValueError(f"unknown controller reference {reference!r}; choose from {sorted(REFERENCES)}")
    reference_function = REFERENCES[reference]
    controller = EnergyController(
        JaxProvider(model) if provider is None else provider,
        use_dissipation=USE_DISSIPATION_FEEDFORWARD if use_dissipation is None else bool(use_dissipation),
        damping_scale=damping_scale,
    )
    gust_rng = None
    gust_force = np.zeros(3)
    gust_series: list[np.ndarray] = []
    white_wind = gust is not None and gust.get("model") == "white_state_dependent"
    if gust is not None and not white_wind and float(gust.get("sigma_fraction_of_weight", 0.0)) > 0.0:
        gust_rng = np.random.default_rng(int(gust.get("seed", 0)))
    # gust = {"model": "white_state_dependent", linear_sigma, speed_gain, angular_sigma, rate_gain, hold_seconds, seed}:
    # the QUADROTOR-DATASET-WINDSDE wind (same law as its generator): every hold a body force m sigma_a(x) xi / sqrt(hold)
    # and torque J sigma_alpha(x) zeta / sqrt(hold), sigma_a = linear_sigma (1 + speed_gain |v_b|),
    # sigma_alpha = angular_sigma (1 + rate_gain |omega_b|); unobserved by the controller.
    white_rng = np.random.default_rng(int(gust.get("seed", 0))) if white_wind else None
    white_force, white_torque, white_series = np.zeros(3), np.zeros(3), []
    total_steps = round(duration_seconds * PHYSICS_HZ)
    full_plan_array = None            # built after the reset: it is anchored to the measured start state
    env = make_environment()
    states, plans, diagnostics = [], [], []
    requested_rpms, applied_rpms, applied_controls, controller_times = [], [], [], []
    maximum_contact_points = 0
    minimum_altitude = np.inf
    failure = None
    try:
        observation, _ = env.reset(seed=seed)
        no_ground_audit = remove_ground_plane(env)
        damping_law = str(vehicle.get("damping_law", "nonlinear"))
        dynamics_audit = configure_contact_free_dynamics(env, damping_law)
        dynamics_audit["no_ground"] = no_ground_audit
        observation, *_ = env.step(np.zeros((1, 4), dtype=np.float64))
        state = observation[0]
        if initial_state is not None:
            # Start at the recorded x_0: pose and twist are written after the settling step, then re-read.
            drone, client = int(env.DRONE_IDS[0]), int(env.CLIENT)
            rotation_0 = np.asarray(initial_state["rotation"], dtype=np.float64).reshape(3, 3)
            pb.resetBasePositionAndOrientation(drone, np.asarray(initial_state["position"], dtype=np.float64).tolist(),
                                               Rotation.from_matrix(rotation_0).as_quat().tolist(), physicsClientId=client)
            pb.resetBaseVelocity(drone, (rotation_0 @ np.asarray(initial_state["velocity_body"], dtype=np.float64)).tolist(),
                                 (rotation_0 @ np.asarray(initial_state["omega_body"], dtype=np.float64)).tolist(),
                                 physicsClientId=client)
            env._updateAndStoreKinematicInformation()
            state = env._getDroneStateVector(0)
        maximum_contact_points = max(maximum_contact_points, contact_point_count(env))
        mixer, mixer_inverse = motor_mixer(env)
        # The reference is anchored to the position the vehicle is actually in at t=0, not to the nominal start:
        # the plant settles a fraction of a millimetre during the reset. Yaw is NOT anchored - the vehicle now
        # starts at the shapes' own heading, so a deliberate initial yaw offset would stay a real tracking error.
        anchor_position = np.asarray(state[:3], dtype=np.float64)

        def anchored(desired: DesiredState) -> DesiredState:
            desired.pos = desired.pos + anchor_position
            return desired

        full_plan_array = np.asarray([desired_row(anchored(reference_function(plan_step * DT)), plan_step * DT)
                                      for plan_step in range(total_steps + 1)])
        for step in range(total_steps + 1):
            current_time = step * DT
            desired = anchored(reference_function(current_time))
            rotation = Rotation.from_quat(state[3:7]).as_matrix()
            omega_body = rotation.T @ state[13:16]
            states.append(trajectory_row(state, omega_body, current_time))
            plans.append(full_plan_array[step])
            minimum_altitude = min(minimum_altitude, float(state[2]))
            if step == total_steps:
                break
            started = time.perf_counter()
            try:
                result = controller.compute(state, desired)
            except BaseException as error:  # noqa: BLE001 - recorded as a benchmark outcome
                failure = {"kind": "controller_exception", "step": step, "time_seconds": current_time, "error": repr(error)}
                break
            controller_times.append(time.perf_counter() - started)
            control = np.asarray(result["physical_control"], dtype=np.float64)
            if not np.all(np.isfinite(control)):
                failure = {"kind": "controller_exception", "step": step, "time_seconds": current_time, "error": "non-finite wrench"}
                break
            motor_force = np.maximum(mixer_inverse @ control, 0.0)
            requested_rpm = np.sqrt(motor_force / env.KF)
            applied_rpm = np.clip(requested_rpm, 0.0, env.MAX_RPM)
            applied_force = mixer @ (applied_rpm**2 * env.KF)
            result["allocation_residual"] = float(np.linalg.norm(applied_force - control))
            diagnostics.append(result)
            requested_rpms.append(requested_rpm)
            applied_rpms.append(applied_rpm)
            applied_controls.append(applied_force)
            try:
                if damping_law == "linear":
                    apply_linear_damping(env, state)
                if white_rng is not None:
                    hold_steps = max(1, int(round(float(gust.get("hold_seconds", DT)) / DT)))
                    if step % hold_steps == 0:
                        rotation_now = np.asarray(pb.getMatrixFromQuaternion(state[3:7]), dtype=np.float64).reshape(3, 3)
                        speed = float(np.linalg.norm(rotation_now.T @ np.asarray(state[10:13], dtype=np.float64)))
                        rate = float(np.linalg.norm(rotation_now.T @ np.asarray(state[13:16], dtype=np.float64)))
                        sigma_a = float(gust["linear_sigma"]) * (1.0 + float(gust["speed_gain"]) * speed)
                        sigma_alpha = float(gust["angular_sigma"]) * (1.0 + float(gust["rate_gain"]) * rate)
                        hold = hold_steps * DT
                        white_force = float(env.M) * sigma_a * white_rng.normal(size=3) / math.sqrt(hold)
                        white_torque = np.asarray(env.J, dtype=np.float64) @ (sigma_alpha * white_rng.normal(size=3)) / math.sqrt(hold)
                    pb.applyExternalForce(int(env.DRONE_IDS[0]), -1, white_force.tolist(), [0.0, 0.0, 0.0],
                                          pb.LINK_FRAME, physicsClientId=int(env.CLIENT))
                    pb.applyExternalTorque(int(env.DRONE_IDS[0]), -1, white_torque.tolist(), pb.LINK_FRAME,
                                           physicsClientId=int(env.CLIENT))
                    white_series.append(np.concatenate([white_force, white_torque]))
                if gust_rng is not None:
                    tau_w = float(gust["time_constant_seconds"])
                    sigma_w = float(gust["sigma_fraction_of_weight"]) * float(env.M) * float(env.G)
                    decay = math.exp(-DT / tau_w)
                    gust_force = gust_force * decay + sigma_w * math.sqrt(1.0 - decay * decay) * gust_rng.normal(size=3)
                    pb.applyExternalForce(int(env.DRONE_IDS[0]), -1, gust_force.tolist(), [0.0, 0.0, 0.0],
                                          pb.LINK_FRAME, physicsClientId=int(env.CLIENT))
                    gust_series.append(gust_force.copy())
                observation, *_ = env.step(requested_rpm.reshape(1, 4))
            except BaseException as error:  # noqa: BLE001
                failure = {"kind": "pybullet_exception", "step": step, "time_seconds": current_time, "error": f"PyBullet step failed: {error!r}"}
                break
            state = observation[0]
            contacts = contact_point_count(env)
            maximum_contact_points = max(maximum_contact_points, contacts)
            if contacts:
                failure = {"kind": "ground_contact", "step": step, "time_seconds": current_time, "error": f"{contacts} contacts"}
                break
        plant = {
            "mass_kg": float(env.M),
            "inertia_kg_m2": np.asarray(env.J, dtype=np.float64).tolist(),
            "gravity_m_per_s2": float(env.G),
            "arm_length_m": float(env.L),
            "kf": float(env.KF),
            "km": float(env.KM),
            "hover_rpm": float(env.HOVER_RPM),
            "maximum_rpm": float(env.MAX_RPM),
            "physics": "Physics.PYB",
        }
    finally:
        env.close()

    state_array = np.asarray(states)
    plan_array = np.asarray(plans[: len(state_array)])
    timing_array = np.asarray(controller_times)
    diagnostic_arrays = {key: np.asarray([entry[key] for entry in diagnostics]) for key in diagnostics[0]} if diagnostics else {}
    requested_rpm_array = np.asarray(requested_rpms) if requested_rpms else np.zeros((0, 4))
    applied_rpm_array = np.asarray(applied_rpms) if applied_rpms else np.zeros((0, 4))
    metrics = tracking_metrics(state_array, plan_array, timing_array)
    exceedance = requested_rpm_array > plant["maximum_rpm"]
    metrics.update({
        "controller_failure": failure is not None,
        "failure": failure,
        "maximum_requested_rpm": float(np.max(requested_rpm_array)) if requested_rpm_array.size else float("nan"),
        "maximum_applied_rpm": float(np.max(applied_rpm_array)) if applied_rpm_array.size else float("nan"),
        "motor_saturation_fraction": float(np.mean(exceedance)) if exceedance.size else float("nan"),
        "any_motor_saturation_step_fraction": float(np.mean(np.any(exceedance, axis=1))) if exceedance.size else float("nan"),
        "mean_allocation_residual": float(np.mean(diagnostic_arrays["allocation_residual"])) if diagnostics else float("nan"),
        "maximum_contact_points": int(maximum_contact_points),
        "minimum_altitude_m": float(minimum_altitude),
    })
    np.savez_compressed(
        output_dir / "controller_rollout.npz",
        gust_force_body=np.asarray(gust_series) if gust_series else np.zeros((0, 3)),
        white_wind_wrench_body=np.asarray(white_series) if white_series else np.zeros((0, 6)),
        s_traj=state_array, s_plan=plan_array, s_plan_full=full_plan_array,
        controller_time_seconds=timing_array, requested_rpm=requested_rpm_array, applied_rpm=applied_rpm_array,
        applied_physical_control=np.asarray(applied_controls) if applied_controls else np.zeros((0, 4)),
        **diagnostic_arrays,
    )
    plots = save_plots(output_dir, model_label, state_array, plan_array, full_plan_array, reference)
    metadata = {
        "status": "completed_with_physical_failure" if failure is not None and failure.get("kind") == "ground_contact"
        else "completed" if failure is None else "failed",
        "experiment": "contact-free closed-loop control with the learned JAX model",
        "model_name": run["metadata"].get("model_name"),
        "model_label": model_label,
        "model_family": "ph-gp",
        "inference_device": device_text,
        "integrator": run["metadata"].get("integrator"),
        "damping_law": str(vehicle.get("damping_law", "nonlinear")),
        "damping_coefficient": DAMPING_COEFFICIENT,
        "dissipation_mode": "on" if controller.use_dissipation else "off",
        "dissipation_input": "Dv: v_b only; Dw: omega_b only",
        "controller_uses_dissipation": bool(controller.use_dissipation),
        "controller_dissipation_equation": (
            "+Dv*v_b and +Dw*omega_b" if controller.use_dissipation
            else "none (reference paper law; learned damping not fed forward)"
        ),
        "wrench_allocation": "u = pinv(M^-1 g) (M^-1 w), gauge-invariant (reference paper: u = pinv(g) w)",
        "control_representation": "physical wrench [T, tau_x, tau_y, tau_z]",
        "actuator_policy": {
            "upper_rpm_clipping": True,
            "nonnegative_rotor_thrust_retained": True,
            "physical_maximum_rpm_reference": plant["maximum_rpm"],
            "unbounded_positive_rpm_diagnostic": False,
            "tilt_limit_enabled": True,
            "numerical_acos_clip_retained": True,
        },
        "duration_seconds": duration_seconds,
        "reference": reference,
        "operator_provider": provider_description,
        "reference_active_seconds": ACTIVE_SECONDS if reference != "diamond" else 15.0,
        "reference_feedforward": "position, velocity and acceleration (analytic)" if reference != "diamond" else "position and velocity; acceleration zero",
        "full_reference_duration_seconds": float(full_plan_array[-1, -1]),
        "full_reference_samples": int(len(full_plan_array)),
        "ground_plane_removed": bool(no_ground_audit["ground_plane_removed"]),
        "collision_environment": no_ground_audit["collision_environment"],
        "controller_hz": PHYSICS_HZ,
        "physics_hz": PHYSICS_HZ,
        "controller_gains": {name: value.tolist() for name, value in GAINS.items()},
        "maximum_tilt_rad": MAXIMUM_TILT,
        "reference_anchor": "measured initial position; vehicle starts at the shape's own heading",
        "initial_xyz": INITIAL_XYZ.tolist(),
        "initial_rpy": INITIAL_RPY.tolist(),
        "initial_state": (None if initial_state is None else
                          {k: np.asarray(v, dtype=np.float64).tolist() for k, v in initial_state.items()}),
        "seed": seed,
        "contact_free": maximum_contact_points == 0,
        "dynamics_audit": dynamics_audit,
        "plant": plant,
        "ground_truth_source": {
            "mass_kg": vehicle["mass"],
            "inertia_kg_m2": np.asarray(vehicle["inertia"]).tolist(),
            "gravity_m_per_s2": vehicle["gravity"],
            "nonlinear_damping_coefficient": DAMPING_COEFFICIENT,
        },
        "metrics": metrics,
        "training_run": str(run["directory"]),
        "checkpoint": str(run["checkpoint"]),
        "checkpoint_name": run["checkpoint"].name,
        "checkpoint_sha256": run["checkpoint_sha256"],
        "training_dataset": str(training_dataset),
        "training_dataset_sha256": run["metadata"].get("dataset_sha256"),
        "plots": plots,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata["gust"] = (None if gust_rng is None else {
        "model": "Ornstein-Uhlenbeck body-frame force on the plant, unobserved by the controller",
        "time_constant_seconds": float(gust["time_constant_seconds"]),
        "sigma_fraction_of_weight": float(gust["sigma_fraction_of_weight"]),
        "sigma_newtons": float(gust["sigma_fraction_of_weight"]) * float(plant["mass_kg"]) * float(plant["gravity_m_per_s2"]),
        "seed": int(gust.get("seed", 0)),
        "rms_applied_newtons": float(np.sqrt(np.mean(np.sum(np.asarray(gust_series) ** 2, axis=1)))) if gust_series else 0.0,
    })
    if white_wind:
        metadata["gust"] = {"model": "white_state_dependent (QUADROTOR-DATASET-WINDSDE law), unobserved by the controller",
                            **{k: gust[k] for k in ("linear_sigma", "speed_gain", "angular_sigma", "rate_gain", "hold_seconds", "seed")},
                            "rms_force_newtons": float(np.sqrt(np.mean(np.sum(np.asarray(white_series)[:, :3] ** 2, axis=1)))) if white_series else 0.0}
    metadata["damping_injection"] = {
        "Kv": controller.KV.tolist(), "Komega": controller.KW.tolist(),
        "scale_over_nominal": list(controller.damping_scale),
        "schedule": damping_schedule or {"kind": "none"},
    }
    (output_dir / "controller_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return {"directory": output_dir, "metadata": metadata,
            "data": {"s_traj": state_array, "s_plan": plan_array, "requested_rpm": requested_rpm_array,
                     "applied_physical_control": np.asarray(applied_controls) if applied_controls else np.zeros((0, 4)),
                     "current_rotation": diagnostic_arrays.get("current_rotation"),
                     "tilt_angle": diagnostic_arrays.get("tilt_angle")}}


VALID_TRACKING_FRACTION = 0.158     # sqrt(0.025): the VPT convention, as a fraction of the shape's own extent


def valid_tracking_seconds(times: np.ndarray, distance: np.ndarray, threshold: float) -> float:
    """First time the flight leaves ``threshold`` of its target, capped at the end of the flight."""
    exceeded = np.flatnonzero(distance > threshold)
    return float(times[exceeded[0]] if exceeded.size else times[-1])


def euler_angles(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll, pitch, yaw (ZYX) in radians for a stack of rotation matrices."""
    roll = np.arctan2(rotations[:, 2, 1], rotations[:, 2, 2])
    pitch = np.arcsin(-np.clip(rotations[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    return roll, pitch, yaw


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def flat_reference_angles(plan: np.ndarray, mass: float, gravity: float, damping: float):
    """Roll and pitch implied by the reference through differential flatness, with the TRUE constants.

    The quadrotor is under-actuated: the reference fixes position and yaw, and the tilt follows from the
    acceleration it demands, b3 ~ m(a_ref + g e3) + m c (1+|v_ref|) v_ref. A real controller must lead or lag
    this attitude to correct position error, so even an exact-physics controller shows a non-zero error
    against it -- that is why the PH-GT column belongs in the table as the attainable floor.
    """
    velocity, acceleration, yaw = plan[:, 3:6], plan[:, 6:9], plan[:, 9]
    force = mass * (acceleration + np.array([0.0, 0.0, gravity])) \
        + mass * damping * (1.0 + np.linalg.norm(velocity, axis=1))[:, None] * velocity
    b3 = force / np.linalg.norm(force, axis=1, keepdims=True)
    roll = np.arctan2(b3[:, 1] * np.cos(yaw) - b3[:, 0] * np.sin(yaw), b3[:, 2])
    pitch = np.arctan2(b3[:, 0] * np.cos(yaw) + b3[:, 1] * np.sin(yaw), b3[:, 2])
    return roll, pitch


def controller_comparison(controller: dict[str, Any]) -> dict[str, Any]:
    """Failure-aware tracking metrics on the measured horizon of the single model."""
    data, metadata = controller["data"], controller["metadata"]
    state, plan = data["s_traj"], data["s_plan"][: len(data["s_traj"])]
    count = len(state)
    position_error = np.linalg.norm(state[:, :3] - plan[:, :3], axis=1)
    velocity_error = np.linalg.norm(state[:, 3:6] - plan[:, 3:6], axis=1)
    requested_duration = float(metadata["duration_seconds"])
    completed_duration = float(state[-1, -1])
    maximum_rpm = float(metadata["plant"]["maximum_rpm"])
    control_count = min(count, len(data["requested_rpm"]))
    requested_rpm = data["requested_rpm"][:control_count]
    failure = metadata["metrics"].get("failure")
    plant = metadata["plant"]
    mass, gravity, arm = float(plant["mass_kg"]), float(plant["gravity_m_per_s2"]), float(plant["arm_length_m"])
    hover_force, hover_torque = mass * gravity, mass * gravity * arm
    extra: dict[str, Any] = {}
    rotations = data.get("current_rotation")
    if rotations is not None and len(rotations):
        shared = min(count, len(rotations))
        roll, pitch, _ = euler_angles(np.asarray(rotations)[:shared])
        roll_reference, pitch_reference = flat_reference_angles(
            plan[:shared], mass, gravity, float(metadata.get("damping_coefficient", 0.0)))
        extra["roll_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(wrap_to_pi(roll - roll_reference) ** 2))))
        extra["pitch_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(wrap_to_pi(pitch - pitch_reference) ** 2))))
    if metadata["metrics"].get("yaw_rmse_rad") is not None:
        extra["yaw_rmse_deg"] = float(np.degrees(metadata["metrics"]["yaw_rmse_rad"]))
    control = data.get("applied_physical_control")
    if control is not None and len(control) > 1:
        control = np.asarray(control)[:control_count]
        extra["thrust_rms_hover"] = float(np.sqrt(np.mean((control[:, 0] / hover_force) ** 2)))
        # Normalised wrench, so thrust (N) and torque (N m) are comparable before differencing.
        normalised = np.column_stack((control[:, 0] / hover_force, control[:, 1:] / hover_torque))
        step = float(np.median(np.diff(state[:, -1]))) if count > 1 else 1.0
        extra["control_chatter_per_s"] = float(np.sqrt(np.mean(np.sum(np.diff(normalised, axis=0) ** 2, axis=1))) / step)
    tilt = data.get("tilt_angle")
    if tilt is not None and len(tilt):
        extra["max_tilt_deg"] = float(np.degrees(np.max(np.asarray(tilt)[:control_count])))
    if metadata["metrics"].get("controller_time_p95_ms") is not None:
        extra["controller_time_p95_ms"] = float(metadata["metrics"]["controller_time_p95_ms"])
    # Valid tracking time against the commanded reference, same threshold convention as the PH-GT version.
    excursion = float(np.sqrt(np.mean(np.sum((plan[:, :3] - plan[0, :3]) ** 2, axis=1))))
    extra["valid_tracking_seconds_vs_reference"] = valid_tracking_seconds(
        state[:, -1], position_error, VALID_TRACKING_FRACTION * excursion)
    return {
        **extra,
        "shared_sample_count": count,
        "shared_horizon_seconds": completed_duration,
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_final_m": float(position_error[-1]),
        "position_max_m": float(np.max(position_error)),
        "velocity_rmse_m_per_s": float(np.sqrt(np.mean(velocity_error**2))),
        "requested_duration_seconds": requested_duration,
        "completed_duration_seconds": completed_duration,
        "completion_fraction": min(completed_duration / requested_duration, 1.0),
        "controller_failure": failure is not None,
        "failure": failure,
        "maximum_contact_points": int(metadata["metrics"]["maximum_contact_points"]),
        "shared_motor_saturation_fraction": float(np.mean(requested_rpm > maximum_rpm)) if requested_rpm.size else float("nan"),
    }

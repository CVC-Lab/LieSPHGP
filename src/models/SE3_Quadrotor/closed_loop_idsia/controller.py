"""The Crazyflie firmware's geometric controller (Mellinger & Kumar 2011 with integral action), as described in
Busetto et al., Control Engineering Practice 172 (2026) 106871, section 4.2, followed by the benchmark's mixer.

Outer loop (paper, section 4.2):
    F_d = K_p (p_d - p) + K_i int(p_d - p) + K_d (v_d - v) + m (a_d + g e_3),      T = F_d . z_B
Desired attitude: z_d along F_d, heading from the yaw reference psi_d.
Inner loop on SO(3):
    e_R = 1/2 (R_d^T R - R^T R_d)^vee,   e_w = w - w_d,
    tau = -K_R e_R - K_w e_w - K_iR int(e_R)
Allocation: [T, tau] = MIX @ Omega^2 with the benchmark's mixer (models/models.py:82-84, K_t, K_c, arm a),
inverted and clipped to the physical rotor-speed range.

The paper gives the control law but not the gain values flown, so the gains here are our own, chosen by
natural frequency and damping for the published mass and inertia (see ``Gains``) and checked on the analytic
rigid body before any learned model is flown.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

MASS, GRAVITY = 0.045, 9.81
INERTIA = np.array([2.3951e-05, 2.3951e-05, 3.2347e-06])
KF, KM, ARM = 3.72e-08, 7.74e-12, 0.0353
SIGN_X = np.array([-1.0, -1.0, 1.0, 1.0])
SIGN_Y = np.array([-1.0, 1.0, 1.0, -1.0])
SIGN_Z = np.array([1.0, -1.0, 1.0, -1.0])
MIX = np.stack([KF * np.ones(4), KF * ARM * SIGN_X, KF * ARM * SIGN_Y, KM * SIGN_Z])   # [T, tau] = MIX Omega^2
MIX_INVERSE = np.linalg.inv(MIX)
OMEGA_MAX = 2600.0                  # rad/s; the melon flights peak at 2490
ROTOR2_SCALE = 4.0 * KF / (MASS * GRAVITY)          # convert_idsia.ROTOR2_SCALE: u_i = 1 at hover


@dataclass(frozen=True)
class Gains:
    """Diagonal gains in SI units, set from a natural frequency w_n and damping ratio zeta per loop:
    position   K_p = m w_n^2, K_d = 2 zeta w_n m        (xy: w_n 4 rad/s, z: w_n 7 rad/s, zeta 0.9)
    attitude   K_R = J w_n^2, K_w = 2 zeta w_n J        (roll/pitch: w_n 25 rad/s, yaw: w_n 10 rad/s, zeta 0.8)
    """
    position_wn: tuple = (4.0, 4.0, 7.0)
    position_zeta: float = 0.9
    position_ki: tuple = (0.08, 0.08, 0.08)          # N / (m s), about the firmware's 0.05 scaled to 45 g
    position_i_limit: float = 2.0                     # m s
    attitude_wn: tuple = (25.0, 25.0, 10.0)
    attitude_zeta: float = 0.8
    attitude_ki_fraction: float = 0.05                # K_iR = fraction * K_R
    attitude_i_limit: float = 1.0                     # rad s

    def arrays(self) -> dict[str, jnp.ndarray]:
        wn_p, wn_r = np.asarray(self.position_wn), np.asarray(self.attitude_wn)
        k_r = INERTIA * wn_r**2
        return {"kp": jnp.asarray(MASS * wn_p**2), "kd": jnp.asarray(2 * self.position_zeta * wn_p * MASS),
                "ki": jnp.asarray(np.asarray(self.position_ki)), "kr": jnp.asarray(k_r),
                "kw": jnp.asarray(2 * self.attitude_zeta * wn_r * INERTIA),
                "kir": jnp.asarray(self.attitude_ki_fraction * k_r),
                "i_limit_p": self.position_i_limit, "i_limit_r": self.attitude_i_limit}


def _vee(matrix):
    return jnp.stack([matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]], axis=-1)


def _normalize(vector):
    return vector / jnp.maximum(jnp.linalg.norm(vector, axis=-1, keepdims=True), 1e-9)


def mellinger(state, integral_p, integral_r, reference, gains: dict, tick: float):
    """One controller tick for a batch.  state (B, >=18); reference entries (B, ...).  Returns
    (Omega^2 (B,4), desired wrench (B,4), new position integral, new attitude integral, saturated (B,))."""
    position = state[:, :3]
    rotation = state[:, 3:12].reshape(-1, 3, 3)
    velocity = jnp.einsum("bij,bj->bi", rotation, state[:, 12:15])
    omega = state[:, 15:18]

    error_p = reference["position"] - position
    integral_p = jnp.clip(integral_p + tick * error_p, -gains["i_limit_p"], gains["i_limit_p"])
    force = (gains["kp"] * error_p + gains["ki"] * integral_p
             + gains["kd"] * (reference["velocity"] - velocity)
             + MASS * (reference["acceleration"] + jnp.array([0.0, 0.0, GRAVITY])))
    thrust = jnp.einsum("bi,bi->b", force, rotation[:, :, 2])

    z_d = _normalize(force)
    yaw = reference["yaw"]
    x_c = jnp.stack([jnp.cos(yaw), jnp.sin(yaw), jnp.zeros_like(yaw)], axis=-1)
    y_d = _normalize(jnp.cross(z_d, x_c))
    x_d = jnp.cross(y_d, z_d)
    rotation_d = jnp.stack([x_d, y_d, z_d], axis=-1)                 # columns

    error_r = 0.5 * _vee(jnp.einsum("bji,bjk->bik", rotation_d, rotation)
                         - jnp.einsum("bji,bjk->bik", rotation, rotation_d))
    omega_d = jnp.stack([jnp.zeros_like(yaw), jnp.zeros_like(yaw), reference["yaw_rate"]], axis=-1)
    integral_r = jnp.clip(integral_r + tick * error_r, -gains["i_limit_r"], gains["i_limit_r"])
    torque = -gains["kr"] * error_r - gains["kw"] * (omega - omega_d) - gains["kir"] * integral_r

    wrench = jnp.concatenate([thrust[:, None], torque], axis=1)
    omega_squared_raw = wrench @ jnp.asarray(MIX_INVERSE).T
    omega_squared = jnp.clip(omega_squared_raw, 0.0, OMEGA_MAX**2)
    saturated = jnp.any(omega_squared != omega_squared_raw, axis=1)
    return omega_squared, wrench, integral_p, integral_r, saturated


def encode(omega_squared, input_mode: str):
    """Rotor speeds squared -> the control columns the plant was trained on (benchmark_protocol.controls)."""
    if input_mode == "rotor2":
        return omega_squared * ROTOR2_SCALE
    if input_mode == "wrench":
        return omega_squared @ jnp.asarray(MIX).T
    raise ValueError(f"unknown input mode {input_mode!r}")

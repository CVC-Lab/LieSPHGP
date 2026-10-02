"""Readable variational-GP port-Hamiltonian model on SE(3)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ._backend import (
    build_gp_setup,
    feature_normalization,
    initialize_variational_parameters,
    kl_divergence,
    physical_inverse_mass_targets,
    physics_from_settings,
    raw_output,
    sample_weights,
)


Array = jax.Array


def exact_gp_input(
    name: str, value: Array, gp_setup: Mapping[str, Any]
) -> Array:
    """Map quadrotor variables to the copied GP's 9D/12D convention."""
    if name == "sigma":
        return value                      # already the diffusion features (diffusion_features)
        return value
    if name == "Dw" and value.shape[-1] == 12:
        return value                      # already [vec(R), omega_b] (gp.dissipation_pose_input)
    if name in ("V", "g"):
        rotation = value[3:12]
        if name == "V" and "_potential_position_only" in gp_setup:
            rotation = jnp.eye(3, dtype=value.dtype).reshape(9)
        return jnp.concatenate([rotation, value[:3]])
    identity = jnp.eye(3, dtype=value.dtype).reshape(9)
    return jnp.concatenate([identity, value])


def _spd_from_levels(value: Array) -> Array:
    """Build L L^T from six numbers: exp() of the first three on the diagonal."""
    diagonal = jnp.exp(value[:3])
    lower = jnp.asarray(
        [
            [diagonal[0], 0.0, 0.0],
            [value[3], diagonal[1], 0.0],
            [value[4], value[5], diagonal[2]],
        ],
        dtype=value.dtype,
    )
    return lower @ lower.T


DIFFUSION_MODES = ("twist_control", "orientation", "full")


def diffusion_mode(gp_setup: Mapping[str, Any]) -> str | None:
    """Which state-dependent diffusion the setup was built with (static: key presence), or None."""
    return next((mode for mode in DIFFUSION_MODES if f"_diffusion_{mode}" in gp_setup), None)


def diffusion_features(state: Array, mode: str) -> Array:
    """(a) twist_control: body twist + rotor inputs; (b) orientation: vec(R); (c) full: x, vec(R), twist, u."""
    if mode == "twist_control":
        return jnp.concatenate([state[12:18], state[18:22]])
    if mode == "orientation":
        return state[3:12]
    return state[:22]


class SampledSE3HamODE(eqx.Module):
    """One coherent GP weight sample defining the complete SE(3) vector field."""

    weights: Mapping[str, Any]
    gp_setup: Mapping[str, Any]

    def inverse_mass_1(self, position: Array) -> Array:
        raw = raw_output("M1", self.weights, self.gp_setup, position).squeeze()
        mu = jnp.exp(self.weights["_M1_log_level"] + raw)
        return mu * jnp.eye(3, dtype=position.dtype)

    def inverse_mass_2(self, rotation_flat: Array) -> Array:
        raw = raw_output("M2", self.weights, self.gp_setup, rotation_flat)
        return _spd_from_levels(self.weights["_M2_level"] + raw)

    def _pose_input(self) -> bool:
        """gp.dissipation_pose_input: key presence is static under tracing, like _direct_control_map."""
        return "_dissipation_pose_input" in self.gp_setup

    def dissipation_v(self, velocity: Array, position: Array | None = None) -> Array:
        raw = raw_output("Dv", self.weights, self.gp_setup, velocity)
        enabled = self.gp_setup["_physics"]["dissipation_enabled"]
        matrix = _spd_from_levels(self.weights["_Dv_level"] + raw)
        if self._pose_input() and position is not None:
            # D_v(x, v_b) = exp(s_v(x)) * SPD(level + f_Dv(v_b)); the positive factor keeps it SPD.
            scale = jnp.exp(raw_output("Dv_pos", self.weights, self.gp_setup, position).squeeze())
            matrix = scale * matrix
        return enabled * matrix

    def dissipation_w(self, omega: Array, rotation_flat: Array | None = None) -> Array:
        value = omega
        if self._pose_input() and rotation_flat is not None:
            # D_w(R, omega_b): the real rotation replaces the identity in the nine rotation slots.
            value = jnp.concatenate([rotation_flat, omega])
        raw = raw_output("Dw", self.weights, self.gp_setup, value)
        enabled = self.gp_setup["_physics"]["dissipation_enabled"]
        return enabled * _spd_from_levels(self.weights["_Dw_level"] + raw)

    def potential(self, pose: Array) -> Array:
        level = self.weights["_V_level"]
        physics = self.gp_setup["_physics"]
        # Structured rigid-body potential: V(x) = g z / mu(x) with the learned position-dependent inverse
        # mass (grad V then carries the -g z grad(mu)/mu^2 term through autodiff). Missing key = off.
        structured = physics.get("structured_potential", jnp.asarray(0.0, dtype=level.dtype))
        gravity_value = physics.get("known_gravity", jnp.asarray(0.0, dtype=level.dtype))
        mu = self.inverse_mass_1(pose[:3])[0, 0]
        structured_value = gravity_value * pose[2] / mu
        learned_value = self._learned_potential(pose)
        return structured * structured_value + (1.0 - structured) * learned_value

    def _learned_potential(self, pose: Array) -> Array:
        level = self.weights["_V_level"]
        physics = self.gp_setup["_physics"]
        # Known-gravity prior: the z slope is g / mu_level so that mu * dV/dz = g at the level
        # (gauge-consistent); off: fully learned slope. Missing keys (older setups) mean off.
        enabled = physics.get("known_gravity_enabled", jnp.asarray(0.0, dtype=level.dtype))
        gravity = physics.get("known_gravity", jnp.asarray(0.0, dtype=level.dtype))
        prior_slope = gravity * jnp.exp(-self.weights["_M1_log_level"])
        z_slope = enabled * prior_slope + (1.0 - enabled) * level[2]
        linear = level[0] * pose[0] + level[1] * pose[1] + z_slope * pose[2]
        return linear + raw_output("V", self.weights, self.gp_setup, pose).squeeze()

    def twist_diffusion_blocks(self, state: Array) -> tuple[Array, Array]:
        """(3,3) linear and angular blocks A_v, A_w of the twist diffusion, so d xi = f dt + blockdiag(A_v, A_w) dW:
        diag(sigma) for the constant model, M^-1(x) diag(sigma(x)) for noise in momentum."""
        if diffusion_mode(self.gp_setup) is not None and "sigma" in self.weights:
            scale = self.diffusion_scale(state)
            return (self.inverse_mass_1(state[:3]) @ jnp.diag(scale[:3]),
                    self.inverse_mass_2(state[3:12]) @ jnp.diag(scale[3:]))
        sigma = self.process_sigma()
        return jnp.diag(sigma[:3]), jnp.diag(sigma[3:])

    def process_sigma(self) -> Array:
        """Diagonal of the diffusion on the body twist, (6,): [sigma_v x3, sigma_omega x3]; zero for ODE weights."""
        log_sigma = self.weights.get("_process_log_sigma")
        if log_sigma is None:
            return jnp.zeros((6,), dtype=self.weights["_g_level"].dtype)
        return jnp.repeat(jnp.exp(log_sigma), 3)

    def diffusion_scale(self, state: Array) -> Array:
        """(6,) scales. Constant SDE: twist units [sigma_v x3, sigma_omega x3]. State-dependent SDE: momentum
        units, a body force (3) and torque (3) scale sigma(x) = softplus(level + GP(x))."""
        mode = diffusion_mode(self.gp_setup)
        if mode is None or "sigma" not in self.weights:
            return self.process_sigma()
        raw = raw_output("sigma", self.weights, self.gp_setup, diffusion_features(state, mode))
        return jax.nn.softplus(self.weights["_sigma_level"] + raw)

    def twist_noise(self, state: Array, noise: Array) -> Array:
        """(6,) twist increment per unit sqrt(dt): Sigma z (constant) or M^-1(x) Sigma(x) z (noise in momentum)."""
        if diffusion_mode(self.gp_setup) is None or "sigma" not in self.weights:
            return self.process_sigma() * noise
        scale = self.diffusion_scale(state)
        force = self.inverse_mass_1(state[:3]) @ (scale[:3] * noise[:3])
        torque = self.inverse_mass_2(state[3:12]) @ (scale[3:] * noise[3:])
        return jnp.concatenate([force, torque])

    def control_matrix(self, pose: Array) -> Array:
        learned = raw_output("g", self.weights, self.gp_setup, pose).reshape(6, 4)
        if "_direct_control_map" in self.gp_setup:
            return learned
        return self.weights["_g_level"] + learned

    def hamiltonian(self, pose_momenta: Array) -> Array:
        position = pose_momenta[:3]
        rotation_flat = pose_momenta[3:12]
        momentum_v = pose_momenta[12:15]
        momentum_w = pose_momenta[15:18]
        return (
            0.5 * momentum_v @ self.inverse_mass_1(position) @ momentum_v
            + 0.5
            * momentum_w
            @ self.inverse_mass_2(rotation_flat)
            @ momentum_w
            + self.potential(pose_momenta[:12])
        )

    def vector_field(self, state: Array) -> Array:
        pose = state[:12]
        position = pose[:3]
        rotation_flat = pose[3:12]
        velocity = state[12:15]
        omega = state[15:18]
        control = state[18:22]

        mass_inv_v = self.inverse_mass_1(position)
        mass_inv_w = self.inverse_mass_2(rotation_flat)
        momentum_v = jnp.linalg.solve(mass_inv_v, velocity)
        momentum_w = jnp.linalg.solve(mass_inv_w, omega)
        pose_momenta = jnp.concatenate([pose, momentum_v, momentum_w])
        gradient = jax.grad(self.hamiltonian)(pose_momenta)
        d_h_dx, d_h_dr = gradient[:3], gradient[3:12]
        d_h_dpv, d_h_dpw = gradient[12:15], gradient[15:18]

        rotation = rotation_flat.reshape(3, 3)
        dx = rotation @ d_h_dpv
        dr = jnp.cross(rotation, d_h_dpw[None, :], axis=-1).reshape(9)
        force_torque = self.control_matrix(pose) @ control
        dpv = (
            jnp.cross(momentum_v, d_h_dpw)
            - rotation.T @ d_h_dx
            - self.dissipation_v(velocity, position) @ d_h_dpv
            + force_torque[:3]
        )
        dpw = (
            jnp.cross(momentum_w, d_h_dpw)
            + jnp.cross(momentum_v, d_h_dpv)
            + jnp.cross(rotation, d_h_dr.reshape(3, 3), axis=-1).sum(axis=0)
            - self.dissipation_w(omega, rotation_flat) @ d_h_dpw
            + force_torque[3:6]
        )

        dmass_inv_v = jax.jvp(
            self.inverse_mass_1, (position,), (dx,)
        )[1]
        dmass_inv_w = jax.jvp(
            self.inverse_mass_2, (rotation_flat,), (dr,)
        )[1]
        dv = mass_inv_v @ dpv + dmass_inv_v @ momentum_v
        dw = mass_inv_w @ dpw + dmass_inv_w @ momentum_w
        return jnp.concatenate(
            [dx, dr, dv, dw, jnp.zeros(4, dtype=state.dtype)]
        )

    def effective_damping(self, state: Array) -> tuple[Array, Array]:
        return (
            self.inverse_mass_1(state[:3])
            @ self.dissipation_v(state[12:15], state[:3]),
            self.inverse_mass_2(state[3:12])
            @ self.dissipation_w(state[15:18], state[3:12]),
        )


class DissipativeSE3HamODE(eqx.Module):
    """Variational model that produces posterior-mean or sampled dynamics."""

    variational: Mapping[str, Any]
    gp_setup: Mapping[str, Any]

    def sample(self, key: Array | None = None) -> SampledSE3HamODE:
        return SampledSE3HamODE(
            sample_weights(self.variational, key),
            self.gp_setup,
        )

    def kl_loss(self) -> Array:
        return kl_divergence(self.variational)


# Compatibility wrappers for checkpoints and callers using the former API.
def inverse_mass_1(weights, gp_setup, position: Array) -> Array:
    return SampledSE3HamODE(weights, gp_setup).inverse_mass_1(position)


def inverse_mass_2(weights, gp_setup, rotation_flat: Array) -> Array:
    return SampledSE3HamODE(weights, gp_setup).inverse_mass_2(rotation_flat)


def vector_field_single(weights, gp_setup, state: Array) -> Array:
    return SampledSE3HamODE(weights, gp_setup).vector_field(state)


def effective_damping_single(weights, gp_setup, state: Array) -> tuple[Array, Array]:
    return SampledSE3HamODE(weights, gp_setup).effective_damping(state)


vector_field = jax.vmap(vector_field_single, in_axes=(None, None, 0))
effective_damping_matrices = jax.vmap(
    effective_damping_single, in_axes=(None, None, 0)
)

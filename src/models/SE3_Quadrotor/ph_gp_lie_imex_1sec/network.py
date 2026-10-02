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
    if name == "M2":
        return value
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

    def dissipation_v(self, velocity: Array) -> Array:
        raw = raw_output("Dv", self.weights, self.gp_setup, velocity)
        enabled = self.gp_setup["_physics"]["dissipation_enabled"]
        return enabled * _spd_from_levels(self.weights["_Dv_level"] + raw)

    def dissipation_w(self, omega: Array) -> Array:
        raw = raw_output("Dw", self.weights, self.gp_setup, omega)
        enabled = self.gp_setup["_physics"]["dissipation_enabled"]
        return enabled * _spd_from_levels(self.weights["_Dw_level"] + raw)

    def potential(self, pose: Array) -> Array:
        linear = jnp.dot(self.weights["_V_level"], pose[:3])
        return linear + raw_output("V", self.weights, self.gp_setup, pose).squeeze()

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
            - self.dissipation_v(velocity) @ d_h_dpv
            + force_torque[:3]
        )
        dpw = (
            jnp.cross(momentum_w, d_h_dpw)
            + jnp.cross(momentum_v, d_h_dpv)
            + jnp.cross(rotation, d_h_dr.reshape(3, 3), axis=-1).sum(axis=0)
            - self.dissipation_w(omega) @ d_h_dpw
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
            @ self.dissipation_v(state[12:15]),
            self.inverse_mass_2(state[3:12])
            @ self.dissipation_w(state[15:18]),
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

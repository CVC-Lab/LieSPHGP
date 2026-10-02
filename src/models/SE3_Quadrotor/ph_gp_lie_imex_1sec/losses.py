"""Trajectory and observation losses for PH-GP-LieIMEX."""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from src.utils.JAX.loss_utils_jax import (
    compute_geodesic_distance_from_two_matrices_safe,
)

Array = jax.Array
ACOS_EPS = 1.0e-6


def _normalize(vector: Array) -> Array:
    magnitude = jnp.maximum(jnp.linalg.norm(vector, axis=-1, keepdims=True), 1.0e-8)
    return vector / magnitude


def project_rotation(rotation_flat: Array) -> Array:
    x_raw = rotation_flat[..., :3]
    y_raw = rotation_flat[..., 3:6]
    x_axis = _normalize(x_raw)
    z_axis = _normalize(jnp.cross(x_axis, y_raw, axis=-1))
    y_axis = jnp.cross(z_axis, x_axis, axis=-1)
    return jnp.stack([x_axis, y_axis, z_axis], axis=-2)


def pose_loss_components(reference: Array, prediction: Array) -> Array:
    """Return ``[total, x, v, omega, geodesic]`` for time-major arrays."""
    position_loss = jnp.mean(jnp.square(reference[..., :3] - prediction[..., :3]))
    velocity_loss = jnp.mean(jnp.square(reference[..., 12:15] - prediction[..., 12:15]))
    omega_loss = jnp.mean(jnp.square(reference[..., 15:18] - prediction[..., 15:18]))

    reference_rotation = project_rotation(reference[..., 3:12])
    predicted_rotation = project_rotation(prediction[..., 3:12])
    relative = reference_rotation @ jnp.swapaxes(predicted_rotation, -1, -2)
    cosine = 0.5 * (jnp.trace(relative, axis1=-2, axis2=-1) - 1.0)
    cosine = jnp.clip(cosine, -1.0 + ACOS_EPS, 1.0 - ACOS_EPS)
    geodesic_loss = jnp.mean(jnp.square(jnp.arccos(cosine)))
    total = position_loss + velocity_loss + omega_loss + geodesic_loss
    return jnp.stack(
        [total, position_loss, velocity_loss, omega_loss, geodesic_loss]
    )


LOG_2PI = math.log(2.0 * math.pi)
LIKELIHOOD_NAMES = (
    "nll_total",
    "nll_position",
    "nll_attitude",
    "nll_linear_velocity",
    "nll_angular_velocity",
    "position_squared_norm",
    "attitude_angle_squared",
    "linear_velocity_squared_norm",
    "angular_velocity_squared_norm",
    "sigma_position",
    "sigma_attitude",
    "sigma_linear_velocity",
    "sigma_angular_velocity",
)


def initialize_likelihood_parameters(initial_sigma: float = 0.05) -> dict[str, Array]:
    """Create the four trainable observation-noise parameters."""
    if initial_sigma <= 0.0:
        raise ValueError("initial observation sigma must be positive")
    value = jnp.log(jnp.asarray(initial_sigma, dtype=jnp.float32))
    return {
        "log_sigma_position": value,
        "log_sigma_attitude": value,
        "log_sigma_linear_velocity": value,
        "log_sigma_angular_velocity": value,
    }


def gaussian_nll_vector(
    observed: Array,
    predicted: Array,
    log_sigma: Array,
    squared_error_scale: float = 1.0,
) -> tuple[Array, Array]:
    """Mean NLL and mean squared norm for one three-vector observation."""
    residual = observed - predicted
    mean_squared_norm = jnp.mean(jnp.sum(jnp.square(residual), axis=-1))
    variance = jnp.exp(2.0 * log_sigma)
    nll = (
        0.5 * squared_error_scale * mean_squared_norm / variance
        + 3.0 * log_sigma
        + 1.5 * LOG_2PI
    )
    return nll, mean_squared_norm


def gaussian_nll_attitude(
    observed_flat: Array,
    predicted_flat: Array,
    log_sigma: Array,
    squared_error_scale: float = 1.0,
) -> tuple[Array, Array]:
    """Mean concentrated-Gaussian NLL on the three-dimensional SO(3) manifold."""
    observed = observed_flat.reshape(-1, 3, 3)
    predicted = predicted_flat.reshape(-1, 3, 3)
    angle = compute_geodesic_distance_from_two_matrices_safe(observed, predicted)
    mean_angle_squared = jnp.mean(jnp.square(angle))
    variance = jnp.exp(2.0 * log_sigma)
    nll = (
        0.5 * squared_error_scale * mean_angle_squared / variance
        + 3.0 * log_sigma
        + 1.5 * LOG_2PI
    )
    return nll, mean_angle_squared


def se3_elbo_nll(
    observed: Array,
    predicted: Array,
    likelihood: Mapping[str, Array],
    squared_error_scale: float = 1.0,
) -> dict[str, Array]:
    """Return the SE(3) rollout NLL; state controls in channels 18:22 are ignored."""
    nll_position, position_squared_norm = gaussian_nll_vector(
        observed[..., :3],
        predicted[..., :3],
        likelihood["log_sigma_position"],
        squared_error_scale,
    )
    nll_attitude, attitude_angle_squared = gaussian_nll_attitude(
        observed[..., 3:12],
        predicted[..., 3:12],
        likelihood["log_sigma_attitude"],
        squared_error_scale,
    )
    nll_linear_velocity, linear_velocity_squared_norm = gaussian_nll_vector(
        observed[..., 12:15],
        predicted[..., 12:15],
        likelihood["log_sigma_linear_velocity"],
        squared_error_scale,
    )
    nll_angular_velocity, angular_velocity_squared_norm = gaussian_nll_vector(
        observed[..., 15:18],
        predicted[..., 15:18],
        likelihood["log_sigma_angular_velocity"],
        squared_error_scale,
    )
    return {
        "nll_total": (
            nll_position
            + nll_attitude
            + nll_linear_velocity
            + nll_angular_velocity
        ),
        "nll_position": nll_position,
        "nll_attitude": nll_attitude,
        "nll_linear_velocity": nll_linear_velocity,
        "nll_angular_velocity": nll_angular_velocity,
        "position_squared_norm": position_squared_norm,
        "attitude_angle_squared": attitude_angle_squared,
        "linear_velocity_squared_norm": linear_velocity_squared_norm,
        "angular_velocity_squared_norm": angular_velocity_squared_norm,
        "sigma_position": jnp.exp(likelihood["log_sigma_position"]),
        "sigma_attitude": jnp.exp(likelihood["log_sigma_attitude"]),
        "sigma_linear_velocity": jnp.exp(
            likelihood["log_sigma_linear_velocity"]
        ),
        "sigma_angular_velocity": jnp.exp(
            likelihood["log_sigma_angular_velocity"]
        ),
    }


def likelihood_vector(values: Mapping[str, Array]) -> Array:
    """Pack a likelihood dictionary into a stable logging order."""
    return jnp.stack([values[name] for name in LIKELIHOOD_NAMES])


def per_window_se3_nll(
    observed: Array,
    predicted: Array,
    likelihood: Mapping[str, Array],
    squared_error_scale: float = 1.0,
) -> Array:
    """Total SE(3) NLL of every window separately; arrays are time-major (T, B, 22)."""

    def one(observed_window: Array, predicted_window: Array) -> Array:
        return se3_elbo_nll(observed_window, predicted_window, likelihood, squared_error_scale)["nll_total"]

    return jax.vmap(one, in_axes=(1, 1))(observed, predicted)

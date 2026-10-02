"""JAX equivalent of the comparison's SE(3) pose/geodesic training loss."""

from __future__ import annotations

import jax
import jax.numpy as jnp


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


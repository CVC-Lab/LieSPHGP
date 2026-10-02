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


def window_health(prediction: Array) -> Array:
    """Per-window finite flag for a time-major ``(time, window, 22)`` rollout.

    A single diverged window makes the plain mean in :func:`pose_loss_components` NaN, which aborts training
    at step 0 on real data (1728 IDSIA test windows, at least one of which leaves the chart at random init).
    Mirrors ``ph_gp_lie_imex_idsia.train`` evaluate_long, which masks the same way.
    """
    return jnp.all(jnp.isfinite(prediction[..., :18]), axis=(0, -1))


def pose_loss_components_masked(reference: Array, prediction: Array, healthy: Array) -> tuple[Array, Array]:
    """``pose_loss_components`` averaged over healthy windows only, plus the masked fraction.

    ``healthy`` is the (window,) flag from :func:`window_health`. Non-finite entries are replaced by zero
    *before* the reduction, so no NaN reaches the average through ``0 * inf``; the divisor counts only the
    healthy windows. With every window healthy this returns exactly what the unmasked function returns.
    """
    weight = healthy.astype(reference.dtype)
    denominator = jnp.maximum(weight.sum(), 1.0)

    def masked_mean(values: Array) -> Array:
        # values: (time, window[, channel]) -> mean over time/channels per window, then over healthy windows
        per_window = jnp.mean(values, axis=tuple(a for a in range(values.ndim) if a != 1))
        per_window = jnp.where(healthy, per_window, 0.0)
        return jnp.sum(per_window * weight) / denominator

    safe = jnp.where(jnp.isfinite(prediction), prediction, 0.0)
    position_loss = masked_mean(jnp.square(reference[..., :3] - safe[..., :3]))
    velocity_loss = masked_mean(jnp.square(reference[..., 12:15] - safe[..., 12:15]))
    omega_loss = masked_mean(jnp.square(reference[..., 15:18] - safe[..., 15:18]))

    reference_rotation = project_rotation(reference[..., 3:12])
    predicted_rotation = project_rotation(safe[..., 3:12])
    relative = reference_rotation @ jnp.swapaxes(predicted_rotation, -1, -2)
    cosine = 0.5 * (jnp.trace(relative, axis1=-2, axis2=-1) - 1.0)
    cosine = jnp.clip(cosine, -1.0 + ACOS_EPS, 1.0 - ACOS_EPS)
    geodesic_loss = masked_mean(jnp.square(jnp.arccos(cosine)))
    total = position_loss + velocity_loss + omega_loss + geodesic_loss
    masked_fraction = 1.0 - jnp.mean(healthy.astype(jnp.zeros(()).dtype))
    return jnp.stack([total, position_loss, velocity_loss, omega_loss, geodesic_loss]), masked_fraction


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


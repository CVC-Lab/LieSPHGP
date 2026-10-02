"""Trajectory and observation losses for PH-GP-LieIMEX (IDSIA variant).

Differences from the original ``ph_gp_lie_imex`` package, both off by default so an unchanged config
reproduces the original behaviour exactly:

1. **Robust likelihood** (``gp.robust_likelihood_dof``).  The Gaussian negative log-likelihood is linear in the
   squared residual, so a single diverging window dominates the batch mean and its gradient.  Both real-flight
   failures were exactly that: a finite objective with ``gradient_norm = inf``, at step 129 in float32 and step
   434 in float64, so widening the floats only delayed it.  With ``dof = nu`` the per-sample likelihood becomes
   a multivariate Student-t with isotropic scale,

       -log p(r) = (nu + d)/2 * log(1 + |r|^2 / (nu sigma^2)) + d log sigma + const,   d = 3,

   whose influence function is bounded: an outlier contributes a *logarithm* of its squared error instead of
   the error itself.  Note this requires the per-sample NLL to be averaged, not the NLL of the average squared
   error, which is why the helpers below return element-wise arrays.  The two agree exactly for a Gaussian.

2. **Horizon weighting** (``training.loss_horizon_weight``).  Transition h of a window is weighted by
   exp(-lambda (h - 1)), normalised to mean one so the loss scale is unchanged.  With lambda = 0 every
   transition is weighted equally, as before.  The IDSIA benchmark uses the same idea with lambda = 0.1.

3. **Observation-scale floor** (``gp.min_observation_sigma``).  Measured on IDSIA, the real driver of the
   divergence: the four learned scales fall monotonically (position 0.300 -> 0.044 -> 0.032 -> 0.027) while the
   diverged-window mask never fires once.  Since the NLL weights each residual by 1/(2 sigma^2), the loss
   surface stiffens without bound.  A floor stops the progression, and it addresses the badly over-confident
   posterior (coverage 0.23 at 2 sigma) for the same reason.

4. **Diverged-window masking** (``training.diverged_window_position_limit``).  A window whose rolled-out
   position leaves that many metres of the measurement, or goes non-finite, is dropped from the batch.  It
   carries no information, only noise.  Masking is done with ``jnp.where`` rather than multiplication, because
   the VJP of a select returns an exact zero, whereas ``0 * inf`` is NaN.
"""

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
BLOCK_DIMENSION = 3.0          # every observation block (position, attitude, v, omega) is three-dimensional


def _normalize(vector: Array) -> Array:
    magnitude = jnp.maximum(jnp.linalg.norm(vector, axis=-1, keepdims=True), 1.0e-8)
    return vector / magnitude


def project_rotation(rotation_flat: Array) -> Array:
    """Row Gram-Schmidt onto SO(3), the same rule the attitude loss applies."""
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
    "masked_window_fraction",
)


def initialize_likelihood_parameters(initial_sigma: float = 0.05) -> dict[str, Array]:
    """Create the four trainable observation-noise parameters."""
    if initial_sigma <= 0.0:
        raise ValueError("initial observation sigma must be positive")
    value = jnp.log(jnp.asarray(initial_sigma, dtype=jnp.zeros(()).dtype))
    return {
        "log_sigma_position": value,
        "log_sigma_attitude": value,
        "log_sigma_linear_velocity": value,
        "log_sigma_angular_velocity": value,
    }


# ---------------------------------------------------------------------------
# Aggregation over the (transition, window) grid
# ---------------------------------------------------------------------------
def horizon_weights(transitions: int, decay: float, dtype) -> Array:
    """exp(-decay * (h - 1)) for h = 1..K, normalised to mean one. decay = 0 gives uniform weights."""
    index = jnp.arange(transitions, dtype=dtype)
    if decay <= 0.0:
        return jnp.ones_like(index)
    weight = jnp.exp(-jnp.asarray(decay, dtype=dtype) * index)
    return weight / jnp.mean(weight)


def window_mask(observed: Array, predicted: Array, limit: float | None) -> Array:
    """True for windows worth learning from: finite, and never more than ``limit`` metres off in position."""
    finite = jnp.all(jnp.isfinite(predicted), axis=(0, 2))
    if limit is None:
        return jax.lax.stop_gradient(finite)
    deviation = jnp.max(jnp.linalg.norm(
        jnp.nan_to_num(predicted[..., :3] - observed[..., :3], nan=jnp.inf), axis=-1), axis=0)
    return jax.lax.stop_gradient(finite & (deviation <= limit))


def _aggregate(values: Array, weights: Array, mask: Array) -> Array:
    """Weighted mean of ``values`` (transitions, windows) over the unmasked windows.

    ``jnp.where`` is used rather than a multiply: the VJP of a select is an exact zero, so a window whose
    rollout produced inf contributes neither a value nor a gradient, while ``0 * inf`` would be NaN.
    """
    keep = jnp.broadcast_to(mask[None, :], values.shape)
    selected = jnp.where(keep, values, jnp.zeros_like(values))
    weight_grid = jnp.where(keep, jnp.broadcast_to(weights[:, None], values.shape),
                            jnp.zeros_like(values))
    total = jnp.sum(weight_grid)
    return jnp.sum(selected * weight_grid) / jnp.maximum(total, jnp.asarray(1.0e-12, values.dtype))


def _block_nll(squared: Array, log_sigma: Array, squared_error_scale: float, dof: float | None) -> Array:
    """Element-wise negative log-likelihood of a three-dimensional block from its squared residual."""
    variance = jnp.exp(2.0 * log_sigma)
    if dof is None:
        return 0.5 * squared_error_scale * squared / variance + 3.0 * log_sigma + 1.5 * LOG_2PI
    nu = jnp.asarray(dof, dtype=squared.dtype)
    normaliser = (
        jax.scipy.special.gammaln(0.5 * (nu + BLOCK_DIMENSION))
        - jax.scipy.special.gammaln(0.5 * nu)
        - 0.5 * BLOCK_DIMENSION * jnp.log(nu * jnp.pi)
    )
    quadratic = squared_error_scale * squared / (nu * variance)
    return (0.5 * (nu + BLOCK_DIMENSION) * jnp.log1p(quadratic)
            + 3.0 * log_sigma - normaliser)


def _vector_squared(observed: Array, predicted: Array) -> Array:
    return jnp.sum(jnp.square(observed - predicted), axis=-1)


def _attitude_squared(observed_flat: Array, predicted_flat: Array) -> Array:
    shape = observed_flat.shape[:-1]
    angle = compute_geodesic_distance_from_two_matrices_safe(
        observed_flat.reshape(-1, 3, 3), predicted_flat.reshape(-1, 3, 3))
    return jnp.square(angle).reshape(shape)


def se3_elbo_nll(
    observed: Array,
    predicted: Array,
    likelihood: Mapping[str, Array],
    squared_error_scale: float = 1.0,
    *,
    decay: float = 0.0,
    position_limit: float | None = None,
    dof: float | None = None,
    minimum_sigma: float | None = None,
) -> dict[str, Array]:
    """The SE(3) rollout NLL over time-major (transitions, windows, 22) arrays; controls are ignored."""
    mask = window_mask(observed, predicted, position_limit)
    weights = horizon_weights(observed.shape[0], decay, observed.dtype)
    # Cut the gradient of a masked window at the prediction itself. Zeroing only the loss is not enough:
    # the reverse pass would still form 0 * d(squared)/d(pred) = 0 * inf = NaN. A select discards it.
    # Masked windows get a finite, gradient-free stand-in (the observation itself): with the diverged
    # prediction left in place, the local derivatives of squared / sigma^2 would be inf and 0 * inf = NaN
    # would reach the observation scales even though the select zeroes the value.
    predicted = jnp.where(mask[None, :, None], predicted, jax.lax.stop_gradient(observed))
    blocks = {
        "position": (_vector_squared(observed[..., :3], predicted[..., :3]), "log_sigma_position"),
        "attitude": (_attitude_squared(observed[..., 3:12], predicted[..., 3:12]), "log_sigma_attitude"),
        "linear_velocity": (_vector_squared(observed[..., 12:15], predicted[..., 12:15]),
                            "log_sigma_linear_velocity"),
        "angular_velocity": (_vector_squared(observed[..., 15:18], predicted[..., 15:18]),
                             "log_sigma_angular_velocity"),
    }
    result: dict[str, Array] = {}
    total = jnp.zeros((), dtype=observed.dtype)
    floor = None if minimum_sigma is None else jnp.log(jnp.asarray(minimum_sigma, dtype=observed.dtype))
    for name, (squared, sigma_key) in blocks.items():
        log_sigma = likelihood[sigma_key]
        if floor is not None:
            # The learned scales collapse on real data (0.300 -> 0.027 on IDSIA), and the NLL weights every
            # residual by 1/(2 sigma^2), so the loss surface stiffens without bound until a gradient overflows
            # even in float64. A floor stops that progression and also stops the posterior band from
            # collapsing, which is the same root cause as the 0.23 coverage at 2 sigma.
            log_sigma = jnp.maximum(log_sigma, floor)
        elementwise = _block_nll(squared, log_sigma, squared_error_scale, dof)
        value = _aggregate(elementwise, weights, mask)
        result[f"nll_{name}"] = value
        total = total + value
        key = "attitude_angle_squared" if name == "attitude" else f"{name}_squared_norm"
        result[key] = _aggregate(squared, jnp.ones_like(weights), mask)
    result["nll_total"] = total
    result["masked_window_fraction"] = 1.0 - jnp.mean(mask.astype(observed.dtype))
    for name in ("position", "attitude", "linear_velocity", "angular_velocity"):
        value = likelihood[f"log_sigma_{name}"]
        result[f"sigma_{name}"] = jnp.exp(value if floor is None else jnp.maximum(value, floor))
    return result


def likelihood_vector(values: Mapping[str, Array]) -> Array:
    """Pack a likelihood dictionary into a stable logging order."""
    return jnp.stack([values[name] for name in LIKELIHOOD_NAMES])

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
import jax.scipy.linalg

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
    # SDE variant only (zero otherwise): learned process-noise scales and 2-sigma coverage of the
    # moment-matched predictive band on the training/eval windows
    "process_sigma_linear",
    "process_sigma_angular",
    "coverage_position_2sigma",
    "coverage_attitude_2sigma",
    "coverage_linear_velocity_2sigma",
    "coverage_angular_velocity_2sigma",
    # horizon-growing observation variance (SDE variant, optional): var_b(h) = sigma_b^2 + gamma_b * t_h
    "gamma_position",
    "gamma_attitude",
    "gamma_linear_velocity",
    "gamma_angular_velocity",
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
    predicted = jnp.where(mask[None, :, None], predicted, jax.lax.stop_gradient(predicted))
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
    """Pack a likelihood dictionary into a stable logging order; names a variant does not produce read as 0."""
    reference = values["nll_total"]
    return jnp.stack([jnp.asarray(values.get(name, 0.0), dtype=reference.dtype) for name in LIKELIHOOD_NAMES])


# ---------------------------------------------------------------------------
# SDE variant: moment-matched predictive likelihood over S sample paths
# ---------------------------------------------------------------------------
def log_so3(rotation: Array, epsilon: float = 1.0e-12) -> Array:
    """Rotation vector of R in SO(3), (..., 3); smooth at the identity (no acos, norm regularised)."""
    skew = 0.5 * (rotation - jnp.swapaxes(rotation, -1, -2))
    vee = jnp.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], axis=-1)      # = sin(theta) * axis
    sine = jnp.sqrt(jnp.sum(vee * vee, axis=-1, keepdims=True) + epsilon)
    cosine = 0.5 * (jnp.trace(rotation, axis1=-2, axis2=-1) - 1.0)[..., None]
    theta = jnp.arctan2(sine, cosine)
    return vee * theta / sine


def _diag_gaussian_nll(residual: Array, variance: Array) -> Array:
    """Element-wise -log N(residual; 0, diag(variance)) summed over the last axis (3 components)."""
    return 0.5 * jnp.sum(residual * residual / variance + jnp.log(variance) + LOG_2PI, axis=-1)


def _coverage(residual: Array, variance: Array, weights: Array, mask: Array) -> Array:
    inside = jnp.all(jnp.abs(residual) <= 2.0 * jnp.sqrt(variance), axis=-1).astype(residual.dtype)
    return _aggregate(inside, jnp.ones_like(weights), mask)


def se3_moment_nll(
    observed: Array,
    samples: Array,
    likelihood: Mapping[str, Array],
    process_log_sigma: Array,
    *,
    decay: float = 0.0,
    position_limit: float | None = None,
    minimum_sigma: float | None = None,
    variance_floor: float = 1.0e-10,
    horizon_seconds: Array | None = None,
) -> dict[str, Array]:
    """Moment-matched Gaussian NLL of the observations under S stochastic rollouts.

    Optional horizon-growing observation variance: when ``likelihood`` carries ``log_gamma_<block>`` and
    ``horizon_seconds`` (transitions,) is given, sigma_obs,b^2 becomes sigma_obs,b^2 + gamma_b * t_h. This is the
    observation-model analogue of process noise (variance linear in the horizon) and gives the band a way to
    widen with h that does not depend on the process scales, which shrink during training.

    ``observed`` is time-major (transitions, windows, 22); ``samples`` is (S, transitions, windows, 22), every
    path with its own Brownian increments and weight sample. Per block b and horizon h the predictive law is
    N(mu_b,h, diag(var_b,h) + sigma_obs,b^2 I), mu the sample mean (chordal mean projected to SO(3) for the
    attitude) and var the unbiased sample variance (in the tangent space at mu for the attitude). Unlike a
    per-sample NLL, the sample spread enters the density, so the process-noise scales receive a data-fit
    signal: the band must be as wide as the residuals, no wider.
    """
    sample_count = samples.shape[0]
    mean_state = jnp.mean(samples, axis=0)
    mean_rotation = project_rotation(mean_state[..., 3:12])                       # (T, B, 3, 3)
    weights = horizon_weights(observed.shape[0], decay, observed.dtype)
    mask = window_mask(observed, mean_state, position_limit) & jnp.all(jnp.isfinite(samples), axis=(0, 1, 3))
    mask = jax.lax.stop_gradient(mask)
    # A masked window (a diverged or non-finite sample path) must not leave NaN/inf in ANY intermediate:
    # the select in _aggregate zeroes its value and cotangent, but the local derivatives of the terms
    # between the samples and that select (residual^2 / variance, log variance) would still be NaN, and
    # 0 * NaN = NaN in the gradient of the observation and process scales (measured: step 176 of the first
    # SDE run). So the masked windows' samples are replaced by a finite, gradient-free stand-in.
    stand_in = jax.lax.stop_gradient(jnp.broadcast_to(observed[None], samples.shape))
    samples = jnp.where(mask[None, None, :, None], samples, stand_in)
    mean_state = jnp.mean(samples, axis=0)
    mean_rotation = project_rotation(mean_state[..., 3:12])

    floor = None if minimum_sigma is None else jnp.log(jnp.asarray(minimum_sigma, dtype=observed.dtype))
    correction = sample_count / max(sample_count - 1, 1)
    result: dict[str, Array] = {}
    total = jnp.zeros((), dtype=observed.dtype)

    def growth(name):
        key_name = f"log_gamma_{name}"
        if horizon_seconds is None or key_name not in likelihood:
            return 0.0
        return (jnp.exp(likelihood[key_name]) * horizon_seconds)[:, None, None]                          # (T, 1, 1)

    def vector_block(name, columns, sigma_key):
        residual = observed[..., columns] - mean_state[..., columns]                                    # (T, B, 3)
        spread = correction * jnp.mean(jnp.square(samples[..., columns] - mean_state[None, ..., columns]), axis=0)
        log_sigma = likelihood[sigma_key] if floor is None else jnp.maximum(likelihood[sigma_key], floor)
        variance = spread + jnp.exp(2.0 * log_sigma) + growth(name) + variance_floor
        result[f"nll_{name}"] = _aggregate(_diag_gaussian_nll(residual, variance), weights, mask)
        result[f"{name}_squared_norm"] = _aggregate(jnp.sum(residual * residual, axis=-1), jnp.ones_like(weights), mask)
        result[f"coverage_{name}_2sigma"] = _coverage(residual, variance, weights, mask)
        result[f"sigma_{name}"] = jnp.exp(log_sigma)

    vector_block("position", slice(0, 3), "log_sigma_position")
    # attitude: residual and spread in the tangent space at the mean rotation
    observed_rotation = observed[..., 3:12].reshape(*observed.shape[:-1], 3, 3)
    sample_rotation = samples[..., 3:12].reshape(*samples.shape[:-1], 3, 3)
    mean_transpose = jnp.swapaxes(mean_rotation, -1, -2)
    residual = log_so3(mean_transpose @ observed_rotation)                                              # (T, B, 3)
    tangents = log_so3(mean_transpose[None] @ sample_rotation)                                           # (S, T, B, 3)
    spread = correction * jnp.mean(jnp.square(tangents), axis=0)
    log_sigma = likelihood["log_sigma_attitude"] if floor is None else jnp.maximum(likelihood["log_sigma_attitude"], floor)
    variance = spread + jnp.exp(2.0 * log_sigma) + growth("attitude") + variance_floor
    result["nll_attitude"] = _aggregate(_diag_gaussian_nll(residual, variance), weights, mask)
    result["attitude_angle_squared"] = _aggregate(jnp.sum(residual * residual, axis=-1), jnp.ones_like(weights), mask)
    result["coverage_attitude_2sigma"] = _coverage(residual, variance, weights, mask)
    result["sigma_attitude"] = jnp.exp(log_sigma)
    vector_block("linear_velocity", slice(12, 15), "log_sigma_linear_velocity")
    vector_block("angular_velocity", slice(15, 18), "log_sigma_angular_velocity")

    for name in ("position", "attitude", "linear_velocity", "angular_velocity"):
        total = total + result[f"nll_{name}"]
    result["nll_total"] = total
    result["masked_window_fraction"] = 1.0 - jnp.mean(mask.astype(observed.dtype))
    result["process_sigma_linear"] = jnp.exp(process_log_sigma[0])
    result["process_sigma_angular"] = jnp.exp(process_log_sigma[1])
    for name in ("position", "attitude", "linear_velocity", "angular_velocity"):
        if f"log_gamma_{name}" in likelihood:
            result[f"gamma_{name}"] = jnp.exp(likelihood[f"log_gamma_{name}"])
    return result


def se3_pseudo_likelihood(model, states: Array, step_size: Array, log_sigma_v: Array, log_sigma_w: Array) -> Array:
    """Per-step Euler-Maruyama pseudo-likelihood of the observed body-twist increments (SO(3) pendulum PL port).

    For every transition t -> t+1 of every window, with the controlled state x_t (its control slot holding the
    transition control, as in the rollout), drift f = vector_field(x_t)[12:18] and diffusion blocks A_v, A_w:
        xi_{t+1} - xi_t ~ N(f dt, blockdiag(A_v A_v^T, A_w A_w^T) dt + 2 sigma_obs^2 I).
    The 2 sigma_obs^2 is the observation noise of the two endpoints. Returns the mean NLL per transition
    (constants dropped); it depends on the diffusion directly, so the process noise cannot hide in sigma_obs.
    """
    controlled = states[:-1].at[..., 18:22].set(states[1:, ..., 18:22])
    flat = controlled.reshape(-1, states.shape[-1])
    drift = jax.vmap(model.vector_field)(flat)[:, 12:18]
    block_v, block_w = jax.vmap(model.twist_diffusion_blocks)(flat)
    increment = (states[1:, ..., 12:18] - states[:-1, ..., 12:18]).reshape(-1, 6)
    residual = increment - drift * step_size
    identity = jnp.eye(3, dtype=states.dtype)

    def block_nll(r, a, log_sigma):
        covariance = a @ a.T * step_size + 2.0 * jnp.exp(2.0 * log_sigma) * identity
        cholesky = jnp.linalg.cholesky(covariance)
        z = jax.scipy.linalg.solve_triangular(cholesky, r, lower=True)
        return 0.5 * jnp.dot(z, z) + jnp.sum(jnp.log(jnp.diag(cholesky)))

    nll_v = jax.vmap(block_nll, in_axes=(0, 0, None))(residual[:, :3], block_v, log_sigma_v)
    nll_w = jax.vmap(block_nll, in_axes=(0, 0, None))(residual[:, 3:], block_w, log_sigma_w)
    return jnp.mean(nll_v + nll_w)


def _transition_moments(model, states: Array, step_size: Array, step_fn):
    """Mean and square-root covariance of the model's OWN one-step twist transition, per transition of every window.

    step_fn(model, state (1, D), h, noise (1, 6)) is the package's stochastic Lie-IMEX step. The noise enters it
    linearly (Ito increment at the step start, then the implicit damping solve), so at zero noise
        xi_{t+1} = m(x_t) + J(x_t) z,   J = d xi_{t+1} / d z  (6 x 6, it already carries sqrt(h) and the damping solve)
    is exact for this step: the transition density is N(m, J J^T). Returns flattened (N, 6) means, (N, 6, 6) J and
    the (N, 6) observed next twists."""
    controlled = states[:-1].at[..., 18:22].set(states[1:, ..., 18:22])
    flat = controlled.reshape(-1, states.shape[-1])
    target = states[1:].reshape(-1, states.shape[-1])[:, 12:18]
    zero = jnp.zeros((6,), dtype=states.dtype)

    def one(x):
        step = lambda z: step_fn(model, x[None], step_size, z[None])[0, 12:18]
        return step(zero), jax.jacfwd(step)(zero)

    mean, jac = jax.vmap(one)(flat)
    return mean, jac, target


def se3_transition_nll(model, states: Array, step_size: Array, log_sigma_v: Array, log_sigma_w: Array, step_fn) -> Array:
    """Exact one-step transition NLL of the observed body twists under the model's own Lie-IMEX-SDE step
    (training.objective = transition). Per transition and per twist dimension, constants included, so the value of
    the true operators is a meaningful floor:
        xi_{t+1} ~ N(m(x_t), J J^T + 2 sigma_obs^2 I),   NLL = [0.5 z^T z + sum log diag L + 3 log(2 pi)] / 6.
    Unlike se3_pseudo_likelihood the mean is the integrator's own step (not an Euler step), which matters when the
    diffusion is small (WINDSDE torque wind 0.01: Euler leaves a 7-9 % excess spread in omega)."""
    mean, jac, target = _transition_moments(model, states, step_size, step_fn)
    obs = jnp.concatenate([jnp.full((3,), jnp.exp(2.0 * log_sigma_v)), jnp.full((3,), jnp.exp(2.0 * log_sigma_w))])

    def nll(m, j, y):
        covariance = j @ j.T + 2.0 * jnp.diag(obs).astype(y.dtype)
        cholesky = jnp.linalg.cholesky(covariance)
        z = jax.scipy.linalg.solve_triangular(cholesky, y - m, lower=True)
        return (0.5 * jnp.dot(z, z) + jnp.sum(jnp.log(jnp.diag(cholesky))) + 3.0 * math.log(2.0 * math.pi)) / 6.0

    return jnp.mean(jax.vmap(nll)(mean, jac, target))


def transition_diffusion(model, states: Array, step_size: Array, step_fn) -> Array:
    """(6,) RMS over transitions of the model's per-axis twist diffusion sqrt(diag(J J^T) / h), in twist units per
    sqrt(s): the learned counterpart of the true M^-1 Sigma (e.g. 0.25 x3, 0.01 x3 on WINDSDEC025A001)."""
    _, jac, _ = _transition_moments(model, states, step_size, step_fn)
    per_axis = jnp.sum(jac * jac, axis=-1) / step_size
    return jnp.sqrt(jnp.mean(per_axis, axis=0))


# ---------------------------------------------------------------------------
# Extended Kalman filter likelihood (training.objective = ekf)
# ---------------------------------------------------------------------------
def _exp_so3(phi: Array) -> Array:
    """Rodrigues' formula with a series-safe second coefficient (exact first order at phi = 0)."""
    theta_squared = jnp.sum(phi * phi) + 1.0e-12
    theta = jnp.sqrt(theta_squared)
    skew = jnp.array([[0.0, -phi[2], phi[1]], [phi[2], 0.0, -phi[0]], [-phi[1], phi[0], 0.0]], dtype=phi.dtype)
    a = jnp.sin(theta) / theta
    b = jnp.where(theta_squared < 1.0e-6, 0.5 - theta_squared / 24.0, (1.0 - jnp.cos(theta)) / theta_squared)
    return jnp.eye(3, dtype=phi.dtype) + a * skew + b * (skew @ skew)


def _retract(state: Array, delta: Array) -> Array:
    """x (+) delta on SE(3) x R^6 with a RIGHT attitude perturbation R Exp(dtheta) (the generator's noise model)."""
    rotation = state[3:12].reshape(3, 3) @ _exp_so3(delta[3:6])
    return jnp.concatenate([state[:3] + delta[:3], rotation.reshape(9), state[12:18] + delta[6:12], state[18:]])


def _difference(state_a: Array, state_b: Array) -> Array:
    """state_a (-) state_b in the 12-dim error coordinates, so that state_a = state_b (+) difference."""
    rotation = state_b[3:12].reshape(3, 3).T @ state_a[3:12].reshape(3, 3)
    return jnp.concatenate([state_a[:3] - state_b[:3], log_so3(rotation), state_a[12:18] - state_b[12:18]])


def se3_ekf_nll(model, windows: Array, step_size: Array, log_sigma_obs: Array, step_fn) -> Array:
    """Negative log marginal likelihood of noisy windows under the SDE, by an error-state extended Kalman filter.

    windows (T, B, 22) time-major; observations y_k = x_k (+) eps_k, eps_k ~ N(0, R_obs) with
    R_obs = diag(sigma_p^2 I3, sigma_R^2 I3, sigma_v^2 I3, sigma_w^2 I3) from log_sigma_obs (4,).
    Per window: x_0 = y_0, P_0 = R_obs; for k = 1..T-1 predict with the model's own stochastic Lie-IMEX step at zero noise,
        x^- = Phi(x, u_k, 0),  P^- = F P F^T + G G^T,   F = d(Phi(x (+) d) (-) x^-)/dd,  G = d(Phi(x, z) (-) x^-)/dz,
    then update with y_k: r = y_k (-) x^-, S = P^- + R_obs, log p(y_k | y_<k) = -0.5 (r^T S^-1 r + log det S + 12 log 2 pi),
    K = P^- S^-1, x = x^- (+) K r, P = (I - K) P^- (I - K)^T + K R_obs K^T (Joseph form).
    The observation noise is white in time while the diffusion accumulates in P, which is what separates sigma_obs from Sigma.
    Returns the mean NLL per predicted step and per error dimension (constants included)."""
    dtype = windows.dtype
    obs_var = jnp.repeat(jnp.exp(2.0 * log_sigma_obs), 3).astype(dtype)
    r_obs = jnp.diag(obs_var)
    identity = jnp.eye(12, dtype=dtype)
    zero_delta, zero_noise = jnp.zeros((12,), dtype), jnp.zeros((6,), dtype)
    log_two_pi = jnp.asarray(math.log(2.0 * math.pi), dtype)

    def one_window(window):
        def body(carry, observation):
            state, covariance = carry
            controlled = state.at[18:22].set(observation[18:22])
            step = lambda delta, noise: step_fn(model, _retract(controlled, delta)[None], step_size, noise[None])[0]
            predicted = step(zero_delta, zero_noise)
            transition = jax.jacfwd(lambda d: _difference(step(d, zero_noise), predicted))(zero_delta)
            noise_map = jax.jacfwd(lambda z: _difference(step(zero_delta, z), predicted))(zero_noise)
            prior = transition @ covariance @ transition.T + noise_map @ noise_map.T
            prior = 0.5 * (prior + prior.T)
            residual = _difference(observation, predicted)
            innovation = prior + r_obs
            cholesky = jnp.linalg.cholesky(innovation)
            whitened = jax.scipy.linalg.solve_triangular(cholesky, residual, lower=True)
            log_likelihood = -0.5 * (jnp.dot(whitened, whitened) + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky))) + 12.0 * log_two_pi)
            gain = jax.scipy.linalg.cho_solve((cholesky, True), prior).T               # P^- S^-1 (S, P symmetric)
            updated = _retract(predicted, gain @ residual)
            factor = identity - gain
            posterior = factor @ prior @ factor.T + gain @ r_obs @ gain.T
            return (updated, 0.5 * (posterior + posterior.T)), log_likelihood

        _, log_likelihoods = jax.lax.scan(body, (window[0], r_obs), window[1:])
        return log_likelihoods

    log_likelihoods = jax.vmap(one_window, in_axes=1, out_axes=1)(windows)          # (T-1, B)
    return -jnp.mean(log_likelihoods) / 12.0

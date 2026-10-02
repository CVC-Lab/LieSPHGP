"""SE(3) pose/geodesic training loss, plus the moment-matched predictive likelihood of the SDE variant."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg


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



def likelihood_vector(values) -> Array:
    """Pack a likelihood dictionary into the stable logging order; absent names read as zero."""
    reference = values["nll_total"]
    return jnp.stack([jnp.asarray(values.get(name, 0.0), dtype=reference.dtype) for name in LIKELIHOOD_NAMES])


def observation_dictionary(log_sigma: Array) -> dict[str, Array]:
    """The module's (4,) log observation scales as the {block: log sigma} mapping se3_moment_nll expects."""
    return {"log_sigma_position": log_sigma[0], "log_sigma_attitude": log_sigma[1],
            "log_sigma_linear_velocity": log_sigma[2], "log_sigma_angular_velocity": log_sigma[3]}


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

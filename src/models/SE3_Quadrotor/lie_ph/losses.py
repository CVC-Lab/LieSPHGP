"""Negative log marginal likelihood of noisy windows under the SDE, by an error-state extended Kalman filter.

Observation model (the generator's): y_k = x_k (+) eps_k, x (+) [a, b, c, d] = (x + a, R Exp(b), v + c, omega + d),
eps_k ~ N(0, R_obs), R_obs = diag(sigma_p^2 I3, sigma_R^2 I3, sigma_v^2 I3, sigma_w^2 I3), all four learned.

Per window, x_0 = y_0 and P_0 = R_obs, then for k = 1..T-1:

    predict  every substep j (step h = dt / substeps, Brownian z_j ~ N(0, I6)):
             x_{j+1} = Psi(x_j, 0),   P_{j+1} = F_j P_j F_j^T + G_j G_j^T,
             [F_j  G_j] = d(Psi(x_j (+) d, z) (-) x_{j+1}) / d(d, z)  at (0, 0)
    update   r = y_k (-) x^-,  S = P^- + R_obs,
             -log p(y_k | y_<k) = 0.5 (r^T S^-1 r + log det S + 12 log 2 pi),
             K = P^- S^-1,  m = K r,  x = x^- (+) m,  P = (I - K) P^- (I - K)^T + K R_obs K^T (Joseph form),
    reset    P <- G P G^T,  G = identity except the attitude block J_r(m_theta): the covariance, computed in the
             tangent space at x^-, is re-expressed at the updated mean (Bourmaud et al. 2015, eq. 64; the full-order
             reset of Maurer et al. 2025). Measured on the pendulum data it changes the NLL by 2e-5.

The normalised innovation squared NIS_k = r^T S^-1 r has mean d for a consistent filter (reported by evaluate.py).

Identical to the pendulum package's filter except for the 12-dim error state [dx, d theta, dv, d omega]. The
Jacobians use the first-order difference vee(skew(R_b^T R_a)) (same derivative as Log at the nominal point,
polynomial second derivative; the true Log at the identity turns float32 rounding into ~1e12 training gradients).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from .integrator import exp_so3, lie_imex_sde_step, log_so3, right_jacobian_so3

Array = jax.Array
ERROR_DIMENSION = 12
NOISE_DIMENSION = 6


def retract(state: Array, delta: Array) -> Array:
    rotation = state[3:12].reshape(3, 3) @ exp_so3(delta[3:6])
    return jnp.concatenate([state[:3] + delta[:3], rotation.reshape(9), state[12:18] + delta[6:12], state[18:]])


def difference(state_a: Array, state_b: Array) -> Array:
    relative = state_b[3:12].reshape(3, 3).T @ state_a[3:12].reshape(3, 3)
    return jnp.concatenate([state_a[:3] - state_b[:3], log_so3(relative), state_a[12:18] - state_b[12:18]])


def local_difference(state_a: Array, state_b: Array) -> Array:
    relative = state_b[3:12].reshape(3, 3).T @ state_a[3:12].reshape(3, 3)
    skew = 0.5 * (relative - relative.T)
    return jnp.concatenate([state_a[:3] - state_b[:3], jnp.stack([skew[2, 1], skew[0, 2], skew[1, 0]]),
                            state_a[12:18] - state_b[12:18]])


def observation_variance(log_sigma: Array) -> Array:
    """(12,) diagonal of R_obs from [log sigma_p, log sigma_R, log sigma_v, log sigma_w]."""
    return jnp.repeat(jnp.exp(2.0 * log_sigma), 3)


def predict(model, state: Array, covariance: Array, control: Array, interval: float, substeps: int):
    step = jnp.asarray(interval / substeps, state.dtype)
    zeros = jnp.zeros((ERROR_DIMENSION + NOISE_DIMENSION,), state.dtype)

    def body(carry, _):
        current, matrix = carry

        def propagate(vector):
            return lie_imex_sde_step(model, retract(current, vector[:ERROR_DIMENSION]), step,
                                     vector[ERROR_DIMENSION:])

        nominal = propagate(zeros)
        jacobian = jax.jacfwd(lambda v: local_difference(propagate(v), nominal))(zeros)
        transition, noise_map = jacobian[:, :ERROR_DIMENSION], jacobian[:, ERROR_DIMENSION:]
        matrix = transition @ matrix @ transition.T + noise_map @ noise_map.T
        return (nominal, 0.5 * (matrix + matrix.T)), None

    (mean, covariance), _ = jax.lax.scan(body, (state.at[18:22].set(control), covariance), None, length=substeps)
    return mean, covariance


def ekf_window_stats(model, window: Array, log_sigma: Array, interval: float, substeps: int) -> tuple[Array, Array]:
    """((T-1,) negative log predictive density, (T-1,) NIS) of y_1..y_{T-1} for one (T, 22) window."""
    dtype = window.dtype
    r_obs = jnp.diag(observation_variance(log_sigma).astype(dtype))
    identity = jnp.eye(ERROR_DIMENSION, dtype=dtype)
    log_two_pi = math.log(2.0 * math.pi)

    def body(carry, observation):
        state, covariance = carry
        predicted, prior = predict(model, state, covariance, observation[18:22], interval, substeps)
        residual = difference(observation, predicted)
        cholesky = jnp.linalg.cholesky(prior + r_obs)
        whitened = jax.scipy.linalg.solve_triangular(cholesky, residual, lower=True)
        nis = whitened @ whitened
        nll = 0.5 * (nis + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky))) + ERROR_DIMENSION * log_two_pi)
        gain = jax.scipy.linalg.cho_solve((cholesky, True), prior).T
        correction = gain @ residual
        updated = retract(predicted, correction)
        factor = identity - gain
        posterior = factor @ prior @ factor.T + gain @ r_obs @ gain.T
        reset = identity.at[3:6, 3:6].set(right_jacobian_so3(correction[3:6]))       # tangent space at the new mean
        posterior = reset @ posterior @ reset.T
        return (updated, 0.5 * (posterior + posterior.T)), (nll, nis)

    _, (nll, nis) = jax.lax.scan(body, (window[0], r_obs), window[1:])
    return nll, nis


def ekf_window_nll(model, window: Array, log_sigma: Array, interval: float, substeps: int) -> Array:
    """(T-1,) negative log predictive density of y_1..y_{T-1} for one (T, 22) window."""
    return ekf_window_stats(model, window, log_sigma, interval, substeps)[0]


def ekf_nll(model, windows: Array, log_sigma: Array, interval: float, substeps: int) -> Array:
    """Mean NLL per predicted observation and per error dimension over time-major (T, B, 22) windows."""
    per_window = jax.vmap(lambda w: ekf_window_nll(model, w, log_sigma, interval, substeps),
                          in_axes=1, out_axes=1)(windows)
    return jnp.mean(per_window) / ERROR_DIMENSION


def rollout_loss(model, windows: Array, interval: float, substeps: int) -> Array:
    """Trajectory loss of Lie-PH-NN-ODE (the SE(3) version of the pendulum package's): roll the deterministic model out
    with Lie-IMEX from the first (noisy) sample of each window with the recorded controls and compare every later sample,

        L = mean_{t>=1, windows} ( |x_hat - x|^2 + theta(R_hat, R)^2 + |v_hat - v|^2 + |omega_hat - omega|^2 ),

    theta the geodesic angle. Windows whose rollout became non-finite are left out of the mean (kept as in the pendulum
    package: the user chose not to fix the NaN gradient this masking can let through)."""
    from .integrator import rollout
    controls = jnp.swapaxes(windows[1:, :, 18:22], 0, 1)                                      # (B, T-1, 4)
    noise = jnp.zeros((windows.shape[0] - 1, substeps, NOISE_DIMENSION), windows.dtype)
    predicted = jax.vmap(lambda x0, u: rollout(model, x0, u, interval, noise))(windows[0], controls)   # (B, T, 22)
    observed = jnp.swapaxes(windows, 0, 1)
    angle = jax.vmap(jax.vmap(lambda a, b: log_so3(b[3:12].reshape(3, 3).T @ a[3:12].reshape(3, 3))))(
        predicted[:, 1:], observed[:, 1:])
    error = (jnp.sum(jnp.square(predicted[:, 1:, :3] - observed[:, 1:, :3]), axis=-1) + jnp.sum(angle * angle, axis=-1)
             + jnp.sum(jnp.square(predicted[:, 1:, 12:18] - observed[:, 1:, 12:18]), axis=-1))
    per_window = jnp.mean(error, axis=1)
    healthy = jax.lax.stop_gradient(jnp.isfinite(per_window))
    safe = jnp.where(healthy, per_window, 0.0)
    return jnp.sum(safe) / jnp.maximum(jnp.sum(healthy), 1.0)


def ekf_consistency(model, windows: Array, log_sigma: Array, interval: float, substeps: int) -> Array:
    """Mean NIS / d over time-major windows: 1 for a statistically consistent filter."""
    nis = jax.vmap(lambda w: ekf_window_stats(model, w, log_sigma, interval, substeps)[1], in_axes=1)(windows)
    return jnp.mean(nis) / ERROR_DIMENSION

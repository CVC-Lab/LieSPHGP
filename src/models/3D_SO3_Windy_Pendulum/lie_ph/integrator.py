"""Second-order Lie-IMEX SDE step on SO(3) x R^3 (damping implicit, everything else explicit).

With a_i = domega/dt at stage i with the damping added back (a_i + K_i omega_i, K = M^-1 D), xi = sqrt(h) Sigma z:

    predictor:  (I + h K_1) omega_p     = omega + h a_1 + xi,                         R_p = R Exp(h omega)
    corrector:  (I + h/2 K_2) omega_+   = omega + h/2 (a_1 + a_2 - K_1 omega) + xi,   R_+ = R Exp(h/2 (omega + omega_p))

The noise is additive on the body twist, so Ito and Stratonovich agree, and the same increment enters both stages.
This is the step of envs/pendulum_so3/windy_pendulum_3d.py::_lie_imex_step and of the quadrotor package.
One observation interval dt is ``substeps`` such steps, each with its own z.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def hat(vector: Array) -> Array:
    x, y, z = vector[0], vector[1], vector[2]
    zero = jnp.zeros_like(x)
    return jnp.stack([jnp.stack([zero, -z, y]), jnp.stack([z, zero, -x]), jnp.stack([-y, x, zero])])


# Below SERIES_LIMIT (angle^2 = 0.01) the coefficients are Taylor polynomials, exact to float precision. The closed
# forms lose their low-order derivatives to cancellation near zero (sin t / t, (1 - cos t) / t^2 in float32), and
# training differentiates through Jacobians of these maps, i.e. through their second derivatives.
SERIES_LIMIT = 1.0e-2


def exp_so3(phi: Array) -> Array:
    """Rodrigues' formula R = I + a hat(phi) + b hat(phi)^2, series coefficients near zero."""
    theta_squared = jnp.sum(phi * phi)
    small = theta_squared < SERIES_LIMIT
    safe = jnp.where(small, 1.0, theta_squared)                 # the unused branch stays finite (double where)
    theta = jnp.sqrt(safe)
    t2 = theta_squared
    a = jnp.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0 - t2 * t2 * t2 / 5040.0, jnp.sin(theta) / theta)
    b = jnp.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0 - t2 * t2 * t2 / 40320.0, (1.0 - jnp.cos(theta)) / safe)
    skew = hat(phi)
    return jnp.eye(3, dtype=phi.dtype) + a * skew + b * (skew @ skew)


def log_so3(rotation: Array) -> Array:
    """Rotation vector of R: vee(skew R) * theta / sin(theta), series in sin^2 near the identity."""
    skew = 0.5 * (rotation - rotation.T)
    vee = jnp.stack([skew[2, 1], skew[0, 2], skew[1, 0]])       # sin(theta) * axis
    s2 = jnp.sum(vee * vee)
    cosine = 0.5 * (jnp.trace(rotation) - 1.0)
    small = (s2 < SERIES_LIMIT) & (cosine > 0.0)
    safe = jnp.where(small, 1.0, jnp.maximum(s2, 1.0e-12))
    sine = jnp.sqrt(safe)
    factor = jnp.where(small, 1.0 + s2 / 6.0 + 3.0 * s2 * s2 / 40.0 + 5.0 * s2 * s2 * s2 / 112.0,
                       jnp.arctan2(sine, cosine) / sine)        # arcsin(s) / s
    return vee * factor


def right_jacobian_so3(phi: Array) -> Array:
    """J_r(phi) = I - a hat(phi) + b hat(phi)^2 with a = (1 - cos t) / t^2, b = (t - sin t) / t^3, so that
    Exp(phi + d) = Exp(phi) Exp(J_r(phi) d) to first order in d. Series coefficients near zero, as in exp_so3."""
    theta_squared = jnp.sum(phi * phi)
    small = theta_squared < SERIES_LIMIT
    safe = jnp.where(small, 1.0, theta_squared)
    theta = jnp.sqrt(safe)
    t2 = theta_squared
    a = jnp.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0 - t2 * t2 * t2 / 40320.0, (1.0 - jnp.cos(theta)) / safe)
    b = jnp.where(small, 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0 - t2 * t2 * t2 / 362880.0,
                  (theta - jnp.sin(theta)) / (safe * theta))
    skew = hat(phi)
    return jnp.eye(3, dtype=phi.dtype) - a * skew + b * (skew @ skew)


def lie_imex_sde_step(model, state: Array, step: Array, noise: Array) -> Array:
    """One step of size ``step``; ``noise`` (3,) standard normal."""
    return lie_imex_increment_step(model, state, step, jnp.sqrt(step) * model.diffusion() * noise)


def lie_imex_increment_step(model, state: Array, step: Array, increment: Array) -> Array:
    """The same step with the omega increment xi given directly, e.g. the simulator's recorded wind (report.py)."""
    rotation, omega, control = state[:9].reshape(3, 3), state[9:12], state[12:15]
    identity = jnp.eye(3, dtype=state.dtype)

    _, rate_1 = model.vector_field(state)
    damping_1 = model.effective_damping(state)
    explicit_1 = rate_1 + damping_1 @ omega
    omega_p = jnp.linalg.solve(identity + step * damping_1, omega + step * explicit_1 + increment)
    predictor = jnp.concatenate([(rotation @ exp_so3(step * omega)).reshape(9), omega_p, control])

    _, rate_2 = model.vector_field(predictor)
    damping_2 = model.effective_damping(predictor)
    explicit_2 = rate_2 + damping_2 @ omega_p
    right = omega + 0.5 * step * (explicit_1 + explicit_2 - damping_1 @ omega) + increment
    omega_next = jnp.linalg.solve(identity + 0.5 * step * damping_2, right)
    rotation_next = rotation @ exp_so3(0.5 * step * (omega + omega_p))
    return jnp.concatenate([rotation_next.reshape(9), omega_next, control])


def interval_step(model, state: Array, control: Array, interval: float, noise: Array) -> Array:
    """Advance one observation interval with ``noise.shape[0]`` Lie-IMEX substeps under the held control."""
    substeps = noise.shape[0]
    step = jnp.asarray(interval / substeps, state.dtype)
    state = state.at[12:15].set(control)

    def body(current, z):
        return lie_imex_sde_step(model, current, step, z), None

    final, _ = jax.lax.scan(body, state, noise)
    return final


def rollout(model, initial_state: Array, controls: Array, interval: float, noise: Array) -> Array:
    """(T, 15) path from the initial state; ``controls`` (T-1, 3) per interval, ``noise`` (T-1, substeps, 3)."""
    def body(current, inputs):
        control, z = inputs
        following = interval_step(model, current, control, interval, z)
        return following, following

    _, path = jax.lax.scan(body, initial_state, (controls, noise))
    return jnp.concatenate([initial_state[None], path], axis=0)

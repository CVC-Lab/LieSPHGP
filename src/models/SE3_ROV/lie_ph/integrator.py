"""Second-order Lie-IMEX SDE step on SE(3) x R^6 (damping implicit, everything else explicit). BlueROV2 copy of the quadrotor package: the control is state[18:] (n_u inputs).

With xi = (v, omega), a_i = dxi/dt at stage i with the damping added back (a_i + K_i xi_i, K = blockdiag(M1^-1 D_v,
M2^-1 D_w)), and the twist increment eta = sqrt(h) Sigma z, z ~ N(0, I6):

    predictor:  (I + h K_1) xi_p     = xi + h a_1 + eta,                       x_p = x + h dx_1,  R_p = R Exp(h omega)
    corrector:  (I + h/2 K_2) xi_+   = xi + h/2 (a_1 + a_2 - K_1 xi) + eta,
                x_+ = x + h/2 (dx_1 + dx_2),   R_+ = R Exp(h/2 (omega + omega_p))

Additive noise on the body twist, so Ito and Stratonovich agree and the same increment enters both stages. Same
scheme as src/models/3D_SO3_Windy_Pendulum/lie_ph/integrator.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array
SERIES_LIMIT = 1.0e-2


def hat(vector: Array) -> Array:
    x, y, z = vector[0], vector[1], vector[2]
    zero = jnp.zeros_like(x)
    return jnp.stack([jnp.stack([zero, -z, y]), jnp.stack([z, zero, -x]), jnp.stack([-y, x, zero])])


def exp_so3(phi: Array) -> Array:
    """Rodrigues' formula R = I + a hat(phi) + b hat(phi)^2, Taylor coefficients below angle^2 = SERIES_LIMIT
    (the closed forms lose their derivatives to cancellation near zero, and training differentiates Jacobians)."""
    theta_squared = jnp.sum(phi * phi)
    small = theta_squared < SERIES_LIMIT
    safe = jnp.where(small, 1.0, theta_squared)
    theta = jnp.sqrt(safe)
    t2 = theta_squared
    a = jnp.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0 - t2 * t2 * t2 / 5040.0, jnp.sin(theta) / theta)
    b = jnp.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0 - t2 * t2 * t2 / 40320.0, (1.0 - jnp.cos(theta)) / safe)
    skew = hat(phi)
    return jnp.eye(3, dtype=phi.dtype) + a * skew + b * (skew @ skew)


def log_so3(rotation: Array) -> Array:
    """Rotation vector of R: vee(skew R) * theta / sin(theta), series in sin^2 near the identity."""
    skew = 0.5 * (rotation - rotation.T)
    vee = jnp.stack([skew[2, 1], skew[0, 2], skew[1, 0]])
    s2 = jnp.sum(vee * vee)
    cosine = 0.5 * (jnp.trace(rotation) - 1.0)
    small = (s2 < SERIES_LIMIT) & (cosine > 0.0)
    safe = jnp.where(small, 1.0, jnp.maximum(s2, 1.0e-12))
    sine = jnp.sqrt(safe)
    factor = jnp.where(small, 1.0 + s2 / 6.0 + 3.0 * s2 * s2 / 40.0 + 5.0 * s2 * s2 * s2 / 112.0,
                       jnp.arctan2(sine, cosine) / sine)
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


def _implicit(matrices: tuple[Array, Array], right: Array, scale: Array) -> Array:
    identity = jnp.eye(3, dtype=right.dtype)
    return jnp.concatenate([jnp.linalg.solve(identity + scale * matrices[0], right[:3]),
                            jnp.linalg.solve(identity + scale * matrices[1], right[3:])])


def _apply(matrices: tuple[Array, Array], twist: Array) -> Array:
    return jnp.concatenate([matrices[0] @ twist[:3], matrices[1] @ twist[3:]])


def lie_imex_sde_step(model, state: Array, step: Array, noise: Array) -> Array:
    """One step of size ``step``; ``noise`` (6,) standard normal."""
    return lie_imex_increment_step(model, state, step, jnp.sqrt(step) * model.diffusion() * noise)


def lie_imex_increment_step(model, state: Array, step: Array, increment: Array) -> Array:
    """The same step with the twist increment eta given directly (e.g. the simulator's recorded wind)."""
    position, rotation = state[:3], state[3:12].reshape(3, 3)
    twist, control = state[12:18], state[18:]

    rate_1 = model.vector_field(state)
    damping_1 = model.effective_damping(state)
    explicit_1 = rate_1[12:18] + _apply(damping_1, twist)
    twist_p = _implicit(damping_1, twist + step * explicit_1 + increment, step)
    predictor = jnp.concatenate([position + step * rate_1[:3], (rotation @ exp_so3(step * twist[3:])).reshape(9),
                                 twist_p, control])

    rate_2 = model.vector_field(predictor)
    damping_2 = model.effective_damping(predictor)
    explicit_2 = rate_2[12:18] + _apply(damping_2, twist_p)
    right = twist + 0.5 * step * (explicit_1 + explicit_2 - _apply(damping_1, twist)) + increment
    twist_next = _implicit(damping_2, right, 0.5 * step)
    rotation_next = rotation @ exp_so3(0.5 * step * (twist[3:] + twist_p[3:]))
    position_next = position + 0.5 * step * (rate_1[:3] + rate_2[:3])
    return jnp.concatenate([position_next, rotation_next.reshape(9), twist_next, control])


def interval_step(model, state: Array, control: Array, interval: float, noise: Array) -> Array:
    """Advance one observation interval with ``noise.shape[0]`` Lie-IMEX substeps under the held control."""
    step = jnp.asarray(interval / noise.shape[0], state.dtype)

    def body(current, z):
        return lie_imex_sde_step(model, current, step, z), None

    final, _ = jax.lax.scan(body, state.at[18:].set(control), noise)
    return final


def rollout(model, initial_state: Array, controls: Array, interval: float, noise: Array) -> Array:
    """(T, 18 + n_u) path; ``controls`` (T-1, n_u) per interval, ``noise`` (T-1, substeps, 6)."""
    def body(current, inputs):
        control, z = inputs
        following = interval_step(model, current, control, interval, z)
        return following, following

    _, path = jax.lax.scan(body, initial_state, (controls, noise))
    return jnp.concatenate([initial_state[None], path], axis=0)

"""Second-order Lie-IMEX integration for the GP SE(3) model, plus its stochastic (SDE) variant.

SDE step. The process noise is additive on the body twist xi = (v, omega) only - forces and torques are
uncertain, kinematics are not - with diagonal diffusion Sigma = diag(sigma_v 1, sigma_omega 1):

    d xi = [ M^-1 ((J - R) grad H + g u) ] dt + Sigma dW,      dx = R v dt,   dR = R hat(omega) dt.

Additive noise makes Ito and Stratonovich coincide, so the deterministic predictor-corrector is reused with
the same Brownian increment sqrt(h) Sigma z, z ~ N(0, I_6), added to both the predictor and corrector
right-hand sides before the implicit damping solve. The rotation update stays the exact exponential, so
sample paths remain on SO(3).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .network import SampledSE3HamODE


Array = jax.Array


def hat(vector: Array) -> Array:
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack(
        [
            jnp.stack([zero, -z, y], axis=-1),
            jnp.stack([z, zero, -x], axis=-1),
            jnp.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )


def exp_so3(phi: Array, epsilon: float = 1.0e-12) -> Array:
    theta_squared = jnp.sum(phi * phi, axis=-1, keepdims=True) + epsilon
    theta = jnp.sqrt(theta_squared)
    skew = hat(phi)
    identity = jnp.broadcast_to(jnp.eye(3, dtype=phi.dtype), skew.shape)
    a = (jnp.sin(theta) / theta)[..., None]
    b = ((1.0 - jnp.cos(theta)) / theta_squared)[..., None]
    return identity + a * skew + b * (skew @ skew)


def _vector_field(model: SampledSE3HamODE, state: Array) -> Array:
    return jax.vmap(model.vector_field)(state)


def _damping_matrices(
    model: SampledSE3HamODE, state: Array
) -> tuple[Array, Array]:
    return jax.vmap(model.effective_damping)(state)


def _apply_damping(
    linear_damping_matrix: Array,
    angular_damping_matrix: Array,
    twist: Array,
) -> Array:
    return jnp.concatenate(
        [
            (linear_damping_matrix @ twist[:, :3, None]).squeeze(-1),
            (angular_damping_matrix @ twist[:, 3:6, None]).squeeze(-1),
        ],
        axis=1,
    )


def _implicit_solve(
    linear_damping_matrix: Array,
    angular_damping_matrix: Array,
    right_hand_side: Array,
    scale: Array,
) -> Array:
    identity = jnp.broadcast_to(
        jnp.eye(3, dtype=right_hand_side.dtype), linear_damping_matrix.shape
    )
    velocity = jnp.linalg.solve(
        identity + scale * linear_damping_matrix, right_hand_side[:, :3, None]
    ).squeeze(-1)
    omega = jnp.linalg.solve(
        identity + scale * angular_damping_matrix,
        right_hand_side[:, 3:6, None],
    ).squeeze(-1)
    return jnp.concatenate([velocity, omega], axis=1)


def lie_imex_step(
    model: SampledSE3HamODE, state: Array, step_size: Array
) -> Array:
    position = state[:, :3]
    rotation = state[:, 3:12].reshape(-1, 3, 3)
    twist = state[:, 12:18]
    control = state[:, 18:22]

    derivative_1 = _vector_field(model, state)
    linear_damping_1, angular_damping_1 = _damping_matrices(model, state)
    damping_1 = _apply_damping(linear_damping_1, angular_damping_1, twist)
    explicit_1 = derivative_1[:, 12:18] + damping_1
    twist_predictor = _implicit_solve(
        linear_damping_1,
        angular_damping_1,
        twist + step_size * explicit_1,
        step_size,
    )

    phi_1 = step_size * twist[:, 3:6]
    predictor = jnp.concatenate(
        [
            position + step_size * derivative_1[:, :3],
            (rotation @ exp_so3(phi_1)).reshape(-1, 9),
            twist_predictor,
            control,
        ],
        axis=1,
    )

    derivative_2 = _vector_field(model, predictor)
    linear_damping_2, angular_damping_2 = _damping_matrices(model, predictor)
    damping_2 = _apply_damping(
        linear_damping_2, angular_damping_2, twist_predictor
    )
    explicit_2 = derivative_2[:, 12:18] + damping_2
    corrector_rhs = twist + 0.5 * step_size * (
        explicit_1 + explicit_2 - damping_1
    )
    twist_next = _implicit_solve(
        linear_damping_2,
        angular_damping_2,
        corrector_rhs,
        0.5 * step_size,
    )

    phi_2 = step_size * twist_predictor[:, 3:6]
    rotation_next = rotation @ exp_so3(0.5 * (phi_1 + phi_2))
    position_next = position + 0.5 * step_size * (
        derivative_1[:, :3] + derivative_2[:, :3]
    )
    return jnp.concatenate(
        [position_next, rotation_next.reshape(-1, 9), twist_next, control],
        axis=1,
    )


def lie_imex_sde_step(
    model: SampledSE3HamODE, state: Array, step_size: Array, noise: Array
) -> Array:
    """One stochastic Lie-IMEX step; ``noise`` is (batch, 6) standard normal for this step."""
    position = state[:, :3]
    rotation = state[:, 3:12].reshape(-1, 3, 3)
    twist = state[:, 12:18]
    control = state[:, 18:22]
    # Ito increment at the start of the step, shared by predictor and corrector: sqrt(h) Sigma z (constant)
    # or sqrt(h) M^-1(x) Sigma(x) z (state-dependent diffusion entering the momentum equations).
    increment = jnp.sqrt(step_size) * jax.vmap(model.twist_noise)(state, noise)

    derivative_1 = _vector_field(model, state)
    linear_damping_1, angular_damping_1 = _damping_matrices(model, state)
    damping_1 = _apply_damping(linear_damping_1, angular_damping_1, twist)
    explicit_1 = derivative_1[:, 12:18] + damping_1
    twist_predictor = _implicit_solve(
        linear_damping_1, angular_damping_1, twist + step_size * explicit_1 + increment, step_size,
    )
    phi_1 = step_size * twist[:, 3:6]
    predictor = jnp.concatenate(
        [position + step_size * derivative_1[:, :3], (rotation @ exp_so3(phi_1)).reshape(-1, 9),
         twist_predictor, control], axis=1,
    )
    derivative_2 = _vector_field(model, predictor)
    linear_damping_2, angular_damping_2 = _damping_matrices(model, predictor)
    damping_2 = _apply_damping(linear_damping_2, angular_damping_2, twist_predictor)
    explicit_2 = derivative_2[:, 12:18] + damping_2
    corrector_rhs = twist + 0.5 * step_size * (explicit_1 + explicit_2 - damping_1) + increment
    twist_next = _implicit_solve(linear_damping_2, angular_damping_2, corrector_rhs, 0.5 * step_size)
    phi_2 = step_size * twist_predictor[:, 3:6]
    rotation_next = rotation @ exp_so3(0.5 * (phi_1 + phi_2))
    position_next = position + 0.5 * step_size * (derivative_1[:, :3] + derivative_2[:, :3])
    return jnp.concatenate([position_next, rotation_next.reshape(-1, 9), twist_next, control], axis=1)


@jax.jit
def rollout_control_sequence_sde(
    model: SampledSE3HamODE,
    initial_state: Array,
    transition_controls: Array,
    step_size: Array,
    noise: Array,
    active: Array | None = None,
) -> Array:
    """Time-major rollout with per-step controls (T-1, B, 4) and standard-normal noise (T-1, B, 6).

    ``active`` (B,) bool, optional: windows marked False are held at their initial state for the whole
    rollout. Their step is still evaluated (from that fixed, sane state, so every intermediate is finite) but
    discarded by a select, which makes their Jacobian the identity. This is how a window whose path is known
    to diverge is kept out of the backward pass: masking its loss alone is not enough, because the zero
    cotangent still traverses the exploded path and 0 * inf = NaN reaches the model weights.
    """
    def scan_step(state, inputs):
        control, step_noise = inputs
        controlled_state = state.at[:, 18:22].set(control)
        next_state = lie_imex_sde_step(model, controlled_state, step_size, step_noise)
        if active is not None:
            next_state = jnp.where(active[:, None], next_state, controlled_state)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, (transition_controls, noise))
    return jnp.concatenate([initial_state[None], trajectory], axis=0)


@partial(jax.jit, static_argnames=("steps",))
def rollout(
    model: SampledSE3HamODE,
    initial_state: Array,
    step_size: Array,
    *,
    steps: int,
) -> Array:
    def scan_step(state, _):
        next_state = lie_imex_step(model, state, step_size)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, None, length=steps)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)


@jax.jit
def rollout_control_sequence(
    model: SampledSE3HamODE,
    initial_state: Array,
    transition_controls: Array,
    step_size: Array,
) -> Array:
    def scan_step(state, control):
        controlled_state = state.at[:, 18:22].set(control)
        next_state = lie_imex_step(model, controlled_state, step_size)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, transition_controls)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)

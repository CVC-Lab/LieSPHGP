"""Second-order Lie-IMEX integration for the GP SE(3) model."""

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


@partial(jax.jit, static_argnames=("substeps",))
def rollout_control_sequence(
    model: SampledSE3HamODE,
    initial_state: Array,
    transition_controls: Array,
    step_size: Array,
    active: Array | None = None,
    substeps: int = 1,
) -> Array:
    """``active`` (B,) bool, optional: windows marked False are held at their initial state (their step is
    evaluated from that fixed, finite state and discarded by a select, so their Jacobian is the identity).
    Used with training.screen_diverged_windows: a window whose rollout is known to diverge must not enter the
    backward pass - masking its loss is not enough, the zero cotangent still traverses the exploded path and
    0 * inf = NaN reaches the weights (measured on the SDE variant, 20 Sep 2026)."""
    inner = step_size / substeps

    def advance(state):
        # ``substeps`` Lie-IMEX steps of step_size/substeps, control held constant across them
        if substeps <= 1:
            return lie_imex_step(model, state, step_size)

        def body(carry, _):
            stepped = lie_imex_step(model, carry, inner)
            return stepped.at[:, 18:22].set(carry[:, 18:22]), None

        out, _ = jax.lax.scan(body, state, None, length=substeps)
        return out

    def scan_step(state, control):
        controlled_state = state.at[:, 18:22].set(control)
        next_state = advance(controlled_state)
        if active is not None:
            next_state = jnp.where(active[:, None], next_state, controlled_state)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, transition_controls)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)

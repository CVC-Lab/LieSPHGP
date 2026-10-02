"""Second-order Lie-IMEX integration for the neural PH model."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .network import effective_damping_blocks, vector_field


Array = jax.Array


def hat(vector: Array) -> Array:
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack(
        [jnp.stack([zero, -z, y], -1), jnp.stack([z, zero, -x], -1),
         jnp.stack([-y, x, zero], -1)],
        axis=-2,
    )


def exp_so3(phi: Array, epsilon: float = 1.0e-12) -> Array:
    theta_squared = jnp.sum(phi * phi, axis=-1, keepdims=True) + epsilon
    theta = jnp.sqrt(theta_squared)
    skew = hat(phi)
    identity = jnp.broadcast_to(jnp.eye(3, dtype=phi.dtype), skew.shape)
    return identity + (jnp.sin(theta) / theta)[..., None] * skew + (
        (1.0 - jnp.cos(theta)) / theta_squared
    )[..., None] * (skew @ skew)


def _apply_damping(block_v: Array, block_w: Array, twist: Array) -> Array:
    return jnp.concatenate(
        [(block_v @ twist[:, :3, None]).squeeze(-1),
         (block_w @ twist[:, 3:6, None]).squeeze(-1)],
        axis=1,
    )


def _implicit_solve(
    block_v: Array, block_w: Array, right_hand_side: Array, scale: Array
) -> Array:
    identity = jnp.broadcast_to(jnp.eye(3, dtype=right_hand_side.dtype), block_v.shape)
    velocity = jnp.linalg.solve(
        identity + scale * block_v, right_hand_side[:, :3, None]
    ).squeeze(-1)
    omega = jnp.linalg.solve(
        identity + scale * block_w, right_hand_side[:, 3:6, None]
    ).squeeze(-1)
    return jnp.concatenate([velocity, omega], axis=1)


def lie_imex_step(model, state: Array, step_size: Array) -> Array:
    position = state[:, :3]
    rotation = state[:, 3:12].reshape(-1, 3, 3)
    twist = state[:, 12:18]
    control = state[:, 18:22]

    derivative_1 = vector_field(model, state)
    block_v_1, block_w_1 = effective_damping_blocks(model, state)
    damping_1 = _apply_damping(block_v_1, block_w_1, twist)
    explicit_1 = derivative_1[:, 12:18] + damping_1
    twist_predictor = _implicit_solve(
        block_v_1, block_w_1, twist + step_size * explicit_1, step_size
    )
    rotation_predictor = rotation @ exp_so3(step_size * twist[:, 3:6])
    position_predictor = position + step_size * derivative_1[:, :3]
    predictor = jnp.concatenate(
        [position_predictor, rotation_predictor.reshape(-1, 9), twist_predictor, control],
        axis=1,
    )

    derivative_2 = vector_field(model, predictor)
    block_v_2, block_w_2 = effective_damping_blocks(model, predictor)
    explicit_2 = derivative_2[:, 12:18] + _apply_damping(
        block_v_2, block_w_2, twist_predictor
    )
    right_hand_side = twist + 0.5 * step_size * (
        explicit_1 + explicit_2 - damping_1
    )
    twist_next = _implicit_solve(
        block_v_2, block_w_2, right_hand_side, 0.5 * step_size
    )
    rotation_next = rotation @ exp_so3(
        0.5 * step_size * (twist[:, 3:6] + twist_predictor[:, 3:6])
    )
    position_next = position + 0.5 * step_size * (
        derivative_1[:, :3] + derivative_2[:, :3]
    )
    return jnp.concatenate(
        [position_next, rotation_next.reshape(-1, 9), twist_next, control], axis=1
    )


step = lie_imex_step


@partial(jax.jit, static_argnames=("steps",))
def rollout(model, initial_state: Array, step_size: Array, *, steps: int) -> Array:
    def scan_step(state, _):
        next_state = lie_imex_step(model, state, step_size)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, None, length=steps)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)


def interval_step(model, state: Array, step_size: Array, substeps: int) -> Array:
    """Advance one DATA interval of ``step_size`` with ``substeps`` Lie-IMEX steps of ``step_size/substeps``.

    The control channels are held constant across the sub-steps (zero-order hold), which is what the recorded
    control is.  substeps == 1 is exactly ``lie_imex_step``.
    """
    if substeps <= 1:
        return lie_imex_step(model, state, step_size)
    inner = step_size / substeps

    def body(carry, _):
        advanced = lie_imex_step(model, carry, inner)
        return advanced.at[:, 18:22].set(carry[:, 18:22]), None

    out, _ = jax.lax.scan(body, state, None, length=substeps)
    return out


@partial(jax.jit, static_argnames=("substeps",))
def rollout_control_sequence(
    model, initial_state: Array, controls: Array, step_size: Array, substeps: int = 1
) -> Array:
    def scan_step(state, control):
        controlled = state.at[:, 18:22].set(control)
        next_state = interval_step(model, controlled, step_size, substeps)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, controls)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)

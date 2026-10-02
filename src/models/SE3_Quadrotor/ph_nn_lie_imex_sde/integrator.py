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


def lie_imex_sde_step(
    model: DissipativeSE3HamNODE, state: Array, step_size: Array, noise: Array
) -> Array:
    """One stochastic Lie-IMEX step; ``noise`` is (batch, 6) standard normal for this step."""
    position = state[:, :3]
    rotation = state[:, 3:12].reshape(-1, 3, 3)
    twist = state[:, 12:18]
    control = state[:, 18:22]
    # Ito increment evaluated at the start of the step, shared by predictor and corrector. Constant model:
    # sqrt(h) Sigma z on the twist; state-dependent model: sqrt(h) M^-1(x) Sigma(x) z (noise in momentum).
    increment = jnp.sqrt(step_size) * jax.vmap(model.twist_noise)(state, noise)

    derivative_1 = vector_field(model, state)
    linear_damping_1, angular_damping_1 = effective_damping_blocks(model, state)
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
    derivative_2 = vector_field(model, predictor)
    linear_damping_2, angular_damping_2 = effective_damping_blocks(model, predictor)
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
    model: DissipativeSE3HamNODE,
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
def rollout(model, initial_state: Array, step_size: Array, *, steps: int) -> Array:
    def scan_step(state, _):
        next_state = lie_imex_step(model, state, step_size)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, None, length=steps)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)


@jax.jit
def rollout_control_sequence(
    model, initial_state: Array, controls: Array, step_size: Array
) -> Array:
    def scan_step(state, control):
        controlled = state.at[:, 18:22].set(control)
        next_state = lie_imex_step(model, controlled, step_size)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, controls)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)

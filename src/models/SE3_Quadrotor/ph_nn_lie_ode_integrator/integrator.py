"""Second-order Lie-Heun integration for the neural PH model."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .network import vector_field


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


def lie_heun_step(model, state: Array, step_size: Array) -> Array:
    position = state[:, :3]
    rotation = state[:, 3:12].reshape(-1, 3, 3)
    twist = state[:, 12:18]
    control = state[:, 18:22]

    derivative_1 = vector_field(model, state)
    rotation_predictor = rotation @ exp_so3(step_size * twist[:, 3:6])
    position_predictor = position + step_size * derivative_1[:, :3]
    twist_predictor = twist + step_size * derivative_1[:, 12:18]
    predictor = jnp.concatenate(
        [position_predictor, rotation_predictor.reshape(-1, 9), twist_predictor, control],
        axis=1,
    )

    derivative_2 = vector_field(model, predictor)
    rotation_next = rotation @ exp_so3(
        0.5 * step_size * (twist[:, 3:6] + twist_predictor[:, 3:6])
    )
    position_next = position + 0.5 * step_size * (
        derivative_1[:, :3] + derivative_2[:, :3]
    )
    twist_next = twist + 0.5 * step_size * (
        derivative_1[:, 12:18] + derivative_2[:, 12:18]
    )
    return jnp.concatenate(
        [position_next, rotation_next.reshape(-1, 9), twist_next, control], axis=1
    )


step = lie_heun_step


@partial(jax.jit, static_argnames=("steps",))
def rollout(model, initial_state: Array, step_size: Array, *, steps: int) -> Array:
    def scan_step(state, _):
        next_state = lie_heun_step(model, state, step_size)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, None, length=steps)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)


@jax.jit
def rollout_control_sequence(
    model, initial_state: Array, controls: Array, step_size: Array
) -> Array:
    def scan_step(state, control):
        controlled = state.at[:, 18:22].set(control)
        next_state = lie_heun_step(model, controlled, step_size)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, controls)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)

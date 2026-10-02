"""RK4 integration for the PH-NODE model."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .network import vector_field


Array = jax.Array


def rk4_step(model, state: Array, step_size: Array) -> Array:
    """Advance one step with fourth-order Runge--Kutta."""
    k1 = vector_field(model, state)
    k2 = vector_field(model, state + 0.5 * step_size * k1)
    k3 = vector_field(model, state + 0.5 * step_size * k2)
    k4 = vector_field(model, state + step_size * k3)
    return state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


step = rk4_step


def interval_step(model, state: Array, step_size: Array, substeps: int) -> Array:
    """Advance one DATA interval of ``step_size`` using ``substeps`` RK4 steps of ``step_size / substeps``.

    The control channels of ``state`` are held fixed across the sub-steps (zero-order hold), matching how the
    recorded control is defined.  substeps == 1 is exactly ``rk4_step``.
    """
    if substeps <= 1:
        return rk4_step(model, state, step_size)
    inner = step_size / substeps
    def body(carry, _):
        advanced = rk4_step(model, carry, inner)
        return advanced.at[..., 18:22].set(carry[..., 18:22]), None
    out, _ = jax.lax.scan(body, state, None, length=substeps)
    return out


@partial(jax.jit, static_argnames=("steps", "substeps"))
def rollout(model, initial_state: Array, step_size: Array, *, steps: int, substeps: int = 1) -> Array:
    def scan_step(state, _):
        next_state = interval_step(model, state, step_size, substeps)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, None, length=steps)
    return jnp.concatenate([initial_state[None], trajectory], axis=0)


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

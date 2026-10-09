"""Classical RK4 on the flattened state [x, vec(R), v, omega] (PH-NODE, the prior work's integrator).

No noise, and R is not kept on SO(3): it drifts off the group at O(h^5) per step. One observation interval dt is
``substeps`` RK4 steps under the held control.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def rk4_step(model, state: Array, step: Array) -> Array:
    control = state[18:22]

    def rate(current):
        return jnp.concatenate([model.vector_field(current), jnp.zeros_like(control)])

    k1 = rate(state)
    k2 = rate(state + 0.5 * step * k1)
    k3 = rate(state + 0.5 * step * k2)
    k4 = rate(state + step * k3)
    return state + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def interval_step(model, state: Array, control: Array, interval: float, noise: Array) -> Array:
    """Advance one observation interval with ``noise.shape[0]`` RK4 substeps (the noise values are ignored; the
    argument keeps the call signature of lie_ph.integrator.interval_step)."""
    step = jnp.asarray(interval / noise.shape[0], state.dtype)

    def body(current, _):
        return rk4_step(model, current, step), None

    final, _ = jax.lax.scan(body, state.at[18:22].set(control), noise)
    return final


def rollout(model, initial_state: Array, controls: Array, interval: float, noise: Array) -> Array:
    """(T, 22) path; ``controls`` (T-1, 4) per interval, ``noise`` (T-1, substeps, 6)."""
    def body(current, inputs):
        control, z = inputs
        following = interval_step(model, current, control, interval, z)
        return following, following

    _, path = jax.lax.scan(body, initial_state, (controls, noise))
    return jnp.concatenate([initial_state[None], path], axis=0)

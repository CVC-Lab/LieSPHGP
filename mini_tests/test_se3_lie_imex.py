"""Mathematical tests for the canonical pure-JAX SE(3) integrators."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.models.SE3_Quadrotor.ph_nn_lie_imex.integrator import lie_imex_step
from src.models.SE3_Quadrotor.ph_nn_lie_imex.network import initialize_parameters
from src.models.SE3_Quadrotor.ph_nn_lie_ode_integrator.integrator import lie_heun_step


def initial_state(batch: int = 2) -> jax.Array:
    state = np.zeros((batch, 22), dtype=np.float32)
    state[:, 3:12] = np.eye(3, dtype=np.float32).reshape(1, 9)
    state[:, 12:18] = np.asarray([1.0, -0.5, 0.25, 0.4, -0.3, 0.2])
    return jnp.asarray(state)


def parameters(*, dissipation: bool) -> dict[str, jax.Array]:
    return initialize_parameters(
        jax.random.PRNGKey(0), hidden_dim=8, initialization_gain=1.0e-3,
        mass_epsilon=1.0e-2, mass_factor_epsilon=1.0,
        dissipation_enabled=dissipation,
    )


class TestJaxLieImex(unittest.TestCase):
    def test_zero_dissipation_reduces_to_lie_heun(self) -> None:
        params = parameters(dissipation=False)
        state = initial_state()
        h = jnp.asarray(1.0 / 240.0, dtype=jnp.float32)
        np.testing.assert_allclose(
            lie_imex_step(params, state, h), lie_heun_step(params, state, h),
            rtol=2.0e-5, atol=2.0e-6,
        )

    def test_rotation_stays_on_so3(self) -> None:
        params = parameters(dissipation=True)
        state = initial_state()
        for _ in range(20):
            state = lie_imex_step(params, state, jnp.asarray(1.0 / 240.0))
        rotation = np.asarray(state[:, 3:12]).reshape(-1, 3, 3)
        np.testing.assert_allclose(
            np.swapaxes(rotation, -1, -2) @ rotation,
            np.broadcast_to(np.eye(3), rotation.shape), rtol=2.0e-5, atol=2.0e-5,
        )
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, rtol=2.0e-5, atol=2.0e-5)

    def test_gradients_through_implicit_solves_are_finite(self) -> None:
        params = parameters(dissipation=True)
        state = initial_state()
        gradient = jax.grad(
            lambda tree: jnp.mean(jnp.square(
                lie_imex_step(tree, state, jnp.asarray(1.0 / 240.0))[:, :18]
            ))
        )(params)
        self.assertTrue(all(np.all(np.isfinite(value)) for value in jax.tree_util.tree_leaves(gradient)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

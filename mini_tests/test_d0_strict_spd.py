"""Strict positive-definite inverse-mass tests for canonical JAX models."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.models.SE3_Quadrotor.ph_nn_lie_imex.network import (
    initialize_parameters, inverse_mass_1, inverse_mass_2,
)


class TestStrictSpd(unittest.TestCase):
    def test_mass_floor_and_gradients(self) -> None:
        epsilon = 1.0e-2
        params = initialize_parameters(
            jax.random.PRNGKey(0), hidden_dim=8, initialization_gain=1.0e-3,
            mass_epsilon=epsilon, mass_factor_epsilon=1.0,
            dissipation_enabled=True,
        )
        position = jax.random.normal(jax.random.PRNGKey(1), (16, 3))
        rotation = jnp.broadcast_to(jnp.eye(3).reshape(1, 9), (16, 9))
        blocks = (
            jax.vmap(inverse_mass_1, in_axes=(None, 0))(params, position),
            jax.vmap(inverse_mass_2, in_axes=(None, 0))(params, rotation),
        )
        for block in blocks:
            np.testing.assert_allclose(block, jnp.swapaxes(block, -1, -2), atol=1.0e-6)
            self.assertGreaterEqual(float(jnp.min(jnp.linalg.eigvalsh(block))), epsilon - 1.0e-6)
        gradient = jax.grad(lambda tree: sum(jnp.mean(value**2) for value in (
            jax.vmap(inverse_mass_1, in_axes=(None, 0))(tree, position),
            jax.vmap(inverse_mass_2, in_axes=(None, 0))(tree, rotation),
        )))(params)
        self.assertTrue(all(np.all(np.isfinite(value)) for value in jax.tree_util.tree_leaves(gradient)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""JAX JVP regression tests for the canonical inverse-mass networks."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.models.SE3_Quadrotor.ph_node.network import (
    initialize_parameters,
    inverse_mass_1,
)


class TestMassDerivativeJvp(unittest.TestCase):
    def test_jvp_matches_explicit_jacobian_contraction(self) -> None:
        params = initialize_parameters(
            jax.random.PRNGKey(3), hidden_dim=8, initialization_gain=1.0e-3,
            mass_epsilon=0.01, mass_factor_epsilon=1.0,
            dissipation_enabled=True,
        )
        position = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        direction = jnp.asarray([0.3, 0.5, -0.2], dtype=jnp.float32)
        function = lambda value: inverse_mass_1(params, value)
        actual = jax.jvp(function, (position,), (direction,))[1]
        expected = jnp.einsum("ijk,k->ij", jax.jacrev(function)(position), direction)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

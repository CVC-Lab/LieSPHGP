"""Regression tests for the now JAX-only quadrotor model path."""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.models.SE3_Quadrotor.ph_nn_lie_imex.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from src.models.SE3_Quadrotor.ph_nn_lie_imex.network import (
    initialize_parameters,
    vector_field,
)
from src.models.SE3_Quadrotor.ph_nn_lie_imex.integrator import rollout as imex_rollout
from src.models.SE3_Quadrotor.ph_nn_lie_ode_integrator.integrator import (
    rollout as lie_heun_rollout,
)
from src.models.SE3_Quadrotor.ph_node.integrator import rollout as rk4_rollout


class TestCanonicalJaxPath(unittest.TestCase):
    def setUp(self) -> None:
        self.params = initialize_parameters(
            jax.random.PRNGKey(5), hidden_dim=8, initialization_gain=1.0e-3,
            mass_epsilon=0.01, mass_factor_epsilon=1.0,
            dissipation_enabled=True,
        )
        state = np.zeros((2, 22), dtype=np.float32)
        state[:, 3:12] = np.eye(3, dtype=np.float32).reshape(1, 9)
        state[:, 12:18] = 0.1
        self.state = jnp.asarray(state)

    def test_vector_field_and_all_rollouts_are_finite(self) -> None:
        self.assertTrue(np.all(np.isfinite(vector_field(self.params, self.state))))
        for rollout in (rk4_rollout, lie_heun_rollout, imex_rollout):
            result = rollout(self.params, self.state, jnp.asarray(0.01), steps=2)
            self.assertEqual(result.shape, (3, 2, 22))
            self.assertTrue(np.all(np.isfinite(result)))

    def test_jax_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            save_checkpoint(path, params=self.params, extra={"step": 7})
            restored = load_checkpoint(path)
        self.assertEqual(restored["extra"]["step"], 7)
        for left, right in zip(
            jax.tree_util.tree_leaves(self.params),
            jax.tree_util.tree_leaves(restored["params"]),
        ):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main(verbosity=2)

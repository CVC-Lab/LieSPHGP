"""The trained quadrotor models of both packages behind one interface, for the comparison tools.

A run belongs to ph_node when its checkpoint config has model.integrator rk4 (PH-NODE) and to lie_ph otherwise. Both
store their parameters the same way (lie_ph.evaluate.load_run reads either).
"""

from __future__ import annotations

from lie_ph.evaluate import load_run  # noqa: F401  (re-exported for the comparison tools)
from lie_ph.integrator import rollout as lie_imex_rollout
from lie_ph.network import model_from_params as lie_ph_model
from ph_node.integrator import rollout as rk4_rollout
from ph_node.network import model_from_params as ph_node_model


def is_ph_node(config: dict) -> bool:
    return str(config["model"].get("integrator", "lie_imex")) == "rk4"


def build_model(params, setup, key, model_config: dict):
    """The model of a run with this model config (key samples GP weights, lie_ph GP family only)."""
    if str(model_config.get("integrator", "lie_imex")) == "rk4":
        return ph_node_model(params, setup, key, model_config)
    return lie_ph_model(params, setup, key, model_config)


def rollout_function(config: dict):
    """rollout(model, x0, controls, interval, noise) of the run's integrator: RK4 (ph_node) or Lie-IMEX (lie_ph)."""
    return rk4_rollout if is_ph_node(config) else lie_imex_rollout

"""PH-NODE baselines on SO(3) x R^3: the port-Hamiltonian neural ODE of the prior work (Duong & Atanasov, RSS 2021).

Two variants (config model.nn_architecture):
    xavier  PH-NODE      the MLP subnetworks of lie_ph's Lie-PH-NN models (NNSO3Model: M^-1(R), D(omega), V(R), G(R),
                         tanh, Xavier start), integrated with RK4 instead of Lie-IMEX
    duong   PH-NODE-ref  the subnetworks of their DissipativeSO3HamNODE: M^-1(R) and D(R) from PSD nets
                         9 -> 20 -> 20 -> 20 -> 6 (L L^T, the diagonal of L plus sqrt(0.1) for M^-1, nothing added for D;
                         their damping depends on the rotation, not on omega), V(R) 9 -> 20 -> 20 -> 1,
                         G(R) 9 -> 20 -> 20 -> 9, tanh, orthogonal weights with gain model.nn_init_gain (their init_gain,
                         0.5 in the pendulum scripts), biases uniform +-1/sqrt(fan_in) (torch.nn.Linear). 3 782 parameters.
The vector field (port-Hamiltonian structure on SO(3)) is lie_ph's; only the subnetworks and the integrator differ.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lie_ph.network import NN_OUTPUTS, NNSO3Model, initialize_parameters, mlp, nn_weights

Array = jax.Array

DUONG_WIDTHS = {"M": [9, 20, 20, 20, 6], "D": [9, 20, 20, 20, 6], "V": [9, 20, 20, 1], "g": [9, 20, 20, 9]}
DUONG_EPSILON = {"M": 0.1, "D": 0.0}                                    # their PSD epsilon: diag(L) + sqrt(epsilon)


def nn_architecture(model_config: Mapping[str, Any]) -> str:
    choice = str(model_config.get("nn_architecture", "xavier"))
    if choice not in ("xavier", "duong"):
        raise ValueError("model.nn_architecture must be xavier or duong")
    return choice


def initialize_orthogonal_mlp(key: Array, widths: list[int], gain: float) -> dict[str, Any]:
    """tanh MLP with torch.nn.init.orthogonal_(gain) weights and torch.nn.Linear's default uniform biases."""
    layers = []
    for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
        key_w, key_b = jax.random.split(jax.random.fold_in(key, index))
        weight = jax.nn.initializers.orthogonal(scale=gain)(key_w, (fan_in, fan_out), jnp.float32)
        bound = 1.0 / fan_in ** 0.5
        layers.append({"w": weight, "b": jax.random.uniform(key_b, (fan_out,), jnp.float32, -bound, bound)})
    return {"layers": layers}


def initialize_node_parameters(model_config: Mapping[str, Any], setup: Mapping[str, Any], key: Array) -> dict[str, Any]:
    """The four subnetworks (no noise parameters: PH-NODE is deterministic and trained without a likelihood)."""
    if nn_architecture(model_config) == "xavier":
        return initialize_parameters(model_config, setup, key, with_likelihood=False)
    return {name: initialize_orthogonal_mlp(jax.random.fold_in(key, index), DUONG_WIDTHS[name],
                                            float(model_config.get("nn_init_gain", 0.5)))
            for index, name in enumerate(NN_OUTPUTS)}


def psd_from_duong(value: Array, epsilon: float) -> Array:
    """Duong & Atanasov's PSD head: L from six numbers (diagonal = first three + sqrt(epsilon), no exp; the last three
    below the diagonal in the order (1,0), (2,0), (2,1)), returns L L^T."""
    diagonal = value[:3] + float(np.sqrt(epsilon))
    zero = jnp.zeros((), value.dtype)
    lower = jnp.stack([
        jnp.stack([diagonal[0], zero, zero]),
        jnp.stack([value[3], diagonal[1], zero]),
        jnp.stack([value[4], value[5], diagonal[2]]),
    ])
    return lower @ lower.T


class DuongSO3Model(NNSO3Model):
    """PH-NODE-ref: their PSD heads (no exp) and D a function of the rotation."""

    def inverse_mass(self, rotation_flat: Array) -> Array:
        return psd_from_duong(mlp(self.weights["M"], rotation_flat), DUONG_EPSILON["M"])

    def dissipation(self, omega: Array, rotation_flat: Array | None = None) -> Array:
        return psd_from_duong(mlp(self.weights["D"], rotation_flat), DUONG_EPSILON["D"])


def model_from_params(params: Mapping[str, Any], setup: Mapping[str, Any], model_config: Mapping[str, Any]) -> NNSO3Model:
    model_class = DuongSO3Model if nn_architecture(model_config) == "duong" else NNSO3Model
    return model_class(nn_weights(params), setup)

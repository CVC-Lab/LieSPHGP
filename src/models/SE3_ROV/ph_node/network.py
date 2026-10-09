"""PH-NODE on SE(3) x R^6 for the BlueROV2: lie_ph's port-Hamiltonian MLP model (NNSE3Model) without process noise,
integrated with RK4 (integrator.py) instead of Lie-IMEX. The vector field is lie_ph's; only the integrator and the loss
differ."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lie_ph.network import NNSE3Model, nn_weights


def model_from_params(params: Mapping[str, Any], setup: Mapping[str, Any], key=None,
                      model_config: Mapping[str, Any] | None = None) -> NNSE3Model:
    """The PH-NODE model (``key`` is ignored: no weight posterior)."""
    return NNSE3Model(nn_weights(params), setup)

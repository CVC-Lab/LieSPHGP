r"""``ph_gp_sde`` **minus the diffusion term** — the deterministic ODE twin.

Identical drift to ``ph_gp_sde`` (structured M/D/g closed forms, GP potential,
optional ``--gp_core`` residuals) with :math:`\Sigma_\theta \equiv 0`, so it
isolates exactly one axis: what the structured stochastic term contributes.
Train it with the SAME flags as the ph_gp_sde run being compared against
(including ``--gp_core``). The all-GP black-box drift is still available via
``--subnet_kind gp`` if an unstructured-GP row is ever wanted.

The physics lives in :mod:`utils.ph_network_nlink`; this module only pins the
two axes that define the variant, so a fix to the shared drift/integrator
reaches every model at once.
"""
from __future__ import annotations

import os
import sys

import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.abspath(os.path.join(_HERE, '..', 'utils'))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

from ph_network_nlink import ArmPortHamiltonian, KeyedArmModel   # noqa: E402,F401

SUBNET_KIND = 'structured'
STOCHASTIC = False
MODEL_NAME = 'ph_gp_ode'


def build_model(*, key, n: int = 2, hidden_dim: int = 32, dtype=jnp.float32,
                **kw) -> ArmPortHamiltonian:
    """Construct the ph_gp_ode network. Extra kwargs pass through unchanged."""
    return ArmPortHamiltonian(key=key, n=n, subnet_kind=SUBNET_KIND,
                              stochastic=STOCHASTIC, hidden_dim=hidden_dim,
                              dtype=dtype, **kw)

r"""MLP subnetworks, **deterministic** ODE on $SO(3)^n$.

The plainest baseline: deterministic MLP/PSD/MatrixNet subnets and no
diffusion. Against ``ph_gp_ode`` it isolates the contribution of the GP prior;
against ``ph_nn_sde`` it isolates the diffusion. Its KL is identically zero, so
``--beta_max`` has no effect.

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

SUBNET_KIND = 'nn'
STOCHASTIC = False
MODEL_NAME = 'ph_nn_ode'


def build_model(*, key, n: int = 2, hidden_dim: int = 32, dtype=jnp.float32,
                **kw) -> ArmPortHamiltonian:
    """Construct the ph_nn_ode network. Extra kwargs pass through unchanged."""
    return ArmPortHamiltonian(key=key, n=n, subnet_kind=SUBNET_KIND,
                              stochastic=STOCHASTIC, hidden_dim=hidden_dim,
                              dtype=dtype, **kw)

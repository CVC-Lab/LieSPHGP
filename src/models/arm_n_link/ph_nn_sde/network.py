r"""MLP subnetworks, **stochastic** SDE on $SO(3)^n$.

MLP subnets with a learned diffusion :math:`\Sigma_\theta(q)\in\mathbb{R}^{3n\times3}`.
The direct ablation of ``ph_gp_sde``: same dynamics and same losses minus the
variational GP prior, so a gap between the two is attributable to the prior.

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
STOCHASTIC = True
MODEL_NAME = 'ph_nn_sde'


def build_model(*, key, n: int = 2, hidden_dim: int = 32, dtype=jnp.float32,
                **kw) -> ArmPortHamiltonian:
    """Construct the ph_nn_sde network. Extra kwargs pass through unchanged."""
    return ArmPortHamiltonian(key=key, n=n, subnet_kind=SUBNET_KIND,
                              stochastic=STOCHASTIC, hidden_dim=hidden_dim,
                              dtype=dtype, **kw)

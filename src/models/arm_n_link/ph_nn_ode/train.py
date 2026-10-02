r"""Train ph_nn_ode on the $n$-link arm — thin wrapper over the shared trainer.

    python src/models/arm_n_link/ph_nn_ode/train.py --n 2 --relative_inputs

Every flag is documented in :mod:`utils.train_common`. Defaults are aligned
with the ph_gp_sde experiment (samples 64, timesteps 30, ``--lambda_pl 1``,
``--random_u_scale 2``) so every variant trains on the same cached dataset;
pass ``--g_diag 0.5 0.7 0.5`` to select the gain used in those runs.

``--beta_max`` is accepted but inert: an MLP has no weight posterior, so the
KL term is identically zero.

The two "how much physics does the NN get told?" experiments
-----------------------------------------------------------
Both are ph_nn_ode — same data, same objective, same integrator. They differ
only in **what the network outputs**:

**(1) Whole matrices.** The net emits every entry of $M^{-1}$, $D$, $g$ and the
scalar $V$; no closed form is used anywhere. `PSD_NN` emits the
$\tfrac{d(d+1)}2$ entries of a Cholesky factor and returns $LL^\top$::

    python src/models/arm_n_link/ph_nn_ode/train.py \
        --n 2 --relative_inputs --loss mse \
        --friction_coeff 0.5 --g_diag 0.5 0.7 0.5 \
        --wind_force_std 0 --obs_noise_std 0 --random_u_scale 2

**(2) Sub-components + closed form.** The net emits only the handful of physical
constants the closed forms take as inputs, and the formulas assemble the
matrices from them:

.. math::
    \mathrm{MLP}(q)\to(m,\mathbb I,\ell,c,d,\gamma,g)\ \longrightarrow\
    M_{jk}=\mathbb I_j\delta_{jk}-\!\!\sum_{i\ge\max(j,k)}\!\! m_i[u_{ij}]_\times R_j^\top R_k[u_{ik}]_\times

so e.g. the diagonal blocks get $\mathbb I_j$ directly while the off-diagonal
blocks are *computed* from $m,\ell,c$ and the relative rotations — never
predicted::

    python src/models/arm_n_link/ph_nn_ode/train.py \
        --n 2 --subnet_kind structured --nn_core --structured_potential \
        --loss mse \
        --friction_coeff 0.5 --g_diag 0.5 0.7 0.5 \
        --wind_force_std 0 --obs_noise_std 0 --random_u_scale 2

``--loss mse`` is the right objective for these two runs: with
``--obs_noise_std 0`` the per-increment likelihood is degenerate (see
``--pl_sigma_obs``), so the plain rollout MSE — mean per-link geodesic$^2$ plus
mean $\lVert\Delta\omega\rVert^2$ — is used instead, matching the
3D_SO3_Windy_Pendulum ph_nn_ode baseline exactly.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.abspath(os.path.join(_HERE, '..', 'utils'))
for _p in (_HERE, _UTILS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_common                                    # noqa: E402
from network import SUBNET_KIND, STOCHASTIC, MODEL_NAME  # noqa: E402

DEFAULT_SAVE_DIR = os.path.join(_HERE, 'data')


if __name__ == '__main__':
    args = train_common.get_args(MODEL_NAME, DEFAULT_SAVE_DIR)
    train_common.train(args, model_name=MODEL_NAME,
                       subnet_kind=SUBNET_KIND, stochastic=STOCHASTIC)

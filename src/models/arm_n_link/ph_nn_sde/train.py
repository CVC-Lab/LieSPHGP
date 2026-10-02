r"""Train ph_nn_sde on the $n$-link arm — thin wrapper over the shared trainer.

    python src/models/arm_n_link/ph_nn_sde/train.py --n 2 --relative_inputs

Every flag is documented in :mod:`utils.train_common`. Defaults are aligned
with the ph_gp_sde experiment (samples 64, timesteps 30, ``--lambda_pl 1``,
``--random_u_scale 2``) so every variant trains on the same cached dataset;
pass ``--g_diag 0.5 0.7 0.5`` to select the gain used in those runs.

``--beta_max`` is accepted but inert: an MLP has no weight posterior, so the
KL term is identically zero.

**Pendulum-parity defaults.** This wrapper defaults to ``--loss mse`` with
``--sigma_rollout_grad`` — exactly how ``3D_SO3_Windy_Pendulum/ph_nn_sde`` was
trained: plain L2+geodesic rollout MSE, with Sigma's only gradient the rollout
path (shrink pressure; its noise amplitude is not identified by this
objective, faithfully mirroring the pendulum baseline). For a diffusion that
is actually fitted to data, pass ``--loss nll --no-sigma_rollout_grad``
(restores NLL + PL and the Sigma detach — the standard grid configuration).
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
    # Pendulum ph_nn_sde parity: rollout MSE, Sigma gradient via the rollout
    # only. CLI flags still override (--loss nll re-enables NLL+PL+detach).
    args = train_common.get_args(MODEL_NAME, DEFAULT_SAVE_DIR,
                                 loss='mse', sigma_rollout_grad=True)
    train_common.train(args, model_name=MODEL_NAME,
                       subnet_kind=SUBNET_KIND, stochastic=STOCHASTIC)

# ph_node — PH-NODE baselines on SO(3) (3D windy pendulum)

The port-Hamiltonian neural ODE of the prior work (Duong & Atanasov, *Hamiltonian-based Neural ODE Networks on the SE(3)
Manifold*, RSS 2021), trained on the same data and budget as the [lie_ph](../lie_ph) models.

| model | subnetworks | integrator | loss | extras |
|---|---|---|---|---|
| PH-NODE | lie_ph's tanh MLPs: $M^{-1}(R)$, $D(\omega)$, $V(R)$, $G(R)$ (Xavier start) | RK4 | trajectory loss (`rollout`) | — |
| PH-NODE-ref | their DissipativeSO3HamNODE: PSD heads, $D(R)$, orthogonal start (gain 0.5), 3 782 parameters | RK4 | their `rotmat_L2_geodesic_loss` (`rollout_reference`) | constant Adam 1e-3, weight decay 1e-4, no clipping; optional $M^{-1}\to I$ pretraining |

- The vector field (the port-Hamiltonian structure on $SO(3)$) is lie_ph's; only the subnetworks and the integrator differ.
- RK4 runs on the flattened state $[\mathrm{vec}(R),\ \omega]$, so $R$ slowly leaves $SO(3)$.
- There is no process noise and no likelihood.

| file | role |
|---|---|
| [network.py](network.py) | PH-NODE-ref subnetworks (`DuongSO3Model`), parameter start, `model_from_params` |
| [integrator.py](integrator.py) | RK4 step and rollout |
| [losses.py](losses.py) | `rollout_loss`, `rollout_loss_reference` |
| [config.py](config.py) | strict config (lie_ph's checks; `model.integrator: rk4`) |
| [train.py](train.py) | trainer (fixed budget, final checkpoint) |
| [evaluate.py](evaluate.py) | ground-truth evaluation (lie_ph's protocol, RK4 rollouts) |

```bash
python src/models/3D_SO3_Windy_Pendulum/ph_node/train.py --config src/models/3D_SO3_Windy_Pendulum/configs/<experiment>/<PH-NODE config>.yaml
python src/models/3D_SO3_Windy_Pendulum/ph_node/evaluate.py --run <run_dir>
```

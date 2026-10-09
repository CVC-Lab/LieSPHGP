# ph_node — PH-NODE baseline on SE(3) (BlueROV2)

The port-Hamiltonian neural ODE of the prior work (Duong & Atanasov, RSS 2021; the real-flight recipe of LieGroupHamDL),
trained on the same Marinarium data as [lie_ph](../lie_ph).

- **Model:** lie_ph's MLP port-Hamiltonian model (`NNSE3Model`), no process noise, no likelihood.
- **Integrator:** classical RK4 on the flattened state $[x,\ \mathrm{vec}(R),\ v,\ \omega]$.
- **Loss:** the trajectory loss (RK4 rollout from each window's first sample, mean of position, geodesic attitude,
  velocity and angular-velocity errors), plus the optional `training.l1_coefficient` $c$:
  $c\,(\overline{|G|}+\overline{|D_v|}+\overline{|D_\omega|})$, the original recipe's sparsity term.

| file | role |
|---|---|
| [network.py](network.py) | `model_from_params`: lie_ph's `NNSE3Model` |
| [integrator.py](integrator.py) | RK4 step and rollout |
| [losses.py](losses.py) | `rollout_loss`, `l1_structure` |
| [config.py](config.py) | strict config: schema `ph_node/rov/v1`, `integrator: rk4`, `family: nn`, `wind: false`, `loss: rollout` |
| [train.py](train.py) | lie_ph's training loop with these pieces and the L1 term |
| [evaluate.py](evaluate.py) | lie_ph's evaluation with this model |

```bash
python src/models/SE3_ROV/ph_node/train.py --config src/models/SE3_ROV/configs/marinarium-05-10-2026/PH-NODE-ORIG_REAL-MARINARIUM-COMMANDS-5s.yaml
python src/models/SE3_ROV/ph_node/evaluate.py --run <run_dir>
```

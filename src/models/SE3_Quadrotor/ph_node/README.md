# ph_node — PH-NODE baseline on SE(3) (quadrotor)

The port-Hamiltonian neural ODE of the prior work (Duong & Atanasov, *Hamiltonian-based Neural ODE Networks on the SE(3)
Manifold*, RSS 2021), trained on the same data, budget and training loop as the [lie_ph](../lie_ph) models.

- **Model:** lie_ph's MLP port-Hamiltonian model (`NNSE3Model`: $M_1^{-1}(x)$, $M_2^{-1}(R)$, $D_v(v)$, $D_\omega(\omega)$,
  $V(x)$, $G([\mathrm{vec}R,x])$, tanh, Xavier start), no process noise, no likelihood.
- **Integrator:** classical RK4 on the flattened state $[x,\ \mathrm{vec}(R),\ v,\ \omega]$, so $R$ slowly leaves $SO(3)$.
- **Loss:** the trajectory loss: RK4 rollout from each window's first (noisy) sample, mean of
  $\lVert\hat x-x\rVert^2+\theta(\hat R,R)^2+\lVert\hat v-v\rVert^2+\lVert\hat\omega-\omega\rVert^2$.

| file | role |
|---|---|
| [network.py](network.py) | `model_from_params`: lie_ph's `NNSE3Model` |
| [integrator.py](integrator.py) | RK4 step and rollout |
| [losses.py](losses.py) | `rollout_loss` |
| [config.py](config.py) | strict config: lie_ph's checks, schema `ph_node/quadrotor/v1`, `integrator: rk4`, `family: nn`, `wind: false` |
| [train.py](train.py) | lie_ph's training loop with these pieces |
| [evaluate.py](evaluate.py) | lie_ph's evaluation with RK4 rollouts |

```bash
python src/models/SE3_Quadrotor/ph_node/train.py --config src/models/SE3_Quadrotor/configs/<experiment>/PH-NODE_<...>.yaml
python src/models/SE3_Quadrotor/ph_node/evaluate.py --run <run_dir>
```

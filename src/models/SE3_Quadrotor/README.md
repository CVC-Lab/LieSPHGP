# SE3_Quadrotor — learning quadrotor dynamics on SE(3)

Models that learn the dynamics of a quadrotor from noisy observations of its position $x$, attitude $R\in SO(3)$, body
velocity $v$ and body angular velocity $\omega$, driven by the wrench $u=[T,\tau]$, with no physical constant given to
them. One model code serves two datasets; only the config differs.

## The system

$$\dot x = R v,\quad \dot R = R\hat\omega,\quad
m\,dv = (m\,v\times\omega - m g R^\top e_3 + T e_3 - D_v v)\,dt + \Sigma_v\,dW,\quad
J\,d\omega = (J\omega\times\omega + \tau - D_\omega\omega)\,dt + \Sigma_\omega\,dW .$$

| dataset | source | damping | wind | observations |
|---|---|---|---|---|
| PyBullet (DampConst-Wind / DampRate-Wind) | `envs/quadrotor_se3_pybullet`: CF2P, DSL PID, 300 flights of 5 s at 50 Hz with vertical excitation | $c\,m\,v$ or $c\,m(1+\lVert v\rVert)v$ (and the same on $\omega$), $c=0.5$ | 0.5 on both twist channels | + noise 0.25 |
| IDSIA | `envs/quadrotor_se3_idsia`: real Crazyflie 2.1 Brushless flights, 100 Hz, wrench from measured rotor speeds | unknown | real | motion capture |

## Models

| model | package | subnetworks | integrator | process noise | loss |
|---|---|---|---|---|---|
| **Lie-PH-GP-SDE** | [lie_ph](lie_ph) | level + variational GP | Lie-IMEX | learned $\Sigma$ | EKF marginal likelihood + KL |
| Lie-PH-GP-ODE | [lie_ph](lie_ph) | level + variational GP | Lie-IMEX | none | EKF marginal likelihood + KL |
| Lie-PH-NN-SDE | [lie_ph](lie_ph) | tanh MLP | Lie-IMEX | learned $\Sigma$ | EKF marginal likelihood |
| Lie-PH-NN-ODE | [lie_ph](lie_ph) | tanh MLP | Lie-IMEX | none | trajectory loss |
| PH-NODE | [ph_node](ph_node) | tanh MLP | RK4 | none | trajectory loss (the prior work's) |

The port-Hamiltonian models learn $M_1^{-1}(x)$, $M_2^{-1}(R)$, $D_v(v)$, $D_\omega(\omega)$, $V(x)$ and $G([\mathrm{vec}R, x])$.
Only products such as $M^{-1}D$, $M^{-1}G$ and $M^{-1}\nabla V$ are identifiable (one scale gauge per block).

## Folders

| folder | content |
|---|---|
| [lie_ph/](lie_ph) | our models: network, Lie-IMEX integrator, EKF / trajectory losses, `train.py`, `evaluate.py` |
| [ph_node/](ph_node) | the PH-NODE baseline: RK4 and its trajectory loss (reuses lie_ph's model class and training loop) |
| [comparison/](comparison) | `campaign.py` (configs + GPU runner), `report.py` (open loop), `closed_loop.py` (tracking in PyBullet), `models.py` |
| [configs/](configs) | `pybullet-campaign-06-10-2026/` (Lie-PH-GP-SDE + PH-NODE, both damping laws, `base.yaml`) and `IDSIA/` (all 5 models) |

A config's `schema` names its trainer: `lie_ph/quadrotor/v1` → `lie_ph/train.py`, `ph_node/quadrotor/v1` → `ph_node/train.py`.

## Usage

Build the data first: `envs/quadrotor_se3_pybullet/README.md` (simulated) or `envs/quadrotor_se3_idsia/README.md` (real).

Train (GPU from the config's `runtime.gpu` unless `CUDA_VISIBLE_DEVICES` is set; 5000 steps, final checkpoint):

```bash
C=src/models/SE3_Quadrotor/configs
python src/models/SE3_Quadrotor/lie_ph/train.py  --config $C/pybullet-campaign-06-10-2026/Lie-PH-GP-SDE_DampConst-Wind_noise0p25.yaml
python src/models/SE3_Quadrotor/ph_node/train.py --config $C/pybullet-campaign-06-10-2026/PH-NODE_DampConst-Wind_noise0p25.yaml
python src/models/SE3_Quadrotor/lie_ph/train.py  --config $C/IDSIA/Lie-PH-GP-SDE_IDSIA.yaml
python src/models/SE3_Quadrotor/ph_node/train.py --config $C/IDSIA/PH-NODE_IDSIA.yaml
```

Evaluate a run (simulated: against the true operators; IDSIA: against the published constants):

```bash
python src/models/SE3_Quadrotor/lie_ph/evaluate.py  --run <run_dir>      # Lie-PH runs
python src/models/SE3_Quadrotor/ph_node/evaluate.py --run <run_dir>      # PH-NODE runs
```

PyBullet campaign and its reports:

```bash
python src/models/SE3_Quadrotor/comparison/campaign.py run --gpu 0
python src/models/SE3_Quadrotor/comparison/report.py --setting DampRate-Wind
JAX_PLATFORMS=cpu python src/models/SE3_Quadrotor/comparison/closed_loop.py --setting DampRate-Wind
```

Results: `experiments/quadrotor/campaign_06-10-2026/` (PyBullet: runs, `results_*.jsonl`, `reports/`) and
`experiments/quadrotor/IDSIA_06-10-2026/` (IDSIA runs).

**Protocol (every model):** float32; no pretraining and no physical prior; a fixed budget of 5000 steps with the final
checkpoint; training reads only the noisy data. GPU training is not bit-reproducible from run to run; on the CPU it is.

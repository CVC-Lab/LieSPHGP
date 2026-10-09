# 3D_SO3_Windy_Pendulum — learning a stochastic pendulum on SO(3)

Models that learn the dynamics of a 3-D pendulum in wind from noisy observations of its attitude $R\in SO(3)$ and body
angular velocity $\omega\in\mathbb R^3$, with no physical constant given to them.

## The system

The simulator is `envs/pendulum_so3/windy_pendulum_3d.py`. The bob hangs on a rod of length $l$ (mass $m$, inertia $J = ml^2 I$),
is driven by a torque $G u$ with $G=\mathrm{diag}(0.5,\,0.7,\,0.8)$, damped by friction and pushed by wind:

$$dR = R\,\hat\omega\,dt,\qquad J\,d\omega = \big(J\omega\times\omega + \tau_g(R) - \tau_D(R,\omega) + G u\big)\,dt + \Sigma\,dW .$$

| damping law | $\tau_D$ | experiments |
|---|---|---|
| constant | $c\,\omega$ | campaign (DampConst) |
| rate-dependent | $c\,(1+\lVert\omega\rVert)\,\omega$ | campaign (DampRate) |
| varying | $c\,(1+0.5\,h+0.5\tanh\lVert\omega\rVert)\,\omega$, $h=(1-R_{33})/2$ | DampVarying-Wind |

Wind: a world-frame force $\sigma\,dW$ on the bob ($\sigma = 0.5$, or 0 for the wind-free "NoWind" sets). Observations
add noise $R\,\mathrm{Exp}(\epsilon_R)$, $\omega+\epsilon_\omega$ with $\epsilon\sim\mathcal N(0, s^2 I)$.

## Models

| model | package | subnetworks | integrator | process noise | loss |
|---|---|---|---|---|---|
| **Lie-PH-GP-SDE** | [lie_ph](lie_ph) | level + variational GP | Lie-IMEX | learned $\Sigma$ | EKF marginal likelihood + KL |
| Lie-PH-GP-ODE | [lie_ph](lie_ph) | level + variational GP | Lie-IMEX | none | EKF marginal likelihood + KL |
| Lie-PH-NN-SDE | [lie_ph](lie_ph) | tanh MLP | Lie-IMEX | learned $\Sigma$ | EKF marginal likelihood |
| Lie-PH-NN-ODE | [lie_ph](lie_ph) | tanh MLP | Lie-IMEX | none | trajectory loss |
| PH-NODE | [ph_node](ph_node) | tanh MLP | RK4 | none | trajectory loss (the prior work's) |
| PH-NODE-ref | [ph_node](ph_node) | Duong & Atanasov's network | RK4 | none | their trajectory loss |
| NeuralSDE | [neural_sde](neural_sde) | unstructured drift + diffusion MLPs | Heun | learned | path MSE + geodesic |

The port-Hamiltonian models share one structure: $p = M(R)\,\omega$ and
$\dot p = p\times\omega + \sum_i r_i\times\partial V/\partial r_i - D\,\omega + G(R)\,u$, with the subnetworks
$M^{-1}(R)$, $D$, $V(R)$, $G(R)$ learned. Only the products $M^{-1}D$, $M^{-1}G$ and $M^{-1}\nabla V$ are
identifiable (gauge $(M^{-1},V,D,G)\to(cM^{-1},V/c,D/c,G/c)$).

## Folders

| folder | content |
|---|---|
| [lie_ph/](lie_ph) | our models: network, Lie-IMEX integrator, EKF / trajectory losses, `train.py`, `evaluate.py` |
| [ph_node/](ph_node) | the PH-NODE baselines: RK4, their network and loss, `train.py`, `evaluate.py` |
| [neural_sde/](neural_sde) | the NeuralSDE baseline: `train.py`, `network.py` |
| [comparison/](comparison) | `campaign.py` (configs + GPU runner), `report.py` (open loop), `closed_loop.py` (swing-up to upright), `closed_loop_figure.py` |
| [configs/](configs) | every config, one folder per experiment: `DampVarying-Wind/`, `old/campaign-04-10-2026/`, `old/gphyper-05-10-2026/`, `old/dev/` |

A config's `schema` names its trainer: `lie_ph/pendulum/v1` → `lie_ph/train.py`, `ph_node/pendulum/v1` →
`ph_node/train.py`; the `NeuralSDE_*.yaml` files hold the arguments of `neural_sde/train.py`.

## Usage

Build the dataset first (see [datasets/README.md](../../../datasets/README.md)):

```bash
python envs/pendulum_so3/datagen/generate_dataset.py --config envs/pendulum_so3/datagen/configs/<name>.yaml
```

Train (GPU from the config's `runtime.gpu` unless `CUDA_VISIBLE_DEVICES` is set; 5000 steps, final checkpoint):

```bash
C=src/models/3D_SO3_Windy_Pendulum/configs/DampVarying-Wind
python src/models/3D_SO3_Windy_Pendulum/lie_ph/train.py     --config $C/Lie-PH-GP-SDE_DampVarying-Wind_noise0p25.yaml
python src/models/3D_SO3_Windy_Pendulum/ph_node/train.py    --config $C/PH-NODE_DampVarying-Wind_noise0p25.yaml
python src/models/3D_SO3_Windy_Pendulum/neural_sde/train.py --config $C/NeuralSDE_DampVarying-Wind_noise0p25.yaml
```

Evaluate a run against the simulator's true operators (the only step that reads the ground truth):

```bash
python src/models/3D_SO3_Windy_Pendulum/lie_ph/evaluate.py  --run <run_dir>      # Lie-PH runs
python src/models/3D_SO3_Windy_Pendulum/ph_node/evaluate.py --run <run_dir>      # PH-NODE runs
```

The campaign (all models × 4 settings × 6 noise levels) and its reports:

```bash
python src/models/3D_SO3_Windy_Pendulum/comparison/campaign.py run --gpu 0
python src/models/3D_SO3_Windy_Pendulum/comparison/report.py --setting DampRate-Wind --noise 0.5
JAX_PLATFORMS=cpu python src/models/3D_SO3_Windy_Pendulum/comparison/closed_loop.py --task upright --setting DampRate-Wind --noise 0.5
```

Results: `experiments/pendulum_so3/campaign_04-10-2026/` (runs, `results_*.jsonl`, `reports/`, `paper_tables/`) and
`experiments/pendulum_so3/varying_08-10-2026/` for the DampVarying-Wind runs.

**Protocol (every model):** float32; no pretraining (except PH-NODE-ref's $M^{-1}\to I$, the reference's own recipe)
and no physical prior; a fixed budget of 5000 steps with the final checkpoint; training reads only the noisy data.
GPU training is not bit-reproducible from run to run; on the CPU it is.

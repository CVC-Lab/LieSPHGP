# LieSPHGP

Lie group stochastic port-Hamiltonian Gaussian processes: learning the dynamics of mechanical systems on $SO(3)$ and
$SE(3)$ from noisy observations, with no physical constant given to the model.

The model is a port-Hamiltonian system $\dot p = J(q,p)\,\nabla H - D\,\nabla_p H + G(q)\,u$ with process noise, whose
inverse mass $M^{-1}$, potential $V$, damping $D$ and input matrix $G$ are each a Matérn random-feature Gaussian process.
It is integrated with **Lie-IMEX** (a Lie group step that keeps $R\in SO(3)$ exactly, with implicit damping) and trained
on the **EKF marginal likelihood** of the observations plus the KL term of the variational GP weights.

## Repository layout

| path | content |
|---|---|
| [envs/](envs) | the four systems and their dataset generators (`envs/<system>/datagen/`, one config per dataset) |
| [datasets/](datasets/README.md) | generated data only, not distributed: rebuild it with the generators |
| [src/models/3D_SO3_Windy_Pendulum/](src/models/3D_SO3_Windy_Pendulum/README.md) | pendulum on $SO(3)$ in wind |
| [src/models/SE3_Quadrotor/](src/models/SE3_Quadrotor/README.md) | quadrotor on $SE(3)$: PyBullet simulation and IDSIA real flights |
| [src/models/SE3_ROV/](src/models/SE3_ROV/README.md) | BlueROV2 Heavy underwater vehicle, real Marinarium recordings |
| [src/utils/](src/utils) | shared JAX helpers (MLP, losses, pickling) used by the NeuralSDE baseline |
| [paper_results/](paper_results/README.md) | the configs behind every table and figure of the paper |

Each system folder under `src/models/` has the same layout:

| folder | content |
|---|---|
| `lie_ph/` | our models: network, Lie-IMEX integrator, EKF loss, `train.py`, `evaluate.py` |
| `ph_node/` | the PH-NODE baseline: RK4 integrator and trajectory loss, `train.py`, `evaluate.py` |
| `comparison/` | open-loop reports, closed-loop control tests, campaign runners |
| `configs/<experiment>/` | one YAML per model and dataset |

A config's `schema` names its trainer: `lie_ph/<system>/v1` → `lie_ph/train.py`, `ph_node/<system>/v1` → `ph_node/train.py`.

## Models

| model | package | subnetworks | integrator | process noise | loss |
|---|---|---|---|---|---|
| **Lie-PH-GP-SDE** (LieSPHGP) | `lie_ph` | variational GP | Lie-IMEX | learned | EKF marginal likelihood + KL |
| Lie-PH-GP-ODE | `lie_ph` | variational GP | Lie-IMEX | none | EKF marginal likelihood + KL |
| Lie-PH-NN-SDE | `lie_ph` | MLP | Lie-IMEX | learned | EKF marginal likelihood |
| Lie-PH-NN-ODE | `lie_ph` | MLP | Lie-IMEX | none | trajectory loss |
| PH-NODE | `ph_node` | MLP | RK4 | none | trajectory loss (prior work) |
| NeuralSDE | `neural_sde` (pendulum) | unstructured drift and diffusion MLPs | Heun | learned | path MSE + geodesic |

## Systems and data

| system | environment | data |
|---|---|---|
| 3-D pendulum in wind, $SO(3)$ | [envs/pendulum_so3](envs/pendulum_so3/README.md) | simulated (Lie-IMEX), constant / rate-dependent / varying damping |
| Crazyflie quadrotor, $SE(3)$ | [envs/quadrotor_se3_pybullet](envs/quadrotor_se3_pybullet/README.md) | simulated in PyBullet with wind and damping (needs a clone of gym-pybullet-drones) |
| Crazyflie 2.1 Brushless, real flights | [envs/quadrotor_se3_idsia](envs/quadrotor_se3_idsia/README.md) | IDSIA nano-drone benchmark (raw data cloned separately) |
| BlueROV2 Heavy, real tank recordings | [envs/rov_se3_marinarium](envs/rov_se3_marinarium/README.md) | KTH Marinarium dataset (raw data cloned separately) |

Third-party simulators and raw data are not redistributed here; each environment README says what to clone and how to
convert it.

## Installation

Tested with Python 3.13.

```bash
pip install -r requirements.txt
pip install "jax[cuda12]"        # optional: JAX on an NVIDIA GPU (the requirements install the CPU version)
```

## Quick start (pendulum)

Commands are run from the repository root.

```bash
# 1. dataset: rate-dependent damping, wind, observation noise levels 0 ... 0.75
python envs/pendulum_so3/datagen/generate_dataset.py \
    --config envs/pendulum_so3/datagen/configs/SDE-DiffusionConstant-0p5-DissipationRateDependent-0p5-5s-G0p5-0p7-0p8.yaml

# 2. train LieSPHGP (5000 steps; GPU from the config's runtime.gpu unless CUDA_VISIBLE_DEVICES is set)
python src/models/3D_SO3_Windy_Pendulum/lie_ph/train.py \
    --config paper_results/configs/pendulum/Lie-PH-GP-SDE_DampRate-Wind_noise0p5.yaml

# 3. evaluate the run against the ground truth
python src/models/3D_SO3_Windy_Pendulum/lie_ph/evaluate.py --run <run_dir>
```

The run folder (printed at the start of training) holds the config copy, the training log and the checkpoints.
The system READMEs list the commands for the other models, the reports and the closed-loop tests.

## Reproducing the paper

[paper_results/README.md](paper_results/README.md) maps every table and figure to its model configs, dataset configs and
report commands.

All experiments use float32, no physical priors or pretraining, a fixed budget of 5000 training steps and the final
checkpoint.

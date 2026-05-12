# LieSPHGP

Stochastic Port-Hamiltonian Neural Networks for learning dynamics on Lie groups (SO(3), SE(3)).

## What's inside

- **envs/** — Gym environment for a 3D windy pendulum on SO(3).
- **datasets/** — Scripts to generate and plot trajectory data.
- **src/models/** — Models trained on the windy pendulum:
  - `ph_nn_ode_v2` — port-Hamiltonian neural ODE
  - `ph_gp_ode_v2` / `ph_gp_sde` — Gaussian-process variants
  - `neural_sde` — neural SDE baseline
- **src/utils/** — Shared helpers, including JAX implementations of GPs, neural nets, and Lie-group integrators.

## Quick start

1. Generate data:
   ```bash
   python datasets/windy_pendulum_3d_datagen.py
   ```
2. Train a model, e.g.:
   ```bash
   python src/models/3D_SO3_Windy_Pendulum/ph_nn_ode_v2/train.py
   ```
3. Compare models:
   ```bash
   python src/models/3D_SO3_Windy_Pendulum/ode_make_comparison_v2.py
   ```

## Requirements

Python 3.10+, NumPy, JAX, PyTorch, Gymnasium.


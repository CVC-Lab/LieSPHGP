# pendulum_so3

3-D windy pendulum on SO(3): state (R, w), J = m l^2 I, torque input G u, friction, a deterministic external force
and a Stratonovich wind noise sigma dW on the bob.

| Path | Role |
|---|---|
| `windy_pendulum_3d.py` | the env. `integrator`: `lie_imex` (default, friction implicit) or `lie_heun` (explicit); `substeps` per `dt` |
| `datagen/generate_dataset.py` + `datagen/config.yaml` | the dataset generator: one config -> `datasets/PENDULUM-DATASET-<name>/<name>_{clean,obs-noise<s>}.pkl` |
| `datagen/windy_pendulum_3d_datagen.py` | sampling + noise code (used by the generator) and the older `get_dataset()` the trainers call |
| `plots/` | dataset phase-space plot (`3d_pendulum_trajectory_dataset_plot.py <pkl>`), friction plot, friction video |
| `assets/` | icons used in renders |

Train on a generated dataset with `--dataset_path datasets/PENDULUM-DATASET-<name>/<name>_obs-noise0p05.pkl`
(ph_gp_ode, ph_gp_sde, ph_nn_ode, ph_nn_sde, neural_sde and ph_gp_ode/verify_losses.py). Without it they keep
using `get_dataset()` and `datasets/windy_pendulum_3d/`.

Lie-IMEX step (per substep h; D = J^-1 diag(f(R, w)), a = all angular accelerations except friction, s = noise increment):
    predictor  (I + h D_1) w_p = w_n + h a_1 + s_1,                       R_p = R_n exp(h w_n)
    corrector  (I + h/2 D_2) w_{n+1} = w_n + h/2 (a_1 + a_2 - D_1 w_n) + (s_1 + s_2)/2,   R_{n+1} = R_n exp(h/2 (w_n + w_p))

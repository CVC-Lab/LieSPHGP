# pendulum_so3

3-D windy pendulum on SO(3): state (R, w), J = m l^2 I, torque input G u, friction, a deterministic external force
and a Stratonovich wind noise sigma dW on the bob.

| Path | Role |
|---|---|
| `windy_pendulum_3d.py` | the env, integrated with Lie-IMEX (friction implicit, the rest explicit); `substeps` per `dt` |
| `datagen/generate_dataset.py` + `datagen/config.yaml` | the dataset generator: one config -> `datasets/PENDULUM-DATASET-<name>/<name>_{clean,obs-noise<s>}.pkl` |
| `datagen/configs/` | the configs of the four released `PENDULUM-DATASET-*-G0p5-0p7-0p8` datasets |
| `datagen/windy_pendulum_3d_datagen.py` | sampling + noise code (used by the generator) and the older `get_dataset()` the trainers call |
| `plots/` | dataset phase-space plot (`3d_pendulum_trajectory_dataset_plot.py <pkl>`), friction plot, friction video (written to `outputs/pendulum_so3/`) |

Requires `numpy`, `gymnasium`, `pyyaml` and `matplotlib`; the two video scripts also need the `ffmpeg` binary.

Generate a dataset:

    python envs/pendulum_so3/datagen/generate_dataset.py --config envs/pendulum_so3/datagen/configs/<name>.yaml

Data layout: `x`, `test_x`, `test_x_noisy` have shape (control batches, T, trajectories, 15); row k is
[vec(R_k) (9, row-major), w_k (3, body frame), u (3)]. The u in row k+1 is the control applied during k -> k+1
(held constant over that interval); row 0 repeats u_0, the control of the first interval.

The models that train on these datasets are in `src/models/3D_SO3_Windy_Pendulum/` (see its README): their configs
name the dataset pickle, e.g. `datasets/PENDULUM-DATASET-<name>/<name>_obs-noise0p25.pkl`.

Lie-IMEX step (per substep h; D = J^-1 diag(f(R, w)), a = all angular accelerations except friction, s = noise increment):
    predictor  (I + h D_1) w_p = w_n + h a_1 + s_1,                       R_p = R_n exp(h w_n)
    corrector  (I + h/2 D_2) w_{n+1} = w_n + h/2 (a_1 + a_2 - D_1 w_n) + (s_1 + s_2)/2,   R_{n+1} = R_n exp(h/2 (w_n + w_p))

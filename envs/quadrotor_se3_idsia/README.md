# quadrotor_se3_idsia

Real Crazyflie 2.1 Brushless flights from the IDSIA nano-quadrotor system-identification benchmark
(motion-capture arena, 100 Hz). There is no simulator here: the flights are recorded data, converted to the
22-column layout the SE(3) models use (`x_w, vec(R), v_b, omega_b, u`).

| File | Role |
|---|---|
| `datagen/convert_idsia.py` | raw benchmark CSVs (`envs/quadrotor_se3_idsia/idsia_raw/data`) -> `datasets/QUADROTOR-DATASET-IDSIA/*.pkl`; `--input wrench` (default) or `rotor2` |
| `datagen/validate_idsia_input_reconstruction.py` | checks that the reconstructed wrench matches the measured motion (wrench dataset only) |

The models are trained on it with `src/models/SE3_Quadrotor/configs/IDSIA/` (see `src/models/SE3_Quadrotor/README.md`).

## Setup

The raw data is not part of this repository and is not redistributed (the benchmark repository has no licence file).
Clone it at the commit the datasets were made with, then convert:

```bash
git clone https://github.com/idsia-robotics/nanodrone-sysid-benchmark envs/quadrotor_se3_idsia/idsia_raw
git -C envs/quadrotor_se3_idsia/idsia_raw checkout 2d921b57d166fe2debe08a5d39bd07297c5abc39
python envs/quadrotor_se3_idsia/datagen/convert_idsia.py
python envs/quadrotor_se3_idsia/datagen/validate_idsia_input_reconstruction.py
```

The conversion needs `pandas` (in `requirements.txt`).

## Citation

If you use this data, cite the benchmark:

R. Busetto, E. Cereda, M. Forgione, G. Maroni, D. Piga, D. Palossi, "Nonlinear System Identification for a Nano-drone
Benchmark", Control Engineering Practice, 2026. https://arxiv.org/abs/2512.14450

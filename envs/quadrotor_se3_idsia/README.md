# quadrotor_se3_idsia

Real Crazyflie 2.1 Brushless flights from the IDSIA nano-quadrotor system-identification benchmark
(motion-capture arena, 100 Hz). There is no simulator here: the flights are recorded data, converted to the
22-column layout the SE(3) models use (`x_w, vec(R), v_b, omega_b, u`).

| File | Role |
|---|---|
| `datagen/convert_idsia.py` | raw benchmark CSVs (`envs/quadrotor_se3_idsia/idsia_raw/data`) -> `datasets/QUADROTOR-DATASET-IDSIA/*.pkl`; `--input wrench` or `rotor2` |
| `datagen/validate_idsia_input_reconstruction.py` | checks that the reconstructed input u matches the measured motion |

The raw clone lives in `envs/quadrotor_se3_idsia/idsia_raw/` (git-ignored). Model-analysis scripts for this dataset are in
`src/models/SE3_Quadrotor/comparision/idsia/`; the conversion is described in `notes/Dataset/real_flight_idsia_experiment.md`.

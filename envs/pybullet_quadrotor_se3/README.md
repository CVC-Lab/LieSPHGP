# pybullet_quadrotor_se3

The gym-pybullet-drones quadrotor: PyBullet physics (`CtrlAviary`, CF2P) driven by `DSLPIDControl`.
Every current quadrotor training and evaluation dataset is generated here.

| Path | Role |
|---|---|
| `gym-pybullet-drones/` | vendored clone (git-ignored), found through `sys.path`, not pip-installed |
| `datagen/generate_dataset.py` + `datagen/config.yaml` | the one dataset generator and its one config: trajectory set (hard / eval / hard+eval), dissipation and wind laws per channel, seeds, noise, output name |
| `datagen/generate_reference_flights.py` | clean PID reference flights → `datasets/QUADROTOR-EVAL-REFERENCE/` |
| `plots/` | dataset plots (HARD clean vs noisy, obs-noise trajectories) |

`src/models/SE3_Quadrotor/comparision/report_controller.py` also uses this simulator as the closed-loop plant.
See `datasets/README.md` for which script made which dataset.

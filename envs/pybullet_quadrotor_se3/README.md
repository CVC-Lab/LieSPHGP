# pybullet_quadrotor_se3

The gym-pybullet-drones quadrotor: PyBullet physics (`CtrlAviary`, CF2P) driven by `DSLPIDControl`.
Every current quadrotor training and evaluation dataset is generated here.

| Path | Role |
|---|---|
| `gym-pybullet-drones/` | vendored clone (git-ignored), found through `sys.path`, not pip-installed |
| `datagen/generate_quadrotor_{hard_v2,evalset,wind,wind25,windsde}.py` | dataset generators; input configs in `datagen/configs/` |
| `datagen/generate_reference_flights.py` | clean PID reference flights → `datasets/QUADROTOR-EVAL-REFERENCE/` |
| `datagen/pybullet_quadrotor_datagen.py` | early D0 PyBullet generator |
| `datagen/real/` | converters and input checks for the real IDSIA and NanoBench flights (`nanobench_pybullet_roundtrip.py` replays them through PyBullet) |
| `plots/` | dataset plots (HARD clean vs noisy, NanoBench analysis, obs-noise trajectories) |

`src/models/SE3_Quadrotor/comparision/report_controller.py` also uses this simulator as the closed-loop plant.
See `datasets/README.md` for which script made which dataset.

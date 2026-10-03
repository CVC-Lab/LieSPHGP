# datasets/

Data only. This folder holds `.pkl` files plus the metadata written alongside them
(`*_config_used.yaml`, `*_generation.log`, `*_audits.json`, `*_dataset_analysis.pdf`).
It contains no code. Every script that writes these files lives under `envs/<system>/datagen/`.
Every script that plots them lives under `envs/<system>/plots/`.
Commands are run from the project root.

| Folder | Made by | Regenerate |
|---|---|---|
| `windy_pendulum_3d/` | `envs/pendulum_so3/datagen/windy_pendulum_3d_datagen.py` (+ `gen_varfric_randu2_5noise.py` for the 5 noise levels) | `python envs/pendulum_so3/datagen/windy_pendulum_3d_datagen.py --help` |
| `windy_pendulum_3d_v2/` | same generator, older settings (`var_fricTrue`, `uScale2p0`) | — |
| `windy_arm_nlink/` | `envs/arm_nlink_so3/datagen/windy_arm_nlink_datagen.py` | `python envs/arm_nlink_so3/datagen/windy_arm_nlink_datagen.py --help` |
| `QUADROTOR-DATASET-HARD/` | `envs/pybullet_quadrotor_se3/datagen/generate_dataset.py` (the former HARD-V5; see `_note` in its `config_used.yaml`) | `trajectory_set: hard`, builtin damping, no wind, `kick_torque: last_step` |
| `QUADROTOR-DATASET-EVALSET/` | same generator | `trajectory_set: eval`, `seeds.eval: [498, 499]` |
| `QUADROTOR-DATASET-WIND/` | same generator | `trajectory_set: hard+eval`, linear `ou` wind, 0.1 of weight |
| `QUADROTOR-DATASET-WIND25/` | same generator | as WIND, 0.25 of weight |
| `QUADROTOR-DATASET-WINDSDE/` | same generator (8 variants, one `*_config_used.yaml` each) | constant damping, white wind (constant or speed/rate dependent) |
| `QUADROTOR-EVAL-REFERENCE/` | `envs/pybullet_quadrotor_se3/datagen/generate_reference_flights.py` | `... generate_reference_flights.py --duration-seconds 3.0` |
| `QUADROTOR-DATASET-IDSIA/` | real flights, converted by `envs/idsia_quadrotor_se3/datagen/convert_idsia.py` (raw data in `tmp/idsia_raw/`) | `... real/convert_idsia.py --help` |

Notes
- Every PyBullet dataset comes from ONE generator and ONE config:
  `python envs/pybullet_quadrotor_se3/datagen/generate_dataset.py --config envs/pybullet_quadrotor_se3/datagen/config.yaml`.
  It writes `datasets/QUADROTOR-DATASET-<name>/<name>_<drone>_<T>s_h<h>_<variant>.pkl`. The settings that rebuild each
  dataset above bit-for-bit are listed at the top of `config.yaml`. Each dataset's own `*_config_used.yaml` is the
  old-format record of how it was first made.
- Model-analysis scripts that used to sit in these folders (IDSIA benchmark protocol, ablations, oracles)
  are now in `src/models/SE3_Quadrotor/comparision/idsia/`.
- Retired data is in `archive/datasets/` (`windy_pendulum_3d_old`, `windy_quadrotor_se3`).

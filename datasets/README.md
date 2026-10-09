# datasets/

Data only. This folder holds `.pkl` files plus the metadata written alongside them
(`*_config_used.yaml`, `*_generation.log`, `*_audits.json`, `*_dataset_analysis.pdf`).
It contains no code. Every script that writes these files lives under `envs/<system>/datagen/`.
Every script that plots them lives under `envs/<system>/plots/`.
Commands are run from the project root.

The datasets are not distributed with the code: build each one with the command in the last column (the configs
are in each generator's `datagen/configs/`). The real-flight sets need the third-party raw data described in the
README of `envs/quadrotor_se3_idsia/` and `envs/rov_se3_marinarium/`.

| Folder | Made by | Regenerate |
|---|---|---|
| `PENDULUM-DATASET-<name>/` | `envs/pendulum_so3/datagen/generate_dataset.py` + `config.yaml` (one pickle per noise level) | `python envs/pendulum_so3/datagen/generate_dataset.py --config <config>` |
| `QUADROTOR-DATASET-<name>/` | `envs/quadrotor_se3_pybullet/datagen/generate_dataset.py` + `config.yaml` (PyBullet Crazyflie; named configs in `datagen/configs/`, the 300-flight training sets and their `-EVAL10s` evaluation companions) | `python envs/quadrotor_se3_pybullet/datagen/generate_dataset.py --config <config>` |
| `ROV-MARINARIUM-DATASET-<name>/` | real BlueROV2 tank recordings: `envs/rov_se3_marinarium/datagen/bag_to_npz.py` then `generate_dataset.py` + `config.yaml` | `PAPER-MANUAL` (thrust input), `PAPER-MANUAL-WRENCH` (wrench input), `PAPER-MANUAL-COMMANDS` (raw PX4 commands, `datagen/configs/PAPER-MANUAL-COMMANDS.yaml`; the only input built without the published thruster geometry / T200 map, used by the SE3_ROV models): the Marinarium paper's split |
| `QUADROTOR-DATASET-IDSIA/` | real flights, converted by `envs/quadrotor_se3_idsia/datagen/convert_idsia.py` (raw data in `envs/quadrotor_se3_idsia/idsia_raw/`) | `python envs/quadrotor_se3_idsia/datagen/convert_idsia.py --help` |

Notes
- Every PyBullet dataset comes from ONE generator, `envs/quadrotor_se3_pybullet/datagen/generate_dataset.py`;
  `datagen/config.yaml` is the documented template of all its settings.
  It writes `datasets/QUADROTOR-DATASET-<name>/<name>_<drone>_<T>s_h<h>_<variant>.pkl`. Each dataset is rebuilt from
  its named config in `envs/quadrotor_se3_pybullet/datagen/configs/<name>.yaml` (`--config` instead of `config.yaml`);
  its own `*_config_used.yaml` records the settings it was made with.

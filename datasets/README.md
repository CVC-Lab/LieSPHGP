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
| `QUADROTOR-DATASET-HARD/` | `envs/quadrotor_se3/datagen/generate_quadrotor_hard_v2.py` (the former HARD-V5; see `_note` in its `config_used.yaml`) | `python envs/quadrotor_se3/datagen/generate_quadrotor_hard_v2.py --config envs/quadrotor_se3/datagen/configs/hard_config.yaml` |
| `QUADROTOR-DATASET-EVALSET/` | `envs/quadrotor_se3/datagen/generate_quadrotor_evalset.py` | `... generate_quadrotor_evalset.py` (default config `configs/evalset_config.yaml`) |
| `QUADROTOR-DATASET-WIND/` | `envs/quadrotor_se3/datagen/generate_quadrotor_wind.py` | default config `configs/wind_config.yaml` |
| `QUADROTOR-DATASET-WIND25/` | `envs/quadrotor_se3/datagen/generate_quadrotor_wind25.py` | default config `configs/wind25_config.yaml` |
| `QUADROTOR-DATASET-WINDSDE/` | `envs/quadrotor_se3/datagen/generate_quadrotor_windsde.py` | one config per variant: `--config configs/windsde_<variant>_config.yaml` |
| `QUADROTOR-EVAL-REFERENCE/` | `envs/quadrotor_se3/datagen/generate_reference_flights.py` | `... generate_reference_flights.py --duration-seconds 3.0` |
| `QUADROTOR-DATASET-IDSIA/` | real flights, converted by `envs/quadrotor_se3/datagen/real/convert_idsia.py` (raw data in `tmp/idsia_raw/`) | `... real/convert_idsia.py --help` |
| `QUADROTOR-DATASET-NANOBENCH/` | real flights, converted by `envs/quadrotor_se3/datagen/real/convert_nanobench.py` (raw data in `tmp/nanobench_raw/`) | `... real/convert_nanobench.py --help` |

Notes
- The quadrotor generators read their input config from `envs/quadrotor_se3/datagen/configs/`.
  They write to `output.directory` in that config, which is relative to the project root.
- `generate_quadrotor_hard_v2.py` always merges the given `--config` over the base `configs/hard_v2_config.yaml`.
  With no `--config` it rebuilds the original HARD-V2 data, not HARD.
- Model-analysis scripts that used to sit in these folders (IDSIA benchmark protocol, ablations, oracles)
  are now in `src/models/SE3_Quadrotor/comparision/idsia/`.
- Retired data is in `archive/datasets/` (`windy_pendulum_3d_old`, `windy_quadrotor_se3`).

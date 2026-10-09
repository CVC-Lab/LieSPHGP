# paper_results — configs and results behind every table and figure of the paper

`configs/` holds the configs that train each model in the paper (current code: `lie_ph/`, `ph_node/`, `neural_sde/`)
and the dataset-generator configs. `results/` holds copies of the report folders from `experiments/` that the paper's
numbers and figures were taken from. Commands are run from the project root.

## Paper item → config → result

| paper item | models (paper name → config) | result (`results/`) |
|---|---|---|
| Pendulum ODE table (`tab:3d-pendulum-comparison`) | PH-NODE → `pendulum/PH-NODE-ref-pretrain_DampConst-NoWind_noise0p25.yaml`; LieSPHGP ($\xi=0$) → `pendulum/Lie-PH-GP-ODE_DampConst-NoWind_noise0p25.yaml` | `pendulum_ode_DampConst-NoWind_noise0p25/open-loop-horizons-summary.json` (rows `PH-NODE-ref-pretrain`, `Lie-PH-GP-ODE`, horizon `0-10s`) |
| Windy pendulum table (`tab:3d-windy-pendulum-comparison`) | PH-NODE → `pendulum/PH-NODE_DampRate-Wind_noise0p5.yaml`; Lie-PH-NN-SDE → `pendulum/Lie-PH-NN-SDE_DampRate-Wind_noise0p5.yaml`; LieSPHGP → `pendulum/Lie-PH-GP-SDE_DampRate-Wind_noise0p5.yaml`; NeuralSDE → `pendulum/NeuralSDE_DampRate-Wind_noise0p5.yaml` | `pendulum_windy_DampRate-Wind_noise0p5/open-loop-horizons-summary.json` (horizon `0-10s`) |
| Pendulum IDA-PBC figure (`fig:ida_pbc`) | the windy-pendulum models above | `pendulum_windy_DampRate-Wind_noise0p5/closed-loop-upright-figure*.png/pdf`, `closed-loop-upright-summary.json` |
| Quadrotor closed-loop table (`tab:quadrotor-closed-loop-comparison`) and figure (`fig:se3_quadrotor_heldout`) | LieSPHGP → `quadrotor_pybullet/Lie-PH-GP-SDE_DampRate-Wind_noise0p25.yaml`; PH-NODE → `quadrotor_pybullet/PH-NODE_DampRate-Wind_noise0p25.yaml` (`base.yaml`: the campaign template) | `quadrotor_pybullet_DampRate-Wind_noise0p25/closed-loop-summary.json` (`table` → `0-10s`), `closed-loop-comparison.pdf` |
| BlueROV2 table (`tab:rov-real`) | LieSPHGP → `rov_marinarium/Lie-PH-GP-SDE_REAL-MARINARIUM-COMMANDS-5s.yaml`; PH-NODE → `rov_marinarium/PH-NODE-ORIG_REAL-MARINARIUM-COMMANDS-5s.yaml` (report uses its step-2000 checkpoint; the run stopped at step 2212) | `rov_marinarium/open-loop-horizons-summary.json` (horizon `0-10s`) |
| Real quadrotor table (`tab:multistep-prediction`) | PH-GP-SDE → `quadrotor_idsia/Lie-PH-GP-SDE_IDSIA.yaml`; PH-NN-SDE → `quadrotor_idsia/Lie-PH-NN-SDE_IDSIA.yaml` | `quadrotor_idsia_protocol_TEST-melon/split_protocol_table.json` (`table7`, `cum`) |

Notes
- Real quadrotor: the numbers in the paper come from the September 2026 runs (old code, since deleted; GP at
  checkpoint step 4000, NN at step 7000). The two `quadrotor_idsia` configs are the current-code configs that
  replicate that recipe (0.5 s windows, batch 256, matmul highest, KL 0.1, fixed GP hyperparameters).
- `experiment.root` in each config still points into `experiments/`: a new run writes a new time-stamped folder there.

## Datasets (`configs/datasets/`)

| config | generator | used by |
|---|---|---|
| `pendulum/ODE-DissipationConstant-0p5-5s-G0p5-0p7-0p8.yaml` | `envs/pendulum_so3/datagen/generate_dataset.py` | pendulum ODE table |
| `pendulum/SDE-DiffusionConstant-0p5-DissipationRateDependent-0p5-5s-G0p5-0p7-0p8.yaml` | same | windy pendulum table, IDA-PBC figure |
| `quadrotor_pybullet/SDE-...-DissipationRateDependent-0p5-5s-NoisyPID0p01-50Hz-300flights.yaml` (+ `-EVAL10s`) | `envs/quadrotor_se3_pybullet/datagen/generate_dataset.py` | quadrotor table and figure |
| `rov_marinarium/PAPER-MANUAL-COMMANDS-5s.yaml` (training), `PAPER-MANUAL-COMMANDS.yaml` (10 s test pieces) | `envs/rov_se3_marinarium/datagen/generate_dataset.py` (after `bag_to_npz.py --all`) | BlueROV2 table |
| no config: `python envs/quadrotor_se3_idsia/datagen/convert_idsia.py` (see `envs/quadrotor_se3_idsia/README.md`) | IDSIA converter | real quadrotor table |

```bash
python envs/pendulum_so3/datagen/generate_dataset.py --config paper_results/configs/datasets/pendulum/<config>.yaml
python envs/quadrotor_se3_pybullet/datagen/generate_dataset.py --config paper_results/configs/datasets/quadrotor_pybullet/<config>.yaml
python envs/rov_se3_marinarium/datagen/generate_dataset.py --config paper_results/configs/datasets/rov_marinarium/<config>.yaml
```

## Training

A config's `schema` names its trainer: `lie_ph/<system>/v1` → `lie_ph/train.py`, `ph_node/<system>/v1` → `ph_node/train.py`;
the NeuralSDE config goes to `neural_sde/train.py`.

```bash
P=paper_results/configs
python src/models/3D_SO3_Windy_Pendulum/lie_ph/train.py    --config $P/pendulum/Lie-PH-GP-SDE_DampRate-Wind_noise0p5.yaml
python src/models/3D_SO3_Windy_Pendulum/ph_node/train.py   --config $P/pendulum/PH-NODE_DampRate-Wind_noise0p5.yaml
python src/models/3D_SO3_Windy_Pendulum/neural_sde/train.py --config $P/pendulum/NeuralSDE_DampRate-Wind_noise0p5.yaml
python src/models/SE3_Quadrotor/lie_ph/train.py  --config $P/quadrotor_pybullet/Lie-PH-GP-SDE_DampRate-Wind_noise0p25.yaml
python src/models/SE3_Quadrotor/ph_node/train.py --config $P/quadrotor_pybullet/PH-NODE_DampRate-Wind_noise0p25.yaml
python src/models/SE3_Quadrotor/lie_ph/train.py  --config $P/quadrotor_idsia/Lie-PH-GP-SDE_IDSIA.yaml
python src/models/SE3_ROV/lie_ph/train.py  --config $P/rov_marinarium/Lie-PH-GP-SDE_REAL-MARINARIUM-COMMANDS-5s.yaml
python src/models/SE3_ROV/ph_node/train.py --config $P/rov_marinarium/PH-NODE-ORIG_REAL-MARINARIUM-COMMANDS-5s.yaml
```

## Reports (the files in `results/`)

```bash
python src/models/3D_SO3_Windy_Pendulum/comparison/report.py --setting DampRate-Wind --noise 0.5
JAX_PLATFORMS=cpu python src/models/3D_SO3_Windy_Pendulum/comparison/closed_loop.py --task upright --setting DampRate-Wind --noise 0.5
python src/models/3D_SO3_Windy_Pendulum/comparison/closed_loop_figure.py --report-dir <report dir> --layout 6x1
python src/models/SE3_Quadrotor/comparison/report.py --setting DampRate-Wind
JAX_PLATFORMS=cpu python src/models/SE3_Quadrotor/comparison/closed_loop.py --setting DampRate-Wind
python src/models/SE3_ROV/comparison/report_real.py --phnode-checkpoint step002000
```

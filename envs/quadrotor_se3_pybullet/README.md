# quadrotor_se3_pybullet

The gym-pybullet-drones quadrotor: PyBullet physics (`CtrlAviary`, CF2P) driven by `DSLPIDControl`.
Every simulated quadrotor training and evaluation dataset is generated here.

| Path | Role |
|---|---|
| `gym-pybullet-drones/` | third-party simulator (git-ignored; see Setup), found through `sys.path` |
| `plant.py` | PyBullet helpers shared with the closed-loop tools: ground-plane removal, contact-free dynamics, CF2P motor mixer |
| `datagen/generate_dataset.py` + `datagen/config.yaml` | the dataset generator and its documented template config: trajectory set (hard / eval / hard+eval), dissipation and wind laws per channel, seeds, noise, output name |
| `datagen/configs/` | one config per dataset in `datasets/QUADROTOR-DATASET-<name>/` |

`src/models/SE3_Quadrotor/comparison/closed_loop.py` uses this simulator as the closed-loop plant.

## Setup

The simulator is not part of this repository. Clone it at the commit the datasets were made with:

```bash
git clone https://github.com/utiasDSL/gym-pybullet-drones envs/quadrotor_se3_pybullet/gym-pybullet-drones
git -C envs/quadrotor_se3_pybullet/gym-pybullet-drones checkout e712698a05a80728b06572819dcf044596707754
pip install -r requirements.txt
```

The scripts add the clone to `sys.path`, so it does not need to be pip-installed. gym-pybullet-drones is MIT-licensed
(Copyright (c) 2020 Jacopo Panerati); its licence is in the clone.

## Generate a dataset

```bash
python envs/quadrotor_se3_pybullet/datagen/generate_dataset.py \
    --config envs/quadrotor_se3_pybullet/datagen/configs/SDE-DiffusionConstant-0p5-DissipationConstant-0p5-5s-NoisyPID0p01-50Hz-300flights.yaml
```

A 300-flight config takes about 10 minutes on one CPU core. See `datasets/README.md` for the output layout.

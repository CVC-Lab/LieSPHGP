# rov_se3_marinarium

Real BlueROV2 Heavy recordings from the KTH Marinarium tank (9 × 5 × 3 m, underwater motion capture), released with
Torroba et al., *Marinarium: A Modular Experimental Facility for Reproducible Maritime and Space-Analog Field Robotics*,
arXiv:2602.23053v2 (2026), Sec. IV. The recordings are converted to the SE(3) layout the models use.
`bluerov2.py` holds the published vehicle model (constants, thruster geometry, T200 thrust map from `data/`), used for
the input conversion and as the published-physics baseline.

| File | Role |
|---|---|
| `marinarium_raw/` | clone of `github.com/ViktorNfa/bluerov2_dynamics` (MIT, commit 5843178): rosbags, their CSVs, baselines (git-ignored; see Setup) |
| `datagen/bag_to_npz.py` | step 1: rosbag2 → `marinarium_raw/npz/<recording>.npz`, raw streams on their own clocks (needs `pip install rosbags`) |
| `datagen/generate_dataset.py` + `datagen/config.yaml` | step 2: npz → `datasets/ROV-MARINARIUM-DATASET-<name>/<name>_clean.pkl`; named variants in `datagen/configs/` (`PAPER-MANUAL-COMMANDS`, `-COMMANDS-5s`, `-WRENCH`) |
| `datagen/validate_marinarium.py` | frame, sensor and input checks of one recording |
| `analysis/sim_to_real_gap.py` | sim-to-real gap of the published physics (paper protocol + clean protocol + per-axis scales) → `experiments/rov_se3/analysis/sim_to_real_gap/` |

## Setup

The raw recordings are not part of this repository. They come from the Marinarium repository, whose rosbags are stored
with Git LFS (about 6 GB on disk after `git lfs pull`: 0.7 GB working tree, the rest in `.git`). From the project root:

```
git clone https://github.com/ViktorNfa/bluerov2_dynamics envs/rov_se3_marinarium/marinarium_raw
git -C envs/rov_se3_marinarium/marinarium_raw checkout 5843178f831e6c10ee9158eaffd7774e3ea83a7a
git -C envs/rov_se3_marinarium/marinarium_raw lfs pull          # needs git-lfs (e.g. apt install git-lfs; git lfs install)
pip install rosbags scipy pyyaml pandas matplotlib               # rosbags is pure Python: no ROS installation needed
```

Then, in this order:

```
python envs/rov_se3_marinarium/datagen/bag_to_npz.py --all                     # 1. rosbag2 -> marinarium_raw/npz/*.npz
python envs/rov_se3_marinarium/datagen/generate_dataset.py --config envs/rov_se3_marinarium/datagen/config.yaml
                                                                                # 2. -> datasets/ROV-MARINARIUM-DATASET-PAPER-MANUAL/
python envs/rov_se3_marinarium/datagen/validate_marinarium.py --recording manual   # optional: the checks below
python envs/rov_se3_marinarium/analysis/sim_to_real_gap.py                     # 3. needs PAPER-MANUAL (input: thrust) from step 2
```

`sim_to_real_gap.py` also re-runs the upstream Fossen baseline (`marinarium_raw/fossen/BlueROV2.py`, imported from the
clone), which is slow; `--skip-their-fossen` skips it (and reuses an earlier result from `results.json` if present).
`bag_to_npz.py` uses the PX4 message definitions shipped in the clone (`rosbags/types/px4_msgs/msg`).

## Recordings

| npz | rosbag | length | notes |
|---|---|---|---|
| `oct30` | `rosbag2_2025_10_30-16_31_20` | 1212 s | full 6-DOF incl. large roll/pitch; commands tied in pairs (`u1=u3`, `u2=u4`, `u5=-u8`): **no sway force at all** |
| `manual` | `rosbag2_2025_11_06-manual` | 916 s | the paper's Table 2 set (chronological 80/20); 18.5 % of motor messages with every motor off (coasting) |
| `stabilized` | `rosbag2_2025_11_06-stabilized` | 917 s | attitude-stabilised flight; least collinear commands |

Streams: motion capture `/mocap/itrl_rov_1/odom` (~100 Hz), PX4 IMU `sensor_combined` (100 Hz), 8 motor commands
`actuator_motors` (100 Hz, `NaN` = motor not commanded), battery voltage `battery_status_v1` (1 Hz; read from the raw CDR
bytes because the bag has no type definition for it; 13.3–16.5 V).

## What the checks found (`validate_marinarium.py`)

- **Frames.** World = motion-capture frame with **z down** (accelerometer matches $-R^\top g$, $g=+9.81\,e_3$, to 0.3 m/s²;
  the z-up hypothesis is off by 11 m/s²). Body forward-right-down. The odom twist is in the **body** frame: over 100 ms
  the pose agrees with the integrated twist to 0.5–1.2 mm and 0.12–0.30° (median). The PX4 gyro is in the same body axes
  (corr 0.96–0.999, bias < 0.003 rad/s).
- **Dropouts.** 95–149 motion-capture gaps > 50 ms per recording (longest 6.45 s). The paper's CSV interpolates across
  them; this converter never does (trajectories are cut between them; streams flag them).
- **Pose glitches.** 42–109 one-step attitude jumps per recording that the gyro does not see, up to 180° (marker
  swaps / flips). Excluded like dropouts (`glitch_threshold_deg`).
- **Clocks.** Motion-capture header stamps are steadier than bag receive times (1–99 % of steps 7.5–12.5 ms vs 6–14 ms) and
  fit the twist slightly better; `mocap_time: header` is the default.
- **Thruster input.** $E\,T_{200}(u,V)$ with the published geometry matches the force the motion needs under the
  published model in surge (corr 0.86, gain 0.95), sway (0.70) and yaw (0.75); heave is weak (0.42) and roll/pitch are
  not explained (≈ 0): the published model (and/or the vertical-thruster geometry) does not fit this vehicle in those axes.
- **Collinearity.** The PX4 mixer commands the thrusters in fixed pairs (`manual`: `u5≈-u6`, `u7≈-u8`; `oct30`: corr ±1.00),
  so the 8 individual columns of $G$ are **not identifiable** from these recordings; the 6-D `wrench` input is better
  conditioned than the 8 `thrust` inputs for learning.
- **Agreement with the paper's CSV.** On the paper's test rows (last 20 % of `manual`, $N=9165$ there, 9164 here) the
  position differs by 0.6 mm and the attitude by 0.05° (median).

## Dataset layout

Rows `[p (3), vec(R) (9), v_b (3), ω_b (3), u (n_u)]`; `n_u = 8` (`commands`, `thrust`) or 6 (`wrench`), recorded in
`settings["control_dim"]`. `u` in row $k$ is the mean input over $(t_{k-1}, t_k]$ (it drives row $k-1 \to k$).
Keys: `train_trajectories`, `test_trajectories` (fixed-length, dropout- and glitch-free), `x`, `test_x` (stored windows),
`train_stream`/`test_stream` (+ `_time`, `_valid`; one continuous series per split, gaps interpolated and flagged in
`_valid`, the layout of the paper's continuous H-step evaluation; with `split: recordings` one `<split>_streams` dict per
recording), `settings` (allocation $E$, thruster geometry, published nominal parameters, coverage audits).

`u` is the mean over the motor messages inside each sample interval. The ~100 Hz motor stream has gaps up to ~40 ms, so
a few 20 ms intervals contain no message and get $u = 0$ (26 of the `PAPER-MANUAL-COMMANDS` input rows); the count is
reported per piece in `settings["splits"][...]["pieces"]` as `input_intervals_without_motor_message`.

## Data source and licence

The recordings, the paper's CSV and the Fossen baseline are from
[github.com/ViktorNfa/bluerov2_dynamics](https://github.com/ViktorNfa/bluerov2_dynamics), released under the MIT
License, "Copyright (c) 2025 Victor Nan Fernandez-Ayala" (see `marinarium_raw/LICENSE`). Datasets derived from them
keep that notice. Please cite the Marinarium paper; the BibTeX given in that repository's README is:

```bibtex
@misc{torroba2026marinariumnewarenabring,
      title={Marinarium: a New Arena to Bring Maritime Robotics Closer to Shore}, 
      author={Ignacio Torroba and David Dorner and Victor Nan Fernandez-Ayala and Mart Kartasev and Joris Verhagen and Elias Krantz and Gregorio Marchesini and Carl Ljung and Pedro Roque and Chelsea Sidrane and Linda Van der Spaa and Nicola De Carli and Petter Ogren and Christer Fuglesang and Jana Tumova and Dimos V. Dimarogonas and Ivan Stenius},
      year={2026},
      eprint={2602.23053},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.23053}, 
}
```

The current arXiv version (v2) carries the title *Marinarium: A Modular Experimental Facility for Reproducible Maritime
and Space-Analog Field Robotics*.

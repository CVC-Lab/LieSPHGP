# SE3_ROV — learning BlueROV2 Heavy dynamics on SE(3) from real tank recordings

Models that learn the dynamics of the BlueROV2 Heavy underwater vehicle from the real KTH Marinarium recordings
(`envs/rov_se3_marinarium`): position $x$, attitude $R\in SO(3)$ (z down), body velocity $v$ and angular velocity
$\omega$, driven by the 8 raw PX4 thruster commands $u\in\mathbb R^8$. No physical constant is given to the models: the
input is the raw command vector (`ROV-MARINARIUM-DATASET-PAPER-MANUAL-COMMANDS-5s`), not a wrench built from the published
thruster geometry and T200 map.

## Models

| model | package | subnetworks | integrator | process noise | loss |
|---|---|---|---|---|---|
| **Lie-PH-GP-SDE** | [lie_ph](lie_ph) | level + variational GP | Lie-IMEX | learned $\Sigma$ | EKF marginal likelihood + KL |
| PH-NODE-ORIG | [ph_node](ph_node) | tanh MLP | RK4 | none | trajectory loss + L1 on $G$, $D_v$, $D_\omega$ (the original real-flight recipe: 0.2 s sequences, full batch, Adam $10^{-3}$, float64) |
| PH-NODE | [ph_node](ph_node) | tanh MLP | RK4 | none | trajectory loss with our 2 s windows (NaN gradients from step 5) |

Both use the same port-Hamiltonian structure on $SE(3)$: learned $M_1^{-1}$ (full SPD here, `model.M1_form: spd`),
$M_2^{-1}(R)$, $D_v(v)$, $D_\omega(\omega)$, $V$ (with a rotation term, `model.V_rotation`) and $G(R,x)\in\mathbb R^{6\times 8}$.

## Folders

| folder | content |
|---|---|
| [lie_ph/](lie_ph) | our model: network, Lie-IMEX integrator, EKF loss, `train.py`, `evaluate.py` (the quadrotor's lie_ph with $n_u$ inputs) |
| [ph_node/](ph_node) | the PH-NODE baseline: RK4, trajectory loss, optional L1 term; reuses lie_ph's model class and training loop |
| [comparison/](comparison) | `report_real.py`: open-loop report on the 10 s held-out pieces (Lie-PH-GP-SDE, PH-NODE, the paper's Koopman EDMDc, published physics, persistence) and the paper's Table 2 protocol; `models.py` |
| [configs/](configs) | `marinarium-05-10-2026/`: the three configs above |

A config's `schema` names its trainer: `lie_ph/rov/v1` → `lie_ph/train.py`, `ph_node/rov/v1` → `ph_node/train.py`.

## Usage

Build the data first (`envs/rov_se3_marinarium/README.md`: clone, `git lfs pull`, `bag_to_npz.py`, `generate_dataset.py`).

```bash
C=src/models/SE3_ROV/configs/marinarium-05-10-2026
python src/models/SE3_ROV/lie_ph/train.py  --config $C/Lie-PH-GP-SDE_REAL-MARINARIUM-COMMANDS-5s.yaml
python src/models/SE3_ROV/ph_node/train.py --config $C/PH-NODE-ORIG_REAL-MARINARIUM-COMMANDS-5s.yaml
python src/models/SE3_ROV/lie_ph/evaluate.py  --run <run_dir>       # learned products, EKF NLL / NIS on the test split
python src/models/SE3_ROV/ph_node/evaluate.py --run <run_dir>
python src/models/SE3_ROV/comparison/report_real.py --phnode-checkpoint step002000
```

Results: `experiments/rov_se3/real_marinarium_05-10-2026/` (runs and `reports/`). The PH-NODE-ORIG run stopped at
step 2212 (non-finite loss); the report uses its step-2000 checkpoint and says so.

**Protocol:** float32 for Lie-PH-GP-SDE (float64 for PH-NODE-ORIG, its original recipe); no pretraining and no physical
prior; a fixed budget of 5000 steps with the final checkpoint; training reads only the measured data.

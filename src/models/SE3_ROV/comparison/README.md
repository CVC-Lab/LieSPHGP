# comparison — BlueROV2 Marinarium report

| file | role |
|---|---|
| [report_real.py](report_real.py) | open-loop report on the 10 s held-out pieces of the real recordings: Lie-PH-GP-SDE, PH-NODE, the Marinarium paper's Koopman EDMDc-RBF, published physics (`envs/rov_se3_marinarium/bluerov2.py`) and persistence; plus the paper's Table 2 protocol |
| [models.py](models.py) | loads a run of either package: model class and integrator (RK4 for ph_node, Lie-IMEX for lie_ph) |

```bash
python src/models/SE3_ROV/comparison/report_real.py --phnode-checkpoint step002000
```

Needs the Marinarium clone (`envs/rov_se3_marinarium/marinarium_raw`, for the Koopman code and the paper's CSVs).
Output: `experiments/rov_se3/real_marinarium_05-10-2026/reports/`.

# comparison — quadrotor model comparison (PyBullet campaign)

The tools that train, evaluate and compare the quadrotor models of [lie_ph](../lie_ph) and [ph_node](../ph_node) on the
simulated PyBullet data. Results: `experiments/quadrotor/campaign_06-10-2026/`.

| file | role |
|---|---|
| [campaign.py](campaign.py) | writes the campaign configs (into `../configs/pybullet-campaign-06-10-2026/`, from its `base.yaml`) and runs them on the GPUs |
| [report.py](report.py) | open-loop horizons report on the 10 s held-out flights (`-EVAL10s` datasets) |
| [closed_loop.py](closed_loop.py) | closed-loop tracking of the held-out flights in the dataset's own PyBullet simulator |
| [models.py](models.py) | loads a run of either package: model class and integrator (RK4 for ph_node, Lie-IMEX for lie_ph) |

```bash
python src/models/SE3_Quadrotor/comparison/campaign.py make
python src/models/SE3_Quadrotor/comparison/campaign.py run --gpu 0
python src/models/SE3_Quadrotor/comparison/report.py --setting DampRate-Wind
JAX_PLATFORMS=cpu python src/models/SE3_Quadrotor/comparison/closed_loop.py --setting DampRate-Wind
```

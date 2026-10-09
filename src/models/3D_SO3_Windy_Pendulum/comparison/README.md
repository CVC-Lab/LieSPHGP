# comparison — pendulum model comparison

The tools that train, evaluate and compare every pendulum model: [lie_ph](../lie_ph) (Lie-PH-GP/NN-SDE/ODE),
[ph_node](../ph_node) (PH-NODE, PH-NODE-ref) and [neural_sde](../neural_sde) (NeuralSDE).
Results: `experiments/pendulum_so3/campaign_04-10-2026/`.

| file | role |
|---|---|
| [campaign.py](campaign.py) | writes every campaign config (into `../configs/old/campaign-04-10-2026/`) and runs them on the GPUs |
| [report.py](report.py) | open-loop horizons report: one PDF per dataset (setting × observation noise) |
| [closed_loop.py](closed_loop.py) | closed-loop swing-up to upright (IDA-PBC with each model's operators) on the true plant |
| [closed_loop_figure.py](closed_loop_figure.py) | the paper's closed-loop figure from the cached flights |
| [models.py](models.py) | loads a run of either package: model class and integrator (RK4 for ph_node, Lie-IMEX for lie_ph) |

```bash
python src/models/3D_SO3_Windy_Pendulum/comparison/campaign.py make
python src/models/3D_SO3_Windy_Pendulum/comparison/campaign.py run --gpu 0
python src/models/3D_SO3_Windy_Pendulum/comparison/report.py --setting DampRate-Wind --noise 0.5
JAX_PLATFORMS=cpu python src/models/3D_SO3_Windy_Pendulum/comparison/closed_loop.py --task upright --setting DampRate-Wind --noise 0.5
```

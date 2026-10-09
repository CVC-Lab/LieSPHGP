# lie_ph — Lie-PH model on SE(3) (BlueROV2, real Marinarium recordings)

The quadrotor's [lie_ph](../../SE3_Quadrotor/lie_ph/README.md) (same algorithm: port-Hamiltonian SDE on $SE(3)\times\mathbb R^6$,
level + variational-GP subnetworks, Lie-IMEX integrator, EKF marginal likelihood + KL, fixed budget, final checkpoint)
with three ROV options:

- `model.control_dim`: $n_u$ inputs (8 raw thruster commands here), $G(R,x)\in\mathbb R^{6\times n_u}$;
- `model.M1_form: spd`: a full SPD translational inverse mass (added mass differs per axis underwater);
- `model.V_rotation`: a rotation term in the potential (buoyancy and gravity act at different points).

| file | role |
|---|---|
| [network.py](network.py) | model, GP features setup, parameter start |
| [integrator.py](integrator.py) | Lie-IMEX step and rollout |
| [losses.py](losses.py) | EKF marginal likelihood, NIS, the trajectory loss (Lie-PH-NN-ODE) |
| [data.py](data.py), [features.py](features.py), [config.py](config.py) | data windows, Matérn random features, strict config (schema `lie_ph/rov/v1`) |
| [train.py](train.py), [evaluate.py](evaluate.py) | trainer; learned gauge-invariant products, EKF NLL / NIS (and ground truth when the dataset has one) |

```bash
python src/models/SE3_ROV/lie_ph/train.py --config src/models/SE3_ROV/configs/marinarium-05-10-2026/Lie-PH-GP-SDE_REAL-MARINARIUM-COMMANDS-5s.yaml
python src/models/SE3_ROV/lie_ph/evaluate.py --run <run_dir>
```

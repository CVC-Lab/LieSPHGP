# quadrotor_se3_port_ham

Our own SE(3) quadrotor simulator, written in port-Hamiltonian form and integrated on the Lie group.
The full derivation is in `quadrotor.md` and `notes/quadrotor-se3-ph-system.md`.

| File | Role |
|---|---|
| `quadrotor.py` | `quadrotor_se3` environment: dynamics, Lie-Heun integrator, rendering |
| `env_config.py` | builds env kwargs from `configs/quadrotor_se3/envs/{ode,sde}.yaml` |
| `datagen/windy_quadrotor_datagen.py` | dataset generator built on this env (default output `datasets/windy_quadrotor_se3/`) |

None of the current `QUADROTOR-DATASET-*` training sets come from this env. They are all made with
PyBullet, see `envs/quadrotor_se3_pybullet/`. Rendering with `render_backend='pybullet'`
borrows the CF2X mesh from `envs/quadrotor_se3_pybullet/gym-pybullet-drones/`.
`mini_tests/compare_pybullet_vs_ours.py` checks that the two simulators agree.

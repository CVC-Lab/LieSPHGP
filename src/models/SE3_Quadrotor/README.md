# Canonical SE(3) quadrotor models

This directory contains four independent model packages, all trained and
evaluated with JAX. Each package owns its network, losses, integrator, data
loading, checkpointing, configuration validation, experiment bookkeeping, and
training loop:

- `ph_node`: neural port-Hamiltonian dynamics with RK4.
- `ph_nn_lie_ode_integrator`: the same neural field with Lie-Heun.
- `ph_nn_lie_imex`: the neural field with Lie-IMEX.
- `ph_gp_lie_imex`: canonical V2 variational GP field with Lie-IMEX.

Every model folder follows the same small layout:

```text
network.py
losses.py
integrator.py
config.py
data.py
checkpoints.py
experiment.py
train.py
```

Each `network.py` presents one readable port-Hamiltonian model object with
named mass, dissipation, potential, and control subnetworks. Each integrator
accepts that model object directly, and each trainer optimizes the model
PyTree.

`ph_gp_lie_imex` also has `gp_model.py`. It is a byte-for-byte copy of
`src/utils/JAX/gp_model.py`; the quadrotor input adapter remains in
`network.py`, while observation likelihood and loss equations remain in
`losses.py`. Its private `_backend.py` contains only feature-bank and
variational-parameter machinery. Setting `gp.model_backend: null` preserves the earlier quadrotor
feature implementation. Setting it to `utils-gp-model` enables the copied GP
implementation and requires all five associated kernel arguments.

The shared dynamics have the port-Hamiltonian form
$f_\theta(z,u)=[J(z)-D_\theta(z)]\nabla H_\theta(z)+G_\theta(z)u$.
Lie methods update rotation with
$R_{k+1}=R_k\exp(\widehat{\phi_k})$.

## Training

Training accepts exactly one command-line argument: a required YAML config.
Production configs live in `configs/`, enumerate every training setting, and
include their creation date and time in the filename.

```bash
python -m src.models.SE3_Quadrotor.ph_node.train \
  --config src/models/SE3_Quadrotor/configs/09-09-2026-23-13_ph_node.yaml
```

Every run creates `experiments/quadrotor/DD-MM-HH-MM_<model-details>/`, copies
the exact input config, writes JAX checkpoints and metrics, then automatically
generates an exactly 47-page model-only PDF.

## Reports

The only report program is `comparision/generate_report.py`. The folder name is
retained for compatibility, but its output does not compare models. A report
config may select one through four experiment directories; each selected model
receives a separate 47-page PDF.

```bash
python -m src.models.SE3_Quadrotor.comparision.generate_report \
  --config src/models/SE3_Quadrotor/configs/<dated-config>.yaml
```

For a standalone report call, set `report.experiment_dirs` in the dated config.
Training supplies its new run directory directly when automatic reporting is
enabled.

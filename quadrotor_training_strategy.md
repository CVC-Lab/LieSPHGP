# SE(3) Quadrotor Training Strategy

## 1. Objective

The goal is to compare the old port-Hamiltonian Neural ODE against the new
$SE(3)$ models under clean dynamics, observation noise, stochastic wind, and
real-world measurements.

The comparison must answer four separate questions:

1. Does removing PH-NODE's identity pretraining make optimization harder?
2. Does the Lie-group integrator improve long-horizon prediction and preserve
   $R\in SO(3)$ better than RK4?
3. Do GP subnetworks improve learning from noisy observations?
4. Do SDE models learn the distribution produced by stochastic wind better
   than deterministic ODE models?

No model should receive ground-truth mass, potential, dissipation, control, or
diffusion targets during trajectory training, except for assumptions already
present in the released old-paper real-world model.

## 2. Model definitions

### PH-NODE

- The released old-paper model.
- Uses its original Euclidean RK4 rollout in the simulated experiment.
- Uses the released identity-pretrained mass-network checkpoint.
- Its model architecture, loss, optimizer, and initialization procedure remain
  unchanged.
- This is the primary old-paper baseline.

### PH-NODE-no-pretraining

- Uses the same architecture, equations, loss, optimizer, and integrator as
  PH-NODE.
- Does not load or generate the identity-pretrained mass checkpoint.
- All trainable subnetworks start from their normal random initialization.
- This model tests whether the released pretraining is important for successful
  optimization.
- It is an ablation of PH-NODE, not a replacement for the official baseline.

### PH-NN-ODE-LieIntegrator

- Port-Hamiltonian neural ODE.
- Uses the $SE(3)$ Lie-group integrator instead of Euclidean RK4.
- Rotation is advanced using $R_{k+1}=R_k\exp([\phi_k]_\times)$.
- It has no process-diffusion network.
- It should not use ground-truth subnetwork pretraining.
- The cleanest comparison is PH-NODE versus PH-NN-ODE-LieIntegrator with all
  other training settings held fixed.

### PH-GP-ODE

- Port-Hamiltonian ODE with GP subnetworks.
- Uses the same Lie-group integration convention as
  PH-NN-ODE-LieIntegrator.
- Has no physical process diffusion.
- It is used primarily to test whether GP modeling improves robustness to
  observation noise.

### PH-NN-SDE

- Port-Hamiltonian neural SDE.
- Uses neural-network physics subnetworks and a learned diffusion subnetwork.
- Uses Lie-Heun integration with the same Wiener increment in the predictor and
  corrector stages.
- Comparing it with PH-NN-ODE-LieIntegrator isolates the SDE contribution.

### PH-GP-SDE

- Full proposed model.
- Uses GP physics subnetworks, learned diffusion, and Lie-Heun integration.
- Represents both GP parameter uncertainty and physical stochasticity.
- Comparing it with PH-GP-ODE isolates the diffusion contribution.
- Comparing it with PH-NN-SDE isolates the GP contribution under the same SDE
  formulation.

### Neural-SDE

- Optional unstructured stochastic baseline has been removed from the clean,
  noisy-ODE, and real-data training plans.
- It remains recommended for the two stochastic-wind datasets because it tests
  whether port-Hamiltonian structure contributes beyond simply using an SDE.

## 3. Training matrix

| Dataset | PH-NODE | PH-NODE-no-pretraining | PH-NN-ODE-LieIntegrator | PH-GP-ODE | PH-NN-SDE | PH-GP-SDE | Neural-SDE |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 official PyBullet | Yes | Yes | Yes | Yes | No | No | No |
| D1 clean custom ODE | Yes | Yes | Yes | Yes | Yes | Yes | No |
| D2 noisy ODE | Yes | Yes | Yes | Yes | Yes | Yes | No |
| D3 windy SDE | Yes | Yes | Yes | Yes | Yes | Yes | Recommended |
| D4 noisy windy SDE | Yes | Yes | Yes | Yes | Yes | Yes | Recommended |
| D5 real quadrotor | Yes, real version | Yes | Yes | Yes | Yes | Yes | No |

## 4. Common protocol for D1-D4

Unless a dataset section overrides a value, use the following common setup:

- State: $z=[x_w,\operatorname{vec}(R),v_b,\omega_b,u]\in\mathbb{R}^{22}$.
- Mass: $m=1$.
- Inertia: $J=\operatorname{diag}(0.5,0.5,1.0)$.
- Arm length: $1$.
- Thrust coefficient: $k_f=1$.
- Yaw-torque coefficient: $k_m=0.1$.
- Linear damping: $d_{\mathrm{lin}}=0.5$.
- Angular damping: $d_{\mathrm{ang}}=0.5$.
- Recorded timestep: $\Delta t=0.05$ seconds.
- Samples per trajectory: $20$.
- Number of trajectories: $120$.
- Split by complete trajectory:
  - $72$ training trajectories,
  - $24$ validation trajectories,
  - $24$ test trajectories.
- Control: geometric PID to a randomly sampled position and yaw target.
- Input convention: the same raw rotor-squared input for every model.
- Deterministic external wind amplitude: $0$.
- Final paper training: at least five paired seeds.
- Development training: seed $0$ may be used until the pipeline is verified.
- Checkpoint selection: best validation checkpoint, never best test checkpoint.

The current environment calculates the deterministic rigid-body drift and the
two stochastic channels in
[`envs/SE3_quadrotor/quadrotor.py`](envs/SE3_quadrotor/quadrotor.py#L409).
The current dataset generator stores clean test states separately from noisy
test observations in
[`datasets/windy_quadrotor_datagen.py`](datasets/windy_quadrotor_datagen.py#L312).

## 5. D0 — Official PyBullet clean dataset

### Purpose

D0 is the direct old-paper reproduction. It answers whether the new ODE models
can outperform PH-NODE on the old paper's own clean simulated quadrotor data.

### Dataset specification

- Source:
  `other_paper_codes/port_ham_neural_ode/LieGroupHamDL/examples/quadrotor/data/pybullet-drone-dataset80.pkl`.
- Simulator: Gym-PyBullet-Drones.
- Vehicle: Crazyflie CF2P.
- Simulation frequency: $240$ Hz.
- Controller frequency: $48$ Hz.
- The control is held for $5$ PyBullet substeps.
- Flight duration: $2.5$ seconds.
- Number of PID target poses: $18$.
- Stored data shape: $18\times5\times120\times22$.
- Training-window length: $5$ samples.
- Effective window duration: approximately $16.7$ ms.
- Explicit observation noise: none.
- Stochastic wind: none.
- Input convention: body wrench $[T,\tau_x,\tau_y,\tau_z]$.
- The old simulated experiment omits dissipation.

The released data collection code is in
[`data_collection.py`](other_paper_codes/port_ham_neural_ode/LieGroupHamDL/examples/quadrotor/data_collection.py#L25).

### Models trained

- PH-NODE with released identity pretraining.
- PH-NODE-no-pretraining.
- PH-NN-ODE-LieIntegrator.
- PH-GP-ODE.

PH-NN-SDE, PH-GP-SDE, and Neural-SDE are not trained on D0 because D0 is the
strict deterministic reproduction benchmark.

### Comparisons

- PH-NODE versus PH-NODE-no-pretraining tests the importance of identity
  pretraining.
- PH-NODE versus PH-NN-ODE-LieIntegrator tests the Lie integrator on the old
  dataset.
- PH-NN-ODE-LieIntegrator versus PH-GP-ODE tests NN versus GP on clean data.

### What to watch

- Use the same wrench input for all four models.
- Do not give one model rotor commands and another model body wrench.
- Keep PH-NODE's released code and initialization unchanged.
- The PH-NODE-no-pretraining row is the only row allowed to remove its
  pretraining.
- The windows are extremely short. Low window loss does not prove accurate
  $2$- or $5$-second prediction.
- Evaluate raw rotation matrices before projection when plotting
  $|\det(R)-1|$ and $\|R^\top R-I\|_F$.

## 6. D1 — Clean custom ODE dataset

### Purpose

D1 is the main controlled deterministic benchmark. It isolates numerical
integration and checks whether the SDE models correctly reduce to nearly
deterministic dynamics.

### Dataset specification

- Uses the common D1-D4 physical constants and split.
- Observation noise: $\sigma_{\mathrm{obs}}=0$.
- Wind-force diffusion: $\sigma_f=0$.
- Wind-torque diffusion: $\sigma_\tau=0$.
- Deterministic external wind: $0$.
- PID control keeps the open-loop-unstable quadrotor in an informative region.
- Every model receives identical controls, initial states, and trajectories.

### Models trained

- PH-NODE.
- PH-NODE-no-pretraining.
- PH-NN-ODE-LieIntegrator.
- PH-GP-ODE.
- PH-NN-SDE.
- PH-GP-SDE.

Neural-SDE is not trained on D1.

### Comparisons

- PH-NODE versus PH-NODE-no-pretraining tests initialization dependence.
- PH-NODE versus PH-NN-ODE-LieIntegrator is the main solver comparison.
- PH-NN-ODE-LieIntegrator versus PH-GP-ODE tests whether GP modeling preserves
  clean-data accuracy.
- PH-NN-SDE and PH-GP-SDE should learn diffusion close to zero.

### What to watch

- The solver-only comparison must use the same network architecture, initial
  weights, controls, loss, optimizer, and number of training steps.
- Do not apply mass-gauge normalization to only one side of the solver
  comparison.
- The custom environment also uses Lie-Heun. Test trained models on an
  independent PyBullet rollout to rule out an integrator-matching advantage.
- Check that PID data excite roll, pitch, and especially yaw.
- A nonzero learned diffusion on this dataset indicates that an SDE is using
  diffusion to hide model error.

## 7. D2 — Noisy deterministic ODE dataset

### Purpose

D2 isolates observation-noise robustness. The physical dynamics remain a
deterministic ODE.

### Dataset specification

- Uses the same clean latent trajectories, controls, initial states, and splits
  as D1.
- Observation noise: $\sigma_{\mathrm{obs}}=0.25$.
- Wind-force diffusion: $\sigma_f=0$.
- Wind-torque diffusion: $\sigma_\tau=0$.
- Deterministic external wind: $0$.
- Rotation noise is applied geometrically as
  $R_{\mathrm{obs}}=R\exp([\epsilon]_\times)$ with
  $\epsilon\sim\mathcal N(0,\sigma_{\mathrm{obs}}^2I_3)$.
- Position, linear velocity, and angular velocity receive additive Gaussian
  observation noise.
- Controls are not corrupted.
- Training observations are noisy, while the primary test target is clean.

The geometric observation-noise implementation is in
[`windy_quadrotor_datagen.py`](datasets/windy_quadrotor_datagen.py#L70).

### Models trained

- PH-NODE.
- PH-NODE-no-pretraining.
- PH-NN-ODE-LieIntegrator.
- PH-GP-ODE.
- PH-NN-SDE.
- PH-GP-SDE.

Neural-SDE is not trained on D2.

### Comparisons

- PH-NN-ODE-LieIntegrator versus PH-GP-ODE isolates the GP contribution under
  observation noise.
- PH-NODE versus PH-GP-ODE is the old-paper versus deterministic-GP comparison.
- PH-NN-SDE and PH-GP-SDE test whether an SDE incorrectly converts sensor noise
  into physical diffusion.

### What to watch

- Use identical noise realizations for all models.
- Use clean latent validation and test targets for the main comparison plots.
- A low loss against noisy observations is not automatically good.
- Learned process diffusion should remain close to zero because D2 has no wind.
- GP posterior uncertainty and SDE process diffusion must be plotted and
  described separately.
- Do not select hyperparameters after inspecting the clean test trajectories.

## 8. D3 — Windy SDE with clean observations

### Purpose

D3 isolates true physical process noise without observation noise.

### Dataset specification

- Uses the common D1-D4 physical constants and split.
- Observation noise: $\sigma_{\mathrm{obs}}=0$.
- Wind-force diffusion: $\sigma_f=0.5$.
- Wind-torque diffusion: $\sigma_\tau=0.1$.
- Deterministic external wind: $0$.
- Each trajectory uses an independent Brownian path.
- Force diffusion is multiplicative through $R^\top$.
- Torque diffusion is additive in body coordinates.

The environment implements the stochastic force and torque increments in
[`quadrotor.py`](envs/SE3_quadrotor/quadrotor.py#L452).

### Models trained

- PH-NODE.
- PH-NODE-no-pretraining.
- PH-NN-ODE-LieIntegrator.
- PH-GP-ODE.
- PH-NN-SDE.
- PH-GP-SDE.
- Neural-SDE is recommended.

### Comparisons

- PH-NN-ODE-LieIntegrator versus PH-NN-SDE isolates learned diffusion.
- PH-GP-ODE versus PH-GP-SDE isolates learned diffusion with GP subnetworks.
- PH-NN-SDE versus PH-GP-SDE isolates NN versus GP under the same SDE setting.
- PH-GP-SDE versus Neural-SDE tests port-Hamiltonian structure.
- PH-NODE versus PH-GP-SDE is the main old-versus-new windy comparison.

### What to watch

- A deterministic ODE predicts a conditional mean; it cannot reproduce every
  random sample path.
- Use shared Wiener increments for pathwise trajectory plots.
- Use independent model samples when comparing predicted distributions.
- Check that learned force diffusion changes with orientation.
- Check that learned torque diffusion is approximately constant.
- Do not use only a single stochastic rollout to rank models.
- Energy is not constant under stochastic forcing. Compare model energy
  evolution with GT energy evolution instead.

## 9. D4 — Noisy windy SDE dataset

### Purpose

D4 is the main difficult synthetic dataset. It combines sensor noise with
physical wind and tests all contributions together.

### Dataset specification

- Uses the same latent system, controls, initial states, and splits as D3.
- Observation noise: $\sigma_{\mathrm{obs}}=0.05$.
- Wind-force diffusion: $\sigma_f=0.5$.
- Wind-torque diffusion: $\sigma_\tau=0.1$.
- Deterministic external wind: $0$.
- Clean latent test trajectories are retained.
- Noisy test observations are retained separately.
- Observation-noise RNG and physical-dynamics RNG must be independent.

### Models trained

- PH-NODE.
- PH-NODE-no-pretraining.
- PH-NN-ODE-LieIntegrator.
- PH-GP-ODE.
- PH-NN-SDE.
- PH-GP-SDE.
- Neural-SDE is recommended.

### Comparisons

- PH-NODE versus PH-NN-ODE-LieIntegrator shows whether geometric integration
  still helps under noise and wind.
- PH-NN-ODE-LieIntegrator versus PH-GP-ODE shows GP noise robustness.
- PH-NN-ODE-LieIntegrator versus PH-NN-SDE shows the stochastic contribution.
- PH-GP-ODE versus PH-GP-SDE shows the same stochastic contribution for GP
  models.
- PH-NN-SDE versus PH-GP-SDE shows the GP contribution under process noise.
- PH-GP-SDE versus PH-NODE is the complete-method headline comparison.

### What to watch

- Observation noise and process noise can be confused by the learned model.
- Do not allow the diffusion network to absorb all observation noise.
- Plot GP uncertainty and physical diffusion separately.
- Keep dynamics seeds and observation-noise seeds fixed across models.
- Use clean latent trajectories for the primary rollout figures.
- Freeze noise levels and hyperparameters before inspecting test results.
- Check both mean prediction and stochastic spread; either one alone is
  incomplete.

## 10. D5 — Released real quadrotor dataset

### Purpose

D5 evaluates the models on real measured quadrotor trajectories and studies how
much nominal rigid-body knowledge is required.

### Dataset specification

- Source:
  `other_paper_codes/real_world_quadrotor_port_ham_neural_ode/LieGroupHamDL/training/examples/quadrotor_px4/data/se3hamdl_dataset_frd_thrust_torque.npz`.
- Samples: $22{,}378$.
- State-control channels: $22$.
- Recorded duration: approximately $108$ seconds.
- Timestep is nonuniform and approximately $4$ to $8$ ms.
- The paper describes $12$ collected flights.
- Recover the original flight boundaries before constructing windows.
- Proposed flight-level split:
  - $8$ flights for training,
  - $2$ flights for validation,
  - $2$ flights for testing.
- Preserve the released real-data thrust/torque input convention.
- Do not add artificial wind or observation noise to the primary real-data
  experiment.

### Models trained

- Released real PH-NODE with fixed nominal $m$, $J$, and $V$.
- PH-NODE-no-pretraining using the same real architecture but random trainable
  subnetwork initialization; its released fixed $m$, $J$, and $V$ remain fixed.
- PH-NN-ODE-LieIntegrator.
- PH-GP-ODE.
- PH-NN-SDE.
- PH-GP-SDE.

Neural-SDE is not trained on D5.

### Comparisons

- Released real PH-NODE versus PH-NODE-no-pretraining tests whether its cached
  initialization matters.
- Released real PH-NODE versus PH-NN-ODE-LieIntegrator compares the old gray-box
  model with geometric learned dynamics.
- PH-NN-ODE-LieIntegrator versus PH-GP-ODE tests GP robustness to real sensor
  noise.
- PH-NN-ODE-LieIntegrator versus PH-NN-SDE tests whether residual real-world
  variability benefits from an SDE.
- PH-GP-ODE versus PH-GP-SDE tests whether a learned physical diffusion improves
  held-out real-flight prediction.

### What to watch

- Do not randomly split overlapping windows. Neighboring windows would leak
  nearly identical samples into train and test.
- Split complete flights before creating windows.
- The released real model fixes nominal mass, inertia, and potential in
  [`SE3HamNODE_PX4.py`](other_paper_codes/real_world_quadrotor_port_ham_neural_ode/LieGroupHamDL/training/se3hamneuralode/SE3HamNODE_PX4.py#L96).
- The released real trainer uses Dopri5 internally, so D5 is not a pure
  RK4-versus-Lie-Heun experiment.
- True real-world diffusion is unknown. Learned uncertainty can be visualized,
  but it cannot be compared to an exact diffusion matrix.
- Do not call nominal mass or inertia real ground truth.
- Handle nonuniform timestamps rather than replacing them silently with one
  constant timestep.
- Do not compare against the old paper's published closed-loop tracking values
  unless the same hardware, controller, and reference trajectories are used.

## 11. Comparison rules shared by all datasets

1. Every model in one report uses the same dataset file.
2. Every model uses the same complete-trajectory split.
3. Every model receives the same control at every transition.
4. PH-NODE remains unchanged in the official-baseline row.
5. Only PH-NODE-no-pretraining removes the old identity-pretraining procedure.
6. The solver-only comparison must not change mass normalization, loss, control
   alignment, architecture, or training budget.
7. Synthetic noisy datasets use clean latent validation and test targets.
8. Rotation-constraint plots use raw $R$ without projection.
9. Potential curves are centered because $V$ is identifiable only up to an
   additive constant.
10. Learned $M$, $V$, $D$, and $G$ are gauge-aligned before comparing their
    magnitudes.
11. Common state-error curves are used to compare models with different native
    objectives.
12. Raw MSE, ELBO, NLL, and KL curves are not treated as the same quantity.
13. Checkpoints are selected using validation data only.
14. SDE models use the same Wiener increments as GT for pathwise plots.
15. Distribution plots use multiple independent stochastic samples.
16. Final comparisons use paired training and evaluation seeds.
17. Both training steps and computational cost should be recorded because RK4
    and Lie-Heun use different numbers of vector-field evaluations.

## 12. Recommended execution order

1. D0 official PyBullet reproduction.
2. D1 clean custom ODE.
3. D2 observation-noise ODE.
4. D3 windy SDE.
5. D4 noisy windy SDE.
6. D5 real quadrotor.

Do not start the full noise and wind sweeps until one seed of D0-D4 completes
successfully and produces comparable reports.

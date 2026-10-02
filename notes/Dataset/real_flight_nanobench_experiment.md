# Real-flight experiment: NanoBench (Crazyflie 2.1), shape-disjoint train/test

19 September 2026. First training of the SE(3) port-Hamiltonian models on **measured** flights instead of the
PyBullet simulator, with a train/test split in which no trajectory shape is shared.

## 1. Why this dataset

Of the public quadrotor sets surveyed (NeuroBEM, Blackbird, Neural-Fly, the IDSIA nano-drone benchmark,
NanoBench), **NanoBench** is the closest fit:

| requirement | NanoBench |
|---|---|
| same vehicle class as our simulator | Crazyflie 2.1, the airframe our CF2 model is built from |
| full state at our step size | Vicon pose, velocity and rate at 100 Hz, i.e. `dt = 0.01` s exactly as in HARD |
| an input we can convert to a wrench | per-motor PWM commands, plus battery voltage |
| several distinct trajectory shapes | 12 families, enough for a shape-disjoint split |
| redistributable | BSD 3-Clause |

Source: Ullah and Baca, *NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor System Identification,
Control, and State Estimation*, arXiv:2603.09908, <https://github.com/syediu/nanobench>.
107 flight CSVs are used, each 52 synchronized columns; the raw clone lives in `tmp/nanobench_raw/`.

## 2. Conversion (`envs/quadrotor_se3/datagen/real/convert_nanobench.py`)

The target is the same 22-channel layout as the simulator datasets, so no trainer or report code changes:
$$s = [\,x_w(3),\ \mathrm{vec}(R)(9),\ v_b(3),\ \omega_b(3),\ u(4)\,]$$

### 2.1 State: Vicon only

$x_w$, $R$ (from the Vicon quaternion), $v_b = R^\top v_w$ and $\omega_b = R^\top \omega_w$; the CSV stores
velocity and angular rate in the world frame. The onboard gyro is deliberately **not** used: the firmware
telemetry shares one radio link and is delivered below its nominal rate and then interpolated, and it matches
$\omega_b$ derived from $\mathrm dR/\mathrm dt$ only at correlation 0.80, with the yaw-rate amplitude 0.63 of
Vicon's, whereas $R^\top\omega_w$ matches at 0.97. Nothing is smoothed. The Vicon rate carries roughly
0.10 rad/s of differentiation jitter, which is exactly what the learned per-block $\sigma_\omega$ is for.

### 2.2 Input: battery-compensated PWM through the CF2X mixer

The stock Förster relation $\mathrm{rpm} = 0.2685\,\mathrm{PWM} + 4070.3$ (the same constants the CF2 URDF and
gym-pybullet-drones use) holds at one supply voltage only. The motor sees duty $\times V_{bat}$, and the
firmware raises PWM as the cell drains, so PWM alone over-states thrust as the battery falls. Using

$$\mathrm{rpm}_i = 0.2685\,\mathrm{PWM}_i\,\frac{V_{bat}}{V_{nom}} + 4070.3,
\qquad f_i = k_f\,\mathrm{rpm}_i^2,\qquad V_{nom} = 3.8\ \mathrm V$$

removes it. Measured on the two battery-drain hover flights, where the true thrust must equal the weight:

| model | hover thrust / $mg$ | CV | first decile / last decile |
|---|---|---|---|
| nominal (no voltage term) | 1.18 – 1.23 | 0.039 – 0.052 | 0.86 – 0.90 |
| battery-compensated | **0.96 – 1.00** | **0.009 – 0.010** | **0.98 – 0.99** |

The wrench then comes from the CF2X mixer (the Crazyflie 2.1 is an X frame), in the gym-pybullet-drones
convention with $a = L/\sqrt2$ and $\kappa = k_m/k_f$:

$$T=\textstyle\sum_i f_i,\quad \tau_x = a(-f_1-f_2+f_3+f_4),\quad \tau_y = a(-f_1+f_2+f_3-f_4),
\quad \tau_z = \kappa(-f_1+f_2-f_3+f_4)$$

The roll and pitch sign patterns are confirmed **from the data**: regressing the Vicon angular acceleration on
the four motor thrusts returns the patterns $(-,-,+,+)$ and $(-,+,+,-)$. Any remaining constant error in this
map is absorbed by the learned control level $\Lambda_0$, since only the product $M^{-1}g$ is identifiable.

### 2.2b Is the reconstructed $u$ actually right?

`validate_input_reconstruction.py` tests it against quantities it was never fitted to. **The thrust channel
passes; the torque channel does not.**

**A. Independent sensor.** The IMU measures specific force, whose body-$z$ component is $T/m$ plus a small drag
term. Over 58 flights the reconstructed $T$ and $m\,g\,a_{imu,z}$ agree with a mean ratio of
**0.978 ± 0.023**. Per-flight correlation median 0.874, capped by the interpolated IMU stream, not by the mean.

**B. Parameter-free translational fit.** With no unknown constants,
$\dot v_w + g e_3 = (T/m)\,R e_3 + \text{drag}/m$, so a least-squares fit of the measured left side on
$T\,R e_3$ returns $1/m$. It gives an implied mass of **39.96 g against the published 40.85 g, −2.2 %**, with
$R^2 = 0.708$ ($x$ 0.742, $y$ 0.729, $z$ 0.252; $z$ is low because the hover balance leaves little variance
there). A wrong attitude convention or a wrong thrust direction would destroy this fit, so magnitude *and*
direction are confirmed.

**D. Rotational consistency.** In integral form, so $\omega$ is never differentiated twice,
$$J\big(\omega(t+W) - \omega(t)\big) = \int_t^{t+W}\!\tau\,\mathrm dt \;-\; D\!\int\!\omega\,\mathrm dt \;-\; \tau_{trim}W .$$
Fitted on the training split with a lag scan for the motor time constant:

| lag | $R^2$ roll | $R^2$ pitch | $R^2$ yaw | $J_{xx}$ | $J_{yy}$ | $J_{zz}$ |
|---|---|---|---|---|---|---|
| 0 ms | 0.237 | 0.105 | 0.000 | 7.9e−6 | 7.8e−6 | 3.2e−6 |
| 30 ms | **0.250** | 0.097 | 0.000 | 1.56e−5 | 1.26e−5 | 3.0e−6 |
| 100 ms | 0.065 | 0.021 | 0.001 | 6.4e−6 | 7.7e−6 | 2.3e−6 |

Three problems, all visible here:
1. A constant **trim torque of $(+2.95, -2.44, +2.46)\times10^{-4}$ N m is needed, which is the size of the
   median $|\tau|$ itself** ($2.71\times10^{-4}$ N m). Real motors are not matched, so the nominal mixer's zero
   is not the vehicle's zero.
2. Even at the best lag the torque explains only **25 % of roll, 10 % of pitch and 0 % of yaw**. The yaw
   channel carries no usable information.
3. The optimum sits at a **20–30 ms lag**, the motor time constant. Commanded PWM is not delivered thrust, and
   torque is a small difference between large motor commands, so it is exactly where that error concentrates.
   The fitted $J_{xx}, J_{yy} \approx 1.3$–$1.6\times10^{-5}$ are then the right order and roughly symmetric,
   which is reassuring about the mixer but not about the signal quality.

**C. Forward integration, the decisive test.** The analytic rigid body (mass published, inertia nominal) driven
from one measured $x_0$ by the reconstructed $u$ alone, against a constant-velocity extrapolation from the same
$x_0$ (position RMSE in metres):

| | 0.5 s | 1.0 s | 3.0 s | VPT median |
|---|---|---|---|---|
| physics, $c = 0$ | diverged | | | |
| physics, $c = 0.5$ | 0.303 | 1.835 | 8.589 | 0.41 s |
| physics, $c = 1.0$ | 0.249 | 1.430 | 6.196 | 0.43 s |
| constant velocity | **0.166** | **0.545** | **2.877** | — |

The physics rollout is **worse than constant-velocity extrapolation at every horizon**.

Note what this first version confounds: it uses the *published* mass but a *nominal* inertia, and the
PyBullet-shaped dissipation $D_v = mc(1+\|v\|)I$, which is a simulator artefact rather than real Crazyflie
aerodynamics. A failure could therefore be blamed on the parameters instead of on $u$.

**C2. The same test with the parameters fitted to the data**, which removes that confound. Fitted on the
training split: $m = 40.20$ g (published 40.85, −1.6 %), linear drag $k_v/m = 0.387\ \mathrm s^{-1}$,
$J = (1.56, 1.26, 0.303)\times10^{-5}$, $k_\omega = (1.89, 1.78, -0.518)\times10^{-4}$ (note the *negative*
yaw damping, which is unphysical and matches yaw $R^2 = 0$), trim as above, with the 30 ms torque lag applied.
Evaluated on the held-out split:

| variant | 0.5 s | 1.0 s | 3.0 s |
|---|---|---|---|
| physics, fitted parameters, full rollout | **NaN** | NaN | NaN |
| physics, fitted parameters, **measured attitude** | **0.056** | **0.191** | **1.368** |
| constant velocity (reference) | 0.166 | 0.545 | 2.877 |

This separates the two channels cleanly:

- Feed the measured attitude and integrate only translation with the reconstructed thrust, and the model beats
  the constant-velocity reference by about **3× at 0.5 s and 1 s** and 2× at 3 s. The thrust channel carries
  real information.
- Integrate the attitude from the reconstructed torque and the rollout **overflows**, even with fitted inertia,
  fitted damping, the trim removed and the best lag applied.

Isolating the rotational channel alone, the integrated body rate departs from the measurement by
**10.8 rad/s within 0.1 s**, against a measured $|\omega|$ of median 0.31 and p90 0.94 rad/s, and it leaves a
0.2 rad/s tube **at the first step on all 25 flights**. Ten substeps per sample change nothing (10.80 versus
10.80 rad/s at 0.1 s), so this is not an explicit-integrator artefact.

**C3. Independent implementation: PyBullet replay.** `pybullet_roundtrip.py` pushes the same motor commands
through gym-pybullet-drones instead of our analytic field: PyBullet applies its own CF2X rotor forces and
torques and integrates the rigid body, with mass overridden to 40.85 g and inertia to the fitted values. Ten
1 kHz physics steps per 100 Hz sample, started from the measured state. Position RMSE (m) over the 25 held-out
flights:

| variant | 0.1 s | 0.5 s | 1.0 s | 3.0 s |
|---|---|---|---|---|
| full, PyBullet integrates attitude | 0.004 | 0.376 | 1.817 | 8.853 |
| **measured attitude written back** | 0.004 | **0.075** | **0.240** | **1.159** |
| constant velocity (reference) | 0.008 | 0.166 | 0.545 | 2.877 |

The two implementations agree closely (our analytic rollout with the measured attitude gave 0.056 / 0.191 /
1.368 against PyBullet's 0.075 / 0.240 / 1.159), so the mismatch is a property of the data and not of our code.
Over the first 0.1 s even the full rollout beats the reference; the attitude error takes about 0.2-0.3 s to
dominate.

**What the source publishes.** Checked in the cloned repository and in the paper (arXiv 2603.09908):

- The paper states the same rigid-body model we use, $m\ddot p = R[0,0,\sum T_i]^\top - mge_3 - D_t\dot p$ and
  $J\dot\omega = \tau - \omega\times J\omega - D_r\omega$ with diagonal drag matrices, which is why a linear
  drag was the right thing to fit.
- It gives the flying mass (40.85 g), a peak thrust of about 0.6 N at 4.2 V and a thrust-to-weight of roughly
  2.2. Our independent calibration gives 0.595 N, which matches.
- It does **not** publish inertia, thrust or torque coefficients, or an arm length. Förster is cited only for
  having "established baseline quadratic thrust parameters"; the physics baseline's "inertia and drag
  parameters are taken from" Busetto et al., the IDSIA brushless-Crazyflie dataset. It mentions "a quadratic
  motor-thrust map" without coefficients.
- It publishes **no control input in physical units**. The repository's own baselines feed normalised PWM,
  `acts = motor_motor_m* / 65535`, straight into black-box MLPs, and its loader still defaults to the 27 g
  airframe mass. `benchmarks/task1_sysid` is an uninitialised submodule, so its converter is not in the clone.
- The paper independently flags both failure modes we measured: "Small thrust-modeling errors compound through
  numerical integration and corrupt the translational state", and "coreless DC motors introduce deadbands and
  response lags absent in brushless propulsion".
- For scale, their own learned MLP dynamics reports (in `debug/sanity_check_dynamics.py`) a rollout position
  error of 0.193 m at 0.5 s and 0.533 m at 1 s, close to the constant-velocity reference and worse than our
  physics rollout with the measured attitude.

**E. Is the conversion itself to blame? No.** The dataset does record a control input, the per-motor PWM
command, so the fair question is whether converting it to a wrench destroyed something. Tested by letting least
squares pick *any* linear combination of the four motor channels, which is the most generous form possible, on
43 flights and 143 758 windows of 0.10 s:

$$\Delta\omega_k \sim [c_1..c_4]\cdot\!\int\!(\text{motor channels}) + d\!\int\!\omega_k + \tau_{trim}W$$

| input parameterisation | $R^2$ roll | $R^2$ pitch | $R^2$ yaw |
|---|---|---|---|
| raw PWM | 0.103 | 0.046 | 0.009 |
| PWM² | 0.104 | 0.046 | 0.009 |
| PWM² × $(V/V_{nom})^2$ | 0.107 | 0.045 | 0.011 |
| battery-compensated thrust $f_i$ (ours) | 0.106 | 0.044 | 0.011 |

All four are the same. Feeding the raw PWM straight in, with no conversion at all, does not help: the rotational
information is absent from the *commanded* PWM, not lost in our mapping.

The identical test on the translational channel, same flights and windows:

$$\Delta v_w + g e_3\,\Delta t \sim [c_1..c_4]\cdot\!\int\!(f_i\,Re_3) + d\!\int\! v_w$$

| axis | $R^2$ |
|---|---|
| x | 0.890 |
| y | 0.910 |
| z | 0.343 (hover balance leaves little variance) |
| **all three jointly** | **0.997**, implied mass 40.2 g against the published 40.85 g |

So the same motor commands explain 99.7 % of the translational velocity change and about 10 %, 4 % and 1 % of
the three rotational ones.

**Why PWM cannot be the model's $u$ directly.** The port-Hamiltonian form is control-affine, $\dot z = (J -
R)\nabla H + g(z)u$, so $g(z)u$ must be *linear* in $u$. The PWM-to-force map is quadratic and depends on the
cell voltage, so raw PWM cannot be $u$ for any constant or state-dependent $g$. What is admissible is any
quantity proportional to per-motor thrust: either the four $f_i$ or, as here, the wrench $M_{mix}f$, since the
two differ by a constant invertible matrix that the learned control level $\Lambda_0$ absorbs. This is the same
choice our own generators expose as `input_mode: wrench` versus `motor`, where motor mode stores
$(\mathrm{rpm}_i/\mathrm{rpm}_{max})^2$, again proportional to thrust.

**F. The paper's own Task 1 results say the same thing.** The benchmark *does* define the rotational problem:
equation (4) is $J\dot\omega = \tau - \omega\times J\omega - D_r\omega$, and Task 1 scores attitude and angular
velocity alongside position and velocity, with the attitude error as a relative-quaternion angle. What their
own baselines achieve on those channels (Table V, MAE at $h = 1, 10, 50$ steps):

| model | $\mathrm{MAE}_R$ (rad) | $\mathrm{MAE}_\omega$ (rad/s) |
|---|---|---|
| Naive (constant state) | 0.0033 / 0.0285 / **0.0915** | 0.0402 / 0.2602 / **0.4524** |
| Physics | 0.0019 / 0.0198 / 0.1425 | 0.0402 / 0.2602 / 0.4524 |
| Residual MLP | 0.0084 / 0.1051 / 0.1971 | 0.2043 / 1.1173 / 1.3046 |
| Physics + residual | 0.0065 / 0.1359 / 0.1825 | 0.8784 / 1.5799 / 1.1781 |
| LSTM | 0.0162 / 0.1339 / 0.3059 | 0.1256 / 0.8159 / 5.1525 |

**Not one of the five baselines beats the naive constant-state model on either rotational channel at
$h = 50$.** On angular velocity the learned models are 2.6 to 11 times worse than naive. And the physics row is
identical to naive by construction, which the paper explains outright:

> "The physics model produces identical angular velocity predictions to naive at all horizons because the
> implementation holds $\dot{\boldsymbol{\omega}} \equiv 0$, which reflects the difficulty of modeling torque
> transients from coreless DC motors without dedicated angular acceleration data."

So the authors' own physics baseline declines to integrate equation (4) at all, for exactly the reason our
test D measures. Their translational numbers match ours too: the physics model reaches 0.3 mm position MAE at
one step, then diverges past about 15 steps (0.15 s) and is 2.2 times worse than naive by 50 steps, with
velocity MAE growing from 51 mm/s to 2.57 m/s because "small thrust-modeling errors compound through numerical
integration". Our PyBullet replay crosses the constant-velocity reference between 0.1 s and 0.5 s, the same
place.

**Consequence for this dataset.** NanoBench supports *translational* identification well and *rotational*
identification poorly, because it records commanded PWM rather than measured rotor speed. A learned $g$ can
absorb the constant trim (through a thrust-to-torque column), but not the 20–30 ms motor lag or the lost
high-frequency torque content. Any claim built on this data must be restricted to the translational operators,
or the data must be replaced by a set with **measured** rotor speeds - the IDSIA nano-drone benchmark
(arXiv 2512.14450, Crazyflie 2.1 Brushless with bidirectional DSHOT eRPM telemetry) is the obvious candidate.

Full record: `NANOBENCH_input_validation.txt`.

### 2.3 Trimming and chunking

Longest contiguous airborne run (above 0.35 m with the motors commanded above idle), minus 0.5 s at each end,
cut into non-overlapping 1001-sample (10.01 s) flights. A chunk is dropped when more than 2 % of its samples
have a motor at the PWM ceiling, because the commanded value is then not what was delivered.

## 3. The shape-disjoint split

| split | trajectory families | flights × 10 s | seconds |
|---|---|---|---|
| train | multi-sine excitation, circle, figure-eight, oval, linear ramp, staircase climb, random waypoints, battery-drain hover | 89 | 890 |
| test | **helix, star, trefoil knot, lissajous** | 25 | 250 |

The converter refuses to run if the two family lists intersect, and the audit records
`families_shared_between_train_and_test: []`. Coverage overlaps while the shapes do not:

| | speed p50 / p90 / max (m/s) | rate p50 / p90 / max (rad/s) | tilt p50 / p90 / max (deg) | $T/mg$ mean |
|---|---|---|---|---|
| train | 0.58 / 1.03 / 1.80 | 0.20 / 0.82 / 5.75 | 3.2 / 9.8 / 26.2 | 0.986 |
| test | 1.02 / 1.17 / 1.63 | 0.31 / 0.94 / 6.00 | 6.7 / 14.6 / 31.4 | 0.985 |

The held-out set is slightly *faster* and more tilted than training, so the generalisation test is conservative.

Files: `NANOBENCH_CF2_10s_h0p01_clean.pkl`, `_audits.json`, `_dataset_analysis.pdf` (6 pages),
`convert_nanobench.py`, `plot_nanobench_dataset.py`, `LICENSE_nanobench`.

## 4. Training recipe

Both models keep their published simulator recipe unchanged; the **only** functional difference from the
reported $\sigma = 0.25$ runs is `data.dataset_path` (verified by diffing the configs). Same window curriculum
($K = 50$ then $K = 100$, stride 10), batch 256, 3000 steps per stage.

| | PH-GP-LieIMEX | PH-NODE-RK4 |
|---|---|---|
| loss | $SE(3)$ NLL, four learned $\sigma_b$ from 0.3 | trajectory MSE |
| latent initial state | MAP, per window | none |
| optimiser | Prodigy on the levels, Adam $10^{-3}$ elsewhere | Adam $5\times10^{-4}$ |
| gradient clipping | global norm 10 | none (`gradient_clip_norm: null`) |
| mass pretraining | none | 200 steps toward the published mass and nominal CF2 inertia |
| physical constants given | none | mass and nominal inertia, through the pretraining |

Configs: `src/models/SE3_Quadrotor/configs/19-09-2026-03-0{0,1}_ph_gp_lie_imex.yaml` and
`19-09-2026-03-1{0,1}_ph_node.yaml`. Chain runner:
`experiments/quadrotor/.chain_logs/run_chain_nanobench.sh`.

No synthetic observation noise is added anywhere: the measurements are the only data, and the GP still starts
its four likelihood scales at a neutral 0.3, so no noise level is supplied to any model.

## 5. Results

### 5.1 PH-NODE-RK4: training diverged

Stage 1 stopped at **step 47** with `objective=nan, gradient_norm=nan`, during the main loop (the 200
mass-pretraining steps and step 0 completed). The recipe is byte-identical to the simulator run that trained
successfully on HARD at $\sigma = 0.25$, so this is a data-driven instability of the unclipped published
recipe, not a configuration slip. It is the fifth NaN from that recipe (`gradient_clip_norm: null`) recorded in
this project. The converted data was checked and is not the cause: every channel is finite, rotations satisfy
$\|R^\top R - I\| < 1.2\times10^{-7}$, and the largest per-sample jumps are 1.9 cm in position, 0.23 m/s in
velocity and 0.027 N in thrust, all consistent with the flight envelope. The angular rate is the noisy channel
(per-sample jumps up to 2 rad/s, i.e. implied $|\dot\omega|$ up to 212 rad/s²), which is genuine Vicon
differentiation jitter.

Not fixed, by standing instruction. The obvious knob is `optimizer.gradient_clip_norm` (the GP recipe uses 10).

### 5.2 PH-GP-LieIMEX: training also diverged, at step 935

Stage 1 stopped at **step 935** with `objective=-5.28` (finite) and `gradient_norm=inf`, so the guard fired on
an overflowing gradient rather than a NaN objective. The trace explains it: the four learned likelihood scales
fall steadily from their neutral start,

| | step 0 | 1/4 | 1/2 | 3/4 | last |
|---|---|---|---|---|---|
| $\sigma_{position}$ | 0.300 | 0.240 | 0.186 | 0.147 | 0.120 |
| $\sigma_{attitude}$ | 0.300 | 0.242 | 0.188 | 0.149 | 0.122 |
| $\sigma_{velocity}$ | 0.300 | 0.239 | 0.186 | 0.149 | 0.123 |
| $\sigma_{\omega}$ | 0.300 | 0.360 | 0.370 | 0.360 | 0.358 |
| NLL total | 38.3 | −2.9 | −4.9 | −7.4 | −8.2 |

and the NLL weights each residual by $1/(2\sigma_b^2)$, so the loss surface stiffens as the fit improves. In
`float32` (the recipe sets `runtime.jax_enable_x64: false`) one backward pass eventually overflowed. Global-norm
clipping at 10 cannot rescue this, because clipping an already-infinite gradient yields NaN. Not fixed, by
standing instruction.

Training curves up to the failure were healthy and still improving: window test loss 0.875 → 0.160, held-out
1 s open-loop position RMS 0.240 → 0.138 m, held-out 3 s 0.962 → 0.651 m.

### 5.3 What the last saved checkpoint achieves on the held-out shapes

The GP run saved `checkpoint_step_00500.pkl` before failing, so the pipeline could be scored end to end.
This is an **early stage-1 checkpoint** (0.5 s windows, one sixth of the planned budget), not the intended
final model, and the numbers should be read as a floor.

`experiments/quadrotor/eval_runs/19-09-00-00_real-data-open-loop_NANOBENCH_GP-step500/`, produced by
`python -m src.models.SE3_Quadrotor.comparision.evaluate_real_dataset --selected-step 500`.
25 held-out flights, 10 s, true $x_0$ and the recorded wrench:

| metric | value |
|---|---|
| position RMSE over 10 s | 1.293 m |
| valid prediction time, median | 0.91 s (ensemble 0.89 s, [0.55, 1.36]) |
| velocity RMSE | 0.988 m/s |
| attitude error, geodesic | 71.6° |
| angular-rate RMSE | 2.150 rad/s |
| SO(3) violation $\|R^\top R - I\|$ | 7.3 × 10⁻⁶ (float32) |
| coverage at 2σ (target 0.954) | 0.566 |
| Spearman ρ(σ, error) | +0.76 |
| plug-in versus predictive-mean gap | 0.527 m |
| rollout compute | 32.6 ms per 10 s flight |

The calibration pattern matches the simulator result: the posterior band is too narrow at long horizon
(coverage 0.57 against a 0.954 target) while σ still ranks the error well (ρ = +0.76), so the uncertainty is
informative but over-confident.

The identification table is deliberately absent everywhere above: a real vehicle has no analytic operators, so
there is nothing to compare the six gauge-invariant products against. What is measured is the honest quantity,
open-loop prediction from one measured initial state driven by the recorded wrench.

## 6. Open items

Both failures are training-stability problems of recipes that were tuned on simulator data, and both are left
untouched by standing instruction. The candidate one-line changes, for a decision:

- PH-NODE-RK4: `optimizer.gradient_clip_norm: null` → a finite value (the GP recipe uses 10).
- PH-GP-LieIMEX: `runtime.jax_enable_x64: true` (the overflow is a float32 range problem), or a floor on the
  learned likelihood scales (`gp.fixed_observation_sigma`, or a minimum on $\sigma_b$), or a smaller
  `optimizer.learning_rate` for the likelihood group.

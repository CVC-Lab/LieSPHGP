# Real-flight experiment 2: IDSIA nano-drone benchmark (Crazyflie 2.1 Brushless)

19 September 2026. Replaces the NanoBench attempt (`real_flight_nanobench_experiment.md`), whose torque channel
was unusable because its input is a *commanded* PWM. Here the input is the **measured** motor angular velocity
from ESC telemetry.

Source: Busetto et al., *Nonlinear System Identification for a Nano-drone Benchmark*, Control Engineering
Practice; <https://github.com/idsia-robotics/nanodrone-sysid-benchmark>. Raw clone in `tmp/idsia_raw/`.

## 1. What it provides that NanoBench did not

| | NanoBench | IDSIA |
|---|---|---|
| input | commanded PWM, 0–65535 | **measured** rotor speed $\Omega_i$ (rad/s) |
| thrust map | none published, we calibrated it | published: $K_t = 3.72\times10^{-8}$, $K_c = 7.74\times10^{-12}$, arm 0.0353 m |
| mass, inertia | mass only (40.85 g) | $m = 0.045$ kg, $J = \mathrm{diag}(2.3951, 2.3951, 0.32347)\times10^{-5}$ |
| motors | coreless DC, deadband and lag | brushless |
| official held-out shape | no | yes, the *melon* trajectory |

The benchmark's own mixer, applied to the measured speeds and kept in physical units rather than their
normalised ones (`models/models.py`):

$$T = K_t\!\sum_i\Omega_i^2,\quad
\tau_x = K_t a\big[(\Omega_3^2+\Omega_4^2)-(\Omega_1^2+\Omega_2^2)\big],\quad
\tau_y = K_t a\big[(\Omega_2^2+\Omega_3^2)-(\Omega_1^2+\Omega_4^2)\big],\quad
\tau_z = K_c\big[(\Omega_1^2+\Omega_3^2)-(\Omega_2^2+\Omega_4^2)\big]$$

None of these constants is given to our models; they are recorded in the settings and used only by the
validation script.

## 2. Conversion (`envs/idsia_quadrotor_se3/datagen/convert_idsia.py`)

State from motion capture: $x_w$, $R$ from the quaternion, $v_b = R^\top v_w$ (the CSV velocity is world frame)
and $\omega_b$ taken directly, since the benchmark's own physics model uses it as a body rate. Flights are cut
into non-overlapping 1001-sample (10.01 s) pieces, the same length as our other datasets. No trimming is
needed: the source files are already restricted to the flying interval.

**Split, the benchmark's own and already shape-disjoint:**

| split | families | flights × 10 s | seconds |
|---|---|---|---|
| train | chirp, random, square | 44 | 440 |
| test | **melon** | 18 | 180 |

| | speed p50/p90/max (m/s) | rate p50/p90/max (rad/s) | tilt p50/p90/max (deg) | $T/mg$ |
|---|---|---|---|---|
| train | 0.87 / 1.76 / 2.72 | 0.29 / 1.18 / 8.38 | 7.9 / 22.8 / 76.8 | 1.024 |
| test | 1.50 / 1.91 / 3.15 | 0.57 / 1.20 / 7.52 | 15.9 / 25.7 / 68.8 | 1.047 |

A much more aggressive envelope than NanoBench (tilt to 77° against 31°, rate to 8.4 against 6.0 rad/s), and
closer to our simulator's.

## 3. Input validation, the same four tests as NanoBench

`validate_input_reconstruction.py`, record in `IDSIA_input_validation.txt`.

| test | IDSIA | NanoBench |
|---|---|---|
| A, thrust against the accelerometer | **0.9996 ± 0.0057** | 0.978 ± 0.023 |
| B, parameter-free mass fit | **45.03 g vs 45.0 published, +0.1 %**, $R^2$ 0.883 | 39.96 vs 40.85, −2.2 %, $R^2$ 0.708 |
| D, rotational $R^2$ roll/pitch/yaw | **0.301 / 0.345 / 0.432 at zero lag** | 0.250 / 0.097 / **0.000** at a 30 ms lag |
| D, trim torque relative to median \|τ\| | **0.35 – 0.42×** | ≈ 1.0× |

Three qualitative wins: the thrust reconstruction is essentially exact, **yaw carries information for the first
time**, and the best lag is **zero**, which is what "measured rotor speed" buys over "commanded PWM" on a
brushless airframe.

**Test C, forward integration, is still the weak point.** Analytic rigid body from one measured $x_0$ driven by
$u$ alone, fitted parameters ($m = 44.74$ g, linear drag 0.360 s⁻¹, $J_{xx,yy} = 3.37, 3.96\times10^{-5}$),
position RMSE in metres over the 18 held-out flights:

| variant | 0.1 s | 0.5 s | 1.0 s | 3.0 s |
|---|---|---|---|---|
| full rollout | 0.011 | 0.382 | 2.69 | 23.9 |
| **measured attitude written back** | **0.011** | **0.074** | **0.206** | **0.580** |
| constant velocity (reference) | 0.020 | 0.484 | 1.412 | 4.239 |

The translational channel is now excellent: with the measured attitude it beats the reference by 6.5× at 0.5 s
and **7.3× at 3 s**, against NanoBench's 1.4× at 3 s. The full rollout still diverges, with a median attitude
error of about 102° at 1 s, and sweeping $J_{zz}$ over its published value, $10^{-5}$ and the CF2 nominal
changes that by less than 3°, so the yaw inertia is not the cause. With $R^2 \approx 0.3$–0.43 on the rate
equation, roughly two thirds of the angular acceleration is still unexplained by a rigid body with constant
$J$ and linear damping, and integrating that for 100 steps is enough to lose the attitude.

**The honest framing of the horizon.** The benchmark's own protocol evaluates open-loop prediction to **0.5 s
(50 steps)**, not ten seconds. At that horizon the reconstructed $u$ plus a rigid body already beats the naive
baseline (0.382 against 0.484 m). Our usual 10 s open-loop convention is far beyond what any model identified
from this data can support, so real-data results should be reported at 0.5 s and 1 s, with the longer horizons
shown only as the divergence they are. The residual at $R^2 \approx 0.35$ is precisely what a learned
state-dependent $M_2^{-1}$, $D_\omega$ and $g$ have the freedom to capture and a constant-parameter rigid body
does not, which is the case for the learned models rather than against them.

## 4. Training

Configs `src/models/SE3_Quadrotor/configs/19-09-2026-04-0{0,1}_ph_gp_lie_imex.yaml` and
`19-09-2026-04-1{0,1}_ph_node.yaml`; the published simulator recipes with only `data.dataset_path` changed.
Chain runner `experiments/quadrotor/.chain_logs/run_chain_nanobench.sh`, GP on GPU 0 and PH-NODE on GPU 1.

### 4.1 PH-NODE-RK4: diverged again

Stage 1 stopped at **step 52** with `objective=nan, gradient_norm=nan`, against step 47 on NanoBench. The same
unclipped published recipe (`gradient_clip_norm: null`) now fails at essentially the same point on two
independent real datasets with different vehicles, different input types and different noise characteristics,
while training cleanly on simulator data. That is strong evidence the recipe, not the data, is the problem.
Not fixed, by standing instruction; the knob is `optimizer.gradient_clip_norm` (the GP recipe uses 10).

### 4.2 PH-GP-LieIMEX: stage 1 completed, stage 2 diverged

**Stage 1 ran all 3000 steps cleanly**, unlike on NanoBench where the GP died at step 935. The held-out
open-loop validation improved monotonically and then plateaued: 1 s position RMS 0.632 m at step 0, 0.246 at
500, 0.208 at 2000, **0.203 m at step 3000**.

**Stage 2 (K = 100, 1 s windows) stopped at step 129** with `objective=336.2` (finite) and
`gradient_norm=inf`. Same signature as the NanoBench GP failure and the same underlying mechanism as
PH-NODE's: one window in one batch makes the unrolled rollout diverge, the gradient overflows in float32, and
global-norm clipping cannot help because clipping an already-infinite gradient yields NaN. Doubling the window
from 0.5 s to 1 s doubles the number of steps over which a rollout can run away, which is why stage 2 is the
fragile one. Not fixed, by standing instruction.

The reported model is therefore the **completed stage-1 GP** (K = 50, 3000 steps).

### 4.3 Held-out result on the melon trajectory

`experiments/quadrotor/eval_runs/19-09-00-30_IDSIA-melon-heldout_GP-stage1_{0.5,1.0}s/`. Position RMSE in
metres over the whole window, averaged across every segment of the 18 held-out flights. The two rigid-body rows
use parameters fitted on the *training* split; the last of them is an oracle, because it is handed the measured
attitude at every step and only integrates translation.

| horizon | constant velocity | rigid body (fitted) | rigid body + **measured attitude** (oracle) | **PH-GP-LieIMEX** |
|---|---|---|---|---|
| 0.5 s (the benchmark's own protocol) | 0.1848 | 0.1539 | *0.0410* | **0.0633** |
| 1.0 s | 0.6707 | 1.2364 | *0.1078* | **0.2235** |

The GP beats constant-velocity extrapolation by 2.9× at 0.5 s and 3.0× at 1 s, and beats the fitted rigid body
by 2.4× and 5.5×. Only the attitude oracle is better, and it is solving an easier problem. This is the first
real-flight result in the project where the learned model wins on an unseen trajectory shape while integrating
its own attitude from the input.

Other channels, at 0.5 s and 1 s: attitude error 10.5° and 15.5°, velocity RMSE 0.368 and 0.681 m/s, angular
rate 0.940 and 1.378 rad/s, $SO(3)$ violation 0.0, compute 8.2 and 18.6 ms per flight.

**Calibration is poor.** Coverage at 2σ is 0.247 and 0.230 against a 0.954 target, noticeably worse than the
0.51–0.57 seen in simulation, so the posterior band is far too narrow on real data. The ranking is still
informative, with Spearman ρ(σ, error) of +0.85 and +0.89. Worth stating as a limitation rather than hiding:
the model knows *where* it is uncertain but badly understates *how* uncertain.

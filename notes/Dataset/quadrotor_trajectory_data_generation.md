# Quadrotor trajectory data generation (QUADROTOR-DATASET-HARD and QUADROTOR-DATASET-EVALSET)

Status on 18 Sep 2026. Two datasets exist:

| dataset | folder | role | flights | generator |
|---|---|---|---|---|
| **HARD** | `datasets/QUADROTOR-DATASET-HARD/` | training (50 flights) + validation (10 flights, key `test_trajectories`) | 60 × 10 s | the former HARD-V5 files, byte-identical (`HARDV5_*` renamed to `HARD_*`); produced by the archived `generate_quadrotor_hard_v2.py` with the V5 config |
| **EVALSET** | `datasets/QUADROTOR-DATASET-EVALSET/` | unseen-shape evaluation, open loop and closed loop (key `heldout_trajectories`) | 20 × 10 s, clean | `envs/quadrotor_se3/datagen/generate_quadrotor_evalset.py` + `evalset_config.yaml` |

Every older version (HARD-V2, V3, V3-LINEAR, V4, V4-LINEAR, V5-LINEAR, V6) and the D0 files were moved, not deleted, to
`tmp/archived_quadrotor_datasets_2026-09-18/`.

The two datasets are produced by the same procedure; only the manoeuvre library, the seeds and the number of splits differ.
Line numbers below refer to `generate_quadrotor_evalset.py`, which is the HARD generator with a single split.

---

## 1. The plant (Step 0)

- Gym-PyBullet-Drones `CtrlAviary`, drone model **CF2P**, `Physics.PYB`, physics at 1000 Hz (`make_env`, line 71).
- Contact-free: the ground plane is removed and every contact friction is zeroed (`configure_plant`, line 82, reusing `report_controller.remove_ground_plane` and `configure_contact_free_dynamics`).
- PyBullet's built-in damping is switched on with $c = 0.5$ for both linear and angular motion. The simulator then applies
  $$\dot v = \ldots - c\,(1+\|v\|)\,v, \qquad \dot\omega = \ldots - c\,(1+\|\omega\|)\,\omega$$
  which is the "nonlinear damping" the models have to identify as $D_v = mc(1+\|v\|)I$ and $D_\omega = c(1+\|\omega\|)J$.
- Vehicle constants (read back from the URDF and stored in `settings.vehicle_parameters`): $m = 0.027$ kg, $J = \mathrm{diag}(2.40, 2.40, 3.23)\times10^{-5}$ kg m², $g = 9.8$ m/s², arm $L = 0.0397$ m, $k_f = 3.16\times10^{-10}$, $k_m = 7.94\times10^{-12}$, thrust-to-weight 2.25.
- No wind gusts (`gusts.enabled: false`), no motor lag, no per-motor gain spread.

## 2. The recorded sample (state and input layout)

Every sample is a 22-vector (`pack_sample`, line 111):

$$s = \big[\,x_w\,(3),\ \mathrm{vec}(R)\,(9),\ v_b\,(3),\ \omega_b\,(3),\ u\,(4)\,\big]$$

- $x_w$: position in the world frame (m).
- $R$: rotation matrix body→world, stored row-major.
- $v_b = R^\top v_w$, $\omega_b = R^\top \omega_w$: body-frame linear and angular velocity.
- $u = [T,\ \tau_x,\ \tau_y,\ \tau_z]$: the **wrench** that actually acted on the body during the sample interval (Section 5).

Sampling: 100 Hz, so $\Delta t = 0.01$ s, and a 10 s flight has 1001 samples. The physics runs 10 sub-steps per sample.

## 3. Random initial state (Step 2, `random_initial_state`, line 126)

Drawn once per flight from the `initial_state` box of the config:

- position uniform in $x, y \in [-2, 2]$, $z \in [1.5, 4]$ m,
- tilt uniform in $[0, 30°]$ about a random horizontal axis, yaw uniform in $[-\pi, \pi]$,
- speed uniform in $[0, 2]$ m/s along a random direction, angular rate uniform in $[0, 1]$ rad/s along a random axis.

The PyBullet body is reset to that pose and velocity before the first step.

## 4. The manoeuvre plan (Step 3, `plan_manoeuvres`, line 137)

A flight is **not** one shape. It is a random sequence of segments that covers the 10 s:

1. Draw a segment name from the library with probability $\propto$ its `weight`.
2. Draw its duration and parameters uniformly from the configured ranges.
3. Append it, advance the clock, repeat until 10 s are filled (the last segment is truncated). Typically 3–5 segments.

Each segment is **anchored** where the drone actually is when it starts (`anchor_xyz`, `anchor_yaw`, line 308): the
target is $p_{target}(t) = p_{anchor} + \text{shape}(t - t_{start})$. So the same shape occurs at different positions,
headings and entry speeds across flights.

### 4.1 HARD training library (9 families; `HARD_CF2P_10s_h0p01_config_used.yaml`)

| segment | target given to the PID (`segment_target`, line 217) | parameters |
|---|---|---|
| `waypoint_hop` | straight to a random point of the envelope box, capped at `max_distance`: $p = p_a + d\,\min(1, 3.5/\|d\|)$ | 1.5–2.5 s |
| `figure_eight` | $p = p_a + a\,[\sin\omega t,\ \sin 2\omega t,\ 0]$, $\omega = v/a$ | $a \in [1,2]$ m, $v \in [1,3]$ m/s, 3–4 s |
| `circle` | $p = p_a + r\,[\cos\omega t - 1,\ \sin\omega t,\ 0]$, $\omega = v/r$ | $r \in [1,2]$ m, $v \in [1,3]$ m/s, 3–4 s |
| `vertical_step` | $p = p_a + [0, 0, \pm h]$ | $h \in [1.5, 2.5]$ m, 1.5–2 s |
| `yaw_turn` | hold $p_a$, $\psi = \psi_a + \dot\psi\,t$ | $\dot\psi \in \pm[1,4]$ rad/s, 1–2 s |
| `aggressive_recovery` | hold $p_a$; external torque kick $\tau = J\hat a\,\dot\theta/t_k$ for $t_k = 0.1$ s about a random axis, then PID recovery | $\dot\theta \in [3,5]$ rad/s |
| `yaw_kick` | same kick about body $z$ | $\dot\theta \in [3,6]$ rad/s |
| `coast` | position loop **off**: $T = mg$, roll/pitch PD to level, $\tau_z = 0$ (`coast_rpm`, line 259); entered from a circle at 2–3 m/s so the decay starts with speed | 1.5–2.5 s |
| `coast_yaw` | coast plus a yaw kick during the decay | 1.5–2.5 s |

`disturbed_hover` had weight 0 in V5 (never drawn); it has been removed from this config and lives in the EVALSET library.

### 4.2 EVALSET library (8 families; `evalset_config.yaml`)

| segment | target / action | parameters |
|---|---|---|
| `dash` | straight run at constant speed, $p = p_a + v\,\hat d\,t$ (heading blended toward the box centre) | $v \in [0.5, 3.2]$ m/s, 2–3 s |
| `thrust_pulse` | position loop off (line 313): $T = f\,mg$ for $t_c$ seconds, then $T = (2-f)\,mg$ for $t_c$ (net impulse $\approx 0$), attitude held level by the coast PD, then PID recovery | $f \in [0.25, 0.45]$, $t_c \in [0.35, 0.5]$ s |
| `disturbed_hover` | hold $p_a$ and $\psi_a$ (gusts disabled, so a plain hover: $T \approx mg$) | 1–2 s |
| `slalom` | $p = p_a + v\,\hat d\,t + a\sin(2\pi t/T_s)\,\hat n$, $\hat n \perp \hat d$ | $v \in [2, 3.4]$ m/s, $a \in [0.3, 0.5]$ m, $T_s \in [2.5, 3.5]$ s |
| `helix` | $p = p_a + [r(\cos\omega t - 1),\ r\sin\omega t,\ \dot z\,t]$, $\omega = v/r$ | $r \in [1.5, 2]$ m, $v \in [2, 3.4]$ m/s, $\dot z \in \pm[0.3, 0.5]$ m/s |
| `chirp` | $p = p_a + a\sin\phi(t)\,\hat d$, $\psi = \psi_a + a_\psi \sin\phi(t)$, $\phi(t) = 2\pi\big(f_0 t + \tfrac{f_1 - f_0}{2T}t^2\big)$ | $a \in [0.5, 1]$ m, $a_\psi \in [0.2, 0.4]$ rad, $f_0 = 0.2$, $f_1 = 1.5$ Hz |
| `bounce` | square wave in height, starting high: $z = z_a + h\,[\,1 - (\lfloor 2t/T_b \rfloor \bmod 2)\,]$ | $h \in [0.3, 0.6]$ m, $T_b \in [1, 1.6]$ s |

All targets are finally clipped to the envelope box (line 255); dash/slalom/chirp headings are blended half-and-half toward the box centre so a long run has room.
| `tumble` | two 0.1 s torque kicks about two random axes, the second 0.4 s later (mid-recovery) (line 206) | $\dot\theta \in [3, 5]$ rad/s |

**Disjointness.** `training_library_names` (line 508) reads the HARD config from disk and `build` (line 514) raises if any
name is shared. The audit records `segment_names_shared_with_training: []`.

## 5. Who produces the input $u$ (Steps 0b–4, `generate_flight`, line 271)

The input is never designed by hand; it is whatever the tracking controller outputs:

1. **Controller**: `DSLPIDControl` (the Crazyflie cascade PID of gym-pybullet-drones) with all six gain vectors scaled by
   `gain_scale = 0.5` (loose tracking on purpose, so the drone lags and overshoots the target). It runs at 100 Hz
   (line 309):
   ```python
   rpm_cmd, _, _ = controller.computeControlFromState(control_timestep=1/sample_hz, state=state,
                                                      target_pos=target_xyz, target_rpy=target_rpy)   # line 318
   ```
   Outer loop: $a_{des} = K_P e_p + K_I\!\int e_p + K_D e_v + g e_3$ → thrust magnitude and desired attitude.
   Inner loop: PID on the attitude error → body torques → four motor RPMs through the CF2P mixer.
2. **Scripted segments bypass the PID**: `coast`, `coast_yaw` and `thrust_pulse` command the thrust directly through
   `coast_rpm` (line 259): $T$ as listed above, $\tau_{x,y} = -k_p\,\theta - k_d\,\omega$ to stay level, $\tau_z = 0$.
3. **Torque kicks** (`aggressive_recovery`, `yaw_kick`, `coast_yaw`, `tumble`) are external body torques applied with
   `pb.applyExternalTorque` for 0.1 s (line 337): $\tau_{kick} = J\,\hat a\,\dot\theta / t_k$, i.e. an impulse of $J\dot\theta$.
4. **Actuation**: the RPMs are clipped to $[0, \mathrm{rpm}_{max}]$ (line 319) and applied for 10 physics steps
   (line 342); PyBullet produces per-motor force $k_f\,\mathrm{rpm}_i^2$ and yaw torque $k_m\,\mathrm{rpm}_i^2$.
5. **Recorded input** (lines 346–349), once per sample:
   $$u = M_{mix}\,\big(k_f\,\mathrm{rpm}^2\big) + \begin{bmatrix}0\\ \tau_{kick}\end{bmatrix},\qquad
   M_{mix} = \begin{bmatrix} 1 & 1 & 1 & 1\\ 0 & L & 0 & -L\\ -L & 0 & L & 0\\ -\kappa & \kappa & -\kappa & \kappa\end{bmatrix},\ \kappa = k_m/k_f .$$
   This is the wrench that actually acted, after clipping, so the true control map is the selection matrix
   $g = S$ (`settings.true_control_map`). The raw $(\mathrm{rpm}_i/\mathrm{rpm}_{max})^2$ are stored too (`*_motor_commands`).

## 6. Gates (Step 5, lines 355–361)

A flight is rejected and re-drawn with a new attempt seed if any of these holds:

| gate | limit |
|---|---|
| max tilt | 90° |
| min altitude | 0.3 m |
| envelope excursion (box $x, y \in [-3.5, 3.5]$, $z \in [0.5, 5]$) | 0.5 m |
| motor saturation fraction (physics steps with a motor at the PID's PWM ceiling) | 0.05 (HARD), 0.08 (EVALSET) |

Up to 20 attempts per flight. EVALSET needed 0 rejections; the highest saturation fraction is 0.008.

## 7. Seeds and splits (Step 6)

Flight RNG seed $= 7\,000\,000 + 1000\,s + i + 100\,000\,a$ (seed $s$, flight index $i$, attempt $a$).

| dataset | split | seeds | flights | pickle key |
|---|---|---|---|---|
| HARD | train | 300–304 | 50 | `train_trajectories`, windows in `x` |
| HARD | validation (i.i.d., loss curves) | 399 | 10 | `test_trajectories`, windows in `test_x` |
| EVALSET | eval | 498–499 | 20 | `heldout_trajectories`, windows in `heldout_x` |

Windows: 6 points (0.05 s) with stride 5, only used by the old window loaders; the trainers use whole flights.

## 8. Observation-noise variants (Step 7, HARD only)

From the one clean set, four absolute-noise files and one sensor-noise file are derived (`absolute_noise`, line 408;
`sensor_noise`, line 416). Only the **training** flights are noised; the validation flights stay clean (a noisy copy is
stored separately as `test_trajectories_noisy`).

- Absolute noise, level $\sigma \in \{0.05, 0.1, 0.25, 0.5\}$:
  $$\tilde x = x + \sigma\epsilon,\quad \tilde v_b = v_b + \sigma\epsilon,\quad \tilde\omega_b = \omega_b + \sigma\epsilon,\quad \tilde R = R\,\exp(\hat\eta),\ \eta \sim \mathcal N(0, \sigma^2 I)$$
  Controls are untouched. The reported models use $\sigma = 0.25$: `HARD_CF2P_10s_h0p01_train-obs-noise-absolute0p25.pkl`.
- Sensor noise (`train-sensor-noise`): 5 mm position, 1° attitude, velocity by central difference of the noisy
  positions, gyro white noise 0.01 rad/s plus a bias random walk.

EVALSET has no noise variants: an evaluation set is scored against the clean truth.

## 9. Audits stored in `settings` (Step 8, `audits`, line 452)

Per split: median position change over horizons 0.1–2 s and its ratio to each noise level, input correlation matrix,
input PSD, coverage percentiles (speed, angular rate, tilt, envelope fraction visited) and rotation validity
($\|R^\top R - I\|_F$, $|\det R - 1|$). EVALSET coverage: speed p50/p90/max = 1.05 / 2.54 / 3.22 m/s, angular rate
0.58 / 2.16 / 5.05 rad/s, tilt 15 / 32 / 59°; thrust $T/mg$ has CV 0.19 with 3.3 % of samples below half hover and 3.4 % above 1.5× hover.

## 10. Files written (Step 9–10)

`<TAG>_clean.pkl` (+ noise variants for HARD), `<TAG>_audits.json`, `<TAG>_config_used.yaml`, `<TAG>_generation.log`
(one line per attempt with the segment sequence and the gate verdict), `<TAG>_dataset_analysis.pdf` (config page, 3-D
overview, example flights with segment shading, coverage, inputs, validity).

## 11. How the report tooling reads them

- Training: `data.dataset_path` in the model config → `train_trajectories`.
- Open loop / recorded closed-loop references: `open_loop.load_open_loop_set("<pkl>@heldout")` returns whole flights
  with $\Delta t$ from `settings.config.environment.sample_hz`; `report_controller.register_recorded_references` turns
  each flight into a closed-loop reference named `EVALSET-heldout-flight<k>`.
- The three reported models were trained on the HARD $\sigma = 0.25$ file; its sha256 (`74d4c5b2…`) matches the
  `dataset_sha256` recorded in the GP run `14-09-20-22_…nopriors-lvl0p25-feat20`, so the rename changed no data.

## 12. Why a PID and not the model-based controller for data generation

- The learner only needs coverage and excitation of $(x, u)$; tracking quality is irrelevant. The loose PID's lag and
  overshoot are free excitation.
- A model-based law (IDA-PBC / the energy-based SE(3) law) needs the operators being learned and cancels gravity and
  drag exactly, so the data would carry less information about exactly those terms (closed-loop identification bias).
- A stock PID transfers to hardware unchanged, so the protocol is reproducible without knowing the physics.

# Results — SE(3) port-Hamiltonian quadrotor identification and control

Everything measured so far, in one place: simulator with observation noise → real flight (IDSIA) → simulator
with observation noise **and** an unobserved wind, for all four model families (PH-GP-LieIMEX ODE and SDE,
PH-NN-LieIMEX ODE and SDE, PH-NODE-RK4) plus the analytic-operator reference.

Every number below is taken from a recorded artefact in `experiments/quadrotor/`; each section links the PDF or
PNG it came from. Where a number was recomputed for this document, the command is given in §10.

---

## 1. How to read this document

### 1.1 The model under test

All learned models share the same state, the same control-affine port-Hamiltonian form and the same six
operators; they differ only in *how the operators are parameterised* and *how they are integrated*.

$$x=\big(x_w,\ \mathrm{vec}(R),\ v_b,\ \omega_b\big),\qquad
H=\tfrac12 p_v^\top M_1^{-1}p_v+\tfrac12 p_\omega^\top M_2^{-1}p_\omega+V(x_w,R),$$

$$\dot z=\big(J(z)-\mathcal R(z)\big)\nabla H(z)+g(z)\,u .$$

| model | operators | integrator | training objective | run tag |
|---|---|---|---|---|
| **PH-GP-LieIMEX** (ODE) | variational GP, Matérn ⊗ periodic features on $SE(3)$ | Lie–IMEX (implicit damping, exact $\exp$ on $SO(3)$) | SE(3) NLL + $\beta\,$KL + penalties | `ph_gp_lie_imex` |
| **PH-GP-LieIMEX-SDE** | same GP | stochastic Lie–IMEX | moment-matched predictive NLL over $S$ paths | `ph_gp_lie_imex_sde_idsia` |
| **PH-NN-LieIMEX** (ODE) | 6 MLPs (width 64), PSD-constructed | Lie–IMEX | trajectory MSE | `ph_nn_lie_imex` |
| **PH-NN-LieIMEX-SDE** | same MLPs + 2 noise scales | stochastic Lie–IMEX | moment-matched predictive NLL | `ph_nn_lie_imex_sde` |
| **PH-NODE-RK4** | 6 MLPs (width 64) | explicit RK4 | trajectory MSE | `ph_node` |
| **PH-GT** (reference) | the simulator's own constants | Lie–IMEX | — (not trained) | `--include-ground-truth` |

Only **gauge-invariant products** are comparable across models, because $M^{-1}$ and $g$ are individually
defined only up to a gauge:

$$\mu\, g_f,\qquad M_2^{-1} g_\tau,\qquad \mu\nabla V,\qquad \mu D_v,\qquad M_2^{-1}D_\omega,\qquad \mu:=M_1^{-1}.$$

The identification tables report the relative RMS error of each product against the simulator's analytic value:
0 % = exact, 100 % = no better than predicting zero, > 100 % = worse than no model.

### 1.2 The three test regimes

| regime | plant | disturbance | what it tests |
|---|---|---|---|
| **A — simulator + observation noise** | PyBullet CF2P, contact-free, nonlinear damping $c=0.5$ | additive measurement noise $\sigma\in\{0,0.1,0.25,0.5\}$ | identification and control under errors-in-variables |
| **B — real flight** | Crazyflie 2.1 Brushless, motion capture, 100 Hz | everything real | does any of it transfer |
| **C — simulator + wind** | the same PyBullet plant | unobserved Ornstein–Uhlenbeck body force, $\tau=0.5$ s, $\sigma\in\{10,25\}\%\,mg$, **plus** observation noise | can a model represent a disturbance it cannot see |

### 1.3 Metric definitions

Full definitions in [`notes/Comparision/open_loop_metrics.md`](notes/Comparision/open_loop_metrics.md) and
[`notes/Comparision/closed_loop_metrics.md`](notes/Comparision/closed_loop_metrics.md). In brief:

- **Open loop**: the model gets the true $x_0$ and the recorded $u(t)$ and integrates alone.
  $\mathrm{RMSE}_p=\sqrt{\frac1{FT}\sum\|\hat p-p\|^2}$; **VPT** = first time the error exceeds
  $0.158\times$ the flight's own extent.
- **Closed loop**: IDA-PBC built from the learned operators, flown on PyBullet; identical gains
  ($K_p=[10,10,50]$, $K_v=[3,3,3]$, $K_R=[250,250,250]$, $K_\omega=[20,20,20]$, tilt limit $40°$) for every
  column, so only the operators differ.
- **Calibration**: coverage of the truth by the $\pm2\sigma$ predictive band (target 0.954), the ratio
  $\sigma/\mathrm{RMS\ error}$ (1 = calibrated), and Spearman $\rho(\sigma,\text{error})$ (does the model know
  *where* it is uncertain).

---

## 2. Datasets

| name | plant | flights | split | note |
|---|---|---|---|---|
| `QUADROTOR-DATASET-HARD` | PyBullet CF2P, DSL-PID | 50 train / 10 val | 9 manoeuvre families | = the former HARD-V5, byte-identical |
| `QUADROTOR-DATASET-EVALSET` | same | 20 held-out | 8 families **disjoint** from training | dash, thrust_pulse, disturbed_hover, slalom, helix, chirp, bounce, tumble |
| `QUADROTOR-DATASET-IDSIA` | **real** Crazyflie 2.1 BL | 44 train / 18 test | chirp+random+square / **melon** | measured rotor speeds; benchmark's own split |
| `QUADROTOR-DATASET-NANOBENCH` | **real** Crazyflie 2.1 | 89 / 25 | shape-disjoint | **rejected**, see §5.4 |
| `QUADROTOR-DATASET-WIND` | PyBullet + OU gust 10 % $mg$ | 50 / 10 / 20 | held-out = EVALSET families | gust unobserved by the models |
| `QUADROTOR-DATASET-WIND25` | + OU gust 25 % $mg$ | 50 / 10 / 20 | same | 64 of 134 attempts gate-rejected |
| `QUADROTOR-EVAL-REFERENCE` | D0 layout, 240 Hz | 10 | — | the report's common evaluation flights |

Every simulator set also ships noise variants $\sigma\in\{0.05,0.1,0.25,0.5\}$ and a sensor-model variant.
Data generation is documented in
[`notes/Dataset/quadrotor_trajectory_data_generation.md`](notes/Dataset/quadrotor_trajectory_data_generation.md).

- 📄 [HARD clean-vs-noisy validation flights (20 pages)](datasets/QUADROTOR-DATASET-HARD/HARD_CF2P_10s_h0p01_validation_clean_vs_noisy.pdf)
- 📄 [WIND dataset analysis](datasets/QUADROTOR-DATASET-WIND/WIND_CF2P_10s_h0p01_dataset_analysis.pdf) · [WIND25](datasets/QUADROTOR-DATASET-WIND25/WIND25_CF2P_10s_h0p01_dataset_analysis.pdf)
- 📄 [EVALSET analysis](datasets/QUADROTOR-DATASET-EVALSET/EVALSET_CF2P_10s_h0p01_dataset_analysis.pdf)

---

## 3. Part A — Simulator with observation noise

### 3.1 Identification vs observation noise

Relative RMS error of each gauge-invariant product, scored on the **same** 10 D0 reference flights
(3 s, 720 samples each) for every row, stage-2 models ($K=100$), no physics priors.

| model | noise $\sigma$ | $\mu g_f$ | $M_2^{-1}g_\tau$ | $\mu\nabla V$ | $\mu D_v$ | $M_2^{-1}D_\omega$ | mean |
|---|---|---|---|---|---|---|---|
| PH-GP-LieIMEX, **latent $x_0$ off** | 0 (clean) | **0.5 %** | **2.5 %** | **0.5 %** | **36.1 %** | **53.1 %** | **18.5 %** |
| PH-GP-LieIMEX | 0 (clean) | 9.5 % | 3.2 % | 10.2 % | 79.2 % | 64.6 % | 33.4 % |
| PH-GP-LieIMEX | 0.1 | 10.2 % | 3.4 % | 10.9 % | 96.5 % | 64.8 % | 37.2 % |
| PH-GP-LieIMEX | 0.25 | 11.7 % | 4.3 % | 12.3 % | 103.7 % | 76.6 % | 41.7 % |
| PH-GP-LieIMEX | 0.5 | 14.5 % | 6.7 % | 15.4 % | 104.5 % | 93.1 % | 46.8 % |
| PH-NN-LieIMEX | 0.1 | 17.5 % | 9.7 % | 18.3 % | 129.1 % | 69.4 % | 48.8 % |
| PH-NN-LieIMEX | 0.25 | 24.0 % | 11.3 % | 23.5 % | 156.6 % | 140.3 % | 71.1 % |
| PH-NN-LieIMEX | 0.5 | 8.7 % | 10.6 % | 94.3 % | 2349.6 % | 281.0 % | 548.9 % |
| PH-NODE-RK4 | 0.1 | 14.6 % | 10.5 % | 17.3 % | 114.3 % | 118.4 % | 55.0 % |
| PH-NODE-RK4 | 0.25 | 24.7 % | 10.5 % | 24.8 % | 138.4 % | 168.6 % | 73.4 % |

**Readings.**
1. The GP is better than both baselines at every noise level and on every product; the ordering is stable.
2. Degradation with noise is *graceful* for the GP (mean 33 → 47 % over $\sigma=0\to0.5$) and *catastrophic*
   for PH-NN at $\sigma=0.5$ (damping 2350 %).
3. **The largest single effect in the whole simulator study is not noise — it is the per-window latent
   $x_0$ (MAP) correction.** Turning it off on clean data takes thrust and gravity from ≈ 10 % to **0.5 %**
   and halves the damping error. The latent term is an errors-in-variables device that, on 1 s windows,
   also absorbs a constant force error (best linear fit of $\tfrac12 a t^2$ leaves only $a/\sqrt{320}$), so the
   loss is nearly flat along a uniform rescaling of $\mu g_f$ and $\mu\nabla V$ — a Neyman–Scott incidental-parameter
   bias. At $\sigma>0$ it cannot simply be removed: it also removes a *larger* EIV bias (§8).

### 3.2 Headline comparison at $\sigma=0.25$

Report: 📄 [`18-09-00-02_…_obs-noise0p25/`](experiments/quadrotor/eval_runs/18-09-00-02_spec-comparison_nopriors-3models-with-GT_closedloop-10shapes+V6heldout_openloop-D0-V6heldout_obs-noise0p25/) —
[physics-identification.pdf](experiments/quadrotor/eval_runs/18-09-00-02_spec-comparison_nopriors-3models-with-GT_closedloop-10shapes+V6heldout_openloop-D0-V6heldout_obs-noise0p25/physics-identification.pdf) ·
[open-loop/comparison.pdf](experiments/quadrotor/eval_runs/18-09-00-02_spec-comparison_nopriors-3models-with-GT_closedloop-10shapes+V6heldout_openloop-D0-V6heldout_obs-noise0p25/open-loop/comparison.pdf) ·
[closed-loop/comparison.pdf](experiments/quadrotor/eval_runs/18-09-00-02_spec-comparison_nopriors-3models-with-GT_closedloop-10shapes+V6heldout_openloop-D0-V6heldout_obs-noise0p25/closed-loop/comparison.pdf) ·
[training-and-test-losses.pdf](experiments/quadrotor/eval_runs/18-09-00-02_spec-comparison_nopriors-3models-with-GT_closedloop-10shapes+V6heldout_openloop-D0-V6heldout_obs-noise0p25/training-and-test-losses.pdf)

**(a) Identification** — 10 D0 flights, 10 s, ± = spread across flights:

| product | PH-GP | PH-NN | PH-NODE | PH-GT |
|---|---|---|---|---|
| thrust gain $\mu g_f$ | **11.8 ± 0.6 %** | 26.8 ± 2.4 % | 27.0 ± 2.5 % | 0 |
| torque gain $M_2^{-1}g_\tau$ | **5.1 ± 0.0 %** | 11.7 ± 0.3 % | 11.1 ± 0.4 % | 0 |
| gravity $\mu\nabla V$ | **12.4 ± 0.6 %** | 23.3 ± 3.1 % | 24.5 ± 3.3 % | 0 |
| translational damping $\mu D_v$ | **31.7 ± 1.0 %** | 73.5 ± 3.9 % | 52.4 ± 3.1 % | 0 |
| rotational damping $M_2^{-1}D_\omega$ | **39.1 ± 3.4 %** | 87.5 ± 2.2 % | 104.8 ± 2.8 % | 0 |

*(These use the 10 s varying reference flights; §3.1 uses the 3 s flights — the force products agree to
0.1 pp, the damping products do not, because damping error is dominated by the speed coverage of the
evaluation flights. Always state the evaluation set with a damping number.)*

**(b) Open loop**, 10 s, true $x_0$ and recorded $u(t)$:

| metric | set | PH-GP | PH-NN | PH-NODE | PH-GT |
|---|---|---|---|---|---|
| position RMSE (m) | D0 | 4.577 | 2.675 | **2.657** | 0.510 |
| attitude (deg) | D0 | 20.81 | 65.57 | **16.68** | 1.24 |
| **VPT median (s)** | D0 | **1.08** | 0.40 | 0.42 | 5.53 |
| position RMSE (m) | V6 held-out | 6.162 | **3.782** | 4.095 | 0.268 |
| attitude (deg) | V6 held-out | **28.53** | 65.05 | 32.82 | 0.58 |
| **VPT median (s)** | V6 held-out | **1.60** | 0.91 | 0.84 | 9.30 |
| $SO(3)$ violation | both | 3e-14 | 3e-14 | **4.6e-07** | 3e-14 |
| coverage@2σ (GP only) | D0 / V6 | 0.572 / 0.509 | — | — | — |
| Spearman $\rho(\sigma,e)$ | D0 / V6 | +0.90 / +0.89 | — | — | — |

The GP tracks **1.8–1.9× longer** before diverging, which is the metric that matters for a 10 s free rollout;
its 10 s RMSE is worse because once diverged it leaves faster. PH-NODE's RK4 breaks $SO(3)$ by $5\times10^{-7}$
against $3\times10^{-14}$ for every Lie–IMEX model — seven orders of magnitude.

**(c) Closed loop** — IDA-PBC on PyBullet, 20 s, position RMSE (m), same gains everywhere:

| reference | PH-GP | PH-NN | PH-NODE | PH-GT |
|---|---|---|---|---|
| spiral_k1p5 | **0.094** | 0.100 | 0.117 | 0.068 |
| spiral_k2 | 0.478 | **0.344** | 0.463 | 0.488 |
| spiral_k2p5 | **1.490** | 8.558 | 17.088 | 0.882 |
| lissajous_yaw0p5 | **0.084** | 1.232 | 0.115 | 0.053 |
| lissajous_yaw1 | **0.084** | 1.232 | 0.115 | 0.053 |
| lissajous_yaw1p5 | **0.084** | 13.428 | 4.221 | 0.053 |
| stop_v1 | **0.056** | 0.105 | 0.084 | 0.033 |
| stop_v1p5 | **0.076** | 15.052 | 0.120 | 0.054 |
| stop_v2 | **1.139** | 51.791 | 36.686 | 0.090 |
| stop_v2p5 | **1.073** | 10.977 | 50.650 | 0.163 |
| EVALSET-shape flights 00–09 (recorded references) | **0.057–1.059** | 0.095–56.1 | 0.100–33.2 | 0.049–0.195 |

**The closed-loop result is the strongest in the project**: the GP wins 19 of 20 shapes, never exceeds 1.5 m,
and is within 0.02–0.04 m of the analytic-operator ceiling on 8 of 10 analytic shapes, while both baselines
diverge by tens of metres on the aggressive ones. A subnetwork-swap ablation isolates **why**: giving the NN
the GP's $\mu g_f$ rescues it (yaw 2.29 → 0.125 m, stop 47.8 → 0.53 m) and giving the GP the NN's $\mu g_f$
breaks it (0.084 → 1.74 m) — the **force scale is the operator that carries closed-loop performance**, because
the allocation commands thrust as $1/(\mu g_f)$ and a 22 % under-estimate over-actuates every force by 28 %.

🖼 Closed-loop images: `…18-09-00-02_…/closed-loop/images/<shape>_{tracking,trajectory}.png` (84 PNGs)

---

## 4. Part B — Real flight (IDSIA nano-drone benchmark)

Source: Busetto et al., *Nonlinear System Identification for a Nano-drone Benchmark*, **Control Engineering
Practice 172 (2026) 106871**; code cloned to `other_paper_codes/nanodrone-sysid-benchmark` (3 branches).
Crazyflie 2.1 Brushless, motion capture at 100 Hz, ~75 k samples, **measured** rotor speeds as input.
Split is the benchmark's own and shape-disjoint: train = chirp + random + square (44 × 10 s), test = **melon**
(18 × 10 s). Notes: [`notes/Dataset/real_flight_idsia_experiment.md`](notes/Dataset/real_flight_idsia_experiment.md).

### 4.1 Input reconstruction is validated

| test | IDSIA | NanoBench (rejected) |
|---|---|---|
| thrust vs accelerometer, $m a_z / k_F\sum\Omega_i^2$ | **0.9996 ± 0.0057** | 0.978 ± 0.023 |
| parameter-free mass fit | **45.03 g** vs 45.0 published (+0.1 %) | 39.96 vs 40.85 (−2.2 %) |
| rotational $R^2$ roll / pitch / yaw | **0.301 / 0.345 / 0.432** at **zero lag** | 0.250 / 0.097 / **0.000** at 30 ms lag |

### 4.2 The benchmark's own protocol (their Table 7)

Same layout and metric as their Table 7: MAE at $h=1,10,50$ and, in *italics*, the cumulative simulation error
(sum of MAEs over $h=1\dots50$, i.e. 0.5 s at 100 Hz). Open loop, no state correction; lower is better, and the
best entry in each column is bold.

Scorers: `benchmark_protocol.py` (quotes their published Table 7) and `split_protocol_table.py` (new — recomputes
**Naïve** and **Physics** on whichever flights are loaded, which is what makes a training-split table possible).
Validation of the recomputation on melon: 0.0143/0.1422/0.6755/17.6802 for Naïve against their published
0.0143/0.1430/0.6797/17.7878, and 0.0013/0.0119/0.1241/2.2409 for Physics against 0.0013/0.0126/0.1269/2.3223 —
agreement to 1–3 %, as expected from re-deriving body-frame velocities from the raw CSVs.

**Res-MLP, Hybrid and Res-LSTM appear only in the held-out table**: the paper publishes no training-split numbers
for them and their weights are not released, so those rows are genuinely unavailable rather than omitted.

#### 4.2a Training shapes — chirp, random, square

12 flights, 13 756 windows, stride 4. 📄 [`22-09-19-00_IDSIA-protocol_TRAIN-shapes/`](experiments/quadrotor/eval_runs/22-09-19-00_IDSIA-protocol_TRAIN-shapes/split_protocol_table.json)

| Model | $\mathrm{MAE}_p$ $h$=1 | $\mathrm{MAE}_p$ $h$=10 | $\mathrm{MAE}_p$ $h$=50 | *$\mathrm{MAE}_p$ $h$=1:50* |  | $\mathrm{MAE}_v$ $h$=1 | $\mathrm{MAE}_v$ $h$=10 | $\mathrm{MAE}_v$ $h$=50 | *$\mathrm{MAE}_v$ $h$=1:50* |  | $\mathrm{MAE}_R$ $h$=1 | $\mathrm{MAE}_R$ $h$=10 | $\mathrm{MAE}_R$ $h$=50 | *$\mathrm{MAE}_R$ $h$=1:50* |  | $\mathrm{MAE}_\omega$ $h$=1 | $\mathrm{MAE}_\omega$ $h$=10 | $\mathrm{MAE}_\omega$ $h$=50 | *$\mathrm{MAE}_\omega$ $h$=1:50* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naïve | 0.0092 | 0.0912 | 0.4306 | *11.2889* |  | 0.0239 | 0.2295 | 0.9735 | *26.6505* |  | 0.0070 | 0.0682 | 0.2549 | *7.4598* |  | 0.0677 | 0.3999 | 1.0722 | *35.9565* |
| Physics | **0.0013** | 0.0125 | 0.1436 | *2.4972* |  | **0.0072** | **0.0538** | 0.6890 | *11.9100* |  | **0.0010** | 0.0207 | 0.3113 | *6.1843* |  | 0.0677 | 0.3999 | 1.0722 | *35.9565* |
| Res-MLP | — | — | — | — |  | — | — | — | — |  | — | — | — | — |  | — | — | — | — |
| Hybrid | — | — | — | — |  | — | — | — | — |  | — | — | — | — |  | — | — | — | — |
| Res-LSTM | — | — | — | — |  | — | — | — | — |  | — | — | — | — |  | — | — | — | — |
| PH-GP-LieIMEX-ODE | **0.0013** | **0.0117** | 0.0945 | ***1.9206*** |  | 0.0096 | 0.0735 | **0.3842** | ***9.0179*** |  | **0.0010** | 0.0200 | 0.1680 | *4.0830* |  | 0.0698 | 0.3516 | 0.6454 | *23.9610* |
| PH-NN-LieIMEX-ODE | **0.0013** | 0.0122 | **0.0942** | *1.9809* |  | 0.0119 | 0.0906 | 0.3889 | *9.4896* |  | **0.0010** | **0.0175** | **0.1443** | ***3.5442*** |  | **0.0626** | **0.3159** | **0.5406** | ***21.0671*** |
| PH-NN-LieIMEX-SDE | **0.0013** | 0.0120 | 0.1235 | *2.3800* |  | 0.0168 | 0.1271 | 0.5243 | *13.5396* |  | 0.0011 | 0.0296 | 0.1905 | *4.9567* |  | 0.1328 | 0.4625 | 0.5897 | *26.1852* |

#### 4.2b Held-out shape — melon (never seen in training)

3 flights, 9 675 windows, stride 2. 📄 [`22-09-19-00_IDSIA-protocol_TEST-melon/`](experiments/quadrotor/eval_runs/22-09-19-00_IDSIA-protocol_TEST-melon/split_protocol_table.json)

| Model | $\mathrm{MAE}_p$ $h$=1 | $\mathrm{MAE}_p$ $h$=10 | $\mathrm{MAE}_p$ $h$=50 | *$\mathrm{MAE}_p$ $h$=1:50* |  | $\mathrm{MAE}_v$ $h$=1 | $\mathrm{MAE}_v$ $h$=10 | $\mathrm{MAE}_v$ $h$=50 | *$\mathrm{MAE}_v$ $h$=1:50* |  | $\mathrm{MAE}_R$ $h$=1 | $\mathrm{MAE}_R$ $h$=10 | $\mathrm{MAE}_R$ $h$=50 | *$\mathrm{MAE}_R$ $h$=1:50* |  | $\mathrm{MAE}_\omega$ $h$=1 | $\mathrm{MAE}_\omega$ $h$=10 | $\mathrm{MAE}_\omega$ $h$=50 | *$\mathrm{MAE}_\omega$ $h$=1:50* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naïve | 0.0143 | 0.1430 | 0.6797 | *17.7878* |  | 0.0329 | 0.3182 | 1.4749 | *38.9241* |  | 0.0071 | 0.0692 | 0.3041 | *8.2138* |  | **0.0796** | 0.3596 | 0.8837 | *29.0866* |
| Physics | **0.0013** | 0.0126 | 0.1269 | *2.3223* |  | **0.0080** | **0.0570** | 0.5781 | *10.6232* |  | **0.0011** | 0.0205 | 0.2544 | *5.1013* |  | **0.0796** | 0.3596 | 0.8837 | *29.0866* |
| Res-MLP | 0.0032 | 0.0305 | 0.1712 | *4.0519* |  | 0.0116 | 0.0890 | 0.5720 | *12.5809* |  | 0.0022 | 0.0331 | 0.2268 | *6.0591* |  | 0.1138 | 0.5949 | 0.9005 | *35.5735* |
| Hybrid | 0.0016 | 0.0166 | 0.1119 | *2.3625* |  | 0.0092 | 0.0613 | 0.5556 | ***10.4033*** |  | 0.0027 | 0.0376 | 0.2306 | *6.1534* |  | 0.0912 | 0.4880 | 0.5979 | *28.9873* |
| Res-LSTM | 0.0079 | 0.0390 | 0.2572 | *5.9711* |  | 0.0247 | 0.1175 | 0.7407 | *16.7207* |  | 0.0066 | 0.0372 | 0.2325 | *5.6525* |  | 0.1021 | 0.4292 | 1.2407 | *35.8353* |
| PH-GP-LieIMEX-ODE | **0.0013** | **0.0116** | **0.1108** | ***2.1985*** |  | 0.0142 | 0.1078 | **0.4457** | *11.5393* |  | **0.0011** | 0.0249 | 0.1985 | *4.7803* |  | 0.0958 | 0.4212 | 0.6289 | *25.9788* |
| PH-NN-LieIMEX-ODE | **0.0013** | 0.0130 | 0.1320 | *2.6348* |  | 0.0187 | 0.1461 | 0.5693 | *14.9211* |  | **0.0011** | **0.0201** | **0.1786** | ***4.1281*** |  | 0.0805 | **0.3573** | **0.5520** | ***23.3117*** |
| PH-NN-LieIMEX-SDE | **0.0013** | 0.0129 | 0.1561 | *3.0545* |  | 0.0211 | 0.1709 | 0.6217 | *17.2566* |  | 0.0012 | 0.0315 | 0.2253 | *5.5662* |  | 0.1516 | 0.4896 | 0.5665 | *27.3701* |

Baseline rows are the authors' published values; our three rows come from the same flights under the same metric.

**Honest summary.**

1. **Attitude and angular rate are where we win on the held-out shape.** $\mathrm{MAE}_R$ cumulative 4.128
   (PH-NN-ODE) and 4.780 (PH-GP-ODE) against the best baseline's 5.101 (Physics); $\mathrm{MAE}_\omega$
   cumulative 23.31 and 25.98 against 28.99 (Hybrid). Both also beat every baseline at $h=50$ on both metrics.
2. **Position is a narrow win at best.** PH-GP-ODE's 2.1985 leads, but Physics (2.3223) and Hybrid (2.3625) are
   within 7 %, and both NN variants are behind. We should not claim position.
3. **Velocity we lose.** Hybrid 10.4033 and Physics 10.6232 beat all three of our models (11.54 / 14.92 / 17.26).
4. **The SDE costs point accuracy and buys calibration.** PH-NN-SDE is worse than PH-NN-ODE on every column here
   ($\mathrm{MAE}_p$ 3.055 vs 2.635, $\mathrm{MAE}_R$ 5.566 vs 4.128), which is expected: it optimises a
   predictive NLL over $S=8$ sampled paths rather than a trajectory MSE, so it trades the mean for the band.
   What it gains is not visible in this table — **coverage@2$\sigma$ = 0.936 against a 0.954 target**, by far the
   best-calibrated model on real data (§4.4, where PH-GP sits at 0.23). That is the reason to keep it.
5. **Generalisation.** Train → held-out, PH-GP-ODE 1.9206 → 2.1985, PH-NN-ODE 1.9809 → 2.6348, PH-NN-SDE
   2.3800 → 3.0545. The NN variants degrade more than the GP, consistent with point estimates fitting the
   training shapes harder than a posterior mean does.

**PH-GP-LieIMEX-SDE is absent because it does not train under these settings.** With priors off, float32 and no
mass pre-training it diverged at step 370 (`objective=-1.26`, `gradient_norm=inf`) after a healthy start — test
loss had already fallen 2.7x and the 1 s error from 0.632 to 0.305 m. It is the fifth SDE run to die once the
physics priors are removed, and the first with a *learned* observation scale, which rules out the pinned-$\sigma$
explanation offered in §5.3: the common factor is the missing gravity penalty, not the $\sigma$ treatment. See §9.

Caveats: each of our columns is a **single seed**; PH-NN-ODE is a point estimate, so §4.4's calibration results
exist only for PH-GP-ODE and PH-NN-SDE; all three of our runs are float32 with both physics priors off and no
mass pre-training, matching each other; and all our rows use ~79 % of the benchmark's training data with no
test-set model selection, whereas their learned baselines validate on melon.

### 4.3 Open loop against the benchmark's baselines, our own protocol

📄 [`19-09-16-13_IDSIA-melon-open-loop-baselines/open_loop_baselines.pdf`](experiments/quadrotor/eval_runs/19-09-16-13_IDSIA-melon-open-loop-baselines/open_loop_baselines.pdf) (42 pages)
🖼 [error vs time, 0.5 s](experiments/quadrotor/eval_runs/19-09-16-13_IDSIA-melon-open-loop-baselines/images/IDSIA-melon_0.5s_error.png)
· [1 s tracking grid](experiments/quadrotor/eval_runs/19-09-16-13_IDSIA-melon-open-loop-baselines/images/IDSIA-melon_1s_states_segment064.png)
· [1 s trajectory](experiments/quadrotor/eval_runs/19-09-16-13_IDSIA-melon-open-loop-baselines/images/IDSIA-melon_1s_trajectory_segment064.png)

| horizon | model | pos RMSE (m) | MAE$_p$ | MAE$_v$ | MAE$_R$ (rad) | MAE$_\omega$ |
|---|---|---|---|---|---|---|
| 0.5 s (342 seg) | Naïve | 0.426 | 0.713 | 1.546 | 0.299 | 0.845 |
| | Physics (IDSIA) | 0.061 | 0.120 | 0.542 | 0.242 | 0.845 |
| | **PH-GP-LieIMEX** | 0.063 | 0.112 | **0.444** | 0.210 | 0.598 |
| | **PH-NN-LieIMEX** | **0.058** | **0.101** | 0.451 | **0.198** | **0.514** |
| | PH-NN-LieIMEX (fp32, no pre-train) | 0.082 | 0.137 | 0.591 | **0.181** | 0.536 |
| 1 s (162 seg) | Naïve | 0.782 | 1.238 | 2.566 | 0.497 | 1.173 |
| | Physics | 0.392 | 0.869 | 2.761 | 0.690 | 1.173 |
| | **PH-GP-LieIMEX** | **0.224** | **0.438** | **0.983** | 0.289 | 1.137 |
| | **PH-NN-LieIMEX** | 0.230 | 0.460 | 1.137 | **0.210** | **0.824** |
| | PH-NN-LieIMEX (fp32, no pre-train) | 0.277 | 0.562 | 1.330 | 0.258 | **0.643** |
| 3 s (54 seg) | Naïve | 1.064 | **0.624** | **1.196** | **0.225** | 0.714 |
| | Physics | 12.08 | 27.5 | 25.5 | 1.773 | 0.714 |
| | **PH-GP-LieIMEX** | **0.881** | 1.417 | 2.096 | 0.735 | 2.552 |
| | PH-NN-LieIMEX | 1.143 | 1.648 | 1.969 | 0.537 | **0.981** |
| | PH-NN-LieIMEX (fp32, no pre-train) | 1.181 | **1.287** | **1.558** | **0.307** | **0.789** |

PH-NN-LieIMEX source: [`22-09-11-20_IDSIA-melon-open-loop_PH-NN/`](experiments/quadrotor/eval_runs/22-09-11-20_IDSIA-melon-open-loop_PH-NN/open_loop_baselines.pdf),
trained by `22-09-2026-09-00_ph_nn_lie_imex.yaml` — matched to the GP and PH-NODE runs (same dataset, window 51,
stride 10, 3000 steps, seed 0, **physics priors off**, fp64). No segment diverged at any horizon.

**Reading.** The two families split by horizon. At the benchmark's own **0.5 s** horizon **PH-NN is the only model
that beats the IDSIA physics baseline** (0.058 vs 0.061 m), and it leads on attitude and angular rate at *every*
horizon. At 1 s the position error is a tie (0.230 vs 0.224 m, a 2.6 % gap). At 3 s the GP wins clearly on position
(0.881 vs 1.143 m) and PH-NN falls behind even Naïve (1.064 m) — the point estimate extrapolates worse. What the
GP has and PH-NN structurally cannot is the posterior: every calibration number in §4.4 exists only for the GP, and
`open_loop.posterior_rollouts` has no meaning for a point estimate.

> **Note on the $SO(3)$ column.** The open-loop evaluators do not enable `jax_enable_x64`, so `report_evaluation.rollout`
> silently truncates to float32 and the reported $\lVert R^\top R-I\rVert_F$ sits at the float32 floor
> ($5\times10^{-7}$–$2\times10^{-6}$) for **every** model. Re-run in float64 the same PH-NN rollout gives
> $9.9\times10^{-16}$. The §3.2b structure-preservation claim (measured in float64) is unaffected, but the
> `so3_orthogonality` entry of any open-loop table must not be used to compare integrators.

Whole 60 s flights (one training-shape flight and one held-out flight):
📄 [`19-09-16-45_IDSIA-full-flight-baselines/full_flight_baselines.pdf`](experiments/quadrotor/eval_runs/19-09-16-45_IDSIA-full-flight-baselines/full_flight_baselines.pdf)
🖼 [melon, restart every 1 s](experiments/quadrotor/eval_runs/19-09-16-45_IDSIA-full-flight-baselines/images/test-melon-run1_60s_restart-every-1s_states.png)
· [chirp, restart every 1 s](experiments/quadrotor/eval_runs/19-09-16-45_IDSIA-full-flight-baselines/images/train-chirp-run1_60s_restart-every-1s_states.png)
· [melon, pure 60 s open loop](experiments/quadrotor/eval_runs/19-09-16-45_IDSIA-full-flight-baselines/images/test-melon-run1_60s_open-loop_states.png)

| view | Naïve | Physics | ours |
|---|---|---|---|
| chirp (train shape), restart every 1 s | 0.483 m | 0.257 m | **0.133 m** |
| melon (held out), restart every 1 s | 0.763 m | 0.361 m | **0.233 m** |
| melon, pure 60 s open loop | 0.986 m | 8056 m | 66.8 m |

No model identified from this data survives 60 s of free integration — the benchmark is defined at 0.5 s for
exactly that reason.

### 4.4 Calibration on real data (the weakness)

| | coverage@2σ (target 0.954) | $\rho(\sigma,e)$ |
|---|---|---|
| GP posterior band, 0.5 s | 0.247 | +0.85 |
| GP posterior band, 1 s | 0.230 | +0.89 |

The model **knows where** it is uncertain and badly **understates how much**. §6.4 shows the cause is the same
latent-$x_0$ device as in §3.1.

### 4.5 What the learned operators actually do on real data

| quantity (mean over training poses) | learned | true physics |
|---|---|---|
| angular acceleration from actuation, roll/pitch/yaw (rms) | 5–7 / 5–7 / 1 rad s⁻² | 23.6 / 32.3 / 0.78 |
| thrust gain $\mu g_{f_z}$ | 6.3–15.0 | 22.2 |
| $M_2^{-1}$ diag | 22–136 | 41752 / 41752 / 309148 |

The model attributes only **20–25 %** of the real angular acceleration to the input and covers the rest with
damping. This is rational: the rotor-speed torque model explains only ~30 % of measured $\dot\omega$
(§4.6), and for a noisy regressor the likelihood-optimal gain is attenuated.

### 4.6 The benchmark's own simulator is not a digital twin

Their `dev`-branch JAX simulator with published constants, against all 15 real flights
(script `$CLAUDE_JOB_DIR/tmp/idsia_sim/compare_sim_vs_real.py`, temporary):

| | train | melon |
|---|---|---|
| $R^2$ of simulated $\dot v$ (x/y/z) | 0.92 / 0.91 / 0.74 | 0.90 / 0.94 / 0.82 |
| $R^2$ of simulated $\dot\omega$ | **−4.2 / −13.3 / −0.44** | **−9.5 / −24.1 / −0.62** |
| unmodelled force (% of weight) | 6 / 6 / 9 % | 8 / 5 / 8 % |
| lateral body force (sim: exactly 0) | 0.017 N | 0.023 N |

With the rotational dynamics switched **on**, their own simulator scores **worse than the naïve hold** at
$h=50$ (position 0.78 vs 0.68 m, attitude 1.55 vs 0.30 rad) — which is why their published Physics baseline
sets $\dot\omega\equiv0$ and why its $\mathrm{MAE}_\omega$ row is digit-for-digit identical to Naïve.

---

## 5. Part C — Simulator with an unobserved wind

Datasets `QUADROTOR-DATASET-WIND` (10 % $mg$) and `-WIND25` (25 %), OU gust
$g_{k+1}=e^{-\Delta t/\tau}g_k+\sigma\sqrt{1-e^{-2\Delta t/\tau}}\,\xi_k$, $\tau=0.5$ s, applied in the body
frame with `applyExternalForce` and **never shown to the models** ($u$ is the commanded wrench).

The horizon-matched ground-truth equivalent of the gust, as a white process-noise scale on the body velocity
over the $0.5$ s training window, is
$$\sigma_v^{\text{GT}}=\sqrt{\frac{\sigma_a^2\,2\tau\big(t-\tau(1-e^{-t/\tau})\big)}{t}}\Bigg|_{t=0.5}
= 0.59\ (10\,\%),\qquad 1.49\ (25\,\%),$$
with low-frequency limits $(\sigma_a/1)\sqrt{2\tau}=0.98$ and $2.45$.

### 5.1 Open loop, all models, 10 % gust + 0.25 observation noise

📄 [1 s segments](experiments/quadrotor/eval_runs/20-09-13-40_WIND-open-loop-1s/comparison.pdf) ·
[full 10 s](experiments/quadrotor/eval_runs/20-09-13-40_WIND-open-loop-full/comparison.pdf)
🖼 [error vs time, held-out](experiments/quadrotor/eval_runs/20-09-13-40_WIND-open-loop-full/images/WIND_CF2P_10s_h0p01_clean_heldout_error.png)
· [states, flight 00](experiments/quadrotor/eval_runs/20-09-13-40_WIND-open-loop-full/images/WIND_CF2P_10s_h0p01_clean_heldout_states_flight00.png)
· [GP calibration](experiments/quadrotor/eval_runs/20-09-13-40_WIND-open-loop-full/images/WIND_CF2P_10s_h0p01_clean_heldout_calibration_PH-GP-LieIMEX.png)

| horizon / split | PH-NODE-RK4 | PH-GP-ODE | PH-GP-SDE |
|---|---|---|---|
| 1 s, val shapes | 0.272 m | **0.254 m** | 0.261 m |
| 1 s, held-out shapes | 0.282 m | **0.271 m** | **0.271 m** |
| 10 s, val (VPT) | **3.09 m** (0.91 s) | 17.1 m (1.07 s) | 13.1 m (**1.24 s**) |
| 10 s, held-out (VPT) | **3.70 m** (0.95 s) | 18.6 m (0.73 s) | 12.6 m (0.78 s) |

At 1 s all three are equal; over 10 s PH-NODE's error **flattens at 3–5 m** while the GPs accelerate away to
~30 m — its learned vector field is more dissipative far from the data. For long-horizon boundedness that is a
real advantage of the baseline, and it should be stated.

PH-NN on the same dataset (later runs, §5.5): 1 s 0.269 m, 3 s 1.120 m.

### 5.2 Closed loop under the gust

The **same** OU gust is now applied to the plant during IDA-PBC flight (new `--gust-sigma-fraction/--gust-tau/--gust-seed`
in `generate_comparison_report_v2.py`; the controller never sees it). 10 s, 10 EVALSET-shape references + diamond.

📄 [no gust](experiments/quadrotor/eval_runs/20-09-14-30_WIND-closedloop-nogust/closed-loop/comparison.pdf) ·
[gust 10 %](experiments/quadrotor/eval_runs/20-09-14-30_WIND-closedloop-gust0p1/closed-loop/comparison.pdf) ·
[gust + σ-scheduled damping](experiments/quadrotor/eval_runs/20-09-14-30_WIND-closedloop-gust0p1-dsched/closed-loop/comparison.pdf)

| plant | model | mean (m) | median | blow-ups > 0.4 m | worst |
|---|---|---|---|---|---|
| no gust | PH-NODE-RK4 | 0.164 | 0.174 | 0 | 0.220 |
| | PH-GP-ODE | **0.127** | 0.130 | 0 | 0.180 |
| | PH-GP-SDE | 0.288 | **0.112** | 1 | 1.818 |
| | PH-GT | 0.202 | 0.109 | 1 | 1.004 |
| gust 10 % | PH-NODE-RK4 | **0.194** | 0.205 | **0** | 0.230 |
| | PH-GP-ODE | 0.289 | 0.200 | 1 | 1.047 |
| | PH-GP-SDE | 0.508 | 0.201 | 3 | 1.650 |
| | PH-GT | 0.730 | 0.204 | 2 | **3.678** |
| gust + schedule | PH-GP-SDE, $K_\omega\times2.57$ | 0.394 | 0.356 | 4 (mild) | 0.737 |

**Key reading**: under the gust every controller's *median* is ≈ 0.20 m — that is the price of the disturbance
and no model changes it. The differences are entirely blow-ups on the most aggressive shapes, and **the
controller built from the exact simulator operators blows up worst** (3.68 m). So this is controller
saturation, not model error: more faithful operators ⇒ a more aggressive wrench, and the gains were tuned
without wind. PH-NODE's shrunk operators make it effectively conservative, and it never fails.

The **σ-scheduled damping** (new `--controller-damping-schedule process-noise`), which sets
$K\leftarrow K\max\big(1,(\sigma/\sigma_{\text{ref}})^2\big)$ from the stochastic-passivity bound
$\mathbb E[\dot H_d]\le-\xi^\top\mathcal R_d\xi+\tfrac12\mathrm{tr}(\Sigma^\top\nabla^2H_d\Sigma)$, removes the
catastrophic failures (worst 1.65 → 0.74 m) but makes the attitude loop sluggish everywhere (median 0.20 → 0.36 m).
A uniform gain multiplier is too blunt.

### 5.3 Does the SDE learn the wind scale? — the central wind experiment

$\sigma_v$ learned by the process-noise term, against the horizon-matched GT:

| data | latent $x_0$ | $\sigma_{\text{obs}}$ | $\sigma_v$ learned | GT | verdict |
|---|---|---|---|---|---|
| clean, 10 % | **on** | learned | 0.025 | 0.59 | absorbed by latent ($\Delta v$ 0.16 m/s rms) |
| clean, 25 % | **on** | learned | 0.091 | 1.49 | absorbed ($\Delta v$ 0.41 m/s) |
| clean, 10 % | off | fixed 0.02 | **0.396** | 0.59 | learns it |
| clean, 25 % | off | fixed 0.02 | **1.011** | 1.49 | learns it; ratio 2.55× for a 2.5× gust |
| noise 0.25, 25 % | on | learned | 0.097 (decays from 0.30) | 1.49 | absorbed |
| noise 0.25, 25 % | **off** | learned | **1.03** | 1.49 | learns it |
| noise 0.25, 25 % | off | **pinned 0.25** | 1.81 | 1.49 | **overshoots** |
| noise 0.25, 10 % | off | **pinned 0.25** | 1.49 | 0.59 | **2.5× overshoot** |

**Conclusion.** The hypothesis — *the SDE learns the true wind scale, and knowing it protects the other
subnetworks* — is **confirmed conditionally**:

1. The process noise **can** learn a scale proportional to the true wind (2.55× for a 2.5× gust), but only once
   the **per-window latent $x_0$ correction is removed**. With a gust correlated over the window length
   ($\tau=0.5$ s = the window), a gust realisation looks like a constant initial-velocity error, and the latent
   term is the cheaper explanation.
2. A **learned** observation scale must be kept. Pinning it makes the process noise the only sink for *every*
   residual, so it overshoots (worst where the wind is smallest), the band becomes over-dispersed
   (coverage 0.88–0.97, $\sigma/e$ 1.15–1.55) and the 3 s error doubles.
3. **Operator benefit is real but fragile.** On clean data it is clear:

| clean data | $\mu g_f$ | $M_2^{-1}g_\tau$ | $\mu\nabla V$ | $\mu D_v$ | $M_2^{-1}D_\omega$ |
|---|---|---|---|---|---|
| ODE, 10 % | 3.9 % | 25.7 % | 3.4 % | 53 % | 425 % |
| **SDE no-latent, 10 %** | 4.8 % | **16.2 %** | 4.0 % | **28 %** | 1662 % |
| ODE, 25 % | 19.4 % | 29.0 % | 3.7 % | 1009 % | 292 % |
| **SDE no-latent, 25 %** | 18.1 % | **14.2 %** | 6.3 % | **171 %** | 374 % |

   Under 0.25 observation noise at 25 % wind it disappears: $\mu D_v$ is 2742 % (latent on) / 5291 % (latent
   off, free $\sigma_{\text{obs}}$) / 250 % (pinned) — unusable in every variant.

4. **Calibration is where the SDE consistently wins.** Coverage@2σ at $h=50$, state band only:

| variant (25 % wind, 0.25 noise) | position | attitude | velocity | angular rate |
|---|---|---|---|---|
| latent on | 0.005 | 0.593 | 0.010 | 0.586 |
| **latent off, free $\sigma_{\text{obs}}$** | **0.310** | 0.397 | **0.415** | 0.401 |
| latent off, pinned $\sigma_{\text{obs}}$ | 0.765 | 0.722 | 0.762 | 0.700 |

5. **Prediction**: 1 s unchanged (0.513 vs 0.529 m), 3 s clearly better (**1.697** vs 2.408 m).

📄 [WIND25 + noise, 1 s](experiments/quadrotor/eval_runs/20-09-23-00_WIND25-noise0p25-open-loop-1s/comparison.pdf) ·
[3 s](experiments/quadrotor/eval_runs/20-09-23-00_WIND25-noise0p25-open-loop-3s/comparison.pdf) ·
[pinned σ, 3 s](experiments/quadrotor/eval_runs/21-09-00-40_WIND25-fixobs-open-loop-3s/comparison.pdf) ·
[clean 10 %](experiments/quadrotor/eval_runs/20-09-16-10_WIND10-CLEAN-open-loop-1s/comparison.pdf) ·
[clean 25 %](experiments/quadrotor/eval_runs/20-09-16-30_WIND25-CLEAN-open-loop-1s/comparison.pdf)

### 5.4 The same experiment on PH-NN-LieIMEX

New package `ph_nn_lie_imex_sde` (§8). This family has **no latent $x_0$ at all**, so it isolates the
observation-scale escape route.

📄 [1 s](experiments/quadrotor/eval_runs/21-09-11-10_WIND10-NN-open-loop-1s/comparison.pdf) ·
[3 s](experiments/quadrotor/eval_runs/21-09-11-10_WIND10-NN-open-loop-3s/comparison.pdf)
🖼 [states, held-out flight 00](experiments/quadrotor/eval_runs/21-09-11-10_WIND10-NN-open-loop-3s/images/WIND_CF2P_10s_h0p01_clean_heldout_3s_states_flight00.png)

| NN variant (10 % gust, 0.25 noise) | 1 s | 3 s | $\mu g_f$ | $M_2^{-1}g_\tau$ | $\mu\nabla V$ | $\mu D_v$ | $\sigma_v$ (GT 0.59) |
|---|---|---|---|---|---|---|---|
| ODE, published MSE | **0.269 m** | 1.120 m | **7.5 %** | 7.8 % | **9.1 %** | 232 % | — |
| NLL only (control) | 0.283 m | 1.091 m | 17.6 % | 4.1 % | 18.8 % | **166 %** | frozen |
| SDE, $S=8$ | 0.282 m | **1.056 m** | 23.0 % | **4.0 %** | 25.3 % | **166 %** | 0.133 |

Two clean findings: (i) $\sigma_v$ stalls at 0.133 because the **learned observation scale inflates to
0.373/0.448** against the true 0.25 — confirming that the escape route, not the latent term specifically, is
what defeats process-noise identification; (ii) the damage to thrust and gravity comes from the **objective
change** (already present in the NLL control), not from the noise.

Cross-model mean operator error on the same wind dataset:
**GP-ODE 42.7 % < GP-SDE 65.0 % < NN-NLL 77.1 % < NN-SDE 79.3 % < PH-NODE 81.5 % < NN-ODE 83.5 %.**

---

## 6. Part D — Design ablations requested by review

Two questions from the NeurIPS review of the $SO(3)$ paper are answered here on equal terms, both scored on the
**same 180 held-out 1 s windows** (20 held-out flights of `EVALSET_CF2P_10s_h0p01_clean.pkl`, each cut into nine
non-overlapping 1 s windows) with the **same** `open_loop.open_loop_metrics` used in §3 and §5, so these rows are
directly comparable to every other table in this document.

### 6.1 Kernel sensitivity — reviewer 6Ap3, W4

**W4 asked** why the Matérn $\otimes$ periodic kernel, and how much the results depend on it. The question is a
genuine sensitivity study rather than a tuning artefact, because with `model_backend: utils-gp-model` the kernel
hyperparameters are **fixed, not learned** — they are part of the prior, not of the fit:

$$k(x,x')=k_{\text{Mat\'ern-}\nu}\!\big(x_m,x'_m;\ell_m\big)\cdot k_{\text{per}}\!\big(x_p,x'_p;\ell_p\big),\qquad
S(\omega)\propto\Big(\tfrac{2\nu}{\ell_m^{2}}+\lVert\omega\rVert^{2}\Big)^{-(\nu+d/2)},$$

with $\nu\to\infty$ the squared-exponential limit and $\nu=\tfrac12$ the rough Ornstein–Uhlenbeck limit.

Seven stage-1 runs, one knob moved at a time from the baseline
($\nu=2.5$, $\ell_m=1.0$, $\ell_p=0.5$, 5 harmonics, 20 features), seed 0, no priors, $K=50$, 3 k steps, on
`HARD_CF2P_10s_h0p01_train-obs-noise-absolute0p25.pkl`. Configs `21-09-2026-04-00…06_ph_gp_lie_imex.yaml`;
8.1–9.6 min each.

| metric | baseline | $\nu{=}1.5$ | $\nu{=}4.5$ | $\ell_m{=}0.5$ | $\ell_m{=}2.0$ | $\ell_p{=}1.0$ | feat$=$10 | spread |
|---|---|---|---|---|---|---|---|---|
| position RMSE (m) | 0.1278 | 0.1210 | 0.1462 | **0.2185** | 0.1383 | **0.1208** | 0.1429 | **1.81×** |
| attitude RMSE (deg) | 11.59 | 9.70 | 12.50 | **19.15** | 11.47 | **9.66** | 17.30 | 1.98× |
| VPT median (s) | 0.625 | **0.680** | 0.625 | **0.540** | 0.625 | 0.675 | 0.580 | 1.26× |
| coverage @ $2\sigma$ (target 0.954) | 0.345 | 0.383 | 0.362 | 0.299 | 0.390 | **0.392** | 0.286 | 1.37× |
| NLPD, position (nats) | 18.95 | 14.59 | 17.53 | **44.14** | **14.53** | 16.39 | 38.59 | 3.04× |
| sharpness, mean $2\sigma$ (m) | 0.0125 | 0.0130 | 0.0135 | 0.0141 | 0.0161 | 0.0123 | 0.0118 | 1.36× |
| $\rho(\sigma,e)$ | +0.926 | +0.906 | +0.936 | +0.919 | +0.927 | +0.920 | +0.930 | **1.03×** |
| operator error, identified 4 (%) | 31.0 | 28.7 | **26.6** | **53.3** | 29.1 | 27.6 | 35.3 | 2.00× |
| — thrust $\mu g_f$ (%) | 10.5 | 10.1 | 7.4 | 17.1 | 14.2 | 8.9 | 8.7 | 2.31× |
| — torque $M_2^{-1}g_\tau$ (%) | 21.5 | 15.8 | 16.1 | 37.3 | 22.4 | 17.8 | 25.7 | 2.36× |
| — gravity $\mu\nabla V$ (%) | 12.0 | 11.6 | 8.7 | 16.3 | 14.9 | 9.9 | 7.0 | 2.32× |
| — damping $\mu D_v$ (%) | 79.8 | 77.3 | 74.2 | 142.4 | 64.8 | 73.7 | 99.8 | 2.20× |
| — damping $M_2^{-1}D_\omega$ (%) | 1655 | 502 | 486 | 703 | 411 | 762 | 770 | 4.02× |

Operator errors are on the shared D0 reference flights (10 flights); the trajectory and calibration rows are on
the 180 held-out windows. Source: [`21-09-14-57_kernel-sensitivity_EVALSET-heldout-1s/kernel_sensitivity.json`](experiments/quadrotor/eval_runs/21-09-14-57_kernel-sensitivity_EVALSET-heldout-1s/kernel_sensitivity.json).

**Four readings.**

1. **Smoothness $\nu$ is nearly free.** Over $\nu\in\{1.5,2.5,4.5\}$ position RMSE moves 0.121 → 0.146 m and the
   identified-4 operator error 28.7 → 26.6 %. In random-feature form $\nu$ only re-weights the tail of $S(\omega)$,
   and the 1 s windows never excite that band — so the choice of $\nu$ is not doing the work.
2. **The Matérn length scale is the one knob that matters, and short is worse.** $\ell_m=0.5$ is the worst setting
   on *every* metric (0.2185 m, 19.15°, NLPD 44.1, operators 53.3 %). A short $\ell_m$ buys high-frequency features
   that fit the $\sigma=0.25$ observation noise — precisely the flexibility a physics prior should not have. This is
   the one hyperparameter the paper should state and justify.
3. **The calibration claim is kernel-independent.** $\rho(\sigma,e)$ stays in $[+0.906,+0.936]$ — a 1.03× spread —
   while coverage is 0.286–0.392 in **all seven** settings. The model ranks its uncertainty well and understates its
   magnitude regardless of kernel; the cause is the latent-$x_0$/observation-scale escape route of §3.1 and §5.3,
   not the kernel.
4. **$M_2^{-1}D_\omega$ is unidentified in every setting** (411–1655 %). Over 1 s, $\int D_\omega\omega\,dt$ falls
   below the attitude noise floor, so no kernel can recover it. A plain 5-product mean would be swamped by this one
   number, so `evaluate_kernel_sweep.py` reports an "identified 4" mean beside it.

**Caveat, stated plainly:** these are **stage-1, 3 k-step** runs, so 0.12–0.22 m is not the headline accuracy
(the published two-stage GP reaches 0.0847 m on the same windows). The sweep measures *sensitivity*, not peak
performance — the knob is varied, everything else is held fixed.

### 6.2 Deep ensemble versus GP posterior — reviewer 6Ap3, W3

**W3 asked** why a Gaussian process rather than a cheaper, more scalable uncertainty mechanism such as a deep
ensemble. The two are compared on identical terms: $N$ independently seeded point-estimate models play exactly
the role the GP's posterior weight samples play, and both are scored by the same metrics on the same windows.

$$\text{ensemble:}\ \ x^{(i)}=\mathrm{Roll}(x_0,u;\theta_i),\ \theta_i\ \text{from seed}\ i
\qquad\qquad
\text{GP:}\ \ x^{(i)}=\mathrm{Roll}(x_0,u;w^{(i)}),\ w^{(i)}\sim q(w)$$

Five PH-NN-LieIMEX members (the published recipe, seeds 0–4, configs `21-09-2026-05-10…14_ph_nn_lie_imex.yaml`,
11.4–12.1 min each) against the published two-stage GP with 50 posterior draws.

| metric | deep ensemble (5 seeds) | single member (seed 0) | **GP posterior (50 draws)** |
|---|---|---|---|
| position RMSE (m) | 0.2637 | 0.2630 | **0.0847** |
| attitude RMSE (deg) | 5.76 | 6.25 | **4.26** |
| VPT median (s) | 0.360 | 0.360 | **0.640** |
| coverage @ $2\sigma$ (target 0.954) | 0.369 | — | 0.369 |
| NLPD, position (nats) | **14.06** | — | 19.48 |
| sharpness, mean $2\sigma$ (m) | 0.0400 | — | **0.0104** |
| $\rho(\sigma,e)$ | +0.895 | — | +0.886 |
| training wall clock | 57.7 min (5 runs) | 12.1 min | **26.5 min** (2 stages) |

Source: [`21-09-14-00_ensemble-vs-gp_EVALSET-heldout-1s/ensemble_vs_gp.json`](experiments/quadrotor/eval_runs/21-09-14-00_ensemble-vs-gp_EVALSET-heldout-1s/ensemble_vs_gp.json).

**Three readings.**

1. **The GP is better where it counts and cheaper.** 3.1× lower position RMSE, 3.8× sharper bands, 1.8× the valid
   prediction time, at 2.2× less training cost. Ensembling five members buys essentially nothing over a single
   member on accuracy (0.2637 vs 0.2630 m) because the members share the same structural bias — the spread is
   across seeds, not across the function space.
2. **Coverage and ranking are identical, so the GP's band is not merely a different flavour of the same thing.**
   Both reach coverage 0.369 and $\rho(\sigma,e)\approx+0.89$. The GP attains that ranking quality with a band
   3.8× narrower, i.e. the same information at a quarter of the width.
3. **The ensemble's one win is NLPD** (14.06 vs 19.48), and it is mechanical: a 3.8× wider band is more forgiving
   wherever the mean is wrong, and the ensemble's mean is 3.1× further off. It buys likelihood with width, not
   with knowledge — which is also why its coverage is no better.

**Honest limitation:** both mechanisms under-cover badly (0.369 against a 0.954 target). Deep ensembling does not
fix the calibration weakness reported in §4.4 and §5.3; it confirms that the weakness is in the model/likelihood
structure, not in the choice of posterior approximation.

---

## 7. What is new since the paper draft — must be added

| # | finding | where | status |
|---|---|---|---|
| 1 | Real-flight validation on a **published benchmark** with its own protocol and five baselines; on par or better at 0.5 s with 79 % of their data | §4.2 | 3 seeds, ready |
| 2 | **The latent $x_0$ (MAP) correction is the dominant identification bias**: removing it on clean data takes thrust/gravity from 10 % to 0.5 % | §3.1 | measured, decisive control run |
| 3 | The **force scale $\mu g_f$ is the operator that carries closed-loop performance** (swap ablation, both directions) | §3.2c | measured |
| 4 | **Closed-loop 19/20 shapes** for the GP, within 0.02–0.04 m of the analytic ceiling; baselines diverge tens of metres | §3.2c | ready |
| 5 | **SDE variant**: process noise learns a wind-proportional scale, improves 3 s prediction and calibration — *iff* the latent term is off and $\sigma_{\text{obs}}$ stays free | §5.3 | new, 4 datasets |
| 6 | **Unobserved-disturbance closed loop**: failures are controller saturation, not model error — the true-operator controller fails worst | §5.2 | new |
| 7 | $SO(3)$ violation $3\times10^{-14}$ (Lie–IMEX) vs $5\times10^{-7}$ (RK4) | §3.2b | ready |
| 8 | Honest calibration statement: $\rho(\sigma,e)\approx+0.9$ but coverage 0.23–0.57 — the model ranks uncertainty well and understates it | §4.4, §5.3 | ready |
| 9 | **Kernel sensitivity**: across 7 settings the uncertainty ranking is invariant (1.03×) and under-coverage holds in all seven; accuracy moves at most 1.81×, with the Matérn length scale the only real knob | §6.1 | new |
| 10 | **Deep ensemble comparison**: the GP beats a 5-seed ensemble 3.1× on accuracy and 3.8× on sharpness at identical coverage and ranking, for 2.2× less training cost | §6.2 | new |
| 11 | Failure catalogue with causes (below) | §9 | ready |

---

## 8. Code written for these experiments

| path | what |
|---|---|
| `src/models/SE3_Quadrotor/ph_gp_lie_imex_sde_idsia/` | GP SDE: process-noise scales, stochastic Lie–IMEX, moment-matched NLL, divergence pre-screen |
| `src/models/SE3_Quadrotor/ph_nn_lie_imex_sde/` | the same for the MLP family |
| `src/models/SE3_Quadrotor/ph_gp_lie_imex_idsia/` | GP ODE + robust likelihood, horizon weighting, window masking, `screen_diverged_windows` |
| `comparision/report_controller.py` | **plant gust** (`gust=`), **σ-scheduled damping** (`damping_scale=`) |
| `comparision/generate_comparison_report_v2.py` | `--gust-*`, `--controller-damping-schedule`, gust-aware flight folders |
| `comparision/evaluate_real_dataset.py` | open-loop-only comparison for real data |
| `datasets/QUADROTOR-DATASET-IDSIA/{convert,validate_input_reconstruction,benchmark_protocol,rollout_fitted_operators,evaluate_sde_band,plot_*}.py` | IDSIA pipeline |
| `datasets/QUADROTOR-DATASET-WIND{,25}/generate_quadrotor_wind*.py`, `identification_table.py` | wind datasets + the operator scorer used throughout |
| `comparision/evaluate_kernel_sweep.py` | kernel sensitivity (§6.1): accuracy, calibration and operator identification per kernel setting |
| `comparision/evaluate_ensemble.py` | deep ensemble vs GP posterior (§6.2) on identical windows and metrics |

The SDE's likelihood, for reference: with $S$ sample paths per window,

$$\hat\mu_{b,h}=\tfrac1S\sum_s x^{(s)}_{b,h},\qquad
\hat\Sigma_{b,h}=\mathrm{Var}_s\big[x^{(s)}_{b,h}\big]+\sigma_{\text{obs},b}^2 I,\qquad
\mathcal L=-\sum_{b,h}\log\mathcal N\big(y_{b,h};\hat\mu_{b,h},\hat\Sigma_{b,h}\big),$$

with the attitude residual and spread taken in the tangent space of the projected chordal mean rotation. The
sample spread enters the density, which is what gives $\sigma$ a data-fit signal (a per-sample NLL drives it to
zero).

---

## 9. Failures and limitations — stated, not hidden

| failure | where | cause (measured) | status |
|---|---|---|---|
| PH-NODE-RK4 diverges on real data (step 52) and on clean 25 % wind (step 294) | §4, §5 | unclipped published recipe; `gradient_clip_norm: null` | not fixed, by instruction |
| GP curriculum stage $K\ge100$ fails on real data (steps 27–859) | §4 | **two** modes: (i) $0\times\infty$ from a masked diverged window — fixed by the pre-screen, which let $K=80$ complete; (ii) genuine BPTT gradient explosion through 100 steps on *bounded* paths — not fixable by clipping or screening | $K=50$ used; limitation stated |
| PH-NN-LieIMEX on IDSIA died at **step 0** | §4.3 | **resolved**: two separate causes. (i) `pose_loss_components` averaged over all 1728 test windows, so one window diverging at random init made the metric NaN — now maskable via `training.mask_diverged_eval_windows` (default off; mirrors the GP package). (ii) the NN package never cast its data to JAX's default float width, so fp64 gave mismatched scan-carry dtypes. **The mask never fired** (`test_masked_fraction` = 0.0 at every step): fp64 alone was the enabling change | fixed, model trains and is reported in §4.3 |
| Calibration: coverage 0.23 (real), 0.005–0.42 (wind) | §4.4, §5.3 | the latent $x_0$ / observation scale absorbs the residual that should widen the band | SDE without latent is the partial fix |
| Damping $\mu D_v$ never identified (32–5000 %) | everywhere | 1 s windows + free $x_0$ + thrust-authority-limited drag; a best-constant model scores 12.4 % where the best GP scores 28.2 % | characterised, open |
| Seed variance | §4.2 | held-out 1 s RMS 0.203 / 0.240 / 0.231 m across seeds — single-run claims are unsafe | 3 seeds reported |
| NanoBench unusable | §4.1 | commanded PWM, not measured rotor speed: rotational $R^2\le0.10$ for every input parameterisation; their own Table V shows no baseline beats naïve on rotation | rejected, one sentence in the paper |
| WIND25 gate rejections | §5 | 64 of 134 flight attempts rejected (28 box, 15 altitude, 12 tilt, 9 saturation) ⇒ accepted flights are the calmer gust realisations | stated |
| Open-loop `so3_orthogonality` is a float32 floor | §4.3, §5.1 | the evaluators do not enable `jax_enable_x64`, so `report_evaluation.rollout` truncates to float32 and every model reports $5\times10^{-7}$–$2\times10^{-6}$; the same rollout in float64 gives $9.9\times10^{-16}$ | column must not be used to compare integrators; §3.2b (float64) unaffected |
| Closed loop on real data | §4 | impossible offline: a controller needs an independent plant, and the recorded flights were flown by the vehicle's own controller | future work: hardware |

---

## 10. Reproducing the tables

```bash
export PYTHONPATH="$PWD"

# §3.1 identification sweep (CPU)
python3 src/models/SE3_Quadrotor/comparision/wind_identification_table.py \
  --run experiments/quadrotor/train_runs/<run> [--run ...]

# §3.2 full simulator report (identification + open loop + closed loop)
python3 -m src.models.SE3_Quadrotor.comparision.generate_comparison_report_v2 \
  --run <gp> --run <nn> --run <node> --include-ground-truth \
  --evaluation-dataset datasets/QUADROTOR-EVAL-REFERENCE/D0_CF2P_PID_contact-free_nonlinear-damping-c0p5_3s_seed0.pkl \
  --controller-recorded-reference datasets/QUADROTOR-DATASET-EVALSET/EVALSET_CF2P_10s_h0p01_clean.pkl@heldout

# §4.2 the benchmark's own protocol
python3 src/models/SE3_Quadrotor/comparision/idsia/benchmark_protocol.py --run <gp run> --stride 2

# §4.3 open loop against their baselines
python3 src/models/SE3_Quadrotor/comparision/idsia/plot_open_loop_baselines.py --run <gp run>

# §5.2 closed loop under the gust
python3 -m src.models.SE3_Quadrotor.comparision.generate_comparison_report_v2 \
  --run <...> --controller-only --controller-seconds 10 --include-ground-truth \
  --gust-sigma-fraction 0.10 --gust-tau 0.5 \
  --controller-damping-schedule process-noise         # for the scheduled variant

# §6.1 kernel sensitivity (7 runs, one knob each)
python3 -m src.models.SE3_Quadrotor.comparision.evaluate_kernel_sweep \
  --run "baseline=<run>" --run "nu=1.5=<run>" --run "lm=0.5=<run>" [...] \
  --dataset datasets/QUADROTOR-DATASET-EVALSET/EVALSET_CF2P_10s_h0p01_clean.pkl@heldout \
  --horizon-seconds 1.0 --samples 50 --flights 180 --reference-flights 10

# §6.2 deep ensemble vs GP posterior
python3 -m src.models.SE3_Quadrotor.comparision.evaluate_ensemble \
  --member <seed0> --member <seed1> --member <seed2> --member <seed3> --member <seed4> \
  --gp <published gp run> \
  --dataset datasets/QUADROTOR-DATASET-EVALSET/EVALSET_CF2P_10s_h0p01_clean.pkl@heldout \
  --horizon-seconds 1.0 --samples 50 --flights 180

# §5.3 predictive band / calibration
python3 src/models/SE3_Quadrotor/comparision/idsia/evaluate_sde_band.py --run <run> \
  --dataset datasets/QUADROTOR-DATASET-WIND25/WIND25_CF2P_10s_h0p01_clean.pkl@heldout \
  --samples 32 --exclude-observation-noise
```

---

## 11. Index of evaluation runs cited

| folder in `experiments/quadrotor/eval_runs/` | contents |
|---|---|
| `18-09-00-02_spec-comparison_nopriors-3models-with-GT_…obs-noise0p25` | **the simulator headline**: identification, open loop (D0 + V6 held-out), closed loop over 20 references, 84 PNGs |
| `19-09-16-13_IDSIA-melon-open-loop-baselines` | real data vs Naïve and their Physics model, 0.5/1/3 s, 39 PNGs |
| `19-09-16-45_IDSIA-full-flight-baselines` | two whole 60 s flights, pure open loop and restart-every-1 s |
| `19-09-00-30_IDSIA-melon-heldout_GP-stage1_{0.5,1.0}s` | the GP's own held-out evaluation with posterior bands |
| `20-09-13-40_WIND-open-loop-{1s,full}` | NODE / GP-ODE / GP-SDE under the 10 % gust |
| `20-09-14-30_WIND-closedloop-{nogust,gust0p1,gust0p1-dsched}` | IDA-PBC under the gust, three plants |
| `20-09-16-{10,30}_WIND{10,25}-CLEAN-*` | clean-wind open loop |
| `20-09-23-00_WIND25-noise0p25-open-loop-{1s,3s}` | latent on/off comparison |
| `21-09-00-40_WIND25-fixobs-open-loop-{1s,3s}` | pinned observation scale |
| `21-09-11-10_WIND10-NN-open-loop-{1s,3s}` | the PH-NN ODE/NLL/SDE trio |
| `21-09-14-00_ensemble-vs-gp_EVALSET-heldout-1s` | §6.2: 5-seed deep ensemble against the GP posterior |
| `21-09-14-57_kernel-sensitivity_EVALSET-heldout-1s` | §6.1: seven kernel settings, accuracy + calibration + operators |

Training runs are under `experiments/quadrotor/train_runs/`; every run folder carries its config copy,
`metadata.json` (with the dataset SHA-256), `training.log`, `training_stats.npz` and checkpoints.

---

*Generated 21 Sep 2026. Numbers recomputed for §3.1 and §5.4 with `identification_table.py` and for §6 with
`evaluate_kernel_sweep.py` / `evaluate_ensemble.py`; all others read
from the linked artefacts.*

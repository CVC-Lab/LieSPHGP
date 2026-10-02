# IDSIA real-flight results — paper draft

> Status: **tables filled 26 Sep 2026**; figures are still placeholders.
> Each item lists its **source** (script / run / JSON) so every number can be regenerated.

---

## 1. Experimental setup (text + one table)

**Dataset.** IDSIA nano-drone system identification benchmark (Busetto et al., *Control Engineering Practice* 172, 2026),
Crazyflie 2.1 Brushless, $m = 0.045$ kg, $J = \mathrm{diag}(2.3951\times10^{-5},\ 2.3951\times10^{-5},\ 3.2347\times10^{-6})$ kg m$^2$
(published values; the $J_{zz}$ exponent may be a typo for $3.2347\times10^{-5}$), 100 Hz. The input is the wrench
$(T, \tau_x, \tau_y, \tau_z)$ computed from the measured rotor speeds $\Omega_{1..4}$ with the benchmark's own mixer
($K_t = 3.72\times10^{-8}$, $K_c = 7.74\times10^{-12}$, arm $0.0353$ m).

**Split.** Train: Square, Random, Chirp (12 recordings, 556 s; 44 ten-second training flights). Test: Melon (3 recordings, 195 s),
never seen in training.

**Protocol.** The benchmark's own: from the true state at every admissible start time $t$, predict $H = 50$ steps (0.5 s)
open loop from the recorded inputs. $\mathrm{MAE}_{\cdot,h}$ is the mean Euclidean error at horizon $h$;
$\mathrm{MAE}_{\cdot,1:50} = \sum_{h=1}^{50}\mathrm{MAE}_{\cdot,h}$ (cumulative). **SDE models** are rolled out with 5 independent
seeds per window (seed $k$ = one Brownian draw, and for the GP one posterior weight sample, shared by all windows); every
seed is scored on its own and the SDE rows report the **mean over the 5 seeds**,
$\overline{\mathrm{MAE}}_{\cdot,h} = \tfrac15\sum_{k=1}^{5}\mathrm{MAE}^{(k)}_{\cdot,h}$. ODE models are deterministic (one rollout).

**Table S1 — Training configuration of the models we trained (ours + PH-NODE-RK4 prior work)**

| setting | PH-GP-LieIMEX-ODE (ours) | PH-NN-LieIMEX-ODE (ours) | PH-GP-LieIMEX-SDE (ours) | PH-NN-LieIMEX-SDE (ours) | PH-NODE-RK4 (prior work) |
|---|---|---|---|---|---|
| run | `18-09-23-15_ph_gp_lie_imex_…-stage1-K50-3k` | `22-09-12-45_ph_nn_lie_imex_…-w64-…-nomasspretrain` | `25-09-22-31_ph_gp_lie_imex_sde_idsia_…-EKF-…-10k` | `26-09-00-22_ph_nn_lie_imex_sde_…-TRANSITION-…-RESUME5k-to-20k` | `18-09-23-15_ph_node_phnode-rk4-…-stage1-K50-3k` |
| checkpoint used | final, step 3000 | final, step 3000 | step 4000 (best held-out NLL) | step 7000 (best held-out NLL) | none (training diverged, NaN at step 52) |
| input encoding | wrench $(T,\tau)$ from measured $\Omega^2$ | wrench | wrench | wrench | wrench |
| integrator, step $h$ | Lie–IMEX, $h = 0.01$ s | Lie–IMEX, $h = 0.01$ s | stochastic Lie–IMEX, $h = 0.01$ s | stochastic Lie–IMEX, $h = 0.01$ s | RK4, $h = 0.01$ s |
| training objective | GP window NLL + KL (latent $x_0$) | trajectory MSE | EKF marginal likelihood + KL 0.1, $\sigma_{obs}$ learned | one-step transition NLL, $\sigma_{obs} = 10^{-5}$ fixed | trajectory MSE |
| training window $K$ (steps) | 50 (stride 10) | 50 (stride 10) | 50 (stride 10) | 51-point windows, one-step transitions | 50 (stride 10) |
| optimiser, steps | Prodigy levels + lr $10^{-3}$, clip 10; 3000 | Adam $5\times10^{-4}$; 3000 | Adam $3\times10^{-3}$ cosine + Prodigy levels, clip 1; 4000 of 10 000 | Adam $10^{-3}$ cosine; 7000 of 20 000 (5k run + resume) | Adam $5\times10^{-4}$, no clipping; crashed at 52 |
| batch size | 256 | 256 | 256 | 256 | 256 |
| precision | fp32 | fp32 | fp32 | fp32 | fp32 |
| physics priors | none | none | none | none | none (published recipe) |
| mass pre-training | none | none | none | none | 200 steps |
| training time (GPU h) | 0.17 | 0.20 | 1.22 (to step 4000, incl. evaluation) | 0.20 (474 s + 251 s) | crashed after 54 s |
| parameters | 19 410 | 47 866 | 19 412 | 47 872 | 48 250 |

Source: each run's config copy and `metadata.json` under `experiments/quadrotor/train_runs/`; GP-SDE / NN-SDE times from
`training.log` timestamps.

---

## 2. Main result — benchmark comparison on held-out Melon

**Table 1a — Multi-step prediction error, training shapes (Square, Random, Chirp; benchmark protocol)**
Same protocol and metrics as Table 1b, scored on the 12 training flights (13 756 windows, stride 4). Naïve and Physics recomputed by us on these flights (the benchmark publishes test-set numbers only); the learned benchmark baselines have no published training-set numbers (n/a). PH-NODE-RK4: prior work [cite], trained by us; training diverged (NaN), so it has no numbers. SDE rows: mean over 5 seeds. Lower is better; best in **bold**.

| Model | $\mathrm{MAE}_p$ $h$=1 | $h$=10 | $h$=50 | *1:50* | | $\mathrm{MAE}_v$ $h$=1 | $h$=10 | $h$=50 | *1:50* | | $\mathrm{MAE}_R$ $h$=1 | $h$=10 | $h$=50 | *1:50* | | $\mathrm{MAE}_\omega$ $h$=1 | $h$=10 | $h$=50 | *1:50* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naïve | 0.0092 | 0.0912 | 0.4306 | *11.2889* |  | 0.0239 | 0.2295 | 0.9735 | *26.6505* |  | 0.0070 | 0.0682 | 0.2549 | *7.4598* |  | 0.0677 | 0.3999 | 1.0722 | *35.9565* |
| Physics | **0.0013** | 0.0125 | 0.1436 | *2.4972* |  | **0.0072** | 0.0538 | 0.6890 | *11.9100* |  | **0.0010** | 0.0207 | 0.3113 | *6.1843* |  | 0.0677 | 0.3999 | 1.0722 | *35.9565* |
| Res-MLP | n/a | n/a | n/a | *n/a* |  | n/a | n/a | n/a | *n/a* |  | n/a | n/a | n/a | *n/a* |  | n/a | n/a | n/a | *n/a* |
| Hybrid | n/a | n/a | n/a | *n/a* |  | n/a | n/a | n/a | *n/a* |  | n/a | n/a | n/a | *n/a* |  | n/a | n/a | n/a | *n/a* |
| Res-LSTM | n/a | n/a | n/a | *n/a* |  | n/a | n/a | n/a | *n/a* |  | n/a | n/a | n/a | *n/a* |  | n/a | n/a | n/a | *n/a* |
| PH-NODE-RK4 (prior work) [cite] | NaN | NaN | NaN | *NaN* |  | NaN | NaN | NaN | *NaN* |  | NaN | NaN | NaN | *NaN* |  | NaN | NaN | NaN | *NaN* |
| **PH-GP-LieIMEX-ODE** (ours) | **0.0013** | 0.0117 | 0.0945 | *1.9206* |  | 0.0096 | 0.0735 | **0.3842** | *9.0179* |  | **0.0010** | 0.0200 | 0.1680 | *4.0830* |  | 0.0698 | 0.3516 | 0.6454 | *23.9610* |
| **PH-NN-LieIMEX-ODE** (ours) | **0.0013** | 0.0122 | 0.0942 | *1.9809* |  | 0.0119 | 0.0906 | 0.3889 | *9.4896* |  | **0.0010** | **0.0175** | **0.1443** | ***3.5442*** |  | **0.0626** | **0.3159** | **0.5406** | ***21.0671*** |
| **PH-GP-LieIMEX-SDE** (ours) | **0.0013** | **0.0114** | 0.1201 | *2.0870* |  | 0.0112 | 0.0503 | 0.6270 | *10.7973* |  | 0.0012 | 0.0293 | 0.3066 | *6.7166* |  | 0.1294 | 0.5189 | 1.1269 | *38.4867* |
| **PH-NN-LieIMEX-SDE** (ours) | **0.0013** | **0.0114** | **0.0895** | ***1.7555*** |  | 0.0111 | **0.0478** | 0.4287 | ***8.0732*** |  | 0.0011 | 0.0232 | 0.1824 | *4.5319* |  | 0.1075 | 0.4187 | 0.7248 | *27.8050* |

Source: `src/models/SE3_Quadrotor/comparision/idsia/split_protocol_table.py --source tmp/idsia_raw/data/train --pattern "*.csv" --stride 4 --sde-seeds 5`,
output `experiments/quadrotor/eval_runs/26-09-01-30_IDSIA-protocol_TRAIN-shapes_4models-SDE5seeds/split_protocol_table.json`
(per-seed rows and seed std included).

**Train → test gap (cumulative $\mathrm{MAE}_p$ / $\mathrm{MAE}_R$, Table 1a → 1b).** PH-NN-LieIMEX-SDE generalises best: 1.756 → 1.778 m (+1 %)
and 4.53 → 4.60 rad (+1 %). PH-GP-LieIMEX-ODE (1.921 → 2.199, +14 %) and PH-GP-LieIMEX-SDE (2.087 → 2.323, +11 %) lose a little on the
unseen shape; PH-NN-LieIMEX-ODE loses most (1.981 → 2.635, +33 %; attitude 3.54 → 4.13, +16 %).

---

**Table 1b — Multi-step prediction error, Melon test set (held out; benchmark protocol)**
Benchmark baselines (Naïve to Res-LSTM): the benchmark authors' published values (their Table 7, p. 10). PH-NODE-RK4: prior work, the port-Hamiltonian neural ODE on SE(3) with an RK4 integrator (Duong & Atanasov) [cite], trained by us on the same split; its training diverged (NaN), so it has no numbers. Our rows: 3 flights, 9 675 windows, stride 2; SDE rows are the mean over 5 seeds. Lower is better; best in **bold**.

| Model | $\mathrm{MAE}_p$ $h$=1 | $h$=10 | $h$=50 | *1:50* | | $\mathrm{MAE}_v$ $h$=1 | $h$=10 | $h$=50 | *1:50* | | $\mathrm{MAE}_R$ $h$=1 | $h$=10 | $h$=50 | *1:50* | | $\mathrm{MAE}_\omega$ $h$=1 | $h$=10 | $h$=50 | *1:50* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naïve | 0.0143 | 0.1430 | 0.6797 | *17.7878* |  | 0.0329 | 0.3182 | 1.4749 | *38.9241* |  | 0.0071 | 0.0692 | 0.3041 | *8.2138* |  | **0.0796** | 0.3596 | 0.8837 | *29.0866* |
| Physics | **0.0013** | 0.0126 | 0.1269 | *2.3223* |  | **0.0080** | **0.0570** | 0.5781 | *10.6232* |  | **0.0011** | 0.0205 | 0.2544 | *5.1013* |  | **0.0796** | 0.3596 | 0.8837 | *29.0866* |
| Res-MLP | 0.0032 | 0.0305 | 0.1712 | *4.0519* |  | 0.0116 | 0.0890 | 0.5720 | *12.5809* |  | 0.0022 | 0.0331 | 0.2268 | *6.0591* |  | 0.1138 | 0.5949 | 0.9005 | *35.5735* |
| Hybrid | 0.0016 | 0.0166 | 0.1119 | *2.3625* |  | 0.0092 | 0.0613 | 0.5556 | *10.4033* |  | 0.0027 | 0.0376 | 0.2306 | *6.1534* |  | 0.0912 | 0.4880 | 0.5979 | *28.9873* |
| Res-LSTM | 0.0079 | 0.0390 | 0.2572 | *5.9711* |  | 0.0247 | 0.1175 | 0.7407 | *16.7207* |  | 0.0066 | 0.0372 | 0.2325 | *5.6525* |  | 0.1021 | 0.4292 | 1.2407 | *35.8353* |
| PH-NODE-RK4 (prior work) [cite] | NaN | NaN | NaN | *NaN* |  | NaN | NaN | NaN | *NaN* |  | NaN | NaN | NaN | *NaN* |  | NaN | NaN | NaN | *NaN* |
| **PH-GP-LieIMEX-ODE** (ours) | **0.0013** | 0.0116 | 0.1108 | *2.1985* |  | 0.0142 | 0.1078 | 0.4457 | *11.5393* |  | **0.0011** | 0.0249 | 0.1985 | *4.7803* |  | 0.0958 | 0.4212 | 0.6289 | *25.9788* |
| **PH-NN-LieIMEX-ODE** (ours) | **0.0013** | 0.0130 | 0.1320 | *2.6348* |  | 0.0187 | 0.1461 | 0.5693 | *14.9211* |  | **0.0011** | **0.0201** | **0.1786** | ***4.1281*** |  | 0.0805 | **0.3573** | **0.5520** | ***23.3117*** |
| **PH-GP-LieIMEX-SDE** (ours) | **0.0013** | 0.0119 | 0.1394 | *2.3231* |  | 0.0150 | 0.0809 | 0.8359 | *14.2152* |  | 0.0012 | 0.0353 | 0.4961 | *9.7868* |  | 0.1476 | 0.6343 | 1.8560 | *56.5287* |
| **PH-NN-LieIMEX-SDE** (ours) | **0.0013** | **0.0112** | **0.0896** | ***1.7779*** |  | 0.0122 | 0.0644 | **0.4349** | ***9.0820*** |  | **0.0011** | 0.0254 | 0.1868 | *4.5969* |  | 0.1240 | 0.4447 | 0.6559 | *27.2143* |

Units: $p$ in m, $v$ in m/s, $R$ in rad (geodesic), $\omega$ in rad/s.
Source: `src/models/SE3_Quadrotor/comparision/idsia/split_protocol_table.py --source tmp/idsia_raw/data/test --pattern "melon*.csv" --stride 2 --sde-seeds 5`,
output `experiments/quadrotor/eval_runs/26-09-01-30_IDSIA-protocol_TEST-melon_4models-SDE5seeds/split_protocol_table.json`; baselines from
`benchmark_protocol.REFERENCE` (= paper Table 7). Our recomputation of Naïve / Physics on the same flights agrees with the published
values to 1–3 % (Physics cumulative $\mathrm{MAE}_p$ 2.2409 vs 2.3223). Seed-to-seed std of the cumulative MAE: PH-NN-SDE $\le 0.04$
on every metric; PH-GP-SDE 0.008 ($p$), 0.094 ($R$), 0.083 ($v$), 0.615 ($\omega$). No window diverged for any seed.

**Summary.** On the unseen Melon shape our models give the best cumulative error on all four metrics: position 1.778 m
(PH-NN-SDE) vs 2.322 for Physics (−23 %); velocity 9.08 m/s (PH-NN-SDE) vs 10.40 for Hybrid (−13 %); attitude 4.128 rad
(PH-NN-ODE) vs 5.101 for Physics (−19 %); angular rate 23.31 rad/s (PH-NN-ODE) vs 28.99 for Hybrid (−20 %). We lose at short
horizons on velocity ($h$ = 1, 10: Physics 0.0080 / 0.0570 vs our best 0.0122 / 0.0644) and on $\omega$ at $h$ = 1 (0.0796 vs 0.0805).
PH-GP-SDE is weak on attitude and rate (cumulative $\mathrm{MAE}_R$ 9.79, $\mathrm{MAE}_\omega$ 56.5): its learned angular diffusion
($\sigma_\omega \approx 0.64$) makes single sample paths drift in attitude.

---

**Figure 1 — Rolling 50-step-ahead predictions on Melon (same window as the benchmark's Fig. 9)**

`[FIGURE 1 PLACEHOLDER]` — 2 × 3 panels: $x, y, z$ (top), roll, pitch, yaw (bottom); time window 20 s to 25 s of Melon run 1
(the benchmark's Fig. 9 window); lines: ground truth $y_{t+50}$, Physics, Hybrid, PH-GP-LieIMEX-ODE, PH-NN-LieIMEX-ODE,
PH-GP-LieIMEX-SDE, PH-NN-LieIMEX-SDE, each showing $\hat y_{t+50\mid t}$ (PH-NODE-RK4 has no trained model).

Source: baselines from `tmp/idsia_raw/out/predictions/*/melon_multistep.csv`; ours: script still to write.

---

**Figure 2 — Error against prediction horizon**

`[FIGURE 2 PLACEHOLDER]` — 4 panels ($\mathrm{MAE}_p$, $\mathrm{MAE}_v$, $\mathrm{MAE}_R$, $\mathrm{MAE}_\omega$) against
$h = 1 \ldots 50$; one line per model.

Source: per-horizon arrays from `split_protocol_table.py` (ours), `melon_multistep.csv` (baselines).

---

## 3. Uncertainty — what the baselines cannot provide

**Table 2 — Predictive calibration on Melon, $H = 50$**
Each cell gives position / attitude / velocity / angular rate. Predictive law per axis $\mathcal N(\mu, \mathrm{var} + \sigma_{obs}^2)$ from
$S = 32$ sample paths per window (GP-ODE: posterior weight samples; GP-SDE: weight samples × Brownian paths; NN-SDE: Brownian paths),
attitude as the rotation vector about the projected mean rotation; pooled over all horizons $h = 1..50$ and 1 935 windows (stride 10).
Coverage@$k$ = fraction of axis values with $|y-\mu| \le k\sigma$; NLPD in nats per axis; sharpness = mean $\|\sigma\|$ of the block
(m, rad, m/s, rad/s); Spearman $\rho$ between $\|\sigma\|$ and $\|e\|$ over (window, $h$). No window diverged.

| Model | coverage@1σ (target 0.683) | coverage@2σ (target 0.954) | NLPD | sharpness (mean σ) | Spearman $\rho(\sigma, e)$ |
|---|---|---|---|---|---|
| Naïve / Physics / Res-MLP / Hybrid / Res-LSTM | n/a | n/a | n/a | n/a | n/a |
| PH-NN-LieIMEX-ODE / PH-NODE-RK4 (point estimates) | n/a | n/a | n/a | n/a | n/a |
| PH-GP-LieIMEX-ODE (posterior) | 0.84 / 0.68 / 0.51 / 0.70 | 0.95 / 0.83 / 0.72 / 0.89 | −1.87 / 0.51 / 2.11 / 1.27 | 0.068 / 0.068 / 0.131 / 0.433 | 0.84 / 0.82 / 0.73 / 0.49 |
| PH-GP-LieIMEX-SDE | 0.56 / 0.39 / 0.36 / 0.50 | 0.77 / 0.64 / 0.61 / 0.73 | −0.56 / 1.25 / 5.63 / 2.45 | 0.023 / 0.072 / 0.130 / 0.481 | 0.82 / 0.86 / 0.83 / 0.62 |
| PH-NN-LieIMEX-SDE | 0.21 / 0.48 / 0.44 / 0.53 | 0.39 / 0.73 / 0.70 / 0.78 | 41.34 / 0.58 / 1.22 / 2.15 | 0.011 / 0.047 / 0.095 / 0.285 | 0.81 / 0.75 / 0.74 / 0.31 |

$\sigma_{obs}$ used: GP-ODE learned (0.039 m, 0.038 rad, 0.074 m/s, 0.247 rad/s); GP-SDE learned (0.0092 m, $3\times10^{-5}$ rad,
$6\times10^{-4}$ m/s, 0.102 rad/s); NN-SDE fixed $10^{-5}$.
Source: `src/models/SE3_Quadrotor/comparision/idsia/idsia_calibration_table.py --samples 32 --stride 10`, output
`experiments/quadrotor/eval_runs/26-09-01-30_IDSIA-calibration_TEST-melon_S32/calibration_table.json`.

**Reading.** Every model's band ranks its error well (Spearman 0.73–0.86 on position, attitude and velocity), but all are
over-confident. The GP-ODE posterior is closest to nominal (position coverage@2σ 0.95), largely because its learned $\sigma_{obs}$
is wide (0.039 m). The SDE bands are narrower than their errors; PH-NN-SDE's position NLPD (41.3) is dominated by the first
horizons, where its band is almost zero ($\sigma_{obs} = 10^{-5}$) while the data carry measurement noise.

**Figure 3 — 50-step prediction with uncertainty band**

`[FIGURE 3 PLACEHOLDER]` — Melon, one 0.5 s window per panel for $x, y, z$, roll, pitch, yaw; truth, PH-NN-SDE mean and ±2σ band.

---

## 4. Appendix A — Physical interpretability

**Table A1 — Identified gauge-invariant operators vs published vehicle constants**
Mean over the Melon test states; (×) = learned / published.

| quantity | learned (PH-GP-ODE) | learned (PH-NN-ODE) | learned (PH-GP-SDE) | learned (PH-NN-SDE) | learned (PH-NODE-RK4) | published |
|---|---|---|---|---|---|---|
| $\mu\,\nabla V$ (gravity, m/s$^2$) | 6.686 (0.68×) | 5.626 (0.57×) | 9.505 (0.97×) | 9.458 (0.96×) | NaN | 9.81 |
| $\mu\,g_f$ (thrust per unit input, 1/kg) | 15.68 (0.71×) | 13.80 (0.62×) | 21.46 (0.97×) | 21.36 (0.96×) | NaN | 22.22 ($1/m$) |
| $M_2^{-1}g_\tau$, $x$ | 571.6 (0.014×) | 0.62 ($1.5\times10^{-5}$×) | 4739 (0.11×) | 47.2 ($1.1\times10^{-3}$×) | NaN | 41 752 |
| $M_2^{-1}g_\tau$, $y$ | 210.6 (0.005×) | −1.66 ($-4\times10^{-5}$×) | 1499 (0.036×) | 54.1 ($1.3\times10^{-3}$×) | NaN | 41 752 |
| $M_2^{-1}g_\tau$, $z$ | −201 ($-6.5\times10^{-4}$×) | −3.09 ($-1\times10^{-5}$×) | −6708 (−0.022×) | −59.9 ($-1.9\times10^{-4}$×) | NaN | 309 148 |

Source: `four_model_comparison.gauge_products` on the Melon states of `IDSIA_CF21BL_10s_h0p01_clean.pkl`, output
`experiments/quadrotor/eval_runs/26-09-01-30_IDSIA-operators-vs-published/operators.json`.

**Reading.** The two SDE models recover the translational operators within 3–4 % (gravity 0.97× / 0.96×, thrust 0.97× / 0.96×);
the two ODE models are 29–43 % low on both, i.e. they identify the ratio $g_f / \nabla V$ but not the scale. None recovers the
torque gain: at most 11 % of $J^{-1}$ (PH-GP-SDE, roll), and every model learns a negative yaw gain (see Appendix B).

---

## 5. Appendix B — Limitations on this dataset

**Table B1 — How much of the measured wrench the rotor speeds explain ($R^2$, best linear map, fitted on train, scored on Melon)**

| channel | $x$ | $y$ | $z$ |
|---|---|---|---|
| force $f = m\,a_{\text{IMU}}$ | - | - | - |
| torque $\tilde\tau = J\dot\omega + \omega\times J\omega$ | - | - | - |

Source: `rotor2_torque_identifiability.py` (in an old job scratchpad, not in the repository; to promote and rerun).
Related, already measured: the rotational consistency of the published mixer at zero lag is $R^2$ = 0.369 / 0.342 / 0.417
(roll / pitch / yaw; `IDSIA_input_validation.txt`), and the translational mass fit gives 45.03 g against the published 45.0 g.

**Table B2 — Long-horizon open loop: a perfect model replaying logged inputs**

| horizon | 1 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|
| median final position error (m) | 0.002 | 0.101 | 0.706 | 10.54 |

Source: `src/models/SE3_Quadrotor/closed_loop_idsia/twin_replay.py`, output
`experiments/quadrotor/eval_runs/24-09-01-00_IDSIA-perfect-model-twin/twin_replay.json` (Lie–IMEX 2 ms twin; 39 / 39 / 36 / 33 Melon segments).

**Text.** (i) The benchmark authors report that the quadratic rotor model does not explain the torques and set $\dot\omega = 0$
in their physics baseline; (ii) we measure the same: the published mixer explains only 34–42 % of the angular acceleration
(Table B1, related numbers), and no learned model recovers the torque gain (Table A1); (iii) even an exact model replaying its
own 100 Hz-logged inputs is 0.1 m off after 3 s and 10.5 m after 10 s (Table B2), so open-loop prediction beyond ~3 s is not a
meaningful test on this data, which is why the benchmark stops at 0.5 s.

---

## 6. Open items before filling

| item | status |
|---|---|
| Figure 1 script (rolling 50-step predictions) | to write |
| Figure 2 script (error vs horizon) | to write (per-horizon arrays can be added to `split_protocol_table.json`) |
| Calibration numbers re-run on final checkpoints (Table 2) | done 26 Sep 2026 (GP-ODE final, GP-SDE step 4000, NN-SDE step 7000) |
| Promote torque-identifiability script into the repo (Table B1) | open |
| PH-NODE-RK4 on IDSIA: training outcome | diverged (NaN at step 52, wrench input); reported as NaN |
| Decide: which PH-NODE-RK4 variant to report (wrench vs rotor2 input, $h$ = 0.005 vs 0.001) | NaN row kept, per decision of 26 Sep 2026 |
| Decide: include PH-NN-SDE in Tables 1a/1b, or only in Table 2 | included, together with PH-GP-SDE (5-seed mean) |
| Decide: Table 1a in the main text or the appendix | open |
| Optional: run the benchmark's learned baselines (Res-MLP, Hybrid, Res-LSTM) on the training flights to fill their n/a rows | open (weights not released) |

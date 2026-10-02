# Open-loop metrics — what each row of the table means

Produced by `src/models/SE3_Quadrotor/comparision/open_loop.py` and written by
`generate_comparison_report_v2.py` to `open-loop/comparison.pdf` (one page per evaluation set) with the
images in `open-loop/images/`.

---

## 0. Setup that every row assumes

* **The experiment.** Each model is handed the *true* initial state $x_0$ and the *recorded* wrench $u(t)$ of a
  held-out flight and integrates $\dot x=f_\theta(x,u)$ forward with its own integrator (Lie-IMEX or RK4).
  Nothing corrects it, so errors compound: this is the test of the learned physics, not of a controller.
* **Evaluation sets.** D0 (10 flights, 240 Hz, $\Delta t=1/240$ s; one reference shape from ten starts) and the
  **HARD-V6 held-out split** (10 flights, 100 Hz, $\Delta t=0.01$ s) whose segment library — slalom, helix, chirp, bounce,
  tumble, plus dash and thrust-pulse — shares no shape with the V5 training library. Both are 10 s and the model is
  integrated at the set's own step. Any V5/V6-style pickle can be added with `--open-loop-dataset PATH@heldout|@test`.
* **PH-GT-LieIMEX.** The simulator's analytic operators in the same integrator (named with its integrator here because, unlike closed loop, it is integrated) — the attainable ceiling. It is not
  exact either (integrator and step differ from PyBullet), which is why it belongs in the table as the floor.
  Never ranked, never bold.
* **Mean vs samples.** The GP has a posterior $q(w)=\prod_i\mathcal N(m_i,s_i^2)$. Point rows score the
  **plug-in rollout** $\hat x^{\text{mean}}=\mathrm{Roll}(x_0,u;\,w=m)$ — the same weights the closed-loop
  controller used. Probabilistic rows score an **ensemble** $\hat x^{(i)}=\mathrm{Roll}(x_0,u;\,w^{(i)})$,
  $w^{(i)}\sim q(w)$, $i=1\dots S$ ($S=50$), through
  $$\mu_j(t)=\frac1S\sum_i\hat x^{(i)}_j(t),\qquad
    \sigma_j^2(t)=\frac1{S-1}\sum_i\big(\hat x^{(i)}_j(t)-\mu_j(t)\big)^2+\sigma_{\text{floor}}^2$$
  with $\sigma_{\text{floor}}=1$ mm, because every sample starts at the same $x_0$ and $\sigma(0)=0$ would
  make the likelihood infinite. The band is **epistemic only** (weights); the learned observation noise is
  not added, because the truth is clean. NN and PH-NODE are point estimates: their probabilistic rows read `—`.
* **Bold.** Best learned model in the row; PH-GT excluded; ties share; rows only one model can fill are not ranked.

Notation: $F$ flights, $T$ steps, $D=3$ position components; $p,R,v,\omega$ position, attitude, body velocity,
body rate; a hat for the model's prediction.

---

## 1. position RMSE (m)
$$\mathrm{RMSE}_p=\sqrt{\frac{1}{FT}\sum_{f,t}\big\|\hat p_f(t)-p_f(t)\big\|_2^2}$$
**Why.** The headline accuracy number — but over 10 s of free integration it is dominated by *when* a model
diverges, so it must be read together with §5.

## 2. attitude error, geodesic (deg)
$$\Theta=\sqrt{\frac{1}{FT}\sum_{f,t}\theta_f(t)^2},\qquad
  \theta=\arccos\!\left(\frac{\operatorname{tr}\big(R^\top\hat R\big)-1}{2}\right)$$
(both matrices projected onto $SO(3)$ first). **Why.** Chart-free rotation error; Euler angles distort near
$\pm90°$ pitch, which diverged rollouts reach.

## 3. velocity RMSE (m/s) and 4. angular-rate RMSE (rad/s)
$$\mathrm{RMSE}_v=\sqrt{\tfrac{1}{FT}\sum\|\hat v-v\|^2},\qquad \mathrm{RMSE}_\omega=\sqrt{\tfrac{1}{FT}\sum\|\hat\omega-\omega\|^2}$$
**Why.** These are the states the operators act on directly; a wrong $D_v$ or $M_1^{-1}$ appears here before
it has integrated into position.

## 5. valid prediction time, median (s)
$$\mathrm{VPT}_f=\min\Big\{t:\ \|\hat p_f(t)-p_f(t)\|_2>\epsilon_f\Big\},\qquad
  \epsilon_f=\sqrt{0.025}\cdot\sqrt{\frac1T\sum_t\|p_f(t)-p_f(0)\|_2^2}$$
reported as the median over flights, capped at the horizon. $\sqrt{0.025}=0.158$ is the Valid Prediction
Time convention of *Which priors matter?* (arXiv 2111.05458), square-rooted from an MSE threshold to a
distance; the second factor is the flight's own extent, so small and large flights are judged equally.
**Why.** The one number that says "how long can this model be trusted". A model can have a *lower* 10-s RMSE
and a *shorter* VPT than another (the NN vs the GP on D0), which is exactly the situation RMSE alone hides.

## 6. error growth rate, median (1/s)
$$\lambda_f=\text{slope of the least-squares fit of }\ \log\|\hat p_f(t)-p_f(t)\|\ \text{ against }t
  \ \text{ on } 0.05\,\text{s}<t\le\mathrm{VPT}_f$$
median over flights (NaN if the window is shorter than 10 samples). **Why.** *How fast* a model diverges,
independent of *when*; a Lyapunov-like exponent of the model error.

## 7. SO(3) violation
$$\frac{1}{FT}\sum_{f,t}\big\|\hat R_f(t)^\top\hat R_f(t)-I\big\|_F$$
**Why.** A Lie-group integrator keeps this at machine precision by construction; a vector-space integrator
(RK4) lets $\hat R$ drift off the manifold. A structural, not a fitted, difference.

## 8. rollout compute (ms per flight)
Median wall-clock time of the full 10-s rollout divided by $F$. **Why.** Pre-empts "the GP is too slow".

---

## 9. NLPD, position (nats) — GP only
$$\mathrm{NLPD}=\frac{1}{FTD}\sum_{f,t,j}\left[\frac{\big(p_{f,j}(t)-\mu_{f,j}(t)\big)^2}{2\sigma_{f,j}^2(t)}
  +\tfrac12\log\big(2\pi\sigma_{f,j}^2(t)\big)\right]$$
**Why.** The standard proper scoring rule for a probabilistic predictor: low only when the model is both
accurate *and* honest about how accurate it is.

## 10. coverage at 2σ (target 0.954) — GP only
$$\mathrm{Cov}_{95}=\frac{1}{FTD}\sum_{f,t,j}\mathbf 1\Big[\,|p_{f,j}(t)-\mu_{f,j}(t)|\le 2\sigma_{f,j}(t)\Big]$$
Best = closest to $0.954$; below it the band is over-confident, above it under-confident. The same at
nominal 50 / 80 / 90 / 95.4 / 99 % is the reliability diagram in `_calibration_<model>.png`.

## 11. sharpness, mean 2σ (m) — GP only
$$\overline{2\sigma}=\frac{1}{FTD}\sum_{f,t,j}2\sigma_{f,j}(t)$$
**Why.** Coverage alone is gameable by an enormous band; coverage *and* sharpness together are not.

## 12. Spearman ρ(σ, error) — GP only
$$\rho=\mathrm{Spearman}\Big(\ \|\sigma_f(t)\|_2\ ,\ \|\hat p_f(t)-p_f(t)\|_2\ \Big)_{f,\,t>0}$$
**Why.** "Does it know when it is wrong?" — a value near 1 means the band widens exactly where the error grows,
which is what makes a variance *useful* rather than merely present.

## 13. ensemble VPT, median [Q1, Q3] (s) — GP only
$\mathrm{VPT}$ of every sampled rollout against the truth, pooled over $S\times F$, reported as median with the
inter-quartile range. **Why.** The GP's own estimate of how long it can be trusted, with a spread.

## 14. plug-in vs predictive-mean gap (m) — GP only
$$\delta=\sqrt{\frac{1}{FT}\sum_{f,t}\big\|\hat p^{\text{mean}}_f(t)-\mu_f(t)\big\|_2^2}$$
**Why.** The Jensen gap between $\mathrm{Roll}(\mathbb E[w])$ and $\mathbb E[\mathrm{Roll}(w)]$. Small means the
choice of point prediction is immaterial; large means the posterior is wide enough that the mean weights are
not representative and the band is the story. Not ranked.

---

## 15. Images

| file | content |
|---|---|
| `<set>_states_flight<k>.png` | one image per flight $k$; one row per model × ($x,y,z$, roll, pitch, yaw, $\|v\|$, $\|\omega\|$); prediction (blue) vs truth (dashed); the GP row carries the $\pm2\sigma$ band |
| `<set>_trajectory_flight<k>.png` | one image per flight; one 3-D panel per model, prediction (blue) vs truth (dashed), each panel scaled to its own flight with the span in the sub-title; the GP panel also shows 20 posterior sample paths faintly |
| `<set>_error.png` | mean-over-flights position error against time on a log axis, every model, with the median VPT threshold |
| `<set>_calibration_<model>.png` | GP only: reliability diagram, band width vs realised error over time, $\sigma$ vs error scatter with $\rho$ |

## 16. Where these are computed

| quantity | code |
|---|---|
| set loading, step size | `open_loop.load_open_loop_set` |
| rollouts and timing | `open_loop.timed_rollout`, `open_loop.posterior_rollouts` |
| §1–14 | `open_loop.open_loop_metrics`, `valid_prediction_times`, `growth_rates` |
| table and bolding | `open_loop.open_loop_table` (via `report_figures_v2._table`) |
| images | `open_loop.states_grid`, `error_figure`, `calibration_figure` |
| constants | `VALID_PREDICTION_FRACTION = 0.158`, `SIGMA_FLOOR_M = 1e-3`, `OPEN_LOOP_SAMPLES = 50` |

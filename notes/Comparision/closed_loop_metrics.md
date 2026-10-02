# Closed-loop metrics — what each row of the table means

This note documents every row of the closed-loop tables produced by
`src/models/SE3_Quadrotor/comparision/generate_comparison_report_v2.py`
(`closed-loop/comparison.pdf`, one page per reference shape).

---

## 0. Setup that every row assumes

* **Plant.** Contact-free PyBullet CF2P, 240 Hz, nonlinear damping $c=0.5$; no ground plane, no obstacles.
* **Controller.** The reference energy-based SE(3) law, identical gains for every column
  ($K_p=[10,10,50]$, $K_v=[3,3,3]$, $K_R=[250,250,250]$, $K_\omega=[20,20,20]$, tilt limit $40°$),
  with the damping feedforward $+D_v v_b$, $+D_\omega \omega_b$ ON for every model.
* **What differs between columns.** Only the six learned operators
  $M_1^{-1}, M_2^{-1}, V, g, D_v, D_\omega$. No integrator is used in the loop: the model is queried
  pointwise at the current state, and the plant does all the integration.
* **PH-GT.** A column whose operators are the simulator's own constants
  ($M_1^{-1}=I/m$, $M_2^{-1}=J^{-1}$, $V=mgz$, $g$ = selection matrix, $D_v=mc(1+\|v\|)I$, $D_\omega=c(1+\|\omega\|)J$).
  It is the attainable ceiling, **never ranked**, and never printed in bold.
* **Recorded references.** Besides the analytic shapes, any evaluation-set flight can be flown as a reference
  (`--controller-recorded-reference PATH[@split]`): position, world velocity and heading are interpolated from the
  recorded flight, the feedforward acceleration and yaw rate are smoothed finite differences (50 ms boxcar), the
  heading is taken relative to the flight's own start, and the shape holds its final point after the flight ends.
  These are the open-loop evaluation shapes flown under the controller, so open- and closed-loop tables share shapes.
* **Start state.** The reference is anchored to the vehicle's measured state, so
  $p_{\text{ref}}(0)=p(0)$ and $\psi_{\text{ref}}(0)=\psi(0)=0$ exactly; no initial-error transient pollutes any row.
* **Horizon.** A flight that ends early is scored on what it actually flew; $N$ below is the number of
  recorded steps of that flight and $\Delta t = 1/240$ s.
* **Bold.** The best *learned* model in that row: smallest for error rows, largest for valid tracking time
  and completion, closest to PH-GT where the target is physical rather than "small". Ties share the bold.

Notation used throughout: $p(t)\in\mathbb{R}^3$ world position, $v(t)$ world velocity,
$R(t)\in SO(3)$ body attitude, $\omega(t)$ body angular rate, $u(t)=[T,\tau_x,\tau_y,\tau_z]$ the applied wrench,
$m$ mass, $g$ gravity, $L$ arm length, and a subscript $\text{ref}$ for the commanded reference.

---

## 1. position RMSE (m)

Average tracking error over the flight.

$$\mathrm{RMSE}_p=\sqrt{\frac{1}{N}\sum_{k=1}^{N}\big\|p(t_k)-p_{\text{ref}}(t_k)\big\|_2^2}$$

* $p(t_k)$ — measured position at step $k$; $p_{\text{ref}}(t_k)$ — commanded position; $N$ — steps flown.

**Why it helps.** The headline number: one scalar for "how well did it fly the shape". It is, however,
dominated by whatever the reference demands that is infeasible (see §8), which is why it never appears alone.

## 2. position max error (m)

The worst instantaneous excursion.

$$e_{\max}=\max_{k\le N}\big\|p(t_k)-p_{\text{ref}}(t_k)\big\|_2$$

**Why it helps.** RMSE hides a single large departure; $e_{\max}$ is what decides whether the vehicle would
have hit something. A model with a good RMSE and a 4 m max error is not safe.

## 3. position final error (m)

Where the flight ends relative to where it was told to be.

$$e_{\text{final}}=\big\|p(t_N)-p_{\text{ref}}(t_N)\big\|_2$$

**Why it helps.** The references hold their final point after 15 s, so this is the steady-state error of the
closed loop. It separates "transiently bad" from "never converged".

## 4. velocity RMSE (m/s)

$$\mathrm{RMSE}_v=\sqrt{\frac{1}{N}\sum_{k=1}^{N}\big\|v(t_k)-v_{\text{ref}}(t_k)\big\|_2^2}$$

**Why it helps.** Velocity error leads position error. A wrong mass or damping estimate shows up here first,
before the position integral has had time to accumulate.

## 5. roll RMSE / 6. pitch RMSE vs the flat reference (deg)

The quadrotor is underactuated: the reference commands position and yaw only, and the tilt is *implied* by
the acceleration it demands (differential flatness), computed with the **true** constants:

$$b_3^{\text{ref}}=\frac{m\big(\ddot p_{\text{ref}}+g e_3\big)+m\,c\,(1+\|v_{\text{ref}}\|)\,v_{\text{ref}}}
{\big\|m(\ddot p_{\text{ref}}+g e_3)+m c (1+\|v_{\text{ref}}\|)v_{\text{ref}}\big\|}$$

$$\phi_{\text{ref}}=\operatorname{atan2}\!\big(b_{3,y}^{\text{ref}}\cos\psi_{\text{ref}}-b_{3,x}^{\text{ref}}\sin\psi_{\text{ref}},\;b_{3,z}^{\text{ref}}\big),
\qquad
\theta_{\text{ref}}=\operatorname{atan2}\!\big(b_{3,x}^{\text{ref}}\cos\psi_{\text{ref}}+b_{3,y}^{\text{ref}}\sin\psi_{\text{ref}},\;b_{3,z}^{\text{ref}}\big)$$

$$\mathrm{RMSE}_\phi=\sqrt{\frac{1}{N}\sum_k \mathrm{wrap}\big(\phi(t_k)-\phi_{\text{ref}}(t_k)\big)^2},
\qquad \text{likewise for }\theta$$

* $b_3^{\text{ref}}$ — thrust direction the reference requires; $e_3=[0,0,1]$; $c=0.5$ the plant's damping
  coefficient; $\phi,\theta$ — measured roll and pitch (ZYX Euler) from $R(t)$;
  $\mathrm{wrap}(\cdot)$ maps to $(-\pi,\pi]$.

**Why it helps.** These are the only attitude quantities with a *model-independent* target, so they are
comparable across columns. **They carry a floor**: a real controller must lead or lag the flat attitude to
correct position error, so even PH-GT scores non-zero (e.g. $29.8°$ roll on `spiral_k1p5`). Read them
against the PH-GT column, never absolutely — that is exactly why PH-GT is in the table.

## 7. yaw RMSE (deg)

Yaw is commanded directly, so it needs no construction:

$$\mathrm{RMSE}_\psi=\sqrt{\frac{1}{N}\sum_k \Big(\arg e^{\,i(\psi(t_k)-\psi_{\text{ref}}(t_k))}\Big)^{2}}$$

* the $\arg e^{i(\cdot)}$ form wraps the difference to $(-\pi,\pi]$ so a $359°$ error reads as $-1°$.

**Why it helps.** It is the one rotational degree of freedom the reference actually specifies, and after the
start-state alignment it measures tracking rather than an initial slew (GP $0.04°$ vs NN $0.19°$, NODE $0.22°$,
PH-GT $0.02°$ on `spiral_k1p5`).

## 8. valid tracking time vs PH-GT (s)

How long the flight stays with the flight the exact-physics controller flew:

$$t_{\text{valid}}=\min\Big\{t_k:\ \big\|p(t_k)-p_{\text{GT}}(t_k)\big\|_2>\epsilon\Big\},
\qquad \epsilon=\sqrt{0.025}\cdot\sqrt{\frac{1}{N}\sum_k\big\|p_{\text{ref}}(t_k)-p_{\text{ref}}(0)\big\|_2^2}$$

capped at the horizon (a flight that never exceeds $\epsilon$ scores the full 20 s).

* $p_{\text{GT}}$ — PH-GT's trajectory on the same shape; the second factor is the RMS **extent** of the shape,
  so the criterion is equally strict on a small spiral and a long dash;
  $\sqrt{0.025}=0.158$ is the Valid Prediction Time convention of *Which priors matter?* (arXiv 2111.05458),
  square-rooted to turn a normalised-MSE threshold into a distance.

Resulting thresholds: spirals $0.126$ m, lissajous $0.221$ m, stops $0.373$–$0.383$ m.

**Why it helps.** It is the sharpest single number in the table, because it compares against what is
*physically attainable* rather than against a command that may be infeasible. On `spiral_k1p5` the GP holds
$20.00$ s while the baselines leave after $1.6$–$1.7$ s — a separation the position RMSE row
($0.094$ / $0.100$ / $0.117$ m) completely hides. The same metric measured against the *reference* is useless:
every model, PH-GT included, breaks the threshold at the same instant, because that measures the command.

## 9. RMS thrust / hover

$$\overline{T}=\sqrt{\frac{1}{N}\sum_k\Big(\frac{T(t_k)}{m g}\Big)^2}$$

* $T$ — collective thrust actually applied; $mg$ — hover thrust. A value of 1 means the flight cost exactly
  hover thrust on average.

**Why it helps.** Control effort exposes a wrong force scale $\mu g_f$ *before* it becomes a position error:
a model that thinks the vehicle is heavier than it is over-thrusts every step. Best is the value closest to
PH-GT's, not the smallest — under-thrusting is as wrong as over-thrusting.

## 10. control chattering (1/s)

Step-to-step roughness of the commanded wrench, on a normalised scale:

$$\tilde u_k=\Big(\frac{T_k}{mg},\ \frac{\tau_{x,k}}{mgL},\ \frac{\tau_{y,k}}{mgL},\ \frac{\tau_{z,k}}{mgL}\Big),
\qquad
\mathrm{chatter}=\frac{1}{\Delta t}\sqrt{\frac{1}{N-1}\sum_{k=1}^{N-1}\big\|\tilde u_{k+1}-\tilde u_k\big\|_2^2}$$

* $L$ — arm length, so thrust and torque are dimensionless and comparable before differencing;
  $\Delta t=1/240$ s.

**Why it helps.** Limit cycles and actuator-destroying oscillation do not always show in RMSE. This row
quantifies the visible buzzing of the PH-NODE ($44.8$ /s on `stop_v2` against PH-GT's $2.4$ /s).

## 11. max tilt (deg)

$$\alpha_{\max}=\max_{k\le N}\arccos\big(b_3(t_k)\cdot e_3\big),\qquad b_3=R(t_k)e_3$$

* $b_3$ — the body thrust axis in world coordinates.

**Why it helps.** A single number for "how close did it come to flipping". Above $90°$ the vehicle is past
horizontal and thrust points downward — a tumble, not a tracking error.

## 12. motor saturation fraction

$$s=\frac{1}{4N}\sum_{k=1}^{N}\sum_{i=1}^{4}\mathbf{1}\big[\text{rpm}^{\text{req}}_{k,i}>\text{RPM}_{\max}\big]$$

* $\text{rpm}^{\text{req}}_{k,i}$ — rotor speed the controller asked for; $\text{RPM}_{\max}$ — the plant's limit.

**Why it helps.** Non-zero saturation means the tracking numbers are partly hardware-limited rather than
model-limited: the model is demanding thrust that does not exist. On `stop_v2` it reads $0.211$ for the GP
against $0.885$ for the NN, which is *why* the NN diverges.

## 13. completed / requested (s) and 14. completion fraction

$$\text{completion}=\min\!\Big(\frac{t_N}{t_{\text{requested}}},\,1\Big)$$

**Why it helps.** Failure-aware bookkeeping: a model that crashed at 3 s of a 20 s flight is scored on those
3 s, so its RMSE looks deceptively small. This row is what stops that from being read as success.

## 15. controller failure

Text: `none`, or the failure kind and the time it occurred (ground contact, PyBullet exception, non-finite command).

**Why it helps.** States plainly whether the flight ended by itself.

## 16. controller time p95 (ms)

The 95th percentile of the wall-clock time of one control computation.

$$t_{95}=\mathrm{percentile}_{95}\big(\{\Delta t^{\text{compute}}_k\}_{k=1}^{N}\big)$$

**Why it helps.** Pre-empts the "a GP with 220 random features cannot run in real time" objection: every
column is $2.5$–$3.2$ ms against the $4.17$ ms budget of a 240 Hz loop.

---

## 17. The vs-PH-GT block (per-model pages)

These rows appear on `closed-loop/<shape>/<model>.pdf` (written only with `--closed-loop-subfolders`).
All of them have a **zero floor**: PH-GT scores 0 against itself by construction, so unlike §5–6 they can be
read absolutely.

$$\mathrm{RMSE}^{\text{GT}}_p=\sqrt{\frac1N\sum_k\big\|p(t_k)-p_{\text{GT}}(t_k)\big\|^2},\qquad
\mathrm{RMSE}^{\text{GT}}_v=\sqrt{\frac1N\sum_k\big\|v(t_k)-v_{\text{GT}}(t_k)\big\|^2}$$

$$\Theta^{\text{GT}}=\sqrt{\frac1N\sum_k\theta_k^2},\qquad
\theta_k=\arccos\!\left(\frac{\operatorname{tr}\big(R(t_k)^\top R_{\text{GT}}(t_k)\big)-1}{2}\right)$$

$$\text{gap}=100\cdot\frac{\mathrm{RMSE}_p-\mathrm{RMSE}_p^{\text{PH-GT}}}{\mathrm{RMSE}_p^{\text{PH-GT}}}\ [\%]$$

* $\theta_k$ — the geodesic (chart-free) angle between the two attitudes, in radians before conversion to
  degrees; roll/pitch versions use the same wrapped-Euler difference as §5–6 but against $R_{\text{GT}}$.

**Why they help.** They answer the question the paper actually asks — *how close is the learned model to the
exact-physics controller?* — instead of *how close is it to a command nobody can follow*. The geodesic
attitude row is the safe companion to the Euler roll/pitch rows, which distort near $\theta=\pm90°$
(a tumbling flight).

---

## 18. Where these are computed

| quantity | code |
|---|---|
| §1–4, 12–15 | `report_controller.controller_comparison` |
| §5–7, 9–11, 16 | `report_controller.controller_comparison` (extras block) and `flat_reference_angles` |
| §8 and §17 | `generate_comparison_report_v2.generate`, in the loop that redraws the plots |
| threshold constant | `report_controller.VALID_TRACKING_FRACTION = 0.158` |
| table layout and bolding | `report_figures_v2.controller_table` |

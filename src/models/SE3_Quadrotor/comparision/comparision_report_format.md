# Comparison report format

Specification for the multi-model comparison PDF produced by
`src/models/SE3_Quadrotor/comparision/generate_comparison_report.py`.

One page type per section. Pages 2 onward are to be defined.

---

## Page 1 — Gauge-invariant physics identification (relative RMS error)

**Purpose.** The headline page. It answers one question: *which model recovered the
physics?* Everything a reader needs to rank the methods is on this page, and no other
page is required to interpret it.

### What goes on the page

A single table. Five rows, one per gauge-invariant product. One column per trained
model, plus a column for the simulator's analytic operators as the floor.

| product | symbol | truth | units |
| --- | --- | --- | --- |
| thrust gain | $\mu\,g_f$ | $(0,\,0,\,1/m)$ | m/s² per N |
| torque gain | $M_2^{-1} g_\tau$ | $J^{-1}$ | rad/s² per N·m |
| gravity | $\mu\,\nabla V$ | $(0,\,0,\,g)$ | m/s² |
| translational damping | $\mu\,D_v$ | $c\,(1+\lVert v_b\rVert)\,I$ | 1/s |
| rotational damping | $M_2^{-1} D_\omega$ | $c\,(1+\lVert \omega_b\rVert)\,I$ | 1/s |

Every cell holds **one number: relative RMS error in percent.** Nothing else.

### Layout

```
        Gauge-invariant physics identification
     <evaluation flight>, horizon <T> s, relative RMS error (%)

                          |  ours  |  base-1  |  base-2  |  GT ops
  thrust gain    mu g_f   |        |          |          |
  torque gain    M2^-1 g  |        |          |          |
  gravity        mu grad V|        |          |          |
  transl. damping mu Dv   |        |          |          |
  rot. damping   M2^-1 Dw |        |          |          |

  0 = exact.  100 = as bad as predicting zero.  >100 = worse than no model.
```

The footer line is part of the page, not optional. It is what lets a reader
interpret a cell without reading the method section.

### The metric

$$\text{rel RMS} \;=\; 100 \times
\sqrt{\frac{\overline{\lVert P - P^\star\rVert^2}}{\overline{\lVert P^\star\rVert^2}}}
\;=\; 100\sqrt{\mathrm{NMSE}}$$

- $P$ is the model's product, $P^\star$ the analytic truth, both evaluated at the same
  state at the same instant along the shared evaluation flight.
- The bar averages over **every time step and every entry** of the vector or matrix.
  For the three $3\times3$ products that means all nine entries, so spurious
  off-diagonal coupling is counted as error.
- Computed in `report_evaluation.product_metrics`; the products themselves come from
  `report_evaluation.gauge_invariant_products`.

### Why these five quantities

The model never observes momenta, so two positive scalars $\beta_v, \beta_\omega$ can
rescale the learned operators without changing any trajectory:

$$\mu \to \beta_v\mu,\quad V \to V/\beta_v,\quad D_v \to D_v/\beta_v,\quad g_f \to g_f/\beta_v$$
$$M_2^{-1} \to \beta_\omega M_2^{-1},\quad D_\omega \to D_\omega/\beta_\omega,\quad g_\tau \to g_\tau/\beta_\omega$$

Each of the five products pairs an inverse mass with a factor carrying the opposite
power of the same $\beta$, so the product cancels **exactly**. These are the only
combinations the data determines. Reporting $M_1^{-1}$ or $D_v$ on their own would be
reporting where the optimizer happened to land, not physics.

Consequence for the page: **no scale fit is applied and none is needed.** $\beta_v$ and
$\beta_\omega$ are never estimated. The page therefore consults the ground truth only to
score the models, never to define the comparison — a reviewer cannot object that the
alignment was chosen to flatter the method.

### Why relative RMS and not the alternatives

- **Not per-axis means.** Undefined on the zero-truth components of thrust gain and
  gravity; hides time variation, because a model 40 % high half the time and 40 % low the
  rest averages to zero error; blind to off-diagonals; and triples the table size.
- **Not MSE.** Its units differ in every row, so the rows cannot share a column. It ranks
  by the magnitude of the physics rather than by fit quality — on one of our runs it calls
  the torque gain the worst operator at $6.9\times10^5$ and the rotational damping the best
  at $0.057$, while the relative RMS values are 3.7 % and 54.6 %, the opposite order. It is
  also not comparable across evaluation flights, since the true damping scales with speed.
- **Not NMSE alongside it.** NMSE is the same number squared. Report one. Relative RMS
  reads faster and carries the interpretable 100 % threshold.

### Rules

1. One evaluation flight per report, named in the page title, held out from training.
2. Every model is rolled out in **the integrator it was trained with**. State that in the
   caption when the models differ.
3. Percentages to one decimal place.
4. Bold the best cell in each row.
5. The analytic-operator column is the achievable floor, not a competitor. Keep it last.

### Reporting note for the paper

Quote per-axis values in prose for gravity and thrust gain only, where the truth is a
known constant and a physical number is what a reader wants (for example, learned gravity
$8.67$ against a true $9.81$ m/s²). Keep them off this page.

---

## Pages 2-6 — One page per gauge-invariant product

One page per row of the page-1 table, in the same order, so a reader following the table
downward finds the pages in the same sequence.

| page | product | panels |
| --- | --- | --- |
| 2 | thrust gain $\mu\,g_f$ | 3 x 1, the x / y / z components |
| 3 | torque gain $M_2^{-1} g_\tau$ | 3 x 3, all nine matrix entries |
| 4 | gravity $\mu\,\nabla V$ | 3 x 1, the x / y / z components |
| 5 | translational damping $\mu\,D_v$ | 3 x 3, all nine matrix entries |
| 6 | rotational damping $M_2^{-1} D_\omega$ | 3 x 3, all nine matrix entries |

**Purpose.** Page 1 says *how much* each model is wrong. These pages say *how*. A constant
offset, a drift, a failure to track the speed dependence and an invented cross-axis
coupling all score as one number on page 1 and look completely different here.

They must work without the reader consulting page 1, because one of them will be lifted
into the paper on its own.

### What is drawn in every panel

- **x-axis is time**, the full evaluation horizon, for all five pages.
- **Ground truth**, the analytic operator of the simulator, evaluated at the same state at
  the same instant.
- **Every model**, and for each model **all 10 evaluation trajectories**, not just one.
- **Posterior uncertainty** for any model that has it.

### Styling rule

This is the rule that makes the page readable, and it is not negotiable:

> **Colour and line style encode the model, never the trajectory.**
> All 10 trajectories of one model share that model's exact colour and line style.
> Ground truth is black dotted, for all 10 of its curves alike.

Each model therefore appears as a *bundle* of 10 like-coloured curves. The width of a
bundle is the spread across initial conditions, which is information: a model whose
operators are genuinely state-independent draws a tight bundle, one that has absorbed
trajectory-specific error draws a wide one. Ten separate colours per model would destroy
this and make the page unreadable.

Draw the trajectory curves at reduced opacity so overlapping bundles stay distinguishable,
and keep the ground-truth bundle at full opacity on top.

### Posterior variance bands

Required on every one of these five pages, for each model that is Bayesian.

- The band is $\mu \pm 2\sigma$ across independent posterior weight samples, the same
  sample count used elsewhere in the report.
- Shade it in the model's own colour, so band and curves are visibly the same model.
- Point-estimate baselines get no band. That absence is the honest visual and should not
  be filled in with anything.
- Evaluating a product under a sampled weight set is a pointwise query with **no
  integration**, so these bands cost far less than the state-page bands and there is no
  reason to omit them.

One judgement call, flagged so it can be revisited: drawing 10 bands on top of 10 bundles
is unreadable, so the band is drawn for **trajectory 0 only** and the legend says so. The
bundle already shows spread across trajectories; the band shows spread across weights.
They are different quantities and should not be visually merged.

### Per-page annotations

1. **Title**: product name, its analytic truth, the evaluation flight, the horizon, and
   the trajectory count.
2. **Units** on every axis label, taken from the page-1 table.
3. **Figure-level legend**, placed outside the panels. It must not sit on top of the data
   in the first panel.
4. **Off-diagonal panels** (matrix pages only) carry one annotation each: the peak value as
   a percentage of the mean diagonal. Panels autoscale independently, so an off-diagonal
   axis can span twenty times less than the diagonal beside it; without this number a
   reader will misjudge the leakage as comparable to the diagonal error.
5. **Corner box** repeating the relative RMS error per model for this product, so the
   figure stands alone.

### Reading guide

| what you see | what it means |
| --- | --- |
| bundle offset from truth, flat | constant bias in the operator |
| bundle tracks shape but wrong amplitude | the functional form is right, the scale is not |
| bundle ignores the truth's variation | the model learned a constant where physics varies |
| non-zero curve in an off-diagonal panel | invented cross-axis coupling, truth is exactly 0 |
| wide bundle | operator error depends on the trajectory |
| wide band | the model is declaring low confidence there |

---

## Page 6a — The paper view of the damping laws  (inserted after page 6)

Pages 2-6 are the diagnostic view: every product against time, every flight drawn, one band per
flight. That view is right for finding faults — it is what exposes the pose-dependent drift of the
$g$ residual at $\sigma=0.1$ — but for the two damping products time is the wrong axis: they depend
on the state, not on the clock. This page re-plots them against the state they actually depend on.

Two panels: $\mu D_v$ against
$\|v_b\|$ and $M_2^{-1}D_\omega$ against $\|\omega_b\|$. Every flight and every instant is pooled
into equal-count quantile bins, which is the legitimate pooling here: two flights at the same instant
sit at different speeds and therefore have different *true* damping, so averaging them at fixed time
would plot the diversity of the test set and call it uncertainty; averaging them at fixed speed does
not. The truth curve is $c(1+\|\cdot\|)$.

Three spreads are kept visually distinct, because a reader who conflates them draws the wrong
conclusion:

| element | meaning |
| --- | --- |
| shaded fill, $\pm1\sigma$ | spread of the model inside the bin: its anisotropy plus its error |
| dotted envelope, $\pm2\sigma$ | spread across posterior weight samples: the epistemic band |
| grey step, right axis | how many samples fall in each bin |

The bin counts matter: most flight time is spent slow, and the low-speed bins are exactly where the
models over-damp, so a figure without the counts overstates how much of the flight is well modelled.

## Page 6b — How the relative RMS error is computed  (closing page of the identification booklet)

A text-and-formula page, no plot. It closes the identification booklet by defining the one number
the whole booklet is scored on, so the file can be read without the reader already knowing the
metric.

1. **What is compared** — the learned product against the analytic truth, at every time step of
   every held-out flight, with no rescaling, because the five products are gauge-free.
2. **The formula**, one number per flight $k$:

$$\mathrm{relRMS}_k = 100\sqrt{\frac{\frac{1}{NC}\sum_t\sum_c (P_{tc}-P^\star_{tc})^2}{\frac{1}{NC}\sum_t\sum_c (P^\star_{tc})^2}}$$

   with $t$ over the $N$ time steps and $c$ over the $C$ components — 3 for a vector product, 9 for
   a matrix one.
3. **The same thing in words** — four bullets: square the difference and average; do the same to the
   truth alone; divide so the units cancel; square-root and multiply by 100.
4. **How to read it** — 0 exact, 100 as bad as predicting zero, above 100 worse than no model.
5. **A worked example**, computed live from the report's own numbers rather than hard-coded: the
   reference model's thrust gain, where only the $z$ entry is non-zero so the $\frac{1}{NC}$
   cancels and the formula reduces to $100\,|P_z-P^\star_z|/|P^\star_z|$. The small residual
   between the hand figure and the table figure is the drift of the learned gain over the flight,
   which the table counts as error and the single averaged value hides.

It closes by restating what the $\pm$ column and the "vs" row on page 1 mean.

## Page 7 — Training curves, all components on one axis

**Purpose.** Show how every model got to where page 1 scores it, on a single axis, so the
reader can compare convergence behaviour without flipping between pages.

### What is drawn

One plot. One page. Five loss components per model, drawn together:

| component | stats key | what it is |
| --- | --- | --- |
| total train loss | `train_loss` | the objective actually minimised |
| position loss | `train_position` | mean squared position error over the window |
| attitude geodesic | `train_attitude` | squared geodesic distance on $SO(3)$ |
| linear-velocity loss | `train_linear_velocity` | mean squared body-velocity error |
| angular-velocity loss | `train_angular_velocity` | mean squared body-rate error |

The last four are the complete decomposition of the window state error, one per state
block, so together they account for everything the total is built from.

### Styling rule

> **Colour encodes the model. Line style encodes the loss component.**

With three models and five components that is 15 lines on one axis. Colour splits them
into three bundles of five; line style separates the five inside each bundle. This is the
densest page in the report, so keep the line width thin and rely on the log axis to spread
the components vertically.

**This reassigns line style relative to pages 2-6**, where colour and style both encode the
model. That is deliberate: there is no trajectory dimension on this page, so style is free.
State it in the caption so a reader carrying over the convention from the plot pages is not
misled.

Use **two separate legends**, not one combined list of 15 entries:

```
  colour:  PH-GP-LieIMEX | PH-NN-LieIMEX | PH-NODE-RK4
  style:   ---- total  | --- position  | -.- attitude  | ..- lin. vel.  | ... ang. vel.
```

Five distinguishable line styles is the practical limit. If they stop being separable at
print size, split the page in two, total plus position and attitude on one, the two
velocity blocks on the other, rather than dropping a component.

### Axes

- **y-axis logarithmic.** The components span orders of magnitude, with total loss near
  $10^0$ and position loss near $10^{-3}$. On a linear axis everything except the total
  collapses onto the zero line.
- **x-axis is the optimizer step of the final curriculum stage**, $K = 100$, starting at 0.
  Earlier stages restart the optimizer and the step counter, so they are not concatenated.
  Say which stage in the title.
- Curves are smoothed with a moving average before plotting; the window goes in the caption.

### What these curves are, and are not

**Every curve on this page is an MSE-type quantity, for every model including the GP.**
`pose_loss_components` in `losses.py` is byte-identical across `ph_gp_lie_imex`,
`ph_nn_lie_imex` and `ph_node`, and all three trainers record its output into the same
statistics keys. The plotted total is

$$\text{total} = \overline{\lVert \Delta x\rVert^2} + \overline{\lVert \Delta v\rVert^2}
+ \overline{\lVert \Delta\omega\rVert^2} + \overline{\theta_{\text{geo}}^2}$$

an unweighted sum of the four component curves. **The negative log-likelihood is never
plotted.** So all five curves *are* comparable between models; they are a common yardstick,
not each model's own objective.

The real caveat is close to the opposite of what it looks like:

> For the neural baselines the plotted total **is** what the optimizer minimises.
> For our model it is **not**. The GP objective is
> $\text{NLL} + \beta\,\mathrm{KL} + \text{penalties}$, recorded separately under
> `objective`, while the plotted total is a diagnostic computed alongside it.

Consequences, to go in a footer line on the page:

1. The GP curve can flatten or tick upward while its training is going fine, because the
   quantity being minimised is not the quantity being drawn.
2. The baselines are scored here on exactly the loss they were tuned to minimise, which
   flatters them on this page relative to a model optimising something else.
3. Ranking therefore still belongs on page 1, which is measured on held-out physics rather
   than on any training window.

---

## Page 8 — Test-window curves, all components on one axis

Identical in construction to page 7. Same single axis, same five components, same log
y-scale, same two-legend layout, and the same styling rule:

> **Colour encodes the model. Line style encodes the loss component.**

Only the source series and three details change.

### What is drawn

| component | stats key | what it is |
| --- | --- | --- |
| total test loss | `test_total` | the objective evaluated on held-out windows |
| position loss | `position` | mean squared position error |
| attitude geodesic | `attitude` | squared geodesic distance on $SO(3)$ |
| linear-velocity loss | `linear_velocity` | mean squared body-velocity error |
| angular-velocity loss | `angular_velocity` | mean squared body-rate error |

The test series carry no `train_` prefix in the recorded statistics. The x-axis is the
evaluation step grid, `step`, which is sampled at the checkpoint interval rather than
every optimizer step.

### The three differences from page 7

1. **No smoothing.** Train curves are smoothed because they are recorded every step and are
   noisy. Test curves are recorded only at checkpoints, so they are already sparse and are
   plotted raw. Say "unsmoothed" in the caption, since page 7 says otherwise.
2. **The targets are clean.** The dataset marks `test_split_clean: true`, with the noise
   applied to the training split only, under a separate seed, before windowing. So these
   curves measure error against the true states while page 7 measures error against noisy
   observations.
3. **This is the page that means something.** Because of point 2, a model can drive page 7
   down by fitting observation noise, and page 8 will not follow. Where the two pages
   disagree is the useful signal.

### The caveat, restated

As on page 7, **every curve here is MSE, for every model including the GP**, produced by
the same `pose_loss_components`. The negative log-likelihood is not plotted anywhere in
this report. All five curves are comparable across models.

What is asymmetric is whose objective is being drawn: the baselines are scored on the loss
they minimise, our model is not. Repeat that footer line here; do not assume the reader saw
it on the previous page.
Repeat the footer line; do not assume the reader saw it on the previous page.

### Reading guide

| what you see | what it means |
| --- | --- |
| test flat while train falls | the model is fitting observation noise, not dynamics |
| test rising late in training | over-fitting; the earlier checkpoint was better |
| test and train falling together | genuine learning |
| one component flat, others falling | that state block is unidentifiable under this loss |
| large train/test gap for one model only | that model is the noise-sensitive one |

Note that a good curve here still does not settle the comparison. Both loss pages are
measured on one-second windows started from a known state. Page 1 is measured on held-out
physics over a full flight, and the two have repeatedly disagreed in this project.

---

## Page 9 — GP-only objective terms

**Purpose.** Pages 7 and 8 draw the common MSE yardstick, which is deliberately *not* what
our model minimises. This page draws what it actually minimises, and the terms that make it
up. It is the only page in the report that is method-specific.

**Emitted only when a GP model is in the report.** With more than one GP variant, repeat
the page once per variant rather than overlaying them.

### Styling rule

> **Colour encodes the loss term.** There is only one model on this page, so colour is free
> again.

Say so in the caption. Pages 2-8 use colour for the model, and a reader arriving here with
that convention will otherwise misread the page.

### Panel A — the objective and its additive terms

| series | stats key | grid | note |
| --- | --- | --- | --- |
| objective | `objective` | eval | what the optimizer actually minimises |
| NLL, train | `train_nll_total` | per-step | the data-fit term, on its own |
| NLL, held out | `test_nll_total` | eval | same term on the test windows |
| KL contribution | `train_kl_contribution` | per-step | $\beta\,\mathrm{KL}/N$, exactly as it enters |
| latent-state KL | `train_latent_kl` | per-step | MAP latent initial state |
| gravity penalty | `train_gravity_penalty` | per-step | omit when its weight is 0 |
| actuation penalty | `train_actuation_penalty` | per-step | omit when its weight is 0 |
| continuity | `train_continuity` | per-step | omit when its weight is 0 |

Every term above is now recorded separately, so this panel is a true decomposition: the
plotted series sum to the objective.

The objective is

$$\mathcal{L} = s\cdot\mathrm{NLL} + \frac{\beta\,\mathrm{KL}}{N}
+ w_{\text{lat}}\frac{\mathrm{KL}_{x_0}}{K} + w_g\,\mathcal{P}_g + w_a\,\mathcal{P}_a
+ w_c\,\mathcal{C}$$

so the panel shows a total and the pieces that sum to it.

**Use a symmetric-log y-axis.** Both the objective and the latent KL go negative; a real
run ends near $-9.1$ on the objective and $-21$ on the latent KL. A plain log axis drops
them silently.

**Draw inactive terms as a labelled flat line at zero, not as nothing.** In the no-prior
runs all three penalties are identically zero, and a reader must be able to see that the
priors were off rather than guess that the curve was forgotten.

### Panel B — KL and its annealing schedule

- Raw weight-space KL on a log axis, left: `train_weight_kl` per step, or `kl` on the
  evaluation grid. It sits near $4\times10^{4}$ and moves little.
- The annealing coefficient on a right-hand twin axis, linear, from `train_kl_beta`. It is
  recorded directly, so do not recompute the schedule when plotting.

Together these show whether the KL is being genuinely traded against the fit or is simply
being held down by a small $\beta$.

### Panel C — Prodigy adaptive step sizes

The six `prodigy_d_<group>` series, one per learned level group: `M1_level`, `M2_level`,
`V_level`, `g_level`, `Dv_level`, `Dw_level`. Log axis.

These are the per-group step sizes Prodigy chose for itself. A group whose value never
leaves its $10^{-3}$ initial value never moved, which is a concrete diagnosis of an
unidentifiable level rather than a vague one. In a real run `Dv_level` and `M2_level` stay
pinned at $10^{-3}$ while `g_level` and `M1_level` climb, and that is worth seeing.

### Panel D — learned observation sigma

The four likelihood $\sigma$ against step, from `train_sigma_position`,
`train_sigma_attitude`, `train_sigma_linear_velocity` and `train_sigma_angular_velocity`.
Held-out values are under the same names with a `test_` prefix.

This panel carries the noise-blind claim. Training starts every $\sigma$ at a neutral
$0.3$ with no knowledge of the true observation noise, so what matters is whether they
separate and settle near the true level. Draw a horizontal reference line at the dataset's
actual noise level and say in the caption that the model was never told it.

A flat $\sigma$ that never leaves $0.3$ means the likelihood never learned anything and the
run is effectively fitting an unweighted MSE.

### One implementation note

**The x-grids differ.** `objective`, `kl`, `test_total` and every `test_*` likelihood series
sit on the evaluation grid, roughly 13 points at the checkpoint interval. The NLL terms, the
sigma, the penalties, the latent KL and the KL contribution are recorded every optimizer
step, 3000 points. Plot each against its own grid; do not resample or interpolate one onto
the other.

### Per-block NLL and its ingredients

Beyond the totals, the trainer records the four per-block NLL terms
(`*_nll_position`, `*_nll_attitude`, `*_nll_linear_velocity`, `*_nll_angular_velocity`) and
the four squared-error norms they are built from (`*_position_squared_norm`,
`*_attitude_angle_squared`, `*_linear_velocity_squared_norm`,
`*_angular_velocity_squared_norm`), for both the `train_` and `test_` prefixes. They are not
required on this page, but they are available if a block-by-block breakdown is wanted later.

Each per-block NLL is the squared norm divided by $2\sigma^2$ plus the $\log\sigma$
normaliser, so a rising NLL term with a falling squared norm means $\sigma$ is shrinking
faster than the fit is improving.

---

## Page 10 — Open-loop rollout error, scalar summary

**Purpose.** The numeric summary of the open-loop forecast, and the bridge between the
identification result on page 1 and the trajectory pages that follow. Page 1 asks whether a
model recovered the physics; this page asks how far its rollout drifts.

**Seven rows, one column per model, plus the analytic operators as the last column.**

### Physical rows, so a reader can picture the error

| row | units | why it is here |
| --- | --- | --- |
| position RMS | m | the headline forecast error |
| position final | m | where it ended, which is what drift means |
| attitude RMS | rad | the second, independent failure mode |

### Unit-free rows, so the blocks compare to each other

Each divides the RMS error by the RMS excursion of the ground truth from its own initial
state over the same horizon.

| row | why it is here |
| --- | --- |
| position RMS / GT excursion | 1.0 means the error equals the true motion |
| attitude RMS / GT excursion | same threshold, different block |
| linear velocity RMS / GT excursion | same |
| angular velocity RMS / GT excursion | same |

These four are the most useful rows on the page and the easiest to overlook. They work like
relative RMS on page 1: unit-free, with 1.0 carrying a fixed meaning. A model reading 6.67 on
position has an error nearly seven times the size of the true motion; the analytic floor on
the same flight reads 0.591.

### What was deliberately left out

The older 23-row version of this table is cut down, not merely reformatted.

1. **The squared rows are gone.** $\lVert\Delta x\rVert^2$ mean and $\lVert\Delta x\rVert$
   RMS are one dataset in two unit systems, since $13.1 = \sqrt{172}$. Metres are readable,
   squared metres are not.
2. **The $SO(3)$ rows are gone**, to pages 13 and 14, where they are shown against time.
3. **The energy row is gone**, to page 15. It is gauge-dependent and does not belong in a
   table that will be read as an accuracy result.
4. **The $\pm$ spread column is gone.** See the note below.

### Required footer

State that the last column is the simulator's own analytic operators in the same integrator
driven by the same recorded $u(t)$, so it is the achievable floor rather than a competitor.
On a 30 s flight that floor is already 1.16 m, which a reader needs in order to judge the
other columns.

### Note on the $\pm$ spread, and why it is omitted

`shared_truth` currently takes one held-out flight and repeats it `count` times with
identical $x_0$ and identical $u(t)$; the copies are bit-identical. Any standard deviation
across them is exactly zero by construction, and printing it implies a reproducibility
result that was never measured.

If `shared_truth` is changed to return genuinely distinct held-out flights, the evaluation
pickle holds 18, then the spread becomes a real across-flight variation and should be
restored to this page as `mean ± std`. The same change is what makes the ten-trajectory
bundles on pages 2-6 and 11 carry information rather than overplotting one curve ten times.

---

## Page 11 — State trajectories, all 10 trajectories

**Layout.** A 4 x 3 grid of panels, one row per state block:

| row | panels |
| --- | --- |
| position | $x$, $y$, $z$ (m) |
| attitude | roll, pitch, yaw (rad), from the projected rotation |
| body velocity | $v_x$, $v_y$, $v_z$ (m/s) |
| body rate | $\omega_x$, $\omega_y$, $\omega_z$ (rad/s) |

**All 10 evaluation trajectories are drawn**, following the same rule as pages 2-6:

> Colour and line style encode the model, never the trajectory. Ground truth is black,
> for all 10 of its curves alike.

Reduced opacity on the curves, ground truth at full opacity on top.

**This changes current behaviour, deliberately.** The report today draws the *ensemble mean*
across the 10 trajectories. Those are 10 physically different flights from 10 different
initial conditions, so their mean position is not a trajectory anything flew, and averaging
hides the divergence it is supposed to show. Bundles of 10 are the honest picture.

**Posterior band** for any Bayesian model, on trajectory 0 only, in the model's colour, as
on pages 2-6. State in the legend that the band is trajectory 0 while the curves are all 10.

**What to look for.** Where a model's bundle separates from the ground-truth bundle is
where the rollout has lost the flight. Read it together with page 16, which quantifies the
same divergence.

---

## Page 12 — State trajectory, first trajectory only

Identical 4 x 3 layout to page 11, restricted to **trajectory 0**.

One curve per model plus ground truth, at full opacity and full line weight, with the
posterior band shaded behind. This is the page to lift into the paper; page 11 shows that
trajectory 0 is representative, page 12 is the one a reader can actually follow.

Say "trajectory 0 of 10" in the title, not "single trajectory".

---

## Page 13 — SO(3) violation, determinant

$\lvert \det R(t) - 1 \rvert$ against time, mean over the 10 trajectories, **log y-axis**.
One curve per model plus a reference line at machine precision.

**This page exists to show a structural property, not an accuracy result.** The Lie-IMEX
models advance attitude by a matrix exponential and stay on the group by construction, so
they sit at round-off, near $10^{-15}$. PH-NODE integrates the nine rotation entries as free
coordinates with no projection, so it drifts: a real run reaches $7\times10^{-9}$ at 0.1
noise, six orders of magnitude worse.

Note in the caption that six orders of magnitude here does **not** translate into six orders
of magnitude of trajectory error over the horizons in this report. The point is that the
violation is bounded by construction for one family and unbounded for the other.

---

## Page 14 — SO(3) violation, orthogonality

Same construction as page 13 with $\lVert R^\top R - I\rVert_F$ on the y-axis. Same log
scale, same reference line, same caption logic.

Keep both pages. The determinant catches volume drift and the Frobenius norm catches shear;
an integrator can be clean on one and not the other.

---

## Page 15 — Physical energy

Two panels on one page, sharing a time axis:

- **top**, ensemble mean across the 10 trajectories with a $\pm 2\sigma$ ground-truth band;
- **bottom**, trajectory 0 alone.

Energy in joules, one curve per model plus ground truth, model colours as elsewhere.

The two panels were separate pages in the older report. Merged here because they carry one
idea and a reader compares them directly.

Energy is a derived diagnostic, not an identifiable quantity: it depends on the learned
$\mu$ and $V$ and therefore on the gauge. A model can track energy well while getting the
physics wrong, and vice versa. Say so in the caption so this page is not read as evidence.

---

## Page 16 — State error against ground truth, all blocks on one axis

**One plot. One page.** Four error series per model, drawn together:

| series | quantity |
| --- | --- |
| position | $\lVert \Delta x\rVert^2$ (m²) |
| attitude | squared geodesic distance on $SO(3)$ (rad²) |
| linear velocity | $\lVert \Delta v\rVert^2$ (m²/s²) |
| angular velocity | $\lVert \Delta\omega\rVert^2$ (rad²/s²) |

Each is the mean over the 10 trajectories, against time.

### Styling rule

> **Colour encodes the model. Line style encodes the error block.**

As on pages 7 and 8, and inverted relative to pages 2-6, 11 and 12. Say so in the caption,
and use two small legends rather than one list of twelve entries.

**Log y-axis**, mandatory: the four blocks differ by orders of magnitude and squared position
error grows without bound while attitude error saturates near $\pi^2$.

This is the quantitative companion to pages 11 and 12. Those show where a rollout diverges;
this shows in which state block it diverged first, which is usually the diagnostic that
matters.

---

## Page 17 — PH-GP-LieIMEX variational posterior uncertainty

**Emitted once per Bayesian model.** A point-estimate baseline gets a placeholder page
stating that an MLP has deterministic weights and therefore no epistemic uncertainty, so the
report keeps a constant length across model families.

**Layout.** A 2 x 3 grid of histograms, one per subnetwork: $M_1^{-1}$, $M_2^{-1}$, $D_v$,
$D_\omega$, $V$, $g$. Each is the distribution of posterior weight standard deviations in
that subnetwork, with its median marked.

**Title, exactly:**

```
PH-GP-LieIMEX variational posterior uncertainty
Weight-space uncertainty; deterministic report rollouts use posterior means
```

The second line is not decoration and must not be dropped. Every rollout, every product and
every subnetwork query elsewhere in the report uses the posterior **mean** weights. The
spread shown here is what the $\pm 2\sigma$ bands on pages 2-6, 11 and 12 are sampled from,
not something that perturbs the reported numbers.

**What it is for.** A subnetwork whose $\sigma$ histogram sits high is one the data did not
constrain. Read alongside page 9, panel C: a level group whose Prodigy step size never moved
and whose posterior $\sigma$ stayed wide is unidentifiable under this loss, and that is a
finding rather than a failure.

---

## Pages 18+ — Closed-loop controller plots

Two image pages per model, in the model order used throughout:

1. **tracking plot** — commanded against achieved position over the 20 s reference;
2. **trajectory plot** — the same flight in 3-D.

These are rendered by the controller module and embedded as images, so they carry their own
styling and do not follow the colour convention of the rest of the report. Title each page
with the model name so there is no ambiguity.

**Different question, different answer.** Every preceding page is open loop: the model
integrates forward from a known state with recorded inputs, and errors accumulate. Here the
model computes control inside a feedback loop with PyBullet advancing the true state, so
feedback corrects what open loop cannot. The two have disagreed repeatedly in this project,
most sharply at 0.5 noise where our model holds 0.068 m closed loop while its open-loop
position error is the larger of the pair. Say this in the caption.

**Available but not specified here:** the failure-aware controller metrics table, with
position RMSE, completion fraction, motor saturation and contact counts per model. Add it as
a page before the image pages if a numeric summary is wanted.

---

## Closed-loop controller references (added 17 Sep 2026)

The controller booklet used to fly one reference, the released diamond — 15 s of straight
segments between waypoints, which is itself a chain of HARD-V5's `waypoint_hop`. The generator now
takes `--controller-reference <name>` (repeatable) and writes **one booklet per shape**,
`closed-loop-controller-<name>.pdf`, each opening with a failure-aware metrics table and then the
two tracking pages per model. `--controller-only` builds only these booklets. The diamond stays the
default so every earlier report is reproduced unchanged.

Five references were added in `report_controller.py`, none of which occur in the HARD-V5 training
or test library (hops, figure-eights, circles, vertical steps, yaw turns, kicks, coasts):

| name | shape | stresses |
| --- | --- | --- |
| `lissajous_3d` | $x = 1.2\sin\omega t,\ y = 1.2\sin(1.5\omega t + \pi/3),\ z = 0.6\sin(0.75\omega t)$, $\omega = 2\pi/10$ | all axes at once, non-repeating |
| `trefoil` | trefoil knot, $a = 0.4$ m, $z = 0.5\sin 3\theta$ | sustained reversing curvature in 3-D |
| `spiral_out_in` | $r = 1.3\sin(\pi t/15)$, $\dot\phi = 2\pi/5$ | speed ramps $0 \to 1.6 \to 0$ m/s |
| `chirp_line` | $x = 0.4\sin 2\pi(0.1t + ct^2)$, 0.1 → 0.6 Hz | bandwidth: inertia against damping |
| `rose_3` | $r = 1.2\cos 3\phi$, $z = 0.4\sin(2\pi t/7.5)$ | sharp petal reversals, no straight segment |

Every shape is analytic (position, velocity and acceleration exact — the diamond supplies no
acceleration), shifted to start at the vehicle's start point, active for 15 s, then held. Yaw is
zero throughout. Peak speeds are 0.8–1.6 m/s against the diamond's 0.4 m/s.

## Page numbering

The page count depends on how many models the report carries and how many of them are
Bayesian. Pages 1-16 are fixed. Page 17 repeats once per model, and the controller section
adds two pages per model.

Pages 6a and 6b were inserted after the rest of this document was written, so every heading
numbered 7 onward sits two pages later in the generated PDF than its own title says: the training
curves described as "page 7" are PDF page 9, and the controller section described as "pages 18+"
begins at PDF page 20. Inline cross-references in this document use the heading numbers, not the PDF
numbers. The booklet table in **Output layout** gives the PDF ranges. Within a booklet the
ordering is unchanged.

## Output layout

One invocation of `generate_comparison_report_v2.py` is one **eval run**. It writes one folder
under `experiments/quadrotor/eval_runs/`, named the way a training run folder under
`experiments/quadrotor/train_runs/` is named:

```
<DD-MM-HH-MM>_spec-comparison_obs-noise<tag>_<evaluation dataset stem>/
```

The pages above are not one file. They are split into four PDFs, one per section, so each can be
opened on its own:

| file | PDF pages | contents |
| --- | --- | --- |
| `physics-identification.pdf` | 1-8 | the product table, the five per-product pages, the damping-law page, and the metric definition |
| `training-and-test-losses.pdf` | 9-11 | train curves, test curves, GP objective terms |
| `open-loop-rollout.pdf` | 12-19 | summary table, states, SO(3) violation, energy, error, posterior |
| `open-loop/comparison.pdf`, `open-loop/images/<set>_states_flight<k>.png` and `<set>_trajectory_flight<k>.png` (one each per flight), `<set>_error.png`, `<set>_calibration_<model>.png` | one page per evaluation set | metrics table (point rows on the posterior-mean weights, probabilistic rows on 50 posterior samples, GP only), the per-model state grid with the GP band, the error-vs-time plot, and the GP calibration page; extra sets via `--open-loop-dataset PATH[@heldout|@test]` (a V5/V6-style pickle defaults to its held-out split when it has one) |
| closed-loop references from recorded flights | — | `--controller-recorded-reference PATH[@split]` registers every flight of an evaluation set as a reference named `<SET>-<split>-flight<k>` (position, velocity and heading as flown, heading-relative, feedforward from smoothed finite differences; active for the flight's duration, then held), flown and reported exactly like the analytic shapes |
| `closed-loop/images/<reference>_trajectory.png` and `_tracking.png`, `closed-loop/comparison.pdf`; with `--closed-loop-subfolders` also `closed-loop/<reference>/<model>.pdf` and `<reference>/comparison.pdf` | 3 / 1 / one per reference | one subfolder per reference shape; inside it one 3-page PDF per model (metrics table, tracking, 3-D trajectory) and a single-page table with every model as a column; the top-level `comparison.pdf` collects those tables |

Alongside them, `report_metadata.json` records the generator, the specification file, the
evaluation dataset, the number of held-out flights, the horizon, and for each model its run
folder, its `model.name`, its observation noise and the checkpoint step that was evaluated. An
eval folder is therefore self-describing in the same way a training run folder is.

`--output-name` sets the folder name, not a file name.

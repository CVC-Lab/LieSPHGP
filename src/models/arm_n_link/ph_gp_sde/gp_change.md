# `gp_core` Stabilisation — Bounded Residuals and a Softer KL Schedule

*2026-07-27. Companion to [`algorithm.md`](algorithm.md). Changes live in
[`../utils/structured_subnets.py`](../utils/structured_subnets.py) and
[`train.py`](train.py).*

Two `--gp_core --beta_max 1.0` runs went NaN:

| run | `num_points` | `mc_samples` | β saturates | first instability | NaN |
|---|---|---|---|---|---|
| A (seed 42) | 9 | 1 | step 500 | ~1100 (PL: 12 → 41 → 101 → 678 → 7223) | 1350 |
| B (seed 0) | 5 | 10 | step 500 | invisible (`eval_every 1000`) | < 2000 |

Run B is the sharper data point: at step 1000 it had the *best fit ever seen on
this problem* (loss 36.9, $\mathrm{nll}_R=-3.05$) and still died within the
next 1000 steps. More MC samples reduce gradient variance but **increase** the
per-step probability of a catastrophic forward pass ($B\times S = 320$ weight
draws per step instead of 32) — the failure is a tail event, not a variance
problem. Hence the two changes below.

---

## 1. The failure mechanism

**The unprotected chain** (was [`structured_subnets.py`](../utils/structured_subnets.py),
`StructuredMass.physical`):

$$
\log m_i(q) \;=\; \log m_i^{\rm base} + \mathrm{GP}_i(q)
\;\xrightarrow{\ \exp\ }\; m_i
\;\longrightarrow\; M(q)
\;\longrightarrow\; M^{-1}(q),\quad p = M(q)\,\omega
$$

and identically for $\log d_i$ in `StructuredDissipation`.

**Step 1 — the KL widens the posteriors.** The five core GPs carry
$\approx 15\,500$ variational weights $w \sim \mathcal N(\mu, \sigma_w^2)$
initialised at $\sigma_w = e^{-2}$, so the initial KL is

$$
\mathrm{KL}_0 \approx 15\,500 \times \tfrac12\big(e^{-4} - 1 + 4\big)
\approx 23\,400
$$

— exactly the value printed at step 0 (23 373.9). The only way to shrink it is

$$
\frac{\partial\,\mathrm{KL}}{\partial\sigma_w} = \sigma_w - \frac{1}{\sigma_w} < 0
\quad\text{for } \sigma_w < 1
\qquad\Longrightarrow\qquad \sigma_w \uparrow 1,\ \mu \downarrow 0 .
$$

Widening $\sigma_w$ injects unit-scale sampled noise into every GP output.

**Step 2 — the exponential makes the noise heavy-tailed.** A Gaussian residual
$r$ on $\log m$ becomes *log-normal multiplicative* noise on $m$: a draw with
$r \sim 10$ (rare per-sample, but sampled $320$ times per step, $10^4$ steps)
gives $m \sim e^{10} \approx 2.2\times10^4$.

**Step 3 — stiffness crosses the integrator's stability limit.** The explicit
Heun scheme is stable only while $h\lambda \lesssim 2$ with
$\lambda \sim \lambda_{\max}\!\big(M^{-1}(q)\,D(q)\big)$. A tail draw on $m$ or
$d$ pushes $h\lambda$ past the limit; the momentum then grows geometrically over
the $4\times10 = 40$ substeps of a window. Weights are (correctly, for the
ELBO) held **fixed per trajectory**, so one bad draw poisons an entire rollout.

**Step 4 — float32 turns growth into NaN.** Overflow at
$\approx 3.4\times10^{38}$, then
$\texttt{clip\_by\_global\_norm}(\mathrm{NaN}) = \mathrm{NaN}$, the parameters
go NaN, and every subsequent step is dead. Gradient clipping cannot help: the
**loss itself** is non-finite in the forward pass.

The PL term is the amplifier that makes the run-up visible first: its quadratic
is $\|r\|^2/s$ with $s = 2\sigma_{\rm obs}^2 = 0.02$, a $\times 50$ multiplier
on any drift error — which is why the `pl` column exploded before `nll`.

---

## 2. Change 1 — bound the GP residuals

**Before:**

$$
\theta(q) \;=\; \theta_{\rm base} + \mathrm{GP}(q)
$$

**After** (`_squash` in [`structured_subnets.py`](../utils/structured_subnets.py)):

$$
\boxed{\;\theta(q) \;=\; \theta_{\rm base}
+ \kappa\tanh\!\Big(\frac{\mathrm{GP}(q)}{\kappa}\Big),
\qquad \kappa = \texttt{RESIDUAL\_KAPPA} = 2.5\;}
$$

applied to **every** `gp_core` residual:

| subnet | squashed residual on | constants affected |
|---|---|---|
| `StructuredMass.physical` | the full GP output vector | $\log m,\ L_{\rm body},\ \ell_0,\ c$ |
| `StructuredDissipation.__call__` | $\log d$ residual | $d_i = e^{\log d_i}$ |
| `StructuredInputMap.__call__` | gain residual | $\gamma_i$ |
| `StructuredSigma.__call__` | lever residual | $v_j$ |

### The guarantee

For every $q$ and every sampled weight draw $w$:

$$
|\theta(q) - \theta_{\rm base}| < \kappa
\qquad\Longrightarrow\qquad
m_i \in \big(e^{-\kappa},\, e^{\kappa}\big)\, m_i^{\rm base},
\quad
d_i \in \big(e^{-\kappa},\, e^{\kappa}\big)\, d_i^{\rm base}
$$

with $e^{\kappa} \approx 12.2$. Combined with the existing inertia floor
$\lambda_{\min}(M) \ge \varepsilon_{\mathbb I} = 0.05$ and bounded levers, the
drift's Lipschitz constant — and hence the per-substep growth factor
$\big(1 + h\lambda + O(h^2\lambda^2)\big)$ — is **uniformly bounded over all
$q$ and all draws**. Worst-case transient growth over a window is
$e^{40\,h\lambda} = O(1)$–$O(10^2)$, dozens of orders of magnitude below float32
overflow. NaN becomes structurally impossible rather than statistically
unlikely.

### Why it costs nothing representationally

The residual `gp_core` exists to capture is the configuration part of the
varying-friction modulation $\rho_i \in [1, 2]$, i.e. a log-residual of at most

$$
|r| \le \ln 2 \approx 0.69 \ll \kappa = 2.5 .
$$

In that regime the squash is nearly the identity:

$$
\kappa\tanh\!\Big(\frac{r}{\kappa}\Big)
= r\Big(1 - \frac{r^2}{3\kappa^2} + O(r^4)\Big)
\;\approx\; 0.974\, r \quad\text{at } r = 0.69 .
$$

The gradient $\mathrm{sech}^2(r/\kappa) \in (0, 1]$ never vanishes, so there
are no dead zones — a saturated residual still receives a restoring gradient.

### Compatibility

- No new arrays: the pytree is unchanged, so existing checkpoints deserialise
  exactly as before. Behaviour differs only where $|\mathrm{GP}(q)| \gtrsim
  \kappa$ — precisely the pathological region.
- `residual_kappa` is now recorded in `history.pkl` alongside `i_epsilon`, so
  a run's architecture is fully reconstructable.
- Non-`gp_core` runs are bit-identical (the squash sits inside the
  `if self.gp_core:` branches).

---

## 3. Change 2 — soften and delay the KL

**Before** ([`train.py`](train.py)):
$\beta(t) = \beta_{\max}\min(1, t/t_{\rm anneal})$ with $t_{\rm anneal} = 500$
— fully on at step 500. Both crashes began 500–1500 steps later.

**After:**

$$
\boxed{\;\beta(t) \;=\; \beta_{\max}\,
\mathrm{clip}\!\Big(\frac{t - t_{\rm warm}}{t_{\rm anneal}},\, 0,\, 1\Big)\;}
\qquad
\begin{aligned}
t_{\rm warm} &= \texttt{--kl\_warmup\_steps} = 2000 \text{ (default)}\\
t_{\rm anneal} &= \texttt{--kl\_anneal\_steps} = 3000 \text{ (default, was 500)}
\end{aligned}
$$

The warmup lets the likelihood localise the weight posteriors before the prior
starts widening them; the slower ramp spreads the widening pressure over
thousands of steps instead of hundreds.

### Why $\beta = 1$ was never "the true ELBO"

For $\beta\,\mathrm{KL}/N$ to be the ELBO weight, the data-fit term must be the
**sum** of log-likelihoods over all observations. But the code's NLL is a
per-snapshot **mean** (`elbo_nll_nlink` averages over samples), while the KL is
divided by the number of windows $N$ — and the windows *overlap* (stride 1), so
$N$ over-counts correlated transitions by a factor of `num_points` anyway. The
units on the two sides of

$$
\mathcal L = \underbrace{\mathcal L_{\rm NLL}}_{\text{per-snapshot mean}}
+ \lambda_{\rm PL}\mathcal L_{\rm PL}
+ \frac{\beta}{N}\underbrace{\mathcal L_{\rm KL}}_{\text{total nats}}
$$

do not match, so $\beta$ is an **empirical regularisation dial**, not a
derived constant. Recommended range with `--gp_core`: $\beta_{\max} \in
[0.05,\ 0.1]$. With Change 1 in place, an over-large $\beta$ now degrades
accuracy instead of killing the run — a recoverable failure mode.

---

## 3b. Change 3 — gauge anchor (follow-up, same day)

The first stabilised run (10 000 steps, no NaN) exposed the next problem: the
**scale gauge ratchet**. The dynamics are exactly invariant under

$$
(M, V, D, g, \Sigma) \;\to\; \beta\,(M, V, D, g, \Sigma),
\qquad p \to \beta p,\qquad \forall \beta > 0,
$$

so the loss has a perfectly flat valley. The rollout NLL prefers small
*observable* diffusion $A = M^{-1}\Sigma$, and inflating $M$'s scale is the
cheapest way to get it — so the optimizer walked the valley to
$\beta \approx 21$: masses 21× truth, $V$ 21× truth (the V-GP paying a visibly
rising KL for the privilege), and $\Sigma$ left chasing a moving target of
$21 \times 1.581 \approx 33$ it never reached (final $\Sigma\Sigma^\top$ error
0.98 while $M^{-1}$ sat at 7.9%).

**Fix** — pin the gauge by convention (`--anchor_m1`, default 1):

$$
\boxed{\;m_1(q) \;\equiv\; m_1^{\rm anchor}\;}
$$

implemented in `StructuredMass.physical` by overwriting `log_m[0]` **after**
the base value, the variational draw and the (squashed) `gp_core` residual are
combined — so no path, including a constant GP offset on $\log m_1$, can
re-open the gauge. The gauge direction is one-dimensional, so freezing this
single number kills it entirely: $\beta$ stays $O(1)$, $\Sigma$'s target
becomes the fixed, reachable $\sigma\sqrt{2\sum_j\|v_j\|^2} \approx 1.58$, and
the V-GP no longer inflates.

**Why this is not "giving the ground truth":** the data determines only
ratios — for *every* value $m_1 = c$ there is a parameter setting reproducing
the observations exactly, so the anchor transfers zero bits about the
data-generating process. It is the multiplicative analogue of centring $V$
(the additive gauge), and standard practice everywhere scale is unobservable
(base inertial parameters in robot ID, monocular SLAM scale, grounding a
circuit node). **Control experiment:** train with `--anchor_m1 2.7`; all
gauge-invariant results must be unchanged, with every raw scale ×2.7. Fine
print: under `gp_core` the anchor also asserts $m_1$ is *constant* in $q$ —
a (physically certain) modeling assumption slightly stronger than pure gauge
fixing.

`anchor_m1` is a static field: recorded in `history.pkl`, mirrored in
`load_run` (`.get('anchor_m1', None)`, so pre-anchor checkpoints load with
the gauge free, exactly as trained).

## 3c. Change 4 — trace anchor (postmortem of Change 3)

**Change 3 failed, and the failure is instructive.** A run with `anchor_m1=1.0`
active still reached $\beta = 23.6$ with `|S|` climbing monotonically to 9.1 —
behaviourally identical to the unanchored run. The reason: the map from
parameters to the function $M(\cdot)$ is many-to-one, and $M$ is **quadratic
in the levers**:

$$
M_{jk} = \mathbb{I}_j\delta_{jk} - \sum_i m_i\,[u_{ij}]_\times R_j^\top R_k\,[u_{ik}]_\times
\qquad\Longrightarrow\qquad
\big(u \to \sqrt{\beta}\,u,\ \mathbb{I} \to \beta\,\mathbb{I},\ m \text{ fixed}\big)
\;\Rightarrow\; M \to \beta M .
$$

With $m_1$ frozen, the optimizer scaled the lever vectors by $\sqrt\beta$ and
the inertias by $\beta$ instead. Freezing any finite set of *parameters*
leaves such routes open; the anchor must act where the gauge acts — on the
**function** $M(q)$. Since the diagonal blocks of $M$ carry no
$q$-dependence, the trace is a clean scale readout, so
(`--anchor_trace`, default $c_0 = 4n$):

$$
\boxed{\;M(q) \;\leftarrow\; \frac{c_0}{\mathrm{tr}\,M(q)}\;M(q)\;}
$$

Now **no parameter combination whatsoever** can rescale $M$ — and a "gauge"
move scaling $V, D, g, \Sigma$ *without* $M$ changes the observable dynamics,
so it is identifiable, not flat. The valley is closed at the function level.
$\Sigma$'s target becomes the fixed constant
$\beta_{c_0}\,\sigma\sqrt{2\sum_j\|v_j\|^2}$ with
$\beta_{c_0} = \mathrm{tr}\,M_{\rm GT}/c_0$ (for this environment
$\mathrm{tr}\,M_{\rm GT} = 8$, so $c_0 = 8$ gives $\beta \approx 1$).
$c_0$ remains a pure units choice — any positive value is gauge-equivalent.

Fine print: under `gp_core` the normalisation pins $\mathrm{tr}\,M$ at
*every* $q$, i.e. it also asserts the overall scale of $M$ is not
configuration-dependent — the same mild, physically-certain assumption class
as Change 3 made. `anchor_m1` is retained (harmless, and it kills the dead
parameter directions' drift); `anchor_trace` is recorded in `history.pkl`
and mirrored in `load_run` (pre-anchor checkpoints load with the gauge free).

**New diagnostic (same commit):** the final eval prints the adjacent
wind-lever alignment $\cos(v_i, v_{i+1})$. One shared wind field must torque
every link the same way, so $+1$ is expected; the $\beta=23.6$ run learned
$v_1 \approx -3.8\,e_z$, $v_2 \approx +2.6\,e_z$ — per-link axes almost
perfect ($|\cos(v_i, e_z)| > 0.98$) but the **relative sign flipped**, which
is a genuine error (anti-correlated wind response), not a gauge, and is why
the gauge-free $AA^\top$ error exceeded 1 while $\|A\|$ was only ~26% low.

## 3d. Change 5 — detach Σ from the rollout NLL (`--sigma_rollout_grad` to undo)

With the trace anchor in place the first anchored run behaved exactly as
designed on the gauge ($\beta = 1.12$, `|S|` flat) — and thereby exposed the
tug-of-war in its pure form. Two forces train $\Sigma$:

* the **PL**, pulling $A = M^{-1}\Sigma$ toward the observed increment
  covariance (informative), and
* the **rollout NLL**, for which model-side diffusion can only *add* variance
  to the residual against an independently-noisy target — pure
  $\Sigma \to 0$ pressure, zero information (this file's §1 and `train.py`'s
  own docstring already said so).

Pre-anchor, the NLL vented that pressure through the cheap gauge direction
(inflate $M$, shrink the observable $A$); post-anchor it acted on $\Sigma$
directly and **won** at $\lambda_{\rm PL} = 1$: measured
$\|v_1\| = 0.349$ vs a target of $1.12$ (31%), $\|v_2\| = 0.019$ vs $0.56$
(collapsed — the model believed the wind exerts no torque on link 2),
$\|\Sigma\|_F$ at 28% of target ⇒ ~8% of the true noise *power*, and
coverage@90 stuck near 0.2.

**Fix**: cut exactly that one gradient path. `stochastic_increment_p` — used
*only* by the rollout integrator; the PL calls `Sigma()` directly — now
applies `stop_gradient` to $\Sigma_\theta(q)$ before multiplying $dW$:

$$
\text{rollout: } \Delta p = \mathrm{sg}\big[\Sigma_\theta(q)\big]\,dW
\qquad\Longrightarrow\qquad
\frac{\partial \mathcal L_{\rm NLL}}{\partial \theta_\Sigma} = 0,\quad
\frac{\partial \mathcal L_{\rm PL}}{\partial \theta_\Sigma}\ \text{unchanged.}
$$

Loss *values* and rollout samples are untouched — only the collapse gradient
is removed, making the stated division of labour literal: the NLL trains the
drift, the PL alone trains the diffusion. `--sigma_rollout_grad` restores the
old behaviour; the resolved boolean is recorded in `history.pkl` and mirrored
in `load_run` (old runs rebuild with `False`, as trained).

Also in this change: the wind-lever diagnostic now prints $\|v_i\|$ and
reports `n/a` for the adjacent cosine when a lever has collapsed
($\|v\| < 10^{-2}$) — the $-0.302$ printed by the anchored run was the
"direction" of a numerically-zero vector, i.e. noise.

## 4. What did *not* change

- The closed forms, the integrator, the NLL/PL/KL definitions, the gauge
  analysis, and all `beta_max = 0` behaviour (bit-identical for non-`gp_core`
  models; `gp_core` forward passes change only through the squash).
- `--lr_sigma` semantics: still `None` ⇒ one optimiser for everything.

## 5. Recommended rerun

```bash
python src/models/arm_n_link/ph_gp_sde/train.py \
  --n 2 --obs_noise_std 0.1 --friction_coeff 0.5 --wind_force_std 0.5 \
  --random_u --random_u_scale 2.0 --samples 64 --timesteps 30 --num_points 5 \
  --hidden_dim 32 --sigma_mode structured --lambda_pl 1.0 --g_diag 0.5 0.7 0.5 \
  --gp_core --beta_max 0.1 --kl_warmup_steps 2000 --kl_anneal_steps 3000 \
  --total_steps 10000 --learn_rate 1e-3 --batch_size 32 --seed 0 \
  --mc_samples 10 --eval_every 200
```

(`--eval_every 200` while validating: run A showed the crash takes only ~200
steps from first spike to NaN, so `eval_every 1000` is blind to the run-up.)

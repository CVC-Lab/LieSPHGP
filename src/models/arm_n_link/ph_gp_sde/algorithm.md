# `ph_gp_sde` on the $n$-Link Arm — Algorithm

A step-by-step mathematical specification of the model in
[`network.py`](network.py), the subnets in
[`../utils/structured_subnets.py`](../utils/structured_subnets.py), the
integrator in [`../utils/lie_integrator_nlink.py`](../utils/lie_integrator_nlink.py),
the losses in [`../utils/elbo_loss_nlink.py`](../utils/elbo_loss_nlink.py) and
the training loop in [`train.py`](train.py).

The **ground-truth system** this model learns is specified separately in
[`multi-joint-ph-system.md`](../../../../multi-joint-ph-system.md) and
implemented in
[`envs/arm_nlink_SO3/arm_nlink_physics.py`](../../../../envs/arm_nlink_SO3/arm_nlink_physics.py).
This document is about the *learned* counterpart.

---

## 1. Data

The environment emits, at every snapshot $k$, a $15n$-dimensional row

$$
y_k \;=\; \big(\underbrace{\mathrm{vec}(R_1),\dots,\mathrm{vec}(R_n)}_{9n},\;
\underbrace{\omega_1,\dots,\omega_n}_{3n},\;
\underbrace{u_1,\dots,u_n}_{3n}\big)\;\in\;\mathbb{R}^{15n}
$$

- $R_i \in SO(3)$ — link $i$'s **absolute** world attitude, row-major flattened.
  Write $q := (\mathrm{vec}(R_1),\dots,\mathrm{vec}(R_n)) \in \mathbb{R}^{9n}$.
- $\omega_i \in \mathbb{R}^3$ — link $i$'s **body-frame** angular velocity.
- $u_i \in \mathbb{R}^3$ — commanded torque at joint $i$, held constant across
  the interval.

> **Torque time convention.** Row $k$ stores the torque that *produced* $y_k$.
> A consumer integrating $t \to t{+}1$ must therefore read row $t{+}1$, i.e.
> `x[1:, 12n:15n]` — not `x[:-1]`. Getting this wrong is silent when torques are
> constant and damaging when they are random.
> See [`elbo_loss_nlink.py:160-165`](../utils/elbo_loss_nlink.py#L160-L165).

Observations are corrupted by noise $\sigma_{\rm obs}$: rotations by a
right-multiplied $\exp([\varepsilon]_\times)$, rates additively.

Training slices trajectories into short overlapping windows of `--num_points`
snapshots (`arrange_data`), because the arm is chaotic and long rollouts have
no usable gradient.

---

## 2. Configuration manifold and state

$$
q \in G = SO(3)^n,\qquad \dim G = 3n
$$

The $9n$-dimensional embedding is **redundant** — it carries $9n$ numbers for
$3n$ degrees of freedom. This is safe because every quantity the model
differentiates is a directional derivative *along the group action* (§5), so
the result is independent of how a function is extended off the manifold.

Velocities are left-trivialized: $\dot R_i = R_i[\omega_i]_\times$. The
conjugate momentum is $p = M(q)\,\omega \in \mathbb{R}^{3n}$.

The integrator carries $(q, p)$; the data and the loss live in
$(q, \omega)$. Conversions happen only at step boundaries.

---

## 3. The learned port-Hamiltonian SDE

$$
H_\theta(q,p) \;=\; \tfrac12\,p^\top M_\theta^{-1}(q)\,p \;+\; V_\theta(q),
\qquad \xi \;:=\; \frac{\partial H_\theta}{\partial p} \;=\; M_\theta^{-1}(q)\,p
$$

$$
\begin{aligned}
dq_i &= R_i[\xi_i]_\times\,dt \\[2pt]
dp &= \Big(\underbrace{\hat P\,\xi}_{\text{gyroscopic } \mathrm{ad}^*}
        \;+\;\underbrace{\mathcal T(H_\theta)}_{\text{gravity + Coriolis}}
        \;-\;\underbrace{D_\theta(q)\,\xi}_{\text{dissipation}}
        \;+\;\underbrace{g_\theta(q)\,u}_{\text{actuation}}\Big)\,dt
      \;+\;\underbrace{\Sigma_\theta(q)\circ dW_t}_{\text{wind}}
\end{aligned}
$$

with $\hat P = \mathrm{blkdiag}([p_1]_\times,\dots,[p_n]_\times)$ and
$W_t$ a **3-dimensional** Wiener process — one shared world wind field drives
every link, which is why $\Sigma_\theta$ is $3n\times 3$ and not $3n\times 3n$.

The $\circ$ denotes Stratonovich. For this system Itô $\equiv$ Stratonovich
exactly: the diffusion depends on $q$ alone and the $q$-rows of the diffusion
are zero, so the correction term vanishes (verified to $0.00\mathrm{e}{+}00$).

### 3.1 The trivialized gradient

$$
\mathcal T_i(F) \;=\; \sum_{k=1}^{3} r_{ik} \times \frac{\partial F}{\partial r_{ik}}
\;=\; -\frac{\partial F}{\partial \phi_i}
$$

where $r_{ik}$ is row $k$ of $R_i$. In code ([`network.py`](network.py),
`drift_p`) this is one `jax.grad` through $H_\theta$ followed by a cross
product — and that **single** gradient produces both gravity *and* the full
inter-link Coriolis field. No Christoffel symbols appear anywhere.

Because $(q,p)$ are the integrated state, $p$ is independent of $q$ in the
autograd graph, so no `stop_gradient` is needed inside $H_\theta$.

---

## 4. Subnetworks

| subnet | parameterisation | quantity | params ($n=2$) |
|---|---|---|---|
| `M_net` | `StructuredMass` | $M_\theta^{-1}(q)\in\mathbb{R}^{3n\times3n}$ | 23 |
| `Dw_net` | `StructuredDissipation` | $D_\theta(q)\in\mathbb{R}^{3n\times3n}$ | 2 |
| `g_net` | `StructuredInputMap` | $g_\theta(q)\in\mathbb{R}^{3n\times3n}$ | 6 |
| `Sigma_net` | `StructuredSigma` *or* `MatrixGP_NLink` | $\Sigma_\theta(q)\in\mathbb{R}^{3n\times3}$ | 6 / 6 336 |
| `V_net` | `GP_NLink` $\to 1$ | $V_\theta(q)\in\mathbb{R}$ | 352 |

$M$, $D$, $g$ and (optionally) $\Sigma$ are **physics-structured**: they learn
the few physical parameters appearing in the closed forms and evaluate the
exact algebra. $V_\theta$ is the only learned *function* left in the drift.

### 4.0 What is learned versus what is computed

This distinction is the whole design, so it is worth stating before the
formulas.

**For a GP subnet** (the old design, and still `V_net` / `Sigma_net` in `full`
mode), the learned object is a set of weights $w$, and the configuration $q$ is
an **input** to a function that must be evaluated at every call:

$$
q \;\xrightarrow{\ \text{random features}\ }\ \varphi(q)
\;\xrightarrow{\ w\ }\ \text{matrix entries}
$$

The network has to *discover* how the output depends on $q$ from data.

**For a structured subnet**, the learned objects are configuration-**independent
constants** — masses, inertias, lever arms, friction coefficients, gains. There
is no network, no features, and $q$ is never an input to anything learned. It
enters only through the exact closed form:

$$
\underbrace{(m,\mathbb{I},\ell,c)}_{\text{learned constants}}
\;+\;\underbrace{q}_{\text{state}}
\;\xrightarrow{\ \text{exact algebra}\ }\ M(q)
$$

So the $q$-dependence is **given**, not learned. Only the physical constants
are fitted. Consequences:

- Rotation invariance, the block sparsity of $T(q)$, the constancy of $M$'s
  diagonal blocks — all exact by construction rather than approximated.
- Positive-definiteness comes from the parameterisation ($m_i > 0$,
  $\mathbb{I}_i = L_iL_i^\top$), so no conditioning floor $\varepsilon$ is
  needed.
- Extrapolation to configurations never visited in training is exact, because
  there is nothing to extrapolate — the formula holds everywhere on $SO(3)^n$.

Each subsection below lists, in order: **learned** (the free parameters),
**given** (what the caller supplies), and **computed** (the formula they feed).

### 4.1 `StructuredMass` — $M_\theta^{-1}(q)$

**Learned** — $2 + 12 + 3 + 6 = 23$ effective numbers at $n=2$, none of them a
function of $q$:

| symbol | code attribute | shape | count | transform |
|---|---|---|---|---|
| $m_i$ — link masses | `log_m` | $(n,)$ | $n$ | $m_i = e^{\log m_i} > 0$ |
| $\mathbb{I}_i$ — inertia about the COM | `L_body` | $(n,3,3)$ | $6n$ | $\mathbb{I}_i = L_iL_i^\top$, $L_i = \mathrm{tril}(\cdot)$ — PSD by construction |
| $\ell_0$ — joint$\to$joint offset | `ell0` | $(3,)$ | $3$ | none |
| $c_i$ — joint$\to$COM offset | `c` | $(n,3)$ | $3n$ | none |

Two masking details: `L_body` is stored as a full $(n,3,3)$ but `jnp.tril`
discards the strict upper triangle, so $6n$ of the $9n$ stored entries are
effective and the rest receive zero gradient. And only $\ell_0$ is free —
$\ell_{n-1}$ never enters the dynamics (the last link has no downstream
neighbour), so `physical()` pads the remaining $\ell_j$ with zeros.

**Given** — $q\in\mathbb{R}^{9n}$, reshaped to $R\in\mathbb{R}^{n\times3\times3}$.

**Computed** — the learned constants are substituted into

$$
\boxed{\;M_{jk}(q)=\underbrace{\mathbb{I}_j}_{\text{learned}}\,\delta_{jk}
-\sum_{i\ge\max(j,k)} \underbrace{m_i}_{\text{learned}}\,
[\underbrace{u_{ij}}_{\text{learned}}]_\times\,
\underbrace{R_j^\top R_k}_{\text{from }q}\,
[\underbrace{u_{ik}}_{\text{learned}}]_\times\;}
\qquad u_{ij}=\begin{cases}\ell_j & j<i\\ c_i & j=i\end{cases}
$$

and the result is inverted: $M_\theta^{-1}(q) = \big(M(q)\big)^{-1}$.

$$
\{\log m,\;L_{\rm body},\;\ell_0,\;c\}
\ \xrightarrow{\ \texttt{physical()}\ }\ \{m,\mathbb{I},\ell,c\}
\ \xrightarrow[\ +\,q\ ]{\ \text{closed form}\ }\ M(q)
\ \xrightarrow{\ \texttt{jnp.linalg.inv}\ }\ M^{-1}(q)
$$

Note the only place $q$ appears is the factor $R_j^\top R_k$ — no learned
quantity touches it.

Derivation: link $i$'s COM sits at $p_i = \sum_{j<i}R_j\ell_j + R_ic_i$;
differentiating with $\dot R_j = R_j[\omega_j]_\times$ and
$a\times b = -[b]_\times a$ gives the block **lower-triangular** Jacobian
$\dot p_i = J_{v,i}(q)\,\omega$, and
$M(q)=\mathrm{blkdiag}(\mathbb{I}_i)+\sum_i m_i J_{v,i}^\top J_{v,i}$.

Three structural facts, all measured:

1. **Diagonal blocks are exactly constant** ($R_j^\top R_j = I$ cancels), std
   $0.0000$ along a trajectory. All variation is inter-link coupling.
2. **$q$ enters only through $R_j^\top R_k$**, so $M(hq)=M(q)$ for any global
   $h\in SO(3)$ — exact by construction here, verified to $4.8\times10^{-7}$.
3. **The spectrum barely moves.** At $n=2$, four of six eigenvalues are
   constant $\{0.382, 1, 1, 2.618\}$; only two vary, and $\det M$ runs $1\to2$
   between aligned and perpendicular links. $M^{-1}$ changes *orientation*, not
   size.

Positive-definiteness is free from
$m_i = e^{\log m_i}>0$ and $\mathbb{I}_i = L_iL_i^\top$, so **no $\varepsilon$
conditioning floor is needed** — unlike the GP it replaced, where $\varepsilon=1$
forced $[M^{-1}]_{ii}\ge1$ while 33% of the true diagonal lies below 1.

> **Identifiability.** $\mathbb{I}_n$ and $m_n[c_n]_\times^\top[c_n]_\times$ only
> ever appear summed (the *base inertial parameters* degeneracy). A supervised
> fit reproduces $M(q)$ to $2\times10^{-5}$ while recovering $m=(1.35,1.10)$
> against a truth of $(1,1)$. **Do not report the individual parameters as
> identified physical quantities** — only their dynamically relevant
> combinations are determined.

### 4.2 `StructuredDissipation` — $D_\theta(q)$

**Learned** — $n$ numbers ($2$ at $n=2$):

| symbol | code attribute | shape | count | transform |
|---|---|---|---|---|
| $d_i$ — joint viscous friction | `log_d` | $(n,)$ | $n$ | $d_i = e^{\log d_i} > 0$ |

**Given** — $q$, from which $T(q)$ is built. $T$ carries **no** learned
parameters: it is pure chain kinematics, block lower-**bi**diagonal with
$T_{ii}=I_3$ and $T_{i,i-1}=-R_i^\top R_{i-1}$.

**Computed** —

$$
D(q) \;=\; \underbrace{T(q)^\top}_{\text{from }q}\,
\mathrm{blkdiag}(\underbrace{d_i}_{\text{learned}} I_3)\,
\underbrace{T(q)}_{\text{from }q},
\qquad
\Omega_i = \omega_i - R_i^\top R_{i-1}\omega_{i-1},\quad \Omega = T(q)\,\omega
$$

$$
\{\log d\}\ \xrightarrow{\ \exp\ }\ \{d_i\}
\ \xrightarrow[\ +\,T(q)\ ]{\ \text{congruence}\ }\ D(q)
$$

So the *shape* of the dissipation as a function of $q$ is entirely given by the
kinematics; only $n$ scalar strengths are fitted.

Friction opposes the **relative** joint rate, which is why $T$ appears rather
than the identity. Since $\det T = 1$, $T$ is always invertible, which is
exactly why joint friction alone damps all $3n$ directions.

### 4.3 `StructuredInputMap` — $g_\theta(q)$

**Learned** — $3n$ numbers ($6$ at $n=2$):

| symbol | code attribute | shape | count | transform |
|---|---|---|---|---|
| $\gamma_i$ — per-axis actuator gain | `gain` | $(n,3)$ | $3n$ | none — a sign flip is physical (a motor wired backwards) and the data identifies it |

**Given** — $q$, from which the same $T(q)$ as in §4.2 is built.

**Computed** — the gain multiplies on the **right**, scaling input *channels*
(columns), not rows:

$$
g(q) \;=\; \underbrace{T(q)^\top}_{\text{from }q,\ \text{no parameters}}\,
\underbrace{\Gamma}_{\text{learned}},\qquad
\Gamma = \mathrm{blkdiag}\big(\mathrm{diag}(\gamma_i)\big)
$$

$$
\{\gamma\}\ \xrightarrow[\ +\,T(q)\ ]{\ \text{column scaling}\ }\ g(q)
$$

Derivation: power at joint $i$ is $u_i^\top\Omega_i$, so
$u^\top\Omega = u^\top T\omega = (T^\top u)^\top\omega$, giving $g = T^\top$ at
unit gain, with

$$
\big(T(q)^\top u\big)_i \;=\; u_i - R_i^\top R_{i+1}\,u_{i+1},\qquad u_{n+1}:=0
$$

The second term is **not** a second control — it is Newton's third law: joint
$i{+}1$'s motor reacts on link $i$.

**How little is actually unknown here.** At $n=2$,

$$
g(q)=\begin{pmatrix}I_3 & -R_1^\top R_2\\ 0 & I_3\end{pmatrix}\Gamma
$$

Of the 36 entries, 27 are the constants $0$ and $1$ and the other 9 are exactly
$-R_1^\top R_2$ — a quantity already present in the state. So the entire
learnable content of $g$ is the $3n$ gains. For contrast, the GP this replaced
spent **12 672 weights** rediscovering $T(q)^\top$ from data and reached only
$0.465$–$0.565$ relative error.

The gain mirrors `g_diag` in the single-pendulum environment. Without it $g$
would be fully determined by the state, leaving nothing to identify and making
the arm a strictly *easier* problem than the pendulum. $\gamma_i=(1,1,1)$
recovers pure kinematics exactly.

$\Gamma$ multiplies on the **right** (scaling input channels). Note
$D_{\rm joint}=g\Lambda g^\top$ holds only at $\Gamma=I$: friction opposes the
physical joint rate regardless of motor gain, so $D$ is always built from $T$.

### 4.4 `StructuredSigma` — $\Sigma_\theta(q)$

Selected by `--sigma_mode structured`; `full` uses a $3n\times3$ GP instead.

**Learned** — $3n$ numbers ($6$ at $n=2$):

| symbol | code attribute | shape | count | transform |
|---|---|---|---|---|
| $v_j$ — wind lever vector | `v` | $(n,3)$ | $3n$ | none |

**Given** — $q$, supplying $R_j^\top$ per link.

**Computed** —

$$
\Sigma(q)_j=\Big(A_j^{>}[\ell_j]_\times+a_j[c_j]_\times\Big)R_j^\top
=\big[\underbrace{A_j^{>}\ell_j+a_jc_j}_{=:\,v_j\ \text{learned}}\big]_\times
\underbrace{R_j^\top}_{\text{from }q}
$$

The collapse to a single vector works because $[\ell]_\times$ and $[c]_\times$
are skew and skew matrices are closed under addition — so the geometry factor
has **no $q$-dependence at all**, and every bit of configuration dependence is
the single explicit $R_j^\top$.

Note $\|\Sigma\|_F^2 = 2\sum_j\|v_j\|^2$, so $v$ also carries the wind
amplitude $\sigma$ — there is no separate scale parameter to fit.

### 4.4a Aside: what the GP mode learns instead

For comparison, `--sigma_mode full` builds a `MatrixGP_NLink`. There the
learned objects are variational weights, and $q$ becomes a function input:

$$
q \;\to\; \varphi(q)\in\mathbb{R}^{D_mD_p}
\;\xrightarrow{\ w\in\mathbb{R}^{D_mD_p\times 3n\cdot3}\ }\;
\text{entries of }\Sigma,\qquad w\sim\mathcal N(\mu,\sigma_w^2)
$$

$6\,336$ weights, against $6$ for the structured form, to represent a function
the structured form represents **exactly**.

### 4.5 `V_net` — the one remaining GP

$V_\theta$ stays a variational random-Fourier-feature GP. Gravity picks out the
world $e_z$, so $V$ is genuinely *not* rotation-invariant and has no comparably
compact closed form to exploit.

**This is the only subnet where "what does the GP predict?" has an answer:** it
predicts the scalar potential energy $V_\theta(q)\in\mathbb{R}$ directly, and
that scalar is what gets plugged into $H_\theta$.

**Learned** — with $D_m$ = `--hidden_dim` $=32$ Matérn features and
$D_p = 1+2m_{\max} = 11$ periodic features:

| symbol | code attribute | shape | count |
|---|---|---|---|
| $\mu$ — variational weight means | `w_mean` | $(D_mD_p,\,1) = (352,1)$ | 352 |
| $\log\sigma_w$ — posterior log-widths | `log_w_covar` | $(352,1)$ | 352 |
| feature frequencies / base rotations | `matern.*` | — | 673 |

Total 1 388 trainable numbers, against 23 + 2 + 6 + 6 = 37 for all four
structured subnets combined.

**Given** — $q$ (absolute attitudes; $V$ is not rotation-invariant, so unlike
§4.1–4.3 the raw $R_i$ genuinely matter, not just $R_j^\top R_k$).

**Computed** —

$$
\varphi_f(q)=\sqrt{\tfrac{2}{F}}\,
\cos\Big(\sum_i \theta^{(i)}_f(R_i)\,\varpi^{(i)}_f + b_f\Big),
\qquad
V_\theta(q)=\sum_{m,p}\varphi_m(q)\,\psi_p(q)\,w_{mp}
$$

$$
q \ \xrightarrow{\ \text{features}\ }\ \varphi(q)
\ \xrightarrow[\ w\sim\mathcal N(\mu,\sigma_w^2)\ ]{\ \text{linear}\ }\
V_\theta(q)\in\mathbb{R}
\ \xrightarrow{\ \text{into } H_\theta\ }\ \mathcal T(H_\theta)
$$

$V_\theta$ never enters the dynamics directly — only through
$\mathcal T_i(H_\theta)$, the trivialized gradient of §3.1. That is why an
additive offset $V\to V+c$ is unobservable (gauge (c) of §7).

Weights are drawn by reparameterisation $w=\mu+\sigma_w\epsilon$, **once per
(batch, MC-sample) trajectory** and held fixed across the predictor, the
corrector and every substep — otherwise the Monte-Carlo estimate is not a draw
from $q_\psi(w)$. With `inference_mode=True` the posterior mean $\mu$ is used
instead, which is what makes reported evaluation numbers deterministic.

---

## 5. Integrator — Stratonovich Lie–Heun

One substep of size $h$, from [`lie_integrator_nlink.py:81`](../utils/lie_integrator_nlink.py#L81):

$$
\begin{aligned}
\textbf{(1)}\quad & \xi^{(1)}=M_\theta^{-1}(q)p,\quad
  \dot p^{(1)} = \text{drift}(q,p,u),\quad
  \Delta p^{(1)}=\Sigma_\theta(q)\,dW,\quad \phi^{(1)}_i=\xi^{(1)}_i h\\
\textbf{(2)}\quad & R^{\rm pr}_i=R_i\exp([\phi^{(1)}_i]_\times),\qquad
  p^{\rm pr}=p+\dot p^{(1)}h+\Delta p^{(1)}\\
\textbf{(3)}\quad & \text{re-evaluate at } (q^{\rm pr},p^{\rm pr})
  \text{ reusing the \emph{same} } dW\\
\textbf{(4)}\quad & \bar\phi_i=\tfrac12(\phi^{(1)}_i+\phi^{(2)}_i),\qquad
  R^{+}_i=R_i\exp([\bar\phi_i]_\times),\\
& p^{+}=p+\tfrac h2\big(\dot p^{(1)}+\dot p^{(2)}\big)
        +\tfrac12\big(\Delta p^{(1)}+\Delta p^{(2)}\big)
\end{aligned}
$$

Two properties matter:

- **Averaging happens in the Lie algebra**, followed by one $\exp$ per link, so
  every $R_i$ stays on $SO(3)$ *by construction* — no reprojection, no drift.
  Measured SO(3) defect: $\sim10^{-14}$ over 60 steps.
- **Reusing $dW$ across both stages** is what makes the scheme Stratonovich
  rather than Itô.

Order is $O(h^2)$, verified by halving $h$ and checking the error ratio exceeds
$2.5$ (measured $4.0$–$4.4$ on energy and $J_z$).

The outer step $\Delta t = 0.05$ is subdivided into `n_substeps = 10`, matching
the environment exactly.

---

## 6. Objective

$$
\mathcal L \;=\; \underbrace{\mathcal L_{\rm NLL}}_{\text{rollout}}
\;+\;\lambda_{\rm PL}\underbrace{\mathcal L_{\rm PL}}_{\text{per-increment}}
\;+\;\frac{\beta}{N}\underbrace{\mathcal L_{\rm KL}}_{\text{GP subnets}}
$$

### 6.1 Rollout NLL

Rotations use the concentrated Gaussian on $SO(3)$, **summed over links**:

$$
-\log p(\tilde q\mid\hat q)=\sum_{i=1}^{n}
\Big[\tfrac{\theta_i^2}{2\sigma_R^2}+3\log\sigma_R+\tfrac32\log2\pi\Big],
\qquad \theta_i = \big\|\log(\hat R_i^\top \tilde R_i)\big\|
$$

The $3$ is the *manifold* dimension of $SO(3)$ (the likelihood is over $R$, so
$Z\approx(2\pi\sigma_R^2)^{3/2}$), not the dimension of the scalar $\theta$.

Rates are a plain $3n$-dimensional isotropic Gaussian:

$$
-\log p(\tilde\omega\mid\hat\omega)
=\tfrac{\lVert\Delta\omega\rVert^2}{2\sigma_\omega^2}
+3n\log\sigma_\omega+\tfrac{3n}{2}\log2\pi
$$

The $n$-scaling of both normalisers is what fixes the relative weight of the
rotation head, the rate head and the KL term. $\sigma_R,\sigma_\omega$ are
learned. Torques are sliced off — they are inputs, not predictions.

### 6.2 Pseudo-likelihood — why it is essential

The rollout NLL gives the diffusion **no usable gradient**: model and
environment have independent Brownian paths, so enlarging $\Sigma_\theta$ only
adds variance to the residual and the optimum is $\Sigma_\theta\to0$.
$\mathcal L_{\rm PL}$ instead evaluates a one-step Euler–Maruyama transition
density between consecutive *observed* snapshots:

$$
\Delta\omega_{\rm obs}\ \sim\ \mathcal N\big(\mu(q_t,\omega_t,u_t)\,\Delta t,\ \Sigma_{\rm eff}\big),
\qquad
\Sigma_{\rm eff}=\Delta t\,\Sigma\Sigma^\top+\underbrace{2\sigma_{\rm obs}^2}_{=:s}I_{3n}
$$

The factor $2$ is because $\Delta\omega_{\rm obs}$ differences two
independently-noisy measurements.

**Woodbury.** $\Sigma\in\mathbb{R}^{3n\times3}$, so $\Sigma\Sigma^\top$ has rank
$3$ in $3n$ dimensions — singular alone; the observation floor is what makes
$\Sigma_{\rm eff}$ invertible. Exploiting the low-rank-plus-scaled-identity
structure keeps the cost $O(n)$ rather than $O(n^3)$:

$$
\begin{aligned}
r^\top\Sigma_{\rm eff}^{-1}r
  &=\tfrac1s\Big[\lVert r\rVert^2-(\Sigma^\top r)^\top K^{-1}(\Sigma^\top r)\Big],
  \qquad K=\tfrac{s}{\Delta t}I_3+\Sigma^\top\Sigma\\
\log\det\Sigma_{\rm eff}&=3n\log s+\log\det\big(I_3+\tfrac{\Delta t}{s}\Sigma^\top\Sigma\big)
\end{aligned}
$$

Both reduce to $3\times3$ problems.

The drift $\mu = \dot\omega$ needed here is *not* what the integrator uses:

$$
\dot\omega \;=\; M^{-1}\dot p \;+\; \dot{\overline{(M^{-1})}}\,p,
\qquad
\dot{\overline{(M^{-1})}} = \text{JVP of } M^{-1} \text{ along } \dot q
$$

with $p = M(q)\omega$ recovered by a solve and `stop_gradient` applied to it, so
the observed $\omega$ does not back-propagate a second, spurious path into
`M_net`.

> **Sensitivity.** $\rho_i := \Delta t\,\varsigma_i(\Sigma)^2 / s$ measures how
> much direction $i$ of the diffusion stands above the observation floor. At
> $n=2$, $\sigma_{\rm obs}=0.1$, $\sigma_{\rm wind}=0.5$: $\rho = (3.13, 3.00,
> 0.12)$ — two directions cleanly identifiable, the third buried.

### 6.3 KL

Closed-form mean-field Gaussian, summed over every variational weight:

$$
\mathrm{KL}\big(\mathcal N(\mu,\sigma_w^2)\,\Vert\,\mathcal N(0,1)\big)
=\tfrac12\big(\sigma_w^2+\mu^2-1-2\log\sigma_w\big)
$$

Structured subnets return exactly $0$ — with a handful of physical parameters a
KL is pointless, and an $\mathcal N(0,I)$ prior would be actively *wrong* (the
true $v_j$, $\gamma_i$, $d_i$ are specific $O(1)$ values, not centred at zero).

$\beta$ defaults to $0$, so **out of the box this is not the ELBO** — it is pure
data-fit $\mathcal L_{\rm NLL}+\lambda_{\rm PL}\mathcal L_{\rm PL}$. Set
`--beta_max > 0` to turn the prior on: $\beta$ is held at $0$ for
`--kl_warmup_steps`, then ramped linearly over `--kl_anneal_steps`,

$$
\beta(t)=\beta_{\max}\,\mathrm{clip}\!\Big(\tfrac{t-t_{\rm warm}}{t_{\rm anneal}},\,0,\,1\Big).
$$

Note $\beta=1$ has **no special status** — $\mathcal L_{\rm NLL}$ is a
per-snapshot *mean* while the KL is scaled per-window, so the units do not
match and $\beta$ is an empirical dial ($0.05$–$0.1$ with `--gp_core`). With
`--gp_core` the GP residuals feeding the closed forms are bounded by
$\kappa\tanh(\cdot/\kappa)$ before use — see [`gp_change.md`](gp_change.md).

---

## 7. Gauge freedoms — required for any evaluation

The model is **not** identifiable pointwise. Three exact symmetries:

**(a) Scale.** For any $\beta>0$,

$$
M^{-1}\to\beta M^{-1},\quad V,D,g,\Sigma\to\tfrac1\beta(\cdot),\quad p\to\tfrac1\beta p
$$

leaves every observable trajectory unchanged (and $\Sigma\Sigma^\top\to\beta^{-2}(\cdot)$).
Comparing a learned subnet to ground truth **without** fitting $\beta$ first is
meaningless. `subnet_errors` fits $\beta$ by least squares on $M^{-1}$.

Training **pins** this gauge with `--anchor_trace` (default $4n$): the
learned mass matrix is normalised so $\mathrm{tr}\,M(q)$ equals the anchor at
every $q$. The anchor must live in *function* space: freezing parameters is
not enough, because $M$ is quadratic in the levers, so the gauge has multiple
parameter realisations ($u \to \sqrt\beta u$, $\mathbb{I} \to \beta\mathbb{I}$
with $m$ fixed) — which is exactly how a run with the earlier `--anchor_m1`
pin still ratcheted to $\beta = 23.6$ under the rollout NLL's small-$A$
pressure, with $\Sigma$ chasing a $\sim\!24\times$ target. Any positive
anchor value is gauge-equivalent (a units choice — zero information about
the data). `subnet_errors` still fits $\beta$ as a diagnostic; with the
default anchor and this environment it should report
$\beta \approx \mathrm{tr}\,M_{\rm GT}/4n = 1$. See
[`gp_change.md`](gp_change.md) §3b–§3c.

**(b) Diffusion rotation.** $\Sigma\to\Sigma O$ for any orthogonal $O$ leaves
the law unchanged, so only $\Sigma\Sigma^\top$ is identifiable — harmless, since
that is exactly what enters the likelihood. Note the *column space* of $\Sigma$
is **not** gauged away and is physically meaningful.

**(c) Potential offset.** $H\to H+c$, so $V$ must be centred before comparison.

A useful gauge-**free** diagnostic is the observable diffusion of $\omega$,

$$
\mathrm{Var}(d\omega)/dt = (M^{-1}\Sigma)(M^{-1}\Sigma)^\top
$$

which is invariant under both (a) and (b) and therefore needs no correction.

---

## 8. Training loop

1. Load/generate the dataset; slice into windows of `--num_points`.
2. Build the model at an explicit `dtype`. **This must not be left implicit**:
   the environment enables `jax_enable_x64`, so a default would silently give
   float64 weights against a float32 batch, surfacing far away as a
   primal/tangent dtype mismatch inside the `jax.jvp` in `drift`.
3. Per step: sample a batch, draw one GP weight set per trajectory, draw
   $dW\sim\mathcal N(0, h)$ of shape $(B, S, T, n_{\rm sub}, 3)$, roll out,
   evaluate $\mathcal L$, clip gradients to `--grad_clip`, Adam.
4. Every `--eval_every`, evaluate on the held-out split in **posterior-mean**
   mode (`inference_mode=True`) so reported numbers are deterministic.
5. Save `model.eqx` plus `history.pkl`.

> **Static fields never reach `model.eqx`.** `sigma_mode`, `n`,
> `structured_subnets` etc. are `eqx.field(static=True)`, so
> `tree_deserialise_leaves` restores only arrays and takes everything else from
> the skeleton. A skeleton built with different flags therefore deserialises
> *cleanly* into a numerically different model. Every architecture-affecting
> flag must be recorded in `history.pkl` and mirrored in `load_run`.

---

## 9. Correctness gates

| gate | what it proves |
|---|---|
| [`../../../../mini_tests/test_gt_pH_matches_arm_env.py`](../../../../mini_tests/test_gt_pH_matches_arm_env.py) | the *environment* is the pH system it claims (13 tests: SO(3) preservation, $M$ vs direct KE, gravity three ways, $\Sigma$ vs virtual work, $O(h^2)$ energy and $J_z$ order, passivity, $n{=}1$ reduction to the pendulum, Itô $\equiv$ Stratonovich, power pairing with and without gain) |
| [`eval_ground_truth_match.py`](eval_ground_truth_match.py) | the *model's* algebra and integrator reproduce the environment when every subnet is replaced by analytic physics — rollout agreement $\sim10^{-16}$ |

If both pass, any remaining error is a **learning** problem, not an algebra
problem. That separation is the whole point of keeping
`structured_subnets.py` free of any `envs.` import: the two closed-form
implementations must stay independent, or the gate is vacuous.

---

## 10. Known limitations

- **$\omega$-dependent friction is unrepresentable.** With
  `--varying_friction`, the true multiplier is
  $\rho_i = 1+\tfrac12\cdot\tfrac12(1-(R_ie_z)_z)+\tfrac12\tanh\lVert\omega_i\rVert$.
  `Dw_net(q)` is a function of $q$ alone, so the $\tanh\lVert\omega_i\rVert$
  term cannot be captured. It is nearly saturated at operating speeds
  ($\lVert\omega_i\rVert\approx3\Rightarrow\tanh\approx0.995$), so most of it
  looks like a constant offset — but the low-speed variation is lost.
- **`subnet_errors` evaluates $D_{\rm GT}$ at $\omega=0$.** Harmless with fixed
  friction ($\rho\equiv1$ exactly); with varying friction it compares against a
  reference $\approx1.4\times$ too small and inflates the reported $D$ error.
- **`--relative_inputs` is inert.** The closed forms contain $R_j^\top R_k$
  exactly, so the approximate invariance prior it provided no longer has a
  subnet to act on. Accepted so older commands still run.
- **Pre-structured checkpoints cannot be loaded.** `load_run` refuses them by
  name rather than failing inside equinox.

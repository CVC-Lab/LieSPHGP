# Dzhanibekov Effect — Full Technical Reference

*Personal reference: every formula, file, decision, and fix. Last updated 2026-06-26.*

[TOC]

---

## 0. Repo layout

```
LieSPHGP/
├── envs/
│   ├── tennis_racket_3d.py              # Gymnasium env (frictionless)
│   └── tennis_racket_3d_friction.py     # Stage B env (with friction)
├── datasets/
│   ├── tennis_racket_3d_datagen.py      # Multi-config dataset generator
│   └── tennis_racket_3d_datagen_friction.py
├── src/
│   ├── utils/
│   │   └── loss_utils.py                # geodesic loss, safe logm
│   └── models/
│       ├── 3D_SO3_Windy_Pendulum/ph_nn_ode_v2/
│       │   └── network.py               # shared DissipativeSO3HamNODE base
│       └── 3D_SO3_Tennis_Racket/ph_nn_ode_v2/
│           ├── network_stageC.py        # 18D state, FixedInertiaFromState
│           ├── train.py                 # Stage A (single config, fix_M)
│           ├── train_e2e.py             # Stage A end-to-end
│           ├── train_stageB.py          # Stage B (friction)
│           ├── train_stageC.py          # Stage C (multi-config)
│           ├── eval_stageC_5th.py       # held-out geometry test
│           ├── controller_stageD.py     # PDBodyFrameController
│           ├── controller_stageF.py     # GeometricAttitudeController
│           ├── simulate_stageD.py       # Stage D simulation + K_p scan
│           ├── simulate_stageE.py       # Stage E (A, B, C scenarios)
│           ├── simulate_stageE_v2.py    # Stage E v2 (+ e1+wind, e3+wind)
│           └── simulate_stageF.py       # Stage F (full (R*, ω*) stabilisation)
└── docs/
    ├── hackmd_advisor.md                # ← advisor-facing summary
    └── hackmd_reference.md              # ← this file
```

**Python env**: `/Users/katesur/Projects/LieSPHGP/venv/bin/python3`  
Always run with `-u` for unbuffered output: `python3 -u script.py`

---

## 1. Physics

### 1.1 Euler's equations (body frame)

$$
I\dot{\omega} = -\omega \times (I\omega) + \tau
$$
$$
\dot{R} = R\,\widehat{\omega}
$$

$R\in SO(3)$, $\omega\in\mathbb{R}^3$ body-frame angular velocity, $I = \mathrm{diag}(I_1,I_2,I_3)$, $\widehat{\omega}$ skew-symmetric hat map.

**Hat map**: $\widehat{v} = \begin{pmatrix}0&-v_3&v_2\\v_3&0&-v_1\\-v_2&v_1&0\end{pmatrix}$, so $\widehat{v}\,w = v\times w$.

**Vee map** (inverse): $\mathrm{vee}(\Omega)_i = \Omega_{jk}$ with $(i,j,k)$ cyclic, concretely $\mathrm{vee}(\Omega) = [\Omega_{32}, \Omega_{13}, \Omega_{21}]^\top$.

### 1.2 Principal axis stability (linearisation)

Steady spin about $e_k$ at rate $\omega_k^*$. Perturb in the plane $\{e_i, e_j\}$ with $(i,j,k)$ distinct.

From Euler: $I_i\delta\dot{\omega}_i = (I_j - I_k)\omega_k^*\,\delta\omega_j$, $I_j\delta\dot{\omega}_j = (I_k - I_i)\omega_k^*\,\delta\omega_i$.

Eigenvalues:

$$
\lambda^2 = \frac{(I_j - I_k)(I_k - I_i)}{I_i\, I_j}\,(\omega_k^*)^2
$$

- $\lambda^2 < 0$: $\lambda$ imaginary, **stable** oscillation ($e_1$ and $e_3$ for our racket)
- $\lambda^2 > 0$: $\lambda$ real, **saddle** unstable ($e_2$ — intermediate axis)

For cfg0 at $\omega_2^* = 2\pi$ rad/s: $\lambda = \pm 2.40$ s$^{-1}$, doubling time $0.29$ s.

### 1.3 Tennis racket geometry (cfg0)

```
head semi-axes: a=0.195 m (long, e₁), b=0.135 m (short, e₂)
handle: L=0.32 m, r=0.012 m
total mass: 0.312 kg
```

| Axis | Inertia (kg·m²) | $1/I_i$ (m$^{-2}$·kg$^{-1}$) |
|------|----------------|-----------------------------|
| $e_1$ (long head) | $5.719\times10^{-3}$ | 174.8 |
| $e_2$ (short head) | $1.121\times10^{-2}$ | 89.2 |
| $e_3$ (handle) | $1.322\times10^{-2}$ | 75.6 |

Asymmetry index $(I_3-I_1)/I_2 = 0.67$ — large enough to produce visible flipping at $\omega_2^*=2\pi$ rad/s.

---

## 2. Port-Hamiltonian structure

### 2.1 Hamiltonian

$$
H(R,\omega) = \underbrace{\tfrac{1}{2}\,\omega^\top M^{-1}(R)\,\omega}_{\text{kinetic}} + \underbrace{V(R)}_{\text{potential}}
$$

For a free body: $M^{-1} = \mathrm{diag}(1/I_1,1/I_2,1/I_3)$ (constant), $V = 0$.

### 2.2 Equations of motion in pH form

$$
\dot{x} = [J(x) - D(x)]\nabla_x H + g(R)\,u
$$

The $\omega$-subsystem (expanding the Lie–Poisson bracket):

$$
I\dot{\omega} = -\widehat{\omega}(I\omega) - D_w\,\omega + g\,u
$$

which is exactly Euler's equations with damping $D_w$ and input coupling $g$.

**Gradients**: $\nabla_\omega H = M^{-1}\omega$ (the angular momentum direction). $\nabla_R H$ involves $\partial V/\partial R$ and $\frac{1}{2}\omega^\top\frac{\partial M^{-1}}{\partial R}\omega$ (only matters when $M^{-1}$ or $V$ depend on $R$; zero for the free body).

### 2.3 Why four sub-networks

Every term in $[J - D]\nabla H + gu$ must be learned:

| Term | Network | Notes |
|------|---------|-------|
| $M^{-1}(R)$ | `M_net` | PSD constraint via Cholesky $LL^\top + \epsilon I$ |
| $V(R)$ | `V_net` | potential energy; must be regularised to 0 for free body |
| $D_w(R)$ | `Dw_net` | PSD; learns friction if present; $\approx 0$ for frictionless |
| $g(R)$ | `g_net` | $3\times3$; converges to $I_3$ for direct-torque actuator |

---

## 3. State representations

| Stage | Dim | Layout |
|-------|-----|--------|
| A, B | 15 | $[\text{vec}(R)_9,\, \omega_3,\, u_3]$ |
| C | 18 | $[\text{vec}(R)_9,\, I_1 I_2 I_3,\, \omega_3,\, u_3]$ |

**Inertia embedding**: writing $I_1,I_2,I_3$ directly into the state vector (cols 9:11) lets one model serve all racket geometries. At training and inference time, `FixedInertiaFromState` reads those values and returns the exact physical $M^{-1}$.

**`strip_inertia`**: the loss function expects 15D input. Before computing loss, remove cols 9:11 from the 18D vector. The `split=[9,3,3]` (vec(R), ω, u) is unchanged after stripping.

---

## 4. File-by-file descriptions

### `envs/tennis_racket_3d.py`
Gymnasium env. Lie-group Heun integrator, 10 substeps per `dt=0.05 s`. Observation: `obs = [vec(R)_9, ω_3]` ∈ ℝ¹². Action: body-frame torque `u` ∈ ℝ³ (clipped to ±2 N·m).

Key: `tau = u + disturbance_torque` — so $g = I_3$ exactly and any `disturbance_torque_std` adds mean-zero noise to the applied torque.

`reset(options={"axis": k})` → random $R_0$, spin $\omega_0 = \omega^*_k e_k + \epsilon$.
`reset(options={"R_init": R})` → use specified $R_0$.
`get_inertia_info()` → dict of $I_1, I_2, I_3$, head dims, mass.

### `network_stageC.py`
Defines `DissipativeSO3HamNODE` for 18D state. Contains `FixedInertiaFromState`:

```python
class FixedInertiaFromState(nn.Module):
    """M_net replacement that reads I from state cols 9:12 exactly."""
    def forward(self, q_ext):     # q_ext: (N, 12)  [vec(R), I1, I2, I3]
        return torch.diag_embed(1.0 / q_ext[:, 9:12])  # (N, 3, 3)
```

Activated by passing `fix_M=True` to the model constructor.

### `train_stageC.py`
Training loop for Stage C. Key flags:

| Flag | Effect |
|------|--------|
| `--fix_M` | Use `FixedInertiaFromState` instead of learned `M_net` |
| `--lambda_V_zero 1.0` | Add $\|V_\theta\|^2$ regularisation (needed for e2e) |
| `--total_steps 5000` | Number of gradient steps |
| `--num_points 5` | Window size for windowed geodesic loss |
| `--n_samples 25` | Trajectories per batch |

Checkpoint path: `data/run_tr3d_stageC_fp32/.../tr3d-so3ham-rk4-5p.tar`

### `controller_stageD.py`
`PDBodyFrameController`: proportional ω-only controller.

```python
def __call__(self, R_flat, omega):
    tau = self.Kp * (self.omega_star - np.asarray(omega))
    return np.clip(tau, -self.clip, self.clip)
```

No model used. `use_model_g=False` (default); setting `True` would apply `g_net⁻¹` via `lstsq`, but since $g \approx I_3$ this makes no difference.

### `controller_stageF.py`
`GeometricAttitudeController`: full $(R^*, \omega^*)$ stabiliser.

Key helpers:
```python
def logm_SO3(R):
    """Rodrigues formula; handles θ≈0 and θ≈π edge cases."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-7:
        return (R - R.T) / 2.0           # first-order: near identity
    if abs(theta - np.pi) < 1e-4:        # antipodal case
        sym = (R + np.eye(3)) / 2.0      # = n nᵀ
        i = int(np.argmax(np.diag(sym)))
        n = sym[:, i] / np.sqrt(sym[i, i])
        return theta * hat(n)
    return theta / (2.0 * np.sin(theta)) * (R - R.T)

def vee(Omega):                           # Omega skew-symmetric
    return np.array([Omega[2,1], Omega[0,2], Omega[1,0]])
```

Controller call:
```python
def __call__(self, R_flat, omega):
    e_R = vee(logm_SO3(self.R_star.T @ R_flat.reshape(3,3)))
    e_w = np.asarray(omega) - self.omega_star
    u   = -self.K_R * e_R - self.K_p * e_w
    return np.clip(u, -self.clip, self.clip)
```

When `K_R = 0`: reduces exactly to `PDBodyFrameController`. Confirmed (Stage F Scenario C = Stage D, 0.30 s, 100%).

### `simulate_stageD.py`
K_p scan, linearisation printout. Bug fix history:
- **Bug 1**: overdamped threshold formula `sqrt(a12*a21)` returned NaN (product < 0). Fix: `sqrt(abs(a12*a21))`.
- **Bug 2**: `best_Kp = max(all_results, ...)` picked first (slowest) not fastest. Fix: filter ≥95% conv, then `min(..., key=conv_times.mean)`.

### `simulate_stageE.py` / `simulate_stageE_v2.py`
E.py: Scenarios A (e₁ redirect), B (e₂ hold no wind), C (e₂ hold with wind).
E_v2.py: adds D (e₁ hold with wind), E (e₃ hold with wind).

Key functions:
```python
def kp_max_zoh(I1, dt):
    return 2.0 * I1 / dt          # = 0.229 for cfg0

def kp_min_e2(I1, I2, I3, omega_star):
    return omega_star * np.sqrt(abs(I2 - I3) * abs(I1 - I2))   # = 0.021
```

Bug fix: `scenario_C` verdict checked `Kp=0.10` hardcoded (was 80% conv). Fix: find best-converging Kp with `late_max < 0.5` across the scan.

### `simulate_stageF.py`
Scenarios: A (R*=I, ω*=0), B (R*=Ry(π/4), ω*=0), C (K_R=0 sanity), D (with wind, K_R scan).

```python
def print_linearised_stability(I1, I2, I3, K_R, K_p, dt):
    # Per axis: ωn = sqrt(K_R/I), ζ = K_p / (2*sqrt(K_R*I))
    # ZOH K_p_max = 2*I1/dt,  ZOH K_R_max = K_p/dt
```

Bug fix: first run labeled wind scenario "UNSTABLE" because `late_eR > theta_thr*5`. Changed to detect truly growing errors (2nd-half mean vs initial value) vs bounded SS offset.

---

## 5. Training stage details

### Stage A — `train.py`

```bash
python3 -u train.py --fix_M --lambda_V_zero 0 --total_steps 5000
```

Dataset: `tr3d_dataset_dist0p0_obs_noise0p0_perturb0p05_ncfg4_steps100.pkl`
Shape: `(4, 100, 25, 15)` — 4 configs, 100 timesteps, 25 trajectories, 15D state.

**Loss terms**:
- `geo_loss`: windowed geodesic loss on $SO(3)$
- `V_loss`: MSE of `V_net` output vs 0 (when `--lambda_V_zero` set)
- `M_loss`: MSE of `M_net` output vs exact $M_\text{tgt}$ (only with `--fix_M`)

Result at 5000 steps: `windowed_geo = 2.03e-6`, `V_loss ≈ 0`, `Dw_loss ≈ 0`.

### Stage B — `train_stageB.py`

Dataset with `fric_coeff=0.01`: `tau_fric = -0.01 * omega`.  
`Dw_net` learns dissipation unsupervised.  
Key: friction makes trajectories contractive → near-zero variance → `Dw_loss = 5.13e-14`.

### Stage C — `train_stageC.py`

Four racket geometries trained jointly. Checkpoint naming:
```
run_tr3d_stageC_fp32/obs0_dist0_cfgALL_lP0_lV0_lB0_lD0_lr0p001_s5000_np5_smp25_T100_rk4_seed0_fixM_YYMMDD-HHMMSS/tr3d-so3ham-rk4-5p.tar
```

`strip_inertia(x)`: removes cols 9:12 from shape `(*, 18)` to get `(*, 15)` for loss.

**Identifiability note**: without `--fix_M`, `M_net` output initialises near $\epsilon I \approx 1 \cdot I_3$ (Cholesky output near zero → $L L^\top \approx 0 + \epsilon I$). True targets are $O(100)$ (e.g. $1/I_1 = 174.8$). This is why a high learning rate `lr=0.1` is needed for `M_net` pretraining, not the default `lr=1e-3`.

### Stage C evaluation — `eval_stageC_5th.py`

Evaluates on a 5th geometry not seen during training. Result: `windowed_geo = 2.03e-6`. This is exact because `FixedInertiaFromState` reads the embedded $I$ directly — generalisation is by construction, not by the network having learned geometry.

---

## 6. Control mathematics

### 6.1 IDA-PBC matching equation

For port-Hamiltonian $\dot{x} = [J-D]\nabla H + gu$, find $u$ such that the closed-loop equals $[J_d - D_d]\nabla H_d$:

$$
g\,u = [J_d - D_d]\nabla H_d - [J - D]\nabla H
$$

For $g = I_3$:

$$
u = [J_d - D_d]\nabla H_d - [J - D]\nabla H
$$

**Stage D choice**: $H_d = H + \frac{1}{2}K_p\|\omega-\omega^*\|^2$, $J_d = J$, $D_d = D + K_p I$.

$\nabla_\omega H_d = M^{-1}\omega + K_p(\omega-\omega^*)$. Substituting, the $[J_d-D_d]\nabla H_d$ and $[J-D]\nabla H$ terms cancel except for:

$$
u = K_p(\omega^* - \omega)
$$

**Stage F choice**: additionally add $\frac{1}{2}K_R\|e_R\|^2$ to $H_d$. Since $\nabla_\omega(\|e_R\|^2) \approx 0$ (orientation error doesn't depend on $\omega$ in the body frame), the additional term enters only through the $\nabla_R H_d$ equation. The result:

$$
u = -K_R e_R - K_p(\omega-\omega^*)
$$

### 6.2 ZOH discrete-time analysis

**First-order (ω loop only, Stage D/E)**:

The discrete map per axis $i$ at $T = dt$:

$$
\omega_i[k+1] = \omega_i[k] - (K_p T / I_i)\,(\omega_i[k] - \omega_i^*) = (1 - K_p T/I_i)\omega_i[k] + \ldots
$$

Stability: $|z_i| = |1 - K_p T/I_i| < 1$, i.e. $0 < K_p T/I_i < 2$:

$$
\boxed{K_{p,\max} = \frac{2I_1}{T} = \frac{2\times5.72\times10^{-3}}{0.05} = 0.229 \text{ N·m·s/rad}}
$$

**Second-order (R + ω loop, Stage F)**:

Characteristic polynomial of the ZOH-sampled 2nd-order system:

$$
z^2 - \underbrace{\left(2 - \frac{K_p T}{I_i}\right)}_{\alpha} z + \underbrace{\left(1 - \frac{K_p T}{I_i} + \frac{K_R T^2}{I_i}\right)}_{\beta} = 0
$$

Schur stability conditions: $|\beta| < 1$ and $|\alpha| < 1 + \beta$. The binding condition from $\beta < 1$:

$$
\boxed{K_{R,\max} = K_p / T = 0.10/0.05 = 2.00 \text{ N·m/rad}}
$$

### 6.3 Saddle stabilisation (Stage E)

Closed-loop linearisation about $\omega^* = \omega_2^* e_2$ with $u = K_p(\omega^*-\omega)$:

$$
A_{cl} = \begin{pmatrix} -K_p/I_1 & (I_2-I_3)\omega_2^*/I_1 \\ (I_1-I_2)\omega_2^*/I_3 & -K_p/I_3 \end{pmatrix}
$$

$\text{tr}(A) = -K_p(1/I_1 + 1/I_3) < 0$ always.

$\det(A) = K_p^2/(I_1 I_3) - (I_2-I_3)(I_1-I_2)(\omega_2^*)^2/(I_1 I_3)$

Note: $(I_2-I_3) < 0$ and $(I_1-I_2) < 0$ so the product is **positive**. For $\det > 0$:

$$
\boxed{K_{p,\min} = \omega_2^*\sqrt{|I_2-I_3|\cdot|I_1-I_2|} = 0.021 \text{ N·m·s/rad}}
$$

Empirical value: $K_p = 0.020$ gives ≥95% conv; $K_p = 0.015$ gives 90%.

### 6.4 Steady-state offset under constant wind

**Stage E** ($\omega^* \neq 0$, proportional ω controller): at SS, $\dot{\omega} = 0$:

$$
0 = -K_p(\omega_{ss} - \omega^*) + d \;\Rightarrow\; \|\delta\omega\|_{ss} = |d|/K_p
$$

The orientation $R$ is unconstrained and drifts freely.

**Stage F** ($\omega^* = 0$, attitude+velocity controller): at SS, $\dot{\omega} = 0$ and $\dot{R} = R\widehat{\omega} = 0$:

$$
0 = -K_R e_R - K_p\,0 + d \;\Rightarrow\; \|e_R\|_{ss} = |d|/K_R, \quad \|\omega\|_{ss} = 0
$$

The angular velocity is zero; the body tilts until the orientation restoring torque balances the wind.

Measured ratios $\|e_R\|_\text{actual}/(|d|/K_R)$ at Stage F Scenario D:

| $K_R$ | Ratio |
|-------|-------|
| 0.10 | 1.000 |
| 0.50 | 1.000 |
| 1.00 | 1.000 |
| 1.50 | 1.000 |

Predicted window for $\|e_R\|_{ss} < \theta_\text{thr} = 0.10$ rad with $|d| \approx 0.082$ N·m:

$$
K_R \in \bigl(0.082/0.10,\; 0.10/0.05\bigr) = (0.82,\; 2.00)
$$

Best operating point: $K_R = 1.50$ (middle of window, 95% conv in 1.80 s).

### 6.5 SO(3) geometry for Stage F

**Matrix logarithm** $\log: SO(3) \to \mathfrak{so}(3)$ via Rodrigues:

$$
\log R = \frac{\theta}{2\sin\theta}(R - R^\top), \quad \theta = \arccos\!\left(\frac{\text{tr}R - 1}{2}\right)
$$

Edge cases:
- $\theta \approx 0$: $\log R \approx (R-R^\top)/2$ (first-order expansion)
- $\theta \approx \pi$ (antipodal): $R = 2\mathbf{n}\mathbf{n}^\top - I$ for rotation axis $\mathbf{n}$; reconstruct $\mathbf{n}$ from symmetric part $(R+I)/2 = \mathbf{n}\mathbf{n}^\top$

**Attitude error**: $e_R = \mathrm{vee}(\log(R^{*\top}R)) \in \mathbb{R}^3$; $\|e_R\| = \theta_\text{err} \in [0,\pi]$.

**Almost-global stability**: $\|e_R\| = \pi$ is the only failure point (antipodal configuration). On SO(3) this is a closed set of measure zero; starting from any smooth distribution of initial conditions the probability of hitting it exactly is zero.

---

## 7. Complete results tables

### Stage D — K_p scan (20 trials, axis=1 start, no wind)

| $K_p$ | Conv time (s) | Final $\|\omega-\omega^*\|$ | % conv | Mode |
|-------|--------------|--------------------------|--------|------|
| 0.01 | 2.97 ± 0.xx | 0.002 | 100 | oscillatory |
| 0.05 | 0.65 | ~0 | 100 | oscillatory |
| **0.10** | **0.30** | **~0** | **100** | overdamped ✓ |
| 0.20 | 0.10 | ~0 | 100 | overdamped |

Overdamped threshold: $K_p \geq 0.071$ (from $\text{disc}(A_{cl}) = 0$).

### Stage E — scenarios

| Scenario | Target | Wind | Kp range | Best Kp | % conv | SS ratio |
|----------|--------|------|----------|---------|--------|---------|
| A | $e_1$ redirect | — | 0.10 | 0.10 | 100 | — |
| B | $e_2$ hold | — | 0.010–0.500 | 0.025 | 100 | — |
| C | $e_2$ hold | 0.05 | 0.05–0.50 | 0.20 | 95 | 1.019 |
| D | $e_1$ hold | 0.05 | 0.05–0.50 | 0.20 | 95 | 0.987 |
| E | $e_3$ hold | 0.05 | 0.05–0.50 | 0.20 | 95 | 0.999 |

ZOH instability at $K_p=0.50$: late_max ≈ 14 rad/s on all three axes.

### Stage F — scenarios

| Scenario | $(R^*, \omega^*)$ | $(K_R, K_p)$ | % conv | Conv time | Notes |
|----------|------------------|-------------|--------|-----------|-------|
| A | $(I_3, \mathbf{0})$ | (0.10, 0.10) | 100 | 2.75 s | |
| B | $(R_y(\pi/4), \mathbf{0})$ | (0.10, 0.10) | 100 | 2.82 s | arbitrary target orientation |
| C (sanity) | $(I_3, (0,0,2\pi))$ | (0, 0.10) | 100 | 0.30 s | = Stage D exactly |
| D wind σ=0.05 | $(I_3, \mathbf{0})$ | (1.50, 0.10) | 95 | 1.80 s | ratio=1.000 |

---

## 8. Run commands (all stages)

```bash
VENV=/Users/katesur/Projects/LieSPHGP/venv/bin/python3
MODEL=src/models/3D_SO3_Tennis_Racket/ph_nn_ode_v2

# ── Data ──────────────────────────────────────────────────────────────────────
$VENV datasets/tennis_racket_3d_datagen.py \
    --ncfg 4 --timesteps 100 --trials 25 \
    --perturb_std 0.05 --dist_std 0.0 --obs_noise 0.0

# ── Stage A ───────────────────────────────────────────────────────────────────
$VENV -u $MODEL/train.py --fix_M --total_steps 5000

# ── Stage A e2e ───────────────────────────────────────────────────────────────
$VENV -u $MODEL/train_e2e.py --lambda_V_zero 1.0 --total_steps 5000

# ── Stage B ───────────────────────────────────────────────────────────────────
$VENV -u $MODEL/train_stageB.py --fix_M --total_steps 5000

# ── Stage C ───────────────────────────────────────────────────────────────────
$VENV -u $MODEL/train_stageC.py --fix_M --total_steps 5000 \
    --num_points 5 --n_samples 25

# ── Stage C eval (held-out 5th geometry) ─────────────────────────────────────
$VENV -u $MODEL/eval_stageC_5th.py

# ── Stage D ───────────────────────────────────────────────────────────────────
$VENV -u $MODEL/simulate_stageD.py --Kp_scan "0.01,0.05,0.10,0.20"

# ── Stage E ───────────────────────────────────────────────────────────────────
$VENV -u $MODEL/simulate_stageE.py                              # A, B, C
$VENV -u $MODEL/simulate_stageE_v2.py                           # A–E incl. wind

# ── Stage F ───────────────────────────────────────────────────────────────────
$VENV -u $MODEL/simulate_stageF.py
$VENV -u $MODEL/simulate_stageF.py --n_steps 400 --theta_thr 0.05   # stricter
```

---

## 9. Known issues / gotchas

**`M_net` init too small**: Cholesky-based PSD output initialises near $\epsilon I$ (values $\approx 10^{-3}$). True $1/I_i \approx O(100)$. Need `lr=0.1` for `M_net` in standalone pretraining; `--fix_M` sidesteps this entirely.

**`split=[9,3,3]` unchanged in Stage C**: The loss `split` always describes the 15D stripped state (vec(R), ω, u). Never include the embedded inertia in `split`.

**`env._disturbance_torque`**: This attribute stores the constant disturbance sampled at each `env.reset()`. Use it to compute the per-trial disturbance norm for the SS offset verification.

**Stage F: `K_R=0.01` gives `BOUNDED_SS` not `CONVERGED`**: With initial $\|e_R\|\approx 2.2$ rad and $\omega_n = 1.32$ rad/s, the convergence time to $\|e_R\|<0.10$ rad is $\approx 2.2/1.32 \times 2\zeta = 2.2/1.32 \times 13.2 \approx 22$ s. Our 15 s window is too short. Run with `--n_steps 600` to confirm convergence.

**`simulate_stageF.py`: `scenario_D_wind` creates a fresh `Env` per $K_R$ value** with the same seed, ensuring all $K_R$ values see identical disturbance vectors — necessary for the clean ratio comparison.

---

## 10. Next possible directions

- **Stage F with `g_net⁻¹`**: For non-trivial input coupling (e.g. reaction wheels), $u = g_\theta(R)^{-1}(-K_R e_R - K_p e_\omega)$ would use the learned model at control time. Currently redundant because $g \approx I_3$.
- **Model-predictive control (MPPI/iLQR)**: Use Stage C neural ODE as a differentiable rollout model for trajectory optimisation.
- **Apply to the Windy Pendulum**: Port Stages D–F controllers to the 3D SO(3) pendulum, which has a non-trivial gravity potential $V(R) \neq 0$ — the learned $V_\text{net}$ would directly enter the IDA-PBC matching equation.
- **Continuous-time robust control**: The ZOH instability suggests that designing in continuous time then discretising is unsafe for this class of systems. A proper discrete-time IDA-PBC design would be more principled.

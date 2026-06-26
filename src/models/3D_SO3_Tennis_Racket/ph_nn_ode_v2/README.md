# Port-Hamiltonian Neural ODE on SO(3) — Tennis Racket / Dzhanibekov Effect

A physics-structured neural ODE that learns the Hamiltonian and dissipation of a free
rigid body from trajectory data, then uses that learned structure to stabilise the
system with an IDA-PBC controller.

---

## 1. The physics

### 1.1 Euler's equations for a free rigid body

A rigid body rotating freely in space obeys Euler's equations in the body frame:

```
I ω̇  =  −ω × (I ω)  +  τ
Ṙ   =  R · hat(ω)
```

where `I = diag(I₁, I₂, I₃)` is the inertia tensor, `ω ∈ ℝ³` is the angular
velocity in the body frame, `R ∈ SO(3)` is the orientation, and `τ ∈ ℝ³` is an
applied body-frame torque.

### 1.2 The Dzhanibekov (intermediate-axis) theorem

The three principal axes have a fundamental stability difference:

| Axis | Inertia | Open-loop stability | Notes |
|------|---------|---------------------|-------|
| e₁ (long head) | I₁ = 0.0057 kg·m² | **stable** (oscillations) | smallest I |
| e₂ (short head) | I₂ = 0.0112 kg·m² | **UNSTABLE** (saddle) | intermediate I |
| e₃ (handle) | I₃ = 0.0132 kg·m² | **stable** (oscillations) | largest I |

Rotation about e₂ is linearly unstable: the linearised free-body eigenvalue is
`λ = ±ω₂ √[(I₂−I₃)(I₁−I₂)/(I₁I₃)]`, which is real positive for intermediate I.
For cfg0 at ω₂ = 2π rad/s: `λ = ±2.40 s⁻¹` (doubling time ≈ 0.29 s).

### 1.3 Port-Hamiltonian structure

The system is cast as a port-Hamiltonian system on SO(3) × ℝ³:

```
H(R, ω) = ½ ωᵀ M⁻¹(R) ω  +  V(R)
```

- `M⁻¹(R)` — effective inverse-inertia (PSD matrix, learned by `M_net`)
- `V(R)` — potential energy (zero for free body in space, learned by `V_net`)
- `Dw(R)` — dissipation matrix (zero for frictionless body, learned by `Dw_net`)
- `g(R)` — input coupling matrix (= I₃ for direct body-frame torque, learned by `g_net`)

The dynamics in port-Hamiltonian form:

```
I ω̇  =  J(R,ω) ∇_ω H  −  Dw ∇_ω H  +  g u
Ṙ   =  R · hat(ω)
```

where `J` is the structure matrix encoding gyroscopic coupling.

---

## 2. Model architecture

### 2.1 State representation

| Stage | Dimension | Layout |
|-------|-----------|--------|
| A, B  | 15D | `[vec(R)₉, ω₃, u₃]` |
| C     | 18D | `[vec(R)₉, I₁I₂I₃₃, ω₃, u₃]` — inertia embedded in state |

The inertia is embedded in the state vector in Stage C so that one model generalises
across multiple racket geometries at inference time.

### 2.2 Sub-networks

All sub-networks take `q_ext = [vec(R), I₁, I₂, I₃]` (12D input in Stage C).

| Network | Output | Constraint | Notes |
|---------|--------|-----------|-------|
| `M_net` | 3×3 matrix | PSD via `L Lᵀ + ε I` | inverse inertia `M⁻¹` |
| `V_net` | scalar | — | potential energy |
| `Dw_net` | 3×3 matrix | PSD | dissipation |
| `g_net` | 3×3 matrix | — | input coupling |

### 2.3 `FixedInertiaFromState` (Stage C)

Instead of training `M_net` to learn the inertia from SO(3) geometry, we pin it to the
exact physical value read from the embedded state:

```python
class FixedInertiaFromState(nn.Module):
    def forward(self, q_ext):           # q_ext: (N, 12)
        return torch.diag_embed(1.0 / q_ext[:, 9:12])   # (N, 3, 3)
```

This resolves the `M → αM` scale invariance (Hamiltonian degeneracy) that arises in
free-body training and makes multi-config generalisation exact by construction.

### 2.4 Integration

The neural ODE is integrated using `torchdiffeq` RK4 (`method='rk4'`).
The physical environment uses a Lie-group Heun integrator with 10 substeps per
`dt = 0.05 s` step.

---

## 3. Training stages

### Stage A — single-config, torque-free (`train.py`)

Train on one racket geometry with zero applied torque.
`--fix_M` pins M_net to the exact inertia (avoids `M→αM` degeneracy).

**Result** — windowed geodesic loss = 2.03×10⁻⁶ (threshold 0.01 rad²).  
Key insight: without `--lambda_V_zero`, the optimizer finds a `(M_wrong, V_compensating)`
saddle — V absorbs what M misses. Force `V → 0` to recover physical M.

---

### Stage A end-to-end (`train_e2e.py`)

Full end-to-end training (M, V, Dw, g jointly).
Uses `--lambda_V_zero 1.0` to break the (M, V) Hamiltonian degeneracy.

---

### Stage B — friction (`train_stageB.py`)

Add physical friction `τ_fric = −fric_coeff · ω` to the dataset.
`Dw_net` learns the dissipation: `Dw_loss` converges to 5.13×10⁻¹⁴ (10 orders of
magnitude below threshold). Friction makes the system contractive, so full-trajectory
variance collapses to near zero.

---

### Stage C — multi-config generalisation (`train_stageC.py`, `network_stageC.py`)

Train jointly on 4 racket geometries with different inertia values.
State extended to 18D: `[vec(R)₉, I₁I₂I₃₃, ω₃, u₃]`.

Key design decisions:
- `FixedInertiaFromState` reads `I` from state cols 9:12, returns `diag(1/I)`.
- `strip_inertia(x)` removes cols 9:12 before passing 18D state to loss.
- `split = [9, 3, 3]` unchanged — loss sees 15D after stripping.
- Per-sample `M_tgt` computed from embedded inertia in `subnet_diagnostics_stageC.py`.

**Result** — windowed geo = 2.03×10⁻⁶ across all 4 configs.  
Held-out 5th geometry (different head/handle dimensions): same geo = 2.03×10⁻⁶.
Generalisation is exact because `FixedInertiaFromState` reads the exact I from state.

Run:
```bash
venv/bin/python3 train_stageC.py --fix_M --total_steps 5000
```

Checkpoint saved to `data/run_tr3d_stageC_fp32/`.

---

### Stage D — IDA-PBC stabilisation (`controller_stageD.py`, `simulate_stageD.py`)

Stabilise spinning about **e₃** (handle axis) starting from Dzhanibekov tumbling (e₂).

Because the environment applies torque directly (`τ_total = u + τ_disturbance`), the
true input coupling is `g = I₃` exactly. The IDA-PBC law simplifies to:

```
u = K_p · (ω* − ω)       ω* = (0, 0, 2π) rad/s
```

Linearised stability about ω* = ω*₃ e₃ (overdamped for K_p ≥ 0.071):

| Mode | Eigenvalue | Time constant |
|------|-----------|---------------|
| δω₃ | −K_p/I₃ = −7.6 s⁻¹ | 0.13 s |
| δω₁,₂ (overdamped) | −10.2, −16.2 s⁻¹ | 0.06–0.10 s |

**Result** (20 trials, no disturbance):

| K_p | Conv time | Final ‖ω−ω*‖ |
|-----|-----------|--------------|
| 0.01 | 2.97 s | 0.002 rad/s |
| 0.05 | 0.65 s | ~0 |
| **0.10** | **0.30 s** | **~0** (recommended) |
| 0.20 | 0.10 s | ~0 |

Run:
```bash
venv/bin/python3 simulate_stageD.py --Kp_scan "0.01,0.05,0.10,0.20"
```

---

### Stage E — e₁, e₂ hold, wind robustness (`simulate_stageE.py`)

Three scenarios using the same `u = K_p(ω*−ω)` controller:

**Scenario A** — redirect from e₂ tumbling → **e₁** (no wind):  
100% convergence in 0.25 s with K_p = 0.10. e₁ is naturally stable; trivial.

**Scenario B** — hold **e₂** (saddle, no wind), K_p scan:  
The saddle instability requires active feedback to counteract:

```
K_p_min = ω*₂ · √(|I₂−I₃| · |I₁−I₂|)  =  0.021 N·m·s/rad   (theory)
                                           ≤ 0.020              (empirical)
```

ZOH stability bound — with control update period T = dt = 0.05 s, the discrete
eigenvalue of the I₁ mode is `z = 1 − K_p T/I₁`. For stability need |z| < 1:

```
K_p_max = 2·I₁/T = 2 × 0.0057/0.05 = 0.229 N·m·s/rad
```

K_p = 0.50 violates this: z = −3.39, final error grows to 13.7 rad/s. Confirmed.

Safe operating range for e₂ hold: **K_p ∈ (0.021, 0.229)**.

**Scenario C** — hold **e₂** with stochastic wind (σ = 0.05 N·m):  
Constant-per-episode disturbance `d ∼ N(0, σ² I₃)` sets a steady-state offset:

```
‖δω_ss‖  ≈  |d| / K_p           (linear prediction)
```

Measured ratio (final error / predicted SS) at K_p = 0.20: **1.019** — clean.  
The wind further narrows the viable K_p window:

```
K_p > |d| / ε_conv = 0.084 / 0.5 = 0.168     (wind-offset requirement)
K_p < K_p_max = 0.229                         (ZOH stability)
→ viable range: (0.17, 0.23)
```

K_p = 0.20 achieves 95% convergence. PASS.

**Scenario D** — hold e₁ with wind (σ = 0.05 N·m):  
e₁ is stable (free-body λ = ±3.31i), so K_p_min = 0. With wind the binding lower
constraint is K_p > |d|/ε_conv ≈ 0.17. K_p = 0.20 achieves 95% convergence, SS
error ratio = 0.987 (nearly perfect linear prediction).

**Scenario E** — hold e₃ with wind (σ = 0.05 N·m):  
e₃ is stable (free-body λ = ±3.05i), same analysis. K_p = 0.20 achieves 95%
convergence, SS error ratio = 0.999 — the most accurate ratio across all axes.

The ZOH bound K_p_max = 2·I₁/T = 0.229 is **the same for all three axes** because
it is set by the smallest inertia I₁, not by which axis you are holding.

SS offset ratios across all axes at K_p = 0.20:

| Scenario | Target | SS error / (|d|/K_p) | Note |
|----------|--------|----------------------|------|
| C | e₂ (unstable) | 1.019 | wind + saddle narrowing |
| D | e₁ (stable) | 0.987 | wind only |
| E | e₃ (stable) | 0.999 | wind only, cleanest |

Run:
```bash
venv/bin/python3 simulate_stageE.py        # original three scenarios (A, B, C)
venv/bin/python3 simulate_stageE_v2.py     # all five scenarios (A–E)
```

---

### Stage F — Geometric attitude control (`controller_stageF.py`, `simulate_stageF.py`)

Extends Stage D to stabilise the **full state (R*, ω*) on SO(3) × ℝ³**, not just ω*.
Stage D/E could drive ω → ω* but had no mechanism to control the orientation R.

Control law:
```
u = −K_R · e_R  −  K_p · e_ω

e_R = vee( logm(R*ᵀ R) )   ∈ ℝ³    geodesic attitude error
e_ω = ω − ω*               ∈ ℝ³    angular velocity error
```

`vee` extracts the axial vector of a skew-symmetric matrix; `logm` is the matrix
logarithm on SO(3) implemented via the Rodrigues formula.  When `K_R = 0` this
reduces exactly to the Stage D proportional controller.

**IDA-PBC interpretation**: desired Hamiltonian
`H_d = H + ½K_R ‖e_R‖² + ½K_p ‖e_ω‖²`, J_d = J (unchanged), R_d = R + K_p I
(added damping).

**Linearised closed-loop** (per axis i — independent 2nd-order oscillator):

| Quantity | Formula | K_R=0.10, K_p=0.10, I=I₁ |
|----------|---------|--------------------------|
| Natural frequency | ωn = √(K_R/Iᵢ) | 4.18 rad/s |
| Damping ratio | ζ = K_p/(2√(K_R·Iᵢ)) | 2.09 (overdamped) |

**ZOH bounds** (from discrete-time Schur stability):
```
K_p < 2·I₁/T = 0.229   (velocity loop — same as Stage D/E)
K_R < K_p/T  = 2.000   (attitude loop — 2nd-order ZOH constraint)
```

**Almost-global stability**: fails only at the antipodal set {‖e_R‖ = π},
which has measure zero on SO(3).

**Results** (20 trials, Dzhanibekov start, random R₀):

| Scenario | Target (R*, ω*) | K_R | Conv time | Notes |
|----------|-----------------|-----|-----------|-------|
| A | I₃, 0 | 0.10 | 2.75 s | 100% ✓ |
| B | Ry(π/4), 0 | 0.10 | 2.82 s | 100% ✓, arbitrary orientation |
| C | I₃, (0,0,2π) | 0 | 0.30 s | 100% ✓, exact Stage D match |
| D (wind σ=0.05) | I₃, 0 | 1.50 | 1.80 s | 95% ✓, SS ‖e_R‖=0.055 rad |

**Orientation–velocity duality under wind** (new finding in Stage F):

With ω* = 0, constant wind d ≠ 0 is balanced in steady state by the **orientation**
restoring torque, not the velocity damping:

```
K_R · e_R = d   →   ‖e_R‖_ss = |d| / K_R      (velocity → 0 exactly)
```

Compare with Stage E (ω* ≠ 0): `‖e_ω‖_ss = |d| / K_p` (orientation unconstrained).
The SS ratio ‖e_R‖_actual / (|d|/K_R) = **1.000** for all tested K_R ∈ [0.10, 1.50].

Viable K_R window with wind to satisfy ‖e_R‖ < θ_thr:
```
K_R_min(wind) = |d| / θ_thr  ≈ 0.082/0.10 = 0.82   (SS offset requirement)
K_R_max(ZOH)  = K_p / T             = 2.00   (discrete stability)
→ viable range: (0.82, 2.00)    best: K_R = 1.50
```

Run:
```bash
venv/bin/python3 simulate_stageF.py
```

---

## 4. Summary of stage results

| Stage | Task | Key metric | Result |
|-------|------|-----------|--------|
| A | Single-config dynamics, fix_M | windowed geo | 2.03×10⁻⁶ ✓ |
| A e2e | End-to-end (M,V,Dw,g) | windowed geo | converges with `--lambda_V_zero` |
| B | Friction identification | Dw_loss | 5.13×10⁻¹⁴ ✓ |
| C | 4-config + 5th held-out | windowed geo | 2.03×10⁻⁶ on all ✓ |
| D | e₃ stabilisation, no wind | conv time | 0.30 s at K_p=0.10 ✓ |
| E-A | e₁ redirect (no wind) | conv time | 0.25 s ✓ |
| E-B | e₂ hold, no wind | K_p_min | 0.020 ≈ theory 0.021 ✓ |
| E-C | e₂ hold, σ=0.05 wind | 95% conv | K_p=0.20, ratio=1.019 ✓ |
| E-D | e₁ hold, σ=0.05 wind | 95% conv | K_p=0.20, ratio=0.987 ✓ |
| E-E | e₃ hold, σ=0.05 wind | 95% conv | K_p=0.20, ratio=0.999 ✓ |
| F-A | rest at R*=I₃, no wind | 95% conv | K_R=0.10, 2.75s ✓ |
| F-B | rest at R*=Ry(π/4), no wind | 95% conv | K_R=0.10, 2.82s ✓ |
| F-C | sanity check K_R=0 | matches Stage D | 0.30s ✓ |
| F-D | rest at R*=I₃, σ=0.05 wind | 95% conv | K_R=1.50, ratio=1.000 ✓ |

---

## 5. Key non-obvious findings

**Hamiltonian degeneracy (Stage A)**: For a free body `V = 0` is exact physics,
but the optimizer finds a `(M_wrong, V_compensating)` saddle: `M_loss` diverges while
`geo_loss` converges. Fix: add `--lambda_V_zero 1.0` to regularise `V → 0`.

**M → αM scale invariance**: The free-body Hamiltonian `H = ½ωᵀM⁻¹ω` is invariant
under `M → αM` because the conserved dynamics only depend on ratios of inertia
components. `FixedInertiaFromState` resolves this by anchoring M to absolute physical
values read from the state.

**ZOH discrete instability** (Stage E): The controller computes `u` once per
`dt = 0.05 s` step (zero-order hold). The discrete-time eigenvalue for the lightest axis
is `z = 1 − K_p T/I₁`. For stability: `K_p < 2I₁/T = 0.229`. K_p = 0.50 gives
`z = −3.39` — the error grows with alternating sign each step, reaching 14 rad/s.

**Narrow viable window for e₂ + wind**: Two competing constraints,
`K_p > |d|/ε_conv` (wind rejection) and `K_p < 2I₁/T` (ZOH), leave a window of
width ≈ 0.06 N·m·s/rad for the default parameters. Reducing dt or increasing I₁
(heavier/larger racket head) widens it.

**Universal ZOH bound across all axes**: K_p_max = 2·I₁/T = 0.229 is the same for
e₁, e₂, and e₃ hold. It is set by I₁ (the smallest inertia — the mode that
reaches instability first), regardless of which axis is the target. Confirmed:
K_p = 0.50 diverges on all three axes with late_max ≈ 14 rad/s.

**SS offset ratio is universal**: The steady-state formula ‖δω_ss‖ ≈ |d|/K_p holds
across all three axes (Stage E).

**Orientation–velocity duality under wind (Stage F)**: With ω* ≠ 0 (Stage E), constant
wind d creates a persistent *velocity* offset ‖e_ω‖_ss = |d|/K_p. With ω* = 0
(Stage F), the body comes to rest but settles at a tilted orientation: the wind torque
is balanced by the orientation restoring torque K_R · e_R, giving ‖e_R‖_ss = |d|/K_R.
The ratio is exactly 1.000 across all K_R values (0.10–1.50 N·m/rad), confirming the
linear prediction is tight in orientation space as well as velocity space.

**ZOH bound for the attitude loop (Stage F)**: Adding orientation feedback creates a
2nd-order discrete system with an additional ZOH constraint: K_R < K_p/T.  For
K_p=0.10, dt=0.05: K_R_max = 2.00 N·m/rad.  Viable wind-rejection window:
K_R ∈ (|d|/θ_thr, K_p/T) ≈ (0.82, 2.00).

---

## 6. File index

| File | Description |
|------|-------------|
| `network_stageC.py` | `DissipativeSO3HamNODE` with 18D state; `FixedInertiaFromState` |
| `train.py` | Stage A training (single-config, torque-free) |
| `train_e2e.py` | Stage A end-to-end (joint M, V, Dw, g) |
| `train_stageB.py` | Stage B (friction) |
| `train_stageC.py` | Stage C (multi-config, 18D state) |
| `subnet_diagnostics_stageC.py` | Per-sample subnet MSE with embedded inertia |
| `eval_stageC_5th.py` | Held-out 5th geometry generalisation test |
| `controller_stageD.py` | `PDBodyFrameController`: u = K_p(ω*−ω) |
| `simulate_stageD.py` | Stage D closed-loop simulation + K_p scan |
| `simulate_stageE.py` | Stage E original — e₁ redirect, e₂ hold no-wind, e₂ hold wind |
| `simulate_stageE_v2.py` | Stage E v2 — adds e₁+wind (Scenario D) and e₃+wind (Scenario E) |
| `controller_stageF.py` | `GeometricAttitudeController`: `hat`, `vee`, `logm_SO3`, `expm_SO3` |
| `simulate_stageF.py` | Stage F — full-state (R*, ω*) stabilisation; 4 scenarios |

Related files outside this directory:

| File | Description |
|------|-------------|
| `envs/tennis_racket_3d.py` | Gymnasium env: Lie-group Heun integrator, direct-torque actuator |
| `envs/tennis_racket_3d_friction.py` | Stage B env with friction |
| `datasets/tennis_racket_3d_datagen.py` | Dataset generator (multi-config) |
| `src/models/3D_SO3_Windy_Pendulum/ph_nn_ode_v2/network.py` | Shared `DissipativeSO3HamNODE` base |
| `src/utils/loss_utils.py` | `rotmat_L2_geodesic_loss_safe`, `traj_rotmat_L2_geodesic_loss_safe` |

---

## 7. How to run

All commands assume the project virtual environment:

```bash
PYTHON=/Users/katesur/Projects/LieSPHGP/venv/bin/python3
```

### Generate dataset

```bash
$PYTHON datasets/tennis_racket_3d_datagen.py \
    --ncfg 4 --timesteps 100 --trials 25 \
    --perturb_std 0.05 --dist_std 0.0 --obs_noise 0.0
```

Output: `data/tennis_data/tr3d_dataset_dist0p0_obs_noise0p0_perturb0p05_ncfg4_steps100.pkl`

### Stage C training

```bash
cd src/models/3D_SO3_Tennis_Racket/ph_nn_ode_v2
$PYTHON -u train_stageC.py --fix_M --total_steps 5000 \
    --num_points 5 --n_samples 25 | tee train_stageC.log
```

Important flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--fix_M` | off | Pin M_net to FixedInertiaFromState |
| `--lambda_V_zero` | 0 | Weight for V→0 regularisation (set 1.0 for e2e) |
| `--total_steps` | 5000 | Training steps |
| `--num_points` | 5 | Window size for windowed loss |

### Stage D / E simulation

```bash
# K_p scan for e₃ stabilisation
$PYTHON -u simulate_stageD.py --Kp_scan "0.01,0.05,0.10,0.20"

# Full Stage E (e₁, e₂ hold, wind)
$PYTHON -u simulate_stageE.py

# Custom convergence threshold or longer run
$PYTHON -u simulate_stageE.py --n_steps 400 --conv_thr 0.25 --wind_std 0.10
```

### Stage E simulation (all five scenarios)

```bash
$PYTHON -u simulate_stageE_v2.py

# Override wind strength or run length
$PYTHON -u simulate_stageE_v2.py --n_steps 400 --wind_std 0.10
```

### Stage F simulation (geometric attitude control)

```bash
$PYTHON -u simulate_stageF.py

# Longer run or tighter attitude threshold
$PYTHON -u simulate_stageF.py --n_steps 400 --theta_thr 0.05
```

### Evaluate held-out geometry

```bash
$PYTHON -u eval_stageC_5th.py   # auto-finds latest checkpoint
```

---

## 8. Extended documentation

Two HackMD-formatted documents in `docs/` cover this work in depth:

| File | Audience | Contents |
|------|----------|----------|
| `docs/hackmd_advisor.md` | Advisor / professor | Full LaTeX derivations, results tables, colored callouts; suitable for a technical overview meeting |
| `docs/hackmd_reference.md` | Future self | Every formula with derivation, every file, every design decision, every bug and fix, complete run commands |

---

## 9. Requirements

```
Python     3.10+
PyTorch    2.12+
torchdiffeq
gymnasium
numpy
```

Install into the project venv:
```bash
pip install torch torchdiffeq gymnasium numpy
```

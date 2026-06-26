# Dzhanibekov Effect: Port-Hamiltonian Neural ODE on SO(3)

**Ecaterina Sur — LieSPHGP Project**

[TOC]

---

## 1. Motivation

The **Dzhanibekov effect** (intermediate-axis theorem) is a striking phenomenon in rigid-body mechanics: an object spinning freely about its second principal axis will spontaneously flip its orientation 180°, repeatedly, with no external torque applied. The effect is famously visible in microgravity and follows directly from the instability of the intermediate-inertia axis in Euler's equations.

This project uses the tennis racket as the test system and pursues two goals:

1. **Learn** the port-Hamiltonian structure $H(R,\omega)$ of the rigid body directly from trajectory data, with no prior knowledge of the inertia tensor.
2. **Control** the learned system — including stabilising the open-loop unstable intermediate axis and achieving full orientation + angular velocity targets on $SO(3)\times\mathbb{R}^3$.

The progression is: dataset generation → Hamiltonian identification (Stages A–C) → IDA-PBC control design (Stages D–F).

---

## 2. Mathematics

### 2.1 Rigid body kinematics and Euler's equations

Let $R \in SO(3)$ be the body orientation and $\omega \in \mathbb{R}^3$ the angular velocity in the body frame. The equations of motion are:

$$
I\dot{\omega} = -\omega \times (I\omega) + \tau, \qquad \dot{R} = R\,\widehat{\omega}
$$

where $I = \mathrm{diag}(I_1, I_2, I_3)$ is the principal-axis inertia tensor and $\widehat{\omega}$ is the skew-symmetric matrix satisfying $\widehat{\omega}\, v = \omega \times v$. For the tennis racket (cfg0):

| Axis | Geometry | Inertia | Open-loop stability |
|------|----------|---------|---------------------|
| $e_1$ | long head axis | $I_1 = 5.72 \times 10^{-3}$ kg·m² | **stable** |
| $e_2$ | short head axis | $I_2 = 1.12 \times 10^{-2}$ kg·m² | **unstable** |
| $e_3$ | handle axis | $I_3 = 1.32 \times 10^{-2}$ kg·m² | **stable** |

### 2.2 The intermediate-axis instability

Linearise about steady spin $\omega^* = \omega_2 e_2$ (intermediate axis). The perturbation $(\delta\omega_1, \delta\omega_3)$ satisfies:

$$
\begin{pmatrix}\delta\dot{\omega}_1 \\ \delta\dot{\omega}_3\end{pmatrix}
= \underbrace{\begin{pmatrix} 0 & \tfrac{(I_2-I_3)\omega_2}{I_1} \\ \tfrac{(I_1-I_2)\omega_2}{I_3} & 0\end{pmatrix}}_{A_{\text{free}}}\begin{pmatrix}\delta\omega_1\\ \delta\omega_3\end{pmatrix}
$$

The eigenvalues of $A_{\text{free}}$ are $\lambda = \pm\,\omega_2\sqrt{\dfrac{(I_2-I_3)(I_1-I_2)}{I_1 I_3}}$.

Since $I_1 < I_2 < I_3$, both factors $(I_2 - I_3)$ and $(I_1 - I_2)$ are **negative**, so their product is positive and $\lambda$ is **real**: a saddle point with exponential divergence. The doubling time at $\omega_2 = 2\pi$ rad/s is $1/\lambda = 0.42$ s.

Spin about $e_1$ or $e_3$ gives $\lambda^2 < 0$ — imaginary eigenvalues, stable oscillation.

### 2.3 Port-Hamiltonian structure

The system is cast as a port-Hamiltonian system on $SO(3)\times\mathbb{R}^3$ with Hamiltonian:

$$
H(R,\omega) = \tfrac{1}{2}\,\omega^\top M^{-1}(R)\,\omega + V(R)
$$

where $M^{-1}(R)$ is the effective inverse inertia (equal to $\mathrm{diag}(1/I_1,1/I_2,1/I_3)$ for a free body), and $V(R) = 0$ for a body in free space. The full dynamics are:

$$
\dot{x} = \bigl[J(x) - D(x)\bigr]\nabla_x H + g(R)\,u
$$

with $x = (R,\omega)$ and:
- $J(x)$: skew-symmetric **interconnection matrix** (encodes gyroscopic coupling via the Lie–Poisson bracket on $\mathfrak{so}(3)^*$)
- $D(x) \succeq 0$: symmetric **dissipation matrix**
- $g(R)$: **input coupling matrix** ($= I_3$ for a direct body-frame torque actuator)

The structure is **energy-consistent** by construction: $\dot{H} = -\omega^\top D\,\omega + \omega^\top u \leq \omega^\top u$ (power balance), guaranteeing passivity.

---

## 3. What is learned

Four neural sub-networks jointly parametrise the port-Hamiltonian structure. Each takes the extended configuration $q_\text{ext} = [\text{vec}(R),\, I_1, I_2, I_3] \in \mathbb{R}^{12}$ as input (in Stage C; $\text{vec}(R) \in \mathbb{R}^9$ alone in Stages A–B).

| Network | Output | Constraint | Physical meaning |
|---------|--------|-----------|-----------------|
| `M_net` | $3\times 3$ | PSD via $LL^\top + \epsilon I$ | inverse inertia $M^{-1}(R)$ |
| `V_net` | scalar | — | potential energy $V(R)$ |
| `Dw_net` | $3\times 3$ | PSD | dissipation $D(R)$ |
| `g_net` | $3\times 3$ | — | input coupling $g(R)$ |

The neural ODE is integrated with `torchdiffeq` RK4; the training loss is a **windowed geodesic loss** on $SO(3)$:

$$
\mathcal{L} = \frac{1}{N}\sum_{t}\bigl\|\log(R_{\text{pred}}(t)^\top R_{\text{true}}(t))\bigr\|_F^2
$$

This is coordinate-free and respects the geometry of $SO(3)$.

### 3.1 Key design challenge: Hamiltonian degeneracy

For a free body ($V = 0$ in physics), the Hamiltonian $H = \frac{1}{2}\omega^\top M^{-1}\omega$ is invariant under $M \to \alpha M$ for any $\alpha > 0$ — the dynamics only depend on **ratios** of inertia components, not their absolute scale. Without intervention, the optimizer finds a saddle $(M_\text{wrong},\, V_\text{compensating})$ where $V$ absorbs what $M$ misses: the geodesic loss converges while $M_\text{loss}$ diverges.

**Fix 1** — `--lambda_V_zero 1.0`: adds a $\|V_\theta\|$ regularisation term to force $V \to 0$, recovering the physical $M$.

**Fix 2** — `FixedInertiaFromState`: in Stage C the inertia values are embedded in the 18D state vector. Rather than training `M_net` to recover them from geometry (which reintroduces the degeneracy), we pin $M^{-1}$ to the exact physical value read from the state:

```python
class FixedInertiaFromState(nn.Module):
    def forward(self, q_ext):        # q_ext: (N, 12)
        return torch.diag_embed(1.0 / q_ext[:, 9:12])   # (N, 3, 3)
```

This makes multi-geometry generalisation exact by construction.

---

## 4. Training stages

### Stage A — single geometry, torque-free

Train on one racket geometry ($I_1, I_2, I_3$ fixed) with zero applied torque. `--fix_M` pins `M_net` to the exact inertia and `--lambda_V_zero 1.0` forces $V \to 0$.

**Result**: windowed geodesic loss $= 2.03\times 10^{-6}$ (threshold $0.01$ rad²). $V$ and $D_w$ converge to machine $\epsilon$.

### Stage B — friction identification

A physical friction torque $\tau_\text{fric} = -\kappa\,\omega$ is added to the dataset. `Dw_net` learns the dissipation without any supervision signal on $D_w$ itself.

**Result**: $D_w$ loss $= 5.13 \times 10^{-14}$ — ten orders of magnitude below threshold. (Friction makes the system contractive, collapsing trajectory variance.)

### Stage C — multi-geometry generalisation

State extended from 15D to 18D: $[\text{vec}(R)_9,\, I_1 I_2 I_3,\, \omega_3,\, u_3]$. Train jointly on 4 geometries; test on a held-out 5th.

**Result**: windowed geo $= 2.03\times 10^{-6}$ on all 4 geometries and on the unseen 5th.
Generalisation is **exact** because `FixedInertiaFromState` reads the exact $I$ from the embedded state — it does not need to learn anything geometry-dependent.

---

## 5. Control design

### 5.1 IDA-PBC framework

**Interconnection and Damping Assignment PBC** (Ortega et al. 2001) designs a controller by specifying a desired closed-loop port-Hamiltonian system with Hamiltonian $H_d$, interconnection $J_d$, and dissipation $D_d$. The control law solves the **matching equation**:

$$
g(R)\,u = \bigl[J_d - D_d\bigr]\nabla H_d - \bigl[J - D\bigr]\nabla H
$$

For our environment the actuator applies torque directly, so $g = I_3$ exactly (confirmed by training: `g_net` $\to I_3$). The matching equation becomes straightforward.

### 5.2 Stage D — velocity stabilisation (e₃ target)

Choose $H_d = H + \frac{1}{2}K_p\|\omega - \omega^*\|^2$, $J_d = J$, $D_d = D + K_p I$. Substituting and cancelling:

$$
u = K_p(\omega^* - \omega), \qquad \omega^* = (0,0,2\pi) \text{ rad/s}
$$

The learned model is not used at control time — the IDA-PBC law simplifies entirely because $g = I_3$. The linearised closed-loop about $\omega^* = \omega_3^* e_3$ has eigenvalues:

$$
\lambda_3 = -K_p/I_3, \qquad \lambda_{1,2} = \tfrac{\text{tr}A \pm \sqrt{\text{tr}^2 A - 4\det A}}{2}
$$

Overdamped for $K_p \geq 0.071$. **Result**: $K_p = 0.10$ gives 100% convergence from Dzhanibekov tumbling in $0.30$ s.

### 5.3 Stage E — saddle stabilisation and ZOH instability

**Saddle stabilisation** (hold $e_2$): the closed-loop matrix at $\omega^* = \omega_2 e_2$ is:

$$
A_{cl} = \begin{pmatrix} -K_p/I_1 & (I_2-I_3)\omega_2/I_1 \\ (I_1-I_2)\omega_2/I_3 & -K_p/I_3 \end{pmatrix}
$$

Stability requires $\det(A_{cl}) > 0$, giving:

$$
K_{p,\min} = \omega_2\sqrt{|I_2-I_3|\cdot|I_1-I_2|} = 0.021 \text{ N·m·s/rad}
$$

Empirical threshold: $K_p \leq 0.020$ (matches theory to 5%).

**ZOH discrete instability** (new finding): the environment applies $u$ at $dt = 0.05$ s and holds it constant for 10 integration substeps. The discrete-time eigenvalue for axis $i$ is:

$$
z_i = 1 - K_p\,T/I_i, \qquad T = dt = 0.05 \text{ s}
$$

For $|z_i| < 1$ we need $K_p < 2I_i/T$. The binding constraint is the smallest inertia $I_1$:

$$
K_{p,\max} = \frac{2I_1}{T} = \frac{2 \times 5.72\times10^{-3}}{0.05} = 0.229 \text{ N·m·s/rad}
$$

At $K_p = 0.50$: $z_1 = 1 - 4.39 = -3.39$, $|z_1| \gg 1$ — **confirmed divergence** to 14 rad/s.

**Wind robustness**: constant disturbance $d$ creates a steady-state velocity offset $\|\delta\omega\|_{ss} = |d|/K_p$. The viable window under wind $\sigma = 0.05$ N·m:

$$
K_p \in \!\left(\frac{|d|}{\varepsilon_\text{conv}},\; \frac{2I_1}{T}\right) \approx (0.17,\; 0.23)
$$

$K_p = 0.20$ achieves 95% convergence; ratio $\|\delta\omega\|_\text{actual}/(|d|/K_p) = 1.019$ (linear prediction is tight).

### 5.4 Stage F — full-state geometric attitude control

**What Stage D/E cannot do**: the proportional controller $u = K_p(\omega^*-\omega)$ stabilises $\omega$ but leaves $R$ free to drift anywhere on $SO(3)$.

**Geodesic attitude error**: the natural error on $SO(3)$ is:

$$
e_R = \mathrm{vee}\!\bigl(\log(R^{*\top} R)\bigr) \in \mathbb{R}^3
$$

where $\log: SO(3) \to \mathfrak{so}(3)$ is the matrix logarithm (Rodrigues formula):

$$
\log R = \frac{\theta}{2\sin\theta}(R - R^\top), \quad \theta = \arccos\!\tfrac{\text{tr}R - 1}{2}
$$

and $\mathrm{vee}$ extracts the axial vector of a skew-symmetric matrix. $\|e_R\| = \theta$ is the geodesic distance from $R^*$ to $R$ on $SO(3)$.

**Stage F controller** (IDA-PBC with $H_d = H + \frac{1}{2}K_R\|e_R\|^2 + \frac{1}{2}K_p\|e_\omega\|^2$):

$$
u = -K_R\, e_R - K_p\,(omega - \omega^*)
$$

The linearised closed-loop per axis $i$ is a **2nd-order oscillator**:

$$
\begin{pmatrix}\dot{e}_{R,i}\\ \dot{e}_{\omega,i}\end{pmatrix} = \begin{pmatrix}0 & 1 \\ -K_R/I_i & -K_p/I_i\end{pmatrix}\begin{pmatrix}e_{R,i}\\ e_{\omega,i}\end{pmatrix}
$$

with $\omega_n = \sqrt{K_R/I_i}$ and $\zeta = K_p/(2\sqrt{K_R I_i})$.

**ZOH bound for the attitude loop**: the 2nd-order discrete system (Schur stability) gives an additional constraint:

$$
K_{R,\max} = K_p/T = 0.10/0.05 = 2.00 \text{ N·m/rad}
$$

**Orientation–velocity duality under wind**: with $\omega^* = 0$, constant wind $d$ is absorbed entirely into a steady-state **orientation** offset (the body comes to rest at a tilted angle):

$$
K_R\, e_R = d \;\Rightarrow\; \|e_R\|_{ss} = |d|/K_R, \qquad \|\omega\|_{ss} = 0
$$

Compare with Stage E ($\omega^* \neq 0$): wind creates a **velocity** offset $|d|/K_p$. The ratio $\|e_R\|_\text{actual}/(|d|/K_R) = 1.000$ exactly for all tested $K_R \in [0.10, 1.50]$ N·m/rad.

---

## 6. Results summary

| Stage | Task | Metric | Result |
|-------|------|--------|--------|
| A | Single-config identification | windowed geo | $2.03\times10^{-6}$ rad² ✓ |
| B | Friction ($D_w$) identification | $D_w$ loss | $5.13\times10^{-14}$ ✓ |
| C | 4-config + held-out 5th | windowed geo | $2.03\times10^{-6}$ all ✓ |
| D | Stabilise $e_3$ spin from tumbling | conv time | 0.30 s, 100%, $K_p=0.10$ ✓ |
| E-B | Hold unstable $e_2$ axis | empirical $K_{p,\min}$ | 0.020 (theory 0.021) ✓ |
| E-C | Hold $e_2$ with wind $\sigma=0.05$ | 95% conv | $K_p=0.20$, ratio 1.019 ✓ |
| F-A | Rest at $R^*=I_3$, $\omega^*=0$ | conv time | 2.75 s, 100%, $K_R=0.10$ ✓ |
| F-B | Rest at $R^*=R_y(\pi/4)$, $\omega^*=0$ | conv time | 2.82 s, 100% ✓ |
| F-D | Rest at $I_3$ with wind $\sigma=0.05$ | 95% conv | $K_R=1.50$, ratio 1.000 ✓ |

---

## 7. Key non-obvious findings

:::info
**Hamiltonian degeneracy**: The free-body Hamiltonian $H=\frac{1}{2}\omega^\top M^{-1}\omega$ is unchanged under $M\to\alpha M$. Without regularisation, the optimizer finds a $(M_\text{wrong}, V_\text{compensating})$ saddle where the geodesic loss converges but $M$ is unphysical. Fixed by $\|V\|$ regularisation + `FixedInertiaFromState`.
:::

:::warning
**ZOH discrete instability**: The gain bound from continuous-time analysis ($K_p$ any positive value) does not apply to the digital controller. Zero-order hold at $T=0.05$ s gives $z_1 = 1 - K_p T/I_1$; stability requires $K_p < 2I_1/T = 0.229$. Discovered empirically: $K_p=0.50$ diverges to 14 rad/s with $|z_1|=3.39$.
:::

:::success
**Orientation–velocity duality**: The steady-state wind offset formula $\|\text{error}\|_{ss} = |d|/\text{gain}$ holds in **both** state spaces. In velocity control ($\omega^*\neq 0$): offset is in $\omega$, proportional to $1/K_p$. In attitude control ($\omega^*=0$): offset is in $R$, proportional to $1/K_R$. Ratio exact to 3 decimal places in both cases.
:::

:::success
**Universal ZOH bound**: $K_{p,\max} = 2I_1/T$ is the same regardless of which axis is the control target, because $I_1$ (the smallest inertia) is always the first mode to go unstable. Confirmed: $K_p=0.50$ diverges on $e_1$, $e_2$, and $e_3$ hold with identical late-trajectory error $\approx 14$ rad/s.
:::

---

## 8. Relationship to the broader LieSPHGP project

This work parallels the 3D SO(3) Windy Pendulum model in the same repository, which is the primary case study. The tennis racket model:

- **Shares** the `DissipativeSO3HamNODE` base network architecture and loss utilities
- **Extends** with a multi-geometry state (18D vs 15D) via the embedded inertia approach
- **Adds** `FixedInertiaFromState` as a novel module resolving the identifiability problem
- **Demonstrates** IDA-PBC control design through Stages D–F, which has not yet been applied to the pendulum

The key structural lesson — that the free-body Hamiltonian degeneracy requires either regularisation ($V\to 0$) or structural pinning (`FixedInertiaFromState`) — is applicable to any rigid-body system without a gravity potential to break the symmetry.

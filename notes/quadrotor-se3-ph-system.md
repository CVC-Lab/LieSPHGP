# Stochastic Port-Hamiltonian Formulation of a Windy Quadrotor on $SE(3)$

**Status:** environment built and verified at *both* levels — the env level (§20.1) and the
port-Hamiltonian-structure level (§20.2, **6/6 passing**).
Code: [`envs/port_ham_quadrotor_se3/quadrotor.py`](../envs/port_ham_quadrotor_se3/quadrotor.py) (class
`quadrotor_se3`, line 144), tests
[`mini_tests/test_gt_pH_matches_quad_env.py`](../mini_tests/test_gt_pH_matches_quad_env.py).
Dataset generator and learned model **not yet started**.
**Purpose:** extend the single-link `windy_pendulum_3d` / `ph_gp_sde` system from $SO(3)$ to the
full rigid-body group $SE(3)$, so the benchmark suite gains a *real-world*, *underactuated*
platform (a quadrotor) while keeping the *same* port-Hamiltonian machinery and an exact
analytic ground truth for every learned block.

---

## Table of contents

0. [Notation](#0-notation)
1. [What we are generalizing from](#1-what-we-are-generalizing-from)
2. [Configuration manifold](#2-configuration-manifold)
3. [Kinematics](#3-kinematics)
4. [The Hamiltonian: $\mathcal{M}$ and $V$](#4-the-hamiltonian-mathcalm-and-v)
5. [Trivialized gradients on $SE(3)$](#5-trivialized-gradients-on-se3)
6. [The port-Hamiltonian system](#6-the-port-hamiltonian-system)
7. [Dissipation $D$](#7-dissipation-d)
8. [Input map $G$](#8-input-map-g)
9. [Wind: deterministic port and stochastic ports](#9-wind-deterministic-port-and-stochastic-ports)
10. [The full SDE, and Itô vs Stratonovich](#10-the-full-sde-and-itô-vs-stratonovich)
11. [Passivity and energy balance](#11-passivity-and-energy-balance)
12. [Correspondence with the $SO(3)$ pendulum](#12-correspondence-with-the-so3-pendulum)
13. [Reduction checks and conserved quantities](#13-reduction-checks-and-conserved-quantities)
14. [Geometric integrator](#14-geometric-integrator)
15. [Observation model and losses](#15-observation-model-and-losses)
16. [What the learned subnetworks become](#16-what-the-learned-subnetworks-become)
17. [Structural priors worth baking in](#17-structural-priors-worth-baking-in)
18. [Practical concerns](#18-practical-concerns)
19. [Remaining design decisions](#19-remaining-design-decisions)
20. [Verification: env level and pH-structure level](#20-verification-env-level-and-ph-structure-level)
21. [Build order](#21-build-order)
22. [Appendix: all environment arguments](#22-appendix-all-environment-arguments) — incl. [config files & CLI](#227-config-files-and-cli-overrides)

---

## 0. Notation

| Symbol | Meaning | Frame | Dimension | Code |
|---|---|---|---|---|
| $x_w$ | position of the center of mass | world | $\mathbb{R}^3$ | `self.x_w` |
| $R$ | attitude, body $\to$ world | — | $SO(3)$ | `self.R` |
| $v_b$ | linear velocity | **body** | $\mathbb{R}^3$ | `self.v_b` |
| $\omega_b$ | angular velocity | **body** | $\mathbb{R}^3$ | `self.omega` |
| $\xi=(v_b,\omega_b)$ | left-trivialized velocity | body | $\mathbb{R}^6$ | — |
| $\mathbf{p}=m\,v_b$ | linear momentum | body | $\mathbb{R}^3$ | — |
| $\Pi=J\,\omega_b$ | angular momentum | body | $\mathbb{R}^3$ | — |
| $m,\ J$ | mass, inertia (diagonal) | — | $\mathbb{R},\ \mathbb{R}^{3\times3}$ | `m`, `J_diag` |
| $k_f,\ k_m$ | thrust gain, yaw-drag gain | — | $\mathbb{R}_{\ge0}$ | `kf_coeff`, `km_coeff` |
| $a$ | rotor lever, $a=\text{arm}/\sqrt2$ | body | $\mathbb{R}$ | `arm` |
| $d_{\text{lin}},\ d_{\text{ang}}$ | linear / angular damping | — | $\mathbb{R}_{\ge0}$ | `linear/angular_damping_coeff` |
| $u$ | squared rotor speeds ($u_i=\text{rpm}_i^2$) | — | $\mathbb{R}^4$ | action |
| $w(t),\ \hat d$ | deterministic wind magnitude, direction | world | $\mathbb{R},\ \mathbb{R}^3$ | `update_wind`, `external_force_direction` |
| $\sigma_f,\ \sigma_\tau$ | stochastic wind force / torque scales | — | $\mathbb{R}_{\ge0}$ | `wind_force_std`, `wind_torque_std` |
| $e_3$ | $(0,0,1)^\top$ | — | $\mathbb{R}^3$ | — |

**Hat / vee** ([`quadrotor.py:14-24`](../envs/port_ham_quadrotor_se3/quadrotor.py#L14-L24)):

$$
[a]_\times=\begin{pmatrix}0&-a_3&a_2\\ a_3&0&-a_1\\ -a_2&a_1&0\end{pmatrix},
\qquad [a]_\times b = a\times b,\qquad [a]_\times^\top=-[a]_\times .
$$

Identity used repeatedly, valid for $R\in SO(3)$:

$$
R^\top(u\times v)=(R^\top u)\times(R^\top v).
$$

---

## 1. What we are generalizing from

The existing benchmark, [`envs/pendulum_so3/windy_pendulum_3d.py`](../envs/pendulum_so3/windy_pendulum_3d.py), is a **single
rigid body pinned at a point** — configuration on $SO(3)$, momentum $p\in\mathbb{R}^3$, and the
left-trivialized pH drift

$$
\dot p \;=\; p\times \tfrac{\partial H}{\partial p}
\;+\;\textstyle\sum_{k=1}^{3} r_k\times\tfrac{\partial H}{\partial r_k}
\;-\;D\,\tfrac{\partial H}{\partial p}\;+\;g(q)\,u .
$$

The quadrotor removes the pin: the body now also **translates**, so the configuration group
grows from $SO(3)$ to

$$
SE(3)=\mathbb{R}^3\rtimes SO(3).
$$

**Key observation.** The drift equation above is the general left-trivialized
**Hamilton–Poincaré** equation on a Lie group; only the group changes. What the quadrotor adds
structurally:

1. a **6-dimensional** momentum $(\mathbf p,\Pi)$ with a richer coadjoint block (§6);
2. genuine **underactuation** — 4 inputs, 6 velocity DOF — through a physically meaningful,
   *constant* control matrix $G\in\mathbb{R}^{6\times4}$ (§8), where the pendulum's `g_net`
   only had to learn a near-identity $3\times3$ map;
3. **two distinct noise channels** — a multiplicative world-frame force and an additive
   body-frame torque (§9) — instead of the pendulum's single lever-arm channel.

Unlike the $n$-joint arm note (formulation only), the environment here **already exists** and
is verified against gym-pybullet-drones (§20).

---

## 2. Configuration manifold

$$
\boxed{\;q=(x_w,\,R)\in G=SE(3),\qquad \dim G=6\;}
$$

stored redundantly as $q\in\mathbb{R}^{12}$ ($x_w$ plus row-major $\mathrm{vec}(R)$), which is
unconstrained in $x_w$ and constrained only through $R\in SO(3)$ — the same redundant-but-safe
embedding as the pendulum (§5 explains why it is safe).

Velocities are **left-trivialized** (body frame): $\xi=(v_b,\omega_b)\in\mathfrak{se}(3)\cong\mathbb{R}^6$.
This choice drives everything that follows:

1. **Every coefficient becomes constant.** In the body frame the mass matrix
   $\mathcal M=\mathrm{blkdiag}(mI_3,\,J)$, the dissipation
   $D=\mathrm{blkdiag}(d_{\text{lin}}I_3,\,d_{\text{ang}}I_3)$ and the control matrix $G$ are
   all **state-independent** ground truths ([`quadrotor.py:147-163`](../envs/port_ham_quadrotor_se3/quadrotor.py#L147-L163)).
   In the world frame each of them would pick up $R$-dependence.
2. **The wrench is linear in the input** because the input is defined as $u_i=\text{rpm}_i^2$:
   propeller physics gives thrust $f_i=k_f\,\text{rpm}_i^2$, so $[F_b;\tau_b]=G\,u$ exactly (§8).
3. **Position never enters the momentum dynamics** (§5) — gravity has a *constant* world-frame
   gradient — so all learned subnetworks can take $R$ alone as input (§17).

---

## 3. Kinematics

$$
\boxed{\;\dot x_w = R\,v_b,\qquad \dot R = R\,[\omega_b]_\times\;}
$$

([`quadrotor.py:414`](../envs/port_ham_quadrotor_se3/quadrotor.py#L414) for the first equation; the second
is realized by the exponential-map update of §14.) The pair is exactly the left-trivialization
of $\dot g = g\,\hat\xi$ on $SE(3)$ with

$$
g=\begin{pmatrix}R & x_w\\ 0 & 1\end{pmatrix},\qquad
\hat\xi=\begin{pmatrix}[\omega_b]_\times & v_b\\ 0 & 0\end{pmatrix}.
$$

No forward-kinematics chain and no configuration-dependent Jacobians exist here — the
quadrotor is a *single* body, so what was $J_{v,i}(q)$ for the arm degenerates to the identity.
The complexity moves instead into the coadjoint structure (§6) and the input map (§8).

---

## 4. The Hamiltonian: $\mathcal{M}$ and $V$

### 4.1 Kinetic energy and the mass matrix

$$
T=\tfrac12\,m\|v_b\|^2+\tfrac12\,\omega_b^\top J\,\omega_b
 =\tfrac12\,\xi^\top \mathcal M\,\xi,
\qquad
\boxed{\;\mathcal M=\mathrm{blkdiag}\big(m I_3,\ J\big)\in\mathbb{R}^{6\times6},\ \text{SPD, constant}\;}
$$

The block-diagonal (no $v$–$\omega$ coupling) is exact because the body frame is attached at
the **center of mass** — the same reason the wind force at the COM produces no deterministic
torque (§9).

### 4.2 Potential energy

$$
\boxed{\;V(q)=m\,g\,e_3^\top x_w\;}
$$

$V$ depends on the configuration **only through the height $z=x_{w,3}$** — not on $R$ at all.
Compare the pendulum, where $V=mg\ell R_{22}$ lived entirely in $R$. Two consequences:

- the gravity port is a **constant world-frame force** $-mge_3$, rotated to body by $R^\top$
  ([`quadrotor.py:411`](../envs/port_ham_quadrotor_se3/quadrotor.py#L411));
- $V$ is invariant under horizontal translations and *all* rotations — the residual symmetry
  group is $SE(2)\times\mathbb{R}$, which yields conserved momenta for testing (§13.3).

### 4.3 Momenta and Hamiltonian

$$
\mathbf p=m\,v_b,\qquad \Pi=J\,\omega_b,\qquad
\boxed{\;H=\frac{1}{2m}\|\mathbf p\|^2+\frac12\,\Pi^\top J^{-1}\Pi+m\,g\,e_3^\top x_w\;}
$$

Both momenta are **quasi-momenta** — conjugate to the body-frame rates, not to a coordinate
chart — the same convention as the pendulum's $(q,p)$ integrator.

---

## 5. Trivialized gradients on $SE(3)$

Perturb along the group action: $x_w\mapsto x_w+R\,\delta$ and $R\mapsto R\exp([\phi]_\times)$,
with $(\delta,\phi)\in\mathbb{R}^6$ a body-frame displacement. For any function $F(x_w,R)$,

$$
\delta F=\underbrace{\big(R^\top\nabla_{x_w}F\big)}_{\text{body-frame force}}\cdot\,\delta
\;+\;\underbrace{\Big(\textstyle\sum_{k=1}^3 \frac{\partial F}{\partial r_k}\times r_k\Big)}_{-\,\mathcal T_R(F)}\cdot\,\phi ,
\qquad
\mathcal T_R(F):=\sum_{k=1}^{3} r_k\times\frac{\partial F}{\partial r_k}.
$$

Applying this to the Hamiltonian of §4.3:

- **translation channel:** $-R^\top\nabla_{x_w}V=-m\,g\,R^\top e_3$ — exactly the gravity term
  in [`quadrotor.py:418-421`](../envs/port_ham_quadrotor_se3/quadrotor.py#L418-L421);
- **rotation channel:** $\mathcal T_R(H)=0$, because $H$ does not depend on $R$ — gravity acting
  at the COM produces **no torque**, so the rotational drift contains no gravity term
  ([`quadrotor.py:425-428`](../envs/port_ham_quadrotor_se3/quadrotor.py#L425-L428)).

As with the pendulum, $\mathcal T_R$ and $R^\top\nabla_{x_w}$ are directional derivatives *along
the group action*, hence independent of how $F$ is extended off the manifold — the redundant
$\mathbb{R}^{12}$ storage of $q$ is safe with no projection needed.

---

## 6. The port-Hamiltonian system

On $\mathfrak{se}(3)^*$ the coadjoint operator is

$$
\mathrm{ad}^*_{(v,\omega)}(\mathbf p,\Pi)
=\big(\,\mathbf p\times\omega,\ \ \Pi\times\omega+\mathbf p\times v\,\big),
$$

richer than the pendulum's single $p\times\omega$: translation and rotation couple through
$\mathbf p\times v$. With $\xi=\partial H/\partial(\mathbf p,\Pi)=(v_b,\omega_b)$ the
Hamilton–Poincaré equations read

$$
\boxed{
\begin{aligned}
\dot x_w &= R\,v_b,\\
\dot R &= R\,[\omega_b]_\times,\\[4pt]
\dot{\mathbf p} &= \underbrace{\mathbf p\times\omega_b}_{\text{frame rotation}}
\;\underbrace{-\,R^\top\nabla_{x_w}V}_{-\,mgR^\top e_3}
\;-\;d_{\text{lin}}\,v_b\;+\;(Gu)_{1:3}\;+\;R^\top F_{\text{wind}},\\[4pt]
\dot{\Pi} &= \underbrace{\Pi\times\omega_b}_{\text{gyroscopic}}
\;+\;\underbrace{\mathbf p\times v_b}_{=\,0}
\;+\;\underbrace{\mathcal T_R(H)}_{=\,0}
\;-\;d_{\text{ang}}\,\omega_b\;+\;(Gu)_{4:6}.
\end{aligned}}
$$

**The elegant simplification:** $\mathbf p\times v_b=m\,v_b\times v_b=0$ — the $SE(3)$
cross-coupling vanishes *because the mass matrix is block-diagonal* ($\mathbf p\parallel v_b$).
This is why [`_compute_rates`](../envs/port_ham_quadrotor_se3/quadrotor.py#L389-L444) contains only
$v_b\times\omega_b$ (line 419) and $(J\omega_b)\times\omega_b$ (line 427), and yet is the *exact*
$SE(3)$ pH drift, not an approximation.

### 6.1 Matrix form

With state $x=(x_w,\ \mathrm{vec}(R),\ \mathbf p,\ \Pi)\in\mathbb{R}^{12}\times\mathbb{R}^{6}$,

$$
\dot x=\big(\mathcal J(x)-\mathcal R\big)\frac{\partial H}{\partial x}+\mathcal G\,u+\text{(wind ports)},
$$

$$
\mathcal J(x)=
\begin{pmatrix}
0 & 0 & R & 0\\
0 & 0 & 0 & B(q)\\
-R^\top & 0 & 0 & [\mathbf p]_\times\\
0 & -B(q)^\top & [\mathbf p]_\times & [\Pi]_\times
\end{pmatrix},
\qquad
\mathcal R=\begin{pmatrix}0&0\\ 0&D\end{pmatrix},
\qquad
\mathcal G=\begin{pmatrix}0_{12\times4}\\ G\end{pmatrix},
$$

where $B(q)=\big([r_1]_\times;\,[r_2]_\times;\,[r_3]_\times\big)\in\mathbb{R}^{9\times3}$ is the
same tangent-of-the-group-action block as in the pendulum/arm formulation.

**Consistency checks.**

- $(x_w,\mathbf p)$ block: $\dot x_w=R\,\partial H/\partial\mathbf p=Rv_b$. ✔
- $(\mathbf p,x_w)$ block: $-R^\top\partial H/\partial x_w=-mgR^\top e_3$. ✔
- Momentum–momentum block $\hat P=\begin{pmatrix}0&[\mathbf p]_\times\\ [\mathbf p]_\times&[\Pi]_\times\end{pmatrix}$
  acting on $(v_b,\omega_b)$: row 1 gives $\mathbf p\times\omega_b$, row 2 gives
  $\mathbf p\times v_b+\Pi\times\omega_b$ — the full $\mathrm{ad}^*$. ✔
- $\hat P^\top=-\hat P$ and both off-diagonal pairings ($R$ vs $-R^\top$, $B$ vs $-B^\top$) are
  antisymmetric, so $\mathcal J=-\mathcal J^\top$. ✔ And $\mathcal R\succeq0$. ✔

Note the extra structure relative to the pendulum: the momentum–momentum skew block now has an
**off-diagonal** $[\mathbf p]_\times$ (silent at runtime since $\mathbf p\parallel v_b$, but part
of the exact structure a pH-constrained model should carry), and the $(q,p)$ coupling block is
$\mathrm{blkdiag}(R,\,B(q))$ instead of $B(q)$ alone.

---

## 7. Dissipation $D$

$$
\boxed{\;D=\mathrm{blkdiag}\big(d_{\text{lin}}\,I_3,\ d_{\text{ang}}\,I_3\big)\succeq0\;}
$$

giving the friction force $-d_{\text{lin}}v_b$
([`quadrotor.py:420`](../envs/port_ham_quadrotor_se3/quadrotor.py#L420)) and torque
$-d_{\text{ang}}\omega_b$ ([`quadrotor.py:427`](../envs/port_ham_quadrotor_se3/quadrotor.py#L427)).
Constant and isotropic — simpler than the pendulum's state-modulated
[`_variable_friction`](../envs/pendulum_so3/windy_pendulum_3d.py#L227-L234) — but the environment replaces
state-modulation with **parameter randomization**:

$$
d\;=\;
\begin{cases}
d_{\text{coeff}} & \text{std}=0 \quad(\text{fixed}),\\[4pt]
\max\!\big(0,\ \mathcal N(d_{\text{coeff}},\,\text{std}^2)\big) & \text{std}>0\quad(\text{varying}),
\end{cases}
$$

applied independently to $k_f,k_m,d_{\text{lin}},d_{\text{ang}}$
([`_draw_coeff`, `quadrotor.py:301-305`](../envs/port_ham_quadrotor_se3/quadrotor.py#L301-L305)), re-drawn
**per step** (`resample_coeffs_every_step=True`, held across the 10 substeps — piecewise-constant
parameter noise) or **per trajectory** (`False` — domain randomization)
([`_resample_coeffs`, `quadrotor.py:307-318`](../envs/port_ham_quadrotor_se3/quadrotor.py#L307-L318)).
The realized values are exposed each step in `info` ([`quadrotor.py:564-570`](../envs/port_ham_quadrotor_se3/quadrotor.py#L564-L570))
— the ground truth a learned $D$ is compared against.

---

## 8. Input map $G$

The action is $u\in\mathbb{R}^4$ = squared rotor speeds. The body wrench is one constant matrix
multiply ([`_build_G`, `quadrotor.py:320-342`](../envs/port_ham_quadrotor_se3/quadrotor.py#L320-L342)):

$$
\begin{bmatrix}F_b\\ \tau_b\end{bmatrix}=G\,u,\qquad
\boxed{\;G=\begin{bmatrix}
0&0&0&0\\ 0&0&0&0\\ k_f&k_f&k_f&k_f\\
-ak_f&-ak_f&ak_f&ak_f\\ -ak_f&ak_f&ak_f&-ak_f\\ -k_m&k_m&-k_m&k_m
\end{bmatrix}\in\mathbb{R}^{6\times4},\quad a=\frac{\text{arm}}{\sqrt2}\;}
$$

Row by row:

- rows 1–2 ($F_x,F_y$) are **zero**: all rotors push along body-$z$ only — the quadrotor is
  **underactuated** (rank $G=4$, but 6 velocity DOF);
- row 3: total thrust $k_f\sum_i u_i$;
- rows 4–5: roll/pitch from thrust *differences* across the lever $a$;
- row 6: yaw from alternating propeller reaction drag $k_m$.

Signs follow the **CF2X X-configuration** of gym-pybullet-drones (`BaseAviary._dynamics`), with
rotor positions $r_0=(+a,-a,0),\ r_1=(-a,-a,0),\ r_2=(-a,+a,0),\ r_3=(+a,+a,0)$
([`_rotor_positions`, `quadrotor.py:344-350`](../envs/port_ham_quadrotor_se3/quadrotor.py#L344-L350)).

Two structural points, mirroring the arm note's §8:

1. **$G$ is constant** — a *consequence* of the body-frame (left-trivialized) formulation, not
   an assumption. In world-frame coordinates the same physics would give $G(q)$ with
   $R$-dependence in the force rows. Where the arm made `g_net` learn a genuine
   state-dependent map, the quadrotor makes it learn a genuinely **rectangular, sparse,
   underactuated** one.
2. **The collocated output** is $y=G^\top\xi\in\mathbb{R}^4$,
   $$
   y_i=k_f\,(v_b)_3\;\pm\;a k_f\big(\text{roll/pitch rates}\big)\;\pm\;k_m\,\omega_{b,3},
   $$
   the power each rotor injects per unit $u_i$ — the natural passive output for §11 and for any
   future IDA-PBC controller. The 4×4 submatrix $A=G_{3:6,:}$ (thrust + torque rows) is
   invertible; the demo hover controller uses exactly $u=A^{-1}[T;\tau]$
   ([`quadrotor.py:928-930`](../envs/port_ham_quadrotor_se3/quadrotor.py#L928-L930)).

---

## 9. Wind: deterministic port and stochastic ports

### 9.1 Deterministic wind

A world-frame force at the COM ([`update_wind`, `quadrotor.py:354-363`](../envs/port_ham_quadrotor_se3/quadrotor.py#L354-L363)),
identical menu to the pendulum:

$$
F_{\text{wind}}(t)=w(t)\,\hat d,\qquad
w(t)=\begin{cases}
\sigma_w\sin(2\pi\,0.5\,t) & \texttt{sine}\\
\sigma_w\,\mathrm{sign}\!\big(\sin(2\pi\,0.5\,t)\big) & \texttt{square}\\
\mathcal N(0,\sigma_w^2)\ \text{per step} & \texttt{random}\\
\sigma_w & \texttt{constant}
\end{cases}
$$

Because it acts **at the COM**, it enters only the force channel — no deterministic torque:

$$
d\mathbf p_{\text{det-wind}}=R^\top\hat d\;w(t)\,dt .
$$

### 9.2 Stochastic wind — two channels

$$
\boxed{\;
d\mathbf p_{\text{stoch}}=\sigma_f\,R^\top dW_f\ \ (\text{multiplicative}),
\qquad
d\Pi_{\text{stoch}}=\sigma_\tau\,dW_\tau\ \ (\text{additive}),
\qquad W_f,W_\tau\in\mathbb{R}^3\ \text{independent}\;}
$$

([`quadrotor.py:434-442`](../envs/port_ham_quadrotor_se3/quadrotor.py#L434-L442)). The full diffusion is

$$
\Xi(q)=\begin{pmatrix}0_{12\times6}\\ \mathrm{blkdiag}\big(\sigma_f R^\top,\ \sigma_\tau I_3\big)\end{pmatrix}
\in\mathbb{R}^{18\times6}.
$$

### 9.3 Why this matters

- **Deterministic and stochastic wind share the same map $R^\top$** — exactly the "shared
  $\Sigma(q)$" structure of the arm note (§9.2 there): one physical port, two signals through it.
- **The force channel is state-dependent (multiplicative) noise** — $R$ appears — which is what
  forces the Stratonovich treatment and the two-stage evaluation in the integrator (§14).
  The torque channel is plain additive noise.
- **A structural subtlety absent from the pendulum:** the force channel's covariance is
  *isotropic*, $\sigma_f^2R^\top R^{-\top}=\sigma_f^2 I_3$ — the $R$-dependence is statistically
  invisible at the level of increment covariances (§15.3, §18.3). The pendulum's lever-arm
  diffusion $\ell\,[e_z]_\times R^\top$ ([`windy_pendulum_3d.py:281`](../envs/pendulum_so3/windy_pendulum_3d.py#L281))
  was rank-deficient and anisotropic, hence identifiable from covariances; here only the
  *scale* $\sigma_f$ is.
- The torque channel has **no deterministic counterpart** — it stands in for unmodelled
  rotational turbulence and is what makes the attitude genuinely stochastic (and the platform
  crash without feedback, §18.1).

---

## 10. The full SDE, and Itô vs Stratonovich

$$
\boxed{
\begin{cases}
dx_w=R\,v_b\,dt,\\[2pt]
dR=R\,[\omega_b]_\times\,dt,\\[4pt]
d\mathbf p=\Big(\mathbf p\times\omega_b-mgR^\top e_3-d_{\text{lin}}v_b+(Gu)_{1:3}+w(t)R^\top\hat d\Big)dt
\;+\;\sigma_f\,R^\top\!\circ dW_f,\\[4pt]
d\Pi=\Big(\Pi\times\omega_b-d_{\text{ang}}\omega_b+(Gu)_{4:6}\Big)dt
\;+\;\sigma_\tau\circ dW_\tau,
\end{cases}}
$$

with $\circ$ Stratonovich — the convention under which the ordinary chain rule holds, hence
the one consistent with exponential-map integration on the manifold.

### 10.1 Itô $=$ Stratonovich here

Same argument as the arm note, and it goes through unchanged: the Stratonovich$\to$Itô
correction is $\tfrac12\sum_k(\partial_x\Xi_{\cdot k})\,\Xi_{\cdot k}$, and

1. $\Xi$ depends on the state only through $q=(x_w,R)$ (in fact only through $R$), so
   $\partial\Xi/\partial(\mathbf p,\Pi)=0$;
2. the $q$-rows of $\Xi$ are zero, so every nonzero entry of $\Xi_{\cdot k}$ sits in the
   momentum block.

Therefore every term in the correction pairs a derivative that is zero with an entry that
could be nonzero, or vice versa:

$$
\boxed{\ \text{The Itô correction vanishes: Itô and Stratonovich drifts coincide.}\ }
$$

Consequence: the Lie–Heun (Stratonovich) integrator needs no correction term, and
ensemble-mean comparisons against an Euler–Maruyama (Itô) reference are valid tests.

---

## 11. Passivity and energy balance

Since $\mathcal J=-\mathcal J^\top$, along deterministic trajectories

$$
\boxed{\;\frac{dH}{dt}
=-\,d_{\text{lin}}\|v_b\|^2-d_{\text{ang}}\|\omega_b\|^2
\;+\;y^\top u
\;+\;\dot x_w^\top\big(w(t)\hat d\big),
\qquad y:=G^\top\xi\in\mathbb{R}^4\;}
$$

(the wind term uses $v_b^\top R^\top\hat d=\dot x_w^\top\hat d$ — the wind does work at the rate
force · world velocity, as it must). With the wind off, the system is **passive** with respect
to the collocated pair $(u,\,y)$, storage function $H$.

Consequences:

- $D=0$, $u=0$, wind off $\Rightarrow$ $H$ exactly conserved — sharp integrator test.
- $D\succ0$, $u=0$, wind off $\Rightarrow$ $\dot H\le0$ strictly — **verified** (§20).
- At hover, $\xi=0\Rightarrow y=0$: the hover input $u^\star=\frac{mg}{4k_f}\mathbf 1_4$ injects
  zero power in steady state — thrust exactly cancels gravity, checked as an *exact* fixed
  point of the integrator ([`quadrotor.py:554`](../envs/port_ham_quadrotor_se3/quadrotor.py#L554) uses
  $u^\star$ in the reward; §20).
- The energy-shaping / IDA-PBC story from the pendulum carries over with $y\in\mathbb{R}^4$; the
  underactuation ($\mathrm{rank}\,G=4<6$) is now *real*, which is exactly the regime where
  IDA-PBC matching conditions become nontrivial.

---

## 12. Correspondence with the $SO(3)$ pendulum

| Block | Pendulum $SO(3)$ | Quadrotor $SE(3)$ |
|---|---|---|
| configuration | $R$, embed $\mathbb{R}^9$ | $(x_w,R)$, embed $\mathbb{R}^{12}$ |
| manifold dim | 3 | 6 |
| momentum | $p\in\mathbb{R}^3$ | $(\mathbf p,\Pi)\in\mathbb{R}^6$ |
| $(q,p)$ coupling | $B(q)$ | $\mathrm{blkdiag}(R,\ B(q))$ |
| $\mathrm{ad}^*$ block | $[p]_\times$ | $\begin{pmatrix}0&[\mathbf p]_\times\\ [\mathbf p]_\times&[\Pi]_\times\end{pmatrix}$, cross-term silent ($\mathbf p\parallel v_b$) |
| $\mathcal M$ | $m\ell^2 I_3$ ([`windy_pendulum_3d.py:179`](../envs/pendulum_so3/windy_pendulum_3d.py#L179)) | $\mathrm{blkdiag}(mI_3,J)$ — constant |
| $V(q)$ | $mg\ell\,R_{22}$ — in $R$ | $mg\,e_3^\top x_w$ — in $x_w$ only |
| gravity port | torque $\mathcal T_R(V)\ne0$ ([`windy_pendulum_3d.py:261`](../envs/pendulum_so3/windy_pendulum_3d.py#L261)) | force $-mgR^\top e_3$, torque $=0$ |
| $D$ | $\kappa I_3$ + state modulation | $\mathrm{blkdiag}(d_{\text{lin}}I_3,d_{\text{ang}}I_3)$ + parameter randomization |
| $g(q)$ | $\approx I_3$, full rank ([`windy_pendulum_3d.py:172`](../envs/pendulum_so3/windy_pendulum_3d.py#L172)) | constant $6\times4$ mixer, **underactuated** |
| diffusion | $\ell[e_z]_\times R^\top$ — rank 2, anisotropic | $\mathrm{blkdiag}(\sigma_fR^\top,\sigma_\tau I_3)$ — full rank, force block isotropic |
| state dim | $9+3=12$ | $12+6=18$ |

**Summary of the change:** the coupling block gains a translation row, the skew momentum block
gains the $SE(3)$ cross-coupling, gravity migrates from the torque channel to the force
channel, and the input map becomes rectangular. Everything else — the trivialized-gradient
operator, the Heun integrator, the Itô$=$Stratonovich property, the passivity identity — keeps
its exact shape. That is the entire content of "lifting the pendulum to $SE(3)$".

Relative to the reference paper's two noise channels (noisy time / model noise), the quadrotor
— like the pendulum and the planned $n$-joint arm — implements **model noise only**, keeping
all existing baselines directly comparable.

---

## 13. Reduction checks and conserved quantities

### 13.1 Rotational subsystem $=$ free rigid body

Ignore $(x_w,\mathbf p)$: the $(R,\Pi)$ equations are Euler's rigid-body dynamics with damping,
input torque, and additive noise — the pendulum's structure with $\mathcal T_R(V)=0$ (no
pin, no lever). Any rotational test written for the pendulum applies verbatim after deleting
the gravity torque.

### 13.2 Hover fixed point (exact, verified)

$$
u^\star=\frac{mg}{4k_f}\,\mathbf 1_4,\qquad R=I,\ v_b=0,\ \omega_b=0
\;\Longrightarrow\;
k_f\!\cdot\!4\!\cdot\!\tfrac{mg}{4k_f}=mg,\ \ \tau(u^\star)=0
$$

— all torque rows cancel by the mixer's sign symmetry, and the state is an **exact fixed point
of the discrete integrator** (drift $=0.0$ measured, §20).

### 13.3 Conserved quantities ($D=0$, $u=0$, wind off)

- **Energy:** $H$ conserved. Measured relative drift $5.5\times10^{-11}$ over $T=1$ s of free
  fall and $5.5\times10^{-10}$ over $T=20$ s in the bounded ($g=0$) regime, scaling as
  $O(h^3)$ — see §14 and §20.2 test 12.
- **Horizontal world momentum:** $P_w=mRv_b$ obeys $\dot P_w=-mge_3$, so
  $e_1^\top P_w,\ e_2^\top P_w$ are constants — from the $\mathbb{R}^2$ translation symmetry
  of $V$.
- **World angular momentum about the COM:** $\frac{d}{dt}(R\,\Pi)=R\big([\omega_b]_\times\Pi+\Pi\times\omega_b\big)=0$ —
  all three components conserved (gravity at the COM exerts no torque). Stronger than the
  pendulum's single conserved vertical component.

Each is an independent sign-convention test, and all three are now verified numerically at
$O(h^2)$ — §20.2 test 13.

---

## 14. Geometric integrator

One Stratonovich Heun substep of size $h=dt/10$ on $SE(3)$
([`_lie_heun_step`, `quadrotor.py:448-509`](../envs/port_ham_quadrotor_se3/quadrotor.py#L448-L509)),
reusing the same $dW_f,dW_\tau\sim\mathcal N(0,hI_3)$ in both stages
(sampled once per substep, [`quadrotor.py:533-540`](../envs/port_ham_quadrotor_se3/quadrotor.py#L533-L540)):

$$
\begin{aligned}
&\textbf{Stage 1:} && (\dot x_1,\dot v_1,\dot\omega_1,dV_1,d\Omega_1)=f(x_w,R,v_b,\omega_b),\qquad \phi_1=\omega_b\,h,\\[4pt]
&\textbf{Stage 2 (predictor):} && R_p=R\exp([\phi_1]_\times),\quad x_p=x_w+\dot x_1h,\quad
v_p=v_b+\dot v_1h+dV_1,\quad \omega_p=\omega_b+\dot\omega_1h+d\Omega_1,\\[4pt]
&\textbf{Stage 3:} && (\dot x_2,\dot v_2,\dot\omega_2,dV_2,d\Omega_2)=f(x_p,R_p,v_p,\omega_p)\ \ (\text{same }dW),\qquad \phi_2=\omega_p\,h,\\[4pt]
&\textbf{Stage 4 (corrector):} && R^+=R\exp\!\big([\tfrac12(\phi_1{+}\phi_2)]_\times\big),\quad
x^+=x_w+\tfrac h2(\dot x_1{+}\dot x_2),\\
&&& v^+=v_b+\tfrac h2(\dot v_1{+}\dot v_2)+\tfrac12(dV_1{+}dV_2),\quad
\omega^+=\omega_b+\tfrac h2(\dot\omega_1{+}\dot\omega_2)+\tfrac12(d\Omega_1{+}d\Omega_2).
\end{aligned}
$$

Properties, all inherited from the pendulum scheme:

- $R$ stays on $SO(3)$ **by construction** — Rodrigues' formula
  ([`_exp_so3`, `quadrotor.py:27-50`](../envs/port_ham_quadrotor_se3/quadrotor.py#L27-L50)), averaging in
  the Lie *algebra* then one exponential; measured drift $\sim10^{-14}$ over 500 noisy steps.
  A safety-net projection exists ([`quadrotor.py:549-551`](../envs/port_ham_quadrotor_se3/quadrotor.py#L549-L551))
  and never triggers in practice.
- Second order in the deterministic part — self-convergence ratio 4.03 measured (§20.1), and
  the momentum-conservation residuals scale at ratio 4.00 (§20.2 test 13).
- **The energy error is one order better still — $O(h^3)$, not $O(h^2)$.** Ratio $8.00$
  measured across four independent sweeps (§20.2 test 12), meaning the leading $O(h^2)$ state
  error lies *tangent to the level set of $H$* and only the $O(h^3)$ component moves the
  energy. Worth stating in the paper: it is a property of this scheme on this system, not a
  generic Heun guarantee, and it is what makes $H$ a usable diagnostic at practical step sizes.
- **Re-evaluating $dV$ at the predicted state is the Stratonovich correction**: the force-channel
  diffusion $\sigma_fR^\top$ depends on $R$, and Heun's two-stage averaging of a
  state-dependent diffusion converges to the Stratonovich solution.
- The translation/velocity components are plain vector-space Heun — $SE(3)$ needs the
  exponential only for $R$ (the position update through $Rv_b$ is already handled by the drift
  averaging; a full $SE(3)$-exponential variant is a possible refinement, §19).

RNG draw order per step is fixed (wind $\to$ coefficients $\to$ $dW$'s,
[`quadrotor.py:519-540`](../envs/port_ham_quadrotor_se3/quadrotor.py#L519-L540)) so trajectories are
bitwise reproducible per seed — verified (§20).

---

## 15. Observation model and losses

### 15.1 Observations

$$
\text{obs}=[\,x_w,\ \mathrm{vec}(R),\ v_b,\ \omega_b\,]\in\mathbb{R}^{18}
$$

([`_get_obs`, `quadrotor.py:367-385`](../envs/port_ham_quadrotor_se3/quadrotor.py#L367-L385)). With
`obs_noise_std` $=\sigma_o>0$, geometric noise on the manifold:

$$
R_{\text{obs}}=R\,\exp([\epsilon]_\times),\ \ \epsilon\sim\mathcal N(0,\sigma_o^2I_3),
\qquad
x_{\text{obs}},v_{\text{obs}},\omega_{\text{obs}}\ \text{additive}\ \mathcal N(0,\sigma_o^2I_3),
$$

the same recipe as `add_proper_noise_3d`
([`windy_pendulum_3d_datagen.py:79`](../envs/pendulum_so3/datagen/windy_pendulum_3d_datagen.py#L79)). A separate
RNG (seeded `seed+1`) generates observation noise, so toggling it never perturbs the dynamics
realization; `get_state()` always returns the clean state
([`quadrotor.py:295-297`](../envs/port_ham_quadrotor_se3/quadrotor.py#L295-L297)).

### 15.2 Rollout likelihood

One $SO(3)$ block plus Euclidean blocks — the concentrated-Gaussian NLL of
[`elbo_loss_jax.py`](../src/utils/JAX/elbo_loss_jax.py) with $n=1$, extended by Gaussian terms
for $x_w$ and $v_b$:

$$
-\log p(\tilde q\mid\hat q)=
\underbrace{\frac{\theta^2}{2\sigma_R^2}+3\log\sigma_R+\tfrac32\log2\pi}_{\theta=\|\log(\hat R^\top\tilde R)^\vee\|}
\;+\;\frac{\|\tilde x-\hat x\|^2}{2\sigma_x^2}+3\log\sigma_x+\tfrac32\log2\pi ,
$$

and likewise for $(\tilde v,\tilde\omega)$. The "3" per block is the manifold dimension.

### 15.3 Per-increment pseudo-likelihood

For the velocity increments over snapshot spacing $\Delta t$, the effective covariance is

$$
\Sigma_{\text{eff}}
=\mathrm{blkdiag}\Big(\tfrac{\sigma_f^2}{m^2}\,R^\top R,\ \ \sigma_\tau^2\,J^{-1}J^{-\top}\Big)\Delta t
+2\sigma_{\text{obs}}^2 I_6
=\mathrm{blkdiag}\Big(\tfrac{\sigma_f^2}{m^2}I_3,\ \ \sigma_\tau^2 J^{-2}\Big)\Delta t
+2\sigma_{\text{obs}}^2 I_6 .
$$

**Two welcome simplifications relative to the arm:** $\Sigma_{\text{eff}}$ is (i)
**state-independent** — $R^\top R=I$ collapses the multiplicative channel at covariance
level — and (ii) **diagonal** (for diagonal $J$). No Woodbury machinery needed; the pendulum's
scalar-variance `pl_loss` generalizes to a $6$-vector of per-axis variances. The flip side is
an identifiability caveat: increment covariances see only $\sigma_f$, never the
$R$-structure of the diffusion (§18.3).

---

## 16. What the learned subnetworks become

| subnet | pendulum ($SO(3)$) | quadrotor ($SE(3)$) | raw output dim | ground truth |
|---|---|---|---|---|
| `M_net` $\to\mathcal M^{-1}$ | $3\times3$ PSD | $6\times6$ PSD (Cholesky) | 21 | $\mathrm{blkdiag}(\tfrac1mI_3,\,J^{-1})$ — **constant** |
| `V_net` $\to V(q)$ | scalar of $R$ | scalar of $(x_w,R)$ | 1 | $mg\,z$ — linear in $z$, no $R$ |
| `Dw_net` $\to D$ | $3\times3$ PSD | $6\times6$ PSD | 21 | $\mathrm{blkdiag}(d_{\text{lin}}I_3,d_{\text{ang}}I_3)$ — constant |
| `g_net` $\to G$ | $3\times3$ | $6\times4$ | 24 | the constant sparse mixer of §8 |
| `sigma_net` | scalar | force scale + torque scale (structured $18\times6$ diffusion) | 2 (structured) | $(\sigma_f R^\top,\ \sigma_\tau I_3)$ |

Naive input dimension: $12$ ($x_w$ + $\mathrm{vec}R$). But see §17 — the physics says the
momentum dynamics need **at most $R$** (9 numbers), and most blocks need nothing at all.

**The kernel.** [`MaternFeatures`](../src/utils/JAX/gp_model.py#L60-L95) computes geodesic-angle
features on $SO(3)$. The natural extension to $SE(3)=\mathbb{R}^3\times SO(3)$ (as a metric
product) is a **product kernel**: standard Euclidean random Fourier features on $x_w$ times the
existing $SO(3)$ Matérn features on $R$,

$$
\varphi_f(q)=\sqrt{\tfrac2F}\,\cos\big(s_f^\top x_w+\theta_f(R)\,\varpi_f+b_f\big),
$$

with $s_f$ drawn from the Euclidean spectral density. If the priors of §17 are adopted, the
$x_w$ factor disappears from every subnet except (optionally) `V_net`'s $z$ input, and the
existing $SO(3)$ features suffice unchanged.

---

## 17. Structural priors worth baking in

Free accuracy, from the physics above, strongest first:

1. **Position never enters the momentum dynamics.** $\nabla_{x_w}V=mge_3$ is constant, and no
   other term touches $x_w$ — so `M_net`, `Dw_net`, `g_net`, `sigma_net` and even the gravity
   port can take **$R$ alone** (or nothing) as input. This is exact, not an approximation, and
   also sidesteps the non-compactness of $\mathbb{R}^3$ for the GP features (§18.2).
2. **$\mathcal M$, $D$, $G$ are constants.** Learning them as free parameters instead of
   networks is the correct model class; learning them as $q$-dependent maps and *checking they
   come out constant* is the honest ablation ("does the model discover the left-trivialized
   structure?").
3. **$V$ is linear in $z$ and independent of $R$.** Feeding `V_net` only $z$ (1 number instead
   of 12) makes the $SE(2)\times\mathbb{R}$ symmetry exact by construction.
4. **Known sparsity/sign pattern of $G$** (§8): zero $F_x,F_y$ rows, alternating torque signs.
   A soft version: penalize the $F_{x,y}$ rows toward zero.
5. **Diffusion structure:** two scalar scales $(\sigma_f,\sigma_\tau)$ on fixed maps
   $(R^\top,\,I_3)$, instead of a free $18\times6$ matrix — matches the environment exactly and
   is the only version identifiable from data (§18.3).
6. **Conditioning floor on $\mathcal M^{-1}$:** keep the diagonal `epsilon` floor from the
   pendulum model ([`network.py:112`](../src/models/3D_SO3_Windy_Pendulum/ph_gp_sde/network.py#L112));
   with $m$ and $J$ entries differing by an order of magnitude the $6\times6$ block is
   better-conditioned than the arm's but still benefits.

Recommendation: implement (1) and (3) as options and ablate — "does the $SE(3)$ structure
help?" is the reviewer's first question, and this answers it directly.

---

## 18. Practical concerns

### 18.1 Open-loop instability — the analogue of the arm's chaos problem

A quadrotor is **open-loop unstable**: stochastic torques tilt it, tilted thrust stops
cancelling gravity, and it crashes within seconds — physically correct, and observed (every
seed crashed until the demo gained its geometric PD controller,
[`quadrotor.py:902-930`](../envs/port_ham_quadrotor_se3/quadrotor.py#L902-L930)). Consequences for the
learning experiments, parallel to the arm note's chaos section:

- long-horizon single-trajectory MSE will explode for **every** method — say so in the paper
  before a reviewer does;
- data generation must choose between: (a) short horizons with random $u$ around
  $u^\star=\frac{mg}{4k_f}\mathbf 1_4$ (persistently exciting, mirrors the pendulum's
  `--random_u`); (b) the hover controller in the loop plus exploration noise (stationary data
  near the operating point, but the input is then state-correlated); (c) free fall/tumbling
  segments (rich rotational data, useless translational data). Recommendation: (a) for the
  headline dataset, (b) as a distribution-shift evaluation;
- evaluation should lean on per-increment NLL and distributional metrics (ensemble spread,
  energy statistics) rather than long rollouts.

### 18.2 Unbounded position

$x_w$ drifts over $\mathbb{R}^3$ — there is no compact invariant region like the pendulum's
sphere. Prior (1) of §17 removes $x_w$ from every learned input, which resolves the modelling
side entirely; on the data side, normalize rollout-error metrics per unit time and report
altitude separately (gravity makes $z$ the only drifting direction under (a)-type data).

### 18.3 Identifiability

- From translational data only the ratios $k_f/m$, $d_{\text{lin}}/m$, $\sigma_f/m$ are visible
  (gravity fixes the scale through the known $g$); torques add $ak_f/J$, $k_m/J$. This is the
  standard wrench-scale degeneracy — compare learned products, not raw coefficients, in the
  ground-truth-match test.
- The force-channel diffusion's $R$-dependence is **invisible in increment covariances**
  ($\sigma_f^2R^\top R=\sigma_f^2I$, §15.3) — only trajectory-level correlations distinguish
  multiplicative from additive noise. Do not expect (or claim) the model to recover the
  $R^\top$ factor from covariance-based losses alone.
- With `resample_coeffs_every_step=True`, parameter jitter acts as extra process noise the
  model will fold into its diffusion. Keep all `*_std = 0` for the headline runs; use per-step
  resampling as a robustness ablation and per-trajectory resampling
  (`resample_coeffs_every_step=False`) as the domain-randomization ablation.
- $V\mapsto V+\text{const}$ unidentifiable and harmless, as always.

### 18.4 Deterministic wind is unmodelled

Same issue and same resolution as the arm note: the learned model has no $w(t)$ port, so keep
`external_force_std = 0` for headline experiments, or add an explicit time input. (The demo's
`sine` wind is for the video, not for training data.)

---

## 19. Remaining design decisions

Group ($SE(3)$), frame convention (left-trivialized/body), and input definition
($u=\text{rpm}^2$) are **fixed** — see §2. Still open:

| # | decision | recommended | alternative |
|---|---|---|---|
| 1 | dataset input policy | random $u$ near hover, short horizons | hover controller + exploration noise (as shift study) |
| 2 | subnet inputs | $R$ only (+ $z$ for `V_net`) — priors (1),(3) | full $(x_w,R)$, ablate |
| 3 | coefficient randomization in data | all `*_std = 0` headline | per-trajectory std $>0$ ablation |
| 4 | integrator refinement | current $SO(3)$-exp Heun (validated) | full $SE(3)$-exponential update for $x_w$ |
| 5 | model backend | mirror `ph_gp_sde` JAX stack | — |
| 6 | paper's noisy-time channel | off (parity with all baselines) | flag + Itô correction |

---

## 20. Verification: env level and pH-structure level

### 20.1 Done — environment level (11/11 smoke tests + cross-validation)

| # | test | result |
|---|---|---|
| 1 | manifold: $\vert\det R-1\vert$, $\Vert R^\top R-I\Vert$ over 500 noisy steps | $<10^{-14}$ ✅ |
| 2 | hover: $u^\star=\frac{mg}{4k_f}\mathbf 1_4$ at level rest | exact fixed point, drift $=0.0$ ✅ |
| 3 | energy: $u=0$, damping on, wind off | $H$ strictly non-increasing ✅ |
| 4 | yaw sign: speed up rotors 1,3 | $\omega_{b,3}>0$, matches $\tau_z$ row ✅ |
| 5 | reproducibility: same seed | bitwise-identical; obs noise never leaks into dynamics ✅ |
| 6 | resampling modes: per-step vs per-trajectory | both behave as specified ✅ |
| 7 | cross-validation vs gym-pybullet-drones (`Physics.DYN`, real CF2X, same RPM sequence) | gap halves exactly when their $h$ halves (ratio 1.99–2.00) — same continuous dynamics ✅ |
| 8 | accuracy vs fine-grid reference ($h=1/3840$) | ours $\sim$17× closer (1.2 mm vs 21 mm over 2 s) ✅ |
| 9 | Heun self-convergence | ratio 4.03 $\approx$ 4 — second order ✅ |
| 10 | Richardson closure | residual gap (1.42e-3 m / 7.93e-4 rad) equals their predicted Euler error to 3 digits ✅ |

Details in [`envs/port_ham_quadrotor_se3/quadrotor.md`](../envs/port_ham_quadrotor_se3/quadrotor.md) §11. Test 7–10
jointly answer the *"you built the simulator in the form your model assumes"* objection: an
independent simulator (gym-pybullet-drones) solves provably the same continuous dynamics.

### 20.2 Done — pH-structure level, **6/6 passing**

Implemented in [`mini_tests/test_gt_pH_matches_quad_env.py`](../mini_tests/test_gt_pH_matches_quad_env.py)
(the SE(3) analogue of [`test_gt_pH_matches_env.py`](../mini_tests/test_gt_pH_matches_env.py)); run
with `python mini_tests/test_gt_pH_matches_quad_env.py`.

| # | test | what it proves | result |
|---|---|---|---|
| 11 | **GT-pH vs env**: assemble $\mathcal J,\mathcal R,\mathcal G,\Xi$ of §6.1 from scratch, form $(\mathcal J-\mathcal R)\partial_xH+\mathcal Gu+$ wind port, compare to `_compute_rates` over 200 random states | the pH algebra — every block of §6.1 — is right | agrees to $1.4\times10^{-14}$ (rel $2\times10^{-16}$); $\mathcal J+\mathcal J^\top=0$ **exactly**; $\mathcal R\succeq0$ ✅ |
| 12 | **Energy conservation**: $D=0$, $u=0$, wind off; both free-fall ($T=1$ s) and bounded $g=0$ ($T=20$ s) regimes | integrator + sign conventions, no secular drift | rel drift $5.5\times10^{-11}$ / $5.5\times10^{-10}$; ratio **8.00** in all four sweeps $\Rightarrow O(h^3)$ ✅ |
| 13 | **Momentum conservation** (§13.3): horizontal $P_w$, ballistic $P_{w,3}+mgt$, all three of $R\Pi$ | gravity/coadjoint terms, independent of 12 | $5.7\times10^{-7}$, $6.3\times10^{-7}$, $4.4\times10^{-8}$ at $h=5\!\times\!10^{-4}$; ratios **3.99–4.00** $\Rightarrow O(h^2)$ ✅ |
| 14 | **Trivialized gradients**: JAX autodiff of $\mathcal T_R(H)$, $R^\top\nabla_{x_w}V$ vs the closed forms of §5, plus an operator check on a non-trivial $F(R)=\mathrm{tr}(A^\top R)$ | the safe-embedding argument; gravity at the COM exerts no torque | operator exact to $3.0\times10^{-10}$ (finite-difference-limited); $\mathcal T_R(H)=0$ and $R^\top\nabla V$ match **to the last bit**; env torque at rest $=0$ exactly ✅ |
| 15 | **Itô $=$ Strat**: Heun vs Euler–Maruyama under *common random numbers*, refining $h$; run beside a deliberately broken control env whose diffusion depends on $v_b$ | §10.1 — the correction genuinely vanishes | pathwise gap $\to0$, **7.9×** per 8× refinement; paired ensemble mean **4.04×** per 4× ($O(h)$); broken control **plateaus** (1.00× / 0.89×) ✅ |
| 16 | **Passivity / balance**: instantaneous identity over 200 random states; $\dot H\le0$ with $u=0$; integrated balance over a driven rollout ($u\ne0$, wind on) | §11 identity, i.e. $\mathcal J$ antisymmetry | identity exact to $1.1\times10^{-13}$ (rel $5.4\times10^{-16}$); $H$ monotone; driven balance closes at ratio **4.00** ($O(\Delta t^2)$ trapezoid) ✅ |

**Two findings that corrected this document's own predictions:**

1. **Test 12 — energy error is $O(h^3)$, not the $O(h^2)$ predicted.** Ratio $8.00$, robust
   across horizons $T=1,4,16$ s and both the free-fall and bounded regimes. The state error
   *is* $O(h^2)$ (test 13, ratio $4.00$), so the leading state error must be tangent to the
   energy level set. §14 now records this.
2. **Test 15 needed reformulating.** "Ensemble means agree at small $h$" is not the claim and
   is false at any finite $h$ — the paired mean difference is the $O(h)$ discretisation gap
   ($\approx2\times10^{-3}$ at $h=1.25\times10^{-3}$, resolvable at $\sim7\sigma$ with only 120
   paths). The testable claim is *convergence to zero*, which is what is now measured. The
   broken-control arm is what gives the test power: it plateaus at its non-zero Itô
   correction while the true env converges, so a non-vanishing correction would be detected.

Together with §20.1, this closes the objection *"you built the simulator in the form your
model assumes"* from both directions: an independent simulator solves the same continuous
dynamics (§20.1), and the pH structure matrices reproduce the env's drift to machine
precision without ever being used to write it (test 11).

---

## 21. Build order

**Phase 1 — environment. ✅ DONE.**

1. ✅ [`envs/port_ham_quadrotor_se3/quadrotor.py`](../envs/port_ham_quadrotor_se3/quadrotor.py) — env with
   mean$\pm$std coefficients, two wind channels, geometric obs noise, Lie–Heun stepper,
   PyBullet/matplotlib rendering, demo video.
2. ✅ Smoke tests + cross-validation vs gym-pybullet-drones (§20.1).
3. ✅ Algorithm documentation, [`envs/port_ham_quadrotor_se3/quadrotor.md`](../envs/port_ham_quadrotor_se3/quadrotor.md).

**Phase 2 — data & ground truth.**

4. ✅ [`mini_tests/test_gt_pH_matches_quad_env.py`](../mini_tests/test_gt_pH_matches_quad_env.py) —
   tests 11–16 of §20.2, all passing (mirror of
   [`mini_tests/test_gt_pH_matches_env.py`](../mini_tests/test_gt_pH_matches_env.py)).
5. `envs/port_ham_quadrotor_se3/datagen/windy_quadrotor_datagen.py` — mirror of
   [`envs/pendulum_so3/datagen/windy_pendulum_3d_datagen.py`](../envs/pendulum_so3/datagen/windy_pendulum_3d_datagen.py):
   random-$u$-near-hover policy (§18.1), obs $\in\mathbb{R}^{18}$, geometric noise, all
   `*_std = 0` for the headline set.

**Phase 3 — model.**

6. `src/utils/JAX/lie_integrator.py` — `lie_heun_sde_step_se3` (add the $x_w$/$v_b$ lanes to
   the existing $SO(3)$ step).
7. `src/utils/JAX/elbo_loss_jax.py` — $SE(3)$ NLL (§15.2); diagonal per-axis `pl_loss`
   (§15.3 — simpler than the pendulum's, no Woodbury needed).
8. `src/utils/JAX/gp_model.py` — product $\mathbb{R}^3\times SO(3)$ features (§16), only if
   the §17 priors are ablated off.
9. `src/models/SE3_Quadrotor/ph_gp_sde/{network,train}.py` — $\mathcal M^{-1}\!:6\times6$,
   $D:6\times6$, $G:6\times4$, structured diffusion; priors (1),(3) of §17 as flags.

**Phase 4 — experiments.** Baselines (`ph-node`, `ph_nn_ode_lieintegrator`,
`ph_nn_sde`, `neural_sde`, `ph_gp_ode`)
re-pointed at the quadrotor dataset; ablations on the §17 priors; instability-aware evaluation
protocol of §18.1.

---

## 22. Appendix: all environment arguments

`quadrotor_se3(...)` takes **25 constructor arguments**
([`quadrotor.py:179-206`](../envs/port_ham_quadrotor_se3/quadrotor.py#L179-L206)); defaults below are
the code's own. The `symbol` column ties each one back to the math in this document.

### 22.1 Rigid-body constants

| argument | default | symbol | meaning |
|---|---|---|---|
| `g` | `9.81` | $g$ | gravitational acceleration; enters only through $V=mg\,z$ (§4.2) |
| `m` | `1.0` | $m$ | mass; the $mI_3$ block of $\mathcal M$ (§4.1) |
| `J_diag` | `(0.5, 0.5, 1.0)` | $\mathrm{diag}(J)$ | body inertia — **diagonal only**; the $J$ block of $\mathcal M$ |
| `arm` | `1.0` | — | rotor arm length; sets the lever $a=\text{arm}/\sqrt2$ in $G$ (§8). **Never randomized** |
| `dt` | `0.05` | $\Delta t$ | env step. Split internally into **10 substeps** of $h=\Delta t/10$ (§14) — `n_substeps` is hardcoded at [line 528](../envs/port_ham_quadrotor_se3/quadrotor.py#L528) and is *not* an argument |

### 22.2 Randomizable coefficients (mean ± std)

Each pair follows $\text{value}=\text{coeff}$ if $\text{std}=0$, else
$\max\!\big(0,\ \mathcal N(\text{coeff},\text{std}^2)\big)$ (§7). Every `*_std` must be
$\ge0$ or the constructor raises ([line 229](../envs/port_ham_quadrotor_se3/quadrotor.py#L229)).

| argument | default | symbol | meaning |
|---|---|---|---|
| `kf_coeff` / `kf_std` | `1.0` / `0.0` | $k_f$ | thrust gain; **every draw rebuilds $G$** |
| `km_coeff` / `km_std` | `0.1` / `0.0` | $k_m$ | yaw reaction-drag gain; also rebuilds $G$ |
| `linear_damping_coeff` / `linear_damping_std` | `0.1` / `0.0` | $d_{\text{lin}}$ | translational friction, the $d_{\text{lin}}I_3$ block of $D$ |
| `angular_damping_coeff` / `angular_damping_std` | `0.1` / `0.0` | $d_{\text{ang}}$ | rotational friction, the $d_{\text{ang}}I_3$ block of $D$ |
| `resample_coeffs_every_step` | `True` | — | `True`: redrawn once per `step()`, held across the 10 substeps → piecewise-constant parameter noise. `False`: drawn once at `reset()` → per-trajectory domain randomization |

The realized values are exposed every step in
`info = {"wind", "kf", "km", "d_lin", "d_ang"}` — the ground truth a learned model is scored
against.

### 22.3 Wind — deterministic and stochastic (§9)

| argument | default | symbol | meaning |
|---|---|---|---|
| `external_force_type` | `"sine"` | $w(t)$ shape | one of `sine`, `square`, `random`, `constant`; anything else raises ([line 363](../envs/port_ham_quadrotor_se3/quadrotor.py#L363)) |
| `external_force_std` | `1.0` | $\sigma_w$ | amplitude of the deterministic wind $w(t)$ |
| `external_force_direction` | `(1.0, 0.0, 0.0)` | $\hat d$ | world-frame direction; **normalized internally**, must be non-zero ([line 244](../envs/port_ham_quadrotor_se3/quadrotor.py#L244)) |
| `wind_force_std` | `0.0` | $\sigma_f$ | stochastic force channel $\sigma_f R^\top dW_f$ — *multiplicative* |
| `wind_torque_std` | `0.0` | $\sigma_\tau$ | stochastic body-torque channel $\sigma_\tau dW_\tau$ — *additive* |

> **Gotcha.** The defaults are `external_force_type="sine"` with `external_force_std=1.0`, so
> **deterministic wind is ON out of the box** while both stochastic channels are off. For the
> headline datasets set `external_force_std=0.0` — the learned model has no $w(t)$ port and
> would otherwise absorb the wind into $V$ and $D$ (§18.4).

### 22.4 Observation and action API

| argument | default | symbol | meaning |
|---|---|---|---|
| `obs_noise_std` | `0.0` | $\sigma_o$ | geometric observation noise $R\exp([\epsilon]_\times)$ plus additive noise on $x_w,v_b,\omega_b$ (§15.1). Uses a **separate RNG** seeded `seed+1`, so toggling it never perturbs the dynamics; `get_state()` always returns the clean state |
| `ori_rep` | `"rotmat"` | — | only `"rotmat"` is accepted; anything else raises ([line 210](../envs/port_ham_quadrotor_se3/quadrotor.py#L210)) |
| `max_u` | `25.0` | — | defines the `action_space` upper bound **only**. The env does **not** clip $u$ (mirrors the pendulum) |

Derived spaces: `action_space = Box(0, max_u, (4,))`,
`observation_space = Box(-inf, inf, (18,))`.

### 22.5 Rendering and seeding

| argument | default | meaning |
|---|---|---|
| `render_mode` | `None` | `None`, `"human"`, or `"rgb_array"` |
| `render_backend` | `"pybullet"` | `"pybullet"` (real CF2X model; **renderer only**, never steps physics) or `"matplotlib"` (schematic fallback); anything else raises ([line 212](../envs/port_ham_quadrotor_se3/quadrotor.py#L212)) (§9 of `quadrotor.md`) |
| `seed` | `None` | seeds the dynamics RNG; the observation-noise RNG gets `seed+1`. Draw order per step is fixed (wind → coefficients → $dW$) so trajectories are bitwise reproducible |

### 22.6 `reset(seed=..., options={...})`

All four initial-condition options are optional; omitted ones are randomized.

| option | default when omitted | notes |
|---|---|---|
| `x_init` | $\mathcal U(-1,1)^3$ | world position |
| `R_init` | uniform on $SO(3)$ (Shoemake) | any supplied matrix is projected onto $SO(3)$ by polar decomposition |
| `v_init` | $\mathcal U(-1,1)^3$ | **body-frame** linear velocity |
| `omega_init` | $\mathcal U(-1,1)^3$ | **body-frame** angular velocity |

Useful fixed point for tests and controllers: $u^\star=\frac{mg}{4k_f}\mathbf 1_4$ with
`R_init=I`, `v_init=0`, `omega_init=0` is an *exact* equilibrium of the integrator (§13.2).

### 22.7 Config files and CLI overrides

Two ready-made configs live in `configs/quadrotor_se3/envs/`; their keys map 1:1 onto the
arguments tabulated above, and any key that is *not* a constructor argument is a hard error
(so typos surface instead of being silently dropped).

| file | role | differs from the other only in |
|---|---|---|
| [`ode.yaml`](../configs/quadrotor_se3/envs/ode.yaml) | deterministic pH-**ODE** — no Brownian channels | $\sigma_f=0$, $\sigma_\tau=0$, $\sigma_o=0$ |
| [`sde.yaml`](../configs/quadrotor_se3/envs/sde.yaml) | stochastic pH-**SDE** — both diffusion channels on (§9.2) | $\sigma_f=0.5$, $\sigma_\tau=0.1$, $\sigma_o=0.05$ |

Both keep `external_force_std: 0.0` — the deterministic wind port is unmodelled by the learned
models (§18.4). Note that raising it keeps the system an *ODE*: it is a deterministic forcing
term, not a diffusion. Only `wind_force_std` / `wind_torque_std` make it an SDE.

Resolution order is **constructor defaults < config file < CLI flags / kwargs**:

```bash
# use a config as-is
python envs/port_ham_quadrotor_se3/quadrotor.py --config configs/quadrotor_se3/envs/sde.yaml

# override anything from the command line
python envs/port_ham_quadrotor_se3/quadrotor.py --config configs/quadrotor_se3/envs/sde.yaml \
    --external_force_type random --wind_force_std 0.8 --seed 7

# see what a config + flags resolve to, without building the env
python envs/port_ham_quadrotor_se3/env_config.py --config configs/quadrotor_se3/envs/ode.yaml
```

```python
env = quadrotor_se3.from_config("configs/quadrotor_se3/envs/ode.yaml")
env = quadrotor_se3.from_config("configs/quadrotor_se3/envs/sde.yaml", seed=7)  # kwargs win
```

The CLI flags are **generated by reflecting over `quadrotor_se3.__init__`**
([`env_config.py`](../envs/port_ham_quadrotor_se3/env_config.py)), so every argument in §22.1–22.5 is
overridable and a new env argument becomes configurable with no edit to the plumbing.

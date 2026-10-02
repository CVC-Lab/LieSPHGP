# Stochastic Port-Hamiltonian Formulation of an $n$-Joint Windy Robotic Arm on $SO(3)^n$

**Status:** formulation only — no code written yet.
**Purpose:** extend the single-link `windy_pendulum_3d` / `ph_gp_sde` system to a multi-joint
arm, so the NeurIPS reviewers' request ("test on a multi-joint pendulum") can be answered
with the *same* port-Hamiltonian machinery, on a *Lie group* rather than in $\mathbb{R}^n$.

---

## Table of contents

0. [Notation](#0-notation)
1. [What we are generalizing from](#1-what-we-are-generalizing-from)
2. [Configuration manifold](#2-configuration-manifold)
3. [Kinematics](#3-kinematics)
4. [The Hamiltonian: $M(q)$ and $V(q)$](#4-the-hamiltonian-mq-and-vq)
5. [Trivialized gradients on $SO(3)^n$](#5-trivialized-gradients-on-so3n)
6. [The port-Hamiltonian system](#6-the-port-hamiltonian-system)
7. [Dissipation $D(q)$](#7-dissipation-dq)
8. [Input map $g(q)$](#8-input-map-gq)
9. [Wind: deterministic port and stochastic port](#9-wind-deterministic-port-and-stochastic-port)
10. [The full SDE, and Itô vs Stratonovich](#10-the-full-sde-and-itô-vs-stratonovich)
11. [Passivity and energy balance](#11-passivity-and-energy-balance)
12. [Correspondence with the reference paper's $n$-DOF arm](#12-correspondence-with-the-reference-papers-n-dof-arm)
13. [Reduction checks](#13-reduction-checks)
14. [Geometric integrator](#14-geometric-integrator)
15. [Observation model and losses](#15-observation-model-and-losses)
16. [What the learned subnetworks become](#16-what-the-learned-subnetworks-become)
17. [Structural priors worth baking in](#17-structural-priors-worth-baking-in)
18. [Practical concerns](#18-practical-concerns)
19. [Remaining design decisions](#19-remaining-design-decisions)
20. [Verification plan](#20-verification-plan)

---

## 0. Notation

| Symbol | Meaning | Dimension |
|---|---|---|
| $n$ | number of links / joints | — |
| $R_i$ | **world** attitude of link $i$ | $SO(3)$ |
| $q$ | $(R_1,\dots,R_n)$, flattened row-major | $\mathbb{R}^{9n}$ |
| $r_{ik}^\top$ | $k$-th **row** of $R_i$ | $\mathbb{R}^{3}$ |
| $\omega_i$ | body-frame angular velocity of link $i$ | $\mathbb{R}^3$ |
| $\omega$ | $(\omega_1,\dots,\omega_n)$ | $\mathbb{R}^{3n}$ |
| $\Omega_i$ | **relative** (joint) angular velocity at joint $i$ | $\mathbb{R}^3$ |
| $p$ | conjugate momentum, $p = M(q)\omega$ | $\mathbb{R}^{3n}$ |
| $\xi$ | $\partial H/\partial p = M^{-1}(q)p$ (equals $\omega$) | $\mathbb{R}^{3n}$ |
| $m_i,\ \mathbb{I}_i$ | mass, COM inertia (body frame) of link $i$ | $\mathbb{R}$, $\mathbb{R}^{3\times3}$ |
| $c_i$ | joint $i$ $\to$ COM of link $i$, in link-$i$ frame | $\mathbb{R}^3$ |
| $\ell_i$ | joint $i$ $\to$ joint $i{+}1$, in link-$i$ frame | $\mathbb{R}^3$ |
| $a_i$ | wind drag area/coefficient of link $i$ | $\mathbb{R}_{\ge0}$ |
| $d_i$ | viscous friction coefficient at joint $i$ | $\mathbb{R}_{\ge0}$ |
| $u_i$ | actuator torque at joint $i$, link-$i$ frame | $\mathbb{R}^3$ |
| $e_z$ | $(0,0,1)^\top$ | $\mathbb{R}^3$ |

**Hat / vee.**

$$
[a]_\times=\begin{pmatrix}0&-a_3&a_2\\ a_3&0&-a_1\\ -a_2&a_1&0\end{pmatrix},
\qquad [a]_\times b = a\times b,\qquad [a]_\times^\top=-[a]_\times .
$$

Two identities used constantly below, valid for $R\in SO(3)$:

$$
R\,[a]_\times R^\top=[Ra]_\times,
\qquad
R^\top(u\times v)=(R^\top u)\times(R^\top v).
$$

---

## 1. What we are generalizing from

Your current model, [`src/models/3D_SO3_Windy_Pendulum/ph_gp_sde/network.py`](src/models/3D_SO3_Windy_Pendulum/ph_gp_sde/network.py), implements
for a **single** rigid body:

$$
H(q,p)=\tfrac12\,p^\top M^{-1}(q)\,p+V(q),
$$

$$
\dot p \;=\; \underbrace{p\times \tfrac{\partial H}{\partial p}}_{\text{gyroscopic}}
\;+\;\underbrace{\textstyle\sum_{k=1}^{3} r_k\times\tfrac{\partial H}{\partial r_k}}_{\text{"grav" term}}
\;-\;D(q)\tfrac{\partial H}{\partial p}\;+\;g(q)u ,
$$

(line 250) with the diffusion hard-coded to the single-pendulum lever arm (lines 189–202):

$$
dp_{\text{stoch}} = R^\top\!\big(\,\ell\,R e_z \times \sigma(q)\,dW\,\big).
$$

**Key observation.** The drift above is *not* pendulum-specific — it is the general
left-trivialized **Hamilton–Poincaré** equation on a Lie group. Only the diffusion encodes
geometry. So the $n$-DOF extension is mostly *structural*: the same equation, repeated
per block, with all inter-link coupling hidden inside $M^{-1}(q)$ and $V(q)$.

---

## 2. Configuration manifold

The arm is a serial chain of $n$ links connected by **ball joints** (3 rotational DOF each),
and each link is described by its **absolute (world-frame) attitude** $R_i\in SO(3)$:

$$
\boxed{\;q=(R_1,\dots,R_n)\in G=SO(3)^n,\qquad \dim G=3n\;}
$$

A serial chain of ball joints has **no kinematic constraints** — each $R_i$ ranges over all of
$SO(3)$ independently — so $G$ is the full product group and the flattened representation
$q\in\mathbb{R}^{9n}$ is redundant but unconstrained.

Three properties of this choice drive everything that follows:

1. **It reduces exactly to the existing environment at $n=1$.** Not approximately — the
   $n=1$ instance is bit-for-bit [`windy_pendulum_3d`](envs/windy_pendulum_3d.py) with the
   parameters of §13.1, which makes it a hard regression test.
2. **$M(q)$ has its cleanest structure in absolute coordinates.** The rotational contribution
   is a *constant* block-diagonal $\mathrm{blkdiag}(\mathbb{I}_1,\dots,\mathbb{I}_n)$ and the
   translational Jacobians are block lower-triangular (§3.3). Since $M^{-1}$ is the hardest
   object the GP has to learn, this matters.
3. **$g(q)$ becomes non-trivial**, and that is a feature. Actuator torques act on *relative*
   joint velocities while the state carries *absolute* rates, so $g(q)=T(q)^\top$ is a genuine
   state-dependent difference operator (§8). `g_net` currently learns a near-identity matrix,
   which is a weak test of the architecture; here it must learn a real map.

Note the distinction between **coordinates** and **velocities**: the configuration is stored
absolutely, but the *relative* joint rate $\Omega_i=\omega_i-R_i^\top R_{i-1}\omega_{i-1}$ is
still the physically meaningful quantity at each joint. It is what friction opposes (§7.1) and
what actuator torque is conjugate to (§8). Both enter through the single matrix $T(q)$.

---

## 3. Kinematics

### 3.1 Configuration and velocity

$$
q=(R_1,\dots,R_n)\in G=SO(3)^n,\qquad \dim G=3n,
$$

$$
\dot R_i = R_i\,[\omega_i]_\times, \qquad i=1,\dots,n .
$$

The Lie algebra is $\mathfrak g=\mathfrak{so}(3)^n\cong\mathbb{R}^{3n}$, and
$\omega=(\omega_1,\dots,\omega_n)$ is the **left-trivialized** (body-frame) velocity.

### 3.2 Forward kinematics

Joint positions in the world frame:

$$
o_1=0,\qquad o_{i+1}=o_i+R_i\ell_i \quad\Longrightarrow\quad o_i=\sum_{j<i}R_j\ell_j .
$$

Centre of mass of link $i$:

$$
\boxed{\;p_i(q)=\sum_{j<i}R_j\ell_j+R_i c_i\;}
$$

### 3.3 Jacobians

Differentiating with $\dot R_j=R_j[\omega_j]_\times$ and $[\,a\,]_\times b=-[b]_\times a$:

$$
\dot p_i=\sum_{j<i}R_j(\omega_j\times \ell_j)+R_i(\omega_i\times c_i)
      =-\sum_{j<i}R_j[\ell_j]_\times\omega_j-R_i[c_i]_\times\omega_i
      \;=:\;J_{v,i}(q)\,\omega .
$$

So $J_{v,i}(q)\in\mathbb{R}^{3\times3n}$ is **block lower-triangular**:

$$
\big[J_{v,i}\big]_j=
\begin{cases}
-R_j[\ell_j]_\times, & j<i,\\[2pt]
-R_i[c_i]_\times, & j=i,\\[2pt]
0, & j>i .
\end{cases}
$$

The rotational Jacobian is trivial — this is the payoff of absolute coordinates:

$$
J_{\omega,i}=\big(0\;\cdots\;I_3\;\cdots\;0\big)\in\mathbb{R}^{3\times 3n}\quad(\text{$I_3$ in block }i).
$$

---

## 4. The Hamiltonian: $M(q)$ and $V(q)$

### 4.1 Kinetic energy and the mass matrix

$$
T=\tfrac12\sum_{i=1}^n\Big(m_i\|\dot p_i\|^2+\omega_i^\top\mathbb{I}_i\,\omega_i\Big)
 =\tfrac12\,\omega^\top M(q)\,\omega ,
$$

$$
\boxed{\;M(q)=\mathrm{blkdiag}\big(\mathbb{I}_1,\dots,\mathbb{I}_n\big)+\sum_{i=1}^n m_i\,J_{v,i}(q)^\top J_{v,i}(q)\;}\qquad \in\mathbb{R}^{3n\times3n},\ \ \text{SPD}.
$$

**Closed form, block $(j,k)$.** Write $u_{ij}:=\ell_j$ if $j<i$ and $u_{ii}:=c_i$. Then

$$
\boxed{\;M_{jk}(q)=\mathbb{I}_j\,\delta_{jk}\;-\;\sum_{i\ge\max(j,k)} m_i\,[u_{ij}]_\times\,R_j^\top R_k\,[u_{ik}]_\times\;}
$$

This is the exact analogue of the $M(q)$ in the reference paper — same object, different
manifold.

> **Invariance (important, §17).** $M_{jk}$ depends on $q$ **only through the relative
> rotations $R_j^\top R_k$**. Hence under a global world rotation $R_i\mapsto h R_i$
> ($h\in SO(3)$ fixed), $M(q)$ is invariant. Physically obvious — kinetic energy does not
> know which way "north" is — and a strong inductive bias for the GP.

### 4.2 Potential energy

$$
V(q)=\sum_{i=1}^n m_i\,g\,\big\langle e_z,\;p_i(q)\big\rangle .
$$

Swapping the order of summation gives a compact closed form:

$$
\boxed{\;V(q)=g\,e_z^\top\sum_{j=1}^n R_j\Big(\mu_j^{>}\,\ell_j+m_j c_j\Big),\qquad
\mu_j^{>}:=\sum_{i>j}m_i\;}
$$

i.e. link $j$ carries its own COM ($m_jc_j$) plus the total downstream mass hanging off its
tip ($\mu_j^{>}\ell_j$).

Unlike $M$, $V$ is **not** globally rotation-invariant — gravity picks out $e_z$. It *is*
invariant under rotation about $e_z$ (a residual $SO(2)$ symmetry, and hence a conserved
vertical angular momentum when wind and friction are off — a useful conservation test).

### 4.3 Momentum and Hamiltonian

$$
p:=M(q)\,\omega\in\mathbb{R}^{3n},\qquad
\boxed{\;H(q,p)=\tfrac12\,p^\top M^{-1}(q)\,p+V(q)\;}
$$

Note $p$ is a **quasi-momentum** — conjugate to the body-frame rate $\omega$, not to a
coordinate chart. This is the same convention the existing `(q,p)` integrator uses
([`lie_integrator.py`](src/utils/JAX/lie_integrator.py)).

---

## 5. Trivialized gradients on $SO(3)^n$

The Hamiltonian is stored as a function of the redundant $q\in\mathbb{R}^{9n}$, but the
manifold is only $3n$-dimensional. The bridge is the **left-trivialized derivative**.

Perturb $R_i\mapsto R_i\exp([\phi_i]_\times)$, $\phi_i\in\mathbb{R}^3$. For a row vector $a^\top$,

$$
a^\top[\phi]_\times = (a\times\phi)^\top ,
$$

so row $k$ of $R_i$ moves by $\delta r_{ik}=r_{ik}\times\phi_i$, and for any $F$,

$$
\delta F=\sum_{k=1}^3\frac{\partial F}{\partial r_{ik}}\cdot(r_{ik}\times\phi_i)
       =\phi_i\cdot\sum_{k=1}^3\Big(\frac{\partial F}{\partial r_{ik}}\times r_{ik}\Big).
$$

Define the **body-frame generalized-force operator**

$$
\boxed{\;\mathcal T_i(F):=\sum_{k=1}^{3} r_{ik}\times\frac{\partial F}{\partial r_{ik}}
\;=\;-\frac{\partial F}{\partial\phi_i}\;}
$$

Two properties:

- $\mathcal T_i$ is exactly the code's
  `grav = jnp.sum(jnp.cross(R_3x3, dHdq_3x3, axis=-1), axis=0)`
  ([network.py:247](src/models/3D_SO3_Windy_Pendulum/ph_gp_sde/network.py#L247)).
- $\mathcal T_i(F)$ is **independent of how $F$ is extended off the manifold**, because it is a
  directional derivative *along the group action*. So there is no constraint/projection issue
  from working in the redundant $9n$ embedding. This is what makes the $\mathbb{R}^{9n}$
  representation safe.

---

## 6. The port-Hamiltonian system

Because $SO(3)^n$ is a **direct product** group, the adjoint acts block-wise,
$\mathrm{ad}_\xi\eta\big|_i=\xi_i\times\eta_i$. Therefore the Hamilton–Poincaré equations
are the single-body equations *repeated per block*, with **all coupling living inside $H$**
through $M^{-1}(q)$ and $V(q)$.

With $\xi:=\partial H/\partial p=M^{-1}(q)p$ (block $i$ equals $\omega_i$):

$$
\boxed{
\begin{aligned}
\dot R_i &= R_i\,[\xi_i]_\times, \\[4pt]
\dot p_i &= \underbrace{p_i\times\xi_i}_{\text{gyroscopic }(\mathrm{ad}^*_\xi p)_i}
\;+\;\underbrace{\mathcal T_i(H)}_{\text{gravity + inter-link Coriolis}}
\;-\;\underbrace{\big(D(q)\xi\big)_i}_{\text{dissipation}}
\;+\;\underbrace{\big(g(q)u\big)_i}_{\text{actuation}} .
\end{aligned}}
$$

### 6.1 Matrix form

With $x=(q,p)\in\mathbb{R}^{9n}\times\mathbb{R}^{3n}$,

$$
\dot x=\big(\mathcal J(x)-\mathcal R(q)\big)\frac{\partial H}{\partial x}+\mathcal G(q)\,u+\text{(noise)},
$$

$$
\mathcal J(x)=\begin{pmatrix}0_{9n\times9n} & B(q)\\[2pt] -B(q)^\top & \hat P\end{pmatrix},
\qquad
\mathcal R(q)=\begin{pmatrix}0&0\\ 0&D(q)\end{pmatrix},
\qquad
\mathcal G(q)=\begin{pmatrix}0\\ B_u(q)\end{pmatrix},
$$

where

$$
B(q)=\mathrm{blkdiag}(B_1,\dots,B_n),\qquad
B_i=\begin{pmatrix}[r_{i1}]_\times\\ [r_{i2}]_\times\\ [r_{i3}]_\times\end{pmatrix}\in\mathbb{R}^{9\times3},
\qquad
\hat P=\mathrm{blkdiag}\big([p_1]_\times,\dots,[p_n]_\times\big).
$$

**Consistency checks.**

- Row $k$ of $B_i\xi_i$ is $[r_{ik}]_\times\xi_i=r_{ik}\times\xi_i$, which is row $k$ of
  $R_i[\xi_i]_\times$. ✔
- Block $i$ of $-B(q)^\top\partial H/\partial q$ is
  $-\sum_k[r_{ik}]_\times^\top\partial H/\partial r_{ik}
  =\sum_k r_{ik}\times\partial H/\partial r_{ik}=\mathcal T_i(H)$. ✔
- $\hat P^\top=-\hat P$ and the off-diagonal pairing is antisymmetric by construction, so
  $\mathcal J=-\mathcal J^\top$. ✔ And $\mathcal R\succeq0$. ✔

### 6.2 Where the hard multibody physics goes

You never write Christoffel symbols. The full Coriolis/centrifugal field is split into
two structurally distinct pieces that the pH form produces automatically:

$$
\underbrace{p_i\times\xi_i}_{\text{single-body gyroscopic}}
\qquad\text{and}\qquad
\underbrace{\mathcal T_i\!\Big(\tfrac12 p^\top M^{-1}(q)p\Big)}_{\text{inter-link coupling}} .
$$

The second term is exactly what makes an $n$-link arm hard, and it is obtained by one
automatic-differentiation call through $M^{-1}(q)$ — precisely what
[`drift_p`](src/models/3D_SO3_Windy_Pendulum/ph_gp_sde/network.py#L211-L253) already does
via `jax.grad(H_of_q)(q)`.

### 6.3 Gravity in closed form

Using §4.2 and the identity $R^\top(u\times v)=(R^\top u)\times(R^\top v)$,

$$
\boxed{\;\mathcal T_j(V)=-g\Big(\mu_j^{>}\,[\ell_j]_\times+m_j\,[c_j]_\times\Big)R_j^\top e_z\;}
$$

Read physically: *torque on link $j$ from the weight of everything downstream, transmitted
through the lever $\ell_j$, plus the torque from its own weight through the lever $c_j$* —
both expressed in link-$j$'s body frame. This closed form is an exact analytic reference
for the ground-truth test (§20).

---

## 7. Dissipation $D(q)$

Two physically distinct sources, both PSD.

### 7.1 Joint friction (acts on *relative* rates)

$$
\Omega_i=\omega_i-R_i^\top R_{i-1}\,\omega_{i-1},\qquad \omega_0:=0
\qquad\Longleftrightarrow\qquad
\Omega=T(q)\,\omega ,
$$

with $T(q)\in\mathbb{R}^{3n\times3n}$ **block lower-bidiagonal**:

$$
T_{ii}=I_3,\qquad T_{i,i-1}=-R_i^\top R_{i-1},\qquad \text{else }0 .
$$

Rayleigh dissipation $\tfrac12\sum_i d_i\|\Omega_i\|^2$ gives

$$
\boxed{\;D_{\mathrm{joint}}(q)=T(q)^\top\,\mathrm{blkdiag}(d_1I_3,\dots,d_nI_3)\,T(q)\;\succeq0\;}
$$

### 7.2 Air drag (acts on *absolute* rates)

$$
D_{\mathrm{air}}=\mathrm{blkdiag}(\kappa_1 I_3,\dots,\kappa_n I_3)\succeq 0 .
$$

### 7.3 Total, with the existing "varying friction" modulation

Keeping the modulation used in the current env
([`_variable_friction`](envs/windy_pendulum_3d.py#L227-L234)) — height term + speed term —
generalized per link:

$$
D(q,\omega)=\sum_{i}\rho_i(q,\omega)\,\Big(\text{contribution of link/joint }i\Big),
\qquad
\rho_i = 1+\tfrac12\underbrace{\tfrac12\big(1-e_z^\top R_ie_z\big)}_{\text{height}}+\tfrac12\tanh\|\omega_i\| .
$$

Note $D_{\mathrm{joint}}$ depends on $q$ only through $R_i^\top R_{i-1}$ — the same
relative-rotation invariance as $M$ (§17).

---

## 8. Input map $g(q)$

Let $u_i\in\mathbb{R}^3$ be the actuator torque at joint $i$, expressed in link-$i$'s body
frame. The mechanical power injected at joint $i$ is $u_i^\top\Omega_i$, so

$$
u^\top\Omega=u^\top T(q)\,\omega=\big(T(q)^\top u\big)^\top\omega
\qquad\Longrightarrow\qquad
\boxed{\;g(q)=B_u(q)=T(q)^\top\;}
$$

Written out:

$$
\big(g(q)u\big)_i=u_i-R_i^\top R_{i+1}\,u_{i+1},\qquad u_{n+1}:=0 .
$$

That is Newton's third law at the joint: the torque applied to link $i$ reacts on its parent.

> **Elegant consequence.** The *same* $T(q)$ appears in both the input map and the joint
> friction:
> $$D_{\mathrm{joint}}(q)=g(q)\,\mathrm{blkdiag}(d_iI_3)\,g(q)^\top .$$
> Joint friction is literally "negative output feedback through the actuation port" —
> $u=-\mathrm{blkdiag}(d_i)\,y$ with $y=g^\top\xi=\Omega$. This is a clean structural fact
> worth stating in the paper, and it is a useful *tie* between `g_net` and `Dw_net` that
> could be imposed as a soft constraint.

**Actuation levels.** Fully actuated: $m=3n$, $u\in\mathbb{R}^{3n}$. Under-actuated: drop
columns of $g$ (e.g. an acrobot-like arm with a passive first joint) — this is where IDA-PBC
becomes interesting.

---

## 9. Wind: deterministic port and stochastic port

### 9.1 The wrench-to-joint-torque map

Given world-frame forces $f_i$ applied at each link's COM, the generalized force is
$\tau=\sum_i J_{v,i}(q)^\top f_i$. Using the block structure of §3.3 and
$[a]_\times^\top=-[a]_\times$:

$$
\boxed{\;\tau_j=\Phi_j\big(q;\{f_i\}\big):=[\ell_j]_\times R_j^\top\!\!\sum_{i>j}f_i\;+\;[c_j]_\times R_j^\top f_j\;}
$$

Every body force in the system flows through this one map:

| force set | result |
|---|---|
| $f_i=-m_i g\,e_z$ | gravity — reproduces §6.3 exactly |
| $f_i=a_i F(t)$ | wind |

### 9.2 Wind field

A single world-frame wind field acts on all links:

$$
F(t)\,dt \;=\; \underbrace{w(t)\,\mathrm{d}\,dt}_{\text{deterministic (sine/square/const)}}\;+\;\underbrace{\sigma(q)\,dW_t}_{\text{stochastic}},\qquad W_t\in\mathbb{R}^3 .
$$

Substituting $f_i=a_iF$ into $\Phi$ gives the **same** matrix for both channels:

$$
\boxed{\;\Sigma(q)\in\mathbb{R}^{3n\times3},\qquad
\Sigma(q)_j=\Big(A_j^{>}[\ell_j]_\times+a_j[c_j]_\times\Big)R_j^\top,
\qquad A_j^{>}:=\sum_{i>j}a_i\;}
$$

so that

$$
\tau^{\mathrm{wind}}_{\text{det}}=\Sigma(q)\,w(t)\,\mathrm d,
\qquad
d p^{\mathrm{wind}}_{\text{stoch}}=\sigma(q)\,\Sigma(q)\,dW_t .
$$

### 9.3 Why this matters

- **The diffusion is low rank.** Only $3$ Brownian motions drive $3n$ momentum dimensions.
  That is physically correct (there is one wind field, not $n$ independent ones), and it
  matches the structured $\xi(X)\,\delta B_t$ channel of the reference paper rather than a
  generic full-rank diffusion.
- **Deterministic and stochastic wind share $\Sigma(q)$** — same lever arms. If we ever model
  the deterministic wind explicitly, the two ports are tied.
- $\Sigma(q)$ depends on $q$ only through the *absolute* $R_j^\top$, so unlike $M$ and $D$ it
  is **not** globally rotation invariant. Correct — the wind direction is a world-frame object.

---

## 10. The full SDE, and Itô vs Stratonovich

$$
\boxed{
\begin{cases}
dR_i=R_i[\xi_i]_\times\,dt, & i=1,\dots,n,\\[6pt]
dp=\Big(\hat P\,\xi+\mathcal T(H)-D(q)\xi+g(q)u+\Sigma(q)\,w(t)\mathrm d\Big)dt
\;+\;\sigma(q)\,\Sigma(q)\circ dW_t ,
\end{cases}}
$$

with $\xi=M^{-1}(q)p$ and $\circ$ denoting Stratonovich (the convention the Heun scheme
integrates).

### 10.1 Itô $=$ Stratonovich here

The general correction from Stratonovich to Itô is
$\tfrac12\sum_{k=1}^{3}\big(\partial_x\Xi_{\cdot k}\big)\,\Xi_{\cdot k}$, where
$\Xi\in\mathbb{R}^{(9n+3n)\times3}$ is the full diffusion matrix:

$$
\Xi(x)=\begin{pmatrix}0_{9n\times3}\\ \sigma(q)\Sigma(q)\end{pmatrix}.
$$

Now: (i) $\Xi$ depends on $x$ only through $q$, so $\partial\Xi/\partial p=0$; and
(ii) the $q$-rows of $\Xi$ are zero, so the only non-zero entries of $\Xi_{\cdot k}$ sit in
the $p$-block. Therefore

$$
\sum_m \frac{\partial \Xi_{\cdot k}}{\partial x_m}\,\Xi_{mk}
=\sum_{m\in p\text{-block}}\underbrace{\frac{\partial \Xi_{\cdot k}}{\partial p_m}}_{=\,0}\Xi_{mk}=0 .
$$

$$
\boxed{\ \text{The Itô correction vanishes: the Stratonovich and Itô drifts coincide.}\ }
$$

This is the same property your current $n=1$ system has, and it means the Lie–Heun
(Stratonovich) integrator needs no correction term. It is **not** true of the reference
paper's eq. (113): its $\sigma^2\partial_x\big(S\partial_xH\big)S\partial_xH$ term comes from
the *noisy-time* channel $S\partial_xH\,\delta W_t$, which our environment does not have
(see §12.2).

---

## 11. Passivity and energy balance

Since $\mathcal J=-\mathcal J^\top$, the term $\nabla H^\top\mathcal J\nabla H$ vanishes and

$$
\boxed{\;\frac{dH}{dt}=-\,\xi^\top D(q)\,\xi\;+\;y^\top u\;+\;\xi^\top\Sigma(q)w(t)\mathrm d,
\qquad y:=g(q)^\top\xi=\Omega\;}
$$

so with the wind off the system is **passive** with respect to the collocated pair
$(u,\,y=\Omega)$ — storage function $H$, dissipation rate $\xi^\top D\xi\ge0$.

Consequences:

- Your IDA-PBC controller
  ([`ph_gp_sde/controller.py`](src/models/3D_SO3_Windy_Pendulum/ph_gp_sde/controller.py))
  generalizes with no structural change — same collocated output, same energy-shaping
  argument, now on $SO(3)^n$.
- With $D=0$, $u=0$, $w=0$, $\sigma=0$: $H$ is exactly conserved $\Rightarrow$ a sharp
  integrator test (§20).
- With additionally $\sigma=0$ and gravity along $e_z$: the vertical component of total
  angular momentum is conserved (residual $SO(2)$ symmetry of $V$) $\Rightarrow$ a second,
  independent test.

---

## 12. Correspondence with the reference paper's $n$-DOF arm

### 12.1 Structure matrices, side by side

The reference paper (Euclidean, $q\in\mathbb{R}^n$, $p=M(q)\dot q$):

$$
J=\begin{pmatrix}0&I_n\\ -I_n&0\end{pmatrix},\quad
R=\begin{pmatrix}0&0\\ 0&D(p,q)\end{pmatrix},\quad
g=\begin{pmatrix}0\\ B(q)\end{pmatrix},\quad S=J-R .
$$

Ours (Lie group, $q\in SO(3)^n$):

| Block | Euclidean $n$-DOF | $SO(3)^n$ $n$-joint |
|---|---|---|
| $(q,p)$ | $I_n$ | $B(q)=\mathrm{blkdiag}(B_i)$, $B_i\in\mathbb{R}^{9\times3}$ — the tangent map of the group action |
| $(p,q)$ | $-I_n$ | $-B(q)^\top$ — automatically yields $\mathcal T_i(H)$ |
| $(p,p)$ | $0$ | $\hat P=\mathrm{blkdiag}([p_i]_\times)$ — **new**: the $\mathrm{ad}^*_\xi p$ gyroscopic term |
| $R$ | $D(p,q)$ | $D(q,\omega)=D_{\mathrm{air}}+T^\top\mathrm{blkdiag}(d_i)T$ |
| $g$ | $B(q)$ | $g(q)=T(q)^\top$ |
| state dim | $2n$ | $9n+3n=12n$ (manifold dim $6n$) |

**Summary of the change:** the identity coupling block $I_n$ becomes the group-action map
$B(q)$, and a *new* skew block $\hat P$ appears. Everything else keeps its shape. This is the
entire content of "lifting the $n$-DOF arm to $SO(3)$".

### 12.2 Noise channels

The paper's eq. (112) has **two** noise channels:

$$
\delta X_t=S\partial_xH\,\delta t+\underbrace{S\partial_xH\,\delta W_t}_{\text{(a) noisy time}}+g u\,\delta t+\underbrace{\xi(X)\delta B_t}_{\text{(b) model noise}} .
$$

Our system implements **(b) only** — a physical wind-force noise with the structured
coefficient $\Sigma(q)$ of §9.2. This is a deliberate choice: it keeps exact parity with
`windy_pendulum_3d` (which has only the wind channel), so all existing baselines
(`ph_nn_ode`, `ph_nn_sde`, `neural_sde`, `ph_gp_ode`) stay directly comparable.

Adding channel (a) — $Z_t=t+\sigma W_t$, "randomly-perturbed time" — is a one-line change to
the drift *plus* the non-vanishing Itô correction
$\sigma^2\partial_x\big(S\partial_xH\big)S\partial_xH$. It is worth doing **only if** we want
to claim coverage of the paper's full model class. Recommendation: keep (b) as default,
expose (a) behind a flag.

---

## 13. Reduction checks

### 13.1 $n=1$ must reproduce `windy_pendulum_3d` exactly

Set $n=1$, $c_1=\ell e_z$, $\mathbb{I}_1=\mathrm{diag}(0,0,m\ell^2)$, $a_1=1$
(and $\ell_1$ irrelevant — no downstream link):

| quantity | our formula | existing code | match |
|---|---|---|---|
| $M$ | $\mathrm{diag}(0,0,m\ell^2)+m\ell^2\,\mathrm{diag}(1,1,0)=m\ell^2 I_3$ | `self.I` [windy_pendulum_3d.py:179](envs/windy_pendulum_3d.py#L179) | ✔ |
| $V$ | $m g\,e_z^\top R c_1=m g \ell\,R_{22}$ | `V_GT` [test_gt_pH_matches_env.py:44](mini_tests/test_gt_pH_matches_env.py#L44) | ✔ |
| $\mathcal T(V)$ | $-mg\ell\,[e_z]_\times R^\top e_z=mg\ell\,(R^\top e_z)\times e_z$ | `tau_g_body` [windy_pendulum_3d.py:261](envs/windy_pendulum_3d.py#L261) | ✔ |
| $\Sigma$ | $[c_1]_\times R^\top\Rightarrow\tau=\ell\,(e_z\times R^\top F)$ | `tau_stoch_body` [windy_pendulum_3d.py:281](envs/windy_pendulum_3d.py#L281) | ✔ |
| $D$ | $\kappa_1 I_3$ (no joint-relative term since $\omega_0=0$) | `D_GT` [test_gt_pH_matches_env.py:54](mini_tests/test_gt_pH_matches_env.py#L54) | ✔ |
| $g$ | $T^\top=I_3$ | `g_diag_mat` [windy_pendulum_3d.py:172](envs/windy_pendulum_3d.py#L172) | ✔ |

Derivation of the $\mathcal T(V)$ row: $\tau_g^{\text{env}}=R^\top(\ell Re_z\times(-mge_z))
=-mg\ell\,(e_z\times R^\top e_z)=mg\ell\,(R^\top e_z)\times e_z$. ✔

This is not an approximation — the $n=1$ instance is *bit-for-bit* the existing environment,
which makes it a hard regression test.

### 13.2 Reduction to a chain of point masses

Set $\mathbb{I}_i=0$, $c_i=\ell_i/2$ (or $c_i=\ell_i$ for a bob at the tip) and the system
becomes the classical $n$-fold spherical pendulum — the standard chaotic benchmark.

---

## 14. Geometric integrator

Generalize `lie_heun_sde_step` ([lie_integrator.py:42](src/utils/JAX/lie_integrator.py#L42))
block-wise. State layout inside the scan:

$$
x=\big(\mathrm{vec}(R_1),\dots,\mathrm{vec}(R_n),\,p\big)\in\mathbb{R}^{9n+3n}.
$$

**One Stratonovich Heun substep of size $h$, reusing the same $dW$ in both stages:**

$$
\begin{aligned}
&\textbf{Stage 1:} && \xi^{(1)}=M^{-1}(q)p,\quad \dot p^{(1)}=\text{drift}(q,p,u),\quad \Delta p^{(1)}_{\text{st}}=\sigma(q)\Sigma(q)\,dW,\\
&&&\phi^{(1)}_i=\xi^{(1)}_i h,\\[4pt]
&\textbf{Stage 2 (predictor):} && R_i^{\text{pr}}=R_i\exp\big([\phi^{(1)}_i]_\times\big),\qquad p^{\text{pr}}=p+\dot p^{(1)}h+\Delta p^{(1)}_{\text{st}},\\[4pt]
&\textbf{Stage 3:} && \xi^{(2)}=M^{-1}(q^{\text{pr}})p^{\text{pr}},\quad \dot p^{(2)}=\text{drift}(q^{\text{pr}},p^{\text{pr}},u),\quad \Delta p^{(2)}_{\text{st}}=\sigma(q^{\text{pr}})\Sigma(q^{\text{pr}})dW,\\[4pt]
&\textbf{Stage 4 (corrector):} && \bar\phi_i=\tfrac12\big(\phi^{(1)}_i+\phi^{(2)}_i\big),\quad
R_i^{+}=R_i\exp\big([\bar\phi_i]_\times\big),\\
&&& p^{+}=p+\tfrac{h}{2}\big(\dot p^{(1)}+\dot p^{(2)}\big)+\tfrac12\big(\Delta p^{(1)}_{\text{st}}+\Delta p^{(2)}_{\text{st}}\big).
\end{aligned}
$$

Properties preserved from the $n=1$ case:

- Every $R_i$ stays on $SO(3)$ **by construction** (single $\exp$ per link per step) — no SVD
  projection inside the loop.
- Second-order accurate in the deterministic part; correct Stratonovich limit in the
  stochastic part (Heun $\equiv$ Stratonovich–Milstein for this noise structure).
- Averaging happens in the Lie *algebra*, then one exponential — not an average of two
  rotation matrices.

Implementation cost: `exp_so3` applied $n$ times (`jax.vmap` over the link axis), everything
else unchanged.

---

## 15. Observation model and losses

### 15.1 Observations

$$
\tilde R_i=R_i\exp\big([\epsilon_i]_\times\big),\ \ \epsilon_i\sim\mathcal N(0,\sigma_{\text{obs}}^2 I_3),
\qquad
\tilde\omega_i=\omega_i+\eta_i,\ \ \eta_i\sim\mathcal N(0,\sigma_{\text{obs}}^2 I_3).
$$

Observation vector: $\mathrm{concat}\big(\mathrm{vec}(\tilde R_1),\dots,\mathrm{vec}(\tilde R_n),\tilde\omega\big)\in\mathbb{R}^{12n}$.
This is exactly `add_proper_noise_3d`
([windy_pendulum_3d_datagen.py:79](datasets/windy_pendulum_3d_datagen.py#L79)) applied
per link.

### 15.2 Rollout likelihood

The concentrated-Gaussian NLL on $SO(3)$ becomes a **sum over links**:

$$
-\log p(\tilde q\mid\hat q,\sigma_R)=\sum_{i=1}^{n}\left[\frac{\theta_i^2}{2\sigma_R^2}+3\log\sigma_R+\tfrac32\log2\pi\right],
\qquad
\theta_i=\big\|\log\big(\hat R_i^\top\tilde R_i\big)^\vee\big\| ,
$$

$$
-\log p(\tilde\omega\mid\hat\omega,\sigma_\omega)=\frac{\|\tilde\omega-\hat\omega\|^2}{2\sigma_\omega^2}+3n\log\sigma_\omega+\tfrac{3n}{2}\log2\pi .
$$

The "$3$" per link is the *manifold* dimension of $SO(3)$ — the same argument as in
[`elbo_loss_jax.py`](src/utils/JAX/elbo_loss_jax.py), applied $n$ times.

### 15.3 Per-increment pseudo-likelihood

`pl_loss` generalizes directly. With snapshot spacing $\Delta t$:

$$
\Delta\omega_{\text{obs}}\ \sim\ \mathcal N\Big(\mu(q_t,\omega_t,u_t)\,\Delta t,\ \ \underbrace{\sigma^2(q_t)\,\Sigma\Sigma^\top(q_t)\,\Delta t+2\sigma_{\text{obs}}^2 I_{3n}}_{\Sigma_{\text{eff}}\in\mathbb{R}^{3n\times3n}}\Big).
$$

**Important difference from $n=1$.** $\Sigma_{\text{eff}}$ is now a $3n\times3n$ matrix, and
$\Sigma\Sigma^\top$ has **rank 3** — the observation-noise floor $2\sigma_{\text{obs}}^2I$ is
what makes it invertible. Use the Woodbury identity to keep the cost $O(n)$ instead of
$O(n^3)$:

$$
\Sigma_{\text{eff}}^{-1}=\frac{1}{s}\Big(I-\Sigma\big(\tfrac{s}{\sigma^2\Delta t}I_3+\Sigma^\top\Sigma\big)^{-1}\Sigma^\top\Big),
\qquad s:=2\sigma_{\text{obs}}^2 ,
$$

and $\log\det\Sigma_{\text{eff}}=3n\log s+\log\det\big(I_3+\tfrac{\sigma^2\Delta t}{s}\Sigma^\top\Sigma\big)$
— only a $3\times3$ determinant. The current scalar-variance version
([elbo_loss_jax.py:214](src/utils/JAX/elbo_loss_jax.py#L214)) is the $n=1$ special case.

### 15.4 ELBO

Unchanged in form:

$$
-\mathrm{ELBO}=\underbrace{L_{\text{NLL}}}_{\text{rollout}}+\underbrace{L_{\text{PL}}}_{\text{per-increment}}+\frac{\beta}{N}\sum_{\text{subnets}}\mathrm{KL}\big(q_\psi(w)\,\|\,\mathcal N(0,I)\big).
$$

---

## 16. What the learned subnetworks become

| subnet | current ($n=1$) | $n$-joint | raw output dim | $n{=}2$ | $n{=}3$ |
|---|---|---|---|---|---|
| `M_net` $\to M^{-1}(q)$ | $3\times3$ PSD | $3n\times3n$ PSD (Cholesky) | $\tfrac{3n(3n+1)}{2}$ | 21 | 45 |
| `V_net` $\to V(q)$ | scalar | scalar | 1 | 1 | 1 |
| `Dw_net` $\to D(q)$ | $3\times3$ PSD | $3n\times3n$ PSD | $\tfrac{3n(3n+1)}{2}$ | 21 | 45 |
| `g_net` $\to g(q)$ | $3\times3$ | $3n\times m$ | $3nm$ | 36 | 81 |
| `sigma_net` | scalar $\sigma$ | $\Sigma_\theta(q)\in\mathbb{R}^{3n\times3}$ (**low rank**) | $9n$ | 18 | 27 |

Input dimension to every subnet: $9n$.

**The one non-trivial change: the kernel.**
[`MaternFeatures`](src/utils/JAX/gp_model.py#L60-L95) currently computes a *single* geodesic
angle against random base rotations $W_f$:

$$
\theta_f(R)=\arccos\!\Big(\tfrac{\mathrm{tr}(W_f^\top R)-1}{2}\Big),
\qquad \varphi_f(R)=\sqrt{\tfrac{2}{F}}\cos\big(\theta_f\,\varpi_f+b_f\big).
$$

On the product manifold the natural metric is
$d(q,q')^2=\sum_{i}\theta(R_i,R_i')^2$, so the random-Fourier features become

$$
\boxed{\;\varphi_f(q)=\sqrt{\tfrac{2}{F}}\cos\Big(\sum_{i=1}^{n}\theta_f^{(i)}(R_i)\,\varpi_f^{(i)}+b_f\Big)\;}
$$

with independent base rotations $W_f^{(i)}$ and frequencies $\varpi_f^{(i)}$ per link. This is
the RFF approximation of the **product Matérn kernel** on $SO(3)^n$ — one extra sum over
links in `__call__`, roughly ten lines. (An additive/separable kernel
$k(q,q')=\sum_i k_i(R_i,R_i')$ is also valid and cheaper, but cannot represent inter-link
products, which is exactly what $M^{-1}(q)$ needs.)

---

## 17. Structural priors worth baking in

These are free accuracy — they come from the physics derived above, not from data.

1. **Global rotation invariance of $M$ and $D_{\mathrm{joint}}$.**
   Both depend on $q$ only through relative rotations $R_j^\top R_k$ (§4.1, §7.1). Feeding
   `M_net` and `Dw_net` the relative rotations $\{R_i^\top R_{i+1}\}$ instead of the raw
   absolute $\{R_i\}$ makes the invariance *exact by construction* and shrinks the input
   from $9n$ to $9(n-1)$.
2. **$V$ is $SO(2)$-invariant about $e_z$.** $V$ depends on $q$ only via $\{R_j^\top e_z\}$
   ($3n$ numbers instead of $9n$) — a large reduction for `V_net`.
3. **$\Sigma$ and gravity share the map $\Phi$ (§9.1).** If we ever expose the deterministic
   wind as a known input, `sigma_net` and the gravity port are tied.
4. **$D_{\mathrm{joint}}=g\,\mathrm{blkdiag}(d_i)\,g^\top$ (§8).** A soft penalty tying
   `Dw_net` to `g_net` encodes Newton's third law.
5. **Conditioning floor on $M^{-1}$.** Keep the `epsilon=1.0` diagonal floor
   ([network.py:112](src/models/3D_SO3_Windy_Pendulum/ph_gp_sde/network.py#L112)); with $3n$
   coupled dimensions the condition number of $M$ grows with $n$, so this matters *more*, not
   less.

Recommendation: implement (1) and (2) as options and ablate them — "does the Lie-group
structure help?" is precisely the question a reviewer will ask, and this gives a direct answer.

---

## 18. Practical concerns

### 18.1 Chaos — the single biggest evaluation risk

A 3D double or triple spherical pendulum is **chaotic**: nearby trajectories separate as
$e^{\lambda t}$ with a Lyapunov time of order a second. Consequences:

- Long-horizon single-trajectory MSE will explode for **every** method, including a perfect
  model integrated with a slightly different scheme. A reviewer who sees only that plot will
  read it as "the model fails".
- The honest protocol, which also happens to favour a well-specified model:
  - short rollout horizons (`num_points` $\in[3,5]$ — what the current trainer already uses);
  - **per-increment** metrics (`pl_loss` NLL) that do not compound;
  - **distributional** metrics: ensemble spread vs. truth, energy statistics
    $H(t)$, invariant-measure comparison over long horizons;
  - report the Lyapunov time of the chosen configuration so the horizon choice is justified.
- Say this explicitly in the paper rather than letting a reviewer say it.

### 18.2 Cost scaling

| item | scaling |
|---|---|
| observation / state | $12n$ |
| $M^{-1}$ Cholesky construction | $O(n^2)$ outputs |
| $\mathcal T(H)$ via `jax.grad` through $M^{-1}$ | $\sim O(n^2)$ |
| Lie–Heun substep | $2\times$ drift $+$ $n$ exponentials |

$n\in\{2,3\}$ is comfortable; $n\ge5$ needs profiling. Suggest $n=2$ (double spherical
pendulum — classic, chaotic, cheap) as the headline result, $n=3$ as a scaling ablation.

### 18.3 Identifiability

- $M$ and $g$ are separately identifiable only if $u$ is persistently exciting — keep
  `--random_u` on.
- $V\mapsto V+\mathrm{const}$ is unidentifiable and harmless.
- With $\sigma_{\text{obs}}>0$, there is a well-known trade-off between process noise
  $\sigma\Sigma$ and observation noise. Keep $\sigma_{\text{obs}}$ **frozen** at the dataset
  value, exactly as the current code does
  ([network.py:78](src/models/3D_SO3_Windy_Pendulum/ph_gp_sde/network.py#L78)) — letting it
  float lets the optimizer hide model bias in it.

### 18.4 Deterministic wind is unmodelled

The learned model has no $w(t)$ port, so with `external_force_std > 0` it must absorb the
wind into $V$ and $D$. Keep `external_force_std = 0` for the headline $n$-DOF experiments (as
the current defaults do), or add an explicit time input.

---

## 19. Remaining design decisions

Joint type (ball) and coordinates (absolute) are **fixed** — see §2. What is still open:

| # | decision | recommended | alternative |
|---|---|---|---|
| 1 | default $n$ | $2$ headline, $3$ ablation | — |
| 2 | env backend | JAX (jit + autodiff gives $\mathcal T(H)$ free, fast datagen) | NumPy (requires hand-derived $\partial M^{-1}/\partial q$) |
| 3 | noise structure | shared wind, $\Sigma(q)\in\mathbb{R}^{3n\times3}$ | $+$ independent per-joint noise ($3n$ Brownian motions) |
| 4 | paper's noisy-time channel | off by default | flag (§12.2), needs the Itô correction |

---

## 20. Verification plan

Each test isolates one failure mode.

| # | test | what it proves | tolerance |
|---|---|---|---|
| 1 | **$n=1$ reduction**: run the $n$-link env with the §13.1 parameters, compare to `windy_pendulum_3d` step-for-step | the generalization is exact, not merely similar | $\|\Delta R\|_F,\|\Delta\omega\|\lesssim10^{-10}$ |
| 2 | **GT-pH vs env**: hand-code $M,V,D,g,\Sigma$ analytically, run through the pH drift, compare to the env rollout (generalizes [`test_gt_pH_matches_env.py`](mini_tests/test_gt_pH_matches_env.py)) | the pH algebra ($\hat P$, $\mathcal T$, $B$) is right | $\lesssim10^{-9}$ over 100 steps |
| 3 | **Energy conservation**: $D=0$, $u=0$, $w=0$, $\sigma=0$ | integrator + $\mathcal T(H)$ sign conventions | $\|\Delta H/H\|=O(h^2)$, no secular drift |
| 4 | **$SO(2)$ momentum conservation**: same, check $e_z^\top\sum_i R_i\mathbb{I}_i\omega_i$-type invariant | gravity term is correct, independently of test 3 | $O(h^2)$ |
| 5 | **Manifold constraint**: $\|R_i^\top R_i-I\|$, $\det R_i-1$ over a long rollout | the exp-map integrator, no SVD needed | $\lesssim10^{-12}$ |
| 6 | **Passivity**: with $D\succ0$, $u=0$, check $\dot H\le0$ numerically | $D\succeq0$ construction | monotone |
| 7 | **Analytic gravity**: compare `jax.grad`-based $\mathcal T(V)$ against the closed form §6.3 | autodiff path matches hand derivation | $\lesssim10^{-12}$ |
| 8 | **Itô$=$Strat**: compare Heun vs Euler–Maruyama ensemble means at small $h$ | §10.1 claim | means agree within MC error |
| 9 | **Independent dynamics cross-check**: compare a short rollout against a Featherstone / MuJoCo model of the same chain | the environment is not "pH-shaped by assumption" | $\lesssim10^{-4}$ |

Test 9 is the one that answers the reviewer objection *"you built the simulator in the same
form your model assumes"*. The answer is that the pH form **is** the exact
Lagrangian mechanics of the chain — not an approximation — but an independent simulator
makes that verifiable rather than merely asserted.

---

## 21. Build order

**Phase 1 — physics (this document's content).**

1. `envs/windy_arm_nlink_so3.py` — the environment. Config: $n$, $\{m_i,\ell_i,c_i,\mathbb{I}_i,a_i,d_i,\kappa_i\}$,
   `varying_friction`, wind type/std, `wind_force_std`, actuation gains. Observation $\in\mathbb{R}^{12n}$.
   Lie–Heun Stratonovich stepper (§14).
2. `mini_tests/test_gt_pH_matches_arm_env.py` — tests 1–8 of §20.
3. `datasets/windy_arm_nlink_datagen.py` — mirror of the existing datagen with per-link
   $SO(3)$ observation noise.

**Phase 2 — model (after Phase 1 is signed off).**

4. `src/utils/JAX/lie_integrator.py` — add `lie_heun_sde_step_nlink` / `_rollout_nlink`.
5. `src/utils/JAX/loss_utils_jax.py`, `elbo_loss_jax.py` — per-link geodesic loss;
   Woodbury form of `pl_loss` (§15.3).
6. `src/utils/JAX/gp_model.py` — product-manifold Matérn features (§16).
7. `src/models/nDOF_SO3_Arm/ph_gp_sde/{network,train}.py` — the model with
   $M^{-1}\!:3n\times3n$, $D:3n\times3n$, $g:3n\times m$, $\Sigma:3n\times3$.

**Phase 3 — experiments.** Baselines (`ph_nn_ode`, `ph_nn_sde`, `neural_sde`, `ph_gp_ode`)
re-pointed at the new dataset; ablations on the structural priors of §17; the chaos-aware
evaluation protocol of §18.1.

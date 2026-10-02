# `quadrotor.py` — SE(3) Quadrotor Environment: Algorithm Explained

A step-by-step explanation of the algorithm in [`quadrotor.py`](quadrotor.py) (class `quadrotor_se3`, line 109).
The env simulates a quadrotor as a **stochastic port-Hamiltonian system on SE(3)**: rigid-body
dynamics with a constant control matrix $G$, friction (damping), deterministic + stochastic wind,
and geometrically-correct observation noise — the SE(3) counterpart of `envs/windy_pendulum_3d.py`.

---

## 1. The state — where the drone is and how it moves

- The drone's full state lives on the Lie group $SE(3) = \mathbb{R}^3 \rtimes SO(3)$ plus its velocities:

| Symbol | Meaning | Frame | Code |
|---|---|---|---|
| $x_w \in \mathbb{R}^3$ | position of center of mass | world | `self.x_w` |
| $R \in SO(3)$ | rotation matrix (body → world) | — | `self.R` |
| $v_b \in \mathbb{R}^3$ | linear velocity | **body** | `self.v_b` |
| $\omega_b \in \mathbb{R}^3$ | angular velocity | **body** | `self.omega` |

- **Why body frame for $v_b$?** Because then every input term (thrust, torques, friction) has
  **constant** coefficients — the ground-truth control matrix $G$ does not depend on the state $q$.
  This is the "left-trivialized" formulation of geometric mechanics.
- The observation is the flattened state (line 325, `_get_obs`):

$$\text{obs} = [\,\underbrace{x_w}_{3},\ \underbrace{\text{vec}(R)}_{9},\ \underbrace{v_b}_{3},\ \underbrace{\omega_b}_{3}\,] \in \mathbb{R}^{18}$$

---

## 2. The control input — 4 rotors, one constant matrix

- The action is $u \in \mathbb{R}^4$ = **squared rotor speeds** ($u_i = \text{rpm}_i^2$ conceptually).
  Squaring is what makes thrust **linear** in the input: propeller physics gives
  $f_i = k_f\,\text{rpm}_i^2$, so with $u_i = \text{rpm}_i^2$ we get $f_i = k_f u_i$.
- The whole body wrench (force + torque) is one matrix multiply (`_build_G`, lines 278–300):

$$\begin{bmatrix} F_b \\ \tau_b \end{bmatrix} = G\,u, \qquad
G = \begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
k_f & k_f & k_f & k_f \\
-a k_f & -a k_f & a k_f & a k_f \\
-a k_f & a k_f & a k_f & -a k_f \\
-k_m & k_m & -k_m & k_m
\end{bmatrix} \in \mathbb{R}^{6\times 4}, \qquad a = \frac{\text{arm}}{\sqrt{2}}$$

- Row by row, in simple words:
  - Rows 1–2 ($F_x, F_y$) are **zero**: all propellers push along the body z-axis only.
    The drone is **underactuated** — 4 inputs, 6 degrees of freedom.
  - Row 3 ($F_z$): total thrust = sum of the four rotor thrusts.
  - Rows 4–5 ($\tau_x, \tau_y$): roll/pitch torques from thrust *differences* between rotors
    on opposite sides (lever arm $a$).
  - Row 6 ($\tau_z$): yaw torque from propeller *reaction* (spin drag) torques, alternating
    sign because rotors alternate spin direction ($k_m$).
- Signs follow the **CF2X X-configuration** of gym-pybullet-drones
  (`BaseAviary.py:838-855`), with rotor body positions
  $r_0 = (+a,-a,0),\ r_1 = (-a,-a,0),\ r_2 = (-a,+a,0),\ r_3 = (+a,+a,0)$
  (`_rotor_positions`, line 302).
- $G$ never depends on the state — only on $k_f, k_m, \text{arm}$.

---

## 3. The dynamics — one Stratonovich SDE on SE(3)

The continuous-time model (`_compute_rates`, lines 347–403):

**Kinematics** (how position/orientation change, given velocities):

$$\dot{x}_w = R\,v_b \qquad\text{(line 372)}, \qquad\qquad \dot{R} = R\,[\omega_b]_\times$$

**Translational dynamics** (Newton in the rotating body frame, line 376):

$$m\,\dot{v}_b = \underbrace{m\,v_b \times \omega_b}_{\text{frame rotation}}
\;+\; \underbrace{R^\top\!\left(-m g e_3 + F_{\text{wind}}(t)\right)}_{\text{world forces} \to \text{body}}
\;+\; \underbrace{e_3\,k_f \textstyle\sum_i u_i}_{\text{thrust } = (Gu)_{1:3}}
\;-\; \underbrace{d_{\text{lin}}\,v_b}_{\text{linear friction}}
\;+\; \sigma_f\,R^\top \!\circ dW_f$$

**Rotational dynamics** (Euler's rigid-body equation, line 384):

$$J\,\dot{\omega}_b = \underbrace{(J\omega_b) \times \omega_b}_{\text{gyroscopic}}
\;+\; \underbrace{\tau(u) = (Gu)_{4:6}}_{\text{rotor torques}}
\;-\; \underbrace{d_{\text{ang}}\,\omega_b}_{\text{rotational friction torque}}
\;+\; \sigma_\tau \circ dW_\tau$$

Point-by-point:

- **$m v_b \times \omega_b$**: appears because $v_b$ is expressed in a rotating frame
  (it is *not* a physical force — it's the chain rule of $v_w = R v_b$).
- **Gravity** $-mge_3$ is a world-frame force, so it gets rotated into the body frame by $R^\top$.
- **Deterministic wind** $F_{\text{wind}}(t) = w(t)\,\hat{d}$ acts at the center of mass in the
  world frame (`update_wind`, line 312). Four types, identical to the pendulum:
  - `sine`: $w(t) = \sigma_w \sin(2\pi\, 0.5\, t)$
  - `square`: $w(t) = \sigma_w\,\text{sign}(\sin(2\pi\, 0.5\, t))$
  - `random`: $w \sim \mathcal{N}(0, \sigma_w^2)$, one draw per env step
  - `constant`: $w(t) = \sigma_w$
  Because it acts at the COM, it produces **no deterministic torque**.
- **Friction** = damping proportional to velocity: force $-d_{\text{lin}} v_b$ and
  torque $-d_{\text{ang}} \omega_b$. This is the dissipation $D$ of the port-Hamiltonian form.
- **Stochastic wind, force channel** (line 393): a world-frame random force
  $\sigma_f\,dW_f$ rotated to body, so the increment is
  $dV = \frac{\sigma_f}{m} R^\top dW_f$ — **state-dependent (multiplicative) noise**,
  because $R$ appears.
- **Stochastic wind, torque channel** (line 398): body-frame random torque, increment
  $d\Omega = J^{-1}\sigma_\tau\,dW_\tau$ — **additive noise** (no state in it).
- $\circ$ marks **Stratonovich** interpretation — the convention under which the ordinary chain
  rule holds, which is what makes integration on a manifold (via the exponential map) consistent.

---

## 4. Randomizable coefficients — mean + std scheme

- Four physical coefficients can be fixed or randomly varying
  (`_draw_coeff`, line 259; `_resample_coeffs`, line 265):

$$\text{value} =
\begin{cases}
\text{coeff} & \text{if } \text{std} = 0 \quad (\text{fixed})\\[4pt]
\max\!\big(0,\ \mathcal{N}(\text{coeff},\ \text{std}^2)\big) & \text{if } \text{std} > 0 \quad (\text{varying})
\end{cases}$$

  applied to: `kf` (thrust gain), `km` (yaw-torque gain), `linear_damping`, `angular_damping`.
  The $\max(0,\cdot)$ clip keeps them physical (no negative thrust/friction).
- **When are they re-drawn?** Controlled by `resample_coeffs_every_step`:
  - `True` (default): re-drawn once per `step()` call (line 482), held constant across the
    10 substeps inside that step → piecewise-constant parameter noise.
  - `False`: drawn once at `reset()` (line 573) and held for the **whole trajectory**
    → domain randomization ("each episode is a slightly different drone").
- Every draw of `kf`/`km` **rebuilds $G$** (line 276), and the current values are exposed in
  `info = {"wind", "kf", "km", "d_lin", "d_ang"}` (line 522) — the ground truth a learned
  model can be compared against.
- `arm` is **always fixed** (argument, default 1.0). Mass $m$ and inertia $J$ are plain
  constructor arguments.

---

## 5. The integrator — Stratonovich Heun on the Lie group

The SDE is integrated with a **geometric Heun (predictor–corrector) scheme**
(`_lie_heun_step`, lines 406–469), 10 substeps of size $h = dt/10$ per env step (line 486).
The key idea: rotations are updated **through the exponential map**, so $R$ stays exactly on
$SO(3)$ — no projection needed in the loop.

For one substep, with the same Wiener increments $dW_f, dW_\tau \sim \mathcal{N}(0, h I_3)$
used in **both** stages (sampled at lines 492, 496):

**Stage 1 — evaluate at the current state:**

$$(\dot{x}_1, \dot{v}_1, \dot{\omega}_1, dV_1, d\Omega_1) = f(x_w, R, v_b, \omega_b), \qquad \phi_1 = \omega_b\,h$$

**Stage 2 — predictor (Euler step on the manifold):**

$$R_p = R\,\exp([\phi_1]_\times), \quad x_p = x_w + \dot{x}_1 h, \quad v_p = v_b + \dot{v}_1 h + dV_1, \quad \omega_p = \omega_b + \dot{\omega}_1 h + d\Omega_1$$

**Stage 3 — re-evaluate at the predicted state (same $dW$):**

$$(\dot{x}_2, \dot{v}_2, \dot{\omega}_2, dV_2, d\Omega_2) = f(x_p, R_p, v_p, \omega_p), \qquad \phi_2 = \omega_p\,h$$

**Stage 4 — corrector (average in the Lie algebra, exponentiate once, line 460):**

$$R^{+} = R\,\exp\!\Big(\big[\tfrac{\phi_1 + \phi_2}{2}\big]_\times\Big), \qquad
x^{+} = x_w + \tfrac{\dot{x}_1 + \dot{x}_2}{2}h$$

$$v^{+} = v_b + \tfrac{\dot{v}_1 + \dot{v}_2}{2}h + \tfrac{dV_1 + dV_2}{2}, \qquad
\omega^{+} = \omega_b + \tfrac{\dot{\omega}_1 + \dot{\omega}_2}{2}h + \tfrac{d\Omega_1 + d\Omega_2}{2}$$

Why each piece matters:

- **$\exp([\phi]_\times)$ (Rodrigues' formula, line 27)** guarantees $R^{+}\in SO(3)$ to machine
  precision — measured drift is $\sim 10^{-14}$ after 500 noisy steps.
- **Averaging $\phi$ before one exponential** (instead of two exponentials) is the correct
  Heun scheme on a Lie group: average in the algebra $\mathfrak{so}(3) \cong \mathbb{R}^3$, map once.
- **Re-evaluating $dV$ in stage 2** is what makes the scheme converge to the **Stratonovich**
  solution — the force-channel noise $\frac{\sigma_f}{m}R^\top dW_f$ depends on $R$, and
  Heun's averaging of a state-dependent diffusion is precisely the Stratonovich correction.
- A safety net re-orthogonalizes $R$ only if floating-point drift ever exceeds $10^{-8}$
  (line 509) — in practice it never triggers.

---

## 6. One `step(u)` call — the full flow (lines 471–530)

1. Read $u \in \mathbb{R}^4$ — **no clipping** (mirrors the pendulum; `max_u` only defines the
   `action_space` API bounds).
2. Advance time $t \mathrel{+}= dt$ and draw the deterministic wind $w(t)$ (line 479).
3. If `resample_coeffs_every_step`: re-draw $k_f, k_m, d_{\text{lin}}, d_{\text{ang}}$ and
   rebuild $G$ (line 482). RNG draw order is fixed (wind → coeffs → $dW$s) for reproducibility.
4. Loop 10 substeps: sample $dW_f, dW_\tau \sim \mathcal{N}(0, h I_3)$, run the Lie–Heun substep.
5. Safety-net orthogonality check on $R$.
6. Reward = negative hover cost (placeholder, line 513):

$$\text{cost} = \|x_w\|^2 + 0.1\big(\|v_b\|^2 + \|\omega_b\|^2\big) + 0.001\,\|u - u_{\text{hover}}\mathbf{1}_4\|^2,
\qquad u_{\text{hover}} = \frac{mg}{4 k_f}$$

7. Return `(obs, reward, False, False, info)` with the ground-truth coefficients in `info`.

**Hover check** (used in tests/demo): with $u = u_{\text{hover}}\mathbf{1}_4$, level attitude and
zero velocity, thrust exactly cancels gravity: $k_f \cdot 4 \cdot \frac{mg}{4k_f} = mg$, and all
torque rows cancel by symmetry — the state is an exact fixed point of the integrator.

---

## 7. Observation noise — geometric, not naive (lines 325–344)

- With `obs_noise_std` $= \sigma_o > 0$, `_get_obs()` returns a corrupted observation while the
  internal state stays clean (`get_state()`, line 253, always returns the true state):

$$R_{\text{obs}} = R\,\exp([\epsilon]_\times), \quad \epsilon \sim \mathcal{N}(0, \sigma_o^2 I_3)
\qquad\text{(noise on the manifold — } R_{\text{obs}} \in SO(3)\text{ still!)}$$

$$x_{\text{obs}} = x_w + \eta_x,\quad v_{\text{obs}} = v_b + \eta_v,\quad \omega_{\text{obs}} = \omega_b + \eta_\omega, \qquad \eta \sim \mathcal{N}(0, \sigma_o^2 I_3)$$

- Same recipe as `add_proper_noise_3d` in `datasets/windy_pendulum_3d_datagen.py:79-133`.
- A **separate RNG** (`self._obs_rng`, seeded `seed+1`) generates observation noise, so turning
  it on/off never changes the dynamics realization.

---

## 8. `reset(seed, options)` — initial conditions (lines 534–577)

- Options (all optional): `x_init`, `R_init` (projected onto $SO(3)$), `v_init`, `omega_init`.
- Defaults: $x_w, v_b, \omega_b \sim U(-1, 1)^3$ and $R$ uniform on $SO(3)$ (Shoemake
  quaternion method, line 89).
- Re-seeds both RNGs, then draws the trajectory's coefficients (`_resample_coeffs`, line 573).

---

## 9. Rendering — gym-pybullet-drones look, our dynamics

- Two backends via the `render_backend` argument (`render()` dispatcher):
  - **`"pybullet"` (default)**: renders the real Crazyflie CF2X model (`cf2x.urdf` + mesh from the
    local gym-pybullet-drones clone) over PyBullet's checkerboard ground plane, with the same
    camera recipe as `BaseAviary` recordings (`computeViewMatrixFromYawPitchRoll`, yaw $-30°$,
    pitch $-30°$, fov $60°$, 640×480, TinyRenderer + shadows). The camera **tracks the drone**
    at distance $6\cdot\text{arm}$, and the model is scaled by $\text{arm}/0.0397$ so its size
    matches the env's arm length.
  - **`"matplotlib"`**: the original zero-dependency schematic (X-frame + rotor circles).
- **Important:** PyBullet is a *renderer only*. Each frame, the env's clean state $(x_w, R)$ is
  pushed in via `resetBasePositionAndOrientation` ($R$ → quaternion by Shepperd's method,
  `_rotmat_to_quat`); `stepSimulation` is **never called**, so the Lie–Heun integrator remains
  the single source of truth for the dynamics.
- The `__main__` demo includes a demo-only geometric hover controller (Lee-style PD on SE(3)):
  a bare quadrotor is **open-loop unstable** — random wind torques tilt it, tilted thrust stops
  cancelling gravity, and it crashes within seconds (physically correct). The controller computes
  $F_{\text{des}} = m(K_p e_x - K_d v_w + g e_3)$, a desired attitude with $b_3 \parallel F_{\text{des}}$,
  torque $\tau = -K_R e_R - K_\omega \omega$ with $e_R = \frac{1}{2}(R_d^\top R - R^\top R_d)^\vee$,
  and inverts the $[F_z; \tau]$ rows of $G$: $u = A^{-1}[T; \tau]$.

## 10. Port-Hamiltonian reading (for the paper)

The env is exactly the paper's Eq. (6) lifted from $SO(3)$ to $SE(3)$. With momenta
$\mathbf{p} = m v_b$, $\boldsymbol{\Pi} = J\omega_b$ and Hamiltonian

$$H = \tfrac{1}{2m}\|\mathbf{p}\|^2 + \tfrac{1}{2}\boldsymbol{\Pi}^\top J^{-1}\boldsymbol{\Pi} + m g\,z,$$

the pieces map as:

| pH component | Ground truth in this env |
|---|---|
| Mass matrix $\mathcal{M}^{-1}$ | $\text{blkdiag}(\frac{1}{m}I_3,\ J^{-1})$ — constant |
| Potential $V(q)$ | $m g\, x_{w,3}$ |
| Dissipation $D$ | $\text{blkdiag}(d_{\text{lin}} I_3,\ d_{\text{ang}} I_3)$ |
| Port matrix $G(q)$ | the constant $6\times4$ mixer of §2 (underactuated) |
| Diffusion $\xi(q)$ | $\big(\sigma_f R^\top,\ \sigma_\tau I_3\big)$ — force channel state-dependent |
| External port | deterministic wind $w(t)\hat d$ through the force channel |

---

## 11. How it was verified

- **Manifold**: $|\det R - 1| < 10^{-14}$, $\|R^\top R - I\| < 10^{-14}$ over 500 noisy steps.
- **Hover**: exact equilibrium (drift $= 0.0$) at $u = \frac{mg}{4k_f}\mathbf{1}_4$.
- **Energy**: $E = \frac{1}{2}m\|v_b\|^2 + \frac{1}{2}\omega_b^\top J \omega_b + mgz$ strictly
  non-increasing with $u=0$, damping on, wind off.
- **Yaw sign**: speeding up rotors 1 and 3 yields $\omega_z > 0$, matching the $\tau_z$ row.
- **Reproducibility**: same seed → bitwise-identical trajectories; obs noise never leaks into dynamics.
- **Cross-validation vs gym-pybullet-drones** (`CtrlAviary`, `Physics.DYN`, real Crazyflie CF2X
  parameters, identical RPM sequence, substep $h = 1/240$):
  - the gap between the two sims halves exactly when $h$ halves (ratio 1.99–2.00) — both solve
    the **same continuous dynamics**; the difference is purely their first-order Euler error;
  - our env lands ~17× closer to a fine-resolution ($h = 1/3840$) reference than their own
    coarse run (1.2 mm vs 21 mm position error over a 2 s flight);
  - our Heun self-converges at ratio 4.03 ≈ 4 (second order, as designed);
  - Richardson check: the residual fine-grid gap (1.42e-3 m / 7.93e-4 rad) equals their
    predicted leftover Euler error to three digits — every bit of discrepancy is accounted for.

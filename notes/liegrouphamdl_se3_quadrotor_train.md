# LieGroupHamDL — SE(3) Quadrotor Training, Explained

**What this is.** A step-by-step explanation of how the reference implementation
trains its SE(3) port-Hamiltonian Neural ODE on a quadrotor: what it learns, how
the data is produced, the loss, every argument, and the design choices in the
reference implementation.

**Paper.** *Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For
Dynamics Learning and Control*, Thai Duong & Nikolay Atanasov, RSS 2021.

**Code.** `LieGroupHamDL/` —
`se3hamneuralode/SE3HamNODE.py` (model), `se3hamneuralode/utils.py` (loss),
`examples/quadrotor/data_collection.py` (data),
`examples/quadrotor/train_quadrotor_SE3.py` (training),
`examples/quadrotor/analyze_quadrotor_SE3.py` (evaluation).

> **Note on scope.** Everything below is read from their source, not from the
> paper. Where I could not verify a claim in code, I say so.

---

## Table of contents

1. [The idea in one page](#1-the-idea-in-one-page)
2. [What the model learns](#2-what-the-model-learns)
3. [The dynamics the network computes](#3-the-dynamics-the-network-computes)
4. [The loss function](#4-the-loss-function)
5. [The data — PID flight in PyBullet](#5-the-data--pid-flight-in-pybullet)
6. [The control input is a WRENCH](#6-the-control-input-is-a-wrench)
7. [The training loop](#7-the-training-loop)
8. [All command-line arguments](#8-all-command-line-arguments)
9. [How they evaluate](#9-how-they-evaluate)
10. [Physical constants](#10-physical-constants)
11. [Design choices vs ours](#11-design-choices-vs-ours)
12. [Issues their setup has](#12-issues-their-setup-has)

---

## 1. The idea in one page

- Fly a real quadrotor model (Crazyflie CF2P) in PyBullet using a PID controller.
- Record short state/control snippets.
- Train a neural network that is **forced to have port-Hamiltonian structure**
  on SE(3).
- Four unknown blocks sit inside: mass, potential energy, damping, input matrix.
- Training only compares **trajectories**.
- Afterwards, plot the learned blocks and check they look right.

Same premise as ours. The differences are all in *how the problem is set up* —
what $u$ means, how it is excited, and how long the training windows are.

---

## 2. What the model learns

`SE3HamNODE.__init__` builds **six** subnetworks (we use four):

| subnet | input | output | true value |
|---|---|---|---|
| `M_net1` | $x$ (3) | $3\times3$ PSD | $\tfrac{1}{m}I_3$ |
| `M_net2` | $R$ (9) | $3\times3$ PSD | $J^{-1}$ |
| `V_net` | pose (12) | scalar | $V = mgz$ |
| `Dv_net` | $x$ (3) | $3\times3$ PSD | translational damping |
| `Dw_net` | $R$ (9) | $3\times3$ PSD | rotational damping |
| `g_net` | pose (12) | $6\times4$ | $[0_{2\times4};\,I_4]$ |

Key structural choices:

- **$\mathcal M$ is split into two networks**, one per channel. This *hard-codes*
  that the mass matrix is block diagonal — no $v$–$\omega$ coupling — and that
  the translational block depends only on position, the rotational only on
  attitude. We learn a single $6\times6$ and must discover that.
- **$D$ is split the same way**, `Dv_net(x)` + `Dw_net(R)`. Note their $D$ nets
  do **not** see momentum, so a friction law depending on $|\omega|$ (as the
  SO(3) pendulum env has) is structurally unrepresentable.
- `M_net1` / `M_net2` output the **inverse** $\mathcal M^{-1}$, same convention
  as ours.
- All PSD nets are $LL^\top + \epsilon I$ with $\epsilon = 0.01$ for the mass
  nets and $0.0$ for the damping nets.
- Width is **400** for every subnet; `init_gain = 0.001`.

State through the solver is 22 numbers, identical layout to ours:

$$x = [\ \underbrace{x_w(3)\ \ \mathrm{vec}R(9)}_{\text{pose } q}\ |\ \underbrace{v_b(3)\ \ \omega_b(3)}_{\xi}\ |\ \underbrace{u(4)}_{\text{wrench}}\ ]$$

---

## 3. The dynamics the network computes

$$H(q,p) \;=\; \tfrac12\,\mathbf p^\top M_1^{-1}\mathbf p \;+\; \tfrac12\,\Pi^\top M_2^{-1}\Pi \;+\; V(q)$$

$$\dot x_w = R\,v_b, \qquad \dot R = R\,[\omega_b]_\times$$

$$\dot{\mathbf p} = \mathbf p\times\omega_b \;-\; R^\top\tfrac{\partial H}{\partial x} \;-\; D_v\,v_b \;+\; (gu)_{1:3}$$

$$\dot{\Pi} = \Pi\times\omega_b \;+\; \mathbf p\times v_b \;+\; \mathcal T_R(H) \;-\; D_\omega\,\omega_b \;+\; (gu)_{4:6}$$

**These are exactly the equations we use.** I checked them term by term against
our `network.py`: the gyroscopic terms, gravity as $-R^\top\partial H/\partial x$,
the SE(3) cross-term $\mathbf p\times v_b$, and the trivialised rotational
gradient $\mathcal T_R(H) = \sum_k r_k\times\partial H/\partial r_k$ all match.

Two implementation differences (ours is faster, theirs is the original):

| | theirs | ours |
|---|---|---|
| $d\mathcal M^{-1}/dt$ | **18** `autograd.grad` calls in a double loop | **1** `torch.func.jvp` call |
| momentum from velocity | `torch.inverse(M_inv)` | `torch.linalg.solve` |

---

## 4. The loss function

`pose_L2_geodesic_loss` in `se3hamneuralode/utils.py`:

$$\mathcal L \;=\; \underbrace{\mathrm{MSE}(x_w)}_{\text{position}} + \underbrace{\mathrm{MSE}(v_b)}_{\text{lin. vel}} + \underbrace{\mathrm{MSE}(\omega_b)}_{\text{ang. vel}} + \underbrace{\mathbb E[\theta^2]}_{\text{attitude}}$$

$$\theta = \arccos\!\left(\frac{\operatorname{tr}(\hat R^\top R) - 1}{2}\right)$$

- **Structurally the same as ours** — four equally-weighted channels, geodesic
  on $SO(3)$ rather than L2 on matrix entries.
- Predicted $R$ is projected back onto $SO(3)$ first
  (`compute_rotation_matrix_from_unnormalized_rotmat`).
- The $u$ channel is split out and discarded.
- **No `arccos` clamp.** Ours adds one, because the gradient of $\arccos$ is
  infinite at $\cos = \pm1$ and can produce NaNs in fp32.
- **No physics terms.** Like ours, the loss never sees the true
  $\mathcal M, V, D, g$.

---

## 5. The data — PID flight in PyBullet

From `examples/quadrotor/data_collection.py`:

- Simulator: **PyBullet** `CtrlAviary`, `Physics.PYB`, Crazyflie **CF2P**
  (plus-configuration: rotors on the body axes).
- Simulation rate **240 Hz**; control rate **48 Hz**, so
  `AGGR_PHY_STEPS = 240/48 = 5` physics substeps per control step.
- Controller: **`DSLPIDControl`** with all gains halved, regulating to setpoints.
- **18 runs**, one per goal position on a $3\times3\times2$ grid
  ($x,y \in \{-1,0,1\}$, $z \in \{0.375, 0.75\}$), each 2.5 s long.
- Initial yaw $(i\bmod3)\cdot30°$; target yaw $(i\bmod3)\cdot15°$.
- Train/test split **0.8** on the control-step axis.

**The clever part — how windows are formed.** The recorded array is

$$\texttt{data\_set}[\ \underbrace{18}_{\text{runs}},\ \underbrace{5}_{\text{substeps}},\ \underbrace{120}_{\text{control steps}},\ 22\ ]$$

and `arrange_data` slices along the **substep** axis. So the ODE "time" axis is
the 5 physics substeps *inside one control step*, and the 120 control steps
become part of the batch. That gives:

- window length $= 4\times\tfrac{1}{240} \approx \mathbf{16.7\ ms}$,
- $u$ **exactly constant** within every window, by construction,
- $18 \times 120 = \mathbf{1728}$ training windows.

Because $u$ is constant across a window, they can integrate the whole window
with a **single `odeint` call**. Ours varies $u$ per step, so we roll out
segment-by-segment. (Our `--u_hold` flag reproduces their arrangement.)

---

## 6. The control input is a WRENCH

This is the single most consequential difference from our setup.

They convert rotor speeds to a wrench **before saving the data**:

```python
r = env.KM / env.KF
conversion_mat = [[1, 1, 1, 1],
                  [0, L, 0, -L],
                  [-L, 0, L, 0],
                  [-r, r, -r, r]]
forces = rpm**2 * env.KF
thrust_torques = conversion_mat @ forces      #  <-- this is u
```

So

$$u = [\,T,\ \tau_x,\ \tau_y,\ \tau_z\,] \qquad\text{not}\qquad u_i = \text{rpm}_i^2$$

**Consequence: their ground-truth $g$ is a selection matrix.**

$$g_{\text{true}} = \begin{bmatrix}0_{2\times4}\\ I_4\end{bmatrix}
= \begin{bmatrix}0&0&0&0\\ 0&0&0&0\\ 1&0&0&0\\ 0&1&0&0\\ 0&0&1&0\\ 0&0&0&1\end{bmatrix}$$

There is **no physics left in $g$** — no $k_f$, no $k_m$, no arm length. Those
constants were read from the URDF and applied by the data pipeline. `g_net` only
has to find a 0/1 pattern.

The two formulations are related by an invertible $4\times4$ map. With
$A = G_{3:6,:}$ (the mixer's non-zero rows):

$$u_{\text{wrench}} = A\,u_{\text{rotor}}, \qquad G_{\text{rotor}}A^{-1} = g_{\text{true}}$$

so it is purely a **change of input coordinates**. The physics does not vanish —
it moves out of the model and into the preprocessing.

---

## 7. The training loop

Per gradient step (`train_quadrotor_SE3.py`):

1. `odeint(model, train_x_cat[0], t_eval, method='rk4')` — **one call** over the
   whole window, from the first sample.
2. `pose_L2_geodesic_loss` against samples $1{:}T$.
3. The same on the test set (evaluated **every step**, not periodically).
4. `backward()`, `optim.step()` — but **skipped at step 0**.

- Optimizer **Adam**, `lr = 5e-4`, **`weight_decay = 0.0`**.
- **No gradient clipping.**
- 5000 steps, full batch (no minibatching).

**Pretraining.** `pretrain()` fits `M_net1` and `M_net2` to the **identity**, not
to the true values: 64 000 grid positions and 250 000 uniform quaternions,
looping `while loss > 1e-6`. In the released trainer this is skipped
(`pretrain=False`) and a cached `...-pre-init.tar` is loaded instead.

> Note the true CF2P values are $1/m \approx 37$ and $J^{-1} \approx 4\times10^4$,
> so pretraining to $I$ is an **arbitrary starting gauge**, not an anchor at the
> truth. Ours pretrains to the true $\mathrm{blkdiag}(I/m, J^{-1})$.

---

## 8. All command-line arguments

`train_quadrotor_SE3.py` exposes **12** arguments (ours has 44):

| argument | type | default | meaning |
|---|---|---|---|
| `--learn_rate` | float | `5e-4` | Adam learning rate |
| `--total_steps` | int | `5000` | gradient steps |
| `--print_every` | int | `100` | print / checkpoint cadence |
| `--num_points` | int | `5` | samples per training window |
| `--solver` | str | `rk4` | `torchdiffeq` solver |
| `--nonlinearity` | str | `tanh` | subnet activation |
| `--float` | int | `32` | fp32 or fp64 |
| `--seed` | int | `0` | random seed |
| `--gpu` | int | `0` | CUDA device |
| `--name` | str | `quadrotor` | run label |
| `--save_dir` | str | script dir | output directory |
| `--verbose` | flag | off | extra printing |

Everything else — architecture, priors, dataset composition, physical
constants — is **hard-coded**. Width 400, `init_gain` 0.001, PSD $\epsilon$ 0.01,
the 18 setpoints, the PID gains, the 2.5 s duration: all literals in the source.

`data_collection.py` has its own four: `--duration_sec` (2.5), `--num_resets`
(1), `--simulation_freq_hz` (240), `--control_freq_hz` (48).

---

## 9. How they evaluate

`analyze_quadrotor_SE3.py` produces plots only. I grepped it: **there is no
ground-truth comparison anywhere** — no true mass, no $9.81$, no reference curve.

What they plot:

| figure | what it checks | the property |
|---|---|---|
| $V(q)$ **against $z$** | is $V$ linear in height? | *shape* |
| $M_1^{-1}$ entries **over time**, off-diagonals overlaid | constant? diagonal? | *shape* |
| $M_2^{-1}$ entries over time | same | *shape* |
| $B_v[0,0], B_v[1,0], B_v[2,0]$ | are the force rows zero? | *structure* |
| $\lVert RR^\top - I\rVert$ | does the rollout stay on $SO(3)$? | constraint |

This is a deliberate and defensible choice, and it is forced by §12.1: absolute
values are not identifiable, so only shape can honestly be reported.

The learned model is then used for **energy-based / IDA-PBC control**
(`controller_energy_based.py`, `control_quadrotor_SE3.py`), which queries
$M_1, M_2, V, g$ from the trained network to synthesise a controller. There are
also **unstructured baselines** (`UnstructuredSE3HamNODE`, `UnstructuredSE3NODE`)
and a ground-truth model (`SE3HamNODEGT`) for comparison.

---

## 10. Physical constants

From `cf2p.urdf` — a real Crazyflie 2.0:

| quantity | value |
|---|---|
| mass $m$ | $0.027$ kg |
| $J$ | $\mathrm{diag}(2.3951, 2.3951, 3.2347)\times10^{-5}$ kg·m² |
| arm | $0.0397$ m |
| $k_f$ | $3.16\times10^{-10}$ |
| $k_m$ | $7.94\times10^{-12}$ |
| thrust/weight | $2.25$ |

Ours uses round numbers ($m=1$, $J=(0.5,0.5,1)$, arm $=1$, $k_f=1$, $k_m=0.1$),
so our $k_m/k_f = 0.1$ against their $\approx 0.025$.

---

## 11. Design choices vs ours

| | LieGroupHamDL | Ours |
|---|---|---|
| $u$ | **wrench** $[T,\tau]$ | **rotor speeds** $\text{rpm}^2$ |
| true $g$ | selection matrix | real CF2X mixer |
| excitation | **PID tracking 18 setpoints** (closed loop) | **open-loop random $u$** near hover |
| attitude coverage | near level, $R\approx I$ | **uniform on $SO(3)$** |
| $dt$ / window | $1/240$ s, **16.7 ms** | $0.05$ s, 200 ms |
| $u$ in window | constant by construction | varies (unless `--u_hold`) |
| rollout | one `odeint` | segment-wise |
| windows | 1728 | 768 |
| $\mathcal M$ | two nets, $M_1(x)$ + $M_2(R)$ | one $6\times6$ (or `--split_M`) |
| $D$ | two nets, no momentum input | one $6\times6$ on $(q,p)$ |
| width / gain / $\epsilon$ | 400 / 0.001 / 0.01 | 64 / 0.5 / 1.0 (configurable) |
| optimizer | Adam 5e-4, wd 0, no clip | Adam 1e-3, wd 1e-4, clip 10 |
| $M$ pretraining | to **identity** | to the **true** $\mathcal M^{-1}$ |
| simulator | PyBullet Euler, quaternion renormalised | Lie–Heun, $SO(3)$ exact by construction |
| stochastic option | none | wind force/torque, obs noise |
| GT comparison | **none** — plots only | numeric MSE, raw + gauge-fixed |

---

## 12. Issues their setup has

### 12.1 The scale gauge — present, unaddressed

The momentum $p$ is latent, so

$$(\mathcal M,\ V,\ D,\ g)\;\longrightarrow\;(\lambda\mathcal M,\ \lambda V,\ \lambda D,\ \lambda g)$$

leaves every observable unchanged. I verified this on **their** `forward`: with
`dv = M_inv1 @ dpv + dM_inv_dt1 @ pv`, the $\lambda$ cancels exactly.

Worse, their `M_net1`/`M_net2` split makes the translation and rotation gauges
**two literally separate networks**, so $\lambda_v$ and $\lambda_w$ are
independent — the coupling term $\mathbf p\times v_b$ that might link them
vanishes identically for block-diagonal $\mathcal M$.

They do not fix it. They **report only shape** (§9), which is the honest
response. Because their $g_{\text{true}}$ has entries of exactly 1, the constants
in their $g$ plots *are* $\lambda_v$ and $\lambda_w$ read off directly.

### 12.2 Gravity can leak into $g$

Writing $u = \bar u + \tilde u$, the term $g_{1:3}(q)\bar u$ has the same form as
gravity $-mgR^\top e_3$, so a state-dependent $g$ can absorb it while $V$ shrinks.
Both conditions hold for them: `g_net` takes the full pose, and $T \approx mg$ at
hover.

**Their case is arguably worse than ours**: because the PID keeps $R\approx I$,
body-frame gravity $-mgR^\top e_3 \approx -mge_3$ is nearly *constant* and points
along body $-z$ — collinear with thrust. Gravity and thrust are then confounded
**even with a constant $g$**. Our full-$SO(3)$ coverage is what makes them
separable at all.

They plot $B_v[0,0], B_v[1,0], B_v[2,0]$ — exactly this leak's signature — so
they watch for it without naming the mechanism. We prevent it structurally with
`--const_G`.

### 12.3 Closed-loop data is weaker for identification

Their $u$ is a **function of the state** (PID feedback), which correlates input
and state and can let the controller mask the very dynamics being identified.
Open-loop excitation (ours) is the textbook-preferred condition.

### 12.4 Short windows carry little information

Over 16.7 ms the state barely changes, so each window weakly constrains the
physics. **This costs them nothing** because their $g$ has nothing to identify —
but we measured that the same window length applied to *our* setup destroys
physics recovery (shape scores fell to the level of an untrained model) while
giving the best trajectory fit of any run.

### 12.5 Minor

- **No `arccos` clamp** in the geodesic loss — a NaN risk in fp32.
- **Test loss computed every step**, which roughly doubles the cost.
- **PyBullet does not preserve $SO(3)$**; quaternions are renormalised. Our
  Lie–Heun integrator keeps $R\in SO(3)$ to $<10^{-13}$ by construction.
- **Almost everything is hard-coded** — reproducing a variant means editing
  source, not passing flags.

---

## 13. Summary

- The **physics equations and the loss are essentially identical** to ours.
- The real differences are in the **problem setup**: a wrench input instead of
  rotor speeds, closed-loop PID data instead of open-loop random excitation, and
  16.7 ms windows instead of 200 ms.
- Those choices make their learning problem **easier**: $g$ is trivial, the data
  is stable, and short windows suffice.
- They carry the **same two identifiability problems** we do (scale gauge,
  gravity leak) and address them by reporting shape rather than absolute values.
- Our additions relative to them: measured gauges, an identifiability split of
  the ratios, structural priors as flags, an exact-$SO(3)$ simulator, stochastic
  (SDE) support, and a verified pH-structure test against the simulator.

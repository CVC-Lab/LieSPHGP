# LieGroupHamDL SE3HamNODE, Reproduced Exactly, on Our Environment's Data

**What this is.** A faithful reproduction of the reference SE(3) model — same
neural networks, same equations, same loss, same optimiser — with **only the
data source replaced** by our simulator. It follows the same format as
[`liegrouphamdl_se3_quadrotor_train.md`](liegrouphamdl_se3_quadrotor_train.md).

**Why it exists.** The removed full-matrix model and the reference differed in *many* ways at
once — input representation, excitation, window length, architecture,
hyperparameters. That makes it impossible to say which difference causes which
result. This build removes every difference except the data, so the comparison
becomes a controlled experiment.

**Code.** [`src/models/SE3_Quadrotor/ph-node/`](../src/models/SE3_Quadrotor/ph-node/)
— `SE3HamNODE.py`, `lgh_utils.py`, `train.py`.

---

## Table of contents

1. [The idea in one page](#1-the-idea-in-one-page)
2. [What is exactly identical](#2-what-is-exactly-identical)
3. [What the model learns](#3-what-the-model-learns)
4. [The dynamics](#4-the-dynamics)
5. [The loss](#5-the-loss)
6. [The data — ours, in their coordinates](#6-the-data--ours-in-their-coordinates)
7. [The rotor-to-wrench conversion](#7-the-rotor-to-wrench-conversion)
8. [The training loop](#8-the-training-loop)
9. [All command-line arguments](#9-all-command-line-arguments)
10. [The deviations — an honest list](#10-the-deviations--an-honest-list)
11. [The NaN bug we hit](#11-the-nan-bug-we-hit)
12. [Two things that had to be worked around](#12-two-things-that-had-to-be-worked-around)
13. [How to run](#13-how-to-run)
14. [Three-way comparison](#14-three-way-comparison)

---

## 1. The idea in one page

- Preserve the reference architecture and equations, with verified
  equation-equivalent implementation speedups.
- Take their loss **verbatim**.
- Take their training loop, optimiser settings and pretraining workflow.
- Feed it data from **our** simulator instead of their PyBullet PID flights.
- Everything that differs is then *only* the data.

If the physics comes out well, our data is fine and the earlier problems were
ours. If it comes out the same as before, the problems are shared.

---

## 2. What is exactly identical

| item | status |
|---|---|
| `SE3HamNODE` architecture/equations | **equivalent**; solve, batched-cross, and JVP speedups are verified numerically |
| `MLP` / `PSD` / `MatrixNet` | already byte-identical in both repos (Symplectic ODE-Net) |
| `pose_L2_geodesic_loss` | copied, one line changed (§11) |
| width / init gain / PSD $\epsilon$ | 400 / 0.001 / 0.01 (mass), 0.0 (damping) |
| optimiser | Adam, `lr = 5e-4`, `weight_decay = 0.0` |
| gradient clipping | **none** |
| rollout | **one** `odeint` per window |
| backward at step 0 | **skipped** |
| test loss | evaluated **every** step |
| steps / solver / `num_points` | 5000 / `rk4` / 5 |
| $M$ pretraining | to the **identity**, then cached as `-pre-init.tar` |
| training windows | **1728** — the same count as theirs |
| window length | **16.7 ms**, $u$ constant within it |

Only the import block of `SE3HamNODE.py` was edited, to point at this repo's
identical copies of the building blocks.

---

## 3. What the model learns

**Six** subnetworks (the removed full-matrix model used four):

| subnet | input | output | true value here |
|---|---|---|---|
| `M_net1` | $x_w$ (3) | $3\times3$ PSD | $\tfrac{1}{m}I_3 = I_3$ |
| `M_net2` | $R$ (9) | $3\times3$ PSD | $J^{-1} = \mathrm{diag}(2,2,1)$ |
| `V_net` | pose (12) | scalar | $V = mgz$ |
| `Dv_net` | $x_w$ (3) | $3\times3$ PSD | $d_{\text{lin}}I_3 = 0.5\,I_3$ |
| `Dw_net` | $R$ (9) | $3\times3$ PSD | $d_{\text{ang}}I_3 = 0.5\,I_3$ |
| `g_net` | pose (12) | $6\times4$ | $[0_{2\times4};\ I_4]$ |

- **$\mathcal M$ split into two networks** hard-codes that the mass matrix is
  block diagonal (no $v$–$\omega$ coupling) and that each block sees only its
  own coordinates. The removed full-matrix model had to *discover* that.
- **$D$ split the same way.** Note `Dv_net` and `Dw_net` do **not** see
  momentum, so a friction law depending on $|\omega|$ is unrepresentable.
- Both mass nets output the **inverse** $\mathcal M^{-1}$.
- **Total: 1,645,249 parameters.**

Interesting accident of our constants: their pretraining drives
$\mathcal M^{-1}\to I$, which is **exactly right** for our translation block
($1/m = 1$) but wrong for rotation (true $J^{-1} = \mathrm{diag}(2,2,1)$). For
their Crazyflie, $1/m \approx 37$, so it is wrong for both.

---

## 4. The dynamics

$$H(q,p) \;=\; \tfrac12\,\mathbf p^\top M_1^{-1}\mathbf p \;+\; \tfrac12\,\Pi^\top M_2^{-1}\Pi \;+\; V(q)$$

$$\dot x_w = R\,v_b, \qquad \dot R = R\,[\omega_b]_\times$$

$$\dot{\mathbf p} = \mathbf p\times\omega_b \;-\; R^\top\frac{\partial H}{\partial x_w} \;-\; D_v\,v_b \;+\; (gu)_{1:3}$$

$$\dot{\Pi} = \Pi\times\omega_b \;+\; \mathbf p\times v_b \;+\; \mathcal T_R(H) \;-\; D_\omega\,\omega_b \;+\; (gu)_{4:6}$$

with $\mathcal T_R(H) = \sum_{k=1}^{3} r_k\times\dfrac{\partial H}{\partial r_k}$.

- These are the same equations used by the removed full-matrix model — verified term by term.
  Gravity is a body-frame force $-R^\top\nabla_{x_w}H$, not a torque.
- $\mathbf p\times v_b$ and $\mathcal T_R(H)$ are **zero for the true physics**
  but computed anyway.
- Velocity is recovered as
  $\dot\xi = \mathcal M^{-1}\dot p + \frac{d\mathcal M^{-1}}{dt}\,p$.

Implementation note: the published code computes $d\mathcal M^{-1}/dt$ with
**18 separate `autograd.grad` calls** and recovers momentum with
`torch.inverse`. This local implementation uses two `torch.func.jvp` calls and
`linalg.solve`; equivalence is covered by `mini_tests/test_se3hamnode_jvp.py`.

---

## 5. The loss

`pose_L2_geodesic_loss`, split `[3, 9, 6, 4]`:

$$\mathcal L \;=\; \underbrace{\mathrm{MSE}(x_w)}_{\text{position}} \;+\; \underbrace{\mathrm{MSE}(v_b)}_{\text{lin.\ vel}} \;+\; \underbrace{\mathrm{MSE}(\omega_b)}_{\text{ang.\ vel}} \;+\; \underbrace{\mathbb E\!\left[\theta^2\right]}_{\text{attitude}}$$

$$\theta = \arccos\!\left(\frac{\operatorname{tr}(\hat R^\top R) - 1}{2}\right)$$

- Four equally-weighted channels; attitude uses the **geodesic angle on
  $SO(3)$**, not L2 on the nine matrix entries.
- The predicted $R$ is re-orthogonalised first
  (`compute_rotation_matrix_from_unnormalized_rotmat`).
- The $u$ channel is split out and discarded.
- **The loss never sees the true $\mathcal M, V, D, g$.**

---

## 6. The data — ours, in their coordinates

Generated by our
[`windy_quadrotor_datagen.py`](../envs/quadrotor_se3/datagen/windy_quadrotor_datagen.py), then the
control column converted to a wrench (§7).

| setting | value | why |
|---|---|---|
| `--dt` | $1/240$ s | their simulation rate |
| `--u_hold` | 5 | their 48 Hz control inside 240 Hz physics |
| `--num_points` | 5 | window $= 4\times\tfrac{1}{240} = 16.7$ ms |
| `--samples` | 96 | 48 train trajectories |
| `--timesteps` | 40 | $48\times(40-4) = \mathbf{1728}$ windows — **their exact count** |
| `--random_u_scale` | 2.0 | open-loop excitation around hover |
| `--friction_coeff` | 0.5 | $d_{\text{lin}} = d_{\text{ang}} = 0.5$ |

Because `u_hold = 5 ≥ num_points`, $u$ is **constant within every window**, which
is what makes a single `odeint` call valid — exactly their arrangement.

What still differs from their data: **open-loop random excitation** instead of
PID setpoint tracking, **uniform $SO(3)$** initial attitudes instead of
near-level flight, our physical constants, and $R$ exactly on the manifold.

---

## 7. The rotor-to-wrench conversion

Our environment's native input is $u_i = \text{rpm}_i^2$. Their model expects the
**body wrench**. The two are related by an invertible $4\times4$ map — the
non-zero rows of the mixer:

$$\underbrace{\begin{bmatrix}T\\\tau_x\\\tau_y\\\tau_z\end{bmatrix}}_{u_{\text{wrench}}}
=\underbrace{\begin{bmatrix}
k_f&k_f&k_f&k_f\\
-ak_f&-ak_f&ak_f&ak_f\\
-ak_f&ak_f&ak_f&-ak_f\\
-k_m&k_m&-k_m&k_m\end{bmatrix}}_{A\ =\ G_{3:6,:}}
\underbrace{\begin{bmatrix}u_1\\u_2\\u_3\\u_4\end{bmatrix}}_{u_{\text{rotor}}},
\qquad a=\frac{\text{arm}}{\sqrt2}$$

- $\det A = -0.8$ for our constants, so the map is **invertible and lossless** —
  it is a change of input coordinates, not a loss of information.
- Since $G = \begin{bmatrix}0_{2\times4}\\ A\end{bmatrix}$, substituting
  $u_{\text{rotor}} = A^{-1}u_{\text{wrench}}$ gives

$$G\,A^{-1} = \begin{bmatrix}0_{2\times4}\\ I_4\end{bmatrix}$$

  which is exactly their ground-truth $g$. Verified numerically.
- **Consequence:** all the physics that lived in $G$ ($k_f$, $k_m$, arm length)
  moves out of the model and into the data pipeline. `g_net` now only has to
  find a 0/1 pattern — the same, much easier, problem they solve.

Enabled by `u_wrench=True` in `get_dataset`; the dataset cache key records it.

---

## 8. The training loop

Per gradient step, exactly their structure:

1. **One** `odeint(model, x[0], t_eval, method='rk4')` over the whole window —
   valid because $u$ is constant inside it.
2. `pose_L2_geodesic_loss` against samples $1{:}T$.
3. The same on the test set — **every step**, not periodically.
4. `backward()`, `optim.step()` — **skipped at step 0**.

- Adam, `lr = 5e-4`, `weight_decay = 0.0`, **no gradient clipping**.
- Full batch: all 1728 windows at once, no minibatching.

**Pretraining workflow.** Their released trainer does *not* call `pretrain()`; it
loads a cached `-pre-init.tar`. We reproduce that: the first run executes
`pretrain()` (fitting $M_1^{-1}, M_2^{-1} \to I$ over a 64 000-point grid and
250 000 uniform quaternions, looping `while loss > 1e-6`), saves the checkpoint,
and later runs load it. Measured: both converge to $\sim5\times10^{-7}$.

---

## 9. All command-line arguments

### 9.1 Theirs, with their defaults

| argument | type | default |
|---|---|---|
| `--learn_rate` | float | `5e-4` |
| `--total_steps` | int | `5000` |
| `--print_every` | int | `100` |
| `--num_points` | int | `5` |
| `--solver` | str | `rk4` |
| `--seed` | int | `0` |
| `--gpu` | int | `0` |
| `--name` | str | `quadrotor` |
| `--save_dir` | str | `ph-node/data` |
| `--verbose` | flag | off |

### 9.2 Added, to point at our dataset

| argument | type | default | meaning |
|---|---|---|---|
| `--data_dir` | str | our dataset dir | where the pickle lives |
| `--samples` | int | `96` | trajectories (half train, half test) |
| `--timesteps` | int | `40` | chosen so windows $= 1728$ |
| `--dt` | float | `1/240` | their simulation rate |
| `--u_hold` | int | `5` | their control rate |
| `--random_u` | flag | **on** | open-loop excitation |
| `--random_u_scale` | float | `2.0` | half-width around hover |
| `--friction_coeff` | float | `0.5` | $d_{\text{lin}} = d_{\text{ang}}$ |
| `--regenerate_data` | flag | off | ignore the cache |
| `--pretrain` / `--no_pretrain` | flag | **on** | their identity pretraining |

---

## 10. The deviations — an honest list

Five differences remain. Four are consequences of using our data; one is a
genuine code change.

**1. $u$ is the wrench, not rotor speeds.** Required: their model is built
around $g_{\text{true}} = [0;I_4]$. Feeding rotor speeds would be a *different,
harder* problem, not the same model. The conversion is lossless (§7).

**2. One line of their loss changed — it NaN'd.** See §11. This is the only
functional edit to their code.

**3. Physical constants are ours** — $m=1$, $J=(0.5,0.5,1)$, arm $=1$,
$k_f=1$, $k_m=0.1$ — not the Crazyflie's. A direct consequence of using our env.

**4. Excitation is ours** — open-loop random $u$ around hover with uniform
$SO(3)$ initial attitudes, not PID setpoint tracking. This *is* the dataset.

**5. Our $R$ is exactly on $SO(3)$**; PyBullet renormalises quaternions instead.
Strictly better data, but different.

---

## 11. The NaN bug we hit

Their geodesic clamps $\cos$ to **exactly** $\pm1$:

```python
cos = torch.min(cos, torch.ones(batch))
cos = torch.max(cos, torch.ones(batch) * -1)
theta = torch.acos(cos)
```

But

$$\frac{d}{d\cos}\arccos(\cos) \;=\; \frac{-1}{\sqrt{1-\cos^2}} \;\longrightarrow\; -\infty \quad\text{as}\ \cos\to\pm1$$

**Measured on our data:** with 16.7 ms windows the prediction is extremely close
to the target, and **99.7% of samples land on $\cos = 1.0$ exactly** in fp32.
Training produced `nan` on the **first gradient step**.

Verified gradient values:

| $\cos$ | their clamp | strict-interior clamp |
|---|---|---|
| $1.0$ | **$-\infty$** | $0$ |
| $1 - 10^{-8}$ | **$-\infty$** | $0$ |
| $0.999999$ | $-702.5$ | $-702.5$ |
| $0.99$ | $-7.09$ | $-7.09$ |

**Fix applied:** clamp to $\pm(1-10^{-6})$, which caps $|{\rm grad}|$ at about
707. This is the same guard our own `se3_loss_utils.py` already had. It is
marked in the file as the single deviation.

> This looks like a **latent bug in the reference implementation** rather than
> something specific to our data — any run whose predictions get very close to
> the targets in fp32 would hit it.

---

## 12. Two things that had to be worked around

**CUDA out-of-memory at 5568 windows.** Our first attempt used
`--timesteps 120`, giving 5568 windows. Their full-batch loop with
`hidden_dim = 400` and the 18-call `dM_inv/dt` graph exhausted a 40 GB A100.
Matching their **1728** windows fixed it — and incidentally shows their setup
sits close to the memory limit.

**`pretrain()` does not free its tensors.** It allocates 250 000-sample tensors
and only partially frees them. An `empty_cache()` was added before the training
graph is built. This is memory hygiene, not a maths change.

---

## 13. How to run

```bash
python src/models/SE3_Quadrotor/ph-node/train.py \
    --total_steps 5000 --print_every 100 --learn_rate 5e-4 \
    --num_points 5 --solver rk4 --seed 0 \
    --samples 96 --timesteps 40 --dt 0.0041666667 --u_hold 5 \
    --random_u_scale 2.0 --friction_coeff 0.5
```

The first run performs the identity pretraining and caches
`quadrotor-se3ham-pre-init.tar`; later runs load it.

**Status: training in progress.** Results will be appended here. Note that
their `dM_inv/dt` double loop makes each step considerably slower than our
`jvp`-based model, so this run takes noticeably longer than the equivalent
removed full-matrix-model run.

---

## 14. Three-way comparison

| | LieGroupHamDL (theirs) | **this reproduction** | removed full-matrix model |
|---|---|---|---|
| model code | reference | **equation-equivalent optimized implementation** | removed full-matrix model |
| $\mathcal M$ | $M_1(x)$ + $M_2(R)$ | **same** | one $6\times6$ (or `--split_M`) |
| $D$ | $D_v(x)$ + $D_\omega(R)$ | **same** | one $6\times6$ on $(q,p)$ |
| width / gain / $\epsilon$ | 400 / 0.001 / 0.01 | **same** | 64 / 0.5 / 1.0 |
| optimiser | Adam 5e-4, wd 0, no clip | **same** | Adam 1e-3, wd 1e-4, clip 10 |
| $u$ | wrench | **wrench** | rotor speeds |
| true $g$ | $[0;I_4]$ | **$[0;I_4]$** | real CF2X mixer |
| $dt$ / window | $1/240$, 16.7 ms | **same** | 0.05 s, 200 ms |
| windows | 1728 | **1728** | 768 |
| rollout | one `odeint` | **one `odeint`** | segment-wise |
| $d\mathcal M^{-1}/dt$ | 18 `autograd.grad` | **18 `autograd.grad`** | 1 `jvp` |
| simulator | PyBullet | **ours** | ours |
| excitation | PID setpoints | **open-loop random** | open-loop random |
| constants | Crazyflie CF2P | **ours** | ours |
| `arccos` clamp | $\pm1$ (NaNs) | $\pm(1-10^{-6})$ | $\pm(1-10^{-6})$ |
| priors available | none | none | `--const_M/D/G`, `--v_height_only` |

**What this buys us.** With everything except the data held fixed, any
difference in outcome between this run and the reference's published behaviour
is attributable to the dataset alone — open-loop random excitation versus PID
flight, and full-$SO(3)$ coverage versus near-level. That is the controlled
experiment the earlier comparisons could not provide.

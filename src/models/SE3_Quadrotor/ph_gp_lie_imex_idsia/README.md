# PH-GP-LieIMEX quadrotor model

Last updated: 14 September 2026 (reflects the noise-blind recipe with MAP latent initial states, gravity and actuation priors, the window-length curriculum, HARD-V3 / HARD-V3-LINEAR data and the 54-page report).

## 1. Short answer

- This is a JAX port-Hamiltonian model on $SE(3)$ trained from noisy trajectories whose noise level is **not** given to the model.
- Its six unknown physical functions are finite-feature variational Gaussian processes (GPs) around learned constant levels: $M_1^{-1}$, $M_2^{-1}$, $D_v$, $D_\omega$, $V$, and $g$. No mass, inertia, damping coefficient or actuator geometry is typed in.
- A sample has the state $s=[x,\operatorname{vec}(R),v_b,\omega_b,u]\in\mathbb R^{22}$; the model step is $h=0.01\,\mathrm s$.
- Training rolls the model through windows of $K$ transitions with the second-order Lie-IMEX solver and minimises an $SE(3)$ negative log-likelihood with four **learned** noise scales (all initialised at a neutral $0.3$), plus the variational KL.
- Each training window has a **latent initial state** (a MAP correction of the noisy first sample, with the learned noise scales as its prior), which is the errors-in-variables treatment that removes most of the damping bias caused by noisy window starts.
- Two soft physics priors use only known constants: a gravity-consistency penalty ($\mu\nabla V\approx g\,e_3$, $g=9.81$) and an actuation-direction penalty (a pure thrust command produces no lateral force and no torque). Both are flags; magnitudes stay learned.
- Training is a two-stage window-length curriculum, $K=50$ (0.5 s) for 3000 steps from scratch, then $K=100$ (1 s) for 3000 steps from that checkpoint with a fresh optimizer, on overlapping windows (stride 10). Prodigy trains the level scalars, Adam ($10^{-3}$) everything else, batch 256, global-norm clip 10.
- The report is 54 pages and includes gauge-invariant products against the analytical ground truth, a speed-binned damping table, posterior bands, and a closed-loop PyBullet controller that uses the learned damping as feedforward.

The readable model is in [`network.py`](network.py) (`SampledSE3HamODE`, lines 56–178), the Lie-IMEX solver in [`integrator.py`](integrator.py) (`lie_imex_step`, lines 82–138; `rollout_control_sequence`, line 156), and the training loop in [`train.py`](train.py) (`train_from_config`, line 336 onward).

## 2. State, control, and conventions

Each stored sample has $22$ channels:

| State part | Meaning | Dimension |
|---|---|---:|
| $x$ | world-frame position | $3$ |
| $R$ | body-to-world rotation matrix, flattened row-wise | $9$ |
| $v_b$ | body-frame linear velocity | $3$ |
| $\omega_b$ | body-frame angular velocity | $3$ |
| $u=[T,\tau_x,\tau_y,\tau_z]$ | applied thrust and body torques | $4$ |

Therefore,

$$s=\begin{bmatrix}x\\\operatorname{vec}(R)\\v_b\\\omega_b\\u\end{bmatrix}\in\mathbb R^{22},\qquad R\in SO(3),\qquad u\in\mathbb R^4.$$

The vector field reads these state parts directly in [`network.py`, lines 128–140](network.py#L128-L140). Its output has the same $22$ channels, but the last four derivatives are zero because the controls are external inputs:

$$\dot s=\begin{bmatrix}\dot x\\\operatorname{vec}(\dot R)\\\dot v_b\\\dot\omega_b\\0_4\end{bmatrix}.$$

For time-varying control, the stored target-state control is used for each transition:

$$u_{k+1}\text{ drives }s_k\longrightarrow s_{k+1}.$$

This indexing is applied in [`integrator.py`, lines 155–168](integrator.py#L155-L168).

## 3. What a GP subnetwork means here

`gp_setup` is the fixed information needed to evaluate the GPs. It contains the feature generators, physical constants, and static model options. It stays separate from `weights`, which contains the sampled trainable GP values.

For each physical subnetwork $a\in\{M_1,M_2,D_v,D_\omega,V,g\}$, fixed features $\phi_a(z)$ are multiplied by a trainable weight matrix $W_a$:

$$f_a(z)=\phi_a(z)^T W_a.$$

The raw multiplication is in [`_backend.py`, lines 628–631](_backend.py#L628-L631).

Every weight has a mean-field Gaussian posterior:

$$q(W_a)=\mathcal N\!\left(M_a,\operatorname{diag}\!\left(e^{2S_a}\right)\right).$$

During training, weights are sampled using

$$W_a=M_a+e^{S_a}\odot\epsilon_a,\qquad \epsilon_a\sim\mathcal N(0,I).$$

The six samples are created together in [`_backend.py`, lines 574–589](_backend.py#L574-L589). One coherent draw defines the whole model for one optimizer update and its complete rollout; see `_gp_components` in [`train.py`, line 287](train.py#L287) and `objective`, line 557. Evaluation passes no random key, so it uses the six posterior means.

### 3.1 Exact utility GP feature map

The current config sets `gp.model_backend: utils-gp-model`. Therefore, all six subnetworks use the `GP_Model` implementation copied exactly from `src/utils/JAX/gp_model.py`. The local copy must remain byte-for-byte identical to that source.

The input adapter converts each quadrotor quantity to the copied GP's $9$- or $12$-dimensional convention:

| GP | Physical input | Input given to `GP_Model` |
|---|---|---|
| $M_1$ | $x\in\mathbb R^3$ | $[\operatorname{vec}(I_3),x]\in\mathbb R^{12}$ |
| $M_2$ | $\operatorname{vec}(R)\in\mathbb R^9$ | $\operatorname{vec}(R)\in\mathbb R^9$ |
| $D_v$ | $v_b\in\mathbb R^3$ | $[\operatorname{vec}(I_3),v_b]\in\mathbb R^{12}$ |
| $D_\omega$ | $\omega_b\in\mathbb R^3$ | $[\operatorname{vec}(I_3),\omega_b]\in\mathbb R^{12}$ |
| $V$ | $x\in\mathbb R^3$ (position only) | $[\operatorname{vec}(I_3),x]\in\mathbb R^{12}$ |
| $g$ | $(x,R)$ | $[\operatorname{vec}(R),x]\in\mathbb R^{12}$ |

This small adapter is in [`network.py`, lines 31–43](network.py#L31-L43).

The Matérn feature map measures the geodesic angle between the input rotation $R$ and a fixed random base rotation $B_j$:

$$\theta_j(R)=\operatorname{atan2}\!\left(\sqrt{1-c_j^2},c_j\right),\qquad c_j=\operatorname{clip}\!\left(\frac{\operatorname{tr}(B_j^TR)-1}{2},-1,1\right).$$

For a $12$-dimensional input, its last three values $z$ are also included. One Matérn feature is

$$\phi_{M,j}(R,z)=\sqrt{\frac{2}{D_M}}\cos\!\left(\omega_{\theta,j}\theta_j(R)+\omega_{z,j}^Tz+b_j\right).$$

For a $9$-dimensional input, the $\omega_{z,j}^Tz$ term is absent. Random rotations, Matérn frequencies, phases, and the safe angle calculation are implemented in [`gp_model.py`, lines 18–95](gp_model.py#L18-L95).

The second feature map is periodic. With maximum harmonic $m_{\max}$, it has

$$D_P=1+2m_{\max}$$

features containing one constant plus sine/cosine pairs. Its coefficients are based on

$$a_m=\exp\!\left(-\frac{2\pi^2\ell_p^2m^2}{P^2}\right).$$

The current values are $P=2\pi$, $\ell_p=0.5$, and $m_{\max}=5$, so $D_P=11$. See [`gp_model.py`, lines 97–130](gp_model.py#L97-L130).

The final feature is the product

$$\phi_a(z)=\phi_{M,a}(z)\otimes\phi_P(z_{d_p}),$$

and the raw GP output is

$$f_a(z)=\sum_{i=1}^{D_M}\sum_{j=1}^{D_P}\phi_{M,a,i}(z)\phi_{P,j}(z_{d_p})W_{a,ij}.$$

The product construction is in [`_backend.py`, lines 604–614](_backend.py#L604-L614), and its equivalent implementation in the copied model is in [`gp_model.py`, lines 167–182](gp_model.py#L167-L182). The current `periodic_dimension` is $d_p=0$, meaning the first value of each adapted GP input supplies the periodic scalar.

## 4. Every subnetwork: input, raw output, physical output, and parameters

The exact-backend GP specifications are created in [`_backend.py`, lines 350–417](_backend.py#L350-L417). Because $D_P=11$, the complete feature dimension is $D=D_MD_P$.

| GP | Physical input | Matérn features $D_M$ | Total features $D$ | Raw output | Physical output | Weight shape |
|---|---|---:|---:|---:|---|---:|
| $M_1$ | $x$ | $220$ | $2420$ | $1$ | $M_1^{-1}(x)\in\mathbb R^{3\times3}$ | $2420\times1$ |
| $M_2$ | $R$ | $220$ | $2420$ | $6$ | $M_2^{-1}(R)\in\mathbb R^{3\times3}$ | $2420\times6$ |
| $D_v$ | $v_b$ | $225$ | $2475$ | $6$ | $D_v(v_b)\in\mathbb R^{3\times3}$ | $2475\times6$ |
| $D_\omega$ | $\omega_b$ | $220$ | $2420$ | $6$ | $D_\omega(\omega_b)\in\mathbb R^{3\times3}$ | $2420\times6$ |
| $V$ | $x$ | $20$ | $220$ | $1$ | $V(x)\in\mathbb R$ | $220\times1$ |
| $g$ | $(x,R)$ | $20$ | $220$ | $24$ | $g(x,R)\in\mathbb R^{6\times4}$ | $220\times24$ |

Each weight entry has a posterior mean and posterior log-standard-deviation. Hence the exact-backend trainable GP parameter count is

$$2\left(2420+2420\cdot6+2475\cdot6+2420\cdot6+220+220\cdot24\right)=103{,}620.$$

The learned levels add $1+6+6+6+3+24=46$ point-estimate scalars ($\lambda_0$ for $M_1$, six each for $M_2$, $D_v$, $D_\omega$, the linear level $\lambda_V$ of $V$, and the $6\times4$ level of $g$), and the `se3-nll` data fit adds four likelihood log-standard-deviations, giving $103{,}670$ trainable scalar values, plus $12$ latent initial-state coordinates per training window. The exact backend keeps its random feature objects fixed and trains only these variational arrays; initialization is in [`_backend.py`, lines 509–543](_backend.py#L509-L543).

The configured initial posterior value is

$$\log\sigma_W=-5,\qquad \sigma_W=e^{-5}\approx0.00673795.$$

Although the untouched copied `GP_Model` has its own internal default of $-2$, the local backend explicitly replaces the optimized posterior log-standard-deviation with the config value $-5$ in [`_backend.py`, lines 536–540](_backend.py#L536-L540).

### 4.1 Translational inverse-mass GP $M_1$

The GP emits one scalar $f(x)$. Together with one learned level $\lambda_0$ it becomes an isotropic positive inverse mass:

$$\mu(x)=e^{\lambda_0+f(x)},$$

$$M_1^{-1}(x)=\mu(x)I_3.$$

No vehicle mass is read into the model. $\lambda_0$ is a trainable scalar stored as `M1/log_level`, initialised at $0$ so that $M_1^{-1}=I_3$ before training; it is a point estimate and is not part of the KL term. Far from the data $f(x)\to0$ and $\mu\to e^{\lambda_0}$, so $1/e^{\lambda_0}$ is the learned mass estimate. The exponential keeps $\mu(x)>0$. See [`network.py`, lines 52–55](network.py#L52-L55) and [`_backend.py`, lines 542–543, 591](_backend.py#L542-L543).

### 4.2 Rotational inverse-mass GP $M_2$

The GP emits $f(R)\in\mathbb R^6$. Together with six learned levels $\lambda\in\mathbb R^6$ (stored as `M2/level`, point estimates, not in the KL) it builds a lower-triangular factor

$$L(R)=\begin{bmatrix}e^{\lambda_1+f_1(R)}&0&0\\ \lambda_4+f_4(R)&e^{\lambda_2+f_2(R)}&0\\ \lambda_5+f_5(R)&\lambda_6+f_6(R)&e^{\lambda_3+f_3(R)}\end{bmatrix},\qquad M_2^{-1}(R)=L(R)L(R)^T.$$

The positive diagonal makes $L$ invertible, so $M_2^{-1}$ is symmetric positive definite without a floor. No vehicle inertia is read into the model. The config value `gp.inverse_mass_2_initial_value` $=c$ sets the start: $\lambda_{1,2,3}=\tfrac12\log c$, $\lambda_{4,5,6}=0$, so $M_2^{-1}=cI_3$ before training (currently $c=1000$; a gauge choice). Far from the data $f\to0$, and the learned inertia estimate is $\hat J=\left(L(\lambda)L(\lambda)^T\right)^{-1}$. See [`network.py`, lines 68–70](network.py#L68-L70) and [`_backend.py`, lines 509–520](_backend.py#L509-L520).

### 4.3 Translational dissipation GP $D_v$

The GP emits $f(v_b)\in\mathbb R^6$ from the body velocity. With six learned levels $\lambda\in\mathbb R^6$ (`Dv/level`, point estimates, not in the KL) it builds the same lower-triangular factor as $M_2^{-1}$:

$$L_v(v_b)=\begin{bmatrix}e^{\lambda_1+f_1}&0&0\\ \lambda_4+f_4&e^{\lambda_2+f_2}&0\\ \lambda_5+f_5&\lambda_6+f_6&e^{\lambda_3+f_3}\end{bmatrix},\qquad D_v(v_b)=L_v(v_b)L_v(v_b)^T\succeq0.$$

No mass, damping coefficient, or damping law is read into the model; all velocity dependence is learned by $f(v_b)$. The config value `gp.dissipation_v_initial_value` $=c_v$ sets the start $D_v=c_vI_3$ (currently $c_v=1$, so the initial damping rate $M_1^{-1}D_v$ is $1\,\mathrm{s^{-1}}$ with $M_1^{-1}=I_3$). The shared factor builder is [`network.py`, lines 43–54](network.py#L43-L54); the matrix is assembled in [`network.py`, lines 72–75](network.py#L72-L75). `model.dissipation_enabled` still gates the matrix.

### 4.4 Rotational dissipation GP $D_\omega$

Identical construction with input $\omega_b$ and levels `Dw/level`:

$$D_\omega(\omega_b)=L_\omega(\omega_b)L_\omega(\omega_b)^T\succeq0.$$

The start is $D_\omega=c_\omega I_3$ from `gp.dissipation_w_initial_value` (currently $c_\omega=10^{-3}$, so the initial rate $M_2^{-1}D_\omega$ is $1\,\mathrm{s^{-1}}$ with $M_2^{-1}=1000I_3$). See [`network.py`, lines 77–80](network.py#L77-L80).

### 4.5 Potential GP $V$ (learned linear level + GP, optional structured form)

The potential is a position-only GP (`potential_include_rotation: false`) on top of a **learned linear level** $\lambda_V\in\mathbb R^3$ (stored as `V/level`, point estimate, not in the KL, zero-initialised):

$$V(x)=\lambda_V^\top x+\phi_V(x)^\top w_V .$$

The linear level is what carries gravity: a GP with Matérn features returns to zero far from the data, so without $\lambda_V$ the gravitational slope could not extrapolate. Because $\partial V/\partial R=0$, the potential produces no torque. Implementation: `_learned_potential` in [`network.py`, lines 93–103](network.py#L93-L103); the level is created in `_backend.py` (`initialize_variational_parameters`) and read in `sample_weights` (`_V_level`).

Three mutually exclusive ways to give the model the constant $g$ exist, selected in the `gp` block of the config:

| option | config | what it does | status |
|---|---|---|---|
| gravity-consistency penalty (**used**) | `gravity_penalty_weight: 30`, `known_gravity: 9.81` | soft loss on the identifiable product, Section 7.4 | final recipe |
| level prior | `gravity_level_prior: true` | replaces the $z$ slope of the level by $g\,e^{-\lambda_0}$ | ablation; the mass GP residual absorbed it |
| structured potential | `structured_potential: true` | $V=g\,z/\mu(x)$, V GP and level frozen | ablation; 5–7 % gravity error because $\nabla(1/\mu)\neq0$ |

With none of them, $V$ is fully learned and the hover flat direction (thrust and gravity cancel) is unconstrained.

### 4.6 Control-map GP $g$

The raw $24$-vector is reshaped into a $6\times4$ matrix and added to a learned constant level $\Lambda_0\in\mathbb R^{6\times4}$ (stored as `g/level`, point estimate, not in the KL):

$$g(x,R)=\Lambda_0+\operatorname{reshape}_{6\times4}\!\left(\phi_g(x,R)^TW_g\right).$$

No selection matrix or actuator geometry is typed in: $\Lambda_0$ is initialised at $0$ and learned, and the GP learns the pose-dependent residual around it. See [`network.py`, lines 105–109](network.py#L105-L109). Because $\Lambda_0$ is free, the data determine only the products $M_1^{-1}g$, $M_2^{-1}g$, $M_1^{-1}\nabla V$, $M_1^{-1}D_v$, $M_2^{-1}D_\omega$ (the port-Hamiltonian gauge); the report compares these gauge-invariant products with the ground truth (page 3 and pages 4–9).

With noisy data the pose-dependent residual of the thrust column leaks thrust into torque and lateral force (up to $0.4\,\mathrm{rad/s^2}$ per hover thrust at $\sigma=0.5$), which makes open-loop rollouts tilt and drift during hover. The **actuation-direction prior** (Section 7.5) suppresses it; it constrains only the direction of the thrust column, so the thrust gain $\mu g_{fz}$, all torque gains and the position dependence remain learned. If `direct_control_map: true`, the level is omitted (ablation).

## 5. Port-Hamiltonian mathematics

### 5.1 Hamiltonian

Let $p_v$ and $p_\omega$ be the translational and rotational body momenta. The Hamiltonian is

$$H(x,R,p_v,p_\omega)=\frac12p_v^TM_1^{-1}(x)p_v+\frac12p_\omega^TM_2^{-1}(R)p_\omega+V(x,R).$$

The observed body velocities satisfy

$$v_b=M_1^{-1}(x)p_v,\qquad \omega_b=M_2^{-1}(R)p_\omega.$$

Consequently, the code reconstructs momenta as

$$p_v=M_1(x)v_b,\qquad p_\omega=M_2(R)\omega_b.$$

The Hamiltonian is in [`network.py`, lines 114–126](network.py#L114-L126), and momentum reconstruction is in [`network.py`, lines 136–140](network.py#L136-L140).

### 5.2 Pose kinematics

Define

$$H_x=\frac{\partial H}{\partial x},\qquad H_R=\frac{\partial H}{\partial R},\qquad H_{p_v}=\frac{\partial H}{\partial p_v},\qquad H_{p_\omega}=\frac{\partial H}{\partial p_\omega}.$$

The pose equations are

$$\dot x=RH_{p_v},$$

$$\dot R=R\widehat{H_{p_\omega}},$$

where $\widehat a b=a\times b$. They are evaluated in [`network.py`, lines 141–147](network.py#L141-L147).

### 5.3 Momentum equations and the drift term

The learned generalized control wrench is

$$\begin{bmatrix}f_u\\\tau_u\end{bmatrix}=g(x,R)u.$$

The momentum dynamics are

$$\dot p_v=p_v\times H_{p_\omega}-R^TH_x-D_v(v_b)H_{p_v}+f_u,$$

$$\dot p_\omega=p_\omega\times H_{p_\omega}+p_v\times H_{p_v}+\sum_{i=1}^{3}R_{i,:}\times(H_R)_{i,:}-D_\omega(\omega_b)H_{p_\omega}+\tau_u.$$

They are implemented term by term in [`network.py`, lines 145–161](network.py#L145-L161).

In control-affine notation,

$$\dot z=f_\theta(z)+G_\theta(z)u.$$

Here, $G_\theta(z)u$ is the `force_torque` value on [`network.py`, line 148](network.py#L148). Everything else in the pose and momentum equations is the drift $f_\theta(z)$. Therefore, the drift is present, but it is constructed from the port-Hamiltonian physics instead of being represented by a separate unconstrained drift network.

Because data store velocities rather than momenta, the product rule gives

$$\dot v_b=M_1^{-1}\dot p_v+\dot M_1^{-1}p_v,$$

$$\dot\omega_b=M_2^{-1}\dot p_\omega+\dot M_2^{-1}p_\omega.$$

JAX Jacobian-vector products compute $\dot M_1^{-1}$ and $\dot M_2^{-1}$ in [`network.py`, lines 163–170](network.py#L163-L170).

### 5.4 Why this is port-Hamiltonian

The pose and cross-product terms exchange energy internally. The positive-semidefinite dissipation matrices remove energy, and the input map supplies external power. Ignoring integration error, the energy balance has the form

$$\dot H=-H_{p_v}^TD_vH_{p_v}-H_{p_\omega}^TD_\omega H_{p_\omega}+\begin{bmatrix}H_{p_v}\\H_{p_\omega}\end{bmatrix}^Tg(x,R)u.$$

For $u=0$,

$$\dot H\le0,$$

because $D_v\succeq0$ and $D_\omega\succeq0$ by construction.

## 6. Lie-IMEX integration

The solver keeps the nonlinear damping implicit and treats the remaining vector field explicitly. Define

$$\xi=\begin{bmatrix}v_b\\\omega_b\end{bmatrix},\qquad K(s)=\operatorname{blkdiag}\!\left(M_1^{-1}D_v,M_2^{-1}D_\omega\right).$$

The two $3\times3$ damping matrices are provided in [`network.py`, lines 175–181](network.py#L175-L181). Write the twist dynamics as

$$\dot\xi=F_{\mathrm{exp}}(s,u)-K(s)\xi.$$

At state $s_n$, the predictor solves

$$\left(I+hK_n\right)\xi^*=\xi_n+hF_{\mathrm{exp},n}.$$

Its pose predictor is

$$x^*=x_n+h\dot x_n,$$

$$R^*=R_n\operatorname{Exp}\!\left(h\widehat{\omega_n}\right).$$

These operations are in [`integrator.py`, lines 85–110](integrator.py#L85-L110). The linear systems are solved independently for linear and angular velocity in [`integrator.py`, lines 63–79](integrator.py#L63-L79).

At the predicted state, let $F_{\mathrm{exp},*}$ and $K_*$ be the new explicit derivative and damping matrix. The corrector implemented by the code is

$$\left(I+\frac h2K_*\right)\xi_{n+1}=\xi_n+\frac h2\left(F_{\mathrm{exp},n}+F_{\mathrm{exp},*}-K_n\xi_n\right).$$

The pose corrector is

$$x_{n+1}=x_n+\frac h2\left(\dot x_n+\dot x^*\right),$$

$$R_{n+1}=R_n\operatorname{Exp}\!\left(\frac12\left(h\widehat{\omega_n}+h\widehat{\omega^*}\right)\right).$$

The corrector is in [`integrator.py`, lines 112–136](integrator.py#L112-L136). The exponential map is implemented with Rodrigues' formula in [`integrator.py`, lines 16–36](integrator.py#L16-L36), so rotations remain on $SO(3)$ up to floating-point error.

## 7. Training objective

### 7.1 $SE(3)$ negative log-likelihood with learned noise scales

Let $(x,R,v_b,\omega_b)$ be an observed target state of a window and $(\hat x,\hat R,\hat v_b,\hat\omega_b)$ its rollout prediction. Each of the four state blocks $b\in\{x,R,v,\omega\}$ has one trainable noise scale $\sigma_b$ (`likelihood/log_sigma_*`). For a Euclidean three-vector block,

$$\mathcal L_{\mathrm{NLL},b}=\frac{1}{2\sigma_b^2}\operatorname{mean}\!\left(\lVert y_b-\hat y_b\rVert_2^2\right)+3\log\sigma_b+\tfrac32\log(2\pi),$$

and for attitude the squared norm is replaced by the squared geodesic angle $\theta(R,\hat R)=\cos^{-1}\!\big(\tfrac{\operatorname{tr}(R\hat R^T)-1}{2}\big)$. The data fit is the sum of the four blocks, averaged over the $K$ transitions of every window in the batch ([`losses.py`, lines 82–179](losses.py#L82-L179)).

**The observation noise is not an input.** All four $\sigma_b$ start at the neutral value `gp.initial_observation_sigma: 0.3` regardless of the dataset and are learned; on HARD-V3 they converge to $0.10$–$0.19$ for the $0.1$-noise data and $0.50$–$0.52$ for the $0.5$-noise data within the first stage. `gp.fixed_observation_sigma` freezes them (ablation only).

### 7.2 Variational KL term

The prior for every GP weight is $\mathcal N(0,I)$; the KL over all six GPs is
$$\mathcal L_{\mathrm{KL}}=\tfrac12\sum_a\sum_j\left(e^{2S_{a,j}}+M_{a,j}^2-1-2S_{a,j}\right)$$
(`_backend.kl_divergence`). It is spread over every training transition $N=B_{\mathrm{train}}K$ and warmed up linearly over `gp.kl_anneal_steps` $=1000$ updates to `gp.kl_beta_max` $=1$. Levels and $\sigma_b$ are point estimates outside the KL.

### 7.3 Latent initial state per window (errors-in-variables treatment)

Rolling a window out from its *noisy* first sample biases the model: the loss then contains a term $(1-dh)^{2k}\sigma^2$ from the initial-state noise, which is cheapest to remove by over-damping, so low-speed damping came out $17$–$30\times$ too high. Every training window $i$ therefore gets a learned correction $m_i\in\mathbb R^{12}$ on the tangent coordinates of its first state,

$$x_0^{(i)}=x_0^{obs}+m_i^{x},\qquad R_0^{(i)}=R_0^{obs}\exp\!\big(\widehat{m_i^{R}}\big),\qquad v_0^{(i)}=v_0^{obs}+m_i^{v},\qquad \omega_0^{(i)}=\omega_0^{obs}+m_i^{\omega},$$

and the rollout starts from the corrected state (`apply_initial_correction`, [`train.py`, line 523](train.py#L523)). The correction is a **MAP** estimate (`latent_initial_state_sample: false`): no sampling, and the prior is the observation model applied to the first sample with the same learned $\sigma_b$,

$$\mathcal L_{x_0}=\frac1K\sum_b\Big(\frac{\lVert m_i^{(b)}\rVert^2}{2\sigma_b^2}+3\log\sigma_b\Big),$$

i.e. the missing $-\log p(x_0^{obs}\mid x_0)$ of the window likelihood (`latent_prior_sigma`, `latent_terms`, [`train.py`, lines 532–555](train.py#L532-L555)). The $m_i$ are trained with their own Adam ($10^{-2}$, `latent_initial_state_learning_rate`), stored in checkpoints, and reset at every curriculum stage because the window set changes.

Two variants exist for ablation and are **not** used: sampling the initial state from $q(x_0)=\mathcal N(x_0^{obs}+m_i,s_i^2)$ with the full Gaussian KL (`latent_initial_state_sample: true`; the sampled perturbation re-injects exactly the noise the bias feeds on), and a floor on $s_i$ (`latent_initial_state_std_floor`; made damping $136\times$ too high). A fixed numeric `latent_initial_state_sigma` assumes a known noise level and is only for ablation.

Flags: `training.latent_initial_state` (on/off), `latent_initial_state_sigma: learned`, `latent_initial_state_kl_weight: 1.0`.

### 7.4 Gravity-consistency prior

On 1024 poses sub-sampled from the batch (`gp.gravity_penalty_points`),

$$\mathcal P_V=\frac{1}{g^2}\,\mathbb E_x\big\lVert\mu(x)\nabla V(x)-g\,e_3\big\rVert^2,\qquad g=9.81,$$

added with weight `gp.gravity_penalty_weight` $=30$ (`gravity_penalty`, [`train.py`, line 452](train.py#L452)). It pins the gauge-invariant gravity product without fixing $V$, $\mu$ or any mass. Reason: near hover the thrust and gravity products cancel, so their common scale is a flat direction of any trajectory loss; with $w=1$ gravity moved from $7.8$ to $8.4$, with $w=30$ to $9.6$–$9.8$.

### 7.5 Actuation-direction prior

With $u_h$ the mean thrust magnitude of the batch and $e_1$ the thrust command,

$$\mathcal P_g=\frac{1}{g^2}\,\mathbb E_x\Big[\big\lVert\mu(x)\,[g(x)e_1]_{xy}\,u_h\big\rVert^2+\big\lVert M_2^{-1}(x)\,[g(x)e_1]_{\tau}\,u_h\big\rVert^2\Big],$$

i.e. the lateral and angular acceleration a pure thrust command would produce, in units of $g$, added with weight `gp.actuation_penalty_weight` $=30$ (`actuation_penalty`, [`train.py`, line 470](train.py#L470)). Physically the wrench directions come from the known motor mixer; the thrust gain, all torque gains and the pose dependence stay learned. It is exact in the simulator; on real vehicles the coupling is a small constant (centre-of-gravity offset, motor tilt), so a residual-only variant that leaves the constant level free would be the robust form (not implemented).

### 7.6 Complete objective

At optimizer update $n$, with a fresh GP weight sample per step shared by all terms,

$$\boxed{\mathcal J_n=\mathcal L_{\mathrm{NLL}}+\frac{\beta_n}{N}\mathcal L_{\mathrm{KL}}+\mathcal L_{x_0}+30\,\mathcal P_V+30\,\mathcal P_g},\qquad \beta_n=\min(n/1000,1).$$

Gradients flow through the whole rollout into the GP parameters and into the latent corrections (`objective`, `training_step`, [`train.py`, lines 557–600](train.py#L557-L600)); the global gradient norm is clipped at 10.

### 7.7 Optimizer

`optimizer.name: prodigy-levels-per-subnetwork` ([`train.py`, lines 112–157](train.py#L112-L157)): the level scalars of $M_1$, $M_2$, $D_v$, $D_\omega$, $V$ and $g$ each get their own learning-rate-free Prodigy group (`level_learning_rate` $=\eta$: $0.5$ at $\sigma=0.1$, $0.25$ at $\sigma=0.5$, where $0.5$ diverged once); GP means, log-stds and the four $\log\sigma_b$ use Adam $10^{-3}$. A frozen group (`optax.set_to_zero`) is used for the $V$ level when the structured potential is on. Mass pretraining (`mass_pretrain_steps`) is kept at $0$: it would require the true mass and inertia.

### 7.8 Window-length curriculum

Long windows make a dynamics bias $b$ visible through $(bT/\sigma)^2$, but training on them from scratch is unstable. Stages are chained with [`experiments/quadrotor/.chain_logs/run_chain.sh`](../../../../experiments/quadrotor/.chain_logs/run_chain.sh), each stage a separate config whose `training.initial_checkpoint` points at the previous stage's `checkpoint_final.pkl` (parameters only: optimizer state and latents are fresh):

| stage | `window_points` | horizon | steps | note |
|---|---:|---:|---:|---|
| 1 | 51 ($K=50$) | 0.5 s | 3000 | from scratch |
| 2 | 101 ($K=100$) | 1 s | 3000 | **final model** |
| (3) | 201 ($K=200$) | 2 s | 4000 | dropped: raised damping, 1 s and 3 s errors in every run |

Windows overlap with `window_stride: 10`, so every measurement is predicted from 10 different starts at horizons 0.1–1 s (a multi-horizon loss by layout); 50 flights of 10 s give 4550 windows per stage. A multiple-shooting continuity penalty between overlapping windows is the natural next step and is not implemented.

## 8. Datasets

### 8.1 HARD-V3 (nonlinear damping, default)

`datasets/QUADROTOR-DATASET-HARD-V3/HARDV3_CF2P_10s_h0p01_*.pkl`, generated by `generate_quadrotor_hard_v2.py` (since merged into [`envs/pybullet_quadrotor_se3/datagen/generate_dataset.py`](../../../../envs/pybullet_quadrotor_se3/datagen/generate_dataset.py)) with `hard_v3_yaw_config.yaml`:

- Gym-PyBullet-Drones CF2P, `Physics.PYB` at 1000 Hz, contact-free (ground plane removed, contact friction zeroed), PyBullet built-in damping $c=0.5$ (translational $a=-c(1+\lVert v\rVert)v$, rotational likewise).
- 50 training flights (seeds 300–304, 10 each) and 10 clean test flights (seed 399), 10 s each sampled at 100 Hz: arrays $50\times1001\times22$ and $10\times1001\times22$.
- Half-gain DSL PID flying random manoeuvre sequences (waypoint hops, figure-eights, circles, vertical steps, yaw turns, aggressive-recovery torque kicks, disturbed hover, **yaw kicks** $J_{zz}\cdot$rate about body $z$ at 3–6 rad/s). Speeds up to 3.3 m/s, tilt up to 62°. The recorded control is the motor wrench after RPM clipping plus the external kick torque.
- Files with absolute observation noise $\sigma\in\{0.05,0.1,0.25,0.5\}$ on all 18 state channels of the training flights (rotations re-projected onto $SO(3)$), the test split clean, plus a sensor-noise variant. `data.observation_noise_override` is `null`: the noise is in the file, never added by the trainer.

Vehicle constants (used only by the report's ground truth): $m=0.027\,\mathrm{kg}$, $J=\operatorname{diag}(2.3951,2.3951,3.2347)\times10^{-5}\,\mathrm{kg\,m^2}$, $g=9.8\,\mathrm{m/s^2}$.

### 8.2 HARD-V3-LINEAR (linear damping)

Same protocol with `environment.damping_law: linear` (`hard_v3_linear_config.yaml`): built-in damping off, an external force $-c\,m\,v$ and torque $-c\,J\,\omega$ applied every physics step ($c=0.5$, verified exact: $v\to v(1-c\,dt)$). Speeds reach 4.2 m/s. Files `datasets/QUADROTOR-DATASET-HARD-V3-LINEAR/HARDV3LIN_CF2P_10s_h0p01_*.pkl`. Truth for the report: $\mu D_v=M_2^{-1}D_\omega=c\,I$.

### 8.3 Report reference flights

Clean 240 Hz D0-protocol PID flights of any length from [`envs/pybullet_quadrotor_se3/datagen/generate_reference_flights.py`](../../../../envs/pybullet_quadrotor_se3/datagen/generate_reference_flights.py) (`--duration-seconds`, `--damping-law`): `datasets/data/pybullet_quadrotor/D0_CF2P_PID_contact-free_{nonlinear,linear}-damping-c0p5_{3s,10s,30s}_seed0.pkl`. The 3 s flight (fast for 1.2 s, then hover) is the default `report.evaluation_dataset`.

### 8.4 Windows

`_time_major_windows` in [`data.py`, line 15](data.py#L15) cuts each flight into windows of `window_points` samples with stride `window_stride`; tensors are $(K+1)\times B\times22$ in `float32`. `load_test_flights` ([`data.py`, line 67](data.py#L67)) returns the clean test flights for the long-horizon metrics.

## 9. Current configuration (final recipe)

Canonical stage-2 configs: [`13-09-2026-03-01_ph_gp_lie_imex.yaml`](../configs/13-09-2026-03-01_ph_gp_lie_imex.yaml) ($\sigma=0.1$) and [`13-09-2026-03-11_ph_gp_lie_imex.yaml`](../configs/13-09-2026-03-11_ph_gp_lie_imex.yaml) ($\sigma=0.5$); stage 1 is `03-00` / `03-10`. Linear-damping twins: `04-0x` / `04-1x`.

| Configuration | Value |
|---|---:|
| Data | HARD-V3, noise file 0.1 or 0.5, test split clean |
| Windows | 101 points ($K=100$, 1 s), stride 10, overlapping |
| Data fit | $SE(3)$ NLL, four learned $\sigma_b$ from 0.3 |
| Latent initial state | on, MAP (no sampling), prior = learned $\sigma_b$, Adam $10^{-2}$ |
| Gravity prior | penalty weight 30, $g=9.81$, 1024 poses |
| Actuation prior | penalty weight 30 |
| Level prior / structured $V$ | off |
| KL | $\beta_{\max}=1$, warm-up 1000 steps, divided by all training transitions |
| Optimizer | Prodigy on levels ($\eta=0.5$ at 0.1 noise, $0.25$ at 0.5), Adam $10^{-3}$ elsewhere, clip 10 |
| Steps / batch | 3000 per stage, 256 windows |
| Evaluation | every 250 steps: window test loss and open-loop 1 s / 3 s position RMS on the clean test flights (`training.long_horizon_test_seconds`) |
| Checkpoints | every 500 steps + final; latents included |
| Initial $M_2^{-1}$, $D_v$, $D_\omega$ | $1000I_3$, $I_3$, $10^{-3}I_3$ (gauge choices) |
| Posterior initial log-std | $-5$ |
| Features | Matérn $\nu=2.5$, $\ell=1$; periodic $P=2\pi$, $\ell_P=0.5$, 5 harmonics; $D_M=220/220/225/220/20/20$ |
| Precision | `float32`, `matmul_precision: highest`, compilation cache on |
| Report | auto after stage 2, 54 pages, controller with damping feedforward (`report.controller_use_dissipation: true`) |

Run one stage from the project root:

```bash
python -m src.models.SE3_Quadrotor.ph_gp_lie_imex.train \
  --config src/models/SE3_Quadrotor/configs/13-09-2026-03-00_ph_gp_lie_imex.yaml
```

or a whole chain (the script substitutes `__PREV_CHECKPOINT__` in each following config):

```bash
bash experiments/quadrotor/.chain_logs/run_chain.sh 0 F1 \
  src/models/SE3_Quadrotor/configs/13-09-2026-03-00_ph_gp_lie_imex.yaml \
  src/models/SE3_Quadrotor/configs/13-09-2026-03-01_ph_gp_lie_imex.yaml
```

Config names must follow `DD-MM-YYYY-HH-MM_<model>.yaml` ([`config.py`](config.py)); with `CUDA_VISIBLE_DEVICES` set, `runtime.device_index` stays 0. The optional keys added since the original design (`training.latent_initial_state*`, `training.long_horizon_test_seconds`, `gp.known_gravity`, `gp.gravity_level_prior`, `gp.gravity_penalty_weight`, `gp.gravity_penalty_points`, `gp.structured_potential`, `gp.actuation_penalty_weight`, `report.evaluation_dataset`, `report.controller_use_dissipation`, `report.posterior_samples`) default to off when absent, so older configs still run.

## 10. Evaluation and report

- During training: `test1s[...]`/`test3s[...]` in the log are open-loop position RMS over the clean test flights from random offsets, the metric used to compare recipes.
- [`../comparision/generate_report.py`](../comparision/generate_report.py) (`--config <run>/<config>.yaml --run <run> [--evaluation-dataset ...]`) produces the 54-page report: 1 summary, 2 subnetwork MSE / NMSE / relative RMS (raw and mass-gauge-fixed), **3 gauge-invariant products** $\mu g_f$, $M_2^{-1}g_\tau$, $\mu\nabla V$, $\mu D_v$, $M_2^{-1}D_\omega$ with a speed-binned $\mu D_v$ table, 4–9 the same products along the reference flight and damping-vs-speed scatter, 10–19 training curves, 20–30 open-loop errors, energy, $SO(3)$ checks, state trajectories with posterior $\pm2\sigma$ bands, phase portraits, 31–42 raw and gauge-fixed subnetwork trajectories, 43–48 compute, controller table, GP uncertainty, controller plots, 49–54 audits. The ground-truth damping law is read from the dataset settings (`damping_law`).
- The closed-loop controller ([`../comparision/report_controller.py`](../comparision/report_controller.py)) flies a 20 s diamond in PyBullet with an energy-based law and gauge-invariant allocation; `use_dissipation` adds the learned $+D_vv_b$, $+D_\omega\omega_b$ feedforward (the demanding test), off gives the reference law.
- Helper diagnostics in `experiments/quadrotor/.chain_logs/`: `damping_bins.py` (speed-binned products, `DAMPING_LAW`, `CLEAN_PICKLE`), `leak_check.py` (thrust-to-torque leak), `openloop_axes.py` / `hover_budget.py` (per-axis drift, `EVAL_PICKLE`), `ctrl_diag.py` (controller with/without feedforward).

## 11. Results snapshot (14 September 2026, stage-2 models with both priors)

| quantity | HARD-V3 $\sigma=0.1$ | HARD-V3 $\sigma=0.5$ | linear $\sigma=0.1$ | linear $\sigma=0.5$ | truth |
|---|---|---|---|---|---|
| learned $\sigma_v$ | 0.125 | 0.504 | 0.137 | 0.507 | 0.1 / 0.5 |
| thrust gain $\mu g_{fz}$ | 36.5 | 36.8 | 36.2 | 36.8 | 37.0 |
| gravity $\mu\partial_zV$ | 9.65 | 9.76 | 9.56 | 9.76 | 9.8 |
| torque gains, rel. RMS | 3.3 % | 5.3 % | 3.9 % | 3.4 % | – |
| $\mu D_v$ ratio, $|v|<0.25$ / $>2$ m/s | 7.4 / 0.7 | 8.1 / 0.8 | 9.6 / 1.0 | 10.3 / 1.9 | 1 |
| roll-pitch / yaw damping ratio | 1.0 / 2.7 | 1.1 / 4.1 | 1.4 / 6.0 | 1.7 / 16 | 1 |
| 3 s report open loop, final | 1.05 m | 0.46 m | 2.53 m | 0.30 m | – |
| controller, feedforward / reference | 0.113 / 0.045 m | 0.115 / 0.044 m | 0.117 / 0.045 m | 0.116 / 0.045 m | – |

Run folders: `experiments/quadrotor/train_runs/13-09-19-32_*finalF{1,5}-stage2*actpen30*` and `13-09-23-06_*HARDV3LIN*linF{1,5}-stage2*`. Known remaining errors: translational damping too high below 0.5 m/s at both noise levels and both damping laws (errors-in-variables residue), yaw damping, and at $\sigma=0.1$ a pose-dependent leak of the $g$ residual between training poses that drives the hover-phase drift of the open-loop plots.

## 12. Source map

- Readable GP model and $SE(3)$ dynamics: [`network.py`](network.py)
- Exact local copy of the utility GP implementation: [`gp_model.py`](gp_model.py)
- Private fixed-feature setup and variational plumbing: [`_backend.py`](_backend.py)
- Trajectory and likelihood equations: [`losses.py`](losses.py)
- Second-order Lie-IMEX solver: [`integrator.py`](integrator.py)
- Dataset loading and window preparation: [`data.py`](data.py)
- Strict config loading and validation: [`config.py`](config.py)
- Checkpoint serialization: [`checkpoints.py`](checkpoints.py)
- Experiment folders and history entries: [`experiment.py`](experiment.py)
- Independent JAX training loop: [`train.py`](train.py)
- Current configs: [`13-09-2026-03-00/01`](../configs/13-09-2026-03-01_ph_gp_lie_imex.yaml) (0.1 noise), `03-10/11` (0.5 noise), `04-xx` (linear damping)
- Chain runner and diagnostics: `experiments/quadrotor/.chain_logs/`
- Dataset generator: [`envs/pybullet_quadrotor_se3/datagen/generate_dataset.py`](../../../../envs/pybullet_quadrotor_se3/datagen/generate_dataset.py) with [`config.yaml`](../../../../envs/pybullet_quadrotor_se3/datagen/config.yaml)
- 54-page single-model report (pure JAX; see [`../comparision/README.md`](../comparision/README.md)): [`../comparision/generate_report.py`](../comparision/generate_report.py), with `report_evaluation.py`, `report_figures.py`, `report_controller.py`
- Utility source that `gp_model.py` must match: [`../../../utils/JAX/gp_model.py`](../../../utils/JAX/gp_model.py)

This package does not import implementation code from another quadrotor model. Its only allowed parent-level runtime dependency is the report generator.

# PH-GP-LieIMEX quadrotor model — multi-horizon variant (`ph_gp_lie_imex_1sec`)

This folder is the main `ph_gp_lie_imex` package plus a multi-horizon training term (section 7.6). The model, GP, levels, optimizer and report are identical; only `data.py`, `integrator.py`, `losses.py`, `config.py` and `train.py` differ.

## 1. Short answer

- This is a JAX port-Hamiltonian model on $SE(3)$.
- Its six unknown physical functions are finite-feature variational Gaussian processes (GPs): $M_1^{-1}$, $M_2^{-1}$, $D_v$, $D_\omega$, $V$, and $g$.
- A sample has the state $s=[x,\operatorname{vec}(R),v_b,\omega_b,u]\in\mathbb R^{22}$.
- The current config uses a model step of $h=0.01\,\mathrm{s}$ and $K=5$ Lie-IMEX transitions per rollout, so each rollout covers $T=Kh=0.05\,\mathrm{s}$.
- Main training uses $5000$ mini-batch Adam updates with batch size $256$.
- Mass pretraining is disabled in the canonical config (`mass_pretrain_steps: 0`); $M_1^{-1}$ starts at $I_3$ with a learned level $\lambda_0$.
- The configured objective is the $SE(3)$ negative log-likelihood with one learned noise scale per state block, plus the variational KL spread over all training transitions (a minibatch negative ELBO). Trajectory MSE remains available as an option.
- Every operator has a learned constant level (no vehicle mass, inertia, or damping is read into the model); the GPs learn the state dependence around those levels.
- The drift is not a separate GP named `drift`. It is assembled inside `SampledSE3HamODE.vector_field()` from the Hamiltonian gradients, rigid-body coupling, learned potential, and dissipation. Only the $g(x,R)u$ term is the control contribution.

The readable model is in [`network.py`, lines 46–197](network.py#L46-L197), the second-order Lie-IMEX solver is in [`integrator.py`, lines 82–136](integrator.py#L82-L136), and the independent JAX training loop is in [`train.py`, lines 248–500](train.py#L248-L500).

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

The six samples are created together in [`_backend.py`, lines 574–589](_backend.py#L574-L589). One coherent draw defines the whole model for one optimizer update and its complete rollout; see [`train.py`, lines 199–233](train.py#L199-L233) and [`train.py`, lines 367–374](train.py#L367-L374). Evaluation passes no random key, so it uses the six posterior means.

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

The learned levels add $1+6+6+6+24=43$ point-estimate scalars ($\lambda_0$ for $M_1$, six each for $M_2$, $D_v$, $D_\omega$, and the $6\times4$ level of $g$), and the configured `se3-nll` data fit adds four likelihood log-standard-deviations, giving $103{,}667$ trainable scalar values. The exact backend keeps its random feature objects fixed and trains only these variational arrays; initialization is in [`_backend.py`, lines 509–543](_backend.py#L509-L543).

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

The positive diagonal makes $L$ invertible, so $M_2^{-1}$ is symmetric positive definite without a floor. No vehicle inertia is read into the model. The config value `gp.inverse_mass_2_initial_value` $=c$ sets the start: $\lambda_{1,2,3}=\tfrac12\log c$, $\lambda_{4,5,6}=0$, so $M_2^{-1}=cI_3$ before training (currently $c=1000$). Far from the data $f\to0$, and the learned inertia estimate is $\hat J=\left(L(\lambda)L(\lambda)^T\right)^{-1}$. See [`network.py`, lines 68–70](network.py#L68-L70) and [`_backend.py`, lines 509–520](_backend.py#L509-L520).

### 4.3 Translational dissipation GP $D_v$

The GP emits $f(v_b)\in\mathbb R^6$ from the body velocity. With six learned levels $\lambda\in\mathbb R^6$ (`Dv/level`, point estimates, not in the KL) it builds the same lower-triangular factor as $M_2^{-1}$:

$$L_v(v_b)=\begin{bmatrix}e^{\lambda_1+f_1}&0&0\\ \lambda_4+f_4&e^{\lambda_2+f_2}&0\\ \lambda_5+f_5&\lambda_6+f_6&e^{\lambda_3+f_3}\end{bmatrix},\qquad D_v(v_b)=L_v(v_b)L_v(v_b)^T\succeq0.$$

No mass, damping coefficient, or damping law is read into the model; all velocity dependence is learned by $f(v_b)$. The config value `gp.dissipation_v_initial_value` $=c_v$ sets the start $D_v=c_vI_3$ (currently $c_v=1$, so the initial damping rate $M_1^{-1}D_v$ is $1\,\mathrm{s^{-1}}$ with $M_1^{-1}=I_3$). The shared factor builder is [`network.py`, lines 43–54](network.py#L43-L54); the matrix is assembled in [`network.py`, lines 72–75](network.py#L72-L75). `model.dissipation_enabled` still gates the matrix.

### 4.4 Rotational dissipation GP $D_\omega$

Identical construction with input $\omega_b$ and levels `Dw/level`:

$$D_\omega(\omega_b)=L_\omega(\omega_b)L_\omega(\omega_b)^T\succeq0.$$

The start is $D_\omega=c_\omega I_3$ from `gp.dissipation_w_initial_value` (currently $c_\omega=10^{-3}$, so the initial rate $M_2^{-1}D_\omega$ is $1\,\mathrm{s^{-1}}$ with $M_2^{-1}=1000I_3$). See [`network.py`, lines 77–80](network.py#L77-L80).

### 4.5 Potential GP $V$

The potential is a position-only GP (`potential_include_rotation: false`, so the adapter feeds $[\operatorname{vec}(I_3),x]$ and the map is Matérn on $x$):

$$V(x)=\phi_V(x)^Tw_V.$$

There is no separately added $mgz$ term and no learned level: a constant in $V$ is invisible to the dynamics, and gravity is learned inside the GP as the linear part of $V$. Because $\partial V/\partial R=0$, the potential produces no torque. The evaluation is in [`network.py`, lines 82–83](network.py#L82-L83) and the adapter in [`network.py`, lines 37–40](network.py#L37-L40).

### 4.6 Control-map GP $g$

The raw $24$-vector is reshaped into a $6\times4$ matrix and added to a learned constant level $\Lambda_0\in\mathbb R^{6\times4}$ (stored as `g/level`, point estimate, not in the KL):

$$g(x,R)=\Lambda_0+\operatorname{reshape}_{6\times4}\!\left(\phi_g(x,R)^TW_g\right).$$

No selection matrix or actuator geometry is typed in: $\Lambda_0$ is initialised at $0$ (no actuation before training) and learned from data, and the GP learns the pose-dependent residual around it. Far from the data the GP term vanishes and $g\to\Lambda_0$. See [`network.py`, lines 84–88](network.py#L84-L88). Because $\Lambda_0$ is free, the data determine only the products $M_1^{-1}g$, $M_2^{-1}g$, $M_1^{-1}\nabla V$, $M_1^{-1}D_v$, $M_2^{-1}D_\omega$; reports compare these gauge-invariant operators with the ground truth. If `direct_control_map: true`, the level is omitted and $g$ is the GP output alone (ablation).

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

## 7. Training loss

### 7.1 Configured $SE(3)$ negative log-likelihood

Let $(x,R,v_b,\omega_b)$ be an observed target state and $(\hat x,\hat R,\hat v_b,\hat\omega_b)$ its rollout prediction. Each of the four state blocks $a\in\{x,R,v,\omega\}$ has one trainable noise scale $\sigma_a$ (stored as `likelihood/log_sigma_*`, initialised at `gp.initial_observation_sigma` $=0.1$). For a Euclidean three-vector block,

$$\mathcal L_{\mathrm{NLL},a}=\frac{1}{2\sigma_a^2}\operatorname{mean}\!\left(\lVert y_a-\hat y_a\rVert_2^2\right)+3\log\sigma_a+\frac32\log(2\pi),$$

and for attitude the squared norm is replaced by the squared geodesic angle

$$\theta(R,\hat R)=\cos^{-1}\!\left(\operatorname{clip}\!\left(\frac{\operatorname{tr}(R\hat R^T)-1}{2},-1+10^{-6},1-10^{-6}\right)\right).$$

The data-fit term is

$$\mathcal L_{\mathrm{NLL}}=\mathcal L_{\mathrm{NLL},x}+\mathcal L_{\mathrm{NLL},R}+\mathcal L_{\mathrm{NLL},v}+\mathcal L_{\mathrm{NLL},\omega}.$$

Because each block is weighted by its own $1/\sigma_a^2$, the four blocks are balanced automatically, and the learned $\sigma_a$ report the residual scale of each block. If `gp.fixed_observation_sigma` is set, the four $\sigma_a$ are frozen at that value instead. Implementation: [`losses.py`, lines 69–179](losses.py#L69-L179); selection in [`train.py`](train.py) via `gp.data_fit`. Controls in channels $18{:}22$ do not contribute.

### 7.2 Variational KL term

The prior for every GP weight is an independent standard normal, $p(W)=\mathcal N(0,I)$. The KL divergence is

$$\mathcal L_{\mathrm{KL}}=\frac12\sum_a\sum_j\left(e^{2S_{a,j}}+M_{a,j}^2-1-2S_{a,j}\right).$$

It is computed across all six GPs in `_backend.py` (`kl_divergence`). The learned levels and the likelihood scales are point estimates and are not part of the KL.

### 7.3 Complete main-training objective

At optimizer update $n$ the KL coefficient is warmed up linearly to `gp.kl_beta_max` $=1$:

$$\beta_n=\min\!\left(\frac{n}{1000},1\right).$$

The KL is spread over every training transition, $N=B_{\mathrm{train}}K=2880\times5=14400$, so the objective is a minibatch estimate of the negative ELBO:

$$\boxed{\mathcal J_n=\mathcal L_{\mathrm{NLL}}+\frac{\beta_n}{14400}\mathcal L_{\mathrm{KL}}}.$$

The code computes $N$ from the loaded training tensor, so this number follows the dataset. See `train.py` (`number_of_observations` and `objective`).

### 7.6 Multi-horizon term (this variant only)

Short windows cannot separate an actuation gain from a damping rate whose time constant is longer than the window. This variant adds a curriculum of long rollouts on a few full flights:

$$\mathcal J_n=\mathcal L_{\mathrm{NLL}}^{\text{windows}}+\frac{\beta_n}{N}\mathcal L_{\mathrm{KL}}+\lambda\,r_n\,\frac{\sum_i m_i\,\ell_i}{\max(1,\sum_i m_i)} .$$

- $\ell_i$ is the per-flight $SE(3)$ NLL of a rollout of horizon $H$ on flight $i$ (`losses.py`, `per_window_se3_nll`), with the likelihood scales $\sigma_a$ held fixed (`stop_gradient`) so the long term cannot loosen the window likelihood.
- $H$ follows `training.long_horizon_schedule`, a list of `[start_step, seconds]`: the term is off before the first start; the canonical schedule is $0.25$ s from step 2000, $0.5$ s from 5000, $1$ s from 10000.
- $\lambda$ is `training.long_horizon_weight` and $r_n=\min(1,(n-n_{\text{start}})/1000)$ ramps the term in after every stage start.
- `training.long_horizon_flights` flights are drawn at random every step from the noisy full training flights (`data.py` now returns the full flights; noise is added to the flights first, rotations re-projected, and the windows are cut from the same noisy flights).
- The rollout is `integrator.rollout_control_sequence_bounded`: after every Lie-IMEX step the body twist is clipped to $\pm$`training.long_horizon_twist_bound` (NaN/inf replaced), so the forward pass cannot overflow; every `training.long_horizon_truncation_steps` transitions the state passes through `stop_gradient` (multiple shooting), so backpropagation spans at most that many steps; $m_i=1$ only for flights that never touched the bound and stayed finite.
- Every evaluation also logs a 1 s open-loop rollout of the clean test flights (`training.long_horizon_test_seconds`): position RMS, its ratio to the ground truth's own excursion, and the masked fraction (`test_long_position_rms`, `test_long_position_relative`, `test_long_masked_fraction` in `training_stats.npz` and in the log line).

Canonical values: 8 flights, weight $0.2$, truncation 25 steps, twist bound $50$, test horizon $1$ s. Measured on the 20000-step run: no masked window at any stage, window test loss $7.5\times10^{-4}$, 1 s test position RMS $3.7$ cm (relative $0.06$).

### 7.4 Mass-pretraining loss

Before trajectory training, the inverse masses can be fitted toward the vehicle values:

$$\mathcal L_{\mathrm{mass}}=\operatorname{mean}\!\left(\lVert M_1^{-1}(x)-I_3/m\rVert_F^2\right)+\operatorname{mean}\!\left(\lVert M_2^{-1}(R)-J^{-1}\rVert_F^2\right).$$

The canonical config sets `mass_pretrain_steps: 0`, so this stage is skipped: the target would require the true mass and inertia, which the model is not given. The code path remains in [`train.py`, lines 147–181](train.py#L147-L181) for the optional non-zero setting.

### 7.5 Optional trajectory MSE

If both `training.loss` and `gp.data_fit` are set to `trajectory-mse`, the data fit becomes

$$\mathcal L_{\mathrm{traj}}=\operatorname{mean}\!\left((x-\hat x)^2\right)+\operatorname{mean}\!\left((v_b-\hat v_b)^2\right)+\operatorname{mean}\!\left((\omega_b-\hat\omega_b)^2\right)+\operatorname{mean}\!\left(\theta(R,\hat R)^2\right),$$

which is the NLL with every $\sigma_a^2$ frozen at $1/2$. Its components are in [`losses.py`, lines 33–48](losses.py#L33-L48) and are still logged as `train`/`test` metrics in every run for comparability.

## 8. Dataset used by the current config

### 8.1 Simulator and timing

- The config points to the clean `D0-CF2P-PIDplusEXC100Hz-PYB1000Hz` dataset; see [`09-09-2026-23-13_ph_gp_lie_imex.yaml`, lines 14–21](../configs/09-09-2026-23-13_ph_gp_lie_imex.yaml#L14-L21).
- It contains contact-free Crazyflie `CF2P` flights simulated with PyBullet physics.
- Native physics runs at $1000\,\mathrm{Hz}$ and stored controls/states run at $100\,\mathrm{Hz}$.
- Therefore the stored and model step is $h=0.01\,\mathrm{s}$, with $10$ native physics substeps per stored transition.
- Each flight lasts $1\,\mathrm{s}$ and contains $100$ transitions or $101$ stored points.
- Nonlinear translational and angular damping use coefficient $c=0.5$.

The trainer reads the time array and computes $h$ from its median difference in [`data.py`, lines 72–81](data.py#L72-L81).

### 8.2 Train and test trajectories

- Training contains $144$ full flights.
- Testing contains $18$ full flights.
- Half of the training flights use original PID input and half use PID plus bounded multisine excitation.
- The clean full-trajectory arrays have shapes $144\times101\times22$ and $18\times101\times22$.
- The stored window arrays have shapes $6\times2880\times22$ and $6\times360\times22$.

The loader requires the `x`, `test_x`, and `t` entries in [`data.py`, lines 44–60](data.py#L44-L60).

### 8.3 Controller and excitation

The excited half adds an independently phased two-frequency signal to each wrench component:

$$\Delta u_j(t)=A_j\left(0.65\sin(2\pi f_{j1}t+\varphi_{j1})+0.35\sin(2\pi f_{j2}t+\varphi_{j2})\right).$$

The requested maximum amplitudes in wrench order $[T,\tau_x,\tau_y,\tau_z]$ are

$$A=[0.04,0.001,0.001,0.0008].$$

The requested wrench is passed through the `CF2P` mixer and scaled when necessary to keep rotor forces feasible. The stored control is the actual post-allocation wrench.

### 8.4 Physical parameters and quality checks

The vehicle values read from dataset metadata are

$$m=0.027\,\mathrm{kg},$$

$$J=\operatorname{diag}\!\left(2.3951\times10^{-5},2.3951\times10^{-5},3.2347\times10^{-5}\right)\,\mathrm{kg\,m^2},$$

$$g=9.8\,\mathrm{m/s^2}.$$

The model extracts mass, diagonal inertia, gravity, and damping from dataset settings, with explicit fallback values, in [`_backend.py`, lines 243–281](_backend.py#L243-L281). The dataset records zero contact events. Its maximum training rotation orthogonality error is approximately $5.49\times10^{-16}$ and maximum absolute determinant error is approximately $4.44\times10^{-16}$.

### 8.5 Observation noise

The stored dataset is clean. The config sets `observation_noise_override: 0.01`, so the loader adds Gaussian noise to the first $18$ channels of every **training** state (the test set and the four control channels are never noised):

$$\widetilde s_{0:18}=s_{0:18}+\epsilon,\qquad \epsilon\sim\mathcal N(0,\sigma^2I),\qquad \sigma=0.01 .$$

Entry-wise noise takes the $3\times3$ rotation off $SO(3)$ (at $\sigma=0.01$ the worst case has $\lVert\widetilde R\widetilde R^T-I\rVert\approx0.1$). The loader therefore projects every noisy rotation back onto $SO(3)$ by row Gram--Schmidt, the same rule the attitude loss uses:

$$r_1=\frac{\widetilde R_{1,:}}{\lVert\widetilde R_{1,:}\rVert},\qquad r_3=\frac{r_1\times\widetilde R_{2,:}}{\lVert r_1\times\widetilde R_{2,:}\rVert},\qquad r_2=r_3\times r_1 .$$

The result is a true rotation that differs from the clean one by a small angle (mean $0.9^\circ$, maximum $2.5^\circ$ at $\sigma=0.01$), so the perturbation is a genuine attitude error. Noise and projection are in [`data.py`](data.py) (`load_dataset`, `_project_rotations`). The noise level is recorded in `metadata.json` as `observation_noise_override`.

### 8.6 Windows used by this model

The current config uses:

- $6$ points per window;
- $K=5$ predicted transitions;
- $h=0.01\,\mathrm{s}$ per transition;
- $T=Kh=0.05\,\mathrm{s}$ per rollout;
- stride $5$ stored transitions;
- no overlapping transitions between adjacent windows.

Each one-second flight supplies $100/5=20$ windows. Therefore,

$$B_{\mathrm{train}}=144\times20=2880,$$

$$B_{\mathrm{test}}=18\times20=360.$$

The tensors supplied to JAX are

$$S_{\mathrm{train}}\in\mathbb R^{6\times2880\times22},\qquad S_{\mathrm{test}}\in\mathbb R^{6\times360\times22}.$$

Window construction and accepted tensor layouts are in [`data.py`, lines 15–41](data.py#L15-L41). Returned arrays are converted to `float32` in [`data.py`, lines 72–81](data.py#L72-L81).

## 9. Exact current training configuration

The canonical config is [`09-09-2026-23-13_ph_gp_lie_imex.yaml`](../configs/09-09-2026-23-13_ph_gp_lie_imex.yaml). Important values are:

| Configuration | Current value |
|---|---:|
| Framework | JAX |
| Numeric type | `float32` |
| Device | GPU $0$, required |
| Model | `ph_gp_lie_imex` |
| Solver | second-order Lie-IMEX |
| Random seed | $0$ |
| Main optimizer | Adam |
| Learning rate | $10^{-3}$ |
| Adam $\beta_1$ | $0.9$ |
| Adam $\beta_2$ | $0.999$ |
| Adam $\epsilon$ | $10^{-8}$ |
| Main optimizer updates | $5000$ |
| Mini-batch windows | $256$ |
| Evaluation interval | $500$ updates |
| Checkpoint interval | $500$ updates |
| Gradient clipping | `None` |
| Weight decay | $0$ |
| Data-fit loss | $SE(3)$ NLL, learned $\sigma_a$ (initial $0.1$) |
| Loss scale | $1$ |
| Maximum KL weight | $1$ (KL divided by all $14400$ training transitions) |
| KL warm-up | $1000$ updates |
| Observation noise (training only) | $\sigma=0.01$, rotations re-projected |
| Initial $M_2^{-1}$, $D_v$, $D_\omega$ | $1000I_3$, $I_3$, $10^{-3}I_3$ |
| Posterior initial log standard deviation | $-5$ |
| Mass-pretraining updates | $0$ |
| Long-horizon term | 8 flights, weight $0.2$, schedule $[2000{:}0.25\,\mathrm s,\ 5000{:}0.5\,\mathrm s,\ 10000{:}1\,\mathrm s]$, truncation 25, twist bound 50 |
| Mass-pretraining learning rate | $10^{-3}$ |
| Mass-pretraining state samples | $4096$ |
| Matérn smoothness $\nu$ | $2.5$ |
| Matérn length scale $\ell_M$ | $1$ |
| Period $P$ | $2\pi$ |
| Periodic length scale $\ell_P$ | $0.5$ |
| Periodic harmonics $m_{\max}$ | $5$ |
| Periodic dimension $d_p$ | $0$ |
| Integrator step | $h=0.01\,\mathrm{s}$ |
| Integration steps per rollout | $K=5$ |
| Duration per rollout | $0.05\,\mathrm{s}$ |
| Recorded controls | time-varying $u_{k+1}$ |
| Initial checkpoint | `None` |
| Resume checkpoint | `None` |
| Optional model subnetwork | `None` |
| Optional GP mean subnetwork | `None` |
| Automatic report | enabled |
| Report length | exactly $47$ pages |

Every argument consumed by training is required in YAML, including every GP argument. The required-path list is enforced in [`config.py`, lines 24–112](config.py#L24-L112). Config names must follow `DD-MM-YYYY-HH-MM_<model>.yaml`; this is validated in [`config.py`, lines 132–142](config.py#L132-L142).

Run training from the project root with:

```bash
python -m src.models.SE3_Quadrotor.ph_gp_lie_imex.train \
  --config src/models/SE3_Quadrotor/configs/09-09-2026-23-13_ph_gp_lie_imex.yaml
```

The `--config` argument is mandatory in [`train.py`, lines 35–38](train.py#L35-L38). A successful run saves checkpoints and report data, appends experiment history, and invokes the single report generator in [`train.py`, lines 429–484](train.py#L429-L484).

The meanings of “step” are:

- simulator physics step: $0.001\,\mathrm{s}$;
- stored sample and model integration step: $h=0.01\,\mathrm{s}$;
- training rollout: $K=5$ model steps, or $0.05\,\mathrm{s}$;
- main training length: $5000$ optimizer updates;
- separate initialization stage: none (mass pretraining disabled).

## 10. Source map

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
- Current complete config: [`09-09-2026-23-13_ph_gp_lie_imex.yaml`](../configs/09-09-2026-23-13_ph_gp_lie_imex.yaml)
- 47-page single-model report (pure JAX; see [`../comparision/README.md`](../comparision/README.md)): [`../comparision/generate_report.py`](../comparision/generate_report.py), with `report_evaluation.py`, `report_figures.py`, `report_controller.py`
- Utility source that `gp_model.py` must match: [`../../../utils/JAX/gp_model.py`](../../../utils/JAX/gp_model.py)

This package does not import implementation code from another quadrotor model. Its only allowed parent-level runtime dependency is the report generator.

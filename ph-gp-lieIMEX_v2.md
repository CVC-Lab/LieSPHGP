# PH-GP-LieIMEX-v2 quadrotor model

## 1. Short answer

- This model is a port-Hamiltonian model on $SE(3)$ whose unknown physical functions are represented by six finite-feature variational Gaussian processes (GPs), not ordinary multilayer perceptrons.
- The six GP subnetworks are $M_1^{-1}$, $M_2^{-1}$, $D_v$, $D_\omega$, $V$, and $g$.
- The state stored in one dataset sample is $s=[x,\operatorname{vec}(R),v_b,\omega_b,u]\in\mathbb R^{22}$.
- The reference v2 training run uses $h=0.01\,\mathrm{s}$, $K=10$ integration steps per rollout window, and $10{,}000$ full-batch Adam updates.
- Before the main training, the two inverse-mass GPs are fitted for $200$ additional mass-pretraining updates.
- The clean and noisy reference runners use trajectory MSE plus a KL regularizer. Although the shared trainer also supports an $SE(3)$ negative log-likelihood, that option is not used by these v2 reference runs.

The v2-specific architecture is implemented in [`model.py`, lines 1–39](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L1), its Lie-IMEX solver is in [`integrator.py`, lines 64–107](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/integrator.py#L64), and the v2 wrapper installs these functions into the shared trainer in [`train.py`, lines 88–114](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/train.py#L88).

## 2. State, control, and conventions

Each stored sample has $22$ channels:

| Block | Meaning | Dimension |
|---|---|---:|
| $x$ | world-frame position | $3$ |
| $R$ | body-to-world rotation matrix, flattened row-wise | $9$ |
| $v_b$ | body-frame linear velocity | $3$ |
| $\omega_b$ | body-frame angular velocity | $3$ |
| $u=[T,\tau_x,\tau_y,\tau_z]$ | applied thrust and body torques | $4$ |

Thus,

$$s=\begin{bmatrix}x\\\operatorname{vec}(R)\\v_b\\\omega_b\\u\end{bmatrix}\in\mathbb R^{22},\qquad R\in SO(3),\qquad u\in\mathbb R^4.$$

The dataset packing performs $v_b=R^T v_w$ and $\omega_b=R^T\omega_w$ before storing the sample; see [`expanded_excitation_h001_dataset.py`, lines 104–112](src/models/SE3_Quadrotor/comparision/expanded_excitation_h001_dataset.py#L104). The v2 vector field reads these exact channel blocks in [`model.py`, lines 301–312](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L301).

## 3. What a GP subnetwork means here

For each subnetwork $a$, the code constructs a fixed feature vector $\phi_a(z)$ and a trainable weight matrix $W_a$. Its raw output is

$$f_a(z)=\phi_a(z)^T W_a.$$

This multiplication is implemented in [`model.py`, lines 200–218](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L200).

The weight posterior is mean-field Gaussian:

$$q(W_a)=\mathcal N\!\left(M_a,\operatorname{diag}\!\left(e^{2S_a}\right)\right),$$

and a training sample is obtained by the reparameterization

$$W_a=M_a+e^{S_a}\odot\epsilon_a,\qquad \epsilon_a\sim\mathcal N(0,I).$$

The posterior sampling is in [`model.py`, lines 171–185](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L171). One coherent draw of all six GPs is used for an entire rollout and a new draw is made at every optimizer update; see [`train.py`, lines 331–360](src/models/SE3_Quadrotor/ph_gp_lieimex/train.py#L331) and [`train.py`, lines 687–705](src/models/SE3_Quadrotor/ph_gp_lieimex/train.py#L687).

### 3.1 Feature types and shared feature configuration

- Euclidean inputs use normalized Matérn-$5/2$ random Fourier features. For $z\in\mathbb R^d$,

  $$\bar z=\frac{z-c}{\sigma_z},\qquad \phi_E(z)=\sqrt{\frac{2}{F}}\cos\!\left(\Omega\frac{\bar z}{\ell}+b\right).$$

  The frequency, phase, normalization, and feature evaluation are defined in [`features.py`, lines 22–61](src/models/SE3_Quadrotor/ph_gp_lieimex/features.py#L22) and [`features.py`, lines 181–186](src/models/SE3_Quadrotor/ph_gp_lieimex/features.py#L181).

- Rotation inputs use a truncated Laplace–Beltrami spectral Matérn-$5/2$ feature map on $SO(3)$. The spectral coefficient for degree $l$ is proportional to

  $$c_l=\left(\frac{2\nu}{\ell^2}+l(l+1)\right)^{-(\nu+3/2)},\qquad \nu=2.5.$$

  This construction is in [`features.py`, lines 101–136](src/models/SE3_Quadrotor/ph_gp_lieimex/features.py#L101) and [`features.py`, lines 203–219](src/models/SE3_Quadrotor/ph_gp_lieimex/features.py#L203).

- An $SE(3)$ feature is the product of a position feature and a rotation feature:

  $$\phi_{SE(3)}(x,R)=\phi_E(x)\otimes\phi_{SO(3)}(R).$$

  See [`features.py`, lines 222–235](src/models/SE3_Quadrotor/ph_gp_lieimex/features.py#L222).

- Euclidean inputs are centered and divided by training-set standard deviations. Position, linear-velocity, and angular-velocity statistics are computed only from the training tensor; see [`model.py`, lines 44–54](src/models/SE3_Quadrotor/ph_gp_lieimex/model.py#L44).

- Every positive kernel length scale is parameterized as

  $$\ell=\operatorname{softplus}(\rho)+0.05.$$

  The floor is defined in [`features.py`, lines 18–20](src/models/SE3_Quadrotor/ph_gp_lieimex/features.py#L18), and the transformation is in [`features.py`, lines 162–170](src/models/SE3_Quadrotor/ph_gp_lieimex/features.py#L162).

## 4. Every subnetwork: input, raw output, physical output, and parameters

The v2 feature-bank defaults are defined in [`model.py`, lines 42–113](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L42). The resulting feature dimensions are computed in [`model.py`, lines 116–133](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L116), and the GP output dimensions are declared in [`model.py`, lines 32–39](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L32).

| GP | Input | Feature configuration | Feature dimension | Raw GP output | Physical output | Weight shape | Length scales |
|---|---|---|---:|---:|---|---:|---:|
| $M_1$ | $x\in\mathbb R^3$ | Euclidean Matérn RFF, requested features $220$ | $220$ | scalar | $M_1^{-1}(x)\in\mathbb R^{3\times3}$ | $220\times1$ | $1$ |
| $M_2$ | $R\in SO(3)$ as $9$ values | $SO(3)$ spectral Matérn, budget $220$ | $165$ | $6$ values | $M_2^{-1}(R)\in\mathbb R^{3\times3}$ | $165\times6$ | $1$ |
| $D_v$ | $v_b\in\mathbb R^3$ | Euclidean Matérn RFF, requested features $225$ | $225$ | $6$ values | $D_v(v_b)\in\mathbb R^{3\times3}$ | $225\times6$ | $1$ |
| $D_\omega$ | $\omega_b\in\mathbb R^3$ | Euclidean Matérn RFF, requested features $220$ | $220$ | $6$ values | $D_\omega(\omega_b)\in\mathbb R^{3\times3}$ | $220\times6$ | $1$ |
| $V$ | $(x,R)\in SE(3)$ | $20$ position RFFs times an $SO(3)$ budget of $11$ | $20\times10=200$ | scalar | potential energy $V(x,R)\in\mathbb R$ | $200\times1$ | $2$ |
| $g$ | $(x,R)\in SE(3)$ | $20$ position RFFs times an $SO(3)$ budget of $11$ | $20\times10=200$ | $24$ values | input map $g(x,R)\in\mathbb R^{6\times4}$ | $200\times24$ | $2$ |

An $SO(3)$ budget is an upper budget, not necessarily the final dimension. Budget $220$ retains degrees $l=0,1,2,3,4$, so its dimension is $1^2+3^2+5^2+7^2+9^2=165$. Budget $11$ retains $l=0,1$, so its dimension is $1^2+3^2=10$. The budget rule is implemented in [`features.py`, lines 101–109](src/models/SE3_Quadrotor/ph_gp_lieimex/features.py#L101).

Each weight entry has both a posterior mean and a posterior log-standard-deviation. Including the learned raw length scales, the number of trainable GP parameters is

$$2(220\cdot1)+1+2(165\cdot6)+1+2(225\cdot6)+1+2(220\cdot6)+1+2(200\cdot1)+2+2(200\cdot24)+2=17{,}768.$$

This agrees with the completed run metadata. The initial posterior value is $\log\sigma_W=-5$, hence $\sigma_W=e^{-5}\approx0.00673795$; see [`model.py`, lines 34–35](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L34) and [`model.py`, lines 136–158](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L136). All length scales start at $1$ in the v2 initializer.

### 4.1 Translational inverse-mass GP $M_1$

The raw GP is a scalar $r_1(x)$. It is converted into a strictly positive isotropic inverse mass:

$$r_1(x)=\phi_{M_1}(x)^T w_{M_1},$$

$$\mu(x)=\epsilon_M+\left(\frac{1}{m}-\epsilon_M\right)e^{\alpha_M r_1(x)},$$

$$M_1^{-1}(x)=\mu(x)I_3,$$

where $\epsilon_M=0.01$, $\alpha_M=0.10$, and the dataset mass is $m=0.027\,\mathrm{kg}$. This is the main v2 mass change and is implemented in [`model.py`, lines 221–236](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L221). The exponential makes $\mu(x)>0$, while the isotropic form uses only one learned scalar instead of six raw factor entries.

### 4.2 Rotational inverse-mass GP $M_2$

The raw output $r_2(R)\in\mathbb R^6$ modifies a lower-triangular factor $L_2(R)$. The final matrix is

$$M_2^{-1}(R)=L_2(R)L_2(R)^T+\epsilon_M I_3.$$

Its physical reference is $J^{-1}$, where

$$J=\operatorname{diag}(2.3951\times10^{-5},\,2.3951\times10^{-5},\,3.2347\times10^{-5})\;\mathrm{kg\,m^2}.$$

The six raw outputs control three positive diagonal entries through exponentials and three lower-triangular off-diagonal entries. The generic factor construction is in [`model.py`, lines 261–287](src/models/SE3_Quadrotor/ph_gp_lieimex/model.py#L261), and v2 applies it to $M_2^{-1}$ in [`model.py`, lines 239–246](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L239). The mass residual scale is $0.10$ and the eigenvalue floor is $0.01$.

### 4.3 Translational dissipation GP $D_v$

This GP receives only $v_b$, not $(x,v_b)$. Its physical diagonal reference is

$$D_{v,0}(v_b)=m c\left(1+\lVert v_b\rVert_2\right)I_3,$$

with $c=0.5$. Its six raw outputs form a lower-triangular $L_v$, and

$$D_v(v_b)=L_v(v_b)L_v(v_b)^T\succeq0.$$

The residual-factor scale is $0.25$ and the eigenvalue floor is $0$. See [`model.py`, lines 249–260](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L249).

### 4.4 Rotational dissipation GP $D_\omega$

This GP receives only $\omega_b$, not $(R,\omega_b)$. Its physical diagonal reference is

$$D_{\omega,0}(\omega_b)=c\left(1+\lVert\omega_b\rVert_2\right)J,$$

and its physical output is

$$D_\omega(\omega_b)=L_\omega(\omega_b)L_\omega(\omega_b)^T\succeq0.$$

It also uses residual-factor scale $0.25$ and zero eigenvalue floor. See [`model.py`, lines 263–273](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L263).

### 4.5 Potential GP $V$

The potential is learned directly from the complete pose:

$$V(x,R)=\phi_{SE(3)}(x,R)^T w_V.$$

There is no separately added $mgz$ term and no residual multiplier. Therefore the GP must learn gravity as part of the complete potential. This is implemented in [`model.py`, lines 276–278](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L276). This direct potential is the second main architectural change from the original PH-GP-LieIMEX model.

### 4.6 Control-map GP $g$

The raw $24$-vector is reshaped into a $6\times4$ residual matrix:

$$g(x,R)=g_0+0.10\,\operatorname{reshape}_{6\times4}\!\left(\phi_{SE(3)}(x,R)^T W_g\right),$$

where

$$g_0=\begin{bmatrix}0&0&0&0\\0&0&0&0\\1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&1\end{bmatrix}.$$

Thus the physical mean maps total thrust to body $z$ force and maps the three torque inputs directly to body torque. The learned GP supplies a pose-dependent residual. See [`model.py`, lines 281–286](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L281).

## 5. Port-Hamiltonian mathematics

### 5.1 Hamiltonian

Let $p_v$ and $p_\omega$ be the translational and rotational body momenta. The Hamiltonian is total kinetic plus potential energy:

$$H(x,R,p_v,p_\omega)=\frac12p_v^TM_1^{-1}(x)p_v+\frac12p_\omega^TM_2^{-1}(R)p_\omega+V(x,R).$$

The observed body velocities and momenta are related by

$$v_b=M_1^{-1}(x)p_v,\qquad \omega_b=M_2^{-1}(R)p_\omega,$$

so the code reconstructs momenta with

$$p_v=M_1(x)v_b,\qquad p_\omega=M_2(R)\omega_b.$$

The Hamiltonian is implemented in [`model.py`, lines 289–298](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L289), and the momentum reconstruction is in [`model.py`, lines 308–313](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L308).

### 5.2 Pose kinematics

Define

$$H_x=\frac{\partial H}{\partial x},\qquad H_R=\frac{\partial H}{\partial R},\qquad H_{p_v}=\frac{\partial H}{\partial p_v},\qquad H_{p_\omega}=\frac{\partial H}{\partial p_\omega}.$$

Then the pose equations used by the model are

$$\dot x=R H_{p_v},$$

$$\dot R=R\widehat{H_{p_\omega}},$$

where $\widehat{a}$ is the skew-symmetric matrix satisfying $\widehat{a}b=a\times b$. The code computes these equations in [`model.py`, lines 313–318](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L313).

### 5.3 Momentum equations

Let the learned generalized wrench be

$$\begin{bmatrix}f_u\\\tau_u\end{bmatrix}=g(x,R)u.$$

The body-momentum dynamics are

$$\dot p_v=p_v\times H_{p_\omega}-R^TH_x-D_v(v_b)H_{p_v}+f_u,$$

$$\dot p_\omega=p_\omega\times H_{p_\omega}+p_v\times H_{p_v}+\sum_{i=1}^{3}R_{i,:}\times(H_R)_{i,:}-D_\omega(\omega_b)H_{p_\omega}+\tau_u.$$

These equations are implemented directly in [`model.py`, lines 319–330](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L319).

Because training data are stored as velocities rather than momenta, the vector field applies the product rule:

$$\dot v_b=M_1^{-1}\dot p_v+\dot M_1^{-1}p_v,$$

$$\dot\omega_b=M_2^{-1}\dot p_\omega+\dot M_2^{-1}p_\omega.$$

The directional matrix derivatives are calculated with JAX Jacobian-vector products in [`model.py`, lines 331–340](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L331).

### 5.4 Why this is port-Hamiltonian

The cross-product and pose-coupling terms exchange energy internally. The positive-semidefinite dissipation matrices remove energy, while $g(x,R)u$ supplies external power. Ignoring numerical integration error, the energy balance has the form

$$\dot H=-H_{p_v}^TD_vH_{p_v}-H_{p_\omega}^TD_\omega H_{p_\omega}+\begin{bmatrix}H_{p_v}\\H_{p_\omega}\end{bmatrix}^{T}g(x,R)u.$$

Therefore, when $u=0$,

$$\dot H\le0,$$

because $D_v\succeq0$ and $D_\omega\succeq0$. This physical sign constraint comes from the $LL^T$ parameterizations.

## 6. Lie-IMEX-v2 integration

The solver treats nonlinear damping implicitly and the remaining vector field explicitly. Let

$$\xi=\begin{bmatrix}v_b\\\omega_b\end{bmatrix},\qquad K(s)=\operatorname{blkdiag}\!\left(M_1^{-1}D_v,\,M_2^{-1}D_\omega\right).$$

The effective damping blocks are formed in [`model.py`, lines 346–355](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L346). If the complete velocity derivative is written as

$$\dot\xi=F_{\mathrm{exp}}(s,u)-K(s)\xi,$$

then the predictor solves

$$\left(I+hK_n\right)\xi^*=\xi_n+hF_{\mathrm{exp},n}.$$

The predicted pose is

$$x^*=x_n+h\dot x_n,$$

$$R^*=R_n\operatorname{Exp}\!\left(h\widehat{\omega_n}\right).$$

These predictor operations are in [`integrator.py`, lines 70–87](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/integrator.py#L70).

At the predicted state, the corrector performs the second implicit solve. In the exact code notation, with $E_1=\dot\xi_n+K_n\xi_n$, $E_2=\dot\xi^*+K_*\xi^*$, and $d_1=K_n\xi_n$,

$$\left(I+\frac h2K_*\right)\xi_{n+1}=\xi_n+\frac h2\left(E_1+E_2-d_1\right).$$

The pose corrector is

$$x_{n+1}=x_n+\frac h2\left(\dot x_n+\dot x^*\right),$$

$$R_{n+1}=R_n\operatorname{Exp}\!\left(\frac12\left(h\widehat{\omega_n}+h\widehat{\omega^*}\right)\right).$$

The corrector is implemented in [`integrator.py`, lines 89–107](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/integrator.py#L89). Because rotation is updated through the exponential map, $R$ stays on $SO(3)$ up to floating-point error. Each transition uses two full vector-field evaluations and two auxiliary damping evaluations.

During rollout, the recorded control at the target state is inserted before each transition:

$$u_{k+1}\text{ drives }s_k\longrightarrow s_{k+1}.$$

This control sequence is applied in [`integrator.py`, lines 120–130](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/integrator.py#L120).

## 7. Training loss

### 7.1 Trajectory data-fit term used by the reference v2 runs

Let $(x,R,v_b,\omega_b)$ be the observed state and $(\hat x,\hat R,\hat v_b,\hat\omega_b)$ be the rollout. The four components are

$$\mathcal L_x=\operatorname{mean}\!\left(\lVert x-\hat x\rVert_{\mathrm{element}}^2\right),$$

$$\mathcal L_v=\operatorname{mean}\!\left(\lVert v_b-\hat v_b\rVert_{\mathrm{element}}^2\right),$$

$$\mathcal L_\omega=\operatorname{mean}\!\left(\lVert\omega_b-\hat\omega_b\rVert_{\mathrm{element}}^2\right),$$

$$\theta(R,\hat R)=\cos^{-1}\!\left(\frac{\operatorname{tr}(R\hat R^T)-1}{2}\right),\qquad \mathcal L_R=\operatorname{mean}\!\left(\theta(R,\hat R)^2\right).$$

The unscaled trajectory loss is

$$\mathcal L_{\mathrm{traj}}=\mathcal L_x+\mathcal L_v+\mathcal L_\omega+\mathcal L_R.$$

The implementation is in [`losses.py`, lines 27–42](src/models/SE3_Quadrotor/ph_nn_ode_lieimex_jax/losses.py#L27). It compares the $10$ predicted target points against the $10$ observed target points and ignores the four control channels; see [`train.py`, lines 331–360](src/models/SE3_Quadrotor/ph_gp_lieimex/train.py#L331).

### 7.2 Variational KL term

The prior for every random-feature weight is an independent standard normal, $p(W)=\mathcal N(0,I)$. For posterior means $M_a$ and log-standard-deviations $S_a$, the KL divergence is

$$\mathcal L_{\mathrm{KL}}=\frac12\sum_a\sum_j\left(e^{2S_{a,j}}+M_{a,j}^2-1-2S_{a,j}\right).$$

This is implemented in [`model.py`, lines 188–197](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py#L188). The kernel length scales are point estimates and are not included in this KL expression.

### 7.3 Complete main-training objective

At optimizer update $n$, the KL coefficient is linearly warmed up:

$$\beta_n=0.01\min\!\left(\frac{n}{1000},1\right).$$

There are $B=1440$ training windows and $K=10$ target transitions, so the trainer defines

$$N_{\mathrm{obs}}=BK=1440\times10=14{,}400.$$

The complete objective used in the reference v2 runs is

$$\boxed{\mathcal J_n=\mathcal L_{\mathrm{traj}}+\frac{\beta_n}{14{,}400}\mathcal L_{\mathrm{KL}}}.$$

The normalization and objective are in [`train.py`, lines 320–328](src/models/SE3_Quadrotor/ph_gp_lieimex/train.py#L320) and [`train.py`, lines 366–388](src/models/SE3_Quadrotor/ph_gp_lieimex/train.py#L366). The annealing schedule is applied in [`train.py`, lines 687–705](src/models/SE3_Quadrotor/ph_gp_lieimex/train.py#L687).

### 7.4 Mass-pretraining loss

Before trajectory training, the inverse masses are fitted toward the dataset vehicle parameters:

$$\mathcal L_{\mathrm{mass}}=\operatorname{mean}\!\left(\lVert M_1^{-1}(x)-I_3/m\rVert_F^2\right)+\operatorname{mean}\!\left(\lVert M_2^{-1}(R)-J^{-1}\rVert_F^2\right).$$

The pretraining uses up to $4096$ evenly spaced training states, $200$ Adam updates, and learning rate $10^{-3}$. See [`train.py`, lines 165–218](src/models/SE3_Quadrotor/ph_gp_lieimex/train.py#L165).

### 7.5 Optional loss supported by the trainer but not used here

The shared trainer can replace $\mathcal L_{\mathrm{traj}}$ with a four-block $SE(3)$ negative log-likelihood. Each Euclidean three-vector block has

$$\mathcal L_{\mathrm{NLL},a}=\frac{\gamma}{2\sigma_a^2}\operatorname{mean}\!\left(\lVert y_a-\hat y_a\rVert_2^2\right)+3\log\sigma_a+\frac32\log(2\pi),$$

and attitude uses the same expression with $\theta(R,\hat R)^2$. The four blocks are summed. This optional implementation is in [`likelihood.py`, lines 53–145](src/models/SE3_Quadrotor/ph_gp_lieimex/likelihood.py#L53). Both v2 launchers explicitly choose `trajectory-mse`, so this NLL is not part of the reference v2 objective; see [`run_clean_training.py`, lines 54–75](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/run_clean_training.py#L54) and [`run_noise_sweep.py`, lines 60–81](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/run_noise_sweep.py#L60).

## 8. How the dataset is generated

### 8.1 Simulator and timing

- The generator is [`tmp/aggressive_quadrotor_datagen.py`](tmp/aggressive_quadrotor_datagen.py).
- It uses one Crazyflie 2.x plus-frame quadrotor (`CF2P`) in Gym-PyBullet-Drones with `Physics.PYB`; see [`aggressive_quadrotor_datagen.py`, lines 137–150](tmp/aggressive_quadrotor_datagen.py#L137).
- PyBullet runs at $1000\,\mathrm{Hz}$, so its native physics step is $0.001\,\mathrm{s}$.
- PID actions and stored samples run at $100\,\mathrm{Hz}$, so the stored/model step is $h=0.01\,\mathrm{s}$.
- One stored transition therefore contains $10$ native Bullet substeps. These constants and the environment configuration are in [`fine_h001_dataset.py`, lines 33–45](src/models/SE3_Quadrotor/comparision/fine_h001_dataset.py#L33) and [`fine_h001_dataset.py`, lines 71–85](src/models/SE3_Quadrotor/comparision/fine_h001_dataset.py#L71).
- Every flight lasts $1\,\mathrm{s}$ and contains $100$ transitions or $101$ stored points.

### 8.2 Train/test trajectories

- Training uses condition seeds $300$ through $307$, with $18$ accepted flights per seed: $8\times18=144$ training flights.
- Testing uses seed $399$, with $18$ accepted flights.
- Therefore the full clean tensors have shapes $144\times101\times22$ for training and $18\times101\times22$ for testing.

The split constants are in [`aggressive_quadrotor_datagen.py`, lines 71–79](tmp/aggressive_quadrotor_datagen.py#L71), and split assembly is in [`aggressive_quadrotor_datagen.py`, lines 327–390](tmp/aggressive_quadrotor_datagen.py#L327).

### 8.3 Initial conditions, controller, and excitation

For every flight, the generator randomly selects:

- roll and pitch in $[-35^\circ,35^\circ]$ and yaw in $[-\pi,\pi]$;
- world-frame $x$ and $y$ in $[-0.6,0.6]\,\mathrm{m}$ and altitude in $[1.0,1.5]\,\mathrm{m}$;
- each initial world-velocity component in $[-1,1]\,\mathrm{m/s}$;
- each initial body-angular-velocity component in $[-3,3]\,\mathrm{rad/s}$.

These settings and samples are in [`aggressive_quadrotor_datagen.py`, lines 81–126](tmp/aggressive_quadrotor_datagen.py#L81).

Control is created as follows:

1. A half-gain `DSLPIDControl` tracks a new position/yaw target every $0.25\,\mathrm{s}$. The position offset is sampled within $\pm0.6\,\mathrm{m}$ and yaw within $\pm60^\circ$. The PID gain scaling is implemented in [`generate_current_pybullet_pid_dataset.py`, lines 100–120](src/models/SE3_Quadrotor/comparision/generate_current_pybullet_pid_dataset.py#L100).
2. A persistent three-component multisine wrench with frequencies from $1$ to $12\,\mathrm{Hz}$ is added throughout the flight.
3. The maximum requested excitation amplitudes are $[0.080\,\mathrm{N},\,0.002\,\mathrm{N\,m},\,0.002\,\mathrm{N\,m},\,0.0016\,\mathrm{N\,m}]$, because the aggressive generator doubles the base amplitudes $[0.040,0.001,0.001,0.0008]$; see [`aggressive_quadrotor_datagen.py`, lines 90–94](tmp/aggressive_quadrotor_datagen.py#L90) and [`expanded_excitation_h001_dataset.py`, lines 67–70](src/models/SE3_Quadrotor/comparision/expanded_excitation_h001_dataset.py#L67).
4. At every sample, the added wrench is converted to rotor forces and scaled down when necessary to keep every rotor force feasible. The actual post-allocation wrench, rather than the requested wrench, is stored; see [`aggressive_quadrotor_datagen.py`, lines 170–200](tmp/aggressive_quadrotor_datagen.py#L170).

The ground is removed, contact friction is disabled, and PyBullet nonlinear linear/angular damping is set to $c=0.5$; see [`damping_ablation_common.py`, lines 399–465](src/models/SE3_Quadrotor/comparision/damping_ablation_common.py#L399). The corresponding audited one-step velocity form is

$$z_{k+1}=z_k-\Delta t\,c\left(1+\lVert z_k\rVert_2\right)z_k,$$

as recorded in [`damping_ablation_common.py`, lines 616–618](src/models/SE3_Quadrotor/comparision/damping_ablation_common.py#L616).

### 8.4 Quality gate

A generated flight is rejected and regenerated if either

$$\max_t\operatorname{tilt}(R_t)>90^\circ$$

or

$$\min_t z_t<0.5\,\mathrm{m}.$$

Up to $20$ sub-seed attempts are allowed. The gate is defined and applied in [`aggressive_quadrotor_datagen.py`, lines 95–98](tmp/aggressive_quadrotor_datagen.py#L95) and [`aggressive_quadrotor_datagen.py`, lines 236–257](tmp/aggressive_quadrotor_datagen.py#L236). Any contact event or non-finite trajectory also causes failure; see [`aggressive_quadrotor_datagen.py`, lines 211–233](tmp/aggressive_quadrotor_datagen.py#L211).

### 8.5 Observation-noise variants

The clean dataset has no observation noise. Two additional training datasets use $\sigma_{\mathrm{obs}}=0.05$ and $\sigma_{\mathrm{obs}}=0.1$.

For the Euclidean channels $x$, $v_b$, and $\omega_b$,

$$\tilde y_j=y_j+\sigma_{\mathrm{obs}}\epsilon_j,\qquad \epsilon_j\sim\mathcal N(0,1).$$

For rotations,

$$\tilde R=R\operatorname{Exp}(\widehat\eta),\qquad \eta\sim\mathcal N(0,\sigma_{\mathrm{obs}}^2I_3).$$

Controls and time stamps are unchanged. Noise is added to complete trajectories before the $K=10$ windows are constructed, and the test data used for evaluation remain clean. See [`absolute_observation_noise_dataset.py`, lines 114–167](src/models/SE3_Quadrotor/jax_4model_comparison/absolute_observation_noise_dataset.py#L114) and [`aggressive_quadrotor_datagen.py`, lines 391–416](tmp/aggressive_quadrotor_datagen.py#L391).

### 8.6 Windows used by this model

The generator also stores an older six-point window view, but v2 deliberately rebuilds its training view from the complete one-second trajectories. It uses:

- $11$ points per window;
- $K=10$ transitions per window;
- $h=0.01\,\mathrm{s}$ per transition;
- $T=Kh=10(0.01)=0.10\,\mathrm{s}$ per rollout;
- stride $10$ stored steps;
- zero overlapping transitions; neighboring windows share only one boundary state.

Each flight supplies $100/10=10$ windows. Therefore,

$$B_{\mathrm{train}}=144\times10=1440,$$

$$B_{\mathrm{test}}=18\times10=180.$$

The tensors passed to training have shapes

$$S_{\mathrm{train}}\in\mathbb R^{11\times1440\times22},\qquad S_{\mathrm{test}}\in\mathbb R^{11\times180\times22}.$$

The window constants and validation are in [`reduced_input_k10_data.py`, lines 42–50](src/models/SE3_Quadrotor/comparision/reduced_input_k10_data.py#L42), while construction and shapes are in [`reduced_input_k10_data.py`, lines 128–180](src/models/SE3_Quadrotor/comparision/reduced_input_k10_data.py#L128). The v2 loader explicitly selects this derived view in [`train.py`, lines 19–25](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/train.py#L19).

## 9. Exact reference training configuration

| Configuration | Reference v2 value |
|---|---:|
| Numeric type | JAX `float32` |
| Random seed | $0$ |
| Main optimizer | Adam |
| Learning rate | $10^{-3}$ |
| Main optimizer updates | $10{,}000$ |
| Gradient clipping | disabled |
| Weight decay | $0$ |
| Data-fit loss | unscaled trajectory MSE |
| Maximum KL weight | $10^{-2}$ |
| KL warm-up | $1000$ updates |
| Posterior initial log standard deviation | $-5$ |
| Mass-pretraining updates | $200$ |
| Mass-pretraining learning rate | $10^{-3}$ |
| Mass-pretraining state samples | $4096$ |
| Integrator step | $h=0.01\,\mathrm{s}$ |
| Integration steps per training rollout | $K=10$ |
| Duration per training rollout | $0.10\,\mathrm{s}$ |
| Training batch | all $1440$ windows |
| Recorded controls | time-varying $u_{k+1}$ |
| Evaluation interval | $1000$ updates |
| Checkpoint interval | $1000$ updates |

These are fixed by the clean launcher in [`run_clean_training.py`, lines 19–28](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/run_clean_training.py#L19) and [`run_clean_training.py`, lines 54–75](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/run_clean_training.py#L54). The noise-sweep launcher uses the same optimization settings for both noise levels; see [`run_noise_sweep.py`, lines 19–29](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/run_noise_sweep.py#L19) and [`run_noise_sweep.py`, lines 60–81](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/run_noise_sweep.py#L60).

The important distinction between the different meanings of “steps” is:

- simulator physics step: $0.001\,\mathrm{s}$;
- stored sample and model integrator step: $h=0.01\,\mathrm{s}$;
- rollout length: $K=10$ integrator steps, or $0.10\,\mathrm{s}$;
- optimizer training length: $10{,}000$ main updates;
- separate initialization stage: $200$ mass-pretraining updates.

## 10. Source map

- V2 GP architecture and port-Hamiltonian vector field: [`src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py`](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/model.py)
- V2 Lie-IMEX integrator: [`src/models/SE3_Quadrotor/ph_gp_lieimex_v2/integrator.py`](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/integrator.py)
- V2-to-shared-trainer bindings: [`src/models/SE3_Quadrotor/ph_gp_lieimex_v2/train.py`](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/train.py)
- Shared variational training loop: [`src/models/SE3_Quadrotor/ph_gp_lieimex/train.py`](src/models/SE3_Quadrotor/ph_gp_lieimex/train.py)
- Matérn RFF and $SO(3)$ spectral features: [`src/models/SE3_Quadrotor/ph_gp_lieimex/features.py`](src/models/SE3_Quadrotor/ph_gp_lieimex/features.py)
- Actual v2 clean-run settings: [`src/models/SE3_Quadrotor/ph_gp_lieimex_v2/run_clean_training.py`](src/models/SE3_Quadrotor/ph_gp_lieimex_v2/run_clean_training.py)
- Dataset generator: [`tmp/aggressive_quadrotor_datagen.py`](tmp/aggressive_quadrotor_datagen.py)
- V2 $K=10$ dataset view: [`src/models/SE3_Quadrotor/comparision/reduced_input_k10_data.py`](src/models/SE3_Quadrotor/comparision/reduced_input_k10_data.py)

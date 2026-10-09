# lie_ph — Lie-PH models on SE(3) (quadrotor: PyBullet simulation and real IDSIA flights)

The pendulum twin is [src/models/3D_SO3_Windy_Pendulum/lie_ph](../../3D_SO3_Windy_Pendulum/lie_ph/README.md): same algorithm, same files, with SO(3) in place of SE(3). The PH-NODE baseline is in [../ph_node](../ph_node), the multi-model reports in [../comparison](../comparison), every config in [../configs](../configs).

## 1. Short answer

- A port-Hamiltonian SDE on $SE(3)\times\mathbb R^6$. Every physical function is a learned level plus a variational GP (or an MLP): $M_1^{-1}$, $M_2^{-1}$, $D_v$, $D_\omega$, $V$ and $G$. The diffusion $\Sigma$ has 6 learned numbers.
- **Integrator:** second-order **Lie-IMEX** SDE step (damping implicit), one step per sample.
- **Loss:** the **EKF marginal likelihood** of noisy windows, plus the variational KL. The four observation-noise scales are learned.
- **No physical prior:** training never reads the mass, inertia, gravity, damping, wind, control map or clean flights. Only `evaluate.py` does, after training.
- **One code for simulated and real data:** only the config differs (see section 4).

| model | `model.family` | `model.wind` | `training.loss` |
|---|---|---|---|
| Lie-PH-GP-SDE | gp | true | ekf |
| Lie-PH-GP-ODE | gp | false | ekf |
| Lie-PH-NN-SDE | nn | true | ekf |
| Lie-PH-NN-ODE | nn | false | rollout (trajectory loss) |

## 2. Model ([network.py](network.py))

State $s=[x,\ \mathrm{vec}(R),\ v,\ \omega,\ u]$, with $u$ the wrench $[T,\tau]$ scaled by its training RMS. The Hamiltonian is $H=\tfrac12p_v^\top M_1^{-1}(x)p_v+\tfrac12p_\omega^\top M_2^{-1}(R)p_\omega+V(x)$, and:

- $\dot x=Rv$, $\ \dot R=R\hat\omega$
- $\dot p_v=p_v\times\omega-R^\top\nabla V-D_v(v)\,v+g_F u$
- $\dot p_\omega=p_\omega\times\omega+p_v\times v+\sum_i r_i\times\partial H/\partial r_i-D_\omega(\omega)\,\omega+g_\tau u$
- $d\xi=[M^{-1}\dot p+\dot{M^{-1}}p]\,dt+\Sigma\,dW$, with $\xi=(v,\omega)$

| subnetwork | form | input |
|---|---|---|
| $M_1^{-1}(x)$ | $e^{\lambda_1+\mathrm{GP}}I_3$ | $x$ |
| $M_2^{-1}(R)$ | $LL^\top$, level (6) + GP | $\mathrm{vec}(R)$ |
| $D_v(v)$, $D_\omega(\omega)$ | $LL^\top$, level (6) + GP | $v$, $\omega$ |
| $V(x)$ | $\lambda_V\cdot x$ + GP | $x$ |
| $g(R,x)$ | $\Lambda_g$ (6×4) + GP | $[\mathrm{vec}(R),x]$ |
| $\Sigma$ | 6 numbers, constant, twist units | — |

The features are smooth chordal Matérn RFF ([features.py](features.py)). The starting values are neutral ($M^{-1}=I$, $D=I$, $\lambda_V=0$, $\Lambda_g=0$, $\Sigma=0.1$, $\sigma_{obs}=0.1$).

## 3. Loss: EKF over the 12-dim error state $[\delta x,\delta\theta,\delta v,\delta\omega]$

This is the pendulum filter with $R_{obs}=\mathrm{diag}(\sigma_p^2I,\sigma_R^2I,\sigma_v^2I,\sigma_\omega^2I)$ and $x\oplus\delta=(x+\delta_x,\ R\,\mathrm{Exp}(\delta_\theta),\ v+\delta_v,\ \omega+\delta_\omega)$. That is the generator's absolute-noise model.

After each update the covariance is reset to the tangent space of the new mean, $P\leftarrow G\,P\,G^\top$ with the attitude block $J_r(m_\theta)$ (Bourmaud et al. 2015; Maurer et al. 2025). `evaluate.py` reports the consistency measure NIS/$d$ (1 = consistent).

The training loss is $\mathcal L=\overline{\mathrm{NLL}}/12+\beta\,\mathrm{KL}/(N_{obs}\cdot12)$.

## 4. Training protocol and the two configs

Fixed budget for every run: 5000 steps, `checkpoint_final.pkl` reported (no best-checkpoint selection), float32; the EKF NLL of the posterior-mean model on the noisy test windows is logged as a monitor only. A non-finite loss or gradient aborts the run.

| | PyBullet campaign ([../configs/pybullet-campaign-06-10-2026](../configs/pybullet-campaign-06-10-2026)) | real IDSIA flights ([../configs/IDSIA](../configs/IDSIA)) |
|---|---|---|
| data | 300 flights of 5 s at 50 Hz, fast manoeuvres + vertical excitation, observation noise 0.25, wind 0.5, damping 0.5 | IDSIA Crazyflie 2.1 Brushless, 100 Hz, wrench from measured rotor speeds; train chirp/random/square, test melon |
| windows | 2 s (101 samples), stride 75, batch 16 | 0.5 s (51 samples), stride 10, batch 256 |
| GP hyperparameters $\tau,\ell$ | learned (Prodigy, $d_0=0.01$) | fixed ($\tau=\ell=1$) |
| level optimiser | Prodigy, $d_0=0.01$, random start spread 0.1 | Prodigy, $d_0=10^{-3}$, start spread 0 |
| KL weight $\beta$ | 1 | 0.1 |
| `model.spd_eigenvalue_floor` | 0 (off) | 0.01: $M_2^{-1}=LL^\top+0.01\,I$ |
| `runtime.matmul_precision` | default | highest |

Both use Adam $3\cdot10^{-3}$ on the GP weights and noise scales, cosine decay to 0.1×, global-norm clip 1, controls divided by their training RMS.

On the real flights the learned $\sigma_{obs}$ becomes tiny and the EKF ill-conditioned: without the eigenvalue floor the run hits NaN at step ~330, with the floor alone at step ~2400; floor + fixed $\tau,\ell$ + $d_0=10^{-3}$ trains all 5000 steps (8 Oct 2026).

## 5. No-prior guarantee

- [data.py](data.py) reads only `train_trajectories` (noisy), `test_trajectories_noisy` and `t`.
- [config.py](config.py) refuses keys containing mass, inertia, gravity, damping_coefficient, pretrain, physics, known, prior, true, ground_truth, penalty, mixer, control_level or arm.
- There is no mass pretraining, no known gravity, no structured potential, no penalty and no checkpoint selection on clean data. `model.spd_eigenvalue_floor` is a numerical conditioning floor, not a fitted constant.

## 6. Numerical choices

1. **EKF Jacobians use the polynomial first-order difference** $\mathrm{vee}(\mathrm{skew}(R_b^\top R_a))$ instead of $\mathrm{Log}$. Their second derivative, which training needs, is clean.
2. **Series $\mathrm{Exp}$/$\mathrm{Log}$** below $\theta^2=0.01$.
3. **Smooth chordal features** instead of geodesic-angle features, which have cones at the base rotations.

On the pendulum, (1) and (3) took the training gradient from noise (float32 vs float64 cosine 0.01, norm about $10^{6}$–$10^{12}$) to exact (cosine 0.99993), compared with Log and geodesic-angle features.

## 7. Usage

```bash
C=src/models/SE3_Quadrotor/configs
python src/models/SE3_Quadrotor/lie_ph/train.py    --config $C/pybullet-campaign-06-10-2026/Lie-PH-GP-SDE_DampConst-Wind_noise0p25.yaml
python src/models/SE3_Quadrotor/lie_ph/train.py    --config $C/IDSIA/Lie-PH-GP-SDE_IDSIA.yaml
python src/models/SE3_Quadrotor/lie_ph/evaluate.py --run <run_dir>          # simulated: vs true operators; IDSIA: vs published constants
python src/models/SE3_Quadrotor/lie_ph/evaluate.py --self-test --config $C/pybullet-campaign-06-10-2026/base.yaml
```

**Self-test (3 Oct 2026):**

- With the true operators and the row-$k{+}1$ control, the one-step twist residual is $0.50/\sqrt{s}$ on all 6 axes, which equals the true wind. With the row-$k$ control it is $1.43/1.39/0.72$ on the torque axes.
- EKF NLL on 128 noisy train windows:

| operators | NLL |
|---|---|
| true | **0.1066** |
| $D_v$ ×4 | 0.1134 |
| $D_\omega$ ×4 | 0.1143 |
| control ×0.9 | 0.1086 |
| gravity ×0.9 | 0.1076 |
| $\Sigma$ ×2 | 0.1179 |
| $\sigma_{obs}$ ×0.5 | 0.817 |
| untrained start | 4.79 |

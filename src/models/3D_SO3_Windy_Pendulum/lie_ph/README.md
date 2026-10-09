# lie_ph — Lie-PH models on SO(3) (3D windy pendulum)

The quadrotor twin is [src/models/SE3_Quadrotor/lie_ph](../../SE3_Quadrotor/lie_ph/README.md): same algorithm, same files, with SE(3) in place of SO(3).

## 1. Short answer

- A port-Hamiltonian SDE on $SO(3)\times\mathbb R^3$. Every physical function is a learned level plus a variational GP: $M^{-1}$, $D$, $V$ and $g$. The diffusion $\Sigma$ is learned too.
- **Integrator:** second-order **Lie-IMEX** SDE step (damping implicit), one step per sample. It is the same scheme as the env's `_lie_imex_step`; with the true operators one step differs from the env's 10 substeps by at most $1.7\cdot10^{-3}$ rad/s per sample, against a wind of 0.11.
- **Loss:** the **EKF marginal likelihood** of the noisy windows, plus the variational KL (negative ELBO). The observation noise $\sigma_{obs}$ is learned.
- **No physical prior:** training never reads the mass, rod length, gravity, friction, wind or the clean data. Only `evaluate.py` does, after training.

## Model variants (config `model.family`, `model.wind`, `training.loss`)

| model | subnetworks | process noise $\Sigma$ | loss |
|---|---|---|---|
| Lie-PH-GP-SDE | level + GP | learned | EKF marginal likelihood + KL |
| Lie-PH-GP-ODE | level + GP | none | EKF marginal likelihood + KL |
| Lie-PH-NN-SDE | tanh MLP | learned | EKF marginal likelihood |
| Lie-PH-NN-ODE | tanh MLP | none | trajectory loss (`losses.rollout_loss`) |

All four use the Lie-IMEX integrator below. The multi-model campaign, reports and closed-loop tests are in
[../comparison](../comparison).

## 2. Model ([network.py](network.py))

State $s=[\mathrm{vec}(R),\ \omega,\ u]$, momentum $p=M(R)\,\omega$, rows $r_i$ of $R$:

- $\dot R = R\,\hat\omega$
- $\dot p = p\times\omega + \sum_i r_i\times\partial V/\partial r_i - D(\omega)\,\omega + g(R)\,u$
- $d\omega = \big[M^{-1}\dot p + (\tfrac{d}{dt}M^{-1})\,p\big]\,dt + \Sigma\,dW$

| subnetwork | form | input |
|---|---|---|
| $M^{-1}(R)$ | $LL^\top$, level (6) + GP | $\mathrm{vec}(R)$ |
| $D(\omega)$ | $LL^\top$, level (6) + GP | $\omega$ |
| $V(R)$ | $\lambda_V\cdot\mathrm{vec}(R)$ + GP | $\mathrm{vec}(R)$ |
| $g(R)$ | $\Lambda_g$ (3×3) + GP | $\mathrm{vec}(R)$ |
| $\Sigma$ | $\mathrm{diag}(e^{\ell_1},e^{\ell_2},e^{\ell_3})$, constant | — |

- **GP:** $\mathrm{GP}(x)=\phi(x)\,w$, with $q(w)=\mathcal N(m,\mathrm{diag}\,s^2)$ and prior $\mathcal N(0,I)$.
- **Features** ([features.py](features.py)): smooth Matérn random Fourier features on $\mathrm{vec}(R)\in\mathbb R^9$, so the kernel uses the chordal distance $\|R-R'\|_F$.
- **Starting values:** neutral, $M^{-1}=I$, $D=I$, $\lambda_V=0$, $\Lambda_g=0$, $\Sigma=0.1$.
- **Gauge:** $(M^{-1},V,D,g)\to(cM^{-1},V/c,D/c,g/c)$ leaves the dynamics unchanged. So only products are identifiable: $M^{-1}D$, $M^{-1}g$, $M^{-1}\nabla V$ and $\Sigma$.

## 3. Integrator ([integrator.py](integrator.py))

With $K=M^{-1}D$, $a_i$ the acceleration at stage $i$ with damping added back, and $\eta=\sqrt h\,\Sigma z$:

- predictor: $(I+hK_1)\,\omega_p=\omega+h a_1+\eta$, and $R_p=R\,\mathrm{Exp}(h\omega)$
- corrector: $(I+\tfrac h2K_2)\,\omega_+=\omega+\tfrac h2(a_1+a_2-K_1\omega)+\eta$, and $R_+=R\,\mathrm{Exp}(\tfrac h2(\omega+\omega_p))$

## 4. Loss ([losses.py](losses.py)): EKF marginal likelihood

**Observation model:** $y_k=x_k\oplus\varepsilon_k$, where $x\oplus[a,b]=(R\,\mathrm{Exp}(a),\ \omega+b)$ and $R_{obs}=\mathrm{diag}(\sigma_R^2I,\sigma_\omega^2I)$. This is exactly the generator's noise.

**Filter, per window:** start at $x_0=y_0$, $P_0=R_{obs}$. Then for each observation:

- predict through the 10 substeps: $P\leftarrow F_jPF_j^\top+G_jG_j^\top$
- update: $r=y_k\ominus x^-$, $S=P^-+R_{obs}$, $\ -\log p=\tfrac12(r^\top S^{-1}r+\log\det S+6\log2\pi)$, then Kalman gain and Joseph form
- reset: $P\leftarrow G\,P\,G^\top$, with $G$ the identity except the attitude block $J_r(m_\theta)$, where $m=Kr$ is the correction. This re-expresses the covariance at the updated mean (Bourmaud et al. 2015; Maurer et al. 2025). On this data it changes the NLL by $2\cdot10^{-5}$.

**Consistency check:** `evaluate.py` reports NIS/$d$, the mean of $r^\top S^{-1}r/d$ on the noisy test windows. A consistent filter gives 1; with the true operators it is 1.001.

**Training loss:** $\mathcal L=\overline{\mathrm{NLL}}/6+\beta\,\mathrm{KL}/(N_{obs}\cdot6)$.

**Why the EKF, and not the old per-step pseudo-likelihood (PL):**

- A noisy increment has variance $\Sigma^2\Delta t+2\sigma_{obs}^2$.
  - At obs noise 0.25 and wind 0.5 that is $0.0125+0.125$, so the noise is 10 times the wind.
  - PL sees only this sum. It cannot separate $\Sigma$ from $\sigma_{obs}$, and with $\sigma_{obs}$ frozen at 0.01 it inflates $\Sigma$ to $\approx1.66$.
- PL also evaluates the drift at the noisy state, which biases the damping toward zero.
- In the EKF the wind accumulates in $P$, while the observation noise stays white in time. That difference is what makes the two separable.

## 5. Training protocol ([../configs/old/dev/03-10-2026-20-00_lie_ph_5s-beta0p1.yaml](../configs/old/dev/03-10-2026-20-00_lie_ph_5s-beta0p1.yaml))

Agreed on 3 October 2026. It is identical for the pendulum and the quadrotor except for the sampling rate and the batch size.

| item | value |
|---|---|
| data | `PENDULUM-DATASET-SDE-DiffusionConstant-0p5-DissipationConstant-0p5-5s`: 100 train and 25 test trajectories of 5 s (101 samples, dt 0.05 s), obs noise 0.25, wind 0.5, damping 0.5 |
| windows | 2 s = 41 samples, stride 30 samples (25 % overlap) → 3 windows per trajectory, 300 in total |
| batch | 64 windows per step |
| integrator | one Lie-IMEX step per sample |
| optimiser | Adam $3\cdot10^{-3}$ on GP weights and noise scales; Prodigy (rate 0.25, $d_0=10^{-3}$) on the level scalars, one instance per subnetwork; both with cosine decay to 0.1× at the last step; clip 1 |
| KL | $\beta$ rises linearly over 1000 steps; pilot configs `beta0p1` ($\beta=0.1$) and `beta1` ($\beta=1$, the ELBO) |
| steps | **5000, fixed** |
| reported model | `checkpoint_final.pkl`, the step-5000 model. There is no best-checkpoint selection. |
| monitor | EKF NLL of the posterior-mean model on the noisy test windows, every 500 steps, logged only |
| precision | float32 |
| speed | about 0.18 s/step alone (15 min per run) |

The earlier 10 000-step runs (config `03-10-2026-12-00`) used shorter windows and kept a best checkpoint; their results are in the run folders.

## 6. Numerical choices that matter

The training gradient goes through the EKF Jacobian $F=\partial\Psi/\partial x$. It therefore needs second and third derivatives of the dynamics with respect to the state.

1. **Jacobians use $\mathrm{vee}(\mathrm{skew}(R_b^\top R_a))$, not $\mathrm{Log}(R_b^\top R_a)$.** The two have the same first derivative. The true $\mathrm{Log}$ at the identity has $1/\sin^2\theta$ terms in its second derivative, which produced gradients of about $10^{12}$.
2. **$\mathrm{Exp}$ and $\mathrm{Log}$ use Taylor coefficients below $\theta^2=0.01$.** In float32 the closed forms lose their derivatives to cancellation near zero.
3. **Smooth chordal features instead of the geodesic-angle features of `src/utils/JAX/gp_model.py`.** A geodesic distance has a cone at each base rotation, so $V(R)$ had kinks. With those features the float32 gradient had cosine 0.01 against float64; it is now 0.99993.

## 7. No-prior guarantee

- [data.py](data.py) reads only `x` (noisy train), `test_x_noisy` and `t`. It never reads `settings`, `test_x` or the env.
- [config.py](config.py) refuses any key containing mass, inertia, gravity, friction, pretrain, physics, known, prior, true, ground_truth or penalty.
- There is no pretraining toward $1/(ml^2)$ and no PSD floor at the true $M^{-1}$. The wind's geometry ($l$, lever arm $Re_z$) is not hard-coded: $\Sigma$ is three free numbers.
- The comparison with true values lives only in [evaluate.py](evaluate.py).

## 8. Control convention

The control of the interval $k\to k+1$ is stored in row $k+1$ (row 0 repeats $u_0$). The self-test shows it: with the
true operators the one-interval $\omega$ residual per $\sqrt{s}$ is $(0.495, 0.493, 0.0)$ with row $k+1$, which equals the
true wind; with row $k$ it is $(0.78, 0.99, 0.12)$.

## 9. Usage

```bash
# self-test: model vs env step, control row, EKF NLL minimal at the true operators (reads ground truth; no training)
python src/models/3D_SO3_Windy_Pendulum/lie_ph/evaluate.py --self-test --config src/models/3D_SO3_Windy_Pendulum/configs/old/dev/03-10-2026-20-00_lie_ph_5s-beta0p1.yaml
# train (GPU from runtime.gpu unless CUDA_VISIBLE_DEVICES is set)
python src/models/3D_SO3_Windy_Pendulum/lie_ph/train.py --config src/models/3D_SO3_Windy_Pendulum/configs/old/dev/03-10-2026-20-00_lie_ph_5s-beta0p1.yaml
# evaluate against the ground truth after training
python src/models/3D_SO3_Windy_Pendulum/lie_ph/evaluate.py --run experiments/pendulum_so3/train_runs/<run>
```

Self-test (3 Oct 2026), EKF NLL per dimension on 256 noisy train windows:

| operators | NLL |
|---|---|
| true | **0.153** |
| damping ×4 | 0.373 |
| control ×0.7 | 0.156 |
| $\Sigma$ ×2 | 0.170 |
| gravity ×0.8 | 0.171 |
| $\sigma_{obs}$ ×0.5 | 0.806 |
| untrained start | 13.37 |

The model with the true operators matches the env's Lie-IMEX step to $5\cdot10^{-8}$.

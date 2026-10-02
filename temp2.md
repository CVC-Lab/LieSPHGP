• The PH-GP-LieIMEX controller did not crash. It completed all $20$ seconds, but the physical trajectory failed to track the reference.

  The primary cause is that the learned physics subnetworks are not mutually scale-consistent for use inside the controller.

  1. Impossible control commands begin immediately

  At $t=0$:

  - Maximum available RPM: $21{,}703$
  - Requested RPM: up to $542{,}299$
  - At least one motor is saturated during $100%$ of the trajectory.
  - Overall motor saturation fraction: $50.34%$.

  The controller computes

  $u=g_\theta(q)^\dagger w_{\mathrm{desired}}$

  at src/models/SE3_Quadrotor/comparision/run_dataset_matched_gt_vs_learned_controller.py:388.

  Once $u$ is converted to motor RPM, the motors cannot generate the requested wrench. The runner records the clipping and allocation error at src/models/
  SE3_Quadrotor/comparision/run_damping_ablation_controller.py:236.

  2. The learned rotational inertia is thousands of times too large

  At the initial state, the learned rotational inverse-mass eigenvalues are approximately

  $\lambda(M_{\omega,\theta}^{-1})=[7.08,;17.26,;26.36]$.

  The real CF2P values are approximately

  $\lambda(J^{-1})=[30{,}915,;41{,}752,;41{,}752]$.

  Therefore, the controller interprets the rotational inertia as approximately

  $\lambda(M_{\omega,\theta})=[0.038,;0.058,;0.141]$,

  while the real inertia is only around

  $\lambda(J)=[2.40,;2.40,;3.23]\times10^{-5}$.

  The learned inertia is consequently roughly $10^3$–$6\times10^3$ times too large. The controller multiplies attitude corrections by this learned inertia
  at src/models/SE3_Quadrotor/comparision/run_dataset_matched_gt_vs_learned_controller.py:380.

  That produces initial desired torques of approximately

  $\tau_{\mathrm{desired}}=[-11.26,;-5.93,;-15.38]$,

  which are completely unrealistic for this small drone.

  3. The port-Hamiltonian gauge factors do not cancel

  The fitted gauge scales are

  $\beta_v=19.43,\qquad\beta_\omega=1920.59$.

  A dynamically equivalent gauge should approximately satisfy

  $M_\theta^{-1}=\frac{1}{\beta}M^{-1},\quad
  V_\theta=\beta V,\quad
  D_\theta=\beta D,\quad
  g_\theta=\beta g$.

  If all subnetworks use the same $\beta$, it cancels inside the dynamics and controller.

  Here, $M_\omega^{-1}$ exhibits a scale near $\beta_\omega\approx1920$, but the learned rotational control map does not have the corresponding scale. Its
  singular values at $t=0$ are only

  $\sigma(g_\theta)=[11.66,;8.26,;6.74,;1.73]$.

  Therefore, the large learned inertia is not cancelled by a proportionally scaled control map. This is the strongest reason for the enormous torque and RPM
  requests.

  4. Potential and dissipation are also physically inaccurate

  The correct initial gravity gradient is

  $\nabla_xV=[0,;0,;mg]=[0,;0,;0.2646]$.

  The GP predicts

  $\nabla_xV_\theta=[-0.237,;0.150,;0.489]$.

  This creates spurious horizontal gravity forces and an excessive vertical component. The controller uses this gradient directly at src/models/
  SE3_Quadrotor/comparision/run_dataset_matched_gt_vs_learned_controller.py:302.

  The learned dissipation is highly anisotropic and nearly rank-deficient. At $t=0$:

  $\lambda(D_v)\approx[1.5\times10^{-12},1.81,4.11]$,

  $\lambda(D_\omega)\approx[2.1\times10^{-9},0.525,10.84]$.

  The controller adds $D_vv_b$ and $D_\omega\omega_b$ at lines 302 and 380. Dissipation is not the initial failure source because $\omega_b=0$ initially,
  but it later adds incorrect forces and torques.

  5. Short-window loss does not guarantee controller correctness

  The GP was trained on five-point windows:

  $T_{\mathrm{train}}=\frac{4}{240}\approx0.0167,\mathrm{s}$.

  Its clean test loss is good because that test also measures short local predictions. The $20$-second controller evaluation is approximately $1200$ times
  longer.

  The trajectory loss observes combinations such as

  $M_\theta^{-1}g_\theta u,\qquad
  M_\theta^{-1}\nabla V_\theta,\qquad
  M_\theta^{-1}D_\theta$,

  rather than supervising every subnetwork independently. Many incorrect $M_\theta$, $V_\theta$, $D_\theta$ and $g_\theta$ combinations can therefore
  produce a good short-term trajectory loss.

  6. The controller enters a feedback loop outside the training region

  The sequence is:

  $ \text{wrong subnetworks}
  \rightarrow \text{huge requested wrench}
  \rightarrow \text{motor saturation}
  \rightarrow \text{tracking error}
  \rightarrow \text{states outside training distribution}
  \rightarrow \text{worse GP predictions}
  \rightarrow \text{larger commands}. $

  By $t\approx4$ seconds, the altitude crosses below zero. Because the benchmark intentionally removed the ground plane, it continues falling until
  approximately

  $z=-45.73,\mathrm{m}$.

  During this extrapolation:

  - $\kappa(g_\theta)$ reaches $3071$.
  - The potential gradient reaches roughly $189$ in magnitude.
  - Requested RPM reaches $4.91\times10^8$.
  - Wrench-allocation error reaches approximately $5.92\times10^6$.

  7. This is not a Lie-IMEX integration failure

  The controller does not use Lie-IMEX to advance the drone. It queries the learned subnetworks, calculates motor commands, and PyBullet advances the
  physical plant at src/models/SE3_Quadrotor/comparision/run_damping_ablation_controller.py:224.

  The corrected GP open-loop Lie-IMEX rollout remains finite and performs reasonably over $0.5$ seconds. The controller failure is primarily caused by
  incorrect and gauge-inconsistent learned physics operators, followed by actuator saturation and out-of-distribution feedback—not the Lie-IMEX solver.
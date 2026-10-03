# rov_se3_port_ham

Our own BlueROV2 Heavy simulator, written in port-Hamiltonian form on SE(3) and integrated on the Lie group. It is
the underwater counterpart of `quadrotor_se3_port_ham` + `quadrotor_se3_pybullet`: exact ground truth for every
operator, one config-driven dataset generator. The real recordings of the same vehicle are in `envs/rov_se3_marinarium/`.

| File | Role |
|---|---|
| `bluerov2.py` | vehicle constants (published), PX4-order thruster allocation $E$, T200 thrust map, pH dynamics, Lie-group Heun step |
| `data/t200_thrust_table.csv` | Blue Robotics T200 public thrust table, 10–20 V, in N |
| `datagen/generate_dataset.py` + `datagen/config.yaml` | one generator, one config → `datasets/ROV-DATASET-<name>/` |
| `datagen/configs/*.yaml` | named dataset configs (ODE / SDE) |
| `mini_tests/test_gt_pH_matches_rov_env.py` | energy balance, $-\nabla V$, SO(3), term-by-term match to the Marinarium Fossen baseline |

## Model

World z **down** (depth), body forward-right-down, as in the Marinarium recordings and Fossen's convention.
With $\nu=(v,\omega)$, $P=M\nu$:

$$H = \tfrac12\,\nu^\top M\nu + V,\qquad V(p,R) = (B-W)\,z + B\,e_3^\top R\,r_b$$

$$\dot p = Rv,\quad \dot R = R\hat\omega,\quad \dot P = J(P)\,\nu - D(\nu)\,\nu - \nabla V + G\,u + \text{wind}$$

- $M = M_{RB}+M_A = \mathrm{diag}(19.86,\ 20.62,\ 32.18,\ 0.449,\ 0.365,\ 0.592)$ (CG at the body origin, so diagonal).
- $J(P)\nu = [P_v\times\omega;\ P_v\times v + P_\omega\times\omega]$: rigid body + added mass, conserves energy.
- $D(\nu)\nu = (D_L + D_Q\,\lvert\nu\rvert)\,\nu$ per axis; $W-B = +0.98$ N, CB 1 cm above the CG.
- $G = E$ (6 × 8, thrusters), $I_6$ (wrench) or the nonlinear T200 map (commands).

Parameters: von Benzon et al., *An open-source benchmark simulator: control of a BlueROV2 underwater robot*,
J. Mar. Sci. Eng. 10(12), 2022, Table A1 (heavy), as used by the Marinarium Fossen baseline (`fossen/BlueROV2.py`);
`M`, `D`, restoring and Coriolis agree with that code to $10^{-14}$. Thruster geometry and order: the PX4 rotor list of
the Marinarium repository, i.e. the motor channels of the real recordings.

## Datasets

| Folder | Config |
|---|---|
| `ROV-DATASET-ODE-DissipationLinearQuadratic` | published damping, no wind |
| `ROV-DATASET-SDE-DiffusionConstant-DissipationLinearQuadratic` | + white wind $\sigma = 0.05$ m s$^{-1.5}$ (linear), 0.1 rad s$^{-1.5}$ (angular) |
| `ROV-DATASET-SDE-DiffusionRateDependent-DissipationLinearQuadratic` | + wind $\sigma(1+k\lVert\cdot\rVert)$, $k$ = 2 (linear), 1 (angular) |

All `hard+eval`, 8-thruster input, 50 Hz samples (physics 500 Hz), 10 s flights: 50 train, 10 test, 20 heldout (eval
library). Checks: replaying a recorded ODE flight through the stored ground truth with the recorded $u$ reproduces it
to $3\times10^{-15}$; the SDE one-step twist residuals divided by $\sigma(x)\sqrt h$ have std 0.96–0.995.

Motion is sized to the real recordings (manual set): median speed 0.32 vs 0.28 m/s, rate 0.25 vs 0.37 rad/s,
tilt 6.8° vs 12.3°. At speed the vehicle pitches noticeably (Munk moment from $M_{A,w}\gg M_{A,u}$ plus the 6 cm
offset of the horizontal thrusters); the real vehicle also reaches 47–117° of tilt.

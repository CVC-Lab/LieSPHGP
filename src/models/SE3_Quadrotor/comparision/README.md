# 47-page single-model report (pure JAX)

`generate_report.py` builds the established 47-page report for one completed
`ph_gp_lie_imex` run. Nothing is imported from the legacy archive; the model is
evaluated with the package's own JAX code in float64.

## Pipeline

| Module | Role |
|---|---|
| `report_evaluation.py` | loads the run (`metadata.json`, `training_stats.npz`, checkpoint, config copy); loads the common clean 240 Hz PID benchmark `datasets/data/pybullet_quadrotor/D0_CF2P_PID_contact-free_nonlinear-damping-c0p5_seed0_260905.pkl`; repeats one held-out flight 10 times (120 states, $h=1/240$ s, identical $z_0$ and $u(t)$); Lie-IMEX rollout with `integrator.rollout_control_sequence`; per-step errors and physical energy; the six operators along the truth trajectory; analytic targets $I/m$, $J^{-1}$, $mc(1+\lVert v\rVert)I$, $c(1+\lVert\omega\rVert)J$, $mgz$, $g_0$; mass-gauge fit $\beta_v,\beta_\omega$; 7-repeat rollout benchmark |
| `report_figures.py` | the page layouts: summary tables (squared reference rows plus RMS rows in physical units and RMS relative to the ground-truth excursion), loss curves (per-step train losses, smoothed with a 101-step window), error/energy/SO(3) pages, state and phase pages, 3×3 and 6×4 subnetwork grids raw and gauge-fixed, computation/controller tables, posterior-std histograms, image and audit pages |
| `report_controller.py` | 20 s closed-loop PyBullet run (CtrlAviary CF2P, `Physics.PYB`, 240 Hz, ground plane removed, contact-free nonlinear damping $c=0.5$) with the reference energy-based SE(3) law ($K_p=[10,10,50]$, $K_v=3$, $K_R=250$, $K_\omega=20$, 40° tilt limit) driven by the learned operators; $\nabla V$ by `jax.grad`; wrench allocated as $u=\mathrm{pinv}(M^{-1}g)\,(M^{-1}w)$, which uses only the identifiable products and is invariant to the free scale of the learned operators (the reference paper's $u=g^{\dagger}w$ is not, and fails when the learned gauge has large force rows); rotor speeds through the plant mixer, clipped at the plant maximum RPM; tracking and 3D plots |
| `generate_report.py` | orchestrates the above, asserts exactly 47 pages, writes `reports/SE3_Quadrotor/<stamp>_report_single_PH-GP-LieIMEX-levels_<tags>_47-page.pdf` plus a JSON sidecar, and copies the PDF into the run folder as `report.output_name` |

## Ground-truth-operator reference model

Every report also rolls out a report-only reference model: the simulator's
analytic operators ($I/m$, $J^{-1}$, $mgz$, the selection matrix,
$mc(1+\lVert v\rVert)I$, $c(1+\lVert\omega\rVert)J$ from the evaluation dataset)
inside the same Lie-IMEX integrator with the same recorded $u(t)$
(`report_evaluation.GroundTruthSE3HamODE`). It appears as a second column on
page 1 and the fairness table, and as a second curve on the error, energy,
SO(3), state and phase pages. Its error is the floor any learned model can
reach with this integrator; the learned model's excess over it is model error.

## Page map

1 summary table · 2 subnetwork MSE (raw / gauge-fixed) · 3–7 train losses ·
8–12 test-window losses · 13–16 error vs GT · 17–18 physical energy ·
19–20 SO(3) violation · 21 state ensemble · 22 single trajectory · 23 phase
portraits · 24–35 $M_1^{-1}$, $M_2^{-1}$, $D_v$, $D_\omega$, $g$, $V$ raw and
gauge-fixed · 36 fairness and computation · 37 compute bars · 38 failure-aware
controller table · 39 GP posterior uncertainty · 40–41 controller plots ·
42–47 audit tables (data/optimizer, architecture, window, checkpoint,
open-loop inference, closed-loop controller).

## Gauge-fixed pages

Every operator has a free learned level, so the data determine only the
products $M_1^{-1}g$, $M_2^{-1}g$, $M_1^{-1}\nabla V$, $M_1^{-1}D_v$, $M_2^{-1}D_\omega$.
The gauge fit is $\beta_v=\arg\min_\beta\lVert\beta M_1^{-1}-I/m\rVert^2$ and
$\beta_\omega$ likewise for $M_2^{-1}$; the gauge-fixed pages show
$\beta_vM_1^{-1}$, $\beta_\omega M_2^{-1}$, $D_v/\beta_v$, $D_\omega/\beta_\omega$,
$g$ with its force rows divided by $\beta_v$ and torque rows by $\beta_\omega$, and
$V/\beta_v+c$.

## Running

The trainer calls `generate_reports(config_copy, experiment_dirs=[run_dir])`
after a completed run when `report.auto_generate` is true. Manually:

```bash
python -m src.models.SE3_Quadrotor.comparision.generate_report \
  --config experiments/quadrotor/<run>/09-09-2026-23-13_ph_gp_lie_imex.yaml
```

Optional: `--selected-step`, `--evaluation-dataset`, `--controller-seconds`,
`--benchmark-repeats`.

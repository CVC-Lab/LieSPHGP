"""GP-SDE stochastic evaluation report (per obs-noise level).

Evaluates ph_gp_sde *as an SDE* — exercising its learned diffusion σ(q) —
rather than the drift-only view in make_comparison_3way.py.

For one fixed initial condition we draw K Wiener-noise realisations and roll
out BOTH:
  - the ground-truth env (wind_force_std σ_GT applied at the tip), and
  - the GP-SDE posterior-mean model (drift + learned σ(q)·dW diffusion),
using the SAME dW per pair, so the comparison is pathwise as well as
distributional.

Pages:
  1. σ(q) along the trajectory vs the GT wind level (β-gauge corrected).
  2. Summary table: σ level, spread-ratio, band coverage, pathwise error.
  3. State ensemble with ±2σ bands (GT vs SDE), euler + ω.
  4. Hamiltonian energy with ±2σ bands.
  5. Uncertainty calibration: ensemble std(t), GT vs SDE.
  6. Pathwise tracking (same-dW) vs free diffusive spread.

Usage:
  python make_sde_stochastic_eval.py --sde_dir <path> --out_pdf r.pdf \
    [--obs_label "obs=0.05"] [--n_samples 30] [--seed 42] [--ic 0]

GT env settings (g_diag, varying_friction, wind_force_std) are auto-detected
from the run-dir name.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import make_comparison_gp_vs_nn as base      # noqa: E402
import make_comparison_3way as m3                 # noqa: E402

GT_C = "black"
SDE_C = "#2ca02c"     # green, matching the 3-way report

_sde_rollout = None
def _ensure_sde_rollout():
    """Import the *stochastic* Lie-Heun rollout (base only loads the ODE one)."""
    global _sde_rollout
    if _sde_rollout is not None:
        return
    base._ensure_jax()
    from src.utils.JAX.lie_integrator import lie_heun_sde_rollout
    _sde_rollout = lie_heun_sde_rollout


def rollout_sde_stochastic(model, R0, omega0, dW_per_outer, dt, n_substeps):
    """Sampled SDE rollout (posterior-mean drift + learned σ·dW), u=0."""
    _ensure_sde_rollout()
    jnp = base.jnp
    h = dt / n_substeps
    x0 = jnp.concatenate([
        jnp.asarray(R0.reshape(-1), dtype=jnp.float32),
        jnp.asarray(omega0,          dtype=jnp.float32),
    ])
    u = jnp.zeros(3, dtype=jnp.float32)
    dW = jnp.asarray(dW_per_outer, dtype=jnp.float32)
    return np.asarray(_sde_rollout(model, x0, u, jnp.float32(h), dW))


def sigma_along(model, traj_12):
    """Learned diffusion scale σ(q) (posterior mean) along a trajectory."""
    jnp = base.jnp
    qs = jnp.asarray(traj_12[:, :9], dtype=jnp.float32)
    s = base.jax.vmap(lambda q: model.sigma(q))(qs)
    return np.asarray(s)


# ── Metric helpers ───────────────────────────────────────────────────────────

def _euler_ensemble(samples):                       # (K, T, 12) -> (K, T, 3)
    return np.array([base.rotmat_to_euler(samples[k, :, :9])
                     for k in range(samples.shape[0])])


def _energy_ensemble(samples):                      # (K, T, 12) -> (K, T)
    return np.array([base.get_energy(samples[k]) for k in range(samples.shape[0])])


def _band(ax, t, ens, color, label, lw=1.8):
    m, s = ens.mean(0), ens.std(0)
    ax.plot(t, m, color=color, lw=lw, label=label)
    ax.fill_between(t, m - 2 * s, m + 2 * s, color=color, alpha=0.18)
    return m, s


# ── Pages ────────────────────────────────────────────────────────────────────

def fig_sigma(model, gt_traj, t_eval, beta, sigma_gt, obs_label):
    s_raw = sigma_along(model, gt_traj)
    s_corr = s_raw / beta
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(t_eval, s_raw, color=SDE_C, lw=1.4, label="σ(q) raw")
    axes[0].axhline(sigma_gt * beta, color="k", ls=":", lw=1.4,
                    label=f"GT·β = {sigma_gt*beta:.3f}")
    axes[0].set_title("Raw learned diffusion σ(q)")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("σ raw")
    axes[1].plot(t_eval, s_corr, color=SDE_C, lw=1.4, label="σ(q)/β  (corrected)")
    axes[1].axhline(sigma_gt, color="k", ls=":", lw=1.6,
                    label=f"GT wind σ = {sigma_gt:g}")
    axes[1].set_title("β-gauge-corrected σ(q)/β vs GT wind level")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("σ / β")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend(fontsize="small")
    fig.suptitle(f"{obs_label} — learned diffusion vs GT wind  "
                 f"(β={beta:.3f}, mean σ/β={s_corr.mean():.3f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, s_corr


def fig_state_bands(t, gt_s, sde_s, obs_label):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    gt_eul, sde_eul = _euler_ensemble(gt_s), _euler_ensemble(sde_s)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    for i in range(3):
        _band(axes[i, 0], t, gt_eul[:, :, i], GT_C, "GT" if i == 0 else None)
        _band(axes[i, 0], t, sde_eul[:, :, i], SDE_C, "SDE" if i == 0 else None)
        _band(axes[i, 1], t, gt_s[:, :, 9 + i], GT_C, "GT" if i == 0 else None)
        _band(axes[i, 1], t, sde_s[:, :, 9 + i], SDE_C, "SDE" if i == 0 else None)
        axes[i, 0].set_ylabel(labels_ang[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel(labels_om[i]); axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="small"); axes[0, 1].legend(fontsize="small")
    fig.suptitle(f"{obs_label} — stochastic rollout, mean ±2σ bands "
                 f"({gt_s.shape[0]} samples, same dW per pair)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_energy_bands(t, gt_s, sde_s, obs_label):
    gt_e, sde_e = _energy_ensemble(gt_s), _energy_ensemble(sde_s)
    fig, ax = plt.subplots(figsize=(11, 6))
    _band(ax, t, gt_e, GT_C, "GT")
    _band(ax, t, sde_e, SDE_C, "SDE")
    # a few thin individual paths for texture
    for k in range(min(5, gt_e.shape[0])):
        ax.plot(t, gt_e[k], color=GT_C, lw=0.5, alpha=0.25)
        ax.plot(t, sde_e[k], color=SDE_C, lw=0.5, alpha=0.25)
    ax.set_title(f"{obs_label} — Hamiltonian energy, mean ±2σ bands")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def fig_calibration(t, gt_s, sde_s, obs_label):
    # per-time ensemble std: ω (Frobenius over components) and euler
    gt_om_std = np.linalg.norm(gt_s[:, :, 9:12].std(0), axis=-1)
    sde_om_std = np.linalg.norm(sde_s[:, :, 9:12].std(0), axis=-1)
    gt_eul, sde_eul = _euler_ensemble(gt_s), _euler_ensemble(sde_s)
    gt_an_std = np.linalg.norm(gt_eul.std(0), axis=-1)
    sde_an_std = np.linalg.norm(sde_eul.std(0), axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(t, gt_om_std, color=GT_C, lw=1.6, label="GT")
    axes[0].plot(t, sde_om_std, color=SDE_C, lw=1.6, label="SDE")
    axes[0].set_title("Ensemble std of ω  (‖std‖)")
    axes[0].set_ylabel("rad/s")
    axes[1].plot(t, gt_an_std, color=GT_C, lw=1.6, label="GT")
    axes[1].plot(t, sde_an_std, color=SDE_C, lw=1.6, label="SDE")
    axes[1].set_title("Ensemble std of Euler angles  (‖std‖)")
    axes[1].set_ylabel("rad")
    for ax in axes:
        ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    fig.suptitle(f"{obs_label} — uncertainty calibration "
                 "(does the SDE's spread match the wind-induced spread?)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, (gt_om_std, sde_om_std)


def fig_pathwise(t, gt_s, sde_s, obs_label):
    # same-dW pairwise geodesic² (does SDE track GT given identical noise?)
    paired = base.geodesic_sq(sde_s, gt_s).mean(0)            # (T,)
    # free spread: geodesic² between GT and a shuffled GT (independent noise)
    perm = np.roll(np.arange(gt_s.shape[0]), 1)
    free = base.geodesic_sq(gt_s[perm], gt_s).mean(0)         # (T,)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t, paired, color=SDE_C, lw=1.8, label="SDE vs GT (same dW) — pathwise")
    ax.plot(t, free, color=GT_C, ls="--", lw=1.6,
            label="GT vs GT (independent dW) — free diffusive spread")
    ax.set_yscale("log")
    ax.set_title(f"{obs_label} — pathwise tracking vs free spread")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("geodesic² (rad²)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize="small")
    fig.tight_layout()
    return fig, (paired, free)


def fig_table(rows, obs_label):
    fig, ax = plt.subplots(figsize=(9, 1.5 + 0.55 * len(rows)))
    ax.axis("off")
    ax.set_title(f"{obs_label} — GP-SDE stochastic-evaluation summary",
                 fontsize=12, fontweight="bold", pad=12)
    tbl = ax.table(cellText=[[v] for _, v in rows],
                   rowLabels=[k for k, _ in rows],
                   colLabels=["value"], cellLoc="center", rowLoc="left",
                   loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.7)
    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sde_dir", required=True)
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--obs_label", default="")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ic", type=int, default=0, help="which IC index to evaluate")
    args = ap.parse_args()

    sde_ckpt = m3.find_sde_ckpt(args.sde_dir)
    assert sde_ckpt, f"No GP-SDE checkpoint in {args.sde_dir}"
    print(f"SDE ckpt : {os.path.basename(sde_ckpt)}")

    g_diag = base._parse_g_from_dirname(args.sde_dir)
    varying_friction = base._parse_varfric_from_dirname(args.sde_dir)
    sigma_gt = m3._parse_wind_from_dirname(args.sde_dir)
    print(f"GT env: g_diag={g_diag}, varying_friction={varying_friction}, "
          f"wind σ_GT={sigma_gt}")
    if sigma_gt <= 0:
        print("WARNING: wind_force_std parsed as 0 — there is no GT diffusion to match.")
    env_kw = dict(base.ENV_KW)
    env_kw.update(g_diag=g_diag, varying_friction=varying_friction,
                  wind_force_std=sigma_gt)

    dt = env_kw["dt"]
    n_sub, n_outer = base.N_SUBSTEPS, base.N_OUTER
    t_eval = np.arange(n_outer + 1) * dt
    sqrt_h = float(np.sqrt(dt / n_sub))

    # Fixed IC from the requested ensemble index.
    env = base.windy_pendulum_3d(seed=args.seed + args.ic, **env_kw)
    env.reset(seed=args.seed + args.ic)
    R0, omega0 = env.R.copy(), env.omega.copy()

    # K independent Wiener-noise realisations, shared between GT and SDE.
    rng = np.random.default_rng(args.seed + 10_000)
    K = args.n_samples
    dW_set = [rng.normal(0.0, sqrt_h, (n_outer, n_sub, 3)) for _ in range(K)]

    # ── Load model ──
    sde_model = m3.load_sde_model(sde_ckpt)

    # ── Sampled rollouts (same dW per pair) ──
    print(f"Rolling out {K} GT and {K} SDE samples (same dW) from IC {args.ic} ...")
    gt_s = np.zeros((K, n_outer + 1, 12))
    sde_s = np.zeros((K, n_outer + 1, 12))
    for k in range(K):
        gt_s[k] = base.rollout_gt(env, R0, omega0, dW_set[k])
        sde_s[k] = rollout_sde_stochastic(sde_model, R0, omega0, dW_set[k],
                                          dt, n_sub)

    # Mean GT path — reference trajectory for σ(q) eval and the M-gauge β
    # (same convention as the 3-way report's subnet pages).
    gt_mean_traj = gt_s.mean(0)
    beta = base.estimate_beta(m3.eval_subnets_sde(sde_model, gt_mean_traj),
                              gt_m_inv_scalar=1.0 / base.I_PERP)

    obs = args.obs_label or os.path.basename(args.sde_dir)
    print(f"Writing PDF: {args.out_pdf}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_pdf)), exist_ok=True)

    with PdfPages(args.out_pdf) as pdf:
        fig_s, s_corr = fig_sigma(sde_model, gt_mean_traj, t_eval, beta,
                                  sigma_gt, obs)
        fig_cal, (gt_om_std, sde_om_std) = fig_calibration(
            t_eval, gt_s, sde_s, obs)
        fig_pw, (paired, free) = fig_pathwise(t_eval, gt_s, sde_s, obs)

        # spread ratio at final time + band coverage on ω
        eps = 1e-12
        om_ratio_final = float(sde_om_std[-1] / (gt_om_std[-1] + eps))
        om_ratio_mean = float(np.mean(sde_om_std / (gt_om_std + eps)))
        gt_e, sde_e = _energy_ensemble(gt_s), _energy_ensemble(sde_s)
        e_ratio_final = float(sde_e.std(0)[-1] / (gt_e.std(0)[-1] + eps))
        # coverage: fraction of GT-sample ω within SDE mean ±2σ, at final time
        sde_m, sde_sd = sde_s[:, -1, 9:12].mean(0), sde_s[:, -1, 9:12].std(0)
        gt_final = gt_s[:, -1, 9:12]
        inside = np.all(np.abs(gt_final - sde_m) <= 2 * sde_sd + eps, axis=1)
        coverage = float(inside.mean())

        rows = [
            ("wind σ_GT", f"{sigma_gt:g}"),
            ("β (M-gauge)", f"{beta:.4f}"),
            ("mean σ(q)/β (learned)", f"{s_corr.mean():.4f}"),
            ("σ/β vs σ_GT (ratio)", f"{s_corr.mean()/ (sigma_gt+eps):.3f}"),
            ("ω-spread ratio SDE/GT (mean)", f"{om_ratio_mean:.3f}"),
            ("ω-spread ratio SDE/GT (final)", f"{om_ratio_final:.3f}"),
            ("energy-spread ratio SDE/GT (final)", f"{e_ratio_final:.3f}"),
            ("ω 2σ-band coverage of GT (final)", f"{coverage:.2f}"),
            ("pathwise geo² @10s (same dW)", f"{paired[-1]:.3e}"),
            ("free geo² @10s (indep dW)", f"{free[-1]:.3e}"),
        ]

        pdf.savefig(fig_s); plt.close(fig_s)
        pdf.savefig(fig_table(rows, obs)); plt.close()
        pdf.savefig(fig_state_bands(t_eval, gt_s, sde_s, obs)); plt.close()
        pdf.savefig(fig_energy_bands(t_eval, gt_s, sde_s, obs)); plt.close()
        pdf.savefig(fig_cal); plt.close(fig_cal)
        pdf.savefig(fig_pw); plt.close(fig_pw)

    print(f"σ/β mean={s_corr.mean():.4f} (GT {sigma_gt})  "
          f"ω-spread ratio mean={om_ratio_mean:.3f}  coverage={coverage:.2f}")
    print("Done.")


if __name__ == "__main__":
    main()

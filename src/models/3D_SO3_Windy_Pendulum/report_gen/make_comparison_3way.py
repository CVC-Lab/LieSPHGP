"""3-way comparison PDF: NN-ODE-fp32 vs GP-ODE-v2 vs GP-SDE.

Generates one comparison report for a single (obs-noise) triple of runs:
  - ph_nn_ode (PyTorch)   — deterministic ODE (no diffusion)
  - ph_gp_ode   (JAX)       — deterministic ODE (no diffusion)
  - ph_gp_sde      (JAX)       — STOCHASTIC: drift + learned σ(q)·dW diffusion,
                                 driven by the SAME Wiener noise as the matching
                                 GT path (pathwise-comparable). Env GT is also
                                 stochastic (wind diffusion). NN / GP-ODE have no
                                 diffusion term so they stay deterministic.

Reuses the helpers/metrics/loaders from make_comparison_gp_vs_nn.py and
adds the GP-SDE model + generalized N-model plotting.

Usage:
  python make_comparison_3way.py \
    --nn_dir <path> --gp_dir <path> --sde_dir <path> \
    --out_pdf report.pdf [--obs_label "obs=0.05"] [--seed 42] [--device cpu]

GT rollouts use the env's stochastic wind (wind_force_std auto-detected from
the run-dir names) so the report reflects the wind-0.5 training setup.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import make_comparison_gp_vs_nn as base   # noqa: E402  (reuse everything)

# ── Per-model display style ──────────────────────────────────────────────────
STYLE = {
    "nn_ode": ("NN-ODE-fp32", base.COLORS["nn_ode"], "--"),   # blue dashed
    "gp_ode": ("GP-ODE-v2",   base.COLORS["gp_ode"], "-"),    # red solid
    "gp_sde": ("GP-SDE (stoch)", "#2ca02c",          ":"),    # green dotted
}


# ── GP-SDE model: loader / rollout / subnet eval ─────────────────────────────

_sde_loaded = False
def _ensure_sde():
    global _sde_loaded, DissipativeSO3HamSDE, lie_heun_sde_rollout
    if _sde_loaded:
        return
    base._ensure_jax()                                # jax/jnp/eqx + lie rollout
    spec = importlib.util.spec_from_file_location(
        "_gp_sde_network", os.path.join(THIS_DIR, "ph_gp_sde", "network.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gp_sde_network"] = mod
    spec.loader.exec_module(mod)
    DissipativeSO3HamSDE = mod.DissipativeSO3HamSDE
    # Stochastic Lie-Heun rollout (base only loads the deterministic ODE one).
    from src.utils.JAX.lie_integrator import lie_heun_sde_rollout as _sde_roll
    lie_heun_sde_rollout = _sde_roll
    _sde_loaded = True


def find_sde_ckpt(run_dir):
    return base._find_latest(run_dir, r"wp3d-so3hamGPSDE-5p-(\d+)\.eqx$")


def load_sde_model(ckpt_path):
    _ensure_sde()
    template = DissipativeSO3HamSDE(
        key=base.jax.random.PRNGKey(0), u_dim=3, init_gain=0.5, friction=True)
    return base.eqx.tree_deserialise_leaves(ckpt_path, template)


def rollout_sde(model, R0, omega0, dW_per_outer, dt, n_substeps):
    """Stochastic SDE rollout (drift + learned σ(q)·dW diffusion), u=0.

    The diffusion term IS exercised. `dW_per_outer` is the SAME Wiener noise
    used for the matching GT rollout, so the SDE path is pathwise-comparable to
    GT (a fair test of drift+diffusion together). NN / GP-ODE remain
    deterministic (they have no diffusion)."""
    _ensure_sde()
    jnp = base.jnp
    h = dt / n_substeps
    x0 = jnp.concatenate([
        jnp.asarray(R0.reshape(-1), dtype=jnp.float32),
        jnp.asarray(omega0,          dtype=jnp.float32),
    ])
    u = jnp.zeros(3, dtype=jnp.float32)
    dW = jnp.asarray(dW_per_outer, dtype=jnp.float32)
    return np.asarray(lie_heun_sde_rollout(model, x0, u, jnp.float32(h), dW))


def eval_subnets_sde(model, traj_12):
    """Returns dicts with M (N,3,3), V (N,), D (N,3,3), B (N,3,3).
    Note: GP-SDE's Dw depends on q only (no momentum input)."""
    jnp = base.jnp
    qs = jnp.asarray(traj_12[:, :9], dtype=jnp.float32)
    omegas = jnp.asarray(traj_12[:, 9:12], dtype=jnp.float32)

    def per_sample(q, omega):
        M_inv = model._M_call(q)                    # (3,3) posterior mean
        V = model._V_call(q)[0]                      # scalar
        D = model._Dw_call(q)                        # (3,3) — q only
        B = model._g_call(q)                         # (3,3)
        return M_inv, V, D, B

    M_inv, V, D, B = base.jax.vmap(per_sample)(qs, omegas)
    return {"M": np.asarray(M_inv), "V": np.asarray(V),
            "D": np.asarray(D), "B": np.asarray(B)}


# ── Dir-name parsing (wind, in addition to base's g/varfric) ──────────────────

def _parse_wind_from_dirname(dirname, default=0.0):
    m = re.search(r"_wind([0-9pn]+)_", os.path.basename(dirname.rstrip("/")))
    if not m:
        return default
    return float(m.group(1).replace("p", ".").replace("n", "-"))


# ── Generalized N-model figures ──────────────────────────────────────────────

def _ax(title):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(title); ax.grid(True, alpha=0.3)
    return fig, ax


def fig_lines(models, key, title, ylab, logy=True, smooth=True, eval_key=False):
    fig, ax = _ax(title)
    any_plotted = False
    for m in models:
        st = m["stats"]
        if not st or key not in st:
            continue
        if eval_key:
            if "eval_step" not in st:
                continue
            ax.plot(st["eval_step"], st[key], color=m["color"], ls=m["ls"],
                    lw=1.4, label=m["label"])
        else:
            arr = np.asarray(st[key])
            if smooth:
                arr = base._smooth(arr)
            ax.plot(arr, color=m["color"], ls=m["ls"], lw=1.2, label=m["label"])
        any_plotted = True
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(ylab)
    if any_plotted:
        ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def fig_summary_table(models, gt_trajs, horizon_s, dt, obs_label):
    n_keep = min(int(round(horizon_s / dt)) + 1, gt_trajs.shape[1])
    gt_sl = gt_trajs[:, :n_keep]

    def _metrics(preds):
        pred_sl = preds[:, :n_keep]
        N, T, _ = gt_sl.shape
        geo = base.geodesic_sq(pred_sl, gt_sl)
        om = base.omega_sq(pred_sl, gt_sl)
        e_gt = np.array([base.get_energy(gt_sl[n]) for n in range(N)])
        e_pr = np.array([base.get_energy(pred_sl[n]) for n in range(N)])
        e_err = np.abs(e_pr - e_gt).mean(axis=1)
        R_pr = pred_sl[..., :9].reshape(N, T, 3, 3)
        det_max = np.abs(np.linalg.det(R_pr) - 1.0).max(axis=1)
        orth_max = np.linalg.norm(
            (np.einsum("ntji,ntjk->ntik", R_pr, R_pr) - np.eye(3)[None, None]
             ).reshape(N, T, 9), axis=-1).max(axis=1)
        return {
            "geo² mean":    base._ms(geo.mean(axis=1)),
            "‖Δω‖² mean":  base._ms(om.mean(axis=1)),
            "geo² final":   base._ms(geo[:, -1]),
            "‖Δω‖² final": base._ms(om[:, -1]),
            "|ΔE| mean":   base._ms(e_err),
            "max|det−1|":  base._ms(det_max),
            "max‖RᵀR−I‖": base._ms(orth_max),
        }

    mets = [_metrics(m["preds"]) for m in models]
    if abs(horizon_s - dt * (gt_trajs.shape[1] - 1)) < 1e-9:
        for key, lab in (("test_geo_loss", "test geo²"),
                         ("test_l2_loss", "test ω-MSE")):
            for m, met in zip(models, mets):
                st = m["stats"]
                if st and key in st:
                    arr = np.asarray(st[key]).ravel()
                    met[f"{lab} (final)"] = (f"{float(arr[-1]):.3e}"
                                             if arr.size else "—")
                else:
                    met[f"{lab} (final)"] = "—"

    row_names = list(mets[0].keys())
    cell = [[met.get(k, "—") for met in mets] for k in row_names]
    fig, ax = plt.subplots(figsize=(3.5 + 2.6 * len(models),
                                    1.5 + 0.55 * len(row_names)))
    ax.axis("off")
    ax.set_title(
        f"{obs_label} — horizon {horizon_s:g}s — "
        f"{gt_trajs.shape[0]}-rollout ensemble vs GT",
        fontsize=12, fontweight="bold", pad=12)
    tbl = ax.table(cellText=cell, rowLabels=row_names,
                   colLabels=[m["label"] for m in models],
                   cellLoc="center", rowLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.6)
    for j in range(len(models)):
        tbl[0, j].set_text_props(weight="bold")
    fig.tight_layout()
    return fig


def fig_subnet_loss_table(models, gt_all, gt_M_inv_flat, gt_B_flat,
                          gt_D_vals, horizon_s):
    """Per-subnet MSE vs GT (β-corrected, ensemble over full horizon)."""
    def _accum(m):
        M_l, V_l, D_l, B_l = [], [], [], []
        for ti in range(gt_all.shape[0]):
            gt_traj = gt_all[ti]
            sub = m["subnet_fn"](gt_traj)
            sub_b = base.apply_beta(sub, m["beta"])
            T = gt_traj.shape[0]
            gt_M = np.tile(gt_M_inv_flat[None], (T, 1, 1))
            gt_B = np.tile(gt_B_flat[None],     (T, 1, 1))
            gt_D = gt_D_vals(gt_traj).reshape(T, 3, 3)
            gt_v = base.GT_M * base.GT_G * base.GT_L * gt_traj[:, 8]
            M_l.append(float(np.mean((sub_b["M"] - gt_M) ** 2)))
            D_l.append(float(np.mean((sub_b["D"] - gt_D) ** 2)))
            B_l.append(float(np.mean((sub_b["B"] - gt_B) ** 2)))
            v = sub_b["V"]
            v_c = v - v.mean() + gt_v.mean()
            V_l.append(float(np.mean((v_c - gt_v) ** 2)))
        return {"M⁻¹ MSE": base._ms(M_l), "V MSE (centred)": base._ms(V_l),
                "D MSE": base._ms(D_l), "B MSE": base._ms(B_l)}

    mets = [_accum(m) for m in models]
    row_names = list(mets[0].keys())
    cell = [[met[k] for met in mets] for k in row_names]
    fig, ax = plt.subplots(figsize=(3.5 + 2.6 * len(models),
                                    1.5 + 0.55 * len(row_names)))
    ax.axis("off")
    ax.set_title(
        f"Subnetwork prediction MSE vs GT — horizon {horizon_s:g}s\n"
        f"(β-gauge corrected, {gt_all.shape[0]}-traj ensemble)",
        fontsize=12, fontweight="bold", pad=12)
    tbl = ax.table(cellText=cell, rowLabels=row_names,
                   colLabels=[m["label"] for m in models],
                   cellLoc="center", rowLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.6)
    for j in range(len(models)):
        tbl[0, j].set_text_props(weight="bold")
    fig.tight_layout()
    return fig


def fig_traj_mse(models, t_eval, gt_all):
    figs = []
    for metric_fn, ylab, title in (
        (base.geodesic_sq, "geodesic² (rad²)", "Geodesic error vs GT"),
        (base.omega_sq,    "‖Δω‖² (rad²/s²)", "Angular velocity MSE vs GT"),
    ):
        fig, ax = _ax(title)
        for m in models:
            err = metric_fn(m["preds"], gt_all)
            ax.plot(t_eval, err.mean(0), color=m["color"], ls=m["ls"],
                    lw=1.6, label=m["label"])
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
        ax.legend(fontsize="small"); fig.tight_layout()
        figs.append(fig)
    return figs


def fig_energy(models, t_eval, gt_all):
    gt_e = np.array([base.get_energy(gt_all[n]) for n in range(gt_all.shape[0])])
    fig1, ax = _ax("Hamiltonian energy — ensemble mean")
    ax.plot(t_eval, gt_e.mean(0), "k-", lw=2, label="GT")
    ax.fill_between(t_eval, gt_e.mean(0) - 2 * gt_e.std(0),
                    gt_e.mean(0) + 2 * gt_e.std(0), color="black", alpha=0.15)
    for m in models:
        me = np.array([base.get_energy(m["preds"][n])
                       for n in range(m["preds"].shape[0])])
        ax.plot(t_eval, me.mean(0), color=m["color"], ls=m["ls"], lw=1.6,
                label=m["label"])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.legend(fontsize="small"); fig1.tight_layout()

    fig2, ax2 = _ax("Hamiltonian energy — single trajectory (traj 0)")
    ax2.plot(t_eval, gt_e[0], "k-", lw=2, label="GT")
    for m in models:
        e0 = base.get_energy(m["preds"][0])
        ax2.plot(t_eval, e0, color=m["color"], ls=m["ls"], lw=1.6,
                 label=m["label"])
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Energy (J)")
    ax2.legend(fontsize="small"); fig2.tight_layout()
    return fig1, fig2


def fig_so3(models, t_eval, gt_all):
    figs = []
    for metric_fn, ylab, title in (
        (base.so3_det_err,  "|det(R)−1|", "SO(3) violation — determinant"),
        (base.so3_orth_err, "‖RᵀR−I‖_F", "SO(3) violation — orthogonality"),
    ):
        fig, ax = _ax(title)
        gt_m = np.array([metric_fn(gt_all[n])
                         for n in range(gt_all.shape[0])]).mean(0)
        ax.plot(t_eval, gt_m, "k--", lw=1.5, label="GT")
        for m in models:
            mm = np.array([metric_fn(m["preds"][n])
                           for n in range(m["preds"].shape[0])]).mean(0)
            ax.plot(t_eval, mm, color=m["color"], ls=m["ls"], lw=1.4,
                    label=m["label"])
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
        ax.legend(fontsize="small"); fig.tight_layout()
        figs.append(fig)
    return figs


def fig_state_ensemble(models, t_eval, gt_all):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    gt_eul = np.array([base.rotmat_to_euler(gt_all[n, :, :9])
                       for n in range(gt_all.shape[0])])
    gt_om = gt_all[:, :, 9:12]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    gt_em, gt_es = gt_eul.mean(0), gt_eul.std(0)
    gt_om_m, gt_om_s = gt_om.mean(0), gt_om.std(0)
    for i in range(3):
        axes[i, 0].plot(t_eval, gt_em[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 0].fill_between(t_eval, gt_em[:, i] - 2 * gt_es[:, i],
                                gt_em[:, i] + 2 * gt_es[:, i],
                                color="black", alpha=0.15)
        axes[i, 1].plot(t_eval, gt_om_m[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 1].fill_between(t_eval, gt_om_m[:, i] - 2 * gt_om_s[:, i],
                                gt_om_m[:, i] + 2 * gt_om_s[:, i],
                                color="black", alpha=0.15)
    for m in models:
        eul = np.array([base.rotmat_to_euler(m["preds"][n, :, :9])
                        for n in range(m["preds"].shape[0])])
        em, om_m = eul.mean(0), m["preds"][:, :, 9:12].mean(0)
        for i in range(3):
            axes[i, 0].plot(t_eval, em[:, i], color=m["color"], ls=m["ls"],
                            lw=1.4, label=m["label"] if i == 0 else None)
            axes[i, 1].plot(t_eval, om_m[:, i], color=m["color"], ls=m["ls"],
                            lw=1.4, label=m["label"] if i == 0 else None)
    for i in range(3):
        axes[i, 0].set_ylabel(labels_ang[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel(labels_om[i]); axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small", ncol=2)
    axes[0, 1].legend(fontsize="x-small", ncol=2)
    fig.suptitle("State trajectories — 10-traj ensemble means",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_state_single(models, t_eval, gt_single):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    gt_eul = base.rotmat_to_euler(gt_single[:, :9])
    for i in range(3):
        axes[i, 0].plot(t_eval, gt_eul[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 1].plot(t_eval, gt_single[:, 9 + i], "k-", lw=2,
                        label="GT" if i == 0 else None)
    for m in models:
        traj = m["preds"][0]
        eul = base.rotmat_to_euler(traj[:, :9])
        for i in range(3):
            axes[i, 0].plot(t_eval, eul[:, i], color=m["color"], ls=m["ls"],
                            lw=1.4, label=m["label"] if i == 0 else None)
            axes[i, 1].plot(t_eval, traj[:, 9 + i], color=m["color"], ls=m["ls"],
                            lw=1.4, label=m["label"] if i == 0 else None)
    for i in range(3):
        axes[i, 0].set_ylabel(labels_ang[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel(labels_om[i]); axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small"); axes[0, 1].legend(fontsize="x-small")
    fig.suptitle("Single trajectory (traj 0) state",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_phase_portraits(models, gt_all):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om = ["Omega X", "Omega Y", "Omega Z"]
    gt_eul = np.array([base.rotmat_to_euler(gt_all[n, :, :9])
                       for n in range(gt_all.shape[0])])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i in range(3):
        axes[i].plot(gt_eul[0, :, i], gt_all[0, :, 9 + i], "k-", lw=2,
                     label="GT" if i == 0 else None)
        for m in models:
            eul = base.rotmat_to_euler(m["preds"][0, :, :9])
            axes[i].plot(eul[:, i], m["preds"][0, :, 9 + i], color=m["color"],
                         ls=m["ls"], lw=1.4,
                         label=m["label"] if i == 0 else None)
        axes[i].set_xlabel(labels_ang[i]); axes[i].set_ylabel(labels_om[i])
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(fontsize="x-small")
    fig.suptitle("Phase portraits — single trajectory (traj 0)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_subnet_matrix(models, t_eval, gt_traj, comp_key, gt_val_fn, title):
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    gt_vals = gt_val_fn(gt_traj) if gt_val_fn else None
    for idx in range(9):
        r, c = divmod(idx, 3); ax = axes[r, c]
        if gt_vals is not None:
            ax.plot(t_eval, gt_vals[:, idx], "k:", lw=1.5,
                    label="GT" if idx == 0 else None)
        for m in models:
            arr = m["sub_b"].get(comp_key)
            if arr is not None and arr.ndim == 3:
                ax.plot(t_eval, arr.reshape(arr.shape[0], 9)[:, idx],
                        color=m["color"], ls=m["ls"], lw=1.0, alpha=0.85,
                        label=m["label"] if idx == 0 else None)
        ax.set_title(f"({r},{c})"); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize="x-small")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_subnet_V(models, t_eval, gt_traj, title):
    gt_v = base.GT_M * base.GT_G * base.GT_L * gt_traj[:, 8]
    fig, ax = _ax(title)
    ax.plot(t_eval, gt_v, "k:", lw=1.5, label="GT")
    for m in models:
        v = m["sub_b"].get("V")
        if v is not None:
            ax.plot(t_eval, v - v.mean() + gt_v.mean(),
                    color=m["color"], ls=m["ls"], lw=1.0, alpha=0.85,
                    label=m["label"])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("V(q)")
    ax.legend(fontsize="small"); fig.tight_layout()
    return fig


def sigma_along(model, traj_12):
    """GP-SDE learned diffusion scale σ(q) (posterior mean) along a trajectory."""
    jnp = base.jnp
    qs = jnp.asarray(traj_12[:, :9], dtype=jnp.float32)
    return np.asarray(base.jax.vmap(lambda q: model.sigma(q))(qs))


def fig_sde_sigma(sde_model, gt_traj, t_eval, beta_sde, sigma_gt):
    """Predicted diffusion σ(q) of the GP-SDE vs the GT wind level.

    The PH gauge that scales M⁻¹ by β also scales the diffusion, so the
    physically-comparable quantity is σ(q)/β (same convention as the V/D/B
    subnet pages). GT reference is the constant wind σ = wind_force_std."""
    s_raw = sigma_along(sde_model, gt_traj)
    s_corr = s_raw / beta_sde
    col = STYLE["gp_sde"][1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(t_eval, s_raw, color=col, lw=1.4, label="σ(q) raw")
    axes[0].axhline(sigma_gt * beta_sde, color="k", ls=":", lw=1.4,
                    label=f"GT·β = {sigma_gt*beta_sde:.3f}")
    axes[0].set_title("GP-SDE raw learned diffusion σ(q)")
    axes[0].set_ylabel("σ (raw)")
    axes[1].plot(t_eval, s_corr, color=col, lw=1.4, label="σ(q)/β (corrected)")
    axes[1].axhline(sigma_gt, color="k", ls=":", lw=1.6,
                    label=f"GT wind σ = {sigma_gt:g}")
    axes[1].set_title("β-corrected σ(q)/β vs GT wind level")
    axes[1].set_ylabel("σ / β")
    for ax in axes:
        ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    fig.suptitle(f"GP-SDE predicted diffusion σ along GT traj 0  "
                 f"(β={beta_sde:.3f}, mean σ/β={s_corr.mean():.3f}, "
                 f"GT wind={sigma_gt:g})", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn_dir",  required=True)
    ap.add_argument("--gp_dir",  required=True)
    ap.add_argument("--sde_dir", required=True)
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--obs_label", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # ── Checkpoints + stats ──
    nn_ckpt = base.find_nn_ckpt(args.nn_dir)
    gp_ckpt = base.find_gp_ckpt(args.gp_dir)
    sde_ckpt = find_sde_ckpt(args.sde_dir)
    assert nn_ckpt, f"No NN checkpoint in {args.nn_dir}"
    assert gp_ckpt, f"No GP-ODE checkpoint in {args.gp_dir}"
    assert sde_ckpt, f"No GP-SDE checkpoint in {args.sde_dir}"
    stats_nn = base.load_stats(args.nn_dir, "wp3d-so3ham-rk4-5p")
    stats_gp = base.load_stats(args.gp_dir, "wp3d-so3hamGPODE-5p")
    stats_sde = base.load_stats(args.sde_dir, "wp3d-so3hamGPSDE-5p")
    print(f"NN  ckpt : {os.path.basename(nn_ckpt)}")
    print(f"GP  ckpt : {os.path.basename(gp_ckpt)}")
    print(f"SDE ckpt : {os.path.basename(sde_ckpt)}")

    # ── Env settings auto-detected from run-dir names ──
    g_diag = base._parse_g_from_dirname(args.gp_dir)
    varying_friction = base._parse_varfric_from_dirname(args.gp_dir)
    wind = _parse_wind_from_dirname(args.gp_dir)
    print(f"GT env: g_diag={g_diag}, varying_friction={varying_friction}, "
          f"wind_force_std={wind}")
    env_kw = dict(base.ENV_KW)
    env_kw["g_diag"] = g_diag
    env_kw["varying_friction"] = varying_friction
    env_kw["wind_force_std"] = wind

    # ── ICs, noise, GT rollouts ──
    rng = np.random.default_rng(args.seed)
    dt = env_kw["dt"]
    dt_sub = dt / base.N_SUBSTEPS
    sqrt_h = float(np.sqrt(dt_sub))
    dW_ensemble = [rng.normal(0.0, sqrt_h, (base.N_OUTER, base.N_SUBSTEPS, 3))
                   for _ in range(base.N_TRAJ_ENSEMBLE)]
    t_eval = np.arange(base.N_OUTER + 1) * dt

    ic_list = []
    for ti in range(base.N_TRAJ_ENSEMBLE):
        env = base.windy_pendulum_3d(seed=args.seed + ti, **env_kw)
        env.reset(seed=args.seed + ti)
        ic_list.append((env.R.copy(), env.omega.copy()))

    print(f"Rolling out {base.N_TRAJ_ENSEMBLE} GT trajectories (wind={wind}) ...")
    gt_all = np.zeros((base.N_TRAJ_ENSEMBLE, base.N_OUTER + 1, 12))
    for ti in range(base.N_TRAJ_ENSEMBLE):
        env = base.windy_pendulum_3d(seed=args.seed + ti, **env_kw)
        env.reset(seed=args.seed + ti)
        gt_all[ti] = base.rollout_gt(env, ic_list[ti][0], ic_list[ti][1],
                                     dW_ensemble[ti])

    # ── Load models ──
    print("Loading models ...")
    base._ensure_torch()
    device = base.torch.device(args.device)
    nn_model = base.load_nn_model(nn_ckpt, device)
    gp_model = base.load_gp_model(gp_ckpt)
    sde_model = load_sde_model(sde_ckpt)

    # ── Model rollouts ──
    # NN / GP-ODE: deterministic (no diffusion term).
    # GP-SDE: stochastic — drift + learned σ(q)·dW, driven by the SAME Wiener
    #         noise as the matching GT path (pathwise-comparable).
    print("Rolling out models (SDE with diffusion, matched GT noise) ...")
    preds_nn = np.zeros_like(gt_all)
    preds_gp = np.zeros_like(gt_all)
    preds_sde = np.zeros_like(gt_all)
    for ti in range(base.N_TRAJ_ENSEMBLE):
        R_ti, om_ti = ic_list[ti]
        preds_nn[ti] = base.rollout_nn(nn_model, R_ti, om_ti, t_eval, device)
        preds_gp[ti] = base.rollout_gp(gp_model, R_ti, om_ti,
                                       base.N_SUBSTEPS, base.N_OUTER, dt)
        preds_sde[ti] = rollout_sde(sde_model, R_ti, om_ti,
                                    dW_ensemble[ti], dt, base.N_SUBSTEPS)

    # ── Subnets on GT traj 0 + β gauge ──
    print("Evaluating subnets on GT traj 0 ...")
    sub_nn = base.eval_subnets_nn(nn_model, gt_all[0], device)
    sub_gp = base.eval_subnets_gp(gp_model, gt_all[0])
    sub_sde = eval_subnets_sde(sde_model, gt_all[0])
    beta_nn = base.estimate_beta(sub_nn, gt_m_inv_scalar=1.0 / base.I_PERP)
    beta_gp = base.estimate_beta(sub_gp, gt_m_inv_scalar=1.0 / base.I_PERP)
    beta_sde = base.estimate_beta(sub_sde, gt_m_inv_scalar=1.0 / base.I_PERP)
    print(f"estimated β  NN={beta_nn:.4f}  GP={beta_gp:.4f}  SDE={beta_sde:.4f}")

    # ── Assemble model list (display order: NN, GP-ODE, GP-SDE) ──
    models = [
        dict(key="nn_ode", stats=stats_nn, preds=preds_nn, beta=beta_nn,
             sub=sub_nn, sub_b=base.apply_beta(sub_nn, beta_nn),
             subnet_fn=lambda traj: base.eval_subnets_nn(nn_model, traj, device)),
        dict(key="gp_ode", stats=stats_gp, preds=preds_gp, beta=beta_gp,
             sub=sub_gp, sub_b=base.apply_beta(sub_gp, beta_gp),
             subnet_fn=lambda traj: base.eval_subnets_gp(gp_model, traj)),
        dict(key="gp_sde", stats=stats_sde, preds=preds_sde, beta=beta_sde,
             sub=sub_sde, sub_b=base.apply_beta(sub_sde, beta_sde),
             subnet_fn=lambda traj: eval_subnets_sde(sde_model, traj)),
    ]
    for m in models:
        m["label"], m["color"], m["ls"] = STYLE[m["key"]]

    # ── GT subnet references ──
    gt_M_inv_flat = (1.0 / base.I_PERP) * np.eye(3)
    gt_B_flat = np.diag(np.asarray(g_diag, dtype=np.float64))

    def gt_M_vals(gt_traj):
        return np.tile(gt_M_inv_flat.reshape(-1), (gt_traj.shape[0], 1))

    def gt_B_vals(gt_traj):
        return np.tile(gt_B_flat.reshape(-1), (gt_traj.shape[0], 1))

    def gt_D_vals(gt_traj):
        if varying_friction:
            D_all = base.gt_D_varying(gt_traj, base.GT_FRICTION)
        else:
            T = gt_traj.shape[0]
            D_all = np.tile((base.GT_FRICTION * np.eye(3))[None], (T, 1, 1))
        return D_all.reshape(D_all.shape[0], 9)

    # ── Write PDF ──
    print(f"Writing PDF: {args.out_pdf}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_pdf)), exist_ok=True)
    obs = args.obs_label or os.path.basename(args.gp_dir)

    with PdfPages(args.out_pdf) as pdf:
        # Summary tables
        for horizon_s in (1.0, 2.0, dt * base.N_OUTER):
            pdf.savefig(fig_summary_table(models, gt_all, horizon_s, dt, obs))
            plt.close()

        # Subnet prediction-loss table (full horizon)
        pdf.savefig(fig_subnet_loss_table(
            models, gt_all, gt_M_inv_flat, gt_B_flat, gt_D_vals,
            dt * base.N_OUTER))
        plt.close()

        # Train loss curves
        for key, title, ylab in (
            ("train_loss",     "Train total loss", "loss"),
            ("train_l2_loss",  "Train L2 (ω)",      "L2"),
            ("train_geo_loss", "Train geodesic²",   "geo²"),
        ):
            pdf.savefig(fig_lines(models, key, title, ylab)); plt.close()

        # NLL (GP-ODE + GP-SDE only; NN has none)
        pdf.savefig(fig_lines(models, "train_nll",
                              "Marginal NLL (training) — GP models",
                              "NLL", logy=False)); plt.close()

        # Eval losses
        for key, title, ylab in (
            ("test_l2_loss",  "Test L2(ω)",   "L2"),
            ("test_geo_loss", "Test geodesic²", "geo²"),
            ("eval_M_loss",   "Eval M⁻¹ MSE",  "MSE"),
            ("eval_V_loss",   "Eval V MSE",     "MSE"),
            ("eval_Dw_loss",  "Eval Dw MSE",    "MSE"),
            ("eval_g_loss",   "Eval g MSE",     "MSE"),
        ):
            pdf.savefig(fig_lines(models, key, title, ylab, eval_key=True))
            plt.close()

        # Trajectory metrics
        for f in fig_traj_mse(models, t_eval, gt_all):
            pdf.savefig(f); plt.close(f)

        # Energy
        e1, e2 = fig_energy(models, t_eval, gt_all)
        pdf.savefig(e1); plt.close(e1)
        pdf.savefig(e2); plt.close(e2)

        # SO(3)
        for f in fig_so3(models, t_eval, gt_all):
            pdf.savefig(f); plt.close(f)

        # State trajectories
        pdf.savefig(fig_state_ensemble(models, t_eval, gt_all)); plt.close()
        pdf.savefig(fig_state_single(models, t_eval, gt_all[0])); plt.close()
        pdf.savefig(fig_phase_portraits(models, gt_all)); plt.close()

        # Subnet evolution (β-corrected) along GT traj 0
        bstr = "  ".join(f"{m['label'].split('-')[0]}={m['beta']:.3f}"
                         for m in models)
        pdf.savefig(fig_subnet_matrix(models, t_eval, gt_all[0], "M", gt_M_vals,
            f"β·M⁻¹(q) along GT traj 0   (β: {bstr})")); plt.close()
        pdf.savefig(fig_subnet_matrix(models, t_eval, gt_all[0], "D", gt_D_vals,
            f"D(q[,p])/β along GT traj 0   (β: {bstr})")); plt.close()
        pdf.savefig(fig_subnet_matrix(models, t_eval, gt_all[0], "B", gt_B_vals,
            f"B(q)/β along GT traj 0   (GT=diag{g_diag}; β: {bstr})")); plt.close()
        pdf.savefig(fig_subnet_V(models, t_eval, gt_all[0],
            f"V(q)/β along GT traj 0   (β: {bstr})")); plt.close()

        # GP-SDE predicted diffusion σ(q) vs GT wind level (SDE-only page).
        pdf.savefig(fig_sde_sigma(sde_model, gt_all[0], t_eval,
                                  beta_sde, wind)); plt.close()

    print("Done.")


if __name__ == "__main__":
    main()

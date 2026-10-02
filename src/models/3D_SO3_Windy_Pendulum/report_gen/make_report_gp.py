"""Single-model GP-ODE-v2 report PDF.

Generates a comprehensive report PDF for a single ph_gp_ode run:
  - summary metric tables (1s / 2s / full horizon) vs GT
  - train / eval / aux-loss curves (incl. L_power, L_V, L_B, L_D)
  - trajectory error vs GT (geodesic², ‖Δω‖²)
  - Hamiltonian energy, SO(3) violations
  - state trajectories (ensemble + single) + phase portraits
  - subnet evolution along the GT trajectory (M⁻¹, D, B, V)

Usage:
  python make_report_gp.py \\
    --gp_dir <run_dir> --out_pdf report.pdf \\
    [--obs_label "obs=0.01"] [--seed 42]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, "../../.."))
GP_ODE_DIR = os.path.join(THIS_FILE_DIR, "ph_gp_ode")

for p in (PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, "src/utils"),
          os.path.join(PROJECT_ROOT, "datasets"),
          os.path.join(PROJECT_ROOT, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

from envs.windy_pendulum_3d import windy_pendulum_3d

# ── Env / rollout constants ──────────────────────────────────────────────────
ENV_KW = dict(
    g=9.81, m=1.0, l=1.0, dt=0.05,
    friction_coeff=0.5, varying_friction=True,
    external_force_type="sine", external_force_std=0.0,
    wind_force_std=0.0,
)


def _parse_g_from_dirname(dirname):
    """Recover (gx, gy, gz) from a run dir name like '..._G0p5-0p7-0p1_...'.
    Returns (1.0, 1.0, 1.0) if no G tag is found."""
    m = re.search(r"_G([0-9pn]+)-([0-9pn]+)-([0-9pn]+)(?:_|$)",
                  os.path.basename(dirname.rstrip("/")))
    if not m:
        return (1.0, 1.0, 1.0)
    def _decode(s):
        return float(s.replace("p", ".").replace("n", "-"))
    return tuple(_decode(g) for g in m.groups())


def _parse_varfric_from_dirname(dirname):
    """True if the run dir name contains '_varfric_' or ends with '_varfric'."""
    base = os.path.basename(dirname.rstrip("/"))
    return ("_varfric_" in base) or base.endswith("_varfric")
N_SUBSTEPS = 10
N_OUTER = 200
N_TRAJ_ENSEMBLE = 10
GT_FRICTION = 0.5
GT_M, GT_L, GT_G = 1.0, 1.0, 9.81
I_PERP = GT_M * GT_L * GT_L

GP_COLOR = "#d62728"   # red
GP_LS = "-"


# ── Lazy JAX import ──────────────────────────────────────────────────────────
_jax_loaded = False
def _ensure_jax():
    global _jax_loaded, jax, jnp, eqx, DissipativeSO3HamODE, lie_heun_ode_rollout
    if _jax_loaded:
        return
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax as _jax
    import jax.numpy as _jnp
    import equinox as _eqx
    spec = importlib.util.spec_from_file_location(
        "_gp_ode_network", os.path.join(GP_ODE_DIR, "network.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gp_ode_network"] = mod
    spec.loader.exec_module(mod)
    from src.utils.JAX.lie_integrator import lie_heun_ode_rollout as _rollout
    jax = _jax
    jnp = _jnp
    eqx = _eqx
    DissipativeSO3HamODE = mod.DissipativeSO3HamODE
    lie_heun_ode_rollout = _rollout
    _jax_loaded = True


# ── Checkpoint / stats discovery ─────────────────────────────────────────────

def _find_latest(run_dir, pattern):
    cands = []
    for fn in os.listdir(run_dir):
        m = re.match(pattern, fn)
        if m:
            cands.append((int(m.group(1)), os.path.join(run_dir, fn)))
    return sorted(cands)[-1][1] if cands else None


def find_gp_ckpt(run_dir):
    return _find_latest(run_dir, r"wp3d-so3hamGPODE-\d+p-(\d+)\.eqx$")


def load_stats(run_dir):
    for fn in os.listdir(run_dir):
        if fn.endswith("-stats.pkl"):
            with open(os.path.join(run_dir, fn), "rb") as f:
                return pickle.load(f)
    return None


# ── Model loading & rollout ──────────────────────────────────────────────────

def load_gp_model(ckpt_path):
    _ensure_jax()
    template = DissipativeSO3HamODE(
        key=jax.random.PRNGKey(0), u_dim=3, init_gain=0.5, friction=True,
        fix_M=False)
    return eqx.tree_deserialise_leaves(ckpt_path, template)


def rollout_gp(model, R0, omega0, n_substeps, n_outer, dt, u_seq):
    """u_seq: (3,) constant or (n_outer, 3) per-outer-step."""
    _ensure_jax()
    h = dt / n_substeps
    x0 = jnp.concatenate([
        jnp.asarray(R0.reshape(-1), dtype=jnp.float32),
        jnp.asarray(omega0,          dtype=jnp.float32),
    ])
    u = jnp.asarray(u_seq, dtype=jnp.float32)
    return np.asarray(
        lie_heun_ode_rollout(model, x0, u, jnp.float32(h), n_substeps, n_outer))


def rollout_gt(env, R0, omega0, dW_per_outer, u_seq):
    """u_seq: (3,) constant or (n_outer, 3) per-outer-step."""
    n_outer, n_sub, _ = dW_per_outer.shape
    h_sub = env.dt / n_sub
    sigma = env.wind_force_std
    u_seq = np.asarray(u_seq, dtype=np.float64)
    if u_seq.ndim == 1:
        u_seq = np.broadcast_to(u_seq[None, :], (n_outer, 3))
    R, omega = R0.copy(), omega0.copy()
    traj = np.zeros((n_outer + 1, 12), dtype=np.float64)
    traj[0, :9] = R.reshape(-1); traj[0, 9:12] = omega
    t = 0.0
    for k in range(n_outer):
        t += env.dt
        w_force = env.update_wind(t)
        for s in range(n_sub):
            R, omega = env._lie_heun_step(
                R, omega, w_force, u_seq[k], h_sub, sigma, dW_per_outer[k, s])
        traj[k + 1, :9] = R.reshape(-1); traj[k + 1, 9:12] = omega
    return traj


# ── Subnet evaluation along a trajectory ─────────────────────────────────────

def eval_subnets_gp(model, traj_12):
    """Returns dict with M (T,3,3), V (T,), D (T,3,3), B (T,3,3)."""
    _ensure_jax()
    qs = jnp.asarray(traj_12[:, :9], dtype=jnp.float32)
    omegas = jnp.asarray(traj_12[:, 9:12], dtype=jnp.float32)

    def per_sample(q, omega):
        M_inv = model.M_net(q, inference_mode=True)
        V = model.V_net(q, inference_mode=True)[0]
        p = jnp.linalg.solve(M_inv, omega)
        D = model._Dw_call(q, p)
        B = model.g_net(q, inference_mode=True)
        return M_inv, V, D, B

    M_inv, V, D, B = jax.vmap(per_sample)(qs, omegas)
    return {
        "M": np.asarray(M_inv),
        "V": np.asarray(V),
        "D": np.asarray(D),
        "B": np.asarray(B),
    }


def estimate_beta(sub, gt_m_inv_scalar=1.0 / I_PERP):
    """Estimate the port-Hamiltonian scale-invariance factor β from M⁻¹.

    Dynamics are invariant under (M⁻¹, V, D, B) → (β·M⁻¹, V/β, D/β, B/β).
    With M_GT⁻¹ = (1/(m·l²))·I, the closed-form least-squares fit is

        β* = gt_m_inv_scalar / mean_t( trace(M_β⁻¹(q_t)) / 3 )

    Multiply the model's M⁻¹ by β, and divide V, D, B by β, to compare on a
    common physical scale with GT (dynamics unchanged).
    """
    if "M" not in sub:
        return 1.0
    diag = np.trace(sub["M"], axis1=1, axis2=2) / 3.0
    mean_diag = float(np.mean(diag))
    if mean_diag < 1e-12:
        return 1.0
    return gt_m_inv_scalar / mean_diag


def apply_beta(sub, beta):
    out = dict(sub)
    if "M" in out: out["M"] = out["M"] * beta
    if "V" in out: out["V"] = out["V"] / beta
    if "D" in out: out["D"] = out["D"] / beta
    if "B" in out: out["B"] = out["B"] / beta
    return out


def gt_D_varying(traj_12, friction_coeff=GT_FRICTION):
    q = traj_12[:, :9]
    omega = traj_12[:, 9:12]
    height_term = 0.5 * (1.0 - q[:, 8])
    speed_term = np.tanh(np.linalg.norm(omega, axis=-1))
    mult = 1.0 + 0.5 * height_term + 0.5 * speed_term
    N = q.shape[0]
    D = np.zeros((N, 3, 3), dtype=np.float64)
    for i in range(3):
        D[:, i, i] = friction_coeff * mult
    return D


# ── Utility ──────────────────────────────────────────────────────────────────

def rotmat_to_euler(R_flat):
    Rs = np.asarray(R_flat).reshape(-1, 3, 3)
    sy = np.sqrt(Rs[:, 0, 0] ** 2 + Rs[:, 1, 0] ** 2)
    near = sy < 1e-6
    roll  = np.where(near, np.arctan2(-Rs[:, 1, 2], Rs[:, 1, 1]),
                           np.arctan2( Rs[:, 2, 1], Rs[:, 2, 2]))
    pitch = np.arctan2(-Rs[:, 2, 0], sy)
    yaw   = np.where(near, 0.0, np.arctan2(Rs[:, 1, 0], Rs[:, 0, 0]))
    return np.stack([roll, pitch, yaw], axis=-1)


def get_energy(traj):
    pe = GT_G * (1.0 - traj[:, 8])
    omega = traj[:, 9:12]
    I = I_PERP * np.eye(3)
    return 0.5 * np.einsum("ti,ij,tj->t", omega, I, omega) + pe


def geodesic_sq(traj_pred, traj_gt):
    R_pred = traj_pred[..., :9].reshape(*traj_pred.shape[:-1], 3, 3)
    R_gt   = traj_gt  [..., :9].reshape(*traj_gt  .shape[:-1], 3, 3)
    M = np.einsum("...ji,...jk->...ik", R_pred, R_gt)
    cos_t = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_t) ** 2


def omega_sq(traj_pred, traj_gt):
    return np.sum((traj_pred[..., 9:12] - traj_gt[..., 9:12]) ** 2, axis=-1)


def so3_det_err(traj):
    R = traj[:, :9].reshape(-1, 3, 3)
    return np.abs(np.linalg.det(R) - 1.0)


def so3_orth_err(traj):
    R = traj[:, :9].reshape(-1, 3, 3)
    diff = np.einsum("tji,tjk->tik", R, R) - np.eye(3)[None]
    return np.linalg.norm(diff.reshape(-1, 9), axis=1)


def _smooth(y, w=21):
    y = np.asarray(y, dtype=np.float64); n = len(y)
    if n < w + 1:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    half = w // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def _ms(x):
    return f"{float(np.mean(x)):.2e}±{float(np.std(x)):.1e}"


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _ax(title):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(title); ax.grid(True, alpha=0.3)
    return fig, ax


def fig_loss(stats, key, title, ylab, logy=True, smooth=True,
              x_key=None, label="GP"):
    fig, ax = _ax(title)
    if stats is None or key not in stats:
        ax.text(0.5, 0.5, f"{key} not in stats", transform=ax.transAxes,
                ha="center", va="center")
        return fig
    arr = np.asarray(stats[key])
    x = np.asarray(stats[x_key]) if x_key and x_key in stats else np.arange(len(arr))
    if smooth:
        arr = _smooth(arr)
    ax.plot(x, arr, color=GP_COLOR, ls=GP_LS, lw=1.4, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(ylab)
    ax.legend(fontsize="small"); fig.tight_layout()
    return fig


def fig_loss_overlay(stats, keys_labels, title, ylab, logy=True, smooth=True,
                     x_key=None):
    """Overlay multiple loss series on the same axes."""
    fig, ax = _ax(title)
    colors = plt.get_cmap("tab10").colors
    plotted = 0
    for i, (key, label) in enumerate(keys_labels):
        if stats is None or key not in stats:
            continue
        arr = np.asarray(stats[key])
        x = np.asarray(stats[x_key]) if x_key and x_key in stats else np.arange(len(arr))
        if smooth:
            arr = _smooth(arr)
        ax.plot(x, arr, color=colors[i % len(colors)], lw=1.4, label=label)
        plotted += 1
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(ylab)
    if plotted:
        ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def fig_summary_table(gt_trajs, preds_gp, stats, horizon_s, dt, obs_label):
    n_keep = min(int(round(horizon_s / dt)) + 1, gt_trajs.shape[1])
    gt_sl   = gt_trajs[:, :n_keep]
    pred_sl = preds_gp[:, :n_keep]
    N, T, _ = gt_sl.shape
    geo  = geodesic_sq(pred_sl, gt_sl)
    om   = omega_sq(pred_sl, gt_sl)
    geo_mean = geo.mean(axis=1)
    om_mean  = om.mean(axis=1)
    e_gt = np.array([get_energy(gt_sl[n]) for n in range(N)])
    e_pr = np.array([get_energy(pred_sl[n]) for n in range(N)])
    e_err = np.abs(e_pr - e_gt).mean(axis=1)
    R_pr = pred_sl[..., :9].reshape(N, T, 3, 3)
    det_max  = np.abs(np.linalg.det(R_pr) - 1.0).max(axis=1)
    orth_max = np.linalg.norm(
        (np.einsum("ntji,ntjk->ntik", R_pr, R_pr) - np.eye(3)[None, None]
         ).reshape(N, T, 9), axis=-1).max(axis=1)
    metrics = {
        "geo² mean":    _ms(geo_mean),
        "‖Δω‖² mean":  _ms(om_mean),
        "geo² final":   _ms(geo[:, -1]),
        "‖Δω‖² final": _ms(om[:, -1]),
        "|ΔE| mean":   _ms(e_err),
        "max|det−1|":  _ms(det_max),
        "max‖RᵀR−I‖": _ms(orth_max),
    }
    if abs(horizon_s - dt * (gt_trajs.shape[1] - 1)) < 1e-9:
        for key, label in (("test_geo_loss", "test geo²"),
                            ("test_l2_loss",  "test ω-MSE")):
            if stats and key in stats:
                arr = np.asarray(stats[key]).ravel()
                metrics[f"{label} (final step)"] = (
                    f"{float(arr[-1]):.3e}" if arr.size else "—")
            else:
                metrics[f"{label} (final step)"] = "—"

    row_names = list(metrics.keys())
    cell = [[metrics[k]] for k in row_names]
    fig, ax = plt.subplots(figsize=(8, 1.5 + 0.55 * len(row_names)))
    ax.axis("off")
    ax.set_title(
        f"{obs_label} — horizon {horizon_s:g}s — {N}-rollout ensemble vs GT",
        fontsize=12, fontweight="bold", pad=12)
    tbl = ax.table(cellText=cell, rowLabels=row_names,
                   colLabels=["GP-ODE-v2"],
                   cellLoc="center", rowLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.6)
    tbl[0, 0].set_text_props(weight="bold")
    fig.tight_layout()
    return fig


def fig_traj_mse(t_eval, gt_all, preds_gp):
    figs = []
    for metric_fn, ylab, title in (
        (geodesic_sq, "geodesic² (rad²)", "Geodesic error vs GT"),
        (omega_sq,    "‖Δω‖² (rad²/s²)", "Angular velocity MSE vs GT"),
    ):
        fig, ax = _ax(title)
        err = metric_fn(preds_gp, gt_all)
        ax.plot(t_eval, err.mean(0), color=GP_COLOR, ls=GP_LS, lw=1.6, label="GP")
        ax.fill_between(t_eval, err.mean(0) - err.std(0), err.mean(0) + err.std(0),
                        color=GP_COLOR, alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
        ax.legend(fontsize="small"); fig.tight_layout()
        figs.append(fig)
    return figs


def fig_energy(t_eval, gt_all, preds_gp):
    gt_e = np.array([get_energy(gt_all[n]) for n in range(gt_all.shape[0])])
    gp_e = np.array([get_energy(preds_gp[n]) for n in range(preds_gp.shape[0])])

    fig1, ax = _ax("Hamiltonian energy — ensemble mean")
    ax.plot(t_eval, gt_e.mean(0), "k-", lw=2, label="GT")
    ax.fill_between(t_eval, gt_e.mean(0)-2*gt_e.std(0), gt_e.mean(0)+2*gt_e.std(0),
                    color="black", alpha=0.15)
    ax.plot(t_eval, gp_e.mean(0), color=GP_COLOR, ls=GP_LS, lw=1.6, label="GP")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.legend(fontsize="small"); fig1.tight_layout()

    fig2, ax2 = _ax("Hamiltonian energy — single trajectory (traj 0)")
    ax2.plot(t_eval, gt_e[0], "k-", lw=2, label="GT")
    ax2.plot(t_eval, gp_e[0], color=GP_COLOR, ls=GP_LS, lw=1.6, label="GP")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Energy (J)")
    ax2.legend(fontsize="small"); fig2.tight_layout()
    return fig1, fig2


def fig_so3(t_eval, gt_all, preds_gp):
    figs = []
    for metric_fn, ylab, title in (
        (so3_det_err,  "|det(R)−1|",   "SO(3) violation — determinant"),
        (so3_orth_err, "‖RᵀR−I‖_F",   "SO(3) violation — orthogonality"),
    ):
        fig, ax = _ax(title)
        gt_m = np.array([metric_fn(gt_all[n]) for n in range(gt_all.shape[0])]).mean(0)
        ax.plot(t_eval, gt_m, "k--", lw=1.5, label="GT")
        m = np.array([metric_fn(preds_gp[n]) for n in range(preds_gp.shape[0])]).mean(0)
        ax.plot(t_eval, m, color=GP_COLOR, ls=GP_LS, lw=1.4, label="GP")
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
        ax.legend(fontsize="small"); fig.tight_layout()
        figs.append(fig)
    return figs


def fig_state_ensemble(t_eval, gt_all, preds_gp):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om  = ["Omega X",    "Omega Y",     "Omega Z"]
    gt_eul = np.array([rotmat_to_euler(gt_all[n, :, :9]) for n in range(gt_all.shape[0])])
    gt_om  = gt_all[:, :, 9:12]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    gt_em = gt_eul.mean(0); gt_es = gt_eul.std(0)
    gt_om_m = gt_om.mean(0); gt_om_s = gt_om.std(0)
    eul = np.array([rotmat_to_euler(preds_gp[n, :, :9]) for n in range(preds_gp.shape[0])])
    em = eul.mean(0); om_m = preds_gp[:, :, 9:12].mean(0)
    for i in range(3):
        axes[i, 0].plot(t_eval, gt_em[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 0].fill_between(t_eval, gt_em[:, i]-2*gt_es[:, i],
                                gt_em[:, i]+2*gt_es[:, i], color="black", alpha=0.15)
        axes[i, 0].plot(t_eval, em[:, i], color=GP_COLOR, ls=GP_LS, lw=1.4,
                        label="GP" if i == 0 else None)
        axes[i, 1].plot(t_eval, gt_om_m[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 1].fill_between(t_eval, gt_om_m[:, i]-2*gt_om_s[:, i],
                                gt_om_m[:, i]+2*gt_om_s[:, i], color="black", alpha=0.15)
        axes[i, 1].plot(t_eval, om_m[:, i], color=GP_COLOR, ls=GP_LS, lw=1.4,
                        label="GP" if i == 0 else None)
        axes[i, 0].set_ylabel(labels_ang[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel(labels_om[i]);  axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small")
    axes[0, 1].legend(fontsize="x-small")
    fig.suptitle("State trajectories — 10-traj ensemble means", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_state_single(t_eval, gt_single, traj_gp):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om  = ["Omega X",    "Omega Y",     "Omega Z"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    gt_eul = rotmat_to_euler(gt_single[:, :9])
    eul = rotmat_to_euler(traj_gp[:, :9])
    for i in range(3):
        axes[i, 0].plot(t_eval, gt_eul[:, i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 0].plot(t_eval, eul[:, i], color=GP_COLOR, ls=GP_LS, lw=1.4,
                        label="GP" if i == 0 else None)
        axes[i, 1].plot(t_eval, gt_single[:, 9+i], "k-", lw=2,
                        label="GT" if i == 0 else None)
        axes[i, 1].plot(t_eval, traj_gp[:, 9+i], color=GP_COLOR, ls=GP_LS, lw=1.4,
                        label="GP" if i == 0 else None)
        axes[i, 0].set_ylabel(labels_ang[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel(labels_om[i]);  axes[i, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)"); axes[2, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize="x-small")
    axes[0, 1].legend(fontsize="x-small")
    fig.suptitle("Single trajectory (traj 0) state", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_phase_portraits(gt_all, preds_gp):
    labels_ang = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    labels_om  = ["Omega X",    "Omega Y",     "Omega Z"]
    gt_eul = rotmat_to_euler(gt_all[0, :, :9])
    eul = rotmat_to_euler(preds_gp[0, :, :9])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i in range(3):
        axes[i].plot(gt_eul[:, i], gt_all[0, :, 9+i], "k-", lw=2,
                     label="GT" if i == 0 else None)
        axes[i].plot(eul[:, i], preds_gp[0, :, 9+i], color=GP_COLOR, ls=GP_LS,
                     lw=1.4, label="GP" if i == 0 else None)
        axes[i].set_xlabel(labels_ang[i]); axes[i].set_ylabel(labels_om[i])
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(fontsize="x-small")
    fig.suptitle("Phase portraits — single trajectory (traj 0)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_subnet_matrix(t_eval, sub_gp, gt_traj, comp_key, gt_val_fn, title):
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    gt_vals = gt_val_fn(gt_traj) if gt_val_fn else None
    arr = sub_gp.get(comp_key)
    for idx in range(9):
        r, c = divmod(idx, 3); ax = axes[r, c]
        if gt_vals is not None:
            ax.plot(t_eval, gt_vals[:, idx], "k:", lw=1.5,
                    label="GT" if idx == 0 else None)
        if arr is not None and arr.ndim == 3:
            ax.plot(t_eval, arr.reshape(arr.shape[0], 9)[:, idx],
                    color=GP_COLOR, ls=GP_LS, lw=1.0, alpha=0.9,
                    label="GP" if idx == 0 else None)
        ax.set_title(f"({r},{c})"); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize="x-small")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_subnet_V(t_eval, sub_gp, gt_traj, title="Potential energy V(q) along GT trajectory (traj 0)"):
    gt_v = GT_M * GT_G * GT_L * gt_traj[:, 8]
    fig, ax = _ax(title)
    ax.plot(t_eval, gt_v, "k:", lw=1.5, label="GT")
    v = sub_gp.get("V")
    if v is not None:
        ax.plot(t_eval, v - v.mean() + gt_v.mean(),
                color=GP_COLOR, ls=GP_LS, lw=1.0, alpha=0.9, label="GP (centred)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("V(q)")
    ax.legend(fontsize="small"); fig.tight_layout()
    return fig


def fig_B_mean_diag_evolution(stats):
    """Diagnostic: G should converge to I₃. Plot eval_g_loss curve prominently."""
    fig, ax = _ax("Eval g MSE — convergence of B(q) toward I₃")
    if stats is None or "eval_g_loss" not in stats or "eval_step" not in stats:
        ax.text(0.5, 0.5, "no eval_g_loss in stats", transform=ax.transAxes,
                ha="center", va="center")
        return fig
    x = np.asarray(stats["eval_step"]); y = np.asarray(stats["eval_g_loss"])
    ax.plot(x, y, color=GP_COLOR, lw=1.6, label="eval_g_loss")
    # Reference line at g_loss for G = 0.3·I₃: (1-0.3)²/3 ≈ 0.163
    ax.axhline(0.163, color="gray", ls=":", lw=1.2,
               label="G = 0.3·I₃  (0.163)")
    ax.axhline(0.0, color="black", ls=":", lw=1.0,
               label="G = I₃  (0.0)")
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("eval_g MSE")
    ax.legend(fontsize="small"); fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp_dir",    required=True,  help="ph_gp_ode run dir")
    ap.add_argument("--out_pdf",   required=True)
    ap.add_argument("--obs_label", default="",     help="title annotation (e.g. obs=0.01)")
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--g_x", type=float, default=None,
                    help="diagonal control gain G[0,0] used in GT env "
                         "(default: parsed from run dir name)")
    ap.add_argument("--g_y", type=float, default=None,
                    help="diagonal control gain G[1,1] used in GT env")
    ap.add_argument("--g_z", type=float, default=None,
                    help="diagonal control gain G[2,2] used in GT env")
    ap.add_argument("--u_x", type=float, default=0.0,
                    help="constant control input u[0] (ignored if --random_u)")
    ap.add_argument("--u_y", type=float, default=0.0,
                    help="constant control input u[1] (ignored if --random_u)")
    ap.add_argument("--u_z", type=float, default=0.0,
                    help="constant control input u[2] (ignored if --random_u)")
    ap.add_argument("--random_u", action="store_true",
                    help="sample u ~ U(-scale, scale) per axis per outer step "
                         "(same sequence shared between GT and GP rollouts)")
    ap.add_argument("--random_u_scale", type=float, default=2.0,
                    help="half-width for --random_u")
    ap.add_argument("--u_seed", type=int, default=12345,
                    help="seed for the random-u sequence (independent of --seed)")
    args = ap.parse_args()
    if args.random_u:
        print(f"rollout u: random  ~ U(-{args.random_u_scale}, {args.random_u_scale}) "
              f"per axis per outer step (u_seed={args.u_seed})")
    else:
        u_const = np.array([args.u_x, args.u_y, args.u_z], dtype=np.float64)
        print(f"rollout u: constant {tuple(u_const.tolist())}")

    parsed_g = _parse_g_from_dirname(args.gp_dir)
    g_diag = (
        parsed_g[0] if args.g_x is None else args.g_x,
        parsed_g[1] if args.g_y is None else args.g_y,
        parsed_g[2] if args.g_z is None else args.g_z,
    )
    varying_friction = _parse_varfric_from_dirname(args.gp_dir)
    print(f"GT g_diag = {g_diag}  (parsed from dirname: {parsed_g})")
    print(f"GT varying_friction = {varying_friction}  (parsed from dirname)")
    env_kw = dict(ENV_KW)
    env_kw["g_diag"] = g_diag
    env_kw["varying_friction"] = varying_friction

    gp_ckpt = find_gp_ckpt(args.gp_dir)
    assert gp_ckpt, f"No GP checkpoint in {args.gp_dir}"
    stats = load_stats(args.gp_dir)
    print(f"GP ckpt : {os.path.basename(gp_ckpt)}")
    print(f"stats   : {'loaded' if stats is not None else 'missing'}")

    rng = np.random.default_rng(args.seed)
    dt = ENV_KW["dt"]
    dt_sub = dt / N_SUBSTEPS
    sqrt_h = float(np.sqrt(dt_sub))
    dW_ensemble = [rng.normal(0.0, sqrt_h, (N_OUTER, N_SUBSTEPS, 3))
                   for _ in range(N_TRAJ_ENSEMBLE)]
    t_eval = np.arange(N_OUTER + 1) * dt

    ic_list = []
    for ti in range(N_TRAJ_ENSEMBLE):
        env = windy_pendulum_3d(seed=args.seed + ti, **env_kw)
        env.reset(seed=args.seed + ti)
        ic_list.append((env.R.copy(), env.omega.copy()))

    # Build per-trajectory u sequences (shared between GT and GP rollouts so
    # the comparison is fair). With --random_u each axis is independently drawn
    # from U(-scale, scale) at every outer step.
    if args.random_u:
        u_rng = np.random.default_rng(args.u_seed)
        s = args.random_u_scale
        u_seqs = u_rng.uniform(-s, s, (N_TRAJ_ENSEMBLE, N_OUTER, 3))
    else:
        u_seqs = np.broadcast_to(u_const[None, None, :],
                                  (N_TRAJ_ENSEMBLE, N_OUTER, 3)).copy()

    print(f"Rolling out {N_TRAJ_ENSEMBLE} GT trajectories ...")
    gt_all = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
    for ti in range(N_TRAJ_ENSEMBLE):
        env = windy_pendulum_3d(seed=args.seed + ti, **env_kw)
        env.reset(seed=args.seed + ti)
        gt_all[ti] = rollout_gt(env, ic_list[ti][0], ic_list[ti][1], dW_ensemble[ti], u_seqs[ti])

    print("Loading GP model ...")
    gp_model = load_gp_model(gp_ckpt)

    print("Rolling out GP model ...")
    preds_gp = np.zeros((N_TRAJ_ENSEMBLE, N_OUTER + 1, 12))
    for ti in range(N_TRAJ_ENSEMBLE):
        R_ti, om_ti = ic_list[ti]
        preds_gp[ti] = rollout_gp(gp_model, R_ti, om_ti, N_SUBSTEPS, N_OUTER, dt, u_seqs[ti])

    print("Evaluating subnets on GT traj 0 ...")
    sub_gp = eval_subnets_gp(gp_model, gt_all[0])

    gt_M_inv_flat = (1.0 / I_PERP) * np.eye(3)
    gt_B_flat = np.diag(np.asarray(g_diag, dtype=np.float64))

    def gt_M_vals(gt_traj):
        T = gt_traj.shape[0]
        return np.tile(gt_M_inv_flat.reshape(-1), (T, 1))

    def gt_B_vals(gt_traj):
        T = gt_traj.shape[0]
        return np.tile(gt_B_flat.reshape(-1), (T, 1))

    def gt_D_vals(gt_traj):
        if varying_friction:
            D_all = gt_D_varying(gt_traj, GT_FRICTION)
        else:
            T = gt_traj.shape[0]
            D_all = np.tile((GT_FRICTION * np.eye(3))[None], (T, 1, 1))
        return D_all.reshape(D_all.shape[0], 9)

    print(f"Writing PDF: {args.out_pdf}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_pdf)) or ".",
                exist_ok=True)
    obs = args.obs_label or os.path.basename(args.gp_dir)

    with PdfPages(args.out_pdf) as pdf:
        # Summary tables at 1s, 2s, full horizon
        for horizon_s in (1.0, 2.0, dt * N_OUTER):
            pdf.savefig(fig_summary_table(
                gt_all, preds_gp, stats, horizon_s, dt, obs))
            plt.close()

        # Training loss curves (individual)
        for key, title, ylab in (
            ("train_loss",     "Train total loss",  "loss"),
            ("train_nll",      "Train NLL",         "NLL"),
            ("train_kl_total", "Train KL total",    "KL"),
            ("train_l2_loss",  "Train L2 (ω)",      "L2"),
            ("train_geo_loss", "Train geodesic²",   "geo²"),
        ):
            pdf.savefig(fig_loss(stats, key, title, ylab)); plt.close()

        # Aux loss overlay (all four together)
        pdf.savefig(fig_loss_overlay(
            stats,
            [("train_L_power", "L_power (λ=0.1)"),
             ("train_L_V",     "L_V (λ=0.1)"),
             ("train_L_B",     "L_B (λ=0.1)"),
             ("train_L_D",     "L_D (λ=0.1)")],
            "Aux physics losses (weighted)", "aux loss"))
        plt.close()
        pdf.savefig(fig_loss_overlay(
            stats,
            [("train_L_power_raw", "L_power raw"),
             ("train_L_V_raw",     "L_V raw")],
            "Aux physics losses (raw, unweighted)", "aux loss raw"))
        plt.close()

        # Eval losses
        for key, title, ylab in (
            ("test_l2_loss",  "Test L2(ω)",      "L2"),
            ("test_geo_loss", "Test geodesic²",   "geo²"),
            ("eval_M_loss",   "Eval M⁻¹ MSE",     "MSE"),
            ("eval_V_loss",   "Eval V MSE",        "MSE"),
            ("eval_Dw_loss",  "Eval Dw MSE",       "MSE"),
        ):
            pdf.savefig(fig_loss(stats, key, title, ylab,
                                  x_key="eval_step" if key.startswith(("test","eval")) else None))
            plt.close()

        # g_loss with reference lines (diagnostic page)
        pdf.savefig(fig_B_mean_diag_evolution(stats)); plt.close()

        # Trajectory metrics
        for f in fig_traj_mse(t_eval, gt_all, preds_gp):
            pdf.savefig(f); plt.close(f)

        # Energy
        e1, e2 = fig_energy(t_eval, gt_all, preds_gp)
        pdf.savefig(e1); plt.close(e1)
        pdf.savefig(e2); plt.close(e2)

        # SO(3)
        for f in fig_so3(t_eval, gt_all, preds_gp):
            pdf.savefig(f); plt.close(f)

        # State trajectories
        pdf.savefig(fig_state_ensemble(t_eval, gt_all, preds_gp)); plt.close()
        pdf.savefig(fig_state_single(t_eval, gt_all[0], preds_gp[0])); plt.close()
        pdf.savefig(fig_phase_portraits(gt_all, preds_gp)); plt.close()

        # ── Subnet evolution along GT traj 0 ─────────────────────────────
        # Port-Hamiltonian dynamics admit a scalar gauge β:
        #   (M⁻¹, V, D, B)  →  (β·M⁻¹, V/β, D/β, B/β)
        # leaves trajectories invariant. We estimate β from M⁻¹ and overlay
        # both raw and β-corrected curves so the learned subnets can be
        # compared on the same physical scale as GT.
        beta = estimate_beta(sub_gp, gt_m_inv_scalar=1.0 / I_PERP)
        sub_gp_beta = apply_beta(sub_gp, beta)
        print(f"estimated β = {beta:.4f}  (M⁻¹·β, V/β, D/β, B/β)")

        pdf.savefig(fig_subnet_matrix(t_eval, sub_gp_beta, gt_all[0],
            "M", gt_M_vals,
            f"β·M⁻¹(q) along GT traj 0  (β = {beta:.4f})")); plt.close()
        d_kind = "varying friction" if varying_friction else f"fixed {GT_FRICTION}·I₃"
        pdf.savefig(fig_subnet_matrix(t_eval, sub_gp_beta, gt_all[0],
            "D", gt_D_vals,
            f"D(q,p)/β along GT traj 0  (GT = {d_kind}, β = {beta:.4f})")); plt.close()
        pdf.savefig(fig_subnet_matrix(t_eval, sub_gp_beta, gt_all[0],
            "B", gt_B_vals,
            f"B(q)/β along GT traj 0  (GT = diag{g_diag}, β = {beta:.4f})")); plt.close()
        pdf.savefig(fig_subnet_V(t_eval, sub_gp_beta, gt_all[0],
            f"V(q)/β along GT trajectory (traj 0)  (β = {beta:.4f})")); plt.close()

    print("Done.")


if __name__ == "__main__":
    main()

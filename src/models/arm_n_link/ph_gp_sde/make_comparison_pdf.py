"""Compare ph_gp_sde against ground truth on the n-link windy SO(3)^n arm.

Mirrors the structure and plotting conventions of
`src/models/3D_SO3_Windy_Pendulum/report_gen/make_comparison_pdf.py`:
a 10-trajectory ensemble for state/energy/SO(3) comparisons (mean ±2σ bands
with every member drawn faintly), per-entry matrix grids for M⁻¹/Dw/g, a scalar
V(q) page, phase portraits, and a 5-trajectory subnetwork-evolution suite.

Generalised from 1 link to n: the state grids get one row-block per link, and
the 3×3 matrix grids become 3n×3n.

**One** torque sequence and **one** Wiener path are drawn per report and reused
by ground truth and by every rollout; across the ensemble only the initial
condition varies. So the comparison is pathwise rather than merely
distributional, and the ensemble spread isolates sensitivity to the starting
state at a fixed forcing history.

Because the members share that forcing they are *correlated*: the ± printed on
the summary page is a spread over initial conditions, not an i.i.d. uncertainty
estimate, and no standard error should be derived from it.

Rollouts run for `N_OUTER` outer steps of dt = 0.05 s (10 s by default), and the
summary tables are evaluated on prefixes of that single rollout — exact, since
the integrator is causal, so a prefix is bit-identical to a shorter rollout
under the same forcing.

Output PDF pages (in order):
  A. Summary
     1.  Trajectory comparison table — horizon 1 s   (10-rollout ensemble)
     2.  Trajectory comparison table — horizon 2 s
     3.  Trajectory comparison table — horizon 5 s
     4.  Trajectory comparison table — horizon 10 s
     5.  Trajectory comparison table — the single rollout shown on pages 15/21

  B. Loss curves
     6.  Rotation NLL
     7.  Angular-rate NLL
     8.  Pseudo-likelihood NLL
     9.  Geodesic² per link
     10. MSE angular velocity
     11. Diffusion magnitude ‖Σ_θ‖_F (vs GT scale)
     12. Total KL
     13. Per-subnet KL (M, V, Dw, g, Sigma)

  C. Ensemble dynamics (10 GT trajectories, 10 model rollouts, matched u and dW)
     14. Hamiltonian energy E(t)             — GT vs ph_gp_sde (mean ±2σ)
     15. Hamiltonian energy — single trajectory
     16. SO(3) violation: |det(R) − 1|
     17. SO(3) violation: ‖RᵀR − I‖_F
     18. Trajectory geodesic² vs GT
     19. Trajectory MSE ω vs GT
     20. State trajectories (Euler angles + ω, per link) — mean ±2σ
     21. State trajectories — single trajectory
     22. Phase portraits (Euler angle vs ω, per link)

  D. Subnetwork evolution along 5 GT trajectories
     23. Inverse mass M⁻¹(q)  — 3n×3n grid (raw)
     24. Inverse mass β·M⁻¹(q) — 3n×3n grid (gauge-corrected)
     25. Dissipation Dw(q)/β  — 3n×3n grid
     26. Control gain g(q)/β  — 3n×3n grid
     27. Potential V(q)/β
     28. Diffusion Σ(q)/β     — 3n×3 grid
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, "..", "..", "..", ".."))
for p in (PKG_ROOT, PROJECT_ROOT, THIS_FILE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from envs.arm_nlink_SO3 import arm_nlink_physics as phys          # noqa: E402
from network import DissipativeArmHamSDE, KeyedArmModel           # noqa: E402
from utils.lie_integrator_nlink import (                          # noqa: E402
    lie_heun_sde_rollout_nlink, exp_so3_batch,
)

# ──────────────────────────────────────────────────────────────────────────
# Constants  (same conventions as the single-pendulum report)
# ──────────────────────────────────────────────────────────────────────────

N_SUBSTEPS = 10
N_OUTER = 200                    # 200 outer steps of dt=0.05 → 10 s
# Summary-table horizons, in seconds. Each gets its own page; the metrics are
# aggregated over 0..t, so the shortest one is the regime the model was trained
# on and the longest is a pure extrapolation test. Horizons beyond
# `--n_outer * dt` are skipped with a note rather than silently truncated.
TABLE_HORIZONS_S = (1.0, 2.0, 5.0, 10.0)
N_TRAJ_ENSEMBLE = 10             # ensemble for state / energy / SO(3) plots
N_SUB_TRAJ = 5                   # trajectories for subnet-evolution plots
N_BETA_SAMPLES = 200             # configurations used to fit the scale gauge

GP_COLOR = "#d62728"             # ph_gp_sde — red, as in the pendulum report
GT_COLOR = "black"


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────

def load_run(run_dir):
    with open(os.path.join(run_dir, "history.pkl"), "rb") as f:
        meta = pickle.load(f)
    a = meta["args"]
    dtype = jnp.float64 if a.get("float64") else jnp.float32

    # Every architecture-affecting flag must be mirrored here, or the skeleton's
    # pytree will not match the checkpoint's and deserialisation fails deep
    # inside equinox with an opaque leaf-path error. `.get` with the old default
    # keeps runs that predate a flag loadable.
    # M/D/g became physics-structured, so a pre-structured checkpoint holds GP
    # weights where the skeleton now holds physical parameters. Deserialisation
    # would fail deep inside equinox; say why instead.
    if not a.get("structured_subnets", False):
        raise RuntimeError(
            f"{run_dir} predates the physics-structured M/D/g subnets "
            f"(no 'structured_subnets' marker in its saved args). Its M_net, "
            f"Dw_net and g_net are GPs, which this code no longer builds, so "
            f"the checkpoint cannot be loaded. Re-train, or check out the "
            f"commit that produced the run to regenerate its report."
        )

    skeleton = DissipativeArmHamSDE(
        key=jax.random.PRNGKey(a["seed"]), n=a["n"], hidden_dim=a["hidden_dim"],
        friction=not a.get("no_friction", False),
        relative_inputs=a.get("relative_inputs", False),
        init_sigma_obs_omega=a["obs_noise_std"],
        sigma_mode=a.get("sigma_mode", "full"),
        # Static, so it only affects gradients at train time — but mirror it
        # anyway so the rebuilt skeleton states the truth about its run.
        sigma_detach_rollout=a.get("sigma_detach_rollout", False),
        variational_core=a.get("variational_core", False),
        core_prior_std=a.get("core_prior_std", 1.0),
        gp_core=a.get("gp_core", False),
        # Runs predating the inertia floor were trained without it; runs
        # predating the gauge anchors were trained with the gauge free.
        i_epsilon=a.get("i_epsilon", 0.0),
        anchor_m1=a.get("anchor_m1", None),
        anchor_trace=a.get("anchor_trace", None), dtype=dtype)
    try:
        model = eqx.tree_deserialise_leaves(
            os.path.join(run_dir, "model.eqx"), skeleton)
    except Exception as exc:                      # noqa: BLE001
        raise RuntimeError(
            f"could not load {run_dir}/model.eqx into the skeleton built from "
            f"its own args (n={a['n']}, hidden_dim={a['hidden_dim']}, "
            f"relative_inputs={a.get('relative_inputs', False)}, "
            f"sigma_mode={a.get('sigma_mode', 'full')}). This is an "
            f"architecture mismatch: some flag that changes the model's shape "
            f"is not being mirrored in load_run()."
        ) from exc

    params = phys.uniform_chain_params(
        a["n"], d=a["friction_coeff"], kappa=a.get("air_drag", 0.0),
        varying_friction=a.get("varying_friction", False),
        g_diag=tuple(a.get("g_diag", (1.0, 1.0, 1.0))))
    return model, meta, params, dtype


# ──────────────────────────────────────────────────────────────────────────
# Rollouts  (GT and model driven by the SAME Wiener path)
# ──────────────────────────────────────────────────────────────────────────

@eqx.filter_jit
def _gt_scan(params, R0, w0, u_seq, dW, sigma, h):
    r"""Whole GT rollout as one `lax.scan`, so it compiles once.

    A Python loop over `phys.lie_heun_substeps` would re-trace on every outer
    step (that function is not jitted), which costs minutes rather than seconds.
    Semantics are unchanged: $p$ is carried inside the substeps and converted
    back to $\omega$ at each outer-step boundary, exactly as the environment does.
    """
    n = R0.shape[0]

    def outer(carry, xs):
        R, p = carry
        u_t, dW_t = xs
        R_new, p_new = phys.lie_heun_substeps(
            params, R, p, u_t.reshape(n, 3), h, jnp.zeros(3), sigma, dW_t)
        w_new = phys.omega_from_momentum(params, R_new, p_new)
        return (R_new, p_new), jnp.concatenate([R_new.reshape(-1),
                                                w_new.reshape(-1)])

    p0 = phys.momentum_from_omega(params, R0, w0)
    _, traj = jax.lax.scan(outer, (R0, p0), (u_seq, dW))
    x0 = jnp.concatenate([R0.reshape(-1), w0.reshape(-1)])
    return jnp.concatenate([x0[None], traj], axis=0)


def rollout_gt(params, R0, w0, u_seq, dW, sigma, h):
    """Ground-truth rollout via the analytic physics. Returns (T+1, 12n)."""
    return np.asarray(_gt_scan(params, jnp.asarray(R0), jnp.asarray(w0),
                               jnp.asarray(u_seq), jnp.asarray(dW),
                               jnp.asarray(sigma), jnp.asarray(h)))


@eqx.filter_jit
def _model_scan(model, x0, u_seq, h, dW):
    keyed = KeyedArmModel(model=model, keys={}, inference_mode=True)
    return lie_heun_sde_rollout_nlink(keyed, x0, u_seq, h, dW)


def rollout_model(model, R0, w0, u_seq, dW, h, dtype):
    """ph_gp_sde rollout with the identical Wiener path. Returns (T+1, 12n)."""
    x0 = jnp.asarray(np.concatenate([np.asarray(R0).reshape(-1),
                                     np.asarray(w0).reshape(-1)]), dtype=dtype)
    return np.asarray(_model_scan(model, x0, jnp.asarray(u_seq, dtype),
                                  jnp.asarray(h, dtype),
                                  jnp.asarray(dW, dtype)))


# ──────────────────────────────────────────────────────────────────────────
# Subnet evaluation
# ──────────────────────────────────────────────────────────────────────────

@eqx.filter_jit
def _subnets_model(model, q):
    return (jax.vmap(model.M_inv)(q),
            jax.vmap(lambda x: model._call(model.V_net, x, None)[0])(q),
            jax.vmap(lambda x: model._call(model.Dw_net,
                                           x, None))(q),
            jax.vmap(lambda x: model._call(model.g_net,
                                           x, None))(q),
            jax.vmap(model.Sigma)(q))


def eval_subnets_model(model, traj, n, dtype):
    """Evaluate every learned subnet along a trajectory. Arrays keyed like the
    single-pendulum report: M (T,d,d), V (T,), D (T,d,d), B (T,d,m), Xi (T,d,3)."""
    M, V, D, B, Xi = _subnets_model(model, jnp.asarray(traj[:, :9 * n], dtype=dtype))
    return {"M": np.asarray(M), "V": np.asarray(V), "D": np.asarray(D),
            "B": np.asarray(B), "Xi": np.asarray(Xi)}


@eqx.filter_jit
def _subnets_gt(params, R, sigma):
    return (jax.vmap(lambda r: jnp.linalg.inv(phys.mass_matrix(params, r)))(R),
            jax.vmap(lambda r: phys.potential(params, r))(R),
            jax.vmap(lambda r: phys.dissipation_matrix(
                params, r, jnp.zeros(r.shape[:1] + (3,))))(R),
            jax.vmap(lambda Rk: phys.input_map(params, Rk))(R),
            sigma * jax.vmap(lambda r: phys.wind_map(params, r))(R))


def eval_subnets_gt(params, traj, n, sigma):
    """Analytic port-Hamiltonian terms along the same trajectory."""
    R = jnp.asarray(traj[:, :9 * n]).reshape(-1, n, 3, 3)
    M, V, D, B, Xi = _subnets_gt(params, R, jnp.asarray(sigma))
    return {"M": np.asarray(M), "V": np.asarray(V), "D": np.asarray(D),
            "B": np.asarray(B), "Xi": np.asarray(Xi)}


def estimate_beta(model, params, R_samples, dtype):
    r"""Port-Hamiltonian scale gauge, fitted from M⁻¹.

    The dynamics are invariant under (M, V, D, B, Σ) → β·(M, V, D, B, Σ) with
    p → β·p, because ∂H_β/∂p_β = M_β⁻¹p_β = M⁻¹p — and ω is all the data sees.
    We fit β so that β·M_β⁻¹ ≈ M_GT⁻¹, exactly as `estimate_beta` does in the
    single-pendulum report, generalised from the isotropic scalar target to
    full-matrix least squares (M_GT⁻¹ is not a multiple of I for n ≥ 2):

        β* = argmin_β ‖β·M̂⁻¹ − M_GT⁻¹‖²_F

    Recovery for plotting:  M⁻¹ → β·M⁻¹,  V, D, B, Σ → ·/β.
    """
    Mh, Mg = [], []
    for R in R_samples:
        q = jnp.asarray(np.asarray(R).reshape(-1), dtype=dtype)
        Mh.append(np.asarray(model.M_inv(q)))
        Mg.append(np.asarray(jnp.linalg.inv(phys.mass_matrix(params, jnp.asarray(R)))))
    Mh, Mg = np.asarray(Mh), np.asarray(Mg)
    denom = float(np.sum(Mh * Mh))
    return float(np.sum(Mh * Mg) / denom) if denom > 1e-30 else 1.0


def _raw_gauge(beta):
    """True when the report is showing uncorrected subnets (β pinned to 1)."""
    return abs(float(beta) - 1.0) < 1e-12


def _gtag(beta):
    return "" if _raw_gauge(beta) else "β·"


def _gdiv(beta):
    return "" if _raw_gauge(beta) else "/β"


def _gsuf(beta):
    return ("  (RAW — no gauge correction)" if _raw_gauge(beta)
            else f"  (β={beta:.3f})")


def apply_beta(comp, beta):
    """Gauge-correct a subnet dict: M⁻¹·β, and V, D, B, Xi divided by β."""
    out = dict(comp)
    if "M" in out:
        out["M"] = out["M"] * beta
    for k in ("V", "D", "B", "Xi"):
        if k in out:
            out[k] = out[k] / beta
    return out


# ──────────────────────────────────────────────────────────────────────────
# Derived quantities
# ──────────────────────────────────────────────────────────────────────────

def rotmat_to_euler(R_flat):
    """Vectorised batch ZYX Euler conversion, returns angles in (T, 3) rad."""
    Rs = np.asarray(R_flat).reshape(-1, 3, 3)
    R00, R10, R20 = Rs[:, 0, 0], Rs[:, 1, 0], Rs[:, 2, 0]
    R21, R22, R12, R11 = Rs[:, 2, 1], Rs[:, 2, 2], Rs[:, 1, 2], Rs[:, 1, 1]
    sy = np.sqrt(R00 ** 2 + R10 ** 2)
    near = sy < 1e-6
    roll = np.where(near, np.arctan2(-R12, R11), np.arctan2(R21, R22))
    pitch = np.arctan2(-R20, sy)
    yaw = np.where(near, 0.0, np.arctan2(R10, R00))
    return np.stack([roll, pitch, yaw], axis=-1)


def euler_all_links(traj, n):
    """(T, n, 3) ZYX Euler angles, one triple per link."""
    return np.stack([rotmat_to_euler(traj[:, 9 * i:9 * (i + 1)])
                     for i in range(n)], axis=1)


def get_energy(params, traj, n):
    """GT Hamiltonian H = ½ ωᵀM(q)ω + V(q) along a trajectory."""
    R = jnp.asarray(traj[:, :9 * n]).reshape(-1, n, 3, 3)
    w = jnp.asarray(traj[:, 9 * n:]).reshape(-1, n, 3)
    return np.asarray(jax.vmap(lambda r, o: phys.total_energy(params, r, o))(R, w))


def hamiltonian_from_subnets(traj, sub, n):
    """Model-self H = ½ pᵀM⁻¹p + V, with p = solve(M⁻¹, ω)."""
    w = traj[:, 9 * n:]
    M_inv, V = sub["M"], sub["V"]
    p = np.linalg.solve(M_inv, w[..., None])[..., 0]
    return 0.5 * np.einsum("ti,tij,tj->t", p, M_inv, p) + V


def so3_orth_residual(traj, n):
    """max over links of ‖RᵀR − I‖_F."""
    R = traj[:, :9 * n].reshape(-1, n, 3, 3)
    d = np.einsum("tnji,tnjk->tnik", R, R) - np.eye(3)[None, None]
    return np.linalg.norm(d.reshape(-1, n, 9), axis=-1).max(axis=-1)


def so3_det_residual_abs(traj, n):
    R = traj[:, :9 * n].reshape(-1, n, 3, 3)
    return np.abs(np.linalg.det(R) - 1.0).max(axis=-1)


def geodesic_sq_all(pred, gt, n):
    """(T,) mean-over-links squared geodesic angle between two trajectories."""
    Rp = pred[:, :9 * n].reshape(-1, n, 3, 3)
    Rg = gt[:, :9 * n].reshape(-1, n, 3, 3)
    M = np.einsum("tnji,tnjk->tnik", Rp, Rg)
    c = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return (np.arccos(c) ** 2).mean(axis=-1)


def omega_sq_all(pred, gt, n):
    r"""Mean-over-links squared angular-velocity error, $\tfrac1n\lVert\Delta\omega\rVert^2$.

    Averaged over links, **not** summed. Two reasons:

    1. :func:`geodesic_sq_all` already averages over links, so summing here
       would put the report's two headline metrics on different scales.
    2. The single-pendulum reports in ``3D_SO3_Windy_Pendulum/report_gen``
       (``omega_sq``) sum over the 3 components of a *single* link, i.e. $n=1$.
       Summing over $3n$ makes the number grow with $n$ and silently breaks any
       comparison against those results — a 2-link arm looks $2\times$ worse
       than an identically-accurate pendulum.

    Broadcasts over leading axes, so it accepts one trajectory `(T, D)` or a
    whole ensemble `(N, T, D)`.
    """
    return np.sum((pred[..., 9 * n:] - gt[..., 9 * n:]) ** 2, axis=-1) / n


# ──────────────────────────────────────────────────────────────────────────
# Plot helpers  (mirroring the single-pendulum report)
# ──────────────────────────────────────────────────────────────────────────

def _smooth(y, w=5):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < w + 1:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    half = w // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def fig_one_curve(x, y, title, ylab, logy=True, smooth=True, hline=None,
                  hlabel=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, _smooth(y) if smooth else y, lw=1.2, color=GP_COLOR,
            label="ph_gp_sde")
    if hline is not None:
        ax.axhline(hline, color="k", linestyle="--", lw=1.5,
                   label=hlabel or "GT")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(fontsize="x-small")
    fig.tight_layout()
    return fig


def fig_n_curves(curves, title, ylab, logy=True, smooth=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    for x, y, label, color in curves:
        if y is None or len(y) == 0:
            continue
        ax.plot(x, _smooth(y) if smooth else y, lw=1.2, label=label, color=color)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(fontsize="x-small")
    fig.tight_layout()
    return fig


def fig_energy_ensemble(t, gt_e_all, model_e):
    """gt_e_all: (N, T). model_e: dict[name -> (mean, std, color)]."""
    fig, ax = plt.subplots(figsize=(10, 6))
    gt_m, gt_s = gt_e_all.mean(0), gt_e_all.std(0)
    ax.plot(t, gt_m, "k-", lw=2, label="GT Mean")
    ax.fill_between(t, gt_m - 2 * gt_s, gt_m + 2 * gt_s,
                    color="black", alpha=0.15, label="GT ±2σ")
    for name, (em, es, col) in model_e.items():
        ax.plot(t, em, color=col, lw=2, label=name)
        ax.fill_between(t, em - 2 * es, em + 2 * es, color=col, alpha=0.2,
                        label=f"{name} ±2σ")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.set_title("Hamiltonian Energy (10-traj ensemble)")
    ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_energy_single(t, gt_e, model_e):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, gt_e, "k-", lw=2, label="GT")
    for name, (e, col) in model_e.items():
        ax.plot(t, e, color=col, lw=2, label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.set_title("Hamiltonian Energy — single trajectory")
    ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_error_band(t, err_all, title, ylab, color=GP_COLOR, floor_all=None,
                   saturation=None):
    r"""Model error vs time, against the **intrinsic chaos floor**.

    On a chaotic system a rising error curve is ambiguous on its own: some of it
    is model error and some is the system's own sensitivity to the unobserved
    wind path. `floor_all` is the same quantity computed between **two
    ground-truth rollouts** from an identical state driven by identical torques,
    differing only in the wind realisation — so it is the error a *perfect*
    model would still incur. Model error below that line is not meaningful;
    above it is.

    `saturation` (geodesic pages only) marks the mean geodesic² between two
    independent uniform-random rotations. Crossing it means the prediction is
    worse than a random guess, which — since ground-truth states are
    concentrated rather than uniform — indicates the model has settled into a
    *different* region, not merely a decorrelated one.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    m, s = err_all.mean(0), err_all.std(0)
    for ni in range(err_all.shape[0]):
        ax.plot(t, err_all[ni], color=color, alpha=0.15, lw=0.8)
    ax.plot(t, m, color=color, lw=2, label="ph_gp_sde (mean)")
    ax.fill_between(t, np.maximum(m - 2 * s, 1e-12), m + 2 * s,
                    color=color, alpha=0.2, label="±2σ")

    if floor_all is not None:
        fm, fs = floor_all.mean(0), floor_all.std(0)
        ax.plot(t, fm, "k--", lw=2,
                label="GT vs GT, different wind (chaos floor)")
        ax.fill_between(t, np.maximum(fm - 2 * fs, 1e-12), fm + 2 * fs,
                        color="black", alpha=0.12)
    if saturation is not None:
        ax.axhline(saturation, color="0.45", ls=":", lw=1.8,
                   label=f"saturation (random rotations) = {saturation:.2f}")

    ax.set_yscale("log")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def random_rotation_saturation(n_samples: int = 20000, seed: int = 5) -> float:
    r"""Mean geodesic² between two independent uniform-random rotations.

    The ceiling for the rotation error: two unrelated attitudes score this on
    average (the hard maximum is $\pi^2\approx9.87$, attained only at antipodes).
    """
    ka, kb = jax.random.split(jax.random.PRNGKey(seed))
    Ra = exp_so3_batch(jax.random.normal(ka, (n_samples, 3), dtype=jnp.float64))
    Rb = exp_so3_batch(jax.random.normal(kb, (n_samples, 3), dtype=jnp.float64))
    M = jnp.einsum("tji,tjk->tik", Ra, Rb)
    c = jnp.clip((jnp.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return float(jnp.mean(jnp.arccos(c) ** 2))


def fig_so3_violation(t, gt_all, model_metric, ylab, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    if gt_all is not None:
        ax.plot(t, gt_all.mean(0), "k--", lw=2, label="GT (mean)")
    for name, (mm, ms, col) in model_metric.items():
        ax.plot(t, mm, color=col, label=f"{name} (mean)")
        ax.plot(t, ms, color=col, linestyle=":", label=f"{name} (single)")
    ax.set_yscale("log"); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_state_ensemble(t, gt_eul, gt_om, model_states, n):
    """(3n)×2 grid: rows = link×axis, cols = Euler angle / ω."""
    fig, axes = plt.subplots(3 * n, 2, figsize=(14, 4 * n + 8), sharex=True,
                             squeeze=False)
    ang_lbl = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    om_lbl = ["Omega X", "Omega Y", "Omega Z"]

    gem, ges = gt_eul.mean(0), gt_eul.std(0)
    gom, gos = gt_om.mean(0), gt_om.std(0)

    for li in range(n):
        for i in range(3):
            r = li * 3 + i
            for ni in range(gt_eul.shape[0]):
                axes[r][0].plot(t, gt_eul[ni, :, li, i], color="black",
                                alpha=0.15, lw=0.8)
                axes[r][1].plot(t, gt_om[ni, :, li, i], color="black",
                                alpha=0.15, lw=0.8)
            first = (r == 0)
            axes[r][0].plot(t, gem[:, li, i], "k-", lw=2,
                            label="GT Mean" if first else None)
            axes[r][0].fill_between(t, gem[:, li, i] - 2 * ges[:, li, i],
                                    gem[:, li, i] + 2 * ges[:, li, i],
                                    color="black", alpha=0.15,
                                    label="GT ±2σ" if first else None)
            axes[r][1].plot(t, gom[:, li, i], "k-", lw=2,
                            label="GT Mean" if first else None)
            axes[r][1].fill_between(t, gom[:, li, i] - 2 * gos[:, li, i],
                                    gom[:, li, i] + 2 * gos[:, li, i],
                                    color="black", alpha=0.15,
                                    label="GT ±2σ" if first else None)

            for name, (eul, om, col) in model_states.items():
                em, es = eul.mean(0), eul.std(0)
                mm, ms = om.mean(0), om.std(0)
                for ni in range(eul.shape[0]):
                    axes[r][0].plot(t, eul[ni, :, li, i], color=col,
                                    alpha=0.15, lw=0.8)
                    axes[r][1].plot(t, om[ni, :, li, i], color=col,
                                    alpha=0.15, lw=0.8)
                axes[r][0].plot(t, em[:, li, i], color=col, lw=2,
                                label=name if first else None)
                axes[r][0].fill_between(t, em[:, li, i] - 2 * es[:, li, i],
                                        em[:, li, i] + 2 * es[:, li, i],
                                        color=col, alpha=0.2,
                                        label=f"{name} ±2σ" if first else None)
                axes[r][1].plot(t, mm[:, li, i], color=col, lw=2,
                                label=name if first else None)
                axes[r][1].fill_between(t, mm[:, li, i] - 2 * ms[:, li, i],
                                        mm[:, li, i] + 2 * ms[:, li, i],
                                        color=col, alpha=0.2,
                                        label=f"{name} ±2σ" if first else None)

            axes[r][0].set_ylabel(f"L{li+1} {ang_lbl[i]}")
            axes[r][1].set_ylabel(f"L{li+1} {om_lbl[i]}")
            axes[r][0].grid(True, alpha=0.3); axes[r][1].grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Time (s)"); axes[-1][1].set_xlabel("Time (s)")
    axes[0][0].legend(fontsize="x-small"); axes[0][1].legend(fontsize="x-small")
    fig.suptitle(f"State Trajectories ({N_TRAJ_ENSEMBLE}-traj ensemble)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    return fig


def fig_state_single(t, gt_traj, model_single, n):
    fig, axes = plt.subplots(3 * n, 2, figsize=(14, 4 * n + 8), sharex=True,
                             squeeze=False)
    ang_lbl = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    om_lbl = ["Omega X", "Omega Y", "Omega Z"]
    gt_eul = euler_all_links(gt_traj, n)
    gt_om = gt_traj[:, 9 * n:].reshape(-1, n, 3)

    for li in range(n):
        for i in range(3):
            r = li * 3 + i
            first = (r == 0)
            axes[r][0].plot(t, gt_eul[:, li, i], "k-", lw=2,
                            label="GT" if first else None)
            axes[r][1].plot(t, gt_om[:, li, i], "k-", lw=2,
                            label="GT" if first else None)
            for name, (tr, col) in model_single.items():
                eul = euler_all_links(tr, n)
                om = tr[:, 9 * n:].reshape(-1, n, 3)
                axes[r][0].plot(t, eul[:, li, i], color=col, lw=2,
                                label=name if first else None)
                axes[r][1].plot(t, om[:, li, i], color=col, lw=2,
                                label=name if first else None)
            axes[r][0].set_ylabel(f"L{li+1} {ang_lbl[i]}")
            axes[r][1].set_ylabel(f"L{li+1} {om_lbl[i]}")
            axes[r][0].grid(True, alpha=0.3); axes[r][1].grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Time (s)"); axes[-1][1].set_xlabel("Time (s)")
    axes[0][0].legend(fontsize="x-small"); axes[0][1].legend(fontsize="x-small")
    fig.suptitle("1 Trajectory State Trajectories", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    return fig


def fig_phase_portraits(gt_eul, gt_om, model_states, n):
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)
    ang_lbl = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    om_lbl = ["Omega X", "Omega Y", "Omega Z"]
    for li in range(n):
        for i in range(3):
            ax = axes[li][i]
            first = (li == 0 and i == 0)
            for ni in range(gt_eul.shape[0]):
                ax.plot(gt_eul[ni, :, li, i], gt_om[ni, :, li, i],
                        color="black", alpha=0.15, lw=0.8)
            ax.plot(gt_eul.mean(0)[:, li, i], gt_om.mean(0)[:, li, i], "k-",
                    lw=2, label="GT Mean" if first else None)
            for name, (eul, om, col) in model_states.items():
                for ni in range(eul.shape[0]):
                    ax.plot(eul[ni, :, li, i], om[ni, :, li, i], color=col,
                            alpha=0.15, lw=0.8)
                ax.plot(eul.mean(0)[:, li, i], om.mean(0)[:, li, i], color=col,
                        lw=2, label=name if first else None)
            ax.set_xlabel(f"L{li+1} {ang_lbl[i]}")
            ax.set_ylabel(f"L{li+1} {om_lbl[i]}")
            ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize="x-small")
    fig.suptitle("Phase Portraits (Angle vs Omega)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    return fig


# ── Subnet evolution along multiple trajectories ──────────────────────────

def plot_matrix_grid_multi(t, comps_model, comps_gt, key, title, rows, cols):
    """rows×cols grid, one panel per matrix entry, all N_SUB_TRAJ overlaid.

    GT is drawn per-trajectory as a dashed black line (it is state-dependent
    here, unlike the constant M⁻¹ of the single pendulum).
    """
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols + 2, 2.4 * rows + 2),
                             squeeze=False)
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            first = (r == 0 and c == 0)
            for ti in range(len(comps_gt)):
                ax.plot(t, comps_gt[ti][key][:, r, c], "k--", lw=1.0, alpha=0.6,
                        label="GT" if (first and ti == 0) else None)
            for ti in range(len(comps_model)):
                ax.plot(t, comps_model[ti][key][:, r, c], color=GP_COLOR,
                        alpha=0.7, lw=1.2,
                        label="ph_gp_sde" if (first and ti == 0) else None)
            ax.set_title(f"({r},{c})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if r == rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)
    axes[0][0].legend(fontsize="x-small")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    return fig


def plot_potential_multi(t, comps_model, comps_gt, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for ti in range(len(comps_gt)):
        ax.plot(t, comps_gt[ti]["V"] - comps_gt[ti]["V"].mean(), "k--", lw=1.0,
                alpha=0.6, label="GT" if ti == 0 else None)
    for ti in range(len(comps_model)):
        v = comps_model[ti]["V"]
        ax.plot(t, v - v.mean(), color=GP_COLOR, alpha=0.7, lw=1.2,
                label="ph_gp_sde" if ti == 0 else None)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("V(q) − mean")
    ax.set_title(title + "   (centred: H is defined up to a constant)")
    ax.legend(fontsize="x-small"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Summary table ─────────────────────────────────────────────────────────

def _compute_table_metrics(params, gt_all, pred_all, n, steps=None):
    """Summary metrics over the first `steps` outer steps (all of them if None).

    Every row aggregates over 0..steps: the "Traj-mean" rows average that
    window and the "Final-step" rows read its last entry. Slicing here rather
    than re-rolling is exact — the rollouts are causal, so a prefix of a long
    rollout is bit-identical to a short rollout with the same forcing.
    """
    N = gt_all.shape[0]
    if steps is not None:
        gt_all = gt_all[:, :steps + 1]
        pred_all = pred_all[:, :steps + 1]
    geo_sq = np.stack([geodesic_sq_all(pred_all[i], gt_all[i], n)
                       for i in range(N)])
    om_sq = omega_sq_all(pred_all, gt_all, n)

    det_max = np.stack([so3_det_residual_abs(pred_all[i], n) for i in range(N)]).max(1)
    orth_max = np.stack([so3_orth_residual(pred_all[i], n) for i in range(N)]).max(1)

    e_gt = np.stack([get_energy(params, gt_all[i], n) for i in range(N)])
    e_pr = np.stack([get_energy(params, pred_all[i], n) for i in range(N)])
    e_err = np.abs(e_pr - e_gt).mean(axis=1)

    def ms(x):
        # A single rollout has no spread to report; printing "± 0.000e+00"
        # would read as a measured uncertainty of zero.
        if np.size(x) == 1:
            return f"{float(np.mean(x)):.3e}"
        return f"{float(np.mean(x)):.3e}  ±  {float(np.std(x)):.3e}"

    return {
        "Traj-mean geodesic² (rad²)": ms(geo_sq.mean(1)),
        "Traj-mean ‖Δω‖²/link (rad²/s²)": ms(om_sq.mean(1)),
        "Final-step geodesic² (rad²)": ms(geo_sq[:, -1]),
        "Final-step ‖Δω‖²/link (rad²/s²)": ms(om_sq[:, -1]),
        "Mean |ΔE| over time (J)": ms(e_err),
        "Max |det(R)−1|": ms(det_max),
        "Max ‖RᵀR−I‖_F": ms(orth_max),
    }


def fig_comparison_table(params, gt_all, models, n, beta, subnet_err,
                         u_label="random u", steps=None, horizon_s=None):
    per_model = {name: _compute_table_metrics(params, gt_all, tr, n, steps)
                 for name, tr in models.items()}
    metric_names = list(next(iter(per_model.values())).keys())
    model_names = list(per_model.keys())
    cell_text = [[per_model[m][k] for m in model_names] for k in metric_names]

    # Gauge-corrected subnet errors appended as extra rows.
    for k, v in subnet_err.items():
        metric_names.append(f"Subnet rel. error: {k}")
        cell_text.append([f"{v:.4f}"] + [""] * (len(model_names) - 1))

    N = gt_all.shape[0]
    single = N == 1
    gauge_note = ("subnet errors RAW — no scale gauge applied (β pinned to 1)"
                  if abs(beta - 1.0) < 1e-12 else
                  f"subnet errors gauge-corrected with β = {beta:.4f}")

    hz = "" if horizon_s is None else f"horizon {horizon_s:g} s — "
    fig, ax = plt.subplots(figsize=(14, 1.8 + 0.55 * len(metric_names)))
    ax.axis("off")
    ax.set_title(
        (f"{hz}single rollout vs GT (n={n} links)"
         if single else
         f"{hz}{N}-rollout ensemble vs GT (n={n} links)")
        + f"\ncontrol: {u_label}\n("
        + ("single trajectory, no aggregation;  " if single
           else "per-trajectory aggregates: mean ± std;  ")
        + gauge_note + ")",
        fontsize=13, fontweight="bold", pad=14)

    table = ax.table(cellText=cell_text, rowLabels=metric_names,
                     colLabels=model_names, cellLoc="center", rowLoc="left",
                     loc="center", colWidths=[0.34] * len(model_names))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)
    for j in range(len(model_names)):
        table[0, j].set_text_props(weight="bold")

    fig.text(0.5, 0.02,
             ("GT and ph_gp_sde are driven by the identical torque sequence and "
              "Wiener path, so the comparison is pathwise, not merely "
              "distributional.\n"
              if single else
              "One torque sequence and one Wiener path drive GT and all "
              f"{N} rollouts; only the initial condition varies. The comparison is "
              "pathwise, not merely distributional.\n"
              "± is therefore the spread over initial conditions at fixed forcing, "
              "not an i.i.d. uncertainty estimate.\n")
             + "Subnet errors are relative Frobenius after fitting the scale gauge β "
               "(V additionally centred).",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def build_u_sequence(u_const, T, n, random_u, random_u_scale, rng):
    r"""Control sequence for one trajectory, `(T, 3n)`.

    `u_const=None` reproduces the training distribution: a fresh draw
    $u_t\sim\mathcal U(-s,s)^{3n}$ per outer step. Otherwise the control is held
    **constant** at the given value for every step and every trajectory, which
    isolates the drift from the torque excitation — useful for seeing how the
    model behaves under a steady load rather than under persistent excitation.

    A constant control is broadcast the same way the dataset generator broadcasts
    its ``us`` entries: length 1 fills all $3n$ components, length 3 is tiled
    once per joint, length $3n$ is used verbatim.
    """
    if u_const is None:
        if not random_u:
            return np.zeros((T, 3 * n))
        return rng.uniform(-random_u_scale, random_u_scale, size=(T, 3 * n))

    v = np.asarray(u_const, dtype=np.float64).reshape(-1)
    if v.size == 1:
        v = np.full(3 * n, float(v[0]))
    elif v.size == 3:
        v = np.tile(v, n)
    elif v.size != 3 * n:
        raise ValueError(f"--u takes 1, 3 or {3 * n} values for n={n}; got {v.size}")
    return np.tile(v, (T, 1))


def _u_label(u_const, random_u, random_u_scale):
    """Human-readable description of the control regime, shown on page 1."""
    if u_const is None:
        return (f"random u ~ U(-{random_u_scale:g}, {random_u_scale:g}) per step"
                if random_u else "u = 0")
    v = np.asarray(u_const, dtype=float).reshape(-1)
    return (f"constant u = {v[0]:g}" if v.size == 1
            else "constant u = [" + ", ".join(f"{x:g}" for x in v) + "]")


def _default_pdf_name(u_const, no_gauge=False):
    """Keep constant-u and raw-gauge reports from overwriting each other."""
    suffix = "_rawgauge" if no_gauge else ""
    if u_const is None:
        return f"comparison_report{suffix}.pdf"
    v = np.asarray(u_const, dtype=float).reshape(-1)
    tag = "_".join(f"{x:g}" for x in v).replace("-", "n").replace(".", "p")
    return f"comparison_report_u{tag}{suffix}.pdf"


def main():
    ap = argparse.ArgumentParser(
        description="ph_gp_sde vs ground truth on the n-link SO(3)^n arm.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_outer", type=int, default=None,
                    help="rollout length in outer steps of dt=0.05. Default: "
                         "just long enough for the longest --horizons entry, so "
                         "`--horizons 1` rolls out 1 s rather than the full "
                         f"{N_OUTER * 0.05:g} s. Pass explicitly to override.")
    ap.add_argument("--horizons", type=float, nargs="+", metavar="SEC",
                    default=list(TABLE_HORIZONS_S),
                    help="summary-table horizons in seconds, one page each "
                         f"(default: {' '.join(f'{x:g}' for x in TABLE_HORIZONS_S)}). "
                         "`--horizons 1` gives a single 1 s page and, unless "
                         "--n_outer says otherwise, shortens the rollout to "
                         "match — the fast path for debugging.")
    ap.add_argument("--tables_only", action="store_true",
                    help="emit only the summary-table pages, skipping the loss "
                         "curves, ensemble dynamics and subnet grids. The 3n x 3n "
                         "grids dominate rendering time, so this is much faster.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--u", type=float, nargs="+", default=None, metavar="VAL",
                    help="hold the control CONSTANT instead of resampling it. "
                         "Give 1 value (broadcast to all 3n components), 3 "
                         "(tiled per joint, matching the datagen's `us`), or 3n "
                         "(used as-is). Omit for the default: a fresh random "
                         "torque per outer step at the run's own "
                         "random_u_scale.")
    ap.add_argument("--no_gauge", action="store_true",
                    help="pin beta = 1 instead of fitting it, i.e. compare the "
                         "RAW learned subnets against ground truth with no "
                         "scale correction. Useful for seeing where each subnet "
                         "actually sits, since the fitted beta is estimated "
                         "from M^-1 alone and is only the right correction for "
                         "subnets that share M^-1's gauge. Note the port-"
                         "Hamiltonian dynamics ARE invariant under "
                         "(M,V,D,g,Sigma) -> beta(.) with p -> beta*p, so a raw "
                         "mismatch is not by itself an error -- read this "
                         "alongside the gauge-corrected run, not instead of it.")
    args = ap.parse_args()

    out = args.out or os.path.join(
        args.run_dir, _default_pdf_name(args.u, args.no_gauge))
    model, meta, params, dtype = load_run(args.run_dir)
    a = meta["args"]
    n = a["n"]
    sigma = a["wind_force_std"]
    dt = 0.05
    h = dt / N_SUBSTEPS
    # Roll out only as far as the requested tables need, unless overridden.
    # The rollouts dominate runtime, so `--horizons 1` must shorten them or the
    # flag saves nothing.
    T = (args.n_outer if args.n_outer is not None
         else max(1, int(round(max(args.horizons) / dt))))
    t_eval = np.arange(T + 1) * dt

    print(f"loaded run: n={n}  steps={a['total_steps']}  "
          f"dtype={jnp.dtype(dtype).name}")
    print(f"  horizon: {T} outer steps x dt={dt} = {T * dt:.2f} s "
          f"(training window was num_points-1 = "
          f"{a.get('num_points', 5) - 1} steps = "
          f"{(a.get('num_points', 5) - 1) * dt:.2f} s)")

    # ── scale gauge ──
    R_beta = np.asarray(exp_so3_batch(jax.random.normal(
        jax.random.PRNGKey(args.seed + 7), (N_BETA_SAMPLES * n, 3),
        dtype=jnp.float64)).reshape(N_BETA_SAMPLES, n, 3, 3))
    if args.no_gauge:
        beta = 1.0
        beta_fit = estimate_beta(model, params, R_beta, dtype)
        print(f"  --no_gauge: beta pinned to 1 (raw comparison). "
              f"The fitted value would have been {beta_fit:.4f}.")
    else:
        beta = estimate_beta(model, params, R_beta, dtype)
        print(f"  fitted beta = {beta:.4f}")

    # ── ensemble rollouts, matched torque and Wiener paths ──
    u_label = _u_label(args.u, a["random_u"], a["random_u_scale"])
    print(f"Rolling out {N_TRAJ_ENSEMBLE} GT + model trajectories "
          f"(one u and one dW shared by GT and every rollout; only the initial "
          f"condition varies; {u_label}) ...")
    rng = np.random.default_rng(args.seed)
    # ONE torque sequence and ONE Wiener path, drawn once and reused by every
    # rollout and by its ground truth. Across the ensemble only the initial
    # condition varies, so the spread isolates sensitivity to the starting
    # state at a fixed forcing history.
    #
    # Consequence for the table: the N_TRAJ_ENSEMBLE samples are *correlated*
    # (they share the forcing), so the reported ± is a spread over initial
    # conditions and NOT an i.i.d. uncertainty estimate. Do not read it as a
    # standard deviation from which a standard error could be derived.
    u_seq = build_u_sequence(args.u, T, n, a["random_u"],
                             a["random_u_scale"], rng)
    dW = rng.normal(0.0, np.sqrt(h), size=(T, N_SUBSTEPS, 3))
    # A SECOND Wiener path, also shared: rolling GT again under this one gives
    # the divergence an exact model would still incur — the chaos floor.
    dW_ref = rng.normal(0.0, np.sqrt(h), size=(T, N_SUBSTEPS, 3))

    gt_all, md_all, ref_all = [], [], []
    for ti in range(N_TRAJ_ENSEMBLE):
        key = jax.random.PRNGKey(args.seed * 1000 + ti)
        R0 = exp_so3_batch(jax.random.normal(key, (n, 3), dtype=jnp.float64))
        w0 = rng.uniform(-1.0, 1.0, size=(n, 3))
        # The SAME u_seq and dW drive GT and model, so every trajectory
        # difference is model error and nothing else.
        gt_all.append(rollout_gt(params, R0, w0, u_seq, dW, sigma, h))
        md_all.append(rollout_model(model, R0, w0, u_seq, dW, h, dtype))
        ref_all.append(rollout_gt(params, R0, w0, u_seq, dW_ref, sigma, h))
    gt_all, md_all, ref_all = (np.stack(gt_all), np.stack(md_all),
                               np.stack(ref_all))

    # ── subnet evolution along N_SUB_TRAJ GT trajectories ──
    print(f"Evaluating subnets along {N_SUB_TRAJ} trajectories ...")
    comps_md_raw = [eval_subnets_model(model, gt_all[i], n, dtype)
                    for i in range(N_SUB_TRAJ)]
    comps_gt = [eval_subnets_gt(params, gt_all[i], n, sigma)
                for i in range(N_SUB_TRAJ)]
    comps_md = [apply_beta(c, beta) for c in comps_md_raw]

    # gauge-corrected subnet errors for the summary table
    def rel(A, B):
        return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-30))

    def subnet_errors_over(cs_md, cs_gt, steps=None):
        """Relative Frobenius error of each subnet over the given trajectories.

        `steps` restricts the comparison to the first `steps` outer steps, so a
        per-horizon table page scores the subnets over the same window its
        trajectory metrics use.
        """
        sl = slice(None) if steps is None else slice(0, steps + 1)
        cat = lambda cs, k: np.concatenate([c[k][sl] for c in cs])
        Mh, Mg = cat(cs_md, "M"), cat(cs_gt, "M")
        Dh, Dg = cat(cs_md, "D"), cat(cs_gt, "D")
        Vh, Vg = cat(cs_md, "V"), cat(cs_gt, "V")
        Bh, Bg = cat(cs_md, "B"), cat(cs_gt, "B")
        Sh, Sg = cat(cs_md, "Xi"), cat(cs_gt, "Xi")
        return {
            "M⁻¹": rel(Mh, Mg),
            "V (centred)": rel(Vh - Vh.mean(), Vg - Vg.mean()),
            "D": rel(Dh, Dg),
            # g is easy to omit and easy to get badly wrong: it is only
            # identifiable when the torque actually moves the system, so a small
            # --random_u_scale leaves it unconstrained while every other subnet
            # still looks fine.
            "g": rel(Bh, Bg),
            "ΣΣᵀ": rel(np.einsum("tab,tcb->tac", Sh, Sh),
                       np.einsum("tab,tcb->tac", Sg, Sg)),
        }

    # One table page per horizon; each scores its own window. Horizons longer
    # than the rollout are dropped rather than silently clipped to it.
    table_steps = [(hs, int(round(hs / dt))) for hs in args.horizons]
    dropped = [hs for hs, k in table_steps if k > T]
    table_steps = [(hs, k) for hs, k in table_steps if k <= T]
    if dropped:
        print(f"  skipping table horizons {dropped} s: beyond the "
              f"{T * dt:g} s rollout (raise --n_outer to include them)")
    subnet_err_by_h = {hs: subnet_errors_over(comps_md, comps_gt, k)
                       for hs, k in table_steps}
    # The single-rollout page scores the subnets along the one trajectory it
    # plots, not the N_SUB_TRAJ pool, so its numbers match the curves shown
    # elsewhere in the report for that same trajectory.
    subnet_err_single = subnet_errors_over(comps_md[:1], comps_gt[:1])

    # ── derived ensemble quantities ──
    # Only the plot pages need these, and `md_e` re-evaluates every subnet along
    # all 10 model rollouts, so skip them entirely under --tables_only.
    if not args.tables_only:
        gt_eul = np.stack([euler_all_links(gt_all[i], n) for i in range(N_TRAJ_ENSEMBLE)])
        md_eul = np.stack([euler_all_links(md_all[i], n) for i in range(N_TRAJ_ENSEMBLE)])
        gt_om = gt_all[..., 9 * n:].reshape(N_TRAJ_ENSEMBLE, T + 1, n, 3)
        md_om = md_all[..., 9 * n:].reshape(N_TRAJ_ENSEMBLE, T + 1, n, 3)

        gt_e = np.stack([get_energy(params, gt_all[i], n) for i in range(N_TRAJ_ENSEMBLE)])
        md_e = np.stack([hamiltonian_from_subnets(
            md_all[i], apply_beta(eval_subnets_model(model, md_all[i], n, dtype), beta), n)
            for i in range(N_TRAJ_ENSEMBLE)])

    geo_err = np.stack([geodesic_sq_all(md_all[i], gt_all[i], n)
                        for i in range(N_TRAJ_ENSEMBLE)])
    om_err = omega_sq_all(md_all, gt_all, n)
    # Same two metrics between the two ground-truth rollouts: the chaos floor.
    geo_floor = np.stack([geodesic_sq_all(ref_all[i], gt_all[i], n)
                          for i in range(N_TRAJ_ENSEMBLE)])
    om_floor = omega_sq_all(ref_all, gt_all, n)
    sat = random_rotation_saturation()
    print(f"  chaos floor (GT vs GT): traj-mean geodesic^2 = "
          f"{geo_floor.mean():.3f}  vs model {geo_err.mean():.3f}   |   "
          f"||dw||^2 = {om_floor.mean():.3f} vs {om_err.mean():.3f}")

    hist = meta["history"]
    xs = np.arange(len(hist)) * (a["total_steps"] / max(len(hist) - 1, 1))
    getc = lambda k: np.array([r[k] for r in hist])

    print(f"Writing {out} ...")
    with PdfPages(out) as pdf:
        # A. summary — one ensemble table per horizon, then the single
        # trajectory that the later single-rollout plots show.
        for hs, k in table_steps:
            pdf.savefig(fig_comparison_table(
                params, gt_all, {"ph_gp_sde": md_all}, n, beta,
                subnet_err_by_h[hs], u_label=u_label, steps=k, horizon_s=hs))
            plt.close()
        pdf.savefig(fig_comparison_table(
            params, gt_all[:1], {"ph_gp_sde": md_all[:1]}, n, beta,
            subnet_err_single, u_label=u_label, horizon_s=T * dt))
        plt.close()

        if args.tables_only:
            # Returning here still closes PdfPages cleanly, so the file is
            # complete — just without sections B/C/D.
            print("  --tables_only: skipped loss curves, ensemble dynamics "
                  "and subnet grids")
            print(f"\nwrote {out}")
            return

        # B. loss curves
        pdf.savefig(fig_one_curve(xs, getc("nll_R"), "Rotation NLL",
                                  "NLL", logy=False)); plt.close()
        pdf.savefig(fig_one_curve(xs, getc("nll_omega"), "Angular-rate NLL",
                                  "NLL", logy=False)); plt.close()
        pdf.savefig(fig_one_curve(xs, getc("pl_loss"), "Pseudo-likelihood NLL",
                                  "NLL", logy=False)); plt.close()
        pdf.savefig(fig_one_curve(xs, getc("mean_theta_sq"),
                                  "Train geodesic² per link", "geodesic² (rad²)"))
        plt.close()
        pdf.savefig(fig_one_curve(xs, getc("mean_omega_sq"),
                                  "Train MSE angular velocity", "‖Δω‖²")); plt.close()
        pdf.savefig(fig_one_curve(xs, getc("sigma_fro"),
                                  "Diffusion magnitude ‖Σ_θ‖_F", "‖Σ‖_F",
                                  hline=float(np.linalg.norm(comps_gt[0]["Xi"][0])),
                                  hlabel="GT ‖Σ‖_F")); plt.close()
        pdf.savefig(fig_one_curve(xs, getc("kl_total_kl"), "Total KL",
                                  "KL")); plt.close()
        pdf.savefig(fig_n_curves(
            [(xs, getc(f"kl_{k}_kl"), k, c) for k, c in
             zip(("M", "V", "Dw", "g", "Sigma"),
                 ("#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"))],
            "Per-subnet KL", "KL")); plt.close()

        # C. ensemble dynamics
        pdf.savefig(fig_energy_ensemble(
            t_eval, gt_e, {"ph_gp_sde": (md_e.mean(0), md_e.std(0), GP_COLOR)}))
        plt.close()
        pdf.savefig(fig_energy_single(
            t_eval, gt_e[0], {"ph_gp_sde": (md_e[0], GP_COLOR)})); plt.close()

        det_md = np.stack([so3_det_residual_abs(md_all[i], n)
                           for i in range(N_TRAJ_ENSEMBLE)])
        orth_md = np.stack([so3_orth_residual(md_all[i], n)
                            for i in range(N_TRAJ_ENSEMBLE)])
        det_gt = np.stack([so3_det_residual_abs(gt_all[i], n)
                           for i in range(N_TRAJ_ENSEMBLE)])
        orth_gt = np.stack([so3_orth_residual(gt_all[i], n)
                            for i in range(N_TRAJ_ENSEMBLE)])
        pdf.savefig(fig_so3_violation(
            t_eval, det_gt, {"ph_gp_sde": (det_md.mean(0), det_md[0], GP_COLOR)},
            "|det(R) − 1|", "SO(3) Violation: determinant")); plt.close()
        pdf.savefig(fig_so3_violation(
            t_eval, orth_gt, {"ph_gp_sde": (orth_md.mean(0), orth_md[0], GP_COLOR)},
            "‖RᵀR − I‖_F", "SO(3) Violation: orthogonality")); plt.close()

        pdf.savefig(fig_error_band(
            t_eval, geo_err,
            "Trajectory geodesic² vs GT (mean over links)",
            "geodesic² (rad²)", floor_all=geo_floor, saturation=sat))
        plt.close()
        pdf.savefig(fig_error_band(
            t_eval, om_err, "Trajectory ‖Δω‖² per link vs GT", "‖Δω‖² / link",
            floor_all=om_floor)); plt.close()

        pdf.savefig(fig_state_ensemble(
            t_eval, gt_eul, gt_om,
            {"ph_gp_sde": (md_eul, md_om, GP_COLOR)}, n)); plt.close()
        pdf.savefig(fig_state_single(
            t_eval, gt_all[0], {"ph_gp_sde": (md_all[0], GP_COLOR)}, n)); plt.close()
        pdf.savefig(fig_phase_portraits(
            gt_eul, gt_om, {"ph_gp_sde": (md_eul, md_om, GP_COLOR)}, n)); plt.close()

        # D. subnet evolution
        d = 3 * n
        pdf.savefig(plot_matrix_grid_multi(
            t_eval, comps_md_raw, comps_gt, "M",
            f"Inverse Mass M⁻¹(q) Along {N_SUB_TRAJ} Trajectories  (raw, no β)",
            d, d)); plt.close()
        pdf.savefig(plot_matrix_grid_multi(
            t_eval, comps_md, comps_gt, "M",
            f"Inverse Mass {_gtag(beta)}M⁻¹(q) Along {N_SUB_TRAJ} Trajectories{_gsuf(beta)}",
            d, d)); plt.close()
        pdf.savefig(plot_matrix_grid_multi(
            t_eval, comps_md, comps_gt, "D",
            f"Dissipation Dw(q){_gdiv(beta)} Along {N_SUB_TRAJ} Trajectories{_gsuf(beta)}",
            d, d)); plt.close()
        pdf.savefig(plot_matrix_grid_multi(
            t_eval, comps_md, comps_gt, "B",
            f"Control gain g(q){_gdiv(beta)} Along {N_SUB_TRAJ} Trajectories{_gsuf(beta)}",
            d, d)); plt.close()
        pdf.savefig(plot_potential_multi(
            t_eval, comps_md, comps_gt,
            f"Potential Energy V(q)/β Along {N_SUB_TRAJ} Trajectories")); plt.close()
        pdf.savefig(plot_matrix_grid_multi(
            t_eval, comps_md, comps_gt, "Xi",
            f"Diffusion Σ(q){_gdiv(beta)} Along {N_SUB_TRAJ} Trajectories{_gsuf(beta)}",
            d, 3)); plt.close()

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

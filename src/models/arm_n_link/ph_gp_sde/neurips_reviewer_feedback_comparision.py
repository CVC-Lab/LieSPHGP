r"""NeurIPS-reviewer-driven evaluation report for a trained `ph_gp_sde` run.

Every section answers a specific criticism from the NeurIPS 2026 reviews of
LieSPHGP (submission 28694), for the **n-link arm** experiment, model only
(no baselines yet — tables are laid out so baseline columns can be added).

=====================  ==================================================
review criticism        section of this report
=====================  ==================================================
"errors look large"     §1  Pathwise error **normalised by the chaos
                            floor**: for a chaotic SDE the correct
                            reference is GT-vs-GT under a different wind
                            realisation — the *irreducible* pathwise
                            error. We report the skill ratio
                            model/floor with bootstrap CIs, per horizon.
UQ never evaluated      §2  Quantitative calibration: empirical coverage
(6Ap3, W2)                  of central predictive intervals at nominal
                            10–90%, a reliability diagram, and CRPS
                            (sample estimator), per horizon.
pathwise saturates      §3  Distributional accuracy: energy distance
                            between model and GT *ensembles* per horizon,
                            plus Hamiltonian-energy envelope overlay.
manifold claims         §4  SO(3) constraint violations over the rollout.
structure claims        §5  Physics recovery: gauge-corrected subnet
                            errors (with fitted β) and the **gauge-free**
                            ω-space diffusion error
                            ‖ÂÂᵀ − AAᵀ‖/‖AAᵀ‖, A = M⁻¹σΣ —
                            invariant under both the scale gauge and
                            Σ → ΣO, so it needs no correction at all.
significance (U4gY)     every table: median [IQR] across initial
                            conditions + percentile-bootstrap CIs.
=====================  ==================================================

Protocol
--------
For each of `--n_ic` initial conditions (random attitudes via the exponential
map, ω ~ U(-1,1) as in the environment's reset) and one shared random torque
sequence u ~ U(-u_scale, u_scale) per outer step:

* **pathwise pair**   — GT and model driven by the *same* Wiener path;
* **chaos floor**     — a second GT run with an *independent* Wiener path;
* **model ensemble**  — `--ensemble` model rollouts, independent winds,
                        posterior-mean (deterministic) drift;
* **GT ensemble**     — `--ensemble` GT rollouts, independent winds.

Calibration targets the *state* predictive distribution (exact initial
condition, no observation noise): the dataset's obs noise is known exactly
and would only convolve every interval with the same known Gaussian, telling
us nothing new about the learned model.

Outputs, written into the run directory:
    neurips_reviewer_feedback_report.pdf    (plots + summary page)
    neurips_reviewer_feedback_metrics.json  (every number, for paper tables)

Usage:
    python neurips_reviewer_feedback_comparision.py <run_dir> \
        [--n_ic 24] [--ensemble 32] [--horizon 1.0] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, ".."))
PROJECT_ROOT = os.path.abspath(
    os.path.join(THIS_FILE_DIR, "..", "..", "..", ".."))
for _p in (PKG_ROOT, PROJECT_ROOT, THIS_FILE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.arm_nlink_so3 import arm_nlink_physics as phys       # noqa: E402
from utils.elbo_loss_nlink import geodesic_distance            # noqa: E402
from utils.lie_integrator_nlink import exp_so3_batch           # noqa: E402
from make_comparison_pdf import (                              # noqa: E402
    load_run, rollout_gt, rollout_model,
    eval_subnets_model, eval_subnets_gt, estimate_beta, apply_beta,
)

DT = 0.05          # outer step — matches the environment and the trainer
N_SUB = 10         # substeps per outer step — matches train.py's hardcoded 10


# ──────────────────────────────────────────────────────────────────────────
# Per-trajectory error curves
# ──────────────────────────────────────────────────────────────────────────

def geo2_per_link(trajA, trajB, n):
    """Mean-over-links squared geodesic distance, per time step. (T+1,)."""
    RA = trajA[:, :9 * n].reshape(-1, 3, 3)
    RB = trajB[:, :9 * n].reshape(-1, 3, 3)
    th = np.asarray(geodesic_distance(jnp.asarray(RA), jnp.asarray(RB)))
    return (th ** 2).reshape(-1, n).mean(axis=1)


def domega2_per_link(trajA, trajB, n):
    """‖Δω‖² summed over 3n components, divided by n. (T+1,)."""
    d = trajA[:, 9 * n:12 * n] - trajB[:, 9 * n:12 * n]
    return (d ** 2).sum(axis=1) / n


# ──────────────────────────────────────────────────────────────────────────
# Statistics helpers (numpy only — no scipy dependency)
# ──────────────────────────────────────────────────────────────────────────

def boot_ci(values, stat=np.median, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI of `stat` over axis 0. Returns (lo, hi)."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = stat(values[idx], axis=1)
    return float(np.quantile(stats, alpha / 2)), \
        float(np.quantile(stats, 1 - alpha / 2))


def med_iqr(values):
    v = np.asarray(values)
    return float(np.median(v)), float(np.quantile(v, 0.25)), \
        float(np.quantile(v, 0.75))


def crps_samples(x, y):
    r"""Sample-based CRPS for scalar y with ensemble x (K,).

    CRPS(F, y) = E|X − y| − ½E|X − X′|  (proper score; lower is better).
    """
    x = np.asarray(x)
    return np.mean(np.abs(x - y)) - 0.5 * np.mean(
        np.abs(x[:, None] - x[None, :]))


def energy_distance(X, Y):
    r"""Energy distance between samples X (K,d) and Y (K,d).

    ED² = 2E‖X−Y‖ − E‖X−X′‖ − E‖Y−Y′‖ ≥ 0, = 0 iff the laws coincide.
    A proper distributional metric — exactly what pathwise error cannot give
    once the chaos floor saturates.
    """
    def pd(A, B):
        return np.mean(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1))
    return float(2 * pd(X, Y) - pd(X, X) - pd(Y, Y))


# ──────────────────────────────────────────────────────────────────────────
# Rollout collection
# ──────────────────────────────────────────────────────────────────────────

def collect(model, params, dtype, n, sigma, T, K, n_ic, u_scale, rng):
    """Run the full protocol. Returns a dict of stacked numpy arrays."""
    h = DT / N_SUB

    def wind():
        return rng.normal(0.0, np.sqrt(h), size=(T, N_SUB, 3))

    out = dict(gt=[], model=[], floor=[], ens_model=[], ens_gt=[])
    for _ in range(n_ic):
        R0 = np.asarray(exp_so3_batch(jnp.asarray(rng.normal(size=(n, 3)))))
        w0 = rng.uniform(-1.0, 1.0, size=(n, 3))
        u_seq = rng.uniform(-u_scale, u_scale, size=(T, 3 * n))

        dW = wind()
        out["gt"].append(rollout_gt(params, R0, w0, u_seq, dW, sigma, h))
        out["model"].append(rollout_model(model, R0, w0, u_seq, dW, h, dtype))
        out["floor"].append(rollout_gt(params, R0, w0, u_seq, wind(), sigma, h))

        out["ens_model"].append(np.stack([
            rollout_model(model, R0, w0, u_seq, wind(), h, dtype)
            for _ in range(K)]))
        out["ens_gt"].append(np.stack([
            rollout_gt(params, R0, w0, u_seq, wind(), sigma, h)
            for _ in range(K)]))

    return {k: np.stack(v) for k, v in out.items()}


# ──────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────

def pathwise_metrics(roll, n, t_grid, horizons, rng):
    """§1 — model-vs-GT and floor curves + per-horizon skill table."""
    n_ic = roll["gt"].shape[0]
    geo_m = np.stack([geo2_per_link(roll["model"][i], roll["gt"][i], n)
                      for i in range(n_ic)])
    geo_f = np.stack([geo2_per_link(roll["floor"][i], roll["gt"][i], n)
                      for i in range(n_ic)])
    om_m = np.stack([domega2_per_link(roll["model"][i], roll["gt"][i], n)
                     for i in range(n_ic)])
    om_f = np.stack([domega2_per_link(roll["floor"][i], roll["gt"][i], n)
                     for i in range(n_ic)])

    table = []
    for t_h in horizons:
        k = int(np.argmin(np.abs(t_grid - t_h)))
        row = {"horizon_s": float(t_grid[k])}
        for name, m_arr, f_arr in (("geo2", geo_m, geo_f),
                                   ("omega2", om_m, om_f)):
            med, q1, q3 = med_iqr(m_arr[:, k])
            fmed, fq1, fq3 = med_iqr(f_arr[:, k])
            # Paired per-IC skill ratio: the floor is high exactly when the
            # trajectory is chaotic, so pairing removes that common variance.
            ratio = m_arr[:, k] / np.maximum(f_arr[:, k], 1e-12)
            rmed = float(np.median(ratio))
            lo, hi = boot_ci(ratio, rng=rng)
            row[name] = {"model_med": med, "model_iqr": [q1, q3],
                         "floor_med": fmed, "floor_iqr": [fq1, fq3],
                         "skill_ratio_med": rmed, "skill_ratio_ci95": [lo, hi]}
        table.append(row)
    curves = {"geo_model": geo_m, "geo_floor": geo_f,
              "om_model": om_m, "om_floor": om_f}
    return table, curves


def calibration_metrics(roll, n, t_grid, horizons, levels):
    """§2 — coverage per nominal level & horizon + CRPS per horizon.

    Coverage uses central intervals from ensemble quantiles per
    (time, component); the fraction of GT values inside is aggregated over
    components and initial conditions. Under-dispersion (e.g. an
    under-scaled diffusion) shows as empirical < nominal.
    """
    ens = roll["ens_model"][:, :, :, 9 * n:12 * n]    # (IC, K, T+1, 3n)
    gt = roll["gt"][:, :, 9 * n:12 * n]               # (IC, T+1, 3n)

    cov = np.zeros((len(levels), len(t_grid)))
    for li, lev in enumerate(levels):
        lo = np.quantile(ens, (1 - lev) / 2, axis=1)
        hi = np.quantile(ens, (1 + lev) / 2, axis=1)
        inside = (gt >= lo) & (gt <= hi)              # (IC, T+1, 3n)
        cov[li] = inside.mean(axis=(0, 2))

    n_ic, K, Tp1, d = ens.shape
    crps_t = np.zeros((n_ic, Tp1))
    for i in range(n_ic):
        for k in range(Tp1):
            crps_t[i, k] = np.mean([crps_samples(ens[i, :, k, c], gt[i, k, c])
                                    for c in range(d)])

    table = []
    for t_h in horizons:
        k = int(np.argmin(np.abs(t_grid - t_h)))
        row = {"horizon_s": float(t_grid[k]),
               "crps_mean": float(crps_t[:, k].mean())}
        for li, lev in enumerate(levels):
            if lev in (0.5, 0.9):
                row[f"coverage@{int(lev * 100)}"] = float(cov[li, k])
        table.append(row)

    # Reliability diagram data: aggregate over t >= first step.
    reliability = {"nominal": [float(l) for l in levels],
                   "empirical": [float(cov[li, 1:].mean())
                                 for li in range(len(levels))]}
    return table, reliability, cov, crps_t


def distributional_metrics(roll, params, n, t_grid, horizons):
    """§3 — per-horizon energy distance between model and GT ω-ensembles,
    plus Hamiltonian-energy envelopes of both ensembles."""
    em = roll["ens_model"][:, :, :, 9 * n:12 * n]
    eg = roll["ens_gt"][:, :, :, 9 * n:12 * n]
    n_ic = em.shape[0]

    table = []
    for t_h in horizons:
        k = int(np.argmin(np.abs(t_grid - t_h)))
        ed = [energy_distance(em[i, :, k], eg[i, :, k]) for i in range(n_ic)]
        # Scale reference: ED of two *GT* half-ensembles — the sampling floor.
        half = eg.shape[1] // 2
        ed0 = [energy_distance(eg[i, :half, k], eg[i, half:, k])
               for i in range(n_ic)]
        med, q1, q3 = med_iqr(ed)
        table.append({"horizon_s": float(t_grid[k]),
                      "energy_dist_med": med, "energy_dist_iqr": [q1, q3],
                      "gt_split_floor_med": float(np.median(ed0))})

    @jax.jit
    def batch_energy(R_flat, om):
        return jax.vmap(lambda r, w: phys.total_energy(
            params, r.reshape(n, 3, 3), w.reshape(n, 3)))(R_flat, om)

    def energies(stack):                              # (IC,K,T+1,12n)
        s = stack.reshape(-1, stack.shape[-1])
        E = np.asarray(batch_energy(jnp.asarray(s[:, :9 * n]),
                                    jnp.asarray(s[:, 9 * n:12 * n])))
        return E.reshape(stack.shape[:-1])            # (IC,K,T+1)

    return table, energies(roll["ens_model"]), energies(roll["ens_gt"])


def so3_metrics(roll, n, t_grid):
    """§4 — constraint violations of the *model* rollouts over time."""
    R = roll["model"][:, :, :9 * n].reshape(
        roll["model"].shape[0], -1, n, 3, 3)          # (IC, T+1, n, 3, 3)
    gram = np.einsum("itnab,itnac->itnbc", R, R) - np.eye(3)
    orth = np.linalg.norm(gram.reshape(*gram.shape[:3], 9), axis=-1).max(2)
    det = np.abs(np.linalg.det(R) - 1.0).max(2)
    return {"orth_max_final": float(orth[:, -1].max()),
            "det_max_final": float(det[:, -1].max()),
            "orth_curve": orth.mean(0), "det_curve": det.mean(0)}


def physics_metrics(model, params, roll, n, sigma, dtype):
    """§5 — gauge-corrected subnet errors + the gauge-free diffusion error."""
    q_states = roll["gt"][:, :, :9 * n].reshape(-1, 9 * n)
    pick = np.random.default_rng(0).choice(len(q_states), 64, replace=False)
    q = q_states[pick]
    R_samples = [q_i.reshape(n, 3, 3) for q_i in q]

    hat = eval_subnets_model(model, np.concatenate(
        [q, np.zeros((len(q), 3 * n))], axis=1), n, dtype)
    gt = eval_subnets_gt(params, np.concatenate(
        [q, np.zeros((len(q), 3 * n))], axis=1), n, sigma)
    beta = estimate_beta(model, params, R_samples, dtype)
    hatb = apply_beta(hat, beta)

    def rel(A, B):
        return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-30))

    Vc_h = hatb["V"] - hatb["V"].mean()
    Vc_g = gt["V"] - gt["V"].mean()
    P_h = np.einsum("kab,kcb->kac", hatb["Xi"], hatb["Xi"])
    P_g = np.einsum("kab,kcb->kac", gt["Xi"], gt["Xi"])

    # Gauge-FREE observable diffusion: A = M⁻¹Σ (ω-space). Invariant under
    # both the scale gauge (β cancels) and Σ → ΣO (only AAᵀ is used), so it
    # needs no correction and directly certifies the learned stochasticity.
    A_h = np.einsum("kab,kbc->kac", hat["M"], hat["Xi"])
    A_g = np.einsum("kab,kbc->kac", gt["M"], gt["Xi"])
    AAT_h = np.einsum("kab,kcb->kac", A_h, A_h)
    AAT_g = np.einsum("kab,kcb->kac", A_g, A_g)

    out = {"beta": float(beta),
           "M_inv": rel(hatb["M"], gt["M"]),
           "V_centred": rel(Vc_h, Vc_g),
           "D": rel(hatb["D"], gt["D"]),
           "g": rel(hatb["B"], gt["B"]),
           "SSt": rel(P_h, P_g),
           "omega_diffusion_AAt": rel(AAT_h, AAT_g),
           "A_fro_model": float(np.sqrt((A_h ** 2).sum(axis=(1, 2)).mean())),
           "A_fro_gt": float(np.sqrt((A_g ** 2).sum(axis=(1, 2)).mean()))}
    # One shared wind field must torque every link the same way, so adjacent
    # lever vectors should be near-parallel. A negative cosine is a relative
    # sign flip — a genuine error (anti-correlated wind response), not gauge.
    if hasattr(model.Sigma_net, "v") and n >= 2:
        vv = np.asarray(model.Sigma_net.v)
        nn = np.linalg.norm(vv, axis=1)
        out["sigma_v_norms"] = [float(x) for x in nn]
        # null when a lever has collapsed: the direction of a ~zero vector is
        # noise, and that failure is a magnitude story told by the norms.
        out["sigma_v_adjacent_cos"] = [
            None if min(nn[i], nn[i + 1]) < 1e-2 else
            float(vv[i] @ vv[i + 1] / (nn[i] * nn[i + 1]))
            for i in range(n - 1)]
    return out


# ──────────────────────────────────────────────────────────────────────────
# Report rendering
# ──────────────────────────────────────────────────────────────────────────

# (header, [lines]) blocks rendered on the two definition pages. Lines may use
# matplotlib mathtext ($...$). Kept next to the metric code so the two cannot
# silently drift apart.
_DEFS_PAGE_1 = [
    ("§1  PATHWISE ERROR vs THE CHAOS FLOOR", [
     r"Rotation error — squared geodesic distance on SO(3), averaged over links:",
     r"    $\theta_i = \arccos\left(\frac{\mathrm{tr}(R_i^{gt\top} R_i^{model})-1}{2}\right),"
     r"\qquad \mathrm{geo}^2 = \frac{1}{n}\sum_i \theta_i^2$",
     r"$\theta_i$ is the angle of the relative rotation — the intrinsic distance on the",
     r"manifold, immune to Euler-angle / chart artifacts.",
     "",
     r"Rate error:  $\frac{1}{n}\| \omega^{model} - \omega^{gt} \|^2$  (summed over all $3n$ components, per link).",
     "",
     r"CHAOS FLOOR — the same two metrics for GT vs GT: identical initial condition and",
     r"torques, but an independent wind realisation. The system is chaotic and stochastic,",
     r"so even the TRUE model incurs this error in expectation. Absolute pathwise numbers",
     r"are therefore meaningless on their own; the floor is the correct denominator.",
     "",
     r"SKILL RATIO — per-initial-condition paired ratio  $\mathrm{err}_{model}/\mathrm{err}_{floor}$",
     r"(pairing cancels the shared per-IC chaos variance). Reported as the median with a",
     r"95% percentile-bootstrap CI over ICs.  1.0 = information-theoretic optimum;",
     r"values are only meaningful BEFORE the curves saturate at the floor (~0.5 s here).",
     ]),
    ("§2  CALIBRATION OF THE PREDICTIVE DISTRIBUTION", [
     r"Predictive ensemble: K rollouts from the same initial condition with independent",
     r"wind paths and posterior-mean (deterministic) drift — the model's own predictive",
     r"law for the state. The target is one held-out GT realisation.",
     "",
     r"COVERAGE at nominal level $\ell$ — per (time, $\omega$-component), the central interval",
     r"between ensemble quantiles $\frac{1-\ell}{2}$ and $\frac{1+\ell}{2}$; coverage = fraction of GT",
     r"values falling inside, pooled over components and ICs. Calibrated $\Leftrightarrow$ coverage $=\ell$.",
     "",
     r"RELIABILITY DIAGRAM — empirical vs nominal coverage across $\ell = 0.1 \ldots 0.9$.",
     r"Below the diagonal = under-dispersed (intervals too narrow / model too confident);",
     r"above = over-dispersed. The distance to the diagonal is the calibration error.",
     "",
     r"CRPS (continuous ranked probability score), sample estimator with ensemble $X$:",
     r"    $\mathrm{CRPS}(F, y) = E_F|X - y| - \frac{1}{2}E_F|X - X'|$",
     r"A strictly proper scoring rule: uniquely minimised in expectation by the true",
     r"predictive distribution, so it rewards accuracy AND honest uncertainty at once",
     r"(a sharp-but-wrong or wide-but-safe forecast both score worse). Generalises MAE",
     r"to distributions; units of $\omega$ (rad/s); lower is better.",
     ]),
]

_DEFS_PAGE_2 = [
    ("§3  DISTRIBUTIONAL ACCURACY (what replaces pathwise error after the floor)", [
     r"ENERGY DISTANCE between the model law $P$ and GT law $Q$ of the $\omega$-vector at",
     r"each horizon, from K-sample ensembles $X \sim P$, $Y \sim Q$:",
     r"    $ED^2(P, Q) = 2E\|X - Y\| - E\|X - X'\| - E\|Y - Y'\|$",
     r"$ED \geq 0$, with $ED = 0$ iff $P = Q$ — a genuine metric between distributions.",
     r"GT SPLIT-HALF FLOOR: ED between two halves of the GT ensemble — the value a",
     r"perfect model would show from finite-sample noise alone.",
     r"HAMILTONIAN ENVELOPE — $H(q,\omega)$ evaluated with GT physics on BOTH ensembles:",
     r"do the two laws transport the same energy statistics (mean and spread) over time?",
     ]),
    ("§4  SO(3) CONSTRAINT SATISFACTION", [
     r"Orthogonality defect $\|R^\top R - I\|_F$ and determinant defect $|\det R - 1|$ along model",
     r"rollouts. Values at float-precision level certify the Lie-group integrator keeps the",
     r"state on the manifold by construction — no projection or renormalisation anywhere.",
     ]),
    ("§5  PHYSICS RECOVERY (gauge-corrected and gauge-free)", [
     r"SCALE GAUGE $\beta$ — the dynamics are invariant under $(M, V, D, g, \Sigma) \to \beta(\cdot)$ with",
     r"$p \to \beta p$: the data determine the physics only up to one global scale. Fitted by",
     r"least squares, $\beta = \mathrm{argmin} \|\beta\hat{M}^{-1} - M_{gt}^{-1}\|_F^2$, then every subnet is compared",
     r"AFTER correction (and $V$ centred: $H$ is defined up to a constant).",
     r"SUBNET ERRORS — relative Frobenius over a stack of sampled configurations:",
     r"    $\mathrm{rel}(A, B) = \|A - B\|_F \, / \, \|B\|_F$",
     r"$SS^T$ compares $\beta^{-2}\hat{\Sigma}\hat{\Sigma}^\top$ vs $\sigma^2\Sigma\Sigma^\top$ because only $\Sigma\Sigma^\top$ is identifiable",
     r"($\Sigma \to \Sigma O$ for orthogonal $O$ leaves the law of the SDE unchanged).",
     r"GAUGE-FREE $\omega$-DIFFUSION — the observable diffusion $A = M^{-1}\Sigma$ satisfies",
     r"$\mathrm{Var}(d\omega)/dt = AA^\top$ and is invariant under BOTH gauges ($\beta$ cancels, only $AA^\top$",
     r"is used). Its relative error needs no correction of any kind and directly certifies",
     r"the learned stochasticity; $\|A\|_F$ values give the noise amplitude in rad/s$^{3/2}$.",
     r"WIND LEVERS — structured diffusion $\Sigma_j = [v_j]_\times R_j^\top$: reported $\|v_j\|$ and adjacent",
     r"$\cos(v_i, v_{i+1})$. One shared wind field should torque every link coherently (+1",
     r"expected); a negative cosine is a relative sign flip (a genuine, identifiable error,",
     r"unlike a global flip which is gauge); n/a = lever collapsed, direction undefined.",
     ]),
    ("STATISTICAL CONVENTIONS", [
     r"Aggregates are median [IQR] across initial conditions (robust to the heavy tails",
     r"chaotic rollouts produce); intervals are 95% percentile-bootstrap CIs (2000",
     r"resamples over ICs). Torques are shared across every rollout of a given IC.",
     r"Calibration targets noiseless states: the observation noise is known exactly, so",
     r"it would only convolve every interval with the same fixed Gaussian.",
     ]),
]


def _definition_pages(pdf):
    """Two text pages defining every metric and its formula (§1–§5)."""
    for pageno, blocks in ((1, _DEFS_PAGE_1), (2, _DEFS_PAGE_2)):
        fig = plt.figure(figsize=(11.5, 8))
        fig.text(0.5, 0.965, f"How to read this report — metric definitions "
                 f"({pageno}/2)", ha="center", fontsize=12, fontweight="bold")
        y = 0.92
        for header, lines in blocks:
            fig.text(0.05, y, header, fontsize=9.5, fontweight="bold",
                     va="top")
            y -= 0.028
            for ln in lines:
                fig.text(0.07, y, ln, fontsize=8.2, va="top")
                y -= 0.023
            y -= 0.010
        pdf.savefig(fig)
        plt.close(fig)

def _fmt_pathwise(table):
    lines = [f"{'t':>5} {'metric':>7} {'model med [IQR]':>28} "
             f"{'floor med [IQR]':>28} {'skill med (95% CI)':>26}"]
    for row in table:
        for m in ("geo2", "omega2"):
            r = row[m]
            lines.append(
                f"{row['horizon_s']:>5.2f} {m:>7} "
                f"{r['model_med']:>10.3e} [{r['model_iqr'][0]:.2e},"
                f"{r['model_iqr'][1]:.2e}] "
                f"{r['floor_med']:>10.3e} [{r['floor_iqr'][0]:.2e},"
                f"{r['floor_iqr'][1]:.2e}] "
                f"{r['skill_ratio_med']:>7.2f} ({r['skill_ratio_ci95'][0]:.2f},"
                f"{r['skill_ratio_ci95'][1]:.2f})")
    return lines


def _fmt_calibration(table):
    lines = [f"{'t':>5} {'cov@50 (nom .50)':>18} {'cov@90 (nom .90)':>18} "
             f"{'CRPS':>10}"]
    for row in table:
        lines.append(f"{row['horizon_s']:>5.2f} "
                     f"{row.get('coverage@50', float('nan')):>18.3f} "
                     f"{row.get('coverage@90', float('nan')):>18.3f} "
                     f"{row['crps_mean']:>10.4f}")
    return lines


def make_pdf(path, meta, t_grid, path_table, curves, cal_table, reliability,
             cov, crps_t, dist_table, E_model, E_gt, so3, phys_m, levels):
    a = meta["args"]
    with PdfPages(path) as pdf:
        # ── Page 1: summary tables ──
        fig = plt.figure(figsize=(11.5, 8))
        fig.text(0.5, 0.97, "ph_gp_sde — NeurIPS-review-driven evaluation "
                 f"(n={a['n']}, horizon {t_grid[-1]:.2f}s)",
                 ha="center", fontsize=13, fontweight="bold")
        txt = ["§1 Pathwise error vs the CHAOS FLOOR (GT vs GT, other wind).",
               "   skill = model/floor per IC; 1.0 = information-theoretic optimum.",
               ""] + _fmt_pathwise(path_table) + [
               "",
               "§2 Calibration of the predictive distribution (held-out ICs).",
               ""] + _fmt_calibration(cal_table) + [
               "",
               "§3 Ensemble energy distance (model vs GT law), with the",
               "   GT split-half sampling floor for scale:",
               ] + [f"   t={r['horizon_s']:.2f}  ED={r['energy_dist_med']:.3f} "
                    f"[{r['energy_dist_iqr'][0]:.3f},{r['energy_dist_iqr'][1]:.3f}]"
                    f"   GT-split floor {r['gt_split_floor_med']:.3f}"
                    for r in dist_table] + [
               "",
               "§4 SO(3) violation (max over ICs, final step): "
               f"det {so3['det_max_final']:.2e}   orth {so3['orth_max_final']:.2e}",
               "",
               "§5 Physics recovery  (gauge-corrected, "
               f"beta={phys_m['beta']:.3f}):",
               f"   M^-1 {phys_m['M_inv']:.4f}   V {phys_m['V_centred']:.4f}   "
               f"D {phys_m['D']:.4f}   g {phys_m['g']:.4f}   "
               f"SS^T {phys_m['SSt']:.4f}",
               f"   gauge-FREE omega-diffusion ||AA^T-err|| : "
               f"{phys_m['omega_diffusion_AAt']:.4f}   "
               f"(||A||_F model {phys_m['A_fro_model']:.3f} vs GT "
               f"{phys_m['A_fro_gt']:.3f})"]
        if "sigma_v_adjacent_cos" in phys_m:
            txt.append("   wind levers ||v_i||: "
                       + ", ".join(f"{x:.3f}" for x in phys_m["sigma_v_norms"])
                       + "   adjacent cos: "
                       + ", ".join("n/a" if c is None else f"{c:+.3f}"
                                   for c in phys_m["sigma_v_adjacent_cos"])
                       + "   (+1 expected; n/a = lever collapsed)")
        fig.text(0.06, 0.90, "\n".join(txt), fontsize=7.6,
                 va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Pages 2-3: metric definitions ──
        _definition_pages(pdf)

        # ── Page 4: pathwise curves vs floor ──
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for ax, key, name in ((axes[0], "geo", "geodesic$^2$ per link"),
                              (axes[1], "om", r"$\|\Delta\omega\|^2$ per link")):
            m, f = curves[f"{key}_model"], curves[f"{key}_floor"]
            ax.plot(t_grid, np.median(m, 0), "r-", label="model vs GT (median)")
            ax.fill_between(t_grid, np.quantile(m, .25, 0),
                            np.quantile(m, .75, 0), color="r", alpha=.2)
            ax.plot(t_grid, np.median(f, 0), "k--",
                    label="chaos floor (GT vs GT)")
            ax.fill_between(t_grid, np.quantile(f, .25, 0),
                            np.quantile(f, .75, 0), color="k", alpha=.15)
            ax.set_yscale("log")
            ax.set_xlabel("time (s)")
            ax.set_title(name)
            ax.legend(fontsize=8)
            ax.grid(alpha=.3)
        fig.suptitle("§1  Pathwise error against the irreducible floor")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: reliability + coverage over time ──
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        axes[0].plot(reliability["nominal"], reliability["empirical"],
                     "ro-", label="model")
        axes[0].set_xlabel("nominal coverage")
        axes[0].set_ylabel("empirical coverage")
        axes[0].set_title("Reliability (all horizons pooled)")
        axes[0].legend()
        axes[0].grid(alpha=.3)
        for li, lev in enumerate(levels):
            if lev in (0.5, 0.9):
                axes[1].plot(t_grid[1:], cov[li, 1:], "-o", ms=3,
                             label=f"nominal {lev:.0%}")
                axes[1].axhline(lev, color="gray", lw=.8, ls=":")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("empirical coverage")
        axes[1].set_title("Coverage vs horizon")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=.3)
        fig.suptitle("§2  Calibration — under-dispersion appears as "
                     "curves below the diagonal / nominal lines")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: CRPS + energy distance vs horizon ──
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        axes[0].plot(t_grid, crps_t.mean(0), "r-o", ms=3)
        axes[0].set_title("CRPS (mean over components & ICs)")
        axes[0].set_xlabel("time (s)")
        axes[0].grid(alpha=.3)
        ed_t = [r["horizon_s"] for r in dist_table]
        axes[1].plot(ed_t, [r["energy_dist_med"] for r in dist_table],
                     "r-o", ms=4, label="model vs GT law")
        axes[1].plot(ed_t, [r["gt_split_floor_med"] for r in dist_table],
                     "k--o", ms=4, label="GT split-half floor")
        axes[1].set_title("Ensemble energy distance")
        axes[1].set_xlabel("time (s)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=.3)
        fig.suptitle("§2/§3  Proper scores over the horizon")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 5: Hamiltonian energy envelopes ──
        fig, ax = plt.subplots(figsize=(11.5, 4.6))
        Eg = E_gt.reshape(-1, E_gt.shape[-1])
        Em = E_model.reshape(-1, E_model.shape[-1])
        for E, c, lab in ((Eg, "k", "GT ensemble"),
                          (Em, "r", "model ensemble")):
            ax.plot(t_grid, E.mean(0), c, lw=2, label=f"{lab} mean")
            ax.fill_between(t_grid, E.mean(0) - 2 * E.std(0),
                            E.mean(0) + 2 * E.std(0), color=c, alpha=.15,
                            label=f"{lab} ±2σ")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("H (GT physics) [J]")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.suptitle("§3  Energy statistics of the two laws "
                     "(computed with GT physics on both ensembles)")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 6: SO(3) violations ──
        fig, ax = plt.subplots(figsize=(11.5, 4.2))
        ax.semilogy(t_grid, so3["det_curve"] + 1e-20,
                    "r-", label="|det R − 1| (mean over ICs)")
        ax.semilogy(t_grid, so3["orth_curve"] + 1e-20,
                    "r--", label=r"$\|R^\top R - I\|_F$")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.suptitle("§4  SO(3) constraint violation of model rollouts")
        pdf.savefig(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NeurIPS-review-driven metrics report for ph_gp_sde.")
    ap.add_argument("run_dir")
    ap.add_argument("--n_ic", type=int, default=24,
                    help="held-out initial conditions")
    ap.add_argument("--ensemble", type=int, default=32,
                    help="rollouts per predictive ensemble")
    ap.add_argument("--horizon", type=float, default=1.0, help="seconds")
    ap.add_argument("--u_scale", type=float, default=None,
                    help="torque range; default = the run's random_u_scale")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, meta, params, dtype = load_run(args.run_dir)
    a = meta["args"]
    n = a["n"]
    sigma = a["wind_force_std"]
    u_scale = a.get("random_u_scale", 1.0) if args.u_scale is None \
        else args.u_scale
    T = int(round(args.horizon / DT))
    t_grid = DT * np.arange(T + 1)
    horizons = [t for t in (0.1, 0.25, 0.5, 1.0) if t <= args.horizon + 1e-9]
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rng = np.random.default_rng(args.seed)

    print(f"run: {args.run_dir}\n"
          f"protocol: {args.n_ic} ICs x (2 GT + 1 model pathwise + "
          f"2x{args.ensemble} ensemble) rollouts, horizon {args.horizon}s, "
          f"u~U(-{u_scale},{u_scale}), sigma={sigma}")

    roll = collect(model, params, dtype, n, sigma, T,
                   args.ensemble, args.n_ic, u_scale, rng)

    path_table, curves = pathwise_metrics(roll, n, t_grid, horizons, rng)
    cal_table, reliability, cov, crps_t = calibration_metrics(
        roll, n, t_grid, horizons, levels)
    dist_table, E_model, E_gt = distributional_metrics(
        roll, params, n, t_grid, horizons)
    so3 = so3_metrics(roll, n, t_grid)
    phys_m = physics_metrics(model, params, roll, n, sigma, dtype)

    print("\n§1 pathwise vs chaos floor")
    print("\n".join(_fmt_pathwise(path_table)))
    print("\n§2 calibration")
    print("\n".join(_fmt_calibration(cal_table)))
    print(f"\n§5 physics recovery: beta={phys_m['beta']:.3f}  "
          f"M^-1 {phys_m['M_inv']:.4f}  V {phys_m['V_centred']:.4f}  "
          f"D {phys_m['D']:.4f}  g {phys_m['g']:.4f}  "
          f"SS^T {phys_m['SSt']:.4f}  "
          f"gauge-free AA^T {phys_m['omega_diffusion_AAt']:.4f}")
    if "sigma_v_adjacent_cos" in phys_m:
        print("   wind levers ||v_i||:",
              ["%.3f" % x for x in phys_m["sigma_v_norms"]],
              " adjacent cos:",
              ["n/a" if c is None else "%+.3f" % c
               for c in phys_m["sigma_v_adjacent_cos"]],
              "(+1 expected; n/a = collapsed)")

    pdf_path = os.path.join(args.run_dir,
                            "neurips_reviewer_feedback_report.pdf")
    make_pdf(pdf_path, meta, t_grid, path_table, curves, cal_table,
             reliability, cov, crps_t, dist_table, E_model, E_gt, so3,
             phys_m, levels)

    metrics = {"protocol": {"n_ic": args.n_ic, "ensemble": args.ensemble,
                            "horizon_s": args.horizon, "u_scale": u_scale,
                            "seed": args.seed, "dt": DT, "n_substeps": N_SUB},
               "pathwise_vs_floor": path_table,
               "calibration": cal_table,
               "reliability": reliability,
               "energy_distance": dist_table,
               "so3": {k: so3[k] for k in ("det_max_final", "orth_max_final")},
               "physics_recovery": phys_m}
    json_path = os.path.join(args.run_dir,
                             "neurips_reviewer_feedback_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {pdf_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()

r"""Compare **two or more `ph_nn_ode` runs** against ground truth on the
$n$-link arm — e.g. "NN predicts whole matrices" vs "NN predicts
sub-components, formula assembles the matrix".

    python src/models/arm_n_link/ph_nn_ode/make_comparison_pdf.py \
        --run_dir <exp1_dir> <exp2_dir>

Relation to the other report generators
---------------------------------------
This is **separate code**, not a refactor of
``ph_gp_sde/make_comparison_pdf.py``, and it deliberately imports nothing from
it. Two reasons:

1. That file carries an explicit warning against re-pointing validated
   ``ph_gp_sde`` code mid-experiment — it produced published numbers.
2. ``ph_gp_sde/network.py`` and ``ph_nn_ode/network.py`` are *both* importable
   as the top-level module name ``network``, and that file does
   ``from network import DissipativeArmHamSDE`` at import time. Importing it
   from here would resolve ``network`` to whichever directory happens to be
   first on ``sys.path`` — a silent, state-dependent wrong-model bug.

The plotting conventions (mean ±2σ bands with every ensemble member drawn
faintly, per-entry matrix grids, ZYX Euler state grids, one row-block per link)
are kept identical to that report so the two are read side by side. Its
structural limits are lifted: everything here is keyed by a **models dict**, and
the port-Hamiltonian scale gauge $\beta$ is fitted **per model** — each run has
its own, so a single shared $\beta$ would mis-scale every subnet of every run
but one.

The comparison environment
--------------------------
Ground truth is built from the specs the runs were **trained** on, read back out
of each ``history.pkl`` (`n`, `friction_coeff`, `varying_friction`, `air_drag`,
`g_diag`, `wind_force_std`). Those specs are then **cross-checked across runs**
and the script refuses to continue if they differ: two models fitted to
different physics have no common ground truth, and a report that quietly picked
one run's environment would look completely normal.

Rollouts
--------
$T$ outer steps of $dt=0.05$ s. **One** torque sequence is drawn per report and
reused by ground truth and by every model, so the comparison is *pathwise*
rather than merely distributional; across the ensemble only the initial
condition varies. The members therefore share their forcing and are
**correlated** — the ± on the summary pages is a spread over initial conditions,
not an i.i.d. uncertainty estimate, and no standard error should be derived
from it.

Summary tables are evaluated on prefixes of that single rollout. This is exact,
not an approximation: the integrator is causal, so a prefix is bit-identical to
a shorter rollout under the same forcing.

What this report does **not** contain, and why
----------------------------------------------
* **No KL pages.** Both variants are point-estimate NNs, so every
  ``weight_kl_loss()`` returns exactly zero — the KL is absent by construction,
  not switched off.
* **No diffusion pages, no $\Sigma\Sigma^\top$ row.** ODE variants have
  $\Sigma_\theta\equiv0$; with ``wind_force_std=0`` the ground-truth wind map is
  scaled by zero too, so both sides of that comparison are identically zero.
* **No chaos floor.** ``ph_gp_sde``'s reports re-roll ground truth under a
  second Wiener path to show the divergence a *perfect* model would still
  incur. With no wind and no observation noise the environment is
  deterministic, so that floor is identically zero and there is nothing to
  plot. Long-horizon divergence is then bounded below by the system's own
  sensitivity to initial conditions, which this report does not estimate —
  read the 5 s and 10 s pages with that in mind.

The omissions are also stated on a note page inside the PDF, so a reader who
only ever sees the PDF is told as well.

Output pages
------------
  A. Summary — one table per horizon (1, 2, 5, 10 s), one column per run,
     trajectory metrics plus that run's own gauge-corrected subnet errors
  A'. Note page: what is omitted and why
  B. Training curves, all runs overlaid — objective, geodesic²/link, MSE ω,
     rotation NLL, rate NLL, pseudo-likelihood
  C. Ensemble dynamics — energy (ensemble + single), |det R − 1|, ‖RᵀR − I‖_F,
     geodesic² vs GT, MSE ω vs GT, state grids (ensemble + single), phase
     portraits
  D. Subnet evolution along N_SUB_TRAJ GT trajectories — M⁻¹ raw, β·M⁻¹,
     D/β, g/β, V/β
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, ".."))
UTILS_DIR = os.path.join(PKG_ROOT, "utils")
PROJECT_ROOT = os.path.abspath(
    os.path.join(THIS_FILE_DIR, "..", "..", "..", ".."))
# NOTE: THIS_FILE_DIR is deliberately NOT added to sys.path. It contains
# `network.py`, and so does every sibling variant directory; putting any of them
# on the path makes the bare name `network` ambiguous. Everything this script
# needs lives in `utils/`, which has no such collision.
MJENV_DIR = os.path.join(PKG_ROOT, "mujoco_env")
for p in (PROJECT_ROOT, PKG_ROOT, UTILS_DIR, MJENV_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

# MuJoCo build: `phys` is this package's own copy of the analytic closed forms
# (used for the GT M, V, D, g matrices on the subnet pages), and the GT
# *trajectories* come from MuJoCo itself -- see `rollout_gt`.
import arm_nlink_physics as phys                                      # noqa: E402
from build_mjcf import ArmSpec                                        # noqa: E402
from mujoco_arm_nlink import MujocoArmEnv                             # noqa: E402
from ph_network_nlink import ArmPortHamiltonian, KeyedArmModel        # noqa: E402
from lie_integrator_nlink import (                                    # noqa: E402
    lie_heun_sde_rollout_nlink, lie_heun_ode_rollout_nlink, exp_so3_batch,
)

# ──────────────────────────────────────────────────────────────────────────
# Constants  (same conventions as ph_gp_sde/make_comparison_pdf.py)
# ──────────────────────────────────────────────────────────────────────────

N_SUBSTEPS = 10                  # must match the environment
DT = 0.05
N_OUTER = 200                    # 200 outer steps of dt=0.05 -> 10 s
TABLE_HORIZONS_S = (1.0, 2.0, 5.0, 10.0)
N_TRAJ_ENSEMBLE = 10             # ensemble for state / energy / SO(3) plots
N_SUB_TRAJ = 5                   # trajectories for subnet-evolution plots
N_BETA_SAMPLES = 200             # configurations used to fit each scale gauge

GT_COLOR = "black"
# Blue first, red second: the pendulum reports use red for the "structured /
# GP" model and blue for the plain NN baseline, and keeping that association
# lets the two report families be read together.
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")

# Physics specs that must agree across the compared runs. `obs_noise_std` is
# excluded on purpose: it changes the *data* the model saw, not the ground truth
# it is scored against, so runs at different noise levels are still comparable
# against one common environment.
# MuJoCo build: the four extra keys matter as much as the original six. Two runs
# that differ in `armature` or `frictionloss` were trained on *different physics*
# -- one inside the structured model class and one outside it -- and comparing
# them against a single ground truth would be exactly the mistake this guard
# exists to prevent. `link_radius` changes the inertia; `dt` changes the flow map.
SHARED_SPEC_KEYS = ("n", "friction_coeff", "varying_friction", "air_drag",
                    "wind_force_std", "g_diag",
                    "link_radius", "armature", "frictionloss", "dt")


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────

def _auto_label(meta):
    """Short human label for a run, from what its architecture actually is.

    Derived from the saved metadata rather than the directory name: the name is
    a convenience string, the metadata is the record.
    """
    a = meta["args"]
    kind = meta.get("subnet_kind", a.get("subnet_kind") or "?")
    # The scale-gauge anchor changes which constants are identifiable at all
    # (see fig_physical_constants), so it belongs in the label: without it two
    # runs differing only in the anchor are indistinguishable and the dedup
    # below would tag them "[1]"/"[2]", which says nothing.
    anch = "anchored" if a.get("anchor_trace") else "no anchor"
    if kind == "nn":
        return f"nn: whole matrices, {anch}"
    if kind == "gp":
        return f"gp: whole matrices, {anch}"
    core = ("nn_core" if a.get("nn_core") else
            "gp_core" if a.get("gp_core") else "constants")
    v = ("+Vshared" if a.get("share_mass_potential") else
         "+V" if a.get("structured_potential") else "")
    if a.get("gravity"):
        v += f", g={a['gravity']:g} fixed"
    return f"structured/{core}{v}: sub-components, {anch}"


def load_run(run_dir):
    """Rebuild an `ArmPortHamiltonian` from a run directory and load its weights.

    Every architecture-affecting flag has to be mirrored here or the skeleton's
    pytree will not match the checkpoint's and `tree_deserialise_leaves` fails
    deep inside equinox with an opaque leaf-path error. `.get` with the old
    default keeps runs that predate a flag loadable.

    `relative_inputs` is passed as the *flag* the run was given, not a resolved
    boolean: the constructor re-derives the effective value (``--nn_core`` and
    ``--gp_core`` force it on, ``n=1`` forces it off), so passing the raw flag
    reproduces training exactly while passing a resolved value could not.
    """
    with open(os.path.join(run_dir, "history.pkl"), "rb") as f:
        meta = pickle.load(f)
    a = meta["args"]
    dtype = jnp.float64 if a.get("float64") else jnp.float32

    subnet_kind = meta.get("subnet_kind") or a.get("subnet_kind")
    if subnet_kind is None:
        raise RuntimeError(
            f"{run_dir}/history.pkl records no 'subnet_kind', so the skeleton "
            f"cannot be rebuilt. It predates the unified ArmPortHamiltonian.")

    # The PL variance scale is a static float, but mirror it anyway so the
    # rebuilt model states the truth about its own run.
    pl_sigma = a.get("pl_sigma_obs")
    if pl_sigma is None:
        pl_sigma = a.get("obs_noise_std", 0.1)

    skeleton = ArmPortHamiltonian(
        key=jax.random.PRNGKey(a["seed"]), n=a["n"],
        subnet_kind=subnet_kind, stochastic=meta["stochastic"],
        hidden_dim=a["hidden_dim"],
        friction=not a.get("no_friction", False),
        relative_inputs=a.get("relative_inputs", False),
        init_gain=a.get("init_gain", 0.5),
        init_sigma_obs_omega=pl_sigma,
        m_epsilon=a.get("m_epsilon", 1.0),
        d_epsilon=a.get("d_epsilon", 0.5),
        i_epsilon=a.get("i_epsilon", None),
        sigma_mode=a.get("sigma_mode", "full"),
        gp_core=a.get("gp_core", False),
        nn_core=a.get("nn_core", False),
        structured_potential=a.get("structured_potential", False),
        share_mass_potential=a.get("share_mass_potential", False),
        gravity=a.get("gravity", None),
        variational_core=a.get("variational_core", False),
        core_prior_std=a.get("core_prior_std", 1.0),
        anchor_m1=a.get("anchor_m1", None),
        anchor_trace=a.get("anchor_trace", None),
        sigma_detach_rollout=a.get("sigma_detach_rollout", True),
        dtype=dtype)
    try:
        model = eqx.tree_deserialise_leaves(
            os.path.join(run_dir, "model.eqx"), skeleton)
    except Exception as exc:                      # noqa: BLE001
        raise RuntimeError(
            f"could not load {run_dir}/model.eqx into the skeleton built from "
            f"its own args (n={a['n']}, subnet_kind={subnet_kind}, "
            f"hidden_dim={a['hidden_dim']}, nn_core={a.get('nn_core')}, "
            f"gp_core={a.get('gp_core')}, "
            f"structured_potential={a.get('structured_potential')}, "
            f"sigma_mode={a.get('sigma_mode', 'full')}). This is an "
            f"architecture mismatch: some flag that changes the model's shape "
            f"is not being mirrored in load_run()."
        ) from exc
    return model, meta, dtype


def comparison_env(metas, run_dirs):
    r"""Ground-truth `ArmParams` for the specs every run was trained on.

    The point of the cross-check: two models fitted to different physics share
    no ground truth, so scoring them against one environment would be
    meaningless — and it would *look* completely normal in the output, which is
    why this raises rather than warns.

    MuJoCo build: the ground truth is whatever MuJoCo's compiler produced, so it
    is recovered by re-compiling the same :class:`ArmSpec` and reading the
    parameters back out — the identical code path
    ``data_gen/mujoco_arm_datagen.py`` used to make the data. Calling
    :func:`phys.uniform_chain_params` here (as the analytic version does) would
    hand back an arm with $\mathbb{I}=\mathrm{diag}(0,0,\cdot)$, which is not
    what was simulated and is not even physically realisable.

    `m`, `link_length` and `g` are not trainer arguments — the datagen defaults
    ($m=1$, $L=1$, $g=9.81$) are the only values the data can have had.

    Returns `(params, spec, ref_args)`.
    """
    ref = metas[0]["args"]
    for meta, rd in zip(metas[1:], run_dirs[1:]):
        a = meta["args"]
        bad = {k: (ref.get(k), a.get(k)) for k in SHARED_SPEC_KEYS
               if tuple(np.atleast_1d(ref.get(k))) != tuple(np.atleast_1d(a.get(k)))}
        if bad:
            detail = "; ".join(f"{k}: {v0!r} vs {v1!r}" for k, (v0, v1) in bad.items())
            raise SystemExit(
                f"refusing to compare runs trained on different physics.\n"
                f"  {run_dirs[0]}\n  {rd}\ndiffer in: {detail}\n"
                f"There is no common ground truth for these, so any table "
                f"would be scoring them against different environments.")
    spec = ArmSpec(
        n=ref["n"],
        link_length=1.0,
        link_radius=ref.get("link_radius", 0.3),
        mass=1.0,
        damping=ref["friction_coeff"],
        gain=tuple(ref.get("g_diag", (1.0, 1.0, 1.0))),
        gravity=9.81,
        timestep=ref.get("mj_timestep", 0.001),
        armature=ref.get("armature", 0.0),
        frictionloss=ref.get("frictionloss", 0.0),
    )
    env = MujocoArmEnv(spec, dt=ref.get("dt", 0.05))
    params = jax.tree_util.tree_map(jnp.asarray, env.params)
    env.close()
    return params, spec, ref


# ──────────────────────────────────────────────────────────────────────────
# Rollouts  (GT and every model driven by the SAME torque sequence)
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


def rollout_gt(env, R0, w0, u_seq):
    r"""Ground-truth rollout **from MuJoCo**. Returns `(T+1, 12n)`.

    This is the substantive difference from the analytic report, and the reason
    the whole package exists: the reference trajectory is produced by an
    independent, widely-used simulator rather than by the same Lie–Heun scheme
    the model is being trained with. A model can no longer look good by
    reproducing our discretisation error, because MuJoCo does not share it.

    A plain Python loop is fine here — the report rolls out a handful of
    trajectories, and MuJoCo steps in microseconds. (The analytic version needed
    a `lax.scan` only because re-tracing an un-jitted JAX function per outer step
    cost minutes.)

    The old signature carried `dW`/`sigma` for a stochastic GT; MuJoCo here is
    deterministic, so they are gone rather than silently ignored.
    """
    obs = env.reset(np.asarray(R0, dtype=np.float64),
                    np.asarray(w0, dtype=np.float64))
    out = [obs]
    for t in range(u_seq.shape[0]):
        out.append(env.step(np.asarray(u_seq[t], dtype=np.float64)))
    return np.stack(out)


def rollout_gt_analytic(params, R0, w0, u_seq, dW, sigma, h):
    """The analytic Lie–Heun reference, kept for integrator cross-checks.

    Not used by the report. `mujoco_env/test_mujoco_matches_gt.py` (T8) shows it
    converges to :func:`rollout_gt` at 2nd order, which is what licenses using
    the closed-form $M,V,D,g$ as ground truth on the subnet pages while the
    trajectories come from MuJoCo.
    """
    return np.asarray(_gt_scan(params, jnp.asarray(R0), jnp.asarray(w0),
                               jnp.asarray(u_seq), jnp.asarray(dW),
                               jnp.asarray(sigma), jnp.asarray(h)))


@eqx.filter_jit
def _model_scan_sde(model, x0, u_seq, h, dW):
    keyed = KeyedArmModel(model=model, keys={}, inference_mode=True)
    return lie_heun_sde_rollout_nlink(keyed, x0, u_seq, h, dW)


@eqx.filter_jit
def _model_scan_ode(model, x0, u_seq, h, n_outer):
    keyed = KeyedArmModel(model=model, keys={}, inference_mode=True)
    return lie_heun_ode_rollout_nlink(keyed, x0, u_seq, h, N_SUBSTEPS, n_outer)


def rollout_model(model, R0, w0, u_seq, dW, h, dtype):
    """One model rollout under the shared forcing. Returns (T+1, 12n).

    The integrator is chosen from the model's own `stochastic` flag rather than
    always taking the SDE path. For a deterministic variant the two are provably
    identical (``test_variants.py::test_ode_equals_sde_zero_noise``, since
    $\\Sigma\\equiv0$ makes every noise increment zero), so this is about
    honesty and cost, not correctness: the ODE path never builds the diffusion
    term at all.
    """
    x0 = jnp.asarray(np.concatenate([np.asarray(R0).reshape(-1),
                                     np.asarray(w0).reshape(-1)]), dtype=dtype)
    if model.stochastic:
        out = _model_scan_sde(model, x0, jnp.asarray(u_seq, dtype),
                              jnp.asarray(h, dtype), jnp.asarray(dW, dtype))
    else:
        out = _model_scan_ode(model, x0, jnp.asarray(u_seq, dtype),
                              jnp.asarray(h, dtype), int(u_seq.shape[0]))
    return np.asarray(out)


# ──────────────────────────────────────────────────────────────────────────
# Subnet evaluation
# ──────────────────────────────────────────────────────────────────────────

@eqx.filter_jit
def _subnets_model(model, q):
    r"""Every learned structure matrix at a batch of configurations.

    `_invariant_q` is what the drift feeds `Dw_net` / `g_net`, so the diagnostic
    must use it too: for the black-box kinds with `relative_inputs` those nets
    were BUILT for the $9(n{-}1)$ relative features and the raw $9n$ vector is a
    shape error, while for the structured kind it returns $q$ unchanged (the
    closed forms contain $T(q)$ literally).
    """
    inv = jax.vmap(model._invariant_q)(q)
    return (jax.vmap(model.M_inv)(q),
            jax.vmap(model.potential)(q),
            jax.vmap(lambda x: model._call(model.Dw_net, x, None))(inv),
            jax.vmap(lambda x: model._call(model.g_net, x, None))(inv),
            jax.vmap(model.Sigma)(q))


def eval_subnets_model(model, traj, n, dtype):
    """Learned subnets along a trajectory, keyed as in the ph_gp_sde report:
    M (T,d,d), V (T,), D (T,d,d), B (T,d,m), Xi (T,d,3)."""
    M, V, D, B, Xi = _subnets_model(
        model, jnp.asarray(traj[:, :9 * n], dtype=dtype))
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
    r"""Port-Hamiltonian scale gauge, fitted from $M^{-1}$ by least squares.

    The dynamics are invariant under $(M,V,D,g,\Sigma)\to\beta(\cdot)$ with
    $p\to\beta p$, because $\partial H_\beta/\partial p_\beta=M_\beta^{-1}p_\beta
    =M^{-1}p$ — and $\omega$ is all the data ever sees. So

    .. math:: \beta^*=\arg\min_\beta\lVert\beta\hat M^{-1}-M_{\rm GT}^{-1}\rVert_F^2

    Recovery for plotting: $M^{-1}\to\beta M^{-1}$ and $V,D,g,\Sigma\to(\cdot)/\beta$.

    **Fitted per model.** Each run sits at its own point along the gauge orbit,
    so one shared $\beta$ would mis-scale every subnet of every run but one.
    """
    Mh, Mg = [], []
    for R in R_samples:
        q = jnp.asarray(np.asarray(R).reshape(-1), dtype=dtype)
        Mh.append(np.asarray(model.M_inv(q)))
        Mg.append(np.asarray(jnp.linalg.inv(
            phys.mass_matrix(params, jnp.asarray(R)))))
    Mh, Mg = np.asarray(Mh), np.asarray(Mg)
    denom = float(np.sum(Mh * Mh))
    return float(np.sum(Mh * Mg) / denom) if denom > 1e-30 else 1.0


def _raw_gauge(beta):
    """True when the report is showing uncorrected subnets (β pinned to 1)."""
    return abs(float(beta) - 1.0) < 1e-12


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
    return np.asarray(jax.vmap(
        lambda r, o: phys.total_energy(params, r, o))(R, w))


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

    Averaged over links, **not** summed, matching `geodesic_sq_all` (which also
    averages) and the single-pendulum reports' `omega_sq` (which sums over the 3
    components of one link, i.e. $n=1$). Summing over $3n$ instead would make a
    2-link arm look $2\times$ worse than an identically-accurate pendulum.

    Broadcasts over leading axes: accepts `(T, D)` or an ensemble `(N, T, D)`.
    """
    return np.sum((pred[..., 9 * n:] - gt[..., 9 * n:]) ** 2, axis=-1) / n


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


# ──────────────────────────────────────────────────────────────────────────
# Plot helpers
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


def fig_curves(curves, title, ylab, logy=True, smooth=True, note=None):
    """One panel, one line per run. `curves`: list of (x, y, label, color)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for x, y, label, color in curves:
        if y is None or len(y) == 0:
            continue
        ax.plot(x, _smooth(y) if smooth else y, lw=1.4, label=label, color=color)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="x-small")
    if note:
        fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=8,
                 style="italic", color="gray")
    # Reserve vertical space per note LINE, or a two-line note overlaps the
    # x-label and the first line runs off the bottom of the page.
    fig.tight_layout(rect=(0, 0.03 * (note.count("\n") + 1) if note else 0.0,
                           1, 1))
    return fig


def fig_energy_ensemble(t, gt_e_all, model_e, n_traj):
    """gt_e_all: (N, T). model_e: dict[label -> (mean, std, color)]."""
    fig, ax = plt.subplots(figsize=(10, 6))
    gt_m, gt_s = gt_e_all.mean(0), gt_e_all.std(0)
    ax.plot(t, gt_m, "k-", lw=2, label="GT Mean")
    ax.fill_between(t, gt_m - 2 * gt_s, gt_m + 2 * gt_s,
                    color="black", alpha=0.15, label="GT ±2σ")
    for name, (em, es, col) in model_e.items():
        ax.plot(t, em, color=col, lw=2, label=name)
        ax.fill_between(t, em - 2 * es, em + 2 * es, color=col, alpha=0.2,
                        label=f"{name} ±2σ")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title(f"Hamiltonian Energy ({n_traj}-traj ensemble)")
    ax.legend(fontsize="x-small")
    ax.grid(True, alpha=0.3)
    fig.text(0.5, 0.012,
             "Model energy is each model's OWN Hamiltonian ½pᵀM⁻¹p + V after its "
             "own gauge correction — not the GT Hamiltonian evaluated at the\n"
             "model's state — so this page compares the learned energy "
             "functions, not just the trajectories they generate.",
             ha="center", va="bottom", fontsize=8, style="italic", color="gray")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return fig


def fig_energy_single(t, gt_e, model_e):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, gt_e, "k-", lw=2, label="GT")
    for name, (e, col) in model_e.items():
        ax.plot(t, e, color=col, lw=2, label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Hamiltonian Energy — single trajectory")
    ax.legend(fontsize="x-small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_error_band_multi(t, err_by_model, title, ylab, saturation=None):
    r"""Per-model error vs time: every ensemble member faint, median bold, 10–90% band.

    `err_by_model`: dict[label -> ((N, T) array, color)].

    **Percentiles, not mean ±2σ** — a deliberate departure from the ph_gp_sde
    report's convention, forced by this page's log axis. These errors span
    decades and are strongly right-skewed, so $\mu-2\sigma$ is *negative* over
    most of the horizon; clamping it to a small positive floor (which is what
    the ±2σ version does) paints a band stretching the full height of the plot
    and hides every curve behind it. The 10th/90th percentiles are positive by
    construction, need no clamp, and describe the same spread honestly. The
    median is plotted for the same reason: with a skewed distribution the mean
    sits above most of the ensemble.

    There is deliberately **no chaos-floor line** (see the module docstring):
    with no wind and no observation noise the environment is deterministic, so
    ground truth re-rolled under a second noise path is bit-identical and the
    floor is exactly zero. A rising curve here is therefore model error *plus*
    the system's intrinsic sensitivity to initial conditions, which this report
    does not separate out.

    `saturation` (geodesic pages only) marks the mean geodesic² between two
    independent uniform-random rotations. Crossing it means the prediction is
    worse than a random guess — and since ground-truth states are concentrated
    rather than uniform, that indicates the model has settled into a *different*
    region, not merely a decorrelated one.
    """
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for name, (err_all, color) in err_by_model.items():
        lo, med, hi = np.percentile(err_all, [10, 50, 90], axis=0)
        for ni in range(err_all.shape[0]):
            ax.plot(t, err_all[ni], color=color, alpha=0.13, lw=0.8)
        ax.plot(t, med, color=color, lw=2, label=f"{name} (median)")
        ax.fill_between(t, lo, hi, color=color, alpha=0.18)
    if saturation is not None:
        ax.axhline(saturation, color="0.45", ls=":", lw=1.8,
                   label=f"saturation (random rotations) = {saturation:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize="x-small", loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.text(0.5, 0.012,
             "Bold = ensemble median, band = 10–90th percentile (not mean ±2σ: "
             "these errors are right-skewed over decades, so μ−2σ is negative\n"
             "and unplottable on a log axis). No chaos-floor line — the "
             "environment is deterministic here (wind_force_std = 0), so GT "
             "re-rolled under a\nsecond noise path is bit-identical and the "
             "floor is exactly 0. Long-horizon growth still includes intrinsic "
             "sensitivity to initial conditions.",
             ha="center", va="bottom", fontsize=8, style="italic", color="gray")
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    return fig


def fig_so3_violation(t, gt_all, model_metric, ylab, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    if gt_all is not None:
        ax.plot(t, gt_all.mean(0), "k--", lw=2, label="GT (mean)")
    for name, (mm, ms, col) in model_metric.items():
        ax.plot(t, mm, color=col, label=f"{name} (mean)")
        ax.plot(t, ms, color=col, linestyle=":", label=f"{name} (single)")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize="x-small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_state_ensemble(t, gt_eul, gt_om, model_states, n, n_traj):
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
            first = (r == 0)
            for ni in range(gt_eul.shape[0]):
                axes[r][0].plot(t, gt_eul[ni, :, li, i], color="black",
                                alpha=0.13, lw=0.8)
                axes[r][1].plot(t, gt_om[ni, :, li, i], color="black",
                                alpha=0.13, lw=0.8)
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
                                    alpha=0.13, lw=0.8)
                    axes[r][1].plot(t, om[ni, :, li, i], color=col,
                                    alpha=0.13, lw=0.8)
                axes[r][0].plot(t, em[:, li, i], color=col, lw=2,
                                label=name if first else None)
                axes[r][0].fill_between(t, em[:, li, i] - 2 * es[:, li, i],
                                        em[:, li, i] + 2 * es[:, li, i],
                                        color=col, alpha=0.18,
                                        label=f"{name} ±2σ" if first else None)
                axes[r][1].plot(t, mm[:, li, i], color=col, lw=2,
                                label=name if first else None)
                axes[r][1].fill_between(t, mm[:, li, i] - 2 * ms[:, li, i],
                                        mm[:, li, i] + 2 * ms[:, li, i],
                                        color=col, alpha=0.18,
                                        label=f"{name} ±2σ" if first else None)

            axes[r][0].set_ylabel(f"L{li+1} {ang_lbl[i]}")
            axes[r][1].set_ylabel(f"L{li+1} {om_lbl[i]}")
            axes[r][0].grid(True, alpha=0.3)
            axes[r][1].grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Time (s)")
    axes[-1][1].set_xlabel("Time (s)")
    axes[0][0].legend(fontsize="x-small")
    axes[0][1].legend(fontsize="x-small")
    fig.suptitle(f"State Trajectories ({n_traj}-traj ensemble)",
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
            axes[r][0].grid(True, alpha=0.3)
            axes[r][1].grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Time (s)")
    axes[-1][1].set_xlabel("Time (s)")
    axes[0][0].legend(fontsize="x-small")
    axes[0][1].legend(fontsize="x-small")
    fig.suptitle("Single-trajectory State Comparison",
                 fontsize=14, fontweight="bold")
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
                        color="black", alpha=0.13, lw=0.8)
            ax.plot(gt_eul.mean(0)[:, li, i], gt_om.mean(0)[:, li, i], "k-",
                    lw=2, label="GT Mean" if first else None)
            for name, (eul, om, col) in model_states.items():
                for ni in range(eul.shape[0]):
                    ax.plot(eul[ni, :, li, i], om[ni, :, li, i], color=col,
                            alpha=0.13, lw=0.8)
                ax.plot(eul.mean(0)[:, li, i], om.mean(0)[:, li, i], color=col,
                        lw=2, label=name if first else None)
            ax.set_xlabel(f"L{li+1} {ang_lbl[i]}")
            ax.set_ylabel(f"L{li+1} {om_lbl[i]}")
            ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize="x-small")
    fig.suptitle("Phase Portraits (Angle vs Omega)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    return fig


# ── Subnet evolution along multiple trajectories ──────────────────────────

def _figure_legend(fig, entries, wrap_at=45):
    r"""Legend on the FIGURE, below the axes. Returns the bottom fraction used.

    Deliberately not an axes legend. ``tight_layout`` folds a child legend into
    its parent axes' tight bbox, so a legend wider than its panel inflates that
    bbox -- and because the grid is uniform, *every* panel then shrinks to make
    room. The result is narrow plots floating in wide gutters. It scales with
    label length, which is why it appeared only once the auto labels grew the
    anchor and gravity suffixes (~55 chars) and never with short explicit
    ``--labels``. A figure legend belongs to no axes and cannot distort a grid.
    """
    handles = [mlines.Line2D([], [], **kw) for kw in entries]
    longest = max(len(h.get_label()) for h in handles)
    ncol = 1 if longest > wrap_at else len(handles)
    nrow = int(np.ceil(len(handles) / ncol))
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               fontsize="small", frameon=True)
    return min(0.32, (0.26 * nrow + 0.22) / fig.get_size_inches()[1])


def is_structured(model):
    """True when the model's subnets are closed forms over physical constants.

    Duck-typed on the readout the constants page needs rather than on
    `subnet_kind`, so a future subnet family that exposes `physical()` is
    included automatically instead of being silently skipped.
    """
    return (model.subnet_kind == 'structured'
            and hasattr(model.M_net, 'physical')
            and hasattr(model.Dw_net, 'physical')
            and hasattr(model.g_net, 'physical'))


@eqx.filter_jit
def _physical_structured(model, q):
    r"""$(m,\operatorname{diag}\mathbb I,\ell,c,d,\gamma)$ at a batch of configurations."""
    def one(qi):
        m, I_body, ell, c = model.M_net.physical(qi, None, inference_mode=True)
        return (m, jnp.diagonal(I_body, axis1=1, axis2=2), ell, c,
                model.Dw_net.physical(qi, None, inference_mode=True),
                model.g_net.physical(qi, None, inference_mode=True))
    return jax.vmap(one)(q)


def eval_physical_structured(model, traj, n, dtype):
    """The physical sub-components a structured model predicts, along a trajectory.

    Read through each subnet's own ``physical()`` rather than re-deriving
    ``exp(log_d + kappa*tanh(core(q)))`` here -- the point of the page is to show
    what the model uses, so the readout must not be a second implementation.

    Gravity is handled separately because the two potential modules expose it
    differently: :class:`SharedStructuredPotential` has ``gravity()`` (no $q$, and
    a Python float when `--gravity` fixed it), while `StructuredPotential` folds
    $g$ into its ``physical()`` tuple and can make it $q$-dependent via its core.
    Returns ``g=None`` for a black-box V_net, which has no gravity to report.
    """
    q = jnp.asarray(traj[:, :9 * n], dtype=dtype)
    m, I_d, ell, c, d, gain = _physical_structured(model, q)

    V = model.V_net
    if hasattr(V, 'gravity'):                      # SharedStructuredPotential
        g = np.full(q.shape[0], float(V.gravity(None, True)))
    elif hasattr(V, 'levers'):                     # StructuredPotential
        g = np.asarray(jax.vmap(
            lambda qi: V.physical(qi, None, True)[3])(q))
    else:
        g = None
    return {'m': np.asarray(m), 'I_diag': np.asarray(I_d),
            'ell': np.asarray(ell), 'c': np.asarray(c), 'd': np.asarray(d),
            'gain': np.asarray(gain), 'g': g}


def fig_physical_constants(t, phys_list, params, n, label, color, n_cols=6):
    r"""One panel per learned scalar: its value along the trajectories vs truth.

    `phys_list` is one dict per trajectory, from :func:`eval_physical_structured`.

    Every panel is a *constant* in the ground truth (dashed black), so any slope
    or wobble in the coloured curves is the ``nn_core`` MLP injecting spurious
    configuration dependence into a quantity that has none.

    .. important::
       These are the **raw** learned values, with no gauge correction -- that is
       what makes the page meaningful, and also why some panels sit off truth
       even when the model reproduces the dynamics essentially exactly.
       ``anchor_trace`` renormalises $M$ by $\alpha$, so scaling
       $(m,\mathbb I,\ell,c)$ such that $M_{\rm raw}$ and $V_{\rm raw}$ both
       scale by $k$ is invisible in the dynamics.
    """
    ell_free = max(n - 1, 1)     # ell_{n-1} never enters the dynamics
    panels = []                  # (title, per-traj series, truth or None)

    def add(title, get, truth):
        panels.append((title, [get(p) for p in phys_list], truth))

    for i in range(n):
        add(f'm  [link {i+1}]', lambda p, i=i: p['m'][:, i], float(params.m[i]))
    for j in range(ell_free):
        for a, ax in enumerate('xyz'):
            add(f'ell_{j+1} {ax}', lambda p, j=j, a=a: p['ell'][:, j, a],
                float(params.ell[j, a]))
    for i in range(n):
        for a, ax in enumerate('xyz'):
            add(f'c_{i+1} {ax}', lambda p, i=i, a=a: p['c'][:, i, a],
                float(params.c[i, a]))
    for i in range(n):
        for a, ax in enumerate('xyz'):
            add(f'I_{i+1} {ax}{ax}', lambda p, i=i, a=a: p['I_diag'][:, i, a],
                float(params.I_body[i, a, a]))
    for i in range(n):
        add(f'd  [joint {i+1}]', lambda p, i=i: p['d'][:, i], float(params.d[i]))
    for i in range(n):
        for a, ax in enumerate('xyz'):
            add(f'gain_{i+1} {ax}', lambda p, i=i, a=a: p['gain'][:, i, a],
                float(params.gain[i, a]))
    if phys_list[0]['g'] is not None:
        add('g  (gravity)', lambda p: p['g'], float(params.g))

    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols + 1, 2.1 * n_rows + 2.4),
                             squeeze=False)
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        if idx >= len(panels):
            ax.axis('off')
            continue
        title, series, truth = panels[idx]
        if truth is not None:
            ax.axhline(truth, color='k', ls='--', lw=1.2)
        for s_ in series:
            ax.plot(t, s_, color=color, lw=1.1, alpha=0.75)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=8)
    # Figure legend, not axes[0][0] -- see _figure_legend.
    bottom = _figure_legend(fig, [dict(color='k', ls='--', lw=1.2,
                                       label='ground truth'),
                                  dict(color=color, lw=1.8, label=label)])
    fig.suptitle(f'Physical sub-components predicted by {label}\n'
                 f'(raw learned values, no gauge correction)',
                 fontsize=13, fontweight='bold')
    fig.text(0.5, bottom - 0.035,
             'Every quantity here is a CONSTANT in the ground truth (dashed), so '
             'any variation in the coloured curves is the nn_core MLP injecting\n'
             'configuration dependence that the true system does not have. '
             'anchor_trace renormalises M by alpha = anchor / tr M_raw, so '
             'scaling (m, I, ell, c) to multiply\nboth M_raw and V_raw by k is '
             'exactly invisible in the dynamics: m, c and I ride that orbit, '
             'while ell_0, d and gain do not.',
             ha='center', va='top', fontsize=8, style='italic', color='gray')
    fig.tight_layout(rect=(0, bottom + 0.05, 1, 0.945))
    return fig


def plot_matrix_grid_multi(t, comps_by_model, comps_gt, key, title, rows, cols,
                           subtitle=None):
    """rows x cols grid, one panel per matrix entry, all trajectories overlaid.

    `comps_by_model`: dict[label -> (list of subnet dicts, color)].
    GT is drawn per-trajectory as a dashed black line -- unlike the single
    pendulum's constant M^-1, it is state-dependent here.

    Legend goes on the figure, not in panel (0,0): see :func:`_figure_legend`.
    `subtitle` (the per-model beta values) is a separate small line rather than
    part of the suptitle, which with three long labels ran to ~200 characters.
    """
    fig, axes = plt.subplots(rows, cols,
                             figsize=(2.6 * cols + 2, 2.4 * rows + 2),
                             squeeze=False)
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            for ti in range(len(comps_gt)):
                ax.plot(t, comps_gt[ti][key][:, r, c], "k--", lw=1.0, alpha=0.6)
            for _, (comps, col) in comps_by_model.items():
                for ti in range(len(comps)):
                    ax.plot(t, comps[ti][key][:, r, c], color=col,
                            alpha=0.7, lw=1.2)
            ax.set_title(f"({r},{c})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if r == rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    entries = [dict(color="k", ls="--", lw=1.2, label="GT")]
    entries += [dict(color=col, lw=1.8, label=name)
                for name, (_, col) in comps_by_model.items()]
    bottom = _figure_legend(fig, entries)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    top = 0.97
    if subtitle:
        fig.text(0.5, 0.962, subtitle, ha="center", va="top", fontsize=8.5,
                 style="italic", color="gray")
        top = 0.94
    fig.tight_layout(rect=(0, bottom, 1, top))
    return fig


def plot_potential_multi(t, comps_by_model, comps_gt, title, subtitle=None):
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for ti in range(len(comps_gt)):
        v = comps_gt[ti]["V"]
        ax.plot(t, v - v.mean(), "k--", lw=1.0, alpha=0.6)
    for _, (comps, col) in comps_by_model.items():
        for ti in range(len(comps)):
            v = comps[ti]["V"]
            ax.plot(t, v - v.mean(), color=col, alpha=0.7, lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("V(q) - mean")
    ax.set_title(title + "   (centred: H is defined up to a constant)")
    ax.grid(True, alpha=0.3)

    entries = [dict(color="k", ls="--", lw=1.2, label="GT")]
    entries += [dict(color=col, lw=1.8, label=name)
                for name, (_, col) in comps_by_model.items()]
    bottom = _figure_legend(fig, entries)
    if subtitle:
        fig.text(0.5, 0.965, subtitle, ha="center", va="top", fontsize=8.5,
                 style="italic", color="gray")
    fig.tight_layout(rect=(0, bottom, 1, 0.93 if subtitle else 1.0))
    return fig


# ── Summary table ─────────────────────────────────────────────────────────

def _compute_table_metrics(params, gt_all, pred_all, n, steps=None):
    """Summary metrics over the first `steps` outer steps (all of them if None).

    Every row aggregates over 0..steps: the "Traj-mean" rows average that window
    and the "Final-step" rows read its last entry. Slicing here rather than
    re-rolling is exact — the rollouts are causal, so a prefix of a long rollout
    is bit-identical to a short rollout with the same forcing.
    """
    N = gt_all.shape[0]
    if steps is not None:
        gt_all = gt_all[:, :steps + 1]
        pred_all = pred_all[:, :steps + 1]
    geo_sq = np.stack([geodesic_sq_all(pred_all[i], gt_all[i], n)
                       for i in range(N)])
    om_sq = omega_sq_all(pred_all, gt_all, n)

    det_max = np.stack([so3_det_residual_abs(pred_all[i], n)
                        for i in range(N)]).max(1)
    orth_max = np.stack([so3_orth_residual(pred_all[i], n)
                         for i in range(N)]).max(1)

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


def _wrap(label, width=24):
    """Break a long column label over lines so the summary table stays legible."""
    return "\n".join(textwrap.wrap(label, width=width)) or label


def fig_comparison_table(params, gt_all, models, n, betas, subnet_err_by_model,
                         u_label="", steps=None, horizon_s=None,
                         param_counts=None):
    """One column per run: trajectory metrics, then that run's subnet errors.

    `models`: dict[label -> (T+1, 12n) or (N, T+1, 12n) predictions.
    `betas` / `subnet_err_by_model`: keyed by the same labels — each run has its
    own scale gauge, so its subnet errors are corrected with its own β.
    """
    model_names = list(models.keys())
    per_model = {name: _compute_table_metrics(params, gt_all, models[name], n,
                                              steps)
                 for name in model_names}
    metric_names = list(next(iter(per_model.values())).keys())
    cell_text = [[per_model[m][k] for m in model_names] for k in metric_names]

    if param_counts:
        metric_names.append("Trainable parameters")
        cell_text.append([f"{param_counts.get(m, 0):,}" for m in model_names])

    # Per-model gauge-corrected subnet errors, one value per column.
    for key in ("M⁻¹", "V (centred)", "D", "g"):
        metric_names.append(f"Subnet rel. error: {key}")
        cell_text.append([f"{subnet_err_by_model[m][key]:.4f}"
                          for m in model_names])
    metric_names.append("Fitted scale gauge β")
    cell_text.append([f"{betas[m]:.4f}" for m in model_names])

    N = gt_all.shape[0]
    single = N == 1
    hz = "" if horizon_s is None else f"horizon {horizon_s:g} s — "
    # Widen with the column count, and leave headroom for wrapped headers.
    fig, ax = plt.subplots(figsize=(6.5 + 3.4 * len(model_names),
                                    2.6 + 0.55 * len(metric_names)))
    ax.axis("off")
    ax.set_title(
        (f"{hz}single rollout vs GT (n={n} links)" if single else
         f"{hz}{N}-rollout ensemble vs GT (n={n} links)")
        + (f"\ncontrol: {u_label}" if u_label else "")
        + "\n("
        + ("single trajectory, no aggregation;  " if single
           else "per-trajectory aggregates: mean ± std;  ")
        + "each column's subnet errors use that column's own β)",
        fontsize=13, fontweight="bold", pad=14)

    wrapped = [_wrap(m) for m in model_names]
    table = ax.table(cellText=cell_text, rowLabels=metric_names,
                     colLabels=wrapped, cellLoc="center", rowLoc="left",
                     loc="center", colWidths=[0.30] * len(model_names))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)
    # The header row keeps the default one-line height, so a wrapped label is
    # drawn outside its own cell and the last line is clipped by the row rule.
    # Scale that row by the number of lines it actually has.
    n_hdr = max(lbl.count("\n") for lbl in wrapped) + 1
    if n_hdr > 1:
        # Iterate the cells that exist: the row-label corner (0, -1) is only
        # created in some matplotlib configurations, so indexing it directly
        # raises KeyError.
        for (r, _), cell in table.get_celld().items():
            if r == 0:
                cell.set_height(cell.get_height() * n_hdr)
    for j in range(len(model_names)):
        table[0, j].set_text_props(weight="bold")

    fig.text(
        0.5, 0.02,
        ("GT and every model are driven by the identical torque sequence, so "
         "the comparison is pathwise, not merely distributional.\n"
         if single else
         "One torque sequence drives GT and all "
         f"{N} rollouts of every model; only the initial condition varies, so "
         "the comparison is pathwise, not merely distributional.\n"
         "± is therefore the spread over initial conditions at fixed forcing, "
         "not an i.i.d. uncertainty estimate.\n")
        + "Subnet errors are relative Frobenius after fitting the scale gauge β "
          "(V additionally centred, since H is defined up to a constant).",
        ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    return fig


def fig_omissions_note(specs, u_label, n_traj, T, phys_skipped=()):
    """A page stating what this report leaves out and why.

    A reader who only ever sees the PDF should not have to infer that the
    missing KL and Σ pages were omitted deliberately rather than lost.
    """
    fig = plt.figure(figsize=(14, 9))
    fig.text(0.5, 0.95, "Scope of this report", ha="center", fontsize=17,
             fontweight="bold")
    body = (
        "COMPARISON ENVIRONMENT — built from the specs the runs were trained on,\n"
        "read back out of each run's history.pkl and cross-checked between runs:\n"
        f"    links n                 : {specs['n']}\n"
        f"    joint friction d        : {specs['friction_coeff']}  "
        f"(varying_friction = {specs.get('varying_friction', False)})\n"
        f"    actuator gain diag(Γ)   : {tuple(specs.get('g_diag', (1.0, 1.0, 1.0)))}\n"
        f"    stochastic wind σ       : {specs.get('wind_force_std', 0.0)}\n"
        f"    air drag κ              : {specs.get('air_drag', 0.0)}\n"
        f"    observation noise (data): {specs.get('obs_noise_std', 0.0)}\n"
        f"    m = 1, L = 1, COM at tip, unit axial inertia, g = 9.81  (env defaults)\n"
        "\n"
        f"ROLLOUTS — {n_traj} initial conditions x {T} outer steps of dt = {DT} s "
        f"= {T * DT:g} s.\n"
        f"    control: {u_label}\n"
        "    One torque sequence, shared by GT and every model: the comparison is\n"
        "    pathwise. Ensemble members share their forcing and are therefore\n"
        "    correlated -- the reported +- is a spread over initial conditions, NOT\n"
        "    an i.i.d. uncertainty estimate. Do not derive a standard error from it.\n"
        "\n"
        "DELIBERATELY OMITTED\n"
        "  * KL pages. Both variants are point-estimate neural networks, so every\n"
        "    weight_kl_loss() returns exactly zero. The KL is absent by\n"
        "    construction, not switched off by --beta_max.\n"
        "  * Diffusion pages and the SS^T subnet row. These are ODE variants, so\n"
        "    Sigma_theta == 0 identically; and with wind_force_std = 0 the GT wind\n"
        "    map is scaled by zero too, so both sides of that comparison vanish.\n"
        "  * Chaos floor. ph_gp_sde's reports re-roll GT under a second Wiener path\n"
        "    to show the divergence a PERFECT model would still incur. With no wind\n"
        "    and no observation noise the environment is deterministic, so that\n"
        "    floor is identically zero and there is nothing to plot.\n"
        "\n"
        + ("  * Physical-constants page for: " + ", ".join(phys_skipped) + ".\n"
           "    Those runs use black-box subnets that emit matrix entries\n"
           "    directly, so they have no physical sub-components to plot.\n"
           if phys_skipped else "")
        + "\n"
        "READ THE LONG HORIZONS WITH CARE\n"
        "    The training window was num_points - 1 = 4 outer steps = 0.2 s, so even\n"
        "    the 1 s table is already a 5x extrapolation. A 2-link arm on SO(3)^2 is\n"
        "    chaotic: at 5 s and 10 s, error growth reflects intrinsic sensitivity to\n"
        "    initial conditions as well as model error, and this report does not\n"
        "    separate the two.\n"
    )
    fig.text(0.06, 0.90, body, ha="left", va="top", fontsize=10.5,
             family="monospace")
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def build_u_sequence(u_const, T, n, random_u, random_u_scale, rng):
    r"""Control sequence for one trajectory, `(T, 3n)`.

    `u_const=None` reproduces the training distribution: a fresh draw
    $u_t\sim\mathcal U(-s,s)^{3n}$ per outer step. Otherwise the control is held
    **constant** for every step, which isolates the drift from the torque
    excitation. Note a *nonzero* constant still excites $g(q)$ — unlike $u=0$,
    which leaves the input map entirely unidentified.
    """
    if u_const is None:
        if not random_u:
            return np.zeros((T, 3 * n))
        return rng.uniform(-random_u_scale, random_u_scale, size=(T, 3 * n))
    u = np.asarray(u_const, dtype=np.float64).reshape(-1)
    if u.size == 1:
        u = np.repeat(u, 3 * n)
    elif u.size == 3:
        u = np.tile(u, n)                    # matching the datagen's `us`
    if u.size != 3 * n:
        raise SystemExit(f"--u needs 1, 3 or {3 * n} values, got {u.size}")
    return np.tile(u[None, :], (T, 1))


def _u_label(u_const, random_u, random_u_scale):
    if u_const is None:
        return (f"random u ~ U(-{random_u_scale:g}, {random_u_scale:g})"
                f"^3n per step (training distribution)"
                if random_u else "u = 0 (constant)")
    u = np.asarray(u_const, dtype=np.float64).reshape(-1)
    if u.size == 1:
        return f"u = {u[0]:g} at every component (constant)"
    return "u = " + np.array2string(u, precision=3) + " (constant)"


def main():
    ap = argparse.ArgumentParser(
        description="Compare two or more ph_nn_ode runs against ground truth "
                    "on the n-link SO(3)^n arm.")
    ap.add_argument("--run_dir", nargs="+", required=True, metavar="DIR",
                    help="two or more run directories, each holding model.eqx "
                         "and history.pkl. Column order follows this order.")
    ap.add_argument("--labels", nargs="+", default=None, metavar="LBL",
                    help="override the auto-generated column labels (one per "
                         "--run_dir). Defaults are derived from each run's "
                         "saved architecture flags, not its directory name.")
    ap.add_argument("--out", default=None,
                    help="output PDF. Default: comparison_report.pdf in the "
                         "FIRST run directory.")
    ap.add_argument("--n_outer", type=int, default=None,
                    help="rollout length in outer steps of dt=0.05. Default: "
                         "just long enough for the longest --horizons entry, "
                         "so `--horizons 1` rolls out 1 s rather than the full "
                         f"{N_OUTER * DT:g} s.")
    ap.add_argument("--horizons", type=float, nargs="+", metavar="SEC",
                    default=list(TABLE_HORIZONS_S),
                    help="summary-table horizons in seconds, one page each "
                         f"(default: "
                         f"{' '.join(f'{x:g}' for x in TABLE_HORIZONS_S)}).")
    ap.add_argument("--tables_only", action="store_true",
                    help="emit only the summary and note pages. The 3n x 3n "
                         "subnet grids dominate rendering time.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--u", type=float, nargs="+", default=(1.0,), metavar="VAL",
                    help="hold the control CONSTANT at this value (default: 1 "
                         "at every component). Give 1 value (broadcast to all "
                         "3n), 3 (tiled per joint, matching the datagen's "
                         "`us`), or 3n (used as-is). Pass --random_u instead "
                         "for a fresh draw per step at the run's own scale.")
    ap.add_argument("--random_u", action="store_true",
                    help="resample u every outer step from the training "
                         "distribution instead of holding it constant.")
    ap.add_argument("--no_gauge", action="store_true",
                    help="pin every beta to 1 instead of fitting it, i.e. show "
                         "the RAW learned subnets. The port-Hamiltonian "
                         "dynamics ARE invariant under (M,V,D,g,Sigma) -> "
                         "beta(.) with p -> beta*p, so a raw mismatch is not by "
                         "itself an error -- read this alongside the "
                         "gauge-corrected report, not instead of it.")
    ap.add_argument("--n_traj", type=int, default=N_TRAJ_ENSEMBLE,
                    help=f"ensemble size (default {N_TRAJ_ENSEMBLE})")
    args = ap.parse_args()

    run_dirs = [os.path.abspath(d.rstrip("/")) for d in args.run_dir]
    if len(run_dirs) < 2:
        raise SystemExit("--run_dir needs at least two directories; use "
                         "ph_gp_sde/make_comparison_pdf.py for a single run.")
    if len(run_dirs) > len(PALETTE):
        raise SystemExit(f"at most {len(PALETTE)} runs can be distinguished by "
                         f"the colour palette; got {len(run_dirs)}.")

    # ── load ──
    loaded = [load_run(d) for d in run_dirs]
    models = [m for m, _, _ in loaded]
    metas = [meta for _, meta, _ in loaded]
    dtypes = [dt for _, _, dt in loaded]

    if args.labels:
        if len(args.labels) != len(run_dirs):
            raise SystemExit(f"--labels needs {len(run_dirs)} entries, "
                             f"got {len(args.labels)}")
        labels = list(args.labels)
    else:
        labels = [_auto_label(m) for m in metas]
    if len(set(labels)) != len(labels):
        # Identical labels would silently collapse dict-keyed columns into one.
        labels = [f"{lb} [{i+1}]" for i, lb in enumerate(labels)]
    colors = {lb: PALETTE[i] for i, lb in enumerate(labels)}

    params, gt_spec, specs = comparison_env(metas, run_dirs)
    gt_env = MujocoArmEnv(gt_spec, dt=specs.get("dt", 0.05))
    n = specs["n"]
    sigma = specs.get("wind_force_std", 0.0)
    h = DT / N_SUBSTEPS
    T = (args.n_outer if args.n_outer is not None
         else max(1, int(round(max(args.horizons) / DT))))
    t_eval = np.arange(T + 1) * DT
    n_traj = max(1, int(args.n_traj))

    out = args.out or os.path.join(run_dirs[0], "comparison_report.pdf")

    param_counts = {lb: sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(mdl, eqx.is_array))) for lb, mdl in zip(labels, models)}

    print(f"comparison environment: n={n}  d={specs['friction_coeff']}  "
          f"g_diag={tuple(specs.get('g_diag', (1., 1., 1.)))}  "
          f"wind_sigma={sigma}  varfric={specs.get('varying_friction', False)}")
    for lb, rd, meta, dt in zip(labels, run_dirs, metas, dtypes):
        print(f"  {lb:<42} {os.path.basename(rd)}  "
              f"steps={meta['args']['total_steps']}  "
              f"loss={meta['args'].get('loss', 'nll')}  "
              f"dtype={jnp.dtype(dt).name}  params={param_counts[lb]:,}")
    print(f"  horizon: {T} outer steps x dt={DT} = {T * DT:.2f} s "
          f"(training window was num_points-1 = "
          f"{specs.get('num_points', 5) - 1} steps = "
          f"{(specs.get('num_points', 5) - 1) * DT:.2f} s)")

    # ── per-model scale gauge ──
    R_beta = np.asarray(exp_so3_batch(jax.random.normal(
        jax.random.PRNGKey(args.seed + 7), (N_BETA_SAMPLES * n, 3),
        dtype=jnp.float64)).reshape(N_BETA_SAMPLES, n, 3, 3))
    betas = {}
    for lb, mdl, dt in zip(labels, models, dtypes):
        fitted = estimate_beta(mdl, params, R_beta, dt)
        betas[lb] = 1.0 if args.no_gauge else fitted
        note = (f" (--no_gauge: pinned to 1; fitted would be {fitted:.4f})"
                if args.no_gauge else "")
        print(f"  beta[{lb}] = {betas[lb]:.4f}{note}")

    # ── rollouts: ONE torque sequence shared by GT and every model ──
    u_const = None if args.random_u else args.u
    u_label = _u_label(u_const, specs.get("random_u", True),
                       specs.get("random_u_scale", 1.0))
    print(f"Rolling out {n_traj} GT + {n_traj} per model "
          f"({u_label}; only the initial condition varies) ...")
    rng = np.random.default_rng(args.seed)
    u_seq = build_u_sequence(u_const, T, n, specs.get("random_u", True),
                             specs.get("random_u_scale", 1.0), rng)
    # Wiener path: inert here (GT sigma is 0 and the models are ODE variants),
    # but generated and threaded so the GT scan signature is unchanged and a
    # future stochastic run needs no new code path.
    dW = rng.normal(0.0, np.sqrt(h), size=(T, N_SUBSTEPS, 3))
    if sigma == 0.0:
        print("  note: wind sigma = 0, so the Wiener path has no effect on GT; "
              "the models are ODE variants, so it has none on them either.")

    gt_all, md_all = [], {lb: [] for lb in labels}
    for ti in range(n_traj):
        key = jax.random.PRNGKey(args.seed * 1000 + ti)
        R0 = exp_so3_batch(jax.random.normal(key, (n, 3), dtype=jnp.float64))
        w0 = rng.uniform(-1.0, 1.0, size=(n, 3))
        gt_all.append(rollout_gt(gt_env, R0, w0, u_seq))
        for lb, mdl, dt in zip(labels, models, dtypes):
            md_all[lb].append(rollout_model(mdl, R0, w0, u_seq, dW, h, dt))
    gt_all = np.stack(gt_all)
    md_all = {lb: np.stack(v) for lb, v in md_all.items()}

    # ── subnet evolution along the first N_SUB_TRAJ GT trajectories ──
    n_sub = min(N_SUB_TRAJ, n_traj)
    print(f"Evaluating subnets along {n_sub} trajectories ...")
    comps_gt = [eval_subnets_gt(params, gt_all[i], n, sigma)
                for i in range(n_sub)]
    comps_raw, comps_md = {}, {}
    for lb, mdl, dt in zip(labels, models, dtypes):
        comps_raw[lb] = [eval_subnets_model(mdl, gt_all[i], n, dt)
                         for i in range(n_sub)]
        comps_md[lb] = [apply_beta(c, betas[lb]) for c in comps_raw[lb]]

    # Physical sub-components: only the structured kind has any. A black-box
    # subnet emits matrix entries directly, so there is nothing to report --
    # said out loud here and on the scope page rather than a page quietly missing.
    phys_by_model, phys_skipped = {}, []
    for lb, mdl, dt in zip(labels, models, dtypes):
        if is_structured(mdl):
            phys_by_model[lb] = [eval_physical_structured(mdl, gt_all[i], n, dt)
                                 for i in range(n_sub)]
        else:
            phys_skipped.append(lb)
    if phys_skipped:
        print("  no physical-constants page for "
              + ", ".join(phys_skipped)
              + " (black-box subnets emit matrix entries, not constants)")

    def rel(A, B):
        return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-30))

    def subnet_errors_over(cs_md, cs_gt, steps=None):
        r"""Relative Frobenius error of each subnet over the given trajectories.

        `cs_gt` is passed explicitly rather than closed over: the
        single-trajectory page scores one model trajectory, and pairing that
        against the whole GT pool is a shape error at best and a silently
        mismatched comparison at worst. `cs_md` and `cs_gt` must cover the SAME
        trajectories in the same order.

        `steps` restricts the comparison to the first `steps` outer steps, so a
        per-horizon page scores the subnets over the same window its trajectory
        metrics use.

        No $\Sigma\Sigma^\top$ row: these are ODE variants, so $\Sigma\equiv0$,
        and with the wind off the ground truth is zero too — the ratio would be
        0/0.
        """
        if len(cs_md) != len(cs_gt):
            raise ValueError(f"model/GT trajectory counts differ: "
                             f"{len(cs_md)} vs {len(cs_gt)}")
        sl = slice(None) if steps is None else slice(0, steps + 1)
        cat = lambda cs, k: np.concatenate([c[k][sl] for c in cs])
        Mh, Mg = cat(cs_md, "M"), cat(cs_gt, "M")
        Dh, Dg = cat(cs_md, "D"), cat(cs_gt, "D")
        Vh, Vg = cat(cs_md, "V"), cat(cs_gt, "V")
        Bh, Bg = cat(cs_md, "B"), cat(cs_gt, "B")
        return {
            "M⁻¹": rel(Mh, Mg),
            "V (centred)": rel(Vh - Vh.mean(), Vg - Vg.mean()),
            "D": rel(Dh, Dg),
            # g is easy to leave unidentified: it only gets signal when the
            # torque actually moves the system, so a weak control leaves it
            # unconstrained while every other subnet still looks fine.
            "g": rel(Bh, Bg),
        }

    table_steps = [(hs, int(round(hs / DT))) for hs in args.horizons]
    dropped = [hs for hs, k in table_steps if k > T]
    table_steps = [(hs, k) for hs, k in table_steps if k <= T]
    if dropped:
        print(f"  skipping table horizons {dropped} s: beyond the "
              f"{T * DT:g} s rollout (raise --n_outer to include them)")
    err_by_h = {hs: {lb: subnet_errors_over(comps_md[lb], comps_gt, k)
                     for lb in labels}
                for hs, k in table_steps}

    # ── derived ensemble quantities ──
    if not args.tables_only:
        gt_eul = np.stack([euler_all_links(gt_all[i], n) for i in range(n_traj)])
        gt_om = gt_all[..., 9 * n:].reshape(n_traj, T + 1, n, 3)
        gt_e = np.stack([get_energy(params, gt_all[i], n) for i in range(n_traj)])

        md_eul, md_om, md_e = {}, {}, {}
        for lb, mdl, dt in zip(labels, models, dtypes):
            md_eul[lb] = np.stack([euler_all_links(md_all[lb][i], n)
                                   for i in range(n_traj)])
            md_om[lb] = md_all[lb][..., 9 * n:].reshape(n_traj, T + 1, n, 3)
            md_e[lb] = np.stack([hamiltonian_from_subnets(
                md_all[lb][i],
                apply_beta(eval_subnets_model(mdl, md_all[lb][i], n, dt),
                           betas[lb]), n) for i in range(n_traj)])

    geo_err = {lb: np.stack([geodesic_sq_all(md_all[lb][i], gt_all[i], n)
                             for i in range(n_traj)]) for lb in labels}
    om_err = {lb: omega_sq_all(md_all[lb], gt_all, n) for lb in labels}
    sat = random_rotation_saturation()
    for lb in labels:
        print(f"  {lb:<42} traj-mean geodesic^2 {geo_err[lb].mean():.4f}   "
              f"||dw||^2/link {om_err[lb].mean():.4f}")

    print(f"Writing {out} ...")
    with PdfPages(out) as pdf:
        # ── A. summary tables, one page per horizon ──
        for hs, k in table_steps:
            pdf.savefig(fig_comparison_table(
                params, gt_all, md_all, n, betas, err_by_h[hs],
                u_label=u_label, steps=k, horizon_s=hs,
                param_counts=param_counts))
            plt.close("all")
        # The single-rollout page scores the subnets along the one trajectory it
        # plots, so its numbers match the curves shown later for that same one.
        pdf.savefig(fig_comparison_table(
            params, gt_all[:1], {lb: v[:1] for lb, v in md_all.items()}, n,
            betas, {lb: subnet_errors_over(comps_md[lb][:1], comps_gt[:1])
                    for lb in labels},
            u_label=u_label, param_counts=param_counts))
        plt.close("all")

        # ── A'. scope / omissions ──
        pdf.savefig(fig_omissions_note(
            dict(specs, obs_noise_std=specs.get("obs_noise_std", 0.0)),
            u_label, n_traj, T, phys_skipped=phys_skipped))
        plt.close("all")

        if args.tables_only:
            print("  --tables_only: skipping curve, dynamics and subnet pages.")
            print(f"Done: {out}")
            return

        # ── B. training curves, all runs overlaid ──
        def curves(key):
            out_c = []
            for lb, meta in zip(labels, metas):
                hist = meta["history"]
                if not hist or key not in hist[0]:
                    continue
                xs = np.arange(len(hist)) * (
                    meta["args"]["total_steps"] / max(len(hist) - 1, 1))
                out_c.append((xs, np.array([r[key] for r in hist]), lb,
                              colors[lb]))
            return out_c

        losses = {meta["args"].get("loss", "nll") for meta in metas}
        mse_note = ("Under --loss mse the trained objective is mean geodesic² + "
                    "mean ‖Δω‖²; the NLL and PL quantities are still computed\n"
                    "each step for comparability, but never enter the gradient — "
                    "read them as diagnostics, not as what was optimised."
                    if losses == {"mse"} else
                    "Runs use different --loss settings "
                    f"({', '.join(sorted(losses))}); NLL/PL enter the gradient "
                    "for some columns and not others.")
        pdf.savefig(fig_curves(curves("total"), "Training objective",
                               "loss", logy=True, note=mse_note))
        pdf.savefig(fig_curves(curves("mean_theta_sq"),
                               "Geodesic² per link (train batch)",
                               "geodesic² (rad²)", logy=True))
        pdf.savefig(fig_curves(curves("mean_omega_sq"),
                               "‖Δω‖² (train batch)", "rad²/s²", logy=True))
        # NLL and PL go negative (they include log-normalisers), so a log axis
        # would silently drop points rather than fail.
        pdf.savefig(fig_curves(curves("nll_R"), "Rotation NLL (diagnostic)",
                               "nats", logy=False, note=mse_note))
        pdf.savefig(fig_curves(curves("nll_omega"),
                               "Angular-rate NLL (diagnostic)", "nats",
                               logy=False, note=mse_note))
        pdf.savefig(fig_curves(curves("pl_loss"),
                               "Pseudo-likelihood NLL (diagnostic)", "nats",
                               logy=False, note=mse_note))
        plt.close("all")

        # ── C. ensemble dynamics ──
        pdf.savefig(fig_energy_ensemble(
            t_eval, gt_e,
            {lb: (md_e[lb].mean(0), md_e[lb].std(0), colors[lb])
             for lb in labels}, n_traj))
        pdf.savefig(fig_energy_single(
            t_eval, gt_e[0], {lb: (md_e[lb][0], colors[lb]) for lb in labels}))
        plt.close("all")

        det_gt = np.stack([so3_det_residual_abs(gt_all[i], n)
                           for i in range(n_traj)])
        orth_gt = np.stack([so3_orth_residual(gt_all[i], n)
                            for i in range(n_traj)])
        det_md = {lb: np.stack([so3_det_residual_abs(md_all[lb][i], n)
                                for i in range(n_traj)]) for lb in labels}
        orth_md = {lb: np.stack([so3_orth_residual(md_all[lb][i], n)
                                 for i in range(n_traj)]) for lb in labels}
        pdf.savefig(fig_so3_violation(
            t_eval, det_gt,
            {lb: (det_md[lb].mean(0), det_md[lb][0], colors[lb])
             for lb in labels},
            "|det(R) − 1|", "SO(3) constraint: determinant"))
        pdf.savefig(fig_so3_violation(
            t_eval, orth_gt,
            {lb: (orth_md[lb].mean(0), orth_md[lb][0], colors[lb])
             for lb in labels},
            "‖RᵀR − I‖_F", "SO(3) constraint: orthogonality"))
        plt.close("all")

        pdf.savefig(fig_error_band_multi(
            t_eval, {lb: (geo_err[lb], colors[lb]) for lb in labels},
            "Trajectory rotation error vs GT", "geodesic² (rad²)",
            saturation=sat))
        pdf.savefig(fig_error_band_multi(
            t_eval, {lb: (om_err[lb], colors[lb]) for lb in labels},
            "Trajectory angular-rate error vs GT", "‖Δω‖²/link (rad²/s²)"))
        plt.close("all")

        pdf.savefig(fig_state_ensemble(
            t_eval, gt_eul, gt_om,
            {lb: (md_eul[lb], md_om[lb], colors[lb]) for lb in labels},
            n, n_traj))
        pdf.savefig(fig_state_single(
            t_eval, gt_all[0],
            {lb: (md_all[lb][0], colors[lb]) for lb in labels}, n))
        pdf.savefig(fig_phase_portraits(
            gt_eul, gt_om,
            {lb: (md_eul[lb], md_om[lb], colors[lb]) for lb in labels}, n))
        plt.close("all")

        # ── D. subnet evolution ──
        d = 3 * n
        raw = _raw_gauge(next(iter(betas.values())))
        bstr = ("RAW — no gauge correction" if raw else
                ", ".join(f"β[{lb}]={betas[lb]:.3f}" for lb in labels))
        t_sub = t_eval
        pdf.savefig(plot_matrix_grid_multi(
            t_sub, {lb: (comps_raw[lb], colors[lb]) for lb in labels}, comps_gt,
            "M", f"Inverse mass M⁻¹(q) — raw, no gauge correction", d, d))
        plt.close("all")
        if not raw:
            pdf.savefig(plot_matrix_grid_multi(
                t_sub, {lb: (comps_md[lb], colors[lb]) for lb in labels},
                comps_gt, "M", "Inverse mass β·M⁻¹(q)", d, d, subtitle=bstr))
            plt.close("all")
        pdf.savefig(plot_matrix_grid_multi(
            t_sub, {lb: (comps_md[lb], colors[lb]) for lb in labels}, comps_gt,
            "D", "Dissipation D(q)/β", d, d, subtitle=bstr))
        plt.close("all")
        pdf.savefig(plot_matrix_grid_multi(
            t_sub, {lb: (comps_md[lb], colors[lb]) for lb in labels}, comps_gt,
            "B", "Input map g(q)/β", d, models[0].u_dim, subtitle=bstr))
        plt.close("all")
        pdf.savefig(plot_potential_multi(
            t_sub, {lb: (comps_md[lb], colors[lb]) for lb in labels}, comps_gt,
            "Potential V(q)/β", subtitle=bstr))
        plt.close("all")

        # ── E. physical sub-components, one page per structured run ──
        for lb in labels:
            if lb not in phys_by_model:
                continue
            pdf.savefig(fig_physical_constants(
                t_eval, phys_by_model[lb], params, n, lb, colors[lb]))
            plt.close("all")

    print(f"Done: {out}")


if __name__ == "__main__":
    main()

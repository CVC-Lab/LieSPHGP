r"""ELBO trainer for the $n$-link variational-GP port-Hamiltonian SDE.

Objective
---------
.. math::
    L \;=\; \underbrace{L_{\rm NLL}}_{\text{rollout}}
        \;+\;\lambda_{\rm PL}\underbrace{L_{\rm PL}}_{\text{per-increment}}
        \;+\;\frac{\beta}{N}\underbrace{L_{\rm KL}}_{\text{5 GP subnets}}

**$\beta$ defaults to 0**, so out of the box this is *not* the ELBO — it is pure
data-fit, $L_{\rm NLL}+\lambda_{\rm PL}L_{\rm PL}$, with the variational prior
switched off. Consequences worth knowing:

* With no KL, nothing opposes $\sigma_w\to0$ in $q_\psi(w)=\mathcal N(\mu,\sigma_w^2)$
  — sampled weights only ever hurt the fit on average — so the GPs collapse
  toward deterministic maximum-likelihood feature regressors. The *dynamics*
  stay stochastic: $\Sigma_\theta(q)$ is a structural part of the SDE, not a
  weight posterior, and is held up by $L_{\rm PL}$.
* $L_{\rm KL}$ is still computed and printed each eval, just weighted by zero,
  so the term remains observable.

Set `--beta_max > 0` to recover the true ELBO.

* $L_{\rm NLL}$ — concentrated-Gaussian likelihood on $SO(3)^n$ plus an
  isotropic Gaussian on the rates, over a Stratonovich Lie–Heun rollout.
* $L_{\rm PL}$ — Euler–Maruyama transition density at consecutive observed
  snapshots. **Essential**: the rollout NLL alone drives $\Sigma_\theta\to0$,
  because model and environment have independent Brownian paths, so enlarging
  the diffusion only adds variance to the residual. $L_{\rm PL}$ is what gives
  the diffusion a non-collapsing data-fit signal.
* $\beta$ is held at $0$ for `--kl_warmup_steps`, then annealed linearly
  $0\to\beta_{\max}$ over `--kl_anneal_steps`, so the likelihood shapes the
  posterior before the prior starts pulling on it. $\beta=1$ is **not** "the
  true ELBO" — the NLL is a per-snapshot *mean* while the KL is per-window, so
  the units do not match and $\beta$ is an empirical dial (see `gp_change.md`,
  which also documents the bounded `gp_core` residuals in
  `structured_subnets.py`).

GP weight samples are drawn **once per (batch, MC-sample) trajectory** and held
fixed across the predictor, the corrector and every substep — otherwise the
Monte-Carlo estimate is not a draw from $q_\psi(w)$.

Default configuration is the target experiment: $n=2$, `obs_noise_std=0.1`,
fixed friction $0.5$, random torques, stochastic wind $\sigma=0.5$.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# x64 stays ON so the *environment* (which the dataset generator runs) keeps its
# float64 physics — that is what makes the ground-truth reference trustworthy.
# The **model** precision is separate and set explicitly by `--float64`
# (default: float32). Leaving model precision implicit is what caused the
# original `jax.jvp` primal/tangent dtype mismatch in `network.drift`: float64
# GP weights met a float32 batch. Both the model and the batch are now pinned to
# the same `DTYPE`, so the two never disagree regardless of the global flag.
jax.config.update('jax_enable_x64', True)

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..', '..', '..', '..'))
for _p in (PKG_ROOT, PROJECT_ROOT, THIS_FILE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.arm_nlink_so3.datagen.windy_arm_nlink_datagen import get_dataset            # noqa: E402
from envs.arm_nlink_so3 import arm_nlink_physics as phys            # noqa: E402

from network import DissipativeArmHamSDE, KeyedArmModel             # noqa: E402
from utils.lie_integrator_nlink import lie_heun_sde_rollout_nlink   # noqa: E402
from utils import structured_subnets                                # noqa: E402
from utils.elbo_loss_nlink import (                                 # noqa: E402
    elbo_nll_nlink, pl_loss_nlink, kl_per_subnet, geodesic_distance,
)

DEFAULT_SAVE_DIR = os.path.join(THIS_FILE_DIR, 'data')
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'windy_arm_nlink')

SUBNETS = ('M', 'V', 'Dw', 'g', 'Sigma')


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description='Train ph_gp_sde on the n-link arm.')
    # ── system / dataset (defaults = the target experiment) ──
    p.add_argument('--n', type=int, default=2)
    p.add_argument('--obs_noise_std', type=float, default=0.1)
    p.add_argument('--friction_coeff', type=float, default=0.5)
    p.add_argument('--varying_friction', action='store_true',
                   help='off by default => fixed friction')
    p.add_argument('--wind_force_std', type=float, default=0.5,
                   help='sigma of the stochastic wind (the dW channel)')
    p.add_argument('--external_force_std', type=float, default=0.0)
    p.add_argument('--air_drag', type=float, default=0.0)
    p.add_argument('--random_u', action='store_true', default=True)
    p.add_argument('--random_u_scale', type=float, default=1.0)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--timesteps', type=int, default=20)
    p.add_argument('--num_points', type=int, default=5,
                   help='rollout window length (short: the arm is chaotic)')

    # ── model ──
    p.add_argument('--hidden_dim', type=int, default=32,
                   help='number of Matern random features per GP')
    p.add_argument('--relative_inputs', action='store_true',
                   help='INERT since M/D/g became physics-structured: those '
                        'closed forms already contain R_j^T R_k exactly, so '
                        'there is no approximate invariance prior left to '
                        'switch on. Accepted so old commands still run.')
    p.add_argument('--g_diag', type=float, nargs=3, default=(1.0, 1.0, 1.0),
                   metavar=('GX', 'GY', 'GZ'),
                   help='per-axis actuator gain, mirroring the single '
                        'pendulum. g(q) = T(q)^T blkdiag(diag(gamma_i)). At '
                        'the default (1,1,1) the input map is pure kinematics '
                        'and there is nothing to identify.')
    p.add_argument('--no_friction', action='store_true')
    p.add_argument('--gp_core', action='store_true',
                   help='predict the physical constants of M/D/g/Sigma with the '
                        'Matern x periodic variational GP instead of learning '
                        'them as constants: GP(q) -> [m, I, ell, c] -> closed '
                        'form -> M(q). A grey-box -- the formula supplies the '
                        'structure, the GP absorbs what rigid-body physics '
                        'cannot. Costs exactness (the GP must relearn that the '
                        'constants ARE constant) and ~8k params per subnet, but '
                        'can represent q-dependent effects such as the height '
                        'term in --varying_friction, which constants provably '
                        'cannot. Implies --relative_inputs for M/D/g so that '
                        'M(hq)=M(q) survives. Supersedes --variational_core '
                        '(the GP is variational already).')
    p.add_argument('--variational_core', action='store_true',
                   help='give the physical constants of M/D/g/Sigma a mean-field '
                        'Gaussian posterior N(mu, sigma^2) instead of point '
                        'estimates, sampled by reparameterisation exactly like '
                        'the GP weights. Bayesian system identification rather '
                        'than max-likelihood. NOTE: the KL is weighted by '
                        '--beta_max, which defaults to 0 -- at that default '
                        'this only injects sampling noise, so set --beta_max>0 '
                        'for it to regularise anything.')
    p.add_argument('--core_prior_std', type=float, default=1.0,
                   help='std of the N(0, s^2) prior on the physical constants '
                        '(only used with --variational_core). Weakly '
                        'informative: the truths here (m=1, ell=c=e_z, d=0.5, '
                        'gamma~0.5) all sit within one std of zero. Raise it if '
                        'the true scale is unknown, since the prior does shrink '
                        'toward zero.')
    p.add_argument('--sigma_rollout_grad', action='store_true',
                   help='let the rollout NLL back-propagate into Sigma (the '
                        'pre-Change-5 behaviour). By default that path is '
                        'stop_gradient-ed: it carries ONLY collapse pressure '
                        '(model-side diffusion can only add variance to the '
                        'rollout residual against an independently-noisy '
                        'target), and with the gauge anchored it crushed '
                        '||Sigma|| to ~30% of truth and zeroed the link-2 '
                        'channel. Detached, the PL term solely owns the '
                        'diffusion. See gp_change.md Change 5.')
    p.add_argument('--anchor_trace', type=float, default=None,
                   help='FUNCTION-SPACE gauge anchor: normalise the learned '
                        'mass matrix so tr M(q) equals this value at every q. '
                        'Default: 4*n. Pass 0 to disable. This supersedes '
                        '--anchor_m1, which proved insufficient: M is '
                        'quadratic in the levers, so the gauge escaped via '
                        'u -> sqrt(beta)*u, I -> beta*I with the masses '
                        'fixed (measured beta=23.6 with m_1 pinned). Pinning '
                        'tr M closes every parameter route at once; any '
                        'positive value is gauge-equivalent (a units choice, '
                        'zero information about the data).')
    p.add_argument('--anchor_m1', type=float, default=1.0,
                   help='gauge anchor: freeze link-1 mass m_1 at this value '
                        '(base, posterior draw and gp_core residual slice all '
                        'overwritten, so no path can re-open the gauge). The '
                        'scale direction (M,V,D,g,Sigma) -> beta*(...) is '
                        'exactly unidentifiable, so this carries no '
                        'information about the data -- it fixes the units, '
                        'stops the NLL-driven gauge ratchet (an unanchored '
                        'run reached beta=21) and gives Sigma a fixed target. '
                        'Any positive value is gauge-equivalent (use e.g. 2.7 '
                        'as a control that the anchor leaks nothing); <=0 '
                        'disables the anchor.')
    p.add_argument('--sigma_mode', choices=('full', 'structured'),
                   default='full',
                   help="diffusion parameterisation. 'full' = free 3n x 3 "
                        "learned map. 'structured' = Sigma_j = [v_j]_x R_j^T, "
                        "the exact physics form: 3n parameters instead of a "
                        "learned function.")

    # ── optimisation ──
    p.add_argument('--learn_rate', type=float, default=1e-3)
    p.add_argument('--lr_sigma', type=float, default=None,
                   help='separate learning rate for Sigma_net. Unset => same '
                        'as --learn_rate, reproducing the single-optimiser '
                        'behaviour exactly. Sigma_net holds 6 numbers out of '
                        '~1433 (structured mode) and converges far more slowly '
                        'than the drift, so it often wants a larger step.')
    p.add_argument('--total_steps', type=int, default=2000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--mc_samples', type=int, default=1)
    p.add_argument('--beta_max', type=float, default=0.0,
                   help='KL weight. Default 0 => the variational prior is OFF '
                        'and the objective is pure data-fit (NLL + PL). '
                        'NOTE: 1.0 is NOT "the true ELBO" -- the NLL is a '
                        'per-snapshot mean while the KL is scaled per-window, '
                        'so the units do not match and beta is an empirical '
                        'dial. 0.05-0.1 is the sane range with --gp_core: '
                        'large beta widens the ~15k weight posteriors, which '
                        'injects sampling noise into the physical constants '
                        '(see gp_change.md).')
    p.add_argument('--kl_anneal_steps', type=int, default=3000,
                   help='linear 0->beta_max ramp, starting after '
                        '--kl_warmup_steps; irrelevant when beta_max=0. '
                        '3000-5000 recommended with --gp_core.')
    p.add_argument('--kl_warmup_steps', type=int, default=2000,
                   help='hold beta = 0 for this many steps before the ramp, so '
                        'the likelihood localises the weight posteriors before '
                        'the prior starts widening them. beta(t) = beta_max * '
                        'clip((t - warmup)/anneal, 0, 1).')
    p.add_argument('--lambda_pl', type=float, default=1.0)
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--eval_every', type=int, default=50)

    # ── misc ──
    p.add_argument('--float64', action='store_true',
                   help='train the model in float64 (default float32). The '
                        'environment/dataset is always float64; this flag only '
                        'changes model + batch precision.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR)
    p.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    p.add_argument('--name', type=str, default='arm')
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════

def arrange_data(x, t, num_points=2):
    """Slice `(num_us, T, N, D)` trajectories into overlapping windows.

    Returns `(num_us, num_points, N_windows, D)` and the window time grid.
    Short windows are what make training on a chaotic system tractable: a long
    rollout's gradient is dominated by exponential trajectory separation rather
    than by model error.
    """
    assert 2 <= num_points <= len(t)
    stack = [x[:, i:x.shape[1] - num_points + i + 1, :, :] if i < num_points - 1
             else x[:, i:, :, :] for i in range(num_points)]
    stack = np.stack(stack, axis=1)
    stack = np.reshape(stack, (x.shape[0], num_points, -1, x.shape[3]))
    return stack, t[:num_points]


def model_dtype(args):
    """Model + batch precision. The dataset itself is always float64."""
    return jnp.float64 if args.float64 else jnp.float32


def load_data(args):
    """Build/load the dataset and reshape to `(T_obs, B_all, 15n)`."""
    data, path = get_dataset(
        n=args.n, seed=args.seed, samples=args.samples,
        timesteps=args.timesteps, test_split=0.5, save_dir=args.data_dir,
        obs_noise_std=args.obs_noise_std, friction_coeff=args.friction_coeff,
        varying_friction=args.varying_friction, air_drag=args.air_drag,
        external_force_std=args.external_force_std,
        wind_force_std=args.wind_force_std, random_u=args.random_u,
        random_u_scale=args.random_u_scale, g_diag=tuple(args.g_diag),
        us=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)),
    )
    print(f'dataset: {path}')

    def to_windows(arr):
        w, _ = arrange_data(arr, data['t'], num_points=args.num_points)
        # (num_us, P, N_win, D) -> (P, num_us*N_win, D)
        w = np.transpose(w, (1, 0, 2, 3))
        return jnp.asarray(w.reshape(w.shape[0], -1, w.shape[-1]),
                           dtype=model_dtype(args))

    train = to_windows(data['x'])
    test = to_windows(data['test_x_noisy'])
    dt = float(data['t'][1] - data['t'][0])
    return train, test, dt, data


# ═════════════════════════════════════════════════════════════════════
# Rollout
# ═════════════════════════════════════════════════════════════════════

def sample_dW(key, B, S, n_outer, n_substeps, h, dtype):
    r"""`(B, S, n_outer, n_substeps, 3)` pre-scaled so $\mathrm{Var}(dW)=h$.

    The trailing axis is **3**, not $3n$: one shared world wind field.
    """
    return jax.random.normal(key, (B, S, n_outer, n_substeps, 3),
                             dtype=dtype) * jnp.sqrt(jnp.asarray(h, dtype=dtype))


def sample_gp_keys(key, B, S):
    """Dict `subnet -> (B, S, 2)` PRNGKeys: one $w$ sample per trajectory."""
    tops = jax.random.split(key, len(SUBNETS))
    return {name: jax.random.split(k, B * S).reshape(B, S, 2)
            for name, k in zip(SUBNETS, tops)}


def rollout_single(model, x_traj, h, dW_one, keys_one, n, n_sub, inference_mode):
    r"""One trajectory. `x_traj : (T_obs, 15n)` $\to$ `(T_obs, 15n)`.

    The applied torque is read **per outer step** from the data, not held at
    $u(0)$ — otherwise a dataset with random per-step torques never excites
    $g_\theta$ and the input map is unidentifiable.

    .. important::
       The slice is ``x_traj[1:]``, not ``x_traj[:-1]``. The generator appends
       ``(obs, curr_u)`` *before* advancing the torque, so **row $k$ stores the
       torque that produced $\mathrm{obs}_k$** — i.e. the one applied on the
       transition $k{-}1\to k$. The torque driving $t\to t{+}1$ therefore lives
       at row $t{+}1$.

       Using ``[:-1]`` duplicates $u_0$ and lags the rest by one step. That is
       invisible when ``random_u=False`` (a shifted constant is the same
       constant) but with per-step random torques it feeds the rollout a
       sequence essentially uncorrelated with the one that generated the data:
       ``g_net`` then trains against the wrong input, and the resulting
       irreducible residual pushes the learned $\sigma_\omega$ far above the
       true observation noise. Verified by replaying data through the analytic
       physics: ``[1:]`` reproduces it to 6e-08, ``[:-1]`` to only 2e-02.
    """
    del n_sub
    x0 = x_traj[0, :12 * n]
    u_per_outer = x_traj[1:, 12 * n:15 * n]
    keyed = KeyedArmModel(model=model, keys=keys_one, inference_mode=inference_mode)
    traj = lie_heun_sde_rollout_nlink(keyed, x0, u_per_outer, h, dW_one)
    u_full = jnp.concatenate([u_per_outer, u_per_outer[-1:]], axis=0)
    return jnp.concatenate([traj, u_full], axis=-1)


def rollout_batch(model, batch, h, dW, keys, n, n_sub, inference_mode=False):
    """Vmap over batch and MC sample. `(T,B,15n)` -> `(T,B,S,15n)`."""
    x_BT = jnp.transpose(batch, (1, 0, 2))

    def per_b(x_b, dW_b, k_b):
        def per_s(dW_s, k_s):
            return rollout_single(model, x_b, h, dW_s, k_s, n, n_sub, inference_mode)
        return jax.vmap(per_s)(dW_b, k_b)

    out = jax.vmap(per_b)(x_BT, dW, keys)              # (B, S, T, 15n)
    return jnp.transpose(out, (2, 0, 1, 3))


# ═════════════════════════════════════════════════════════════════════
# Loss
# ═════════════════════════════════════════════════════════════════════

def loss_fn(model, batch, h, dW, keys, beta, N, lambda_pl, dt_outer,
            n, n_sub, inference_mode=False):
    """Negative ELBO plus the weighted pseudo-likelihood. Returns `(loss, aux)`."""
    traj = rollout_batch(model, batch, h, dW, keys, n, n_sub, inference_mode)
    target = jnp.broadcast_to(batch[:, :, None, :], traj.shape)

    # Drop t=0: the initial condition is supplied, not predicted.
    nll = elbo_nll_nlink(target[1:], traj[1:],
                         model.log_sigma_R, model.log_sigma_omega, n)
    kl = kl_per_subnet(model)
    pl = pl_loss_nlink(model, batch, dt_outer, model.sigma_obs_omega, keys, n,
                       inference_mode=inference_mode)

    total = (nll['nll_total'] + (beta / N) * kl['total_kl']
             + lambda_pl * pl['pl_loss'])

    aux = dict(nll)
    aux.update({f'kl_{k}': v for k, v in kl.items()})
    aux.update({'pl_loss': pl['pl_loss'],
                'pl_residual_sq': pl['mean_residual_sq'],
                'sigma_fro': pl['mean_sigma_fro'],
                'beta': beta, 'total': total})
    return total, aux


# ═════════════════════════════════════════════════════════════════════
# Optimiser
# ═════════════════════════════════════════════════════════════════════

def make_optimizer(model, args):
    r"""Adam(W) over everything, optionally with a separate rate for `Sigma_net`.

    Why a second rate is worth having: with the physics-structured subnets the
    parameter counts are wildly unequal — at $n=2$, `Sigma_net` holds **6**
    numbers out of ~1 433 — and one global step size has to serve both. In
    practice the drift plateaus by ~step 4 000 while $\lVert\Sigma\rVert_F$ is
    still climbing monotonically at step 10 000, i.e. the two are not converging
    on the same timescale.

    `--lr_sigma` unset (or equal to `--learn_rate`) reproduces the single-leg
    optimiser **exactly**, so this cannot perturb existing results.

    Implementation note: `optax.multi_transform` needs a label pytree matching
    the *filtered* parameter tree, so the labels are built from
    `eqx.filter(model, eqx.is_array)` — filtering replaces every non-array leaf
    with `None`, which JAX treats as an empty subtree and skips.
    """
    def leg(lr):
        return optax.chain(optax.clip_by_global_norm(args.grad_clip),
                           optax.adamw(lr))

    lr_sigma = args.learn_rate if args.lr_sigma is None else float(args.lr_sigma)
    if lr_sigma == args.learn_rate or getattr(model, 'Sigma_net', None) is None:
        return leg(args.learn_rate)

    arrays = eqx.filter(model, eqx.is_array)
    labels = jax.tree_util.tree_map(lambda _: 'base', arrays)
    labels = eqx.tree_at(
        lambda t: t.Sigma_net,
        labels,
        replace=jax.tree_util.tree_map(
            lambda _: 'sigma', eqx.filter(model.Sigma_net, eqx.is_array)),
    )
    return optax.multi_transform(
        {'base': leg(args.learn_rate), 'sigma': leg(lr_sigma)}, labels)


# ═════════════════════════════════════════════════════════════════════
# Diagnostics: learned subnets vs analytic ground truth
# ═════════════════════════════════════════════════════════════════════

def subnet_errors(model, q_samples, params, n, sigma_true):
    r"""Relative error of each learned subnet against the analytic physics,
    **after removing the two unobservable gauges**.

    1. *Scale.* Momenta are never measured, so the dynamics are invariant under
       $(M,V,D,g,\Sigma)\mapsto\beta\,(M,V,D,g,\Sigma)$ with $p\mapsto\beta p$:
       $H_\beta=\beta H$ but $\partial H_\beta/\partial p_\beta=\partial H/\partial p$,
       so $\omega$ — all the data sees — is unchanged. $\beta$ is fitted from
       $M^{-1}$ by least squares, then $M^{-1}\!\to\!\beta M^{-1}$ and
       $V,D,\Sigma\to\cdot/\beta$ (with $\Sigma\Sigma^\top$ taking $\beta^2$).
    2. *Offset.* $H\mapsto H+c$ is also free, so $V$ is centred before comparison.

    Skipping either makes the numbers measure the gauge rather than the physics —
    raw errors can exceed 1.0 for a model that is exactly right up to scale.

    Returns `([M, V, D, g, ΣΣᵀ], beta)` — relative Frobenius errors,
    gauge-corrected.

    .. note::
       $g$ was absent from this list until now, which made every claim about the
       input map unfalsifiable. It matters: at ``random_u_scale=1`` the measured
       $g$ error was 1.035 — *worse than predicting zero* — while $M$, $V$ and
       $D$ all looked reasonable, so the failure was completely invisible.
    """
    def batch(fn):
        return jax.vmap(fn)(q_samples)

    R_all = q_samples.reshape(-1, n, 3, 3)
    M_gt = jax.vmap(lambda R: jnp.linalg.inv(phys.mass_matrix(params, R)))(R_all)
    V_gt = jax.vmap(lambda R: phys.potential(params, R))(R_all)
    D_gt = jax.vmap(lambda R: phys.dissipation_matrix(
        params, R, jnp.zeros((n, 3))))(R_all)
    g_gt = jax.vmap(lambda R: phys.input_map(params, R))(R_all)   # T(q)^T Gamma
    S_gt = sigma_true * jax.vmap(lambda R: phys.wind_map(params, R))(R_all)

    M_hat = batch(model.M_inv)
    V_hat = batch(lambda q: model._call(model.V_net, q, None)[0])
    # Structured subnets take the FULL q: they need T(q), and any
    # invariant projection their internal GP wants is applied inside.
    # Passing _invariant_q(q) here silently worked only while
    # relative_inputs was False (it is the identity then) and crashes
    # the moment it is on.
    D_hat = batch(lambda q: model._call(model.Dw_net, q, None))
    g_hat = batch(lambda q: model._call(model.g_net, q, None))
    S_hat = batch(model.Sigma)

    # β from M⁻¹: argmin_β ‖β·M̂⁻¹ − M_gt⁻¹‖²
    beta = jnp.sum(M_hat * M_gt) / (jnp.sum(M_hat * M_hat) + 1e-30)

    Vc_hat, Vc_gt = V_hat - jnp.mean(V_hat), V_gt - jnp.mean(V_gt)
    P_hat = jnp.einsum('kab,kcb->kac', S_hat, S_hat)
    P_gt = jnp.einsum('kab,kcb->kac', S_gt, S_gt)

    def rel(A, B):
        return jnp.linalg.norm(A - B) / (jnp.linalg.norm(B) + 1e-30)

    return jnp.array([
        rel(beta * M_hat, M_gt),
        rel(Vc_hat / beta, Vc_gt),
        rel(D_hat / beta, D_gt),
        rel(g_hat / beta, g_gt),
        rel(P_hat / beta ** 2, P_gt),
    ]), beta


# ═════════════════════════════════════════════════════════════════════
# Train
# ═════════════════════════════════════════════════════════════════════

def train(args):
    key = jax.random.PRNGKey(args.seed)
    n = args.n
    n_sub = 10                       # must match the environment's n_substeps
    # Resolve the trace anchor's n-dependent default HERE so history.pkl
    # records the actual number (load_run rebuilds the skeleton from it).
    if args.anchor_trace is None:
        args.anchor_trace = 4.0 * n

    train_x, test_x, dt_outer, raw = load_data(args)
    h = dt_outer / n_sub
    n_outer = args.num_points - 1
    N_train = train_x.shape[1]
    print(f'train windows: {train_x.shape}   test: {test_x.shape}   '
          f'dt={dt_outer:.3f}  h={h:.4f}')

    dtype = model_dtype(args)
    key, k_model = jax.random.split(key)
    model = DissipativeArmHamSDE(
        key=k_model, n=n, hidden_dim=args.hidden_dim,
        friction=not args.no_friction,
        relative_inputs=args.relative_inputs,
        init_sigma_obs_omega=args.obs_noise_std,
        sigma_mode=args.sigma_mode,
        sigma_detach_rollout=not args.sigma_rollout_grad,
        variational_core=args.variational_core,
        core_prior_std=args.core_prior_std,
        gp_core=args.gp_core,
        anchor_m1=args.anchor_m1,
        anchor_trace=args.anchor_trace,
        dtype=dtype,
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_array)))
    print(f'model: n={n}  u_dim={model.u_dim}  relative_inputs='
          f'{model.relative_inputs}  sigma_mode={model.sigma_mode}  '
          f'anchor_m1={model.M_net.anchor_m1}  '
          f'anchor_trace={model.M_net.anchor_trace}  '
          f'sigma_detach={model.sigma_detach_rollout}  '
          f'dtype={jnp.dtype(dtype).name}  '
          f'trainable params={n_params}')

    optim = make_optimizer(model, args)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Ground-truth params for the diagnostic (never touches the loss).
    gt_params = phys.uniform_chain_params(
        n, d=args.friction_coeff, kappa=args.air_drag,
        varying_friction=args.varying_friction, g_diag=tuple(args.g_diag))

    @eqx.filter_jit
    def train_step(model, opt_state, batch, dW, keys, beta):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, batch, h, dW, keys, beta, N_train, args.lambda_pl,
            dt_outer, n, n_sub)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss, aux

    @eqx.filter_jit
    def eval_step(model, batch, dW, keys):
        # Posterior-mean path: the reported numbers are deterministic.
        _, aux = loss_fn(model, batch, h, dW, keys, jnp.asarray(0.0), N_train,
                         args.lambda_pl, dt_outer, n, n_sub, inference_mode=True)
        return aux

    os.makedirs(args.save_dir, exist_ok=True)
    run = (f'{args.name}{n}link_obs{args.obs_noise_std:g}_fric{args.friction_coeff:g}'
           f'_wind{args.wind_force_std:g}_s{args.total_steps}'
           f'{"_rel" if args.relative_inputs else ""}_{time.strftime("%y%m%d-%H%M%S")}')
    run_dir = os.path.join(args.save_dir, run)
    os.makedirs(run_dir, exist_ok=True)
    print(f'run dir: {run_dir}\n')

    hdr = (f'{"step":>6} {"loss":>11} {"nll_R":>10} {"nll_w":>10} {"pl":>10} '
           f'{"KL":>11} {"beta":>6} {"geo2":>9} {"w2":>9} {"sig_R":>7} '
           f'{"sig_w":>7} {"|S|":>7}')
    print(hdr)
    print('-' * len(hdr))

    history = []
    for step in range(args.total_steps + 1):
        key, k_b, k_dw, k_gp = jax.random.split(key, 4)
        idx = jax.random.choice(k_b, N_train,
                                (min(args.batch_size, N_train),), replace=False)
        batch = train_x[:, idx, :]
        B = batch.shape[1]

        dW = sample_dW(k_dw, B, args.mc_samples, n_outer, n_sub, h, batch.dtype)
        keys = sample_gp_keys(k_gp, B, args.mc_samples)
        # beta(t) = beta_max * clip((t - warmup)/anneal, 0, 1): held at 0 while
        # the likelihood localises the posteriors, then ramped linearly. Both
        # NaN crashes started ~500-1500 steps after beta saturated at 1.0.
        beta = jnp.asarray(
            args.beta_max * min(1.0, max(0, step - args.kl_warmup_steps)
                                / max(args.kl_anneal_steps, 1)))

        model, opt_state, loss, aux = train_step(model, opt_state, batch, dW, keys, beta)

        if step % args.eval_every == 0:
            print(f'{step:>6} {float(loss):>11.3f} {float(aux["nll_R"]):>10.3f} '
                  f'{float(aux["nll_omega"]):>10.3f} {float(aux["pl_loss"]):>10.3f} '
                  f'{float(aux["kl_total_kl"]):>11.1f} {float(beta):>6.3f} '
                  f'{float(aux["mean_theta_sq"]):>9.4f} '
                  f'{float(aux["mean_omega_sq"]):>9.4f} '
                  f'{float(aux["sigma_R"]):>7.4f} {float(aux["sigma_omega"]):>7.4f} '
                  f'{float(aux["sigma_fro"]):>7.3f}')
            history.append({k: float(v) for k, v in aux.items()
                            if jnp.ndim(v) == 0})

    # ── final evaluation ──
    key, k_dw, k_gp = jax.random.split(key, 3)
    B_te = test_x.shape[1]
    dW_te = sample_dW(k_dw, B_te, 1, n_outer, n_sub, h, test_x.dtype)
    keys_te = sample_gp_keys(k_gp, B_te, 1)
    te = eval_step(model, test_x, dW_te, keys_te)

    q_samp = train_x[0, :64, :9 * n].astype(dtype)
    err, beta = subnet_errors(model, q_samp, gt_params, n, args.wind_force_std)

    print('\n' + '=' * 62)
    print('final (test set, posterior mean)')
    print('=' * 62)
    print(f'  geodesic^2 per link : {float(te["mean_theta_sq"]):.5f}  '
          f'(rms angle {np.sqrt(float(te["mean_theta_sq"])):.4f} rad)')
    print(f'  ||dw||^2            : {float(te["mean_omega_sq"]):.5f}')
    print(f'  pl_loss             : {float(te["pl_loss"]):.4f}')
    print(f'  sigma_R / sigma_w   : {float(te["sigma_R"]):.4f} / '
          f'{float(te["sigma_omega"]):.4f}')
    print(f'\nsubnet relative error vs analytic ground truth  '
          f'(gauge-corrected, beta={float(beta):.4f})')
    print(f'  M^-1 : {float(err[0]):.4f}')
    print(f'  V    : {float(err[1]):.4f}   (centred: H is defined up to a constant)')
    print(f'  D    : {float(err[2]):.4f}')
    print(f'  g    : {float(err[3]):.4f}   (needs excitation: raise --random_u_scale)')
    print(f'  SS^T : {float(err[4]):.4f}   (Sigma also free up to Sigma->Sigma O)')
    # One shared wind field must torque every link the SAME way: adjacent
    # lever vectors should be nearly parallel (cos ~ +1). A global sign flip
    # is gauge; a NEGATIVE adjacent cosine is a real error (anti-correlated
    # wind response) that inflates AA^T error past 1 and wrecks calibration.
    if hasattr(model.Sigma_net, 'v') and n >= 2:
        v_lev = np.asarray(model.Sigma_net.v)
        v_nrm = np.linalg.norm(v_lev, axis=1)
        # The cosine is meaningless when a lever has collapsed (direction of
        # a ~zero vector is noise) — that is a *magnitude* failure, reported
        # by the norms themselves, not an alignment failure.
        cos_adj = ['n/a' if min(v_nrm[i], v_nrm[i + 1]) < 1e-2 else
                   '%+.3f' % (v_lev[i] @ v_lev[i + 1]
                              / (v_nrm[i] * v_nrm[i + 1]))
                   for i in range(n - 1)]
        print(f'  wind levers ||v_i|| : {["%.3f" % x for x in v_nrm]}   '
              f'adjacent cos: {cos_adj}   (+1 expected; negative = relative '
              f'sign flip; n/a = lever collapsed, direction undefined)')
    print('=' * 62)

    eqx.tree_serialise_leaves(os.path.join(run_dir, 'model.eqx'), model)
    with open(os.path.join(run_dir, 'history.pkl'), 'wb') as f:
        # Architecture markers that are *static* eqx fields never reach
        # model.eqx, so a skeleton built differently would silently deserialise
        # into another model. Record them; `load_run` refuses runs whose marker
        # does not match the code.
        pickle.dump({'history': history,
                     'args': dict(vars(args), structured_subnets=True,
                                  i_epsilon=structured_subnets.I_EPSILON,
                                  residual_kappa=structured_subnets.RESIDUAL_KAPPA,
                                  sigma_detach_rollout=not args.sigma_rollout_grad),
                     'final_test': {k: float(v) for k, v in te.items()
                                    if jnp.ndim(v) == 0},
                     'subnet_err': np.asarray(err).tolist(),
                     'beta': float(beta)}, f)
    print(f'\nsaved to {run_dir}')
    return model


if __name__ == '__main__':
    train(get_args())

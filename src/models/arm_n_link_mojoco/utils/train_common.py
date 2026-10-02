r"""Shared trainer for the $n$-link port-Hamiltonian variants.

One trainer drives ``ph_gp_ode``, ``ph_nn_ode`` and ``ph_nn_sde``; each of those
packages is a thin CLI wrapper that selects `subnet_kind` and `stochastic`.

Objective
---------
.. math::
    L \;=\; \underbrace{L_{\rm NLL}}_{\text{rollout}}
        \;+\;\lambda_{\rm PL}\underbrace{L_{\rm PL}}_{\text{per-increment}}
        \;+\;\frac{\beta}{N}\underbrace{L_{\rm KL}}_{\text{GP variants only}}

* $L_{\rm NLL}$ — concentrated Gaussian on $SO(3)^n$ + isotropic Gaussian on the
  rates, over a Lie–Heun rollout (Stratonovich for SDE, deterministic for ODE).
* $L_{\rm PL}$ — Euler–Maruyama transition density at consecutive snapshots.
  For SDE variants this is what stops $\Sigma_\theta\to0$; for ODE variants
  $\Sigma\equiv0$ makes it a fixed-variance Gaussian on $\Delta\omega$, i.e. a
  strong single-step drift signal. Same code either way.
* $L_{\rm KL}$ — zero for the NN variants by construction (their modules return
  zero), so ``--beta_max`` simply has no effect there.

.. warning::
   ``ph_gp_sde`` has its own copy of this trainer and does **not** import from
   here — it was validated and produced results before this module existed, and
   re-pointing it mid-experiment risks silently changing those numbers. Any fix
   made here must be mirrored into ``ph_gp_sde/train.py`` (and vice versa) until
   the two are deliberately merged.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, '..'))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..'))
# MuJoCo build: the dataset and the analytic reference physics both come from
# inside this package, not from the project-level datasets/ and envs/ trees.
# That is the ONLY structural difference from arm_n_link/utils/train_common.py.
_MJENV = os.path.abspath(os.path.join(_PKG, 'mujoco_env'))
_DATAGEN = os.path.abspath(os.path.join(_PKG, 'data_gen'))
for _p in (_HERE, _PKG, _ROOT, _MJENV, _DATAGEN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# x64 stays ON so the environment/dataset keep float64 physics; the *model*
# precision is separate and chosen by --float64 (default float32). See the note
# in ph_gp_sde/train.py: leaving model precision implicit caused a jax.jvp
# primal/tangent dtype mismatch when float64 GP weights met a float32 batch.
jax.config.update('jax_enable_x64', True)

from mujoco_arm_datagen import get_dataset                           # noqa: E402
import arm_nlink_physics as phys                                     # noqa: E402

import structured_subnets                                            # noqa: E402
from ph_network_nlink import ArmPortHamiltonian, KeyedArmModel, SUBNET_NAMES  # noqa: E402
from lie_integrator_nlink import (                                   # noqa: E402
    lie_heun_sde_rollout_nlink, lie_heun_ode_rollout_nlink,
)
from elbo_loss_nlink import elbo_nll_nlink, pl_loss_nlink, kl_per_subnet  # noqa: E402

DEFAULT_DATA_DIR = os.path.join(_PKG, 'data', 'mujoco_arm_nlink')
r"""Lie–Heun substeps per outer step. **Its meaning changes with MuJoCo data.**

Against the analytic env this had to *match the generator* — the data came from
the very same Lie–Heun scheme at the same $h$, so a perfect model could
reproduce the data exactly, discretisation error and all.

MuJoCo integrates with RK4 at $10^{-3}$ instead, so there is nothing to match:
both schemes merely approximate the same continuous flow, and their difference
is our model's irreducible error floor. From the convergence study in
``mujoco_env/test_mujoco_matches_gt.py`` (2nd order, $\varepsilon\propto h^2$):

======  ========  ==================  ======================
 subs    h         $|\Delta\omega|$    floor on $\|d\omega\|^2$
======  ========  ==================  ======================
 10      0.0050    9.3e-4              ~9e-7
 20      0.0025    2.2e-4              ~5e-8
 40      0.0013    5.3e-5              ~3e-9
======  ========  ==================  ======================

For scale, the structured model reaches $\|d\omega\|^2\approx2\times10^{-5}$ on
analytic data. At 10 substeps the floor sits within an order of magnitude of
that, so the model would be limited by *our integrator* rather than by its own
capacity — and "structured did worse on MuJoCo" would be an artefact. 20
substeps puts the floor ~400x below the signal for 2x the compute, which is why
it is the default here. Override with ``--n_substeps``.
"""
N_SUBSTEPS = 20


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def get_args(model_name: str, default_save_dir: str, **default_overrides):
    """Build the shared CLI. `default_overrides` lets a variant wrapper change
    flag *defaults* (e.g. ph_nn_sde defaults to the pendulum-parity loss);
    anything the user passes on the command line still wins."""
    p = argparse.ArgumentParser(description=f'Train {model_name} on the n-link arm.')
    p.add_argument('--n', type=int, default=2)
    p.add_argument('--obs_noise_std', type=float, default=0.1)
    p.add_argument('--friction_coeff', type=float, default=0.5)
    p.add_argument('--varying_friction', action='store_true')
    p.add_argument('--wind_force_std', type=float, default=0.5)
    p.add_argument('--external_force_std', type=float, default=0.0)
    p.add_argument('--air_drag', type=float, default=0.0)
    p.add_argument('--random_u', action='store_true', default=True)
    # Defaults aligned with the ph_gp_sde experiment so all four variants
    # train on the SAME cached dataset and are directly comparable.
    p.add_argument('--random_u_scale', type=float, default=2.0)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--timesteps', type=int, default=30)
    p.add_argument('--num_points', type=int, default=5)
    p.add_argument('--g_diag', type=float, nargs=3, default=(1.0, 1.0, 1.0),
                   metavar=('GX', 'GY', 'GZ'),
                   help='per-axis actuator gain of the DATASET (must match the '
                        'ph_gp_sde run being compared against, e.g. 0.5 0.7 '
                        '0.5). Previously missing here, which silently '
                        'selected a different dataset than ph_gp_sde trained '
                        'on.')

    p.add_argument('--hidden_dim', type=int, default=32)
    p.add_argument('--relative_inputs', action='store_true')
    p.add_argument('--gp_core', action='store_true',
                   help='structured kind only: predict the physical constants '
                        'of M/D/g/Sigma with the Matern x periodic GP instead '
                        'of learning them as constants (see ph_gp_sde). '
                        'Implies --relative_inputs for the core GPs. Ignored '
                        'by the gp/nn kinds.')
    p.add_argument('--nn_core', action='store_true',
                   help='structured kind only: the MLP twin of --gp_core. An '
                        'MLP(q) predicts the physical SUB-COMPONENTS (m, I, '
                        'ell, c, d, gamma) as a bounded residual on the base '
                        'constants, and the closed forms assemble M/D/g from '
                        'them — as opposed to --subnet_kind nn, where the net '
                        'emits every matrix entry directly. Residual is zero '
                        'at init (output layer zero-initialised), so this is a '
                        'strict superset of the constants-only model. Implies '
                        '--relative_inputs. Mutually exclusive with --gp_core; '
                        'ignored by the gp/nn kinds.')
    p.add_argument('--structured_potential', action='store_true',
                   help='structured kind only: replace the black-box V_net '
                        'with the closed-form gravity potential V(q) = '
                        'sum_j [R_j g(mu_j^> ell_j + m_j c_j)]_z, learning '
                        '(log m, ell, c, log g) — 12 numbers at n=2 — plus the '
                        '--nn_core / --gp_core residual. Off by default so V '
                        'stays the GP that ph_gp_sde uses (bit-parity).')
    p.add_argument('--share_mass_potential', action='store_true',
                   help='structured kind only, requires --structured_potential: '
                        'build V from M_net\'s OWN (m, ell, c) instead of '
                        're-estimating them, so V_net learns only log g (1 '
                        'parameter instead of 12 + a 2060-weight core). They '
                        'are the same physical quantities, and the two '
                        'functions pin down different things about them: M is '
                        'quadratic in the levers so it CANNOT determine their '
                        'sign, while V is linear and can; V only sees the '
                        'combination g*(mu_j ell_j + m_j c_j) so it cannot '
                        'separate m from g, while M can. Unshared, the two '
                        'heads were measured 120%% apart on that combination '
                        'and disagreed on its sign.')
    p.add_argument('--gravity', type=float, default=None,
                   help='requires --share_mass_potential: FIX gravity at this '
                        'known value (e.g. 9.81) instead of learning log g. '
                        'This closes the lever-SCALE gauge: with g learned, '
                        'ell,c -> L*(ell,c) with I -> L^2*I and g -> L*g '
                        '(g scales the SAME way as the levers) sends both '
                        'M_raw and V_raw to L^2 times themselves, so the one '
                        'anchor factor alpha -> alpha/L^2 cancels both and the '
                        'observables are exactly invariant (measured: 0 and '
                        '3e-16 at L=2). Pinning g makes the same move change V '
                        'by 50%, so the lever magnitude becomes identifiable. '
                        'Costs no generality: the pH scale stays free through '
                        'the independent m -> mu*m, I -> mu*I direction (closed '
                        'by --anchor_m1). V_net then holds ZERO learnable '
                        'parameters. 0 or unset = learn it.')
    p.add_argument('--variational_core', action='store_true',
                   help='structured kind only: mean-field Gaussian posteriors '
                        'on the physical constants (see ph_gp_sde).')
    p.add_argument('--core_prior_std', type=float, default=1.0)
    p.add_argument('--no_friction', action='store_true')
    p.add_argument('--init_gain', type=float, default=0.5,
                   help='orthogonal-init gain (NN variants only)')
    p.add_argument('--m_epsilon', type=float, default=1.0,
                   help='PSD diagonal floor on M_net (gp/nn kinds: PSD_NN / '
                        'PSD_GP_NLink add sqrt(eps) to the Cholesky diagonal)')
    p.add_argument('--d_epsilon', type=float, default=0.5,
                   help='PSD diagonal floor on Dw_net. Previously not exposed, '
                        'so it was always the 0.5 default.')
    p.add_argument('--i_epsilon', type=float, default=None,
                   help='structured kind only: inertia conditioning floor, '
                        'I_i = L_iL_i^T + eps*I, which GUARANTEES '
                        'lambda_min(M) >= eps at every q. Default '
                        f'{structured_subnets.I_EPSILON} (see structured_'
                        'subnets.I_EPSILON). It is also an accuracy floor: the '
                        'true I = diag(0,0,mL^2) is rank 1, so eps biases the '
                        'two axial directions upward and cannot be removed by '
                        'the core net. Lower it (e.g. 0.01) for a tighter fit '
                        'at the cost of conditioning.')
    p.add_argument('--pl_sigma_obs', type=float, default=None,
                   help='variance scale of the one-step PL term, i.e. '
                        'sigma_obs in Sigma_eff = dt*AA^T + 2*sigma_obs^2*I. '
                        'Unset => the dataset --obs_noise_std, which is the '
                        'statistically correct choice. REQUIRED when '
                        '--obs_noise_std is 0: the density is then degenerate, '
                        'elbo_loss_nlink clamps s at its 1e-6 floor and the PL '
                        'weight 1/(2s) = 5e5 swamps the rollout NLL by ~1000x, '
                        'turning training into biased one-step EULER '
                        'regression. Inert under --loss mse (lambda_pl = 0), '
                        'where it only rescales the printed pl column.')
    p.add_argument('--sigma_mode', choices=('full', 'structured'),
                   default='full',
                   help="diffusion parameterisation. 'full' = free 3n x 3 "
                        "learned map. 'structured' = Sigma_j = [v_j]_x R_j^T, "
                        "the exact physics form: 3n parameters instead of a "
                        "learned function. No effect on ODE variants.")

    p.add_argument('--learn_rate', type=float, default=1e-3)
    p.add_argument('--lr_sigma', type=float, default=None,
                   help='separate learning rate for Sigma_net (SDE variants '
                        'only). Unset => same as --learn_rate.')
    p.add_argument('--total_steps', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--mc_samples', type=int, default=1)
    p.add_argument('--beta_max', type=float, default=0.0,
                   help='KL weight; no effect on the NN variants (KL is 0). '
                        'NOTE: 1.0 is NOT "the true ELBO" (per-snapshot-mean '
                        'NLL vs per-window KL — mismatched units); 0.05-0.1 '
                        'is the sane range for GP variants. See '
                        'ph_gp_sde/gp_change.md.')
    p.add_argument('--kl_anneal_steps', type=int, default=3000,
                   help='linear 0->beta_max ramp after --kl_warmup_steps')
    p.add_argument('--kl_warmup_steps', type=int, default=2000,
                   help='hold beta = 0 for this many steps first, so the '
                        'likelihood localises the posteriors before the prior '
                        'widens them. beta(t) = beta_max * '
                        'clip((t - warmup)/anneal, 0, 1).')
    p.add_argument('--anchor_m1', type=float, default=1.0,
                   help='parameter-space pin on m_1 (structured kind only; '
                        'harmless but insufficient alone — see gp_change.md '
                        '§3c). 0 disables.')
    p.add_argument('--anchor_trace', type=float, default=None,
                   help='FUNCTION-SPACE gauge anchor, default 4*n. Structured '
                        'kind: tr M(q) pinned inside StructuredMass. gp/nn '
                        'kinds: tr M^-1(q) pinned at the M_inv() wrapper. '
                        'Either kills the same scale gauge; any positive '
                        'value is a units choice. 0 disables.')
    p.add_argument('--sigma_rollout_grad', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='let the rollout loss back-propagate into Sigma. By '
                        'default that path is stop_gradient-ed: it carries '
                        'only Sigma->0 collapse pressure; the PL alone owns '
                        'the diffusion (gp_change.md Change 5). No effect on '
                        'ODE variants. ph_nn_sde defaults this ON for '
                        'pendulum parity — pass --no-sigma_rollout_grad '
                        'there to restore the detach.')
    p.add_argument('--subnet_kind', choices=('structured', 'gp', 'nn'),
                   default=None,
                   help='override the variant folder\'s default subnet kind — '
                        'e.g. run ph_gp_ode with --subnet_kind structured to '
                        'get the pure diffusion ablation of ph_gp_sde instead '
                        'of the black-box GP baseline.')
    p.add_argument('--loss', choices=('nll', 'mse'), default='nll',
                   help="training objective. 'nll' (default) = the likelihood "
                        "family: rollout NLL + lambda_pl*PL + (beta/N)*KL. "
                        "'mse' = the pendulum baselines' plain rollout "
                        "L2+geodesic MSE (rotmat_L2_geodesic_loss): "
                        "mean theta^2 + mean ||d_omega||^2 with no learned "
                        "sigmas, no KL, and --lambda_pl forced to 0 — exact "
                        "parity with 3D_SO3_Windy_Pendulum/ph_nn_ode. "
                        "Available to every variant for loss ablations.")
    p.add_argument('--lambda_pl', type=float, default=1.0)
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--eval_every', type=int, default=200)

    p.add_argument('--float64', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_dir', type=str, default=default_save_dir)
    p.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    p.add_argument('--name', type=str, default='arm')

    # ── MuJoCo-only knobs (this package; absent from arm_n_link) ──
    p.add_argument('--link_radius', type=float, default=0.3,
                   help='capsule radius. Sets the axial inertia '
                        'I_zz = m r^2 / 2, which is the light direction of '
                        'M(q) and therefore the conditioning knob: cond(M) '
                        'grows like 1/r^2 as r -> 0. MuJoCo rejects the '
                        'analytic default diag(0, 0, .) as non-physical, so '
                        'this cannot simply be set to zero.')
    p.add_argument('--armature', type=float, default=0.0,
                   help='rotor inertia added to the JOINT-space diagonal, i.e. '
                        'T^T alpha T in absolute coordinates. That is NOT of '
                        'the form blkdiag(I_i) + sum_i m_i J_vi^T J_vi, so no '
                        'ArmParams represents it: any value > 0 puts the truth '
                        'outside the structured model class. This is the knob '
                        'for the misspecified experiment.')
    p.add_argument('--frictionloss', type=float, default=0.0,
                   help='Coulomb joint friction. Non-smooth, whereas D(q) is '
                        'linear viscous — also outside the model class. The '
                        'second misspecification knob.')
    p.add_argument('--n_substeps', type=int, default=N_SUBSTEPS,
                   help='Lie-Heun substeps per outer dt in the MODEL rollout. '
                        'With MuJoCo data this is an accuracy knob, not a '
                        'matching requirement: the error floor falls as h^2, '
                        'so 10 -> 20 -> 40 substeps gives a floor on ||dw||^2 '
                        'of roughly 9e-7 -> 5e-8 -> 3e-9. Keep it well below '
                        'the loss you expect to reach or you are measuring the '
                        'integrator, not the model.')
    p.add_argument('--mj_timestep', type=float, default=0.001,
                   help='MuJoCo internal RK4 step. The outer --dt is what gets '
                        'stored; this only controls integration accuracy.')
    p.add_argument('--dt', type=float, default=0.05,
                   help='outer sampling interval, i.e. the spacing of stored '
                        'observations. Must match the analytic env (0.05) for '
                        'cross-dataset comparisons to mean anything.')
    if default_overrides:
        p.set_defaults(**default_overrides)
    return p.parse_args()


def model_dtype(args):
    return jnp.float64 if args.float64 else jnp.float32


def core_tag(args):
    """Short marker for the structured kind's core mode, e.g. ``'+nncore+V'``.

    Goes into both the console banner and the run-directory name: without it the
    "NN predicts whole matrices" and "NN predicts sub-components" experiments
    produce *identical* directory names inside ``ph_nn_ode/data/`` and silently
    overwrite each other's place in the listing.
    """
    tag = ''
    if getattr(args, 'gp_core', False):
        tag += '+gpcore'
    if getattr(args, 'nn_core', False):
        tag += '+nncore'
    if getattr(args, 'structured_potential', False):
        tag += '+Vshared' if getattr(args, 'share_mass_potential', False) else '+V'
    if getattr(args, 'gravity', None):
        tag += 'gfix'
    return tag


# ═════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════

def arrange_data(x, t, num_points=2):
    """Slice `(num_us, T, N, D)` trajectories into overlapping windows."""
    assert 2 <= num_points <= len(t)
    stack = [x[:, i:x.shape[1] - num_points + i + 1, :, :] if i < num_points - 1
             else x[:, i:, :, :] for i in range(num_points)]
    stack = np.stack(stack, axis=1)
    stack = np.reshape(stack, (x.shape[0], num_points, -1, x.shape[3]))
    return stack, t[:num_points]


def load_data(args):
    r"""Load the MuJoCo dataset and the ground-truth `ArmParams` that made it.

    Returns `(train, test, dt_outer, gt_params)` — one element longer than the
    analytic version. The extra element is load-bearing: with MuJoCo the truth
    is a property of the compiled ``mjModel``, so it must travel *with the
    data*. Reconstructing it by calling
    :func:`arm_nlink_physics.uniform_chain_params` would silently hand every
    diagnostic the wrong arm, because that helper's default inertia
    $\mathbb{I}=\mathrm{diag}(0,0,\cdot)$ is not physically realisable and is
    not what MuJoCo simulated (see ``mujoco_env/build_mjcf.py``).
    """
    data, path = get_dataset(
        n=args.n, seed=args.seed, samples=args.samples,
        timesteps=args.timesteps, test_split=0.5, save_dir=args.data_dir,
        obs_noise_std=args.obs_noise_std, friction_coeff=args.friction_coeff,
        varying_friction=args.varying_friction, air_drag=args.air_drag,
        external_force_std=args.external_force_std,
        wind_force_std=args.wind_force_std, random_u=args.random_u,
        random_u_scale=args.random_u_scale, g_diag=tuple(args.g_diag),
        us=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)),
        link_radius=args.link_radius, armature=args.armature,
        frictionloss=args.frictionloss, mj_timestep=args.mj_timestep,
        dt=args.dt,
    )
    print(f'dataset: {path}')
    st = data['settings']
    print(f'  source={st.get("source", "analytic")}  matched={st.get("matched")}'
          f'  armature={st.get("armature")}  frictionloss={st.get("frictionloss")}')

    def to_windows(arr):
        w, _ = arrange_data(arr, data['t'], num_points=args.num_points)
        w = np.transpose(w, (1, 0, 2, 3))
        return jnp.asarray(w.reshape(w.shape[0], -1, w.shape[-1]),
                           dtype=model_dtype(args))

    return (to_windows(data['x']), to_windows(data['test_x_noisy']),
            float(data['t'][1] - data['t'][0]),
            gt_params_from_settings(st))


def gt_params_from_settings(settings) -> 'phys.ArmParams':
    """Rebuild the MuJoCo ground-truth `ArmParams` from a dataset's settings."""
    raw = settings.get('arm_params')
    if raw is None:
        raise KeyError(
            "dataset has no 'arm_params'; it was not produced by "
            "data_gen/mujoco_arm_datagen.py. Regenerate it -- the MuJoCo "
            "ground truth cannot be reconstructed from the CLI flags alone.")
    return phys.ArmParams(**{k: jnp.asarray(v) for k, v in raw.items()})


# ═════════════════════════════════════════════════════════════════════
# Rollout
# ═════════════════════════════════════════════════════════════════════

def sample_dW(key, B, S, n_outer, n_sub, h, dtype):
    r"""`(B,S,n_outer,n_sub,3)`, pre-scaled so $\mathrm{Var}(dW)=h$.

    Trailing axis is 3, not $3n$: one shared world wind field.
    """
    return jax.random.normal(key, (B, S, n_outer, n_sub, 3),
                             dtype=dtype) * jnp.sqrt(jnp.asarray(h, dtype))


def sample_gp_keys(key, B, S):
    tops = jax.random.split(key, len(SUBNET_NAMES))
    return {name: jax.random.split(k, B * S).reshape(B, S, 2)
            for name, k in zip(SUBNET_NAMES, tops)}


def rollout_single(model, x_traj, h, dW_one, keys_one, n, n_outer, inference_mode):
    r"""One trajectory, `(T_obs, 15n)` in and out.

    The torque slice is ``x_traj[1:]``: the generator appends ``(obs, curr_u)``
    before advancing the torque, so row $k$ holds the torque that produced
    $\mathrm{obs}_k$ and the one driving $t\to t{+}1$ lives at row $t{+}1$.
    Using ``[:-1]`` lags the sequence by a step whenever `random_u` is on.
    """
    x0 = x_traj[0, :12 * n]
    u_per_outer = x_traj[1:, 12 * n:15 * n]
    keyed = KeyedArmModel(model=model, keys=keys_one, inference_mode=inference_mode)

    if model.stochastic:
        traj = lie_heun_sde_rollout_nlink(keyed, x0, u_per_outer, h, dW_one)
    else:
        traj = lie_heun_ode_rollout_nlink(keyed, x0, u_per_outer, h,
                                          N_SUBSTEPS, n_outer)
    u_full = jnp.concatenate([u_per_outer, u_per_outer[-1:]], axis=0)
    return jnp.concatenate([traj, u_full], axis=-1)


def rollout_batch(model, batch, h, dW, keys, n, n_outer, inference_mode=False):
    x_BT = jnp.transpose(batch, (1, 0, 2))

    def per_b(x_b, dW_b, k_b):
        def per_s(dW_s, k_s):
            return rollout_single(model, x_b, h, dW_s, k_s, n, n_outer,
                                  inference_mode)
        return jax.vmap(per_s)(dW_b, k_b)

    return jnp.transpose(jax.vmap(per_b)(x_BT, dW, keys), (2, 0, 1, 3))


# ═════════════════════════════════════════════════════════════════════
# Loss
# ═════════════════════════════════════════════════════════════════════

def loss_fn(model, batch, h, dW, keys, beta, N, lambda_pl, dt_outer,
            n, n_outer, inference_mode=False, loss_kind='nll'):
    traj = rollout_batch(model, batch, h, dW, keys, n, n_outer, inference_mode)
    target = jnp.broadcast_to(batch[:, :, None, :], traj.shape)

    nll = elbo_nll_nlink(target[1:], traj[1:],
                         model.log_sigma_R, model.log_sigma_omega, n)
    kl = kl_per_subnet(model)
    pl = pl_loss_nlink(model, batch, dt_outer, model.sigma_obs_omega, keys, n,
                       inference_mode=inference_mode)

    if loss_kind == 'mse':
        # Pendulum-parity objective (`rotmat_L2_geodesic_loss_safe`): plain
        # rollout MSE, mean per-link geodesic^2 + mean ||Delta omega||^2 (the
        # pendulum's L2 also spans the control channels, which the rollout
        # copies through — their contribution is identically zero, so this IS
        # the exact n-link generalisation). The learned sigmas, the KL and the
        # PL never enter the gradient; they are still computed above so the
        # printed columns stay comparable across --loss modes.
        total = nll['mean_theta_sq'] + nll['mean_omega_sq']
    else:
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
# Diagnostics
# ═════════════════════════════════════════════════════════════════════

def subnet_errors(model, q_samples, params, n, sigma_true):
    r"""Gauge-corrected relative error of every subnet vs the analytic physics.

    Two unobservable gauges are removed first: the port-Hamiltonian **scale**
    $\beta$ (fitted from $M^{-1}$; momenta are never measured, so
    $(M,V,D,g,\Sigma)\mapsto\beta(\cdot)$ leaves $\omega$ unchanged) and the
    **additive** offset on $V$ (from $H\mapsto H+c$). Without both, the numbers
    measure the gauge rather than the physics.

    Returns `([M, V, D, g, ΣΣᵀ], beta)`. $g$ is included because it is the
    subnet most easily left unidentified — it only gets signal when the torque
    actually moves the system.
    """
    def batch(fn):
        return jax.vmap(fn)(q_samples)

    R_all = q_samples.reshape(-1, n, 3, 3)
    M_gt = jax.vmap(lambda R: jnp.linalg.inv(phys.mass_matrix(params, R)))(R_all)
    V_gt = jax.vmap(lambda R: phys.potential(params, R))(R_all)
    D_gt = jax.vmap(lambda R: phys.dissipation_matrix(
        params, R, jnp.zeros((n, 3))))(R_all)
    g_gt = jax.vmap(lambda R: phys.input_map(params, R))(R_all)
    S_gt = sigma_true * jax.vmap(lambda R: phys.wind_map(params, R))(R_all)

    M_hat = batch(model.M_inv)
    # `model.potential` is the single entry point: with share_mass_potential
    # V_net alone is not evaluable (it holds only log g and needs M_net's
    # (m, ell, c) plus the gauge factor alpha).
    V_hat = batch(lambda q: model.potential(q))
    # `_invariant_q` is the right featuriser for BOTH families and is exactly
    # what `drift_p` feeds these two subnets, so the diagnostic must use it too:
    #   * 'structured' -> returns q unchanged (the closed forms contain T(q) and
    #     R_j^T R_k literally, and any invariant projection their core net wants
    #     is applied inside the subnet);
    #   * 'gp' / 'nn' with --relative_inputs -> returns the 9(n-1) relative
    #     features the nets were BUILT for (dim_inv), so passing the raw 9n q
    #     here is a hard shape error, not a silent inaccuracy.
    # Hard-coding raw q covered the first case and broke the second.
    D_hat = batch(lambda q: model._call(model.Dw_net,
                                        model._invariant_q(q), None))
    g_hat = batch(lambda q: model._call(model.g_net,
                                        model._invariant_q(q), None))
    S_hat = batch(model.Sigma)

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

def train(args, *, model_name: str, subnet_kind: str, stochastic: bool):
    # Against MuJoCo data the substep count is an accuracy knob rather than a
    # property that must match the generator -- see the N_SUBSTEPS docstring.
    global N_SUBSTEPS
    N_SUBSTEPS = int(args.n_substeps)

    key = jax.random.PRNGKey(args.seed)
    n = args.n
    dtype = model_dtype(args)
    # CLI override of the folder's default kind (e.g. structured-minus-
    # diffusion via `ph_gp_ode/train.py --subnet_kind structured`).
    if args.subnet_kind:
        subnet_kind = args.subnet_kind
    # NN subnets are point estimates: the KL is identically zero, so any
    # beta_max would be silently inert. Force it to 0 (and say so) rather
    # than record a flag value that did nothing.
    if subnet_kind == 'nn' and args.beta_max != 0.0:
        print(f'note: --beta_max {args.beta_max:g} forced to 0 for '
              f'subnet_kind=nn (no weight posterior -> KL == 0).')
        args.beta_max = 0.0
    # The core flags only exist on the structured subnets; gp/nn emit matrix
    # entries directly and have nothing to plug into a closed form. Say so
    # rather than record a flag that did nothing.
    if subnet_kind != 'structured' and (args.gp_core or args.nn_core):
        which = ' '.join(f'--{f}' for f in ('gp_core', 'nn_core')
                         if getattr(args, f))
        print(f'note: {which} ignored for subnet_kind={subnet_kind} (no closed '
              f'form to feed; use --subnet_kind structured).')
        args.gp_core = args.nn_core = False
    # --loss mse is the pendulum baselines' plain rollout MSE: no one-step PL
    # term and no KL enter the objective. Force the weights to 0 (and say so)
    # rather than record flag values that did nothing.
    if args.loss == 'mse':
        if args.lambda_pl != 0.0:
            print(f'note: --lambda_pl {args.lambda_pl:g} forced to 0 by '
                  f'--loss mse (pendulum ph_nn_ode parity: no PL term).')
            args.lambda_pl = 0.0
        if args.beta_max != 0.0:
            print(f'note: --beta_max {args.beta_max:g} forced to 0 by '
                  f'--loss mse (no KL in the MSE objective).')
            args.beta_max = 0.0
        if stochastic and not args.sigma_rollout_grad:
            print('note: --loss mse with the Sigma rollout detach active '
                  'means Sigma receives NO gradient at all (frozen at init). '
                  'Pass --sigma_rollout_grad for pendulum ph_nn_sde parity '
                  '(Sigma trained by the rollout MSE, i.e. collapse-pressure '
                  'only), or use --loss nll for a data-identified diffusion.')
    # Resolve the n-dependent anchor default HERE so history.pkl records the
    # actual number a future load_run must rebuild with.
    if args.anchor_trace is None:
        args.anchor_trace = 4.0 * n

    train_x, test_x, dt_outer, gt_params_mj = load_data(args)
    h = dt_outer / N_SUBSTEPS
    n_outer = args.num_points - 1
    N_train = train_x.shape[1]
    print(f'train windows: {train_x.shape}   test: {test_x.shape}   '
          f'dt={dt_outer:.3f}  h={h:.4f}')

    # The PL's variance scale is a separate knob from the dataset noise (see
    # --pl_sigma_obs): at obs_noise_std = 0 the transition density is degenerate
    # and the raw value would clamp to the 1e-6 floor.
    pl_sigma = (args.obs_noise_std if args.pl_sigma_obs is None
                else float(args.pl_sigma_obs))
    if pl_sigma != args.obs_noise_std:
        print(f'note: PL variance scale sigma_obs={pl_sigma:g}, decoupled from '
              f'--obs_noise_std {args.obs_noise_std:g}.')
    elif pl_sigma == 0.0 and args.lambda_pl != 0.0 and args.loss != 'mse':
        print('WARNING: --obs_noise_std 0 with the PL term active. s clamps to '
              'the 1e-6 floor, so the PL weight is 5e5 and dominates the '
              'rollout NLL. Pass --pl_sigma_obs (e.g. 0.1) or --loss mse.')

    key, k_model = jax.random.split(key)
    model = ArmPortHamiltonian(
        key=k_model, n=n, subnet_kind=subnet_kind, stochastic=stochastic,
        hidden_dim=args.hidden_dim, friction=not args.no_friction,
        relative_inputs=args.relative_inputs, init_gain=args.init_gain,
        init_sigma_obs_omega=pl_sigma, m_epsilon=args.m_epsilon,
        d_epsilon=args.d_epsilon, i_epsilon=args.i_epsilon,
        sigma_mode=args.sigma_mode,
        gp_core=args.gp_core, nn_core=args.nn_core,
        structured_potential=args.structured_potential,
        share_mass_potential=args.share_mass_potential,
        gravity=args.gravity,
        variational_core=args.variational_core,
        core_prior_std=args.core_prior_std,
        anchor_m1=args.anchor_m1, anchor_trace=args.anchor_trace,
        sigma_detach_rollout=not args.sigma_rollout_grad, dtype=dtype)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_array)))
    print(f'model: {model_name}  n={n}  subnets={subnet_kind}{core_tag(args)}  '
          f'stochastic={stochastic}  sigma_mode={model.sigma_mode}  '
          f'relative_inputs={model.relative_inputs}  '
          f'anchor_trace={model.anchor_trace}  '
          f'sigma_detach={model.sigma_detach_rollout}  '
          f'V={type(model.V_net).__name__}  '
          f'dtype={jnp.dtype(dtype).name}  trainable params={n_params}')

    # AdamW everywhere, optionally with a separate rate for Sigma_net (SDE
    # variants only) — mirrors ph_gp_sde.make_optimizer.
    def leg(lr):
        return optax.chain(optax.clip_by_global_norm(args.grad_clip),
                           optax.adamw(lr))

    lr_sigma = args.learn_rate if args.lr_sigma is None else float(args.lr_sigma)
    if lr_sigma == args.learn_rate or model.Sigma_net is None:
        optim = leg(args.learn_rate)
    else:
        arrays = eqx.filter(model, eqx.is_array)
        labels = jax.tree_util.tree_map(lambda _: 'base', arrays)
        labels = eqx.tree_at(
            lambda t: t.Sigma_net, labels,
            replace=jax.tree_util.tree_map(
                lambda _: 'sigma', eqx.filter(model.Sigma_net, eqx.is_array)))
        optim = optax.multi_transform(
            {'base': leg(args.learn_rate), 'sigma': leg(lr_sigma)}, labels)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # MuJoCo build: gt_params comes from the dataset (see load_data), NOT from
    # uniform_chain_params -- the arm MuJoCo simulated has a real capsule
    # inertia, not the analytic default diag(0, 0, .).
    gt_params = gt_params_mj

    @eqx.filter_jit
    def train_step(model, opt_state, batch, dW, keys, beta):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, batch, h, dW, keys, beta, N_train, args.lambda_pl,
            dt_outer, n, n_outer, loss_kind=args.loss)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss, aux

    @eqx.filter_jit
    def eval_step(model, batch, dW, keys):
        _, aux = loss_fn(model, batch, h, dW, keys, jnp.asarray(0.0), N_train,
                         args.lambda_pl, dt_outer, n, n_outer,
                         inference_mode=True, loss_kind=args.loss)
        return aux

    os.makedirs(args.save_dir, exist_ok=True)
    # `subnet_kind` + core mode + loss are part of the identity of a run: two
    # experiments that differ only in those would otherwise be indistinguishable
    # from the directory name alone.
    run = (f'{args.name}{n}link_{subnet_kind}{core_tag(args)}_{args.loss}'
           f'_obs{args.obs_noise_std:g}_fric{args.friction_coeff:g}'
           f'_wind{args.wind_force_std:g}_s{args.total_steps}'
           # The scale-gauge anchor is part of a run's identity: with it off the
           # pH gauge is flat, so the recovered constants mean something
           # different. Two runs differing only in that must not differ only by
           # timestamp.
           f'{"_noanchor" if model.anchor_trace is None else ""}'
           # model.relative_inputs, not args: --nn_core / --gp_core force it ON
           # and n=1 forces it OFF, so the flag as typed is not what the model
           # used. A directory that says "absolute inputs" about a run that used
           # relative ones is worse than no tag at all.
           f'{"_rel" if model.relative_inputs else ""}_{time.strftime("%y%m%d-%H%M%S")}')
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

        dW = sample_dW(k_dw, B, args.mc_samples, n_outer, N_SUBSTEPS, h, batch.dtype)
        keys = sample_gp_keys(k_gp, B, args.mc_samples)
        # beta(t) = beta_max * clip((t - warmup)/anneal, 0, 1) — see
        # ph_gp_sde/gp_change.md Change 2.
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
            history.append({k: float(v) for k, v in aux.items() if jnp.ndim(v) == 0})

    key, k_dw, k_gp = jax.random.split(key, 3)
    B_te = test_x.shape[1]
    te = eval_step(model, test_x,
                   sample_dW(k_dw, B_te, 1, n_outer, N_SUBSTEPS, h, test_x.dtype),
                   sample_gp_keys(k_gp, B_te, 1))

    q_samp = train_x[0, :64, :9 * n].astype(dtype)
    err, beta_fit = subnet_errors(model, q_samp, gt_params, n, args.wind_force_std)

    print('\n' + '=' * 62)
    print(f'final ({model_name}, test set, posterior mean)')
    print('=' * 62)
    print(f'  geodesic^2 per link : {float(te["mean_theta_sq"]):.5f}  '
          f'(rms angle {np.sqrt(float(te["mean_theta_sq"])):.4f} rad)')
    print(f'  ||dw||^2            : {float(te["mean_omega_sq"]):.5f}')
    print(f'  pl_loss             : {float(te["pl_loss"]):.4f}')
    print(f'  sigma_R / sigma_w   : {float(te["sigma_R"]):.4f} / '
          f'{float(te["sigma_omega"]):.4f}')
    print(f'\nsubnet relative error vs analytic ground truth  '
          f'(gauge-corrected, beta={float(beta_fit):.4f})')
    for lbl, v in zip(('M^-1', 'V   ', 'D   ', 'g   ', 'SS^T'), err):
        print(f'  {lbl} : {float(v):.4f}')
    if not stochastic:
        # Sigma_hat == 0 for an ODE variant, so the SS^T row measures nothing:
        # it reads 1.0 against a non-zero truth and 0.0 when the wind is off too
        # (rel(0, 0) = 0). Say which, rather than claim a fixed value.
        print(f'  (SS^T is meaningless here: deterministic ODE variant, so '
              f'Sigma == 0'
              f'{" and the dataset wind is off as well" if args.wind_force_std == 0 else ""})')
    print('=' * 62)

    eqx.tree_serialise_leaves(os.path.join(run_dir, 'model.eqx'), model)
    with open(os.path.join(run_dir, 'history.pkl'), 'wb') as f:
        # Static fields never reach model.eqx: every architecture-affecting
        # marker must be recorded here so a loader can rebuild the skeleton.
        # `i_epsilon` used to be pinned to the module constant here, which was
        # right while it was ONLY a module constant. Now that --i_epsilon can
        # override it, record the value the run actually used: the constant is
        # merely the default, and a loader that rebuilt with 0.05 after a
        # `--i_epsilon 0.01` run would silently reproduce different physics
        # (i_epsilon is a static float, so nothing would fail to deserialise).
        pickle.dump({'history': history,
                     'args': dict(vars(args),
                                  i_epsilon=(structured_subnets.I_EPSILON
                                             if args.i_epsilon is None
                                             else float(args.i_epsilon)),
                                  residual_kappa=structured_subnets.RESIDUAL_KAPPA,
                                  sigma_detach_rollout=not args.sigma_rollout_grad),
                     'model_name': model_name, 'subnet_kind': subnet_kind,
                     'stochastic': stochastic,
                     'structured_potential': args.structured_potential,
                     'share_mass_potential': args.share_mass_potential,
                     'gravity': args.gravity,
                     'nn_core': args.nn_core, 'gp_core': args.gp_core,
                     'final_test': {k: float(v) for k, v in te.items()
                                    if jnp.ndim(v) == 0},
                     'subnet_err': np.asarray(err).tolist(),
                     'beta': float(beta_fit)}, f)
    print(f'\nsaved to {run_dir}')
    return model

r"""Trajectory dataset generator for the $n$-link windy arm on $SO(3)^n$.

Mirrors ``envs/pendulum_so3/datagen/windy_pendulum_3d_datagen.py`` — same dict keys, same
train/test split, same pickle conventions — so the existing trainers and
comparison scripts need only a path change. At $n=1$ the produced arrays have
exactly the same shape and layout as the single-pendulum dataset.

Per-sample layout (last axis, width $15n$)
------------------------------------------
    [ 0     : 9n  )   $\mathrm{vec}(R_1),\dots,\mathrm{vec}(R_n)$   attitudes
    [ 9n    : 12n )   $\omega_1,\dots,\omega_n$                     body rates
    [ 12n   : 15n )   $u_1,\dots,u_n$                               joint torques

.. important:: Torque time convention
   Row $k$ stores the torque that **produced** $\mathrm{obs}_k$ — the one applied
   on the transition $k{-}1\to k$ — because the sampling loop appends
   ``(obs, curr_u)`` before drawing the next torque. (Row 0 stores the torque
   about to be applied, so it doubles as the $0\to1$ torque.)

   Consumers integrating $t\to t{+}1$ must therefore read the torque at
   **row $t{+}1$**, i.e. ``x[1:, 12n:15n]`` — not ``x[:-1, …]``. The wrong slice
   is silently harmless when ``random_u=False`` (a shifted constant is the same
   constant) and badly wrong when it is on.

Observation noise (§15.1)
-------------------------
Applied per link, geometrically:

.. math::
    \tilde R_i = R_i\exp([\epsilon_i]_\times),\ \epsilon_i\sim\mathcal N(0,\sigma_{\rm obs}^2I_3),
    \qquad \tilde\omega_i = \omega_i + \eta_i .

Torques are never corrupted — they are known inputs, not measurements.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.arm_nlink_so3.windy_arm_nlink_so3 import windy_arm_nlink_so3    # noqa: E402


# ─────────────────── Pickle helpers ───────────────────

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved data to {path}')


def from_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f'Loaded data from {path}')
    return data


# ─────────────────── Observation noise ───────────────────

def _exp_so3_batch_np(eps: np.ndarray) -> np.ndarray:
    r"""Vectorized Rodrigues formula, `(..., 3) -> (..., 3, 3)`.

    .. math::
        \exp([\epsilon]_\times) = I + \frac{\sin\theta}{\theta}[\epsilon]_\times
            + \frac{1-\cos\theta}{\theta^2}[\epsilon]_\times^2,\quad \theta=\lVert\epsilon\rVert

    with the small-angle Taylor branch (`1 - θ²/6`, `1/2 - θ²/24`) selected by
    `np.where` so `θ = 0` is safe. Fully vectorized — the single-pendulum
    generator looped in Python, which is far too slow at $n$ links.
    """
    theta_sq = np.sum(eps ** 2, axis=-1, keepdims=True)         # (..., 1)
    theta = np.sqrt(theta_sq)
    small = theta_sq < 1e-20
    theta_safe = np.where(small, 1.0, theta)

    A = np.where(small, 1.0 - theta_sq / 6.0, np.sin(theta_safe) / theta_safe)
    B = np.where(small, 0.5 - theta_sq / 24.0,
                 (1.0 - np.cos(theta_safe)) / (theta_safe ** 2))

    zero = np.zeros(eps.shape[:-1])
    K = np.stack([
        np.stack([zero, -eps[..., 2], eps[..., 1]], axis=-1),
        np.stack([eps[..., 2], zero, -eps[..., 0]], axis=-1),
        np.stack([-eps[..., 1], eps[..., 0], zero], axis=-1),
    ], axis=-2)                                                  # (..., 3, 3)

    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + A[..., None] * K + B[..., None] * (K @ K)


def add_proper_noise_nlink(clean_data: np.ndarray, obs_noise_std: float,
                           n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Geometrically-correct observation noise on `(..., 15n)` data.

    Rotations are perturbed **on the manifold** via the right-multiplied
    exponential map (so $\tilde R_i$ stays in $SO(3)$ exactly), rates additively,
    torques not at all.
    """
    noisy = np.copy(clean_data)
    base_shape = clean_data.shape[:-1]

    R_flat = clean_data[..., :9 * n].reshape(base_shape + (n, 3, 3))
    omega = clean_data[..., 9 * n:12 * n]

    eps = rng.normal(0.0, obs_noise_std, size=base_shape + (n, 3))
    R_noisy = R_flat @ _exp_so3_batch_np(eps)
    noisy[..., :9 * n] = R_noisy.reshape(base_shape + (9 * n,))

    noisy[..., 9 * n:12 * n] = omega + rng.normal(
        0.0, obs_noise_std, size=omega.shape)
    # noisy[..., 12n:] (the torques) intentionally left untouched.
    return noisy


# ─────────────────── Sampling ───────────────────

def sample_windy_arm(
    n=2,
    seed=0,
    timesteps=75,
    trials=50,
    u=None,
    random_u=False,
    random_u_scale=1.0,
    **env_kwargs,
):
    r"""Roll out `trials` trajectories.

    Returns `(trajs, tspan)` with `trajs` of shape
    `(timesteps, trials, 15n)` — the time-major layout the existing trainers
    expect — and `tspan` of shape `(timesteps,)`.

    Trajectories whose $\lVert\omega\rVert_\infty$ exceeds `env.max_speed` are
    rejected and resampled with a fresh seed. This is a soft validity filter
    only (the environment does not clip), matching the single-pendulum
    generator; with strong wind and weak friction, rejections can be frequent.
    """
    env = windy_arm_nlink_so3(n=n, render_mode=None, **env_kwargs)
    act_dim = 3 * n
    u0 = np.zeros(act_dim) if u is None else np.asarray(u, dtype=np.float64).reshape(act_dim)

    trajs = []
    main_seed = int(seed)

    for _ in range(trials):
        for attempt in range(51):
            obs, _ = env.reset(seed=main_seed)
            action_rng = np.random.default_rng(main_seed + 12345)

            def draw_u():
                if random_u:
                    return action_rng.uniform(-random_u_scale, random_u_scale, size=act_dim)
                return u0

            curr_u = draw_u()
            traj = [np.concatenate((obs, curr_u.astype(np.float32)))]

            for _ in range(timesteps - 1):
                obs, _, terminated, truncated, _ = env.step(curr_u)
                traj.append(np.concatenate((obs, curr_u.astype(np.float32))))
                curr_u = draw_u()
                if terminated or truncated:
                    break

            traj = np.stack(traj, axis=0)                     # (timesteps, 15n)
            omega = traj[:, 9 * n:12 * n]
            if (not np.isnan(traj).any()
                    and np.max(np.abs(omega)) < env.max_speed - 1e-3):
                break

            print('  rejected trajectory (nan or speed limit), resampling...')
            main_seed += 10
        else:
            raise RuntimeError(
                'Too many retries generating a valid trajectory — reduce '
                'wind_force_std / random_u_scale or raise max_speed.')

        trajs.append(traj)
        main_seed += 1

    env.close()

    trajs = np.stack(trajs, axis=0)                           # (trials, T, 15n)
    trajs = np.transpose(trajs, (1, 0, 2))                    # (T, trials, 15n)
    return trajs, np.arange(timesteps) * env.dt


# ─────────────────── Dataset ───────────────────

def _fmt(x):
    return str(x).replace('.', 'p')


def get_dataset(
    n=2,
    seed=0,
    samples=50,
    test_split=0.5,
    save_dir=None,
    us=((0.0, 0.0, 0.0),),
    timesteps=75,
    obs_noise_std=0.0,
    friction_coeff=0.5,
    varying_friction=False,
    air_drag=0.0,
    external_force_type='sine',
    external_force_std=0.0,
    wind_force_std=0.0,
    random_u=False,
    random_u_scale=1.0,
    link_length=1.0,
    m=1.0,
    g=9.81,
    g_diag=(1.0, 1.0, 1.0),
    **env_kwargs,
):
    r"""Build (or load) the dataset pickle.

    Returns `(data, path)` where `data` has the same keys as the single-pendulum
    dataset::

        x            : (num_us, T, N_train, 15n)   train, possibly noisy
        test_x       : (num_us, T, N_test,  15n)   test,  clean
        test_x_noisy : (num_us, T, N_test,  15n)   test,  noisy
        t            : (T,)
        settings     : dict of everything above (incl. `n`)

    `us` is a list of constant torque vectors, each of length $3n$ (or length 3,
    which is broadcast to every joint) — one batch of trajectories per entry.
    """
    if save_dir is None:
        raise ValueError('save_dir must be specified.')
    os.makedirs(save_dir, exist_ok=True)

    name = (f'arm{n}link_extforce-{external_force_type}-std{_fmt(external_force_std)}'
            f'_fric{_fmt(friction_coeff)}_varfric{varying_friction}'
            f'_drag{_fmt(air_drag)}_obs_noise{_fmt(obs_noise_std)}'
            f'_wind{_fmt(wind_force_std)}'
            f'_randu{random_u}{"_uScale" + _fmt(random_u_scale) if random_u else ""}'
            # Only tag non-unit gain, so every dataset generated before the
            # actuator gain existed keeps its exact filename and is still found.
            f'{"" if tuple(g_diag) == (1.0, 1.0, 1.0) else "_G" + "-".join(_fmt(v) for v in g_diag)}'
            f'_steps{timesteps}.pkl')
    out_path = os.path.join(save_dir, name)

    try:
        return from_pickle(out_path), out_path
    except FileNotFoundError:
        print(f'Building dataset at {out_path}...')

    per_u = []
    for i, u_i in enumerate(us):
        u_vec = np.asarray(u_i, dtype=np.float64).reshape(-1)
        if u_vec.size == 3:
            u_vec = np.tile(u_vec, n)                       # same torque at every joint
        if u_vec.size != 3 * n:
            raise ValueError(f'us[{i}] has size {u_vec.size}, expected 3 or {3 * n}')

        trajs, tspan = sample_windy_arm(
            n=n, seed=int(seed) + i * 10000, timesteps=timesteps, trials=samples,
            u=u_vec, random_u=random_u, random_u_scale=random_u_scale,
            friction_coeff=friction_coeff, varying_friction=varying_friction,
            air_drag=air_drag, external_force_type=external_force_type,
            external_force_std=external_force_std, wind_force_std=wind_force_std,
            link_length=link_length, m=m, g=g, g_diag=tuple(g_diag),
            **env_kwargs)
        per_u.append(trajs)

    all_clean = np.stack(per_u, axis=0)                      # (num_us, T, N, 15n)

    split_ix = (int(samples * 0.5) if test_split >= 0.5
                else int(samples * (1.0 - test_split)))
    train_clean = all_clean[:, :, :split_ix, :]
    test_clean = all_clean[:, :, split_ix:, :]

    data = {
        't': tspan,
        'settings': {
            'n': n, 'seed': seed, 'samples': samples, 'test_split': test_split,
            'us': us, 'timesteps': timesteps, 'obs_noise_std': obs_noise_std,
            'friction_coeff': friction_coeff, 'varying_friction': varying_friction,
            'air_drag': air_drag, 'external_force_type': external_force_type,
            'external_force_std': external_force_std,
            'wind_force_std': wind_force_std, 'random_u': random_u,
            'random_u_scale': random_u_scale, 'link_length': link_length,
            'm': m, 'g': g,
        },
    }

    if obs_noise_std > 0.0:
        print(f'Applying per-link SO(3) observation noise (std={obs_noise_std})...')
        data['x'] = add_proper_noise_nlink(
            train_clean, obs_noise_std, n, np.random.default_rng(seed + 999))
        data['test_x'] = test_clean
        data['test_x_noisy'] = add_proper_noise_nlink(
            test_clean, obs_noise_std, n, np.random.default_rng(seed + 1999))
    else:
        print('No observation noise added.')
        data['x'] = train_clean
        data['test_x'] = test_clean
        data['test_x_noisy'] = test_clean

    to_pickle(data, out_path)
    return data, out_path


# ─────────────────── CLI ───────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Generate an n-link windy arm dataset on SO(3)^n.')
    p.add_argument('--n', type=int, default=2, help='number of links / ball joints')
    p.add_argument('--save_dir', type=str,
                   default='datasets/windy_arm_nlink')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=10)
    p.add_argument('--timesteps', type=int, default=100)
    p.add_argument('--test_split', type=float, default=0.5)
    p.add_argument('--friction_coeff', type=float, default=0.5)
    p.add_argument('--varying_friction', action='store_true')
    p.add_argument('--air_drag', type=float, default=0.0,
                   help='kappa_i; present in the model but off by default')
    p.add_argument('--external_force_type', type=str, default='sine',
                   choices=['sine', 'square', 'random', 'constant'])
    p.add_argument('--external_force_std', type=float, default=0.0,
                   help='deterministic wind; off by default so wind enters '
                        'only through the dW channel')
    p.add_argument('--wind_force_std', type=float, default=0.1,
                   help='sigma of the stochastic wind (the dW channel)')
    p.add_argument('--obs_noise_std', type=float, default=0.05)
    p.add_argument('--random_u', action='store_true')
    p.add_argument('--random_u_scale', type=float, default=1.0)
    p.add_argument('--link_length', type=float, default=1.0)
    args = p.parse_args()

    data, path = get_dataset(
        n=args.n, seed=args.seed, samples=args.samples, timesteps=args.timesteps,
        test_split=args.test_split, save_dir=args.save_dir,
        friction_coeff=args.friction_coeff, varying_friction=args.varying_friction,
        air_drag=args.air_drag, external_force_type=args.external_force_type,
        external_force_std=args.external_force_std,
        wind_force_std=args.wind_force_std, obs_noise_std=args.obs_noise_std,
        random_u=args.random_u, random_u_scale=args.random_u_scale,
        link_length=args.link_length,
        us=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)),
    )
    print('Done.')
    print(f"  n = {args.n}   per-sample width = {15 * args.n}")
    print(f"  Train (x):           {data['x'].shape}")
    print(f"  Test (clean):        {data['test_x'].shape}")
    print(f"  Test (noisy):        {data['test_x_noisy'].shape}")

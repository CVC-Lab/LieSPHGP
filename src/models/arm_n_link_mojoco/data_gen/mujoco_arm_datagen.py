r"""Trajectory dataset generator backed by **MuJoCo** instead of our own
integrator.

This is a drop-in replacement for ``datasets/windy_arm_nlink_datagen.py``: same
function name, same dict keys, same array layout, same torque-time convention.
That is deliberate — the dataset is the *seam* at which MuJoCo enters the
project. Everything downstream (``utils/train_common.py``, the trainers, the
comparison report) is untouched and cannot tell the difference.

Per-sample layout (last axis, width $15n$)
------------------------------------------
::

    [ 0     : 9n  )   vec(R_1), ..., vec(R_n)      attitudes  (world <- body)
    [ 9n    : 12n )   omega_1, ..., omega_n        body rates
    [ 12n   : 15n )   u_1, ..., u_n                joint torques

.. important:: Torque time convention — copied verbatim, and easy to get wrong.
   Row $k$ stores the torque that **produced** observation $k$, i.e. the one
   applied on the transition $k{-}1\to k$. A consumer integrating $t\to t{+}1$
   must therefore read row $t{+}1$ (``x[1:, 12n:15n]``), which is exactly what
   ``train_common.rollout_batch`` does. Reading ``x[:-1, ...]`` instead is
   silently harmless when ``random_u=False`` — a shifted constant is the same
   constant — and badly wrong when it is on.

What MuJoCo cannot do
---------------------
The analytic env supports wind, air drag and state-dependent friction. MuJoCo
here has none of them, and rather than silently ignore those arguments this
module **raises** if they are non-default. A dataset that quietly dropped the
wind term would train a model against the wrong physics and look merely
"worse", not broken.

Ground truth travels with the data
----------------------------------
``settings['arm_params']`` carries the exact :class:`ArmParams` read back out of
the compiled ``mjModel``. Downstream code must use *that* rather than calling
:func:`arm_nlink_physics.uniform_chain_params`, whose default
$\mathbb{I}=\mathrm{diag}(0,0,\cdot)$ is not the arm MuJoCo simulated — see
``mujoco_env/build_mjcf.py`` for why the inertia had to change.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MJENV = os.path.abspath(os.path.join(_HERE, '..', 'mujoco_env'))
for _p in (_HERE, _MJENV):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_mjcf import ArmSpec                                    # noqa: E402
from mujoco_arm_nlink import MujocoArmEnv, random_state           # noqa: E402

DEFAULT_SAVE_DIR = os.path.abspath(
    os.path.join(_HERE, '..', 'data', 'mujoco_arm_nlink'))


# ─────────────────── Pickle helpers ───────────────────

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved data to {path}')


def from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────── Observation noise ───────────────────

def _exp_so3_batch_np(phi: np.ndarray) -> np.ndarray:
    """Rodrigues on a batch of `(..., 3)` rotation vectors → `(..., 3, 3)`."""
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)
    small = theta < 1e-8
    safe = np.where(small, 1.0, theta)
    A = np.where(small, 1.0 - theta ** 2 / 6.0, np.sin(safe) / safe)[..., 0]
    B = np.where(small, 0.5 - theta ** 2 / 24.0,
                 (1.0 - np.cos(safe)) / safe ** 2)[..., 0]
    K = np.zeros(phi.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -phi[..., 2], phi[..., 1]
    K[..., 1, 0], K[..., 1, 2] = phi[..., 2], -phi[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -phi[..., 1], phi[..., 0]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + A[..., None, None] * K + B[..., None, None] * (K @ K)


def add_proper_noise_nlink(clean_data, obs_noise_std, n, rng):
    r"""Geometrically-correct observation noise on `(..., 15n)` data.

    .. math::
        \tilde R_i = R_i\exp([\epsilon_i]_\times), \qquad
        \tilde\omega_i = \omega_i + \eta_i

    Rotations are perturbed **on the manifold** (right-multiplied exponential),
    so $\tilde R_i$ stays in $SO(3)$ exactly rather than needing re-projection.
    Torques are never corrupted — they are known inputs, not measurements.
    """
    noisy = np.copy(clean_data)
    base = clean_data.shape[:-1]
    R_flat = clean_data[..., :9 * n].reshape(base + (n, 3, 3))
    omega = clean_data[..., 9 * n:12 * n]

    eps = rng.normal(0.0, obs_noise_std, size=base + (n, 3))
    noisy[..., :9 * n] = (R_flat @ _exp_so3_batch_np(eps)).reshape(base + (9 * n,))
    noisy[..., 9 * n:12 * n] = omega + rng.normal(
        0.0, obs_noise_std, size=omega.shape)
    return noisy


# ─────────────────── Sampling ───────────────────

def sample_mujoco_arm(spec: ArmSpec, *, seed=0, timesteps=75, trials=50,
                      dt=0.05, u=None, random_u=False, random_u_scale=1.0,
                      max_speed=8.0):
    r"""Roll out `trials` MuJoCo trajectories → `(timesteps, trials, 15n)`.

    Trajectories whose $\lVert\omega\rVert_\infty$ exceeds `max_speed` are
    rejected and resampled with a fresh seed, mirroring the analytic generator.
    The filter is what keeps both datasets in the same dynamic regime, so it
    must stay even though MuJoCo is perfectly happy to spin fast.
    """
    env = MujocoArmEnv(spec, dt=dt, max_speed=max_speed)
    n = spec.n
    act = 3 * n
    u0 = np.zeros(act) if u is None else np.asarray(u, np.float64).reshape(act)

    trajs, main_seed = [], int(seed)
    for _ in range(trials):
        for _attempt in range(51):
            rng = np.random.default_rng(main_seed)
            action_rng = np.random.default_rng(main_seed + 12345)
            R0, w0 = random_state(rng, n)
            obs = env.reset(R0, w0)

            def draw_u():
                if random_u:
                    return action_rng.uniform(-random_u_scale, random_u_scale,
                                              size=act)
                return u0

            curr_u = draw_u()
            # Row 0 pairs obs_0 with the torque about to be applied; every later
            # row pairs obs_k with the torque that produced it. See the module
            # docstring -- this ordering is load-bearing.
            traj = [np.concatenate((obs, curr_u))]
            for _ in range(timesteps - 1):
                obs = env.step(curr_u)
                traj.append(np.concatenate((obs, curr_u)))
                curr_u = draw_u()

            traj = np.stack(traj, axis=0)
            omega = traj[:, 9 * n:12 * n]
            if (not np.isnan(traj).any()
                    and np.max(np.abs(omega)) < max_speed - 1e-3):
                break
            print('  rejected trajectory (nan or speed limit), resampling...')
            main_seed += 10
        else:
            raise RuntimeError(
                'Too many retries generating a valid trajectory — reduce '
                'random_u_scale or raise max_speed.')
        trajs.append(traj)
        main_seed += 1

    env.close()
    trajs = np.transpose(np.stack(trajs, axis=0), (1, 0, 2))   # (T, trials, 15n)
    return trajs, np.arange(timesteps) * dt


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
    # ── MuJoCo-specific ──
    link_radius=0.3,
    armature=0.0,
    frictionloss=0.0,
    mj_timestep=0.001,
    dt=0.05,
    **env_kwargs,
):
    r"""Build (or load) the MuJoCo dataset pickle. Returns `(data, path)`.

    The signature matches ``windy_arm_nlink_datagen.get_dataset`` argument for
    argument so ``train_common.load_data`` needs no special-casing, plus four
    MuJoCo-only knobs (`link_radius`, `armature`, `frictionloss`, `mj_timestep`).

    Keys of the returned dict::

        x            : (num_us, T, N_train, 15n)   train, possibly noisy
        test_x       : (num_us, T, N_test,  15n)   test,  clean
        test_x_noisy : (num_us, T, N_test,  15n)   test,  noisy
        t            : (T,)
        settings     : dict, including 'arm_params' (the MuJoCo ground truth)
                       and 'arm_spec'

    Raises on any argument MuJoCo cannot honour, rather than ignoring it.
    """
    if save_dir is None:
        save_dir = DEFAULT_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Fail loudly rather than silently simulating different physics.
    if varying_friction:
        raise ValueError(
            'varying_friction is not representable in MuJoCo here: joint '
            'damping is a constant, not a function of height and speed.')
    if air_drag:
        raise ValueError(
            f'air_drag={air_drag} unsupported. The analytic kappa opposes the '
            f'*absolute* rate; MuJoCo joint damping opposes the *relative* '
            f'rate. Adding one as the other would be wrong physics.')
    if wind_force_std or external_force_std:
        raise ValueError(
            f'wind_force_std={wind_force_std}, '
            f'external_force_std={external_force_std}: there is no wind field '
            f'in the MuJoCo model. Use the analytic env for stochastic forcing.')

    spec = ArmSpec(n=n, link_length=link_length, link_radius=link_radius,
                   mass=m, damping=friction_coeff, gain=tuple(g_diag),
                   gravity=g, timestep=mj_timestep, armature=armature,
                   frictionloss=frictionloss)

    name = (f'mj_{spec.tag()}_obs{_fmt(obs_noise_std)}'
            f'_randu{random_u}{"_uScale" + _fmt(random_u_scale) if random_u else ""}'
            f'_dt{_fmt(dt)}_steps{timesteps}_s{seed}_N{samples}.pkl')
    out_path = os.path.join(save_dir, name)

    try:
        return from_pickle(out_path), out_path
    except FileNotFoundError:
        print(f'Building MuJoCo dataset at {out_path}...')

    per_u = []
    for i, u_i in enumerate(us):
        u_vec = np.asarray(u_i, dtype=np.float64).reshape(-1)
        if u_vec.size == 3:
            u_vec = np.tile(u_vec, n)          # same torque at every joint
        if u_vec.size != 3 * n:
            raise ValueError(
                f'us[{i}] has size {u_vec.size}, expected 3 or {3 * n}')
        trajs, tspan = sample_mujoco_arm(
            spec, seed=int(seed) + i * 10000, timesteps=timesteps,
            trials=samples, dt=dt, u=u_vec, random_u=random_u,
            random_u_scale=random_u_scale)
        per_u.append(trajs)

    all_clean = np.stack(per_u, axis=0)                  # (num_us, T, N, 15n)

    split_ix = (int(samples * 0.5) if test_split >= 0.5
                else int(samples * (1.0 - test_split)))
    train_clean = all_clean[:, :, :split_ix, :]
    test_clean = all_clean[:, :, split_ix:, :]

    env = MujocoArmEnv(spec, dt=dt)
    params = env.params
    env.close()

    data = {
        't': tspan,
        'settings': {
            'n': n, 'seed': seed, 'samples': samples, 'test_split': test_split,
            'us': us, 'timesteps': timesteps, 'obs_noise_std': obs_noise_std,
            'friction_coeff': friction_coeff,
            'varying_friction': varying_friction, 'air_drag': air_drag,
            'external_force_type': external_force_type,
            'external_force_std': external_force_std,
            'wind_force_std': wind_force_std, 'random_u': random_u,
            'random_u_scale': random_u_scale, 'link_length': link_length,
            'm': m, 'g': g, 'g_diag': tuple(g_diag),
            # ── MuJoCo provenance ──
            'source': 'mujoco',
            'link_radius': link_radius, 'armature': armature,
            'frictionloss': frictionloss, 'mj_timestep': mj_timestep, 'dt': dt,
            'matched': spec.matched,
            'arm_spec': spec._asdict(),
            # The ground truth, read back out of the compiled mjModel. Consumers
            # MUST use this instead of uniform_chain_params -- see the module
            # docstring.
            'arm_params': {k: np.asarray(v)
                           for k, v in params._asdict().items()},
        },
    }

    rng = np.random.default_rng(seed + 777)
    if obs_noise_std > 0.0:
        print(f'Applying per-link SO(3) observation noise '
              f'(std={obs_noise_std})...')
        data['x'] = add_proper_noise_nlink(train_clean, obs_noise_std, n, rng)
        data['test_x_noisy'] = add_proper_noise_nlink(
            test_clean, obs_noise_std, n, rng)
    else:
        data['x'] = train_clean
        data['test_x_noisy'] = test_clean
    data['test_x'] = test_clean

    to_pickle(data, out_path)
    return data, out_path


# ─────────────────── CLI ───────────────────

def main():
    p = argparse.ArgumentParser(
        description='Generate a MuJoCo ball-joint arm dataset.')
    p.add_argument('--n', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--timesteps', type=int, default=30)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--obs_noise_std', type=float, default=0.0)
    p.add_argument('--friction_coeff', type=float, default=0.5)
    p.add_argument('--random_u', action='store_true', default=True)
    p.add_argument('--no_random_u', dest='random_u', action='store_false')
    p.add_argument('--random_u_scale', type=float, default=2.0)
    p.add_argument('--g_diag', type=float, nargs=3, default=[0.5, 0.7, 0.5])
    p.add_argument('--link_length', type=float, default=1.0)
    p.add_argument('--link_radius', type=float, default=0.3,
                   help='capsule radius; the conditioning knob for M(q)')
    p.add_argument('--m', type=float, default=1.0)
    p.add_argument('--g', type=float, default=9.81)
    p.add_argument('--armature', type=float, default=0.0,
                   help='>0 puts the truth OUTSIDE the model class '
                        '(misspecified experiment)')
    p.add_argument('--frictionloss', type=float, default=0.0,
                   help='>0 adds Coulomb friction, also outside the class')
    p.add_argument('--mj_timestep', type=float, default=0.001)
    p.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR)
    a = p.parse_args()

    data, path = get_dataset(
        n=a.n, seed=a.seed, samples=a.samples, timesteps=a.timesteps,
        test_split=0.5, save_dir=a.save_dir, obs_noise_std=a.obs_noise_std,
        friction_coeff=a.friction_coeff, random_u=a.random_u,
        random_u_scale=a.random_u_scale, g_diag=tuple(a.g_diag),
        link_length=a.link_length, link_radius=a.link_radius, m=a.m, g=a.g,
        armature=a.armature, frictionloss=a.frictionloss,
        mj_timestep=a.mj_timestep, dt=a.dt,
        us=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)),
    )
    x = data['x']
    print(f'\nx            {x.shape}   (num_us, T, N_train, 15n)')
    print(f'test_x       {data["test_x"].shape}')
    print(f't            {data["t"].shape}  dt = {data["t"][1] - data["t"][0]:g}')
    print(f'matched      {data["settings"]["matched"]}')
    print(f'|omega|_max  {np.abs(x[..., 9 * a.n:12 * a.n]).max():.3f}')
    print(f'path         {path}')


if __name__ == '__main__':
    main()

import numpy as np
import pickle
import os
import argparse

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.tennis_racket_3d import tennis_racket_3d


# ─────────────────── Helper Functions ───────────────────
# (identical to windy_pendulum_3d_datagen)

def to_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved data to {path}")


def from_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded data from {path}")
    return data


def arrange_data(x, t, num_points=2):
    """Arrange data to feed into neural ODE in small chunks.

    x : (num_configs, T, N, D)
    t : (T,)

    Returns:
        x_stack : (num_configs, num_points, N_windows, D)
        t_eval  : (num_points,)
    """
    assert num_points >= 2 and num_points <= len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points - 1:
            x_stack.append(x[:, i:-num_points + i + 1, :, :])
        else:
            x_stack.append(x[:, i:, :, :])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack, (x.shape[0], num_points, -1, x.shape[3]))
    t_eval  = t[0:num_points]
    return x_stack, t_eval


def _project_to_so3(R):
    """Numpy SVD projection to SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1.0
        Rp = U @ Vt
    return Rp


def add_proper_noise_3d(clean_data, obs_noise_std, rng):
    """
    Apply geometrically-correct observation noise to SO(3) rigid-body data.

    Identical to windy_pendulum_3d_datagen.add_proper_noise_3d:
      R_noisy   = R @ expm(hat(eps)),  eps ~ N(0, sigma²)
      ω_noisy   = ω + noise
      action is NOT corrupted.

    clean_data : (..., D)  where D = 9 (R) + 3 (ω) + 3 (u) = 15
    """
    noisy_data = np.copy(clean_data)
    base_shape = clean_data.shape[:-1]

    R_flat = clean_data[..., :9]
    omega  = clean_data[..., 9:12]
    action = clean_data[..., 12:]

    # Rotation noise via exponential map
    eps = rng.normal(0.0, obs_noise_std, size=base_shape + (3,))

    R_flat_2d = R_flat.reshape(-1, 9)
    eps_2d    = eps.reshape(-1, 3)
    R_noisy_flat = np.zeros_like(R_flat_2d)

    for i in range(R_flat_2d.shape[0]):
        R = R_flat_2d[i].reshape(3, 3)
        e = eps_2d[i]
        ex = np.array([[0,    -e[2],  e[1]],
                       [e[2],  0,    -e[0]],
                       [-e[1], e[0],  0   ]], dtype=np.float64)
        theta = np.linalg.norm(e)
        if theta < 1e-10:
            R_perturb = np.eye(3) + ex
        else:
            R_perturb = (np.eye(3)
                         + (np.sin(theta) / theta) * ex
                         + ((1 - np.cos(theta)) / theta**2) * (ex @ ex))
        R_noisy_flat[i] = (R @ R_perturb).flatten()

    noisy_data[..., :9]  = R_noisy_flat.reshape(R_flat.shape)
    noisy_data[..., 9:12] = omega + rng.normal(0.0, obs_noise_std, size=base_shape + (3,))
    noisy_data[..., 12:]  = action  # action untouched
    return noisy_data


# ─────────────────── Racket geometry configs ───────────────────

def make_racket_configs(config_list):
    """
    Validate and return a list of racket geometry dicts.

    Each entry in config_list is a dict with keys matching the tennis_racket_3d
    constructor kwargs for geometry:
        head_a, head_b, handle_length, handle_radius,
        total_mass, head_mass_ratio

    Missing keys are filled with the env defaults.
    """
    defaults = dict(
        head_a=0.195,
        head_b=0.135,
        handle_length=0.25,
        handle_radius=0.015,
        total_mass=0.290,
        head_mass_ratio=0.70,
    )
    out = []
    for cfg in config_list:
        full = {**defaults, **cfg}
        out.append(full)
    return out


# ─────────────────── Sampling ───────────────────

def sample_tennis_racket_3d(
    seed=0,
    timesteps=75,
    trials=50,
    disturbance_torque_std=0.0,
    omega_0_scale=2 * np.pi,
    perturb_std=0.05,
    axis_weights=(0.15, 0.70, 0.15),
    # [WIND] wind_force_std=0.0,
    # [FRICTION] friction_coeff=0.0,
    # [FRICTION] varying_friction=False,
    racket_kwargs=None,
    **kwargs
):
    """
    Sample trajectories from the tennis_racket_3d environment.

    Returns
    -------
    trajs : (timesteps, trials, obs_dim + act_dim)   =  (T, N, 15)
    tspan : (timesteps,)

    The action stored in the last 3 dims is the disturbance torque that was
    active during that trajectory (constant per trajectory, zero if
    disturbance_torque_std == 0).  This mirrors the pendulum convention of
    storing the applied torque alongside the state.
    """
    if racket_kwargs is None:
        racket_kwargs = {}

    env = tennis_racket_3d(
        disturbance_torque_std=disturbance_torque_std,
        omega_0_scale=omega_0_scale,
        perturb_std=perturb_std,
        axis_weights=axis_weights,
        # [WIND] wind_force_std=wind_force_std,
        # [FRICTION] friction_coeff=friction_coeff,
        # [FRICTION] varying_friction=varying_friction,
        render_mode=None,
        **racket_kwargs,
        **kwargs
    )

    obs_dim = env.observation_space.shape[0]   # 12
    act_dim = env.action_space.shape[0]        # 3
    dt      = env.dt
    timesteps = int(timesteps)
    trials    = int(trials)

    trajs     = []
    main_seed = int(seed)

    for trial in range(trials):
        valid = False
        retry_count = 0

        while not valid:
            if retry_count > 50:
                raise RuntimeError(
                    "Too many retries generating a valid trajectory "
                    "(NaN or solver instability)."
                )

            obs, info = env.reset(seed=main_seed)

            # The disturbance torque is fixed for this episode; record it as
            # the "action" so the dataset format matches the pendulum exactly.
            curr_u = env._disturbance_torque.copy().astype(np.float32)

            traj = []
            x_init = np.concatenate((obs, curr_u))
            traj.append(x_init)

            for t in range(timesteps - 1):
                # Pass zero control — the disturbance lives inside the env.
                # To generate controlled trajectories, replace with your policy.
                obs, reward, terminated, truncated, info = env.step(
                    np.zeros(act_dim, dtype=np.float32)
                )
                x = np.concatenate((obs, curr_u))
                traj.append(x)

                if terminated or truncated:
                    break

            traj = np.stack(traj, axis=0)   # (timesteps, 15)

            if np.isnan(traj).any():
                retry_count += 1
                main_seed   += 10
                print(f"NaN detected in trial {trial}, retrying (seed={main_seed})...")
                continue

            valid = True

        trajs.append(traj)
        main_seed += 1

    env.close()

    trajs = np.stack(trajs, axis=0)          # (trials, timesteps, 15)
    trajs = np.transpose(trajs, (1, 0, 2))   # (timesteps, trials, 15)
    tspan = np.arange(timesteps) * dt

    return trajs, tspan


# ─────────────────── Dataset ───────────────────

def get_dataset(
    seed=0,
    samples=50,
    test_split=0.5,
    save_dir=None,
    racket_configs=(None,),       # list of dicts (or None for default)
    disturbance_torque_std=0.0,
    omega_0_scale=2 * np.pi,
    perturb_std=0.05,
    axis_weights=(0.15, 0.70, 0.15),
    obs_noise_std=0.0,
    timesteps=75,
    # [WIND] wind_force_std=0.0,
    # [FRICTION] friction_coeff=0.0,
    # [FRICTION] varying_friction=False,
    **kwargs
):
    """
    Build (or load) one pickle per racket geometry configuration.

    Each pickle contains:
        data = {
            "x":            (num_configs, T, N_train, 15),   # train (possibly noisy)
            "test_x":       (num_configs, T, N_test,  15),   # test clean
            "test_x_noisy": (num_configs, T, N_test,  15),   # test noisy
            "t":            (T,),
            "inertia_info": list of dicts (one per config),
            "settings":     {...}
        }

    The leading num_configs axis mirrors the num_us axis in the pendulum
    datagen — each racket geometry is treated like a different "force" regime.

    Returns
    -------
    data     : dict (as above)
    out_path : str
    """
    if save_dir is None:
        raise ValueError("save_dir must be specified.")
    os.makedirs(save_dir, exist_ok=True)

    # ── Build canonical geometry list ──
    resolved_configs = make_racket_configs(
        [c if c is not None else {} for c in racket_configs]
    )

    # ── Filename mirrors pendulum convention ──
    dist_str    = f"dist{str(disturbance_torque_std).replace('.', 'p')}"
    obs_str     = f"obs_noise{str(obs_noise_std).replace('.', 'p')}"
    perturb_str = f"perturb{str(perturb_std).replace('.', 'p')}"
    ncfg_str    = f"ncfg{len(resolved_configs)}"
    steps_str   = f"steps{timesteps}"
    # [WIND]     wind_str = f"wind{str(wind_force_std).replace('.', 'p')}"
    # [FRICTION] fric_str = f"fric{str(friction_coeff).replace('.', 'p')}"
    filename = (
        f"tr3d_dataset_{dist_str}_{obs_str}_{perturb_str}"
        f"_{ncfg_str}_{steps_str}.pkl"
    )
    out_path = os.path.join(save_dir, filename)

    try:
        data = from_pickle(out_path)
        return data, out_path
    except FileNotFoundError:
        print(f"Building dataset at {out_path}...")

    # ── Collect trajectories for each geometry config ──
    trajs_per_config = []
    inertia_info_list = []

    for i, rcfg in enumerate(resolved_configs):
        current_seed = int(seed) + (i * 10000)
        print(f"  Config {i+1}/{len(resolved_configs)}: {rcfg}")

        trajs, tspan = sample_tennis_racket_3d(
            seed=current_seed,
            timesteps=timesteps,
            trials=samples,
            disturbance_torque_std=disturbance_torque_std,
            omega_0_scale=omega_0_scale,
            perturb_std=perturb_std,
            axis_weights=axis_weights,
            # [WIND] wind_force_std=wind_force_std,
            # [FRICTION] friction_coeff=friction_coeff,
            # [FRICTION] varying_friction=varying_friction,
            racket_kwargs=rcfg,
            **kwargs
        )
        trajs_per_config.append(trajs)

        # Record inertia info from a throw-away env instance
        _env = tennis_racket_3d(**rcfg, render_mode=None)
        inertia_info_list.append(_env.get_inertia_info())
        _env.close()

    # (num_configs, T, N, D)
    all_clean_x = np.stack(trajs_per_config, axis=0)

    # ── Train / test split ──
    if test_split >= 0.5:
        split_ix = int(samples * 0.5)
    else:
        split_ix = int(samples * (1.0 - test_split))

    train_clean_x = all_clean_x[:, :, :split_ix, :]
    test_clean_x  = all_clean_x[:, :, split_ix:, :]

    # ── Observation noise ──
    data = {}
    data['t']            = tspan
    data['inertia_info'] = inertia_info_list
    data['settings'] = {
        'seed':                   seed,
        'samples':                samples,
        'test_split':             test_split,
        'racket_configs':         resolved_configs,
        'disturbance_torque_std': disturbance_torque_std,
        'omega_0_scale':          omega_0_scale,
        'perturb_std':            perturb_std,
        'axis_weights':           list(axis_weights),
        'obs_noise_std':          obs_noise_std,
        'timesteps':              timesteps,
        # [WIND]     'wind_force_std':    wind_force_std,
        # [FRICTION] 'friction_coeff':    friction_coeff,
        # [FRICTION] 'varying_friction':  varying_friction,
    }

    if obs_noise_std > 0.0:
        print(f"Applying SO(3) observation noise (std={obs_noise_std})...")
        rng = np.random.default_rng(seed + 999)

        data['x']            = add_proper_noise_3d(train_clean_x, obs_noise_std, rng)
        data['test_x']       = test_clean_x
        data['test_x_noisy'] = add_proper_noise_3d(
            test_clean_x, obs_noise_std,
            np.random.default_rng(seed + 1999)
        )
    else:
        print("No observation noise added.")
        data['x']            = train_clean_x
        data['test_x']       = test_clean_x
        data['test_x_noisy'] = test_clean_x

    to_pickle(data, out_path)
    return data, out_path


# ─────────────────── CLI ───────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a tennis-racket free-rigid-body dataset on SO(3)."
    )
    parser.add_argument("--save_dir",    type=str,   default="data/tennis_data")
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--samples",     type=int,   default=50)
    parser.add_argument("--timesteps",   type=int,   default=100)
    parser.add_argument("--test_split",  type=float, default=0.5)
    parser.add_argument("--obs_noise_std",          type=float, default=0.0)
    parser.add_argument("--perturb_std",            type=float, default=0.05)
    parser.add_argument("--disturbance_torque_std", type=float, default=0.0)
    parser.add_argument("--omega_0_scale",          type=float, default=6.2832)
    # [WIND]     parser.add_argument("--wind_force_std",    type=float, default=0.0)
    # [FRICTION] parser.add_argument("--friction_coeff",    type=float, default=0.0)
    # [FRICTION] parser.add_argument("--varying_friction",  action="store_true")

    # Geometry overrides (applied to ALL configs when using CLI)
    parser.add_argument("--head_a",          type=float, default=0.195)
    parser.add_argument("--head_b",          type=float, default=0.135)
    parser.add_argument("--handle_length",   type=float, default=0.25)
    parser.add_argument("--handle_radius",   type=float, default=0.015)
    parser.add_argument("--total_mass",      type=float, default=0.290)
    parser.add_argument("--head_mass_ratio", type=float, default=0.70)

    args = parser.parse_args()

    # ── Define geometry sweep ──
    # Each dict overrides only the keys you care about; the rest use defaults.
    # This example sweeps over three racket sizes (light/medium/heavy head).
    # Edit freely — or pass a single {} for the default geometry only.
    racket_configs = [
        # Default geometry
        {},
        # Heavier head (more pronounced intermediate-axis instability)
        {"head_mass_ratio": 0.80, "total_mass": 0.300},
        # Lighter head / longer handle
        {"head_mass_ratio": 0.60, "handle_length": 0.28, "total_mass": 0.280},
        # Wider head (larger a → I1 further from I2)
        {"head_a": 0.210, "head_b": 0.135},
    ]

    # Override the first config with any CLI geometry flags
    racket_configs[0] = dict(
        head_a=args.head_a,
        head_b=args.head_b,
        handle_length=args.handle_length,
        handle_radius=args.handle_radius,
        total_mass=args.total_mass,
        head_mass_ratio=args.head_mass_ratio,
    )

    data, path = get_dataset(
        seed=args.seed,
        samples=args.samples,
        timesteps=args.timesteps,
        test_split=args.test_split,
        save_dir=args.save_dir,
        racket_configs=racket_configs,
        disturbance_torque_std=args.disturbance_torque_std,
        omega_0_scale=args.omega_0_scale,
        perturb_std=args.perturb_std,
        obs_noise_std=args.obs_noise_std,
        # [WIND]     wind_force_std=args.wind_force_std,
        # [FRICTION] friction_coeff=args.friction_coeff,
        # [FRICTION] varying_friction=args.varying_friction,
    )

    print("\nDone.")
    print(f"Train (x):          {data['x'].shape}")
    print(f"Test (clean):       {data['test_x'].shape}")
    print(f"Test (noisy):       {data['test_x_noisy'].shape}")
    print(f"Timesteps:          {data['t'].shape}")
    print("\nInertia tensors per config:")
    for i, info in enumerate(data['inertia_info']):
        print(
            f"  [{i}]  I1={info['I1']:.5f}  I2={info['I2']:.5f}  "
            f"I3={info['I3']:.5f}  kg·m²   "
            f"(head_a={info['head_a']}, mass={info['total_mass']}kg)"
        )

"""Stage C held-out generalization test — 5th racket geometry.

Generates fresh trajectories for a racket geometry not seen during training,
loads the saved Stage C --fix_M checkpoint, and evaluates geodesic + subnet MSE.

The 4 training geometries all have head_a ∈ {0.195, 0.210} and head_b=0.135.
The 5th config uses head_a=0.225, head_b=0.115 (longer and narrower head),
plus different mass / handle parameters — genuinely out-of-distribution.

Pass criterion (same as Stage C windowed training metric):
  windowed geo_loss < 0.01 rad²
"""
import torch, glob, argparse
import numpy as np
import os, sys

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))

PENDULUM_ODE_DIR = os.path.join(
    PROJECT_ROOT, 'src/models/3D_SO3_Windy_Pendulum/ph_nn_ode_v2')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/utils'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'datasets'))
sys.path.insert(0, PENDULUM_ODE_DIR)
sys.path.insert(0, THIS_FILE_DIR)

from torchdiffeq import odeint
from network_stageC import DissipativeSO3HamNODE, FixedInertiaFromState
from subnet_diagnostics_stageC import subnet_physics_mse_stageC
from tennis_racket_3d_datagen import sample_tennis_racket_3d, arrange_data
from train_stageC import augment_with_inertia, strip_inertia
from loss_utils import (
    rotmat_L2_geodesic_loss_safe as rotmat_L2_geodesic_loss,
    traj_rotmat_L2_geodesic_loss_safe as traj_rotmat_L2_geodesic_loss,
)


# ── Training config inertia for reference ─────────────────────────────────────
TRAIN_CONFIGS = [
    dict(I1=0.005719, I2=0.011212, I3=0.013221,
         head_a=0.195, head_b=0.135, handle_length=0.25, total_mass=0.29, head_mass_ratio=0.70),
    dict(I1=0.006757, I2=0.008643, I3=0.011019,
         head_a=0.195, head_b=0.135, handle_length=0.25, total_mass=0.30, head_mass_ratio=0.80),
    dict(I1=0.004738, I2=0.014832, I3=0.016495,
         head_a=0.195, head_b=0.135, handle_length=0.28, total_mass=0.28, head_mass_ratio=0.60),
    dict(I1=0.006336, I2=0.012067, I3=0.014693,
         head_a=0.210, head_b=0.135, handle_length=0.25, total_mass=0.29, head_mass_ratio=0.70),
]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path',   type=str,   default=None,
                   help='path to .tar checkpoint; defaults to latest Stage C fixM run')
    p.add_argument('--head_a',      type=float, default=0.225,
                   help='head semi-axis along e₁ (long), m')
    p.add_argument('--head_b',      type=float, default=0.115,
                   help='head semi-axis along e₂ (short), m')
    p.add_argument('--handle_length', type=float, default=0.27)
    p.add_argument('--total_mass',    type=float, default=0.31)
    p.add_argument('--head_mass_ratio', type=float, default=0.75)
    p.add_argument('--n_samples',   type=int,   default=25)
    p.add_argument('--timesteps',   type=int,   default=100)
    p.add_argument('--num_points',  type=int,   default=5,
                   help='window size for windowed eval (must match training)')
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--solver',      type=str,   default='rk4')
    return p.parse_args()


def find_latest_ckpt():
    run_base = os.path.join(THIS_FILE_DIR, 'data', 'run_tr3d_stageC_fp32')
    subdirs  = sorted(glob.glob(os.path.join(run_base, '*fixM*')))
    if not subdirs:
        raise FileNotFoundError(f"No fixM Stage C runs under {run_base}")
    latest = subdirs[-1]
    final  = os.path.join(latest, 'tr3d-so3ham-rk4-5p.tar')
    if not os.path.exists(final):
        raise FileNotFoundError(f"Final checkpoint not found: {final}")
    return final


def main():
    args = get_args()
    float_type = torch.float32
    torch.set_default_dtype(torch.float32)
    device = torch.device('cpu')

    # ── 5th config geometry ────────────────────────────────────────────────
    racket_kwargs = dict(
        head_a=args.head_a,
        head_b=args.head_b,
        handle_length=args.handle_length,
        total_mass=args.total_mass,
        head_mass_ratio=args.head_mass_ratio,
    )

    # Instantiate env just to read inertia (no data generated here)
    from envs.tennis_racket_3d import tennis_racket_3d as Env
    env5 = Env(**racket_kwargs)
    iinfo = env5.get_inertia_info()
    env5.close()
    I1, I2, I3 = iinfo['I1'], iinfo['I2'], iinfo['I3']

    print("=" * 60)
    print("Stage C — 5th config held-out generalization test")
    print("=" * 60)
    print(f"\n5th config geometry:")
    print(f"  head_a={args.head_a:.3f}  head_b={args.head_b:.3f}  "
          f"handle={args.handle_length:.3f}  mass={args.total_mass:.3f}  "
          f"hmr={args.head_mass_ratio:.2f}")
    print(f"  I1={I1:.6f}  I2={I2:.6f}  I3={I3:.6f}  kg·m²")
    print(f"  I2/I1={I2/I1:.2f}  asym=(I3-I1)/I2={(I3-I1)/I2:.2f}")

    print(f"\nTraining configs (for comparison):")
    for ci, c in enumerate(TRAIN_CONFIGS):
        print(f"  cfg{ci}: I1={c['I1']:.6f}  I2={c['I2']:.6f}  I3={c['I3']:.6f}  "
              f"head_a={c['head_a']:.3f}  head_b={c['head_b']:.3f}  "
              f"asym={(c['I3']-c['I1'])/c['I2']:.2f}")

    # ── Generate trajectories ──────────────────────────────────────────────
    print(f"\nGenerating {args.n_samples} trajectories "
          f"(T={args.timesteps}, seed={args.seed + 99999})...")
    trajs, tspan = sample_tennis_racket_3d(
        seed=args.seed + 99999,
        timesteps=args.timesteps,
        trials=args.n_samples,
        perturb_std=0.05,
        racket_kwargs=racket_kwargs,
    )
    # trajs: (T, N, 15)
    print(f"  Generated: {trajs.shape}  t ∈ [{tspan[0]:.3f}, {tspan[-1]:.3f}]s")

    # Augment with embedded inertia: (T, N, 15) → (T, N, 18)
    trajs_aug = augment_with_inertia(trajs, I1, I2, I3)

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt_path = args.ckpt_path or find_latest_ckpt()
    print(f"\nLoading checkpoint:\n  {ckpt_path}")

    model = DissipativeSO3HamNODE(
        device=device, u_dim=3, init_gain=0.5, inertia_dim=3).to(device)
    model.M_net = FixedInertiaFromState().to(device).to(float_type)

    state_dict = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  WARNING: unexpected keys in checkpoint: {unexpected}")
    if missing:
        print(f"  INFO: keys not in checkpoint (expected for FixedInertiaFromState): {missing}")
    model.eval()

    # ── Windowed eval (same 5-point window as training) ────────────────────
    x_arr, t_eval = arrange_data(trajs_aug[np.newaxis], tspan,
                                  num_points=args.num_points)
    x_cat  = np.concatenate(x_arr, axis=1)    # (num_points, B_windows, 18)
    x_cat_t = torch.tensor(x_cat,  dtype=float_type)
    t_ev_t  = torch.tensor(t_eval, dtype=float_type)

    with torch.no_grad():
        x_hat = odeint(model, x_cat_t[0], t_ev_t, method=args.solver)
        tgt     = strip_inertia(x_cat_t[1:])
        tgt_hat = strip_inertia(x_hat[1:])
        w_loss, w_l2, w_geo = rotmat_L2_geodesic_loss(tgt, tgt_hat, split=[9, 3, 3])
        subnet = subnet_physics_mse_stageC(model, x_hat)

    print(f"\n--- Windowed eval  ({args.num_points}-point window, "
          f"N={x_cat.shape[1]} windows) ---")
    print(f"  geo={w_geo:.4e}  L2={w_l2:.4e}  total={w_loss:.4e}")
    print(f"  subnet MSE  M={subnet['M_loss']:.3e}  V={subnet['V_loss']:.3e}  "
          f"Dw={subnet['Dw_loss']:.3e}  g={subnet['g_loss']:.3e}")

    # ── Full-trajectory eval ───────────────────────────────────────────────
    x_full_t = torch.tensor(trajs_aug, dtype=float_type)   # (T, N, 18)
    t_full_t = torch.tensor(tspan,     dtype=float_type)

    with torch.no_grad():
        x_full_hat = odeint(model, x_full_t[0], t_full_t, method=args.solver)
        tl, ll, gl = traj_rotmat_L2_geodesic_loss(
            strip_inertia(x_full_t), strip_inertia(x_full_hat), split=[9, 3, 3])

    gl_sum = gl.sum(dim=0)   # (N,) — geo summed over all T timesteps
    print(f"\n--- Full-trajectory eval  (T={args.timesteps}, N={args.n_samples}) ---")
    print(f"  geo = {gl_sum.mean():.4e} ± {gl_sum.std():.4e}")
    print(f"  min = {gl_sum.min():.4e}  max = {gl_sum.max():.4e}")

    # ── Verdict ────────────────────────────────────────────────────────────
    passed = w_geo.item() < 0.01
    print(f"\n{'PASS ✓' if passed else 'FAIL ✗'}  windowed geo = {w_geo:.4e} "
          f"(threshold 0.01 rad²)")

    return {
        'I1': I1, 'I2': I2, 'I3': I3,
        'windowed_geo': w_geo.item(),
        'windowed_M':   subnet['M_loss'],
        'windowed_V':   subnet['V_loss'],
        'windowed_Dw':  subnet['Dw_loss'],
        'full_traj_geo_mean': gl_sum.mean().item(),
        'full_traj_geo_std':  gl_sum.std().item(),
    }


if __name__ == '__main__':
    main()

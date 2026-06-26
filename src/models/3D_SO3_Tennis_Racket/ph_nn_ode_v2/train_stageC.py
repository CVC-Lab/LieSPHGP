"""Stage C training: multi-config generalization with inertia-augmented state.

State extended from 15D to 18D: (vec(R)[9], I₁I₂I₃[3], ω[3], u[3]).
All four geometry configs are trained jointly; the model learns the mapping
(R, I₁,I₂,I₃) → M⁻¹(R,I) = diag(1/I₁,1/I₂,1/I₃).

Key differences from train.py (Stage A):
  - Uses network_stageC.DissipativeSO3HamNODE (inertia_dim=3, rotmatdim=12).
  - Uses FixedInertiaFromState instead of FixedInertia (reads I from state).
  - augment_with_inertia() inserts (I₁,I₂,I₃) into every state vector.
  - All 4 configs are concatenated along the batch axis for joint training.
  - Per-config eval reported separately so generalisation is visible.
  - loss_utils.py is still imported from the pendulum directory (unchanged).

Stage C pass criteria:
  1. eval_M_loss < 1e-4  across all 4 configs (FixedInertiaFromState gives 0)
  2. test_geo_loss < 0.01 rad² for all 4 configs
  3. eval_V_loss, eval_Dw_loss small for all configs
  4. Generalisation: held-out 5th config (unseen geometry) achieves geo < 0.01
"""
import torch, argparse
import numpy as np
import os, sys
import time
import pickle

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))

PENDULUM_ODE_DIR = os.path.join(
    PROJECT_ROOT, 'src/models/3D_SO3_Windy_Pendulum/ph_nn_ode_v2')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/utils'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'datasets'))
sys.path.insert(0, PENDULUM_ODE_DIR)
sys.path.insert(0, THIS_FILE_DIR)   # for network_stageC

from torchdiffeq import odeint

from ode_utils import to_pickle
from subnet_diagnostics_stageC import subnet_physics_mse_stageC
from tennis_racket_3d_datagen import arrange_data
from network_stageC import DissipativeSO3HamNODE, FixedInertiaFromState
from loss_utils import (
    rotmat_L2_geodesic_loss_safe as rotmat_L2_geodesic_loss,
    traj_rotmat_L2_geodesic_loss_safe as traj_rotmat_L2_geodesic_loss,
    power_balance_loss,
    consistency_subnet_losses,
)


DEFAULT_SAVE_DIR = os.path.join(THIS_FILE_DIR, 'data', 'run_tr3d_stageC_fp32')
DEFAULT_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'data/tennis_data/'
    'tr3d_dataset_dist0p0_obs_noise0p0_perturb0p05_ncfg4_steps100.pkl'
)


# FixedInertiaFromState is imported from network_stageC — reads I from state.


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learn_rate',   default=1e-3, type=float)
    parser.add_argument('--total_steps',  default=10000, type=int)
    parser.add_argument('--eval_every',   default=50, type=int,
                        help='windowed eval + diagnostics + checkpoint cadence')
    parser.add_argument('--name',         default='tr3d', type=str)
    parser.add_argument('--verbose',      action='store_true')
    parser.add_argument('--seed',         default=0, type=int)
    parser.add_argument('--save_dir',     default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument('--data_path',    default=DEFAULT_DATA_PATH, type=str,
                        help='path to the tennis-racket .pkl dataset')
    parser.add_argument('--gpu',          type=int, default=0)
    parser.add_argument('--num_points',   type=int, default=5)
    parser.add_argument('--solver',       default='rk4', type=str)
    parser.add_argument('--init_gain',    default=0.5, type=float)

    parser.add_argument('--samples',      type=int, default=25)
    parser.add_argument('--timesteps',    type=int, default=100)
    parser.add_argument('--obs_noise_std',          type=float, default=0.0)
    parser.add_argument('--disturbance_torque_std', type=float, default=0.0)
    parser.add_argument('--perturb_std',            type=float, default=0.05)

    # M_net pretraining: anchors M_net near diag(1/I1,1/I2,1/I3) before joint training.
    parser.add_argument('--pretrain_M_steps',       type=int,   default=200)
    parser.add_argument('--pretrain_M_lr',          type=float, default=1e-3)
    parser.add_argument('--pretrain_M_print_every', type=int,   default=20)

    # Physics-informed auxiliary losses (off by default).
    parser.add_argument('--lambda_power', type=float, default=0.0,
                        help='weight for power-balance loss; 0 disables')
    parser.add_argument('--lambda_V',     type=float, default=0.0,
                        help='weight for V back-solving loss; 0 disables')
    parser.add_argument('--lambda_B',     type=float, default=0.0,
                        help='weight for B (input coupling) back-solving loss; 0 disables')
    parser.add_argument('--lambda_D',     type=float, default=0.0,
                        help='weight for D (dissipation) back-solving loss; 0 disables')

    # Pin M_net to the analytic ground-truth diag(1/I1,1/I2,1/I3).
    parser.add_argument('--fix_M', action='store_true',
                        help='use FixedInertia (non-learnable M_net) with true I from the data')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def augment_with_inertia(x_np, I1, I2, I3):
    """Insert (I1,I2,I3) into every state vector after vec(R).

    Input:  x_np  (T, N, 15)  — [vec(R)(9), ω(3), u(3)]
    Output: x_aug (T, N, 18)  — [vec(R)(9), I1I2I3(3), ω(3), u(3)]
    """
    T, N, _ = x_np.shape
    I_col = np.array([I1, I2, I3], dtype=x_np.dtype)
    I_tile = np.tile(I_col, (T, N, 1))     # (T, N, 3)
    return np.concatenate([x_np[:, :, :9], I_tile, x_np[:, :, 9:]], axis=2)


def strip_inertia(x):
    """Remove the embedded I₁I₂I₃ columns, (*, 18) → (*, 15).

    The 18D Stage C state is [vec(R)(9), I₁I₂I₃(3), ω(3), u(3)].
    Removing positions 9:12 gives the 15D form [vec(R), ω, u] that
    rotmat_L2_geodesic_loss_safe expects with split=[9,3,3].
    Works for any leading dimensions (T, B, …).
    """
    return torch.cat([x[..., :9], x[..., 12:]], dim=-1)


def get_model_parm_nums(model):
    return sum(p.nelement() for p in model.parameters())


def _fmt_num(x):
    """Format a number for filesystem paths: 0.01 → 0p01, -1.0 → n1."""
    s = f"{x:g}"
    return s.replace('.', 'p').replace('-', 'n').replace('+', '')


def build_run_name(args):
    """Build a run-folder name encoding training specs + a YYMMDD-HHMM stamp."""
    parts = [
        f"obs{_fmt_num(args.obs_noise_std)}",
        f"dist{_fmt_num(args.disturbance_torque_std)}",
        "cfgALL",
        f"lP{_fmt_num(args.lambda_power)}",
        f"lV{_fmt_num(args.lambda_V)}",
        f"lB{_fmt_num(args.lambda_B)}",
        f"lD{_fmt_num(args.lambda_D)}",
        f"lr{_fmt_num(args.learn_rate)}",
        f"s{args.total_steps}",
        f"np{args.num_points}",
        f"smp{args.samples}",
        f"T{args.timesteps}",
        f"{args.solver}",
        f"seed{args.seed}",
    ]
    if args.fix_M:
        parts.append('fixM')
    stamp = time.strftime('%y%m%d-%H%M%S')
    return '_'.join(parts) + '_' + stamp


def _state_dict(model):
    return (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict()


def _inner(model):
    """Unwrap a possibly torch.compile-wrapped model."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model


# ─────────────────────────────────────────────────────────────────────────────
# M_net pretraining
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_M_net(model, q_samples, n_steps, lr, print_every):
    """Pretrain M_net toward diag(1/I₁,1/I₂,1/I₃) using per-sample targets.

    q_samples must be (N, 12) = (vec(R), I₁,I₂,I₃).  Inertia values are read
    from columns 9:12 so each sample gets the correct target for its config.
    """
    if n_steps <= 0:
        return

    inner  = _inner(model)
    device = q_samples.device
    dtype  = q_samples.dtype

    q_no_grad = q_samples.detach()
    I_vals    = q_no_grad[:, 9:12]                              # (N, 3)
    target    = torch.diag_embed(1.0 / I_vals).detach()        # (N, 3, 3)

    print(f"\nPretraining M_net for {n_steps} steps  (lr={lr})")
    print(f"  {q_no_grad.shape[0]} q-samples, per-sample targets from embedded inertia")

    optim = torch.optim.Adam(inner.M_net.parameters(), lr=lr, weight_decay=1e-4)

    initial_loss = None
    for step in range(n_steps):
        M_pred = inner.M_net(q_no_grad)
        loss   = (M_pred - target).pow(2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        if initial_loss is None:
            initial_loss = loss.item()
        if step % max(1, print_every) == 0 or step == n_steps - 1:
            print(f"  pretrain step {step:>4d}: loss={loss.item():.3e}")

    with torch.no_grad():
        M_check   = inner.M_net(q_no_grad[:1])
        deviation = (M_check - target[:1]).abs().max().item()
    print(f"  pretrain done.  initial={initial_loss:.3e}  "
          f"final={loss.item():.3e}  max|M(q₀)−target|={deviation:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    float_type = torch.float32
    torch.set_default_dtype(torch.float32)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    run_name   = build_run_name(args)
    args.save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Run dir : {args.save_dir}")

    if args.verbose:
        print(f"Start training (fp32)  num_points={args.num_points}  "
              f"solver={args.solver}  eval_every={args.eval_every}  device={device}")

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading dataset: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    all_inertia = data['inertia_info']
    num_configs = len(all_inertia)
    print(f"\nDataset: {num_configs} geometry configs")
    for ci, info in enumerate(all_inertia):
        I1c, I2c, I3c = info['I1'], info['I2'], info['I3']
        print(f"  cfg{ci}: I1={I1c:.6f}  I2={I2c:.6f}  I3={I3c:.6f}  kg·m²  "
              f"(I2/I1={I2c/I1c:.2f}  asym=(I3-I1)/I2={(I3c-I1c)/I2c:.2f})")

    # Augment each config's data with its embedded inertia: (T,N,15) → (T,N,18)
    x_aug_all = np.stack([
        augment_with_inertia(data['x'][ci],
                             all_inertia[ci]['I1'], all_inertia[ci]['I2'], all_inertia[ci]['I3'])
        for ci in range(num_configs)
    ])   # (num_configs, T, N, 18)

    x_test_aug_all = np.stack([
        augment_with_inertia(data['test_x'][ci],
                             all_inertia[ci]['I1'], all_inertia[ci]['I2'], all_inertia[ci]['I3'])
        for ci in range(num_configs)
    ])   # (num_configs, T, N, 18)

    # ── Build model ───────────────────────────────────────────────────────
    model = DissipativeSO3HamNODE(
        device=device, u_dim=3, init_gain=args.init_gain, inertia_dim=3).to(device)

    if args.fix_M:
        _inner(model).M_net = FixedInertiaFromState().to(device).to(float_type)
        print("M_net fixed to FixedInertiaFromState — reads I₁,I₂,I₃ from state.")

    print(f'Model: {get_model_parm_nums(model)} parameters')

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # ── Arrange data ──────────────────────────────────────────────────────
    # arrange_data slices along time; all 4 config batch dims are concatenated.
    train_x, t_eval = arrange_data(x_aug_all,      data['t'], num_points=args.num_points)
    test_x,  _      = arrange_data(x_test_aug_all, data['t'], num_points=args.num_points)
    train_x_cat = np.concatenate(train_x, axis=1)   # (num_points, B_all, 18)
    test_x_cat  = np.concatenate(test_x,  axis=1)   # (num_points, B_all, 18)

    train_x_cat = torch.tensor(train_x_cat, requires_grad=True,
                                dtype=float_type).to(device)
    test_x_cat  = torch.tensor(test_x_cat,  requires_grad=True,
                                dtype=float_type).to(device)
    t_eval      = torch.tensor(t_eval, requires_grad=True,
                                dtype=float_type).to(device)

    # ── M_net pretraining ─────────────────────────────────────────────────
    if args.pretrain_M_steps > 0 and not args.fix_M:
        q_pretrain = train_x_cat.detach().reshape(-1, 18)[:, :12]   # (N, 12)
        pretrain_M_net(
            model=model,
            q_samples=q_pretrain,
            n_steps=args.pretrain_M_steps,
            lr=args.pretrain_M_lr,
            print_every=args.pretrain_M_print_every,
        )

    split = [9, 3, 3]

    stats = {
        'train_loss': [], 'train_l2_loss': [], 'train_geo_loss': [],
        'train_power_loss': [],
        'train_V_cons_loss': [], 'train_B_cons_loss': [], 'train_D_cons_loss': [],
        'forward_time': [], 'backward_time': [], 'nfe': [],
        # Eval (windowed) — recorded every eval_every steps
        'eval_step':  [],
        'test_loss':  [], 'test_l2_loss': [], 'test_geo_loss': [],
        'eval_M_loss': [], 'eval_V_loss': [],
        'eval_Dw_loss': [], 'eval_g_loss': [],
        # Config metadata saved alongside training stats
        'num_configs': num_configs,
        'all_inertia': all_inertia,
    }

    dt_train = (t_eval[1] - t_eval[0]).detach().item()

    os.makedirs(args.save_dir, exist_ok=True)
    label      = '-so3ham'
    stats_path = (f'{args.save_dir}/{args.name}{label}'
                  f'-{args.solver}-{args.num_points}p-stats.pkl')

    loss_buffer = []
    fwd_buffer  = []
    bwd_buffer  = []
    nfe_buffer  = []

    for step in range(args.total_steps + 1):

        # ── Training step ────────────────────────────────────────────────
        t = time.time()
        train_x_hat = odeint(model, train_x_cat[0, :, :], t_eval, method=args.solver)
        forward_time = time.time() - t

        target     = strip_inertia(train_x_cat[1:, :, :])
        target_hat = strip_inertia(train_x_hat[1:, :, :])
        train_loss, train_l2_loss, train_geo_loss = rotmat_L2_geodesic_loss(
            target, target_hat, split=split)

        if args.lambda_power > 0.0:
            L_power = power_balance_loss(model, train_x_cat, dt_train)
        else:
            L_power = torch.zeros((), device=device, dtype=float_type)

        if args.lambda_V > 0.0 or args.lambda_B > 0.0 or args.lambda_D > 0.0:
            L_V, L_B, L_D = consistency_subnet_losses(model, train_x_cat, dt_train)
        else:
            L_V = torch.zeros((), device=device, dtype=float_type)
            L_B = torch.zeros((), device=device, dtype=float_type)
            L_D = torch.zeros((), device=device, dtype=float_type)

        total_loss = (train_loss
                      + args.lambda_power * L_power
                      + args.lambda_V     * L_V
                      + args.lambda_B     * L_B
                      + args.lambda_D     * L_D)

        t = time.time()
        total_loss.backward()
        optim.step()
        optim.zero_grad()
        backward_time = time.time() - t

        loss_buffer.append(torch.stack([
            total_loss.detach(), train_l2_loss.detach(), train_geo_loss.detach(),
            L_power.detach(), L_V.detach(), L_B.detach(), L_D.detach()
        ]))
        fwd_buffer.append(forward_time)
        bwd_buffer.append(backward_time)
        nfe = getattr(model, 'nfe',
                      getattr(getattr(model, '_orig_mod', model), 'nfe', 0))
        nfe_buffer.append(nfe)

        # ── Windowed eval + diagnostics + checkpoint ─────────────────────
        if step % args.eval_every == 0:
            with torch.no_grad():
                test_x_hat = odeint(
                    model, test_x_cat[0, :, :], t_eval, method=args.solver)
                tgt     = strip_inertia(test_x_cat[1:, :, :])
                tgt_hat = strip_inertia(test_x_hat[1:, :, :])
                test_loss, test_l2_loss, test_geo_loss = rotmat_L2_geodesic_loss(
                    tgt, tgt_hat, split=split)
                subnet = subnet_physics_mse_stageC(model, test_x_hat)

            if loss_buffer:
                drained = torch.stack(loss_buffer, dim=0).cpu().numpy()  # (N, 7)
                stats['train_loss'].extend(drained[:, 0].tolist())
                stats['train_l2_loss'].extend(drained[:, 1].tolist())
                stats['train_geo_loss'].extend(drained[:, 2].tolist())
                stats['train_power_loss'].extend(drained[:, 3].tolist())
                stats['train_V_cons_loss'].extend(drained[:, 4].tolist())
                stats['train_B_cons_loss'].extend(drained[:, 5].tolist())
                stats['train_D_cons_loss'].extend(drained[:, 6].tolist())
                stats['forward_time'].extend(fwd_buffer)
                stats['backward_time'].extend(bwd_buffer)
                stats['nfe'].extend(nfe_buffer)
                loss_buffer = []; fwd_buffer = []; bwd_buffer = []; nfe_buffer = []

            test_pack = torch.stack([
                test_loss.detach(), test_l2_loss.detach(), test_geo_loss.detach()
            ]).cpu().numpy()
            train_total = stats['train_loss'][-1]
            train_l2    = stats['train_l2_loss'][-1]
            train_geo   = stats['train_geo_loss'][-1]

            stats['eval_step'].append(step)
            stats['test_loss'].append(float(test_pack[0]))
            stats['test_l2_loss'].append(float(test_pack[1]))
            stats['test_geo_loss'].append(float(test_pack[2]))
            stats['eval_M_loss'].append(subnet['M_loss'])
            stats['eval_V_loss'].append(subnet['V_loss'])
            stats['eval_Dw_loss'].append(subnet['Dw_loss'])
            stats['eval_g_loss'].append(subnet['g_loss'])

            train_pow = stats['train_power_loss'][-1]
            train_LV  = stats['train_V_cons_loss'][-1]
            train_LB  = stats['train_B_cons_loss'][-1]
            train_LD  = stats['train_D_cons_loss'][-1]

            print(f"[step {step:>6d}]")
            print(f"  train: total={train_total:.4e}  "
                  f"L2={train_l2:.4e}  geo={train_geo:.4e}  "
                  f"power={train_pow:.4e}")
            print(f"  cons : L_V={train_LV:.4e}  "
                  f"L_B={train_LB:.4e}  L_D={train_LD:.4e}")
            print(f"  test : total={test_pack[0]:.4e}  "
                  f"L2={test_pack[1]:.4e}  geo={test_pack[2]:.4e}")
            print(f"  subnet MSE  M={subnet['M_loss']:.3e}  "
                  f"V={subnet['V_loss']:.3e}  Dw={subnet['Dw_loss']:.3e}  "
                  f"g={subnet['g_loss']:.3e}  | nfe={nfe}")

            ckpt = (f'{args.save_dir}/{args.name}{label}'
                    f'-{args.solver}-{args.num_points}p-{step}.tar')
            torch.save(_state_dict(model), ckpt)
            to_pickle(stats, stats_path)

    # ── Final per-config per-trajectory eval ─────────────────────────────
    t_full = torch.tensor(data['t'], requires_grad=True,
                           dtype=float_type).to(device)

    all_train_x_hat, all_test_x_hat = [], []
    all_train_geo,   all_test_geo   = [], []

    print("\n=== Final trajectory eval (all configs) ===")
    for ci in range(num_configs):
        I = all_inertia[ci]
        I1c, I2c, I3c = I['I1'], I['I2'], I['I3']

        x_aug  = augment_with_inertia(data['x'][ci],      I1c, I2c, I3c)  # (T, N, 18)
        xt_aug = augment_with_inertia(data['test_x'][ci], I1c, I2c, I3c)  # (T, N, 18)

        cfg_x_t  = torch.tensor(x_aug,  requires_grad=True,
                                 dtype=float_type).to(device)   # (T, N, 18)
        cfg_xt_t = torch.tensor(xt_aug, requires_grad=True,
                                 dtype=float_type).to(device)   # (T, N, 18)

        with torch.no_grad():
            tr_hat = odeint(model, cfg_x_t[0],  t_full, method=args.solver)
            te_hat = odeint(model, cfg_xt_t[0], t_full, method=args.solver)

            tr_loss, tr_l2, tr_geo = traj_rotmat_L2_geodesic_loss(
                strip_inertia(cfg_x_t), strip_inertia(tr_hat), split=split)
            te_loss, te_l2, te_geo = traj_rotmat_L2_geodesic_loss(
                strip_inertia(cfg_xt_t), strip_inertia(te_hat), split=split)

        def _mean_std(t):
            v = t.detach()
            return v.mean().item(), v.std().item()

        tr_geo_sum = tr_geo.sum(dim=0)
        te_geo_sum = te_geo.sum(dim=0)

        print(f"  cfg{ci} (I1={I1c:.4f} I2={I2c:.4f} I3={I3c:.4f}):")
        print(f"    train geo {_mean_std(tr_geo_sum)[0]:.4e} ± {_mean_std(tr_geo_sum)[1]:.4e}")
        print(f"    test  geo {_mean_std(te_geo_sum)[0]:.4e} ± {_mean_std(te_geo_sum)[1]:.4e}")

        all_train_x_hat.append(tr_hat.detach().cpu().numpy())
        all_test_x_hat.append(te_hat.detach().cpu().numpy())
        all_train_geo.append(tr_geo_sum.detach().cpu().numpy())
        all_test_geo.append(te_geo_sum.detach().cpu().numpy())

    all_train_geo_np = np.concatenate(all_train_geo)
    all_test_geo_np  = np.concatenate(all_test_geo)
    print(f"\nAll configs — train geo {all_train_geo_np.mean():.4e} ± {all_train_geo_np.std():.4e}")
    print(f"All configs — test  geo {all_test_geo_np.mean():.4e} ± {all_test_geo_np.std():.4e}")

    stats['traj_train_geo'] = all_train_geo_np
    stats['traj_test_geo']  = all_test_geo_np
    stats['train_x_hat']    = all_train_x_hat
    stats['test_x_hat']     = all_test_x_hat
    stats['t_eval']         = t_full.detach().cpu().numpy()
    return model, stats


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    os.makedirs(args.save_dir, exist_ok=True)
    label      = '-so3ham'
    final_ckpt = (f'{args.save_dir}/{args.name}{label}'
                  f'-{args.solver}-{args.num_points}p.tar')
    torch.save(_state_dict(model), final_ckpt)
    final_stats = (f'{args.save_dir}/{args.name}{label}'
                   f'-{args.solver}-{args.num_points}p-stats.pkl')
    print("Saved final stats:", final_stats)
    to_pickle(stats, final_stats)

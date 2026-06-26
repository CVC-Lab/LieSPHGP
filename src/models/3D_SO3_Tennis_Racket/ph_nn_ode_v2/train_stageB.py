"""Stage B training: viscous friction data, learn Dw_net → friction_coeff · I₃.

Adapted from train.py. Single change from Stage A:
  - --data_path defaults to the friction dataset (fric0p01).
  - --friction_coeff is read and passed to subnet_physics_mse_tennis so the
    Dw diagnostic target is friction_coeff · I₃ instead of 0.
  - --fix_M still recommended (M is pinned to ground truth, same as Stage A).

Stage B pass criteria:
  1. eval_M_loss  = 0   (pinned via --fix_M)
  2. test_geo_loss < 0.01 rad²
  3. eval_Dw_loss converges toward (friction_coeff² / 3) — the MSE of a
     perfectly-learned scalar-times-identity dissipation matrix goes to 0,
     so eval_Dw_loss should be much smaller than friction_coeff²
  4. det(R̂) ≈ 1 throughout
"""
import torch, argparse
import numpy as np
import os, sys
import time
import pickle

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))

# Shared network.py and loss_utils.py live in the pendulum model directory.
# The architecture is identical; we reuse without copying.
PENDULUM_ODE_DIR = os.path.join(
    PROJECT_ROOT, 'src/models/3D_SO3_Windy_Pendulum/ph_nn_ode_v2')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/utils'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'datasets'))
sys.path.insert(0, PENDULUM_ODE_DIR)

from torchdiffeq import odeint

from ode_utils import to_pickle
from subnet_diagnostics_tennis import subnet_physics_mse_tennis
from tennis_racket_3d_datagen import arrange_data
from network import DissipativeSO3HamNODE
from loss_utils import (
    rotmat_L2_geodesic_loss_safe as rotmat_L2_geodesic_loss,
    traj_rotmat_L2_geodesic_loss_safe as traj_rotmat_L2_geodesic_loss,
    power_balance_loss,
    consistency_subnet_losses,
)


DEFAULT_SAVE_DIR = os.path.join(THIS_FILE_DIR, 'data', 'run_tr3d_stageB_fp32')
DEFAULT_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'data/tennis_data/'
    'tr3d_dataset_dist0p0_obs_noise0p0_perturb0p05_fric0p01_ncfg4_steps100.pkl'
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed inverse-inertia module (drop-in M_net for Stage A sanity checks)
# ─────────────────────────────────────────────────────────────────────────────

class FixedInertia(torch.nn.Module):
    """Drop-in M_net that returns diag(1/I1, 1/I2, 1/I3) — no learnable parameters.

    M_net outputs M⁻¹ (the inverse inertia tensor).  For the tennis racket this
    is diag(1/I1, 1/I2, 1/I3).  Use --fix_M to pin M_net here and let the
    optimizer focus on V_net, Dw_net, g_net.
    """
    def __init__(self, I1: float, I2: float, I3: float):
        super().__init__()
        I_inv = torch.tensor([1.0/I1, 1.0/I2, 1.0/I3], dtype=torch.float32)
        self.register_buffer('M_inv', torch.diag(I_inv))

    def forward(self, q):
        return self.M_inv.unsqueeze(0).expand(q.shape[0], 3, 3)


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

    # Which geometry config in the dataset to train on (0-indexed).
    parser.add_argument('--config_idx', type=int, default=0,
                        help='index into data[inertia_info] to select (0-3 for the default 4-config dataset)')

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

    # Friction coefficient — must match the dataset used.
    # Passed to subnet_physics_mse_tennis so Dw diagnostic target = friction_coeff·I₃.
    parser.add_argument('--friction_coeff', type=float, default=0.01,
                        help='viscous friction coefficient used in data generation')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

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
        f"cfg{args.config_idx}",
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

def pretrain_M_net(model, q_samples, n_steps, lr, print_every, I1, I2, I3):
    """Pretrain M_net to output the true inverse inertia diag(1/I1, 1/I2, 1/I3).

    For a free rigid body the inertia tensor is constant (independent of R), so
    M_net needs to learn a constant diagonal PSD matrix.  Pretraining anchors it
    near the correct value before joint training begins.
    """
    if n_steps <= 0:
        return

    inner  = _inner(model)
    device = q_samples.device
    dtype  = q_samples.dtype

    I_inv_diag = torch.tensor([1.0/I1, 1.0/I2, 1.0/I3], dtype=dtype, device=device)
    target_mat = torch.diag(I_inv_diag)
    target     = target_mat.unsqueeze(0).expand(q_samples.shape[0], 3, 3)

    print(f"\nPretraining M_net for {n_steps} steps  (lr={lr})")
    print(f"  target: diag(1/I1,1/I2,1/I3) = "
          f"diag({1/I1:.4f}, {1/I2:.4f}, {1/I3:.4f})")
    print(f"  {q_samples.shape[0]} q-samples drawn from training data")

    optim = torch.optim.Adam(inner.M_net.parameters(), lr=lr, weight_decay=1e-4)
    q_no_grad = q_samples.detach()

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
        deviation = (M_check - target_mat).abs().max().item()
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

    inertia = data['inertia_info'][args.config_idx]
    I1, I2, I3 = inertia['I1'], inertia['I2'], inertia['I3']
    print(f"\nConfig {args.config_idx}:  I1={I1:.6f}  I2={I2:.6f}  I3={I3:.6f}  kg·m²")
    print(f"  I2/I1={I2/I1:.2f}  I3/I1={I3/I1:.2f}  "
          f"(I3-I1)/I2={(I3-I1)/I2:.2f}  (asymmetry index for Dzhanibekov strength)")

    # Slice to single config: (num_configs, T, N, 15) → (1, T, N, 15)
    cfg_x    = data['x'][[args.config_idx]]
    cfg_test = data['test_x'][[args.config_idx]]

    # ── Build model ───────────────────────────────────────────────────────
    model = DissipativeSO3HamNODE(
        device=device, u_dim=3, init_gain=args.init_gain).to(device)

    if args.fix_M:
        _inner(model).M_net = FixedInertia(
            I1=I1, I2=I2, I3=I3).to(device).to(float_type)
        print(f"M_net fixed to diag(1/I1,1/I2,1/I3) — not trained.")

    print(f'Model: {get_model_parm_nums(model)} parameters')

    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # ── Arrange data ──────────────────────────────────────────────────────
    train_x, t_eval = arrange_data(cfg_x,    data['t'], num_points=args.num_points)
    test_x,  _      = arrange_data(cfg_test, data['t'], num_points=args.num_points)
    train_x_cat = np.concatenate(train_x, axis=1)
    test_x_cat  = np.concatenate(test_x,  axis=1)

    train_x_cat = torch.tensor(train_x_cat, requires_grad=True,
                                dtype=float_type).to(device)
    test_x_cat  = torch.tensor(test_x_cat,  requires_grad=True,
                                dtype=float_type).to(device)
    t_eval      = torch.tensor(t_eval, requires_grad=True,
                                dtype=float_type).to(device)

    # ── M_net pretraining ─────────────────────────────────────────────────
    if args.pretrain_M_steps > 0 and not args.fix_M:
        q_pretrain = train_x_cat.detach().reshape(-1, 15)[:, :9]
        pretrain_M_net(
            model=model,
            q_samples=q_pretrain,
            n_steps=args.pretrain_M_steps,
            lr=args.pretrain_M_lr,
            print_every=args.pretrain_M_print_every,
            I1=I1, I2=I2, I3=I3,
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
        'config_idx': args.config_idx,
        'I1': I1, 'I2': I2, 'I3': I3,
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

        target     = train_x_cat[1:, :, :]
        target_hat = train_x_hat[1:, :, :]
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
                tgt     = test_x_cat[1:, :, :]
                tgt_hat = test_x_hat[1:, :, :]
                test_loss, test_l2_loss, test_geo_loss = rotmat_L2_geodesic_loss(
                    tgt, tgt_hat, split=split)
                subnet = subnet_physics_mse_tennis(
                    model, test_x_hat, I1=I1, I2=I2, I3=I3,
                    friction_coeff=args.friction_coeff)

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

    # ── Final per-trajectory eval ─────────────────────────────────────────
    cfg_x_full    = torch.tensor(cfg_x,    requires_grad=True,
                                  dtype=float_type).to(device)
    cfg_test_full = torch.tensor(cfg_test, requires_grad=True,
                                  dtype=float_type).to(device)
    t_full = torch.tensor(data['t'], requires_grad=True,
                           dtype=float_type).to(device)

    train_loss_l, test_loss_l = [], []
    train_l2_l,   test_l2_l  = [], []
    train_geo_l,  test_geo_l  = [], []
    train_data_hat, test_data_hat = [], []

    for i in range(cfg_x_full.shape[0]):   # single iteration for Stage A
        train_x_hat = odeint(
            model, cfg_x_full[i, 0, :, :], t_full, method=args.solver)
        total_loss, l2_loss, geo_loss = traj_rotmat_L2_geodesic_loss(
            cfg_x_full[i, :, :, :], train_x_hat, split=split)
        train_loss_l.append(total_loss)
        train_l2_l.append(l2_loss)
        train_geo_l.append(geo_loss)
        train_data_hat.append(train_x_hat.detach().cpu().numpy())

        test_x_hat = odeint(
            model, cfg_test_full[i, 0, :, :], t_full, method=args.solver)
        total_loss, l2_loss, geo_loss = traj_rotmat_L2_geodesic_loss(
            cfg_test_full[i, :, :, :], test_x_hat, split=split)
        test_loss_l.append(total_loss)
        test_l2_l.append(l2_loss)
        test_geo_l.append(geo_loss)
        test_data_hat.append(test_x_hat.detach().cpu().numpy())

    def _per_traj(loss_list):
        return torch.sum(torch.cat(loss_list, dim=1), dim=0)

    train_loss_pt = _per_traj(train_loss_l)
    test_loss_pt  = _per_traj(test_loss_l)
    train_l2_pt   = _per_traj(train_l2_l)
    test_l2_pt    = _per_traj(test_l2_l)
    train_geo_pt  = _per_traj(train_geo_l)
    test_geo_pt   = _per_traj(test_geo_l)

    print('Final trajectory train loss {:.4e} +/- {:.4e}\n'
          'Final trajectory test loss  {:.4e} +/- {:.4e}'.format(
              train_loss_pt.mean().item(), train_loss_pt.std().item(),
              test_loss_pt.mean().item(),  test_loss_pt.std().item()))
    print('Final trajectory train geo  {:.4e} +/- {:.4e}\n'
          'Final trajectory test geo   {:.4e} +/- {:.4e}'.format(
              train_geo_pt.mean().item(), train_geo_pt.std().item(),
              test_geo_pt.mean().item(),  test_geo_pt.std().item()))

    stats['traj_train_loss'] = train_loss_pt.detach().cpu().numpy()
    stats['traj_test_loss']  = test_loss_pt.detach().cpu().numpy()
    stats['train_x']         = cfg_x_full.detach().cpu().numpy()
    stats['test_x']          = cfg_test_full.detach().cpu().numpy()
    stats['train_x_hat']     = np.array(train_data_hat)
    stats['test_x_hat']      = np.array(test_data_hat)
    stats['t_eval']          = t_full.detach().cpu().numpy()
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

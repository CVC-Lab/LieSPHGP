r"""Visualize the $n$-link windy arm on $SO(3)^n$.

Drives :class:`envs.arm_nlink_SO3.windy_arm_nlink_so3.windy_arm_nlink_so3` for one rollout and
produces two artefacts:

**(A) MP4 animation** — the chain drawn joint-to-joint via the environment's own
`render()`, so what you see is exactly the simulated state (no re-derivation).

**(B) Diagnostics PNG** — four panels over the same rollout:

    1. $H(t)$                          energy: flat when conservative,
                                       monotone down when damped
    2. $J_z(t) = e_z^\top\sum_i R_ip_i$ the Noether charge of the residual
                                       $SO(2)$ symmetry about $e_z$
    3. $\lVert\omega_i(t)\rVert$        per-link body rate
    4. $\lVert R_i^\top R_i - I\rVert_F$ manifold defect (log axis; should sit
                                       at machine precision ~1e-15)

Outputs go to ``videos/windy_arm_nlink/`` named ``mm-dd-hh-mm-ss-<name>.<ext>``.

Examples
--------
    # conservative 2-link arm: H and J_z should both be flat
    python envs/arm_nlink_SO3/render_arm_nlink_video.py --n 2 --friction_coeff 0.0

    # damped 3-link arm under stochastic wind, random torques
    python envs/arm_nlink_SO3/render_arm_nlink_video.py --n 3 --friction_coeff 0.5 \
        --wind_force_std 0.5 --random_u --random_u_scale 1.0

    # diagnostics only (skips the ~30 s MP4 encode)
    python envs/arm_nlink_SO3/render_arm_nlink_video.py --n 2 --no_video
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.animation import FuncAnimation        # noqa: E402

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.arm_nlink_SO3.windy_arm_nlink_so3 import windy_arm_nlink_so3    # noqa: E402


OUT_DIR = os.path.join(PROJECT_ROOT, 'videos', 'windy_arm_nlink')


# ══════════════════════════════════════════════════════════════════════
# Output naming
# ══════════════════════════════════════════════════════════════════════

def make_stamp() -> str:
    """Timestamp prefix ``mm-dd-hh-mm-ss``, fixed once per run.

    Computed a single time in :func:`main` so the MP4 and the PNG of the same
    rollout share a prefix and sort together in the directory listing.
    """
    return time.strftime('%m-%d-%H-%M-%S')


def out_path(stamp: str, name: str, ext: str) -> str:
    """``videos/windy_arm_nlink/<stamp>-<name>.<ext>`` (directory auto-created)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, f'{stamp}-{name}.{ext}')


# ══════════════════════════════════════════════════════════════════════
# Rollout
# ══════════════════════════════════════════════════════════════════════

def rollout(env, steps, rng, random_u=False, random_u_scale=1.0,
            capture_frames=True, print_every=50):
    r"""Run `steps` environment steps, recording frames and diagnostics.

    Returns a dict of `(steps+1,)`-length arrays (per-link quantities have shape
    `(steps+1, n)`), plus the RGB frames if `capture_frames`.

    Torques are drawn i.i.d. per step when `random_u`, giving the persistent
    excitation that makes the trajectory interesting to look at (and that
    identification needs — §18.3 of `multi-joint-ph-system.md`).
    """
    n = env.n
    frames = []
    t, H, Jz, defect = [], [], [], []
    omega_norm = []

    def record():
        t.append(env.t)
        H.append(env.energy())
        Jz.append(env.vertical_momentum())
        defect.append(env.manifold_defect()[0])
        omega_norm.append(np.linalg.norm(env.omega, axis=-1))
        if capture_frames:
            f = env.render()
            if f is not None:
                frames.append(f)

    record()
    for k in range(steps):
        u = (rng.uniform(-random_u_scale, random_u_scale, size=(n, 3))
             if random_u else np.zeros((n, 3)))
        env.step(u)
        record()

        if print_every and k % print_every == 0:
            print(f'  step {k:4d}/{steps}   H={H[-1]:+.5f}   '
                  f'Jz={Jz[-1]:+.5f}   |omega|={np.linalg.norm(env.omega):.3f}   '
                  f'defect={defect[-1]:.2e}')

    return {
        't': np.array(t),
        'H': np.array(H),
        'Jz': np.array(Jz),
        'defect': np.array(defect),
        'omega_norm': np.array(omega_norm),          # (steps+1, n)
        'frames': frames,
    }


# ══════════════════════════════════════════════════════════════════════
# (A) MP4
# ══════════════════════════════════════════════════════════════════════

def save_video(frames, path, fps=30):
    """Encode captured RGB frames to MP4 (ffmpeg), falling back to GIF."""
    if not frames:
        print('No frames captured — skipping video.')
        return None

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis('off')
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / fps, blit=True)
    try:
        anim.save(path, writer='ffmpeg', fps=fps, dpi=100)
    except Exception as exc:                                  # noqa: BLE001
        gif_path = os.path.splitext(path)[0] + '.gif'
        print(f'ffmpeg unavailable ({exc}); writing GIF instead.')
        anim.save(gif_path, writer='pillow', fps=fps)
        path = gif_path
    finally:
        plt.close(fig)

    print(f'Saved video      : {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# (B) Diagnostics figure
# ══════════════════════════════════════════════════════════════════════

def save_diagnostics(log, env, path, title_suffix=''):
    r"""Four-panel diagnostics figure for the rollout.

    Energy and $J_z$ are plotted as *relative* deviations from their initial
    values, so a conservative run reads as a flat line near zero at the $O(h^2)$
    truncation scale rather than as an arbitrary offset.
    """
    n = env.n
    t = log['t']
    H, Jz = log['H'], log['Jz']

    H_scale = max(abs(H[0]), 1e-9)
    Jz_scale = max(abs(Jz[0]), 1e-3)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # ── 1. Energy ──
    ax = axes[0, 0]
    ax.plot(t, H, color='#c0392b', linewidth=1.6)
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$H = \frac{1}{2}\omega^\top M(q)\omega + V(q)$')
    ax.set_title(f'Energy   (drift {abs(H[-1] - H[0]) / H_scale:.2e} rel.)')
    ax.grid(True, alpha=0.3)

    # ── 2. Vertical angular momentum ──
    ax = axes[0, 1]
    ax.plot(t, Jz, color='#2c7fb8', linewidth=1.6)
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$J_z = e_z^\top \sum_i R_i p_i$')
    ax.set_title(f'Vertical angular momentum   '
                 f'(drift {abs(Jz[-1] - Jz[0]) / Jz_scale:.2e} rel.)')
    ax.grid(True, alpha=0.3)

    # ── 3. Per-link body rates ──
    ax = axes[1, 0]
    for i in range(n):
        ax.plot(t, log['omega_norm'][:, i], linewidth=1.3, label=f'link {i + 1}')
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$\|\omega_i\|$  [rad/s]')
    ax.set_title('Per-link angular rate')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── 4. Manifold defect ──
    ax = axes[1, 1]
    ax.semilogy(t, np.maximum(log['defect'], 1e-18), color='#27ae60', linewidth=1.3)
    ax.axhline(1e-15, color='#999999', linestyle='dashed', linewidth=1.0,
               label='machine precision')
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$\max_i \|R_i^\top R_i - I\|_F$')
    ax.set_title('SO(3) constraint defect (exp-map integrator, no re-projection)')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8)

    fig.suptitle(f'{n}-link windy arm on SO(3)^{n}{title_suffix}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f'Saved diagnostics: {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description='Render and diagnose one n-link windy arm rollout.')
    p.add_argument('--n', type=int, default=2, help='number of links / ball joints')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--friction_coeff', type=float, default=0.2,
                   help='joint friction d_i')
    p.add_argument('--varying_friction', action='store_true')
    p.add_argument('--air_drag', type=float, default=0.0, help='kappa_i')

    p.add_argument('--wind_force_std', type=float, default=0.0,
                   help='sigma of the stochastic wind (the dW channel)')
    p.add_argument('--external_force_std', type=float, default=0.0,
                   help='deterministic wind amplitude')
    p.add_argument('--external_force_type', type=str, default='sine',
                   choices=['sine', 'square', 'random', 'constant'])

    p.add_argument('--random_u', action='store_true',
                   help='i.i.d. joint torques each step (otherwise u = 0)')
    p.add_argument('--random_u_scale', type=float, default=1.0)

    p.add_argument('--link_length', type=float, default=1.0)
    p.add_argument('--name', type=str, default=None,
                   help='filename stem; defaults to arm<n>link')
    p.add_argument('--no_video', action='store_true',
                   help='diagnostics only (skips frame capture and encoding)')
    return p.parse_args()


def main():
    args = get_args()
    stamp = make_stamp()
    name = args.name or f'arm{args.n}link'

    env = windy_arm_nlink_so3(
        n=args.n,
        render_mode=None if args.no_video else 'rgb_array',
        dt=args.dt,
        link_length=args.link_length,
        friction_coeff=args.friction_coeff,
        varying_friction=args.varying_friction,
        air_drag=args.air_drag,
        external_force_type=args.external_force_type,
        external_force_std=args.external_force_std,
        wind_force_std=args.wind_force_std,
        seed=args.seed,
    )
    env.reset(seed=args.seed)

    print(f'{args.n}-link arm  |  d={args.friction_coeff}  '
          f'kappa={args.air_drag}  sigma_wind={args.wind_force_std}  '
          f'u={"random" if args.random_u else "0"}  seed={args.seed}')
    print(f'  obs dim = {env.observation_space.shape[0]}   '
          f'act dim = {env.action_space.shape[0]}')

    log = rollout(env, args.steps, np.random.default_rng(args.seed),
                  random_u=args.random_u, random_u_scale=args.random_u_scale,
                  capture_frames=not args.no_video)
    env.close()

    suffix = (f'   (d={args.friction_coeff}, '
              f'$\\sigma_{{wind}}$={args.wind_force_std}, '
              f'u={"random" if args.random_u else "0"})')
    save_diagnostics(log, env, out_path(stamp, name, 'png'), title_suffix=suffix)

    if not args.no_video:
        save_video(log['frames'], out_path(stamp, name, 'mp4'), fps=args.fps)

    print(f'Output directory : {OUT_DIR}')


if __name__ == '__main__':
    main()

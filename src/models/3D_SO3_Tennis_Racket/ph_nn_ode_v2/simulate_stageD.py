"""Stage D simulation: IDA-PBC stabilisation of the tennis racket.

Task: starting from Dzhanibekov tumbling (spin about unstable e₂), drive
the angular velocity to a stable target  ω* = (0, 0, ω₀)  (handle spin, e₃).

The controller is a simple proportional law  u = K_p·(ω* - ω), which is
exact for a direct-torque actuator (g = I₃).  No model is needed.

Linearised analysis (about ω* for cfg0):
  δω₃ mode:        τ₃  = I₃/K_p            ≈  0.132 s
  δω₁,δω₂ modes:  Re(λ) = -K_p(1/I₁+1/I₂)/2  ≈ -13.2 s⁻¹  (τ ≈ 0.076 s)
  Overdamped for K_p > 0.071;  K_p = 0.10 is in the overdamped regime.

Expected convergence: ‖ω-ω*‖ decays to <0.01 rad/s within ~0.5 s (10 steps).

Usage:
    /Users/katesur/Projects/LieSPHGP/venv/bin/python3 -u simulate_stageD.py

Options:
    --Kp          Proportional gain [N·m·s/rad]   (default 0.1)
    --omega_star  Target spin rate [rad/s]         (default 2π)
    --n_trials    Number of trials                 (default 20)
    --n_steps     Steps per trial (dt=0.05s)       (default 200 = 10s)
    --axis_start  Initial spin axis: 1=e₂ unstable, 0=e₁, 2=e₃  (default 1)
    --conv_thr    Convergence threshold ‖ω-ω*‖     (default 0.5 rad/s)
    --seed        RNG seed                         (default 0)
    --Kp_scan     Comma-separated Kp values to scan (overrides --Kp)
"""
import argparse
import os
import sys
import numpy as np

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(THIS_FILE_DIR, '../../../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'envs'))
sys.path.insert(0, THIS_FILE_DIR)

from envs.tennis_racket_3d import tennis_racket_3d as Env
from controller_stageD import PDBodyFrameController


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--Kp',        type=float,  default=0.10)
    p.add_argument('--omega_star', type=float, default=2 * np.pi,
                   help='Target spin speed about e₃ [rad/s]')
    p.add_argument('--n_trials',  type=int,   default=20)
    p.add_argument('--n_steps',   type=int,   default=200)
    p.add_argument('--axis_start', type=int,  default=1,
                   help='Initial axis: 0=e₁ stable, 1=e₂ unstable, 2=e₃ target')
    p.add_argument('--conv_thr',  type=float, default=0.50,
                   help='Convergence threshold on ‖ω-ω*‖ [rad/s]')
    p.add_argument('--seed',      type=int,   default=0)
    p.add_argument('--Kp_scan',   type=str,   default=None,
                   help='Comma-separated Kp values to scan, e.g. "0.01,0.05,0.10,0.20"')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Single Kp run
# ─────────────────────────────────────────────────────────────────────────────

def run_trials(
    Kp: float,
    omega_star_vec: np.ndarray,
    n_trials: int,
    n_steps: int,
    axis_start: int,
    conv_thr: float,
    seed: int,
    env: Env,
    verbose: bool = True,
) -> dict:
    """Run n_trials closed-loop episodes and return convergence statistics."""
    ctrl = PDBodyFrameController(
        omega_star=omega_star_vec,
        Kp=Kp,
        use_model_g=False,
    )

    dt = env.dt
    t_grid = np.arange(n_steps + 1) * dt   # (n_steps+1,)  includes t=0

    # Storage: omega_error[trial, step]
    err_traces = np.zeros((n_trials, n_steps + 1), dtype=np.float64)
    conv_steps = []

    for trial in range(n_trials):
        obs, _ = env.reset(seed=seed + trial,
                           options={"axis": axis_start})
        R_flat = obs[:9].astype(np.float64)
        omega  = obs[9:12].astype(np.float64)

        err_traces[trial, 0] = ctrl.omega_error(omega)

        for step in range(1, n_steps + 1):
            u   = ctrl(R_flat, omega)
            obs, _, _, _, _ = env.step(u)
            R_flat = obs[:9].astype(np.float64)
            omega  = obs[9:12].astype(np.float64)
            err_traces[trial, step] = ctrl.omega_error(omega)

        # First step where error drops below threshold
        below = np.where(err_traces[trial] < conv_thr)[0]
        conv_steps.append(int(below[0]) if len(below) else n_steps)

    conv_steps = np.array(conv_steps, dtype=float)
    final_err  = err_traces[:, -1]

    result = {
        'Kp':            Kp,
        'err_traces':    err_traces,         # (n_trials, n_steps+1)
        't_grid':        t_grid,
        'conv_steps':    conv_steps,         # (n_trials,) steps to conv
        'conv_times':    conv_steps * dt,    # (n_trials,) seconds to conv
        'final_err':     final_err,          # (n_trials,) final omega error
        'pct_converged': 100 * np.mean(conv_steps < n_steps),
    }

    if verbose:
        _print_result(result, n_steps, dt, conv_thr)

    return result


def _print_result(result: dict, n_steps: int, dt: float, conv_thr: float):
    Kp       = result['Kp']
    ct       = result['conv_times']
    fe       = result['final_err']
    err0     = result['err_traces'][:, 0]
    pct      = result['pct_converged']

    print(f"\n  Kp = {Kp:.4f}")
    print(f"  Initial   ‖ω−ω*‖: {err0.mean():.3f} ± {err0.std():.3f} rad/s")
    print(f"  Final     ‖ω−ω*‖: {fe.mean():.4f} ± {fe.std():.4f} rad/s")
    print(f"  Time to ‖ω−ω*‖ < {conv_thr:.2f}: "
          f"{ct.mean():.3f} ± {ct.std():.3f} s   "
          f"(min={ct.min():.3f}  max={ct.max():.3f})")
    print(f"  Converged: {pct:.0f}% of trials "
          f"within {n_steps*dt:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Linearisation sanity check (analytical, no simulation needed)
# ─────────────────────────────────────────────────────────────────────────────

def print_linearisation(I1, I2, I3, Kp, omega_star):
    """Print eigenvalues of the linearised closed-loop at ω*."""
    # δω₃ mode
    lam3 = -Kp / I3
    tau3 = -1.0 / lam3

    # δω₁,δω₂ modes
    # A = [-Kp/I1   w*(I2-I3)/I1]
    #     [w*(I3-I1)/I2  -Kp/I2 ]
    a11 = -Kp / I1
    a12 = omega_star * (I2 - I3) / I1
    a21 = omega_star * (I3 - I1) / I2
    a22 = -Kp / I2

    tr_A  = a11 + a22
    det_A = a11 * a22 - a12 * a21
    disc  = tr_A**2 - 4 * det_A

    if disc >= 0:
        lam_real = (tr_A + np.sqrt(disc)) / 2, (tr_A - np.sqrt(disc)) / 2
        print(f"  δω₁,₂ modes: overdamped  λ = {lam_real[0]:.3f}, {lam_real[1]:.3f} s⁻¹  "
              f"(τ_slow = {-1/lam_real[0]:.4f} s,  τ_fast = {-1/lam_real[1]:.4f} s)")
    else:
        Re_lam = tr_A / 2
        Im_lam = np.sqrt(-disc) / 2
        freq_hz = Im_lam / (2 * np.pi)
        print(f"  δω₁,₂ modes: oscillatory  λ = {Re_lam:.3f} ± {Im_lam:.3f}i s⁻¹  "
              f"(τ = {-1/Re_lam:.4f} s,  f = {freq_hz:.3f} Hz)")
    print(f"  δω₃  mode:  λ = {lam3:.3f} s⁻¹  (τ = {tau3:.4f} s)")
    # Overdamped when disc = Kp²(1/I1-1/I2)² - 4|a12·a21| ≥ 0
    # Threshold: Kp_od = 2√|a12·a21| / |1/I1-1/I2|  (a12,a21 already carry ω*)
    overdamped_kp = 2 * np.sqrt(abs(a12 * a21)) / abs(1/I1 - 1/I2)
    print(f"  Overdamped threshold Kp ≥ {overdamped_kp:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    omega_star_vec = np.array([0.0, 0.0, args.omega_star], dtype=np.float64)

    print("=" * 62)
    print("Stage D — IDA-PBC Tennis Racket Stabilisation")
    print("=" * 62)
    print(f"\nTarget:  ω* = (0, 0, {args.omega_star:.4f}) rad/s  [{args.omega_star/(2*np.pi):.3f} rev/s]")
    print(f"Initial: axis {args.axis_start}  "
          + {0: "(e₁ stable)", 1: "(e₂ UNSTABLE — Dzhanibekov)", 2: "(e₃ target)"}[args.axis_start])
    print(f"Trials:  {args.n_trials}  ×  {args.n_steps} steps  "
          f"(dt=0.05s  →  {args.n_steps*0.05:.1f}s per trial)")
    print(f"Convergence threshold: ‖ω-ω*‖ < {args.conv_thr:.2f} rad/s")

    # Build env (default cfg0 geometry — head_a=0.195, head_b=0.135, etc.)
    env = Env(disturbance_torque_std=0.0, seed=args.seed)
    iinfo = env.get_inertia_info()
    I1, I2, I3 = iinfo['I1'], iinfo['I2'], iinfo['I3']

    print(f"\nRacket (cfg0):")
    print(f"  I1={I1:.6f}  I2={I2:.6f}  I3={I3:.6f}  kg·m²")
    print(f"  I2/I1={I2/I1:.2f}   asym=(I3-I1)/I2={(I3-I1)/I2:.2f}")
    print(f"  head_a={iinfo['head_a']:.3f}  head_b={iinfo['head_b']:.3f}  "
          f"handle={iinfo['handle_length']:.3f}  mass={iinfo['total_mass']:.3f} kg")

    # ── Linearisation analysis ─────────────────────────────────────────────
    print(f"\n── Linearised stability (Kp={args.Kp:.4f}) ──")
    print_linearisation(I1, I2, I3, args.Kp, args.omega_star)

    # ── Simulation ────────────────────────────────────────────────────────
    Kp_list = ([float(k) for k in args.Kp_scan.split(',')]
               if args.Kp_scan else [args.Kp])

    all_results = {}
    for Kp in Kp_list:
        if args.Kp_scan:
            print(f"\n── Scan Kp = {Kp:.4f} ──")
            print_linearisation(I1, I2, I3, Kp, args.omega_star)
        result = run_trials(
            Kp=Kp,
            omega_star_vec=omega_star_vec,
            n_trials=args.n_trials,
            n_steps=args.n_steps,
            axis_start=args.axis_start,
            conv_thr=args.conv_thr,
            seed=args.seed,
            env=env,
            verbose=True,
        )
        all_results[Kp] = result

    # ── Convergence curve summary (profile at select timesteps) ───────────
    # Among fully-converged Kps, prefer fastest convergence time; else fallback to last
    fully_conv = [k for k, r in all_results.items() if r['pct_converged'] >= 95.0]
    if fully_conv:
        best_Kp = min(fully_conv, key=lambda k: all_results[k]['conv_times'].mean())
    else:
        best_Kp = Kp_list[-1]
    best    = all_results[best_Kp]

    print(f"\n── Error profile (best Kp={best_Kp:.4f}, avg over {args.n_trials} trials) ──")
    print(f"{'t [s]':>8}  {'mean ‖ω-ω*‖':>14}  {'max ‖ω-ω*‖':>14}")
    profile_steps = list(range(0, min(21, args.n_steps + 1), 2))
    if args.n_steps not in profile_steps:
        profile_steps.append(args.n_steps)
    for step in profile_steps:
        t_s   = step * env.dt
        mean_ = best['err_traces'][:, step].mean()
        max_  = best['err_traces'][:, step].max()
        print(f"  {t_s:6.2f}   {mean_:14.6f}   {max_:14.6f}")

    # ── Verdict ────────────────────────────────────────────────────────────
    final_mean = best['final_err'].mean()
    pct_conv   = best['pct_converged']
    passed     = pct_conv >= 95.0 and final_mean < 0.10

    print(f"\n{'='*62}")
    print(f"{'PASS ✓' if passed else 'FAIL ✗'}  "
          f"Kp={best_Kp:.4f}  |  "
          f"{pct_conv:.0f}% converged  |  "
          f"final ‖ω-ω*‖ = {final_mean:.4f} rad/s")
    print(f"{'='*62}")

    env.close()
    return all_results


if __name__ == '__main__':
    main()

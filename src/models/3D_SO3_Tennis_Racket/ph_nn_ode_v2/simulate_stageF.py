"""Stage F simulation: full-state (R*, ω*) stabilisation on SO(3) × ℝ³.

Stage D/E only controlled angular velocity ω → ω*.  Stage F adds an orientation
target R* ∈ SO(3), using the geodesic attitude error e_R = vee(logm(R*ᵀ R)).

Control law:  u = −K_R · e_R − K_p · e_ω

Four scenarios:
  A. Rest at identity:    R* = I₃,       ω* = 0.       K_R scan.
  B. Rest at 45° pitch:   R* = Ry(π/4),  ω* = 0.       K_R = 0.10.
  C. Sanity check (K_R=0 → Stage D):
                          R* = I₃,       ω* = (0,0,2π). Must match Stage D.
  D. Rest at identity with wind  (σ = 0.05 N·m):
                          R* = I₃,       ω* = 0.        K_R = 0.10.

All scenarios start from Dzhanibekov tumbling (axis=1, e₂ unstable) with a random
initial orientation R₀ ∈ SO(3).  Convergence requires simultaneously:
    ‖e_R‖  < theta_thr  [rad]   (default 0.10 rad ≈ 5.7°)
    ‖e_ω‖  < omega_thr  [rad/s] (default 0.50 rad/s)

Usage:
    /Users/katesur/Projects/LieSPHGP/venv/bin/python3 -u simulate_stageF.py

Options:
    --omega_star   Spin rate for Scenario C [rad/s]   (default 2π)
    --n_trials     Trials per configuration           (default 20)
    --n_steps      Steps per trial (dt=0.05s)         (default 300 = 15s)
    --hold_perturb Not used for axis starts; kept for compatibility
    --wind_std     Disturbance std for Scenario D [N·m] (default 0.05)
    --theta_thr    Attitude convergence threshold [rad] (default 0.10)
    --omega_thr    Velocity convergence threshold [rad/s] (default 0.50)
    --seed         RNG seed                           (default 0)
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
from controller_stageF import GeometricAttitudeController, expm_SO3, hat


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--omega_star',  type=float, default=2 * np.pi)
    p.add_argument('--n_trials',    type=int,   default=20)
    p.add_argument('--n_steps',     type=int,   default=300)
    p.add_argument('--wind_std',    type=float, default=0.05)
    p.add_argument('--theta_thr',   type=float, default=0.10,
                   help='Attitude convergence threshold ‖e_R‖ [rad]')
    p.add_argument('--omega_thr',   type=float, default=0.50,
                   help='Velocity convergence threshold ‖e_ω‖ [rad/s]')
    p.add_argument('--seed',        type=int,   default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Linearised stability helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_linearised_stability(I1, I2, I3, K_R, K_p, dt):
    """Print ωn, ζ, mode type for each principal axis.

    The linearised closed-loop per axis i is an independent 2nd-order system:
        [ė_R_i]   =  [0      1  ] [e_R_i]
        [ė_ω_i]      [-K_R/Iᵢ  -K_p/Iᵢ] [e_ω_i]

    Natural frequency:  ωn = √(K_R / Iᵢ)
    Damping ratio:      ζ  = K_p / (2 √(K_R · Iᵢ))
    Critical K_p:       K_p_cd = 2 √(K_R · Iᵢ)
    """
    names = ["e₁ (I₁, smallest)", "e₂ (I₂, middle)", "e₃ (I₃, largest)"]
    I_vals = [I1, I2, I3]
    kp_max_zoh = 2.0 * I1 / dt
    print(f"  {'Axis':<22}  {'ωn [rad/s]':>12}  {'ζ':>8}  {'mode':>12}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*8}  {'-'*12}")
    for name, I in zip(names, I_vals):
        if K_R < 1e-15:
            print(f"  {name:<22}  {'—':>12}  {'—':>8}  {'K_R=0 (Stage D)':>12}")
            continue
        wn  = np.sqrt(K_R / I)
        zet = K_p / (2.0 * np.sqrt(K_R * I))
        mode = "overdamped" if zet > 1.0 else ("critically" if abs(zet - 1) < 0.05
                                                else "oscillatory")
        print(f"  {name:<22}  {wn:12.3f}  {zet:8.3f}  {mode:>12}")
    kr_max_zoh = K_p / dt      # 2nd-order ZOH bound for K_R: K_R < K_p/dt
    print(f"  ZOH K_p_max = 2·I₁/dt = {kp_max_zoh:.4f}  [current K_p={K_p:.4f} — "
          + ("OK ✓" if K_p < kp_max_zoh else "EXCEEDS BOUND ✗") + "]")
    print(f"  ZOH K_R_max = K_p/dt  = {kr_max_zoh:.4f}  [current K_R={K_R:.4f} — "
          + ("OK ✓" if K_R < kr_max_zoh else "EXCEEDS BOUND ✗") + "]")


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_trials(
    env: Env,
    ctrl: GeometricAttitudeController,
    n_trials: int,
    n_steps: int,
    theta_thr: float,
    omega_thr: float,
    seed: int,
    ic_axis: int = 1,
) -> dict:
    """Run n_trials closed-loop episodes.  Returns convergence statistics.

    Convergence criterion: ‖e_R‖ < theta_thr AND ‖e_ω‖ < omega_thr
    simultaneously.
    """
    dt = env.dt

    err_R_traces = np.zeros((n_trials, n_steps + 1), dtype=np.float64)
    err_w_traces = np.zeros((n_trials, n_steps + 1), dtype=np.float64)
    disturbance_norms = np.zeros(n_trials, dtype=np.float64)
    conv_steps = []

    for trial in range(n_trials):
        obs, _ = env.reset(seed=seed + trial, options={"axis": ic_axis})
        R_flat = obs[:9].astype(np.float64)
        omega  = obs[9:12].astype(np.float64)
        disturbance_norms[trial] = float(np.linalg.norm(env._disturbance_torque))

        eR, ew = ctrl.errors(R_flat, omega)
        err_R_traces[trial, 0] = eR
        err_w_traces[trial, 0] = ew

        for step in range(1, n_steps + 1):
            u = ctrl(R_flat, omega)
            obs, _, _, _, _ = env.step(u)
            R_flat = obs[:9].astype(np.float64)
            omega  = obs[9:12].astype(np.float64)
            eR, ew = ctrl.errors(R_flat, omega)
            err_R_traces[trial, step] = eR
            err_w_traces[trial, step] = ew

        # First step both errors simultaneously below threshold
        conv_mask = ((err_R_traces[trial] < theta_thr)
                     & (err_w_traces[trial] < omega_thr))
        below = np.where(conv_mask)[0]
        conv_steps.append(int(below[0]) if len(below) else n_steps)

    conv_steps = np.array(conv_steps, dtype=float)

    return {
        'err_R_traces':      err_R_traces,          # (n_trials, n_steps+1) [rad]
        'err_w_traces':      err_w_traces,           # (n_trials, n_steps+1) [rad/s]
        't_grid':            np.arange(n_steps + 1) * dt,
        'conv_steps':        conv_steps,
        'conv_times':        conv_steps * dt,        # [s]
        'final_eR':          err_R_traces[:, -1],
        'final_ew':          err_w_traces[:, -1],
        'late_eR':           err_R_traces[:, n_steps // 2:].max(axis=1),
        'late_ew':           err_w_traces[:, n_steps // 2:].max(axis=1),
        'pct_converged':     100.0 * np.mean(conv_steps < n_steps),
        'disturbance_norms': disturbance_norms,
    }


def print_trial_result(r: dict, n_steps: int, dt: float,
                       theta_thr: float, omega_thr: float,
                       K_R: float, K_p: float,
                       show_dist: bool = False):
    ct   = r['conv_times']
    feR  = r['final_eR']
    few  = r['final_ew']
    lmR  = r['late_eR']
    lmw  = r['late_ew']
    pct  = r['pct_converged']
    eR0  = r['err_R_traces'][:, 0]
    ew0  = r['err_w_traces'][:, 0]

    # Distinguish truly unstable (errors grow) from bounded SS offset.
    # Compare 2nd-half mean vs 1st-step value — growing if 2nd-half > initial.
    truly_unstable = (lmR.mean() > r['err_R_traces'][:, 0].mean() * 1.5
                      or lmw.mean() > r['err_w_traces'][:, 0].mean() * 1.5)
    has_large_ss = lmR.mean() > theta_thr * 5 or lmw.mean() > omega_thr * 5
    label = ("UNSTABLE"   if truly_unstable
             else "BOUNDED_SS" if has_large_ss
             else "CONVERGED"  if pct >= 95
             else "PARTIAL")

    print(f"\n  K_R={K_R:.4f}  K_p={K_p:.4f}  [{label}]")
    print(f"    Initial  ‖e_R‖: {eR0.mean():.3f} ± {eR0.std():.3f} rad")
    print(f"    Initial  ‖e_ω‖: {ew0.mean():.3f} ± {ew0.std():.3f} rad/s")
    print(f"    Final    ‖e_R‖: {feR.mean():.4f} ± {feR.std():.4f} rad")
    print(f"    Final    ‖e_ω‖: {few.mean():.4f} ± {few.std():.4f} rad/s")
    print(f"    Late max ‖e_R‖: {lmR.mean():.4f} ± {lmR.std():.4f} rad  (2nd half)")
    print(f"    Late max ‖e_ω‖: {lmw.mean():.4f} ± {lmw.std():.4f} rad/s (2nd half)")
    if pct > 0:
        print(f"    Conv (‖e_R‖<{theta_thr:.2f} AND ‖e_ω‖<{omega_thr:.2f}): "
              f"{ct.mean():.3f} ± {ct.std():.3f} s  "
              f"(min={ct.min():.2f}  max={ct.max():.2f})")
    print(f"    Converged: {pct:.0f}% of {len(feR)} trials within {n_steps*dt:.1f}s")
    if show_dist:
        dn = r['disturbance_norms']
        print(f"    |d|: {dn.mean():.4f} ± {dn.std():.4f} N·m  "
              f"(SS ‖e_ω‖ ≈ {dn.mean()/K_p:.3f} rad/s;  "
              f"SS ‖e_R‖ ≈ {dn.mean()/K_R:.3f} rad)")


def print_error_profile(r: dict, env_dt: float, n_steps: int, n_trials: int):
    """Print mean attitude and velocity error at selected timesteps."""
    print(f"\n  Error profile (avg over {n_trials} trials):")
    print(f"  {'t[s]':>6}  {'‖e_R‖ mean':>12}  {'‖e_R‖ max':>12}  "
          f"{'‖e_ω‖ mean':>12}  {'‖e_ω‖ max':>12}")
    steps = list(range(0, min(21, n_steps + 1), 2))
    if n_steps not in steps:
        steps.append(n_steps)
    for s in steps:
        t    = s * env_dt
        eRm  = r['err_R_traces'][:, s].mean()
        eRx  = r['err_R_traces'][:, s].max()
        ewm  = r['err_w_traces'][:, s].mean()
        ewx  = r['err_w_traces'][:, s].max()
        print(f"  {t:6.2f}  {eRm:12.6f}  {eRx:12.6f}  {ewm:12.6f}  {ewx:12.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runners
# ─────────────────────────────────────────────────────────────────────────────

def scenario_A(args, I1, I2, I3, env_clean):
    """Rest at identity: R* = I₃, ω* = 0.  K_R scan."""
    print(f"\n{'─'*62}")
    print(f"Scenario A: Rest at identity  R* = I₃,  ω* = 0")
    print(f"  IC: Dzhanibekov tumbling (axis=1, random R₀)")
    print(f"  K_p = 0.10 fixed;  scanning K_R")
    print(f"{'─'*62}")

    R_star     = np.eye(3)
    omega_star = np.zeros(3)
    K_p        = 0.10
    Kp_max     = 2.0 * I1 / env_clean.dt
    K_R_list   = [0.01, 0.05, 0.10, 0.20, 0.50]

    results = {}
    for K_R in K_R_list:
        print(f"\n  ── Stability analysis (K_R={K_R:.4f}, K_p={K_p:.4f}) ──")
        print_linearised_stability(I1, I2, I3, K_R, K_p, env_clean.dt)

        ctrl = GeometricAttitudeController(R_star, omega_star, K_R=K_R, K_p=K_p)
        r = run_trials(
            env=env_clean, ctrl=ctrl,
            n_trials=args.n_trials, n_steps=args.n_steps,
            theta_thr=args.theta_thr, omega_thr=args.omega_thr,
            seed=args.seed,
        )
        print_trial_result(r, args.n_steps, env_clean.dt,
                           args.theta_thr, args.omega_thr, K_R, K_p)
        results[K_R] = r

    # Pick best: fastest convergence among fully converged
    fully_conv = [k for k, r in results.items() if r['pct_converged'] >= 95]
    if fully_conv:
        best_KR = min(fully_conv, key=lambda k: results[k]['conv_times'].mean())
    else:
        best_KR = K_R_list[-1]

    print(f"\n  ── Error profile for best K_R = {best_KR:.4f} ──")
    print_error_profile(results[best_KR], env_clean.dt, args.n_steps, args.n_trials)

    return results, best_KR


def scenario_B(args, I1, I2, I3, env_clean):
    """Rest at 45° pitch: R* = Ry(π/4), ω* = 0.  K_R = 0.10."""
    K_R  = 0.10
    K_p  = 0.10
    R_star     = expm_SO3(hat([0.0, np.pi / 4.0, 0.0]))   # 45° rotation about e₂
    omega_star = np.zeros(3)

    print(f"\n{'─'*62}")
    print(f"Scenario B: Rest at 45° pitch  R* = Ry(π/4),  ω* = 0")
    print(f"  Target R*:\n{R_star.round(4)}")
    print(f"  IC: Dzhanibekov tumbling (axis=1, random R₀)")
    print(f"  K_R = {K_R:.4f},  K_p = {K_p:.4f}")
    print(f"{'─'*62}")

    print(f"\n  ── Stability analysis ──")
    print_linearised_stability(I1, I2, I3, K_R, K_p, env_clean.dt)

    ctrl = GeometricAttitudeController(R_star, omega_star, K_R=K_R, K_p=K_p)
    r = run_trials(
        env=env_clean, ctrl=ctrl,
        n_trials=args.n_trials, n_steps=args.n_steps,
        theta_thr=args.theta_thr, omega_thr=args.omega_thr,
        seed=args.seed,
    )
    print_trial_result(r, args.n_steps, env_clean.dt,
                       args.theta_thr, args.omega_thr, K_R, K_p)
    print_error_profile(r, env_clean.dt, args.n_steps, args.n_trials)
    return r


def scenario_C_sanity(args, I1, I2, I3, env_clean):
    """Sanity check: K_R = 0 must reproduce Stage D (K_p=0.10, ω*=(0,0,2π))."""
    K_R        = 0.0
    K_p        = 0.10
    R_star     = np.eye(3)
    omega_star = np.array([0.0, 0.0, args.omega_star])

    print(f"\n{'─'*62}")
    print(f"Scenario C (sanity): K_R=0  →  Stage D  "
          f"[ω*=(0,0,{args.omega_star:.2f}), K_p={K_p}]")
    print(f"  Expected: ~100% convergence in ~0.3s  (matches Stage D result)")
    print(f"{'─'*62}")

    ctrl = GeometricAttitudeController(R_star, omega_star, K_R=K_R, K_p=K_p)
    r = run_trials(
        env=env_clean, ctrl=ctrl,
        n_trials=args.n_trials, n_steps=args.n_steps,
        theta_thr=np.pi,          # K_R=0 → no R criterion; use π so it never blocks
        omega_thr=args.omega_thr,
        seed=args.seed,
    )
    # For sanity, report only the velocity convergence
    ct  = r['conv_times']
    few = r['final_ew']
    pct = r['pct_converged']
    print(f"\n  Conv time (‖e_ω‖<{args.omega_thr:.2f}): "
          f"{ct.mean():.3f} ± {ct.std():.3f} s")
    print(f"  Final ‖e_ω‖: {few.mean():.4f} ± {few.std():.4f} rad/s")
    print(f"  Converged: {pct:.0f}% of {args.n_trials} trials  "
          f"({'PASS ✓' if pct >= 95 else 'FAIL ✗'}  matches Stage D)")
    return r


def scenario_D_wind(args, I1, I2, I3, wind_std):
    """Rest at identity with stochastic wind.  R* = I₃, ω* = 0.  K_R scan.

    Key physics (differs from Stage E):
      With ω* = 0, constant wind d ≠ 0 is balanced in steady state by the
      orientation restoring torque:
          K_R · e_R  =  d   →   ‖e_R‖_ss ≈ |d| / K_R

      The angular velocity converges to zero (K_p damps ω → 0), so the wind
      creates an ORIENTATION offset, not a velocity offset.  This is the dual
      of Stage E, where ω* ≠ 0 and wind created a VELOCITY offset |d|/K_p.

    ZOH bound for K_R (2nd-order attitude loop):
      The discrete-time I₁ mode has characteristic polynomial
          z² − (2 − K_p T/I₁) z + (1 − K_p T/I₁ + K_R T²/I₁) = 0
      Schur stability requires K_R < K_p / T.
      For K_p=0.10, T=0.05: K_R_max = 0.10/0.05 = 2.00  N·m/rad.

    Viable K_R with wind to satisfy ‖e_R‖ < θ_thr:
          K_R_min(wind) = |d| / θ_thr   (SS offset requirement)
          K_R_max(ZOH)  = K_p / T       (discrete stability)
    """
    K_p        = 0.10
    R_star     = np.eye(3)
    omega_star = np.zeros(3)
    KR_max_zoh = K_p / 0.05          # = 2.00 for K_p=0.10, dt=0.05

    print(f"\n{'─'*62}")
    print(f"Scenario D: Rest at identity with wind")
    print(f"  R* = I₃,  ω* = 0,  wind σ = {wind_std:.3f} N·m")
    print(f"  K_p = {K_p:.4f}   K_R scan")
    print(f"  ZOH K_R_max = K_p/T = {KR_max_zoh:.4f} N·m/rad")
    print(f"  SS orientation offset: ‖e_R‖_ss ≈ |d|/K_R  (wind balanced by K_R e_R)")
    print(f"{'─'*62}")

    # Build a fresh wind env per K_R (same disturbances due to fixed seed)
    K_R_list = [0.10, 0.50, 1.00, 1.50]
    results  = {}
    d_mean   = None

    for K_R in K_R_list:
        env_wind = Env(disturbance_torque_std=wind_std, seed=args.seed)
        ctrl = GeometricAttitudeController(R_star, omega_star, K_R=K_R, K_p=K_p)
        r = run_trials(
            env=env_wind, ctrl=ctrl,
            n_trials=args.n_trials, n_steps=args.n_steps,
            theta_thr=args.theta_thr, omega_thr=args.omega_thr,
            seed=args.seed,
        )
        if d_mean is None:
            d_mean = r['disturbance_norms'].mean()
        print_trial_result(r, args.n_steps, env_wind.dt,
                           args.theta_thr, args.omega_thr, K_R, K_p,
                           show_dist=(K_R == K_R_list[0]))
        env_wind.close()
        results[K_R] = r

    # SS-offset prediction table
    print(f"\n  SS orientation error vs predicted |d|/K_R  (|d| ≈ {d_mean:.4f} N·m):")
    print(f"  {'K_R':>6}  {'‖e_R‖ actual':>14}  {'|d|/K_R pred':>14}  "
          f"{'ratio':>7}  {'‖e_ω‖ actual':>14}")
    for K_R, r in results.items():
        feR   = r['final_eR'].mean()
        few   = r['final_ew'].mean()
        pred  = d_mean / K_R
        ratio = feR / pred if pred > 0 else float('nan')
        print(f"  {K_R:6.2f}  {feR:14.4f}  {pred:14.4f}  {ratio:7.3f}  {few:14.4f}")

    # Find best K_R (max convergence % within ZOH bound)
    best_KR = max(
        (k for k in results if k < KR_max_zoh * 0.99),
        key=lambda k: results[k]['pct_converged'],
        default=None,
    )

    print(f"\n  ZOH window for tight holding: K_R ∈ ({d_mean/args.theta_thr:.3f}, "
          f"{KR_max_zoh:.3f})  [θ_thr={args.theta_thr:.2f} rad]")
    if best_KR:
        r_best = results[best_KR]
        print(f"  Best K_R = {best_KR:.2f}: {r_best['pct_converged']:.0f}% conv,  "
              f"late ‖e_R‖ = {r_best['late_eR'].mean():.4f} rad")

    return results, best_KR


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    print("=" * 62)
    print("Stage F — Geometric Attitude Control on SO(3) × ℝ³")
    print("=" * 62)

    env_clean = Env(disturbance_torque_std=0.0, seed=args.seed)
    iinfo = env_clean.get_inertia_info()
    I1, I2, I3 = iinfo['I1'], iinfo['I2'], iinfo['I3']

    print(f"\nRacket (cfg0):")
    print(f"  I1={I1:.6f}  I2={I2:.6f}  I3={I3:.6f}  kg·m²")
    print(f"  K_p_max (ZOH) = 2·I₁/dt = {2*I1/env_clean.dt:.4f} N·m·s/rad")
    print(f"\nTrials: {args.n_trials}  ×  {args.n_steps} steps "
          f"(dt={env_clean.dt:.2f}s → {args.n_steps*env_clean.dt:.1f}s per trial)")
    print(f"Convergence: ‖e_R‖ < {args.theta_thr:.2f} rad  AND  "
          f"‖e_ω‖ < {args.omega_thr:.2f} rad/s  (simultaneously)")

    resA, best_KR_A  = scenario_A(args, I1, I2, I3, env_clean)
    resB             = scenario_B(args, I1, I2, I3, env_clean)
    resC             = scenario_C_sanity(args, I1, I2, I3, env_clean)
    resD, best_KR_D  = scenario_D_wind(args, I1, I2, I3, args.wind_std)

    # ── Final verdict ──────────────────────────────────────────────────────
    A_pass = resA[best_KR_A]['pct_converged'] >= 95

    B_pass = resB['pct_converged'] >= 95

    C_pass = resC['pct_converged'] >= 95

    # Scenario D: converges if best K_R achieves ≥95% within ZOH bound
    D_r_best = resD[best_KR_D] if best_KR_D is not None else resD[min(resD)]
    D_pass   = D_r_best['pct_converged'] >= 95

    KR_max = K_p_val = 0.10 / env_clean.dt   # K_p=0.10 at dt=0.05

    print(f"\n{'='*62}")
    print(f"Stage F summary")
    print(f"  A — rest at I₃     (no wind):    {'PASS ✓' if A_pass else 'FAIL ✗'}"
          f"  [best K_R={best_KR_A:.3f}]")
    print(f"  B — rest at Ry(π/4)(no wind):    {'PASS ✓' if B_pass else 'FAIL ✗'}")
    print(f"  C — Stage D sanity (K_R=0):       {'PASS ✓' if C_pass else 'FAIL ✗'}")
    print(f"  D — rest at I₃ (wind {args.wind_std:.2f} N·m):  "
          f"{'PASS ✓' if D_pass else 'FAIL ✗'}"
          + (f"  [best K_R={best_KR_D:.2f}]" if best_KR_D else ""))
    print(f"{'='*62}")

    env_clean.close()


if __name__ == '__main__':
    main()

"""Stage E simulation: e₁ stabilisation, e₂ hold, wind robustness.

Three scenarios:
  A. Redirect + stabilise e₁ (ω* = (2π,0,0), start from e₂ tumbling, no wind)
  B. Hold e₂  (ω* = (0,2π,0), start near e₂ with small perturbation, no wind)
       — Kp scan to find empirical minimum Kp
  C. Hold e₂  with stochastic wind (disturbance_torque_std=0.05 N·m)

Linearised stability (free body, Euler's equations):

  For ω* = ω* eₙ, perturbations in the orthogonal plane form a 2D system.
  Off-diagonal Euler coupling → eigenvalues:

    λ² = ω*² (I_L - I_T)(I_S - I_T) / (I_L · I_S)

  where I_T = inertia about target axis, I_S, I_L = smaller/larger inertia.

  e₁ (smallest I): I_T < I_S < I_L → (I_L-I_T)>0, (I_S-I_T)>0, product>0 but wait
    Actual formula for ω* = (ω,0,0):
      δω̇₂ = (I₃-I₁)ω/I₂ · δω₃
      δω̇₃ = (I₁-I₂)ω/I₃ · δω₂
      λ² = (I₃-I₁)(I₁-I₂)ω²/(I₂I₃)  → λ² < 0 (stable oscillations)

  e₂ (middle I): λ² > 0 (saddle, unstable) — Dzhanibekov
  e₃ (largest I): λ² < 0 (stable oscillations)

  Kp_min for e₂ stabilisation:
    det(A_cl) > 0 requires: Kp² > |(I₂-I₃)(I₁-I₂)| · ω*₂²
    Kp_min = ω*₂ · √(|I₂-I₃| · |I₁-I₂|)

Usage:
    /Users/katesur/Projects/LieSPHGP/venv/bin/python3 -u simulate_stageE.py

Options:
    --omega_star     Spin rate [rad/s]             (default 2π)
    --n_trials       Trials per Kp value           (default 20)
    --n_steps        Steps per trial (dt=0.05s)    (default 200 = 10s)
    --hold_perturb   ‖δω‖ for hold IC [rad/s]      (default 0.3)
    --wind_std       Disturbance std [N·m]          (default 0.05)
    --conv_thr       Convergence threshold          (default 0.5 rad/s)
    --seed           RNG seed                       (default 0)
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
    p = argparse.ArgumentParser()
    p.add_argument('--omega_star',    type=float, default=2 * np.pi)
    p.add_argument('--n_trials',      type=int,   default=20)
    p.add_argument('--n_steps',       type=int,   default=200)
    p.add_argument('--hold_perturb',  type=float, default=0.30,
                   help='std of ω perturbation for hold IC [rad/s]')
    p.add_argument('--wind_std',      type=float, default=0.05,
                   help='disturbance_torque_std for wind scenario')
    p.add_argument('--conv_thr',      type=float, default=0.50)
    p.add_argument('--seed',          type=int,   default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Stability helpers
# ─────────────────────────────────────────────────────────────────────────────

def free_body_eigenvalue(I1, I2, I3, omega_star, target_axis):
    """Eigenvalue of the free-body linearisation about ω* = ω_star · e_axis.

    Returns the real part λ_real and imaginary part λ_imag of one eigenvalue.
    If λ_real > 0: unstable.  If λ_real = 0: neutral (oscillatory).
    """
    # δω̇_a = (I_c - I_t) ω* / I_a · δω_b
    # δω̇_b = (I_t - I_b) ω* / I_b ... use component form
    I = [I1, I2, I3]
    # Indices of the two orthogonal axes
    a, b = [(target_axis + 1) % 3, (target_axis + 2) % 3]
    It = I[target_axis]

    # From Euler equations, linearised:
    # Ia * δω̇_a = (Ib - It) * ω* * δω_b   [from Ia ω̇_a = (Ij - Ik) ωj ωk]
    # Ib * δω̇_b = (It - Ia) * ω* * δω_a
    Ia, Ib = I[a], I[b]
    m12 = (Ib - It) * omega_star / Ia
    m21 = (It - Ia) * omega_star / Ib

    lam_sq = m12 * m21   # = (Ib-It)(It-Ia)ω*² / (IaIb)

    if lam_sq > 1e-15:
        lam = np.sqrt(lam_sq)
        return lam, 0.0           # real ±λ  (unstable saddle)
    elif lam_sq < -1e-15:
        return 0.0, np.sqrt(-lam_sq)  # ±i|λ| (stable oscillation)
    else:
        return 0.0, 0.0           # borderline (marginal)


def kp_min_e2(I1, I2, I3, omega_star):
    """Minimum Kp to stabilise spinning about the intermediate axis.

    From det(A_cl) > 0:
        Kp² > |(I₂-I₃)(I₁-I₂)| · ω*²
        Kp_min = ω* · √(|I₂-I₃| · |I₁-I₂|)
    """
    return omega_star * np.sqrt(abs(I2 - I3) * abs(I1 - I2))


def kp_max_zoh(I1, dt):
    """Maximum Kp before ZOH discrete-time instability on the I₁ (smallest) axis.

    With ZOH period T = dt (env.dt), the discrete-time eigenvalue for the I₁ axis is:
        z = 1 - Kp · T / I₁
    Stable iff |z| < 1, i.e.  Kp < 2 · I₁ / T.
    """
    return 2.0 * I1 / dt


def print_stability_table(I1, I2, I3, omega_star, dt):
    """Print free-body eigenvalues and Kp bounds for all three axes."""
    names = ["e₁ (smallest I, long head)", "e₂ (middle I, short head — UNSTABLE)",
             "e₃ (largest I, handle)"]
    kp_max = kp_max_zoh(I1, dt)
    print(f"  {'Axis':<36}  {'λ_free':>12}  {'K_p_min':>10}  {'Status'}")
    print(f"  {'-'*36}  {'-'*12}  {'-'*10}  {'-'*15}")
    for axis in range(3):
        lr, li = free_body_eigenvalue(I1, I2, I3, omega_star, axis)
        if lr > 1e-10:
            lam_str = f"+{lr:.3f} (real)"
            status  = "UNSTABLE"
        elif li > 1e-10:
            lam_str = f"±{li:.3f}i"
            status  = "stable"
        else:
            lam_str = "0"
            status  = "marginal"

        kp_m = (kp_min_e2(I1, I2, I3, omega_star) if axis == 1 else 0.0)
        print(f"  {names[axis]:<36}  {lam_str:>12}  {kp_m:>10.4f}  {status}")
    print(f"\n  ZOH stability bound (K_p_max = 2·I₁/T):  {kp_max:.4f} N·m·s/rad  "
          f"[T = dt = {dt:.3f} s]")
    print(f"  Safe operating range for e₂ hold: "
          f"({kp_min_e2(I1, I2, I3, omega_star):.4f}, {kp_max:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_trials(
    env: Env,
    omega_star_vec: np.ndarray,
    Kp: float,
    n_trials: int,
    n_steps: int,
    conv_thr: float,
    seed: int,
    ic_mode: str = "axis",    # "axis" or "hold"
    ic_axis: int = 1,         # for ic_mode="axis"
    hold_perturb: float = 0.3,  # for ic_mode="hold"
    rng_seed_offset: int = 0,
) -> dict:
    """
    ic_mode = "axis":  env.reset(options={"axis": ic_axis})
    ic_mode = "hold":  omega_init = omega_star_vec + N(0, hold_perturb²)
                       with a random R from env.reset()
    """
    ctrl = PDBodyFrameController(omega_star=omega_star_vec, Kp=Kp)
    rng  = np.random.default_rng(seed + rng_seed_offset)

    dt = env.dt
    err_traces = np.zeros((n_trials, n_steps + 1), dtype=np.float64)
    disturbance_norms = np.zeros(n_trials, dtype=np.float64)
    conv_steps = []

    for trial in range(n_trials):
        if ic_mode == "axis":
            obs, _ = env.reset(seed=seed + trial,
                               options={"axis": ic_axis})
        else:  # hold
            noise   = rng.normal(0.0, hold_perturb, 3)
            omega_0 = omega_star_vec + noise
            # Reset gets a random R but we override omega
            obs, _ = env.reset(seed=seed + trial,
                               options={"omega_init": omega_0})

        disturbance_norms[trial] = float(
            np.linalg.norm(env._disturbance_torque))

        R_flat = obs[:9].astype(np.float64)
        omega  = obs[9:12].astype(np.float64)
        err_traces[trial, 0] = ctrl.omega_error(omega)

        for step in range(1, n_steps + 1):
            u   = ctrl(R_flat, omega)
            obs, _, _, _, _ = env.step(u)
            R_flat = obs[:9].astype(np.float64)
            omega  = obs[9:12].astype(np.float64)
            err_traces[trial, step] = ctrl.omega_error(omega)

        below = np.where(err_traces[trial] < conv_thr)[0]
        conv_steps.append(int(below[0]) if len(below) else n_steps)

    conv_steps = np.array(conv_steps, dtype=float)
    final_err  = err_traces[:, -1]
    late_max   = err_traces[:, n_steps // 2:].max(axis=1)  # max in 2nd half

    return {
        'Kp':            Kp,
        'err_traces':    err_traces,
        't_grid':        np.arange(n_steps + 1) * dt,
        'conv_steps':    conv_steps,
        'conv_times':    conv_steps * dt,
        'final_err':     final_err,
        'late_max':      late_max,    # stability indicator
        'disturbance_norms': disturbance_norms,
        'pct_converged': 100 * np.mean(conv_steps < n_steps),
    }


def print_trial_result(r: dict, n_steps: int, dt: float, conv_thr: float,
                       show_dist: bool = False):
    Kp  = r['Kp']
    ct  = r['conv_times']
    fe  = r['final_err']
    lm  = r['late_max']
    pct = r['pct_converged']
    err0 = r['err_traces'][:, 0]

    label = ("UNSTABLE" if (pct < 50 or lm.mean() > 1.0)
             else "CONVERGED" if pct >= 95 else "PARTIAL")

    print(f"\n  Kp = {Kp:.4f}  [{label}]")
    print(f"    Initial  ‖ω−ω*‖: {err0.mean():.3f} ± {err0.std():.3f} rad/s")
    print(f"    Final    ‖ω−ω*‖: {fe.mean():.4f} ± {fe.std():.4f} rad/s")
    print(f"    Late max ‖ω−ω*‖: {lm.mean():.4f} ± {lm.std():.4f} rad/s  "
          f"(2nd half of traj)")
    if pct > 0:
        print(f"    Conv time (<{conv_thr:.1f}): "
              f"{ct.mean():.3f} ± {ct.std():.3f} s  "
              f"(min={ct.min():.2f}  max={ct.max():.2f})")
    print(f"    Converged: {pct:.0f}% of {len(fe)} trials "
          f"within {n_steps*dt:.1f}s")
    if show_dist:
        dn = r['disturbance_norms']
        print(f"    Disturbance |d|: {dn.mean():.4f} ± {dn.std():.4f} N·m  "
              f"(expected SS offset ≈ |d|/Kp = {dn.mean()/Kp:.3f} rad/s)")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runners
# ─────────────────────────────────────────────────────────────────────────────

def scenario_A(args, I1, I2, I3, env_clean):
    """Redirect from e₂ tumbling → e₁. No wind."""
    print(f"\n{'─'*60}")
    print(f"Scenario A: Stabilise e₁  (from e₂ tumbling, no wind)")
    print(f"  Target ω* = ({args.omega_star:.4f}, 0, 0) rad/s")
    print(f"  IC: axis=1 (Dzhanibekov tumbling)   Kp = 0.10")
    print(f"{'─'*60}")

    omega_star = np.array([args.omega_star, 0.0, 0.0])
    Kp = 0.10

    r = run_trials(
        env=env_clean, omega_star_vec=omega_star, Kp=Kp,
        n_trials=args.n_trials, n_steps=args.n_steps,
        conv_thr=args.conv_thr, seed=args.seed,
        ic_mode="axis", ic_axis=1,
    )
    print_trial_result(r, args.n_steps, env_clean.dt, args.conv_thr)

    # Error profile
    print(f"\n  Error profile (avg over {args.n_trials} trials):")
    print(f"  {'t [s]':>8}  {'mean ‖ω-ω*‖':>14}  {'max ‖ω-ω*‖':>14}")
    steps = list(range(0, min(21, args.n_steps + 1), 2))
    if args.n_steps not in steps:
        steps.append(args.n_steps)
    for step in steps:
        t_s  = step * env_clean.dt
        mean_ = r['err_traces'][:, step].mean()
        max_  = r['err_traces'][:, step].max()
        print(f"  {t_s:8.2f}  {mean_:14.6f}  {max_:14.6f}")

    return r


def scenario_B(args, I1, I2, I3, env_clean):
    """Hold e₂ from near-target IC. No wind. K_p scan."""
    print(f"\n{'─'*60}")
    print(f"Scenario B: Hold e₂  (start near ω*, no wind)")
    print(f"  Target ω* = (0, {args.omega_star:.4f}, 0) rad/s")
    print(f"  IC: ω* + N(0, {args.hold_perturb:.2f}²)  [perturbation about e₂]")
    Kp_th = kp_min_e2(I1, I2, I3, args.omega_star)
    print(f"  Theoretical K_p_min = {Kp_th:.4f} N·m·s/rad")
    print(f"{'─'*60}")

    omega_star = np.array([0.0, args.omega_star, 0.0])

    # Scan: below threshold, near threshold, well above, and beyond ZOH bound
    Kp_list = [0.010, 0.015, 0.020, 0.025, 0.030, 0.050, 0.100, 0.500]
    results = {}
    for Kp in Kp_list:
        r = run_trials(
            env=env_clean, omega_star_vec=omega_star, Kp=Kp,
            n_trials=args.n_trials, n_steps=args.n_steps,
            conv_thr=args.conv_thr, seed=args.seed,
            ic_mode="hold", hold_perturb=args.hold_perturb,
        )
        print_trial_result(r, args.n_steps, env_clean.dt, args.conv_thr)
        results[Kp] = r

    # Find empirical threshold (smallest Kp with ≥95% convergence)
    kp_emp = None
    for Kp in sorted(results):
        r = results[Kp]
        if r['pct_converged'] >= 95 and r['late_max'].mean() < 0.5:
            kp_emp = Kp
            break

    kp_max = kp_max_zoh(I1, env_clean.dt)
    print(f"\n  Summary:")
    print(f"    Theoretical K_p_min (saddle)  = {Kp_th:.4f}")
    print(f"    Theoretical K_p_max (ZOH)     = {kp_max:.4f}  "
          f"[2·I₁/T = 2×{I1:.4f}/{env_clean.dt:.3f}]")
    if kp_emp is not None:
        print(f"    Empirical   K_p_min ≤ {kp_emp:.4f}  "
              f"(first Kp with ≥95% convergence + stable hold)")
    else:
        print(f"    Empirical   K_p_min > {max(Kp_list):.4f}  "
              "(no Kp in scan converged reliably)")
    # Check ZOH instability empirically
    if 0.500 in results:
        r_zoh = results[0.500]
        zoh_unstable = r_zoh['late_max'].mean() > 1.0 and r_zoh['final_err'].mean() > 1.0
        print(f"    ZOH instability at K_p=0.50: "
              f"{'CONFIRMED ✓' if zoh_unstable else 'not observed'}"
              f"  (late_max={r_zoh['late_max'].mean():.2f} rad/s)")

    return results


def scenario_C(args, I1, I2, I3, wind_std):
    """Hold e₂ with stochastic wind. K_p scan."""
    print(f"\n{'─'*60}")
    print(f"Scenario C: Hold e₂ with wind")
    print(f"  Target ω* = (0, {args.omega_star:.4f}, 0) rad/s")
    print(f"  IC: ω* + N(0, {args.hold_perturb:.2f}²)")
    print(f"  Wind: disturbance_torque_std = {wind_std:.3f} N·m  (constant per episode)")
    Kp_th = kp_min_e2(I1, I2, I3, args.omega_star)
    print(f"  K_p_min (no-wind) = {Kp_th:.4f} N·m·s/rad")
    print(f"{'─'*60}")

    omega_star = np.array([0.0, args.omega_star, 0.0])
    Kp_list = [0.05, 0.10, 0.20, 0.50]

    results = {}
    for Kp in Kp_list:
        # Build a fresh env with wind enabled
        env_wind = Env(disturbance_torque_std=wind_std, seed=args.seed)
        r = run_trials(
            env=env_wind, omega_star_vec=omega_star, Kp=Kp,
            n_trials=args.n_trials, n_steps=args.n_steps,
            conv_thr=args.conv_thr, seed=args.seed,
            ic_mode="hold", hold_perturb=args.hold_perturb,
        )
        print_trial_result(r, args.n_steps, env_wind.dt, args.conv_thr,
                           show_dist=True)
        env_wind.close()
        results[Kp] = r

    # Compare actual final error vs predicted steady-state offset
    print(f"\n  Final-error vs predicted SS offset (|d|/Kp):")
    print(f"  {'Kp':>8}  {'final err (mean)':>18}  {'predicted SS':>14}  {'ratio':>8}")
    for Kp, r in results.items():
        dn   = r['disturbance_norms'].mean()
        pred = dn / Kp
        fe   = r['final_err'].mean()
        ratio = fe / pred if pred > 0 else float('nan')
        print(f"  {Kp:8.4f}  {fe:18.4f}  {pred:14.4f}  {ratio:8.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    print("=" * 62)
    print("Stage E — e₁ stabilisation, e₂ hold, wind robustness")
    print("=" * 62)

    # Default env (no wind, cfg0)
    env_clean = Env(disturbance_torque_std=0.0, seed=args.seed)
    iinfo = env_clean.get_inertia_info()
    I1, I2, I3 = iinfo['I1'], iinfo['I2'], iinfo['I3']

    print(f"\nRacket (cfg0):")
    print(f"  I1={I1:.6f}  I2={I2:.6f}  I3={I3:.6f}  kg·m²")
    print(f"  I2/I1={I2/I1:.2f}   asym=(I3-I1)/I2={(I3-I1)/I2:.2f}")
    print(f"\nTrials: {args.n_trials}  ×  {args.n_steps} steps "
          f"(dt={env_clean.dt:.2f}s → {args.n_steps*env_clean.dt:.1f}s per trial)")
    print(f"Convergence threshold: ‖ω-ω*‖ < {args.conv_thr:.2f} rad/s")

    # Stability table
    print(f"\n── Free-body stability (ω* = {args.omega_star:.4f} rad/s, cfg0) ──")
    print_stability_table(I1, I2, I3, args.omega_star, env_clean.dt)

    # Run all three scenarios
    resA = scenario_A(args, I1, I2, I3, env_clean)
    resB = scenario_B(args, I1, I2, I3, env_clean)
    resC = scenario_C(args, I1, I2, I3, args.wind_std)

    # ── Final verdict ──────────────────────────────────────────────────────
    A_pass = resA['pct_converged'] >= 95
    # Scenario B: K_p=0.10 is always in the scan
    B_r010 = resB.get(0.100, resB[max(resB)])
    B_pass = B_r010['pct_converged'] >= 95 and B_r010['late_max'].mean() < 0.5
    # Find best Kp in Scenario C (highest convergence + stable hold)
    C_best_kp = max(
        (k for k in resC if resC[k]['late_max'].mean() < 0.5),
        key=lambda k: resC[k]['pct_converged'],
        default=None,
    )
    C_r_best = resC[C_best_kp] if C_best_kp is not None else resC[min(resC)]
    C_pass = C_r_best['pct_converged'] >= 95

    print(f"\n{'='*62}")
    print(f"Stage E summary  (Kp=0.10 unless noted)")
    print(f"  A — e₁ stabilisation:          {'PASS ✓' if A_pass else 'FAIL ✗'}")
    print(f"  B — e₂ hold (no wind):         {'PASS ✓' if B_pass else 'FAIL ✗'}")
    kp_max = kp_max_zoh(I1, env_clean.dt)
    print(f"  C — e₂ hold (wind {args.wind_std:.2f} N·m):   "
          f"{'PASS ✓' if C_pass else 'FAIL ✗'}"
          + (f"  [best Kp={C_best_kp:.3f}]" if C_best_kp else ""))
    print(f"{'='*62}")

    env_clean.close()


if __name__ == '__main__':
    main()

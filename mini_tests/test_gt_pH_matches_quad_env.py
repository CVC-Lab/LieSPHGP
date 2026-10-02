"""Ground-truth port-Hamiltonian structure tests for the SE(3) quadrotor env.

Implements tests 11-16 of  notes/quadrotor-se3-ph-system.md  §20.2, i.e. the
"pH-structure level" verification that was still pending after the env-level
smoke tests and the gym-pybullet-drones cross-validation (§20.1).

The SE(3) analogue of mini_tests/test_gt_pH_matches_env.py: instead of replacing the
*model* subnetworks with ground truth, we rebuild the *environment* drift from
the hand-derived port-Hamiltonian structure matrices and check that the two
agree.

    state    x = ( x_w(3),  vec(R)(9),  p(3),  Pi(3) )  in R^18
    energy   H = |p|^2/(2m) + Pi^T J^-1 Pi / 2 + m g z
    drift    xdot = (Jcal(x) - Rcal) dH/dx + Gcal u + wind port
    noise    Xi(q) = [ 0_{12x6} ; blkdiag(sigma_f R^T, sigma_tau I3) ]

Tests
-----
11  GT-pH vs env         structure matrices reproduce _compute_rates exactly
12  Energy conservation   D=0, u=0, wind off  ->  H conserved; error O(h^3)
13  Momentum conservation horizontal P_w, ballistic P_w,z, world L_w = R Pi
14  Trivialized gradients autodiff T_R(H) and R^T grad_x V vs closed forms
15  Ito == Stratonovich   Heun vs Euler-Maruyama under common random numbers
16  Passivity / balance   dH/dt == -xi^T D xi + y^T u + xdot_w . F_wind

Run:  python mini_tests/test_gt_pH_matches_quad_env.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
ENVS_DIR = os.path.join(PROJECT_ROOT, "envs")
QUAD_DIR = os.path.join(ENVS_DIR, "SE3_quadrotor")
for _p in (PROJECT_ROOT, ENVS_DIR, QUAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quadrotor import quadrotor_se3, _hat, _exp_so3, _random_rotation  # noqa: E402

E3 = np.array([0.0, 0.0, 1.0])

# Collected (name, passed, headline) rows for the closing summary table.
RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, headline: str) -> None:
    RESULTS.append((name, bool(passed), headline))
    print(f"\n  -> {'PASS' if passed else 'FAIL'}   {headline}")


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _ratios(errs: list[float]) -> list[float]:
    return [errs[i] / max(errs[i + 1], 1e-300) for i in range(len(errs) - 1)]


# ══════════════════════════════════════════════════════════════════════════
# Ground-truth port-Hamiltonian building blocks  (notes §4, §6.1, §7, §8, §9)
# ══════════════════════════════════════════════════════════════════════════

def B_of_R(R: np.ndarray) -> np.ndarray:
    """B(q) = [[r1]x ; [r2]x ; [r3]x] in R^{9x3}, where r_k^T is row k of R.

    This is the tangent map of the group action: (B omega) reshaped to 3x3
    equals R [omega]_x = Rdot.
    """
    return np.vstack([_hat(R[k, :]) for k in range(3)])


def M_GT(m: float, J: np.ndarray) -> np.ndarray:
    """Mass matrix  Mcal = blkdiag(m I3, J)  -- constant (notes §4.1)."""
    Z = np.zeros((3, 3))
    return np.block([[m * np.eye(3), Z], [Z, J]])


def V_GT(x_w: np.ndarray, m: float, g: float) -> float:
    """Potential  V = m g e3^T x_w  -- depends on height only (notes §4.2)."""
    return m * g * float(x_w[2])


def H_GT(x_w, p, Pi, m, g, J_inv) -> float:
    """H = |p|^2/(2m) + Pi^T J^-1 Pi / 2 + m g z   (notes §4.3)."""
    return 0.5 * float(p @ p) / m + 0.5 * float(Pi @ (J_inv @ Pi)) + m * g * float(x_w[2])


def dH_dx_GT(x_w, R, p, Pi, m, g, J_inv) -> np.ndarray:
    """dH/dx for x = (x_w, vec(R), p, Pi).  Note dH/dvec(R) = 0 exactly."""
    out = np.zeros(18)
    out[0:3] = m * g * E3
    out[3:12] = 0.0                      # H does not depend on R  (notes §5)
    out[12:15] = p / m                   # = v_b
    out[15:18] = J_inv @ Pi              # = omega_b
    return out


def struct_J_GT(R, p, Pi) -> np.ndarray:
    """The 18x18 skew structure matrix Jcal(x) of notes §6.1."""
    Jc = np.zeros((18, 18))
    B = B_of_R(R)
    Jc[0:3, 12:15] = R                   # xdot_w   = R v_b
    Jc[3:12, 15:18] = B                  # vec(Rdot) = B(q) omega
    Jc[12:15, 0:3] = -R.T                # gravity force, body frame
    Jc[12:15, 15:18] = _hat(p)           # p x omega
    Jc[15:18, 3:12] = -B.T               # T_R(H)   (zero here)
    Jc[15:18, 12:15] = _hat(p)           # p x v_b  (silently zero)
    Jc[15:18, 15:18] = _hat(Pi)          # Pi x omega
    return Jc


def struct_R_GT(d_lin: float, d_ang: float) -> np.ndarray:
    """Rcal = blkdiag(0_12, D),  D = blkdiag(d_lin I3, d_ang I3)  (notes §7)."""
    Rc = np.zeros((18, 18))
    Rc[12:15, 12:15] = d_lin * np.eye(3)
    Rc[15:18, 15:18] = d_ang * np.eye(3)
    return Rc


def struct_G_GT(G: np.ndarray) -> np.ndarray:
    """Gcal = [0_{12x4} ; G]  with the constant 6x4 mixer of notes §8."""
    Gc = np.zeros((18, 4))
    Gc[12:18, :] = G
    return Gc


def Xi_GT(R: np.ndarray, sigma_f: float, sigma_tau: float) -> np.ndarray:
    """Diffusion Xi(q) in R^{18x6}: force channel multiplicative, torque additive."""
    Xi = np.zeros((18, 6))
    Xi[12:15, 0:3] = sigma_f * R.T
    Xi[15:18, 3:6] = sigma_tau * np.eye(3)
    return Xi


def ph_rhs_GT(x_w, R, v_b, omega, u, w_mag, d_hat,
              m, g, J, G, d_lin, d_ang):
    """Full pH right-hand side assembled from the structure matrices.

    Returns (xdot_18, Jcal) where xdot_18 = (Jcal - Rcal) dH/dx + Gcal u + port.
    """
    J_inv = np.linalg.inv(J)
    p = m * v_b
    Pi = J @ omega

    dHdx = dH_dx_GT(x_w, R, p, Pi, m, g, J_inv)
    Jc = struct_J_GT(R, p, Pi)
    Rc = struct_R_GT(d_lin, d_ang)
    Gc = struct_G_GT(G)

    port = np.zeros(18)                       # deterministic wind (notes §9.1)
    port[12:15] = R.T @ (w_mag * d_hat)

    xdot = (Jc - Rc) @ dHdx + Gc @ u + port
    return xdot, Jc


# ══════════════════════════════════════════════════════════════════════════
# Test 11 -- GT-pH structure matrices vs the environment's _compute_rates
# ══════════════════════════════════════════════════════════════════════════

def test_11_gt_ph_vs_env(n_states: int = 200, seed: int = 11) -> bool:
    _banner("Test 11  --  GT port-Hamiltonian structure vs env  (notes §6.1)")
    rng = np.random.default_rng(seed)

    env = quadrotor_se3(
        m=1.3, J_diag=(0.4, 0.6, 0.9), arm=0.8, g=9.81,
        kf_coeff=1.1, kf_std=0.2,
        km_coeff=0.13, km_std=0.03,
        linear_damping_coeff=0.35, linear_damping_std=0.1,
        angular_damping_coeff=0.22, angular_damping_std=0.05,
        external_force_type="sine", external_force_std=2.5,
        external_force_direction=(0.3, -0.8, 0.5),
        wind_force_std=0.7, wind_torque_std=0.25,
        seed=seed,
    )

    err_x, err_v, err_om, err_R, err_dV, err_dOm = [], [], [], [], [], []
    err_M, err_V = [], []
    skew_err, psd_min = [], []
    scale_v, scale_om = [], []

    for _ in range(n_states):
        # Random state, input, wind phase and (randomised) coefficients
        env._resample_coeffs()
        x_w = rng.uniform(-5.0, 5.0, size=3)
        R = _random_rotation(rng)
        v_b = rng.uniform(-3.0, 3.0, size=3)
        omega = rng.uniform(-4.0, 4.0, size=3)
        u = rng.uniform(0.0, 20.0, size=4)
        w_mag = float(rng.uniform(-3.0, 3.0))
        dW_f = rng.normal(0.0, 0.1, size=3)
        dW_tau = rng.normal(0.0, 0.1, size=3)

        # ── environment ──
        x_dot_e, v_dot_e, om_dot_e, dV_e, dOm_e = env._compute_rates(
            x_w, R, v_b, omega, w_mag, u, dW_f, dW_tau
        )

        # ── ground-truth pH ──
        xdot, Jc = ph_rhs_GT(
            x_w, R, v_b, omega, u, w_mag, env.external_force_direction,
            env.m, env.g, env.J, env.G, env._d_lin, env._d_ang,
        )
        x_dot_g = xdot[0:3]
        vecRdot_g = xdot[3:12]
        v_dot_g = xdot[12:15] / env.m                 # pdot = m vdot
        om_dot_g = np.linalg.inv(env.J) @ xdot[15:18]  # Pidot = J omegadot

        # Diffusion: Xi dW mapped from (p, Pi) to (v, omega)
        Xi = Xi_GT(R, env.wind_force_std, env.wind_torque_std)
        dPI = Xi @ np.hstack([dW_f, dW_tau])
        dV_g = dPI[12:15] / env.m
        dOm_g = np.linalg.inv(env.J) @ dPI[15:18]

        # Mcal (§4.1) must map the body velocity to the momenta used above,
        # and dH/dx_w must be exactly grad V_GT (§4.2).
        xi = np.hstack([v_b, omega])
        err_M.append(np.abs(M_GT(env.m, env.J) @ xi
                            - np.hstack([env.m * v_b, env.J @ omega])).max())
        dV_num = ((V_GT(x_w + 1e-6 * E3, env.m, env.g)
                   - V_GT(x_w - 1e-6 * E3, env.m, env.g)) / 2e-6)
        err_V.append(abs(dV_num - env.m * env.g))

        err_x.append(np.abs(x_dot_e - x_dot_g).max())
        err_v.append(np.abs(v_dot_e - v_dot_g).max())
        err_om.append(np.abs(om_dot_e - om_dot_g).max())
        err_R.append(np.abs(vecRdot_g - (R @ _hat(omega)).reshape(9)).max())
        err_dV.append(np.abs(dV_e - dV_g).max())
        err_dOm.append(np.abs(dOm_e - dOm_g).max())

        skew_err.append(np.abs(Jc + Jc.T).max())
        psd_min.append(np.linalg.eigvalsh(
            struct_R_GT(env._d_lin, env._d_ang)).min())

        scale_v.append(np.abs(v_dot_e).max())
        scale_om.append(np.abs(om_dot_e).max())

    e = dict(x=max(err_x), v=max(err_v), om=max(err_om),
             R=max(err_R), dV=max(err_dV), dOm=max(err_dOm))
    sv, so = max(scale_v), max(scale_om)

    print(f"  random states tested        : {n_states}")
    print(f"  max |Mcal xi - (p, Pi)|     : {max(err_M):.3e}   (§4.1 mass matrix)")
    print(f"  max |dV_GT/dz - m g|        : {max(err_V):.3e}   (§4.2 potential)")
    print(f"  max |xdot_w  env - GT|      : {e['x']:.3e}")
    print(f"  max |vec(Rdot) GT - R[w]x|  : {e['R']:.3e}   (B(q) block)")
    print(f"  max |vdot_b  env - GT|      : {e['v']:.3e}   (scale {sv:.2f}"
          f"  -> rel {e['v'] / sv:.3e})")
    print(f"  max |omdot_b env - GT|      : {e['om']:.3e}   (scale {so:.2f}"
          f"  -> rel {e['om'] / so:.3e})")
    print(f"  max |dV_stoch  env - Xi dW| : {e['dV']:.3e}")
    print(f"  max |dOm_stoch env - Xi dW| : {e['dOm']:.3e}")
    print(f"  max |Jcal + Jcal^T|         : {max(skew_err):.3e}   (skew-symmetry)")
    print(f"  min eig(Rcal)               : {min(psd_min):.3e}   (>= 0 required)")

    tol = 1e-12
    passed = (max(e.values()) < tol and max(err_M) < tol and max(err_V) < 1e-6
              and max(skew_err) < 1e-15 and min(psd_min) >= 0.0)
    _record("11 GT-pH vs env", passed,
            f"all drift/diffusion blocks agree to {max(e.values()):.1e} "
            f"(tol {tol:.0e}); Jcal skew, Rcal PSD")
    env.close()
    return passed


# ══════════════════════════════════════════════════════════════════════════
# Test 12 -- energy conservation (D = 0, u = 0, wind off)
# ══════════════════════════════════════════════════════════════════════════

def _conservative_env(dt: float, seed: int = 12, g: float = 9.81) -> quadrotor_se3:
    return quadrotor_se3(
        m=1.0, J_diag=(0.5, 0.5, 1.0), arm=1.0, g=g, dt=dt,
        kf_coeff=1.0, km_coeff=0.1,
        linear_damping_coeff=0.0, angular_damping_coeff=0.0,
        external_force_type="constant", external_force_std=0.0,
        wind_force_std=0.0, wind_torque_std=0.0,
        seed=seed,
    )


_IC = dict(x_init=[0.0, 0.0, 10.0],
           v_init=[0.6, -0.4, 0.3],
           omega_init=[1.2, -0.7, 0.9])
_R0 = _exp_so3(np.array([0.3, -0.2, 0.7]))


def _energy(env: quadrotor_se3) -> float:
    x, R, v, om = env.get_state()
    return (0.5 * env.m * float(v @ v)
            + 0.5 * float(om @ (env.J @ om))
            + env.m * env.g * float(x[2]))


def test_12_energy_conservation(seed: int = 12) -> bool:
    _banner("Test 12  --  Energy conservation, D=0, u=0, wind off  (notes §13.3)")
    u0 = np.zeros(4)

    def sweep(g, T, dts, label):
        print(f"  {label}")
        drifts, rels = [], []
        for dt in dts:
            env = _conservative_env(dt, seed, g=g)
            env.reset(seed=seed, options=dict(_IC, R_init=_R0))
            H0 = _energy(env)
            worst = 0.0
            for _ in range(int(round(T / dt))):
                env.step(u0)
                worst = max(worst, abs(_energy(env) - H0))
            env.close()
            drifts.append(worst)
            rels.append(worst / abs(H0))
            print(f"      dt={dt:<7g} h={dt / 10:<8.4g} H0={H0:8.4f} J   "
                  f"max|H-H0| = {worst:.4e} J   (rel {worst / abs(H0):.3e})")
        rr = _ratios(drifts)
        print(f"      error ratios per halving of h : "
              + ", ".join(f"{r:.2f}" for r in rr))
        return drifts, rels, rr

    # 12a -- gravity on: the drone free-falls, PE <-> KE exchange is exercised
    _, rel_a, r_a = sweep(9.81, 1.0, [0.02, 0.01, 0.005],
                          "12a  gravity on (free fall), T = 1 s:")
    # 12b -- g = 0: |v_b| and omega stay bounded, so a long horizon is a clean
    #        secular-drift check uncontaminated by the growing free-fall state
    _, rel_b, r_b = sweep(0.0, 20.0, [0.02, 0.01, 0.005],
                          "12b  g = 0 (bounded state), T = 20 s, secular-drift check:")

    print("      NOTE ratios ~ 8 mean the ENERGY error is O(h^3) -- one order")
    print("           better than the O(h^2) state error (test 13, and the")
    print("           Heun self-convergence ratio 4.03 of §20.1).  The leading")
    print("           O(h^2) state error is tangent to the energy level set.")

    passed = (all(6.5 < r < 9.5 for r in r_a + r_b)
              and rel_a[-1] < 1e-9 and rel_b[-1] < 1e-9)
    _record("12 Energy conservation", passed,
            f"rel drift {rel_a[-1]:.1e} (free fall, T=1s) / {rel_b[-1]:.1e} "
            f"(bounded, T=20s); ratios "
            + "/".join(f"{r:.1f}" for r in r_a + r_b) + " ~ 8, i.e. O(h^3)")
    return passed


# ══════════════════════════════════════════════════════════════════════════
# Test 13 -- momentum conservation (D = 0, u = 0, wind off)
# ══════════════════════════════════════════════════════════════════════════

def test_13_momentum_conservation(T: float = 1.0, seed: int = 13) -> bool:
    _banner("Test 13  --  Momentum conservation, D=0, u=0, wind off  (notes §13.3)")
    u0 = np.zeros(4)

    dts = [0.02, 0.01, 0.005]
    err_hz, err_ball, err_L = [], [], []

    for dt in dts:
        env = _conservative_env(dt, seed)
        env.reset(seed=seed, options=dict(_IC, R_init=_R0))
        x0, R_0, v0, om0 = env.get_state()
        P0 = env.m * (R_0 @ v0)             # world linear momentum
        L0 = R_0 @ (env.J @ om0)            # world angular momentum
        e_hz = e_ball = e_L = 0.0
        for k in range(int(round(T / dt))):
            env.step(u0)
            x, R, v, om = env.get_state()
            P = env.m * (R @ v)
            L = R @ (env.J @ om)
            t = (k + 1) * dt
            e_hz = max(e_hz, np.abs(P[:2] - P0[:2]).max())
            e_ball = max(e_ball, abs((P[2] + env.m * env.g * t) - P0[2]))
            e_L = max(e_L, np.abs(L - L0).max())
        err_hz.append(e_hz)
        err_ball.append(e_ball)
        err_L.append(e_L)
        print(f"  dt={dt:<7g}  max|dP_xy| = {e_hz:.3e}   "
              f"max|dP_z + m g t| = {e_ball:.3e}   max|d(R Pi)| = {e_L:.3e}")
        env.close()

    r_hz, r_ball, r_L = _ratios(err_hz), _ratios(err_ball), _ratios(err_L)
    print(f"  ratios  P_xy  (const)      : " + ", ".join(f"{r:.2f}" for r in r_hz))
    print(f"  ratios  P_z   (ballistic)  : " + ", ".join(f"{r:.2f}" for r in r_ball))
    print(f"  ratios  R Pi  (world L)    : " + ", ".join(f"{r:.2f}" for r in r_L))
    print("  All three are conserved in continuous time; the residual is the")
    print("  integrator's O(h^2) STATE error, hence ratio 4 (not 8 as for H).")

    passed = all(3.2 < r < 4.8 for r in r_hz + r_ball + r_L)
    _record("13 Momentum conservation", passed,
            f"P_xy {err_hz[-1]:.1e}, ballistic P_z {err_ball[-1]:.1e}, "
            f"R.Pi {err_L[-1]:.1e} at h=5e-4 s; all ratios ~ 4 (O(h^2))")
    return passed


# ══════════════════════════════════════════════════════════════════════════
# Test 14 -- trivialized gradients on SE(3)
# ══════════════════════════════════════════════════════════════════════════

def test_14_trivialized_gradients(seed: int = 14) -> bool:
    _banner("Test 14  --  Trivialized gradients on SE(3)  (notes §5)")
    rng = np.random.default_rng(seed)

    try:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        have_jax = True
    except Exception as exc:                                   # pragma: no cover
        print(f"  [jax unavailable: {exc}; falling back to central differences]")
        have_jax = False

    m, g = 1.3, 9.81
    J = np.diag([0.4, 0.6, 0.9])
    J_inv = np.linalg.inv(J)

    def T_R(dF_dvecR: np.ndarray, R: np.ndarray) -> np.ndarray:
        """T_R(F) = sum_k r_k x dF/dr_k  (the code's cross(R, dHdq).sum)."""
        A = dF_dvecR.reshape(3, 3)
        return np.sum(np.cross(R, A, axis=-1), axis=0)

    # ── 14a: validate the operator itself on a NON-trivial test function ──
    # F(R) = tr(A^T R)  =>  dF/dr_k = a_k, and the group-action derivative
    #   d/dt F(R exp(t[phi]x))|_0  must equal  -T_R(F) . phi   (notes §5).
    err_op = 0.0
    for _ in range(50):
        A = rng.normal(size=(3, 3))
        R = _random_rotation(rng)
        phi = rng.normal(size=3)

        T_closed = T_R(A.reshape(9), R)
        eps = 1e-6
        F = lambda M: float(np.sum(A * M))
        d_num = (F(R @ _exp_so3(eps * phi)) - F(R @ _exp_so3(-eps * phi))) / (2 * eps)
        err_op = max(err_op, abs(d_num - (-T_closed @ phi)))
    print(f"  14a operator check on F(R)=tr(A^T R)  (central differences, eps=1e-6):")
    print(f"      max | d/dt F(R exp(t[phi]x)) + T_R(F).phi | = {err_op:.3e}")

    # ── 14b: the quadrotor's own claim -- T_R(H) = 0 and R^T grad_x V = m g R^T e3 ──
    err_TRH, err_gradV = 0.0, 0.0
    for _ in range(50):
        x_w = rng.uniform(-5, 5, size=3)
        R = _random_rotation(rng)
        p = rng.uniform(-3, 3, size=3)
        Pi = rng.uniform(-3, 3, size=3)

        if have_jax:
            def H_flat(z):
                xw, vr, pp, PP = z[0:3], z[3:12], z[12:15], z[15:18]
                return (0.5 * jnp.dot(pp, pp) / m
                        + 0.5 * jnp.dot(PP, jnp.linalg.solve(jnp.asarray(J), PP))
                        + m * g * xw[2]) + 0.0 * jnp.sum(vr)
            z = jnp.asarray(np.hstack([x_w, R.reshape(9), p, Pi]))
            gradH = np.asarray(jax.grad(H_flat)(z))
        else:                                                  # pragma: no cover
            z = np.hstack([x_w, R.reshape(9), p, Pi])
            gradH = np.zeros(18)
            eps = 1e-7
            for i in range(18):
                zp, zm = z.copy(), z.copy()
                zp[i] += eps
                zm[i] -= eps
                fp = H_GT(zp[0:3], zp[12:15], zp[15:18], m, g, J_inv)
                fm = H_GT(zm[0:3], zm[12:15], zm[15:18], m, g, J_inv)
                gradH[i] = (fp - fm) / (2 * eps)

        err_TRH = max(err_TRH, np.abs(T_R(gradH[3:12], R)).max())
        err_gradV = max(err_gradV,
                        np.abs(R.T @ gradH[0:3] - m * g * (R.T @ E3)).max())

    print(f"  14b quadrotor Hamiltonian ({'jax autodiff' if have_jax else 'central diff'}):")
    print(f"      max |T_R(H)|                        = {err_TRH:.3e}   "
          f"(claim: exactly 0 -- gravity at the COM exerts no torque)")
    print(f"      max |R^T grad_x V - m g R^T e3|     = {err_gradV:.3e}")

    # ── 14c: the same claim read straight off the environment ──
    env = quadrotor_se3(m=m, J_diag=(0.4, 0.6, 0.9), g=g,
                        kf_coeff=1.0, km_coeff=0.1,
                        linear_damping_coeff=0.0, angular_damping_coeff=0.0,
                        external_force_type="constant", external_force_std=0.0,
                        wind_force_std=0.0, wind_torque_std=0.0, seed=seed)
    err_env_f, err_env_t = 0.0, 0.0
    for _ in range(50):
        R = _random_rotation(rng)
        z3 = np.zeros(3)
        _, v_dot, om_dot, _, _ = env._compute_rates(
            rng.uniform(-5, 5, size=3), R, z3, z3, 0.0, np.zeros(4), z3, z3)
        err_env_f = max(err_env_f, np.abs(v_dot - (-g * (R.T @ E3))).max())
        err_env_t = max(err_env_t, np.abs(om_dot).max())
    env.close()
    print(f"  14c env at rest, u=0, wind off:")
    print(f"      max |vdot_b + g R^T e3|             = {err_env_f:.3e}")
    print(f"      max |omdot_b|  (no gravity torque)  = {err_env_t:.3e}")

    passed = (err_op < 1e-8 and err_TRH < 1e-12
              and err_gradV < 1e-12 and err_env_f < 1e-14 and err_env_t < 1e-14)
    _record("14 Trivialized gradients", passed,
            f"operator exact to {err_op:.1e} (FD-limited); T_R(H)={err_TRH:.1e}, "
            f"R^T grad V matches to {err_gradV:.1e}")
    return passed


# ══════════════════════════════════════════════════════════════════════════
# Test 15 -- Ito == Stratonovich  (the vanishing-correction claim of §10.1)
# ══════════════════════════════════════════════════════════════════════════

class _BrokenItoQuad(quadrotor_se3):
    """Deliberately broken control variant: the force-channel diffusion is
    scaled by |v_b|, so it depends on the MOMENTUM and the Ito correction is
    genuinely non-zero.  Used only to show test 15 can detect such a term.
    """

    def _compute_rates(self, x_w, R, v, omega, w, u, dW_f, dW_tau):
        x_dot, v_dot, om_dot, dV, dOm = super()._compute_rates(
            x_w, R, v, omega, w, u, dW_f, dW_tau)
        return x_dot, v_dot, om_dot, dV * float(np.linalg.norm(v)), dOm


def _em_step(env, x_w, R, v, om, w, u, h, dW_f, dW_tau):
    """One Euler-Maruyama (Ito) step: diffusion frozen at the left endpoint."""
    x_dot, v_dot, om_dot, dV, dOm = env._compute_rates(
        x_w, R, v, om, w, u, dW_f, dW_tau)
    return (x_w + x_dot * h,
            R @ _exp_so3(om * h),
            v + v_dot * h + dV,
            om + om_dot * h + dOm)


def _roll(env, integrator, X0, u, T, n_steps, dW_f, dW_tau):
    """Roll one path with a prescribed Wiener increment sequence."""
    x_w, R, v, om = (X0[0].copy(), X0[1].copy(), X0[2].copy(), X0[3].copy())
    h = T / n_steps
    for k in range(n_steps):
        if integrator == "heun":
            x_w, R, v, om = env._lie_heun_step(
                x_w, R, v, om, 0.0, u, h, dW_f[k], dW_tau[k])
        else:
            x_w, R, v, om = _em_step(
                env, x_w, R, v, om, 0.0, u, h, dW_f[k], dW_tau[k])
    return x_w, R, v, om


def _coarsen(fine, n_steps):
    """Sum fine Wiener increments into n_steps coarse ones (same path)."""
    grp = fine.shape[0] // n_steps
    return fine.reshape(n_steps, grp, 3).sum(axis=1)


def _gap_vec(env, X0, u, T, n_steps, dW_f, dW_tau):
    a = _roll(env, "heun", X0, u, T, n_steps, dW_f, dW_tau)
    b = _roll(env, "em", X0, u, T, n_steps, dW_f, dW_tau)
    return np.hstack([a[0] - b[0], a[2] - b[2], a[3] - b[3]])


def test_15_ito_equals_stratonovich(T: float = 0.5, seed: int = 15) -> bool:
    _banner("Test 15  --  Ito == Stratonovich  (notes §10.1)")
    rng = np.random.default_rng(seed)

    kw = dict(m=1.0, J_diag=(0.5, 0.5, 1.0), arm=1.0, g=9.81,
              kf_coeff=1.0, km_coeff=0.1,
              linear_damping_coeff=0.3, angular_damping_coeff=0.2,
              external_force_type="constant", external_force_std=0.0,
              wind_force_std=2.0, wind_torque_std=0.8, seed=seed)
    env = quadrotor_se3(**kw)
    broken = _BrokenItoQuad(**kw)

    X0 = (np.array([0.0, 0.0, 10.0]),
          _exp_so3(np.array([0.2, -0.4, 0.1])),
          np.array([1.0, -0.5, 0.4]),
          np.array([0.5, -0.3, 0.8]))
    u = np.array([3.0, 2.5, 3.2, 2.8])

    print("  Claim: the Stratonovich->Ito correction vanishes, so Heun (Strat)")
    print("  and Euler-Maruyama (Ito) must converge to the SAME solution.")
    print("  A 'broken' control env whose diffusion depends on v_b (non-zero")
    print("  correction) is run alongside to show the test can detect one.")

    # ── 15a: strong gap under common random numbers, as h -> 0 ──
    n_fine = 3200
    h_fine = T / n_fine
    fine_f = rng.normal(0.0, np.sqrt(h_fine), size=(n_fine, 3))
    fine_t = rng.normal(0.0, np.sqrt(h_fine), size=(n_fine, 3))

    levels = [100, 200, 400, 800]
    print("\n  15a  single-path gap  ||X_Heun(T) - X_EM(T)||  (common noise)")
    print(f"       {'h':>10}  {'true env':>12}  {'broken control':>16}")
    gaps, gaps_b = [], []
    for n in levels:
        dW, dTq = _coarsen(fine_f, n), _coarsen(fine_t, n)
        gi = float(np.linalg.norm(_gap_vec(env, X0, u, T, n, dW, dTq)))
        gb = float(np.linalg.norm(_gap_vec(broken, X0, u, T, n, dW, dTq)))
        gaps.append(gi)
        gaps_b.append(gb)
        print(f"       {T / n:>10.2e}  {gi:>12.3e}  {gb:>16.3e}")
    shrink = gaps[0] / max(gaps[-1], 1e-300)
    shrink_b = gaps_b[0] / max(gaps_b[-1], 1e-300)
    print(f"       shrinkage over 8x refinement: true env {shrink:.1f}x"
          f"  |  broken {shrink_b:.2f}x")

    # ── 15b: paired ensemble mean of the same difference, vs h ──
    # At finite h the paired mean is NOT zero -- it is the O(h) discretisation
    # gap.  The claim is that it CONVERGES TO ZERO with h, which is what is
    # measured here (and what the broken control fails to do).
    n_paths = 120
    levels_b = [100, 200, 400]
    print(f"\n  15b  paired ensemble mean |mean(X_Heun - X_EM)|, {n_paths} paths,")
    print(f"       identical noise reused across every h")
    print(f"       {'h':>10}  {'true env':>12} {'(std err)':>11}"
          f"  {'broken':>12} {'(std err)':>11}")
    means, means_b = [], []
    for n in levels_b:
        acc, acc_b = np.zeros((n_paths, 9)), np.zeros((n_paths, 9))
        r2 = np.random.default_rng(4242)
        for i in range(n_paths):
            ff = r2.normal(0.0, np.sqrt(h_fine), size=(n_fine, 3))
            ft = r2.normal(0.0, np.sqrt(h_fine), size=(n_fine, 3))
            dW, dTq = _coarsen(ff, n), _coarsen(ft, n)
            acc[i] = _gap_vec(env, X0, u, T, n, dW, dTq)
            acc_b[i] = _gap_vec(broken, X0, u, T, n, dW, dTq)
        m_i = float(np.abs(acc.mean(axis=0)).max())
        s_i = float((acc.std(axis=0, ddof=1) / np.sqrt(n_paths)).max())
        m_b = float(np.abs(acc_b.mean(axis=0)).max())
        s_b = float((acc_b.std(axis=0, ddof=1) / np.sqrt(n_paths)).max())
        means.append(m_i)
        means_b.append(m_b)
        print(f"       {T / n:>10.2e}  {m_i:>12.3e} {s_i:>11.2e}"
              f"  {m_b:>12.3e} {s_b:>11.2e}")
    m_shrink = means[0] / max(means[-1], 1e-300)
    m_shrink_b = means_b[0] / max(means_b[-1], 1e-300)
    print(f"       shrinkage over 4x refinement: true env {m_shrink:.2f}x"
          f"  |  broken {m_shrink_b:.2f}x   (4x = O(h) -> 0)")

    env.close()
    broken.close()

    passed = (shrink > 5.0 and shrink_b < 1.5
              and m_shrink > 3.0 and m_shrink_b < 1.5)
    _record("15 Ito == Stratonovich", passed,
            f"gap -> 0 as O(h): pathwise {shrink:.1f}x per 8x refinement, "
            f"paired mean {m_shrink:.1f}x per 4x; broken control plateaus "
            f"({shrink_b:.2f}x / {m_shrink_b:.2f}x)")
    return passed


# ══════════════════════════════════════════════════════════════════════════
# Test 16 -- passivity and the energy-balance identity
# ══════════════════════════════════════════════════════════════════════════

def _power(env, u, w_mag):
    """Instantaneous supply rate  -xi^T D xi + y^T u + xdot_w . F_wind."""
    x, R, v, om = env.get_state()
    x_dot = R @ v
    xi = np.hstack([v, om])
    y = env.G.T @ xi
    return (-env._d_lin * float(v @ v) - env._d_ang * float(om @ om)
            + float(y @ u)
            + float(x_dot @ (w_mag * env.external_force_direction)))


def test_16_passivity(n_states: int = 200, seed: int = 16) -> bool:
    _banner("Test 16  --  Passivity / energy balance  (notes §11)")
    rng = np.random.default_rng(seed)

    env = quadrotor_se3(
        m=1.3, J_diag=(0.4, 0.6, 0.9), arm=0.8, g=9.81,
        kf_coeff=1.1, km_coeff=0.13,
        linear_damping_coeff=0.35, angular_damping_coeff=0.22,
        external_force_type="sine", external_force_std=2.5,
        external_force_direction=(0.3, -0.8, 0.5),
        wind_force_std=0.0, wind_torque_std=0.0, seed=seed,
    )

    # ── 16a: instantaneous identity, exact algebra ──
    err, scale = 0.0, 0.0
    for _ in range(n_states):
        x_w = rng.uniform(-5, 5, size=3)
        R = _random_rotation(rng)
        v_b = rng.uniform(-3, 3, size=3)
        om = rng.uniform(-4, 4, size=3)
        u = rng.uniform(0.0, 20.0, size=4)
        w_mag = float(rng.uniform(-3, 3))
        z3 = np.zeros(3)

        x_dot, v_dot, om_dot, _, _ = env._compute_rates(
            x_w, R, v_b, om, w_mag, u, z3, z3)

        # dH/dt read off the env's own rates
        Hdot_env = (env.m * float(v_b @ v_dot)
                    + float(om @ (env.J @ om_dot))
                    + env.m * env.g * float(x_dot[2]))

        # dH/dt predicted by the pH identity of §11
        xi = np.hstack([v_b, om])
        y = env.G.T @ xi                                # collocated output
        Hdot_pH = (-env._d_lin * float(v_b @ v_b)
                   - env._d_ang * float(om @ om)
                   + float(y @ u)
                   + float(x_dot @ (w_mag * env.external_force_direction)))

        err = max(err, abs(Hdot_env - Hdot_pH))
        scale = max(scale, abs(Hdot_env))
    env.close()

    print(f"  16a instantaneous balance over {n_states} random states:")
    print(f"      max |dH/dt(env) - (-xi^T D xi + y^T u + xdot_w.F_wind)| "
          f"= {err:.3e}")
    print(f"      typical |dH/dt| scale = {scale:.2f}   -> relative {err / scale:.3e}")

    # ── 16b: monotone decay with u = 0 (the hypothesis of §11) ──
    env2 = quadrotor_se3(
        m=1.0, J_diag=(0.5, 0.5, 1.0), arm=1.0, g=9.81, dt=0.02,
        kf_coeff=1.0, km_coeff=0.1,
        linear_damping_coeff=0.6, angular_damping_coeff=0.4,
        external_force_type="constant", external_force_std=0.0,
        wind_force_std=0.0, wind_torque_std=0.0, seed=seed,
    )
    env2.reset(seed=seed, options=dict(
        x_init=[0.0, 0.0, 5.0], R_init=_R0,
        v_init=[1.5, -1.0, 0.8], omega_init=[2.0, -1.5, 1.0]))
    Hs = [_energy(env2)]
    for _ in range(200):
        env2.step(np.zeros(4))
        Hs.append(_energy(env2))
    env2.close()
    dH = np.diff(np.asarray(Hs))
    print(f"  16b passivity check, u = 0, wind off, damping on:")
    print(f"      H: {Hs[0]:.4f} -> {Hs[-1]:.4f} J    "
          f"max step increment = {dH.max():+.3e} J   (must be <= 0)")

    # ── 16c: integrated balance over a driven rollout (u != 0, wind on) ──
    # H(T) - H(0) must equal the time integral of the supply rate.  Sampling
    # the integrand only at the env's snapshot times makes this a trapezoid
    # quadrature, so the residual is O(dt^2) -- ratio 4 per halving.
    u_drive = np.array([2.9, 2.3, 3.1, 2.5])
    w_const = 1.5
    print(f"  16c integrated balance, u != 0 and constant wind, T = 1 s:")
    gaps = []
    for dt in [0.02, 0.01, 0.005]:
        env3 = quadrotor_se3(
            m=1.0, J_diag=(0.5, 0.5, 1.0), arm=1.0, g=9.81, dt=dt,
            kf_coeff=1.0, km_coeff=0.1,
            linear_damping_coeff=0.5, angular_damping_coeff=0.3,
            external_force_type="constant", external_force_std=w_const,
            external_force_direction=(0.3, -0.8, 0.5),
            wind_force_std=0.0, wind_torque_std=0.0, seed=seed,
        )
        env3.reset(seed=seed, options=dict(
            x_init=[0.0, 0.0, 5.0], R_init=_R0,
            v_init=[0.5, -0.3, 0.2], omega_init=[0.4, -0.2, 0.3]))
        H_start = _energy(env3)
        Ps = [_power(env3, u_drive, w_const)]
        for _ in range(int(round(1.0 / dt))):
            env3.step(u_drive)
            Ps.append(_power(env3, u_drive, w_const))
        H_end = _energy(env3)
        env3.close()
        Ps = np.asarray(Ps)
        integral = dt * (Ps[0] / 2 + Ps[1:-1].sum() + Ps[-1] / 2)
        gap = abs((H_end - H_start) - integral)
        gaps.append(gap)
        print(f"      dt={dt:<7g}  dH = {H_end - H_start:+10.5f} J   "
              f"int P dt = {integral:+10.5f} J   |residual| = {gap:.3e}")
    r_c = _ratios(gaps)
    print(f"      residual ratios per halving of dt : "
          + ", ".join(f"{r:.2f}" for r in r_c) + "   (4 = trapezoid O(dt^2))")

    passed = (err < 1e-12) and (dH.max() <= 0.0) and all(3.2 < r < 4.8 for r in r_c)
    _record("16 Passivity / balance", passed,
            f"identity exact to {err:.1e} (rel {err / scale:.1e}); H monotone "
            f"with u=0 (max {dH.max():+.1e}); driven balance closes at O(dt^2)")
    return passed


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("SE(3) quadrotor -- port-Hamiltonian structure tests")
    print("(tests 11-16 of notes/quadrotor-se3-ph-system.md §20.2)")

    test_11_gt_ph_vs_env()
    test_12_energy_conservation()
    test_13_momentum_conservation()
    test_14_trivialized_gradients()
    test_15_ito_equals_stratonovich()
    test_16_passivity()

    _banner("SUMMARY")
    width = max(len(n) for n, _, _ in RESULTS)
    for name, ok, headline in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<{width}}  {headline}")
    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\n  {n_ok}/{len(RESULTS)} tests passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)

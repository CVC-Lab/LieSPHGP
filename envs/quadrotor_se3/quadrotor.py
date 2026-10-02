from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────────────────────────────────────
# SO(3) Lie group utilities
# ──────────────────────────────────────────────────────────────────────────────

def _hat(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix (hat map) for w in R^3.  hat: R^3 -> so(3)."""
    wx, wy, wz = w
    return np.array([[0.0, -wz,  wy],
                     [wz,  0.0, -wx],
                     [-wy, wx,  0.0]], dtype=np.float64)


def _vee(W: np.ndarray) -> np.ndarray:
    """Inverse hat map (vee).  vee: so(3) -> R^3."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    """Matrix exponential on so(3) via Rodrigues' formula.

    Given phi in R^3, computes exp([phi]_x) in SO(3).

    exp([phi]_x) = I + (sin(theta)/theta) [phi]_x
                     + ((1 - cos(theta))/theta^2) [phi]_x^2

    where theta = ||phi||.  Uses Taylor expansions for small theta
    to avoid division by zero.
    """
    theta_sq = np.dot(phi, phi)
    theta = np.sqrt(theta_sq)
    Phi = _hat(phi)

    if theta < 1e-10:
        # Taylor: sin(t)/t ≈ 1 - t²/6,  (1-cos(t))/t² ≈ 1/2 - t²/24
        A = 1.0 - theta_sq / 6.0
        B = 0.5 - theta_sq / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta_sq

    return np.eye(3) + A * Phi + B * (Phi @ Phi)


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithmic map on SO(3).  log: SO(3) -> R^3.

    Returns phi such that R = exp([phi]_x).
    """
    cos_theta = 0.5 * (np.trace(R) - 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        # Near identity: log(R) ≈ (R - R^T)/2
        return _vee(0.5 * (R - R.T))
    elif abs(theta - np.pi) < 1e-6:
        # Near pi: need special handling
        # Find the column of (R + I) with largest norm
        M = R + np.eye(3)
        norms = np.linalg.norm(M, axis=0)
        k = np.argmax(norms)
        v = M[:, k] / norms[k]
        return v * theta
    else:
        return _vee(theta / (2.0 * np.sin(theta)) * (R - R.T))


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the nearest rotation matrix via polar decomposition.
    (Kept as a safety net, but should rarely be needed with the exp map integrator.)
    """
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1.0
        Rproj = U @ Vt
    return Rproj


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [x, y, z, w] (PyBullet convention).

    Shepperd's method: pick the largest of (trace, R00, R11, R22) for
    numerical stability.
    """
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix via random unit quaternion (Shoemake)."""
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class quadrotor_se3(gym.Env):
    """Quadrotor on SE(3) with wind, damping (friction), and stochastic forcing.

    State (body-frame formulation):
        x_w   in R^3    : position of the center of mass, world frame
        R     in SO(3)  : rotation body -> world
        v_b   in R^3    : linear velocity, BODY frame
        omega in R^3    : angular velocity, BODY frame

    Control input u in R^4 = squared rotor speeds (rpm^2 conceptually),
    mapped to a body wrench through the constant control matrix G in R^{6x4}
    (CF2X X-configuration mixer built from kf, km, arm).

    Stratonovich SDE (body frame):
        x_w_dot   = R v_b
        R_dot     = R [omega]_x
        m v_b_dot = m v_b x omega + R^T(-m g e3 + F_wind(t)) + e3 kf sum(u)
                    - d_lin v_b  + sigma_f R^T o dW_f
        J omega_dot = (J omega) x omega + tau(u) - d_ang omega
                    + sigma_tau o dW_tau

    Uses a geometrically exact Lie group integrator:
      - Rotation updates use the exponential map (Rodrigues), so R stays on SO(3)
        by construction — no SVD projection needed in the integration loop.
      - Stratonovich SDE is integrated via Heun's method on the Lie algebra.

    Randomizable coefficients: kf, km, linear damping, angular damping each
    have a *_coeff (mean) and *_std argument.  std == 0 -> fixed constant;
    std > 0 -> value ~ max(0, N(coeff, std^2)), re-sampled every step
    (resample_coeffs_every_step=True) or once per trajectory at reset
    (resample_coeffs_every_step=False).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        render_backend: str = "pybullet",
        g: float = 9.81,
        m: float = 1.0,
        J_diag: Tuple[float, float, float] = (0.5, 0.5, 1.0),
        arm: float = 1.0,
        dt: float = 0.05,
        kf_coeff: float = 1.0,
        kf_std: float = 0.0,
        km_coeff: float = 0.1,
        km_std: float = 0.0,
        linear_damping_coeff: float = 0.1,
        linear_damping_std: float = 0.0,
        angular_damping_coeff: float = 0.1,
        angular_damping_std: float = 0.0,
        external_force_type: str = "sine",
        external_force_std: float = 1.0,
        external_force_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        wind_force_std: float = 0.0,
        wind_torque_std: float = 0.0,
        resample_coeffs_every_step: bool = True,
        obs_noise_std: float = 0.0,
        max_u: float = 25.0,
        ori_rep: str = "rotmat",
        seed: Optional[int] = None,
    ):
        super().__init__()

        if ori_rep != "rotmat":
            raise ValueError("Only ori_rep='rotmat' is supported for the SE(3) quadrotor env.")
        if render_backend not in ("pybullet", "matplotlib"):
            raise ValueError("render_backend must be 'pybullet' or 'matplotlib'.")

        self.render_mode = render_mode
        self.render_backend = render_backend
        self.g = float(g)
        self.m = float(m)
        self.J = np.diag(np.asarray(J_diag, dtype=np.float64))
        self.J_inv = np.linalg.inv(self.J)
        self.arm = float(arm)
        self.dt = float(dt)

        # Coefficient means and stds.  std == 0 -> fixed at coeff;
        # std > 0 -> value ~ max(0, N(coeff, std^2)).
        for name, std in [("kf_std", kf_std), ("km_std", km_std),
                          ("linear_damping_std", linear_damping_std),
                          ("angular_damping_std", angular_damping_std)]:
            if float(std) < 0.0:
                raise ValueError(f"{name} must be >= 0.")
        self.kf_coeff = float(kf_coeff)
        self.kf_std = float(kf_std)
        self.km_coeff = float(km_coeff)
        self.km_std = float(km_std)
        self.linear_damping_coeff = float(linear_damping_coeff)
        self.linear_damping_std = float(linear_damping_std)
        self.angular_damping_coeff = float(angular_damping_coeff)
        self.angular_damping_std = float(angular_damping_std)
        self.resample_coeffs_every_step = bool(resample_coeffs_every_step)

        self.external_force_type = str(external_force_type)
        self.external_force_std = float(external_force_std)
        wdir = np.array(external_force_direction, dtype=np.float64)
        if np.linalg.norm(wdir) < 1e-12:
            raise ValueError("external_force_direction must be non-zero.")
        self.external_force_direction = wdir / np.linalg.norm(wdir)

        self.wind_force_std = float(wind_force_std)
        self.wind_torque_std = float(wind_torque_std)

        self.obs_noise_std = float(obs_noise_std)

        self.ori_rep = ori_rep

        self.max_u = float(max_u)

        # NOTE: action_space uses self.max_u to define the API bounds for the
        # agent's squared-rotor-speed inputs. Kept active because gym.Env
        # requires a well-defined action_space; the env does NOT clip u.
        self.action_space = spaces.Box(
            low=0.0, high=self.max_u, shape=(4,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )

        # Dynamics rng and a separate observation-noise rng, so enabling
        # obs_noise_std never perturbs the dynamics noise stream.
        self._np_rng = np.random.default_rng(seed)
        self._obs_rng = np.random.default_rng(None if seed is None else seed + 1)

        self.t = 0.0
        self.last_u = np.zeros(4, dtype=np.float64)
        self.last_w = 0.0

        self.x_w = np.zeros(3, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.v_b = np.zeros(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

        # Draw initial coefficient values (and build G).
        self._resample_coeffs()

        self._fig = None
        self._ax = None
        # PyBullet render backend state (renderer ONLY — never steps physics)
        self._pb_client = None
        self._pb_drone_id = None

    # ── Construction from a YAML config ───────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str, **overrides) -> "quadrotor_se3":
        """Build the env from a YAML config file, e.g.

            configs/quadrotor_se3/envs/ode.yaml   (deterministic, pH-ODE)
            configs/quadrotor_se3/envs/sde.yaml   (stochastic,   pH-SDE)

        Precedence: constructor defaults < config file < ``overrides``.
        Unknown keys in either the file or ``overrides`` raise, so typos are
        caught instead of silently ignored.  See env_config.py for the CLI.
        """
        import os as _os
        import sys as _sys
        _d = _os.path.dirname(_os.path.abspath(__file__))
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        from env_config import env_kwargs_from_config  # lazy: yaml only if used

        return cls(**env_kwargs_from_config(config_path, **overrides))

    # ── Seeding & state access ────────────────────────────────────────────

    def seed(self, seed: Optional[int] = None):
        self._np_rng = np.random.default_rng(seed)
        self._obs_rng = np.random.default_rng(None if seed is None else seed + 1)

    def get_state(self):
        """Always returns the CLEAN state (x_w, R, v_b, omega)."""
        return self.x_w.copy(), self.R.copy(), self.v_b.copy(), self.omega.copy()

    # ── Randomizable coefficients & control matrix G ──────────────────────

    def _draw_coeff(self, coeff: float, std: float) -> float:
        """value = coeff if std == 0 else max(0, N(coeff, std^2))."""
        if std == 0.0:
            return coeff
        return max(0.0, float(self._np_rng.normal(loc=coeff, scale=std)))

    def _resample_coeffs(self):
        """Draw current kf, km, d_lin, d_ang and rebuild G.

        Called in __init__ and reset() always; additionally called once per
        step() when resample_coeffs_every_step=True (held constant across
        the substeps within one step).
        """
        self._kf = self._draw_coeff(self.kf_coeff, self.kf_std)
        self._km = self._draw_coeff(self.km_coeff, self.km_std)
        self._d_lin = self._draw_coeff(self.linear_damping_coeff, self.linear_damping_std)
        self._d_ang = self._draw_coeff(self.angular_damping_coeff, self.angular_damping_std)
        self.G = self._build_G(self._kf, self._km)

    def _build_G(self, kf: float, km: float) -> np.ndarray:
        """Constant control matrix G in R^{6x4}:  [F_b; tau_b] = G u.

        CF2X X-configuration (signs match gym-pybullet-drones BaseAviary
        _dynamics for DroneModel.CF2X).  With a = arm/sqrt(2):

            F_x   = [   0,    0,    0,    0 ]
            F_y   = [   0,    0,    0,    0 ]
            F_z   = [  kf,   kf,   kf,   kf ]
            tau_x = a[-kf,  -kf,   kf,   kf ]
            tau_y = a[-kf,   kf,   kf,  -kf ]
            tau_z = [ -km,   km,  -km,   km ]

        Rotor body positions (rendering + torque signs):
            r0 = (+a, -a, 0), r1 = (-a, -a, 0), r2 = (-a, +a, 0), r3 = (+a, +a, 0)
        """
        a = self.arm / np.sqrt(2.0)
        G = np.zeros((6, 4), dtype=np.float64)
        G[2, :] = [kf, kf, kf, kf]
        G[3, :] = [-a * kf, -a * kf, a * kf, a * kf]
        G[4, :] = [-a * kf, a * kf, a * kf, -a * kf]
        G[5, :] = [-km, km, -km, km]
        return G

    def _rotor_positions(self) -> np.ndarray:
        """Body-frame rotor positions (4, 3), consistent with _build_G."""
        a = self.arm / np.sqrt(2.0)
        return np.array([[+a, -a, 0.0],
                         [-a, -a, 0.0],
                         [-a, +a, 0.0],
                         [+a, +a, 0.0]], dtype=np.float64)

    # ── Wind model ────────────────────────────────────────────────────────

    def update_wind(self, t: float) -> float:
        if self.external_force_type == "sine":
            return self.external_force_std * np.sin(2 * np.pi * 0.5 * t)
        if self.external_force_type == "square":
            return self.external_force_std * (1.0 if np.sin(2 * np.pi * 0.5 * t) >= 0 else -1.0)
        if self.external_force_type == "random":
            return float(self._np_rng.normal(loc=0, scale=self.external_force_std))
        if self.external_force_type == "constant":
            return float(self.external_force_std)
        raise ValueError(f"Unknown external_force_type: {self.external_force_type}")

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """obs = [x_w (3), R.flatten() (9), v_b (3), omega (3)] in R^18.

        If obs_noise_std > 0, returns geometrically-noised observations:
            R_noisy = R @ exp([eps]_x),  eps ~ N(0, std^2 I_3)
            x, v, omega get additive N(0, std^2) noise.
        The internal state stays clean (see get_state()).
        """
        if self.obs_noise_std > 0.0:
            std = self.obs_noise_std
            eps = self._obs_rng.normal(0.0, std, size=3)
            R_o = self.R @ _exp_so3(eps)
            x_o = self.x_w + self._obs_rng.normal(0.0, std, size=3)
            v_o = self.v_b + self._obs_rng.normal(0.0, std, size=3)
            w_o = self.omega + self._obs_rng.normal(0.0, std, size=3)
        else:
            R_o, x_o, v_o, w_o = self.R, self.x_w, self.v_b, self.omega
        obs = np.hstack((x_o, R_o.reshape(-1), v_o, w_o)).astype(np.float32)
        return obs

    # ── Dynamics: compute rates and stochastic increments ─────────────────

    def _compute_rates(
        self, x_w: np.ndarray, R: np.ndarray, v: np.ndarray, omega: np.ndarray,
        w: float, u: np.ndarray, dW_f: np.ndarray, dW_tau: np.ndarray
    ):
        """Compute deterministic rates and stochastic increments at a state.

        Returns:
            x_dot          : R^3  (world-frame position rate, = R v)
            v_dot_det      : R^3  (deterministic body-frame linear acceleration)
            omega_dot_det  : R^3  (deterministic body-frame angular acceleration)
            dV_stoch       : R^3  (stochastic linear velocity increment)
            dOmega_stoch   : R^3  (stochastic angular velocity increment)
        """
        e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Body wrench from the control matrix:  [F_u; tau_u] = G u
        wrench = self.G @ u
        F_u = wrench[:3]
        tau_u = wrench[3:]

        # World-frame external forces: gravity + deterministic wind at the COM
        F_wind_world = w * self.external_force_direction
        F_ext_world = -self.m * self.g * e3 + F_wind_world

        # Kinematics
        x_dot = R @ v

        # Translational dynamics (body frame):
        #   m v_dot = m v x omega + R^T F_ext_world + F_u - d_lin v
        v_dot_det = (
            np.cross(v, omega)
            + (R.T @ F_ext_world + F_u - self._d_lin * v) / self.m
        )

        # Rotational dynamics (Euler's rigid body equation, body frame):
        #   J omega_dot = (J omega) x omega + tau_u - d_ang omega
        Jw = self.J @ omega
        omega_dot_det = self.J_inv @ (
            np.cross(Jw, omega) + tau_u - self._d_ang * omega
        )

        # Stochastic forcing
        # Force channel: world-frame random force at the COM -> body frame.
        # State-dependent through R (multiplicative noise), so it must be
        # re-evaluated in both Heun stages.
        if self.wind_force_std > 0.0:
            dV_stoch = (self.wind_force_std / self.m) * (R.T @ dW_f)
        else:
            dV_stoch = np.zeros(3)
        # Torque channel: additive body-frame random torque.
        if self.wind_torque_std > 0.0:
            dOmega_stoch = self.J_inv @ (self.wind_torque_std * dW_tau)
        else:
            dOmega_stoch = np.zeros(3)

        return x_dot, v_dot_det, omega_dot_det, dV_stoch, dOmega_stoch

    # ── Lie group Heun integrator (Stratonovich) ──────────────────────────

    def _lie_heun_step(
        self, x_w: np.ndarray, R: np.ndarray, v: np.ndarray, omega: np.ndarray,
        w: float, u: np.ndarray, h: float,
        dW_f: np.ndarray, dW_tau: np.ndarray
    ):
        """One substep of Stratonovich Heun integration on SE(3).

        Rotation update uses the exponential map so R stays on SO(3)
        by construction.  The Heun scheme averages two Lie algebra
        elements before exponentiating, giving second-order accuracy
        in the deterministic part.

        Algorithm:
        ---------
        1. At (x_n, R_n, v_n, omega_n), compute rates_1 and the Lie algebra
           element phi_1 = omega_n * h  (body angular displacement).

        2. Predictor:
             R_pred     = R_n * exp([phi_1]_x)
             x_pred     = x_n + x_dot_1 * h
             v_pred     = v_n + v_dot_1 * h + dV_stoch_1
             omega_pred = omega_n + omega_dot_1 * h + dOmega_stoch_1

        3. At the predicted state, compute rates_2 (reuse same dW) and
           phi_2 = omega_pred * h.

        4. Corrector (average in Lie algebra, then exponentiate once):
             phi_avg     = (phi_1 + phi_2) / 2
             R_{n+1}     = R_n * exp([phi_avg]_x)
             x_{n+1}     = x_n + (x_dot_1 + x_dot_2)/2 * h
             v_{n+1}     = v_n + (v_dot_1 + v_dot_2)/2 * h
                           + (dV_stoch_1 + dV_stoch_2) / 2
             omega_{n+1} = omega_n + (omega_dot_1 + omega_dot_2)/2 * h
                           + (dOmega_stoch_1 + dOmega_stoch_2) / 2
        """
        # ── Stage 1: evaluate at current state ──
        x_dot_1, v_dot_1, omega_dot_1, dV_1, dOmega_1 = self._compute_rates(
            x_w, R, v, omega, w, u, dW_f, dW_tau
        )
        phi_1 = omega * h  # Lie algebra element for rotation

        # ── Stage 2: predictor (Euler on the manifold) ──
        R_pred = R @ _exp_so3(phi_1)
        x_pred = x_w + x_dot_1 * h
        v_pred = v + v_dot_1 * h + dV_1
        omega_pred = omega + omega_dot_1 * h + dOmega_1

        # ── Stage 3: evaluate at predicted state (reuse same dW) ──
        x_dot_2, v_dot_2, omega_dot_2, dV_2, dOmega_2 = self._compute_rates(
            x_pred, R_pred, v_pred, omega_pred, w, u, dW_f, dW_tau
        )
        phi_2 = omega_pred * h

        # ── Stage 4: corrector (average in Lie algebra, single exp) ──
        phi_avg = 0.5 * (phi_1 + phi_2)
        R_new = R @ _exp_so3(phi_avg)

        x_new = x_w + 0.5 * (x_dot_1 + x_dot_2) * h
        v_new = v + 0.5 * (v_dot_1 + v_dot_2) * h + 0.5 * (dV_1 + dV_2)
        omega_new = omega + 0.5 * (omega_dot_1 + omega_dot_2) * h + 0.5 * (dOmega_1 + dOmega_2)

        return x_new, R_new, v_new, omega_new

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, u):
        u = np.asarray(u, dtype=np.float64).reshape(4)
        # NOTE: no clipping of u by self.max_u — the parameter only defines
        # the action_space API bounds (mirrors the pendulum env).
        self.last_u = u.copy()

        self.t += self.dt
        # rng draw order per step (reproducibility): wind -> coeffs -> dW's
        w = self.update_wind(self.t)
        self.last_w = float(w)

        if self.resample_coeffs_every_step:
            self._resample_coeffs()

        # Substep settings
        n_substeps = 10
        dt_sub = self.dt / n_substeps

        for _ in range(n_substeps):
            # Sample Wiener increments once per substep (independent channels)
            if self.wind_force_std > 0.0:
                dW_f = self._np_rng.normal(0.0, np.sqrt(dt_sub), size=3)
            else:
                dW_f = np.zeros(3)
            if self.wind_torque_std > 0.0:
                dW_tau = self._np_rng.normal(0.0, np.sqrt(dt_sub), size=3)
            else:
                dW_tau = np.zeros(3)

            # Lie group Heun step
            self.x_w, self.R, self.v_b, self.omega = self._lie_heun_step(
                self.x_w, self.R, self.v_b, self.omega, w, u, dt_sub, dW_f, dW_tau
            )

        # Periodic re-orthogonalization as a safety net
        # (numerical drift from floating-point accumulation over many steps)
        if abs(np.linalg.det(self.R) - 1.0) > 1e-8 or \
           np.linalg.norm(self.R.T @ self.R - np.eye(3)) > 1e-8:
            self.R = _project_to_so3(self.R)

        # ── Reward (placeholder hover cost about x* = 0) ──
        u_hover = self.m * self.g / (4.0 * max(self._kf, 1e-12))
        pos_cost = float(np.dot(self.x_w, self.x_w))
        vel_cost = 0.1 * float(np.dot(self.v_b, self.v_b) + np.dot(self.omega, self.omega))
        act_cost = 0.001 * float(np.sum((u - u_hover) ** 2))
        cost = pos_cost + vel_cost + act_cost

        obs = self._get_obs()
        reward = -cost
        terminated = False
        truncated = False
        info = {
            "wind": self.last_w,
            "kf": self._kf,
            "km": self._km,
            "d_lin": self._d_lin,
            "d_ang": self._d_ang,
        }

        return obs, reward, terminated, truncated, info

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.t = 0.0
        self.last_u = np.zeros(4, dtype=np.float64)
        self.last_w = 0.0

        if options is None:
            options = {}

        if "x_init" in options:
            x0 = np.asarray(options["x_init"], dtype=np.float64).reshape(3)
        else:
            x0 = self._np_rng.uniform(low=-1.0, high=1.0, size=3)

        if "R_init" in options:
            R0 = np.asarray(options["R_init"], dtype=np.float64).reshape(3, 3)
            R0 = _project_to_so3(R0)
        else:
            R0 = _random_rotation(self._np_rng)

        if "v_init" in options:
            v0 = np.asarray(options["v_init"], dtype=np.float64).reshape(3)
        else:
            v0 = self._np_rng.uniform(low=-1.0, high=1.0, size=3)

        if "omega_init" in options:
            w0 = np.asarray(options["omega_init"], dtype=np.float64).reshape(3)
        else:
            w0 = self._np_rng.uniform(low=-1.0, high=1.0, size=3)

        self.x_w = x0
        self.R = R0
        self.v_b = v0
        self.omega = w0

        # Draw the trajectory's coefficients (one draw per trajectory when
        # resample_coeffs_every_step=False; otherwise re-drawn each step).
        self._resample_coeffs()

        return self._get_obs(), {}

    # ── Rendering ─────────────────────────────────────────────────────────
    #
    # Two backends:
    #   "pybullet"   (default) — renders the actual CF2X model + ground plane
    #                 with PyBullet's camera, exactly like gym-pybullet-drones
    #                 BaseAviary recordings (DIRECT mode + getCameraImage).
    #                 PyBullet is a RENDERER ONLY: the state is pushed in with
    #                 resetBasePositionAndOrientation; stepSimulation is never
    #                 called, so the Lie-Heun dynamics stay the single source
    #                 of truth.
    #   "matplotlib" — zero-dependency schematic fallback (X-frame drawing).

    _CF2X_ARM = 0.0397  # arm length of the cf2x.urdf visual model [m]

    def render(self):
        if self.render_mode is None:
            return
        if self.render_backend == "pybullet":
            return self._render_pybullet()
        return self._render_matplotlib()

    # ── PyBullet backend (gym-pybullet-drones style) ──────────────────────

    def _init_render_pybullet(self):
        import os
        import pybullet as p
        import pybullet_data

        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._pb_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._pb_client)
        p.loadURDF("plane.urdf", physicsClientId=self._pb_client)

        urdf = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party",
            "gym-pybullet-drones", "gym_pybullet_drones", "assets", "cf2x.urdf",
        )
        if not os.path.isfile(urdf):
            raise FileNotFoundError(
                f"CF2X model not found at {urdf} — clone gym-pybullet-drones "
                "into third_party/ or use render_backend='matplotlib'."
            )
        # Scale the visual model so its arm length matches this env's arm.
        self._pb_drone_id = p.loadURDF(
            urdf,
            basePosition=self.x_w,
            baseOrientation=_rotmat_to_quat(self.R),
            globalScaling=self.arm / self._CF2X_ARM,
            physicsClientId=self._pb_client,
        )
        if self.render_mode == "human":
            for flag in [p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                         p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                         p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(flag, 0, physicsClientId=self._pb_client)

    def _render_pybullet(self):
        import pybullet as p

        if self._pb_client is None:
            self._init_render_pybullet()

        # Push the current (clean) state into the render scene
        p.resetBasePositionAndOrientation(
            self._pb_drone_id, self.x_w, _rotmat_to_quat(self.R),
            physicsClientId=self._pb_client,
        )

        # BaseAviary-style camera (yaw -30, pitch -30) tracking the drone so it
        # never leaves the frame; distance 6*arm keeps the model clearly visible.
        cam_dist = 6.0 * self.arm
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist, cameraYaw=-30, cameraPitch=-30,
                cameraTargetPosition=self.x_w, physicsClientId=self._pb_client,
            )
            return None

        vid_w, vid_h = 640, 480
        view = p.computeViewMatrixFromYawPitchRoll(
            distance=cam_dist, yaw=-30, pitch=-30, roll=0,
            cameraTargetPosition=self.x_w, upAxisIndex=2,
            physicsClientId=self._pb_client,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=vid_w / vid_h, nearVal=0.1, farVal=1000.0,
        )
        w, h, rgb, dep, seg = p.getCameraImage(
            width=vid_w, height=vid_h, shadow=1,
            viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._pb_client,
        )
        img = np.reshape(np.asarray(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]
        return img.copy()

    # ── Matplotlib backend (schematic fallback) ───────────────────────────

    def _init_render(self):
        import matplotlib
        if self.render_mode == "rgb_array":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111, projection="3d")

    def _render_matplotlib(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init_render()

        ax = self._ax
        ax.cla()

        center = self.x_w
        rotors_body = self._rotor_positions()
        rotors_world = np.array([center + self.R @ r for r in rotors_body])

        # X-frame arms: rotor 0 <-> rotor 2 and rotor 1 <-> rotor 3
        for i, j in [(0, 2), (1, 3)]:
            ax.plot(
                [rotors_world[i, 0], rotors_world[j, 0]],
                [rotors_world[i, 1], rotors_world[j, 1]],
                [rotors_world[i, 2], rotors_world[j, 2]],
                color="#555555", linewidth=3, solid_capstyle="round",
            )

        # Rotor discs (colored by spin direction from the tau_z row of G)
        spin_colors = ["#cc4d4d", "#4d7ccc", "#cc4d4d", "#4d7ccc"]  # CW, CCW, CW, CCW
        r_disc = 0.3 * self.arm
        theta = np.linspace(0, 2 * np.pi, 30)
        circ_body = np.stack([r_disc * np.cos(theta), r_disc * np.sin(theta),
                              np.zeros_like(theta)], axis=1)
        for k in range(4):
            circ_world = rotors_world[k] + (self.R @ circ_body.T).T
            ax.plot(circ_world[:, 0], circ_world[:, 1], circ_world[:, 2],
                    color=spin_colors[k], linewidth=1.5)
        ax.scatter(*center, color="black", s=60, depthshade=True, zorder=5)

        # Body axes
        axis_len = 0.6 * self.arm
        axis_colors = ["#e74c3c", "#2ecc71", "#3498db"]
        axis_labels = ["x_b", "y_b", "z_b"]
        for i in range(3):
            e_body = np.zeros(3)
            e_body[i] = 1.0
            tip = center + axis_len * (self.R @ e_body)
            ax.quiver(
                center[0], center[1], center[2],
                tip[0] - center[0], tip[1] - center[1], tip[2] - center[2],
                color=axis_colors[i], linewidth=2, arrow_length_ratio=0.15,
            )
            ax.text(tip[0], tip[1], tip[2], f" {axis_labels[i]}",
                    color=axis_colors[i], fontsize=8, fontweight="bold")

        # Wind arrow
        if abs(self.last_w) > 1e-6:
            w_vec = self.last_w * self.external_force_direction
            w_scale = 0.75 * self.arm
            ax.quiver(
                center[0], center[1], center[2],
                w_vec[0] * w_scale, w_vec[1] * w_scale, w_vec[2] * w_scale,
                color="#00bcd4", linewidth=2.5, arrow_length_ratio=0.18,
                label=f"wind = {self.last_w:+.2f}",
            )

        # Gravity arrow
        g_len = 0.6 * self.arm
        ax.quiver(
            center[0], center[1], center[2],
            0, 0, -g_len,
            color="#999999", linewidth=1.5, arrow_length_ratio=0.15,
            linestyle="dashed", label="gravity",
        )

        # Ground disc at z = 0 (under the drone)
        theta_g = np.linspace(0, 2 * np.pi, 60)
        r_circle = 2.5 * self.arm
        xc = center[0] + r_circle * np.cos(theta_g)
        yc = center[1] + r_circle * np.sin(theta_g)
        zc = np.zeros_like(theta_g)
        verts = [list(zip(xc, yc, zc))]
        ground = Poly3DCollection(
            verts, alpha=0.08, facecolor="#b0bec5",
            edgecolor="#78909c", linewidth=0.8,
        )
        ax.add_collection3d(ground)

        # Axis limits track the drone so it stays in frame
        lim = 2.0 * self.arm
        ax.set_xlim(center[0] - lim, center[0] + lim)
        ax.set_ylim(center[1] - lim, center[1] + lim)
        ax.set_zlim(center[2] - lim, center[2] + lim)
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.set_zlabel("Z", fontsize=9)
        ax.set_title("Quadrotor on SE(3) — Lie Group Integrator", fontsize=12, fontweight="bold")
        ax.set_box_aspect([1, 1, 1])

        v_mag = float(np.linalg.norm(self.v_b))
        omega_mag = float(np.linalg.norm(self.omega))
        hud = (
            f"t = {self.t:.2f}s    "
            f"wind = {self.last_w:+.2f}    "
            f"alt = {self.x_w[2]:+.2f}    "
            f"|v| = {v_mag:.2f}    "
            f"|omega| = {omega_mag:.2f}"
        )
        ax.text2D(
            0.02, 0.96, hud, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.001)
        else:
            self._fig.canvas.draw()
            buf = self._fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3].copy()
            return img

    def close(self):
        if self._pb_client is not None:
            import pybullet as p
            p.disconnect(physicsClientId=self._pb_client)
            self._pb_client = None
            self._pb_drone_id = None
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        self._fig = None
        self._ax = None


# ──────────────────────────────────────────────────────────────────────────────
# Demo / video generation
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from env_config import (build_parser, kwargs_from_args, describe,
                            is_stochastic, load_config, SDE_CONFIG)

    # Every env argument is overridable on the command line, e.g.
    #   python envs/quadrotor_se3/quadrotor.py \
    #       --config configs/quadrotor_se3/envs/ode.yaml \
    #       --external_force_type random --external_force_std 1.5
    _parser = build_parser(
        description="SE(3) quadrotor demo: hover under wind, recorded to mp4.",
        default_config=SDE_CONFIG)
    _parser.add_argument("--steps", type=int, default=300, help="env steps to record")
    _parser.add_argument("--out", type=str,
                         default="videos/quadrotor_se3_lie_group.mp4",
                         help="output mp4 path")
    _parser.add_argument("--fwd_speed", type=float, default=2.0,
                         help="forward_spin: target world +x speed [m/s]")
    _parser.add_argument("--yaw_rate", type=float, default=1.0,
                         help="forward_spin: clockwise yaw rate [rad/s] "
                              "(clockwise seen from above = negative omega_z)")
    _parser.add_argument("--controller", choices=["pd", "none", "forward_spin"], default="pd",
                         help="pd = geometric hover controller (keeps it airborne); "
                              "none = open loop at hover thrust (unstable, it will "
                              "tumble - that is physically correct)")
    _parser.add_argument("--random_u", action="store_true",
                         help="add random excitation to u each step, like the "
                              "pendulum datagen's --random_u (notes s18.1)")
    _parser.add_argument("--random_u_scale", type=float, default=1.0,
                         help="std of the random excitation added to u")
    _parser.add_argument("--ylim_z", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="pin the altitude panel's y-axis; use the SAME values across "
                              "runs to compare them (panels autoscale by default, which "
                              "makes proportional trajectories look identical)")
    _parser.add_argument("--ylim_vz", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="pin the vertical-velocity panel's y-axis; see --ylim_z")
    _parser.add_argument("--u_const", type=float, nargs="+", default=None, metavar="U",
                         help="hold the rotor input constant: ONE value broadcast to "
                              "all 4 rotors (e.g. --u_const 1), or four values. "
                              "Overrides --controller. Equal rotor inputs cancel every "
                              "torque row of G, so the drone stays level and only "
                              "translates; u = m*g/(4*kf) is exact hover.")
    _args = _parser.parse_args()

    N_STEPS = _args.steps
    SAVE_PATH = _args.out

    # The demo has to render.  The shipped configs set `render_mode: null`
    # (right for datagen/training), and an explicit null is a *present* key,
    # so a setdefault would not catch it -- force a mode unless the user asked
    # for one with --render_mode.
    ENV_KWARGS = kwargs_from_args(_args)
    if ENV_KWARGS.get("render_mode") is None:
        ENV_KWARGS["render_mode"] = "rgb_array"
    print(describe(ENV_KWARGS, _args.config,
                   load_config(_args.config)[1] if _args.config else None))
    print("-" * 62)
    U_CONST = None
    if _args.u_const is not None:
        if len(_args.u_const) == 1:
            U_CONST = np.full(4, float(_args.u_const[0]))
        elif len(_args.u_const) == 4:
            U_CONST = np.asarray(_args.u_const, dtype=np.float64)
        else:
            _parser.error("--u_const takes either 1 or 4 values")

    if U_CONST is not None:
        _ctrl = f"constant u = {np.array2string(U_CONST, precision=3)}"
    else:
        _ctrl = {"pd": "PD hover", "none": "open loop",
             "forward_spin": f"forward {_args.fwd_speed:g} m/s + spin "
                             f"{_args.yaw_rate:g} rad/s CW, hold z"}[_args.controller]
    if _args.random_u:
        _ctrl += f" + random_u(scale={_args.random_u_scale:g})"
    print(f"  -> {'SDE (stochastic)' if is_stochastic(ENV_KWARGS) else 'ODE (deterministic)'}"
          f"   |   u: {_ctrl}"
          f"   |   {N_STEPS} steps -> {SAVE_PATH}")
    print("=" * 62)

    env = quadrotor_se3(**ENV_KWARGS)
    DEMO_SEED = ENV_KWARGS.get("seed", 0)
    # Start near hover: level attitude, at rest, 3 m altitude
    obs, _ = env.reset(seed=DEMO_SEED, options={
        "x_init": [0.0, 0.0, 3.0],
        "R_init": np.eye(3),
        "v_init": [0.0, 0.0, 0.0],
        "omega_init": [0.0, 0.0, 0.0],
    })
    assert env._kf > 1e-9, "kf too small for the hover demo"

    # ── Demo-only geometric hover controller (Lee-style PD on SE(3)) ──
    # A bare quadrotor is open-loop unstable: without feedback, the stochastic
    # wind torques tilt it and it crashes within seconds (physically correct).
    # This minimal position-hold controller keeps the demo airborne so the
    # video shows the drone leaning into wind gusts and recovering.
    X_TARGET = np.array([0.0, 0.0, 3.0])
    KP_POS, KD_POS = 2.0, 2.5     # position PD
    KR, KOMEGA = 10.0, 3.0        # attitude PD
    KV_FWD = 2.0                  # forward_spin: world velocity P gain

    def hover_controller(env):
        x, R, v_b, omega = env.get_state()
        v_w = R @ v_b
        e3 = np.array([0.0, 0.0, 1.0])
        # Desired world force (PD + gravity feed-forward)
        F_des = env.m * (KP_POS * (X_TARGET - x) - KD_POS * v_w + env.g * e3)
        # Desired attitude: body z along F_des, yaw toward world x
        b3 = F_des / max(np.linalg.norm(F_des), 1e-9)
        b2 = np.cross(b3, np.array([1.0, 0.0, 0.0]))
        b2 /= max(np.linalg.norm(b2), 1e-9)
        b1 = np.cross(b2, b3)
        R_des = np.stack([b1, b2, b3], axis=1)
        # Thrust = projection of desired force on actual body z
        T = float(F_des @ (R @ e3))
        # Attitude error on SO(3) and PD torque
        e_R = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
        tau = -KR * e_R - KOMEGA * omega
        # Invert the (invertible) [F_z; tau] rows of G:  u = A^{-1} [T; tau]
        A = env.G[2:6, :]
        return np.linalg.solve(A, np.hstack([T, tau]))

    def forward_spin_controller(env):
        """Fly forward at a fixed world speed, hold altitude, spin clockwise.

        Same geometric skeleton as hover_controller, but the desired attitude
        is built from two independent pieces:
          * b3  from the desired force  (altitude PD + forward velocity PD)
                -> the drone tilts into the direction it should accelerate;
          * the yaw reference c1 spins at -yaw_rate about world z, so the
            heading rotates while b3 (hence the world-frame tilt) stays put.
        Keeping b3 fixed while yaw spins is what makes the path a straight
        line rather than a spiral.  omega_des feeds the spin forward so the
        attitude PD tracks it instead of fighting it.
        """
        x, R, v_b, omega = env.get_state()
        v_w = R @ v_b
        e3 = np.array([0.0, 0.0, 1.0])

        a_x = KV_FWD * (_args.fwd_speed - v_w[0])   # drive world +x velocity
        a_y = -KV_FWD * v_w[1]                      # kill lateral drift
        a_z = KP_POS * (X_TARGET[2] - x[2]) - KD_POS * v_w[2]   # hold altitude
        F_des = env.m * np.array([a_x, a_y, env.g + a_z])

        b3 = F_des / max(np.linalg.norm(F_des), 1e-9)
        psi = -_args.yaw_rate * env.t               # clockwise from above
        c1 = np.array([np.cos(psi), np.sin(psi), 0.0])
        b2 = np.cross(b3, c1)
        b2 /= max(np.linalg.norm(b2), 1e-9)
        b1 = np.cross(b2, b3)
        R_des = np.stack([b1, b2, b3], axis=1)

        T = float(F_des @ (R @ e3))
        e_R = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
        omega_des = R.T @ np.array([0.0, 0.0, -_args.yaw_rate])   # spin feedforward
        tau = -KR * e_R - KOMEGA * (omega - omega_des)
        return np.linalg.solve(env.G[2:6, :], np.hstack([T, tau]))

    def record_state(env):
        """History row: t, x_w(3), yaw, v_world(3), omega(3)."""
        x, R, v_b, omega = env.get_state()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        v_w = R @ v_b
        return np.hstack([env.t, x, yaw, v_w, omega])

    frames = []
    history = [record_state(env)]
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    # u = rpm^2 is the control input, so it is physically non-negative; the
    # excitation is clipped into the action-space box [0, max_u].  (The env
    # itself never clips - that clipping belongs to the policy, not the plant.)
    U_HOVER = np.full(4, env.m * env.g / (4.0 * env._kf))
    u_rng = np.random.default_rng(DEMO_SEED)

    for step in range(N_STEPS):
        if U_CONST is not None:
            action = U_CONST.copy()
        elif _args.controller == "pd":
            action = hover_controller(env)
        elif _args.controller == "forward_spin":
            action = forward_spin_controller(env)
        else:
            action = U_HOVER.copy()
        if _args.random_u:
            action = action + u_rng.normal(0.0, _args.random_u_scale, size=4)
        action = np.clip(action, 0.0, env.max_u)
        obs, reward, terminated, truncated, info = env.step(action)
        history.append(record_state(env))
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if step % 20 == 0:
            x, R, v, omega = env.get_state()
            det_R = np.linalg.det(R)
            orth_err = np.linalg.norm(R.T @ R - np.eye(3))
            print(
                f"Step {step:3d}/{N_STEPS}  |  reward={reward:.3f}  |  "
                f"wind={info['wind']:+.2f}  |  alt={x[2]:+.2f}  |  "
                f"det(R)={det_R:.8f}  |  orth_err={orth_err:.2e}"
            )

    env.close()
    history = np.stack(history, axis=0)   # (N+1, 11)
    t_hist = history[:, 0]

    if not frames:
        raise SystemExit(
            "no frames were captured - the env produced no images. "
            "Pass --render_mode rgb_array (or human) to the demo.")

    print(f"\nSaving {len(frames)} frames to {SAVE_PATH} ...")

    import os
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ── Compose video: PyBullet render (left) + telemetry plots (right)
    #    + env-argument stats footer (bottom) ──
    # Right panel mirrors the gym-pybullet-drones Logger layout:
    #   column 1 "Position/Yaw": x, y, z, yaw      (dashed = reference)
    #   column 2 "Velocity":     vx, vy, vz, omega

    def env_args_text(env, seed):
        """All env constructor arguments, formatted as a compact footer."""
        d = env.external_force_direction
        return "\n".join([
            (f"g={env.g:g}   m={env.m:g}   "
             f"J_diag=({env.J[0, 0]:g}, {env.J[1, 1]:g}, {env.J[2, 2]:g})   "
             f"arm={env.arm:g}   dt={env.dt:g}   max_u={env.max_u:g}   "
             f"ori_rep={env.ori_rep}   seed={seed}"),
            (f"kf={env.kf_coeff:g} ± {env.kf_std:g}   "
             f"km={env.km_coeff:g} ± {env.km_std:g}   "
             f"linear_damping={env.linear_damping_coeff:g} ± {env.linear_damping_std:g}   "
             f"angular_damping={env.angular_damping_coeff:g} ± {env.angular_damping_std:g}   "
             f"resample_coeffs_every_step={env.resample_coeffs_every_step}"),
            (f"external_force_type={env.external_force_type}   "
             f"external_force_std={env.external_force_std:g}   "
             f"external_force_direction=({d[0]:g}, {d[1]:g}, {d[2]:g})   "
             f"wind_force_std={env.wind_force_std:g}   "
             f"wind_torque_std={env.wind_torque_std:g}   "
             f"obs_noise_std={env.obs_noise_std:g}   "
             f"render_backend={env.render_backend}"),
        ])

    fig_vid = plt.figure(figsize=(16, 6.8))
    gs = fig_vid.add_gridspec(1, 2, width_ratios=[1.0, 1.15],
                              left=0.02, right=0.98, top=0.93, bottom=0.21,
                              wspace=0.08)
    fig_vid.text(
        0.5, 0.015, env_args_text(env, DEMO_SEED),
        ha="center", va="bottom", fontsize=8.5, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5f5",
                  edgecolor="#b0b0b0", linewidth=0.8),
    )
    ax_img = fig_vid.add_subplot(gs[0])
    ax_img.axis("off")
    im = ax_img.imshow(frames[0])

    gs_plots = gs[1].subgridspec(4, 2, hspace=0.45, wspace=0.35)
    T_TOTAL = t_hist[-1]

    # (history column, ylabel, reference value or None)
    pos_specs = [(1, "x (m)", X_TARGET[0]), (2, "y (m)", X_TARGET[1]),
                 (3, "z (m)", X_TARGET[2]), (4, "yaw (rad)", 0.0)]
    vel_specs = [(5, "x (m/s)", None), (6, "y (m/s)", None),
                 (7, "z (m/s)", None)]

    axes, lines = [], []
    for row, (col_idx, ylabel, ref) in enumerate(pos_specs):
        ax = fig_vid.add_subplot(gs_plots[row, 0])
        if row == 0:
            ax.set_title("Position/Yaw", fontsize=10)
        y = history[:, col_idx]
        pad = 0.1 * max(y.max() - y.min(), 1e-3)
        ax.set_xlim(0, T_TOTAL)
        if ylabel.startswith("z") and _args.ylim_z is not None:
            ax.set_ylim(*_args.ylim_z)
        else:
            ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.6)
        if ref is not None:
            ax.axhline(ref, color="tab:orange", linestyle="--", linewidth=1.0)
        (ln,) = ax.plot([], [], color="tab:blue", linewidth=1.0)
        axes.append(ax)
        lines.append((ln, t_hist, y))
    axes[-1].set_xlabel("Time (s)", fontsize=8)

    for row, (col_idx, ylabel, ref) in enumerate(vel_specs):
        ax = fig_vid.add_subplot(gs_plots[row, 1])
        if row == 0:
            ax.set_title("Velocity", fontsize=10)
        y = history[:, col_idx]
        pad = 0.1 * max(y.max() - y.min(), 1e-3)
        ax.set_xlim(0, T_TOTAL)
        if ylabel.startswith("z") and _args.ylim_vz is not None:
            ax.set_ylim(*_args.ylim_vz)
        else:
            ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.axhline(0.0, color="tab:orange", linestyle="--", linewidth=1.0)
        (ln,) = ax.plot([], [], color="tab:blue", linewidth=1.0)
        axes.append(ax)
        lines.append((ln, t_hist, y))

    # omega panel: all three body-rate components together
    ax_om = fig_vid.add_subplot(gs_plots[3, 1])
    om = history[:, 8:11]
    pad = 0.1 * max(om.max() - om.min(), 1e-3)
    ax_om.set_xlim(0, T_TOTAL)
    ax_om.set_ylim(om.min() - pad, om.max() + pad)
    ax_om.set_ylabel(r"$\omega$ (rad/s)", fontsize=8)
    ax_om.set_xlabel("Time (s)", fontsize=8)
    ax_om.tick_params(labelsize=7)
    ax_om.grid(True, linewidth=0.3, alpha=0.6)
    ax_om.axhline(0.0, color="tab:orange", linestyle="--", linewidth=1.0)
    for k, color in enumerate(["tab:blue", "tab:green", "tab:red"]):
        (ln,) = ax_om.plot([], [], color=color, linewidth=0.9)
        lines.append((ln, t_hist, om[:, k]))

    def _update(i):
        im.set_data(frames[i])
        artists = [im]
        for ln, tt, yy in lines:
            ln.set_data(tt[:i + 1], yy[:i + 1])
            artists.append(ln)
        return artists

    anim = FuncAnimation(fig_vid, _update, frames=len(frames), interval=1000 / 30, blit=True)
    anim.save(SAVE_PATH, writer="ffmpeg", fps=30, dpi=100)
    plt.close(fig_vid)

    print(f"Done! Video saved to: {SAVE_PATH}")

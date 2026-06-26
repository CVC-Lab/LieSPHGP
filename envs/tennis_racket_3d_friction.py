from __future__ import annotations

"""Free rigid body (tennis racket) environment on SO(3).

Models a tennis racket as an elliptical hoop (head) + cylindrical rod (handle).
The principal moments of inertia are computed analytically from geometry.
The equations of motion are Euler's equations for a torque-free (or torqued)
rigid body, integrated with the same Lie-group Heun scheme as windy_pendulum_3d.

State  : (R ∈ SO(3), ω ∈ ℝ³)   →   obs vector length 12  (same as pendulum)
Action : τ ∈ ℝ³ (body-frame torque)                         length  3

Body-frame principal axes convention
-------------------------------------
  e₁  —  long axis of head  (largest semi-axis a,  SMALLEST moment  I₁)
  e₂  —  short axis of head (semi-axis b,           INTERMEDIATE    I₂)
  e₃  —  handle axis        (out-of-plane,           LARGEST moment  I₃)

This ordering guarantees  I₁ < I₂ < I₃  for realistic racket geometries,
so rotation about e₂ is the unstable intermediate axis (Dzhanibekov effect).

Inertia formulae (thin-shell / thin-rod approximations)
---------------------------------------------------------
Elliptical hoop (mass m_h, semi-axes a, b):
    I_hoop_1 = (1/2) m_h b²            (about e₁)
    I_hoop_2 = (1/2) m_h a²            (about e₂)
    I_hoop_3 = (1/2) m_h (a² + b²)     (about e₃)

Thin rod (mass m_r, length L, radius r_r ≪ L):
    COM of rod is offset d = a + L/2 from racket COM (along -e₁ / handle dir)
    I_rod_1  = (1/12) m_r L²  +  m_r d²   (parallel axis, about e₁)
    I_rod_2  = (1/12) m_r L²  +  m_r d²   (same by symmetry)
    I_rod_3  = (1/2)  m_r r_r²             (about handle axis ≈ 0 for thin rod)

Total: I_i = I_hoop_i + I_rod_i  for i = 1, 2, 3.

Wind / friction (commented out — ready to enable)
---------------------------------------------------
The wind block mirrors windy_pendulum_3d exactly.  To enable, uncomment the
sections marked  # [WIND]  and  # [FRICTION]  and pass the relevant kwargs.
"""

from typing import Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────────────────────────────────────
# SO(3) Lie group utilities  (identical to windy_pendulum_3d)
# ──────────────────────────────────────────────────────────────────────────────

def _hat(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix (hat map) for w in R^3."""
    wx, wy, wz = w
    return np.array([[0.0, -wz,  wy],
                     [wz,  0.0, -wx],
                     [-wy, wx,  0.0]], dtype=np.float64)


def _vee(W: np.ndarray) -> np.ndarray:
    """Inverse hat map (vee)."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    """Matrix exponential on so(3) via Rodrigues' formula."""
    theta_sq = np.dot(phi, phi)
    theta = np.sqrt(theta_sq)
    Phi = _hat(phi)
    if theta < 1e-10:
        A = 1.0 - theta_sq / 6.0
        B = 0.5  - theta_sq / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta_sq
    return np.eye(3) + A * Phi + B * (Phi @ Phi)


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithmic map on SO(3)."""
    cos_theta = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return _vee(0.5 * (R - R.T))
    elif abs(theta - np.pi) < 1e-6:
        M = R + np.eye(3)
        norms = np.linalg.norm(M, axis=0)
        k = np.argmax(norms)
        v = M[:, k] / norms[k]
        return v * theta
    else:
        return _vee(theta / (2.0 * np.sin(theta)) * (R - R.T))


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project to nearest rotation matrix via SVD (safety net)."""
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1.0
        Rp = U @ Vt
    return Rp


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation via Shoemake's method."""
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1)     * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1)     * np.cos(2 * np.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Inertia computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_racket_inertia(
    head_a: float,
    head_b: float,
    handle_length: float,
    handle_radius: float,
    total_mass: float,
    head_mass_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the principal inertia tensor for the tennis-racket model.

    The racket is modelled as:
      • An elliptical hoop (head) with semi-axes  a (long) × b (short),
        lying in the e₁-e₂ plane.
      • A thin cylindrical rod (handle) of length L along  -e₁,
        with its proximal end at  x = -a  from the racket COM.

    Parameters
    ----------
    head_a        : semi-axis along e₁ (long direction of head),  metres
    head_b        : semi-axis along e₂ (short direction of head), metres
    handle_length : length of handle rod,                          metres
    handle_radius : radius of handle rod (only affects I₃ of rod), metres
    total_mass    : total racket mass,                             kg
    head_mass_ratio : fraction of total mass in the head hoop,    (0, 1)

    Returns
    -------
    I     : (3,3) diagonal inertia tensor  [I₁, I₂, I₃] in body frame
    I_inv : inverse of I
    """
    m_h = total_mass * head_mass_ratio          # head mass
    m_r = total_mass * (1.0 - head_mass_ratio)  # handle mass

    # ── Head (elliptical hoop) ──
    # Thin hoop: all mass on the ellipse perimeter.
    # For a uniform elliptical ring the MOI about its own axes are:
    #   about e₁ (long axis in plane) : I = (1/2) m b²
    #   about e₂ (short axis in plane): I = (1/2) m a²
    #   about e₃ (normal to plane)    : I = (1/2) m (a²+b²)
    I_h1 = 0.5 * m_h * head_b**2
    I_h2 = 0.5 * m_h * head_a**2
    I_h3 = 0.5 * m_h * (head_a**2 + head_b**2)

    # ── Handle (thin rod) ──
    # Rod COM is at distance d = head_a + handle_length/2 from racket COM
    # (along -e₁).  Parallel-axis theorem shifts the rod's own MOI.
    d = head_a + 0.5 * handle_length           # offset along e₁
    I_rod_own = (1.0 / 12.0) * m_r * handle_length**2   # rod about its midpoint

    I_r1 = I_rod_own + m_r * d**2   # about e₁ (⊥ to rod, in head plane)
    I_r2 = I_rod_own + m_r * d**2   # about e₂ (same by symmetry)
    I_r3 = 0.5 * m_r * handle_radius**2      # about e₃ ≈ 0 for thin rod

    # ── Total in original geometric axes [e1_geom, e2_geom, e3_geom] ──
    I_geom = np.array([
        I_h1 + I_r1,
        I_h2 + I_r2,
        I_h3 + I_r3,
    ], dtype=np.float64)

    # Sort so body-frame axes are always labeled:
    # axis 0 = smallest I, axis 1 = intermediate I, axis 2 = largest I
    perm = np.argsort(I_geom)
    I_sorted = I_geom[perm]

    # Permutation matrix: columns are old geometric axes selected for new body axes
    P = np.eye(3)[:, perm]

    I_diag = np.diag(I_sorted)
    return I_diag, np.linalg.inv(I_diag), P


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class tennis_racket_3d(gym.Env):
    """Free rigid body (tennis racket) on SO(3).

    Equations of motion (body frame):
        I ω̇ = -ω × (I ω) + τ          (Euler's equations)
        Ṙ   =  R · hat(ω)

    No gravity (free body in space).  Optional external torque (constant
    per trajectory, sampled from N(0, disturbance_torque_std²·I)) models
    slow aerodynamic disturbances.

    State obs (12,): [R.flatten() (9), ω (3)]
    Action    (3,):  body-frame torque τ

    Principal axes convention (body frame):
        axis 0 (e₁): long head axis  — smallest I  — stable rotation
        axis 1 (e₂): short head axis — middle  I  — UNSTABLE rotation
        axis 2 (e₃): handle axis     — largest I  — stable rotation
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        # ── Geometry & mass ──────────────────────────────────────────────
        head_a: float = 0.195,           # head semi-axis (long),   m
        head_b: float = 0.135,           # head semi-axis (short),  m
        handle_length: float = 0.25,     # handle length,            m
        handle_radius: float = 0.015,    # handle radius,            m
        total_mass: float = 0.290,       # total mass,               kg
        head_mass_ratio: float = 0.70,   # fraction of mass in head
        # ── Integration ──────────────────────────────────────────────────
        dt: float = 0.05,
        max_torque: float = 2.0,         # action space bound (API only)
        # ── Initial condition ─────────────────────────────────────────────
        omega_0_scale: float = 2 * np.pi,  # base spin rate, rad/s (~1 rev/s)
        perturb_std: float = 0.05,          # perturbation std, rad/s
        axis_weights: Tuple[float, float, float] = (0.15, 0.70, 0.15),
        # ── Disturbance torque (constant per trajectory) ──────────────────
        disturbance_torque_std: float = 0.0,  # 0 = off; e.g. 0.05 N·m
        # ── Wind / friction (disabled by default) ─────────────────────────
        # [WIND] wind_force_std: float = 0.0,
        friction_coeff: Union[float, Tuple] = 0.0,
        varying_friction: bool = False,
        # ── Misc ──────────────────────────────────────────────────────────
        ori_rep: str = "rotmat",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if ori_rep != "rotmat":
            raise ValueError("Only ori_rep='rotmat' is supported.")

        # ── Geometry ──
        self.head_a         = float(head_a)
        self.head_b         = float(head_b)
        self.handle_length  = float(handle_length)
        self.handle_radius  = float(handle_radius)
        self.total_mass     = float(total_mass)
        self.head_mass_ratio = float(head_mass_ratio)

        # ── Inertia ──
        self.I, self.I_inv, self.P_body_from_geom = compute_racket_inertia(
            head_a, head_b, handle_length, handle_radius,
            total_mass, head_mass_ratio
        )
        # Convenience: diagonal values in order [I1, I2, I3]
        self.I_diag = np.diag(self.I)

        # ── Integration ──
        self.dt         = float(dt)
        self.max_torque = float(max_torque)
        self.ori_rep    = ori_rep

        # ── Initial condition ──
        self.omega_0_scale = float(omega_0_scale)
        self.perturb_std   = float(perturb_std)
        axis_w = np.array(axis_weights, dtype=np.float64)
        self.axis_weights  = axis_w / axis_w.sum()   # normalise to sum=1

        # ── Disturbance torque ──
        self.disturbance_torque_std = float(disturbance_torque_std)
        self._disturbance_torque    = np.zeros(3, dtype=np.float64)

        # ── Wind / friction (disabled) ──
        # [WIND]     self.wind_force_std = float(wind_force_std)
        self.friction_coeff = np.broadcast_to(
            np.asarray(friction_coeff, dtype=np.float64), (3,)).copy()
        self.varying_friction = bool(varying_friction)

        # ── Spaces ──
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque,
            shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # ── State ──
        self._np_rng = np.random.default_rng(seed)
        self.t       = 0.0
        self.last_u  = np.zeros(3, dtype=np.float64)
        self.R       = np.eye(3,    dtype=np.float64)
        self.omega   = np.zeros(3,  dtype=np.float64)

        self._fig = None
        self._ax  = None
        self.render_mode = render_mode

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def intermediate_axis(self) -> int:
        """Index of the intermediate (unstable) principal axis (always 1)."""
        return 1

    def get_state(self):
        return self.R.copy(), self.omega.copy()

    def get_inertia_info(self) -> dict:
        """Return a dict summarising the inertia tensor for logging."""
        return {
            "I1": self.I_diag[0],
            "I2": self.I_diag[1],
            "I3": self.I_diag[2],
            "head_a": self.head_a,
            "head_b": self.head_b,
            "handle_length": self.handle_length,
            "total_mass": self.total_mass,
            "head_mass_ratio": self.head_mass_ratio,
        }

    # ── Seeding ───────────────────────────────────────────────────────────

    def seed(self, seed: Optional[int] = None):
        self._np_rng = np.random.default_rng(seed)

    # ── Wind model (disabled — uncomment to enable) ───────────────────────

    # [WIND] def update_wind(self, t: float) -> np.ndarray:
    # [WIND]     """Return a 3-vector stochastic wind torque increment."""
    # [WIND]     if self.wind_force_std > 0.0:
    # [WIND]         return self._np_rng.normal(0.0, self.wind_force_std, size=3)
    # [WIND]     return np.zeros(3)

    # ── Friction (disabled — uncomment to enable) ─────────────────────────

    def _variable_friction(self, omega: np.ndarray) -> np.ndarray:
        if not self.varying_friction:
            return self.friction_coeff
        speed_term = np.tanh(np.linalg.norm(omega))
        return self.friction_coeff * (1.0 + 0.5 * speed_term)

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.hstack((self.R.reshape(-1), self.omega)).astype(np.float32)

    # ── Dynamics ──────────────────────────────────────────────────────────

    def _compute_omega_rates(
        self,
        omega: np.ndarray,
        u: np.ndarray,
        # [WIND]     dW: np.ndarray, sigma: float
    ):
        """Compute deterministic ω̇ for given angular velocity and torque.

        Euler's equations (body frame):
            I ω̇ = τ_total - ω × (I ω)

        where τ_total includes the control torque u, the per-trajectory
        constant disturbance, and (when enabled) friction and wind.
        """
        tau = u + self._disturbance_torque

        fric = self._variable_friction(omega)
        tau  = tau - fric * omega

        # [WIND] if sigma > 0.0:
        # [WIND]     tau = tau + sigma * dW   # additive stochastic torque

        Iw = self.I @ omega
        omega_dot = self.I_inv @ (tau - np.cross(omega, Iw))
        return omega_dot

    # ── Lie-group Heun integrator ─────────────────────────────────────────

    def _lie_heun_step(
        self,
        R: np.ndarray,
        omega: np.ndarray,
        u: np.ndarray,
        h: float,
        # [WIND] sigma: float, dW: np.ndarray
    ):
        """One Stratonovich-Heun substep on SO(3) × ℝ³.

        Identical structure to windy_pendulum_3d._lie_heun_step, but
        the dynamics are Euler's equations for a free rigid body (no
        gravity, no pivot constraint).
        """
        # Stage 1
        omega_dot_1 = self._compute_omega_rates(omega, u)
        phi_1 = omega * h

        # Predictor
        R_pred     = R @ _exp_so3(phi_1)
        omega_pred = omega + omega_dot_1 * h

        # Stage 2
        omega_dot_2 = self._compute_omega_rates(omega_pred, u)
        phi_2 = omega_pred * h

        # Corrector
        phi_avg = 0.5 * (phi_1 + phi_2)
        R_new   = R @ _exp_so3(phi_avg)
        omega_new = omega + 0.5 * (omega_dot_1 + omega_dot_2) * h

        return R_new, omega_new

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, u):
        u = np.asarray(u, dtype=np.float64).reshape(3)
        self.last_u = u.copy()
        self.t += self.dt

        n_substeps = 10
        dt_sub = self.dt / n_substeps

        # [WIND] sigma = self.wind_force_std

        for _ in range(n_substeps):
            # [WIND] dW = self._np_rng.normal(0.0, np.sqrt(dt_sub), size=3) if sigma > 0 else np.zeros(3)
            self.R, self.omega = self._lie_heun_step(
                self.R, self.omega, u, dt_sub,
                # [WIND] sigma, dW
            )

        # Periodic re-orthogonalisation (numerical safety net)
        if (abs(np.linalg.det(self.R) - 1.0) > 1e-8 or
                np.linalg.norm(self.R.T @ self.R - np.eye(3)) > 1e-8):
            self.R = _project_to_so3(self.R)

        # ── Reward: penalise deviation from intermediate-axis rotation ──
        # Ideal state: ω aligned with e₂ (body axis 1), R arbitrary.
        # We reward |ω · e₂| / |ω| — how close spin is to the unstable axis.
        e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        omega_norm = float(np.linalg.norm(self.omega))
        if omega_norm > 1e-8:
            axis_align = float(np.dot(self.omega / omega_norm, e2)) ** 2
        else:
            axis_align = 0.0
        act_cost = 0.001 * float(np.dot(u, u))
        reward = axis_align - act_cost

        obs  = self._get_obs()
        info = {"disturbance_torque": self._disturbance_torque.copy()}

        return obs, reward, False, False, info

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.

        options keys (all optional)
        ---------------------------
        "R_init"     : (3,3) initial rotation matrix  (projected to SO(3))
        "omega_init" : (3,)  initial angular velocity
        "axis"       : 0, 1, or 2  — which principal axis to spin about;
                       if absent, sampled according to self.axis_weights
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.t      = 0.0
        self.last_u = np.zeros(3, dtype=np.float64)

        if options is None:
            options = {}

        # ── Orientation ──
        if "R_init" in options:
            R0 = _project_to_so3(
                np.asarray(options["R_init"], dtype=np.float64).reshape(3, 3)
            )
        else:
            R0 = _random_rotation(self._np_rng)

        # ── Angular velocity ──
        if "omega_init" in options:
            w0 = np.asarray(options["omega_init"], dtype=np.float64).reshape(3)
        else:
            # Choose rotation axis
            if "axis" in options:
                axis = int(options["axis"])
            else:
                axis = int(self._np_rng.choice(3, p=self.axis_weights))

            # Spin magnitude (positive or negative with equal probability)
            sign  = self._np_rng.choice([-1.0, 1.0])
            speed = sign * self.omega_0_scale

            # Base angular velocity along chosen axis + small perturbation
            w0       = np.zeros(3, dtype=np.float64)
            w0[axis] = speed
            w0      += self._np_rng.normal(0.0, self.perturb_std, size=3)

        # ── Per-trajectory disturbance torque ──
        # Sampled once per episode; held constant throughout.
        # Represents slow aerodynamic asymmetry.
        if self.disturbance_torque_std > 0.0:
            self._disturbance_torque = self._np_rng.normal(
                0.0, self.disturbance_torque_std, size=3
            )
        else:
            self._disturbance_torque = np.zeros(3, dtype=np.float64)

        self.R     = R0
        self.omega = w0

        return self._get_obs(), {}

    # ── Rendering ─────────────────────────────────────────────────────────

    def _init_render(self):
        import matplotlib
        if self.render_mode == "rgb_array":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self._fig = plt.figure(figsize=(7, 7))
        self._ax  = self._fig.add_subplot(111, projection="3d")

    def render(self):
        if self.render_mode is None:
            return

        import matplotlib.pyplot as plt

        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init_render()

        ax = self._ax
        ax.cla()

        # Draw principal axes of the body frame
        axis_len    = 0.25
        axis_colors = ["#e74c3c", "#2ecc71", "#3498db"]
        axis_labels = ["e₁ (stable, long)", "e₂ (unstable)", "e₃ (stable, handle)"]
        origin = np.zeros(3)
        for i in range(3):
            e_body = np.zeros(3); e_body[i] = 1.0
            tip = axis_len * (self.R @ e_body)
            ax.quiver(
                *origin, *tip,
                color=axis_colors[i], linewidth=2.5,
                arrow_length_ratio=0.18,
            )
            ax.text(*(tip * 1.1), axis_labels[i],
                    color=axis_colors[i], fontsize=7, fontweight="bold")

        # Draw a rough ellipse for the racket head
        theta = np.linspace(0, 2 * np.pi, 80)
        head_pts_geom = np.column_stack([
            self.head_a * np.cos(theta),
            self.head_b * np.sin(theta),
            np.zeros_like(theta),
        ])  # (80, 3) in geometric frame

        head_pts_body = (self.P_body_from_geom.T @ head_pts_geom.T).T
        head_pts_world = (self.R @ head_pts_body.T).T
        ax.plot(head_pts_world[:, 0], head_pts_world[:, 1], head_pts_world[:, 2],
                color="#888888", linewidth=1.5, alpha=0.8)

        # Draw handle rod
        handle_start_geom = np.array([-self.head_a, 0.0, 0.0])
        handle_end_geom   = np.array([-self.head_a - self.handle_length, 0.0, 0.0])

        handle_start_body = self.P_body_from_geom.T @ handle_start_geom
        handle_end_body   = self.P_body_from_geom.T @ handle_end_geom
        hs = self.R @ handle_start_body
        he = self.R @ handle_end_body
        ax.plot([hs[0], he[0]], [hs[1], he[1]], [hs[2], he[2]],
                color="#555555", linewidth=4)

        # Draw ω vector
        omega_scale = 0.08
        ov = self.omega * omega_scale
        ax.quiver(*origin, *ov, color="#ff9800", linewidth=2,
                  arrow_length_ratio=0.2, label=f"|ω|={np.linalg.norm(self.omega):.2f}")

        lim = self.head_a * 1.8
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("Tennis Racket — Free Rigid Body on SO(3)", fontweight="bold")
        ax.set_box_aspect([1, 1, 1])

        omega_norm = np.linalg.norm(self.omega)
        e2 = np.array([0.0, 1.0, 0.0])
        align = float(np.dot(self.omega / (omega_norm + 1e-12), e2)) ** 2
        hud = (
            f"t={self.t:.2f}s   "
            f"|ω|={omega_norm:.2f} rad/s   "
            f"e₂-align={align:.3f}"
        )
        ax.text2D(0.02, 0.96, hud, transform=ax.transAxes,
                  fontsize=9, fontfamily="monospace",
                  verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

        if self.render_mode == "human":
            plt.draw(); plt.pause(0.001)
        else:
            self._fig.canvas.draw()
            buf = self._fig.canvas.buffer_rgba()
            return np.asarray(buf)[:, :, :3].copy()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        self._fig = None
        self._ax  = None


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import os

    N_STEPS   = 400
    SAVE_PATH = "videos/tennis_racket_3d.mp4"

    env = tennis_racket_3d(
        disturbance_torque_std=0.0,
        render_mode="rgb_array",
        seed=42,
    )
    # Start spinning about the intermediate axis with a small perturbation
    obs, _ = env.reset(seed=42, options={"axis": 1})
    print("Inertia:", env.get_inertia_info())
    print(f"I1={env.I_diag[0]:.5f}  I2={env.I_diag[1]:.5f}  I3={env.I_diag[2]:.5f} kg·m²")

    frames = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    for step in range(N_STEPS):
        obs, reward, _, _, info = env.step([0.0, 0.0, 0.0])
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if step % 40 == 0:
            R, omega = env.get_state()
            print(
                f"Step {step:4d}  |  reward={reward:.4f}  |  "
                f"|ω|={np.linalg.norm(omega):.3f}  |  "
                f"det(R)={np.linalg.det(R):.8f}"
            )

    env.close()

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    fig_vid, ax_vid = plt.subplots(figsize=(7, 7))
    ax_vid.axis("off")
    im = ax_vid.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return [im]

    anim = FuncAnimation(fig_vid, _update, frames=len(frames),
                         interval=1000 / 30, blit=True)
    anim.save(SAVE_PATH, writer="ffmpeg", fps=30, dpi=100)
    plt.close(fig_vid)
    print(f"Saved to {SAVE_PATH}")


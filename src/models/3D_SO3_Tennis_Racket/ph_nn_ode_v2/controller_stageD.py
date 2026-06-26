"""Stage D IDA-PBC controller for tennis racket stabilization.

The tennis racket environment applies the action directly as a body-frame torque:
  τ_total = u + τ_disturbance
  ω̇ = M⁻¹(τ_total - ω × Mω)

Because the input coupling is the identity (direct torque → ω̇ via M⁻¹), the
port-Hamiltonian g_net converges to I₃.  The IDA-PBC control law simplifies to:

  u = K_p · (ω* - ω)          [pure proportional on angular-velocity error]

Stability analysis at ω* = ω*₃ e₃ (spinning about handle):
  - δω₃ mode:       λ = -K_p/I₃                (always stable, τ ≈ I₃/K_p)
  - δω₁,δω₂ modes: Re(λ) = -(K_p/2)(1/I₁+1/I₂) (stable for all K_p > 0)
  - For K_p > I₁I₂/(I₃-I₁)(I₃-I₂)·ω*₃²·(something): overdamped
  Empirical recommendation for cfg0: K_p ≥ 0.07 for no oscillations.

The model is used only to read g_net when `use_model_g=True`.  For Stage D with
g=I₃ (known physics), set `use_model_g=False` (the default).
"""
import torch
import numpy as np


class PDBodyFrameController:
    """Proportional controller for stabilising ω → ω*.

    Parameters
    ----------
    omega_star : array-like, shape (3,)
        Target angular velocity in body frame [rad/s].
    Kp : float
        Proportional gain [N·m·s/rad].
    clip : float, optional
        Hard torque clip [N·m] (matches env max_torque=2.0).
    use_model_g : bool
        If True, use the model's g_net as input coupling:
          u = g⁺(R,I) · K_p(ω* - ω)
        If False (default), assume g = I₃:
          u = K_p(ω* - ω)  [exact for direct-torque actuator]
    model : DissipativeSO3HamNODE, optional
        Required only when use_model_g=True.
    I1, I2, I3 : float, optional
        Principal inertia values. Required when use_model_g=True.
    """
    def __init__(
        self,
        omega_star,
        Kp: float = 0.10,
        clip: float = 2.0,
        use_model_g: bool = False,
        model=None,
        I1: float = None,
        I2: float = None,
        I3: float = None,
    ):
        self.omega_star = np.asarray(omega_star, dtype=np.float64).reshape(3)
        self.Kp         = float(Kp)
        self.clip       = float(clip)
        self.use_model_g = use_model_g
        self.model      = model
        self.I1         = I1
        self.I2         = I2
        self.I3         = I3

    # ── Internal helpers ───────────────────────────────────────────────────

    def _tau_desired(self, omega: np.ndarray) -> np.ndarray:
        """Compute desired body-frame torque from current ω."""
        return self.Kp * (self.omega_star - omega)

    def _invert_g(self, R_flat: np.ndarray, tau_des: np.ndarray) -> np.ndarray:
        """Compute u = g⁺ · τ_des using the learned g_net.

        Uses least-squares pseudo-inverse: min_u ‖g·u − τ_des‖².
        """
        with torch.no_grad():
            q_ext = torch.tensor(
                np.concatenate([R_flat, [self.I1, self.I2, self.I3]]),
                dtype=torch.float32,
            ).unsqueeze(0)                    # (1, 12)
            g_mat = self.model.g_net(q_ext)   # (1, 3, 3)
            g_np  = g_mat.squeeze(0).numpy()  # (3, 3)

        u, _, _, _ = np.linalg.lstsq(g_np, tau_des, rcond=None)
        return u

    # ── Public API ─────────────────────────────────────────────────────────

    def __call__(
        self,
        R_flat: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        """Compute the control torque for the current state.

        Parameters
        ----------
        R_flat : (9,) float64 — flattened rotation matrix (row-major)
        omega  : (3,) float64 — body-frame angular velocity [rad/s]

        Returns
        -------
        u : (3,) float64 — body-frame torque to apply [N·m]
        """
        tau_des = self._tau_desired(np.asarray(omega, dtype=np.float64))

        if self.use_model_g and self.model is not None:
            u = self._invert_g(np.asarray(R_flat, dtype=np.float64), tau_des)
        else:
            u = tau_des

        # Clip to actuator limits
        return np.clip(u, -self.clip, self.clip)

    # ── Diagnostics ────────────────────────────────────────────────────────

    def omega_error(self, omega: np.ndarray) -> float:
        """‖ω - ω*‖₂  in rad/s."""
        return float(np.linalg.norm(np.asarray(omega) - self.omega_star))

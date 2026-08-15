"""IDA-PBC attitude controller for the windy pendulum: the racket's
`GeometricAttitudeController` PD law plus an exact feedforward cancellation
of the gravity torque at the CURRENT orientation.

Without cancellation, a bare PD controller holding any target away from the
two natural gravity equilibria (straight down / straight up) would settle
with a permanent steady-state droop -- the PD term only drives orientation
error to zero, it has no way to know it also needs to fight a persistent
disturbance. Subtracting the actual gravity torque at every step is what
makes this genuine IDA-PBC ("damping assignment" from the PD term,
"interconnection assignment" from cancelling gravity's own contribution and
replacing it with a torque whose only remaining equilibrium is R_star)
rather than a controller that only happens to work in the gravity-free case
the racket's own Stage F was validated for.
"""
import numpy as np

from controllers import GeometricAttitudeController
from pendulum_dynamics import gravity_torque_body


class GravityCompensatedAttitudeController:
    """u = -K_R*vee(logm(R_star.T @ R)) - K_p*(omega - omega_star)
        - tau_gravity_body(R, com_body, total_mass, g)

    The minus sign on the gravity term is exact cancellation, not a typo:
    the real dynamics are I*wdot = tau_gravity(R) + u - w x (Iw), so setting
    u's gravity piece to -tau_gravity(R) leaves exactly tau_pd - w x (Iw),
    identical in form to the racket's gravity-free controlled dynamics.
    """

    def __init__(self, R_star, omega_star, com_body, total_mass, g=9.81, K_R=0.10, K_p=0.10, clip=2.0):
        # Internal PD term is deliberately unclipped -- clipping must happen
        # once, on the final combined torque, not on the PD piece alone
        # before gravity cancellation is added (clipping first would leave
        # the gravity term uncancelled whenever the PD piece alone already
        # saturated, breaking the exact-cancellation property this class
        # exists for).
        self._base = GeometricAttitudeController(R_star, omega_star, K_R, K_p, clip=np.inf)
        self.com_body = np.asarray(com_body, dtype=np.float64).reshape(3)
        self.total_mass = float(total_mass)
        self.g = float(g)
        self.clip = float(clip)

    def __call__(self, R, omega):
        """Return body-frame torque u (3,) given current orientation R (3,3)
        and angular velocity omega (3,)."""
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        tau_pd = self._base(R, omega)
        tau_g = gravity_torque_body(R, self.com_body, self.total_mass, self.g)
        return np.clip(tau_pd - tau_g, -self.clip, self.clip)


def default_gains_for_attitude_control(I, target_omega_n=2.0, damping_ratio=1.0):
    """K_R/K_p scaled to the ACTUAL current inertia, not a fixed constant --
    same reasoning as controllers.py's default_kp_for_axis_control: this
    system's I_transverse (the moment governing the actual visible swing;
    I[1] and I[2] are always equal, see pendulum_geometry.py) varies by
    roughly 240x across the geometry sliders' own range (a light, short
    pendulum vs. a heavy, long one). A single hardcoded gain pair tuned for
    one geometry would be badly overdamped for a light/short pendulum and
    underdamped/oscillatory for a heavy/long one -- dragging the geometry
    sliders would silently change how well Control converges. Scaling by
    the live I keeps the closed-loop response (natural frequency
    target_omega_n, damping_ratio) the same regardless of geometry.

    Standard 2nd-order system relations: K_R = omega_n^2 * I,
    K_p = 2*damping_ratio*omega_n*I. damping_ratio=1.0 (critical damping) is
    the textbook "converges as fast as possible with no overshoot" choice.

    @param I [I1,I2,I3] (ascending; I[1]==I[2] for this system)
    @param target_omega_n rad/s, the desired closed-loop natural frequency
    @param damping_ratio 1.0 = critically damped
    @returns (K_R, K_p)
    """
    I_transverse = max(I[1], I[2])
    K_R = target_omega_n ** 2 * I_transverse
    K_p = 2.0 * damping_ratio * target_omega_n * I_transverse
    return K_R, K_p

"""Gravity + friction torques for the windy pendulum, as a `torque_fn` for
`rigid_body.py`'s `integrate_full` (which already accepts an arbitrary
`torque_fn(t, w, q)` -- no changes needed there, see its docstring).

Ported from the exact gravity-torque formula in LieSPHGP's
`envs/windy_pendulum_3d.py` (`_compute_omega_rates`), generalized from that
file's point-mass-at-distance-l assumption to this project's actual rod+bob
center of mass (`pendulum_geometry.py`'s `com`), so it's correct for whichever
axis `eigh` happens to label as the body frame's own COM direction.
"""
import numpy as np


def gravity_torque_body(R, com_body, total_mass, g=9.81):
    """Body-frame torque from gravity acting at the center of mass, for a
    rigid body pivoting about a fixed point (the origin) that is NOT its COM.

    tau_g = R^T @ (r_world x F_g), where r_world = R @ com_body is the COM's
    current position in world coordinates and F_g = -total_mass * g * z_hat
    (gravity pulls along world -z). Zero exactly when the COM is aligned with
    +/- world z (hanging straight down or balanced straight up) -- both true
    equilibria of a gravity pendulum, one stable, one unstable.

    @param R (3,3) current orientation
    @param com_body (3,) center of mass in the body (principal-axis) frame
    @param total_mass float
    @param g float gravitational acceleration
    @returns (3,) body-frame torque
    """
    r_world = R @ np.asarray(com_body, dtype=np.float64)
    Fg = np.array([0.0, 0.0, -total_mass * g])
    return R.T @ np.cross(r_world, Fg)


def friction_torque(omega, friction_coeff):
    """Simple linear (viscous) damping opposing angular velocity.

    @param omega (3,) angular velocity, body frame
    @param friction_coeff float
    @returns (3,) body-frame torque
    """
    return -friction_coeff * np.asarray(omega, dtype=np.float64)

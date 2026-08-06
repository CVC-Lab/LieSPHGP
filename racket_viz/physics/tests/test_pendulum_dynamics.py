import numpy as np
import pytest

from pendulum_dynamics import gravity_torque_body, friction_torque

# Rotation mapping body +x to world +z exactly (R_y(-90 deg)): used below to
# place a com_body lying along the body x-axis (as pendulum_geometry.py's
# build_pendulum always constructs it) at world position straight "up".
R_X_TO_WORLD_Z = np.array([
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
])


def test_zero_torque_when_com_points_straight_up_or_down():
    """Both hanging straight down and balanced straight up are equilibria of a
    gravity pendulum (zero torque) -- one stable, one unstable, but both
    exactly zero net torque since the COM offset is parallel to gravity."""
    com_body = np.array([0.3, 0.0, 0.0])
    tau_up = gravity_torque_body(R_X_TO_WORLD_Z, com_body, total_mass=1.0, g=9.81)
    np.testing.assert_allclose(tau_up, [0.0, 0.0, 0.0], atol=1e-10)

    # 180 deg rotation about world x, composed on top, flips the com's world
    # position from +z to -z (hanging down instead of balanced up) while
    # staying a valid rotation (det=+1).
    R_flip_z = np.diag([1.0, -1.0, -1.0])
    R_down = R_flip_z @ R_X_TO_WORLD_Z
    tau_down = gravity_torque_body(R_down, com_body, total_mass=1.0, g=9.81)
    np.testing.assert_allclose(tau_down, [0.0, 0.0, 0.0], atol=1e-10)


def test_nonzero_torque_off_equilibrium():
    com_body = np.array([0.3, 0.0, 0.0])
    tau = gravity_torque_body(np.eye(3), com_body, total_mass=1.0, g=9.81)
    assert np.linalg.norm(tau) > 1e-6


def test_torque_scales_with_mass_and_g():
    com_body = np.array([0.3, 0.0, 0.0])
    tau1 = gravity_torque_body(np.eye(3), com_body, total_mass=1.0, g=9.81)
    tau2 = gravity_torque_body(np.eye(3), com_body, total_mass=2.0, g=9.81)
    np.testing.assert_allclose(tau2, 2.0 * tau1, atol=1e-12)


def test_friction_torque_opposes_motion():
    omega = np.array([1.0, -2.0, 3.0])
    tau = friction_torque(omega, friction_coeff=0.5)
    np.testing.assert_allclose(tau, -0.5 * omega)


def test_friction_torque_zero_at_rest():
    tau = friction_torque(np.zeros(3), friction_coeff=0.5)
    np.testing.assert_allclose(tau, np.zeros(3))

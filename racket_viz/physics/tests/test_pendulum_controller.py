import numpy as np
import pytest

from pendulum_controller import GravityCompensatedAttitudeController, default_gains_for_attitude_control
from pendulum_dynamics import gravity_torque_body


COM_BODY = np.array([0.3, 0.0, 0.0])
TOTAL_MASS = 1.0
G = 9.81


def test_holds_an_off_equilibrium_target_by_exactly_cancelling_gravity():
    """A stationary target away from either gravity equilibrium (e.g. the
    pendulum held out horizontally) can ONLY be a true steady state
    (omega stays exactly 0 forever) if the controller supplies exactly
    -tau_gravity there -- this is what makes it genuine IDA-PBC (shaping the
    closed-loop equilibrium to sit at R_star) rather than blind PD, which
    would leave a permanent steady-state droop against gravity instead."""
    R_star = np.eye(3)  # com_body=[0.3,0,0] points world +x here -- sideways, not aligned with gravity's z-axis, so tau_gravity != 0
    omega_star = np.zeros(3)
    controller = GravityCompensatedAttitudeController(
        R_star, omega_star, COM_BODY, TOTAL_MASS, g=G, K_R=0.5, K_p=0.5, clip=1e6
    )

    # At the target, at rest: the PD terms (e_R, e_w) are exactly zero, so the
    # controller's entire output must be pure gravity cancellation.
    u = controller(R_star, omega_star)
    tau_g_at_target = gravity_torque_body(R_star, COM_BODY, TOTAL_MASS, G)
    np.testing.assert_allclose(u, -tau_g_at_target, atol=1e-10)
    assert np.linalg.norm(u) > 1e-6  # sanity: this target really isn't a free equilibrium


def test_reduces_to_pd_plus_cancellation_away_from_target():
    """Off-target, the output is exactly the base PD law plus the (current-R)
    gravity cancellation term -- verifies the composition, not just the
    zero-error special case above."""
    R_star = np.eye(3)
    omega_star = np.zeros(3)
    controller = GravityCompensatedAttitudeController(
        R_star, omega_star, COM_BODY, TOTAL_MASS, g=G, K_R=0.5, K_p=0.5, clip=1e6
    )

    # A different orientation, with some angular velocity.
    theta = 0.4
    R = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ])
    omega = np.array([0.1, -0.2, 0.05])

    from controllers import GeometricAttitudeController
    base = GeometricAttitudeController(R_star, omega_star, K_R=0.5, K_p=0.5, clip=1e6)
    expected = base(R, omega) - gravity_torque_body(R, COM_BODY, TOTAL_MASS, G)

    u = controller(R, omega)
    np.testing.assert_allclose(u, expected, atol=1e-10)


def test_clip_still_applies_to_combined_torque():
    R_star = np.eye(3)
    omega_star = np.zeros(3)
    controller = GravityCompensatedAttitudeController(
        R_star, omega_star, COM_BODY, TOTAL_MASS, g=G, K_R=50.0, K_p=50.0, clip=0.5
    )
    theta = 1.2
    R = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ])
    u = controller(R, np.array([3.0, -3.0, 3.0]))
    assert np.all(np.abs(u) <= 0.5 + 1e-9)


def test_default_gains_scale_with_inertia_not_fixed():
    """A heavier/longer pendulum (larger I_transverse) must get proportionally
    larger gains -- otherwise the same fixed gain pair would respond far
    more sluggishly for it than for a light/short one (a real risk this
    function exists specifically to avoid)."""
    K_R_small, K_p_small = default_gains_for_attitude_control([0.01, 0.1, 0.1])
    K_R_large, K_p_large = default_gains_for_attitude_control([0.01, 10.0, 10.0])
    assert K_R_large == pytest.approx(K_R_small * 100)
    assert K_p_large == pytest.approx(K_p_small * 100)


def test_default_gains_are_critically_damped_by_default():
    """damping_ratio = K_p / (2*sqrt(K_R*I)) should equal exactly 1.0 (the
    requested default) for whatever I is passed in."""
    I = [0.02, 0.9, 0.9]
    I_transverse = 0.9
    K_R, K_p = default_gains_for_attitude_control(I)
    damping_ratio = K_p / (2 * np.sqrt(K_R * I_transverse))
    assert damping_ratio == pytest.approx(1.0)


def test_default_gains_scale_with_requested_natural_frequency():
    K_R_slow, K_p_slow = default_gains_for_attitude_control([0.01, 1.0, 1.0], target_omega_n=1.0)
    K_R_fast, K_p_fast = default_gains_for_attitude_control([0.01, 1.0, 1.0], target_omega_n=2.0)
    assert K_R_fast == pytest.approx(K_R_slow * 4)  # K_R ~ omega_n^2
    assert K_p_fast == pytest.approx(K_p_slow * 2)  # K_p ~ omega_n

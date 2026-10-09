"""Tests for cartpole_dynamics.py -- the Hamiltonian-formulation cart-pole
core (state (x, theta, p_x, p_theta), NOT gymnasium's (x, x_dot, theta,
theta_dot)). See that module's docstring for the theta=0-is-upright
convention and why H is conserved exactly when F=0 and there's no wind.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import cartpole_dynamics as cpd

DEFAULT = dict(mp=0.1, mc=1.0, l=0.5, g=9.8)
DEFAULT_NO_G = dict(mp=0.1, mc=1.0, l=0.5)


def test_velocities_and_momenta_are_exact_inverses():
    """`velocities`/`momenta` must be exact inverse maps for any theta --
    this is the property a reset that samples (x, x_dot, theta, theta_dot)
    uniformly and converts to (x, theta, p_x, p_theta) relies on, so it
    doesn't silently distort the initial-condition distribution."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        theta = rng.uniform(-np.pi, np.pi)
        xdot, thdot = rng.uniform(-2, 2, size=2)
        px, pth = cpd.momenta(theta, xdot, thdot, **DEFAULT_NO_G)
        xdot2, thdot2 = cpd.velocities(theta, px, pth, **DEFAULT_NO_G)
        np.testing.assert_allclose([xdot2, thdot2], [xdot, thdot], atol=1e-10)


def test_mass_matrix_symmetric_positive_definite():
    """A physically valid mass matrix must be symmetric and positive
    definite for every theta -- otherwise `velocities`'s implicit M^-1
    isn't even a well-defined kinetic-energy metric."""
    for theta in np.linspace(-np.pi, np.pi, 13):
        M = cpd.mass_matrix(theta, **DEFAULT_NO_G)
        np.testing.assert_allclose(M, M.T)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)


def test_velocities_at_upright_matches_hand_derived_closed_form():
    """At theta=0 (upright), M(0) is a fixed 2x2 matrix whose closed-form
    inverse this project relies on elsewhere (cartpole_diagnostics.py's
    Gain A/B) -- pin raw `velocities` output against that same closed form
    directly so a future edit to either can't silently drift apart."""
    mp, mc, l = DEFAULT_NO_G["mp"], DEFAULT_NO_G["mc"], DEFAULT_NO_G["l"]
    mtot = mc + mp
    a, b, d = (4.0 / 3.0) * mp * l ** 2, -mp * l, mtot
    det = mtot * a - b ** 2
    px, pth = 0.3, -0.7
    xdot, thdot = cpd.velocities(0.0, px, pth, **DEFAULT_NO_G)
    assert xdot == pytest.approx((a * px + b * pth) / det)
    assert thdot == pytest.approx((b * px + d * pth) / det)


def test_fixed_points_are_stationary_at_any_cart_position():
    """theta in {0, pi, -pi} with zero momenta and zero force is a true
    fixed point at ANY cart position x -- the frictionless track has no
    potential in x, so this must hold regardless of x (translation
    invariance)."""
    for theta in [0.0, np.pi, -np.pi]:
        for x in [-1.0, 0.0, 2.0]:
            z = np.array([x, theta, 0.0, 0.0])
            dz = cpd.drift(z, F=0.0, **DEFAULT)
            np.testing.assert_allclose(dz, np.zeros(4), atol=1e-12)


def test_upright_is_unstable_hanging_is_stable():
    """A tiny push off theta=0 (upright) must grow -- off theta=pi
    (hanging) must stay bounded -- verified by direct integration rather
    than assumed from the diagnostics formulas (which this test is
    independent of). T=1.0s is about 4 instability e-folds (see
    cartpole_diagnostics.py's instability_time ~= 0.25s at these defaults);
    a bare theta-only perturbation splits roughly evenly onto the growing
    and decaying eigenmodes, so the observed growth (numerically confirmed
    ~26x at these settings) is closer to (1/2)*e^(t/T) than a full e^(t/T) --
    the >5x threshold below has wide margin either way."""
    def rhs(t, z):
        return cpd.drift(z, F=0.0, **DEFAULT)

    z0_up = np.array([0.0, 1e-3, 0.0, 0.0])
    sol_up = solve_ivp(rhs, [0, 1.0], z0_up, max_step=1e-3)
    assert abs(sol_up.y[1, -1]) > 5 * 1e-3

    z0_down = np.array([0.0, np.pi + 1e-3, 0.0, 0.0])
    sol_down = solve_ivp(rhs, [0, 0.5], z0_down, max_step=1e-3)
    assert abs(sol_down.y[1, -1] - np.pi) < 0.5


def test_hamiltonian_conserved_when_force_free():
    """No friction term exists anywhere in `drift` (unlike the windy
    pendulum's explicit friction_torque) -- so with F=0 and no wind, H must
    be EXACTLY conserved, the same free-scenario invariant already used to
    validate the racket and pendulum."""
    def rhs(t, z):
        return cpd.drift(z, F=0.0, **DEFAULT)

    z0 = np.array([0.3, 0.4, 0.2, -0.3])
    H0 = cpd.hamiltonian(z0, **DEFAULT)
    sol = solve_ivp(rhs, [0, 3.0], z0, max_step=1e-3, dense_output=True)
    for t in np.linspace(0, 3.0, 50):
        Ht = cpd.hamiltonian(sol.sol(t), **DEFAULT)
        assert Ht == pytest.approx(H0, abs=1e-6)


def test_constant_force_changes_hamiltonian_at_the_rate_of_the_work_it_does():
    """A nonzero F does real work on the system (power = F*x_dot), so H
    must NOT be conserved -- and its instantaneous rate of change must
    equal that power exactly, cross-checked via finite difference (mirrors
    the racket's/pendulum's own controlled-scenario-does-work invariant)."""
    z0 = np.array([0.0, 0.2, 0.5, -0.3])
    F = 2.0
    dt = 1e-6
    H0 = cpd.hamiltonian(z0, **DEFAULT)
    dz = cpd.drift(z0, F=F, **DEFAULT)
    z1 = z0 + dz * dt
    H1 = cpd.hamiltonian(z1, **DEFAULT)
    xdot, _ = cpd.velocities(z0[1], z0[2], z0[3], **DEFAULT_NO_G)
    expected_rate = F * xdot
    actual_rate = (H1 - H0) / dt
    assert actual_rate == pytest.approx(expected_rate, rel=1e-2)


def test_termination_bounds_match_reference_env():
    """Pin the exact failure thresholds from `summer-2026/cartpole.py` as a
    regression -- x_threshold=2.4 m, theta_threshold=12 deg -- since these
    feed directly into the sidebar's failure/"Survived" readout."""
    assert cpd.X_THRESHOLD == pytest.approx(2.4)
    assert cpd.THETA_THRESHOLD_RADIANS == pytest.approx(12 * 2 * np.pi / 360)

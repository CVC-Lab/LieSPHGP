"""Tests for cartpole_diagnostics.py -- closed-form fragility metrics for
the upright equilibrium (Gain A, Gain B, Instability Time), all fully
ANALYTIC properties of the plant (mp, mc, l, g). None of this depends on
any trained controller.

Reference values are cross-checked two independent ways: against the
coworker's live demo's displayed numbers at its default sliders
(https://claude.ai/code/artifact/9697efad-c379-4930-a69c-889cd971f315:
pole mass 0.10 kg, cart mass 1.00 kg, pole half-length 0.50 m, gravity
9.8 m/s^2 -> Gain A=32.2, Gain B=1.46, Instability Time=0.252s), AND against
the eigenvalues of a numerically-differentiated full 4x4 Jacobian of
`cartpole_dynamics.drift` at the upright fixed point. Two independent
derivations agreeing to several significant figures is real evidence the
closed forms are right, not just plausible-looking.
"""
import numpy as np
import pytest

import cartpole_diagnostics as diag
import cartpole_dynamics as cpd

DEFAULT = dict(mp=0.1, mc=1.0, l=0.5, g=9.8)


def test_gain_a_matches_demo_reference_value():
    assert diag.gain_a(**DEFAULT) == pytest.approx(32.195, abs=0.01)


def test_gain_b_matches_demo_reference_value():
    assert diag.gain_b(**DEFAULT) == pytest.approx(1.4634, abs=0.001)


def test_instability_time_matches_demo_reference_value():
    assert diag.instability_time(**DEFAULT) == pytest.approx(0.2518, abs=0.001)


def test_gains_independent_of_gravity():
    """Gain A/B are pure mass-matrix entries at theta=0 -- gravity plays no
    role in either of them, only in Instability Time."""
    a1, b1 = diag.gain_a(**DEFAULT), diag.gain_b(**DEFAULT)
    other_g = dict(DEFAULT, g=50.0)
    a2, b2 = diag.gain_a(**other_g), diag.gain_b(**other_g)
    assert a1 == pytest.approx(a2)
    assert b1 == pytest.approx(b2)


def test_instability_time_increases_with_pole_length():
    """A longer pole falls MORE slowly from upright (matches the everyday
    intuition that a broomstick is easier to balance than a pencil) --
    verified across the demo's own pole-half-length slider range."""
    short = diag.instability_time(mp=0.1, mc=1.0, l=0.2, g=9.8)
    long = diag.instability_time(mp=0.1, mc=1.0, l=2.0, g=9.8)
    assert long > short


def test_instability_time_decreases_with_gravity():
    """Stronger gravity means a stronger destabilizing torque at any given
    tilt -- the pole should fall faster, not slower."""
    weak_g = diag.instability_time(mp=0.1, mc=1.0, l=0.5, g=4.9)
    strong_g = diag.instability_time(mp=0.1, mc=1.0, l=0.5, g=19.6)
    assert strong_g < weak_g


def test_gains_and_instability_time_match_full_jacobian_eigenvalues():
    """Cross-check against a SECOND, independent derivation: the full 4x4
    numerical Jacobian of `cartpole_dynamics.drift` at the upright fixed
    point has eigenvalues {0, 0, +lambda, -lambda} (the two zeros are the
    free cart-position/momentum directions on a frictionless track) with
    lambda == sqrt(gain_a * mp*g*l) == 1/instability_time."""
    mp, mc, l, g = DEFAULT["mp"], DEFAULT["mc"], DEFAULT["l"], DEFAULT["g"]
    eps = 1e-6
    z0 = np.zeros(4)

    def drift(z):
        return cpd.drift(z, F=0.0, mp=mp, mc=mc, l=l, g=g)

    J = np.zeros((4, 4))
    for i in range(4):
        dz = np.zeros(4)
        dz[i] = eps
        J[:, i] = (drift(z0 + dz) - drift(z0 - dz)) / (2 * eps)

    eigvals = np.linalg.eigvals(J)
    lam = max(eigvals.real)
    assert lam == pytest.approx(1.0 / diag.instability_time(**DEFAULT), rel=1e-4)
    assert lam ** 2 == pytest.approx(diag.gain_a(**DEFAULT) * mp * g * l, rel=1e-4)

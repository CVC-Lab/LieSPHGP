"""Tests for cartpole_wind.py -- the per-step stochastic Heun
predictor-corrector, ported from `summer-2026/cartpole.py`'s `diffusion`/
`integrator` methods. See that module's docstring for why the random
number STREAM itself isn't cross-validated against the eventual JS port,
only the deterministic mechanics and the noise's statistical calibration.
"""
import numpy as np
import pytest

import cartpole_dynamics as cpd
import cartpole_wind as wind

DEFAULT = dict(mp=0.1, mc=1.0, l=0.5, g=9.8)
DEFAULT_NO_G = dict(mp=0.1, mc=1.0, l=0.5)


class _FixedGenerator:
    """Stub with the one method `heun_step` needs, always returning a fixed
    value -- lets a test compute the EXACT expected output by hand instead
    of relying on statistics."""
    def __init__(self, value):
        self.value = value

    def standard_normal(self):
        return self.value


def test_diffusion_equals_sigma_gust_at_rest_with_no_turbulence_term():
    assert wind.diffusion(0.0, sigma_gust=0.002, sigma_turb=0.0) == pytest.approx(0.002)


def test_diffusion_increases_with_angular_velocity():
    lo = wind.diffusion(0.1, sigma_gust=0.002, sigma_turb=0.001)
    hi = wind.diffusion(5.0, sigma_gust=0.002, sigma_turb=0.001)
    assert hi > lo


def test_diffusion_has_zero_derivative_at_origin_not_a_kink():
    """sqrt(theta_dot^2 + smooth_eps^2) is smooth (differentiable, zero
    slope) at theta_dot=0 -- unlike a bare |theta_dot|, whose one-sided
    secant slope stays roughly CONSTANT as the step shrinks (a genuine
    kink). Comparing the one-sided secant at two very different step sizes
    tells them apart: a smooth zero-derivative point shrinks the secant
    roughly proportionally to the step; a kink's secant does not shrink at
    all. (A naive left-vs-right-secant comparison does NOT distinguish
    these two cases -- both are even functions with a symmetric sign flip
    either way; this is deliberately not that check.)"""
    f = lambda td: wind.diffusion(td, sigma_gust=0.0, sigma_turb=1.0, smooth_eps=1e-3)
    h1, h2 = 1e-4, 1e-6
    secant_h1 = (f(h1) - f(0.0)) / h1
    secant_h2 = (f(h2) - f(0.0)) / h2
    assert secant_h2 / secant_h1 == pytest.approx(h2 / h1, rel=0.2)
    assert secant_h2 / secant_h1 < 0.1  # a true kink would give ~1.0 here


def test_heun_step_matches_hand_computed_value_with_fixed_noise():
    """Inject a fake generator returning a fixed standard-normal draw and
    hand-compute the exact predictor-corrector result -- an exact-value
    test of the Heun mechanics, independent of any statistical claim."""
    z0 = np.array([0.1, 0.05, 0.2, -0.1])
    F = 1.0
    dt = 0.02
    sigma_gust, sigma_turb, smooth_eps = 0.002, 0.001, 1e-3

    dW = np.sqrt(dt) * 0.7
    G = np.array([0.0, 0.0, 0.0, 1.0])
    _, thdot0 = cpd.velocities(z0[1], z0[2], z0[3], **DEFAULT_NO_G)
    s0 = wind.diffusion(thdot0, sigma_gust, sigma_turb, smooth_eps)
    f0 = cpd.drift(z0, F, **DEFAULT)
    z_t = z0 + f0 * dt + s0 * dW * G
    _, thdot1 = cpd.velocities(z_t[1], z_t[2], z_t[3], **DEFAULT_NO_G)
    s1 = wind.diffusion(thdot1, sigma_gust, sigma_turb, smooth_eps)
    f1 = cpd.drift(z_t, F, **DEFAULT)
    expected = z0 + 0.5 * (f0 + f1) * dt + 0.5 * (s0 + s1) * dW * G

    actual = wind.heun_step(z0, F, dt, _FixedGenerator(0.7), sigma_gust=sigma_gust,
                             sigma_turb=sigma_turb, smooth_eps=smooth_eps, **DEFAULT)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_heun_step_reduces_to_plain_heun_ode_step_when_sigma_zero():
    """With sigma_gust=sigma_turb=0, the noise term vanishes identically
    regardless of what the generator draws -- confirming wind can be fully
    disabled, not just made small."""
    z0 = np.array([0.0, 0.3, 0.1, -0.2])
    F = 0.5
    dt = 0.02

    out_a = wind.heun_step(z0, F, dt, _FixedGenerator(5.0), sigma_gust=0.0, sigma_turb=0.0, **DEFAULT)
    out_b = wind.heun_step(z0, F, dt, _FixedGenerator(-5.0), sigma_gust=0.0, sigma_turb=0.0, **DEFAULT)
    np.testing.assert_allclose(out_a, out_b, atol=1e-12)

    f0 = cpd.drift(z0, F, **DEFAULT)
    z_t = z0 + f0 * dt
    f1 = cpd.drift(z_t, F, **DEFAULT)
    expected = z0 + 0.5 * (f0 + f1) * dt
    np.testing.assert_allclose(out_a, expected, atol=1e-12)


def test_heun_step_reproducible_with_independently_seeded_generators():
    """Two independent `np.random.default_rng(42)` instances, stepped the
    same number of times, must produce IDENTICAL trajectories -- matching
    the existing convention (WindyPendulumEnv, DECISIONS.md 2026-07-28)
    that a run is reproducible from its seed alone."""
    def rollout(rng):
        z = np.array([0.0, 0.05, 0.0, 0.0])
        for _ in range(50):
            z = wind.heun_step(z, 0.0, 0.02, rng, sigma_gust=0.002, sigma_turb=0.001, **DEFAULT)
        return z

    z_a = rollout(np.random.default_rng(42))
    z_b = rollout(np.random.default_rng(42))
    np.testing.assert_allclose(z_a, z_b, atol=1e-12)


def test_different_seeds_diverge():
    def rollout(rng):
        z = np.array([0.0, 0.05, 0.0, 0.0])
        for _ in range(50):
            z = wind.heun_step(z, 0.0, 0.02, rng, sigma_gust=0.002, sigma_turb=0.001, **DEFAULT)
        return z

    z_a = rollout(np.random.default_rng(1))
    z_b = rollout(np.random.default_rng(2))
    assert not np.allclose(z_a, z_b)


def test_wind_noise_variance_scales_with_dt():
    """The dominant term in one Heun step's noise is s0*dW ~ N(0, s0^2*dt) --
    over many independent one-step trials from the same z0, the sample
    variance of the resulting p_theta kick should scale linearly with dt
    (the defining signature of a diffusion term, not a bug that's secretly
    proportional to dt^2 or dt^0). Seeded for a reproducible test; the
    tolerance is generous (this is a statistical check, not an exact one) --
    confirmed numerically beforehand to land near a ratio of 4.0 for a 4x
    change in dt, well clear of this tolerance."""
    z0 = np.array([0.0, 0.05, 0.0, 0.0])
    sigma_gust, sigma_turb = 0.002, 0.001
    n = 4000

    def sample_pth_kick(dt, seed):
        rng = np.random.default_rng(seed)
        kicks = np.empty(n)
        for i in range(n):
            z1 = wind.heun_step(z0, 0.0, dt, rng, sigma_gust=sigma_gust, sigma_turb=sigma_turb, **DEFAULT)
            kicks[i] = z1[3] - z0[3]
        return kicks

    var_small = np.var(sample_pth_kick(0.005, seed=10))
    var_large = np.var(sample_pth_kick(0.02, seed=11))
    ratio = var_large / var_small
    assert ratio == pytest.approx(4.0, rel=0.25)

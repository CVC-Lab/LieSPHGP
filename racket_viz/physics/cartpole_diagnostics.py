"""Closed-form fragility metrics for the cart-pole's upright equilibrium:
Gain A (wind -> theta_dot), Gain B (force -> theta_dot), and Instability
Time (e-folding time of a small tip-over from upright).

All three are ANALYTIC properties of the plant alone (mp, mc, l, g) --
none of this depends on any trained controller, and none of it needs the
`ct_sac` policy to exist. This is exactly the sidebar diagnostics block
shown in the coworker's live demo
(https://claude.ai/code/artifact/9697efad-c379-4930-a69c-889cd971f315):
"WIND -> THETA (GAIN A)", "FORCE -> THETA (GAIN B)", "T INSTABILITY TIME".

Derivation: linearize `cartpole_dynamics.drift` about the upright fixed
point (theta=0, p_x=p_theta=0, F=0). Gain A/B are the (theta_dot-row)
entries of M(0)^-1 directly:
    M(0)^-1 = (1/det0) * [[ d0, -b0 ], [ -b0, a0 ]]
    a0 = (4/3)*mp*l^2,  b0 = mp*l,  d0 = mc+mp,  det0 = mtot*a0 - b0^2
    Gain A = d0 / det0     (d theta_dot / d p_theta)
    Gain B = b0 / det0     (d theta_dot / d p_x, magnitude)
The full 4x4 Jacobian at that fixed point has eigenvalues {0, 0, +lambda,
-lambda} -- the two zeros are the free cart-position/momentum directions on
a frictionless track (translation invariance); lambda is the real growth
rate of the unstable mode:
    lambda^2 = Gain A * mp*g*l   =>   Instability Time = 1/lambda
(reduced-order derivation: linearizing the (theta, p_theta) subsystem alone,
holding p_x=0, gives theta_ddot = Gain_A * mp*g*l * theta directly).

These closed forms were cross-checked two independent ways before being
hardcoded as this module's contract: (1) against the coworker's live demo's
own displayed numbers at its default sliders (mp=0.10, mc=1.00, l=0.50,
g=9.8: Gain A=32.2, Gain B=1.46, Instability Time=0.252s) and (2) against
the eigenvalues of a numerically-differentiated full 4x4 Jacobian of
`cartpole_dynamics.drift` -- see `test_cartpole_diagnostics.py`'s
`test_gains_and_instability_time_match_full_jacobian_eigenvalues`.
"""
import numpy as np


def _upright_mass_matrix_entries(mp, mc, l):
    """(a0, b0, d0, det0), the raw M(0) entries and its determinant --
    shared by gain_a/gain_b/instability_time so the three can never
    silently drift apart from each other's version of this algebra."""
    mtot = mc + mp
    a0 = (4.0 / 3.0) * mp * l ** 2
    b0 = mp * l
    d0 = mtot
    det0 = mtot * a0 - b0 ** 2
    return a0, b0, d0, det0


def gain_a(mp=0.1, mc=1.0, l=0.5, g=9.8):
    """d(theta_dot)/d(p_theta) at the upright fixed point, in rad/s per
    unit p_theta. Independent of g (kept as a parameter only so every
    diagnostic function shares one signature).

    @returns float
    """
    _, _, d0, det0 = _upright_mass_matrix_entries(mp, mc, l)
    return d0 / det0


def gain_b(mp=0.1, mc=1.0, l=0.5, g=9.8):
    """d(theta_dot)/d(p_x) at the upright fixed point (magnitude), in rad/s
    per unit p_x. Independent of g.

    @returns float
    """
    _, b0, _, det0 = _upright_mass_matrix_entries(mp, mc, l)
    return b0 / det0


def instability_time(mp=0.1, mc=1.0, l=0.5, g=9.8):
    """e-folding time of a small perturbation from upright: how long it
    takes a tiny tip-over to grow by a factor of e, with no force and no
    wind. Smaller = falls faster = harder to control in real time.

    @returns float, seconds
    """
    lam = np.sqrt(gain_a(mp, mc, l, g) * mp * g * l)
    return 1.0 / lam

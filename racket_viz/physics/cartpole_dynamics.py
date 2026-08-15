"""Hamiltonian-formulation cart-pole core: state (x, theta, p_x, p_theta),
NOT gymnasium's classic (x, x_dot, theta, theta_dot).

Ported from `summer-2026/cartpole.py`'s `_velocities`/`_momenta`/`_drift`
methods (the "windy cart-pole for ct-rl" env) -- that file is itself already
a from-scratch Hamiltonian rewrite of Sutton's classic cart-pole, and this
module extracts those three methods into plain, testable functions (mass and
geometry passed as arguments, not read off `self`) so they can be a real
pytest-verified oracle independent of any gymnasium/RL machinery.

Convention: theta=0 is the pole balanced UPRIGHT (the classic cart-pole's
control target), theta=+-pi is hanging straight down -- the OPPOSITE of the
tennis racket/windy pendulum's convention (theta measured from hanging
down). Confirmed directly from the source: `theta_threshold_radians` fails
the episode at +-12 deg off upright, matching Barto/Sutton/Anderson's
original problem statement, not this site's other two systems.

Mass matrix (x_dot, theta_dot) = M(theta)^-1 @ (p_x, p_theta):
    M(theta) = [[ mc+mp,          mp*l*cos(theta) ],
                [ mp*l*cos(theta), (4/3)*mp*l^2    ]]
`velocities` applies M^-1, `momenta` applies M; they are exact inverses.
`l` is HALF the pole's length (matches `summer-2026/cartpole.py`'s own
`self.length` convention, itself inherited from Sutton's original code).

No friction term appears anywhere in `drift` (unlike the windy pendulum's
explicit `friction_torque`) -- this is a genuinely conservative system when
F=0 and there is no wind: H(t) is then exactly conserved, the same
free-scenario invariant already used to validate the racket and pendulum.
A nonzero F does real external work (power = F*x_dot), breaking that
conservation exactly like a controlled racket/pendulum scenario does.
"""
import numpy as np

# Failure thresholds, pinned from `summer-2026/cartpole.py` (lines 171-173):
# the episode terminates once the cart leaves +-X_THRESHOLD or the pole
# tips more than +-THETA_THRESHOLD_RADIANS off upright.
X_THRESHOLD = 2.4
THETA_THRESHOLD_RADIANS = 12 * 2 * np.pi / 360


def mass_matrix(theta, mp=0.1, mc=1.0, l=0.5):
    """(2, 2) mass matrix M(theta) relating (p_x, p_theta) to (x_dot, theta_dot).

    @param theta float, pole angle (0 = upright)
    @param mp float, pole mass (kg)
    @param mc float, cart mass (kg)
    @param l float, HALF the pole's length (m)
    @returns (2, 2) ndarray, symmetric positive-definite for any theta
    """
    c = np.cos(theta)
    return np.array([
        [mc + mp, mp * l * c],
        [mp * l * c, (4.0 / 3.0) * mp * l ** 2],
    ])


def velocities(theta, px, pth, mp=0.1, mc=1.0, l=0.5):
    """Apply M(theta)^-1 to convert momenta -> velocities.

    @returns (x_dot, theta_dot)
    """
    mtot, c = mc + mp, np.cos(theta)
    det = mtot * (4.0 / 3.0) * mp * l ** 2 - (mp * l * c) ** 2
    a, b, d = (4.0 / 3.0) * mp * l ** 2, -mp * l * c, mtot
    return (a * px + b * pth) / det, (b * px + d * pth) / det


def momenta(theta, xdot, thdot, mp=0.1, mc=1.0, l=0.5):
    """Apply M(theta) to convert velocities -> momenta. Exact inverse of
    `velocities` for the same theta/mp/mc/l.

    @returns (p_x, p_theta)
    """
    mtot, c = mc + mp, np.cos(theta)
    return (
        mtot * xdot + mp * l * c * thdot,
        mp * l * c * xdot + (4.0 / 3.0) * mp * l ** 2 * thdot,
    )


def drift(z, F, mp=0.1, mc=1.0, l=0.5, g=9.8):
    """Hamilton's equations RHS: dz/dt for z = (x, theta, p_x, p_theta).

    @param z (4,) array-like, current state
    @param F float, external force on the cart (N); 0 for the free system
    @returns (4,) ndarray, dz/dt
    """
    x, theta, px, pth = z
    s = np.sin(theta)
    xdot, thdot = velocities(theta, px, pth, mp=mp, mc=mc, l=l)
    return np.array([xdot, thdot, F, mp * g * l * s - mp * l * s * xdot * thdot])


def hamiltonian(z, mp=0.1, mc=1.0, l=0.5, g=9.8):
    """H(z) = 0.5 * p^T M(theta)^-1 p + mp*g*l*cos(theta).

    Exactly conserved when F=0 and there is no wind; changes under an
    applied force at rate dH/dt = F * x_dot (the power the force delivers).

    @returns float
    """
    x, theta, px, pth = z
    xdot, thdot = velocities(theta, px, pth, mp=mp, mc=mc, l=l)
    kinetic = 0.5 * (px * xdot + pth * thdot)
    return kinetic + mp * g * l * np.cos(theta)

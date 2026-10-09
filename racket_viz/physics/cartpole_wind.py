"""Per-step stochastic Heun predictor-corrector for the windy cart-pole,
ported directly from `summer-2026/cartpole.py`'s `diffusion`/`integrator`
methods.

Genuinely different from this project's existing racket/pendulum wind
(`wind.js`: ONE disturbance torque sampled once and held constant for the
whole run) -- this is a real continuous SDE, re-sampled every step, with
diffusion s = sigma_gust + sigma_turb*|theta_dot| that GROWS with the
pole's own angular velocity (so a fast swing is noisier than a slow one).
The noise enters as an impulse on p_theta only (never directly on p_x) via
the fixed vector G = [0, 0, 0, 1]. No Python module in this repo integrates
a continuous SDE yet, so there's no existing in-repo pattern to mirror
beyond the reference env itself.

Per this project's established convention (see `wind.js`'s own note in
ARCHITECTURE.md), the RANDOM NUMBER STREAM itself is not, and does not need
to be, cross-validated bit-for-bit against the eventual JS port -- numpy
and JS use different PRNG algorithms, so exact cross-language trajectory
matching isn't a meaningful target. What IS cross-validated: the
deterministic Heun-step mechanics (exercised in tests via an injected fake
generator that returns a fixed value, so the expected output can be
computed by hand) and the noise's STATISTICAL calibration (does the
variance scale the way a diffusion term should).
"""
import numpy as np


def diffusion(thdot, sigma_gust, sigma_turb, smooth_eps=1e-3):
    """s(theta_dot) = sigma_gust + sigma_turb * sqrt(theta_dot^2 + smooth_eps^2).

    The sqrt(...+eps^2) form (not a bare |theta_dot|) keeps this smooth
    (differentiable) at theta_dot=0, unlike the sharp kink |theta_dot|
    would have there.

    @returns float, the 1-sigma diffusion scale for this step
    """
    return sigma_gust + sigma_turb * np.sqrt(thdot ** 2 + smooth_eps ** 2)


def heun_step(z, F, dt, rng, mp=0.1, mc=1.0, l=0.5, g=9.8,
              sigma_gust=0.002, sigma_turb=0.001, smooth_eps=1e-3):
    """Advance one step of the SDE dz = drift(z,F)*dt + diffusion(z)*dW*G
    via a stochastic Heun (predictor-corrector) scheme:

        dW = sqrt(dt) * rng.standard_normal()
        s0, f0 = diffusion(z),   drift(z, F)          # at the start point
        z_predictor = z + f0*dt + s0*dW*G
        s1, f1 = diffusion(z_predictor), drift(z_predictor, F)
        return z + 0.5*(f0+f1)*dt + 0.5*(s0+s1)*dW*G

    @param z (4,) array-like, current state (x, theta, p_x, p_theta)
    @param F float, applied force (N)
    @param dt float, step size (s)
    @param rng object with a `.standard_normal()` method (e.g. a
        `numpy.random.Generator`, or a test stub) -- drawn from exactly
        once per call, so a run is reproducible from the generator's own
        seed alone
    @returns (4,) ndarray, the next state
    """
    # Deferred import: cartpole_wind.py is the SDE wrapper around
    # cartpole_dynamics.py's deterministic drift, not a re-derivation of it.
    from cartpole_dynamics import drift, velocities

    z = np.asarray(z, dtype=float)
    G = np.array([0.0, 0.0, 0.0, 1.0])
    dW = np.sqrt(dt) * rng.standard_normal()

    _, thdot0 = velocities(z[1], z[2], z[3], mp=mp, mc=mc, l=l)
    s0 = diffusion(thdot0, sigma_gust, sigma_turb, smooth_eps)
    f0 = drift(z, F, mp=mp, mc=mc, l=l, g=g)
    z_predictor = z + f0 * dt + s0 * dW * G

    _, thdot1 = velocities(z_predictor[1], z_predictor[2], z_predictor[3], mp=mp, mc=mc, l=l)
    s1 = diffusion(thdot1, sigma_gust, sigma_turb, smooth_eps)
    f1 = drift(z_predictor, F, mp=mp, mc=mc, l=l, g=g)

    return z + 0.5 * (f0 + f1) * dt + 0.5 * (s0 + s1) * dW * G

/**
 * Per-step stochastic Heun predictor-corrector for the windy cart-pole,
 * ported from physics/cartpole_wind.py (itself ported from
 * summer-2026/cartpole.py's `diffusion`/`integrator` methods).
 *
 * Genuinely different from this site's existing racket/pendulum wind
 * (wind.js: ONE disturbance torque sampled once and held constant for the
 * whole run) -- this is a real continuous SDE, re-sampled every step, with
 * diffusion that GROWS with the pole's own angular velocity. The noise
 * enters as an impulse on p_theta only, via the fixed vector G=[0,0,0,1].
 *
 * Per this project's established convention (see wind.js's own note in
 * ARCHITECTURE.md), the random number STREAM itself is not cross-validated
 * bit-for-bit against the Python oracle -- only the deterministic Heun-step
 * mechanics (exercised here with an injectable `rng`, defaulting to
 * Math.random) and the noise's statistical calibration are.
 */
import { drift, velocities } from "./cartpoleDynamics.js";

/** Box-Muller standard-normal draw from a uniform [0,1) source. */
function standardNormal(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * s(theta_dot) = sigma_gust + sigma_turb * sqrt(theta_dot^2 + smooth_eps^2).
 * The sqrt(...+eps^2) form keeps this smooth (differentiable, zero slope)
 * at theta_dot=0, unlike a bare |theta_dot|'s kink there.
 * @returns {number} the 1-sigma diffusion scale for this step
 */
export function diffusion(thdot, sigmaGust, sigmaTurb, smoothEps = 1e-3) {
  return sigmaGust + sigmaTurb * Math.sqrt(thdot ** 2 + smoothEps ** 2);
}

/**
 * Advance one step of the SDE dz = drift(z,F)*dt + diffusion(z)*dW*G via a
 * stochastic Heun (predictor-corrector) scheme.
 * @param {number[]} z current state (x, theta, p_x, p_theta)
 * @param {number} F applied force (N)
 * @param {number} dt step size (s)
 * @param {() => number} [rng] uniform [0,1) source, injectable for
 *   deterministic tests; defaults to Math.random. Drawn from exactly once
 *   per call (via `standardNormal`), so a run is reproducible from the
 *   rng's own seed alone.
 * @param {{mp?, mc?, l?, g?, sigmaGust?, sigmaTurb?, smoothEps?}} [opts]
 * @returns {number[]} the next state
 */
export function heunStep(z, F, dt, rng = Math.random, {
  mp = 0.1, mc = 1.0, l = 0.5, g = 9.8,
  sigmaGust = 0.002, sigmaTurb = 0.001, smoothEps = 1e-3,
} = {}) {
  const G = [0, 0, 0, 1];
  const dW = Math.sqrt(dt) * standardNormal(rng);
  const opts = { mp, mc, l, g };

  const [, thdot0] = velocities(z[1], z[2], z[3], opts);
  const s0 = diffusion(thdot0, sigmaGust, sigmaTurb, smoothEps);
  const f0 = drift(z, F, opts);
  const zPredictor = z.map((v, i) => v + f0[i] * dt + s0 * dW * G[i]);

  const [, thdot1] = velocities(zPredictor[1], zPredictor[2], zPredictor[3], opts);
  const s1 = diffusion(thdot1, sigmaGust, sigmaTurb, smoothEps);
  const f1 = drift(zPredictor, F, opts);

  return z.map((v, i) => v + 0.5 * (f0[i] + f1[i]) * dt + 0.5 * (s0 + s1) * dW * G[i]);
}

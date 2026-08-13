/**
 * Closed-form fragility metrics for the cart-pole's upright equilibrium:
 * Gain A (wind -> theta_dot), Gain B (force -> theta_dot), and Instability
 * Time. Ported from physics/cartpole_diagnostics.py -- see that module's
 * docstring for the derivation and the two independent cross-checks
 * (coworker's live demo numbers + a from-scratch 4x4 Jacobian eigenvalue
 * check) that validated the closed forms before they were hardcoded here.
 * All three are ANALYTIC properties of the plant (mp, mc, l, g) -- none of
 * this depends on any trained controller.
 */

/** @returns {[number, number, number, number]} [a0, b0, d0, det0] */
function uprightMassMatrixEntries(mp, mc, l) {
  const mtot = mc + mp;
  const a0 = (4.0 / 3.0) * mp * l ** 2;
  const b0 = mp * l;
  const d0 = mtot;
  const det0 = mtot * a0 - b0 ** 2;
  return [a0, b0, d0, det0];
}

/**
 * d(theta_dot)/d(p_theta) at the upright fixed point, rad/s per unit
 * p_theta. Independent of g (kept as a param only so every diagnostic
 * function shares one signature).
 * @returns {number}
 */
export function gainA({ mp = 0.1, mc = 1.0, l = 0.5, g = 9.8 } = {}) {
  const [, , d0, det0] = uprightMassMatrixEntries(mp, mc, l);
  return d0 / det0;
}

/**
 * d(theta_dot)/d(p_x) at the upright fixed point (magnitude), rad/s per
 * unit p_x. Independent of g.
 * @returns {number}
 */
export function gainB({ mp = 0.1, mc = 1.0, l = 0.5, g = 9.8 } = {}) {
  const [, b0, , det0] = uprightMassMatrixEntries(mp, mc, l);
  return b0 / det0;
}

/**
 * e-folding time of a small perturbation from upright: how long it takes a
 * tiny tip-over to grow by a factor of e, with no force and no wind.
 * @returns {number} seconds
 */
export function instabilityTime({ mp = 0.1, mc = 1.0, l = 0.5, g = 9.8 } = {}) {
  const lambda = Math.sqrt(gainA({ mp, mc, l, g }) * mp * g * l);
  return 1.0 / lambda;
}

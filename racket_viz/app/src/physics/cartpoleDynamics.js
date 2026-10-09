/**
 * Hamiltonian-formulation cart-pole core: state (x, theta, p_x, p_theta),
 * NOT gymnasium's classic (x, x_dot, theta, theta_dot). Ported from
 * physics/cartpole_dynamics.py -- see that module's docstring for the
 * theta=0-is-upright convention (opposite of this site's racket/pendulum)
 * and why H is conserved exactly when F=0 and there's no wind.
 */

/** Failure thresholds, pinned from summer-2026/cartpole.py. */
export const X_THRESHOLD = 2.4;
export const THETA_THRESHOLD_RADIANS = (12 * 2 * Math.PI) / 360;

/**
 * (2, 2) mass matrix M(theta) relating (p_x, p_theta) to (x_dot, theta_dot).
 * @param {number} theta pole angle (0 = upright)
 * @param {{mp?: number, mc?: number, l?: number}} [opts] mp=pole mass (kg),
 *   mc=cart mass (kg), l=HALF the pole's length (m)
 * @returns {number[][]} 2x2, symmetric positive-definite for any theta
 */
export function massMatrix(theta, { mp = 0.1, mc = 1.0, l = 0.5 } = {}) {
  const c = Math.cos(theta);
  return [
    [mc + mp, mp * l * c],
    [mp * l * c, (4.0 / 3.0) * mp * l ** 2],
  ];
}

/**
 * Apply M(theta)^-1 to convert momenta -> velocities.
 * @returns {[number, number]} [x_dot, theta_dot]
 */
export function velocities(theta, px, pth, { mp = 0.1, mc = 1.0, l = 0.5 } = {}) {
  const mtot = mc + mp;
  const c = Math.cos(theta);
  const det = mtot * (4.0 / 3.0) * mp * l ** 2 - (mp * l * c) ** 2;
  const a = (4.0 / 3.0) * mp * l ** 2;
  const b = -mp * l * c;
  const d = mtot;
  return [(a * px + b * pth) / det, (b * px + d * pth) / det];
}

/**
 * Apply M(theta) to convert velocities -> momenta. Exact inverse of
 * `velocities` for the same theta/mp/mc/l.
 * @returns {[number, number]} [p_x, p_theta]
 */
export function momenta(theta, xdot, thdot, { mp = 0.1, mc = 1.0, l = 0.5 } = {}) {
  const mtot = mc + mp;
  const c = Math.cos(theta);
  return [
    mtot * xdot + mp * l * c * thdot,
    mp * l * c * xdot + (4.0 / 3.0) * mp * l ** 2 * thdot,
  ];
}

/**
 * Hamilton's equations RHS: dz/dt for z = (x, theta, p_x, p_theta).
 * @param {number[]} z current state
 * @param {number} F external force on the cart (N); 0 for the free system
 * @returns {number[]} dz/dt
 */
export function drift(z, F, { mp = 0.1, mc = 1.0, l = 0.5, g = 9.8 } = {}) {
  const [, theta, px, pth] = z;
  const s = Math.sin(theta);
  const [xdot, thdot] = velocities(theta, px, pth, { mp, mc, l });
  return [xdot, thdot, F, mp * g * l * s - mp * l * s * xdot * thdot];
}

/**
 * H(z) = 0.5 * p^T M(theta)^-1 p + mp*g*l*cos(theta). Exactly conserved
 * when F=0 and there is no wind; changes under an applied force at rate
 * dH/dt = F * x_dot (the power the force delivers).
 * @returns {number}
 */
export function hamiltonian(z, { mp = 0.1, mc = 1.0, l = 0.5, g = 9.8 } = {}) {
  const [, theta, px, pth] = z;
  const [xdot, thdot] = velocities(theta, px, pth, { mp, mc, l });
  const kinetic = 0.5 * (px * xdot + pth * thdot);
  return kinetic + mp * g * l * Math.cos(theta);
}

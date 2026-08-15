/**
 * Gravity + friction torques for the windy pendulum, as a torqueFn for
 * rigidBody.js's integrateFull (which already accepts an arbitrary
 * torqueFn(t, w, q) -- see its docstring, no changes needed there).
 * Ported from physics/pendulum_dynamics.py.
 */

/** @param {number[][]} R @param {number[]} v @returns {number[]} R @ v */
function matVec(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

/** @param {number[][]} R @param {number[]} v @returns {number[]} R^T @ v */
function matTVec(R, v) {
  return [
    R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
    R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
    R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
  ];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/**
 * Body-frame torque from gravity acting at the center of mass, for a rigid
 * body pivoting about a fixed point (the origin) that is NOT its COM. Zero
 * exactly when the COM is aligned with +/- world z (hanging straight down or
 * balanced straight up) -- both true equilibria of a gravity pendulum.
 * @param {number[][]} R current orientation
 * @param {number[]} comBody center of mass in the body (principal-axis) frame
 * @param {number} totalMass
 * @param {number} [g=9.81]
 * @returns {number[]} body-frame torque
 */
export function gravityTorqueBody(R, comBody, totalMass, g = 9.81) {
  const rWorld = matVec(R, comBody);
  const Fg = [0, 0, -totalMass * g];
  return matTVec(R, cross(rWorld, Fg));
}

/**
 * Simple linear (viscous) damping opposing angular velocity.
 * @param {number[]} omega
 * @param {number} frictionCoeff
 * @returns {number[]}
 */
export function frictionTorque(omega, frictionCoeff) {
  return omega.map((w) => -frictionCoeff * w);
}

/**
 * Total mechanical energy per frame: kinetic (0.5*sum(M_i^2/I_i), same form
 * as the racket's ControlPanel.js computeH) plus gravitational potential
 * (totalMass*g*height, height = the COM's world z-coordinate = the z-row of
 * R dotted with comBody). NOT the racket's computeH -- that has no potential
 * term at all (correctly, since the free racket has none) -- kept as its own
 * function here rather than generalizing ControlPanel.js's, so the racket's
 * already-validated energy panel is never at risk of a pendulum-driven edit.
 * @param {number[][]} quats scalar-first quaternions, one per frame
 * @param {number[][]} MBody angular momentum in the body frame, one per frame
 * @param {number[]} I principal moments of inertia
 * @param {number[]} comBody center of mass in the body frame
 * @param {number} totalMass
 * @param {number} [g=9.81]
 * @returns {number[]} H(t)
 */
export function computeH(quats, MBody, I, comBody, totalMass, g = 9.81) {
  return MBody.map((M, i) => {
    const KE = 0.5 * (M[0] ** 2 / I[0] + M[1] ** 2 / I[1] + M[2] ** 2 / I[2]);
    const q = quats[i];
    const [q0, q1, q2, q3] = q;
    // z-row of the rotation matrix for a scalar-first quaternion, dotted with
    // comBody -- avoids building the full 3x3 matrix just to read one row.
    const rz = [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)];
    const height = rz[0] * comBody[0] + rz[1] * comBody[1] + rz[2] * comBody[2];
    return KE + totalMass * g * height;
  });
}

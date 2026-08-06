/**
 * IDA-PBC attitude controller for the windy pendulum: the racket's
 * GeometricAttitudeController PD law plus an exact feedforward cancellation
 * of the gravity torque at the CURRENT orientation. Ported from
 * physics/pendulum_controller.py -- see that file's docstring for why the
 * minus sign on the gravity term is exact cancellation, not a typo.
 */
import { GeometricAttitudeController } from "./controllers.js";
import { gravityTorqueBody } from "./pendulumDynamics.js";

function clipVec(v, clip) {
  return v.map((x) => Math.max(-clip, Math.min(clip, x)));
}

export class GravityCompensatedAttitudeController {
  /**
   * @param {number[][]} Rstar @param {number[]} omegaStar
   * @param {number[]} comBody @param {number} totalMass @param {number} [g=9.81]
   * @param {number} [KR=0.10] @param {number} [Kp=0.10] @param {number} [clip=2.0]
   */
  constructor(Rstar, omegaStar, comBody, totalMass, g = 9.81, KR = 0.1, Kp = 0.1, clip = 2.0) {
    // Internal PD term deliberately unclipped -- see pendulum_controller.py's
    // docstring: clipping must happen once, on the final combined torque.
    this._base = new GeometricAttitudeController(Rstar, omegaStar, KR, Kp, Infinity);
    this.comBody = [...comBody];
    this.totalMass = totalMass;
    this.g = g;
    this.clip = clip;
  }

  /** @param {number[][]} R @param {number[]} omega @returns {number[]} body-frame torque */
  call(R, omega) {
    const tauPd = this._base.call(R, omega);
    const tauG = gravityTorqueBody(R, this.comBody, this.totalMass, this.g);
    return clipVec(
      tauPd.map((t, i) => t - tauG[i]),
      this.clip
    );
  }
}

/**
 * K_R/K_p scaled to the ACTUAL current inertia, not a fixed constant --
 * same reasoning as controllers.js's defaultKpForAxisControl. See
 * pendulum_controller.py's default_gains_for_attitude_control docstring for
 * the full derivation (standard 2nd-order system relations, critically
 * damped by default) -- ported line-for-line.
 * @param {number[]} I [I1,I2,I3] (ascending; I[1]==I[2] for this system)
 * @param {number} [targetOmegaN=2.0] rad/s, desired closed-loop natural frequency
 * @param {number} [dampingRatio=1.0] 1.0 = critically damped
 * @returns {{KR: number, Kp: number}}
 */
export function defaultGainsForAttitudeControl(I, targetOmegaN = 2.0, dampingRatio = 1.0) {
  const ITransverse = Math.max(I[1], I[2]);
  const KR = targetOmegaN ** 2 * ITransverse;
  const Kp = 2.0 * dampingRatio * targetOmegaN * ITransverse;
  return { KR, Kp };
}

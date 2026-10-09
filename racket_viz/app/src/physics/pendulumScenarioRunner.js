/**
 * Builds one pendulum scenario -- a single continuous trajectory -- from
 * sidebar parameters, entirely in JS. Mirrors scenarioRunner.js's shape
 * (same {meta, frames, geometry} contract for reusing DataLoader/Playback/
 * TimeSeriesPanel/ControlPanel unchanged), but with pendulum-specific
 * torques (gravity + friction, see pendulumDynamics.js) instead of the
 * racket's torque-free/wind-only case, and no `background` key -- there's no
 * Casimir sphere for a gravity pendulum (see DECISIONS.md).
 */
import { buildPendulum } from "./pendulumGeometry.js";
import { integrateFull, quatToRotationMatrix } from "./rigidBody.js";
import { gravityTorqueBody, frictionTorque, computeH } from "./pendulumDynamics.js";
import { GravityCompensatedAttitudeController, defaultGainsForAttitudeControl } from "./pendulumController.js";
import { sampleDisturbanceTorque } from "./wind.js";

export const DEFAULT_G = 9.81;

/**
 * Any quaternion q with quatToRotationMatrix(q) @ aBody == bWorld (up to
 * normalization) -- used to build a "hanging straight down" (or any other)
 * starting orientation from the pendulum's own COM direction, rather than
 * assuming identity means anything in particular (unlike the racket, which
 * never needs this: q0 defaults to identity there because a free body has no
 * preferred starting orientation at all).
 * @param {number[]} aBody unit-ish vector in the body frame
 * @param {number[]} bWorld unit-ish vector in the world frame
 * @returns {number[]} scalar-first quaternion
 */
export function quaternionAligning(aBody, bWorld) {
  const na = Math.hypot(aBody[0], aBody[1], aBody[2]);
  const a = [aBody[0] / na, aBody[1] / na, aBody[2] / na];
  const nb = Math.hypot(bWorld[0], bWorld[1], bWorld[2]);
  const b = [bWorld[0] / nb, bWorld[1] / nb, bWorld[2] / nb];
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  if (dot > 1 - 1e-9) return [1, 0, 0, 0];

  if (dot < -1 + 1e-9) {
    // 180deg: any axis perpendicular to `a` works -- Gram-Schmidt an
    // arbitrary non-parallel vector against a to get one.
    let axis = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const d = axis[0] * a[0] + axis[1] * a[1] + axis[2] * a[2];
    axis = [axis[0] - d * a[0], axis[1] - d * a[1], axis[2] - d * a[2]];
    const n = Math.hypot(axis[0], axis[1], axis[2]);
    return [0, axis[0] / n, axis[1] / n, axis[2] / n];
  }

  const axis = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const naxis = Math.hypot(axis[0], axis[1], axis[2]);
  const axisN = [axis[0] / naxis, axis[1] / naxis, axis[2] / naxis];
  const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
  const half = angle / 2;
  const s = Math.sin(half);
  return [Math.cos(half), axisN[0] * s, axisN[1] * s, axisN[2] * s];
}

/**
 * @param {object} params
 * @param {number} [params.rodLength] @param {number} [params.bobRadius]
 * @param {number} [params.totalMass] @param {number} [params.bobFraction]
 * @param {number} [params.frictionCoeff=0.02]
 * @param {number[]} [params.q0] initial orientation; if omitted, defaults to
 *   hanging `startingAngleDeg` degrees off straight-down (COM tilted away
 *   from world -z) -- an EXACT 0deg start sits precisely at the stable
 *   equilibrium (see pendulum_dynamics.py's zero-torque-at-equilibrium
 *   property), which never moves at all under free swing, same reasoning as
 *   the racket's own `startingPerturbation` existing to avoid a boring exact
 *   on-axis start.
 * @param {number} [params.startingAngleDeg=25]
 * @param {boolean} params.controlOn
 * @param {number[][]} [params.Rstar] target orientation, required if controlOn
 * @param {boolean} params.windOn @param {number} [params.windStd]
 * @param {number} [params.g=9.81]
 * @param {number} [params.T=10.0] @param {number} [params.N=1000]
 * @param {() => number} [params.rng] injectable uniform RNG for deterministic tests
 * @param {string} [params.label] @param {string} [params.caption]
 * @returns {{ meta: object, frames: object, geometry: object }}
 */
export function buildPendulumScenario(params) {
  const {
    rodLength,
    bobRadius,
    totalMass,
    bobFraction,
    frictionCoeff = 0.02,
    q0,
    startingAngleDeg = 25,
    controlOn,
    Rstar,
    windOn,
    windStd,
    g = DEFAULT_G,
    T = 10.0,
    N = 1000,
    rng = Math.random,
    label = controlOn ? "Swing-Up Control" : "Free Swing",
    caption = "",
    dotColor,
  } = params;

  const geometry = buildPendulum({ rodLength, bobRadius, totalMass, bobFraction });
  // Resolved (buildPendulum defaults an undefined totalMass to 1.0
  // internally) -- everything below must use THIS value, not the possibly-
  // undefined `totalMass` param, or gravity torque silently becomes NaN.
  const { I, com, totalMass: resolvedTotalMass } = geometry;

  // Tilting the TARGET direction (rather than composing an extra rotation
  // onto the "hanging down" quaternion) reuses quaternionAligning directly --
  // no separate quaternion-multiply helper needed.
  const angleRad = (startingAngleDeg * Math.PI) / 180;
  const tiltedDownDir = [Math.sin(angleRad), 0, -Math.cos(angleRad)];
  const q0Resolved = q0 ?? quaternionAligning(com, tiltedDownDir);
  const windVec = windOn ? sampleDisturbanceTorque(windStd, rng) : [0, 0, 0];

  // GravityCompensatedAttitudeController's own default clip (2.0, inherited
  // from the racket's direct-torque-actuator scale) is far too small here --
  // the worst-case gravity torque alone is ~totalMass*g*|com|, easily several
  // N*m for a real pendulum, so a clip that small chronically saturates the
  // controller, leaving an uncancelled gravity residual that behaves like an
  // uncontrolled swing instead of actually converging. No clip (unlimited
  // torque) by default -- this is a teaching visualization, not a model of a
  // specific motor's torque limit.
  const { KR, Kp } = defaultGainsForAttitudeControl(I);
  const controller = controlOn
    ? new GravityCompensatedAttitudeController(Rstar, [0, 0, 0], com, resolvedTotalMass, g, KR, Kp, Infinity)
    : null;

  const torqueFn = (t, w, q) => {
    const R = quatToRotationMatrix(q);
    const tauG = gravityTorqueBody(R, com, resolvedTotalMass, g);
    const tauFric = frictionTorque(w, frictionCoeff);
    const tauCtrl = controller ? controller.call(R, w) : [0, 0, 0];
    return [
      tauG[0] + tauFric[0] + tauCtrl[0] + windVec[0],
      tauG[1] + tauFric[1] + tauCtrl[1] + windVec[1],
      tauG[2] + tauFric[2] + tauCtrl[2] + windVec[2],
    ];
  };

  const { omegas, quats } = integrateFull([0, 0, 0], I, T, N, { torqueFn, q0: q0Resolved });
  const t = Array.from({ length: N }, (_, i) => (N > 1 ? (T * i) / (N - 1) : 0));
  const M_body = omegas.map((w) => [w[0] * I[0], w[1] * I[1], w[2] * I[2]]);

  // Same rationale as scenarioRunner.js's own controllerTorque: recorded
  // post-hoc from the controller's actual output at each sample's (R, omega),
  // controller-only (excludes wind/gravity/friction), matching what "the
  // control function" means to the user.
  const controllerTorque = controlOn
    ? omegas.map((w, i) => controller.call(quatToRotationMatrix(quats[i]), w))
    : omegas.map(() => [0, 0, 0]);

  const H = computeH(quats, M_body, I, com, resolvedTotalMass, g);

  const meta = {
    mode: controlOn ? "controlled" : "free",
    I,
    segments: [
      {
        start: 0,
        end: N,
        label,
        caption,
        dot_color: dotColor ?? (controlOn ? "#ffaa00" : "#44ee66"),
      },
    ],
    wind_on: windOn,
  };
  if (controlOn) {
    meta.target_R = Rstar;
    // omega_star is always [0,0,0] here (a resting target attitude), so the
    // desired H is pure potential energy: totalMass*g*height, height being
    // the target orientation's own COM z-coordinate (Rstar's z-row . com).
    const heightAtTarget = Rstar[2][0] * com[0] + Rstar[2][1] * com[1] + Rstar[2][2] * com[2];
    meta.desired_H = resolvedTotalMass * g * heightAtTarget;
    // Steady-state torque needed to HOLD Rstar forever, not just to reach it:
    // once converged (omega=0, R=Rstar) the PD term vanishes exactly, so the
    // controller's entire output is pure gravity cancellation -- every target
    // except "hanging straight down" is not a free equilibrium, so this is
    // never zero.
    const tauGAtTarget = gravityTorqueBody(Rstar, com, resolvedTotalMass, g);
    meta.holding_torque = Math.hypot(tauGAtTarget[0], tauGAtTarget[1], tauGAtTarget[2]);
  }

  return {
    meta,
    frames: { t, quaternion: quats, M_body, controller_torque: controllerTorque, H },
    geometry: {
      rod_length: geometry.rodLength,
      bob_radius: geometry.bobRadius,
      com: geometry.com,
    },
  };
}

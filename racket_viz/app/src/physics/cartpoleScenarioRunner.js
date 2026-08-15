/**
 * Drives the cart-pole in REAL TIME, one physics tick per call -- genuinely
 * different from scenarioRunner.js/pendulumScenarioRunner.js's "precompute
 * one continuous trajectory, then scrub it with a Playback cursor" pattern.
 * That pattern doesn't fit here: "you" (manual/keyboard) mode needs a force
 * that depends on what the user does DURING the run, which isn't knowable
 * ahead of time -- there is no fixed trajectory to precompute. "none" mode
 * COULD still be precomputed (F=0 is known in advance), but stepping it the
 * same way as "you" keeps one code path for both instead of two, and this
 * system's wind is already a per-step SDE (see cartpole_wind.py), not the
 * racket/pendulum's "sample once, hold constant" model -- there's no long-
 * lived "doc" object for it to live on in the first place.
 *
 * cartpoleMain.js's own animate() loop calls `advance` once per rendered
 * frame; this module owns none of the UI/rendering, only the physics tick
 * and its own sub-stepping.
 */
import { drift, velocities, X_THRESHOLD, THETA_THRESHOLD_RADIANS } from "./cartpoleDynamics.js";
import { heunStep } from "./cartpoleWind.js";
import { forward, initWindow, pushFrame, windowToObservation } from "./cartpolePolicy.js";

// The real trained checkpoint's own training rate (see cartpole_policy.js's
// docstring / POLICY_SPEC.md) -- NOT a tunable, since the 16-frame
// observation window's meaning (0.16s of history) and the policy's own
// learned dynamics are both tied to exactly this step size. Unlike
// `advance`'s MAX_STEP (an accuracy cap, sub-stepped freely), this is a
// DECISION cadence: one policy forward-pass + one full physics step per
// tick, matching how the training env itself stepped (a single
// Euler-Maruyama/Heun step per env.step(), not further subdivided).
export const CT_SAC_DT = 0.01;

// Same reasoning as rigidBody.js's MAX_STEP: cap the actual RK/Heun step
// size rather than using a fixed substep COUNT, so a real-time frame's
// variable dt (a slow frame, a tab regaining focus after being backgrounded)
// doesn't silently under-resolve the integration.
const MAX_STEP = 0.005;

/**
 * @param {{x0?: number, thetaDeg?: number}} [opts]
 * @returns {number[]} initial state [x, theta, p_x, p_theta] -- momenta are
 *   always 0 since the sidebar only ever specifies a starting POSE (angle,
 *   cart position), never a starting velocity.
 */
export function initialState({ x0 = 0, thetaDeg = 5 } = {}) {
  return [x0, (thetaDeg * Math.PI) / 180, 0, 0];
}

/**
 * True once the episode has failed: the cart left the track (`xThreshold`)
 * or the pole tipped past `thetaThreshold`.
 *
 * Both thresholds default to the canonical constants (the classic +-12deg
 * task boundary and the classic +-2.4m rail) but are real parameters, not
 * just documentation -- cartpoleMain.js passes wider/narrower values so
 * (a) a failure past ct_sac's trained envelope is actually watchable
 * instead of insta-terminating (see DECISIONS.md), and (b) Rail Length is
 * a live sidebar slider, not a fixed constant. The DEFAULTS stay the
 * canonical values so every existing call site/test, none of which pass
 * these args, is unaffected.
 * @param {number[]} z
 * @param {number} [thetaThreshold]
 * @param {number} [xThreshold]
 * @returns {boolean}
 */
export function isTerminated(z, thetaThreshold = THETA_THRESHOLD_RADIANS, xThreshold = X_THRESHOLD) {
  const [x, theta] = z;
  return Math.abs(x) > xThreshold || Math.abs(theta) > thetaThreshold;
}

/**
 * Which boundary a terminated state actually failed, for user-facing
 * display ("Terminated — survived Xs" plus a reason). Checked in the same
 * order/priority as `isTerminated`'s own `||` -- if a single tick somehow
 * crosses both at once, "rail" wins, since running off the physical track
 * is the more fundamental fact (the pole's own angle stops being
 * meaningful once the cart itself has left the modeled world). Returns
 * `null` if `z` isn't actually terminated at all, so callers can use this
 * directly without a separate `isTerminated` check first.
 * @param {number[]} z
 * @param {number} [thetaThreshold]
 * @param {number} [xThreshold]
 * @returns {"rail" | "angle" | null}
 */
export function terminationReason(z, thetaThreshold = THETA_THRESHOLD_RADIANS, xThreshold = X_THRESHOLD) {
  const [x, theta] = z;
  if (Math.abs(x) > xThreshold) return "rail";
  if (Math.abs(theta) > thetaThreshold) return "angle";
  return null;
}

/**
 * Advances one real-time tick, sub-stepping internally so a large `dtReal`
 * (e.g. the first frame after a tab regains focus) can't blow through
 * MAX_STEP and under-resolve the integration -- same fix already applied to
 * rigidBody.js's integrateFull/integrateM for the exact same reason.
 * @param {number[]} z current state
 * @param {number} F applied force (N) -- constant for the whole tick
 * @param {number} dtReal elapsed real time (s) to advance by
 * @param {() => number} rng uniform [0,1) source, forwarded to heunStep
 * @param {object} params {mp, mc, l, g, sigmaGust, sigmaTurb, windOn}
 * @returns {number[]} next state
 */
export function advance(z, F, dtReal, rng, params) {
  const { windOn = true, sigmaGust = 0, sigmaTurb = 0, ...rest } = params;
  const substeps = Math.max(1, Math.ceil(dtReal / MAX_STEP));
  const h = dtReal / substeps;
  let state = z;
  for (let i = 0; i < substeps; i++) {
    state = heunStep(state, F, h, rng, {
      ...rest,
      sigmaGust: windOn ? sigmaGust : 0,
      sigmaTurb: windOn ? sigmaTurb : 0,
    });
  }
  return state;
}

/**
 * Initial state for ct_sac mode: the physics state PLUS the observation
 * window (reset-filled with the initial pose, per cartpolePolicy.js's
 * initWindow) and a real-time accumulator (see advanceCtSac) -- bundled
 * together because they're only ever meaningful in lockstep with each
 * other, never independently.
 * @param {{x0?: number, thetaDeg?: number}} [opts]
 * @returns {{z: number[], window: number[][], accumulator: number, force: number}}
 */
export function initCtSacState({ x0 = 0, thetaDeg = 5 } = {}) {
  const z = initialState({ x0, thetaDeg });
  return { z, window: initWindow(z[1], z[0]), accumulator: 0, force: 0 };
}

/**
 * Advances ct_sac state by real elapsed time `dtReal`, running as many
 * whole CT_SAC_DT ticks as have accumulated -- a standard fixed-timestep
 * accumulator (real-time rendering at ~16ms/frame is coarser than the
 * policy's own 10ms decision rate, so most rendered frames need either 1 or
 * 2 ticks; a slow frame needs more, none is skipped and none is double-
 * counted, unlike naively stepping once per rendered frame at whatever its
 * own dt happens to be). Each tick: sample the CURRENT window -> one policy
 * forward pass -> one full CT_SAC_DT physics step with that force held
 * constant -> push the new measurement into the window. Stops early
 * (without consuming the rest of the accumulated time) if a tick causes
 * termination, so a caller that keeps calling this after failure -- which
 * it shouldn't, see cartpoleMain.js's `!terminated` gate on every mode --
 * still can't run the policy past that point.
 * @param {{z: number[], window: number[][], accumulator: number, force: number}} state
 * @param {number} dtReal elapsed real time (s) to advance by
 * @param {() => number} rng uniform [0,1) source, forwarded to heunStep
 * @param {object} params {mp, mc, l, g, sigmaGust, sigmaTurb, windOn}
 * @param {object} policy parsed cartpole_policy.json
 * @param {number} [thetaThreshold] forwarded to the internal early-stop
 *   check below -- see isTerminated's own docstring. Defaults to the
 *   canonical +-12deg, same as isTerminated itself, so existing callers are
 *   unaffected; cartpoleMain.js passes its own wider value.
 * @param {number} [xThreshold] same idea, for the rail -- defaults to the
 *   canonical X_THRESHOLD; cartpoleMain.js passes the live Rail Length
 *   slider's value.
 * @returns {{z: number[], window: number[][], accumulator: number, force: number, workDelta: number}}
 *   workDelta is the work done DURING THIS CALL ONLY (0 if no tick ran) --
 *   the caller accumulates it into its own running total, same convention
 *   as `advance`'s callers do for none/manual mode.
 */
export function advanceCtSac(state, dtReal, rng, params, policy, thetaThreshold = THETA_THRESHOLD_RADIANS, xThreshold = X_THRESHOLD) {
  const { windOn = true, sigmaGust = 0, sigmaTurb = 0, ...rest } = params;
  let { z, window, accumulator, force } = state;
  accumulator += dtReal;
  let workDelta = 0;

  while (accumulator >= CT_SAC_DT) {
    const obs = windowToObservation(window);
    force = forward(obs, policy);
    // Power = force * cart velocity (dH/dt = F*x_dot), evaluated at the
    // state BEFORE this tick's step -- same identity `advance`'s callers
    // use, just measured per-tick here instead of per-rendered-frame,
    // which is actually the more natural fit: the force is exactly
    // constant for the whole tick by construction.
    const [xDotBefore] = velocities(z[1], z[2], z[3], rest);
    z = heunStep(z, force, CT_SAC_DT, rng, {
      ...rest,
      sigmaGust: windOn ? sigmaGust : 0,
      sigmaTurb: windOn ? sigmaTurb : 0,
    });
    workDelta += force * xDotBefore * CT_SAC_DT;
    window = pushFrame(window, z[1], z[0]);
    accumulator -= CT_SAC_DT;
    if (isTerminated(z, thetaThreshold, xThreshold)) break;
  }

  return { z, window, accumulator, force, workDelta };
}

/** Re-exported for callers that want the raw drift (e.g. computing H or a
 * one-off power/force*velocity readout) without a second import path. */
export { drift };

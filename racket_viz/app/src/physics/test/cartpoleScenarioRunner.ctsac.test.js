import { describe, expect, it } from "vitest";

import { initCtSacState, advanceCtSac, isTerminated, CT_SAC_DT } from "../cartpoleScenarioRunner.js";
import { heunStep } from "../cartpoleWind.js";
import { initWindow, pushFrame } from "../cartpolePolicy.js";

const DEFAULT = { mp: 0.1, mc: 1.0, l: 0.5, g: 9.8, windOn: false, sigmaGust: 0, sigmaTurb: 0 };

/** A fake policy that outputs a KNOWN CONSTANT force regardless of the
 * observation -- all-zero weights (tanh(0)=0) and low===high, so
 * `low + (high-low)*(tanh(0)+1)/2` collapses to exactly `low` no matter
 * what the 48-vector input is. Lets these tests hand-verify the tick
 * mechanics (cadence, window updates, physics chaining) against a known
 * value instead of the real 229,500-weight network. */
function constantForcePolicy(force) {
  return {
    layers: [{ name: "only", in: 48, out: 1, activation: "linear", W: [Array(48).fill(0)], b: [0] }],
    output: { rescale: { low: force, high: force } },
  };
}

describe("initCtSacState", () => {
  it("builds the physics state and a reset-filled window together", () => {
    const state = initCtSacState({ x0: 0.3, thetaDeg: 10 });
    const thetaRad = (10 * Math.PI) / 180;
    expect(state.z[0]).toBeCloseTo(0.3, 9);
    expect(state.z[1]).toBeCloseTo(thetaRad, 9);
    expect(state.window).toHaveLength(16);
    for (const frame of state.window) {
      expect(frame[0]).toBeCloseTo(Math.cos(thetaRad), 9);
      expect(frame[1]).toBeCloseTo(Math.sin(thetaRad), 9);
      expect(frame[2]).toBeCloseTo(0.3, 9);
    }
    expect(state.accumulator).toBe(0);
    expect(state.force).toBe(0);
  });
});

describe("advanceCtSac", () => {
  it("does not tick at all if dtReal is below CT_SAC_DT -- just accumulates", () => {
    const state = initCtSacState({ x0: 0, thetaDeg: 5 });
    const rng = () => 0.5;
    const next = advanceCtSac(state, CT_SAC_DT * 0.3, rng, DEFAULT, constantForcePolicy(3));
    expect(next.accumulator).toBeCloseTo(CT_SAC_DT * 0.3, 12);
    expect(next.z).toEqual(state.z);
    expect(next.window).toEqual(state.window);
    expect(next.force).toBe(0); // no tick happened, so no new force was ever computed
  });

  it("runs exactly one tick for dtReal===CT_SAC_DT, matching a direct heunStep call", () => {
    const state = initCtSacState({ x0: 0, thetaDeg: 5 });
    const rng = () => 0.5;
    const force = 3;
    const next = advanceCtSac(state, CT_SAC_DT, rng, DEFAULT, constantForcePolicy(force));

    const expectedZ = heunStep(state.z, force, CT_SAC_DT, () => 0.5, DEFAULT);
    for (let i = 0; i < 4; i++) expect(next.z[i]).toBeCloseTo(expectedZ[i], 9);
    expect(next.force).toBe(force);
    expect(next.accumulator).toBeCloseTo(0, 12);

    const expectedWindow = pushFrame(state.window, expectedZ[1], expectedZ[0]);
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 3; j++) expect(next.window[i][j]).toBeCloseTo(expectedWindow[i][j], 9);
    }
  });

  it("runs multiple ticks for a larger dtReal, leaving the correct remainder", () => {
    const state = initCtSacState({ x0: 0, thetaDeg: 5 });
    const rng = () => 0.5;
    const force = 2;
    const dtReal = CT_SAC_DT * 2.5;
    const next = advanceCtSac(state, dtReal, rng, DEFAULT, constantForcePolicy(force));

    let expectedZ = state.z;
    for (let i = 0; i < 2; i++) expectedZ = heunStep(expectedZ, force, CT_SAC_DT, () => 0.5, DEFAULT);
    for (let i = 0; i < 4; i++) expect(next.z[i]).toBeCloseTo(expectedZ[i], 9);
    expect(next.accumulator).toBeCloseTo(CT_SAC_DT * 0.5, 9);
  });

  it("accumulates leftover time correctly across repeated small calls (no drift)", () => {
    let state = initCtSacState({ x0: 0, thetaDeg: 5 });
    const rng = () => 0.5;
    const force = 1;
    // 7 calls of 0.3*CT_SAC_DT each = 2.1*CT_SAC_DT total -- should produce
    // exactly 2 ticks overall, regardless of how the real time was chopped
    // into individual animate() frames.
    for (let i = 0; i < 7; i++) {
      state = advanceCtSac(state, CT_SAC_DT * 0.3, rng, DEFAULT, constantForcePolicy(force));
    }
    let expectedZ = initCtSacState({ x0: 0, thetaDeg: 5 }).z;
    for (let i = 0; i < 2; i++) expectedZ = heunStep(expectedZ, force, CT_SAC_DT, () => 0.5, DEFAULT);
    for (let i = 0; i < 4; i++) expect(state.z[i]).toBeCloseTo(expectedZ[i], 6);
  });

  it("stops early on termination, without consuming the rest of the accumulated ticks", () => {
    // Start right at the failure boundary and push hard with a huge
    // constant force. Confirmed numerically first (not assumed): with this
    // force/starting angle, the push swings theta toward the OPPOSITE
    // boundary (through 0), crossing it on tick 4 of what would otherwise
    // be 5 -- so the property to check is "stopped before consuming all of
    // dtReal", not a specific hardcoded tick count.
    const nearThreshold = { z: [0, 0.2093, 0, 0], window: initWindow(0.2093, 0), accumulator: 0, force: 0 };
    const rng = () => 0.5;
    const hugeForce = 500;
    const dtReal = CT_SAC_DT * 5; // would be 5 ticks if nothing stopped it
    const next = advanceCtSac(nearThreshold, dtReal, rng, DEFAULT, constantForcePolicy(hugeForce));

    expect(isTerminated(next.z)).toBe(true);
    // If all 5 ticks had run, accumulator would be ~0 -- stopping early
    // (confirmed: exactly 4 of 5 ticks ran) leaves a whole tick's worth of
    // real time un-consumed instead.
    expect(next.accumulator).toBeCloseTo(CT_SAC_DT, 9);
    expect(next.accumulator).toBeGreaterThan(0);
  });

  it("a wider thetaThreshold override lets it keep ticking right where the default would have stopped", () => {
    // Same near-threshold start and huge force as the test above (confirmed
    // numerically first, not assumed): with the DEFAULT threshold this stops
    // early after only 4 of 10.5 possible ticks (accumulator left at 0.065,
    // i.e. 6.5 ticks' worth un-consumed). Passing a wide threshold --
    // cartpoleMain.js's own VISUAL_THETA_THRESHOLD_RADIANS use case -- lets
    // the same swing run to completion instead: all 10 whole ticks consumed,
    // leaving only the expected half-tick remainder, and no early stop.
    const nearThreshold = { z: [0, 0.2093, 0, 0], window: initWindow(0.2093, 0), accumulator: 0, force: 0 };
    const rng = () => 0.5;
    const hugeForce = 500;
    const dtReal = CT_SAC_DT * 10.5;
    const wideThreshold = 10; // far past anything this swing reaches
    const next = advanceCtSac(nearThreshold, dtReal, rng, DEFAULT, constantForcePolicy(hugeForce), wideThreshold);

    expect(isTerminated(next.z, wideThreshold)).toBe(false);
    expect(next.accumulator).toBeCloseTo(CT_SAC_DT * 0.5, 9); // all 10 whole ticks consumed, none stopped early
  });

  it("also accepts an xThreshold override -- the Rail Length slider's use case -- with the same early-stop behavior", () => {
    // A gentle constant rightward push from rest (confirmed numerically
    // first: x crosses 0.015 on tick 8 of what would otherwise be 10.5).
    // A narrow xThreshold stops it there; a wide one lets the same push
    // run to completion, mirroring the thetaThreshold test above exactly.
    const state = { z: [0, 0, 0, 0], window: initWindow(0, 0), accumulator: 0, force: 0 };
    const rng = () => 0.5;
    const force = 5;
    const dtReal = CT_SAC_DT * 10.5;
    const wideTheta = 100; // theta stays ~0 here regardless; keep it out of play

    const narrow = advanceCtSac(state, dtReal, rng, DEFAULT, constantForcePolicy(force), wideTheta, 0.015);
    expect(isTerminated(narrow.z, wideTheta, 0.015)).toBe(true);
    expect(narrow.accumulator).toBeCloseTo(CT_SAC_DT * 2.5, 9); // stopped after 8 of 10.5 ticks

    const wide = advanceCtSac(state, dtReal, rng, DEFAULT, constantForcePolicy(force), wideTheta, 100);
    expect(isTerminated(wide.z, wideTheta, 100)).toBe(false);
    expect(wide.accumulator).toBeCloseTo(CT_SAC_DT * 0.5, 9); // all 10 whole ticks consumed
  });

  it("workDelta is 0 when no tick ran, and matches force*x_dot*CT_SAC_DT for a single tick", () => {
    const state = initCtSacState({ x0: 0, thetaDeg: 5 });
    const rng = () => 0.5;
    const force = 4;

    const noTick = advanceCtSac(state, CT_SAC_DT * 0.3, rng, DEFAULT, constantForcePolicy(force));
    expect(noTick.workDelta).toBe(0);

    // x_dot is exactly 0 at this state (px=0), so a single tick's work
    // should be exactly 0 too -- a clean, hand-checkable case rather than
    // needing to reproduce the full velocities() formula in the test.
    const oneTick = advanceCtSac(state, CT_SAC_DT, rng, DEFAULT, constantForcePolicy(force));
    expect(oneTick.workDelta).toBeCloseTo(0, 9);
  });

  it("workDelta accumulates only over ticks that actually ran within the call", () => {
    // Give the cart real velocity first (one tick), then check the SECOND
    // call's workDelta is nonzero and scales with how many ticks ran.
    const state = initCtSacState({ x0: 0, thetaDeg: 5 });
    const rng = () => 0.5;
    const force = 4;
    const afterOne = advanceCtSac(state, CT_SAC_DT, rng, DEFAULT, constantForcePolicy(force));

    const oneMoreTick = advanceCtSac(afterOne, CT_SAC_DT, rng, DEFAULT, constantForcePolicy(force));
    const twoMoreTicks = advanceCtSac(afterOne, CT_SAC_DT * 2, rng, DEFAULT, constantForcePolicy(force));
    // Same force, same starting state -- two ticks should do roughly (not
    // exactly, since x_dot itself changes tick to tick) twice the work of one.
    expect(Math.abs(twoMoreTicks.workDelta)).toBeGreaterThan(Math.abs(oneMoreTick.workDelta));
  });

  it("is reproducible with the same seeded generator (wind on)", () => {
    const windParams = { ...DEFAULT, windOn: true, sigmaGust: 0.002, sigmaTurb: 0.001 };
    function seededRng(seed) {
      let a = seed;
      return function () {
        a |= 0;
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }
    function rollout(rng) {
      let state = initCtSacState({ x0: 0, thetaDeg: 5 });
      for (let i = 0; i < 20; i++) {
        state = advanceCtSac(state, CT_SAC_DT, rng, windParams, constantForcePolicy(0.5));
      }
      return state.z;
    }
    const a = rollout(seededRng(7));
    const b = rollout(seededRng(7));
    for (let i = 0; i < 4; i++) expect(a[i]).toBeCloseTo(b[i], 12);
  });
});

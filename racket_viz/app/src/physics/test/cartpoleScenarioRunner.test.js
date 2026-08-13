import { describe, expect, it } from "vitest";

import { initialState, isTerminated, terminationReason, advance } from "../cartpoleScenarioRunner.js";
import { heunStep } from "../cartpoleWind.js";
import { X_THRESHOLD, THETA_THRESHOLD_RADIANS } from "../cartpoleDynamics.js";

const DEFAULT = { mp: 0.1, mc: 1.0, l: 0.5, g: 9.8 };

describe("initialState", () => {
  it("converts thetaDeg to radians and zeros both momenta", () => {
    const z = initialState({ x0: 0.3, thetaDeg: 90 });
    expect(z[0]).toBeCloseTo(0.3, 9);
    expect(z[1]).toBeCloseTo(Math.PI / 2, 9);
    expect(z[2]).toBeCloseTo(0, 9);
    expect(z[3]).toBeCloseTo(0, 9);
  });

  it("defaults to a small nonzero tilt (theta=0 exactly is a true fixed point)", () => {
    const z = initialState();
    expect(z[1]).not.toBeCloseTo(0, 3);
  });
});

describe("isTerminated", () => {
  it("is false well within both bounds", () => {
    expect(isTerminated([0, 0, 0, 0])).toBe(false);
  });

  it("is true once the cart passes X_THRESHOLD", () => {
    expect(isTerminated([X_THRESHOLD + 0.01, 0, 0, 0])).toBe(true);
    expect(isTerminated([-X_THRESHOLD - 0.01, 0, 0, 0])).toBe(true);
  });

  it("is true once the pole passes THETA_THRESHOLD_RADIANS", () => {
    expect(isTerminated([0, THETA_THRESHOLD_RADIANS + 0.001, 0, 0])).toBe(true);
    expect(isTerminated([0, -THETA_THRESHOLD_RADIANS - 0.001, 0, 0])).toBe(true);
  });

  it("accepts a wider thetaThreshold override, past which THETA_THRESHOLD_RADIANS alone no longer terminates", () => {
    // cartpoleMain.js passes a wider value than the canonical 12deg so a
    // failure past ct_sac's trained envelope is actually watchable -- see
    // DECISIONS.md. The default (no override) must stay exactly as before.
    const wider = THETA_THRESHOLD_RADIANS * 2;
    expect(isTerminated([0, THETA_THRESHOLD_RADIANS + 0.001, 0, 0], wider)).toBe(false);
    expect(isTerminated([0, wider + 0.001, 0, 0], wider)).toBe(true);
    expect(isTerminated([0, THETA_THRESHOLD_RADIANS + 0.001, 0, 0])).toBe(true); // default unchanged
  });

  it("accepts an xThreshold override, for the Rail Length slider -- default unchanged", () => {
    const wider = X_THRESHOLD * 2;
    expect(isTerminated([X_THRESHOLD + 0.01, 0, 0, 0], THETA_THRESHOLD_RADIANS, wider)).toBe(false);
    expect(isTerminated([wider + 0.01, 0, 0, 0], THETA_THRESHOLD_RADIANS, wider)).toBe(true);
    expect(isTerminated([X_THRESHOLD + 0.01, 0, 0, 0])).toBe(true); // default unchanged
  });
});

describe("terminationReason", () => {
  it("is null when not actually terminated", () => {
    expect(terminationReason([0, 0, 0, 0])).toBeNull();
  });

  it("is 'rail' once the cart passes xThreshold", () => {
    expect(terminationReason([X_THRESHOLD + 0.01, 0, 0, 0])).toBe("rail");
    expect(terminationReason([-X_THRESHOLD - 0.01, 0, 0, 0])).toBe("rail");
  });

  it("is 'angle' once the pole passes thetaThreshold", () => {
    expect(terminationReason([0, THETA_THRESHOLD_RADIANS + 0.001, 0, 0])).toBe("angle");
    expect(terminationReason([0, -THETA_THRESHOLD_RADIANS - 0.001, 0, 0])).toBe("angle");
  });

  it("prefers 'rail' when both boundaries are crossed at once", () => {
    expect(terminationReason([X_THRESHOLD + 0.01, THETA_THRESHOLD_RADIANS + 0.001, 0, 0])).toBe("rail");
  });

  it("honors both threshold overrides, same as isTerminated", () => {
    const widerTheta = THETA_THRESHOLD_RADIANS * 2;
    const widerX = X_THRESHOLD * 2;
    expect(terminationReason([X_THRESHOLD + 0.01, 0, 0, 0], widerTheta, widerX)).toBeNull();
    expect(terminationReason([0, widerTheta + 0.001, 0, 0], widerTheta, widerX)).toBe("angle");
    expect(terminationReason([widerX + 0.01, 0, 0, 0], widerTheta, widerX)).toBe("rail");
  });
});

describe("advance", () => {
  it("matches a single heunStep call when dtReal is already <= the substep cap", () => {
    const z0 = [0.1, 0.05, 0.0, 0.0];
    const dt = 0.002; // below MAX_STEP=0.005, so this should be exactly 1 substep
    const rng = () => 0.42;
    const expected = heunStep(z0, 1.5, dt, rng, { ...DEFAULT, sigmaGust: 0.002, sigmaTurb: 0.001 });
    const actual = advance(z0, 1.5, dt, rng, { ...DEFAULT, sigmaGust: 0.002, sigmaTurb: 0.001, windOn: true });
    for (let i = 0; i < 4; i++) expect(actual[i]).toBeCloseTo(expected[i], 12);
  });

  it("forces sigma to 0 when windOn is false, regardless of the sigma values passed in", () => {
    const z0 = [0.1, 0.05, 0.0, 0.0];
    const dt = 0.002;
    const rng = () => 0.9; // an extreme draw -- would visibly perturb the result if wind leaked through
    const withWindOff = advance(z0, 0.0, dt, rng, { ...DEFAULT, sigmaGust: 0.5, sigmaTurb: 0.5, windOn: false });
    const trulyZeroSigma = heunStep(z0, 0.0, dt, rng, { ...DEFAULT, sigmaGust: 0, sigmaTurb: 0 });
    for (let i = 0; i < 4; i++) expect(withWindOff[i]).toBeCloseTo(trulyZeroSigma[i], 12);
  });

  it("sub-steps a large dtReal into several MAX_STEP-sized calls, not one big one", () => {
    const z0 = [0.1, 0.05, 0.0, 0.0];
    const dt = 0.03; // well above the 0.005 cap -- should be 6 substeps of 0.005
    const rng = () => 0.5;

    const stepped = advance(z0, 0.0, dt, rng, { ...DEFAULT, sigmaGust: 0, sigmaTurb: 0, windOn: true });

    let manual = z0;
    for (let i = 0; i < 6; i++) {
      manual = heunStep(manual, 0.0, dt / 6, rng, { ...DEFAULT, sigmaGust: 0, sigmaTurb: 0 });
    }
    for (let i = 0; i < 4; i++) expect(stepped[i]).toBeCloseTo(manual[i], 9);
  });
});

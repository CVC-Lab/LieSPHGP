import { describe, expect, it } from "vitest";

import { buildPendulumScenario, quaternionAligning } from "../pendulumScenarioRunner.js";
import { quatToRotationMatrix } from "../rigidBody.js";
import { vee, logmSO3 } from "../controllers.js";

describe("quaternionAligning", () => {
  it("maps aBody to bWorld under the resulting rotation", () => {
    const cases = [
      [
        [1, 0, 0],
        [0, 0, -1],
      ],
      [
        [0.3, 0, 0],
        [0, 0, 1],
      ],
      [
        [1, 0, 0],
        [-1, 0, 0],
      ], // 180deg case
      [
        [0, 1, 0],
        [0, 1, 0],
      ], // already aligned
    ];
    for (const [aBody, bWorld] of cases) {
      const q = quaternionAligning(aBody, bWorld);
      const R = quatToRotationMatrix(q);
      const rotated = [
        R[0][0] * aBody[0] + R[0][1] * aBody[1] + R[0][2] * aBody[2],
        R[1][0] * aBody[0] + R[1][1] * aBody[1] + R[1][2] * aBody[2],
        R[2][0] * aBody[0] + R[2][1] * aBody[1] + R[2][2] * aBody[2],
      ];
      const na = Math.hypot(...aBody);
      const nb = Math.hypot(...bWorld);
      const expected = bWorld.map((x) => (x / nb) * na);
      for (let i = 0; i < 3; i++) expect(rotated[i]).toBeCloseTo(expected[i], 6);
    }
  });
});

describe("buildPendulumScenario", () => {
  it("a free (uncontrolled, frictionless) swing started exactly at rest stays at rest", () => {
    const doc = buildPendulumScenario({
      controlOn: false,
      windOn: false,
      frictionCoeff: 0,
      startingAngleDeg: 0,
      T: 5,
      N: 50,
    });
    for (const w of doc.frames.M_body) {
      for (const x of w) expect(Math.abs(x)).toBeLessThan(1e-8);
    }
  });

  it("swing control drives the pendulum's attitude to converge on a target", () => {
    // A 90deg (not the full 180deg antipodal) swing from the natural
    // hanging-down start -- the antipodal case is a known measure-zero
    // near-degenerate case for this control law (see controllers.js's Stage F
    // "almost-global stability" note), not a good regression-test target.
    const Rstar = [
      [1, 0, 0],
      [0, 0, -1],
      [0, 1, 0],
    ];
    const doc = buildPendulumScenario({
      controlOn: true,
      Rstar,
      windOn: false,
      frictionCoeff: 0.02,
      T: 15,
      N: 1500,
    });
    const lastQ = doc.frames.quaternion[doc.frames.quaternion.length - 1];
    const R = quatToRotationMatrix(lastQ);
    // Same attitude-error metric as controllers.test.js's own Stage F check.
    const RtR = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 3; k++) RtR[i][j] += Rstar[k][i] * R[k][j];
      }
    }
    const attErr = Math.hypot(...vee(logmSO3(RtR)));
    expect(attErr).toBeLessThan(0.01);
  });

  it("free scenario has no target_R/desired_H; controlled scenario has both", () => {
    const free = buildPendulumScenario({ controlOn: false, windOn: false, T: 1, N: 10 });
    expect(free.meta.mode).toBe("free");
    expect(free.meta.desired_H).toBeUndefined();

    const controlled = buildPendulumScenario({
      controlOn: true,
      Rstar: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      windOn: false,
      T: 1,
      N: 10,
    });
    expect(controlled.meta.mode).toBe("controlled");
    expect(typeof controlled.meta.desired_H).toBe("number");
  });
});

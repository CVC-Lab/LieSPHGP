import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { gainA, gainB, instabilityTime } from "../cartpoleDiagnostics.js";
import { drift } from "../cartpoleDynamics.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT = { mp: 0.1, mc: 1.0, l: 0.5, g: 9.8 };

function loadFixture(name) {
  const fixturePath = path.join(__dirname, "../../../../physics/tests/fixtures", name);
  return JSON.parse(readFileSync(fixturePath, "utf-8"));
}

describe("gainA / gainB / instabilityTime", () => {
  it("match the coworker's live-demo reference values at default sliders", () => {
    expect(gainA(DEFAULT)).toBeCloseTo(32.195, 2);
    expect(gainB(DEFAULT)).toBeCloseTo(1.4634, 3);
    expect(instabilityTime(DEFAULT)).toBeCloseTo(0.2518, 3);
  });

  it("match Python's cartpole_diagnostics across several plants", () => {
    const cases = loadFixture("cartpole_diagnostics_cases.json");
    for (const c of cases) {
      expect(gainA(c.params)).toBeCloseTo(c.gain_a, 6);
      expect(gainB(c.params)).toBeCloseTo(c.gain_b, 6);
      expect(instabilityTime(c.params)).toBeCloseTo(c.instability_time, 6);
    }
  });

  it("gains are independent of gravity", () => {
    const a1 = gainA(DEFAULT);
    const b1 = gainB(DEFAULT);
    const other = { ...DEFAULT, g: 50.0 };
    expect(gainA(other)).toBeCloseTo(a1, 9);
    expect(gainB(other)).toBeCloseTo(b1, 9);
  });

  it("instability time increases with pole length", () => {
    const short = instabilityTime({ mp: 0.1, mc: 1.0, l: 0.2, g: 9.8 });
    const long = instabilityTime({ mp: 0.1, mc: 1.0, l: 2.0, g: 9.8 });
    expect(long).toBeGreaterThan(short);
  });

  it("instability time decreases with gravity", () => {
    const weakG = instabilityTime({ mp: 0.1, mc: 1.0, l: 0.5, g: 4.9 });
    const strongG = instabilityTime({ mp: 0.1, mc: 1.0, l: 0.5, g: 19.6 });
    expect(strongG).toBeLessThan(weakG);
  });

  it("match the eigenvalues of a from-scratch numerical 4x4 Jacobian of drift at upright", () => {
    const { mp, mc, l, g } = DEFAULT;
    const eps = 1e-6;
    const z0 = [0, 0, 0, 0];
    const rhs = (z) => drift(z, 0.0, DEFAULT);

    // Build the 4x4 Jacobian by central differences, column by column.
    const J = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ];
    for (let i = 0; i < 4; i++) {
      const zPlus = [...z0];
      const zMinus = [...z0];
      zPlus[i] += eps;
      zMinus[i] -= eps;
      const fPlus = rhs(zPlus);
      const fMinus = rhs(zMinus);
      for (let row = 0; row < 4; row++) {
        J[row][i] = (fPlus[row] - fMinus[row]) / (2 * eps);
      }
    }

    // The nonzero eigenvalues live entirely in the (theta, p_theta)
    // subsystem (rows/cols 1 and 3) -- solve that 2x2's char. polynomial
    // directly rather than a general 4x4 eigensolver.
    const a = J[1][1];
    const b = J[1][3];
    const c = J[3][1];
    const d = J[3][3];
    const trace = a + d;
    const det = a * d - b * c;
    const disc = Math.sqrt(trace * trace - 4 * det);
    const lambda = Math.max((trace + disc) / 2, (trace - disc) / 2);

    expect(lambda).toBeCloseTo(1 / instabilityTime(DEFAULT), 4);
    expect(lambda * lambda).toBeCloseTo(gainA(DEFAULT) * mp * g * l, 4);
  });
});

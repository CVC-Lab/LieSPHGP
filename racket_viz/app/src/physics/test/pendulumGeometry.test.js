import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { buildPendulum } from "../pendulumGeometry.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function camelParams(p) {
  return {
    rodLength: p.rod_length,
    bobRadius: p.bob_radius,
    totalMass: p.total_mass,
    bobFraction: p.bob_fraction,
  };
}

describe("buildPendulum", () => {
  it("axial (imin) moment is much smaller than the two equal transverse moments", () => {
    const result = buildPendulum();
    const [I1, I2, I3] = result.I;
    expect(I1).toBeLessThan(I2);
    expect(I2).toBeCloseTo(I3, 9);
  });

  it("evecs are orthonormal, proper rotation", () => {
    const result = buildPendulum();
    const evecs = result.evecs;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let dot = 0;
        for (let k = 0; k < 3; k++) dot += evecs[k][i] * evecs[k][j];
        expect(dot).toBeCloseTo(i === j ? 1 : 0, 9);
      }
    }
  });

  it("mass splits according to bobFraction", () => {
    const result = buildPendulum({ totalMass: 2.0, bobFraction: 0.75 });
    expect(result.bobMass).toBeCloseTo(1.5, 9);
    expect(result.rodMass).toBeCloseTo(0.5, 9);
  });

  it("matches Python's pendulum_geometry.build_pendulum() across several geometries", () => {
    const fixturePath = path.join(
      __dirname,
      "../../../../physics/tests/fixtures/pendulum_geometry_cases.json"
    );
    const cases = JSON.parse(readFileSync(fixturePath, "utf-8"));

    for (const { params, I, com, rod_mass, bob_mass } of cases) {
      const result = buildPendulum(camelParams(params));
      for (let i = 0; i < 3; i++) {
        expect(result.I[i]).toBeCloseTo(I[i], 6);
        expect(result.com[i]).toBeCloseTo(com[i], 6);
      }
      expect(result.rodMass).toBeCloseTo(rod_mass, 9);
      expect(result.bobMass).toBeCloseTo(bob_mass, 9);
    }
  });
});

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  massMatrix,
  velocities,
  momenta,
  drift,
  hamiltonian,
  X_THRESHOLD,
  THETA_THRESHOLD_RADIANS,
} from "../cartpoleDynamics.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT = { mp: 0.1, mc: 1.0, l: 0.5, g: 9.8 };

function loadFixture(name) {
  const fixturePath = path.join(__dirname, "../../../../physics/tests/fixtures", name);
  return JSON.parse(readFileSync(fixturePath, "utf-8"));
}

describe("massMatrix", () => {
  it("is symmetric and positive definite for a range of theta", () => {
    for (const theta of [-3, -1.5, 0, 0.7, 2.5]) {
      const M = massMatrix(theta, DEFAULT);
      expect(M[0][1]).toBeCloseTo(M[1][0], 12);
      const trace = M[0][0] + M[1][1];
      const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
      // Both eigenvalues positive <=> trace > 0 and det > 0 for a 2x2 symmetric matrix.
      expect(trace).toBeGreaterThan(0);
      expect(det).toBeGreaterThan(0);
    }
  });
});

describe("velocities / momenta", () => {
  it("are exact inverse maps", () => {
    const cases = [
      [0.3, 0.5, -0.7],
      [-1.1, -0.2, 0.4],
      [2.8, 1.0, 1.0],
    ];
    for (const [theta, xdot, thdot] of cases) {
      const [px, pth] = momenta(theta, xdot, thdot, DEFAULT);
      const [xdot2, thdot2] = velocities(theta, px, pth, DEFAULT);
      expect(xdot2).toBeCloseTo(xdot, 9);
      expect(thdot2).toBeCloseTo(thdot, 9);
    }
  });

  it("matches Python's cartpole_dynamics across many (params, theta, p_x, p_theta) cases", () => {
    const cases = loadFixture("cartpole_dynamics_cases.json");
    for (const c of cases) {
      const [xdot, thdot] = velocities(c.theta, c.px, c.pth, c.params);
      expect(xdot).toBeCloseTo(c.velocities[0], 9);
      expect(thdot).toBeCloseTo(c.velocities[1], 9);
    }
  });
});

describe("drift", () => {
  it("is stationary at theta in {0, pi, -pi} with zero momenta, at any cart position", () => {
    for (const theta of [0.0, Math.PI, -Math.PI]) {
      for (const x of [-1.0, 0.0, 2.0]) {
        const dz = drift([x, theta, 0.0, 0.0], 0.0, DEFAULT);
        for (const d of dz) expect(d).toBeCloseTo(0, 10);
      }
    }
  });

  it("matches Python's cartpole_dynamics.drift across many cases", () => {
    const cases = loadFixture("cartpole_dynamics_cases.json");
    for (const c of cases) {
      const dz = drift([0.0, c.theta, c.px, c.pth], c.F, c.params);
      for (let i = 0; i < 4; i++) expect(dz[i]).toBeCloseTo(c.drift[i], 9);
    }
  });
});

describe("hamiltonian", () => {
  it("matches Python's cartpole_dynamics.hamiltonian across many cases", () => {
    const cases = loadFixture("cartpole_dynamics_cases.json");
    for (const c of cases) {
      const H = hamiltonian([0.0, c.theta, c.px, c.pth], c.params);
      expect(H).toBeCloseTo(c.hamiltonian, 9);
    }
  });

  it("is conserved under free (F=0, no-wind) fixed-step integration, mirroring the Python oracle's exact-conservation test", () => {
    // A plain forward-Heun rollout (no wind) -- not claiming numerical
    // exactness like scipy's adaptive solve_ivp does in the Python test,
    // just that a small fixed dt keeps drift well within a loose bound
    // over a short horizon.
    let z = [0.3, 0.4, 0.2, -0.3];
    const H0 = hamiltonian(z, DEFAULT);
    const dt = 1e-4;
    for (let i = 0; i < 20000; i++) {
      const f0 = drift(z, 0.0, DEFAULT);
      const zMid = z.map((v, k) => v + f0[k] * dt);
      const f1 = drift(zMid, 0.0, DEFAULT);
      z = z.map((v, k) => v + 0.5 * (f0[k] + f1[k]) * dt);
    }
    const H1 = hamiltonian(z, DEFAULT);
    expect(Math.abs(H1 - H0)).toBeLessThan(1e-4);
  });
});

describe("termination thresholds", () => {
  it("match summer-2026/cartpole.py's reference values", () => {
    expect(X_THRESHOLD).toBeCloseTo(2.4, 12);
    expect(THETA_THRESHOLD_RADIANS).toBeCloseTo((12 * 2 * Math.PI) / 360, 12);
  });
});

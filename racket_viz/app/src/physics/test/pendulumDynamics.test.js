import { describe, expect, it } from "vitest";

import { gravityTorqueBody, frictionTorque } from "../pendulumDynamics.js";

// Rotation mapping body +x to world +z exactly (R_y(-90deg)) -- see
// physics/tests/test_pendulum_dynamics.py for why this is the natural probe
// for a com_body that pendulum_geometry.js always places along body +x.
const R_X_TO_WORLD_Z = [
  [0, 0, -1],
  [0, 1, 0],
  [1, 0, 0],
];

function matMul(A, B) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[i][k] * B[k][j];
      out[i][j] = s;
    }
  }
  return out;
}

describe("gravityTorqueBody", () => {
  const comBody = [0.3, 0, 0];
  const totalMass = 1.0;
  const g = 9.81;

  it("is zero when the COM points straight up or straight down", () => {
    const tauUp = gravityTorqueBody(R_X_TO_WORLD_Z, comBody, totalMass, g);
    for (const x of tauUp) expect(x).toBeCloseTo(0, 9);

    const RflipZ = [
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, -1],
    ];
    const Rdown = matMul(RflipZ, R_X_TO_WORLD_Z);
    const tauDown = gravityTorqueBody(Rdown, comBody, totalMass, g);
    for (const x of tauDown) expect(x).toBeCloseTo(0, 9);
  });

  it("is nonzero off equilibrium", () => {
    const identity = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const tau = gravityTorqueBody(identity, comBody, totalMass, g);
    const mag = Math.sqrt(tau[0] ** 2 + tau[1] ** 2 + tau[2] ** 2);
    expect(mag).toBeGreaterThan(1e-6);
  });

  it("scales linearly with mass", () => {
    const identity = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const tau1 = gravityTorqueBody(identity, comBody, 1.0, g);
    const tau2 = gravityTorqueBody(identity, comBody, 2.0, g);
    for (let i = 0; i < 3; i++) expect(tau2[i]).toBeCloseTo(2 * tau1[i], 9);
  });
});

describe("frictionTorque", () => {
  it("opposes angular velocity", () => {
    const tau = frictionTorque([1, -2, 3], 0.5);
    expect(tau).toEqual([-0.5, 1, -1.5]);
  });

  it("is zero at rest", () => {
    for (const x of frictionTorque([0, 0, 0], 0.5)) expect(x).toBeCloseTo(0, 9);
  });
});

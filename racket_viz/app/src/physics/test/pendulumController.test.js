import { describe, expect, it } from "vitest";

import { GravityCompensatedAttitudeController, defaultGainsForAttitudeControl } from "../pendulumController.js";
import { gravityTorqueBody } from "../pendulumDynamics.js";
import { GeometricAttitudeController } from "../controllers.js";

const COM_BODY = [0.3, 0, 0];
const TOTAL_MASS = 1.0;
const G = 9.81;
const IDENTITY = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

describe("GravityCompensatedAttitudeController", () => {
  it("holds an off-equilibrium target by exactly cancelling gravity", () => {
    const Rstar = IDENTITY;
    const omegaStar = [0, 0, 0];
    const controller = new GravityCompensatedAttitudeController(
      Rstar,
      omegaStar,
      COM_BODY,
      TOTAL_MASS,
      G,
      0.5,
      0.5,
      1e6
    );

    const u = controller.call(Rstar, omegaStar);
    const tauGAtTarget = gravityTorqueBody(Rstar, COM_BODY, TOTAL_MASS, G);
    for (let i = 0; i < 3; i++) expect(u[i]).toBeCloseTo(-tauGAtTarget[i], 9);
    const mag = Math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2);
    expect(mag).toBeGreaterThan(1e-6);
  });

  it("equals base PD plus (current-R) gravity cancellation off-target", () => {
    const Rstar = IDENTITY;
    const omegaStar = [0, 0, 0];
    const controller = new GravityCompensatedAttitudeController(
      Rstar,
      omegaStar,
      COM_BODY,
      TOTAL_MASS,
      G,
      0.5,
      0.5,
      1e6
    );

    const theta = 0.4;
    const R = [
      [Math.cos(theta), 0, Math.sin(theta)],
      [0, 1, 0],
      [-Math.sin(theta), 0, Math.cos(theta)],
    ];
    const omega = [0.1, -0.2, 0.05];

    const base = new GeometricAttitudeController(Rstar, omegaStar, 0.5, 0.5, 1e6);
    const tauPd = base.call(R, omega);
    const tauG = gravityTorqueBody(R, COM_BODY, TOTAL_MASS, G);
    const expected = tauPd.map((t, i) => t - tauG[i]);

    const u = controller.call(R, omega);
    for (let i = 0; i < 3; i++) expect(u[i]).toBeCloseTo(expected[i], 9);
  });

  it("clips the combined torque", () => {
    const Rstar = IDENTITY;
    const omegaStar = [0, 0, 0];
    const controller = new GravityCompensatedAttitudeController(
      Rstar,
      omegaStar,
      COM_BODY,
      TOTAL_MASS,
      G,
      50,
      50,
      0.5
    );
    const theta = 1.2;
    const R = [
      [Math.cos(theta), 0, Math.sin(theta)],
      [0, 1, 0],
      [-Math.sin(theta), 0, Math.cos(theta)],
    ];
    const u = controller.call(R, [3, -3, 3]);
    for (const x of u) expect(Math.abs(x)).toBeLessThanOrEqual(0.5 + 1e-9);
  });
});

describe("defaultGainsForAttitudeControl", () => {
  it("scales with inertia, not fixed", () => {
    const small = defaultGainsForAttitudeControl([0.01, 0.1, 0.1]);
    const large = defaultGainsForAttitudeControl([0.01, 10.0, 10.0]);
    expect(large.KR).toBeCloseTo(small.KR * 100, 6);
    expect(large.Kp).toBeCloseTo(small.Kp * 100, 6);
  });

  it("is critically damped by default", () => {
    const I = [0.02, 0.9, 0.9];
    const ITransverse = 0.9;
    const { KR, Kp } = defaultGainsForAttitudeControl(I);
    const dampingRatio = Kp / (2 * Math.sqrt(KR * ITransverse));
    expect(dampingRatio).toBeCloseTo(1.0, 9);
  });

  it("scales with the requested natural frequency", () => {
    const slow = defaultGainsForAttitudeControl([0.01, 1.0, 1.0], 1.0);
    const fast = defaultGainsForAttitudeControl([0.01, 1.0, 1.0], 2.0);
    expect(fast.KR).toBeCloseTo(slow.KR * 4, 6);
    expect(fast.Kp).toBeCloseTo(slow.Kp * 2, 6);
  });
});

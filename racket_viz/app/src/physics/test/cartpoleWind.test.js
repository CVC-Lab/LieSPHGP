import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { diffusion, heunStep } from "../cartpoleWind.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT = { mp: 0.1, mc: 1.0, l: 0.5, g: 9.8 };

function loadFixture(name) {
  const fixturePath = path.join(__dirname, "../../../../physics/tests/fixtures", name);
  return JSON.parse(readFileSync(fixturePath, "utf-8"));
}

/** Deterministic seedable PRNG (mulberry32), same as wind.test.js, so wind
 * tests are reproducible. Returns a uniform [0,1) source. */
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

/** Always returns the same uniform value -- used to hand-derive an exact
 * expected standardNormal draw for exact-value tests. */
function constantRng(value) {
  return () => value;
}

describe("diffusion", () => {
  it("equals sigma_gust at rest with no turbulence term", () => {
    expect(diffusion(0.0, 0.002, 0.0)).toBeCloseTo(0.002, 12);
  });

  it("increases with angular velocity", () => {
    expect(diffusion(5.0, 0.002, 0.001)).toBeGreaterThan(diffusion(0.1, 0.002, 0.001));
  });

  it("has zero derivative at the origin, not a kink (see cartpole_wind.py's test for why a naive left/right-secant check is wrong)", () => {
    const f = (td) => diffusion(td, 0.0, 1.0, 1e-3);
    const h1 = 1e-4;
    const h2 = 1e-6;
    const secant1 = (f(h1) - f(0.0)) / h1;
    const secant2 = (f(h2) - f(0.0)) / h2;
    expect(secant2 / secant1).toBeLessThan(0.1); // a true kink would give ~1.0
  });

  it("matches Python's cartpole_wind.diffusion across several cases", () => {
    const { diffusion_cases } = loadFixture("cartpole_wind_deterministic_cases.json");
    for (const c of diffusion_cases) {
      expect(diffusion(c.theta_dot, c.sigma_gust, c.sigma_turb, c.smooth_eps)).toBeCloseTo(c.value, 9);
    }
  });
});

describe("heunStep", () => {
  it("matches Python's cartpole_wind.heun_step (sigma=0, deterministic path) across several cases", () => {
    const { heun_zero_sigma_cases } = loadFixture("cartpole_wind_deterministic_cases.json");
    for (const c of heun_zero_sigma_cases) {
      const zNext = heunStep(c.z0, c.F, c.dt, Math.random, {
        ...c.params,
        sigmaGust: 0.0,
        sigmaTurb: 0.0,
      });
      for (let i = 0; i < 4; i++) expect(zNext[i]).toBeCloseTo(c.z_next[i], 9);
    }
  });

  it("reduces to a plain deterministic Heun ODE step when sigma is zero, regardless of the rng draw", () => {
    const z0 = [0.0, 0.3, 0.1, -0.2];
    const F = 0.5;
    const dt = 0.02;
    const outA = heunStep(z0, F, dt, constantRng(0.9), { ...DEFAULT, sigmaGust: 0.0, sigmaTurb: 0.0 });
    const outB = heunStep(z0, F, dt, constantRng(0.1), { ...DEFAULT, sigmaGust: 0.0, sigmaTurb: 0.0 });
    for (let i = 0; i < 4; i++) expect(outA[i]).toBeCloseTo(outB[i], 12);
  });

  it("is reproducible with the same seeded generator", () => {
    function rollout(rng) {
      let z = [0.0, 0.05, 0.0, 0.0];
      for (let i = 0; i < 50; i++) {
        z = heunStep(z, 0.0, 0.02, rng, { ...DEFAULT, sigmaGust: 0.002, sigmaTurb: 0.001 });
      }
      return z;
    }
    const zA = rollout(seededRng(42));
    const zB = rollout(seededRng(42));
    for (let i = 0; i < 4; i++) expect(zA[i]).toBeCloseTo(zB[i], 12);
  });

  it("different seeds diverge", () => {
    function rollout(rng) {
      let z = [0.0, 0.05, 0.0, 0.0];
      for (let i = 0; i < 50; i++) {
        z = heunStep(z, 0.0, 0.02, rng, { ...DEFAULT, sigmaGust: 0.002, sigmaTurb: 0.001 });
      }
      return z;
    }
    const zA = rollout(seededRng(1));
    const zB = rollout(seededRng(2));
    const allClose = zA.every((v, i) => Math.abs(v - zB[i]) < 1e-9);
    expect(allClose).toBe(false);
  });

  it("wind noise variance scales linearly with dt", () => {
    const z0 = [0.0, 0.05, 0.0, 0.0];
    const n = 4000;

    function samplePthKick(dt, seed) {
      const rng = seededRng(seed);
      const kicks = [];
      for (let i = 0; i < n; i++) {
        const z1 = heunStep(z0, 0.0, dt, rng, { ...DEFAULT, sigmaGust: 0.002, sigmaTurb: 0.001 });
        kicks.push(z1[3] - z0[3]);
      }
      return kicks;
    }

    function variance(xs) {
      const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
      return xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
    }

    const varSmall = variance(samplePthKick(0.005, 10));
    const varLarge = variance(samplePthKick(0.02, 11));
    const ratio = varLarge / varSmall;
    expect(ratio).toBeGreaterThan(3.0);
    expect(ratio).toBeLessThan(5.0);
  });
});

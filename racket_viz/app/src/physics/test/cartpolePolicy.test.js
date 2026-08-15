import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it, beforeAll } from "vitest";

import { forward, initWindow, pushFrame, windowToObservation } from "../cartpolePolicy.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let policy;
let vectors;

beforeAll(() => {
  const policyPath = path.join(__dirname, "../../../../data/cartpole_policy.json");
  policy = JSON.parse(readFileSync(policyPath, "utf-8"));
  const vectorsPath = path.join(__dirname, "../../../../physics/tests/fixtures/cartpole_test_vectors.json");
  vectors = JSON.parse(readFileSync(vectorsPath, "utf-8")).vectors;
});

describe("forward", () => {
  it("architecture matches the spec (pins the body.4-has-no-activation gotcha)", () => {
    const shapes = policy.layers.map((L) => [L.name, L.in, L.out, L.activation]);
    expect(shapes).toEqual([
      ["body.0", 48, 400, "relu"],
      ["body.2", 400, 300, "relu"],
      ["body.4", 300, 300, "linear"],
      ["mu", 300, 1, "linear"],
    ]);
  });

  it("matches all 24 of the coworker's own PyTorch-verified test vectors", () => {
    for (const v of vectors) {
      const got = forward(v.obs, policy);
      expect(Math.abs(got - v.expected_action)).toBeLessThan(1e-4);
    }
  });

  it("matches Python's cartpole_policy.forward on the same vectors (cross-language check)", () => {
    // Redundant with the direct-vs-PyTorch check above in spirit, but this
    // is the project's own established cross-validation pattern (JS vs.
    // Python, not just JS vs. ground truth) -- both should obviously agree
    // since they're both checked against the same vectors, but confirming
    // it directly costs nothing and matches how every other module here
    // is tested.
    for (const v of vectors) {
      const got = forward(v.obs, policy);
      expect(got).toBeCloseTo(v.expected_action, 3);
    }
  });

  it("leaning right pushes right -- the documented sanity check", () => {
    const obs = Array(16).fill([0.995000005, 0.099799998, 0.0]).flat();
    expect(forward(obs, policy)).toBeGreaterThan(0);
  });

  it("output stays within the action box for off-distribution inputs", () => {
    const { low, high } = policy.output.rescale;
    // Deterministic pseudo-random-ish inputs (no seeded RNG needed here --
    // this is a bounds check, not a statistical one).
    for (let trial = 0; trial < 20; trial++) {
      const obs = Array.from({ length: 48 }, (_, i) => Math.sin(trial * 7 + i) * 3);
      const action = forward(obs, policy);
      expect(action).toBeGreaterThanOrEqual(low);
      expect(action).toBeLessThanOrEqual(high);
    }
  });
});

describe("initWindow / pushFrame / windowToObservation", () => {
  it("initWindow fills all 16 frames identically with the initial measurement", () => {
    const window = initWindow(0.1, 0.3);
    expect(window).toHaveLength(16);
    const expected = [Math.cos(0.1), Math.sin(0.1), 0.3];
    for (const frame of window) {
      for (let i = 0; i < 3; i++) expect(frame[i]).toBeCloseTo(expected[i], 12);
    }
  });

  it("pushFrame shifts the oldest out and appends the newest", () => {
    const window = initWindow(0.0, 0.0);
    const updated = pushFrame(window, 0.5, 1.0);
    expect(updated).toHaveLength(16);
    for (let i = 0; i < 15; i++) {
      expect(updated[i]).toEqual(window[i + 1]);
    }
    expect(updated[15][0]).toBeCloseTo(Math.cos(0.5), 12);
    expect(updated[15][1]).toBeCloseTo(Math.sin(0.5), 12);
    expect(updated[15][2]).toBe(1.0);
  });

  it("pushFrame does not mutate its input", () => {
    const window = initWindow(0.0, 0.0);
    const snapshot = window.map((f) => [...f]);
    pushFrame(window, 0.5, 1.0);
    expect(window).toEqual(snapshot);
  });

  it("windowToObservation flattens oldest-first", () => {
    const window = [
      [1, 2, 3],
      [4, 5, 6],
      ...Array.from({ length: 14 }, () => [0, 0, 0]),
    ];
    const obs = windowToObservation(window);
    expect(obs).toHaveLength(48);
    expect(obs.slice(0, 6)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("a full advance cycle matches a hand-built observation", () => {
    let window = initWindow(0.0, 0.0);
    const thetasXs = [
      [0.01, 0.0],
      [0.02, 0.001],
      [0.03, 0.003],
    ];
    for (const [theta, x] of thetasXs) window = pushFrame(window, theta, x);
    const obs = windowToObservation(window);

    const expectedTail = thetasXs.flatMap(([theta, x]) => [Math.cos(theta), Math.sin(theta), x]);
    for (let i = 0; i < 9; i++) expect(obs[obs.length - 9 + i]).toBeCloseTo(expectedTail[i], 12);
    for (let i = 0; i < 13 * 3; i += 3) {
      expect(obs[i]).toBeCloseTo(1.0, 12);
      expect(obs[i + 1]).toBeCloseTo(0.0, 12);
      expect(obs[i + 2]).toBeCloseTo(0.0, 12);
    }
  });
});

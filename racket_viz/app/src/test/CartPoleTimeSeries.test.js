import { describe, expect, it } from "vitest";

import { computeCartPoleStateDomain } from "../scenes/TimeSeriesPanel.js";

describe("computeCartPoleStateDomain", () => {
  it("spans both series with padding", () => {
    const [min, max] = computeCartPoleStateDomain([0, 0.2, -0.1], [0, 5, -3], 0);
    expect(min).toBeCloseTo(-3, 9);
    expect(max).toBeCloseTo(5, 9);
  });

  it("pads a flat (constant) pair of series so the domain isn't zero-width", () => {
    const [min, max] = computeCartPoleStateDomain([0.05, 0.05], [0.05, 0.05]);
    expect(max).toBeGreaterThan(min);
  });
});

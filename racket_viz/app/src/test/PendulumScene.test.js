import { describe, expect, it } from "vitest";

import { createPendulumMesh, updatePendulumOrientation } from "../scenes/PendulumScene.js";

describe("createPendulumMesh", () => {
  it("builds rod + bob + pivot, with the bob at com-direction * rodLength", () => {
    const geometry = { com: [0.9, 0, 0], rod_length: 1.0, bob_radius: 0.15 };
    const group = createPendulumMesh(geometry);
    expect(group.children.length).toBe(3);

    const [rodMesh, bobMesh] = group.children;
    expect(bobMesh.position.x).toBeCloseTo(1.0, 9);
    expect(bobMesh.position.y).toBeCloseTo(0, 9);
    expect(bobMesh.position.z).toBeCloseTo(0, 9);
    // Rod sits at the pivot-to-bob midpoint.
    expect(rodMesh.position.x).toBeCloseTo(0.5, 9);
  });

  it("orients the bob correctly for a non-axis-aligned com direction", () => {
    const geometry = { com: [0, 0.9, 0], rod_length: 1.0, bob_radius: 0.15 };
    const group = createPendulumMesh(geometry);
    const [, bobMesh] = group.children;
    expect(bobMesh.position.x).toBeCloseTo(0, 9);
    expect(bobMesh.position.y).toBeCloseTo(1.0, 9);
    expect(bobMesh.position.z).toBeCloseTo(0, 9);
  });
});

describe("updatePendulumOrientation", () => {
  it("sets the group's quaternion from a scalar-first quaternion frame", () => {
    const geometry = { com: [0.9, 0, 0], rod_length: 1.0, bob_radius: 0.15 };
    const group = createPendulumMesh(geometry);
    // 90deg about y: scalar-first [cos45, 0, sin45, 0].
    const c = Math.cos(Math.PI / 4);
    const s = Math.sin(Math.PI / 4);
    updatePendulumOrientation(group, [c, 0, s, 0]);
    expect(group.quaternion.x).toBeCloseTo(0, 9);
    expect(group.quaternion.y).toBeCloseTo(s, 9);
    expect(group.quaternion.z).toBeCloseTo(0, 9);
    expect(group.quaternion.w).toBeCloseTo(c, 9);
  });
});

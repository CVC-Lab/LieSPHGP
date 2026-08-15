import * as THREE from "three";
import { describe, expect, it } from "vitest";

import {
  poleDirection,
  createCartPoleMesh,
  updateCartPoleFrame,
  resizePole,
  cameraHalfExtents,
  createTrackDecoration,
  updateTrackDecoration,
  CART_HEIGHT,
  CART_WIDTH,
} from "../scenes/CartPoleScene.js";

describe("poleDirection", () => {
  it("points straight up (+y) at theta=0 (upright)", () => {
    const [dx, dy] = poleDirection(0);
    expect(dx).toBeCloseTo(0, 9);
    expect(dy).toBeCloseTo(1, 9);
  });

  it("points toward +x at theta=+90deg", () => {
    const [dx, dy] = poleDirection(Math.PI / 2);
    expect(dx).toBeCloseTo(1, 9);
    expect(dy).toBeCloseTo(0, 9);
  });

  it("points straight down (-y) at theta=+-180deg (hanging)", () => {
    const [dx, dy] = poleDirection(Math.PI);
    expect(dx).toBeCloseTo(0, 9);
    expect(dy).toBeCloseTo(-1, 9);
  });
});

describe("updateCartPoleFrame", () => {
  it("sets the group's x position directly from the cart position", () => {
    const mesh = createCartPoleMesh();
    updateCartPoleFrame(mesh, 1.7, 0.0);
    expect(mesh.group.position.x).toBeCloseTo(1.7, 9);
  });

  it("rotates the pole group so its world direction matches poleDirection(theta), for several angles", () => {
    const mesh = createCartPoleMesh();
    for (const theta of [0, 0.3, -0.8, Math.PI / 2, Math.PI - 0.1, -Math.PI]) {
      updateCartPoleFrame(mesh, 0, theta);
      // The pole mesh's local +y axis, rotated by poleGroup's world matrix,
      // should equal poleDirection(theta) in the x-y plane.
      mesh.poleGroup.updateMatrixWorld(true);
      const localUp = new THREE.Vector3(0, 1, 0);
      const worldDir = localUp.applyQuaternion(mesh.poleGroup.quaternion);
      const [expectedX, expectedY] = poleDirection(theta);
      expect(worldDir.x).toBeCloseTo(expectedX, 9);
      expect(worldDir.y).toBeCloseTo(expectedY, 9);
      expect(worldDir.z).toBeCloseTo(0, 9);
    }
  });

  it("rolls the wheel spokes without slipping: rotation = -distance/radius, the only visible sign a wheel is turning", () => {
    const mesh = createCartPoleMesh();
    const [wheel] = mesh.group.children.filter((c) => c.geometry?.type === "CircleGeometry");
    const wheelRadius = wheel.geometry.parameters.radius;

    for (const x of [0, 0.83, -1.4]) {
      updateCartPoleFrame(mesh, x, 0);
      const expectedRotation = -x / wheelRadius;
      for (const spokes of mesh.wheelSpokes) {
        expect(spokes.rotation.z).toBeCloseTo(expectedRotation, 9);
      }
    }
  });
});

describe("createCartPoleMesh / resizePole", () => {
  it("builds a pole of length 2*poleHalfLength, positioned above the pivot", () => {
    const mesh = createCartPoleMesh({ poleHalfLength: 0.7 });
    expect(mesh.poleMesh.geometry.parameters.height).toBeCloseTo(1.4, 9);
    expect(mesh.poleMesh.position.y).toBeCloseTo(0.7, 9); // half the length
  });

  it("cart body's top sits exactly at the pivot height (CART_HEIGHT)", () => {
    const mesh = createCartPoleMesh();
    const bodyTop = mesh.cartMesh.position.y + mesh.cartMesh.geometry.parameters.height / 2;
    expect(bodyTop).toBeCloseTo(CART_HEIGHT, 9);
  });

  it("wheels sit flush ON the rail (bottom edge at y=0), never dipping below it", () => {
    const mesh = createCartPoleMesh();
    const wheels = mesh.group.children.filter((c) => c.geometry?.type === "CircleGeometry");
    expect(wheels).toHaveLength(2);
    for (const wheel of wheels) {
      const wheelBottom = wheel.position.y - wheel.geometry.parameters.radius;
      expect(wheelBottom).toBeCloseTo(0, 9);
      expect(wheel.position.y).toBeGreaterThan(0); // center is above the rail, not on it
    }
    const xs = wheels.map((w) => w.position.x).sort((a, b) => a - b);
    expect(xs[0]).toBeLessThan(0);
    expect(xs[1]).toBeGreaterThan(0);
  });

  it("body bottom edge sits exactly at the wheels' top (no gap, no overlap)", () => {
    const mesh = createCartPoleMesh();
    const wheels = mesh.group.children.filter((c) => c.geometry?.type === "CircleGeometry");
    const wheelTop = wheels[0].position.y + wheels[0].geometry.parameters.radius;
    const bodyBottom = mesh.cartMesh.position.y - mesh.cartMesh.geometry.parameters.height / 2;
    expect(bodyBottom).toBeCloseTo(wheelTop, 9);
  });

  it("resizePole updates an existing mesh's pole geometry in place", () => {
    const mesh = createCartPoleMesh({ poleHalfLength: 0.5 });
    resizePole(mesh, 1.2);
    expect(mesh.poleMesh.geometry.parameters.height).toBeCloseTo(2.4, 9);
    expect(mesh.poleMesh.position.y).toBeCloseTo(1.2, 9);
  });

  it("builds one wheelSpokes LineSegments per wheel, positioned to match its own wheel exactly", () => {
    const mesh = createCartPoleMesh();
    const wheels = mesh.group.children.filter((c) => c.geometry?.type === "CircleGeometry");
    expect(mesh.wheelSpokes).toHaveLength(2);
    const wheelXs = wheels.map((w) => w.position.x).sort((a, b) => a - b);
    const spokeXs = mesh.wheelSpokes.map((s) => s.position.x).sort((a, b) => a - b);
    expect(spokeXs[0]).toBeCloseTo(wheelXs[0], 9);
    expect(spokeXs[1]).toBeCloseTo(wheelXs[1], 9);
    for (const spokes of mesh.wheelSpokes) {
      expect(spokes.position.y).toBeCloseTo(wheels[0].position.y, 9); // same height as the wheels
      expect(spokes).toBeInstanceOf(THREE.LineSegments);
    }
  });
});

describe("cameraHalfExtents", () => {
  it("half-width comfortably exceeds half the cart's own width (includes pole reach and margin)", () => {
    const { halfWidth } = cameraHalfExtents({ poleHalfLength: 1.5 });
    expect(halfWidth).toBeGreaterThan(CART_WIDTH / 2);
  });

  it("no longer depends on xThreshold/track position at all -- this frame is centered on the cart, not the track", () => {
    // See this module's own docstring: the camera now PANS to follow the
    // cart's live x position (cartpoleMain.js's job), so this function only
    // ever needs to frame the cart+pole's own extent, regardless of where
    // on the track the cart currently is -- a call site can no longer even
    // pass an xThreshold (it's not part of the function's signature at
    // all), and the resulting halfWidth is far smaller than it would be if
    // a +-2.4m track still had to fit in the same frame.
    const { halfWidth } = cameraHalfExtents({ poleHalfLength: 1.0, margin: 1.15 });
    const oldStyleHalfWidthWithTrack = (2.4 + CART_WIDTH / 2 + 2.0) * 1.15;
    expect(halfWidth).toBeLessThan(oldStyleHalfWidthWithTrack);
  });

  it("top grows with the pole's CURRENT length (not a fixed slider max); bottom does not (only depends on the cart)", () => {
    const short = cameraHalfExtents({ poleHalfLength: 0.5 });
    const long = cameraHalfExtents({ poleHalfLength: 1.5 });
    expect(long.top).toBeGreaterThan(short.top);
    expect(long.bottom).toBeCloseTo(short.bottom, 9);
  });

  it("bottom is much smaller than top -- only needs to show the cart, not any pole swing", () => {
    // The whole point of the asymmetric framing: within the reachable
    // +-90deg range, the pole tip is always at or above the pivot (see the
    // module's own docstring), so there's no pole-swing budget to reserve
    // below it at all -- confirmed by this being a large ratio, not just
    // "somewhat smaller".
    const { top, bottom } = cameraHalfExtents({ poleHalfLength: 1.0 });
    expect(top / bottom).toBeGreaterThan(2);
  });

  it("the reachable half-circle (|theta| <= 90deg) stays within the frame at any such angle", () => {
    const poleHalfLength = 1.5;
    const { halfWidth, top, bottom } = cameraHalfExtents({ poleHalfLength });
    const poleLength = 2 * poleHalfLength;
    for (const theta of [0, 0.5, Math.PI / 2, -Math.PI / 2, 1.2, -1.4]) {
      const [dx, dy] = poleDirection(theta);
      const tipYRelativeToPivot = dy * poleLength;
      // Reachable tip height is always >= 0 relative to the pivot (never
      // below it) for this range, well within `top`; `bottom` only needs
      // to clear the cart, which this doesn't even touch.
      expect(tipYRelativeToPivot).toBeGreaterThanOrEqual(-1e-9);
      expect(tipYRelativeToPivot).toBeLessThan(top);
      expect(Math.abs(dx * poleLength)).toBeLessThan(halfWidth);
    }
    expect(bottom).toBeGreaterThan(0); // still real headroom for the cart itself
  });

  it("fill fraction (rendered content height / total framed height) is exactly 1/margin, at any geometry", () => {
    // The elegant consequence of top and bottom both scaling off the SAME
    // margin: the fraction of the frame actually occupied by "cart + fully
    // upright pole" never depends on the actual geometry values, only on
    // margin -- confirmed here at two very different pole lengths, so
    // margin=1.2 (cartpoleMain.js's CAMERA_MARGIN) reliably delivers the
    // user's "80% if not more" ask regardless of what the sliders are set to.
    for (const poleHalfLength of [0.3, 1.0]) {
      const margin = 1.2;
      const { top, bottom } = cameraHalfExtents({ poleHalfLength, margin });
      const poleLength = 2 * poleHalfLength;
      const contentHeight = poleLength + CART_HEIGHT;
      const totalFramedHeight = top + bottom;
      expect(contentHeight / totalFramedHeight).toBeCloseTo(1 / margin, 9);
      expect(contentHeight / totalFramedHeight).toBeGreaterThan(0.8);
    }
  });
});

describe("createTrackDecoration / updateTrackDecoration", () => {
  it("builds 4 children: rail, negative tick, positive tick, ground marks, in that order", () => {
    const group = createTrackDecoration(2.4);
    expect(group.children).toHaveLength(4);
    const [rail, tickNeg, tickPos, groundMarks] = group.children;
    // Float32Array-backed (THREE's BufferGeometry) -- only float32
    // precision, not float64, hence the looser tolerance than this file's
    // other geometry assertions.
    const railXs = rail.geometry.attributes.position.array;
    expect(railXs[0]).toBeCloseTo(-2.4 * 1.15, 5); // default trackHalfLength
    expect(railXs[3]).toBeCloseTo(2.4 * 1.15, 5);
    expect(tickNeg.geometry.attributes.position.array[0]).toBeCloseTo(-2.4, 5);
    expect(tickPos.geometry.attributes.position.array[0]).toBeCloseTo(2.4, 5);
    // Ground marks span the same +-trackHalfLength as the rail, as
    // disconnected dashes (LineSegments, not Line) -- an even number of
    // points (2 per dash), each dash's own center within one spacing of
    // the rail's own end (a dash's OUTER edge can overshoot by its own
    // half-width, a small fixed cosmetic amount, never by a whole spacing).
    expect(groundMarks).toBeInstanceOf(THREE.LineSegments);
    const gmXs = groundMarks.geometry.attributes.position.array;
    expect(gmXs.length % 6).toBe(0); // 2 points * 3 components (x,y,z) per dash
    expect(gmXs.length).toBeGreaterThan(0);
    const maxDashX = Math.max(...gmXs.filter((_, i) => i % 3 === 0).map(Math.abs));
    expect(maxDashX).toBeLessThanOrEqual(2.4 * 1.15 + 0.05 + 1e-6); // + GROUND_MARK_HALF_WIDTH
  });

  it("updateTrackDecoration moves an existing group's geometry in place, matching a fresh build at the new xThreshold", () => {
    const group = createTrackDecoration(2.4);
    updateTrackDecoration(group, 4.0);

    const fresh = createTrackDecoration(4.0);
    expect(group.children).toHaveLength(4); // same 4 children, not rebuilt
    for (let i = 0; i < 4; i++) {
      const updatedXs = group.children[i].geometry.attributes.position.array;
      const freshXs = fresh.children[i].geometry.attributes.position.array;
      expect(Array.from(updatedXs)).toEqual(Array.from(freshXs));
    }
  });
});

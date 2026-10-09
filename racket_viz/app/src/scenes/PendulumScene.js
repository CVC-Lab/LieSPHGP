/**
 * 3D rod+bob pendulum render. Unlike RacketScene.js's custom tube+hoop point
 * cloud (needed because the racket's shape comes from a numerical point
 * discretization), pendulumGeometry.js's closed-form geometry lets this use
 * plain THREE.js primitives directly -- a cylinder for the rod, a sphere for
 * the bob -- no custom mesh-building needed.
 */
import * as THREE from "three";
import { THEME } from "../theme.js";
import { quaternionToMatrix } from "./RacketScene.js";

const ROD_RADIUS = 0.03;

/**
 * Builds the pendulum mesh once, in the body frame. The rod+bob both lie
 * exactly along the COM direction from the pivot (pendulumGeometry.js's
 * construction guarantees this) -- using `com`'s own direction here, rather
 * than assuming a specific body axis index, keeps this correct regardless of
 * which axis eigh happens to label the axial one.
 * @param {object} geometry the scenario's `geometry` block: { com, rod_length, bob_radius }
 * @returns {THREE.Group}
 */
export function createPendulumMesh(geometry) {
  const { com, rod_length: rodLength, bob_radius: bobRadius } = geometry;
  const dir = new THREE.Vector3(...com).normalize();
  const bobCenter = dir.clone().multiplyScalar(rodLength);

  const group = new THREE.Group();

  const rodMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(ROD_RADIUS, ROD_RADIUS, rodLength, 12),
    new THREE.MeshBasicMaterial({ color: THEME.racketTube })
  );
  // CylinderGeometry's axis is local +Y -- rotate that onto `dir`, then
  // position at the pivot-to-bob midpoint (a cylinder is centered on its own
  // local origin).
  rodMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  rodMesh.position.copy(dir).multiplyScalar(rodLength / 2);
  group.add(rodMesh);

  const bobMesh = new THREE.Mesh(
    new THREE.SphereGeometry(bobRadius, 24, 16),
    new THREE.MeshBasicMaterial({ color: THEME.racketFaceA })
  );
  bobMesh.position.copy(bobCenter);
  group.add(bobMesh);

  // Small marker at the pivot: since it sits at the group's own local
  // origin, a pure rotation (this group only ever gets .quaternion set, never
  // .position) leaves it fixed at world origin regardless of orientation --
  // no separate non-rotating parent needed, unlike the racket scene's
  // world-frame axes (which DO need to live outside racketGroup, since they
  // aren't at that group's origin).
  const pivotMesh = new THREE.Mesh(
    new THREE.SphereGeometry(ROD_RADIUS * 1.8, 12, 8),
    new THREE.MeshBasicMaterial({ color: THEME.border })
  );
  group.add(pivotMesh);

  return group;
}

/** Applies a scalar-first quaternion frame to an already-built pendulum group. */
export function updatePendulumOrientation(group, q) {
  group.quaternion.setFromRotationMatrix(quaternionToMatrix(q));
}

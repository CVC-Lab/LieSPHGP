/**
 * Flat, side-on cart-pole render -- deliberately NOT a fuller 3D scene like
 * the racket/pendulum (confirmed with the user): the physics is genuinely
 * planar (x and theta only, nothing ever leaves a plane), and the coworker's
 * own live demo (the one this system is meant to be presented alongside)
 * renders it flat too. Still built with THREE.js/WebGL for consistency with
 * the rest of the app's rendering pipeline -- cartpoleMain.js frames it with
 * an orthographic camera looking straight down -z, which removes perspective
 * foreshortening entirely rather than merely aiming a perspective camera at
 * a planar scene.
 *
 * World-frame convention (chosen for rendering only -- the physics itself,
 * cartpole_dynamics.py/js, is agnostic to how theta is drawn):
 *   x = cart position (matches physics x directly)
 *   y = height, track sits at y=0
 *   theta=0 (upright) -> pole points along +y; positive theta tips the pole
 *     toward +x. Pole direction = (sin(theta), cos(theta), 0).
 * This is the standard "angle from vertical, clockwise" convention, like a
 * clock hand measured from 12 o'clock.
 */
import * as THREE from "three";
import { THEME } from "../theme.js";

/** Visual-only constants (do not affect physics). Doubled again from the
 * pygame reference render's literal 0.4x0.24m (cartwidth=50,cartheight=30 at
 * a 600px/4.8m scale) per explicit user request ("at least double the size
 * of the cart"). CART_HEIGHT is the PIVOT height above the rail -- camera
 * framing, track decoration, and cartpoleMain.js's camera y-position all
 * work off this same constant. */
export const CART_WIDTH = 1.1;
export const CART_HEIGHT = 0.68;
export const POLE_RADIUS = 0.025;
const PIVOT_MARKER_RADIUS = POLE_RADIUS * 1.8;
// Wheels sit ON the rail (bottom edge flush with y=0), not dipping below it
// -- an earlier version had them dip below per a misreading of "bring it
// down;" the user confirmed they DO want wheels visible, just not extending
// past the track line.
const WHEEL_RADIUS = CART_HEIGHT * 0.16;
const WHEEL_SPOKE_COUNT = 6;

// Spacing/size of the ground's diagonal "motion" hashes -- see
// createTrackDecoration's own docstring for why these exist at all (the
// chase-cam keeps the cart centered with no rail-end in view, so without
// SOME fixed-world-position marker scrolling past, the cart can look
// perfectly still even while moving). Deliberately much smaller than
// tickHeight (the actual failure-boundary markers) so they read as
// "distance markers," not "danger zone."
const GROUND_MARK_SPACING = 0.4;
const GROUND_MARK_HALF_WIDTH = 0.05;
const GROUND_MARK_HEIGHT = CART_HEIGHT * 0.25;

/**
 * Unit direction the pole points in the world x-y plane, for the rendering
 * convention documented above.
 * @param {number} theta
 * @returns {[number, number]} [x, y] components (z is always 0, planar)
 */
export function poleDirection(theta) {
  return [Math.sin(theta), Math.cos(theta)];
}

/**
 * A single diagonal dash's endpoints, for one ground mark at world-x `x`.
 * Factored out so createTrackDecoration and updateTrackDecoration build
 * byte-identical geometry from the same trackHalfLength.
 */
function groundMarkPoints(trackHalfLength) {
  const points = [];
  for (let x = -trackHalfLength; x <= trackHalfLength; x += GROUND_MARK_SPACING) {
    points.push(new THREE.Vector3(x - GROUND_MARK_HALF_WIDTH, 0, 0));
    points.push(new THREE.Vector3(x + GROUND_MARK_HALF_WIDTH, GROUND_MARK_HEIGHT, 0));
  }
  return points;
}

/**
 * Builds the cart-pole mesh once. `group`'s position.x is the only thing
 * `updateCartPoleFrame` ever sets on it (cart translation); `poleGroup`
 * (a child of `group`, pivoted at the cart's top-center) has its
 * rotation.z set each frame for the pole's tilt -- see that function for
 * the sign convention.
 * @param {{poleHalfLength?: number}} [opts] poleHalfLength: HALF the pole's
 *   length (m), matching cartpole_dynamics.js's own `l` convention; the
 *   rendered pole is the full length, 2*l, same as the reference env's
 *   pygame render (`polelen = scale * (2 * length)`).
 * @returns {{group: THREE.Group, poleGroup: THREE.Group, cartMesh: THREE.Mesh,
 *   poleMesh: THREE.Mesh, wheelSpokes: THREE.LineSegments[]}}
 */
export function createCartPoleMesh({ poleHalfLength = 0.5 } = {}) {
  const group = new THREE.Group();

  // Body occupies [2*WHEEL_RADIUS, CART_HEIGHT], not [0, CART_HEIGHT] --
  // leaves exactly enough room for the wheels below it (which span
  // [0, 2*WHEEL_RADIUS], center at WHEEL_RADIUS -- their TOP, not their
  // center, is what the body's bottom edge needs to meet).
  const bodyDepth = CART_WIDTH * 0.6;
  const bodyHeight = CART_HEIGHT - 2 * WHEEL_RADIUS;
  const cartMesh = new THREE.Mesh(
    new THREE.BoxGeometry(CART_WIDTH, bodyHeight, bodyDepth),
    new THREE.MeshBasicMaterial({ color: THEME.cartpoleCart })
  );
  cartMesh.position.set(0, 2 * WHEEL_RADIUS + bodyHeight / 2, 0);
  group.add(cartMesh);

  // Wheel CENTER at y=WHEEL_RADIUS, so the wheel spans [0, 2*WHEEL_RADIUS]
  // -- bottom edge flush with the rail, never dipping below it. Placed just
  // in front of the body's own front face so they're never partially
  // hidden behind it in the orthographic view.
  const wheelZ = bodyDepth / 2 + 0.005;
  const wheelX = CART_WIDTH / 2 - WHEEL_RADIUS * 1.6;
  const wheelSpokes = [];
  for (const sign of [-1, 1]) {
    const wheelMesh = new THREE.Mesh(
      new THREE.CircleGeometry(WHEEL_RADIUS, 16),
      new THREE.MeshBasicMaterial({ color: THEME.border })
    );
    wheelMesh.position.set(sign * wheelX, WHEEL_RADIUS, wheelZ);
    group.add(wheelMesh);

    // Radial spokes, rotated in updateCartPoleFrame as the cart moves --
    // the ONLY visible cue this wheel is rolling at all, since the disc
    // itself (a flat, uniformly-colored CircleGeometry) looks identical at
    // any rotation. A hair closer to the camera than the disc (wheelZ +
    // 0.001) so they render on top of it rather than z-fighting.
    const spokePoints = [];
    for (let i = 0; i < WHEEL_SPOKE_COUNT; i++) {
      const angle = (i / WHEEL_SPOKE_COUNT) * Math.PI * 2;
      spokePoints.push(new THREE.Vector3(0, 0, 0));
      spokePoints.push(new THREE.Vector3(Math.cos(angle) * WHEEL_RADIUS, Math.sin(angle) * WHEEL_RADIUS, 0));
    }
    const spokes = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(spokePoints),
      new THREE.LineBasicMaterial({ color: THEME.panelBg })
    );
    spokes.position.set(sign * wheelX, WHEEL_RADIUS, wheelZ + 0.001);
    group.add(spokes);
    wheelSpokes.push(spokes);
  }

  const poleGroup = new THREE.Group();
  poleGroup.position.set(0, CART_HEIGHT, 0);
  group.add(poleGroup);

  const poleLength = 2 * poleHalfLength;
  const poleMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(POLE_RADIUS, POLE_RADIUS, poleLength, 12),
    new THREE.MeshBasicMaterial({ color: THEME.cartpolePole })
  );
  // CylinderGeometry's axis is local +Y and it's centered on its own
  // origin -- theta=0 should put the pole's FAR end at +poleLength along
  // +Y from the pivot, so shift it up by half its length.
  poleMesh.position.set(0, poleLength / 2, 0);
  poleGroup.add(poleMesh);

  const pivotMesh = new THREE.Mesh(
    new THREE.SphereGeometry(PIVOT_MARKER_RADIUS, 12, 8),
    new THREE.MeshBasicMaterial({ color: THEME.border })
  );
  poleGroup.add(pivotMesh);

  return { group, poleGroup, cartMesh, poleMesh, wheelSpokes };
}

/**
 * Rebuilds just the pole's geometry in place when poleHalfLength changes
 * (a geometry-slider live-preview, mirroring racketGeometry's
 * previewRacketGeometry() pattern) -- avoids rebuilding the whole mesh
 * group for a change that only affects one dimension.
 * @param {{poleMesh: THREE.Mesh}} mesh from createCartPoleMesh
 * @param {number} poleHalfLength
 */
export function resizePole(mesh, poleHalfLength) {
  const poleLength = 2 * poleHalfLength;
  mesh.poleMesh.geometry.dispose();
  mesh.poleMesh.geometry = new THREE.CylinderGeometry(POLE_RADIUS, POLE_RADIUS, poleLength, 12);
  mesh.poleMesh.position.set(0, poleLength / 2, 0);
}

/**
 * Applies one physics frame's (x, theta) to an already-built mesh.
 * rotation.z = -theta because THREE's rotation.z is a standard right-handed
 * rotation about +z (RotZ(phi) maps local +y to (-sin(phi), cos(phi), 0)),
 * and this module's convention wants +theta to tip the pole toward +x, i.e.
 * direction (sin(theta), cos(theta), 0) -- setting phi=-theta reconciles
 * the two (see poleDirection's tests for the direct numeric check).
 *
 * Also rolls the wheel spokes: rolling-without-slipping means the wheel
 * turns through angle = distance/radius (arc length = radius * angle) --
 * a real physical relation, not a cosmetic guess, and (like the pole's own
 * rotation above) negated for the same right-handed-rotation reason: moving
 * in +x is rolling to the right, which is a CLOCKWISE turn as drawn (top of
 * the wheel moves toward +x), i.e. a NEGATIVE rotation.z in THREE's
 * standard convention.
 * @param {{group: THREE.Group, poleGroup: THREE.Group, wheelSpokes?: THREE.LineSegments[]}} mesh
 * @param {number} x cart position
 * @param {number} theta pole angle (0 = upright)
 */
export function updateCartPoleFrame(mesh, x, theta) {
  mesh.group.position.x = x;
  mesh.poleGroup.rotation.z = -theta;
  const wheelRotation = -x / WHEEL_RADIUS;
  for (const spokes of mesh.wheelSpokes ?? []) spokes.rotation.z = wheelRotation;
}

/**
 * Static track decoration (the horizontal rail + the two failure-boundary
 * tick marks at +-xThreshold + a run of diagonal "motion" ground marks) --
 * built once by `createTrackDecoration`, but (now that Rail Length is a
 * live sidebar slider, not a fixed constant) updatable in place afterward
 * via `updateTrackDecoration`, same dispose-and-reassign pattern as
 * `resizePole`.
 *
 * The ground marks exist because the chase-cam (see cameraHalfExtents's
 * own docstring) keeps the cart centered at all times -- with no rail-end
 * ever in view, the cart can look perfectly motionless even while
 * genuinely moving, since there's nothing else on screen for the eye to
 * measure it against. Fixed at regular WORLD-space x positions (not
 * attached to the cart), they scroll past exactly as the cart moves,
 * giving a cheap, always-available motion cue -- same idea as a road's
 * lane-divider dashes telling you your own speed.
 * @param {number} xThreshold e.g. cartpole_dynamics.js's X_THRESHOLD, or the
 *   sidebar's live Rail Length slider (halved -- see cartpoleMain.js's
 *   railHalfLengthFromSlider, mirroring poleHalfLengthFromSlider's own
 *   full-length-in-the-UI/half-length-in-the-physics convention)
 * @param {number} [trackHalfLength] how far the visible rail (and its
 *   ground marks) extend past the boundary markers; defaults to
 *   xThreshold*1.15, an arbitrary but fixed margin -- NOT tied to the
 *   camera anymore (the chase-cam no longer frames the track at all, see
 *   cameraHalfExtents's own docstring)
 * @returns {THREE.Group}
 */
export function createTrackDecoration(xThreshold, trackHalfLength = xThreshold * 1.15) {
  const group = new THREE.Group();

  const trackMaterial = new THREE.LineBasicMaterial({ color: THEME.cartpoleTrack });
  // Deliberately brighter than trackMaterial -- the rail/tick lines above
  // are subtle by design (they're not meant to draw the eye), but these
  // marks' entire JOB is to be noticeable at a glance (that's the whole
  // point of a motion cue) -- `muted`, not `cartpoleTrack`/`border`, is the
  // closest already-in-palette color with real contrast against panelBg
  // that isn't already claimed by the cart, pole, or wheels.
  const groundMarkMaterial = new THREE.LineBasicMaterial({ color: THEME.muted });
  const trackGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-trackHalfLength, 0, 0),
    new THREE.Vector3(trackHalfLength, 0, 0),
  ]);
  group.add(new THREE.Line(trackGeom, trackMaterial));

  const tickHeight = CART_HEIGHT * 1.5;
  for (const sign of [-1, 1]) {
    const tickGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(sign * xThreshold, 0, 0),
      new THREE.Vector3(sign * xThreshold, tickHeight, 0),
    ]);
    group.add(new THREE.Line(tickGeom, trackMaterial));
  }

  // LineSegments (not Line): each consecutive PAIR of points is its own
  // disconnected dash, unlike the rail/tick Lines above which are each
  // meant to be one continuous stroke.
  group.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(groundMarkPoints(trackHalfLength)), groundMarkMaterial));

  return group;
}

/**
 * Updates an existing track decoration's geometry in place when the Rail
 * Length slider changes -- mirrors `resizePole`'s own dispose-and-reassign
 * pattern exactly, rather than rebuilding (and re-adding to the scene) a
 * whole new group for a change that only affects 4 line geometries.
 * @param {THREE.Group} group from createTrackDecoration -- children are
 *   [rail, negative tick, positive tick, ground marks], in that exact
 *   creation order
 * @param {number} xThreshold
 * @param {number} [trackHalfLength] see createTrackDecoration
 */
export function updateTrackDecoration(group, xThreshold, trackHalfLength = xThreshold * 1.15) {
  const [rail, tickNeg, tickPos, groundMarks] = group.children;
  const tickHeight = CART_HEIGHT * 1.5;

  rail.geometry.dispose();
  rail.geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-trackHalfLength, 0, 0),
    new THREE.Vector3(trackHalfLength, 0, 0),
  ]);

  for (const [tick, sign] of [[tickNeg, -1], [tickPos, 1]]) {
    tick.geometry.dispose();
    tick.geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(sign * xThreshold, 0, 0),
      new THREE.Vector3(sign * xThreshold, tickHeight, 0),
    ]);
  }

  groundMarks.geometry.dispose();
  groundMarks.geometry = new THREE.BufferGeometry().setFromPoints(groundMarkPoints(trackHalfLength));
}

/**
 * Half-width, plus ASYMMETRIC top/bottom half-heights (all measured from
 * the pivot, y=CART_HEIGHT), an orthographic camera needs to frame this
 * system's CART+POLE ONLY, at the pole's CURRENT length -- deliberately
 * NOT the full +-xThreshold track anymore (see below for why).
 *
 * Sized for the CURRENT pole length, not the Pole Length slider's worst
 * case (unlike every other camera in this app) -- per explicit user
 * request ("cart and upright pole should fill the display to 80% if not
 * more"). `cartpoleMain.js` recomputes this (and re-runs `resize()`) every
 * time the Pole Length slider changes, same moment the existing
 * `resizePole` live-preview call already fires.
 *
 * `halfWidth` USED to also add `xThreshold` here, to keep the entire rail
 * in frame at all times -- but that made the panel need a ~4-6:1
 * width:height aspect ratio to hit any real fill fraction, which no plain
 * grid cell is ever shaped like (measured ~27-45% fill even after
 * reworking the panel into a banner layout -- see DECISIONS.md, since
 * reverted, the user correctly called the banner "unnatural": every other
 * panel on every system is a plain 2x2 grid cell, and this was the odd one
 * out). Dropping `xThreshold` and instead having `cartpoleMain.js` pan the
 * camera horizontally to track the cart's live x position (see its
 * `resize()`/`animate()` call sites) fixes both problems at once: the
 * frame only ever needs to be as wide as the cart+pole themselves (this
 * function), so a normal square-ish panel comfortably satisfies it, AND
 * the rail/boundary markers now scroll past naturally as the cart moves --
 * which the user explicitly said was fine ("I'm ok if we have to scroll
 * the background as the cart moves").
 *
 * Still vertically asymmetric for the same reason as before: the Starting
 * Angle slider (now +-20deg, narrowed from an original +-90deg once we
 * confirmed empirically that ct_sac's trained behavior falls off a cliff
 * well before 90 anyway -- see DECISIONS.md) never exceeds the range where
 * cos(theta) >= 0, so the pole's tip height is ALWAYS >= the pivot height
 * -- `bottom` only ever needs to show the cart itself, never any of the
 * pole's swing. (This was true with a lot more headroom to spare back when
 * the slider still went to +-90deg -- narrowing it only made the assumption
 * safer, never invalidated it.)
 *
 * The fill fraction this produces is exactly `1/margin` whenever height is
 * the binding constraint (nearly always now, since `halfWidth` no longer
 * has to accommodate the track) -- both `top` and `bottom` scale off the
 * SAME `margin`, and the rendered content height is exactly
 * `poleLength + CART_HEIGHT`, so it cancels out algebraically. margin=1.2
 * (cartpoleMain.js's CAMERA_MARGIN) was chosen specifically because
 * 1/1.2 = 83.3%, comfortably matching "80% if not more."
 * @param {{poleHalfLength: number, margin?: number}} opts
 *   poleHalfLength: the CURRENT slider value (not a fixed maximum)
 * @returns {{halfWidth: number, top: number, bottom: number}}
 */
export function cameraHalfExtents({ poleHalfLength, margin = 1.15 }) {
  const poleLength = 2 * poleHalfLength;
  // The pole can point at any angle within the reachable range (see
  // above), so a pole lying flat horizontal reaches poleLength
  // horizontally from the cart's OWN center, regardless of where the cart
  // is on the track -- this frame is centered on the cart (see this
  // function's own docstring), so the cart's absolute track position
  // never enters this calculation.
  const halfWidth = (CART_WIDTH / 2 + poleLength) * margin;
  const top = poleLength * margin;
  const bottom = CART_HEIGHT * margin;
  return { halfWidth, top, bottom };
}

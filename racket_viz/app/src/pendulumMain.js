/**
 * Windy pendulum system: same structure as main.js (own Three.js renderer/
 * scene, own sidebar wiring, own rAF loop gated by pause()/resume() for the
 * carousel), reusing the racket's generic panel modules (ControlPanel's
 * drawEnergyPanel, TimeSeriesPanel, ControlMetrics' torque/work panel)
 * directly -- see DECISIONS.md for why this is a second copy-pasted module
 * rather than a shared generic "system" class.
 *
 * main.js's own document-wide `.explain-switch` wiring already covers this
 * system's 4 Explain toggles too (it queries the whole document, not scoped
 * to the racket's markup) -- nothing pendulum-specific needed for that.
 */
import * as THREE from "three";

import { Playback } from "./core/Playback.js";
import { createPendulumMesh, updatePendulumOrientation } from "./scenes/PendulumScene.js";
import { computeOmega, computeOmegaDomain, drawTimeSeries } from "./scenes/TimeSeriesPanel.js";
import { drawEnergyPanel } from "./scenes/ControlPanel.js";
import {
  computeCumulativeWork,
  computeTorqueMagnitude,
  computeAchievedIndexAttitude,
  computeSettledIndex,
  computeAttitudeErrorDeg,
  drawTorquePanel,
} from "./scenes/ControlMetrics.js";
import { buildPendulum } from "./physics/pendulumGeometry.js";
import { buildPendulumScenario, quaternionAligning } from "./physics/pendulumScenarioRunner.js";
import { quatToRotationMatrix } from "./physics/rigidBody.js";
import { THEME } from "./theme.js";

const DEFAULT_DURATION_SECONDS = 60;

const pendulumCanvas = document.getElementById("pendulum-canvas");
const energyCanvas = document.getElementById("pendulum-energy-canvas");
const energyCtx = energyCanvas.getContext("2d");
const omegaCanvas = document.getElementById("pendulum-omega-canvas");
const omegaCtx = omegaCanvas.getContext("2d");
const torqueCanvas = document.getElementById("pendulum-torque-canvas");
const torqueCtx = torqueCanvas.getContext("2d");

const playPauseBtn = document.getElementById("pendulum-pause-button");
const timeReadout = document.getElementById("pendulum-time-readout");
const pendulumLabel = document.getElementById("pendulum-label");
const energyLabel = document.getElementById("pendulum-energy-label");
const metricsText = document.getElementById("pendulum-metrics-text");
const inertiaReadout = document.getElementById("pendulum-inertia-readout");

// ── Sidebar elements ────────────────────────────────────────────────────────
const sliderIds = ["rodLength", "bobRadius", "pendulumTotalMass", "bobFraction", "startingAngleDeg", "pendulumTargetAngleDeg", "frictionCoeff", "pendulumWindStd"];
const sliders = Object.fromEntries(sliderIds.map((id) => [id, document.getElementById(id)]));
const sliderVals = Object.fromEntries(sliderIds.map((id) => [id, document.getElementById(`${id}-val`)]));
const controlOnCheckbox = document.getElementById("pendulumControlOn");
const windOnCheckbox = document.getElementById("pendulumWindOn");
const controlSubfields = document.getElementById("pendulum-control-subfields");
const windSubfields = document.getElementById("pendulum-wind-subfields");
const runButton = document.getElementById("pendulum-run-button");
const resetButton = document.getElementById("pendulum-reset-button");

const DECIMALS_BY_SLIDER = { bobRadius: 3, frictionCoeff: 3, startingAngleDeg: 0, pendulumTargetAngleDeg: 0 };
function refreshSliderLabels() {
  for (const id of sliderIds) {
    sliderVals[id].textContent = Number(sliders[id].value).toFixed(DECIMALS_BY_SLIDER[id] ?? 2);
  }
}
function refreshSubfieldVisibility() {
  controlSubfields.style.display = controlOnCheckbox.checked ? "flex" : "none";
  windSubfields.style.display = windOnCheckbox.checked ? "flex" : "none";
  metricsText.style.display = controlOnCheckbox.checked ? "block" : "none";
}
for (const id of sliderIds) sliders[id].addEventListener("input", refreshSliderLabels);
controlOnCheckbox.addEventListener("change", refreshSubfieldVisibility);
windOnCheckbox.addEventListener("change", refreshSubfieldVisibility);
refreshSliderLabels();
refreshSubfieldVisibility();

function formatInertia(I) {
  const rows = [
    ["I₁ (Axial)", I[0]],
    ["I₂ (Transverse)", I[1]],
    ["I₃ (Transverse)", I[2]],
  ];
  const rowsHtml = rows
    .map(
      ([label, value]) =>
        `<div style="display:flex;justify-content:space-between;">${label}<span>${value.toFixed(4)}</span></div>`
    )
    .join("");
  return `<div style="margin-bottom:4px;">Moments of Inertia</div>${rowsHtml}`;
}

function readGeometryFromSidebar() {
  return {
    rodLength: Number(sliders.rodLength.value),
    bobRadius: Number(sliders.bobRadius.value),
    totalMass: Number(sliders.pendulumTotalMass.value),
    bobFraction: Number(sliders.bobFraction.value),
  };
}

/** Rebuilds just the pendulum *shape* from the current geometry sliders, so
 * the mesh updates live while dragging -- mirrors main.js's own
 * previewRacketGeometry. */
function previewPendulumGeometry() {
  const geometry = buildPendulum(readGeometryFromSidebar());
  const previousQuaternion = pendulumGroup ? pendulumGroup.quaternion.clone() : null;

  clearScene(pendulumScene);
  pendulumGroup = createPendulumMesh({ com: geometry.com, rod_length: geometry.rodLength, bob_radius: geometry.bobRadius });
  if (previousQuaternion) pendulumGroup.quaternion.copy(previousQuaternion);
  pendulumScene.add(pendulumGroup);

  inertiaReadout.innerHTML = formatInertia(geometry.I);
}
for (const id of ["rodLength", "bobRadius", "pendulumTotalMass", "bobFraction"]) {
  sliders[id].addEventListener("input", previewPendulumGeometry);
}

/** Shared tilt convention for both the "Starting angle" and "Target angle"
 * sliders: 0deg is hanging straight down, +-180deg is inverted (upright),
 * +-90deg is horizontal, with the sign choosing which side of the vertical
 * the pendulum swings toward -- one continuous dial rather than three
 * discrete presets, since nothing in the physics actually limits a target
 * to an axis-aligned pose. */
function tiltedDownDirForAngle(angleDeg) {
  const angleRad = (angleDeg * Math.PI) / 180;
  return [Math.sin(angleRad), 0, -Math.cos(angleRad)];
}

/** Live-previews the tilted starting pose as "Starting angle" is dragged --
 * mirrors the racket's own "Starting axis"/"Starting perturbation" sliders,
 * which preview a starting STATE (unlike gain/friction/wind sliders, which
 * are abstract parameters with no single static picture to show). Skipped
 * while a run is actively playing: the per-frame animate() loop already
 * owns pendulumGroup's orientation every frame, and fighting it here would
 * just flicker. Safe while paused or before the first Run, since nothing
 * else is writing to the orientation at that moment. */
function previewStartingOrientation() {
  if (playback && playback.playing) return;
  previewingPose = true;
  const geometry = buildPendulum(readGeometryFromSidebar());
  const q0 = quaternionAligning(geometry.com, tiltedDownDirForAngle(Number(sliders.startingAngleDeg.value)));
  updatePendulumOrientation(pendulumGroup, q0);
}
sliders.startingAngleDeg.addEventListener("input", previewStartingOrientation);

/** Same idea as previewStartingOrientation, for the "Target angle" slider --
 * only meaningful while Control is on, since there's no target otherwise. */
function previewTargetOrientation() {
  if (playback && playback.playing) return;
  if (!controlOnCheckbox.checked) return;
  previewingPose = true;
  const geometry = buildPendulum(readGeometryFromSidebar());
  const q = quaternionAligning(geometry.com, tiltedDownDirForAngle(Number(sliders.pendulumTargetAngleDeg.value)));
  updatePendulumOrientation(pendulumGroup, q);
}
sliders.pendulumTargetAngleDeg.addEventListener("input", previewTargetOrientation);

/** Builds the target orientation matrix for the sidebar's continuous "Target
 * angle" slider, from the CURRENT geometry's own COM direction (so it stays
 * correct as geometry sliders change) -- reuses the same alignment helper
 * the scenario runner uses for its default "hanging down" start. */
function targetRotationForAngle(angleDeg, comBody) {
  const q = quaternionAligning(comBody, tiltedDownDirForAngle(angleDeg));
  return quatToRotationMatrix(q);
}

// ── Three.js setup ───────────────────────────────────────────────────────────
const pendulumRenderer = new THREE.WebGLRenderer({ canvas: pendulumCanvas, antialias: true });
pendulumRenderer.setPixelRatio(window.devicePixelRatio);

const pendulumScene = new THREE.Scene();
pendulumScene.background = new THREE.Color(THEME.panelBg);

// Gravity is world -z (see pendulumDynamics.js's gravityTorqueBody -- Fg =
// [0,0,-mg]), NOT -y like Three.js's own up-axis convention -- so the
// camera needs its own up vector set to +z, or "down" reads as sideways on
// screen instead of as a vertical pendulum swing.
//
// Framed statically for the WORST CASE (rodLength/bobRadius sliders' own
// max, see index.html), pivot centered rather than pinned -- not a
// per-frame auto-fit. The bob can swing to ANY point on a sphere of radius
// MAX_EXTENT around the pivot (straight down for a free swing, but also
// straight UP for a swing-up-to-inverted Control run, or anywhere in
// between) -- centering on the pivot and sizing for that full sphere is
// what actually guarantees nothing goes off-screen, unlike an earlier
// version pinned near the top (sized for the hang-down case only), which
// looked right for a free swing but clipped the bob for an inverted target.
const MAX_ROD_LENGTH = 2.0; // matches index.html's #rodLength max
const MAX_BOB_RADIUS = 0.3; // matches index.html's #bobRadius max
const MAX_EXTENT = MAX_ROD_LENGTH + MAX_BOB_RADIUS;
const FRAME_MARGIN = 1.15; // headroom so the bob doesn't touch the frame edge
const VERTICAL_FOV_DEG = 45;

const halfHeight = MAX_EXTENT * FRAME_MARGIN;
const cameraDistance = halfHeight / Math.tan((VERTICAL_FOV_DEG / 2) * (Math.PI / 180));
// Same viewing direction as before (mostly -y, slight x/z offset for a bit
// of depth), just rescaled to the new distance.
const VIEW_DIR = new THREE.Vector3(0.5, -2.8, 0.7).normalize();

const pendulumCamera = new THREE.PerspectiveCamera(VERTICAL_FOV_DEG, 1, 0.1, 100);
pendulumCamera.up.set(0, 0, 1);
pendulumCamera.position.copy(VIEW_DIR).multiplyScalar(cameraDistance);
pendulumCamera.lookAt(0, 0, 0);

let pendulumGroup = null;
let playback = null;
let currentDoc = null;
let omegaSeries = null;
let omegaDomain = null;
let torqueMagnitudeSeries = null;
let cumulativeWorkSeries = null;
let lastTorqueTextMs = null;
let achievedIndex = null;
let settledIndex = null;
// True while the "Starting angle" or "Target angle" preview owns
// pendulumGroup's orientation -- updateFrame() runs every animate() frame
// regardless of play/pause (so readouts stay live while paused), which
// would silently overwrite the preview on the very next frame otherwise.
// Cleared on Run/Play/Reset, all of which should show the actual doc's
// orientation again.
let previewingPose = false;

function resize2DCanvas(canvas, ctx) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resize() {
  const w = pendulumCanvas.clientWidth;
  const h = pendulumCanvas.clientHeight;
  pendulumRenderer.setSize(w, h, false);
  pendulumCamera.aspect = w / h;
  pendulumCamera.updateProjectionMatrix();

  resize2DCanvas(energyCanvas, energyCtx);
  resize2DCanvas(omegaCanvas, omegaCtx);
  resize2DCanvas(torqueCanvas, torqueCtx);
}
window.addEventListener("resize", resize);

function clearScene(scene) {
  while (scene.children.length) scene.remove(scene.children[0]);
}

function segmentAt(index) {
  const segs = currentDoc.meta.segments;
  return segs.find((s) => index >= s.start && index < s.end) ?? segs[segs.length - 1];
}

function updateFrame(index) {
  const q = currentDoc.frames.quaternion[index];
  if (!previewingPose) updatePendulumOrientation(pendulumGroup, q);

  const seg = segmentAt(index);
  pendulumLabel.textContent = seg.label;
  timeReadout.textContent = `t = ${currentDoc.frames.t[index].toFixed(1)}s`;

  drawTimeSeries(omegaCtx, {
    t: currentDoc.frames.t,
    omega: omegaSeries,
    domain: omegaDomain,
    currentIndex: index,
    width: omegaCanvas.clientWidth,
    height: omegaCanvas.clientHeight,
  });

  const references =
    currentDoc.meta.mode === "controlled"
      ? [{ value: currentDoc.meta.desired_H, color: THEME.target, label: "Desired H" }]
      : [];
  drawEnergyPanel(energyCtx, {
    t: currentDoc.frames.t,
    H: currentDoc.frames.H,
    references,
    currentIndex: index,
    width: energyCanvas.clientWidth,
    height: energyCanvas.clientHeight,
  });

  if (currentDoc.meta.mode === "controlled") {
    drawTorquePanel(torqueCtx, {
      t: currentDoc.frames.t,
      torqueMagnitude: torqueMagnitudeSeries,
      currentIndex: index,
      width: torqueCanvas.clientWidth,
      height: torqueCanvas.clientHeight,
      withAxes: true,
    });
    updateMetricsText(index);
  } else {
    torqueCtx.clearRect(0, 0, torqueCanvas.width, torqueCanvas.height);
    torqueCtx.fillStyle = THEME.panelBg;
    torqueCtx.fillRect(0, 0, torqueCanvas.width, torqueCanvas.height);
  }
}

function updateMetricsText(index) {
  const nowMs = performance.now();
  if (lastTorqueTextMs !== null && nowMs - lastTorqueTextMs < 500) return;
  lastTorqueTextMs = nowMs;

  const workSoFar = cumulativeWorkSeries[index];
  const currentTorque = torqueMagnitudeSeries[index];
  let html = `Work: ${workSoFar.toFixed(3)} J<br>Torque: ${currentTorque.toFixed(3)} N&middot;m`;
  if (achievedIndex !== null && index >= achievedIndex) {
    const holdingTorque = currentDoc.meta.holding_torque ?? 0;
    // Both "hanging straight down" and "inverted straight up" are torque-
    // free (com collinear with gravity either way) -- sign(desired_H)
    // distinguishes them: desired_H = mass*g*height, and height < 0 means
    // the target sits BELOW the pivot (the true minimum-energy, stable
    // equilibrium), while height > 0 (straight up) is the maximum-energy,
    // unstable one -- the swing-up-to-inverted case, which never actually
    // takes "no effort" to hold in practice, just no STEADY torque.
    const holdingText =
      holdingTorque >= 1e-6
        ? `${holdingTorque.toFixed(3)} N&middot;m`
        : currentDoc.meta.desired_H < 0
          ? "no torque (stable equilibrium)"
          : "no torque (unstable equilibrium)";
    html += `<br><span class="done-message">Done! Achieved at t = ${currentDoc.frames.t[achievedIndex].toFixed(2)}s</span>`;
    html += `<br>Holding requires ${holdingText}`;
  } else if (settledIndex !== null && index >= settledIndex) {
    // No integral term in the controller, so a constant wind bias leaves a
    // permanent steady-state attitude error -- the pendulum genuinely comes
    // to rest (that's what triggered this branch), just not AT the angle
    // you dialed in, so this is worded as "settled", not "achieved".
    const offsetDeg = computeAttitudeErrorDeg(currentDoc.meta.target_R, currentDoc.frames.quaternion[index]);
    const holdingTorque = torqueMagnitudeSeries[index];
    html += `<br><span class="done-message">Settled with wind — holding ${holdingTorque.toFixed(3)} N&middot;m, ${offsetDeg.toFixed(1)}&deg; off-target</span>`;
  }
  metricsText.innerHTML = html;
}

function renderDoc(doc) {
  currentDoc = doc;

  clearScene(pendulumScene);
  pendulumGroup = createPendulumMesh(doc.geometry);
  pendulumScene.add(pendulumGroup);

  omegaSeries = computeOmega(doc.frames.M_body, doc.meta.I);
  omegaDomain = computeOmegaDomain(omegaSeries);

  energyLabel.textContent = doc.meta.mode === "controlled" ? "Current vs. Desired (H)" : "Energy (H)";

  torqueMagnitudeSeries = computeTorqueMagnitude(doc.frames.controller_torque);
  cumulativeWorkSeries = computeCumulativeWork(doc.frames.controller_torque, doc.frames.M_body, doc.meta.I, doc.frames.t);
  achievedIndex =
    doc.meta.mode === "controlled" && !doc.meta.wind_on
      ? computeAchievedIndexAttitude(doc.frames.quaternion, doc.meta.target_R, doc.frames.t)
      : null;
  // Wind means "achieved" (exactly at Rstar) is no longer the right question
  // -- see computeSettledIndex's docstring -- so instead detect "came to
  // rest" on its own, then report however far off-target that rest point is.
  settledIndex =
    doc.meta.mode === "controlled" && doc.meta.wind_on
      ? computeSettledIndex(doc.frames.M_body, doc.meta.I, doc.frames.t)
      : null;
  lastTorqueTextMs = null;

  inertiaReadout.innerHTML = formatInertia(doc.meta.I);

  playback = new Playback(doc.frames.t);
  previewingPose = false;
  updateFrame(0);
  playback.play();
  playPauseBtn.textContent = "Pause";
}

function runFromSidebar() {
  const controlOn = controlOnCheckbox.checked;
  const windOn = windOnCheckbox.checked;
  const geom = readGeometryFromSidebar();
  const geometry = buildPendulum(geom);

  const params = {
    ...geom,
    startingAngleDeg: Number(sliders.startingAngleDeg.value),
    frictionCoeff: Number(sliders.frictionCoeff.value),
    controlOn,
    windOn,
    windStd: Number(sliders.pendulumWindStd.value),
    T: DEFAULT_DURATION_SECONDS,
    N: Math.round(DEFAULT_DURATION_SECONDS * 100),
  };
  if (controlOn) {
    params.Rstar = targetRotationForAngle(Number(sliders.pendulumTargetAngleDeg.value), geometry.com);
  }

  const doc = buildPendulumScenario(params);
  renderDoc(doc);
}

let lastFrameMs = null;
let running = false;

/** See main.js's identical pause()/resume() -- same carousel contract. */
export function pause() {
  running = false;
}
export function resume() {
  if (running) return;
  running = true;
  lastFrameMs = null;
  requestAnimationFrame(animate);
}

function animate(nowMs) {
  if (!running) return;
  requestAnimationFrame(animate);

  const dt = lastFrameMs === null ? 0 : (nowMs - lastFrameMs) / 1000;
  lastFrameMs = nowMs;

  if (playback) {
    if (playback.playing) {
      const alive = playback.tick(dt);
      if (!alive) playPauseBtn.textContent = "Replay";
    }
    updateFrame(playback.frameIndexAtTime(playback.currentTime));
  }

  pendulumRenderer.render(pendulumScene, pendulumCamera);
}

playPauseBtn.addEventListener("click", () => {
  if (!playback) return;
  if (playback.playing) {
    playback.pause();
    playPauseBtn.textContent = "Play";
    return;
  }
  if (playPauseBtn.textContent === "Replay") {
    playback.seek(0);
  }
  previewingPose = false;
  playback.play();
  playPauseBtn.textContent = "Pause";
});

runButton.addEventListener("click", runFromSidebar);

resetButton.addEventListener("click", () => {
  if (!playback) return;
  playback.pause();
  playback.seek(0);
  playPauseBtn.textContent = "Play";
  previewingPose = false;
  updateFrame(playback.frameIndexAtTime(playback.currentTime));
});

resize();
previewPendulumGeometry();
runFromSidebar();

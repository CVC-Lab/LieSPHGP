/**
 * Windy cart-pole system: same file-per-system structure as main.js/
 * pendulumMain.js, but with a genuinely different animate() loop -- see
 * cartpoleScenarioRunner.js's docstring for why. There is no precomputed
 * trajectory to scrub with a Playback cursor here: "you" (manual/keyboard)
 * mode needs a force that depends on what the user does DURING the run, so
 * animate() itself steps the physics live, one tick per rendered frame,
 * appending to a rolling history buffer instead of indexing into a
 * fixed-length precomputed array.
 *
 * main.js's document-wide `.explain-switch` wiring already covers this
 * system's 4 Explain toggles too (see pendulumMain.js's identical note).
 */
import * as THREE from "three";

import {
  createCartPoleMesh,
  updateCartPoleFrame,
  resizePole,
  createTrackDecoration,
  updateTrackDecoration,
  cameraHalfExtents,
  CART_HEIGHT,
} from "./scenes/CartPoleScene.js";
import { computeCartPoleStateDomain, drawCartPoleStatePanel } from "./scenes/TimeSeriesPanel.js";
import { drawEnergyPanel } from "./scenes/ControlPanel.js";
import { drawTorquePanel } from "./scenes/ControlMetrics.js";
import { hamiltonian, velocities } from "./physics/cartpoleDynamics.js";
import { gainA, gainB, instabilityTime } from "./physics/cartpoleDiagnostics.js";
import {
  initialState,
  isTerminated,
  terminationReason,
  advance,
  initCtSacState,
  advanceCtSac,
} from "./physics/cartpoleScenarioRunner.js";
import { THEME } from "./theme.js";

// Deliberately NOT the Pole Length slider's max -- per explicit user
// request, this camera is framed for the CURRENT pole length, recomputed
// every time that slider changes (see updateCameraFraming below), not a
// fixed worst-case like every other camera in this app. margin=1.2 makes
// "cart + fully upright pole" fill exactly 1/1.2 = 83.3% of the frame,
// regardless of the actual geometry (see CartPoleScene.js's
// cameraHalfExtents docstring for why that fraction is margin-independent
// of the geometry values) -- comfortably matching "80% if not more."
const CAMERA_MARGIN = 1.2;
const HISTORY_SECONDS = 30; // rolling retention, independent of the 5s *display* window

// Widened from the canonical +-12deg (THETA_THRESHOLD_RADIANS -- ct_sac's
// actual training/task boundary, still what the angle-warning text above
// refers to) specifically so a failure past ct_sac's trained envelope is
// actually watchable on screen, rather than insta-terminating the instant
// the Starting Angle slider passes 12deg -- per explicit user request. This
// does NOT change THETA_THRESHOLD_RADIANS itself (still the accurate fact
// cited in the angle-warning text and POLICY_SPEC.md) -- only what THIS app
// treats as "run over," independent of ct_sac's own trained notion of
// success.
//
// Per-mode, not one shared number: "none" has no force at all, so nothing
// stops the pole from swinging anywhere within the camera's own safe range
// -- confirmed empirically (a free F=0 fall barely moves x at all, so
// there's no risk of the rail cutting the fall short first) -- so it gets
// the full 90deg, the actual limit of the camera's cos(theta)>=0 assumption
// (see CartPoleScene.js's cameraHalfExtents docstring), per explicit user
// request ("allow it to fall all the way... the full 90 degrees"). Manual
// and ct_sac keep the narrower 45deg from the previous round -- generous
// enough to watch ct_sac's failures play out (see DECISIONS.md) without
// wandering into the confusing full-rotation regime past 90deg.
const THETA_TERMINATION_RADIANS_BY_MODE = {
  none: (90 * Math.PI) / 180,
  manual: (45 * Math.PI) / 180,
  ctsac: (45 * Math.PI) / 180,
};

const cartpoleCanvas = document.getElementById("cartpole-canvas");
const energyCanvas = document.getElementById("cartpole-energy-canvas");
const energyCtx = energyCanvas.getContext("2d");
const thetaCanvas = document.getElementById("cartpole-theta-canvas");
const thetaCtx = thetaCanvas.getContext("2d");
const forceCanvas = document.getElementById("cartpole-force-canvas");
const forceCtx = forceCanvas.getContext("2d");

const timeReadout = document.getElementById("cartpole-time-readout");
const cartpoleLabel = document.getElementById("cartpole-label");
const metricsText = document.getElementById("cartpole-metrics-text");
const diagnosticsBlock = document.getElementById("cartpole-diagnostics");
const diagnosticsText = document.getElementById("cartpole-diagnostics-text");

// ── Sidebar elements ─────────────────────────────────────────────────────────
const sliderIds = [
  "poleMass", "cartMass", "poleLength", "gravity", "forceLimit", "railLength",
  "cartpoleStartingAngleDeg", "cartpoleStartingX", "sigmaGust", "sigmaTurb",
];
const sliders = Object.fromEntries(sliderIds.map((id) => [id, document.getElementById(id)]));
const sliderVals = Object.fromEntries(sliderIds.map((id) => [id, document.getElementById(`${id}-val`)]));
const windOnCheckbox = document.getElementById("cartpoleWindOn");
const windSubfields = document.getElementById("cartpole-wind-subfields");
const runButton = document.getElementById("cartpole-run-button");
const pauseButton = document.getElementById("cartpole-pause-button");
const resetButton = document.getElementById("cartpole-reset-button");

// ── Controller (None / Manual / ct_sac) ─────────────────────────────────────
// A discrete 3-way segmented control, not a boolean switch or a dropdown --
// see DECISIONS.md 2026-08-12: all 3 are always-valid modes worth showing at
// a glance, matching the coworker's own live demo's control layout.
const segButtons = Array.from(document.querySelectorAll("#cartpole-controller-segmented .seg-btn"));
const ctsacButton = document.getElementById("cartpole-ctsac-btn");
const ctsacLoadingNote = document.getElementById("cartpole-ctsac-loading");
const controllerSubfields = {
  none: document.getElementById("cartpole-none-subfields"),
  manual: document.getElementById("cartpole-manual-subfields"),
  ctsac: document.getElementById("cartpole-ctsac-subfields"),
};
let controllerMode = "none"; // the sidebar's CURRENT selection (updates immediately on click)
// The mode actually driving the LIVE run -- captured once, in resetRun(),
// from whatever controllerMode was selected AT THAT MOMENT. Deliberately
// separate from controllerMode: switching the Controller selector must
// have zero effect on a run already in progress (matching the racket/
// pendulum's Control toggle, which also only takes effect on the next Run)
// -- both because that's the behavior the user explicitly asked for, and
// because using the live `controllerMode` directly inside animate() would
// let switching to "ctsac" mid-run reach `advanceCtSac(ctSacState, ...)`
// while `ctSacState` is still null (only ever initialized in resetRun()).
let activeControllerMode = "none";
let ctSacPolicy = null;

// Same snapshot-at-resetRun() idea as activeControllerMode, for the same
// reason: dragging the Rail Length slider mid-run already live-updates the
// rendered track (see its own "input" listener) as a preview, exactly like
// Pole Length's `resizePole` -- but the FAILURE boundary an in-progress
// run actually gets checked against shouldn't silently change out from
// under it. Captured once per Reset/Run, used at both isTerminated call
// sites below.
let activeXThreshold = railHalfLengthFromSlider();

// Fetched once at load -- a real trained checkpoint (2.7MB/229,500 numbers,
// see DECISIONS.md 2026-08-12), not something to bundle into the JS module
// graph. `data/` is Vite's configured publicDir, so this is served at the
// site root regardless of dev or the deployed build. The button stays
// disabled until this resolves -- selecting a mode with no weights to run
// would silently do nothing, indistinguishable from broken.
fetch("/cartpole_policy.json")
  .then((r) => r.json())
  .then((data) => {
    ctSacPolicy = data;
    ctsacButton.disabled = false;
    ctsacLoadingNote.style.display = "none";
  })
  .catch(() => {
    ctsacLoadingNote.textContent = "Failed to load weights.";
  });

const DECIMALS_BY_SLIDER = {
  poleMass: 2, cartMass: 2, poleLength: 2, gravity: 1, forceLimit: 1, railLength: 1,
  cartpoleStartingAngleDeg: 0, cartpoleStartingX: 1, sigmaGust: 3, sigmaTurb: 4,
};
function refreshSliderLabels() {
  for (const id of sliderIds) {
    sliderVals[id].textContent = Number(sliders[id].value).toFixed(DECIMALS_BY_SLIDER[id] ?? 2);
  }
}
function refreshSubfieldVisibility() {
  for (const mode of Object.keys(controllerSubfields)) {
    controllerSubfields[mode].style.display = mode === controllerMode ? "flex" : "none";
  }
  windSubfields.style.display = windOnCheckbox.checked ? "flex" : "none";
  const controllerEngaged = controllerMode !== "none";
  diagnosticsBlock.style.display = controllerEngaged ? "block" : "none";
  metricsText.style.display = controllerEngaged ? "block" : "none";
}
for (const id of sliderIds) sliders[id].addEventListener("input", refreshSliderLabels);

/** ct_sac's own training data never has |theta| > 12deg (see POLICY_SPEC.md
 * -- "cos theta never leaves [0.9989, 1.0]... the network was trained that
 * way"), confirmed live: a closed-loop probe of the real checkpoint (no
 * termination cutoff, see DECISIONS.md) recovers cleanly up to ~15deg but
 * degrades into uncontrolled full rotations by 20deg -- a real cliff in the
 * trained weights, not a gradual struggle. This warning is a heads-up, not
 * a hard stop: it applies regardless of which Controller is selected (None
 * and Manual don't care about ct_sac's training distribution at all), since
 * the slider is shared across all three and the user may well switch to
 * ct_sac after setting it. */
const ANGLE_WARNING_THRESHOLD_DEG = 15;
const angleWarning = document.getElementById("cartpole-angle-warning");
function refreshAngleWarning() {
  const deg = Math.abs(Number(sliders.cartpoleStartingAngleDeg.value));
  angleWarning.style.display = deg > ANGLE_WARNING_THRESHOLD_DEG ? "block" : "none";
}
sliders.cartpoleStartingAngleDeg.addEventListener("input", refreshAngleWarning);
refreshAngleWarning();

// Selecting a mode only updates which mode WILL run next time you press
// Run -- matches the racket/pendulum's own Control toggle exactly (changing
// it has zero effect on whatever's already playing). An earlier version of
// this auto-restarted on every selection (reasoning: Free mode terminates
// almost instantly, so by the time you reach for the arrow keys after
// picking Manual, the page-load run had usually already frozen) -- reverted
// per explicit user request to match the other systems' convention instead.
function selectControllerMode(mode) {
  if (mode === controllerMode) return;
  controllerMode = mode;
  for (const btn of segButtons) btn.classList.toggle("active", btn.dataset.mode === mode);
  refreshSubfieldVisibility();
}
for (const btn of segButtons) {
  btn.addEventListener("click", () => selectControllerMode(btn.dataset.mode));
}
windOnCheckbox.addEventListener("change", refreshSubfieldVisibility);
refreshSliderLabels();
refreshSubfieldVisibility();

/** Sidebar exposes full Pole Length; the physics layer's own convention
 * (cartpoleDynamics.js's `l`, matching summer-2026/cartpole.py's `self.length`)
 * is HALF that -- converted right here, at the UI boundary, so the
 * already-tested physics functions never need to know the sidebar changed
 * units. */
function poleHalfLengthFromSlider() {
  return Number(sliders.poleLength.value) / 2;
}

/** Same full-length-in-the-UI/half-length-in-the-physics convention as
 * poleHalfLengthFromSlider -- X_THRESHOLD (cartpoleDynamics.js) is itself a
 * HALF-width (the cart fails once |x| exceeds it), so "Rail Length" in the
 * sidebar is the FULL track, matching how a real rail's length is normally
 * described, not "distance from center to one end." Default (4.8) is
 * exactly 2*X_THRESHOLD, so an untouched slider reproduces the canonical
 * +-2.4m rail ct_sac was actually trained on -- see index.html. */
function railHalfLengthFromSlider() {
  return Number(sliders.railLength.value) / 2;
}

function readParamsFromSidebar() {
  return {
    mp: Number(sliders.poleMass.value),
    mc: Number(sliders.cartMass.value),
    l: poleHalfLengthFromSlider(),
    g: Number(sliders.gravity.value),
    forceLimit: Number(sliders.forceLimit.value),
    sigmaGust: Number(sliders.sigmaGust.value),
    sigmaTurb: Number(sliders.sigmaTurb.value),
    windOn: windOnCheckbox.checked,
  };
}

/** Rebuilds just the pole's length live while dragging, mirroring the other
 * systems' geometry-slider live-preview pattern -- safe at any time (unlike
 * an orientation preview, this never fights a per-frame write, since
 * updateCartPoleFrame only ever touches position/rotation, never the pole
 * mesh's own geometry/length). */
sliders.poleLength.addEventListener("input", () => {
  resizePole(cartMesh, poleHalfLengthFromSlider());
  // Camera framing is sized for the CURRENT pole length (not the slider's
  // max -- see updateCameraFraming/CAMERA_MARGIN's own comments), so it
  // has to be recomputed on every change, not just once at load.
  updateCameraFraming();
  resize();
});

/** Live-previews the rail's own length while dragging, same pattern as
 * poleLength above -- the chase-cam (see updateCameraChase) never frames
 * the track itself, only the cart+pole, so unlike Pole Length this needs
 * no camera/resize follow-up at all, just the decoration mesh. */
sliders.railLength.addEventListener("input", () => {
  updateTrackDecoration(trackGroup, railHalfLengthFromSlider());
});

/** Live-previews the starting pose while paused -- mirrors the pendulum's
 * previewStartingOrientation. Skipped while actually running: animate()
 * already owns the mesh's position/rotation every frame in that case. */
function previewStartingPose() {
  if (physicsRunning) return;
  const thetaRad = (Number(sliders.cartpoleStartingAngleDeg.value) * Math.PI) / 180;
  const x0 = Number(sliders.cartpoleStartingX.value);
  updateCartPoleFrame(cartMesh, x0, thetaRad);
  updateCameraChase(x0);
}
sliders.cartpoleStartingAngleDeg.addEventListener("input", previewStartingPose);
sliders.cartpoleStartingX.addEventListener("input", previewStartingPose);

// ── Three.js setup ───────────────────────────────────────────────────────────
const cartpoleRenderer = new THREE.WebGLRenderer({ canvas: cartpoleCanvas, antialias: true });
cartpoleRenderer.setPixelRatio(window.devicePixelRatio);

const cartpoleScene = new THREE.Scene();
cartpoleScene.background = new THREE.Color(THEME.panelBg);
// Built at the slider's own current value (not the canonical X_THRESHOLD
// constant), matching every other geometry slider's live-preview
// convention -- see railHalfLengthFromSlider and its "input" listener
// above, which keeps this in sync afterward via updateTrackDecoration.
const trackGroup = createTrackDecoration(railHalfLengthFromSlider());
cartpoleScene.add(trackGroup);

const cartMesh = createCartPoleMesh({ poleHalfLength: poleHalfLengthFromSlider() });
cartpoleScene.add(cartMesh.group);

// Recomputed (not a fixed worst-case constant) every time the Pole Length
// slider changes -- see its own "input" listener above and CAMERA_MARGIN's
// comment for why.
let REQUIRED;
function updateCameraFraming() {
  REQUIRED = cameraHalfExtents({
    poleHalfLength: poleHalfLengthFromSlider(),
    margin: CAMERA_MARGIN,
  });
}
updateCameraFraming();
const CAMERA_DISTANCE = 10; // flat/orthographic -- distance only needs to clear the mesh, doesn't affect scale
const cartpoleCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
cartpoleCamera.position.set(0, CART_HEIGHT, CAMERA_DISTANCE);
cartpoleCamera.lookAt(0, CART_HEIGHT, 0);

/** Chase-cam: pans the camera to keep the cart horizontally centered, so
 * `cameraHalfExtents` only ever needs to frame the cart+pole themselves,
 * not the whole +-2.4m track (see that function's own docstring for why).
 * The track/boundary markers (createTrackDecoration) scroll past
 * naturally as the cart moves, per explicit user request ("I'm ok if we
 * have to scroll the background as the cart moves"). Called every frame
 * alongside updateCartPoleFrame, right after it, so the two never drift
 * out of sync. Only position.x changes -- y/z and the "look straight
 * down -z" orientation are set once above and never need revisiting,
 * since this is a pure horizontal pan, not a rotation. */
function updateCameraChase(x) {
  cartpoleCamera.position.x = x;
}

function resize2DCanvas(canvas, ctx) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resize() {
  const w = cartpoleCanvas.clientWidth;
  const h = cartpoleCanvas.clientHeight;
  cartpoleRenderer.setSize(w, h, false);

  // "Contain" fit, asymmetric: `bottom` (the cart's own headroom below the
  // pivot) is ALWAYS held fixed at REQUIRED.bottom, regardless of aspect
  // ratio -- this is what keeps the rail sitting at a consistent, low
  // position in the panel (see CartPoleScene.js's cameraHalfExtents
  // docstring) rather than drifting toward center. Any slack from the
  // canvas's own aspect ratio always goes to `top` (more headroom above
  // for the pole) or `halfWidth` (wider view of the track), never to
  // `bottom` -- a pure letterbox, never a crop, same guarantee as before,
  // just no longer symmetric.
  const canvasAspect = w / h;
  const requiredTotalHeight = REQUIRED.top + REQUIRED.bottom;
  const requiredAspect = (REQUIRED.halfWidth * 2) / requiredTotalHeight;
  const bottom = REQUIRED.bottom;
  let halfWidth, top;
  if (canvasAspect > requiredAspect) {
    // Canvas wider than needed -- height is the binding constraint.
    top = REQUIRED.top;
    halfWidth = ((top + bottom) * canvasAspect) / 2;
  } else {
    // Canvas narrower -- width is binding; all extra vertical room goes to `top`.
    halfWidth = REQUIRED.halfWidth;
    top = (halfWidth * 2) / canvasAspect - bottom;
  }
  cartpoleCamera.left = -halfWidth;
  cartpoleCamera.right = halfWidth;
  cartpoleCamera.top = top;
  cartpoleCamera.bottom = -bottom;
  cartpoleCamera.updateProjectionMatrix();

  resize2DCanvas(energyCanvas, energyCtx);
  resize2DCanvas(thetaCanvas, thetaCtx);
  resize2DCanvas(forceCanvas, forceCtx);
}
window.addEventListener("resize", resize);

// ── Live run state ───────────────────────────────────────────────────────────
let z = initialState({ x0: 0, thetaDeg: 5 });
// Only populated/read in ct_sac mode -- bundles z together with its
// observation window and real-time tick accumulator (see
// cartpoleScenarioRunner.js's initCtSacState/advanceCtSac). `z` above stays
// the single source of truth for rendering/history in EVERY mode; this is
// kept in sync with `ctSacState.z` after each ct_sac step rather than
// duplicating which one panels/mesh code should read.
let ctSacState = null;
let params = readParamsFromSidebar();
let physicsRunning = false;
let terminated = false;
let survivedSeconds = null;
let terminationCause = null; // "rail" | "angle" -- see terminationReason
let simTime = 0;

// Rolling history for the 3 chart panels -- unbounded arrays would leak
// memory over a long session, so history older than HISTORY_SECONDS is
// trimmed in batches (not every frame -- an O(n) splice every tick would be
// wasteful; trimming once the buffer is 2x over budget amortizes that cost).
let hist = { t: [], theta: [], thetaDot: [], H: [], forceMag: [], work: [] };
let cumulativeWork = 0;

// Arrow-key manual force -- only has any effect in Manual mode, and only
// while this system's own animate() loop is actually being scheduled
// (app.js's pause()/resume() already gates that; a keydown while a
// DIFFERENT slide is active just updates this harmless bit of state and is
// never read).
const keysHeld = { ArrowLeft: false, ArrowRight: false };
window.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  keysHeld[e.key] = true;
  if (controllerMode === "manual") e.preventDefault();
});
window.addEventListener("keyup", (e) => {
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") keysHeld[e.key] = false;
});

/** Manual-mode force only -- ct_sac's force comes from advanceCtSac's own
 * policy forward pass instead, and "none" applies nothing. Reads the
 * ACTIVE run's mode, not the sidebar's live selection -- see
 * activeControllerMode's own comment. */
function currentForce() {
  if (activeControllerMode !== "manual") return 0;
  const left = keysHeld.ArrowLeft ? -1 : 0;
  const right = keysHeld.ArrowRight ? 1 : 0;
  return (left + right) * params.forceLimit;
}

/** Cart velocity at the current state -- shared by pushHistory's theta_dot
 * readout and animate()'s work-rate integration, so the mass-matrix algebra
 * lives in exactly one place (cartpoleDynamics.velocities) rather than
 * being re-derived inline at each call site. */
function currentVelocities() {
  const [, theta, px, pth] = z;
  return velocities(theta, px, pth, params);
}

function pushHistory(F) {
  const [, theta] = z;
  const [, thetaDot] = currentVelocities();
  hist.t.push(simTime);
  hist.theta.push(theta);
  hist.thetaDot.push(thetaDot);
  hist.H.push(hamiltonian(z, params));
  hist.forceMag.push(Math.abs(F));
  hist.work.push(cumulativeWork);

  if (hist.t.length > 4000) {
    const dropCount = hist.t.length - 2000;
    for (const key of Object.keys(hist)) hist[key] = hist[key].slice(dropCount);
  }
}

/** "rail" -> the cart left the track; "angle" -> the pole tipped past this
 * mode's own threshold -- see terminationCause's own comment. Both branches
 * read the SAME snapshotted values (activeXThreshold,
 * THETA_TERMINATION_RADIANS_BY_MODE[activeControllerMode]) the actual
 * isTerminated check that ended the run used, not live slider reads, so
 * this can never describe a boundary other than the one that was actually
 * crossed. */
function formatTerminationCause() {
  if (terminationCause === "rail") {
    return `left the ±${activeXThreshold.toFixed(1)}m rail`;
  }
  const thetaDeg = (THETA_TERMINATION_RADIANS_BY_MODE[activeControllerMode] * 180) / Math.PI;
  return `pole exceeded ±${thetaDeg.toFixed(0)}°`;
}

function formatDiagnostics() {
  const gA = gainA(params).toFixed(2);
  const gB = gainB(params).toFixed(2);
  const tInstab = instabilityTime(params).toFixed(3);
  const survivedLine = terminated
    ? `<span class="done-message">Terminated — survived ${survivedSeconds.toFixed(2)}s</span><br>` +
      `<span class="done-message-reason">(${formatTerminationCause()})</span>`
    : `Survived: ${simTime.toFixed(2)}s (ongoing)`;
  return (
    `Instability Time: ${tInstab}s<br>` +
    `Gain A (wind→θ̇): ${gA} rad/s per unit p_θ<br>` +
    `Gain B (force→θ̇): ${gB} rad/s per unit p_x<br>` +
    `${survivedLine}`
  );
}

let lastMetricsTextMs = null;
function updateMetricsText() {
  const nowMs = performance.now();
  if (lastMetricsTextMs !== null && nowMs - lastMetricsTextMs < 500) return;
  lastMetricsTextMs = nowMs;
  const currentForceMag = hist.forceMag.length ? hist.forceMag[hist.forceMag.length - 1] : 0;
  const workSoFar = hist.work.length ? hist.work[hist.work.length - 1] : 0;
  metricsText.innerHTML = `Work: ${workSoFar.toFixed(3)} J<br>Force: ${currentForceMag.toFixed(3)} N`;
  diagnosticsText.innerHTML = formatDiagnostics();
}

function drawPanels() {
  const n = hist.t.length;
  if (n === 0) return;
  const currentIndex = n - 1;

  const domain = computeCartPoleStateDomain(hist.theta, hist.thetaDot);
  drawCartPoleStatePanel(thetaCtx, {
    t: hist.t, theta: hist.theta, thetaDot: hist.thetaDot, domain, currentIndex,
    width: thetaCanvas.clientWidth, height: thetaCanvas.clientHeight,
  });

  const references = [
    { value: params.mp * params.g * params.l, color: THEME.unstable, label: "H(Upright)" },
    { value: -params.mp * params.g * params.l, color: THEME.stable, label: "H(Hanging)" },
  ];
  drawEnergyPanel(energyCtx, {
    t: hist.t, H: hist.H, references, currentIndex,
    width: energyCanvas.clientWidth, height: energyCanvas.clientHeight,
  });

  // Always drawn now, regardless of mode -- a genuinely-zero force (e.g.
  // "None" mode, or "Manual"/"ct_sac" before anything has actually pushed
  // yet) is still real, meaningful data (computeTorqueDomain's fallback
  // range handles the all-zero case correctly), not something to hide.
  // Blanking this out entirely for "None" (an earlier version did) read as
  // a bug -- a panel with literally nothing in it, not even axes -- rather
  // than the intended "nothing is pushing right now."
  drawTorquePanel(forceCtx, {
    t: hist.t, torqueMagnitude: hist.forceMag, currentIndex,
    width: forceCanvas.clientWidth, height: forceCanvas.clientHeight,
    withAxes: true, yLabel: "Force (N)",
  });
  if (activeControllerMode !== "none") updateMetricsText();
}

function resetRun() {
  // Capture the sidebar's current selection as the mode that will actually
  // drive this run -- see activeControllerMode's own comment for why this
  // must be a snapshot, not a live read of controllerMode.
  activeControllerMode = controllerMode;
  activeXThreshold = railHalfLengthFromSlider();
  params = readParamsFromSidebar();
  const startOpts = {
    x0: Number(sliders.cartpoleStartingX.value),
    thetaDeg: Number(sliders.cartpoleStartingAngleDeg.value),
  };
  if (activeControllerMode === "ctsac") {
    ctSacState = initCtSacState(startOpts);
    z = ctSacState.z;
  } else {
    ctSacState = null;
    z = initialState(startOpts);
  }
  simTime = 0;
  terminated = false;
  survivedSeconds = null;
  terminationCause = null;
  cumulativeWork = 0;
  lastMetricsTextMs = null;
  hist = { t: [], theta: [], thetaDot: [], H: [], forceMag: [], work: [] };
  pushHistory(0);
  updateCartPoleFrame(cartMesh, z[0], z[1]);
  updateCameraChase(z[0]);
  cartpoleLabel.textContent = { none: "Free", manual: "Driving", ctsac: "ct_sac" }[activeControllerMode];
  drawPanels();
}

function runFromSidebar() {
  resetRun();
  physicsRunning = true;
  pauseButton.textContent = "Pause";
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

  const dtReal = lastFrameMs === null ? 0 : Math.min((nowMs - lastFrameMs) / 1000, 0.1);
  lastFrameMs = nowMs;

  if (physicsRunning && !terminated && dtReal > 0) {
    let F;
    if (activeControllerMode === "ctsac") {
      // Fixed-cadence path: advanceCtSac owns its own dt=0.01 ticking
      // (see cartpoleScenarioRunner.js) -- 0, 1, or several ticks may run
      // for a given real dtReal, each with its own policy forward pass.
      ctSacState = advanceCtSac(
        ctSacState,
        dtReal,
        Math.random,
        params,
        ctSacPolicy,
        THETA_TERMINATION_RADIANS_BY_MODE.ctsac,
        activeXThreshold
      );
      z = ctSacState.z;
      F = ctSacState.force;
      cumulativeWork += ctSacState.workDelta;
    } else {
      // Continuous real-time path: "none" (F always 0) and "manual"
      // (F from arrow keys) share this, since both are known instantaneously
      // rather than needing a fixed decision cadence.
      F = currentForce();
      // Power delivered THIS tick = force * cart velocity (dH/dt = F*x_dot,
      // the same work-rate identity cartpole_dynamics.py's own test suite
      // verifies) -- measured at the state BEFORE stepping, consistent with
      // a forward-Euler power estimate over the tick.
      const [xDotBefore] = currentVelocities();
      cumulativeWork += F * xDotBefore * dtReal;
      z = advance(z, F, dtReal, Math.random, params);
    }
    simTime += dtReal;

    const thetaThresholdNow = THETA_TERMINATION_RADIANS_BY_MODE[activeControllerMode];
    if (isTerminated(z, thetaThresholdNow, activeXThreshold)) {
      terminated = true;
      survivedSeconds = simTime;
      terminationCause = terminationReason(z, thetaThresholdNow, activeXThreshold);
    }

    pushHistory(F);
    updateCartPoleFrame(cartMesh, z[0], z[1]);
    updateCameraChase(z[0]);
  }

  timeReadout.textContent = `t = ${simTime.toFixed(1)}s`;
  drawPanels();
  cartpoleRenderer.render(cartpoleScene, cartpoleCamera);
}

pauseButton.addEventListener("click", () => {
  physicsRunning = !physicsRunning;
  pauseButton.textContent = physicsRunning ? "Pause" : "Play";
});

runButton.addEventListener("click", runFromSidebar);

resetButton.addEventListener("click", () => {
  physicsRunning = false;
  pauseButton.textContent = "Play";
  resetRun();
});

resize();
runFromSidebar();

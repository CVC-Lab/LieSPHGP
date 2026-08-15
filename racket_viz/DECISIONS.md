# Decisions log

## 2026-07-24 — Project kickoff, architecture, test-first scaffold

**Context.** Building a reusable four-panel Dzhanibekov-effect visualization
(racket 3D, Casimir sphere, angular velocity vs. time, current-vs-desired
control), styled after a greeting-card physics illustration. A prior attempt on
branch `feature/tennis-racket-visualizer` stalled; going slower and more
deliberately this time.

**Why the prior attempt likely failed.** It ran a live FastAPI + WebSocket
backend doing RK4 integration every frame, coupled to a loaded PyTorch NN
checkpoint (for an H_nn-vs-H_analytical comparison) and a full IDA-PBC
controller — a live research instrument and a public visualization at once. Left
untouched on its branch as historical reference; not deleted or merged.

**Decisions made:**
- Location: `LieSPHGP/racket_viz/`, new branch `feature/racket-viz` off `main`.
- Architecture: Python precomputes all trajectory/phase-portrait data once to
  static JSON; Three.js only loads and replays it. No live server, no
  WebSocket, no NN-checkpoint coupling at runtime.
- Tooling: pytest (physics), Vite + Vitest (app).
- Panel 4 reuses the *already-validated* analytic controllers from LieSPHGP's
  open PR #1 (`tennis-racket-effect` branch, `controller_stageD.py` /
  `controller_stageF.py`) rather than anything NN-based. Toggling "control on"
  + a target axis/alignment selects between a small number of *precomputed*
  controlled scenarios — no live simulation needed.
- Energy/angular-momentum conservation is a hard invariant only for **free**
  (torque-free) scenarios. Controlled scenarios must show `H(t)`/`|M(t)|`
  actually changing (the yellow dot leaving the initial Casimir sphere as the
  racket physically flips to the new target configuration) — this is flagged as
  the main new engineering work, not just glue code, and has its own explicit
  tests (`test_controlled_energy_is_not_conserved`,
  `test_controlled_angular_momentum_changes`) precisely so a bug that decouples
  the controller from the integrator doesn't silently look like the free case.
- Caption text is authored fresh per scenario/segment, not copied from
  `dzhanibekov_display.py`'s hardcoded `CAPTIONS_LEFT`/`CAPTIONS_RIGHT`.

**Process for this stage:** proposed the test list + architecture + docs plan,
got it approved, then wrote the full test suite (physics: 28 pytest cases across
5 files; app: 17 Vitest cases across 5 files) against stub modules that raise
`NotImplementedError` — confirmed red phase (all failing for the right reason,
no import/collection errors) before writing any real implementation. See
`ARCHITECTURE.md` for the data contract these tests assume.

**Known caveat in the current test suite**: a few `DataLoader.test.js` cases
assert `expect(() => validateScenario(bad)).toThrow()` — against the current
stub (which throws unconditionally) these pass "for free," without yet proving
the *right* validation logic fires. They'll need re-checking once
`validateScenario` is implemented (e.g. asserting on the error message) to make
sure they're still testing what they claim to.

**Next step:** implement `physics/` module bodies until `pytest` is green, then
`app/src/core` + `app/src/scenes` until `npm test` is green — in that order,
before any visual/styling work.

## 2026-07-24 — Physics implementation, `pytest` green (28/28)

Implemented `rigid_body.py`, `racket_geometry.py`, `phase_portrait.py`,
`controllers.py`, `scenarios.py`, `export.py` in that order (each was the direct
extraction/adaptation described in `ARCHITECTURE.md`). Two real issues surfaced
along the way, both fixed in the test file (not worked around in the
implementation) since the implementation was verified correct independently:

1. **`np.testing.assert_allclose` broadcasting quirk** (numpy 2.4.6): comparing
   a `(N, 3)` array against a `(3,)` array reports "not equal" even when every
   row is exactly identical — confirmed in isolation with a synthetic example,
   not a translation-layer bug. Fixed by using `np.allclose` (which broadcasts
   correctly) in `test_fixed_points_are_stationary` instead.

2. **The Dzhanibekov-flip test's original IC didn't actually flip** within any
   reasonable time. Root cause: the saddle at the unstable (imid) fixed point
   has a specific growing eigendirection — a fixed *ratio* between the
   perturbations on the two stable axes (≈11.6:1 for this geometry, derived
   from `sqrt((1/I_max−1/I_mid)/(1/I_mid−1/I_min))`), not a 1:1 split. The
   original test perturbed both stable axes by an equal amount *in M-space*,
   which landed almost exactly on the *decaying* eigendirection instead of the
   growing one, so the flip would technically still happen but on an
   impractically long timescale. Verified physically first (checked that a pure
   single-axis spin-up on each of the 3 axes shows axis 1 — the intermediate
   moment — growing perturbations by a factor of ~5×10⁴ over 20s while axes 0
   and 2 stay bounded, confirming the ODE implementation itself was correct)
   before changing the test's IC to an equal-in-*omega* perturbation, which
   generically avoids landing on the decaying eigendirection and flips within
   ~1s.

3. **Stage D/F convergence numbers in the test file were placeholders** copied
   from the LieSPHGP memory notes, which used a completely different inertia
   scale (I ~ 0.006–0.013 there vs. this project's I ~ 0.075–0.95). Recalibrated
   numerically against this project's actual inertia: Stage D needs Kp=1.0 over
   T=6s (not Kp=0.10/T=0.30s) to converge to <1% error here; Stage F needs
   K_R=K_p=1.0 over T=8s. Both now documented inline in `test_controllers.py` as
   `STAGE_D_KP`/`STAGE_D_T` constants with a comment explaining the mismatch, so
   a future reader doesn't mistake them for copy-paste of the PR's numbers.

`python export.py` runs end-to-end and produces real scenario JSON in `data/`
(free narrative: 800 frames, 6 segments, 8 background trajectories + 8
separatrices; controlled scenarios include `desired_H`/`desired_L`). Confirmed
the default racket geometry's inertia (`I ≈ [0.075, 0.875, 0.950]`) happens to
closely match the arbitrary `I_DEFAULT` used across the rigid_body/phase_portrait/
controllers test files — those were chosen independently (to loosely match the
whiteboard sketch's axis labels), so this is a coincidence worth knowing about,
not a hidden coupling between the test files and `racket_geometry.py`.

**Visual sanity check (throwaway, not committed):** before starting on the JS
renderer, rendered the actual exported JSON (not a separate hand-computed case)
with a matplotlib script in the same style as `dzhanibekov_display.py`, to catch
anything the numeric tests couldn't see (e.g. a quaternion sign/handedness bug
that passes every scalar invariant but looks wrong). Confirmed: the racket's
face color visibly flips red→green across "The flip" segment in the free
scenario, in lockstep with the yellow dot tracing the orange separatrix on the
sphere; and in the controlled-to-imax scenario, the racket visibly flips and the
dot converges exactly onto the target (green) fixed point by the end. No bugs
found — gives confidence the remaining work is a rendering-correctness problem
(quaternion convention, camera, styling), not a physics problem.

**Next step:** implement `app/src/core` (`DataLoader.js`, `Playback.js`) and
`app/src/scenes` (`RacketScene.js`, `SphereScene.js`) until `npm test` is green,
re-checking the `DataLoader.test.js` "should throw" cases flagged above along
the way. Still no visual/styling work.

## 2026-07-24 — App layer implementation, `npm test` green (21/21)

Implemented `DataLoader.js`, `Playback.js`, `RacketScene.js::quaternionToMatrix`,
`SphereScene.js::dotPosition` in that order, mirroring the physics-side
implementation order (data validation → playback state → rendering math).

- `DataLoader.validateScenario` mirrors `physics/export.py::scenario_to_json`'s
  checks (same required keys, same length-consistency logic) so both sides
  reject the same malformed shapes. Tightened the `DataLoader.test.js` "should
  throw" cases flagged as a caveat earlier today to assert on the actual error
  message (e.g. `/missing required keys.*R_cas/`), not just "throws something"
  — the earlier version would have passed against any stub. Added one more
  case (`accepts a valid controlled scenario with desired_H/desired_L
  present`) since the negative cases alone didn't prove the positive path for
  `mode: "controlled"` was reachable at all.
- `quaternionToMatrix` reindexes our scalar-first `[w,x,y,z]` convention into
  THREE.Quaternion's scalar-last `(x,y,z,w)` constructor, then lets THREE build
  the rotation matrix — passed all 6 cross-check cases against the Python
  `quat_to_R` fixture on the first try, including the two "hard" non-axis-
  aligned cases (120° about (1,1,1), an unnormalized input). No handedness or
  scalar-order bug.
- `Playback` implements `frameIndexAtTime` as a "last index ≤ t" binary search,
  and `tick()` returns `false` (and clamps, stops) once `currentTime` reaches
  the last frame — matches the free scenario's "plays once then stops" intent
  from `dzhanibekov_display.py` and works unchanged for a controlled scenario's
  frames too (nothing here depends on whether the scenario conserves energy).

**Still not done:** no wiring into `main.js`, no actual `THREE.Scene`/renderer/
camera, no visible output yet — `index.html` still just shows placeholder text.
Skipped a browser preview this round since there's nothing new to observe there
yet; the app becomes visually checkable once the scenes are wired into a real
render loop.

## 2026-07-24 — Render loop wired up, verified in-browser across all 4 scenarios

Before this could be wired up, found a real gap: the exported scenario JSON had
no static racket *mesh* (only per-frame quaternions), so Three.js had nothing to
build a shape from. Fixed by adding a `geometry` block to the schema
(`verts_body`/`handle_idx`/`hoop_idx`/`face_thickness`/`face_normal_body`) —
already fully resolved in the same body frame `frames.quaternion` rotates, so
the app never needs to know the racket's raw construction parameters or repeat
the inertia-tensor/eigenvector computation. Added
`racket_geometry.py::build_racket`'s `face_normal_body` (`evecs.T @ [0,0,1]`)
along with two regression tests proving it lands *exactly* on the imax axis
(not approximately) — guaranteed by the perpendicular-axis theorem since the
racket is planar. Propagated through `scenarios.py`, `export.py` (validated +
serialized like every other required key), `DataLoader.js` (mirrored
validation), and the JS sample fixture. Physics: 31/31 pytest. JS: 22/22 Vitest.

Built `RacketScene.createRacketMesh`/`updateRacketOrientation` and
`SphereScene.createSphereScene`/`updateDot`, and wired everything into
`main.js`: a scenario selector, play/pause, and two `THREE.WebGLRenderer`
panels sharing one `Playback` clock via `requestAnimationFrame`. Key
implementation choice: the racket's two offset faces are built *once* in
body-frame local space (`vertex ± face_thickness * face_normal_body`) rather
than recomputed in world space every frame like the matplotlib reference does —
valid because rotation commutes with a body-fixed offset (`R(v+tn) = Rv+tRn`),
and much cheaper (one `group.quaternion` assignment per frame instead of
rebuilding geometry).

Verified live in the Browser pane against all four generated scenarios:
- **Free narrative**: face color visibly flips red→green during "The flip",
  dot traces the separatrix in sync — matches the earlier matplotlib check.
- **Controlled→imax**: dot starts at the unstable (red) point, ends immediately
  adjacent to the target (green) point; racket ends in a fast clean spin.
- **Alignment** (ω\*=0, R\*=identity): the subtlest correctness check available
  — since the target angular momentum is exactly zero, the dot must converge
  not to a point *on* the sphere but to its exact *center*. It does.

One tool quirk, not an app bug: `computer.left_click` on the Play button
sometimes didn't register through the Browser pane despite correct coordinates
(confirmed via `getBoundingClientRect`); dispatching the click via
`element.click()` in `javascript_tool` worked every time. Noting in case it
recurs during later UI work.

Added a `racket-viz-app` entry to `LieSPHGP/.claude/launch.json` (alongside the
untouched old `tennis-racket-visualizer` entry) for `npm run dev` in
`racket_viz/app`; also set `vite.config.js`'s `publicDir: "../data"` so the
generated scenario JSON is served directly with no copying/duplication.

**Next step:** panels 3 (angular velocity vs. time) and 4 (current-vs-desired
control) are still unbuilt — currently a 2-panel app, not 4. Still no
layout/styling/design pass; that was always meant to come last, once the
simulation itself was verified correct, which it now is for panels 1 and 2.

## 2026-07-24 — Panel 3 (angular velocity vs. time)

Built `TimeSeriesPanel.js`: `computeOmega`/`computeOmegaDomain` as pure,
tested functions (3 new Vitest cases), plus an untested `drawTimeSeries` canvas
renderer — same split as `RacketScene`/`SphereScene` (test the math, verify the
drawing visually), since canvas/WebGL output isn't meaningfully unit-testable
in jsdom.

This panel is plain HTML5 `<canvas>` 2D, not Three.js — a 2D line chart doesn't
need a 3D scene, and no charting library was part of the approved architecture,
so it's ~100 lines of direct canvas drawing instead of a new dependency.

Color convention: ω[imin] green, ω[imid] red, ω[imax] royalblue — reusing the
sphere panel's existing stable(green)/unstable(red) fixed-point colors rather
than inventing a new palette, and matching the whiteboard sketch's ω1=green/
ω2=red/ω3=blue legend (ω2 = imid is always the unstable axis in this codebase's
convention, so "red" means the same thing in both panels).

Wired into `main.js` as a third canvas + panel; the omega series/domain are
computed once per scenario load (not per frame), and each frame draws the
three lines, a vertical time cursor, and a small marker per line at the current
value — mirroring the "yellow = current state" convention already used for the
sphere's moving dot. Verified live: cursor and markers track correctly in sync
with the racket flip and the sphere dot across the free-narrative scenario.

Physics: 31/31 pytest (unchanged). JS: 25/25 Vitest (+3).

**Next step:** panel 4 (current-vs-desired control) is the last panel —
requires deciding how to present "current H(t)/|M(t)| vs. desired_H/desired_L"
for controlled scenarios (and presumably nothing / a placeholder for free
scenarios, which have no target). After that: 4-panel grid layout, and only
then the design/styling pass explicitly deferred since the start.

## 2026-07-24 — Panel 4 (current vs. desired) — the 4th and final panel

User's answer to the "what does panel 4 show for free scenarios?" question:
unify around always plotting the current `H(t)`, varying only which reference
lines get drawn against it:
- **free**: 3 dashed lines at the natural equilibrium energies (`H_axis`),
  reusing panel 3's green(imin)/red(imid=separatrix)/blue(imax) colors — shows
  which energy band the conserved H(t) sits in.
- **controlled**: 1 dashed line at `desired_H` — a literal current-vs-desired
  convergence plot, directly answering the user's question about whether this
  should already show the IDA-PBC target (yes, via the already-exported
  `desired_H`).

This required adding `H_axis` (the 3 equilibrium energies) to free scenarios'
exported meta, mirroring `desired_H`/`desired_L`'s existing treatment for
controlled ones — validated in both `export.py` and `DataLoader.js` the same
way (required key, checked length), with matching pytest/Vitest coverage and a
new `test_export_rejects_free_scenario_missing_H_axis`.

Built `ControlPanel.js`: `computeH`/`referencesForScenario` as pure, tested
functions (4 new Vitest cases) + an untested `drawEnergyPanel` canvas renderer
(same test/verify split as every other scene module). Switched `#panels` from
a 1-row flex to a 2x2 CSS grid now that all 4 panels exist, arranged to match
the whiteboard sketch's layout (control top-left, racket top-right, sphere
bottom-left, angular velocity bottom-right).

Verified live for both modes:
- **Free narrative**: H(t) starts at `H_axis[imin]` (the first segment's
  energy), steps down for the remaining segments — all near `H_axis[imid]`/
  `H_axis[imax]`, correctly clustered near the bottom reference lines since
  only the imin-axis segment has much higher energy on this scale.
- **Controlled→imax**: H(t) dips first (redirecting the spin costs energy
  before the target axis picks up speed) then converges exactly onto the
  `desired_H` line, in sync with panel 3's ω-components crossing over and the
  sphere dot approaching the target fixed point. This is the clearest possible
  demonstration that all 4 panels are telling one physically consistent story.

Physics: 32/32 pytest (+1). JS: 30/30 Vitest (+5).

**All 4 panels now exist and are verified correct.** Nothing left on the
"physically accurate" side of the original brief. Remaining work is the
explicitly-deferred part: layout/visual design/styling (the greeting-card
aesthetic), and — if there's time — the "swipe to another system" idea from
the original request.

## 2026-07-28 — Interactive parameter sidebar + JS physics port

**Context.** User wants a real control surface (mass/geometry sliders,
start/end axis, control on/off, wind on/off, duration) with a "Run" button —
matching `summer-2026/rotate.py`'s interactive spirit plus the LieSPHGP
Stage D/E/F control/wind extensions. This breaks the "Python precomputes once,
JS replays a static file" assumption, since physics now has to run for values
nobody precomputed ahead of time.

**Decisions, discussed and confirmed before writing any code** (see the
approved plan for full reasoning):
- **Physics ported to JavaScript** — not Pyodide/WASM (keeps single-source
  Python but costs a large download), not a backend (zero-rewrite but
  reintroduces the server process this project deliberately avoided). The
  existing `physics/` Python package is kept as the frozen, already-tested
  oracle the JS port is cross-validated against — same fixture-based pattern
  already used for `quaternionToMatrix`, now extended.
- **Wind** = LieSPHGP Stage E's actual `disturbance_torque_std` model: one
  random body-frame torque sampled *once* per run and held constant (verified
  by reading `envs/tennis_racket_3d.py` on the `tennis-racket-effect` branch
  directly, not assumed) — not a per-step stochastic process. Confirmed to
  work in both free and controlled modes.
- **Start/end position** = axis pickers (imin/imid/imax), not free-form
  orientation.
- **Presets pre-fill sliders**, they don't fork the code path — a preset
  click and a manual slider tweak both end up calling the exact same
  `buildScenario(params)`.
- **"Run" produces one continuous trajectory**, not the old 6-segment
  bookended narrative. A real simplification (loses the per-phase captions),
  traded for genuine parameterizability.

**Implementation, module-for-module mirroring `physics/`** (eigen3x3 →
rigidBody → racketGeometry → phasePortrait → controllers → wind →
scenarioRunner, same dependency order as the original Python build): every
module's tests passed on the **first implementation attempt** — a very
different experience from the original Python build, and worth noting why:
the hard physics questions (the flip IC's growing-eigendirection subtlety, the
Stage D/F gain calibration, the numpy `assert_allclose` broadcast quirk) were
already found and fixed once, in Python, and this port could just reuse those
answers instead of rediscovering them. `eigen3x3.js` (no Python line to
translate — no JS equivalent of `numpy.linalg.eigh`) also passed immediately,
including its near-degenerate-eigenvalue test case.

Physics: 32/32 pytest (+1, `test_export_rejects_free_scenario_missing_H_axis`-style
coverage extended). JS: 74/74 Vitest (+41 new: eigen3x3, rigidBody,
racketGeometry, phasePortrait, controllers, wind, scenarioRunner).

**One real bug, found in the browser, not by any test**: `scenarioRunner.js`
returned `phasePortrait.buildStaticCurves()`'s camelCase output
(`stableFixedPoints`) unrenamed into the `background` field, but the schema
(and `SphereScene.js`) expect snake_case (`stable_fixed_points`). Every unit
test passed anyway because `DataLoader.validateScenario` never actually
checked `background`'s shape — only `meta`/`frames`/`geometry`. The failure
was silent (no console error visible via the automated browser tools; found
by manually chaining `buildScenario` → `validateScenario` →
`createSphereScene` step-by-step in the page console until one threw). Fixed
in two places: the actual key-renaming bug in `scenarioRunner.js`, and the
real gap in `DataLoader.validateScenario` (now validates `background`'s keys
and fixed-point shapes too) — plus a regression test for each, one asserting
the renamed keys directly and one asserting `validateScenario` now catches
this exact mistake if it recurs.

**Verified live** across combinations that had never been exercised before:
free+wind (confirmed no errors, first-ever run of this combination in either
language), a custom non-preset geometry/mass combination (still produces a
correct periodic Dzhanibekov flip), and the Control→imax preset (full `H(t)`
convergence onto `desired_H`, `ω` crossover, dot reaching the target fixed
point — all in sync, same as the earlier precomputed-scenario verification).

**Next step:** the app is now feature-complete for "physically accurate and
parameterizable." Only the explicitly-deferred design/styling pass (and the
optional "swipe to another system" stretch goal) remain.

## 2026-07-28 — First hands-on feedback round: 8 fixes, no styling yet

User tried the sidebar directly and reported 8 issues/requests, discussed one
at a time before any code changed (per their explicit request to "talk it out
first" on the ambiguous ones):

1. **Jagged/fuzzy Casimir sphere curves** (sphere wireframe itself was fine,
   explicitly keep that). Root cause: background trajectory/separatrix curves
   sit at exactly `|M|/R_cas == 1`, the same radius as the sphere mesh —
   classic WebGL z-fighting (coincident depth), not a data/integration
   problem. Fixed by rendering curves at `1.006x` the sphere's radius
   (`SphereScene.js`). Confirmed visually: curves are now clean thin
   anti-aliased lines; remaining visual density is legitimately from multiple
   overlapping sign-variant curves (same design as `dzhanibekov_display.py`),
   not an artifact.
2. **Dot "rides the separatrix down and up again."** Real bug, not a taste
   issue: `scenarioRunner.js`'s flip perturbation added a fixed `eps=0.02` in
   raw omega-space, symmetric on both stable axes — dimensionally disconnected
   from `R_cas` (order-of-magnitude too close to the exact separatrix for
   typical geometries) *and* structurally different from
   `dzhanibekov_display.py`'s actual construction (an M-space kick, dominant
   on ONE stable axis, tiny on the other — not symmetric). Rewrote to match
   that construction with `NEAR_SEP_PERTURB = 0.02` (fraction of `R_cas`) —
   between the too-close-to-tell-apart previous behavior and the reference
   script's `0.05`, per the user's "happy medium" ask.
3. **Control on, start==end axis → looks broken.** Kept the behavior (holding
   steady is correct), added an explicit caption ("Already at target — holding
   steady, nothing to correct") instead of hiding or blocking the choice.
4. **"Shouldn't H be an ellipsoid?"** — untangled a real conflation: the
   energy *level set* `H(M)=const` in 3D M-space is genuinely an ellipsoid,
   already implicit in panel 2 (the curves are its intersection with the
   Casimir sphere); panel 4's `H(t)` is a different, correctly-flat-for-free-
   motion scalar trace. Panel 4 unchanged. Added a new opt-in "Hamiltonian 3D"
   toggle in panel 2 instead (`SphereScene.createEnergyEllipsoid` /
   `updateEnergyEllipsoid`, semi-axes `sqrt(2*H*I_k)/R_cas`, rescaled per
   frame) — off by default since the user correctly predicted it'd get
   visually busy over time.
5. **Live slider preview + real line thickness**, both without a server:
   - `previewRacketGeometry()` in `main.js` rebuilds just `buildRacket()` and
     swaps the mesh on `input` events for handle/hoop/width/hoop-fraction
     sliders (not total mass — confirmed it cancels out of the COM math
     entirely and never touches vertex positions). No simulation involved.
   - Handle/hoop switched from `THREE.Line`/`LineLoop` to `TubeGeometry`
     meshes: `LineBasicMaterial.linewidth` is silently ignored by nearly
     every browser/GPU (long-standing WebGL spec quirk), so the old `2`
     genuinely did nothing. Purely a rendering change — physics still treats
     the racket as idealized 1D curves regardless of drawn thickness.

Also two smaller items folded in along the way:
- **A**: panels reflected across the vertical axis per the user's request —
  racket top-left, panel 4 top-right, angular velocity bottom-left, Casimir
  sphere bottom-right (previously the mirror image of this).
- **B**: consolidated "Control" from two places (preset button labels +
  a checkbox/dropdown pair lower down) into one real toggle-switch UI element
  (custom CSS, matching the existing Wind toggle's pattern exactly): Ending
  axis now hides/shows with the Control switch; Starting axis stays always
  visible (confirmed with the user — it's meaningful for free spin too, e.g.
  the Free flip preset has no controller at all but still needs a start axis).
- **Run now auto-plays** (previously left the user needing to separately hit
  Play after Run, which read as "nothing happened" — user flagged this
  directly): `renderDoc()` calls `playback.play()` immediately.

Physics: 32/32 pytest (unchanged — none of this touched `physics/`). JS:
74/74 Vitest (unchanged count; these were all rendering/UI/tuning changes, not
new pure-function surface, so no new unit tests were added for this batch —
verified each visually in the browser instead, including a debug-console
sanity check of the ellipsoid's actual scale values before confirming it
render correctly).

**Explicitly still deferred**: labels and general aesthetic polish — the user
reiterated they want the functional issues resolved first.

## 2026-07-29 — Presets/duration removed, Stop button, control readouts, rolling time windows

**Presets and Duration temporarily removed.** The user found the 4 preset
buttons unintuitive as a UI element (not obvious what each one *meant*) and
the Duration slider pointless once runs can just be stopped manually. Both
are commented out, not deleted, in `index.html` and `main.js` (with comments
explaining why), so they're easy to restore later. Runs now default to a
long fixed `DEFAULT_DURATION_SECONDS = 300`; a new **Stop** button
(pause + seek to t=0) sits below **Run**, distinct from **Pause** (freeze in
place). This was also motivated by wanting to exercise long integrations,
which is how the RK4 drift bug below (deferred, not yet fixed) was found.

**New: information back to the user for controlled runs.** Added to
`app/src/scenes/ControlMetrics.js`:
- `isAchievable(meta)` — an "achieved target" flag is only well-posed for
  controlled + no-wind + stable-target-axis scenarios (wind never lets
  corrective torque relax to zero; holding the unstable imid axis against a
  perturbation doesn't either). Everything else gets an ongoing effort
  readout only, never a "Done!" flag — confirmed explicitly with the user.
- `computeCumulativeWork` — running trapezoidal integral of controller
  torque · ω. Rectangle rule was tried first but failed a real cross-check
  test (`work(t) ≈ H(t)-H(0)` for a no-wind controlled run, since dH/dt =
  torque·ω exactly) by 0.063 vs. a required 0.05 precision — switched to
  trapezoidal, test passed.
- `computeAchievedIndex` — sustained-tolerance detection (must stay within
  5% of target for ≥1s, not just pass through momentarily).
- A small sidebar "Control function" panel (`drawTorquePanel`) plots torque
  magnitude vs. time with the same rolling window as the other time-series
  panels, plus a live "work done thus far" / "current torque" text readout
  (throttled to update every 500ms via `performance.now()`, not every
  animation frame) and the green "Done!" message when applicable.
- Also added: a live moments-of-inertia readout (`I1, I2, I3 = ...`) in the
  sidebar, updated on every geometry slider change.

**New: 10-second rolling time window for all 3 time-series panels.** The
angular-velocity and H(t) panels previously plotted the *entire* run (up to
300s) compressed into one fixed-width canvas, which the user described as
"compressed"/"overwhelmed with colors." Added `scenes/rollingWindow.js`
(`computeVisibleWindow`) — an oscilloscope-style window that grows from 0 to
10s then slides, with "now" always pinned at the right edge — and applied it
uniformly to `TimeSeriesPanel.js`, `ControlPanel.js`, and the new torque
panel.

**Two real bugs found during browser verification of the above (not present
in the physics — pure rendering/lifecycle bugs), both fixed:**
1. **Torque canvas never draws anything the first time Control is turned
   on.** `resize2DCanvas` sizes a canvas from its own `clientWidth`/
   `clientHeight`, but the torque canvas's container starts `display: none`
   (clientWidth/Height = 0) and was only ever resized once, at page load.
   Fixed by calling `resize2DCanvas(torqueCanvas, torqueCtx)` inside
   `refreshSubfieldVisibility()` whenever Control is switched on, so it's
   re-measured at the moment it actually becomes visible.
2. **Panel grid can permanently blow up to ~30,000px tall after a resize to
   a very small viewport.** Root cause: CSS grid items default to
   `min-height: auto`, which for a `<canvas>` child equals the canvas's own
   `width`/`height` *attributes* — and those attributes are themselves set
   from the panel's `clientHeight` (times devicePixelRatio) by
   `resize2DCanvas`/`renderer.setSize`. That's a feedback loop: one resize
   event with a transiently-small or off-by-a-factor clientHeight inflates
   the canvas attribute, which inflates the grid track's min-content size,
   which inflates the next resize's clientHeight, snowballing until the
   layout gets stuck huge — confirmed reproducible by resizing the browser
   to 309×383 and back to 1400×800. Fixed with `min-height: 0; min-width: 0;`
   on `.panel` in `index.html`, the standard fix for this class of
   flex/grid-vs-replaced-element bug. Re-verified the same shrink/grow
   sequence afterward: panel heights now correctly settle back to ~382px
   each instead of ~29,777px.

Physics: unchanged (32/32 pytest). JS: 92/92 Vitest (11 new tests for
`ControlMetrics.js`, 4 for `rollingWindow.js`; the two rendering bugs above
were caught by live browser verification, not unit tests, since they're
CSS-layout/canvas-lifecycle issues outside what Vitest's jsdom environment
exercises).

**Still deferred, not yet approved for implementation**: the long-integration
RK4 drift in `rigidBody.js`'s `integrateM` (fixed 10 substeps/frame; measured
~47% relative `|M|` drift by T=220s, matching `T_SEP_STATIC` — raising
substeps to 100 brings this down to ~0.0007% in a quick check). This is what
made the separatrix/trajectory curves look like "wiring" on long runs. The
user asked to diagnose and hold off implementing until given the go-ahead.

## 2026-07-29 (cont.) — Fixed the RK4 drift

Went with the second option floated above rather than just raising the fixed
substep count: `integrateFull` and `integrateM` no longer take "substeps per
output sample" as a bare fixed divisor. Both now default to
`Math.ceil(dtSample / MAX_STEP)` substeps, where `MAX_STEP = 0.001` is a cap
on the actual RK4 step size `h` — chosen because it's exactly the effective
`h` the main animated trajectory already used by default (T=300s, N=30000,
dtSample~0.01s, 10 substeps), which was already visually/numerically verified
good. This makes the fix a no-op for that path and only densifies the
under-resolved case: `integrateM`'s separatrix curves (T=220, N=900,
dtSample~0.245s) now get ~245 substeps instead of a fixed 10, and the
trajectory-family curves (T=80, N=900) get ~89 instead of 10.

An explicit `opts.substepsPerSample` override still works for both functions
(used by one new regression test to check the override path, and available
if a future perf issue calls for coarsening a specific call site
deliberately).

Added a regression test (`rigidBody.test.js`) reproducing the exact
production separatrix config (`T=220, N=900`, near-separatrix IC) and
asserting `|M|` stays within 1% of its initial value for the whole run — this
would have failed outright under the old fixed-10-substep code (~47% drift)
and passes now. Also re-verified visually in the browser: a long free-spin
run's separatrix curves are now clean and close on themselves instead of
visibly spiraling outward.

Physics (`physics/`, the frozen Python oracle) untouched — this was purely a
JS-port fixed-step-integrator bug, since the Python side already uses
`solve_ivp`'s adaptive RK45 and never had this problem. JS: 94/94 Vitest (2
new tests).

## 2026-07-30/31 — Geometry preview stability, rolling-window ramp-up, playback controls, analytic separatrix

**Head-to-mass ratio no longer moves the racket in the live preview.**
`hoopFraction` only changes mass distribution, not shape, but the preview
recentered on the mass-weighted center of mass, which shifts by up to a full
unit across the slider's range (confirmed: `com[0]` goes from -0.2 to 0.7 as
hoopFraction goes 0.2→0.8, with `evecs` staying exactly identity — a pure
translation, not a rotation). Fix: `buildRacket()` now also returns
`comUnweighted` (the plain average of vertex positions, ignoring mass —
depends only on shape, never on hoopFraction/totalMass), and
`previewRacketGeometry()` in `main.js` recenters on that instead. Verified in
Node that previewed vertex positions are now identical across hoopFraction
values to floating-point precision (1e-16). The real simulation (`Run`) is
untouched — it must still rotate about the true center of mass. Trade-off:
since preview and simulation now use different anchors, there's a one-time
jump the first time you touch a geometry slider after a run (switching from
"watching physics" to "editing shape" mode) — not further movement on
continued dragging, just that one switch.

**Rolling time windows no longer "compress" during ramp-up, and are now 5s.**
`computeVisibleWindow` used to map `[t0, currentTime]` onto the full panel
width while the run was younger than the window, so the effective
seconds-per-pixel scale shrank every frame until the window filled up —
visible as data compressing leftward. Fixed: the window is now always
exactly `windowSeconds` wide from the very first frame
(`windowEnd = max(t0+windowSeconds, currentT)`, `windowStart = windowEnd -
windowSeconds`), so the scale never changes. The "now" markers in
`TimeSeriesPanel.js`/`ControlPanel.js`/`ControlMetrics.js` no longer hardcode
to the panel's right edge either — they're drawn at `toX(t[currentIndex])`,
so they visibly slide in from the left and reach the edge exactly when the
window starts sliding. Default `windowSeconds` dropped from 10 to 5 per the
user's request (10 packed in too many oscillations to read comfortably).

**Playback controls consolidated into 3 identical sidebar buttons.** Removed
the top-left Play/Pause button (user found it hard to notice) and the
size/weight mismatch between Run and Stop (only `#run-button` had the bold
styling). Sidebar now has **Run** / **Pause** / **Reset**, all
`.sidebar-action` (same padding/weight/width), stacked where Run/Stop used to
be. Pause is the relocated old Play/Pause toggle (freezes in place, keeps all
graphs/readouts). Reset is exactly the old Stop behavior (pause + seek to
t=0), renamed since that's what it actually does.

**Separatrix curves replaced with the exact analytic solution (see prior
discussion in this session for the fuzziness diagnosis).**
`analyticSeparatrixCurves` in `phasePortrait.js` substitutes
`M[imid]=circleSign*R*tanh(lambda*t)`, `M[imin]=arcSign*A*sech(lambda*t)`,
`M[imax]=circleSign*arcSign*B*sech(lambda*t)` into Euler's equations and
solves for `lambda,A,B` in closed form — verified by direct substitution
(residual ~1e-11, floating-point noise) for arbitrary I/R_cas. No
integration, so no possible drift or the old ~9x pole-to-pole bouncing.

Two real bugs found and fixed during implementation, both worth recording:
1. **Wrong sign pairing.** Only `(circleSign, arcSign)` combinations where
   the sign relationship keeps `M[imin]` and `M[imax]` "correlated" the right
   way are genuine solutions of Euler's equations — of the 4 naive sign
   combinations tried first, half looked completely fine at first (conserved
   `|M|` and `H` exactly, since those are algebraic consequences of the
   ansatz regardless of sign) but had an Euler's-equation residual of ~2.8,
   i.e. were not real trajectories at all. **Conserving |M| and H is
   necessary but not sufficient evidence a curve is a genuine solution** —
   this is now called out explicitly in both the code comment and a
   regression test. The correct pairing was confirmed by direct numerical
   integration (`integrateM`) starting exactly on each candidate branch: the
   genuinely valid ones keep an exactly constant `M[imin]/M[imax]` ratio for
   the entire run, the invalid one doesn't correspond to any real trajectory
   at all.
2. **4 curves are actually needed, not 2.** The full sign-symmetry group has
   4 valid elements (a Klein four-group), not 2 — the true separatrix is 2
   great circles (`M[imin] = +r*M[imax]` and `M[imin] = -r*M[imax]`), each
   only half-traced by a single sign combination (one arc pole-to-pole
   through one quadrant); the other arc of the same circle needs the second
   valid combination. Missing this initially produced only 2 curves that
   were both (unknowingly) on the *same* circle.
3. **Rendered as thin tubes, not `THREE.Line`.** Even once correct, the
   curves were confirmed present (pixel-sampled directly off the canvas:
   thousands of correctly-colored pixels) but essentially invisible in
   practice — for geometries where the separatrix's excursion off the
   `[imid,imax]` plane is small relative to R_cas, a 1px line all but
   vanishes against the sphere's own wireframe grid. Switched to a thin
   `TubeGeometry` (`SEPARATRIX_TUBE_RADIUS = 0.01`), the same fix already
   used for the racket's handle/hoop and for the identical underlying reason
   (WebGL ignores `LineBasicMaterial.linewidth` on nearly every browser/GPU).

JS: 103/103 Vitest (7 new tests for `analyticSeparatrixCurves`, including one
that documents the invalid-sign-pairing bug directly, plus 1 new
`comUnweighted` regression test and updated `rollingWindow.js` tests).
Verified visually in the browser across two different geometries: 4 sharp
orange curves forming the classic "X" pattern through both unstable poles,
correctly reshaping with geometry changes.

## 2026-07-31 (cont.) — Starting perturbation for any axis, deduped blue curves, R_cas scale fix

**Blue trajectory curves deduped 8→4.** `familyCurves()` looped over both
`sa` and `sb` signs per stable axis, but `(sa, sb=+1)` and `(sa, sb=-1)`
trace the *same* closed orbit from two different starting phases (confirmed
numerically: one's IC lands within ~0.1% of R_cas of a point already on the
other's path). Drawing both wastefully integrated and rendered every loop
twice, and since each copy accumulates its own slightly-different RK4 error,
the near-duplicate pair didn't perfectly coincide -- rendering as a visibly
thicker/fuzzier line than a single clean pass, which is what the user was
noticing when comparing them unfavorably to the newly-sharp tube-rendered
separatrix. Fixed by fixing `sb=1` and only looping `sa`, giving exactly one
curve per actual fixed point (4 total: `+imin, -imin, +imax, -imax`).

**Starting perturbation now works from any axis, not just imid.** The old
kick was hardcoded to imin-dominant-plus-tiny-imax, applied only when
`startAxis===imid && !controlOn && !windOn`. Generalized into `kickedIC()`
(phasePortrait.js): dominant component stays on whichever axis is
`startAxis`, a user-controlled `kickFraction`-sized component goes on one of
the other two (still preferring imin, matching the exact original imid
tuning; falls back to imid when imin itself is the start axis), and a tiny
epsilon (now proportional to the kick, not a fixed constant) on the third.
Removed the old conditional entirely -- the kick is now always applied,
governed by a new **Starting perturbation** slider (0-0.1, step 0.002,
default 0.02) right after Starting axis. Wind is untouched, still layered on
independently on top.

Key design decision: the kick is scaled relative to `I[startAxis]*spinRate`
(that axis's own baseline angular momentum), **not R_cas**. R_cas is built
from I[imid] specifically; for imin/imax it can differ from that axis's own
baseline by an order of magnitude (verified: for this app's default I,
I[imin]*spinRate ≈ 0.47 vs R_cas ≈ 5.50). Scaling by R_cas uniformly would
have silently changed the baseline spin rate for every imin/imax scenario.
Scaling by the axis's own baseline instead means `kickFraction=0` exactly
reproduces today's plain `w0[axis]=spinRate` for any axis, and at imid's
default 0.02 it reproduces the original NEAR_SEP_PERTURB tuning almost
exactly (I[imid]*spinRate coincides with R_cas to ~5 significant figures).

**New live-updating "current orbit" background curve**, same blue as the
trajectory family, showing the actual orbit the current Starting axis/
perturbation choice will trace -- `currentAxisOrbitCurve()` integrates
`kickedIC`'s exact M0 for a short, adaptively-chosen duration derived from
the linearized rate `R_cas*sqrt(|(1/I[axis]-1/I[a])*(1/I[axis]-1/I[b])|)`
(imid: a real saddle growth rate, same formula `analyticSeparatrixCurves`
uses; imin/imax: a real linearized oscillation frequency, since those are
elliptic fixed points, not hyperbolic). imid's duration scales like
-ln(kickFraction)/rate (verified against actual integrated crossing times,
not just the bare formula, since small errors compound badly near a saddle);
imin/imax get a couple of real periods. Both are deliberately short --
verified nowhere near the point where a near-separatrix orbit would start
the multi-bounce fuzziness the exact separatrix curves had before being
replaced with a closed form. `buildBackgroundWithCurrentOrbit()` folds this
into `trajectories` (no schema changes) and is called both from
`buildScenario` (the real run) and a new `previewSphereBackground()` in
main.js (mirroring the existing live racket-shape preview), so the sphere
panel updates live as Starting axis / Starting perturbation / geometry
sliders change, without needing Run.

**Real bug found and fixed while verifying this**: the dot for a free
imin/imax start rendered deep inside the sphere instead of near its own
fixed point, because R_cas was unconditionally computed from imid
(`casimirRadius(I, imid, spinRate)`) regardless of which axis actually
started spinning. This was a pre-existing inconsistency (not introduced by
this work) that the new live background curves made obvious for the first
time, since they anchor prominently at the correct R_cas-scaled position
while the dot did not. Fixed for free runs: R_cas is now computed from the
actual `startAxis`.

**Extended to controlled runs too, same session.** Initially left controlled
scenarios on the imid-based convention (matching the user's earlier "don't
mess with it for now" on the related wind/P-controller steady-state-error
discussion), but on reflection this R_cas question is a genuinely separate,
purely-cosmetic issue -- confirmed `desired_H`/`desired_L` (what
`ControlMetrics.js`'s achieved-detection and panel 4's reference line
actually use) are computed directly from `omegaStar`/`I`, never from R_cas,
so changing R_cas's basis cannot affect whether control actually works,
Kp calibration, or the "Done!" logic -- only where things land in the sphere
panel. Since the controller's real target is `I[endAxis]*spinRate` (exactly
what `desired_L` already is), basing R_cas on `endAxis` for controlled runs
means a successful run's dot now lands exactly on the target's own
green/red marker instead of an unrelated imid-based radius -- and a
wind-driven steady-state offset now reads as what it actually is (a small,
visible miss right next to the target marker) instead of being swamped by a
scale mismatch an order of magnitude bigger than the real effect. Verified
in the browser: a no-wind imax->imin controlled run's dot now sits exactly
on imin's marker at convergence ("Done! Achieved at t=2.79s" unchanged); the
same run with wind on shows the dot close to but visibly offset from the
marker, with a nonzero "current torque" reading, matching the expected P-only
steady-state error exactly.

Physics unchanged. JS: 118/118 Vitest (13 new/updated tests: `kickedIC`,
`currentAxisOrbitCurve`, `buildBackgroundWithCurrentOrbit`, the trajectory
dedup, and the R_cas fixes for both free and controlled runs). Verified
visually in the browser: imid free-spin kick, imin/imax near-stable wobbles,
and controlled runs (with and without wind) all render correctly, with the
dot now landing at its actual fixed point instead of near the sphere's
center or an unrelated radius.

## 2026-07-31 (cont.) — Panel 4 becomes a real 3D Hamiltonian view, with a 2D/3D toggle

Panel 4 ("Current vs. desired (H)") was a flat H(t) line chart only; the
energy-ellipsoid visualization (semi-axes sqrt(2*H*I_k), the H=const level
set in M-space) lived as an optional overlay toggle ON the sphere panel
instead, added back when it was still an experiment (see the "Shouldn't H be
an ellipsoid?" entry). Promoted it into panel 4 as a proper second view:

- **New toggle** ("3D", top-right of panel 4) swaps between the existing 2D
  canvas and a new dedicated 3D one (`control-3d-canvas`), each with its own
  THREE.js scene/camera/renderer -- reusing the exact lesson from the torque
  panel's own display:none/block transition (a canvas sized from
  clientWidth/Height while hidden reads 0, so the toggle handler re-measures
  it the moment it becomes visible, not just on window resize).
- **Removed** the "Hamiltonian 3D" toggle and ellipsoid entirely from the
  sphere panel (`SphereScene.js`'s `createSphereScene` no longer builds or
  returns one) -- panel 4 is its dedicated home now, no redundant control.
- **New `createHamiltonianScene()` / `updateEnergyEllipsoid()`** (moved into
  `ControlPanel.js`, since panel 4 is now their only consumer): a faint
  reference sphere at R_cas (context -- the ellipsoid's own scale, normalized
  the same way, is directly comparable to it), a "current" ellipsoid that's
  ALWAYS visible and updates every frame (constant-sized exactly when H truly
  is conserved -- free, no wind -- and visibly resizing otherwise, since wind
  or control both do real work; this needed no mode-specific branching at
  all, since `updateEnergyEllipsoid` just reflects whatever H(t) actually
  is), a "desired" ellipsoid shown only for controlled scenarios (fixed at
  meta.desired_H, set once in `renderDoc` rather than every frame like
  `currentEllipsoid`, since a controlled target doesn't change mid-run), and
  a dot at the current M-position (reusing SphereScene.js's own
  `dotPosition`/`updateDot`, no duplication).

Verified in the browser: a free-spin run shows one white ellipsoid tracking
H(t) with the dot on its surface; a controlled run shows both ellipsoids
distinct early on (confirmed numerically too: H(0)=17.35 vs
desired_H=18.75 for a representative imid->imax run) and visually converging
into one orange-tinted shape once achieved (H(t) reaches desired_H exactly),
matching the "Done!" 2D-panel state. Toggling back to 2D restores the flat
chart correctly. No console errors across free/controlled/wind combinations.

JS: 121/121 Vitest (5 new tests: `createHamiltonianScene`'s default shape/
visibility, `updateEnergyEllipsoid`'s scaling and negative-H clamping).

## 2026-07-31 (cont.) — Auto-fit camera for panel 4's 3D view

User-reported: the energy ellipsoid could "blow way out of proportion" of
its panel, inconsistently. Reproduced directly (not guessed): with geometry
sliders pushed toward their minimums (all within normal ranges -- handle=1,
hoop length=1, hoop width=0.5, mass=0.3, head-to-mass=0.2) starting on imin,
`I=[0.0019, 0.0765, 0.0784]`, giving semi-axes `[0.99, 6.33, 6.41]` -- over
6x the reference sphere's radius. Root cause: for a free run, R_cas is now
based on the actual starting axis (see the earlier "R_cas scale fix" entry).
When that axis's own moment of inertia is disproportionately small relative
to the other two, R_cas becomes tiny, but the ellipsoid's other semi-axes
(governed by the larger moments) don't shrink to match -- confirmed H stays
exactly conserved throughout, so this is a real, correct consequence of the
normalization choice, not a data bug or rendering glitch.

Fix: `computeAutoFitDistance(maxExtent, fovDegrees, margin=1.3)` (new,
`ControlPanel.js`) computes how far back a perspective camera needs to sit
to frame content of a given bounding radius; `margin=1.3` was chosen so the
common maxExtent=1 case reproduces the original hardcoded distance (~3.17)
closely, so typical scenarios see no visible change. Every frame, `main.js`
takes the max of {reference sphere radius 1, both ellipsoids' max semi-axis,
the dot's distance from origin} and smoothly (10%/frame lerp, not a hard
snap -- wind can make the ellipsoid's scale jitter a bit frame to frame)
moves `controlCamera` along its original fixed viewing direction to the
right distance. Camera distance resets to the neutral default at the start
of every new Run, rather than carrying over a wildly different previous
run's distance.

Verification hit a real testing-harness snag worth recording: samples taken
a few seconds apart initially showed the camera distance frozen well short
of its target, looking like a bug in the smoothing logic. Turned out to be
Chrome throttling `requestAnimationFrame` for the *non-fronted* browser tab
being tested -- fronting the tab (`tabs_select`) immediately showed the
distance fully converged to the target, and the ellipsoid framed correctly
with no overflow. Good reminder for future verification: a frozen-looking
animated value during automated testing is worth checking against tab focus
before assuming the underlying logic is wrong.

JS: 124/124 Vitest (3 new tests for `computeAutoFitDistance`: reproduces the
original default distance, scales linearly with maxExtent, never returns
non-positive for a degenerate input).

## 2026-08-03 — Aesthetics pass (Observatory → Blueprint) and Cloudflare Pages deployment

Ran an aesthetics-mockup exploration (four static HTML color-theme previews
of the real panel layout, generated without touching any app code) so the
user could pick a direction before implementation. Landed on "Blueprint"
(deep navy, chalk-white ink, gold/cyan/coral accents) after trying
"Observatory" (dark neon) first and deciding against it. Implementation:
`app/src/theme.js` centralizes every canvas/WebGL color as one exported
object; `index.html`'s `:root` CSS variables mirror the same palette for
chrome (backgrounds, borders, sliders, buttons) -- the two are hand-kept in
sync (documented in both files) since there's no build-time bridge between
JS and CSS here. Swapping themes going forward means editing those two
blocks, not hunting hex literals across scene files.

Alongside the theme, fixed several rendering issues the user caught by
inspection: the WebGL panels (racket, sphere, 3D Hamiltonian) were defaulting
to THREE's black clear color instead of matching the 2D panels' own
`ctx.fillStyle`, so all three scenes now get `scene.background` set
explicitly to `THEME.panelBg`. The four grid panels could render at visibly
different heights depending on how many legend items each one wrapped to
(more legend items → more wrapped lines → less height left for the actual
`.panel` canvas box within the same fixed-height grid cell) -- fixed with a
fixed (not min-) height on `.panel-legend` plus an empty matching-height
spacer under the racket panel (which has no legend of its own). Removed the
"close to the tipping point" racket caption entirely per request (the
underlying `scenarioRunner.js` caption field is left alone -- unused
metadata now, same as the pre-existing `dot_color` field, not worth a
schema change for a UI-only removal). The energy panel's y-axis floored at
a slightly-negative padded minimum instead of exactly 0 even though H is
provably non-negative (a sum of squares over positive inertias), leaving a
dead strip of unreachable space below the "0" gridline -- now
`yMin = Math.max(0, yMin - pad)` so zero sits exactly at the plot's
bottom-left corner when the data justifies it. Both time-series panels
(`ControlPanel.js`, `TimeSeriesPanel.js`) also picked up small top/right
margins in `drawAxes`'s plot-area bounds -- the topmost y-tick and
rightmost x-tick labels were being drawn exactly at the canvas edge
(`plotTop=0`, `plotRight=width`), so half the label rendered off-canvas.

Separately, got `racket_viz` committed and pushed for the first time (it had
existed only locally until now) and set up Cloudflare Pages for a
shareable deployment link, hitting two unrelated blockers worth recording:

1. Cloudflare's build clones with submodules recursively, and this repo has
   had a dangling gitlink (`src/models/3D_SO3_Windy_Pendulum/ph_gp_ode_v2/JaxJD`,
   commit-mode tree entry with no matching `.gitmodules` entry) since its
   very first commit, on every branch -- so the clone step failed outright
   with "error occurred while updating repository submodules" before
   Cloudflare ever got to checking whether `racket_viz/` existed. Fixed by
   `git rm --cached`-ing the dangling gitlink, scoped to `feature/racket-viz`
   only (not `main`), since it's pre-existing and unrelated to this feature.
2. Cloudflare's GitHub OAuth login (used to sign in) is a separate grant
   from the GitHub App *installation* that actually authorizes repo access
   -- for an org-owned repo (`CVC-Lab/LieSPHGP`), installing/approving that
   app requires an org owner, not just the repo's contributors. Confirmed
   via GitHub's own settings (`Installed GitHub Apps` empty on the personal
   account, org-level installations page 404s for a non-admin) rather than
   guessed. Resolved once an org admin approved the pending install request.

Build settings landed on: root directory `racket_viz/app`, build command
`npm run build`, deploy command `npx wrangler pages deploy dist` (Cloudflare's
newer unified Workers+Pages UI needs an explicit Wrangler deploy command even
for a plain static site -- leaving it blank errors with "need deploy
command"). Also added `app/wrangler.jsonc` (name + `assets.directory`) so
`wrangler deploy` (not the old Pages-only `wrangler pages deploy`) can find
the project and static output automatically -- and passed
`--autoconfig=false`, since `wrangler deploy`'s default framework
auto-detection tries to wire up `@cloudflare/vite-plugin` and hard-errors
on anything older than Vite 6, which this project doesn't need at all for
a plain static build.

One more gotcha worth recording: Cloudflare's "Retry build" replays the
exact build-settings snapshot from when that build entry was originally
queued, not the project's current live settings -- so updating and saving
Settings → Build has no effect on a build you retry after the fact. Only a
genuinely new build (fresh push, or a new manually-triggered deployment)
picks up updated settings.

## 2026-08-05 — Second system: windy pendulum, multi-system carousel

Started the site's second physical system (branch `feature/second-system`,
off `feature/racket-viz`), per the original 2026-07-24 plan's "swipe between
systems" idea. Deliberately not pushed during this build -- every push to a
Cloudflare-connected branch triggers a build against a limited quota, so
work happened entirely against the local dev server until ready for review.

**Skipped the ML pipeline entirely.** Researched LieSPHGP's
`src/models/3D_SO3_Windy_Pendulum/` (6 parallel training variants) before
starting: zero trained checkpoints anywhere, empty training logs, two
controller scripts that exist as source but can't run (both require
checkpoints that don't exist). Confirmed `envs/windy_pendulum_3d.py` is
fully-actuated (direct 3-axis body torque) exactly like the racket's `g=I₃`
case -- same reason the racket's own controller never needed a trained
model. Ported the analytic physics directly instead (gravity + friction
torque, a hand-derived gravity-compensated IDA-PBC controller), matching how
the racket viz already worked. See `ARCHITECTURE.md`'s "Windy pendulum:
physics differences from the racket" for the physics/controller details.

**Carousel, not a toggle.** User explicitly wants the transition to
genuinely SLIDE (not flip/rotate, to avoid the same blur bug the racket's
Explain flip-cards needed a real fix for) and wants it built for a *third*
system later, not hardcoded to two -- see `ARCHITECTURE.md`'s "Multi-system
carousel" section for the `#system-viewport`/`#system-track`/`.system-slide`
structure and the deliberate choice NOT to build a shared "System" class
(each system is its own independently-written module, copy-pasting the
established pattern, per the user's own stated expectation that adding
systems "should be a lot of copying and pasting").

**Panel 4 (was the racket's Casimir Sphere) is a promoted torque/work
readout instead.** Discussed two options with the user first: (a) a
"bob-position sphere" reusing the sphere-panel's visual language (genuinely
analogous physics -- swing-vs-flip-over is topologically the same
stable/unstable-separatrix structure as the racket's stable-spin-vs-flip,
even though the underlying invariant differs), or (b) promote the racket's
existing sidebar torque/work readout to a full panel. User picked (b)
specifically "so both functions can be seen easier than hiding in the side
panel" -- reuses `ControlMetrics.js` unchanged, no new physics/design needed.

**Two real bugs found only by actually running it in-browser** (both now
documented in `ARCHITECTURE.md` so they aren't rediscovered by a future
system): (1) copy-pasting `DataLoader.js::validateScenario(doc)` into the
new system's `renderDoc` -- that validator enforces the racket's own exact
schema (`R_cas`, `background`, racket-shaped `geometry`), so it threw at
module-load time for the pendulum's legitimately-different doc shape,
which silently broke `app.js`'s import chain and prevented EVERY system's
`resume()` from ever being called, not just the pendulum's. (2) Reusing
`GravityCompensatedAttitudeController`'s default `clip=2.0` (inherited from
the racket's small-torque scale) for a real pendulum scenario: the
worst-case gravity torque alone is several N·m, so the controller was
chronically saturated and could never fully cancel gravity, producing a
residual disturbance that looked exactly like an uncontrolled swing instead
of converging -- fixed by passing an unclipped (`Infinity`) controller from
`pendulumScenarioRunner.js` instead of trusting the class's own default.

Also added a "Starting angle" slider (defaults to 25°, mirroring the
racket's own `startingPerturbation`) after noticing the free-swing demo sat
completely motionless by default -- starting exactly at the hanging-down
equilibrium is a true zero-torque fixed point, same reasoning as why the
racket needs a kick off its own unstable axis.

## 2026-08-06 — Pendulum control UX pass, panel-text rewrites, Title Case convention

**Removed the K_R/K_p ("attitude gain"/"rate gain") sliders entirely; gains
now auto-scale from the pendulum's own live inertia.** User asked why they
existed at all, since nothing else in the sidebar exposes a raw controller
gain. Same reasoning as the racket's own `defaultKpForAxisControl`: I₂/I₃
(transverse inertia) varies roughly 240x across the geometry sliders' own
ranges, so a fixed gain pair would be badly overdamped for a light/short
pendulum and sluggish/underdamped for a heavy/long one. Added
`default_gains_for_attitude_control`/`defaultGainsForAttitudeControl`
(Python + JS, ported line-for-line, both test-covered) implementing the
standard 2nd-order relations `K_R = omega_n^2 * I`, `K_p = 2*zeta*omega_n*I`,
critically damped (`zeta=1`) by default -- `pendulumScenarioRunner.js` now
computes these internally from the resolved geometry instead of accepting
them as scenario params, exactly mirroring how the racket's own
`buildScenario` computes its Kp internally with no Kp param in its public
API.

**Target dropdown replaced with a continuous "Target angle" slider
(-180°..180°); "Starting angle" widened to -80°..80°.** The three-option
dropdown (hanging down / horizontal / inverted) was an artificial UI
restriction, not a physics limit -- both sliders now reuse one shared
`tiltedDownDirForAngle()` helper in `pendulumMain.js` (0°=hanging down,
±180°=inverted, sign picks which side), replacing three near-duplicate
tilt-vector blocks with one. Both sliders live-preview the pose they'd
produce while paused, via a single `previewingPose` flag (renamed from the
starting-angle-only `previewingStart`) so dragging either one doesn't fight
`updateFrame`'s per-frame orientation write.

**Distinguished "achieved" (reached the exact target) from "settled"
(came to rest, but off-target) -- and fixed a stable/unstable equilibrium
mislabel in the process.** With wind on, the controller's lack of an
integral term means a constant disturbance torque leaves a permanent
steady-state attitude error (confirmed numerically: ~2° off-target for
windStd=0.05 in one test run) -- so `computeAchievedIndexAttitude`'s tight
attitude-error tolerance is gated on `!wind_on`, and a separate, looser
`computeSettledIndex` (sustained near-zero angular velocity, independent of
*where* it settles) drives a distinct "Settled with wind -- holding N·m,
X° off-target" readout instead. Both live in `ControlMetrics.js`, shared
with the racket's own `computeAchievedIndex`/`isAchievable` pattern.
Separately: the "holding requires no torque" case can mean either the
*stable* equilibrium (hanging straight down) or the *unstable* one
(inverted straight up) -- both are torque-free (com collinear with gravity
either way), so the original code's blanket "stable equilibrium" label was
wrong whenever the target was upright. Fixed by checking `sign(desired_H)`
(negative = below the pivot = stable; positive = above = unstable), caught
by the user spotting "unstable equilibrium" mislabeled as "stable" on an
inverted-target run.

**All four pendulum panel explanations rewritten, collaboratively, one at a
time.** Angular velocity: expanded to properly explain the axial vs.
transverse axes (an SVG diagram was drawn for this conversation to work out
the explanation, not committed to the repo) and to correct "the two
transverse axes go through the bob" to the physically-accurate "through the
pivot" (all three principal axes pass through the pivot, since inertia here
is about the pivot, not the pendulum's COM). Hamiltonian: rewritten to open
by defining what a Hamiltonian *is* (mirroring the racket panel's own
opening sentence) rather than assuming the reader already knows, and to use
only complete sentences throughout (no colon-fragments). Free swing:
tightened, dropped the racket cross-reference so the panel reads standalone.
Control torque & work: expanded substantially (this panel had the most
spare vertical room of the four) to name and explain IDA-PBC
(Interconnection and Damping Assignment Passivity-Based Control) in plain
terms -- reshaping the energy landscape so the target becomes the new low
point, via live gravity cancellation plus a spring-and-damper pull toward
the target.

**Established a per-panel "shrink font until it exactly fits, no
scrollbar" convention** for these longer flip-card explanations (the
`.flip-back` back-face has real `overflow-y: auto`, so an overlong
explanation doesn't visibly break anything -- it just silently requires a
scroll the user won't discover from a static view). Checked by temporarily
overriding `.flip-back`'s `font-size` in-browser and comparing
`scrollHeight` to `clientHeight` at each candidate size, then hard-coding
the largest one that fits exactly as a scoped CSS override
(`#pendulum-<panel>-flip .flip-back.centered { font-size: ...px; }`) --
landed on 12px (Angular velocity), 13.5px (Hamiltonian), 12.5px (Control
torque & work), and the shared 14px default (Free swing, short enough as
rewritten).

**Fixed a real y-axis label overlap bug, not a "small window" illusion.**
The rotated y-axis unit label (e.g. "Hamiltonian (H)") in `axisTicks.js`'s
`drawAxes` was positioned at a fixed pixel offset from the panel edge,
independent of how wide the adjacent tick-value numbers actually were --
most visible on the pendulum's Energy panel specifically, since its
negative-decimal H values (e.g. "-8.60") produce the widest tick labels
anywhere in the app. Fixed by measuring the actual rendered tick-label
width (`ctx.measureText`) each draw and positioning the rotated label clear
of it, plus bumping the shared `marginLeft` (46px -> 54px, `ControlPanel.js`
/ `TimeSeriesPanel.js` / `ControlMetrics.js`) so there's room for the
now-dynamic gap. Confirmed the bug reproduces identically regardless of
window size (all these offsets are fixed canvas-pixel values, not
CSS-relative), then confirmed the fix holds at both a normal and a
deliberately narrow window width.

**Adopted Title Case for every UI label, heading, panel title, legend
item, dropdown option, and button across both systems** (standard rule:
capitalize each "important" word, lowercase minor words -- "of", "to",
"and", "vs." -- unless first/last in the string), per explicit user request
("Starting angle" -> "Starting Angle", "Moments of inertia" -> "Moments of
Inertia"). Deliberately excludes two categories, which stay sentence-case:
the four flip-card explanation paragraphs (prose, not titles), and live
status/readout text like "Work: 12.3 J", "Done! Achieved at t = 3.01s", and
"Settled with wind -- holding..." (messages, not titles -- title-casing
"Done! Achieved At T =" would read as broken English). This is now the
standing convention for any future label/heading/legend addition to either
system, not a one-time cleanup.

## 2026-08-11 — Third system kickoff: windy cart-pole, and a real trained NN this time

Started the site's third physical system (branch `feature/third-system`, off
`feature/second-system`, which stays the up-to-date deployed branch). Source
physics: `summer-2026/cartpole.py`, a from-scratch Hamiltonian rewrite of the
classic Sutton cart-pole ("windy cart-pole for ct-rl") with state
`(x, theta, p_x, p_theta)` -- POSITIONS then MOMENTA, not gym's
`(x, x_dot, theta, theta_dot)` -- and a continuous per-step wind SDE.

**Convention flip from the racket/pendulum:** here `theta=0` is the pole
balanced UPRIGHT (the classic cart-pole's actual control target), not
hanging down. Confirmed directly from the source (`theta_threshold_radians`
fails the episode at +-12 deg off upright, matching Barto/Sutton/Anderson's
original problem statement) rather than assumed by analogy to the other two
systems.

**This system's wind is NOT the racket/pendulum's wind model.** `wind.js`
samples ONE disturbance torque and holds it constant for the whole run.
`cartpole.py`'s wind is a genuine continuous SDE -- re-sampled every step via
a stochastic Heun predictor-corrector, with diffusion
`s = sigma_gust + sigma_turb*|theta_dot|` that grows with the pole's own
angular velocity, entering as an impulse on `p_theta` only. No existing
`wind.js`/`wind.py` pattern applies; ported directly from `cartpole.py`'s own
`diffusion`/`integrator` methods into a new `cartpole_wind.py`.

**Checked for the pendulum's "checkpointless ML" trap -- and did NOT find
it this time.** A coworker sent two live-demo links
(`claude.ai/code/artifact/285103b6-...` and `.../9697efad-...`, the second
being the broader "two plants" version) showing a REAL trained SAC policy
(`ct_sac_cartpole_top_300000_steps.pth`, half-precision weights, confirmed
directly off the demo's own footer text) running live client-side, plus
`none` (no control) and `you` (arrow-key manual drive) modes. Decision,
discussed with the user: port the real weights as a hand-written JS
forward-pass (small MLP, no PyTorch/ONNX at runtime -- still fits "no live
backend," since it's static data + a pure function, same spirit as the
physics port itself), rather than skipping it -- explicitly to respect the
coworker's actual contribution, since this is meant to be presented as
joint work. **Blocked on the coworker actually sending the checkpoint (or
exported weights) and the policy's architecture/observation-normalization
details** -- they were slow to respond, so the build is split into two
phases so the rest of the system isn't blocked waiting:

- **Phase 1 (this entry, no external dependency):** physics
  (`cartpole_dynamics.py`, `cartpole_wind.py`), scene, sidebar, all 4 panels,
  and a real interim "Control" experience via the `you` (manual/keyboard)
  mode -- not a placeholder, one of the coworker's actual three modes, and
  it exercises the same force/work panel and diagnostics block the eventual
  `ct_sac` mode will use. Ships as a complete, deployable third system with
  `none`/`you` as the only controller options.
- **Phase 2 (blocked):** `ctSacPolicy.js`, a small isolated forward-pass
  module, added as a third controller option once the weights arrive. Deliberately
  isolated: panel 4 and the sidebar diagnostics block are architected to read
  "whatever force is being applied this instant," not to know or care where
  that force comes from, so Phase 2 shouldn't require touching Phase 1's code.

**Panel/diagnostics split, decided with the user:** panel 4 (main 4-panel
grid) stays the established `ControlMetrics`-style force/work panel
(reusing the pendulum's pattern, force in N instead of torque in N*m -- the
cart-pole's actuator is a linear force, not a torque). The coworker's own
fragility-specific readouts (difficulty, instability time, wind/force gains,
survived, rolling gust plot) do NOT go in the 4-panel grid; they go in a new
**sidebar-only** block shown when Control is on, mirroring the racket's
original sidebar-only "Control function" readout from 2026-07-29 (later
promoted to a full panel for the pendulum -- this system deliberately does
NOT repeat that promotion, per the user's explicit choice).

**Three of the four sidebar diagnostics are fully analytic, not learned --
verified two independent ways before being trusted.** Gain A (wind->theta_dot),
Gain B (force->theta_dot), and Instability Time are closed-form properties of
the linearized plant at the upright fixed point (mp, mc, l, g only -- no
controller involved at all). Derived by hand, then cross-checked against (1)
the coworker's own demo's displayed numbers at its default sliders (32.2,
1.46, 0.252s) and (2) the eigenvalues of an independently-computed numerical
4x4 Jacobian of `cartpole_dynamics.drift` at the upright point -- agreement
to several significant figures both ways. **Explicitly did NOT attempt to
reverse-engineer "Difficulty Lambda"** (the demo's fourth number, 0.0088 at
default sliders) -- unlike the other three, it didn't yield to a first-
principles derivation with the same confidence from one data point, so it's
deferred until the coworker can give the actual formula, rather than
shipping a plausible-looking guess dressed up as verified physics.

**Test-first, per the established process, with one design bug caught before
it shipped:** wrote the full physics test suite (24 pytest cases across
`test_cartpole_dynamics.py`, `test_cartpole_diagnostics.py`,
`test_cartpole_wind.py`) against `NotImplementedError` stubs first, confirmed
red (22 failed on `NotImplementedError`, 1 already-green constant-only test,
zero import/collection errors, all 50 pre-existing tests still passing)
before writing any real implementation. While designing the wind module's
"is the diffusion function smooth at theta_dot=0, not kinked" test, the first
draft compared raw left/right secant slopes at +-h -- verified numerically
that this is WRONG (both a genuinely smooth minimum and a real kink are even
functions, so both give a trivial sign flip either way, proving nothing).
Fixed by comparing how the one-sided secant shrinks as the step size shrinks
(shrinks proportionally to h for a true zero derivative; stays ~constant for
a real kink) -- verified numerically before locking in the corrected test.
Also verified the wind variance-scaling test's expected ratio (~4.02 for a 4x
change in dt) numerically before writing its tolerance, so it wouldn't be
flaky on the first real run.

**Next step:** implement the three stub modules to green, then the JS port
(`cartpole*.js`, Vitest, cross-validated fixtures) once Python is green, per
the established order.

## 2026-08-11 (cont.) — Physics implementation, pytest green (72/72)

Implemented `cartpole_dynamics.py`, `cartpole_diagnostics.py`,
`cartpole_wind.py` in that order (each the direct extraction/closed-form
derivation described in this file's kickoff entry above and in the modules'
own docstrings). `hamiltonian()` deliberately reuses `velocities()`
internally (`0.5*(px*xdot + pth*thdot)`, equal to `0.5 * p^T M^-1 p` since
`(xdot,thdot) = M^-1 p`) rather than re-deriving `M^-1` a second time, so the
energy formula can't silently drift out of sync with the velocity map it
depends on.

One test (not implementation) issue found on the first real run, fixed in
the test file: `test_upright_is_unstable_hanging_is_stable` assumed a bare
theta-only perturbation would grow at least 10x over 0.5s of free fall, but
the actual observed growth was only ~3.7x -- confirmed numerically that this
is because a pure `(delta_theta, delta_p_theta=0)` kick splits roughly
evenly onto the linearization's growing (+lambda) and decaying (-lambda)
eigenmodes, so the visible growth tracks `(1/2)*e^(lambda*t)`, not a full
`e^(lambda*t)`. Fixed by extending the integration window to T=1.0s
(~4 instability e-folds, observed growth ~26x) and loosening the threshold
to 5x, with the reasoning written into the test's own docstring so a future
reader doesn't mistake this for a fragile magic number.

Physics: 72/72 pytest (50 pre-existing + 22 new, 0 regressions).

**Next step:** generate cross-validation fixtures from this Python oracle,
then the JS port (`cartpole*.js`, Vitest) test-first against them, per the
established order.

## 2026-08-11 (cont.) — Cross-validation fixtures + JS port, Vitest green (196/196)

Generated three fixtures from the now-green Python oracle (no generator
script committed, matching the existing precedent for
`pendulum_geometry_cases.json`/`racket_geometry_cases.json` -- one-off,
just the JSON output is checked in): `cartpole_dynamics_cases.json` (48
cases across 4 plants x 4 states x 3 forces: velocities/drift/hamiltonian),
`cartpole_diagnostics_cases.json` (5 plants: gain_a/gain_b/instability_time),
`cartpole_wind_deterministic_cases.json` (12 diffusion cases + 6 sigma=0
Heun-step cases -- deliberately NOT sampling any actual noise, per this
system's wind convention that the random stream itself isn't cross-language
validated, only the deterministic mechanics).

Ported `cartpoleDynamics.js`, `cartpoleDiagnostics.js`, `cartpoleWind.js`
module-for-module from their Python counterparts, test-first (23 new Vitest
cases against `throw new Error("not implemented")` stubs, confirmed red --
22 failed for that reason, 1 already-green constant-only test, matching the
Python side's red-phase shape exactly -- before implementing). Every new
test passed on the **first implementation attempt**, the same experience the
pendulum's JS port had and for the same reason: the hard physics (the gain/
instability-time derivation, the wind smoothness-test design bug) was
already found and fixed once, in Python, and this port just reused those
answers.

**New JS-side convention decided for this module group:** state args
(`z`/`theta`/`px`/`pth`/`F`) stay positional, but the shared `{mp, mc, l,
g, ...}` group -- repeated across every function in all three modules --
is a trailing options object with camelCase keys and inline defaults
(`{ mp = 0.1, mc = 1.0, l = 0.5, g = 9.8 } = {}`), mirroring
`pendulumGeometry.js`'s existing `buildPendulum({ rodLength, ... })` pattern
rather than `pendulumDynamics.js`'s all-positional
`gravityTorqueBody(R, comBody, totalMass, g)` (that one only has 4 args
total, not enough to motivate an options bag). `cartpoleWind.js`'s
`heunStep` keeps its own local Box-Muller `standardNormal(rng)` helper
(uniform-source-in, standard-normal-out) rather than importing `wind.js`'s
private equivalent -- same primitive, but the two wind features are
independent per-system modules by this project's explicit "no shared code
across systems' own physics" convention, and a shared math primitive this
small (6 lines) isn't worth breaking that for.

Physics (Python): 72/72 pytest, unchanged this round. JS: 196/196 Vitest
(173 pre-existing + 23 new, 0 regressions).

**Next step:** `CartPoleScene.js` (mesh + update function), sized statically
for the geometry sliders' worst case per `ARCHITECTURE.md`'s camera-framing
rule, then the sidebar + `cartpoleMain.js` (geometry/wind sliders, Motion
section, Control toggle with `none`/`you` only for now, Run/Pause/Reset,
the 4 flip-card panels, and the new sidebar-only fragility-diagnostics
block) per the established build order -- still nothing blocked on the
coworker's NN weights.

## 2026-08-11 (cont.) — CartPoleScene.js: flat 2D-style render, not fuller 3D

**Confirmed with the user before writing any scene code**: unlike the
racket/pendulum, `CartPoleScene.js` renders FLAT (side-on, no perspective
depth) rather than a fuller-3D angled view. Justified twice over -- the
physics itself is genuinely planar (x, theta only; nothing ever leaves a
plane, unlike the racket's/pendulum's real 3D rotation), and the coworker's
own live demo (which this system will sit alongside) renders it flat too.
Still built on THREE.js/WebGL for pipeline consistency with the rest of the
app; `cartpoleMain.js` will frame it with an `OrthographicCamera` (removes
perspective foreshortening entirely) rather than a `PerspectiveCamera`
merely aimed at a planar scene.

**Rendering convention chosen** (documented in the module since the physics
itself doesn't care how theta is drawn): x=cart position, y=height,
track at y=0, theta=0 (upright) -> pole points +y, positive theta tips
toward +x -- the standard "clock hand from 12" convention. Cart/pole visual
proportions (0.4m x 0.24m cart, thin pole) match `summer-2026/cartpole.py`'s
own pygame render pixel ratios exactly, not picked freehand.

**Real camera-framing bug caught by the module's own test, before it ever
rendered:** the first draft of `cameraHalfExtents` sized `halfWidth` from
the track boundary and cart width alone. `CartPoleScene.test.js`'s "a full
pole swing circle... stays within the frame at any angle" test failed
immediately (`expected 3 to be less than 2.99`) -- because the pole can
point in ANY direction (a free/wind-driven run isn't limited to the RL
+-12deg band), a pole lying flat sideways reaches `maxPoleLength` further
out than the track alone, including with the cart already at the boundary.
This is exactly the same CLASS of bug `ARCHITECTURE.md` already warns about
for the pendulum's camera (sized only for "hanging down," clipped an
inverted target) -- caught here by a test before a single frame was ever
rendered, rather than by spotting a clipped pole visually after the fact.
Fixed by adding the `maxPoleLength` term to `halfWidth` too, not just
`halfHeight`.

Added `cartpoleCart`/`cartpolePole`/`cartpoleTrack`/`cartpoleGust` to
`theme.js`, reusing the existing Blueprint accent hexes (cyan/coral/gold)
rather than inventing new colors, matching how `racketTube`/`racketFaceA`/
`racketFaceB` already reuse the same accent palette under per-system names.

JS: 207/207 Vitest (196 + 11 new, 0 regressions).

**Next step:** sidebar + `cartpoleMain.js` (geometry/wind sliders, Motion
section, Control toggle with `none`/`you` only for now, Run/Pause/Reset, the
4 flip-card panels, and the sidebar-only fragility-diagnostics block), then
carousel wiring -- still nothing blocked on the coworker's NN weights.

## 2026-08-11 (cont.) — Sidebar + cartpoleMain.js wired in, verified live: third system ships

Built `index.html`'s third `.system-slide` (Cart-Pole Geometry / Motion /
Control+diagnostics / Wind sections, 4 flip-card panels), `cartpoleMain.js`,
and wired both into `app.js`'s `systems` array. Two small, deliberately
minimal additions to shared modules rather than a fork, per
`ARCHITECTURE.md`'s "reuse where it genuinely fits" rule: `ControlMetrics.js`'s
`drawTorquePanel` gained an optional `yLabel` param (defaults to the
existing "Torque (N·m)", so the pendulum's call site is untouched) so
cart-pole's call site can pass "Force (N)" -- same drawing logic, different
actuator units, a real reuse, not a relabeling hack. `TimeSeriesPanel.js`
gained a NEW function, `drawCartPoleStatePanel` (theta and theta-dot, not a
generalized version of the existing 3-component omega drawer) -- these are
two different physical quantities, not 3 components of one vector, a real
schema difference per the same rule.

**cartpoleMain.js's animate() loop is architecturally different from
main.js's/pendulumMain.js's**, matching `cartpoleScenarioRunner.js`'s own
docstring: there is no precomputed trajectory to scrub with a `Playback`
cursor, since "you" mode's force depends on live keyboard input during the
run. animate() itself steps the physics once per rendered frame (via
`cartpoleScenarioRunner.advance`, sub-stepping internally), appending to a
rolling history buffer (trimmed in batches past 4000 samples, not every
frame) instead of indexing into a fixed-length array. Arrow-key state is
tracked via plain `keydown`/`keyup` listeners on `window` -- harmless while
a different carousel slide is active, since nothing reads that state unless
this system's own animate() loop is actually being scheduled.

**Camera**: `OrthographicCamera`, framed via `CartPoleScene.cameraHalfExtents`
at its worst-case static extents, with a "contain" letterbox fit computed
in `resize()` (expand whichever of half-width/half-height the canvas's own
aspect ratio has slack on) so the worst case is fully framed at any panel
aspect ratio -- same spirit as `ControlPanel.js`'s `computeAutoFitDistance`
for panel 4's 3D view, adapted for two independent axis extents instead of
one perspective distance.

**Scope deliberately trimmed for this round** (all noted so they aren't
mistaken for finished): the sidebar's Fragility Diagnostics block shows
Instability Time / Gain A / Gain B / a live "Survived" readout, but NOT yet
the rolling gust plot or "Last 20 Episodes" history (would need an
auto-restart-on-termination episode loop, not built this round) -- and
Difficulty Lambda still shows "pending", per the earlier entry's honest
placeholder rather than a guessed formula.

**One real bug found live, not by any test, then given a real regression
test:** the Force & Work panel's y-axis showed garbled, overlapping digit
strings (e.g. "00000011") before any key was ever pressed. Root cause:
cart-pole's force is genuinely, exactly 0.0 for as long as nothing drives it
(unlike the racket/pendulum's torque, which floating-point noise alone
keeps just barely nonzero) -- `drawTorquePanel`'s degenerate-range floor
(`Math.max(torqueMagnitude, 1e-9)`) then produces a ~1e-10 tick step, and
`axisTicks.js`'s `formatStepTick` had no cap on decimals, asking
`toFixed(10)` -- several such labels landing at nearly the same y-pixel row
read as garbled overlapping text. Fixed at the root (capped at 6 decimals in
`formatStepTick` itself, benefiting every panel that shares it, not just
this one) rather than special-casing cart-pole's call site, plus a
regression test reproducing the exact `1e-10`-step case.

**Verified live in the browser** (not just unit tests, per this project's
own standing rule that type-checks/unit-tests aren't sufficient on their
own): free-mode instability (5deg start, no control/wind -- grows and
correctly freezes at termination, ~0.4s, consistent with the ~0.25s
instability-time formula plus the "growth mode splits roughly evenly"
correction from the physics tests); Control-on manual driving (arrow-key
keydown produced an immediate, correctly-signed Force reading matching the
Force Limit slider exactly, with sensible Energy/Work panel response); Wind
on (no crash, no NaN, terminates on its own without any manual input, as
expected given the short instability timescale). Zero console errors
throughout. Reset/Run/Pause all confirmed working. Physics: 72/72 pytest
(unchanged this round). JS: 218/218 Vitest (+1 regression test for the axis
bug above).

**Next step**: get the coworker's actual `ct_sac` weights + architecture +
observation normalization (still blocked, per the kickoff entry), then wire
in `ctSacPolicy.js` as the system's third controller option -- everything
else in this round was built specifically so that addition wouldn't need to
touch the panels, diagnostics, or scene. Separately, still deferred by
choice: the gust rolling plot, "Last 20 Episodes" history, and Difficulty
Lambda's real formula.

## 2026-08-11 (cont.) — First hands-on feedback round: 4 fixes

User tried it live and reported 4 issues, all addressed:

1. **"Driving does not work."** Root cause, found by reasoning through the
   actual likely user flow rather than guessing: (a) unlike the racket/
   pendulum, this system did NOT auto-run on load (`resetRun()` at the
   bottom of the file instead of `runFromSidebar()`) -- a real inconsistency
   with the established site convention, fixed. (b) Even after that fix,
   toggling Control on does NOT itself restart the run on the racket/
   pendulum (confirmed: both call only `refreshSubfieldVisibility` on
   toggle, matching this project's existing "toggles take effect on the next
   Run click" convention) -- but cart-pole's Free mode terminates almost
   instantly (~0.25-0.4s, the genuine, already-verified physics), so by the
   time a user actually reaches for the arrow keys after flipping Control
   on, the auto-played free run from page-load has almost always already
   frozen. Toggling Control now explicitly calls `runFromSidebar()`,
   deliberately diverging from the other two systems' convention for this
   one control, with the reasoning written inline. Verified live: toggling
   Control now visibly restarts (t resets to 0), and an arrow-key press
   immediately produces a nonzero Force reading and a correct
   Work/Energy/diagnostics response.

2. **Starting Angle widened is wrong -- narrowed instead, to -90..90deg**
   (was -180..180). Not arbitrary: past +-90deg the pole's own tip is BELOW
   its pivot height, which for this system's flat, ground-level rendering
   means the pole visually dips through the track line in the pre-run
   preview -- a real "doesn't make physical sense" visual bug, not a taste
   preference. Confirmed the fix is exact, not just approximate: pole tip
   height above the pivot is `poleLength * cos(theta)`, which is >= 0 for
   all |theta| <= 90deg exactly.

3. **Display too small -- tightened the camera's worst-case framing.**
   `MAX_POLE_HALF_LENGTH` (drives `cameraHalfExtents`) dropped from 1.5 to
   1.0, and a new `CAMERA_MARGIN=1.05` (down from `cameraHalfExtents`'s own
   default 1.15) tightens the headroom -- both still leave genuine clearance
   at the new worst case, just less of it, since the old values left the
   common default geometry looking small/zoomed-out inside its panel.

4. **"Pole Half-Length" renamed to "Pole Length."** Half-length is an
   internal physics convention this project inherited directly from
   `summer-2026/cartpole.py`'s own `self.length` (itself inherited from
   Sutton's original code) -- meaningful to carry through the tested
   physics layer unchanged, but not something a sidebar user should have to
   think in. Converted at the UI boundary only
   (`poleHalfLengthFromSlider() = sliderValue / 2` in `cartpoleMain.js`);
   `cartpole_dynamics.py`/`cartpoleDynamics.js`'s own `l` parameter and
   every test that exercises it are untouched. New slider range 0.4-2.0m
   (full length), default 1.0m -- exactly double the old 0.2-1.5
   half-length range's own bounds, now at the tightened
   `MAX_POLE_HALF_LENGTH=1.0` from fix 3.

JS: 218/218 Vitest (unchanged -- all four fixes were UI/wiring, no new pure-
function surface). Verified live for all four; no console errors.

**Still open, flagged for the user rather than silently changed**: even
with fix 1, the underlying instability timescale (~0.25-0.4s at default
sliders) is objectively fast -- verified correct, matching both the from-
scratch physics derivation and the coworker's own demo's displayed number,
not a bug -- so successfully catching a fall in "you" mode still requires a
quick, correctly-directed push. Deliberately did not soften the default
physics (e.g. a gentler starting angle or weaker gravity) without checking
first, since that would trade away real physical accuracy for playability
without being asked to.

## 2026-08-12 (cont.) — The real ct_sac files arrived; steps 1-5 built, test-first

User reported the reaction window is still too small to catch a fall by
hand even with the toggle-restart fix above. **Decision: pause "you" mode
here rather than keep tuning it**, and pivot to the real controller -- the
coworker's actual files arrived (`POLICY_SPEC.md`, `export_policy.py`,
`cartpole_policy.json` -- the exported weights, 2.7MB/229,500 numbers --
and `cartpole_test_vectors.json` -- 24 observation->action pairs,
independently verified against live PyTorch, tolerance 1e-4 N).

**Verified the actual weights file against the spec before trusting either
one** (this project's standing practice: check, don't assume a doc matches
reality): loaded `cartpole_policy.json` directly and ran the exact forward
pass described in `POLICY_SPEC.md` against all 24 test vectors BEFORE
writing a single line into this repo. All 24/24 matched, worst error
2.13e-6 N against the stated 1e-4 N tolerance -- confirms both the
document and the weights are internally consistent and correct, not just
plausible-looking.

**Two facts from the spec that change how this mode has to be driven,
neither of which is a bug to fix:**
1. **Trained at dt=0.01**, not this project's other dt values -- the
   16-frame observation window encodes exactly 0.16s of history, and the
   policy's own learned dynamics are tied to that rate. `ct_sac` mode
   therefore needs its own fixed-cadence decision loop, distinct from
   `none`/`you` mode's continuous real-time stepping.
2. **Only ever seen one plant** (this project's own default mp/mc/l/g/
   force-limit/wind values, confirmed to match exactly) -- dragging the
   geometry/wind sliders while `ct_sac` is active is deliberately left
   live rather than locked, since watching the trained policy degrade
   out-of-distribution IS the coworker's own demo's whole point ("what
   changes is how far the problem has drifted from what they were built
   for").

**Files placed where they're actually used, not just dropped in**:
`cartpole_policy.json` -> `data/cartpole_policy.json` (Vite's configured
`publicDir`, so the browser can `fetch()` it at runtime with zero build
config changes) -- required adding a targeted exception to
`data/.gitignore`'s blanket `*.json` rule (`!cartpole_policy.json`),
documented inline: unlike every other file in `data/`, this one is an
external asset that must actually be committed, not a regenerable
`physics/export.py` output. `cartpole_test_vectors.json` ->
`physics/tests/fixtures/`, the same cross-validation-fixture location
every other module in this project already uses. `POLICY_SPEC.md`/
`export_policy.py` were NOT copied in (the exporter depends on the
coworker's separate training codebase and can't run here) -- cited by
name/path for provenance instead, with their key facts folded into
`cartpole_policy.py`'s own docstring.

**Built steps 1-5 of the agreed plan, test-first, all pytest/Vitest
green:**
- `physics/cartpole_policy.py` / `cartpolePolicy.js`: the forward pass
  (Linear+ReLU body, bare-linear `body.4` -- the one documented gotcha --
  bare-linear `mu` head, tanh+rescale) plus the observation-window helpers
  (`init_window`/`push_frame`/`window_to_observation`). **Deliberately
  skipped the usual NotImplementedError-stub red phase for the forward
  pass itself** -- the exact same logic was already verified ad hoc
  against these same 24 vectors (see above) before being written as the
  real implementation, so staging a fake red phase for code already known
  correct would have been theater. The window helpers ARE genuinely new
  logic and got real tests.
- `cartpoleScenarioRunner.js`: `initCtSacState`/`advanceCtSac`, a
  fixed-timestep accumulator (standard game-loop pattern) running exactly
  as many whole `CT_SAC_DT=0.01` ticks as have accumulated per call --
  each tick samples the window, runs one policy forward pass, takes one
  full `heunStep` at `dt=CT_SAC_DT` with that force held constant (matching
  how the training env itself stepped -- one Euler-Maruyama/Heun step per
  decision, not further subdivided), then pushes the new measurement into
  the window. Stops early on termination mid-loop rather than consuming
  the rest of the accumulated ticks.
- Tests use a **trivial constant-output fake policy** (all-zero weights,
  `low===high` so tanh's nonlinearity collapses to one known number
  regardless of input) to hand-verify the tick mechanics against directly-
  chained `heunStep` calls -- same "inject a known/fixed value instead of
  the real complex thing" pattern as `cartpoleWind.test.js`'s
  `_FixedGenerator`.
- **One test-design mistake caught by running the numbers, not assumed
  correct:** the termination early-stop test first assumed a huge constant
  force at a near-threshold starting angle would terminate on the very
  FIRST tick. Numerically traced the actual trajectory (`node` one-liner)
  before asserting anything and found it actually swings theta toward the
  OPPOSITE boundary (through zero) and crosses on tick 4 of 5, not tick 1 --
  a genuine, sensible physical result (force couples into theta_dot with a
  particular sign), not a bug. Fixed the test to check the general
  property (stopped before consuming all of `dtReal`) instead of a
  specific miscounted tick number.

Physics (Python): 81/81 pytest (+9 new). JS: 235/235 Vitest (+17 new).

**Next step (paused here per the user's own request)**: wire `ct_sac` into
the sidebar as a real third Controller option (`none`/`you`/`ct_sac`,
replacing the current boolean Control toggle) -- a UI change, discussed
before building per the user's standing preference to go slow on UI/
wording.

## 2026-08-12 — Pausing "you" mode, pivoting to the real ct_sac weights

**Status check-in after the fixes above**: the auto-restart-on-toggle fix
made Control actually responsive, but the user reports the reaction window
is still too small to catch a fall by hand even so -- consistent with the
"still open" note left above (the ~0.25-0.4s instability timescale is
verified-correct physics, not a bug, but that doesn't make it humanly
playable). **Decision: pause manual/"you" mode here rather than keep tuning
it, and pivot to the real controller** -- the user now has the actual files
from their coworker (the `ct_sac` checkpoint/weights, architecture, and
observation-normalization details) that Phase 2 (see the 2026-08-11 kickoff
entry) was blocked on. Priority order flipped: get the "normal" experience
(Free vs. an actually-controlled mode, matching the racket/pendulum's own
pattern) working with the REAL trained policy first, then return to
manual driving as a stretch/secondary mode once the primary experience is
solid -- not abandoning "you" mode, just deprioritizing it.

**Not yet started**: still waiting on the user to actually hand over the
coworker's files in this conversation (checked the project directory and
this session's scratchpad first -- nothing new landed yet, so nothing to
build against until they're actually shared). Once they land, next steps
per the original Phase 2 plan: confirm the policy's exact architecture
(layer sizes, activation(s), output squashing/rescaling) and observation
normalization directly from their source rather than guessing, then write
`ctSacPolicy.js` as a small hand-written forward-pass (Python reference +
JS port, test-first, cross-validated fixtures -- same process as every
other physics module in this project), and wire it in as the `Control`
toggle's real behavior (replacing/supplementing the manual "you" mode that
currently occupies that slot).

## 2026-08-12 (cont.) — Controller selector wired in: None / Manual / ct_sac

Sketched the 3-way selector as an interactive mockup before touching real
code (per the user's standing preference to go slow on UI). Two decisions
made there, carried into the real build:

- **Segmented 3-button control, not a boolean switch or a dropdown** --
  mirrors the coworker's own live demo's control layout exactly, and with
  only 3 always-valid options there's no reason to hide two behind a click.
  New `.segmented`/`.seg-btn` CSS in `index.html`.
- **"You" renamed to "Manual"** (user's own request, after brainstorming
  alternatives: Manual/Drive/Keyboard/Pilot) -- standard control-systems
  terminology, reads cleanly next to "None"/"ct_sac" without being cute.
  `controllerMode` internally is `"none" | "manual" | "ctsac"`.
- **`ct_sac` button disabled until the weights finish loading** (a 2.7MB
  fetch, momentary but not instant) -- selecting a mode with nothing to
  run would silently do nothing, indistinguishable from broken. Sketched
  and confirmed via an interactive preview before building the real
  fetch-then-enable wiring in `cartpoleMain.js`.

**Real architecture change in `cartpoleMain.js`'s `animate()`**: it now
branches per mode -- `none`/`manual` share the existing continuous real-
time `advance()` path (force is knowable instantaneously either way), while
`ctsac` uses the new fixed-cadence `advanceCtSac()` path, maintaining its
own bundled `{z, window, accumulator, force, workDelta}` state
(`ctSacState`) kept in sync with the single `z` every other mode/the
scene/history code reads. `advanceCtSac` gained a `workDelta` return field
(the work done during just that call, using the same `dH/dt = F*x_dot`
identity as `advance`'s callers, but evaluated per-tick instead of per-
rendered-frame -- more natural here since the force is exactly constant
for each whole tick by construction) -- 2 new tests added for it.
Fragility Diagnostics now gates on `controllerMode !== "none"` (either
engaged controller, not just one specific mode).

**Verified live, with one real debugging trail worth recording.** Selecting
`ct_sac` immediately produced correct, sensible values -- Force: 5.513N,
then 1.146N, Work accumulating, Fragility Diagnostics all populated,
panel label "ct_sac", zero thrown console errors. But the simulation then
appeared to freeze (`t` stuck, "Survived: 0.44s (ongoing)" never updating)
even after several more seconds of real waiting. Debugged systematically
rather than guessing:
1. Installed a `window.error`/`unhandledrejection` trap directly in the
   page -- caught nothing, ruling out a silent exception.
2. Confirmed the main thread wasn't blocked (an infinite loop inside
   `advanceCtSac` would hang synchronous JS execution) -- other JS
   evaluations kept succeeding fine.
3. Patched `window.requestAnimationFrame` to count calls -- 0 calls in a
   2-second window, even after explicitly fronting the tab
   (`tabs_select`).
4. Checked `document.hidden`/`document.visibilityState` directly: `true`/
   `"hidden"`, despite `document.hasFocus()` reporting `true` and this
   being the tool's only, active tab. Reproduced identically on a brand
   new tab.

**Conclusion: a characteristic of this Browser-pane tool's environment**
(pages render with `document.hidden = true` regardless of the tool's own
tab-activation bookkeeping), not an app bug -- Chrome suspends/throttles
`requestAnimationFrame` for hidden documents, which is exactly the stall
observed, and matches (extends) the RAF-throttling gotcha already recorded
in the 2026-08-05 entry (that one was fixable by fronting the tab; this one
isn't, at least not by any method tried). A real user's actual browser tab
has a genuinely correct `visibilityState`, so this shouldn't reproduce
there. Recorded here so a future verification attempt in this same tool
doesn't waste time re-diagnosing the same environment limitation, and
doesn't mistake it for a real animate()-loop bug.

Physics (Python): 81/81 pytest (unchanged this round -- no physics changes,
only UI/wiring). JS: 237/237 Vitest (+2 for `workDelta`).

**Next up**: none of the site's 3 systems' Controller UIs have been asked
about together yet -- the racket/pendulum's own Control toggle is still a
boolean, unlike this system's new 3-way selector. Not proposing to unify
them (per the explicit no-shared-abstraction convention), just noting the
inconsistency exists across systems now, in case it's ever worth asking
the user about.

## 2026-08-12 (cont.) — ct_sac's "oscillates, settles, back and forth" behavior: confirmed real, not a bug

User's own hands-on report after the controller selector landed: watching
`ct_sac` run, it oscillates, settles, then keeps going back and forth
rather than coming fully to rest. Investigated rather than assumed either
way (bug vs. expected):

Ran two independent 60-second CLOSED-LOOP simulations directly in `node`
(bypassing the browser/rendering entirely -- a `node -e` one-liner calling
`initCtSacState`/`advanceCtSac` exactly as `cartpoleMain.js` does), one
with wind off, one with wind on matching the coworker's own demo defaults
(sigma_gust=0.002, sigma_turb=0.001):
- **theta (pole angle) stayed bounded the whole run in both cases** --
  max ~0.087 rad (~5deg) in the wind-off run, ~0.089 rad (~5.1deg) with
  wind on -- essentially never exceeding its own 5deg starting tilt, and
  never once crossing the 12deg failure threshold in either 60s run.
- **x (cart position) wandered substantially in both cases** -- up to
  +-1.6m (wind off) and +-1.9m (wind on), well inside the +-2.4m track,
  continuously drifting back and forth rather than re-centering.
- Also opened the coworker's own live demo directly (same checkpoint) and
  watched it run for ~15s: same qualitative signature -- small pole tilt,
  visible cart drift, no failure.

**Conclusion: this is genuine, reproducible behavior of the trained
policy, not a porting bug.** The pole staying up is the actual invariant
being maintained; cart position drifting is weakly regularized (or not
explicitly rewarded to re-center) in whatever the checkpoint was actually
trained on, so the policy satisfices on "don't tip over" without bothering
to recenter x as long as it stays inside the track. What reads as "keeps
going back and forth" from the flat side-on camera view is this ongoing
drift-and-correct cycle, not repeated near-failures. No code change made
-- confirmed correct via direct simulation rather than either dismissing
the report or chasing a bug that isn't there.

Worth folding into the upcoming aesthetics/wording pass: this is a genuinely
interesting, presentable characteristic of the real trained policy (a
concrete illustration of reward under-specification -- "the policy learned
exactly what it was rewarded for, and re-centering wasn't part of that"),
not just a footnote to explain away.

## 2026-08-12 (cont.) — First real-Safari feedback: 3 fixes, scoped to "ship the regular modes first"

User confirmed the build works in their own Safari and gave 3 concrete
fixes, explicitly scoping out Manual mode ("we will continue to edit later")
and asking to get None/ct_sac ship-shape first:

1. **"Difficulty Lambda: pending" removed entirely**, not just left as a
   placeholder -- the user confirmed they never got the real formula from
   the coworker, so showing a permanent "pending" forever is worse than not
   showing the line at all. `formatDiagnostics()` now only shows
   Instability Time / Gain A / Gain B / Survived.

2. **The Force & Work axis-label bug was only half-fixed by the earlier
   `formatStepTick` decimal cap.** That fix stopped the GARBLED overlapping
   digits, but the user's screenshot showed the real remaining problem: a
   wall of a dozen-plus identical "0.000000" rows stacked on top of each
   other, still clearly broken-looking even though no longer garbled.
   Root cause (same family as before, deeper this time): a genuinely-zero
   data range still drives `computeNiceStep` into a ~1e-10-scale major/minor
   step, which subdivides the tiny range into dozens of ticks -- capping
   decimals stopped the GARBLING but not the redundant repetition. Fixed at
   the actual source this time: extracted `computeTorqueDomain` (new,
   tested pure function in `ControlMetrics.js`) which falls back to a fixed
   `[0, 1.0]` range when the max magnitude is at or below a 1e-6 "is this
   actually zero" threshold -- comfortably below any real torque/force value
   this app ever produces, so the racket/pendulum's own genuinely-small-but-
   nonzero torque behavior is completely unchanged (confirmed via a test
   asserting a tiny-but-real value like 0.001 still scales normally, not
   just the exactly-zero case).

3. **Cart enlarged and given wheels that dip below the rail**, per direct
   feedback that the pole "cannot go under the cart" (confirmed: the +-90deg
   Starting Angle limit already guarantees the pole tip never drops below
   the pivot height, so there's no geometric conflict with extending the
   cart's own body downward) and the whole assembly read as too small in
   its panel. `CART_WIDTH` 0.4->0.55, `CART_HEIGHT` (still the PIVOT height
   -- unchanged meaning, so camera framing/track decoration/
   cartpoleMain.js's camera position all just work off the same constant
   with no other code changes needed) 0.24->0.34. Body now occupies
   `[WHEEL_RADIUS, CART_HEIGHT]` instead of `[0, CART_HEIGHT]`, with two new
   wheel circles (`THREE.CircleGeometry`) centered exactly on the rail
   (y=0, so they visibly dip to `-WHEEL_RADIUS` below it) and positioned
   just in front of the body's own front face so they're never partially
   hidden behind it. 4 new/updated tests in `CartPoleScene.test.js`
   (body-top-at-pivot-height, wheel positions/dip, body-bottom-meets-wheel-
   top with no gap or overlap) -- confirmed the camera's existing worst-case
   margin already had enough headroom below the rail for this without
   needing any framing changes (verified by the existing pole-swing-circle
   test still passing unchanged).

**Verification note**: tried to visually confirm the wheels via ad hoc
canvas-zooming hacks (CSS transform/scale tricks) in the Browser pane, but
these introduced their own transform-math and scaling artifacts that made
them unreliable for fine visual QA -- abandoned that approach in favor of
trusting the automated geometry tests (which check exact positions/radii
directly, not by eyeballing a screenshot) and asked the user to confirm the
actual look themselves in their own browser, rather than presenting a
possibly-misleading zoomed screenshot as if it were confirmed correct.

Physics: 81/81 pytest (unchanged -- no physics touched this round). JS:
242/242 Vitest (+5: 3 for computeTorqueDomain, 2 updated + 2 new for
CartPoleScene's wheels/sizing... see test file for the exact count).

**Explicitly still deferred, per the user's own scoping**: Manual mode's
reaction-window difficulty. Not forgotten -- just sequenced after "regular"
modes (None/ct_sac) are fully ship-shape.

## 2026-08-12 (cont.) — Corrected misread: camera framing, not cart geometry

User corrected the previous entry's fix #3: the ask was never a literal
"wheels dip below the rail" -- that was this assistant's own
misinterpretation of "bring it down... wheels are lower". The actual ask:
move the whole rail+cart+pole system down within its panel, and make the
whole thing bigger. Reverted the wheel-dip cart geometry back to a plain
box (`[0, CART_HEIGHT]`, no separate wheel meshes) and fixed the real
issue instead: **camera framing**.

**Root cause of "looks small and stuck in the middle"**: `cameraHalfExtents`
framed a full circle of radius `maxPoleLength` around the pivot, symmetric
above AND below it -- but the +-90deg Starting Angle limit (added earlier
this same day, precisely to keep the pole from visually dipping below its
own pivot) already guarantees `cos(theta) >= 0` for the entire reachable
range, so the pole's tip height is ALWAYS >= the pivot height. The full
lower half of that circle was framed space that could physically never be
used -- exactly a pole-length's worth of dead space below the rail, which
is what pushed the rail toward the panel's vertical center and made
everything look small.

**Fix: made the vertical framing asymmetric on purpose.** `cameraHalfExtents`
now returns `{halfWidth, top, bottom}` instead of `{halfWidth, halfHeight}` --
`top = maxPoleLength*margin` (unchanged, full headroom for the pole),
`bottom = CART_HEIGHT*margin` (new -- only enough to show the cart itself).
In the binding (height-constrained) case this HALVES the total framed
vertical extent, a genuine 2x zoom-in, not just a cosmetic tweak.
`cartpoleMain.js`'s `resize()` letterbox-fit logic updated to match:
`bottom` is always held fixed at its required value regardless of aspect
ratio (this is what keeps the rail's position in-frame consistent), and any
slack from the canvas's own aspect ratio always expands `top` or
`halfWidth`, never `bottom` -- still a pure letterbox (never crops), just
no longer symmetric.

Updated `CartPoleScene.test.js` to match: removed the wheel-specific tests
(geometry reverted), rewrote the `cameraHalfExtents` tests for the new
asymmetric shape -- including a new explicit check that `top/bottom > 3`
(the framing is genuinely lopsided, not just slightly adjusted) and that
the reachable half-circle (`|theta| <= 90deg`, not the old full circle)
stays within `top` with real margin to spare.

Verified live: the rail now sits low and consistently in the panel with
real headroom above for the pole, and the whole assembly reads
noticeably bigger -- confirmed via a clean-reload error trap (zero errors)
rather than screenshot-eyeballing, since the earlier round's ad hoc canvas-
zoom screenshots had already proven unreliable for this kind of visual
check in this tool.

JS: 241/241 Vitest (net unchanged count -- 2 wheel tests removed, 4
cameraHalfExtents tests added/rewritten in their place). Physics
unaffected.

## 2026-08-12 (cont.) — Second correction: wheels back, and the REAL sizing fix

User's actual screenshot made clear the previous round still wasn't it:
they DO want wheels visible (this assistant over-corrected by removing
them entirely, misreading "I don't want wheels below the rail" as "remove
wheels"), and -- the real substance of this round -- the cart+pole was
still much smaller than intended. Explicit target given: "cart and upright
pole should fill the display to 80% if not more... it's ok if we have to
allow the display to scroll... we have a lot of space we're not using."

**Re-added wheels, this time flush with the rail (not dipping below, not
absent).** Wheel center at `y=WHEEL_RADIUS` (so the wheel spans
`[0, 2*WHEEL_RADIUS]` -- bottom edge exactly at the rail, never past it).
Body now occupies `[2*WHEEL_RADIUS, CART_HEIGHT]`. One real bug caught by
the test suite before it shipped: the first draft set the body's bottom
edge at the wheel's CENTER (`WHEEL_RADIUS`) instead of its TOP
(`2*WHEEL_RADIUS`), silently overlapping the top half of each wheel with
the bottom of the body -- caught by
`test("body bottom edge sits exactly at the wheels' top...")` failing with
an exact 2x discrepancy, not a vague near-miss, which is what made the
off-by-a-factor-of-two bug obvious immediately.

**The actual size fix: found and removed the real waste.** Root cause
this time (distinct from the previous round's rail-position fix):
`cameraHalfExtents` was still being called with a fixed
`MAX_POLE_HALF_LENGTH` constant sized for the Pole Length slider's
maximum (2.0m) -- so at the DEFAULT pole length (1.0m), the pole only
ever filled HALF of the vertical space the camera was framed for, no
matter how the rail's own position was fixed up. This is a genuine,
deliberate departure from this app's usual "static worst-case framing,
never recompute" convention (used by the racket, pendulum, and this
system's own horizontal framing) -- confirmed acceptable specifically
because the user explicitly asked for it, trading "never needs
recomputing" for "always looks appropriately large at whatever the
sliders are currently set to."

Implementation: `cartpoleMain.js` now recomputes `REQUIRED` (renamed
internally accurate: it's the CURRENT requirement, not a fixed maximum)
every time the Pole Length slider fires its "input" event -- same moment
`resizePole`'s existing live-preview call already fires, so no new event
wiring, just one more thing done at that moment. `CartPoleScene.js`'s
`cameraHalfExtents` renamed its own param from `maxPoleHalfLength` to
`poleHalfLength` to match this changed contract, with the docstring
rewritten to explain the trade-off explicitly rather than leaving the
"why does this look different from every other camera in this app"
question unanswered for a future reader.

**A clean, testable consequence, verified before committing to a margin
value:** because both `top` and `bottom` scale off the exact same
`margin`, and the rendered content's real height is always exactly
`poleLength + CART_HEIGHT`, the fraction of the frame actually filled
works out to precisely `1/margin` -- independent of whatever the
geometry sliders are set to. Confirmed with a quick throwaway Python
calculation before picking a value (not guessed): margin=1.2 gives
exactly 83.3% fill, comfortably inside "80% if not more" at any pole
length or cart size, not just the current defaults. New test asserts this
identity directly at two very different pole lengths.

Also, per the same screenshot's implicit ask, doubled `CART_WIDTH`
(0.55->1.1) and `CART_HEIGHT` (0.34->0.68) again, per "at least double the
size of the cart."

Verified live: zero console errors on a clean reload (error-trap method,
not screenshot-eyeballing, per the same reasoning as last round), and a
direct screenshot this time DOES clearly show both the enlarged cart and
its wheel circles sitting on the rail -- worth actually looking at rather
than only trusting the geometry tests, now that the fill-fraction identity
gives high confidence the numbers are right independent of the visual
check.

JS: 244/244 Vitest (+3 net this round). Physics unaffected.

**Meta-note for future rounds**: two corrections in a row on this same
"make the cart-pole bigger" ask, both from initially solving the wrong
layer of the problem (cart geometry vs. camera framing vs. worst-case
framing basis). Worth explicitly re-confirming the CONCRETE target (a
percentage, a comparison screenshot) before implementing, rather than
inferring intent from a short phrase like "bring it down" -- which is
exactly what the user's own follow-up ("cart and upright pole should fill
the display to 80%") did, and which resolved this in one pass once
supplied.

## 2026-08-12 (cont.) — Third hands-on round: three more fixes, incl. the real panel-layout root cause

Three issues from one more real-Safari round:

**1) ct_sac auto-started on selection, instead of waiting for Run.**
Root cause: `selectControllerMode` (the sidebar's Controller segmented
control) called `runFromSidebar()` directly, from an earlier round's fix
for a *different* bug (restarting a run that had already terminated when
switching modes) -- but that fix over-reached: it restarted on EVERY mode
click, not just a terminated one, breaking the "selecting a controller is
just a selection, Run is what starts it" contract every other mode (and
every other system) follows. Fixed by removing that call entirely from
`selectControllerMode`; introduced `activeControllerMode`, a snapshot of
`controllerMode` taken once in `resetRun()`, to drive the live run
(`animate()`'s stepping branch, `currentForce()`'s manual-input gate, the
panel label, the diagnostics-visibility gate) -- so switching the
sidebar's selection mid-run no longer risks calling `advanceCtSac` against
a `ctSacState` that's still `null` (the crash `runFromSidebar()` was
papering over), and so a genuinely-still-running "None" run is left alone
if the user clicks "ct_sac" without also clicking Reset/Run, exactly as
seen live (t kept ticking, label stayed "Free").

**2) Force & Work panel showed nothing at all in "None" mode --
looked broken, not "waiting for input."** `drawPanels()` used to skip
`drawTorquePanel` entirely whenever `controllerMode === "none"`, blanking
the canvas instead. But zero force IS real data (nothing is pushing, which
is correct and worth showing with axes), not an absence of data --
`computeTorqueDomain`'s existing all-zero fallback (`[0, 1.0]`, from the
earlier "wall of 0.000000 labels" fix) already handles this domain
correctly. Now `drawTorquePanel` always runs, in every mode; only the
`Work:`/`Force:` text overlay (`updateMetricsText`) stays gated on
`activeControllerMode !== "none"`, since a literal "Work: 0.000 J" readout
before anything has run is genuinely uninteresting, unlike the axes
themselves.

**3) The cart still looked far too small ("double it again") -- and this
time the actual root cause was neither cart geometry nor camera framing
(both already correct/tested), but the PANEL SHAPE itself.** Quantified
before touching anything: `cameraHalfExtents` needs `halfWidth` to
comfortably include the full +-2.4m track, which drives a required
width:height aspect ratio of ~3.7-6.2 (shorter poles need relatively more
width margin) depending on the Pole Length slider. The cart-pole scene
panel, though, was sharing the same plain 2x2 `.panels-grid` as every
other system, giving it a roughly square-ish real aspect (~1.5-2.6
measured live) -- nowhere near wide enough, so the camera was forced to
zoom out far past what the cart+pole alone need, just to keep the track's
ends in frame. This is *why* the fill-fraction identity from the previous
round (`1/margin`, 83.3% at margin=1.2) never showed up live: that
identity only holds when height is the binding constraint, and a
square-ish panel against a ~4-6:1 required aspect means width is always
binding instead, capping real fill around 27-45% no matter how the camera
math is written.

Fix: `#cartpole-panels` gets its own grid override (`.cartpole-panels-banner`,
scoped by ID so it doesn't touch the racket/pendulum's shared `.panels-grid`),
reflowing from a plain 2x2 into a 3-column layout with the scene panel
spanning the full top row as a short, wide banner (`grid-template-rows: 1fr
2fr`) and Energy/Angle/Force sharing the row below, three across instead of
two. Measured live: scene canvas aspect went from 2.56 (naive 3fr/2fr split,
first attempt) to 5.13 (1fr/2fr) -- above the required aspect (4.70) at the
default Pole Length, which flips the binding constraint to height and
actually delivers the `1/margin` = 83.3% fill the previous round's math
promised but the panel shape was silently preventing. Confirmed at the
slider's extremes too, not just the default: 83.3% at Pole Length >= 1.0m,
tapering to ~69% at the slider's minimum (0.4m, the most width-hungry
case) -- still a large improvement over the old ~27-35% range, and no
setting requires scrolling to see the full track.

Verified live end-to-end: screenshot at default settings shows the cart
filling most of the banner with the track visible edge-to-edge; Force &
Work panel shows axes and a real "Work: 0.000 J / Force: 0.000 N" readout
immediately after Reset in ct_sac mode, before Run is pressed; clicking
ct_sac mid-run left an in-progress "None" run untouched (t kept advancing,
label stayed "Free"); Reset-then-Run under ct_sac correctly waits at
t=0.0s until Run is clicked, then drives normally.

JS: 244/244 Vitest (unchanged -- this round's panel-layout fix was pure
CSS/HTML, no new logic to test; the `activeControllerMode` and
always-draw-torque-panel changes were exercised live rather than adding
new unit tests, since both are UI-wiring behavior already covered
indirectly by existing `drawTorquePanel`/`computeTorqueDomain` tests).

## 2026-08-12 (cont.) — Correction: the banner layout was wrong; chase-cam instead

User feedback on the banner layout, with a screenshot of the windy
pendulum's plain 4-panel grid for comparison: "the point was to have four
panels like this. Can we resize the panel so it's back that way? ... I'm
ok if we have to scroll the background as the cart moves, but now it
looks unnatural." Correct call -- the banner made cart-pole the one
system on the whole site whose panel grid didn't match the others, and
the previous entry's own trade-off analysis had already surfaced
scrolling as an option (the user had said as much even earlier: "it's ok
if we have to allow the display to scroll") without my actually
pursuing it in favor of reshaping the grid instead. This round properly
takes that earlier offer at face value.

**Reverted**: `#cartpole-panels`/`.cartpole-scene-wrap`/
`.cartpole-panels-banner` all removed from `index.html`; cart-pole is
back on the exact same plain 2x2 `.panels-grid` as the racket and
pendulum.

**The actual fix: a chase-cam, not a scrollbar.** Re-examined what
"scroll the background as the cart moves" most naturally means -- not a
manually-dragged scrollbar on a wider-than-the-panel canvas, but the
camera panning to keep the cart centered as it translates along the
rail, the same convention as any side-scrolling game's camera. This
also directly resolves the root tension from the previous entry: the
ONLY reason `cameraHalfExtents`'s `halfWidth` ever needed to be so large
(driving the ~4-6:1 required aspect ratio no plain grid cell can match)
was showing the ENTIRE +-2.4m track at once, at every cart position
simultaneously. A camera that instead follows the cart never needs that
-- it only ever has to frame the cart+pole's own extent, which is a
completely different, much more modest, aspect requirement (~1.6-2.5:1
depending on Pole Length), comfortably satisfied by a plain, roughly
square grid cell.

Implementation: `cameraHalfExtents({poleHalfLength, margin})` dropped its
`xThreshold` parameter entirely and no longer adds it to `halfWidth` --
the function now only knows about the cart+pole, never the track's
absolute extent (`CartPoleScene.js`'s docstring rewritten to explain
this). `cartpoleMain.js` gained `updateCameraChase(x)`, called every
frame right alongside `updateCartPoleFrame` (all 3 call sites: the
paused-preview path, `resetRun()`, and the live `animate()` loop) --
sets `cartpoleCamera.position.x = x` (a pure horizontal pan; THREE's
ortho `left`/`right` are offsets relative to the camera's own position,
so they don't need to change). The track/boundary-marker decoration
(`createTrackDecoration`) is untouched -- it's still drawn across the
full +-2.4m in world space, it just now scrolls past naturally as the
chase-cam pans, including the boundary tick marks becoming visible
exactly when the cart actually approaches them, which if anything
communicates the failure boundary better than always having it in frame
from the start.

Verified live: at the default Starting Position (0), the cart fills the
panel similarly to before. Set Starting Position to 2.0 (near the
+-2.4m boundary) and Reset -- the cart is rendered PERFECTLY centered in
the panel regardless, confirming the pan tracks the cart's live position
rather than assuming it starts at 0. (Animating a live run to watch the
background visibly scroll frame-by-frame hit this tool's own
already-documented `document.hidden`-linked RAF throttling -- see the
existing note on this -- so this was confirmed via the reset path
instead, which exercises the identical `updateCameraChase` call.)

Test suite: rewrote the `cameraHalfExtents` describe block for the new
content-only contract -- dropped all `xThreshold` args from existing
assertions, and replaced the old "half-width exceeds the track boundary"
test (no longer a meaningful thing to assert) with one confirming the
new `halfWidth` is far smaller than the old xThreshold-inclusive formula
would have produced for the same pole length. All other invariants (top
scales with current pole length, bottom is cart-only and much smaller
than top, the reachable +-90deg swing stays framed, the `1/margin` fill
identity) still hold under the new formula and needed no logic changes,
only dropping the now-removed parameter from their calls.

JS: 245/245 Vitest (+1 net: replaced one test, added one new one for the
dropped parameter).

## 2026-08-12 (cont.) — Physics discussion: why extreme starting angles fail, and whether "hold at an arbitrary angle" is possible

User question, discussed before touching any code: why doesn't starting
the pole at a more extreme angle and letting ct_sac run show it actually
*trying* to recover? And separately, could we add a "target angle" so it
holds some arbitrary tilt instead of always upright?

**Empirically checked, not guessed:** ran the real checkpoint closed-loop
via a throwaway script calling `advanceCtSac` directly (deleted after),
first with the normal termination check, then with it removed entirely
just to watch the raw dynamics:

- 5-12deg: recovers cleanly, holds for the full run (matches everything
  already known about the cart-drift behavior).
- **15deg: still recovers** -- theta reaches ~0 by 0.5s and stays there,
  slightly beyond the policy's own stated +-12deg training envelope.
- **20deg: falls off a cliff** -- initial force is weak (5.9N, not even
  saturated), theta blows through 42deg by t=1s and keeps climbing into
  uncontrolled full rotations (232deg, 300deg...), cart position diverges
  past +-19m with the termination check removed. 30-90deg: same collapse.

This matches `POLICY_SPEC.md` exactly: *"the pole is confined to +-12deg
over a real episode... cos(theta) never leaves [0.9989, 1.0]... don't fix
it, the network was trained that way."* Past ~15-20deg the input is so far
outside the training distribution that the output stops being a
controller and becomes close to noise -- a real cliff in the weights, not
a bug in the port.

Separately, identified TWO independent reasons extreme angles looked
broken before this: (1) `THETA_THRESHOLD_RADIANS` (12deg, pinned from the
coworker's gym env, `cartpoleDynamics.js`) is the actual termination
boundary, but (2) the Starting Angle slider went to +-90deg -- a range
chosen purely for camera-framing reasons several rounds ago, unrelated to
what's survivable. Anything started past 12deg was already failing before
the sim took a single step, which is what "doesn't even try" actually was.

**Target angle, discussed and deferred (not enough time for a new
controller, but the physics was worth working through):**
1. The existing ct_sac network can't be repointed at a nonzero target --
   its 48 inputs are `(cos theta, sin theta, x)` with no target channel;
   the weights were optimized purely for theta->0. Feeding it
   `(cos(theta-T), sin(theta-T))` as a hack is physically wrong: gravity's
   torque depends on the true absolute angle, not a shifted proxy, so the
   network would think it's balanced while actually falling.
2. A literal static hold at any theta != 0, pi is impossible on this rail
   regardless of controller: holding theta constant requires a sustained
   constant cart acceleration (a pseudo-gravity term canceling real
   gravity's torque at that angle, the same idea as a pendulum tilted
   inside a constantly-accelerating train car) -- constant acceleration
   means x grows without bound, which a finite +-2.4m rail can't sustain.
   Upright and hanging are the system's only two true equilibria; this is
   structural (one horizontal-force actuator, no direct pivot torque), not
   a software gap. Left genuinely open: whether an oscillatory strategy
   (cart swinging, pole wobbling around a nonzero mean angle) could hold
   the *average* angle with bounded x -- not checked either way.
3. Conclusion: a real "target angle" feature needs a new, purpose-built
   controller, not a tweak to ct_sac's frozen weights. Out of scope for
   now; parked as a real idea if there's ever time for it.

**Decided, and implemented this round:** narrow the Starting Angle slider
from +-90deg to +-20deg (`index.html`) -- past 90 was never meaningful
(camera-driven, not physics-driven) and now the whole range is at least in
the neighborhood of what's survivable. Added a standing warning readout
(`#cartpole-angle-warning`, styled in the site's existing `--unstable`
color, not a plain `.readout`) that appears whenever
`|Starting Angle| > 15` -- the user's own chosen cutoff, sitting between
the empirically-confirmed-good 15deg and the confirmed-collapses 20deg.
Worth flagging explicitly: the warning's 15deg cutoff and the actual hard
termination boundary (12deg) are two different numbers by design -- the
warning is about ct_sac's trained envelope specifically (which tolerates a
few degrees past 12 in practice), not a restatement of the environment's
own pass/fail line, and it shows regardless of which Controller is
selected (None/Manual don't care about ct_sac's training distribution,
but the slider is shared and the user may switch modes after setting it).

`CartPoleScene.js`'s `cameraHalfExtents` docstring updated to stop
referencing the old +-90deg range (stale after this change) -- the
underlying `cos(theta) >= 0` assumption is unaffected, just even more
comfortably true now with a narrower slider.

JS: 245/245 Vitest (no test changes needed -- this was a slider-range/UI
change plus a physics discussion, not new library logic).

## 2026-08-12 (cont.) — Relax the termination boundary so a ct_sac failure is actually watchable

Follow-up to the previous entry's own flagged caveat: even with the
Starting Angle slider widened to +-20deg and the >15deg warning added,
anything past the real hard boundary (THETA_THRESHOLD_RADIANS, 12deg)
still insta-terminated at t~0 -- so you'd never actually get to WATCH a
15-20deg run diverge, only see it freeze immediately. User asked to relax
this so the failure plays out on screen.

**Design: two thresholds, not one.** `THETA_THRESHOLD_RADIANS` (12deg) in
`cartpoleDynamics.js` is left completely unchanged -- it's the real,
citable training/task boundary (still exactly what the angle-warning text
and `POLICY_SPEC.md` reference), not something to fudge. Instead:
- `isTerminated(z, thetaThreshold = THETA_THRESHOLD_RADIANS)` and
  `advanceCtSac(..., thetaThreshold = THETA_THRESHOLD_RADIANS)` both gained
  an optional trailing param (default unchanged, so every existing call
  site/test needed zero changes).
- `cartpoleMain.js` defines its own `VISUAL_THETA_THRESHOLD_RADIANS = 45deg`
  and passes it at both of its call sites (the outer `animate()` check and
  the `advanceCtSac` call) -- this is what the app actually treats as "run
  over," independent of what the trained network's own pass/fail line is.

**Why 45deg specifically:** confirmed against the same empirical probe
from the previous entry -- at a 20deg start, theta is still only ~43deg by
t=1.0s (not yet in the uncontrolled-rotation regime, which only kicks in
around t=1.5s+), so 45deg lets a real divergence play out for close to a
full second before ending, long enough to see the Angle/Force panels
actually climb rather than freeze on frame one. It also stays safely
inside the camera's `cos(theta) >= 0` assumption (valid to 90deg, with the
narrowed +-20deg Starting Angle slider giving even more headroom) -- so
the pole tip never dips below the pivot and needs no camera changes, and
it's comfortably below the point where the render would start showing
full uncontrolled rotations (visually confusing, and not what "watch it
fail" was asking for).

Verified live: Starting Angle=20, ct_sac, Run -- the pole visibly leans
further and further (Angle panel climbing, Force panel saturating near
+-10N, Energy panel climbing well above the upright reference) for about a
second, then freezes with "Terminated — survived 1.05s" in the Fragility
Diagnostics readout. Exactly the "attempt, then visible failure" the
slider-widening round couldn't actually show on its own.

Test-first: added a test confirming `isTerminated` accepts and honors the
override (default behavior provably unchanged), and a paired
`advanceCtSac` test reusing the existing "stops early on termination" scenario's exact setup -- with the default threshold it stops after 4 of
10.5 possible ticks (confirmed numerically, not assumed, same as the
existing test's own diligence), and with a wide override it consumes all
10 whole ticks with no early stop.

JS: 247/247 Vitest (+2 net).

## 2026-08-12 (cont.) — Per-mode termination, and diagnosing a real "why did this fail" question

Two requests: let "None" mode fall the complete +-90deg (not just the
45deg ct_sac now uses), and explain why a -20deg ct_sac run terminates
even though "it seems to be doing good."

**Diagnosed the second one empirically first** (a throwaway probe script
against the real checkpoint, deleted after): at a -20deg start, theta
recovers to near 0 within 0.3s and stays there, oscillating gently in a
+-1 to 7deg band, for the ENTIRE run -- the pole genuinely is "doing
good," the whole time. What actually ends the run is `x`: the cart
steadily wanders back and forth along the rail (ct_sac's own known weak
x-regulation, already documented -- see the "oscillates... confirmed
real" entry from the ct_sac wiring round), reaching -2.15m, drifting back
out past +2.0m, and eventually crossing -2.4m at t=11.91s. **The pole
never once came close to the angle boundary in this run -- it ran out of
physical rail, not out of balance.** Confirmed this is the real mechanism,
not a fluke, by logging both `xTerm`/`thetaTerm` flags at every step.

This directly answers "what are the conditions for success": survive
without EITHER `|x| > X_THRESHOLD` (2.4m, the physical rail) OR
`|theta| > THETA_TERMINATION_RADIANS_BY_MODE[mode]`. For a
well-recovered ct_sac run specifically, the rail is now the practically
relevant constraint far more often than the angle ever is -- a real,
somewhat sobering fact about this checkpoint: it cannot actually balance
*indefinitely* on a finite track given its own trained behavior, only
until its own wandering happens to cross the boundary. "Let it go on
longer" is already happening here (12s, not an artificial cutoff) --
there's no angle-side timeout left to relax for a good recovery. The only
further lever would be widening X_THRESHOLD itself (the actual rail
length) -- flagged to the user as a materially different, bigger change
than the theta-threshold work above (X_THRESHOLD is a real physical
parameter shared with camera framing, track decoration, and the Starting
Position slider bounds, not a purely cosmetic display choice), left for a
future round if wanted rather than decided unilaterally here.

**Implemented the first request**: `THETA_TERMINATION_RADIANS_BY_MODE`
replaces the single shared `VISUAL_THETA_THRESHOLD_RADIANS` constant --
`{ none: 90deg, manual: 45deg, ctsac: 45deg }`. "None" gets the full
90deg (confirmed via the same kind of probe: an F=0 free fall barely
moves `x` at all -- under 5cm by the time theta reaches 90 -- so there's
no risk of the rail cutting the fall short first) -- 90deg is also
exactly the camera's own `cos(theta) >= 0` safe limit (see
`cameraHalfExtents`'s docstring), so no camera changes needed. Manual and
ct_sac keep last round's 45deg unchanged.

Verified live: None mode, Starting Angle=20, Run -- the pole visibly
swings all the way to and past horizontal before the run ends (a larger
overshoot than expected was visible in THIS testing tool specifically,
traced to its own severe requestAnimationFrame throttling -- a single
throttled frame's clamped dtReal can be up to 0.1s, and `advance()` only
checks termination once at the end of that whole clamped step, not per
substep, so a fast-falling pole can swing tens of extra degrees past 90
before the next check catches it. At a real 60fps, dtReal is ~0.016s per
frame and this is imperceptible -- confirmed separately via the same
probe technique used above, which found only a 0.35deg overshoot at a
realistic substep size. Documented here rather than "fixed" since it's
this tool's own known throttling limitation, same category as the
already-documented `document.hidden` RAF note, not a real per-frame bug).

JS: 247/247 Vitest (no test changes -- per-mode threshold selection is
`cartpoleMain.js`-only UI wiring over the same already-tested
`isTerminated`/`advanceCtSac` parameter from the previous entry).

## 2026-08-12 (cont.) — Termination-cause readout, and Rail Length as a real slider

Two follow-ups from the previous entry's diagnosis: show WHY a run
terminated (rail vs. angle), and make rail length itself adjustable since
it turned out to be the actual binding constraint for good ct_sac runs.

**`terminationReason(z, thetaThreshold, xThreshold)`** added to
`cartpoleScenarioRunner.js` alongside `isTerminated` (same default
params, same reasoning for keeping them explicit args rather than folding
into the physics `params` object -- these are task/termination concepts,
not dynamics). Checks in the same priority as `isTerminated`'s own `||`:
"rail" wins if both cross in the same tick, since the cart leaving the
modeled world is the more fundamental fact. Returns `null` if not
terminated, so callers don't need a separate `isTerminated` check first.

**Rail Length**: `isTerminated`/`advanceCtSac` both gained a third
`xThreshold` param (default X_THRESHOLD, same backward-compatible pattern
as `thetaThreshold` -- every existing call/test needed zero changes).
`index.html` gained a Rail Length slider under Cart-Pole Geometry (2.0-10.0m,
default 4.8 -- exactly 2*X_THRESHOLD, so untouched it reproduces the
canonical +-2.4m rail ct_sac actually trained on, same "full dimension in
the UI, half in the physics" convention as Pole Length). `CartPoleScene.js`
gained `updateTrackDecoration` (dispose-and-reassign, mirroring
`resizePole` exactly) so the rendered rail/tick-marks can live-update
while dragging, same as every other geometry slider -- convenient that
the chase-cam (see two entries up) never framed the track at all, so
widening/narrowing the rail needs zero camera changes.

**Snapshotted at Reset, not read live**: added `activeXThreshold`,
snapshotted in `resetRun()` exactly like `activeControllerMode` -- the
Rail Length slider updates the VISUAL track live while dragging (a
preview, matching Pole Length's own `resizePole` live-preview), but the
boundary an in-progress run is actually checked against must not silently
change out from under it just because the sidebar moved.

**Display**: `formatDiagnostics()` now appends a second line under
"Terminated — survived Xs" -- `(left the ±X.Xm rail)` or `(pole exceeded
±Ndeg)` -- reading the same snapshotted `activeXThreshold` and
`THETA_TERMINATION_RADIANS_BY_MODE[activeControllerMode]` the actual
check used, so it can never describe a different boundary than the one
genuinely crossed. New `.done-message-reason` CSS class (muted, smaller,
not the bold `--stable` of the message above it) for the sub-detail.

**Verified**: 256/256 Vitest, including new `terminationReason` tests and
an `advanceCtSac` xThreshold early-stop test built the same way as the
existing thetaThreshold one -- numerically probed first (a gentle
constant push crosses a narrow xThreshold on tick 8 of 10.5), not
assumed. Also reproduced the user's exact reported scenario directly (a
throwaway script, deleted after): Starting Angle=-20, ct_sac, default
rail -- confirmed `terminationReason` returns "angle" only once genuinely
past the mode's threshold, "rail" well before that in the actual failing
run. Shortening Rail Length to 2.0m in the same scenario terminates in
1.1s via "rail" with theta at essentially 0deg at that moment -- a clean,
fast repro of exactly the "it was doing great, it just ran out of track"
story from the previous entry.

**Not independently re-verified with a live animated run this round**:
this session's browser tool reported `document.hidden=true` for the tab
and the physics loop did not advance past its first frame no matter how
long real time was allowed to pass -- worse than the milder RAF
throttling noted in earlier entries, and not reproduced via the direct
simulation checks above (which matched the exact reported scenario
numerically). Logged here rather than chased further; the user was
already interacting with the real page in parallel and can confirm the
live rendering/label text directly.

**Standing note for future rounds**: the user asked directly to hand
visual-confirmation checks to them going forward rather than fighting this
tool's RAF/visibility throttling -- for cart-pole specifically (whose
whole rendering loop depends on real elapsed time), verify logic via
targeted Node scripts/unit tests as usual, then ask the user to eyeball
the actual pixels rather than spending turns on browser-tool waits.

## 2026-08-13 — Motion cues: diagonal ground marks + rolling wheel spokes

User feedback: with the chase-cam (see the two entries above) always
keeping the cart centered and no rail-end ever in view, "there's almost
no way to tell the motion of the cart." Asked for two fixes: diagonal
ground marks, and wheels with spokes.

**Ground marks** (`groundMarkPoints`, used by both `createTrackDecoration`
and `updateTrackDecoration`): a run of short diagonal dashes at fixed
0.4m (`GROUND_MARK_SPACING`) intervals across the full rail, built
as ONE `THREE.LineSegments` (not individual `Line` objects, and not
individually managed per-dash) so it's a single draw call regardless of
rail length. The core idea: these sit at FIXED world-x positions, never
attached to the cart -- so as the chase-cam pans to follow the cart, they
scroll past exactly like a road's lane-divider dashes telling you your
own speed, which is the entire point (a cart that's actually moving but
always centered on screen has nothing else to move relative to). Track
decoration's group gained a 4th child (`[rail, tickNeg, tickPos,
groundMarks]`) -- `updateTrackDecoration` extended to dispose/rebuild it
too, same pattern as the other three.

Given a more visible color than reused: `trackMaterial`
(`THEME.cartpoleTrack`, == `THEME.border`) is deliberately subtle -- fine
for a rail that's not meant to draw the eye, wrong for something whose
entire job IS to be noticed. Ground marks get their own `THEME.muted`
material instead -- the closest already-in-palette color with real
contrast against `panelBg` that isn't already claimed by the cart, pole,
or wheels.

**Wheel spokes**: `createCartPoleMesh` now also returns `wheelSpokes` (2
`THREE.LineSegments`, `WHEEL_SPOKE_COUNT = 6` radial lines each, in
`THEME.panelBg` for contrast against the wheel disc's own `THEME.border`
fill), positioned to match each wheel exactly, a hair closer to the camera
(`wheelZ + 0.001`) to avoid z-fighting with the disc. `updateCartPoleFrame`
rotates them every frame: `rotation.z = -x / WHEEL_RADIUS` -- a genuine
rolling-without-slipping relation (arc length = radius * angle), not a
cosmetic guess, negated for the same right-handed-rotation reason the
pole's own `-theta` already is (moving +x is a clockwise turn as drawn,
i.e. negative in THREE's convention). The wheel DISC itself is a flat,
uniformly-colored circle -- identical at any rotation -- so the spokes are
the only thing that actually shows a wheel turning at all.

Test-first: `createTrackDecoration`'s test extended for the new 4th
child (dash count divisible by 6 -- 2 points * xyz per dash -- and every
dash within its own half-width of the rail's actual extent, not a whole
spacing past it); `createCartPoleMesh` gained a test that `wheelSpokes`
is positioned to match its own wheel exactly; `updateCartPoleFrame`
gained a rolling test confirming `rotation.z = -x/radius` at several `x`
values, radius read from the wheel's own geometry (not a re-guessed
constant).

258/258 Vitest. Not independently screenshot-verified this round --
per the user's own explicit request (mid-round), visual confirmation
jobs go to them from now on rather than burning turns on this session's
already-documented RAF/visibility throttling.

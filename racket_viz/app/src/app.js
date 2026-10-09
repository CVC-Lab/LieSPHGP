/**
 * Top-level entry point: drives the multi-system carousel and owns which
 * system's animation loop is actually running. Each system is its own
 * self-contained module (main.js for the racket, following the identical
 * copy-pasted pattern for later systems -- see DECISIONS.md) exporting
 * pause()/resume(); this file never touches a system's internals, only
 * calls those two functions and slides the CSS track.
 *
 * Sliding via `transform`, not display:none/block -- see index.html's
 * #system-viewport comment for why that matters for canvas sizing.
 */
import { pause as pauseRacket, resume as resumeRacket } from "./main.js";
import { pause as pausePendulum, resume as resumePendulum } from "./pendulumMain.js";
import { pause as pauseCartpole, resume as resumeCartpole } from "./cartpoleMain.js";

// Index must match the .system-slide elements' DOM order in index.html.
// Add a { pause, resume } entry here (imported from that system's own
// module) at the same time a new .system-slide is added to the markup.
const systems = [
  { pause: pauseRacket, resume: resumeRacket },
  { pause: pausePendulum, resume: resumePendulum },
  { pause: pauseCartpole, resume: resumeCartpole },
];

const track = document.getElementById("system-track");
const label = document.getElementById("system-label");
const prevBtn = document.getElementById("system-prev");
const nextBtn = document.getElementById("system-next");
const slides = document.querySelectorAll(".system-slide");

let active = 0;

function updateNav() {
  prevBtn.disabled = active === 0;
  nextBtn.disabled = active === slides.length - 1;
  label.textContent = slides[active].dataset.label;
}

function goTo(index) {
  if (index === active || index < 0 || index >= slides.length) return;
  systems[active]?.pause();
  active = index;
  track.style.transform = `translateX(-${active * 100}%)`;
  updateNav();
  systems[active]?.resume();
}

prevBtn.addEventListener("click", () => goTo(active - 1));
nextBtn.addEventListener("click", () => goTo(active + 1));

updateNav();
systems[active]?.resume();

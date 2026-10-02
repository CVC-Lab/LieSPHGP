"""Open-loop and closed-loop reports, in the four_model_comparison format, for the HARD-V5 obs-noise-0.25 models of the
18-09-00-02 spec comparison (PH-GP-LieIMEX, PH-NN-LieIMEX, PH-NODE-RK4 + PH-GT). four_model_comparison.py is imported,
not changed; this file adds only what those models and that plant need:
    * the older GP package (ph_gp_lie_imex): mean model and posterior weight samples via report_evaluation
    * PH-GT with the plant's NONLINEAR damping D_v = m c (1 + |v|) I, D_w = c (1 + |omega|) J (no wind: gusts disabled)
    * page 1 against those true products, and a closed-loop fly / report for this model set

    python -m src.models.SE3_Quadrotor.comparision.hardv5_spec_reports horizons    --model L=RUN@STEP ... --output OUT
    python -m src.models.SE3_Quadrotor.comparision.hardv5_spec_reports closed-loop --model L=RUN@STEP ... --output OUT/closed-loop-test-flights
Open loop: the HARD-V6 held-out flights (disjoint shape library). Closed loop: the 10 analytic shapes of the old booklet
flown from hover, and the held-out flights flown from their recorded x_0, CF2P contact-free with nonlinear damping, no wind.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import four_model_comparison as F

DATASET = Path("tmp/archived_quadrotor_datasets_2026-09-18/QUADROTOR-DATASET-HARD-V6/HARDV6_CF2P_10s_h0p01_clean.pkl")
SPLIT = "heldout"
GT_LABEL = "PH-GT"
ANALYTIC_REFERENCES = ("spiral_k1p5", "spiral_k2", "spiral_k2p5", "lissajous_yaw0p5", "lissajous_yaw1", "lissajous_yaw1p5",
                       "stop_v1", "stop_v1p5", "stop_v2", "stop_v2p5")
for _label, _colour in (("PH-GP-LieIMEX", "tab:orange"), ("PH-NN-LieIMEX", "tab:cyan"), ("PH-NODE-RK4", "tab:pink"), (GT_LABEL, "0.35")):
    F.COLORS.setdefault(_label, _colour)


class NonlinearTruthModel(F.TruthModel):
    """The simulator's operators with the plant's nonlinear damping a = -c (1 + |v|) v, omega_dot = -c (1 + |omega|) omega."""

    def dissipation_v(self, velocity, position=None):
        return self.mass * self.damping_v * (1.0 + jnp.linalg.norm(velocity)) * jnp.eye(3, dtype=velocity.dtype)

    def dissipation_w(self, omega, rotation_flat=None):
        return self.damping_w * (1.0 + jnp.linalg.norm(omega)) * jnp.asarray(self.inertia, dtype=omega.dtype)


class GPv5Entry(F.Entry):
    """The older PH-GP-LieIMEX package (ph_gp_lie_imex): posterior-mean model and weight samples, rolled out with the same
    Lie-IMEX integrator (identical to report_evaluation.rollout, checked to 0.0)."""

    def __init__(self, label: str, run: dict[str, Any]):
        self.label, self.run, self.kind, self.wind_same = label, run, "gp_ode", False
        self.mean, self.params, self.gp_setup = F.evaluation.build_model(run)
        self.step = int(run["selected_step"])
        self.substeps = int(run["config"]["model"].get("integration_substeps", 1) or 1)

    def sample_paths(self, windows: np.ndarray, samples: int, seed: int) -> np.ndarray:
        states = jnp.asarray(windows)
        controls, h = states[1:, :, 18:22], jnp.asarray(F.STEP)
        keys = jax.random.split(jax.random.PRNGKey(seed), samples)

        def one(key):
            model = F.evaluation.network.DissipativeSE3HamODE(self.params, self.gp_setup).sample(key)
            return F.gp_ode_rollout(model, states[0], controls, h, substeps=self.substeps)

        return np.asarray(jax.lax.map(one, keys))


def make_entry(label: str, spec: str, settings: dict[str, Any]):
    if spec == "truth":
        entry = F.Entry(label, "truth", settings)
        base = entry.mean
        entry.mean = NonlinearTruthModel(**{k: getattr(base, k) for k in
                                            ("weights", "gp_setup", "mass", "gravity", "damping_v", "damping_w", "inertia", "wind")})
        entry.sample_paths = lambda *a, **k: None                  # no wind: one deterministic path
        entry.describe = lambda: "simulator constants, nonlinear damping c (1 + |v|), no wind (no training)"
        return entry
    directory, _, step = spec.partition("@")
    run = F.evaluation.load_run(Path(directory), int(step) if step else None)
    if run["config"]["model"]["name"] == "ph_gp_lie_imex":
        return GPv5Entry(label, run)
    return F.Entry(label, spec, settings)


def load_data() -> tuple[np.ndarray, dict[str, Any]]:
    with open(DATASET, "rb") as handle:
        data = pickle.load(handle)
    return np.asarray(data[f"{SPLIT}_trajectories"], dtype=np.float64), data["settings"]


def true_products(settings: dict[str, Any], states: np.ndarray) -> dict[str, np.ndarray]:
    """Per-state true products of the nonlinear-damping plant (M^-1 D depends on |v_b| and |omega_b|)."""
    vehicle = settings["vehicle_parameters"]
    mass, inertia = float(vehicle["mass"]), np.asarray(vehicle["inertia"], dtype=np.float64)
    c_v = float(settings["linear_damping_coefficient"]); c_w = float(settings["angular_damping_coefficient"])
    n = len(states)
    speed, rate = np.linalg.norm(states[:, 12:15], axis=1), np.linalg.norm(states[:, 15:18], axis=1)
    return {"thrust": np.tile([0.0, 0.0, 1.0 / mass], (n, 1)), "torque": np.tile(np.linalg.inv(inertia), (n, 1, 1)),
            "gravity": np.tile([0.0, 0.0, float(vehicle["gravity_acceleration"])], (n, 1)),
            "damping_v": (c_v * (1.0 + speed))[:, None, None] * np.eye(3), "damping_w": (c_w * (1.0 + rate))[:, None, None] * np.eye(3)}


def gauge_rows(entries, settings: dict[str, Any], states: np.ndarray) -> dict[str, Any]:
    truth = true_products(settings, states)
    rows = {"Ground truth": {"values": {key: float(np.mean(pick(truth[F.ERROR_KEYS[key]]))) for key, _, pick in F.GAUGE_COLUMNS},
                             "errors": None, "sigma": None}}
    for e in entries:
        ops = F.gauge_products(e.mean, states)
        errors = {}
        for key, _, _ in F.GAUGE_COLUMNS:
            learned, target = ops[F.ERROR_KEYS[key]], truth[F.ERROR_KEYS[key]]
            axes = tuple(range(1, learned.ndim))
            errors[key] = float(np.mean(np.sqrt(np.sum((learned - target) ** 2, axis=axes)) / np.sqrt(np.sum(target ** 2, axis=axes))))
        rows[e.label] = {"values": {key: float(np.mean(pick(ops[F.ERROR_KEYS[key]]))) for key, _, pick in F.GAUGE_COLUMNS},
                         "errors": errors, "sigma": None}
    return rows


def horizons_title(arguments, entries, flights: int, steps: int) -> plt.Figure:
    lines = [f"dataset   {DATASET.name}  split={SPLIT}  ({flights} flights, shape library disjoint from training)",
             "models    trained on HARD-V5 with observation noise 0.25 (the 18-09-00-02 spec comparison); test flights are clean",
             f"protocol  ONE nonstop open-loop rollout per flight, from the true state at t = 0 to t = {steps * F.STEP:g} s",
             f"          (h = {F.STEP} s, driven by the recorded wrench, no restarts, no feedback). Scored over 0-1, 0-3, 0-5, 0-10 s.",
             "          Nothing is filtered: a diverged rollout keeps its (large or infinite) error.",
             f"line/band PH-GP-LieIMEX: {arguments.samples} posterior weight samples, line = their mean, bands = mean +- 1 / 2 sigma.",
             "          PH-NN-LieIMEX, PH-NODE-RK4 and PH-GT are deterministic (single path, no band).",
             "plant     PyBullet CF2P, contact-free, NONLINEAR damping a = -c (1 + |v|) v, c = 0.5, no wind (gusts disabled)",
             "", "models"]
    for e in entries:
        lines.append(f"  {e.label:14s} {e.describe()}")
    lines += ["", "PH-GT = the simulator's operators (with the nonlinear damping) in the same vector field: the attainable floor,",
              "never ranked. Table A highlights the best of the three learned models per row."]
    return F.text_page("Open-loop horizons: HARD-V5 obs-noise-0.25 models on the HARD-V6 held-out flights", lines)


def horizons(arguments) -> None:
    flights, settings = load_data()
    flights = flights[:arguments.flights]
    entries = [make_entry(label, spec, settings) for label, spec in F.parse_models(arguments.model)] + \
        [make_entry(GT_LABEL, "truth", settings)]
    F.SPLIT_LABEL = "Held-out"
    steps = min(int(round(arguments.seconds / F.STEP)), flights.shape[1] - 1)
    truth = np.transpose(flights[:, :steps + 1], (1, 0, 2))
    t = np.arange(steps + 1) * F.STEP
    results = {}
    for entry in entries:
        print(f"[horizons] {entry.label} ({entry.kind}): {truth.shape[1]} flights x {steps} steps nonstop", flush=True)
        mean = entry.mean_path(truth)
        paths = entry.sample_paths(truth, arguments.samples, arguments.seed)
        centre = F.centre_line(mean, paths)
        results[entry.label] = {"mean": centre, "paths": paths, "errors": F.error_series(truth, centre),
                                "calib": F.calibration_stats(truth, paths, centre),
                                "so3": F.so3_violations(mean if paths is None else paths),
                                "calib_flight": F.calibration_per_flight(truth, paths, centre)}
    marks = [m for m in F.HORIZON_MARKS if m <= t[-1] + 1e-9]
    table = {}
    for entry in entries:
        table[entry.label] = {}
        for block, series in results[entry.label]["errors"].items():
            table[entry.label][block] = {}
            for mark in marks:
                count = int(round(mark / F.STEP))
                per_flight = series[1:count + 1].mean(axis=0)
                table[entry.label][block][f"0-{mark:g}s"] = {"median_over_flights": float(np.median(per_flight)),
                                                              "at_end_median": float(np.median(series[count]))}
    table_a = {e.label: F.table_a_values(results[e.label], marks) for e in entries}
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "open-loop-horizons-comparison.pdf"
    page_arguments = SimpleNamespace(dataset=DATASET, split=SPLIT)
    with PdfPages(pdf_path) as pdf:
        rows = gauge_rows(entries, settings, flights[:, :steps + 1].reshape(-1, 22)[::5])
        pdf.savefig(F.gauge_page(rows, "Physics recovery vs ground truth - HARD-V6 held-out, nonlinear damping")); plt.close("all")
        for mark in marks:
            pdf.savefig(F.table_a_page(table_a, entries, mark, truth.shape[1], page_arguments)); plt.close("all")
        pdf.savefig(horizons_title(arguments, entries, truth.shape[1], steps)); plt.close("all")
        pdf.savefig(F.horizon_error_page(t, results, entries, marks)); plt.close("all")
        pdf.savefig(F.horizon_bar_page(table, entries, marks)); plt.close("all")
        pdf.savefig(F.calibration_figure(t, {e.label: results[e.label]["calib"] for e in entries}, entries, marks[:-1],
                                         f"  ({truth.shape[1]} flights, nonstop from t = 0)")); plt.close("all")
        for f in range(truth.shape[1]):
            per_model = [[{"t": t, "mean": results[e.label]["mean"][:, f],
                           "paths": None if results[e.label]["paths"] is None else results[e.label]["paths"][:, :, f]}] for e in entries]
            flight = flights[f, :steps + 1]
            pdf.savefig(F.path3d_page(flight, per_model, entries, f)); plt.close("all")
            for block in ("p", "euler", "v", "w"):
                pdf.savefig(F.block_page(flight, per_model, entries, f, block, [0], marks=marks[:-1])); plt.close("all")
    summary = {"generated_at": datetime.now().astimezone().isoformat(), "dataset": str(DATASET), "split": SPLIT,
               "flights": int(truth.shape[1]), "seconds": float(t[-1]), "samples": arguments.samples,
               "models": {e.label: {"kind": e.kind, "source": e.describe()} for e in entries},
               "error_table": table, "table_a": table_a, "gauge_table": rows}
    (output / "open-loop-horizons-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[horizons] wrote {pdf_path}", flush=True)


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------
def references(arguments) -> list[str]:
    return list(ANALYTIC_REFERENCES) + [f"{SPLIT}-flight{k:02d}" for k in range(arguments.recorded)]


def fly(arguments) -> None:
    from . import report_controller as rc
    flights, settings = load_data()
    label, _, spec = arguments.model_spec.partition("=")
    entry = make_entry(label, spec, settings)
    initial_state = None
    index = F.recorded_flight_index(arguments.reference)
    if arguments.reference not in rc.REFERENCES and index is not None:
        flight = flights[index]
        rc.REFERENCES[arguments.reference] = rc.make_recorded_reference(flight, float(settings.get("sample_dt", F.STEP)), relative_yaw=False)
        initial_state = F.recorded_initial_state(flight)
    run = entry.run if entry.run is not None else F.stub_run(DATASET)
    rc.run_controller(run, entry.mean, arguments.output, model_label=label, training_dataset=DATASET,
                      vehicle=F.evaluation.vehicle_constants(settings), duration_seconds=arguments.seconds, seed=0,
                      use_dissipation=True, reference=arguments.reference, gust=None,
                      provider_description=f"{label}: {entry.describe()}", initial_state=initial_state)


def closed_loop(arguments) -> None:
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    models = F.parse_models(arguments.model) + [(GT_LABEL, "truth")]
    jobs = [(label, spec, reference, output / "flights" / reference / label / "seed0")
            for label, spec in models for reference in references(arguments)]
    jobs = [job for job in jobs if arguments.refly or not (job[3] / "controller_rollout.npz").is_file()]
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print(f"[closed-loop] {len(jobs)} flights to fly with {arguments.workers} workers", flush=True)

    def launch(job):
        label, spec, reference, folder = job
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "src.models.SE3_Quadrotor.comparision.hardv5_spec_reports", "fly",
                   "--model-spec", f"{label}={spec}", "--reference", reference, "--seconds", str(arguments.seconds),
                   "--output", str(folder)]
        with open(folder / "fly.log", "w") as log:
            code = subprocess.call(command, cwd=F.PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if code:
            print(f"[closed-loop] {reference:24s} {label:14s} exit={code}", flush=True)
        return code

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        codes = list(pool.map(launch, jobs))
    print(f"[closed-loop] flown; {sum(1 for c in codes if c)} exited non-zero", flush=True)
    for statistic in ("mean", "median"):
        closed_loop_report(arguments, [label for label, _ in models], statistic)


def _relabel(figure: plt.Figure, replacements: dict[str, str]) -> plt.Figure:
    text = figure._suptitle.get_text()
    for old, new in replacements.items():
        text = text.replace(old, new)
    figure._suptitle.set_text(text)
    for legend in figure.legends:                          # one deterministic flight: no seed mean, no band
        for label in legend.get_texts():
            label.set_text({"flown (seed mean)": "flown", "+- 1 std over seeds": "(no spread: no wind, one flight)",
                            "seed mean": "flown", "each wind seed": "flown"}.get(label.get_text(), label.get_text()))
    return figure


def closed_loop_title(arguments, labels) -> plt.Figure:
    from . import report_controller as rc
    gains = ", ".join(f"{name} = {np.array2string(value, separator=' ')}" for name, value in rc.GAINS.items())
    lines = ["models     the 18-09-00-02 spec comparison: trained on HARD-V5 with observation noise 0.25",
             "references (1) the old booklet's 10 analytic shapes (spiral k 1.5/2/2.5, lissajous yaw 0.5/1/1.5, stop v 1/1.5/2/2.5),",
             "           flown from hover; (2) the first 10 HARD-V6 held-out flights as flown, from their RECORDED x_0",
             "plant      gym-pybullet-drones CF2P, Physics.PYB, 240 Hz, contact-free, NONLINEAR damping c (1 + |v|), c = 0.5; no wind",
             "           (the dataset's gusts are disabled), so every flight is deterministic: one seed",
             "controller the SAME energy-based SE(3) law for every model (report_controller.EnergyController) on the model's",
             "           mean operators; allocation u = pinv(M^-1 g)(M^-1 w); dissipation feed-forward on",
             f"           {gains}; tilt limit {np.degrees(rc.MAXIMUM_TILT):.0f} deg",
             f"duration   {arguments.seconds:g} s per flight (shapes are active 15 s, recorded flights 10 s, then hold their end point)",
             "", "pages      Table C over the shapes, Table C over the held-out flights, then per reference: table, tracking, 3-D", "", "models"]
    lines += [f"  {spec}" for spec in arguments.model]
    lines.append(f"  {GT_LABEL}=simulator constants with the nonlinear damping (the controller's ceiling, not ranked)")
    return F.text_page("Closed-loop tracking: HARD-V5 obs-noise-0.25 models", lines)


def closed_loop_report(arguments, labels: list[str], statistic: str = "mean") -> None:
    F.TRUTH_LABEL = GT_LABEL                                   # the unranked reference column of the shared table page
    note = F.CLOSED_LOOP_NOTE.replace("Analytical-SDE", GT_LABEL)
    F.CLOSED_LOOP_ROWS = tuple((r[0].replace("Analytical-SDE", GT_LABEL),) + tuple(r[1:]) for r in F.CLOSED_LOOP_ROWS)
    output = arguments.output
    names = references(arguments)
    flights = {ref: {label: [F.load_flight(output / "flights" / ref / label / "seed0")] for label in labels} for ref in names}
    for ref in names:
        F.add_valid_tracking(flights[ref], labels)
    data_flights, settings = load_data()
    median = statistic == "median"
    pdf_path = output / ("closed_loop_comparison_median.pdf" if median else "closed-loop-comparison.pdf")
    how = "MEDIAN [25th, 75th percentile]" if median else "mean +- std"
    summary: dict[str, Any] = {"table_c_statistic": how, "table_c": {}, "flights": {}}
    replacements = {"recorded test flight (reference)": "reference", "line = mean over 1 wind seeds, band = +- 1 std over seeds; ": "",
                    "same recorded start state x_0, reference, controller gains and wind seeds in every panel":
                        "same start state, reference and controller gains in every panel; no wind"}
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(closed_loop_title(arguments, labels)); plt.close("all")
        for group, members in (("the 10 analytic shapes (from hover)", list(ANALYTIC_REFERENCES)),
                               (f"the {arguments.recorded} HARD-V6 held-out flights (from recorded x_0)",
                                [n for n in names if n not in ANALYTIC_REFERENCES])):
            figure, summary["table_c"][group] = F.closed_loop_table_page(
                f"Table C{' (median)' if median else ''} - closed-loop tracking over {group}",
                f"{DATASET.name}: one deterministic flight per model and reference (no wind); cells = {how} over the "
                f"{len(members)} references.\n{arguments.seconds:g} s per flight.",
                {label: [flights[ref][label] for ref in members] for label in labels}, labels, note,
                statistic="median" if median else "mean")
            pdf.savefig(figure); plt.close("all")
        for ref in names:
            index = F.recorded_flight_index(ref)
            chain = F.flight_segments(settings, SPLIT, index) if index is not None else ""
            origin = "flown from its recorded x_0" if index is not None else "analytic shape, flown from hover"
            figure, summary["flights"][ref] = F.closed_loop_table_page(
                f"{ref} - closed-loop tracking, {arguments.seconds:g} s",
                f"{'segments: ' + chain + chr(10) if chain else ''}{origin}; one deterministic flight per model (no wind, so no spread).",
                {label: [flights[ref][label]] for label in labels}, labels, note)
            pdf.savefig(figure); plt.close("all")
            if any(f is not None for label in labels for f in flights[ref][label]):
                pdf.savefig(_relabel(F.closed_loop_tracking_page(flights[ref], labels, ref, arguments.seconds), replacements)); plt.close("all")
                pdf.savefig(_relabel(F.closed_loop_trajectory_page(flights[ref], labels, ref, arguments.seconds), replacements)); plt.close("all")
    (output / ("closed-loop-summary-median.json" if median else "closed-loop-summary.json")).write_text(
        json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[closed-loop] wrote {pdf_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    z = sub.add_parser("horizons")
    z.add_argument("--model", action="append", required=True, help="LABEL=RUN_DIR@STEP")
    z.add_argument("--output", type=Path, required=True)
    z.add_argument("--flights", type=int, default=10)
    z.add_argument("--seconds", type=float, default=10.0)
    z.add_argument("--samples", type=int, default=32)
    z.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("closed-loop")
    c.add_argument("--model", action="append", required=True, help="LABEL=RUN_DIR@STEP")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--recorded", type=int, default=10)
    c.add_argument("--seconds", type=float, default=20.0)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--refly", action="store_true")
    c.add_argument("--report-only", action="store_true")
    f = sub.add_parser("fly")
    f.add_argument("--model-spec", required=True)
    f.add_argument("--reference", required=True)
    f.add_argument("--seconds", type=float, default=20.0)
    f.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "horizons":
        horizons(arguments)
    elif arguments.command == "fly":
        fly(arguments)
    elif arguments.report_only:
        labels = [label for label, _ in F.parse_models(arguments.model)] + [GT_LABEL]
        for statistic in ("mean", "median"):
            closed_loop_report(arguments, labels, statistic)
    else:
        closed_loop(arguments)


if __name__ == "__main__":
    main()

"""Open-loop horizons report for the REAL IDSIA Crazyflie 2.1 Brushless dataset, in the four_model_comparison format.

Same pages as ``four_model_comparison.py horizons`` (whose page builders are imported, not copied or changed), with the
two pages that assume a simulator replaced: there is no ground truth and no recorded wind on real data, so
    page 1   learned gauge-invariant products next to the benchmark's PUBLISHED constants (a reference, not ground truth;
             damping and diffusion of the real vehicle are unknown), plus the learned diffusion and observation noise
    title    the IDSIA protocol and the flights used (source recording and 10 s chunk of each)
then Table A (0-1 / 0-3 / 0-5 / 0-10 s), error vs time, error bars, calibration, and per flight a 3-D page and the
position / attitude / velocity / angular-velocity pages. One nonstop open-loop rollout per flight from t = 0, driven by
the recorded wrench; line = mean of the sample paths, bands = mean +- 1 / 2 sigma.

    python -m src.models.SE3_Quadrotor.comparision.idsia_open_loop_report --model "GP-SDE=RUN_DIR@4000" \
        [--model "NN-SDE=RUN_DIR@7000" ...] --output OUT_DIR
writes OUT_DIR/open_loop_idsia_train.pdf and OUT_DIR/open_loop_idsia_test.pdf (+ a summary JSON each).
"""
from __future__ import annotations

import argparse
import json
import pickle
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

DATASET = Path("datasets/QUADROTOR-DATASET-IDSIA/IDSIA_CF21BL_10s_h0p01_clean.pkl")
# train: 10 flights over the three shapes and ten different recordings (chirp runs 1-3, random runs 1, 2, 4, all four
# square runs, which have one 10 s chunk each); test: 10 melon flights spread over the three melon recordings
FLIGHTS = {"train": (0, 7, 14, 21, 28, 35, 40, 41, 42, 43), "test": (0, 2, 4, 6, 8, 9, 11, 13, 15, 17)}
PDF_NAMES = {"train": "open_loop_idsia_train.pdf", "test": "open_loop_idsia_test.pdf"}


def flight_sources(settings: dict[str, Any], split: str) -> list[str]:
    """Source recording and 10 s chunk of every flight of a split, in the order the converter stacked them."""
    names = []
    for audit in settings[f"{split}_flight_audits"]:
        names += [f"{audit['family']}: {audit['file']} chunk {chunk}" for chunk in range(int(audit["kept_chunks"]))]
    return names


def published_products(settings: dict[str, Any]) -> dict[str, np.ndarray]:
    """The benchmark's published rigid-body values of the products the model can identify (no damping, no wind)."""
    vehicle = settings["vehicle_parameters"]
    mass, gravity = float(vehicle["mass"]), float(vehicle["gravity_acceleration"])
    return {"thrust": np.array([0.0, 0.0, 1.0 / mass]), "torque": np.linalg.inv(np.asarray(vehicle["inertia"], dtype=np.float64)),
            "gravity": np.array([0.0, 0.0, gravity])}


def model_physics(entry, flights: np.ndarray, reference: dict[str, np.ndarray]) -> dict[str, Any]:
    """One model's products (mean over the flights' states), error vs the published values, diffusion, sigma_obs."""
    states = flights.reshape(-1, 22)[::5]
    ops = F.gauge_products(entry.mean, states)
    values = {key: float(np.mean(pick(ops[F.ERROR_KEYS[key]]))) for key, _, pick in F.GAUGE_COLUMNS}
    errors = {}
    for name, target in reference.items():                    # mean over states of ||O_hat - O|| / ||O||
        learned = ops[name]
        axes = tuple(range(1, learned.ndim))
        errors[name] = float(np.mean(np.sqrt(np.sum((learned - target) ** 2, axis=axes)) / np.sqrt(np.sum(target ** 2))))
    starts = np.arange(0, flights.shape[1] - 1, 38)
    pairs = np.stack([flights[:, starts], flights[:, starts + 1]], axis=0).reshape(2, -1, 22)
    step_fn = F.nn_sde_step if entry.kind == "nn_sde" else F.gp_sde_step          # the model's own Lie-IMEX step
    diffusion = np.asarray(F.transition_diffusion(entry.mean, jnp.asarray(pairs, dtype=F.model_dtype(entry.mean)),
                                                  jnp.asarray(F.STEP), step_fn))
    likelihood = entry.params.get("likelihood", {}) if isinstance(entry.params, dict) else {}
    observation = {name: float(np.exp(np.asarray(likelihood[f"log_sigma_{name}"]).ravel()[0]))
                   for name in ("position", "attitude", "linear_velocity", "angular_velocity") if f"log_sigma_{name}" in likelihood}
    fixed = ((entry.run or {}).get("config") or {}).get("sde", {}).get("fixed_observation_sigma")
    return {"values": values, "errors": errors, "diffusion": (float(np.mean(diffusion[:3])), float(np.mean(diffusion[3:]))),
            "observation_sigma": observation, "fixed_observation_sigma": fixed}


def rotor_products(model, states: np.ndarray) -> dict[str, np.ndarray]:
    """Rotor-input (u_i = Omega_i^2 / Omega_hover^2) products: the whole learned mixer M1^-1 g_f (3 x 4) and M2^-1 g_tau
    (3 x 4), gravity M1^-1 grad V and the damping products, on (N, 22) states."""
    dtype = F.model_dtype(model)

    def one(x):
        m1, m2 = model.inverse_mass_1(x[:3]), model.inverse_mass_2(x[3:12])
        control = model.control_matrix(x[:12])
        grad_v = jax.grad(model.potential)(x[:12])[:3]
        return (m1 @ control[:3], m2 @ control[3:], m1 @ grad_v,
                m1 @ model.dissipation_v(x[12:15], x[:3]), m2 @ model.dissipation_w(x[15:18], x[3:12]))

    names = ("force", "torque", "gravity", "damping_v", "damping_w")
    return dict(zip(names, (np.asarray(v, dtype=np.float64) for v in jax.vmap(one)(jnp.asarray(states, dtype=dtype)))))


def published_mixer(settings: dict[str, Any]) -> dict[str, np.ndarray]:
    """Published M^-1 g for u_i = Omega_i^2 / Omega_hover^2, Omega_hover^2 = m g / (4 K_t): per rotor a thrust K_t Omega_h^2
    = m g / 4 and the benchmark's mixer torques (convert_idsia.py), mapped through 1/m and J^-1."""
    vehicle, motor = settings["vehicle_parameters"], settings["motor_model"]
    mass, gravity = float(vehicle["mass"]), float(vehicle["gravity_acceleration"])
    arm, kt, kc = float(motor["arm"]), float(motor["kt"]), float(motor["kc"])
    per_rotor = mass * gravity / 4.0
    force = np.zeros((3, 4)); force[2] = per_rotor / mass
    signs = np.array([[-1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0]])
    torque = np.linalg.inv(np.asarray(vehicle["inertia"], dtype=np.float64)) @ (
        np.array([per_rotor * arm, per_rotor * arm, per_rotor * kc / kt])[:, None] * signs)
    return {"force": force, "torque": torque, "gravity": np.array([0.0, 0.0, gravity])}


def relative_error(learned: np.ndarray, target: np.ndarray) -> float:
    axes = tuple(range(1, learned.ndim))
    return float(np.mean(np.sqrt(np.sum((learned - target) ** 2, axis=axes)) / np.sqrt(np.sum(target ** 2))))


def rotor_physics_page(entries, flights: np.ndarray, settings: dict[str, Any], split: str) -> tuple[plt.Figure, dict[str, Any]]:
    """Page 1 for rotor inputs: the learned 6 x 4 mixer against the one implied by the published constants."""
    reference = published_mixer(settings)
    states = flights.reshape(-1, 22)[::5]
    models = {}
    for e in entries:
        ops = rotor_products(e.mean, states)
        base = model_physics(e, flights, {})               # diffusion and observation noise
        models[e.label] = {"force_mean": ops["force"].mean(axis=0), "torque_mean": ops["torque"].mean(axis=0),
                           "errors": {"force": relative_error(ops["force"], reference["force"]),
                                      "torque": relative_error(ops["torque"], reference["torque"]),
                                      "gravity": relative_error(ops["gravity"], reference["gravity"])},
                           "gravity": float(ops["gravity"][:, 2].mean()),
                           "damping_v": float(np.mean(np.trace(ops["damping_v"], axis1=1, axis2=2) / 3.0)),
                           "damping_w": float(np.mean(np.trace(ops["damping_w"], axis1=1, axis2=2) / 3.0)),
                           "diffusion": base["diffusion"], "observation_sigma": base["observation_sigma"],
                           "fixed_observation_sigma": base["fixed_observation_sigma"]}
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(0.03, 0.95, f"Physics recovery vs the PUBLISHED constants - {DATASET.stem} ({split} split)", fontsize=13, weight="bold")
    figure.text(0.03, 0.91, "Rotor inputs u_i = Omega_i^2 / Omega_hover^2 (Omega_hover^2 = m g / (4 K_t)): g is a 6 x 4 MIXER. Reference = the mixer "
                "implied by the benchmark's published m, J, g, K_t, K_c, arm\n(a reference, not ground truth; damping and diffusion unknown). "
                "(x %) = mean over states of ||O_hat - O|| / ||O||; per-rotor thrust = (M1^-1 g_f)_z of each rotor.", fontsize=8.5, va="top")
    header = ["model", "total thrust\nsum_i (M1^-1 g_f)_z", "per-rotor thrust\n(M1^-1 g_f)_z, rotors 1-4", "M1^-1 g_f\nerror",
              "M2^-1 g_tau\nmixer error", "M1^-1 grad V  z", "M1^-1 D_v\ndiag", "M2^-1 D_w\ndiag", "sigma_v\n(force)", "sigma_w\n(torque)"]
    rows = [["Published constants\n(reference, not truth)", f"{reference['force'][2].sum():.4g}",
             " / ".join(f"{v:.3g}" for v in reference["force"][2]), "-", "-", f"{reference['gravity'][2]:.4g}",
             "unknown", "unknown", "unknown", "unknown"]]
    for e in entries:
        m = models[e.label]
        rows.append([f"{e.label} @ step {e.step}", f"{m['force_mean'][2].sum():.4g}", " / ".join(f"{v:.3g}" for v in m["force_mean"][2]),
                     f"{m['errors']['force'] * 100:.1f} %", f"{m['errors']['torque'] * 100:.1f} %",
                     f"{m['gravity']:.4g}\n({m['errors']['gravity'] * 100:.1f} %)", f"{m['damping_v']:.4g}", f"{m['damping_w']:.4g}",
                     f"{m['diffusion'][0]:.4g}", f"{m['diffusion'][1]:.4g}"])
    axis = figure.add_axes([0.02, 0.50, 0.96, 0.34]); axis.axis("off")
    widths = [0.13, 0.09, 0.17, 0.08, 0.09, 0.09, 0.08, 0.08, 0.09, 0.09]
    table = axis.table(cellText=rows, colLabels=header, colWidths=widths, loc="center", cellLoc="center",
                       cellColours=[["#e8e8e8"] * len(header)] + [["white"] * len(header)] * len(entries))
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 3.0)
    for column in range(len(header)):
        table[0, column].set_text_props(weight="bold"); table[0, column].set_facecolor("#d0d8e8")

    def matrix_lines(matrix: np.ndarray) -> list[str]:
        return ["    " + "  ".join(f"{v:>10.4g}" for v in row) for row in matrix]

    lines = ["Torque mixer M2^-1 g_tau (rows = roll / pitch / yaw, columns = rotors 1-4), mean over states:", "  published:"]
    lines += matrix_lines(reference["torque"])
    for e in entries:
        lines += [f"  {e.label}:"] + matrix_lines(models[e.label]["torque_mean"])
    units = {"position": "m", "attitude": "rad", "linear_velocity": "m/s", "angular_velocity": "rad/s"}
    lines += ["", "Observation noise sigma_obs per model:"]
    for e in entries:
        m = models[e.label]
        lines.append(f"  {e.label:8s} " + ("learned: " + ", ".join(f"{k.replace('_', ' ')} {v:.3g} {units[k]}" for k, v in m["observation_sigma"].items())
                                          if m["observation_sigma"] else f"fixed at {m['fixed_observation_sigma']} (not learned)"))
    jzz = float(np.asarray(settings["vehicle_parameters"]["inertia"])[2, 2])
    lines += [f"Published J_zz = {jzz:.4g} kg m^2 as stored (possibly a typo for 3.2347e-5), so the published yaw row may be 10x too large."]
    figure.text(0.05, 0.46, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
    return figure, {"published": {k: v.tolist() for k, v in reference.items()},
                    "models": {k: {n: (v.tolist() if isinstance(v, np.ndarray) else v) for n, v in m.items()} for k, m in models.items()}}


def physics_page(entries, flights: np.ndarray, settings: dict[str, Any], split: str) -> tuple[plt.Figure, dict[str, Any]]:
    """Page 1: every model's learned products vs the published constants; learned diffusion and observation noise."""
    reference = published_products(settings)
    published = {key: (float(pick(reference[F.ERROR_KEYS[key]][None]).item()) if F.ERROR_KEYS[key] in reference else None)
                 for key, _, pick in F.GAUGE_COLUMNS}
    models = {e.label: model_physics(e, flights, reference) for e in entries}

    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(0.03, 0.95, f"Physics recovery vs the PUBLISHED constants - {DATASET.stem} ({split} split)", fontsize=15, weight="bold")
    figure.text(0.03, 0.91, "Real flight data: there is no ground truth. Reference = the benchmark's published rigid-body values "
                "(m = 45 g, J, g = 9.81; models/models.py), a reference only; the real vehicle's damping and\nwind/diffusion are "
                "unknown. Model value = mean over the shown flights' states; (x %) = mean over states of ||O_hat - O|| / ||O|| "
                "against the published value.", fontsize=8.5, va="top")
    header = ["model"] + [label for _, label, _ in F.GAUGE_COLUMNS] + ["sigma_v (force)", "sigma_w (torque)"]
    rows = [["Published constants\n(reference, not truth)"] + [
        "unknown" if published[key] is None else f"{published[key]:.4g}" for key, _, _ in F.GAUGE_COLUMNS] + ["unknown", "unknown"]]
    for e in entries:
        m = models[e.label]
        rows.append([f"{e.label} @ step {e.step}"] + [
            f"{m['values'][key]:.4g}" + (f"\n({m['errors'][F.ERROR_KEYS[key]] * 100:.1f} %)" if F.ERROR_KEYS[key] in m["errors"] else "")
            for key, _, _ in F.GAUGE_COLUMNS] + [f"{m['diffusion'][0]:.4g}", f"{m['diffusion'][1]:.4g}"])
    axis = figure.add_axes([0.02, 0.43, 0.96, 0.40]); axis.axis("off")
    table = axis.table(cellText=rows, colLabels=header, loc="center", cellLoc="center",
                       cellColours=[["#e8e8e8"] * len(header)] + [["white"] * len(header)] * len(entries))
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1.0, 3.0)
    for column in range(len(header)):
        table[0, column].set_text_props(weight="bold"); table[0, column].set_facecolor("#d0d8e8")
    units = {"position": "m", "attitude": "rad", "linear_velocity": "m/s", "angular_velocity": "rad/s"}
    lines = ["Observation noise sigma_obs per model:"]
    for e in entries:
        m = models[e.label]
        if m["observation_sigma"]:
            lines.append(f"  {e.label:8s} learned (EKF likelihood): " + ", ".join(
                f"{name.replace('_', ' ')} {value:.3g} {units[name]}" for name, value in m["observation_sigma"].items()))
        else:
            lines.append(f"  {e.label:8s} fixed at {m['fixed_observation_sigma']} (not learned)")
    jzz = float(np.asarray(settings["vehicle_parameters"]["inertia"])[2, 2])
    lines += ["", "Diffusion = sqrt(diag(J J^T) / h) of each model's own Lie-IMEX step, in twist units per sqrt(s).",
              f"Published J_zz = {jzz:.4g} kg m^2 as stored (the benchmark value; J_xx / J_zz = {2.3951e-05 / jzz:.1f},",
              "possibly a typo for 3.2347e-5), so the M2^-1 g_tau zz reference may be 10x too large."]
    figure.text(0.05, 0.38, "\n".join(lines), fontsize=9, va="top", family="monospace")
    return figure, {"published": published, "models": models}


def title_page(arguments, entries, split: str, indices, sources: list[str], steps: int) -> plt.Figure:
    lines = [f"dataset   {DATASET.name}  split={split}  ({len(indices)} flights)",
             "          real Crazyflie 2.1 Brushless flights (IDSIA nano-drone benchmark); wrench from the MEASURED rotor speeds",
             f"protocol  ONE nonstop open-loop Lie-IMEX rollout per flight, from the recorded state at t = 0 to t = {steps * F.STEP:g} s",
             f"          (h = {F.STEP} s, driven by the recorded wrench, no restarts, no feedback). Scored over 0-1, 0-3, 0-5, 0-10 s.",
             "          Nothing is filtered: a diverged rollout keeps its (large or infinite) error.",
             f"line/band {arguments.samples} sample paths (posterior weight samples x Brownian paths): line = their mean,",
             "          dark = mean +- 1 sigma, light = mean +- 2 sigma. The band is the LATENT state (no observation noise added).",
             "", "models"] + [f"  {e.label:7s} {e.describe()}" for e in entries] + [
             "  no Analytical-SDE (the real vehicle's operators are unknown) and no -wind-same column (no recorded wind)",
             "", "flights (index in the split: source recording, 10 s chunk)"]
    lines += [f"  {k:2d} -> {index:2d}: {sources[index]}" for k, index in enumerate(indices)]
    lines += ["", "note  the benchmark's own protocol scores open-loop prediction up to 0.5 s; 10 s open loop on real data",
              "      is far beyond it, so the long horizons mostly show how the error grows."]
    return F.text_page(f"Open-loop horizons on real IDSIA flights ({split} split), nonstop from t = 0", lines)


def single_model_block_page(*args, **kwargs) -> plt.Figure:
    """four_model_comparison.block_page indexes axes[row, col]; with ONE model column plt.subplots would squeeze the grid
    to 1-D, so it is called with squeeze=False (the original file is left unchanged)."""
    original = plt.subplots

    def subplots(*a, **k):
        k.setdefault("squeeze", False)
        return original(*a, **k)

    plt.subplots = subplots
    try:
        figure = F.block_page(*args, **kwargs)
    finally:
        plt.subplots = original
    # the shared title is one long line written for a 7-column page: break it so it fits one column
    title = figure._suptitle.get_text().replace(" (single path if none), ", ",\n").replace("; 'clipped'", ";\n'clipped'")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure


def report(arguments, entries, data: dict[str, Any], split: str) -> None:
    settings = data["settings"]
    indices = FLIGHTS[split]
    sources = flight_sources(settings, split)
    flights = np.asarray(data[f"{split}_trajectories"], dtype=np.float64)[list(indices)]
    steps = min(int(round(arguments.seconds / F.STEP)), flights.shape[1] - 1)
    truth = np.transpose(flights[:, :steps + 1], (1, 0, 2))                        # (T, B, 22)
    t = np.arange(steps + 1) * F.STEP
    F.SPLIT_LABEL = split.capitalize()
    results = {}
    for entry in entries:
        print(f"[idsia] {split}: {entry.label} on {truth.shape[1]} flights x {steps} steps nonstop, {arguments.samples} samples", flush=True)
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
    page_arguments = SimpleNamespace(dataset=DATASET, split=split)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / PDF_NAMES[split]
    with PdfPages(pdf_path) as pdf:
        page = rotor_physics_page if settings.get("input_mode") == "rotor2" else physics_page
        figure, physics = page(entries, flights[:, :steps + 1], settings, split)
        pdf.savefig(figure); plt.close("all")
        for mark in marks:
            pdf.savefig(F.table_a_page(table_a, entries, mark, truth.shape[1], page_arguments)); plt.close("all")
        pdf.savefig(title_page(arguments, entries, split, indices, sources, steps)); plt.close("all")
        pdf.savefig(F.horizon_error_page(t, results, entries, marks)); plt.close("all")
        pdf.savefig(F.horizon_bar_page(table, entries, marks)); plt.close("all")
        pdf.savefig(F.calibration_figure(t, {e.label: results[e.label]["calib"] for e in entries}, entries, marks[:-1],
                                         f"  ({truth.shape[1]} flights, nonstop from t = 0)")); plt.close("all")
        for f, index in enumerate(indices):
            per_model = [[{"t": t, "mean": results[e.label]["mean"][:, f],
                           "paths": None if results[e.label]["paths"] is None else results[e.label]["paths"][:, :, f]}] for e in entries]
            flight = flights[f, :steps + 1]
            name = f"{index} ({sources[index]})"
            pdf.savefig(F.path3d_page(flight, per_model, entries, name)); plt.close("all")
            for block in ("p", "euler", "v", "w"):
                pdf.savefig(single_model_block_page(flight, per_model, entries, name, block, [0], marks=marks[:-1])); plt.close("all")
    summary = {"generated_at": datetime.now().astimezone().isoformat(), "dataset": str(DATASET), "split": split,
               "flights": {int(i): sources[i] for i in indices}, "seconds": float(t[-1]), "samples": arguments.samples,
               "models": {e.label: {"source": e.describe(), "band": F.BAND_TEXT[e.kind]} for e in entries},
               "protocol": "one nonstop open-loop rollout per flight from t = 0, recorded wrench, no restarts, no filtering",
               "physics_vs_published": physics, "error_table": table, "table_a": table_a}
    (output / f"open_loop_idsia_{split}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[idsia] wrote {pdf_path}", flush=True)


def main() -> None:
    global DATASET
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="LABEL=RUN_DIR@STEP (repeat for more models)")
    parser.add_argument("--dataset", type=Path, default=DATASET, help="IDSIA pickle (wrench or rotor2 input)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    DATASET = arguments.dataset
    with open(DATASET, "rb") as handle:
        data = pickle.load(handle)
    entries = [F.Entry(label, spec, data["settings"]) for label, spec in F.parse_models(arguments.model)]
    for k, e in enumerate(entries):                     # a label without a colour in the shared palette gets one
        F.COLORS.setdefault(e.label, ("tab:cyan", "tab:orange", "tab:blue", "tab:red")[k % 4])
    for split in (("train", "test") if arguments.split == "both" else (arguments.split,)):
        report(arguments, entries, data, split)


if __name__ == "__main__":
    main()

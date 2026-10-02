"""Closed-loop subnetwork-swap ablation.

Fly a BASE model with ONE gauge-invariant product replaced by a DONOR model's (see HybridProvider), on a set of
references, in both directions. The swap that breaks the base names the operator that carries the closed-loop
difference; the swaps that do not, rule theirs out. Every swap is flown and reported.

Usage:
    python -m src.models.SE3_Quadrotor.comparision.controller_swap_ablation \
        --run-a <gp_run> --run-b <nn_run> --reference lissajous_yaw1p5 --reference stop_v2 [--dissipation on]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import report_evaluation as evaluation
from . import report_figures as figures
from .generate_comparison_report_v2 import EVAL_ROOT
from .generate_report import _model_label, _noise_level
from .report_controller import SWAP_PRODUCTS, HybridProvider, JaxProvider, controller_comparison, run_controller
from ..ph_gp_lie_imex.config import load_config, resolve_project_path

PRODUCT_TEXT = {"force_scale": "mu g_f (thrust / force scale)", "torque_gain": "M2^-1 g_tau (torque gain)",
                "gravity": "mu grad V (gravity)", "damping_v": "mu Dv (translational damping)",
                "damping_w": "M2^-1 Dw (rotational damping)"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", type=Path, required=True); parser.add_argument("--run-b", type=Path, default=None)
    parser.add_argument("--donor-ground-truth", action="store_true",
                        help="donor = the simulator's analytic operators (M1^-1=I/m, M2^-1=J^-1, V=mgz, g=S, Dv=mc(1+|v|)I, Dw=c(1+|w|)J)")
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--dissipation", choices=("on", "off"), default="on")
    parser.add_argument("--controller-seconds", type=float, default=20.0)
    parser.add_argument("--evaluation-dataset", type=Path, default=None)
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def _load(run_dir, device):
    run = evaluation.load_run(Path(run_dir), None); config = run["config"]
    model, params, gp_setup = evaluation.build_model(run)
    label = _model_label(config, _noise_level(config)).replace("\n", " ")
    return {"run": run, "config": config, "model": model, "label": label, "short": label.split()[0], "dir": Path(run_dir)}


def main():
    args = parse_args(); device = jax.devices()[0]
    first = load_config(sorted(Path(args.run_a).glob("*.yaml"))[0])
    dataset = Path(args.evaluation_dataset) if args.evaluation_dataset else Path(resolve_project_path(first["report"]["evaluation_dataset"]))
    data = evaluation.load_common_dataset(dataset); vehicle = evaluation.vehicle_constants(data["settings"])
    use_dissipation = args.dissipation == "on"
    a = _load(args.run_a, device); b = _load(args.run_b, device) if args.run_b else None
    report_dir = EVAL_ROOT / (args.output_name or f"{datetime.now().astimezone().strftime('%d-%m-%H-%M')}_controller-swap-ablation_ff-{args.dissipation}")
    report_dir.mkdir(parents=True, exist_ok=True)
    results = {}          # (base short, donor short, product or "none", reference) -> comparison dict + plots
    if args.donor_ground_truth:
        gt = {"model": evaluation.ground_truth_model(vehicle), "label": "ground-truth operators (simulator constants)", "short": "GT",
              "config": a["config"], "dir": report_dir / "ground-truth-operators",
              "run": {"metadata": {"model_name": "ground_truth", "integrator": "n/a", "dataset_sha256": None},
                      "directory": "n/a", "checkpoint": Path("ground-truth"), "checkpoint_sha256": "n/a"}}
        pairs = [(m, gt) for m in (a, b) if m is not None]
        for reference in args.reference:            # the reference row: the simulator's own operators in the same controller
            print(f"flying GT | ground-truth operators alone | {reference}", flush=True)
            out = run_controller(gt["run"], gt["model"], gt["dir"] / f"controller_{reference}_ff-{args.dissipation}",
                                 model_label="ground-truth operators", training_dataset=dataset, vehicle=vehicle,
                                 duration_seconds=args.controller_seconds, seed=int(a["config"]["report"]["random_seed"]),
                                 device_text=f"{device.platform}:{device.id}", use_dissipation=use_dissipation, reference=reference,
                                 provider=JaxProvider(gt["model"]), provider_description="the simulator's analytic operators")
            results[("GT", "GT", "none", reference)] = {"comparison": controller_comparison(out), "plots": out["metadata"]["plots"], "text": "ground-truth operators alone"}
    else:
        if b is None:
            raise SystemExit("--run-b is required unless --donor-ground-truth is given")
        gt = None; pairs = [(a, b), (b, a)]
    for base, donor in pairs:
        providers = {"none": (JaxProvider(base["model"]), f"{base['short']} own operators")}
        for product in SWAP_PRODUCTS:
            providers[product] = (HybridProvider(JaxProvider(base["model"]), JaxProvider(donor["model"]), product),
                                  f"{base['short']} with {PRODUCT_TEXT[product]} from {donor['short']}")
        for reference in args.reference:
            for product, (provider, text) in providers.items():
                folder = base["dir"] / (f"controller_{reference}_ff-{args.dissipation}" if product == "none"
                                        else f"controller_swap-{product}-from-{donor['short']}_{reference}_ff-{args.dissipation}")
                if product == "none" and (folder / "controller_metadata.json").exists():
                    meta = json.loads((folder / "controller_metadata.json").read_text())
                    z = np.load(folder / "controller_rollout.npz")
                    out = {"directory": folder, "metadata": meta, "data": {"s_traj": z["s_traj"], "s_plan": z["s_plan"], "requested_rpm": z["requested_rpm"]}}
                    print(f"reusing {folder.name}", flush=True)
                else:
                    print(f"flying {base['short']} | {text} | {reference}", flush=True)
                    out = run_controller(base["run"], base["model"], folder, model_label=text,
                                         training_dataset=resolve_project_path(base["config"]["data"]["dataset_path"]),
                                         vehicle=vehicle, duration_seconds=args.controller_seconds,
                                         seed=int(base["config"]["report"]["random_seed"]), device_text=f"{device.platform}:{device.id}",
                                         use_dissipation=use_dissipation, reference=reference, provider=provider, provider_description=text)
                results[(base["short"], donor["short"], product, reference)] = {"comparison": controller_comparison(out), "plots": out["metadata"]["plots"], "text": text}

    # ---- report ----
    pages = []
    for base, donor in pairs:
        rows = ["none"] + list(SWAP_PRODUCTS)
        row_labels = [f"{base['short']} own operators"] + [f"{PRODUCT_TEXT[p]} <- {donor['short']}" for p in SWAP_PRODUCTS]
        if gt is not None:
            rows.append("GT"); row_labels.append("ground-truth operators alone (reference)")
            for reference in args.reference:
                results[(base["short"], donor["short"], "GT", reference)] = results[("GT", "GT", "none", reference)]
        cols, cells = [], []
        for reference in args.reference:
            for key, name in (("position_rmse_m", "RMSE"), ("position_max_m", "max"), ("shared_motor_saturation_fraction", "sat")):
                cols.append(f"{reference}\n{name}")
        for product in rows:
            line = []
            for reference in args.reference:
                c = results[(base["short"], donor["short"], product, reference)]["comparison"]
                line += [f"{c['position_rmse_m']:.3f}", f"{c['position_max_m']:.2f}", f"{c['shared_motor_saturation_fraction']:.2f}"]
            cells.append(line)
        fig, axis = plt.subplots(figsize=(15, 7.5)); axis.axis("off")
        table = axis.table(cellText=cells, rowLabels=row_labels, colLabels=cols, cellLoc="center", rowLoc="left", loc="center")
        table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1.0, 2.0)
        for i in range(len(cols)): table[0, i].set_text_props(weight="bold")
        for r in range(1, len(rows) + 1): table[r, -1].set_text_props(weight="bold" if r == 1 else "normal")
        axis.set_title(f"Subnetwork-swap ablation — base {base['label']} flying with ONE product taken from {donor['label']}\n"
                       f"damping feedforward {args.dissipation} for every flight; same controller, gains, references and start point",
                       fontsize=11, fontweight="bold", pad=14)
        axis.text(0.5, 0.02, "Row 1 is the base model unchanged. Each other row replaces exactly one gauge-invariant product with the donor's, "
                  "re-expressed in the base gauge (HybridProvider).\nThe row that departs from row 1 names the operator carrying the "
                  "closed-loop difference; rows that stay put rule theirs out.  RMSE and max in metres, sat = motor saturation fraction.",
                  transform=axis.transAxes, ha="center", va="bottom", fontsize=8.5)
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 1.0)); pages.append(fig)
        # bar chart of RMSE per swap per reference
        fig, axes = plt.subplots(1, len(args.reference), figsize=(6.5 * len(args.reference), 5.5), squeeze=False)
        for axis, reference in zip(axes[0], args.reference):
            vals = [results[(base["short"], donor["short"], p, reference)]["comparison"]["position_rmse_m"] for p in rows]
            colours = ["#555555"] + ["#CC79A7" if base is a else "#009E73"] * len(SWAP_PRODUCTS) + (["#000000"] if gt is not None else [])
            axis.bar(range(len(rows)), vals, color=colours); axis.set_yscale("log")
            axis.set_xticks(range(len(rows))); axis.set_xticklabels(["own"] + list(SWAP_PRODUCTS) + (["GT alone"] if gt is not None else []), rotation=25, ha="right", fontsize=9)
            axis.set_ylabel("position RMSE (m), log"); axis.set_title(f"{base['short']} on {reference}: which donor product breaks it?", fontsize=10); axis.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"base {base['short']}, donor {donor['short']}", fontweight="bold"); fig.tight_layout(rect=(0, 0, 1, 0.94)); pages.append(fig)
        for reference in args.reference:
            for product in rows:
                r = results[(base["short"], donor["short"], product, reference)]
                pages.append(figures.image_page(Path(r["plots"]["labeled_trajectory_plot"]), f"{r['text']} — {reference}"))
    out_pdf = report_dir / "controller-swap-ablation.pdf"
    with PdfPages(out_pdf) as pdf:
        for fig in pages:
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    (report_dir / "swap_results.json").write_text(json.dumps(
        {"|".join(k): {"text": v["text"], **{kk: vv for kk, vv in v["comparison"].items() if not isinstance(vv, dict)}} for k, v in results.items()},
        indent=2, default=str) + "\n")
    print(f"PDF: {out_pdf}  ({len(pages)} pages)"); print(f"Eval run: {report_dir}")


if __name__ == "__main__":
    sys.exit(main())

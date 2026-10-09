"""Quadrotor model-comparison campaign (6 Oct 2026): Lie-PH-GP-SDE against PH-NODE on two SE(3) datasets.

    python src/models/SE3_Quadrotor/comparison/campaign.py make
    python src/models/SE3_Quadrotor/comparison/campaign.py run --gpu 0      (and --gpu 1)

Datasets (envs/quadrotor_se3_pybullet/datagen/configs; 300 training flights of 5 s at 50 Hz with fast manoeuvres and
vertical excitation, PID reads 0.01-noise measurements, observation noise 0.25, constant wind 0.5 on both twist channels,
damping c = 0.5), evaluated on the 10 s held-out flights of their -EVAL10s companions:
    DampRate-Wind   rate-dependent damping -c m (1+|v_b|) v_b, -c J (1+|w_b|) w_b
    DampConst-Wind  constant damping -c m v_b, -c J w_b
Models (all: float32, no pretraining, no priors; the protocol of the base config: 2 s windows with 25 % overlap,
batch 16, 5000 steps, final model, beta 1):
    Lie-PH-GP-SDE  GP, wind, Lie-IMEX, EKF loss; learned GP amplitude / length-scale, Prodigy levels and
                   hyperparameters (d0 0.01), random start spread 0.1
    PH-NODE        MLP, no wind, RK4, rollout loss (the prior work's loss), random start spread 1
MODELS also defines Lie-PH-GP-ODE, Lie-PH-NN-SDE and Lie-PH-NN-ODE (report.py / closed_loop.py --models).
One shared priority list (DampRate-Wind first); every GPU runs one job at a time and claims the next unclaimed one.
Every run is its own process with a time limit and is evaluated right after training; one JSON line per run in
experiments/quadrotor/campaign_06-10-2026/results_gpu<N>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
PACKAGES = {"lie_ph": THIS_DIR.parent / "lie_ph", "ph_node": THIS_DIR.parent / "ph_node"}
SCHEMAS = {"lie_ph": "lie_ph/quadrotor/v1", "ph_node": "ph_node/quadrotor/v1"}
CONFIG_DIR = THIS_DIR.parent / "configs" / "pybullet-campaign-06-10-2026"      # every config of this campaign
BASE_CONFIG = CONFIG_DIR / "base.yaml"
RESULT_DIR = PROJECT_ROOT / "experiments" / "quadrotor" / "campaign_06-10-2026"
TRAINING_NOTE = ("training flights with vertical excitation (rms v_bz / v_bxy 0.61 const, 0.90 rate), {count} noisy flights, "
                 "2 s windows, 25 % overlap, batch 16, 5000 steps, final model, float32, no priors, no pretraining. "
                 "Lie-PH-GP-SDE: learned GP amplitude / length-scale (type-II ML), Prodigy levels d0 0.01 and hyperparameters "
                 "d0 0.01, random start spread 0.1. PH-NODE: MLP, RK4, rollout loss, spread 1.")
NOISE = 0.25

MODELS = {   # name: (family, wind, integrator, loss)
    "Lie-PH-GP-SDE": ("gp", True, "lie_imex", "ekf"),
    "Lie-PH-GP-ODE": ("gp", False, "lie_imex", "ekf"),
    "Lie-PH-NN-SDE": ("nn", True, "lie_imex", "ekf"),
    "Lie-PH-NN-ODE": ("nn", False, "lie_imex", "rollout"),
    "PH-NODE": ("nn", False, "rk4", "rollout"),
}
CAMPAIGN_MODELS = ("Lie-PH-GP-SDE", "PH-NODE")
MODEL_OVERRIDES = {   # on top of the base config and the family / wind / integrator / loss above
    "Lie-PH-GP-SDE": {"model": {"initial_randomness": 0.1, "learn_gp_hyperparameters": True},
                      "optimizer": {"level_initial_distance": 0.01, "hyper_optimizer": "prodigy", "hyper_initial_distance": 0.01}},
    "PH-NODE": {"model": {"initial_randomness": 1.0}},
}
SETTINGS = {  # name: dataset folder name (without QUADROTOR-DATASET-), wind in the data
    "DampRate-Wind": ("SDE-DiffusionConstant-0p5-DissipationRateDependent-0p5-5s-NoisyPID0p01-50Hz-300flights", True),
    "DampConst-Wind": ("SDE-DiffusionConstant-0p5-DissipationConstant-0p5-5s-NoisyPID0p01-50Hz-300flights", True),
}
PRIORITY = ("DampRate-Wind", "DampConst-Wind")
TIME_LIMIT = {"ekf": 120 * 60, "rollout": 60 * 60}
EVAL_TIME_LIMIT = 30 * 60
FINISHED = ("done", "eval_failed", "train_failed", "timeout")


def run_name(model: str, setting: str, level: float = NOISE) -> str:
    return f"{model}_{setting}_noise{level:g}".replace(".", "p")


def dataset_path(setting: str, kind: str = "noisy") -> str:
    name = SETTINGS[setting][0]
    suffix = "train-obs-noise-absolute0p25" if kind == "noisy" else "clean"
    return f"datasets/QUADROTOR-DATASET-{name}/{name}_CF2P_5s_h0p02_{suffix}.pkl"


def package_of(integrator: str) -> str:
    """rk4 models (PH-NODE) are ph_node's, everything else lie_ph's."""
    return "ph_node" if integrator == "rk4" else "lie_ph"


def make() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for setting in PRIORITY:
        for model in CAMPAIGN_MODELS:
            family, wind, integrator, loss = MODELS[model]
            name = run_name(model, setting)
            config = json.loads(json.dumps(base))
            config["created_at"] = "2026-10-06T00:50:00"
            config["data"]["dataset_path"] = dataset_path(setting)
            config["model"].update({"family": family, "wind": wind, "integrator": integrator, "nn_hidden": 20, "nn_layers": 2})
            config["training"]["loss"] = loss
            for section, values in MODEL_OVERRIDES[model].items():
                config[section].update(values)
            config["experiment"]["root"] = str((RESULT_DIR / "runs").relative_to(PROJECT_ROOT))
            config["experiment"]["name_suffix"] = name
            config["experiment"]["notes"] = [f"Quadrotor campaign 6 Oct 2026: {model} on {setting}, observation noise {NOISE:g}."]
            config["schema"] = SCHEMAS[package_of(integrator)]
            path = CONFIG_DIR / f"{name}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            entries.append(str(path.relative_to(PROJECT_ROOT)))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "queue_shared.txt").write_text("\n".join(entries) + "\n")
    print(f"queue_shared: {len(entries)} runs")


def summary(evaluation: dict) -> dict:
    out = {}
    learned = evaluation.get("learned", {})
    for key, value in learned.get("operators", {}).items():
        if isinstance(value, dict) and "relative_error" in value:
            out[f"{key}_rel_err"] = round(value["relative_error"], 4)
    for key in ("test_ekf_nll_noisy", "test_nis_per_dimension", "sigma_obs", "diffusion"):
        if key in learned:
            out[key] = learned[key]
    return out


def finished_configs() -> set[str]:
    done = set()
    for path in RESULT_DIR.glob("results_*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("status") in FINISHED:
                    done.add(record["config"])
    return done


def run_one(config: str, gpu: int) -> dict:
    cfg = yaml.safe_load((PROJECT_ROOT / config).read_text())
    name, loss = cfg["experiment"]["name_suffix"], cfg["training"]["loss"]
    package = PACKAGES[package_of(str(cfg["model"].get("integrator", "lie_imex")))]
    record = {"config": config, "name": name, "gpu": gpu, "started": datetime.now().isoformat(timespec="seconds")}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    logs = RESULT_DIR / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"{name}.log"
    tic = time.time()
    try:
        with log_path.open("w") as handle:
            proc = subprocess.run([sys.executable, str(package / "train.py"), "--config", str(PROJECT_ROOT / config)],
                                  stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT, timeout=TIME_LIMIT[loss])
        record["status"] = "trained" if proc.returncode == 0 else "train_failed"
    except subprocess.TimeoutExpired:
        record["status"] = "timeout"
    record["train_minutes"] = round((time.time() - tic) / 60, 1)
    text = log_path.read_text(errors="replace")
    run_dirs = [line.split("] run ", 1)[1].strip() for line in text.splitlines() if "] run " in line]
    record["run_dir"] = run_dirs[0] if run_dirs else None
    record["non_finite_steps"] = text.count("non-finite")
    if record["status"] == "trained" and record["run_dir"]:
        tic = time.time()
        try:
            proc = subprocess.run([sys.executable, str(package / "evaluate.py"), "--run", record["run_dir"]],
                                  capture_output=True, text=True, env=env, cwd=PROJECT_ROOT, timeout=EVAL_TIME_LIMIT)
            evaluation = Path(record["run_dir"]) / "evaluation_final.json"
            if evaluation.exists():
                record["status"] = "done"
                record.update(summary(json.loads(evaluation.read_text())))
            else:
                record["status"] = "eval_failed"
                record["eval_error"] = (proc.stdout + proc.stderr)[-500:]
        except subprocess.TimeoutExpired:
            record["status"] = "eval_failed"
            record["eval_error"] = "evaluation timeout"
        record["eval_minutes"] = round((time.time() - tic) / 60, 1)
    if record["status"] == "train_failed":
        record["error_tail"] = text[-500:]
    return record


def run_shared(gpu: int) -> None:
    entries = [line.strip() for line in (RESULT_DIR / "queue_shared.txt").read_text().splitlines() if line.strip()]
    claims = RESULT_DIR / "claims"
    claims.mkdir(exist_ok=True)
    for config in entries:
        if config in finished_configs():
            continue
        try:
            os.close(os.open(claims / Path(config).stem, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue                                          # the other GPU has it
        record = run_one(config, gpu)
        with (RESULT_DIR / f"results_gpu{gpu}.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[{datetime.now():%H:%M:%S}] {record['status']:12s} {record['name']}  train {record.get('train_minutes')} min", flush=True)
    print(f"[{datetime.now():%H:%M:%S}] GPU {gpu}: no unclaimed jobs left", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make")
    runner = sub.add_parser("run")
    runner.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    make() if args.command == "make" else run_shared(args.gpu)


if __name__ == "__main__":
    main()

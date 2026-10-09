"""Pendulum model-comparison campaign (agreed 4 Oct 2026): configs, a shared GPU job list, an unattended runner.

    python src/models/3D_SO3_Windy_Pendulum/comparison/campaign.py make
    python src/models/3D_SO3_Windy_Pendulum/comparison/campaign.py run --gpu 0     (and --gpu 1)

Models (all: float32, no pretraining, no priors, random start spread 1 for levels and noise scales, the common protocol
of the base config: 100 trajectories of 5 s, 2 s windows with 25 % overlap, batch 64, 5000 steps, final model):
    Lie-PH-GP-SDE  GP, wind, Lie-IMEX, EKF loss            Lie-PH-GP-ODE  GP, no wind, Lie-IMEX, EKF loss
    Lie-PH-NN-SDE  MLP, wind, Lie-IMEX, EKF loss           Lie-PH-NN-ODE  MLP, no wind, Lie-IMEX, rollout loss
    PH-NODE        MLP, no wind, RK4, rollout loss (the prior work's loss)
The Lie-PH models are trained by ../lie_ph, the PH-NODE baselines by ../ph_node (configs in
../configs/old/campaign-04-10-2026), the NeuralSDE by ../neural_sde.
Settings: damping constant / rate-dependent (c = 0.5), each without and with constant wind (0.5); the SDE models are
trained on the wind settings only. Observation noise 0, 0.01, 0.05, 0.25, 0.5, 0.75 (0 = the clean pickle).
Datasets: input matrix G = diag(0.5, 0.7, 0.8) (folder suffix ``-G0p5-0p7-0p8``; the env default).

One shared priority list: DampRate-Wind, then DampConst-Wind, DampRate-NoWind, DampConst-NoWind; every GPU runs one job
at a time and claims the next unclaimed one (claim files), so the first setting finishes first. Every run is its own
process with a time limit; it is evaluated right after training; one JSON line per run is appended to
experiments/pendulum_so3/campaign_04-10-2026/results_gpu<N>.jsonl (status, run folder, key metrics). 8 of the runs
(GP-SDE / NN-SDE, wind settings, noise 0.25 / 0.75) come from the identical G-check runs, recorded in
results_reused.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
PACKAGES = {"lie_ph": THIS_DIR.parent / "lie_ph", "ph_node": THIS_DIR.parent / "ph_node"}
SCHEMAS = {"lie_ph": "lie_ph/pendulum/v1", "ph_node": "ph_node/pendulum/v1"}
CONFIG_ROOT = THIS_DIR.parent / "configs"         # every config of every model, one folder per experiment
BASE_CONFIG = CONFIG_ROOT / "old" / "dev" / "03-10-2026-20-00_lie_ph_5s-beta1.yaml"
CAMPAIGN_CONFIGS = CONFIG_ROOT / "old" / "campaign-04-10-2026"
RESULT_DIR = PROJECT_ROOT / "experiments" / "pendulum_so3" / "campaign_04-10-2026"

MODELS = {   # name: (family, wind, integrator, loss)
    "Lie-PH-GP-SDE": ("gp", True, "lie_imex", "ekf"),
    "Lie-PH-GP-ODE": ("gp", False, "lie_imex", "ekf"),
    "Lie-PH-NN-SDE": ("nn", True, "lie_imex", "ekf"),
    "Lie-PH-NN-ODE": ("nn", False, "lie_imex", "rollout"),
    "PH-NODE": ("nn", False, "rk4", "rollout"),
    "PH-NODE-ref": ("nn", False, "rk4", "rollout_reference"),     # Duong & Atanasov's network, loss and optimizer (6 Oct 2026)
    "PH-NODE-ref-pretrain": ("nn", False, "rk4", "rollout_reference"),   # the same + their M^-1 -> I pretraining
    # NeuralSDE (7 Oct 2026, user option A): the unstructured neural SDE of src/models/3D_SO3_Windy_Pendulum/neural_sde
    # trained by ITS OWN trainer (path MSE + geodesic loss, 5-point windows, Heun); report.py loads its .eqx checkpoint.
    "NeuralSDE": ("neural", True, "heun", "path_mse"),
}
# Added after the campaign: shown in a report only where a run of them is recorded (not "training failed" elsewhere).
OPTIONAL_MODELS = ("PH-NODE-ref", "PH-NODE-ref-pretrain", "NeuralSDE")
# PH-NODE-ref (added 6 Oct 2026, after the campaign): their DissipativeSO3HamNODE trained their way on our data:
# 5-sample chunks (4 RK4 steps of 0.05 s), full batch, Adam 1e-3 constant with weight decay 1e-4, no clipping,
# orthogonal init gain 0.5, 3 782 parameters; our budget of 5000 steps, float32, no pretraining.
MODEL_OVERRIDES = {
    "PH-NODE-ref": {"data": {"window_points": 5, "window_stride": 1},
                    "model": {"nn_architecture": "duong", "nn_init_gain": 0.5},
                    "training": {"batch_size": 1000000},
                    "optimizer": {"learning_rate": 1.0e-3, "schedule": "constant", "weight_decay": 1.0e-4,
                                  "gradient_clip_norm": None}},
}
MODEL_OVERRIDES["PH-NODE-ref-pretrain"] = {**MODEL_OVERRIDES["PH-NODE-ref"],
                                           "model": {**MODEL_OVERRIDES["PH-NODE-ref"]["model"], "pretrain_inverse_mass_identity": True}}
SETTINGS = {  # setting name: (dataset name, wind in the data)
    "DampConst-Wind": ("SDE-DiffusionConstant-0p5-DissipationConstant-0p5-5s-G0p5-0p7-0p8", True),
    "DampConst-NoWind": ("ODE-DissipationConstant-0p5-5s-G0p5-0p7-0p8", False),
    "DampRate-Wind": ("SDE-DiffusionConstant-0p5-DissipationRateDependent-0p5-5s-G0p5-0p7-0p8", True),
    "DampRate-NoWind": ("ODE-DissipationRateDependent-0p5-5s-G0p5-0p7-0p8", False),
}
CREATED_AT = "2026-10-04T15:00:00"
PRIORITY = ("DampRate-Wind", "DampConst-Wind", "DampRate-NoWind", "DampConst-NoWind")     # shared-queue order
NOISE_ORDER = (0.25, 0.05, 0.5, 0.01, 0.75, 0.0)
TIME_LIMIT = {"ekf": 45 * 60, "rollout": 30 * 60, "rollout_reference": 30 * 60}
EVAL_TIME_LIMIT = 20 * 60


def package_of(integrator: str) -> str:
    """rk4 models (PH-NODE, PH-NODE-ref) are ph_node's, everything else lie_ph's."""
    return "ph_node" if integrator == "rk4" else "lie_ph"


def noise_tag(level: float) -> str:
    return "clean" if level == 0 else f"obs-noise{level:g}".replace(".", "p")


def run_name(model: str, setting: str, level: float) -> str:
    return f"{model}_{setting}_noise{level:g}".replace(".", "p")


def dataset_name(setting: str) -> str:
    return SETTINGS[setting][0]


def make() -> None:
    """Write the configs and the shared priority list (queue_shared.txt) in PRIORITY order, setting-major."""
    base = yaml.safe_load(BASE_CONFIG.read_text())
    jobs = [(level, setting) for setting in PRIORITY for level in NOISE_ORDER]
    root = str(RESULT_DIR.relative_to(PROJECT_ROOT) / "runs")
    label = "Campaign 4 Oct 2026 (G = diag(0.5, 0.7, 0.8))"
    entries: list[str] = []
    for level, setting in jobs:
        dataset, data_wind = dataset_name(setting), SETTINGS[setting][1]
        for model, (family, wind, integrator, loss) in MODELS.items():
            if model in OPTIONAL_MODELS:
                continue                              # added after the campaign; trained separately
            if wind and not data_wind:
                continue                              # no SDE models on the wind-free data
            name = run_name(model, setting, level)
            config = json.loads(json.dumps(base))
            config["created_at"] = CREATED_AT
            config["data"]["dataset_path"] = f"datasets/PENDULUM-DATASET-{dataset}/{dataset}_{noise_tag(level)}.pkl"
            config["model"].update({"family": family, "wind": wind, "integrator": integrator,
                                    "initial_randomness": 1.0, "nn_hidden": 20, "nn_layers": 2})
            config["training"]["loss"] = loss
            for section, values in MODEL_OVERRIDES.get(model, {}).items():
                config[section].update(values)
            config["experiment"]["root"] = root
            config["experiment"]["name_suffix"] = name
            config["experiment"]["notes"] = [f"{label}: {model} on {setting}, observation noise {level:g}."]
            package = package_of(integrator)
            config["schema"] = SCHEMAS[package]
            CAMPAIGN_CONFIGS.mkdir(parents=True, exist_ok=True)
            path = CAMPAIGN_CONFIGS / f"{name}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            entries.append(str(path.relative_to(PROJECT_ROOT)))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "queue_shared.txt").write_text("\n".join(entries) + "\n")
    print(f"queue_shared: {len(entries)} runs")


def summary(evaluation: dict) -> dict:
    """The few numbers worth a glance in the results table."""
    out = {}
    learned = evaluation.get("learned", {})
    ops = learned.get("operators", {})
    for key in ("damping_M_inv_D", "control_M_inv_g", "gravity_acceleration"):
        if key in ops:
            out[f"{key}_rel_err"] = round(ops[key]["relative_error"], 4)
    for key in ("test_ekf_nll_noisy", "test_nis_per_dimension", "sigma_obs", "diffusion"):
        if key in learned:
            out[key] = learned[key]
    loop = learned.get("open_loop_clean_test", {})
    for horizon in ("t=1.00s", "t=2.00s", "t=5.00s"):
        if horizon in loop:
            out[f"omega_rmse_{horizon}"] = round(loop[horizon]["omega_rmse"], 4)
    return out


FINISHED = ("done", "eval_failed", "train_failed", "timeout")


def finished_configs(result_dir: Path) -> set[str]:
    """Configs already recorded as finished in any results_*.jsonl of the campaign (resume support)."""
    done = set()
    for path in result_dir.glob("results_*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("status") in FINISHED:
                    done.add(record["config"])
    return done


def run_one(config: str, record: dict, env: dict, logs: Path) -> dict:
    """Train one config (own process, time limit), evaluate it, return the filled-in record."""
    cfg = yaml.safe_load((PROJECT_ROOT / config).read_text())
    name, loss = cfg["experiment"]["name_suffix"], cfg["training"]["loss"]
    package = PACKAGES[package_of(str(cfg["model"].get("integrator", "lie_imex")))]
    record.update({"config": config, "name": name, "started": datetime.now().isoformat(timespec="seconds")})
    log_path = logs / f"{name}.log"
    tic = time.time()
    try:
        with log_path.open("w") as handle:
            proc = subprocess.run([sys.executable, str(package / "train.py"), "--config", str(PROJECT_ROOT / config)],
                                  stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT,
                                  timeout=TIME_LIMIT[loss])
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


def gpu_env(gpu: int) -> dict:
    return {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def run_shared(gpu: int) -> None:
    """Every GPU walks the one priority list and runs the first job that is neither finished nor claimed by another
    GPU (claim = exclusive creation of claims/<name>), one job at a time."""
    import os
    result_dir = RESULT_DIR
    entries = [line.strip() for line in (result_dir / "queue_shared.txt").read_text().splitlines() if line.strip()]
    results = result_dir / f"results_gpu{gpu}.jsonl"
    logs, claims = result_dir / "logs", result_dir / "claims"
    logs.mkdir(exist_ok=True)
    claims.mkdir(exist_ok=True)
    for config in entries:
        if config in finished_configs(result_dir):
            continue
        try:
            os.close(os.open(claims / Path(config).stem, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue                                         # the other GPU has it
        record = run_one(config, {"gpu": gpu}, gpu_env(gpu), logs)
        with results.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[{datetime.now():%H:%M:%S}] {record['status']:12s} {record['name']}  train {record.get('train_minutes')} min", flush=True)
    print(f"[{datetime.now():%H:%M:%S}] GPU {gpu}: no unclaimed jobs left", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make")
    sub.add_parser("run").add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.command == "make":
        make()
    else:
        run_shared(args.gpu)


if __name__ == "__main__":
    main()

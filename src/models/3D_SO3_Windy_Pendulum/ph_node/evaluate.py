"""Ground-truth evaluation of a trained PH-NODE run (the same protocol and output as lie_ph/evaluate.py).

    python src/models/3D_SO3_Windy_Pendulum/ph_node/evaluate.py --run <run_dir>

The learned model is rolled out with RK4; the "analytical" reference (lie_ph's model class with the true operators)
with Lie-IMEX, as in lie_ph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent, PROJECT_ROOT / "envs" / "pendulum_so3"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lie_ph.config import resolve_project_path  # noqa: E402
from lie_ph.data import load_noisy_windows  # noqa: E402
from lie_ph.evaluate import analytical_model, analytical_params, load_run, load_truth, make_env, open_loop, operator_errors  # noqa: E402
from lie_ph.integrator import rollout as lie_imex_rollout  # noqa: E402
from lie_ph.losses import ekf_consistency, ekf_nll  # noqa: E402
from ph_node.integrator import rollout as rk4_rollout  # noqa: E402
from ph_node.network import model_from_params  # noqa: E402


def evaluate_run(run: Path, which: str, paths: int) -> dict:
    config, setup, params, payload = load_run(run, which)
    dataset = resolve_project_path(config["data"]["dataset_path"])
    substeps = int(config["data"]["substeps"])
    _, noisy_test, interval = load_noisy_windows(dataset, config["data"].get("window_points"),
                                                 config["data"].get("window_stride"))
    truth = load_truth(dataset)
    env = make_env(truth)
    clean = truth["clean_test"]
    states = clean[:, :, :].reshape(-1, 15)[:: max(1, clean.shape[0] * clean.shape[1] // 2000)]
    analytical = analytical_params(truth, setup, config["model"])
    report = {"run": str(run), "checkpoint": which, "step": int(payload["step"]), "dataset": str(dataset)}
    report["model"] = {k: config["model"].get(k) for k in ("family", "wind", "integrator")} | {"loss": config["training"].get("loss")}

    learned = lambda p, key: model_from_params(p, setup, config["model"])           # noqa: E731
    report["learned"] = {"operators": operator_errors(learned(params, None), states, truth, env),
                         "open_loop_clean_test": open_loop(params, learned, clean, interval, substeps, paths, 0,
                                                           rollout_fn=rk4_rollout)}

    reference = lambda p, key: analytical_model(p, setup, truth, key)               # noqa: E731
    entry = {"operators": operator_errors(reference(analytical, None), states, truth, env),
             "open_loop_clean_test": open_loop(analytical, reference, clean, interval, substeps, paths, 0,
                                               rollout_fn=lie_imex_rollout)}
    nll = jax.jit(lambda p, w: ekf_nll(reference(p, None), w, p["likelihood"]["log_sigma"], interval, substeps))
    nis = jax.jit(lambda p, w: ekf_consistency(reference(p, None), w, p["likelihood"]["log_sigma"], interval, substeps))
    chunks = [jnp.asarray(noisy_test[:, i:i + 128]) for i in range(0, noisy_test.shape[1], 128)]
    weights = np.asarray([c.shape[1] for c in chunks], float)
    entry["sigma_obs"] = np.exp(np.asarray(analytical["likelihood"]["log_sigma"])).round(4).tolist()
    entry["test_ekf_nll_noisy"] = float(np.average([float(nll(analytical, c)) for c in chunks], weights=weights))
    entry["test_nis_per_dimension"] = float(np.average([float(nis(analytical, c)) for c in chunks], weights=weights))
    if "process" in analytical:
        entry["diffusion"] = np.exp(np.asarray(analytical["process"]["log_sigma"])).round(4).tolist()
    report["analytical"] = entry
    report["truth"] = {"sigma_obs": truth["obs_noise"], "diffusion": truth["diffusion"].tolist(),
                       "damping_rate": (truth["friction"] / truth["inertia"]).tolist(), "friction_law": truth["friction_law"],
                       "control_gain": (truth["g_diag"] / truth["inertia"]).tolist()}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, nargs="+", required=True, help="run directories (or checkpoint files)")
    parser.add_argument("--checkpoint", default="final", choices=("final", "best"))
    parser.add_argument("--paths", type=int, default=32, help="sampled paths for the band coverage (zero width here)")
    args = parser.parse_args()
    for run in args.run:
        try:
            report = evaluate_run(run, args.checkpoint, args.paths)
        except Exception as error:             # one broken run must not stop the rest
            print(f"FAILED {run}: {type(error).__name__}: {error}", flush=True)
            continue
        out = (run if run.is_dir() else run.parent) / f"evaluation_{args.checkpoint}.json"
        out.write_text(json.dumps(report, indent=1))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

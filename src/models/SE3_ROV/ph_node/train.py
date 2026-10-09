"""Train the BlueROV2 PH-NODE baseline (MLP port-Hamiltonian model, RK4, the prior work's trajectory loss).

    python src/models/SE3_ROV/ph_node/train.py --config src/models/SE3_ROV/configs/<experiment>/<PH-NODE file>.yaml

The training loop is lie_ph's (lie_ph/train.py: same data windows, Adam with clipping and cosine decay, fixed budget,
final checkpoint, a non-finite step aborts the run) with PH-NODE's model, RK4 rollout loss and strict config.
Optional training.l1_coefficient (default None = off): adds c * losses.l1_structure (the original PH-NODE real-flight
code's L1 term on G, D_v, D_omega) to the objective; the logged data fit and the monitor exclude it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=None, help="override training.total_steps (smoke tests)")
    parser.add_argument("--root", type=Path, default=None, help="override experiment.root (smoke tests)")
    return parser.parse_args()


ARGS = _parse() if __name__ == "__main__" else None
if ARGS is not None:   # the GPU must be chosen before jax is imported
    _runtime = yaml.safe_load(ARGS.config.read_text())["runtime"]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_runtime["gpu"]))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from lie_ph.train import train as train_loop  # noqa: E402
from ph_node.config import load_config  # noqa: E402
from ph_node.losses import l1_structure, rollout_loss  # noqa: E402
from ph_node.network import model_from_params  # noqa: E402


def l1_term(config: dict, log):
    """training.l1_coefficient c -> fn(model, batch) = c * l1_structure(model, batch); None when not set."""
    coefficient = config["training"].get("l1_coefficient")
    if coefficient is None:
        return None
    log(f"L1 term on G, D_v, D_omega: coefficient {float(coefficient):g}")
    return lambda model, batch: float(coefficient) * l1_structure(model, batch)


COMPONENTS = {"load_config": load_config, "model_from_params": model_from_params,
              "data_fit": {"rollout": rollout_loss}, "extra_term": l1_term, "package": "ph_node"}


def train(config_path: Path, total_steps_override: int | None = None, root_override: Path | None = None) -> Path:
    return train_loop(config_path, total_steps_override, root_override, COMPONENTS)


if __name__ == "__main__":
    train(ARGS.config, ARGS.total_steps, ARGS.root)

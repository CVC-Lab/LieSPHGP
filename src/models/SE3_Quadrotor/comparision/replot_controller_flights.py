"""Redraw the tracking and trajectory PNGs of recorded closed-loop flights.

The flight itself is never repeated: everything the two figures need (s_traj, s_plan, s_plan_full) is already
inside controller_rollout.npz, so a change to save_plots can be rolled out over existing flights in seconds.

    python -m src.models.SE3_Quadrotor.comparision.replot_controller_flights <dir> [<dir> ...]

Every directory is searched recursively for controller_rollout.npz; the PNGs are rewritten in place, so the
plot paths already recorded in controller_metadata.json stay valid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.models.SE3_Quadrotor.comparision.report_controller import save_plots


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", help="flight folders, or any parent of them")
    parser.add_argument("--baseline-root", default=None,
                        help="folder holding the PH-GT flights (controller_<reference>_ff-<mode>/); when given, "
                             "each flight is redrawn with the matching ground-truth flight underneath it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flights = sorted({path.parent for directory in args.directories
                      for path in Path(directory).rglob("controller_rollout.npz")})
    if not flights:
        raise SystemExit("no controller_rollout.npz found under the given directories")
    for index, directory in enumerate(flights, start=1):
        metadata_path = directory / "controller_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        payload = np.load(directory / "controller_rollout.npz")
        baseline = None
        if args.baseline_root is not None and directory.name != Path(args.baseline_root).name:
            candidate = Path(args.baseline_root) / directory.name
            if (candidate / "controller_rollout.npz").exists() and candidate.resolve() != directory.resolve():
                baseline = np.load(candidate / "controller_rollout.npz")["s_traj"]
        plots = save_plots(directory, metadata.get("model_label", directory.parent.name),
                           payload["s_traj"], payload["s_plan"], payload["s_plan_full"],
                           metadata.get("reference"), baseline=baseline)
        if metadata:
            metadata["plots"] = plots            # same paths, rewritten for completeness
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[{index}/{len(flights)}] {metadata.get('reference', '?'):>18s}  {directory}", flush=True)


if __name__ == "__main__":
    main()

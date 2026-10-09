"""Evaluate a trained BlueROV2 PH-NODE run (lie_ph/evaluate.py's protocol and output with PH-NODE's model).

    python src/models/SE3_ROV/ph_node/evaluate.py --run <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lie_ph.evaluate import main  # noqa: E402
from ph_node.network import model_from_params  # noqa: E402

if __name__ == "__main__":
    main(model_from_params)

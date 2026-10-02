"""Timestamped experiment directories and the global quadrotor run history."""

from __future__ import annotations

import fcntl
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_project_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-_")
    return cleaned or "run"


def create_experiment(config: dict[str, Any]) -> Path:
    settings = config["experiment"]
    root = resolve_project_path(settings["root"])
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%d-%m-%H-%M")
    detail = _slug(f"{config['model']['name']}_{settings['name_suffix']}")
    candidate = root / f"{stamp}_{detail}"
    counter = 2
    while candidate.exists():
        candidate = root / f"{stamp}_{detail}_{counter:02d}"
        counter += 1
    candidate.mkdir(parents=False, exist_ok=False)
    config_path = Path(config["_config_path"])
    shutil.copy2(config_path, candidate / config_path.name)
    return candidate


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def append_history(
    config: dict[str, Any],
    experiment_dir: Path,
    *,
    action: str,
    status: str,
    details: str = "",
) -> None:
    history = resolve_project_path(config["experiment"]["history_file"])
    assert history is not None
    history.parent.mkdir(parents=True, exist_ok=True)
    changes = config["experiment"]["changes"] or ["baseline; no model change"]
    entry = (
        f"{utc_now()} | folder={experiment_dir.name} | "
        f"model={config['model']['name']} | action={action} | status={status} | "
        f"changes={'; '.join(str(item) for item in changes)} | details={details}\n"
    )
    with history.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(entry)
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


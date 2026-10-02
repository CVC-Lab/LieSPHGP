"""The plants the controller flies: the analytic rigid body and every learned model, behind one step function.

A learned plant is restored exactly as the open-loop evaluation restores it (``comparision/report_evaluation``):
posterior mean for the GP, the module itself for PH-NN / PH-NN-SDE (the SDE's drift, i.e. its mean dynamics), and
RK4 for PH-NODE.  The input encoding (wrench or rotor2) is read from the settings of the pickle the run trained on,
and the integrator step is the run's own training step (0.01 / integration_substeps), sub-stepped to fit the
controller tick.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..comparision import report_evaluation as evaluation
from ..ph_gp_lie_imex.integrator import lie_imex_step
from ..ph_node.integrator import rk4_step
from . import controller as ctl

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_STEP = 0.01


@dataclass
class Plant:
    label: str
    model: Any
    input_mode: str
    training_step: float
    rk4: bool
    run_dir: str | None = None
    selected_test: dict | None = None


def analytic_plant() -> Plant:
    """The benchmark's textbook vehicle: published m, J, g, no damping, wrench input through the selection matrix."""
    model = evaluation.ground_truth_model({"mass": ctl.MASS, "gravity": ctl.GRAVITY, "damping": 0.0,
                                           "inertia": np.diag(ctl.INERTIA)})
    return Plant("Analytic rigid body (published constants)", model, "wrench", DATA_STEP, False)


def learned_plant(label: str, run_dir: Path) -> Plant:
    run = evaluation.load_run(Path(run_dir), None)
    model, _params, _setup = evaluation.build_model(run)
    with (PROJECT_ROOT / run["config"]["data"]["dataset_path"]).open("rb") as handle:
        input_mode = pickle.load(handle)["settings"].get("input_mode", "wrench")
    substeps = int(run["config"]["model"].get("integration_substeps") or 1)
    return Plant(label, model, input_mode, DATA_STEP / substeps, evaluation.is_rk4_model(model),
                 str(Path(run_dir).resolve()), run["selected_test"])


def make_step(plant: Plant, tick: float):
    """state (B,22) with the control in 18:22 -> state after one controller tick, control held constant."""
    single = rk4_step if plant.rk4 else lie_imex_step
    count = max(1, int(round(tick / plant.training_step)))
    inner = jnp.asarray(tick / count, dtype=jnp.float64)

    def step(state):
        def body(carry, _):
            return single(plant.model, carry, inner).at[:, 18:22].set(carry[:, 18:22]), None

        out, _ = jax.lax.scan(body, state, None, length=count)
        return out

    return step, count

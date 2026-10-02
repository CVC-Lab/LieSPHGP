"""Closed-loop melon test: fly the benchmark's controller on each plant and compare with the recorded flights.

For every plant and every melon run the simulation starts from the recorded state at t = 0 and the controller
(``controller.mellinger``, 500 Hz) tracks the recorded path; the plant integrates each 2 ms tick with the rotor
speeds held.  Nothing is reset from the data after t = 0.  Three things are compared with the recording:

  * position and attitude of the simulated flight               (does it fly the same path?)
  * the rotor speeds the controller needed                      (does the plant respond like the real drone?)
  * survival time: a run is frozen and flagged once the state goes non-finite or strays > 10 m from the path.
    Divergence is reported, never repaired.

Usage (from the project root):
    python -m src.models.SE3_Quadrotor.closed_loop_idsia.run --output-name NAME [--seconds 65] [--run LABEL=DIR ...]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from . import controller as ctl  # noqa: E402
from . import plants as pl  # noqa: E402
from . import reference as rf  # noqa: E402

PROJECT_ROOT = pl.PROJECT_ROOT
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
TRAIN = PROJECT_ROOT / "experiments/quadrotor/train_runs"
TICKS = 5                                    # 500 Hz controller on the 100 Hz data grid
TICK = rf.SAMPLE_STEP / TICKS
ESCAPE_METRES = 10.0
WINDOWS = (5.0, 10.0, 30.0)

DEFAULT_RUNS = [
    ("PH-GP-LieIMEX-ODE", TRAIN / "18-09-23-15_ph_gp_lie_imex_gp-lieimex-IDSIA-real-stage1-K50-3k-scratch-melon-heldout"),
    ("PH-GP-LieIMEX-ODE rotor2", TRAIN / "22-09-17-54_ph_gp_lie_imex_idsia_gp-lieimex-ODE-IDSIA-rotor2-ONEINIT-stage1-K50-3k-nopriors-fp32-nopretrain-h0p01"),
    ("PH-NN-LieIMEX-ODE", TRAIN / "22-09-12-45_ph_nn_lie_imex_nn-lieimex-w64-IDSIA-stage1-K50-3k-nopriors-fp32-nomasspretrain-maskeval"),
    ("PH-NN-LieIMEX-ODE rotor2", TRAIN / "22-09-16-48_ph_nn_lie_imex_nn-lieimex-w64-IDSIA-rotor2-stage1-K50-3k-nopriors-fp32-nopretrain-substeps1-h0p01"),
    ("PH-NN-LieIMEX-SDE (drift)", TRAIN / "22-09-13-40_ph_nn_lie_imex_sde_nn-lieimex-SDE-S8-IDSIA-stage1-K50-3k-nopriors-fp32-nopretrain"),
    ("PH-NODE-RK4", TRAIN / "22-09-14-33_ph_node_phnode-rk4-IDSIA-stage1-K50-3k-BODYTWIST-substeps10-h0p001"),
]


def simulate(plant: pl.Plant, flights: list[dict], samples: int, gains: dict) -> dict[str, np.ndarray]:
    """Batched over flights.  Returns states (B, samples, 18), rotor speeds (B, samples-1, 4) averaged over each
    data interval, desired wrench, saturation fraction and the first diverged sample (or -1)."""
    step, count = pl.make_step(plant, TICK)
    refs = [rf.at_controller_rate(rf.reference_signals(f), TICKS, samples) for f in flights]
    reference = {k: jnp.asarray(np.stack([r[k] for r in refs], axis=2)) for k in refs[0]}   # (S-1, T, B, ...)
    initial = np.concatenate([np.stack([f["states"][0] for f in flights]), np.zeros((len(flights), 4))], axis=1)
    batch = len(flights)

    def tick_body(carry, ref_tick):
        state, integral_p, integral_r, alive = carry
        omega2, wrench, integral_p, integral_r, saturated = ctl.mellinger(
            state, integral_p, integral_r, ref_tick, gains, TICK)
        controlled = state.at[:, 18:22].set(ctl.encode(omega2, plant.input_mode))
        advanced = step(controlled)
        bad = (~jnp.all(jnp.isfinite(advanced[:, :18]), axis=1)
               | (jnp.linalg.norm(advanced[:, :3] - ref_tick["position"], axis=1) > ESCAPE_METRES))
        alive_next = alive & ~bad
        state = jnp.where(alive_next[:, None], advanced, controlled)
        return (state, integral_p, integral_r, alive_next), (jnp.sqrt(omega2), wrench, saturated)

    def sample_body(carry, ref_sample):
        carry, (speeds, wrench, saturated) = jax.lax.scan(tick_body, carry, ref_sample)
        return carry, (carry[0][:, :18], speeds.mean(axis=0), wrench.mean(axis=0),
                       saturated.mean(axis=0), carry[3])

    @jax.jit
    def run(initial_state):
        carry = (initial_state, jnp.zeros((batch, 3)), jnp.zeros((batch, 3)), jnp.ones(batch, dtype=bool))
        _, outputs = jax.lax.scan(sample_body, carry, reference)
        return outputs

    start = time.perf_counter()
    states, speeds, wrench, saturated, alive = jax.block_until_ready(run(jnp.asarray(initial)))
    states = np.concatenate([initial[None, :, :18], np.asarray(states)], axis=0).transpose(1, 0, 2)
    alive = np.asarray(alive).T                                        # (B, S-1)
    first_dead = np.where(alive.all(axis=1), -1, np.argmin(alive, axis=1) + 1)
    return {"states": states, "rotor_speeds": np.asarray(speeds).transpose(1, 0, 2),
            "wrench": np.asarray(wrench).transpose(1, 0, 2), "saturated": np.asarray(saturated).T,
            "first_dead": first_dead, "substeps_per_tick": count, "seconds": time.perf_counter() - start}


def geodesic(r_true: np.ndarray, r_pred: np.ndarray) -> np.ndarray:
    cos = (np.trace(np.einsum("...ji,...jk->...ik", r_true, r_pred), axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cos, -1.0, 1.0))


def metrics(sim: dict, flights: list[dict], samples: int) -> dict:
    """Per window [0, W]: RMSE over flights, computed only on the samples each flight was still alive."""
    truth = np.stack([f["states"][:samples] for f in flights])
    recorded = np.stack([f["rotor_speeds"][1:samples] for f in flights])
    position_error = np.linalg.norm(sim["states"][:, :, :3] - truth[:, :, :3], axis=-1)
    attitude_error = np.degrees(geodesic(truth[:, :, 3:12].reshape(*truth.shape[:2], 3, 3),
                                         sim["states"][:, :, 3:12].reshape(*truth.shape[:2], 3, 3)))
    speed_error = np.linalg.norm(sim["rotor_speeds"] - recorded, axis=-1) / 2.0     # RMS over the 4 rotors
    alive = np.ones(position_error.shape, dtype=bool)
    for b, dead in enumerate(sim["first_dead"]):
        if dead >= 0:
            alive[b, dead:] = False
    out = {}
    for window in WINDOWS + (samples * rf.SAMPLE_STEP,):
        n = min(samples, int(round(window / rf.SAMPLE_STEP)) + 1)
        mask = alive[:, 1:n]
        survived = bool(mask.all())
        pick = lambda values: values[:, 1:n][mask] if mask.any() else np.array([np.nan])   # noqa: E731
        out[f"0-{window:g}s"] = {
            "all_flights_survived": survived,
            "position_rmse_m": float(np.sqrt(np.mean(pick(position_error) ** 2))),
            "position_final_error_m": [float(position_error[b, n - 1]) if alive[b, n - 1] else None
                                       for b in range(len(flights))],
            "attitude_rmse_deg": float(np.sqrt(np.mean(pick(attitude_error) ** 2))),
            "rotor_speed_rmse_rads": float(np.sqrt(np.mean(speed_error[:, :n - 1][mask] ** 2)))
            if mask.any() else float("nan"),
            "saturated_fraction": float(sim["saturated"][:, :n - 1][mask].mean()) if mask.any() else float("nan"),
        }
    out["survival_seconds"] = [float((d if d >= 0 else samples - 1) * rf.SAMPLE_STEP) for d in sim["first_dead"]]
    out["errors"] = {"position": position_error, "attitude": attitude_error, "speed": speed_error, "alive": alive}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", help="LABEL=DIR; default: the six IDSIA models")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "tmp/idsia_raw/data/test")
    parser.add_argument("--pattern", default="melon*.csv")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 = the whole recorded flight")
    parser.add_argument("--analytic-only", action="store_true", help="fly only the analytic plant (gain check)")
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    flights = rf.load_flights(arguments.source, arguments.pattern)
    length = min(f["states"].shape[0] for f in flights)
    samples = length if arguments.seconds <= 0 else min(length, int(round(arguments.seconds / rf.SAMPLE_STEP)) + 1)
    gains_config = ctl.Gains()
    gains = gains_config.arrays()

    plants = [pl.analytic_plant()]
    if not arguments.analytic_only:
        specs = ([tuple(s.rpartition("=")[::2]) for s in arguments.run] if arguments.run else DEFAULT_RUNS)
        plants += [pl.learned_plant(label, Path(d)) for label, d in specs]

    folder = EVAL_ROOT / arguments.output_name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{len(flights)} flights, {samples} samples ({(samples - 1) * rf.SAMPLE_STEP:.2f} s), "
          f"controller {1 / TICK:.0f} Hz\n")
    header = (f"{'plant':<44}{'enc':>7}{'survive (s)':>20}" + "".join(
        f"{'pos 0-' + format(w, 'g') + 's':>12}" for w in WINDOWS) + f"{'pos full':>11}"
        + f"{'att 0-10s':>11}{'Omega 0-10s':>13}{'sat 0-10s':>11}")
    print(header)
    results, arrays = {}, {}
    for plant in plants:
        sim = simulate(plant, flights, samples, gains)
        m = metrics(sim, flights, samples)
        errors = m.pop("errors")
        full = f"0-{samples * rf.SAMPLE_STEP:g}s"
        cells = "".join(f"{m[f'0-{w:g}s']['position_rmse_m']:>11.3f}{'' if m[f'0-{w:g}s']['all_flights_survived'] else '*'}"
                        .rjust(12) for w in WINDOWS)
        print(f"{plant.label:<44}{plant.input_mode:>7}{'/'.join(f'{s:.1f}' for s in m['survival_seconds']):>20}"
              + cells + f"{m[full]['position_rmse_m']:>10.3f}{'' if m[full]['all_flights_survived'] else '*'}"
              + f"{m['0-10s']['attitude_rmse_deg']:>11.2f}{m['0-10s']['rotor_speed_rmse_rads']:>13.1f}"
              + f"{m['0-10s']['saturated_fraction']:>11.3f}", flush=True)
        results[plant.label] = {"run": plant.run_dir, "input_mode": plant.input_mode,
                                "training_step": plant.training_step, "rk4": plant.rk4,
                                "integrator_steps_per_tick": sim["substeps_per_tick"],
                                "wall_seconds": sim["seconds"], **m}
        arrays[plant.label] = {**sim, **errors}
    print("\n* = at least one flight diverged inside the window; RMSE then covers only the samples before it")

    (folder / "closed_loop.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(),
        "flights": [f["name"] for f in flights], "samples": samples, "controller_hz": 1 / TICK,
        "gains": {k: (np.asarray(v).tolist() if hasattr(v, "shape") else v) for k, v in gains.items()},
        "gain_design": gains_config.__dict__,
        "reference": "recorded path (position, world velocity), smoothed finite-difference acceleration and yaw rate",
        "escape_metres": ESCAPE_METRES, "results": results,
    }, indent=2, default=float) + "\n")
    np.savez_compressed(folder / "closed_loop_arrays.npz", **{
        f"{label}|{key}": value for label, entry in arrays.items() for key, value in entry.items()
        if isinstance(value, np.ndarray)})
    print(f"written {folder}")

    from .report import write_pdf
    write_pdf(folder / "closed_loop_melon.pdf", flights, samples, arrays, results)
    print(f"written {folder / 'closed_loop_melon.pdf'}")


if __name__ == "__main__":
    main()

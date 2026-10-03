"""Gauge-invariant comparison of trained rotor2 models: the operator PRODUCTS the data identifies.

Individual factors (M1^-1, M2^-1, g, V, D) are not identifiable -- the same dynamics come from many factorisations --
so only these products are compared, evaluated on the held-out melon states (u in rotor2 hover units, u_i = 1 at hover):

    gravity   a_g  = M1^-1 grad_x V                  (m/s^2)          published: [0, 0, 9.81]
    thrust    B_f  = M1^-1 g[0:3, :]   (3x4)         per unit u_i      published: z-row K_t/(s m) = 2.4525 each, sum 9.81
    torque    B_t  = M2^-1 g[3:6, :]   (3x4)         per unit u_i      published: J^-1 * mixer / s  (162.7 roll/pitch, 7.10 yaw)
    damping   K_v  = M1^-1 D_v, K_w = M2^-1 D_w      (1/s)             published: unknown (real vehicle)

Torque gains are reported as the projection of each row of B_t onto the published mixer sign pattern, divided by
the published magnitude (ratio 1 = matches the published mixer, negative = opposite sign), plus the fraction of the
row that lies OFF that pattern (cross-coupling the published mixer does not have).

Usage:
    python gauge_products_compare.py --run LABEL=DIR[@STEP] [...] --output-name NAME
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import benchmark_protocol as bp  # noqa: E402

PROJECT_ROOT = bp.PROJECT_ROOT
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
MASS, GRAVITY = 0.045, 9.81
INERTIA = np.array([2.3951e-05, 2.3951e-05, 3.2347e-06])
KT, KC, ARM = 3.72e-08, 7.74e-12, 0.0353
SCALE = 4.0 * KT / (MASS * GRAVITY)
SIGNS = np.array([[-1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0]])
TORQUE_TRUE = np.array([KT * ARM / SCALE / INERTIA[0], KT * ARM / SCALE / INERTIA[1], KC / SCALE / INERTIA[2]])


def products(model, poses: np.ndarray, twists: np.ndarray) -> dict[str, np.ndarray]:
    def one(pose, twist):
        m1 = model.inverse_mass_1(pose[:3])
        m2 = model.inverse_mass_2(pose[3:12])
        g = model.control_matrix(pose)
        grad_v = jax.grad(model.potential)(pose)[:3]
        return (m1 @ grad_v, m1 @ g[:3], m2 @ g[3:6],
                m1 @ model.dissipation_v(twist[:3], pose[:3]), m2 @ model.dissipation_w(twist[3:], pose[3:12]))
    out = jax.vmap(one)(jnp.asarray(poses), jnp.asarray(twists))
    return dict(zip(("gravity", "thrust", "torque", "damping_v", "damping_w"), map(np.asarray, out)))


def summarise(p: dict[str, np.ndarray]) -> dict[str, float]:
    s = {}
    s["gravity_z"] = float(p["gravity"][:, 2].mean())
    s["gravity_lateral"] = float(np.linalg.norm(p["gravity"][:, :2], axis=1).mean())
    s["thrust_z_sum"] = float(p["thrust"][:, 2, :].sum(axis=1).mean())
    s["thrust_z_spread"] = float((p["thrust"][:, 2, :].std(axis=1) / np.abs(p["thrust"][:, 2, :].mean(axis=1))).mean())
    s["thrust_lateral_frac"] = float((np.linalg.norm(p["thrust"][:, :2, :], axis=(1, 2))
                                      / np.linalg.norm(p["thrust"], axis=(1, 2))).mean())
    s["hover_balance"] = s["thrust_z_sum"] / s["gravity_z"]
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        row = p["torque"][:, axis, :]                                    # (N, 4)
        sign = SIGNS[axis]
        on = row @ sign / 4.0                                            # mean gain along the published pattern
        off = row - on[:, None] * sign[None, :]
        s[f"torque_{name}_ratio"] = float(on.mean() / TORQUE_TRUE[axis])
        s[f"torque_{name}_offpattern"] = float((np.linalg.norm(off, axis=1) / np.linalg.norm(row, axis=1)).mean())
    for key in ("damping_v", "damping_w"):
        diag = np.diagonal(p[key], axis1=1, axis2=2)
        for i, a in enumerate("xyz"):
            s[f"{key}_{a}"] = float(diag[:, i].mean())
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="LABEL=DIR or LABEL=DIR@STEP")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation
    flights = bp.load_melon(PROJECT_ROOT / "envs/quadrotor_se3_idsia/idsia_raw/data/test", "rotor2")
    states = np.concatenate(flights)
    pick = np.linspace(0, len(states) - 1, arguments.samples).astype(int)
    poses, twists = states[pick, :12], states[pick, 12:18]

    rows, labels = {}, []
    for spec in arguments.run:
        label, _, target = spec.rpartition("=")
        run_dir, _, step = target.partition("@")
        run = evaluation.load_run(Path(run_dir), int(step) if step else None)
        model, *_ = evaluation.build_model(run)
        rows[label] = {"run": str(Path(run_dir).resolve()), "step": run["selected_step"],
                       "test_loss": run["selected_test"]["total"], **summarise(products(model, poses, twists))}
        labels.append(label)

    published = {"gravity_z": GRAVITY, "gravity_lateral": 0.0, "thrust_z_sum": GRAVITY, "thrust_z_spread": 0.0,
                 "thrust_lateral_frac": 0.0, "hover_balance": 1.0,
                 **{f"torque_{n}_ratio": 1.0 for n in ("roll", "pitch", "yaw")},
                 **{f"torque_{n}_offpattern": 0.0 for n in ("roll", "pitch", "yaw")}}
    keys = [k for k in rows[labels[0]] if k not in ("run", "step")]
    width = max(len(l) for l in labels) + 2
    print(f"\n{'quantity':<26}" + "".join(f"{l:>{width}}" for l in labels) + f"{'published':>12}")
    for k in keys:
        pub = published.get(k)
        print(f"{k:<26}" + "".join(f"{rows[l][k]:>{width}.4g}" for l in labels)
              + (f"{pub:>12.4g}" if pub is not None else f"{'unknown':>12}"))
    folder = EVAL_ROOT / arguments.output_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gauge_products.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(), "samples": int(len(pick)),
        "published": published, "torque_true_per_unit_u": TORQUE_TRUE.tolist(), "models": rows}, indent=2) + "\n")
    print(f"\nwritten {folder / 'gauge_products.json'}")


if __name__ == "__main__":
    main()

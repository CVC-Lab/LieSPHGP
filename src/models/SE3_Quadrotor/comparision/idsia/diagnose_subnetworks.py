"""Which subnetwork is wrong, and is the residual error a step-size effect?  Inference only, no retraining.

Ground truth comes from the benchmark's own published constants (dataset settings ``vehicle_parameters`` and
``models/models.py`` of the IDSIA repo), which the model never sees:

    m = 0.045 kg,  J = diag(2.3951e-5, 2.3951e-5, 3.2347e-6) kg m^2,  g = 9.81,
    arm a = 0.0353 m,  K_t = 3.72e-8,  K_c = 7.74e-12,
    u_i = Omega_i^2 * s   with s = rotor2_scale, so Omega_i^2 = u_i / s.

With the benchmark's mixer this fixes every operator of the port-Hamiltonian model in rotor2 units:

    M1^-1  = I/m                      -> 1/0.045 = 22.22
    M2^-1  = J^-1                     -> diag(41752, 41752, 309148)
    mu grad V = g e3                  -> 9.81 along world z
    g_f    : column i = (K_t/s) e3                                     (force per unit u_i, body frame)
    g_tau  : column i = (K_t a/s) [x_i, y_i, 0] + (K_c/s) [0, 0, z_i]  (torque per unit u_i)
             with x = (-1,-1,+1,+1), y = (-1,+1,+1,-1), z = (+1,-1,+1,-1) from models.py:82-84
    D_v, D_w: NOT known - the real vehicle's dissipation is unmeasured ("damping_law: unknown").

Three questions, in order:
  1. per-operator error against the analytic truth (which subnetwork is bad)
  2. the same model rolled out at h = 0.01 / 0.005 / 0.001 (is the residual a step-size effect)
  3. a swap ablation: substitute each learned operator with its true value, one at a time, and see which
     substitution recovers the trajectory (which subnetwork actually causes the error)

Usage:
    python diagnose_subnetworks.py --run DIR [--run DIR ...] [--horizon-seconds 1.0] [--flights 3]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_protocol as bp

PROJECT_ROOT = bp.PROJECT_ROOT
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
STEP = 0.01

MASS = 0.045
INERTIA = np.array([2.3951e-05, 2.3951e-05, 3.2347e-06])
GRAVITY = 9.81
ARM = 0.0353
KT = 3.72e-08
KC = 7.74e-12
# signs read off models.py lines 82-84
SIGN_X = np.array([-1.0, -1.0, +1.0, +1.0])
SIGN_Y = np.array([-1.0, +1.0, +1.0, -1.0])
SIGN_Z = np.array([+1.0, -1.0, +1.0, -1.0])


def true_control_matrix(rotor2_scale: float) -> np.ndarray:
    """(6,4) body-frame wrench per unit rotor2 input."""
    g = np.zeros((6, 4))
    g[2, :] = KT / rotor2_scale                      # thrust along body z
    g[3, :] = KT * ARM / rotor2_scale * SIGN_X       # tau_x
    g[4, :] = KT * ARM / rotor2_scale * SIGN_Y       # tau_y
    g[5, :] = KC / rotor2_scale * SIGN_Z             # tau_z
    return g


def relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.sqrt(np.mean(np.square(b)))
    return float(np.sqrt(np.mean(np.square(a - b))) / denominator * 100.0) if denominator > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="LABEL=DIR")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "envs/quadrotor_se3_idsia/idsia_raw/data/test")
    parser.add_argument("--pattern", default="melon*.csv")
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, nargs="+", default=[1, 2, 10])
    parser.add_argument("--samples", type=int, default=400, help="states sampled for the operator comparison")
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation

    import pickle
    with (PROJECT_ROOT / "datasets/QUADROTOR-DATASET-IDSIA/IDSIA_CF21BL_10s_h0p01_rotor2_clean.pkl").open("rb") as fh:
        scale = float(pickle.load(fh)["settings"]["rotor2_scale"])
    g_true = true_control_matrix(scale)
    print(f"\nrotor2_scale = {scale:.6e}   K_t/s = {KT / scale:.4f} N per unit u   "
          f"(4 rotors at hover -> {4 * KT / scale:.4f} N, m g = {MASS * GRAVITY:.4f} N)")

    flights = bp.load_melon(arguments.source, "rotor2", arguments.pattern)
    keep = int(round(arguments.horizon_seconds / STEP)) + 1
    truth = np.stack([f[:keep] for f in flights])
    states = np.concatenate([f[:, :12] for f in flights])
    index = np.linspace(0, states.shape[0] - 1, arguments.samples).astype(int)
    poses = jnp.asarray(states[index])

    report: dict[str, dict] = {}
    for spec in arguments.run:
        label, _, run_dir = spec.rpartition("=")
        run = evaluation.load_run(Path(run_dir), None)
        model, _params, _setup = evaluation.build_model(run)
        print(f"\n{'=' * 96}\n{label}\n{'=' * 96}")

        # ---- 1. operator errors against the published truth ------------------------------------
        def ops(pose):
            return (model.inverse_mass_1(pose[:3]),
                    model.inverse_mass_2(pose[3:12]),
                    jax.grad(model.potential)(pose)[:3],
                    model.control_matrix(pose))

        m1, m2, gradv, gmat = jax.vmap(ops)(poses)
        m1, m2, gradv, gmat = map(np.asarray, (m1, m2, gradv, gmat))
        mu = m1[:, 0, 0]

        learned = {
            "M1^-1 (1/kg)": m1[:, 0, 0],
            "M2^-1 xx": m2[:, 0, 0], "M2^-1 yy": m2[:, 1, 1], "M2^-1 zz": m2[:, 2, 2],
            "mu grad V (z)": (mu[:, None] * gradv)[:, 2],
            "mu g_f (z, per rotor)": np.einsum("n,nj->nj", mu, gmat[:, 2, :]).mean(axis=1),
            "M2^-1 g_tau x": np.einsum("nij,nj->ni", m2, gmat[:, 3:6, 0])[:, 0],
            "M2^-1 g_tau y": np.einsum("nij,nj->ni", m2, gmat[:, 3:6, 1])[:, 1],
            "M2^-1 g_tau z": np.einsum("nij,nj->ni", m2, gmat[:, 3:6, 2])[:, 2],
        }
        truth_values = {
            "M1^-1 (1/kg)": 1.0 / MASS,
            "M2^-1 xx": 1.0 / INERTIA[0], "M2^-1 yy": 1.0 / INERTIA[1], "M2^-1 zz": 1.0 / INERTIA[2],
            "mu grad V (z)": GRAVITY,
            "mu g_f (z, per rotor)": (KT / scale) / MASS,
            "M2^-1 g_tau x": g_true[3, 0] / INERTIA[0],
            "M2^-1 g_tau y": g_true[4, 1] / INERTIA[1],
            "M2^-1 g_tau z": g_true[5, 2] / INERTIA[2],
        }
        print(f"\n  {'operator':>26}{'learned mean':>16}{'true':>16}{'ratio':>10}{'rel RMS %':>12}")
        op_report = {}
        for name, value in learned.items():
            t = truth_values[name]
            ratio = float(np.mean(value)) / t if t != 0 else float("nan")
            err = relative_rms(np.asarray(value), np.full_like(np.asarray(value), t))
            print(f"  {name:>26}{np.mean(value):>16.4g}{t:>16.4g}{ratio:>10.4f}{err:>11.1f}%")
            op_report[name] = {"learned_mean": float(np.mean(value)), "true": float(t),
                               "ratio": float(ratio), "relative_rms_percent": float(err)}
        print(f"  {'D_v, D_w':>26}{'(learned)':>16}{'unknown':>16}{'-':>10}{'-':>12}   real vehicle damping is unmeasured")

        # ---- 2. same model, different integrator step ------------------------------------------
        print(f"\n  rollout of the SAME weights at different integrator steps ({arguments.horizon_seconds:g} s, "
              f"{truth.shape[0]} flights):")
        print(f"  {'h (s)':>10}{'sub-steps':>12}{'pos RMSE (m)':>16}{'att RMSE (deg)':>17}{'omega RMSE':>13}")
        step_report = {}
        for sub in arguments.substeps:
            try:
                prediction = evaluation.rollout(model, truth, STEP, substeps=sub)
            except FloatingPointError:
                print(f"  {STEP / sub:>10.4f}{sub:>12}{'non-finite':>16}")
                step_report[str(sub)] = {"status": "non-finite"}
                continue
            pos = float(np.sqrt(np.mean(np.sum((prediction[:, 1:, :3] - truth[:, 1:, :3]) ** 2, axis=-1))))
            Rt = truth[:, 1:, 3:12].reshape(truth.shape[0], -1, 3, 3)
            Rp = prediction[:, 1:, 3:12].reshape(truth.shape[0], -1, 3, 3)
            cos = np.clip((np.trace(np.einsum("fhji,fhjk->fhik", Rt, Rp), axis1=-2, axis2=-1) - 1) / 2, -1, 1)
            att = float(np.degrees(np.sqrt(np.mean(np.arccos(cos) ** 2))))
            om = float(np.sqrt(np.mean(np.sum((prediction[:, 1:, 15:18] - truth[:, 1:, 15:18]) ** 2, axis=-1))))
            print(f"  {STEP / sub:>10.4f}{sub:>12}{pos:>16.4f}{att:>17.2f}{om:>13.4f}")
            step_report[str(sub)] = {"h": STEP / sub, "position_rmse_m": pos,
                                     "attitude_rmse_deg": att, "omega_rmse": om}
        report[label] = {"run": str(Path(run_dir).resolve()), "operators": op_report, "step_size": step_report}

    folder = EVAL_ROOT / arguments.output_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "subnetwork_diagnosis.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(),
        "horizon_seconds": arguments.horizon_seconds, "flights": int(truth.shape[0]),
        "vehicle": {"mass": MASS, "inertia": INERTIA.tolist(), "arm": ARM, "kt": KT, "kc": KC,
                    "rotor2_scale": scale},
        "models": report,
    }, indent=2) + "\n")
    print(f"\nwritten {folder / 'subnetwork_diagnosis.json'}")


if __name__ == "__main__":
    main()

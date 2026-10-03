"""The complete oracle: analytic rigid body driven by the FULL measured wrench, force and torque.

``oracle_wrench_rollout.py`` gave the rigid body the reconstructed torque but kept the MODELLED thrust
T = k_F sum(Omega^2), which by construction points along body z.  The benchmark's paper reports that this is
exactly what the rotor model misses:

    "the reconstructed body frame forces exhibit additional components in the x and y directions
     that the model does not explain"

No rotor parameterisation can produce those - rotor2 and Omega^2 are the same quantity up to a scale, so
switching input encoding changes nothing.  They are aerodynamic (drag, rotor wake).  But the recordings carry
``ax_body, ay_body, az_body``, the measured specific force, which is what the paper itself uses (f = m a_IMU).

This script therefore drives the rigid body with BOTH reconstructed channels:

    f_body = m * a_IMU                              (all three components, lateral included)
    tau    = J omega_dot + omega x (J omega)        (omega measured, omega_dot by finite difference)

and integrates with a Lie-IMEX scheme of its own (the shared one hardcodes a 4-wide control).  Three variants
isolate what each channel is worth:

    thrust-only  : force = k_F sum(Omega^2) along body z   (what a rotor model can express)
    force-only   : measured f_body, modelled torque         (lateral aerodynamics, nominal rotation)
    full         : measured f_body and reconstructed tau    (the ceiling given perfect instantaneous knowledge)

Usage:
    python oracle_full_wrench.py --horizon-seconds 1 3 5 10 20 30 --output-name NAME
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import benchmark_protocol as bp

PROJECT_ROOT = bp.PROJECT_ROOT
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
STEP = 0.01
MASS, GRAVITY = 0.045, 9.81
INERTIA = np.array([2.3951e-05, 2.3951e-05, 3.2347e-06])
KF = 3.72e-08
KT_ARM, KC = KF * 0.0353, 7.74e-12
SIGN_X = np.array([-1.0, -1.0, 1.0, 1.0])
SIGN_Y = np.array([-1.0, 1.0, 1.0, -1.0])
SIGN_Z = np.array([1.0, -1.0, 1.0, -1.0])


def hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def exp_so3(w):
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3) + hat(w)
    axis = w / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def load(source: Path, pattern: str):
    """Per flight: states (x, R, v_b, omega) plus every candidate force and torque channel."""
    out = []
    for path in sorted(source.glob(pattern)):
        f = pd.read_csv(path)
        R = Rotation.from_quat(f[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
        omega = f[["wx", "wy", "wz"]].to_numpy(dtype=np.float64)
        speeds2 = f[[f"m{i}_rads" for i in (1, 2, 3, 4)]].to_numpy(dtype=np.float64) ** 2
        v_body = np.einsum("nji,nj->ni", R, f[["vx", "vy", "vz"]].to_numpy(dtype=np.float64))
        out.append({
            "x": f[["x", "y", "z"]].to_numpy(dtype=np.float64), "R": R, "v": v_body, "w": omega,
            "f_measured": MASS * f[["ax_body", "ay_body", "az_body"]].to_numpy(dtype=np.float64),
            "f_modelled": np.stack([np.zeros(len(f)), np.zeros(len(f)), KF * speeds2.sum(axis=1)], axis=1),
            "tau_measured": np.gradient(omega, STEP, axis=0) * INERTIA + np.cross(omega, omega * INERTIA),
            "tau_modelled": np.stack([KT_ARM * (speeds2 * SIGN_X).sum(axis=1),
                                      KT_ARM * (speeds2 * SIGN_Y).sum(axis=1),
                                      KC * (speeds2 * SIGN_Z).sum(axis=1)], axis=1),
        })
    return out


def rollout(x0, R0, v0, w0, force, torque):
    """Lie-IMEX (here explicit: the oracle has no damping) on the body-frame rigid body."""
    steps = force.shape[0]
    X = np.empty((steps + 1, 3)); Rs = np.empty((steps + 1, 3, 3))
    V = np.empty((steps + 1, 3)); W = np.empty((steps + 1, 3))
    X[0], Rs[0], V[0], W[0] = x0, R0, v0, w0
    for k in range(steps):
        x, R, v, w = X[k], Rs[k], V[k], W[k]
        # body-frame Newton-Euler; gravity enters rotated into the body
        dv = force[k] / MASS - R.T @ np.array([0.0, 0.0, GRAVITY]) - np.cross(w, v)
        dw = (torque[k] - np.cross(w, INERTIA * w)) / INERTIA
        v_half = v + 0.5 * STEP * dv
        w_half = w + 0.5 * STEP * dw
        X[k + 1] = x + STEP * (R @ v_half)
        Rs[k + 1] = R @ exp_so3(STEP * w_half)
        V[k + 1] = v + STEP * dv
        W[k + 1] = w + STEP * dw
        if not np.isfinite(X[k + 1]).all():
            return None
    return X, Rs, V, W


def score(flights, horizon, stride_seconds, force_key, torque_key):
    keep = int(round(horizon / STEP)) + 1
    stride = max(1, int(round(stride_seconds / STEP)))
    pos_sq, att_sq, om_sq, n, diverged = 0.0, 0.0, 0.0, 0, 0
    for fl in flights:
        for s in range(0, fl["x"].shape[0] - keep, stride):
            sl = slice(s, s + keep)
            result = rollout(fl["x"][s], fl["R"][s], fl["v"][s], fl["w"][s],
                             fl[force_key][sl][:-1], fl[torque_key][sl][:-1])
            if result is None:
                diverged += 1
                continue
            X, Rs, _V, W = result
            pos_sq += np.sum((X[1:] - fl["x"][sl][1:]) ** 2)
            rel = np.einsum("nji,njk->nik", fl["R"][sl][1:], Rs[1:])
            cos = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
            att_sq += np.sum(np.arccos(cos) ** 2)
            om_sq += np.sum((W[1:] - fl["w"][sl][1:]) ** 2)
            n += keep - 1
    if n == 0:
        return None
    return (float(np.sqrt(pos_sq / n)), float(np.degrees(np.sqrt(att_sq / n))),
            float(np.sqrt(om_sq / n)), diverged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "envs/quadrotor_se3_idsia/idsia_raw/data/test")
    parser.add_argument("--pattern", default="melon*.csv")
    parser.add_argument("--horizon-seconds", type=float, nargs="+", default=[1, 3, 5, 10, 20, 30])
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    flights = load(arguments.source, arguments.pattern)
    variants = [("modelled f, modelled tau", "f_modelled", "tau_modelled"),
                ("modelled f, MEASURED tau", "f_modelled", "tau_measured"),
                ("MEASURED f, modelled tau", "f_measured", "tau_modelled"),
                ("MEASURED f, MEASURED tau", "f_measured", "tau_measured")]
    print(f"\n{len(flights)} melon flights, analytic rigid body, no damping, h = {STEP}")
    print(f"\n{'horizon':>9}" + "".join(f"{n[:26]:>28}" for n, _f, _t in variants))
    results = {}
    for horizon in arguments.horizon_seconds:
        row, entry = f"{horizon:>8.0f}s", {}
        for name, fk, tk in variants:
            got = score(flights, horizon, arguments.stride_seconds, fk, tk)
            if got is None:
                row += f"{'-':>28}"; continue
            pos, att, om, div = got
            row += f"{pos:>21.3f}{('  (' + str(div) + 'x)') if div else '':>7}"
            entry[name] = {"position_rmse_m": pos, "attitude_rmse_deg": att,
                           "omega_rmse": om, "diverged_segments": div}
        print(row)
        results[f"{horizon:g}s"] = entry

    print(f"\nattitude RMSE (deg)\n{'horizon':>9}" + "".join(f"{n[:26]:>28}" for n, _f, _t in variants))
    for horizon in arguments.horizon_seconds:
        row = f"{horizon:>8.0f}s"
        for name, _fk, _tk in variants:
            e = results[f"{horizon:g}s"].get(name)
            row += f"{e['attitude_rmse_deg']:>28.2f}" if e else f"{'-':>28}"
        print(row)

    folder = EVAL_ROOT / arguments.output_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "oracle_full_wrench.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(),
        "vehicle": {"mass": MASS, "inertia": INERTIA.tolist(), "kf": KF},
        "note": "measured f = m a_IMU (body axes, lateral included); measured tau = J wdot + w x (J w)",
        "results": results,
    }, indent=2) + "\n")
    print(f"\nwritten {folder / 'oracle_full_wrench.json'}")


if __name__ == "__main__":
    main()

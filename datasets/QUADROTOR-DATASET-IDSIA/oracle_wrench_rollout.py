"""Can the dataset be simulated at all, given perfect knowledge?  Separates two different 'ground truths'.

Two quite different questions get conflated by "assume we know the ground truth":

  (A) rigid-body truth   - m, J, g, and the RIGHT-HAND-SIDE torque tau(t) itself
  (B) actuator truth     - the map Omega^2 -> tau, i.e. the quadratic rotor model with k_F, k_M

The benchmark's own paper shows (B) fails: the reconstructed roll/pitch/yaw torques "depart significantly"
from the quadratic prediction, with "a lower-frequency part that the model cannot account for", which is why
their physics baseline sets omega_dot = 0.

But (A) is available from the measurements, exactly as the paper derives it:

    tau_tilde = J omega_dot + omega x (J omega),     omega measured, omega_dot by finite difference
    T_tilde   = k_F sum(Omega_i^2)                   (the translational fit is accurate to ~0.5%)

This script rolls the analytic rigid body forward on that reconstructed wrench, through the SAME Lie-IMEX
integrator used everywhere else, and compares to the recorded trajectory.  The result bounds what any model
driven by rotor speeds could achieve: if the ORACLE reproduces the flight, the rigid body is right and only
the actuator map is missing; if it does not, the state itself is not explained by rigid-body dynamics.

Usage:
    python oracle_wrench_rollout.py --horizon-seconds 1 3 5 --output-name NAME
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
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


class RigidBody(eqx.Module):
    """The analytic port-Hamiltonian rigid body: no learning anywhere, control is the wrench [T, tau]."""

    damping_v: jax.Array
    damping_w: jax.Array

    def inverse_mass_1(self, position):
        return jnp.eye(3) / MASS

    def inverse_mass_2(self, rotation_flat):
        return jnp.diag(jnp.asarray(1.0 / INERTIA))

    def potential(self, pose):
        return MASS * GRAVITY * pose[2]

    measured_force: bool = eqx.field(static=True, default=False)

    def control_matrix(self, pose):
        if self.measured_force:
            return jnp.eye(6)                         # u = [f_body(3), tau(3)] applied directly
        g = jnp.zeros((6, 4))
        g = g.at[2, 0].set(1.0)                       # u0 = total thrust along body z
        g = g.at[3, 1].set(1.0).at[4, 2].set(1.0).at[5, 3].set(1.0)   # u1..u3 = body torques
        return g

    def dissipation_v(self, velocity):
        return self.damping_v

    def dissipation_w(self, omega):
        return self.damping_w

    def hamiltonian(self, pose_momenta):
        position, rotation_flat = pose_momenta[:3], pose_momenta[3:12]
        pv, pw = pose_momenta[12:15], pose_momenta[15:18]
        return (0.5 * pv @ self.inverse_mass_1(position) @ pv
                + 0.5 * pw @ self.inverse_mass_2(rotation_flat) @ pw
                + self.potential(pose_momenta[:12]))

    def vector_field(self, state):
        pose, position, rotation_flat = state[:12], state[:3], state[3:12]
        velocity, omega, control = state[12:15], state[15:18], state[18:22]
        mass_inv_v = self.inverse_mass_1(position)
        mass_inv_w = self.inverse_mass_2(rotation_flat)
        momentum_v = jnp.linalg.solve(mass_inv_v, velocity)
        momentum_w = jnp.linalg.solve(mass_inv_w, omega)
        pose_momenta = jnp.concatenate([pose, momentum_v, momentum_w])
        gradient = jax.grad(self.hamiltonian)(pose_momenta)
        d_h_dx, d_h_dr = gradient[:3], gradient[3:12]
        d_h_dpv, d_h_dpw = gradient[12:15], gradient[15:18]
        rotation = rotation_flat.reshape(3, 3)
        dx = rotation @ d_h_dpv
        dr = jnp.cross(rotation, d_h_dpw[None, :], axis=-1).reshape(9)
        force_torque = self.control_matrix(pose) @ control
        dpv = (jnp.cross(momentum_v, d_h_dpw) - rotation.T @ d_h_dx
               - self.dissipation_v(velocity) @ d_h_dpv + force_torque[:3])
        dpw = (jnp.cross(momentum_w, d_h_dpw) + jnp.cross(momentum_v, d_h_dpv)
               + jnp.cross(rotation, d_h_dr.reshape(3, 3), axis=-1).sum(axis=0)
               - self.dissipation_w(omega) @ d_h_dpw + force_torque[3:6])
        dv = mass_inv_v @ dpv
        dw = mass_inv_w @ dpw
        return jnp.concatenate([dx, dr, dv, dw, jnp.zeros(4, dtype=state.dtype)])

    def effective_damping(self, state):
        return (self.inverse_mass_1(state[:3]) @ self.damping_v,
                self.inverse_mass_2(state[3:12]) @ self.damping_w)


def load_with_reconstructed_wrench(source: Path, pattern: str, use_measured_force: bool = False):
    """Flights whose control columns hold [T_tilde, tau_tilde] reconstructed from the measurements."""
    out = []
    for path in sorted(source.glob(pattern)):
        frame = pd.read_csv(path)
        rotation = Rotation.from_quat(frame[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
        omega = frame[["wx", "wy", "wz"]].to_numpy(dtype=np.float64)
        speeds = frame[[f"m{i}_rads" for i in (1, 2, 3, 4)]].to_numpy(dtype=np.float64)
        omega_dot = np.gradient(omega, STEP, axis=0)                       # as the paper does
        tau = omega_dot * INERTIA + np.cross(omega, omega * INERTIA)       # J wdot + w x (J w)
        if use_measured_force:
            # the paper's own reconstruction: f = m a_IMU, the measured specific force in BODY axes.
            # This carries the lateral x/y components that no rotor model can produce.
            force = MASS * frame[["ax_body", "ay_body", "az_body"]].to_numpy(dtype=np.float64)
        else:
            thrust = KF * np.sum(speeds ** 2, axis=1)                      # the accurate translational fit
            force = np.zeros((len(frame), 3)); force[:, 2] = thrust
        velocity_world = frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
        out.append(np.concatenate([
            frame[["x", "y", "z"]].to_numpy(dtype=np.float64), rotation.reshape(-1, 9),
            np.einsum("nji,nj->ni", rotation, velocity_world), omega,
            force, tau], axis=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "tmp/idsia_raw/data/test")
    parser.add_argument("--pattern", default="melon*.csv")
    parser.add_argument("--horizon-seconds", type=float, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--damping", type=float, nargs=2, default=[0.0, 0.0],
                        help="isotropic D_v and D_w for the oracle (default: none)")
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation

    model = RigidBody(damping_v=jnp.eye(3) * arguments.damping[0],
                      damping_w=jnp.eye(3) * arguments.damping[1])
    flights = load_with_reconstructed_wrench(arguments.source, arguments.pattern)
    print(f"\n{len(flights)} flights, wrench reconstructed as tau = J wdot + w x (J w), T = k_F sum(Omega^2)")
    print(f"oracle damping: D_v = {arguments.damping[0]}, D_w = {arguments.damping[1]}")
    print(f"\n{'horizon':>9}{'segs':>7}{'pos RMSE (m)':>15}{'att RMSE (deg)':>17}{'omega RMSE':>13}{'VPT (s)':>10}")

    results = {}
    for horizon in arguments.horizon_seconds:
        keep = int(round(horizon / STEP)) + 1
        stride = max(1, int(round(arguments.stride_seconds / STEP)))
        pieces = [f[s:s + keep] for f in flights for s in range(0, f.shape[0] - keep, stride)]
        truth = np.stack(pieces)
        try:
            prediction = evaluation.rollout(model, truth, STEP)
        except FloatingPointError:
            print(f"{horizon:>8.0f}s{truth.shape[0]:>7}{'non-finite':>15}")
            results[f"{horizon:g}s"] = {"status": "non-finite"}
            continue
        pos = float(np.sqrt(np.mean(np.sum((prediction[:, 1:, :3] - truth[:, 1:, :3]) ** 2, axis=-1))))
        Rt = truth[:, 1:, 3:12].reshape(truth.shape[0], -1, 3, 3)
        Rp = prediction[:, 1:, 3:12].reshape(truth.shape[0], -1, 3, 3)
        cos = np.clip((np.trace(np.einsum("fhji,fhjk->fhik", Rt, Rp), axis1=-2, axis2=-1) - 1) / 2, -1, 1)
        att = float(np.degrees(np.sqrt(np.mean(np.arccos(cos) ** 2))))
        om = float(np.sqrt(np.mean(np.sum((prediction[:, 1:, 15:18] - truth[:, 1:, 15:18]) ** 2, axis=-1))))
        error = np.linalg.norm(prediction[:, 1:, :3] - truth[:, 1:, :3], axis=-1)
        extent = np.maximum(1e-6, np.linalg.norm(truth[:, :, :3] - truth[:, :1, :3], axis=-1).max(axis=1))
        within = error <= 0.158 * extent[:, None]
        vpt = float(np.median(np.where(within.all(axis=1), within.shape[1],
                                       np.argmin(within, axis=1)) * STEP))
        print(f"{horizon:>8.0f}s{truth.shape[0]:>7}{pos:>15.4f}{att:>17.2f}{om:>13.4f}{vpt:>10.2f}")
        results[f"{horizon:g}s"] = {"segments": int(truth.shape[0]), "position_rmse_m": pos,
                                    "attitude_rmse_deg": att, "omega_rmse": om, "vpt_median_s": vpt}

    folder = EVAL_ROOT / arguments.output_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "oracle_wrench.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(),
        "note": "analytic rigid body driven by the wrench reconstructed from the measurements",
        "vehicle": {"mass": MASS, "inertia": INERTIA.tolist(), "kf": KF},
        "damping": arguments.damping, "results": results,
    }, indent=2) + "\n")
    print(f"\nwritten {folder / 'oracle_wrench.json'}")


if __name__ == "__main__":
    main()

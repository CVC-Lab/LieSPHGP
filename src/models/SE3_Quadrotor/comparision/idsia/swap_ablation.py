"""Swap ablation on a trained PH-NN-LieIMEX model: replace operators with the benchmark's published truth.

Inference only - no retraining, no weight updates.  A wrapper exposes the same operator interface as the
trained model and forwards every call to it, except the ones selected for substitution, which return the
analytic value instead.  The Lie-IMEX rollout is then run unchanged, so any change in error is attributable
to the substituted operator alone.

Truth (dataset ``vehicle_parameters`` + the benchmark's mixer, models.py:82-84), in rotor2 units u_i = Omega_i^2 s:

    M1^-1 = I/m,   M2^-1 = J^-1,   V = m g z  (so mu grad V = g e3),
    g[:3, i] = (K_t/s) e3,
    g[3:, i] = (K_t a/s)[x_i, y_i, 0] + (K_c/s)[0, 0, z_i],  x=(-1,-1,1,1) y=(-1,1,1,-1) z=(1,-1,1,-1)

D_v and D_w are never substituted: the real vehicle's dissipation is unmeasured, so no truth exists for them.

Usage:
    python swap_ablation.py --run DIR [--horizon-seconds 1.0]
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import benchmark_protocol as bp

PROJECT_ROOT = bp.PROJECT_ROOT
EVAL_ROOT = PROJECT_ROOT / "experiments/quadrotor/eval_runs"
STEP = 0.01

MASS, GRAVITY, ARM, KT, KC = 0.045, 9.81, 0.0353, 3.72e-08, 7.74e-12
INERTIA = np.array([2.3951e-05, 2.3951e-05, 3.2347e-06])
SIGN_X = np.array([-1.0, -1.0, 1.0, 1.0])
SIGN_Y = np.array([-1.0, 1.0, 1.0, -1.0])
SIGN_Z = np.array([1.0, -1.0, 1.0, -1.0])


class SwappedModel(eqx.Module):
    """Delegates to ``base`` except for the operators named in the static flags."""

    base: eqx.Module
    m1_true: jax.Array
    m2_true: jax.Array
    g_true: jax.Array
    swap_m1: bool = eqx.field(static=True)
    swap_m2: bool = eqx.field(static=True)
    swap_v: bool = eqx.field(static=True)
    swap_gf: bool = eqx.field(static=True)
    swap_gtau: bool = eqx.field(static=True)
    preserve_damping: bool = eqx.field(static=True, default=False)

    # ---- operators ---------------------------------------------------------------------------
    def inverse_mass_1(self, position):
        return self.m1_true if self.swap_m1 else self.base.inverse_mass_1(position)

    def inverse_mass_2(self, rotation_flat):
        return self.m2_true if self.swap_m2 else self.base.inverse_mass_2(rotation_flat)

    def potential(self, pose):
        # V = m g z reproduces mu grad V = g e3 exactly, since mu = 1/m
        return MASS * GRAVITY * pose[2] if self.swap_v else self.base.potential(pose)

    def control_matrix(self, pose):
        learned = self.base.control_matrix(pose)
        force = self.g_true[:3] if self.swap_gf else learned[:3]
        torque = self.g_true[3:] if self.swap_gtau else learned[3:]
        return jnp.concatenate([force, torque], axis=0)

    def dissipation_v(self, velocity):
        return self.base.dissipation_v(velocity)

    def dissipation_w(self, omega):
        return self.base.dissipation_w(omega)

    # ---- gauge-consistent damping -------------------------------------------------------------
    # Only the PRODUCT M^-1 D is identifiable from data.  Substituting a true M^-1 without rescaling D
    # multiplies that product by M_true^-1 / M_learned^-1 (a factor of ~2.4e4 for the rotational block),
    # which annihilates the twist and freezes the attitude.  These keep M^-1 D at its learned value:
    #     D_new = (M_true^-1)^-1 @ M_learned^-1 @ D_learned
    def damping_v_at(self, position, velocity):
        d = self.base.dissipation_v(velocity)
        if not (self.preserve_damping and self.swap_m1):
            return d
        return jnp.linalg.solve(self.m1_true, self.base.inverse_mass_1(position) @ d)

    def damping_w_at(self, rotation_flat, omega):
        d = self.base.dissipation_w(omega)
        if not (self.preserve_damping and self.swap_m2):
            return d
        return jnp.linalg.solve(self.m2_true, self.base.inverse_mass_2(rotation_flat) @ d)

    # ---- the three methods the integrator needs, copied so they call SELF's operators ----------
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
               - self.damping_v_at(position, velocity) @ d_h_dpv + force_torque[:3])
        dpw = (jnp.cross(momentum_w, d_h_dpw) + jnp.cross(momentum_v, d_h_dpv)
               + jnp.cross(rotation, d_h_dr.reshape(3, 3), axis=-1).sum(axis=0)
               - self.damping_w_at(rotation_flat, omega) @ d_h_dpw + force_torque[3:6])
        dmass_inv_v = jax.jvp(self.inverse_mass_1, (position,), (dx,))[1]
        dmass_inv_w = jax.jvp(self.inverse_mass_2, (rotation_flat,), (dr,))[1]
        dv = mass_inv_v @ dpv + dmass_inv_v @ momentum_v
        dw = mass_inv_w @ dpw + dmass_inv_w @ momentum_w
        return jnp.concatenate([dx, dr, dv, dw, jnp.zeros(4, dtype=state.dtype)])

    def effective_damping(self, state):
        return (self.inverse_mass_1(state[:3]) @ self.damping_v_at(state[:3], state[12:15]),
                self.inverse_mass_2(state[3:12]) @ self.damping_w_at(state[3:12], state[15:18]))


def all_true(base, dataset=None, preserve_damping=False):
    """The trained model with all five analytic operators substituted (D_v, D_w stay learned)."""
    path = dataset or (PROJECT_ROOT / "datasets/QUADROTOR-DATASET-IDSIA/IDSIA_CF21BL_10s_h0p01_rotor2_clean.pkl")
    with Path(path).open("rb") as handle:
        scale = float(pickle.load(handle)["settings"]["rotor2_scale"])
    g = np.zeros((6, 4))
    g[2, :] = KT / scale
    g[3, :] = KT * ARM / scale * SIGN_X
    g[4, :] = KT * ARM / scale * SIGN_Y
    g[5, :] = KC / scale * SIGN_Z
    return SwappedModel(base=base, m1_true=jnp.eye(3) / MASS,
                        m2_true=jnp.diag(jnp.asarray(1.0 / INERTIA)), g_true=jnp.asarray(g),
                        swap_m1=True, swap_m2=True, swap_v=True, swap_gf=True, swap_gtau=True,
                        preserve_damping=preserve_damping)


def errors(truth, prediction):
    pos = float(np.sqrt(np.mean(np.sum((prediction[:, 1:, :3] - truth[:, 1:, :3]) ** 2, axis=-1))))
    Rt = truth[:, 1:, 3:12].reshape(truth.shape[0], -1, 3, 3)
    Rp = prediction[:, 1:, 3:12].reshape(truth.shape[0], -1, 3, 3)
    cos = np.clip((np.trace(np.einsum("fhji,fhjk->fhik", Rt, Rp), axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    att = float(np.degrees(np.sqrt(np.mean(np.arccos(cos) ** 2))))
    om = float(np.sqrt(np.mean(np.sum((prediction[:, 1:, 15:18] - truth[:, 1:, 15:18]) ** 2, axis=-1))))
    return pos, att, om


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "envs/quadrotor_se3_idsia/idsia_raw/data/test")
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--output-name", required=True)
    arguments = parser.parse_args()

    from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation

    with (PROJECT_ROOT / "datasets/QUADROTOR-DATASET-IDSIA/IDSIA_CF21BL_10s_h0p01_rotor2_clean.pkl").open("rb") as fh:
        scale = float(pickle.load(fh)["settings"]["rotor2_scale"])
    g_true = np.zeros((6, 4))
    g_true[2, :] = KT / scale
    g_true[3, :] = KT * ARM / scale * SIGN_X
    g_true[4, :] = KT * ARM / scale * SIGN_Y
    g_true[5, :] = KC / scale * SIGN_Z

    run = evaluation.load_run(Path(arguments.run), None)
    base, _p, _s = evaluation.build_model(run)
    flights = bp.load_melon(arguments.source, "rotor2", "melon*.csv")
    keep = int(round(arguments.horizon_seconds / STEP)) + 1
    truth = np.stack([f[:keep] for f in flights])
    print(f"\n{truth.shape[0]} melon flights x {arguments.horizon_seconds:g} s, inference only, h = {STEP}")

    def wrapped(**flags):
        return SwappedModel(base=base,
                            m1_true=jnp.eye(3) / MASS,
                            m2_true=jnp.diag(jnp.asarray(1.0 / INERTIA)),
                            g_true=jnp.asarray(g_true),
                            swap_m1=flags.get("m1", False), swap_m2=flags.get("m2", False),
                            swap_v=flags.get("v", False), swap_gf=flags.get("gf", False),
                            swap_gtau=flags.get("gtau", False))

    cases = [
        ("learned (no swap)", {}),
        ("+ true M2^-1", {"m2": True}),
        ("+ true g_tau", {"gtau": True}),
        ("+ true M2^-1 and g_tau  [ROTATIONAL]", {"m2": True, "gtau": True}),
        ("+ true M1^-1", {"m1": True}),
        ("+ true V (gravity)", {"v": True}),
        ("+ true g_f (thrust)", {"gf": True}),
        ("+ true M1^-1, V, g_f  [TRANSLATIONAL]", {"m1": True, "v": True, "gf": True}),
        ("+ ALL five true", {"m1": True, "m2": True, "v": True, "gf": True, "gtau": True}),
    ]
    print(f"\n  {'substitution':>40}{'pos RMSE (m)':>15}{'att RMSE (deg)':>16}{'omega RMSE':>13}")
    out = {}
    for name, flags in cases:
        model = wrapped(**flags)
        try:
            prediction = evaluation.rollout(model, truth, STEP)
        except FloatingPointError:
            print(f"  {name:>40}{'non-finite':>15}")
            out[name] = {"status": "non-finite"}
            continue
        pos, att, om = errors(truth, prediction)
        print(f"  {name:>40}{pos:>15.4f}{att:>16.2f}{om:>13.4f}")
        out[name] = {"position_rmse_m": pos, "attitude_rmse_deg": att, "omega_rmse": om, "flags": flags}

    folder = EVAL_ROOT / arguments.output_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "swap_ablation.json").write_text(json.dumps({
        "run": str(Path(arguments.run).resolve()), "generated_at": datetime.now().astimezone().isoformat(),
        "horizon_seconds": arguments.horizon_seconds, "flights": int(truth.shape[0]),
        "vehicle": {"mass": MASS, "inertia": INERTIA.tolist(), "arm": ARM, "kt": KT, "kc": KC,
                    "rotor2_scale": scale},
        "cases": out,
    }, indent=2) + "\n")
    print(f"\nwritten {folder / 'swap_ablation.json'}")


if __name__ == "__main__":
    main()

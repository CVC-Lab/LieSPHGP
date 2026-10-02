"""Roll out a port-Hamiltonian model built from the FITTED constant operators, Lie-IMEX, over all IDSIA flights.

This answers a narrow question: if we hand the SE(3) port-Hamiltonian model the best constant-operator rigid
body we can fit to this real dataset, and integrate it with the same Lie-IMEX solver the learned models use,
how close does it come to the measured trajectories?

The six operators, fitted on the TRAINING split only (see validate_input_reconstruction.py):

    M1^-1 = I / m,                      m     = 44.74 g          (published 45.0 g)
    M2^-1 = diag(1/Jxx, 1/Jyy, 1/Jzz),  Jxx,Jyy = 3.37, 3.96e-5  (published 2.3951e-5 both)
                                        Jzz   = published 3.2347e-6, since the fit for it is ill-posed
    V     = m g z                       g     = 9.81
    D_v   = m * kv * I,                 kv    = 0.360 1/s        so that M1^-1 D_v = 0.360 I, a LINEAR drag
    D_w   = J * diag(kw / J),           kw    = (3.2, 6.3, 0.11)e-4
    g     = selection matrix, plus a thrust-to-torque column carrying the fitted constant TRIM torque
            (+6.8, +8.1, -0.06)e-5 N m, which is motor mismatch and is exactly what a g coupling represents.

Everything is compared against a constant-velocity extrapolation from the same initial state, so the numbers
say whether the physics is adding anything at each horizon.

Usage: python rollout_fitted_operators.py [--horizons 0.1 0.5 1.0 3.0 10.0]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]  # comparision/idsia -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation  # noqa: E402
from src.models.SE3_Quadrotor.ph_gp_lie_imex import network  # noqa: E402

MASS = 0.04474
GRAVITY = 9.81
INERTIA = np.array([3.37e-5, 3.96e-5, 3.2347e-6])
DRAG_V = 0.360                                   # M1^-1 D_v, 1/s
DRAG_W = np.array([3.2e-4, 6.3e-4, 1.1e-5])      # the k_omega of J dw = tau - k_omega w - trim
TRIM = np.array([6.8e-5, 8.1e-5, -6.1e-7])       # N m


class FittedRealSE3HamODE(network.SampledSE3HamODE):
    """The fitted constant operators inside the same port-Hamiltonian vector field and Lie-IMEX solver."""

    mass: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    inertia: jax.Array
    drag_v: float = eqx.field(static=True)
    drag_w: jax.Array
    trim: jax.Array

    def inverse_mass_1(self, position):
        return jnp.eye(3, dtype=position.dtype) / self.mass

    def inverse_mass_2(self, rotation_flat):
        return jnp.diag(1.0 / self.inertia).astype(rotation_flat.dtype)

    def dissipation_v(self, velocity, position=None):
        # M1^-1 D_v = drag_v * I  =>  D_v = m * drag_v * I   (linear, no (1+|v|) shape: none is resolvable)
        return self.mass * self.drag_v * jnp.eye(3, dtype=velocity.dtype)

    def dissipation_w(self, omega, rotation_flat=None):
        # J dw = tau - k_omega w  =>  D_w = diag(k_omega)
        return jnp.diag(self.drag_w).astype(omega.dtype)

    def potential(self, pose):
        return self.mass * self.gravity * pose[2]

    def control_matrix(self, pose):
        physical = jnp.zeros((6, 4), dtype=pose.dtype)
        physical = physical.at[2, 0].set(1.0)                 # thrust along body z
        physical = physical.at[3:, 1:].set(jnp.eye(3, dtype=pose.dtype))
        # Motor mismatch: at hover thrust this column subtracts the fitted constant trim torque.
        return physical.at[3:, 0].set(-self.trim / (self.mass * self.gravity))


def build() -> FittedRealSE3HamODE:
    return FittedRealSE3HamODE(
        weights={}, gp_setup={}, mass=float(MASS), gravity=float(GRAVITY),
        inertia=jnp.asarray(INERTIA, dtype=jnp.float64),
        drag_v=float(DRAG_V), drag_w=jnp.asarray(DRAG_W, dtype=jnp.float64),
        trim=jnp.asarray(TRIM, dtype=jnp.float64))


def segment(flights: np.ndarray, keep: int) -> np.ndarray:
    pieces = flights.shape[1] // keep
    return flights[:, : pieces * keep].reshape(-1, keep, flights.shape[2])


def errors(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    position = np.sqrt(np.nanmean(np.sum((prediction[..., :3] - truth[..., :3]) ** 2, axis=-1)))
    velocity = np.sqrt(np.nanmean(np.sum((prediction[..., 12:15] - truth[..., 12:15]) ** 2, axis=-1)))
    omega = np.sqrt(np.nanmean(np.sum((prediction[..., 15:18] - truth[..., 15:18]) ** 2, axis=-1)))
    a = truth[..., 3:12].reshape(*truth.shape[:2], 3, 3)
    b = prediction[..., 3:12].reshape(*prediction.shape[:2], 3, 3)
    cosine = np.clip((np.trace(np.einsum("...ji,...jk->...ik", a, b), axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    attitude = np.degrees(np.sqrt(np.nanmean(np.arccos(cosine) ** 2)))
    final = np.sqrt(np.nanmean(np.sum((prediction[:, -1, :3] - truth[:, -1, :3]) ** 2, axis=-1)))
    return {"position": float(position), "final": float(final), "velocity": float(velocity),
            "attitude_deg": float(attitude), "omega": float(omega)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets/QUADROTOR-DATASET-IDSIA/IDSIA_CF21BL_10s_h0p01_clean.pkl")
    parser.add_argument("--horizons", type=float, nargs="+", default=[0.1, 0.5, 1.0, 3.0, 10.0])
    arguments = parser.parse_args()
    with arguments.dataset.open("rb") as handle:
        payload = pickle.load(handle)
    step = payload["settings"]["sample_dt"]
    model = build()
    print(f"\nFitted-operator port-Hamiltonian model, Lie-IMEX, h = {step} s")
    print(f"  M1^-1 = {1 / MASS:.3f} I    M2^-1 = diag({', '.join(f'{1 / j:.0f}' for j in INERTIA)})")
    print(f"  mu grad V = (0, 0, {GRAVITY})    mu D_v = {DRAG_V} I    "
          f"M2^-1 D_w = ({', '.join(f'{k / j:.2f}' for k, j in zip(DRAG_W, INERTIA))})")
    print(f"  trim torque in g: {np.array2string(TRIM, formatter={'float_kind': lambda v: f'{v:+.1e}'})} N m\n")

    for split in ("train", "test"):
        flights = np.asarray(payload[f"{split}_trajectories"], dtype=np.float64)
        label = f"{split} ({'chirp/random/square' if split == 'train' else 'melon, held out'})"
        print(f"=== {label}: {flights.shape[0]} flights of {(flights.shape[1] - 1) * step:.2f} s")
        print(f"   {'horizon':>8}{'segments':>10}{'position':>10}{'final':>9}{'velocity':>10}{'attitude':>10}"
              f"{'omega':>9}   {'const-v position':>17}")
        for horizon in arguments.horizons:
            keep = int(round(horizon / step)) + 1
            if keep > flights.shape[1]:
                continue
            pieces = segment(flights, keep)
            prediction = evaluation.rollout(model, pieces, step)
            prediction = np.asarray(prediction)
            measured = errors(pieces, prediction)
            start = np.einsum("fij,fj->fi", pieces[:, 0, 3:12].reshape(-1, 3, 3), pieces[:, 0, 12:15])
            constant = pieces[:, :1, :3] + start[:, None, :] * (np.arange(keep) * step)[None, :, None]
            reference = float(np.sqrt(np.nanmean(np.sum((constant - pieces[..., :3]) ** 2, axis=-1))))
            print(f"   {horizon:>7.1f}s{pieces.shape[0]:>10d}{measured['position']:>10.4f}{measured['final']:>9.4f}"
                  f"{measured['velocity']:>10.4f}{measured['attitude_deg']:>10.2f}{measured['omega']:>9.3f}"
                  f"{reference:>17.4f}")
        print()
    print("  position/final in m, velocity in m/s, attitude in deg (geodesic), omega in rad/s")
    print("  'position' is the RMSE over the whole window; 'final' is the error at its last sample\n")


if __name__ == "__main__":
    main()

"""JAX evaluation for the 47-page PH-GP-LieIMEX report.

Everything here is pure JAX / NumPy: loading the completed run, rolling the
posterior-mean model out on the common clean 240 Hz PID benchmark, querying the
six learned operators along the truth trajectory, comparing them with the
analytic ground truth of the simulator, fitting the mass gauge, and timing the
rollout.  No legacy report code is imported.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from ..ph_gp_lie_imex import network
from ..ph_gp_lie_imex.checkpoints import load_checkpoint
from ..ph_gp_lie_imex.integrator import rollout_control_sequence
from ..ph_node.integrator import rollout_control_sequence as rk4_rollout_control_sequence
from ..ph_gp_lie_imex.integrator import lie_imex_step
from ..ph_node.integrator import rk4_step
from ..ph_node.network import DissipativeSE3HamNODE as PhNodeModule


PHYSICS_HZ = 240
PID_HZ = 48
HOLD_STEPS = PHYSICS_HZ // PID_HZ
DT = 1.0 / PHYSICS_HZ
DAMPING_COEFFICIENT = 0.5
REPORT_TRAJECTORIES = 10
GP_NAMES = ("M1", "M2", "Dv", "Dw", "V", "g")
SUBNETWORK_LABELS = (
    ("M1⁻¹", "m1"),
    ("M2⁻¹", "m2"),
    ("Dv", "dv"),
    ("Dω", "dw"),
    ("V", "potential"),
    ("g", "control"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Common clean evaluation dataset (D0 layout, 240 Hz PID, 5-step held control)
# ---------------------------------------------------------------------------
def load_common_dataset(path: Path) -> dict[str, Any]:
    dataset_path = Path(path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Common evaluation dataset is missing: {dataset_path}")
    with dataset_path.open("rb") as handle:
        data = pickle.load(handle)
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("The common evaluation dataset must carry settings metadata")
    # Split sizes come from the pickle itself: the original D0 file is 96 train / 24 test
    # windows (0.5 s report horizon); the reference-flight files put every window in test_x.
    try:
        flights = len(settings["initial_xyz"])
        train_windows = int(settings["train_windows"])
        test_windows = int(settings["test_windows"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"Common evaluation dataset settings lack the split sizes: {error}") from error
    if train_windows + test_windows != int(settings.get("control_windows", train_windows + test_windows)):
        raise ValueError("Common evaluation dataset split sizes do not add up to control_windows")
    expected = {
        "x": (flights, HOLD_STEPS, train_windows, 22),
        "test_x": (flights, HOLD_STEPS, test_windows, 22),
        "t": (HOLD_STEPS,),
    }
    for key, shape in expected.items():
        actual = None if key not in data else np.asarray(data[key]).shape
        if actual != shape:
            raise ValueError(f"Dataset {key!r} has shape {actual}; expected {shape}")
    requirements = {
        "released_d0_pickle_used": False,
        "control_layout": "total thrust, body tau_x, body tau_y, body tau_z",
        "drone_model": "CF2P",
        "physics": "Physics.PYB",
        "control_hold_steps": HOLD_STEPS,
        "contact_free": True,
        "zero_contact_events": True,
        "contact_friction_disabled": True,
        "damping_law": ("nonlinear", "linear"),
        "linear_damping_coefficient": DAMPING_COEFFICIENT,
        "angular_damping_coefficient": DAMPING_COEFFICIENT,
    }
    failures = {
        key: settings.get(key)
        for key, value in requirements.items()
        if ((settings.get(key) not in value) if isinstance(value, tuple) else (settings.get(key) != value))
    }
    if failures:
        raise ValueError(f"Common evaluation dataset validation failed: {failures}")
    return data


def chronological_trajectories(x: np.ndarray) -> np.ndarray:
    """Convert (flight, phase, control-window, channel) to physical time order."""
    x = np.asarray(x, dtype=np.float64)
    return np.transpose(x, (0, 2, 1, 3)).reshape(x.shape[0], -1, x.shape[-1])


def shared_truth(data: dict[str, Any], count: int = REPORT_TRAJECTORIES) -> np.ndarray:
    """The first ``count`` distinct held-out flights, the same set for every model.

    Each flight carries its own initial state and its own recorded u(t).  Every reported metric
    therefore has a real across-flight spread, and two models can be compared paired per flight.

    Cross-model fairness comes from every model being scored on this identical set, not from the
    flights resembling one another.  An earlier version repeated one flight ``count`` times, which
    made every standard deviation zero by construction and overstated the effective sample size of
    the speed-binned damping statistics by a factor of ``count``.
    """
    held_out = chronological_trajectories(data["test_x"])
    if held_out.shape[0] < count:
        raise ValueError(
            f"Evaluation dataset holds {held_out.shape[0]} flights; the report needs {count} distinct ones"
        )
    return held_out[:count].copy()


# ---------------------------------------------------------------------------
# Completed training run
# ---------------------------------------------------------------------------
def load_run(run_dir: Path, selected_step: int | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" and selected_step is None:
        # A failed or running run can still be reported at an explicit intermediate checkpoint.
        raise ValueError(f"Training run is not complete: {run_dir}; pass --selected-step for an intermediate checkpoint")
    stats = {key: np.asarray(value) for key, value in np.load(run_dir / "training_stats.npz").items()}
    config_files = sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
    if not config_files:
        raise FileNotFoundError(f"No config copy inside {run_dir}")
    config = yaml.safe_load(config_files[0].read_text(encoding="utf-8"))
    total_steps = int(metadata.get("total_steps", config["training"]["total_steps"]))
    if selected_step is None:
        selected_step = total_steps
    checkpoint = (
        run_dir / "checkpoint_final.pkl"
        if selected_step == total_steps and (run_dir / "checkpoint_final.pkl").is_file()
        else run_dir / f"checkpoint_step_{selected_step:05d}.pkl"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    eval_steps = stats["step"].astype(int)
    if selected_step not in eval_steps:
        raise ValueError(f"Step {selected_step} was never evaluated in {run_dir}")
    index = int(np.where(eval_steps == selected_step)[0][0])
    return {
        "directory": run_dir,
        "metadata": metadata,
        "config": config,
        "config_path": config_files[0],
        "stats": stats,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "selected_step": selected_step,
        "total_steps": total_steps,
        "selected_test": {
            "total": float(stats["test_total"][index]),
            "position": float(stats["position"][index]),
            "linear_velocity": float(stats["linear_velocity"][index]),
            "angular_velocity": float(stats["angular_velocity"][index]),
            "attitude": float(stats["attitude"][index]),
        },
        "best_test": {
            "step": int(eval_steps[int(np.nanargmin(stats["test_total"]))]),
            "clean_test_mse": float(np.nanmin(stats["test_total"])),
        },
    }


def _to_float64(tree: Any) -> Any:
    def convert(leaf: Any) -> Any:
        if hasattr(leaf, "dtype") and np.issubdtype(np.asarray(leaf).dtype, np.floating):
            return jnp.asarray(np.asarray(leaf), dtype=jnp.float64)
        return leaf

    return jax.tree_util.tree_map(convert, tree)


def is_gp_checkpoint(payload: dict[str, Any]) -> bool:
    """GP runs store the fixed feature setup alongside the variational parameters; NN runs do not."""
    return isinstance(payload.get("extra"), dict) and "gp_setup" in payload["extra"]


def build_model(run: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Posterior-mean (GP) or point-estimate (NN) model in float64 from the selected checkpoint."""
    payload = load_checkpoint(run["checkpoint"])
    params = _to_float64(payload["params"])
    if not is_gp_checkpoint(payload):
        # PH-NN checkpoints store the equinox module itself; it exposes the same six subnetwork methods.
        return params, params, {}
    gp_setup = _to_float64(payload["extra"]["gp_setup"])
    model = network.DissipativeSE3HamODE(params, gp_setup).sample()
    return model, params, gp_setup


class GroundTruthSE3HamODE(network.SampledSE3HamODE):
    """The simulator's analytic operators inside the same port-Hamiltonian vector field.

    Report-only upper bound: identical Lie-IMEX integrator, identical recorded
    controls, but M1^-1 = I/m, M2^-1 = J^-1, V = m g z, g = selection matrix,
    Dv = m c (1+|v|) I, Dw = c (1+|omega|) J from the evaluation dataset.
    """

    mass: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    inertia: jax.Array

    def inverse_mass_1(self, position):
        return jnp.eye(3, dtype=position.dtype) / self.mass

    def inverse_mass_2(self, rotation_flat):
        return jnp.linalg.inv(self.inertia).astype(rotation_flat.dtype)

    def dissipation_v(self, velocity, position=None):
        # position is accepted and ignored: the simulator's damping has no pose dependence.
        return self.mass * self.damping * (1.0 + jnp.linalg.norm(velocity)) * jnp.eye(3, dtype=velocity.dtype)

    def dissipation_w(self, omega, rotation_flat=None):
        return self.damping * (1.0 + jnp.linalg.norm(omega)) * self.inertia.astype(omega.dtype)

    def potential(self, pose):
        return self.mass * self.gravity * pose[2]

    def control_matrix(self, pose):
        physical = jnp.zeros((6, 4), dtype=pose.dtype)
        physical = physical.at[2, 0].set(1.0)
        return physical.at[3:, 1:].set(jnp.eye(3, dtype=pose.dtype))


def ground_truth_model(vehicle: dict[str, Any]) -> GroundTruthSE3HamODE:
    return GroundTruthSE3HamODE(
        weights={}, gp_setup={},
        mass=float(vehicle["mass"]), gravity=float(vehicle["gravity"]), damping=float(vehicle["damping"]),
        inertia=jnp.asarray(vehicle["inertia"], dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# Open-loop rollout on the shared benchmark and its errors
# ---------------------------------------------------------------------------
def is_rk4_model(model: Any) -> bool:
    """Published PH-NODE: trained with RK4, so it must be rolled out with RK4, not Lie-IMEX."""
    return isinstance(model, PhNodeModule)


def _substep_rollout(model, initial, controls, step_size, substeps: int):
    """One data interval = ``substeps`` integrator steps of ``step_size / substeps``, control held constant."""
    single = rk4_step if is_rk4_model(model) else lie_imex_step
    inner = step_size / substeps

    def scan_step(state, control):
        controlled = state.at[:, 18:22].set(control)

        def inner_body(carry, _):
            advanced = single(model, carry, inner)
            return advanced.at[:, 18:22].set(carry[:, 18:22]), None

        next_state, _ = jax.lax.scan(inner_body, controlled, None, length=substeps)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial, controls)
    return jnp.concatenate([initial[None], trajectory], axis=0)


def rollout(model: network.SampledSE3HamODE, truth: np.ndarray, step_size: float = DT,
            substeps: int = 1) -> np.ndarray:
    """Autoregressive rollout driven by the recorded wrench u[k+1], in the model's own integrator.

    ``substeps`` > 1 integrates each data interval in that many smaller steps; the emitted trajectory still has
    one entry per data sample, so metrics and plots are unaffected.
    """
    initial = jnp.asarray(truth[:, 0], dtype=jnp.float64)
    controls = jnp.asarray(np.transpose(truth[:, 1:, 18:22], (1, 0, 2)), dtype=jnp.float64)
    if substeps > 1:
        trajectory = _substep_rollout(model, initial, controls, jnp.asarray(step_size, jnp.float64), int(substeps))
    else:
        advance = rk4_rollout_control_sequence if is_rk4_model(model) else rollout_control_sequence
        trajectory = advance(model, initial, controls, jnp.asarray(step_size, jnp.float64))
    trajectory.block_until_ready()
    prediction = np.transpose(np.asarray(trajectory), (1, 0, 2))[..., :18]
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite open-loop report rollout")
    return prediction


def posterior_sample_rollouts(
    params: dict[str, Any], gp_setup: dict[str, Any], truth: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, int]:
    """Open-loop rollouts of ``count`` independent posterior weight samples on trajectory 0 of ``truth``.

    Returns (samples, finite_count): samples has shape (finite_count, time, 18); a sample whose rollout
    leaves the finite range is dropped and counted.  The spread across samples is the model's
    epistemic (weight-space) uncertainty propagated through the Lie-IMEX integrator.
    """
    variational = network.DissipativeSE3HamODE(params, gp_setup)
    rollouts = []
    for index in range(int(count)):
        sampled = variational.sample(jax.random.PRNGKey(int(seed) + index))
        try:
            rollouts.append(rollout(sampled, truth[:1])[0])
        except FloatingPointError:
            continue
    if not rollouts:
        raise FloatingPointError("Every posterior-sample rollout was non-finite")
    return np.stack(rollouts, axis=0), len(rollouts)


def benchmark(model: network.SampledSE3HamODE, truth: np.ndarray, repeats: int, device: Any) -> dict[str, Any]:
    rollout(model, truth[:, : min(9, truth.shape[1])])  # compile and warm up
    timings = []
    prediction = None
    peak = 0
    for _ in range(repeats):
        started = time.perf_counter()
        current = rollout(model, truth)
        timings.append(time.perf_counter() - started)
        if prediction is None:
            prediction = current
        stats = getattr(device, "memory_stats", lambda: None)()
        if stats:
            peak = max(peak, int(stats.get("peak_bytes_in_use", 0)))
    transitions = truth.shape[0] * (truth.shape[1] - 1)
    # RK4 evaluates the vector field four times per step; the second-order Lie-IMEX stepper twice.
    stages = 4.0 if is_rk4_model(model) else 2.0
    return {
        "median_seconds": float(np.median(timings)),
        "minimum_seconds": float(np.min(timings)),
        "timings_seconds": [float(value) for value in timings],
        "transitions_per_second": float(transitions / np.median(timings)),
        "nfe": int(stages * (truth.shape[1] - 1)),
        "nfe_per_transition": stages,
        "auxiliary_damping_queries_per_transition": stages,
        "peak_device_memory_bytes": peak,
        "prediction": prediction,
    }


def project_rotation(rotations: np.ndarray) -> np.ndarray:
    flat = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    u, _, vt = np.linalg.svd(flat)
    projected = u @ vt
    negative = np.linalg.det(projected) < 0
    if np.any(negative):
        u[negative, :, -1] *= -1
        projected[negative] = u[negative] @ vt[negative]
    return projected.reshape(np.asarray(rotations).shape)


def attitude_error_squared(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    ref = project_rotation(reference[..., 3:12].reshape(*reference.shape[:-1], 3, 3))
    pred = project_rotation(prediction[..., 3:12].reshape(*prediction.shape[:-1], 3, 3))
    relative = np.swapaxes(ref, -1, -2) @ pred
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine) ** 2


def vehicle_constants(settings: dict[str, Any]) -> dict[str, Any]:
    vehicle = settings["vehicle_parameters"]
    return {
        "mass": float(vehicle["mass"]),
        "inertia": np.asarray(vehicle["inertia"], dtype=np.float64),
        "gravity": float(vehicle["gravity_acceleration"]),
        "damping": float(settings.get("linear_damping_coefficient", DAMPING_COEFFICIENT)),
        "damping_law": str(settings.get("damping_law", "nonlinear")),
    }


def damping_factor(magnitude: np.ndarray, vehicle: dict[str, Any]) -> np.ndarray:
    """Scalar damping law: c(1+|.|) for the PyBullet built-in nonlinear law, c for the linear law."""
    c = vehicle["damping"]
    if vehicle.get("damping_law", "nonlinear") == "linear":
        return np.full_like(np.asarray(magnitude, dtype=np.float64), c)
    return c * (1.0 + np.asarray(magnitude, dtype=np.float64))


def physical_energy(states: np.ndarray, vehicle: dict[str, Any]) -> np.ndarray:
    velocity = states[..., 12:15]
    omega = states[..., 15:18]
    translational = 0.5 * vehicle["mass"] * np.sum(velocity * velocity, axis=-1)
    rotational = 0.5 * np.einsum("...i,ij,...j->...", omega, vehicle["inertia"], omega)
    potential = vehicle["mass"] * vehicle["gravity"] * states[..., 2]
    return translational + rotational + potential


def rollout_errors(reference: np.ndarray, prediction: np.ndarray, vehicle: dict[str, Any]) -> dict[str, np.ndarray]:
    position = np.sum((prediction[..., :3] - reference[..., :3]) ** 2, axis=-1)
    velocity = np.sum((prediction[..., 12:15] - reference[..., 12:15]) ** 2, axis=-1)
    omega = np.sum((prediction[..., 15:18] - reference[..., 15:18]) ** 2, axis=-1)
    attitude = attitude_error_squared(reference, prediction)
    energy = np.abs(physical_energy(prediction, vehicle) - physical_energy(reference, vehicle))
    rotations = prediction[..., 3:12].reshape(*prediction.shape[:-1], 3, 3)
    determinant = np.abs(np.linalg.det(rotations) - 1.0)
    orthogonality = np.linalg.norm(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3), axis=(-2, -1))
    return {
        "position": position,
        "attitude": attitude,
        "velocity": velocity,
        "omega": omega,
        "energy": energy,
        "determinant": determinant,
        "orthogonality": orthogonality,
    }


def euler_angles(states: np.ndarray) -> np.ndarray:
    rotations = project_rotation(states[..., 3:12].reshape(*states.shape[:-1], 3, 3))
    angles = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_euler("xyz")
    return np.unwrap(angles.reshape(*states.shape[:-1], 3), axis=1)


# ---------------------------------------------------------------------------
# Learned operators along the truth trajectory versus analytic ground truth
# ---------------------------------------------------------------------------
def query_subnetworks(model: network.SampledSE3HamODE, states: np.ndarray) -> dict[str, np.ndarray]:
    shape = states.shape[:2]
    flat = jnp.asarray(states.reshape(-1, states.shape[-1]), dtype=jnp.float64)
    m1 = jax.vmap(model.inverse_mass_1)(flat[:, :3])
    m2 = jax.vmap(model.inverse_mass_2)(flat[:, 3:12])
    dv = jax.vmap(model.dissipation_v)(flat[:, 12:15], flat[:, :3])
    dw = jax.vmap(model.dissipation_w)(flat[:, 15:18], flat[:, 3:12])
    potential = jax.vmap(model.potential)(flat[:, :12])
    control = jax.vmap(model.control_matrix)(flat[:, :12])
    return {
        "m1": np.asarray(m1).reshape(*shape, 3, 3),
        "m2": np.asarray(m2).reshape(*shape, 3, 3),
        "dv": np.asarray(dv).reshape(*shape, 3, 3),
        "dw": np.asarray(dw).reshape(*shape, 3, 3),
        "potential": np.asarray(potential).reshape(*shape),
        "control": np.asarray(control).reshape(*shape, 6, 4),
    }


# ---------------------------------------------------------------------------
# Gauge-invariant products (need no scale fit): mu*g_f, M2^-1 g_tau, mu*grad V, mu*Dv, M2^-1 Dw
# ---------------------------------------------------------------------------
PRODUCT_LABELS = (
    ("thrust_gain", "thrust gain  μ·g_f  (body accel. per unit thrust, m/s² per N)"),
    ("torque_gain", "torque gain  M2⁻¹·g_τ  (angular accel. per unit torque, rad/s² per N·m)"),
    ("gravity", "gravity  μ·∇V  (m/s²)"),
    ("damping_v", "translational damping  μ·Dv  (1/s)"),
    ("damping_w", "rotational damping  M2⁻¹·Dω  (1/s)"),
)
SPEED_BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0))


def gauge_invariant_products(model: network.SampledSE3HamODE, states: np.ndarray) -> dict[str, np.ndarray]:
    """Products of the learned operators along the report trajectories; every entry is identifiable from data."""
    shape = states.shape[:2]
    flat = jnp.asarray(states.reshape(-1, states.shape[-1]), dtype=jnp.float64)

    def products(s):
        mu = model.inverse_mass_1(s[:3])[0, 0]
        m2 = model.inverse_mass_2(s[3:12])
        g = model.control_matrix(s[:12])
        grad_v = jax.grad(model.potential)(s[:12])[:3]
        return (mu * g[:3, 0], m2 @ g[3:6, 1:4], mu * grad_v, mu * model.dissipation_v(s[12:15], s[:3]), m2 @ model.dissipation_w(s[15:18], s[3:12]))

    thrust, torque, gravity, damping_v, damping_w = jax.vmap(products)(flat)
    return {
        "thrust_gain": np.asarray(thrust).reshape(*shape, 3),
        "torque_gain": np.asarray(torque).reshape(*shape, 3, 3),
        "gravity": np.asarray(gravity).reshape(*shape, 3),
        "damping_v": np.asarray(damping_v).reshape(*shape, 3, 3),
        "damping_w": np.asarray(damping_w).reshape(*shape, 3, 3),
    }


def product_targets(truth: np.ndarray, vehicle: dict[str, Any]) -> dict[str, np.ndarray]:
    shape = truth.shape[:2]
    mass, inertia, gravity, c = vehicle["mass"], vehicle["inertia"], vehicle["gravity"], vehicle["damping"]
    speed = np.linalg.norm(truth[..., 12:15], axis=-1)
    rate = np.linalg.norm(truth[..., 15:18], axis=-1)
    thrust = np.zeros((*shape, 3)); thrust[..., 2] = 1.0 / mass
    grav = np.zeros((*shape, 3)); grav[..., 2] = gravity
    return {
        "thrust_gain": thrust,
        "torque_gain": np.broadcast_to(np.linalg.inv(inertia), (*shape, 3, 3)).copy(),
        "gravity": grav,
        "damping_v": damping_factor(speed, vehicle)[..., None, None] * np.eye(3),
        "damping_w": damping_factor(rate, vehicle)[..., None, None] * np.eye(3),
    }


def product_metrics(products: dict[str, np.ndarray], targets: dict[str, np.ndarray], truth: np.ndarray) -> dict[str, Any]:
    """Per product: NMSE / relative RMS error per trajectory, per-axis mean model vs truth; speed-binned mu*Dv."""
    out: dict[str, Any] = {}
    for key, _ in PRODUCT_LABELS:
        pred, target = products[key], targets[key]
        axes = tuple(range(1, pred.ndim))
        mse = np.mean((pred - target) ** 2, axis=axes)
        power = np.mean(target ** 2, axis=axes)
        if pred.ndim == 4:   # 3x3: report the diagonal per axis, off-diagonal as a fraction of the diagonal
            model_axis = np.stack([pred[..., i, i].mean() for i in range(3)])
            truth_axis = np.stack([target[..., i, i].mean() for i in range(3)])
            off = pred - pred * np.eye(3)
            offdiag = float(np.sqrt(np.mean(off ** 2) * 9 / 6) / np.mean(truth_axis))
        else:
            model_axis = pred.mean(axis=(0, 1)); truth_axis = target.mean(axis=(0, 1)); offdiag = float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_axis = np.where(np.abs(truth_axis) > 1e-12, model_axis / truth_axis - 1.0, np.nan)
        out[key] = {
            "nmse": mse / power, "relative_rms_error_percent": 100.0 * np.sqrt(mse / power),
            "model_axis_mean": model_axis, "truth_axis_mean": truth_axis, "relative_error_axis": rel_axis,
            "offdiagonal_over_diagonal": offdiag,
        }
    speed = np.linalg.norm(truth[..., 12:15], axis=-1).reshape(-1)
    diag = np.stack([products["damping_v"][..., i, i] for i in range(3)], -1).mean(-1).reshape(-1)
    truth_diag = targets["damping_v"][..., 0, 0].reshape(-1)
    bins = []
    for low, high in SPEED_BINS:
        mask = (speed >= low) & (speed < high)
        bins.append({"low": low, "high": high, "count": int(mask.sum()),
                     "model": float(diag[mask].mean()) if mask.any() else float("nan"),
                     "truth": float(truth_diag[mask].mean()) if mask.any() else float("nan")})
    out["damping_v_speed_bins"] = bins
    return out


def analytic_targets(truth: np.ndarray, vehicle: dict[str, Any]) -> dict[str, np.ndarray]:
    shape = truth.shape[:2]
    mass, inertia, gravity, c = vehicle["mass"], vehicle["inertia"], vehicle["gravity"], vehicle["damping"]
    m1 = np.broadcast_to(np.eye(3) / mass, (*shape, 3, 3)).copy()
    m2 = np.broadcast_to(np.linalg.inv(inertia), (*shape, 3, 3)).copy()
    potential = mass * gravity * truth[..., 2]
    control = np.zeros((*shape, 6, 4), dtype=np.float64)
    control[..., 2, 0] = 1.0
    control[..., 3:, 1:] = np.eye(3)
    speed = np.linalg.norm(truth[..., 12:15], axis=-1)
    rate = np.linalg.norm(truth[..., 15:18], axis=-1)
    dv = (mass * damping_factor(speed, vehicle))[..., None, None] * np.eye(3)
    dw = damping_factor(rate, vehicle)[..., None, None] * inertia
    return {"m1": m1, "m2": m2, "dv": dv, "dw": dw, "potential": potential, "control": control}


def scale_fit(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sum(prediction * prediction))
    return float(np.sum(prediction * target) / denominator) if denominator > 0 else 1.0


def trajectory_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"Subnetwork shape mismatch: {prediction.shape} != {target.shape}")
    axes = tuple(range(1, prediction.ndim))
    return np.mean((prediction - target) ** 2, axis=axes)


def align_subnetworks(raw: dict[str, np.ndarray], targets: dict[str, np.ndarray]):
    """Apply the mass-derived port-Hamiltonian gauge, not per-operator fits."""
    beta_v = scale_fit(raw["m1"], targets["m1"])
    beta_w = scale_fit(raw["m2"], targets["m2"])
    potential_scaled = raw["potential"] / beta_v
    potential_offset = float(np.mean(targets["potential"] - potential_scaled))
    control = raw["control"].copy()
    control[..., :3, :] /= beta_v
    control[..., 3:, :] /= beta_w
    aligned = {
        "m1": beta_v * raw["m1"],
        "m2": beta_w * raw["m2"],
        "dv": raw["dv"] / beta_v,
        "dw": raw["dw"] / beta_w,
        "potential": potential_scaled + potential_offset,
        "control": control,
    }
    fits = {"beta_v": beta_v, "beta_w": beta_w, "potential_offset": potential_offset}
    metrics: dict[str, np.ndarray] = {}
    for label, key in SUBNETWORK_LABELS:
        # NMSE = MSE / mean of the squared ground-truth entries over the same horizon (unit-free, per trajectory);
        # sqrt(NMSE) is the relative RMS error of the subnetwork.
        target_power = trajectory_mse(np.zeros_like(targets[key]), targets[key])
        raw_mse = trajectory_mse(raw[key], targets[key])
        fixed_mse = trajectory_mse(aligned[key], targets[key])
        metrics[f"{label} MSE — raw"] = raw_mse
        metrics[f"{label} MSE — gauge-fixed"] = fixed_mse
        metrics[f"{label} NMSE — raw"] = raw_mse / target_power
        metrics[f"{label} NMSE — gauge-fixed"] = fixed_mse / target_power
        metrics[f"{label} relative RMS error (%) — gauge-fixed"] = 100.0 * np.sqrt(fixed_mse / target_power)
    return aligned, fits, metrics


def posterior_standard_deviations(params: Any) -> dict[str, np.ndarray]:
    if not isinstance(params, dict) or "M1" not in params:
        return {}                       # point-estimate model: no posterior widths
    return {name: np.exp(np.asarray(params[name]["log_std"], dtype=np.float64)).ravel() for name in GP_NAMES}


def learned_levels(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict) or "M1" not in params:
        return {"note": "point-estimate network: no learned levels or likelihood sigmas"}
    return {
        "M1_log_level": float(np.asarray(params["M1"]["log_level"])),
        "M2_level": np.asarray(params["M2"]["level"], dtype=np.float64).tolist(),
        "Dv_level": np.asarray(params["Dv"]["level"], dtype=np.float64).tolist(),
        "Dw_level": np.asarray(params["Dw"]["level"], dtype=np.float64).tolist(),
        "g_level": np.asarray(params["g"]["level"], dtype=np.float64).tolist(),
        "V_level": np.asarray(params["V"].get("level", np.zeros(3)), dtype=np.float64).tolist(),
        "likelihood_sigma": {
            key.replace("log_sigma_", ""): float(np.exp(np.asarray(value)))
            for key, value in params.get("likelihood", {}).items()
        },
    }

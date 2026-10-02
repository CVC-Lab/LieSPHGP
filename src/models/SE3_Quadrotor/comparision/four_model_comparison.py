"""NN-ODE / NN-SDE / GP-ODE / GP-SDE on one dataset: an open-loop PDF and a closed-loop PDF, plots first.

Models are given as LABEL=RUN_DIR@STEP; the package is read from the run's config (model.name):
    ph_nn_lie_imex            NN-ODE   point estimate, one deterministic path
    ph_nn_lie_imex_sde        NN-SDE   point estimate + diffusion subnetwork (Brownian paths)
    ph_gp_lie_imex_idsia      GP-ODE   posterior mean path; band = posterior weight samples
    ph_gp_lie_imex_sde_idsia  GP-SDE   posterior mean path; band = weight samples x Brownian paths
Analytical-SDE reference models are added from the dataset settings (--analytical plain|same|different): the simulator's exact drift (M1^-1 = I/m,
M2^-1 = J^-1, V = m g z, g = selection, linear damping c m v / c J omega) and, for a white_state_dependent
wind dataset, its exact diffusion M^-1 Sigma(x) = diag(sigma_a(x) I, sigma_alpha(x) I). Its band is the
spread any perfect SDE must show; its mean-path error is the irreducible open-loop error of the wind.

Subcommands
    open-loop      rollouts restarted every --horizon steps along held-out flights -> open-loop-comparison.pdf
    closed-loop    every model (and Analytical-SDE) flies each --reference in PyBullet under the SAME energy controller,
                   with the dataset's wind on the plant, for --seeds wind realisations -> closed-loop-comparison.pdf
    fly            one closed-loop flight (used internally by closed-loop, one process per flight)

All rollouts use each package's own network and integrator; nothing is imported from the legacy archive.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from . import report_evaluation as evaluation  # noqa: E402
from ..ph_gp_lie_imex_idsia import network as gp_ode_network  # noqa: E402
from ..ph_gp_lie_imex_idsia.integrator import rollout_control_sequence as gp_ode_rollout  # noqa: E402
from ..ph_gp_lie_imex_sde_idsia import network as gp_sde_network  # noqa: E402
from ..ph_gp_lie_imex_sde_idsia.integrator import rollout_control_sequence_sde as gp_sde_rollout  # noqa: E402
from ..ph_nn_lie_imex.integrator import rollout_control_sequence as nn_ode_rollout  # noqa: E402
from ..ph_nn_lie_imex_sde.integrator import rollout_control_sequence_sde as nn_sde_rollout  # noqa: E402
from ..ph_nn_lie_imex_sde.integrator import lie_imex_sde_step as nn_sde_step  # noqa: E402
from ..ph_gp_lie_imex_sde_idsia.integrator import lie_imex_sde_step as gp_sde_step  # noqa: E402
from ..ph_nn_lie_imex_sde.losses import transition_diffusion  # noqa: E402
from ..ph_node.integrator import rollout_control_sequence as ph_node_rollout  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
STEP = 0.01
SPLIT_LABEL = "Held-out"          # page-title label of the flight split, set by each command
COLORS = {"NN-ODE": "tab:blue", "NN-SDE": "tab:cyan", "NN-SDE-wind-same": "tab:olive", "GP-SDE-wind-same": "tab:brown", "GP-ODE": "tab:red", "GP-SDE": "tab:orange", "Analytical-SDE": "0.35",
          "Analytical-SDE-wind-same": "tab:green", "Analytical-SDE-wind-different": "tab:purple", "PH-NODE-RK4": "tab:pink"}
KIND_BY_PACKAGE = {"ph_nn_lie_imex": "nn_ode", "ph_nn_lie_imex_sde": "nn_sde",
                   "ph_gp_lie_imex_idsia": "gp_ode", "ph_gp_lie_imex_sde_idsia": "gp_sde", "ph_node": "ph_node"}
BAND_TEXT = {"nn_ode": "none (deterministic point estimate)",
             "ph_node": "none (deterministic point estimate, RK4)",
             "nn_sde": "Brownian paths of the learned diffusion",
             "gp_ode": "posterior weight samples (epistemic only)",
             "gp_sde": "posterior weight samples x Brownian paths",
             "truth": "Brownian paths of the TRUE diffusion (the irreducible spread); line = the zero-wind path",
             "analytical_same": "none: ONE path driven by the simulator's RECORDED wind dW_t (the same draw as the data)",
             "analytical_diff": "Brownian paths of the TRUE diffusion, independent of the data's wind (dW_t different)"}
ANALYTICAL_KINDS = ("truth", "analytical_same", "analytical_diff")
ANALYTICAL_SPECS = {"plain": ("Analytical-SDE", "truth"), "same": ("Analytical-SDE-wind-same", "analytical_same"),
                    "different": ("Analytical-SDE-wind-different", "analytical_diff")}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TruthModel(gp_sde_network.SampledSE3HamODE):
    """The simulator's exact operators inside the GP-SDE package's port-Hamiltonian vector field."""

    mass: float = 0.0
    gravity: float = 0.0
    damping_v: float = 0.0
    damping_w: float = 0.0
    inertia: Any = None
    wind: Any = None            # (linear_sigma, speed_gain, angular_sigma, rate_gain) or None

    def inverse_mass_1(self, position):
        return jnp.eye(3, dtype=position.dtype) / self.mass

    def inverse_mass_2(self, rotation_flat):
        return jnp.linalg.inv(jnp.asarray(self.inertia, dtype=rotation_flat.dtype))

    def dissipation_v(self, velocity, position=None):
        return self.mass * self.damping_v * jnp.eye(3, dtype=velocity.dtype)

    def dissipation_w(self, omega, rotation_flat=None):
        return self.damping_w * jnp.asarray(self.inertia, dtype=omega.dtype)

    def potential(self, pose):
        return self.mass * self.gravity * pose[2]

    def control_matrix(self, pose):
        physical = jnp.zeros((6, 4), dtype=pose.dtype).at[2, 0].set(1.0)
        return physical.at[3:, 1:].set(jnp.eye(3, dtype=pose.dtype))

    def diffusion_scale(self, state):
        if self.wind is None:
            return jnp.zeros((6,), dtype=state.dtype)
        linear_sigma, speed_gain, angular_sigma, rate_gain = self.wind
        sigma_a = linear_sigma * (1.0 + speed_gain * jnp.linalg.norm(state[12:15]))
        sigma_alpha = angular_sigma * (1.0 + rate_gain * jnp.linalg.norm(state[15:18]))
        return jnp.concatenate([jnp.full((3,), sigma_a), jnp.full((3,), sigma_alpha)])

    def twist_noise(self, state, noise):
        # M^-1 Sigma(x) z with Sigma = diag(m sigma_a, J sigma_alpha) -> diag(sigma_a, sigma_alpha) z in twist units
        return self.diffusion_scale(state) * noise


class RecordedWindModel(TruthModel):
    """Analytical-SDE driven by the simulator's recorded wind: the noise argument already holds the twist increment
    [F_wind / m, J^-1 tau_wind] sqrt(dt) per unit sqrt(dt), so the diffusion term reproduces the data's own dW_t."""

    def twist_noise(self, state, noise):
        return noise


class RecordedWindWrapper(eqx.Module):
    """A learned model driven by the RECORDED wind: every operator is the learned one, only twist_noise returns the
    given increment (the recorded [F/m, J^-1 tau] sqrt(h)) instead of the learned diffusion. Tests the learned drift
    pathwise, independent of its diffusion."""

    base: Any

    def vector_field(self, state):
        return self.base.vector_field(state)

    def effective_damping(self, state):
        return self.base.effective_damping(state)

    def twist_noise(self, state, noise):
        return noise


def load_wind(path: Path, split: str, settings: dict[str, Any]) -> np.ndarray | None:
    """(B, T, 6) recorded wind as twist accelerations [F/m, J^-1 tau] (body frame); row k acts on the interval k-1 -> k."""
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    prefix = "" if split == "train" else f"{split}_"
    if f"{prefix}gust_force" not in data or f"{prefix}gust_torque" not in data:
        return None
    vehicle = settings["vehicle_parameters"]
    force = np.asarray(data[f"{prefix}gust_force"], dtype=np.float64) / float(vehicle["mass"])
    torque = np.asarray(data[f"{prefix}gust_torque"], dtype=np.float64) @ np.linalg.inv(np.asarray(vehicle["inertia"])).T
    return np.concatenate([force, torque], axis=-1)


def wind_of(settings: dict[str, Any]) -> tuple[float, float, float, float] | None:
    gusts = (settings.get("config") or {}).get("gusts") or {}
    if gusts.get("model") != "white_state_dependent" or not gusts.get("enabled", True):
        return None
    return (float(gusts["linear_sigma"]), float(gusts["speed_gain"]), float(gusts["angular_sigma"]), float(gusts["rate_gain"]))


def wind_hold_seconds(settings: dict[str, Any]) -> float:
    return float(((settings.get("config") or {}).get("gusts") or {}).get("hold_seconds", 1.0 / 240.0))


def truth_model(settings: dict[str, Any]) -> TruthModel:
    vehicle = settings["vehicle_parameters"]
    return TruthModel(
        weights={}, gp_setup={}, mass=float(vehicle["mass"]), gravity=float(vehicle["gravity_acceleration"]),
        damping_v=float(settings.get("linear_damping_coefficient", 0.5)),
        damping_w=float(settings.get("angular_damping_coefficient", settings.get("linear_damping_coefficient", 0.5))),
        inertia=np.asarray(vehicle["inertia"], dtype=np.float64), wind=wind_of(settings))


class PhNodeOperators(eqx.Module):
    """PH-NODE's six operators with the (x / v_b, omega_b) argument convention the report tools use.

    The shared tools call dissipation_v(v_b, x) and dissipation_w(omega_b, R); PH-NODE's own methods let the SECOND
    argument win (published pose-dependent damping), which is wrong for a body-twist model (dissipation_inputs:
    body-twist), whose vector field evaluates D_v(v_b), D_w(omega_b) (ph_node/network.py vector_field)."""
    model: Any

    def inverse_mass_1(self, position):
        return self.model.inverse_mass_1(position)

    def inverse_mass_2(self, rotation_flat):
        return self.model.inverse_mass_2(rotation_flat)

    def potential(self, pose):
        return self.model.potential(pose)

    def control_matrix(self, pose):
        return self.model.control_matrix(pose)

    def dissipation_v(self, velocity, position=None):
        body = self.model.body_twist_dissipation or position is None
        return self.model.dissipation_v(velocity if body else position)

    def dissipation_w(self, omega, rotation_flat=None):
        body = self.model.body_twist_dissipation or rotation_flat is None
        return self.model.dissipation_w(omega if body else rotation_flat)


class Entry:
    """One model: its mean dynamics, a sampler for its band, and the metadata the pages print."""

    def __init__(self, label: str, spec: str | None, settings: dict[str, Any]):
        self.label = label
        if spec is None or spec in ANALYTICAL_KINDS:
            self.kind, self.run, self.step = spec or "truth", None, None
            self.mean = truth_model(settings)
            if self.kind == "analytical_same":
                self.mean = RecordedWindModel(**{k: getattr(self.mean, k) for k in
                                                 ("weights", "gp_setup", "mass", "gravity", "damping_v", "damping_w", "inertia", "wind")})
            self.params = self.gp_setup = None
            return
        # spec = RUN_DIR[@STEP|@best][+windsame]: @best loads checkpoint_best.pkl; +windsame replays the recorded wind
        self.wind_same = spec.endswith("+windsame")
        spec = spec[: -len("+windsame")] if self.wind_same else spec
        directory, _, step = spec.partition("@")
        if step == "best":
            self.run = evaluation.load_run(Path(directory), None)
            self.run["checkpoint"] = Path(directory).resolve() / "checkpoint_best.pkl"
            self.run["checkpoint_sha256"] = evaluation.sha256(self.run["checkpoint"])
        else:
            self.run = evaluation.load_run(Path(directory), int(step) if step else None)
        self.kind = KIND_BY_PACKAGE[self.run["config"]["model"]["name"]]
        payload = evaluation.load_checkpoint(self.run["checkpoint"])
        self.step = int(payload.get("extra", {}).get("step", self.run["selected_step"]))
        self.run["selected_step"] = self.step
        self.params = evaluation._to_float64(payload["params"])
        self.gp_setup = evaluation._to_float64(payload["extra"]["gp_setup"]) if self.kind.startswith("gp") else None
        self.substeps = int(self.run["config"]["model"].get("integration_substeps", 1) or 1)
        if self.kind == "ph_node":
            self.mean = PhNodeOperators(self.params)
        elif self.kind == "gp_ode":
            self.mean = gp_ode_network.DissipativeSE3HamODE(self.params, self.gp_setup).sample(None)
        elif self.kind == "gp_sde":
            self.mean = gp_sde_network.DissipativeSE3HamODE(self.params, self.gp_setup).sample(None)
        else:
            self.mean = self.params

    def describe(self) -> str:
        if getattr(self, "wind_same", False):
            return f"{self.run['directory'].name} @ step {self.step}, driven by the RECORDED wind (learned drift, data's dW_t)"
        if self.kind in ANALYTICAL_KINDS:
            return {"truth": "all subnetworks = simulator values; zero-wind line, random-wind band",
                    "analytical_same": "all subnetworks = simulator values; fed the RECORDED wind of each flight",
                    "analytical_diff": "all subnetworks = simulator values; an independent wind draw"}[self.kind]
        return f"{self.run['directory'].name} @ step {self.step}"

    # ---- rollouts on time-major windows (T, B, 22), controls taken from the data ----
    def mean_path(self, windows: np.ndarray, wind: np.ndarray | None = None) -> np.ndarray:
        """``wind`` (T-1, B, 6): the recorded twist accelerations for each interval, used by analytical_same only."""
        states = jnp.asarray(windows)
        controls, h = states[1:, :, 18:22], jnp.asarray(STEP)
        if getattr(self, "wind_same", False):
            if wind is None:
                raise ValueError("+windsame needs the dataset's recorded wind (gust_force / gust_torque)")
            replay = RecordedWindWrapper(self.mean)
            rollout_fn = nn_sde_rollout if self.kind.startswith("nn") else gp_sde_rollout
            return np.asarray(rollout_fn(replay, states[0], controls, h, jnp.asarray(wind) * jnp.sqrt(h)))
        if self.kind == "analytical_same":
            if wind is None:
                raise ValueError("Analytical-SDE-wind-same needs the dataset's recorded wind (gust_force / gust_torque)")
            return np.asarray(gp_sde_rollout(self.mean, states[0], controls, h, jnp.asarray(wind) * jnp.sqrt(h)))
        if self.kind == "analytical_diff":            # one independent draw, fixed seed so every page shows the same one
            noise = jax.random.normal(jax.random.PRNGKey(20260925), controls.shape[:2] + (6,))
            return np.asarray(gp_sde_rollout(self.mean, states[0], controls, h, noise))
        if self.kind == "nn_ode":
            return np.asarray(nn_ode_rollout(self.mean, states[0], controls, h))
        if self.kind == "ph_node":                    # its own RK4 with the training sub-steps (ph_node/train.py _trajectory)
            return np.asarray(ph_node_rollout(self.params, states[0], controls, h, self.substeps))
        if self.kind == "gp_ode":
            return np.asarray(gp_ode_rollout(self.mean, states[0], controls, h, substeps=self.substeps))
        noise = jnp.zeros(controls.shape[:2] + (6,))
        if self.kind == "nn_sde":
            return np.asarray(nn_sde_rollout(self.mean, states[0], controls, h, noise))
        return np.asarray(gp_sde_rollout(self.mean, states[0], controls, h, noise))    # gp_sde and truth

    def sample_paths(self, windows: np.ndarray, samples: int, seed: int) -> np.ndarray | None:
        if self.kind in ("nn_ode", "ph_node", "analytical_same") or getattr(self, "wind_same", False):
            return None
        states = jnp.asarray(windows)
        controls, h = states[1:, :, 18:22], jnp.asarray(STEP)
        keys = jax.random.split(jax.random.PRNGKey(seed), samples)

        def one(key):
            weight_key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, controls.shape[:2] + (6,))
            if self.kind == "gp_ode":
                model = gp_ode_network.DissipativeSE3HamODE(self.params, self.gp_setup).sample(weight_key)
                return gp_ode_rollout(model, states[0], controls, h, substeps=self.substeps)
            if self.kind == "gp_sde":
                model = gp_sde_network.DissipativeSE3HamODE(self.params, self.gp_setup).sample(weight_key)
                return gp_sde_rollout(model, states[0], controls, h, noise)
            if self.kind == "nn_sde":
                return nn_sde_rollout(self.mean, states[0], controls, h, noise)
            return gp_sde_rollout(self.mean, states[0], controls, h, noise)

        return np.asarray(jax.lax.map(one, keys))


def load_dataset(path: Path, split: str) -> tuple[np.ndarray, dict[str, Any]]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return np.asarray(data[f"{split}_trajectories"], dtype=np.float64), data["settings"]


def build_entries(arguments, settings: dict[str, Any]) -> list["Entry"]:
    entries = [Entry(label, spec, settings) for label, spec in parse_models(arguments.model or [])]
    for name in (arguments.analytical or ["plain"]):
        label, kind = ANALYTICAL_SPECS[name]
        entries.append(Entry(label, kind, settings))
    return entries


def parse_models(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for spec in specs:
        label, _, rest = spec.partition("=")
        out.append((label, rest))
    return out


# ---------------------------------------------------------------------------
# Open loop
# ---------------------------------------------------------------------------
def geodesic(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    a = reference.reshape(*reference.shape[:-1], 3, 3)
    b = prediction.reshape(*prediction.shape[:-1], 3, 3)
    relative = np.swapaxes(a, -1, -2) @ b
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def project_matrices(matrices: np.ndarray) -> np.ndarray:
    """Nearest rotation (SVD) of each (..., 3, 3) matrix; a diverged (non-finite or overflowing) one stays NaN."""
    matrices = np.asarray(matrices, dtype=np.float64)
    out = np.full(matrices.shape, np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        ok = np.all(np.isfinite(matrices), axis=(-2, -1)) & (np.max(np.abs(matrices), axis=(-2, -1)) < 1e100)
    if np.any(ok):
        u, _, vt = np.linalg.svd(matrices[ok])
        d = np.sign(np.linalg.det(u @ vt))
        u[..., :, -1] *= d[..., None]
        out[ok] = u @ vt
    return out


def project(rotation_flat: np.ndarray) -> np.ndarray:
    return project_matrices(rotation_flat.reshape(*rotation_flat.shape[:-1], 3, 3)).reshape(*rotation_flat.shape)


def euler_deg(rotation_flat: np.ndarray) -> np.ndarray:
    shape = rotation_flat.shape[:-1]
    matrices = project(rotation_flat.reshape(-1, 9)).reshape(-1, 3, 3)
    angles = np.full((len(matrices), 3), np.nan)
    ok = np.all(np.isfinite(matrices), axis=(-2, -1))
    if np.any(ok):
        angles[ok] = Rotation.from_matrix(matrices[ok]).as_euler("xyz", degrees=True)
    return angles.reshape(*shape, 3)


def ensemble_mean(paths: np.ndarray) -> np.ndarray:
    """Mean over the sample axis (0) of (S, ..., 22) paths at every time; the mean rotation is projected onto SO(3)."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(paths, axis=0)
    rotation = np.nan_to_num(mean[..., 3:12], nan=0.0, posinf=0.0, neginf=0.0)
    mean[..., 3:12] = project(rotation + 1e-12 * np.eye(3).reshape(9))
    return mean


def centre_line(mean_path: np.ndarray, paths: np.ndarray | None) -> np.ndarray:
    """What every page draws and scores as the prediction: the ensemble mean of the sample paths when the model has
    them, otherwise its single deterministic path."""
    return mean_path if paths is None else ensemble_mean(paths)


CALIBRATION_BLOCKS = (("p", slice(0, 3)), ("v", slice(12, 15)), ("w", slice(15, 18)))


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Ensemble CRPS per element: E|X - y| - 0.5 E|X - X'| with X over axis 0 (sorted-sample identity, O(S log S))."""
    S = samples.shape[0]
    x = np.sort(samples, axis=0)
    first = np.mean(np.abs(x - truth[None]), axis=0)
    weights = (2.0 * np.arange(1, S + 1) - S - 1).reshape((S,) + (1,) * (samples.ndim - 1))
    spread = 2.0 * np.sum(weights * x, axis=0) / (S * S)
    return first - 0.5 * spread


def calibration_stats(truth: np.ndarray, paths: np.ndarray | None, centre: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Per block and per time step, pooled over windows/flights (axis 1) and the block's 3 coordinates:
    coverage of mean +- 1 sigma and +- 2 sigma, spread-skill ratio RMSE(mean) / RMS(sigma), and CRPS. A model without
    samples is a point forecast: only its CRPS (= absolute error) is defined."""
    out = {}
    for key, cols in CALIBRATION_BLOCKS:
        y = truth[..., cols]
        if paths is None:
            out[key] = {"crps": np.nanmean(np.abs(centre[..., cols] - y), axis=(1, 2))}
            continue
        x = paths[..., cols]
        with np.errstate(invalid="ignore"):
            mu, sigma = np.nanmean(x, axis=0), np.nanstd(x, axis=0, ddof=1)
            z = np.abs(y - mu) / np.maximum(sigma, 1e-12)
            out[key] = {"cover1": np.nanmean(z <= 1.0, axis=(1, 2)), "cover2": np.nanmean(z <= 2.0, axis=(1, 2)),
                        "spread_skill": np.sqrt(np.nanmean((y - mu) ** 2, axis=(1, 2)) / np.maximum(np.nanmean(sigma ** 2, axis=(1, 2)), 1e-24)),
                        "sigma": np.sqrt(np.nanmean(sigma ** 2, axis=(1, 2))),
                        "crps": np.nanmean(crps_ensemble(np.nan_to_num(x, nan=1e6), y), axis=(1, 2))}
    return out


def plottable(values) -> np.ndarray:
    """For log-scale plots only: a diverged (inf / NaN / > 1e12) value becomes NaN, i.e. a gap, instead of breaking the
    axis; the tables keep the true (infinite) value."""
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(values) & (np.abs(values) < 1e12), values, np.nan)


def calibration_figure(t, calib, entries, marks=(), title_extra="") -> plt.Figure:
    """4 x 3 grid: rows = coverage of +-1 sigma, coverage of +-2 sigma, spread-skill ratio, CRPS; columns = p, v, omega."""
    figure, axes = plt.subplots(4, 3, figsize=(11.69, 8.27), sharex=True)
    names = {"p": "position", "v": "body velocity", "w": "body angular velocity"}
    for col, (key, _) in enumerate(CALIBRATION_BLOCKS):
        for e in entries:
            stats = calib[e.label][key]
            style = dict(color=COLORS[e.label], lw=1.8 if e.kind not in ANALYTICAL_KINDS else 1.5,
                         ls="--" if e.kind in ANALYTICAL_KINDS else "-", label=e.label)
            if "cover1" in stats:
                axes[0, col].plot(t, stats["cover1"], **style)
                axes[1, col].plot(t, stats["cover2"], **style)
                axes[2, col].plot(t[1:], plottable(stats["spread_skill"][1:]), **style)
            axes[3, col].plot(t[1:], plottable(stats["crps"][1:]), **style)
        axes[0, col].axhline(0.683, color="k", lw=0.8, ls=":"); axes[0, col].set_ylim(0, 1.02)
        axes[1, col].axhline(0.954, color="k", lw=0.8, ls=":"); axes[1, col].set_ylim(0, 1.02)
        axes[2, col].axhline(1.0, color="k", lw=0.8, ls=":"); axes[2, col].set_yscale("log")
        axes[3, col].set_yscale("log")
        axes[0, col].set_title(f"{names[key]}: coverage of mean +- 1 sigma", fontsize=9)
        axes[1, col].set_title(f"{names[key]}: coverage of mean +- 2 sigma", fontsize=9)
        axes[2, col].set_title(f"{names[key]}: spread-skill RMSE / sigma", fontsize=9)
        axes[3, col].set_title(f"{names[key]}: CRPS (lower is better)", fontsize=9)
        for row in range(4):
            for mark in marks:
                axes[row, col].axvline(mark, color="0.5", lw=0.6, ls="--")
            axes[row, col].grid(alpha=0.3, which="both")
        axes[3, col].set_xlabel("horizon (s)")
    axes[3, 0].legend(fontsize=7)
    figure.suptitle("Calibration of the sample paths vs horizon (pooled over windows and axes). Dotted = ideal: 0.683, 0.954, 1.\n"
                    "Spread-skill > 1 = over-confident, < 1 = under-confident. CRPS of a deterministic model = its absolute error."
                    + title_extra, fontsize=10)
    figure.tight_layout()
    return figure


def healthy(paths: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """(B,) windows whose every path stays finite and within 100 m of the data: a blow-up guard only. Physical spread
    is NOT filtered - open loop in the WINDSDE wind even the true SDE drifts ~9 m (median) from the data in 3 s."""
    ok = np.isfinite(paths).all(axis=tuple(i for i in range(paths.ndim) if i != paths.ndim - 2))
    distance = np.nan_to_num(np.linalg.norm(paths[..., :3] - truth[..., :3], axis=-1), nan=np.inf)
    ok &= distance.reshape(-1, distance.shape[-1]).max(axis=0) <= 100.0
    return ok


def open_loop(arguments) -> None:
    flights, settings = load_dataset(arguments.dataset, arguments.split)
    entries = build_entries(arguments, settings)
    wind = load_wind(arguments.dataset, arguments.split, settings)
    H, S = arguments.horizon, arguments.samples
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    # ---- statistics over sliding windows of every flight ----
    starts = list(range(0, flights.shape[1] - H, arguments.stride))
    windows = np.stack([flights[f, s:s + H + 1] for f in range(flights.shape[0]) for s in starts], axis=1)   # (T, B, 22)
    wind_windows = None if wind is None else np.stack([wind[f, s + 1:s + H + 1] for f in range(flights.shape[0]) for s in starts], axis=1)
    stats: dict[str, dict[str, Any]] = {}
    for entry in entries:
        print(f"[open-loop] statistics {entry.label} ({entry.kind}) on {windows.shape[1]} windows", flush=True)
        mean = entry.mean_path(windows, wind_windows)
        record: dict[str, Any] = {"mean": mean}
        paths = entry.sample_paths(windows, S, arguments.seed)
        ok = healthy(mean[None], windows) if paths is None else healthy(mean[None], windows) & healthy(paths, windows)
        record["ok"] = ok
        if paths is not None:
            record["paths"] = paths
        stats[entry.label] = record
    common = np.logical_and.reduce([r["ok"] for r in stats.values()])
    print(f"[open-loop] windows kept by every model: {int(common.sum())}/{common.size}", flush=True)
    truth = windows[:, common]
    horizon_s = np.arange(H + 1) * STEP
    errors, calib = {}, {}
    for entry in entries:
        r = stats[entry.label]
        paths = r["paths"][:, :, common] if "paths" in r else None
        mean = centre_line(r["mean"][:, common], paths)
        calib[entry.label] = calibration_stats(truth, paths, mean)
        errors[entry.label] = {
            "p": np.median(np.linalg.norm(mean[..., :3] - truth[..., :3], axis=-1), axis=1),
            "R": np.degrees(np.median(geodesic(truth[..., 3:12], project(mean[..., 3:12])), axis=1)),
            "v": np.median(np.linalg.norm(mean[..., 12:15] - truth[..., 12:15], axis=-1), axis=1),
            "w": np.median(np.linalg.norm(mean[..., 15:18] - truth[..., 15:18], axis=-1), axis=1),
        }

    # ---- per-flight restarted rollouts for the trajectory pages ----
    flight_ids = list(range(min(arguments.flights, flights.shape[0])))
    restarts = list(range(0, flights.shape[1] - 1, H))
    segments = {}
    for entry in entries:
        per_flight = []
        for f in flight_ids:
            segs = []
            for s in restarts:
                end = min(s + H, flights.shape[1] - 1)
                window = flights[f, s:end + 1][:, None]
                mean = entry.mean_path(window, None if wind is None else wind[f, s + 1:end + 1][:, None])[:, 0]
                paths = entry.sample_paths(window, S, arguments.seed + 1000 * f + s)
                segs.append({"t": np.arange(s, end + 1) * STEP,
                             "mean": centre_line(mean, None if paths is None else paths[:, :, 0]),
                             "paths": None if paths is None else paths[:, :, 0]})
            per_flight.append(segs)
        segments[entry.label] = per_flight

    pdf_path = output / "open-loop-comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(open_loop_title(arguments, entries, windows.shape[1], int(common.sum()), restarts)); plt.close("all")
        pdf.savefig(error_page(horizon_s, errors, entries)); plt.close("all")
        pdf.savefig(calibration_figure(horizon_s, calib, entries)); plt.close("all")
        for index, f in enumerate(flight_ids):
            flight = flights[f]
            pdf.savefig(path3d_page(flight, [segments[e.label][index] for e in entries], entries, f)); plt.close("all")
            for block in ("p", "euler", "v", "w"):
                pdf.savefig(block_page(flight, [segments[e.label][index] for e in entries], entries, f, block, restarts))
                plt.close("all")
    summary = {"generated_at": datetime.now().astimezone().isoformat(), "dataset": str(arguments.dataset),
               "split": arguments.split, "horizon_steps": H, "samples": S, "stride": arguments.stride,
               "windows": int(windows.shape[1]), "windows_kept": int(common.sum()),
               "models": {e.label: {"kind": e.kind, "source": e.describe(), "band": BAND_TEXT[e.kind]} for e in entries},
               "median_error_at_horizon": {k: {b: float(v[b][-1]) for b in v} for k, v in errors.items()},
               "calibration_at_horizon": {k: {b: {m: float(x[-1]) for m, x in v[b].items()} for b in v} for k, v in calib.items()}}
    (output / "open-loop-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[open-loop] wrote {pdf_path}", flush=True)


def text_page(title: str, lines: list[str]) -> plt.Figure:
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(0.05, 0.93, title, fontsize=17, weight="bold")
    figure.text(0.05, 0.88, "\n".join(lines), fontsize=10, va="top", family="monospace")
    return figure


def open_loop_title(arguments, entries, n_windows, n_kept, restarts) -> plt.Figure:
    lines = [f"dataset   {arguments.dataset.name}  split={arguments.split}",
             f"protocol  open-loop Lie-IMEX rollouts at h = {STEP} s driven by the RECORDED wrench; each rollout starts from the",
             f"          true state and runs {arguments.horizon} steps ({arguments.horizon * STEP:.1f} s) with no feedback.",
             f"          Statistics: {n_windows} windows (stride {arguments.stride}), {n_kept} kept (every model finite and within 100 m).",
             f"          Trajectory pages: restart from the true state every {arguments.horizon * STEP:.1f} s (dotted lines), {len(restarts)} segments.",
             f"line/band {arguments.samples} sample paths per model: line = their mean at every t, dark = mean +- 1 sigma,",
             "          light = mean +- 2 sigma (sigma = sample std at every t). Models without samples: their single path.", "", "models"]
    for e in entries:
        lines.append(f"  {e.label:7s} {e.kind:7s} {e.describe()}")
        lines.append(f"          samples: {BAND_TEXT[e.kind]}")
    lines += ["", "how to read",
              "  Analytical-SDE = every subnetwork set to the simulator's value. The data are ONE wind realisation: fed that SAME",
              "  recorded wind (-wind-same) it must lie on the data; with a DIFFERENT wind draw (-wind-different) it cannot, and its",
              "  mean path drifts away from them; its grey band is the spread a perfect SDE must reproduce. An ODE",
              "  cannot represent it; a good SDE band should look like the Analytical-SDE band and contain the black data line."]
    return text_page("Open-loop comparison: NN-ODE / NN-SDE / GP-ODE / GP-SDE", lines)


def error_page(horizon_s, errors, entries) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    names = {"p": "position error |p - p_true| (m)", "R": "attitude error (deg)", "v": "body velocity error (m/s)",
             "w": "body angular-velocity error (rad/s)"}
    for axis, key in zip(axes.flat, ("p", "R", "v", "w")):
        for e in entries:
            axis.plot(horizon_s, errors[e.label][key], color=COLORS[e.label], lw=2 if e.kind not in ANALYTICAL_KINDS else 1.5,
                      ls="--" if e.kind in ANALYTICAL_KINDS else "-", label=f"{e.label}")
        axis.set_title(names[key]); axis.set_xlabel("horizon (s)"); axis.grid(alpha=0.3)
    axes[0, 0].legend()
    figure.suptitle("Median open-loop error of the prediction (mean of the sample paths) vs horizon (dashed = Analytical-SDE variants)")
    figure.tight_layout()
    return figure


def path3d_page(flight, per_model, entries, index) -> plt.Figure:
    columns = 3 if len(entries) <= 6 else 4                    # 2 x 3 up to six models, 2 x 4 for seven or eight
    figure = plt.figure(figsize=(11.69 if columns == 3 else 14.5, 8.27))
    for k, (e, segs) in enumerate(zip(entries, per_model)):
        axis = figure.add_subplot(math.ceil(len(entries) / columns), columns, k + 1, projection="3d")
        axis.plot(*flight[:, :3].T, color="k", lw=1.2, label="data")
        for j, seg in enumerate(segs):
            if seg["paths"] is not None:
                for path in seg["paths"][:8]:
                    axis.plot(*path[:, :3].T, color=COLORS[e.label], lw=0.4, alpha=0.35)
            axis.plot(*seg["mean"][:, :3].T, color=COLORS[e.label], lw=1.6, label=e.label if j == 0 else None)
            axis.scatter(*seg["mean"][0, :3], color="k", s=8)
        axis.set_title(f"{e.label}  (bold: mean of samples, thin: 8 samples)", fontsize=9)
        lo, hi = flight[:, :3].min(axis=0) - 0.5, flight[:, :3].max(axis=0) + 0.5
        axis.set_xlim(lo[0], hi[0]); axis.set_ylim(lo[1], hi[1]); axis.set_zlim(lo[2], hi[2])
        axis.set_xlabel("x"); axis.set_ylabel("y"); axis.set_zlabel("z")
    figure.suptitle(f"{SPLIT_LABEL} flight {index}: 3-D path, restarted open-loop segments (black = data, dots = restarts)")
    figure.tight_layout()
    return figure


BLOCKS = {"p": (slice(0, 3), ["x (m)", "y (m)", "z (m)"], "Position (world)"),
          "euler": (None, ["roll (deg)", "pitch (deg)", "yaw (deg)"], "Attitude (Euler xyz, from R)"),
          "v": (slice(12, 15), ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)"], "Body linear velocity"),
          "w": (slice(15, 18), ["w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"], "Body angular velocity")}


def block_values(states: np.ndarray, block: str) -> np.ndarray:
    if block == "euler":
        return euler_deg(states[..., 3:12])
    return states[..., BLOCKS[block][0]]


def block_page(flight, per_model, entries, index, block, restarts, marks=()) -> plt.Figure:
    _, labels, title = BLOCKS[block]
    # page width grows with the model count (2.7 in per column) so seven or more columns stay readable
    figure, axes = plt.subplots(3, len(entries), figsize=(max(11.69, 2.7 * len(entries)), 8.27), sharex=True)
    t_all = np.arange(flight.shape[0]) * STEP
    data = block_values(flight, block)
    for col, (e, segs) in enumerate(zip(entries, per_model)):
        for row in range(3):
            axis = axes[row, col]
            extent = [data[:, row].min(), data[:, row].max()]          # this panel's y-range: data + its line + its band

            def grow(values):
                finite = np.asarray(values)[np.isfinite(values)]
                if finite.size:
                    extent[0], extent[1] = min(extent[0], finite.min()), max(extent[1], finite.max())

            axis.plot(t_all, data[:, row], color="k", lw=0.9, label="data")
            for seg in segs:
                if seg["paths"] is not None:
                    values = block_values(seg["paths"], block)[..., row]
                    with np.errstate(invalid="ignore"):
                        mu, sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=1)
                    axis.fill_between(seg["t"], mu - 2 * sd, mu + 2 * sd, color=COLORS[e.label], alpha=0.15, lw=0)
                    axis.fill_between(seg["t"], mu - sd, mu + sd, color=COLORS[e.label], alpha=0.30, lw=0)
                    grow(mu - 2 * sd); grow(mu + 2 * sd)
                    mean_values = mu
                else:
                    mean_values = block_values(seg["mean"], block)[:, row]
                axis.plot(seg["t"], mean_values, color=COLORS[e.label], lw=1.3)
                grow(mean_values)
            # a diverged path or a very wide band must not flatten the data: cap the range at data +- 3x its width
            data_low, data_high = data[:, row].min(), data[:, row].max()
            reach = 3.0 * max(data_high - data_low, 1e-3)
            clipped = extent[0] < data_low - reach or extent[1] > data_high + reach
            extent = [max(extent[0], data_low - reach), min(extent[1], data_high + reach)]
            pad = 0.05 * max(extent[1] - extent[0], 1e-3)
            axis.set_ylim(extent[0] - pad, extent[1] + pad)
            if clipped:
                axis.text(0.98, 0.97, "clipped", transform=axis.transAxes, ha="right", va="top", fontsize=7, color="0.3")
            for s in restarts[1:]:
                axis.axvline(s * STEP, color="k", lw=0.5, ls=":")
            for mark in marks:
                axis.axvline(mark, color="0.5", lw=0.6, ls="--")
            axis.grid(alpha=0.25)
            if col == 0:
                axis.set_ylabel(labels[row])
            if row == 0:
                axis.set_title(e.label.replace("-wind-", "-\nwind-"), color=COLORS[e.label], weight="bold")
            if row == 2:
                axis.set_xlabel("t (s)")
    figure.suptitle(f"{SPLIT_LABEL} flight {index}: {title}\nblack = data, line = mean of the sample paths (single path if none), "
                    "dark = mean +- 1 sigma, light = mean +- 2 sigma; 'clipped' = y-axis capped at the data range +- 3x its width")
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------
def stub_run(settings_path: Path) -> dict[str, Any]:
    return {"directory": Path("ground-truth-operators"), "checkpoint": Path("none"), "checkpoint_sha256": None,
            "metadata": {"model_name": "ground_truth", "integrator": "n/a"}, "config": {}}


def recorded_flight_index(reference: str) -> int | None:
    """``<split>-flightNN`` names a recorded flight of the dataset split used as a closed-loop reference."""
    head, _, index = reference.rpartition("-flight")
    return int(index) if head and index.isdigit() else None


def recorded_initial_state(flight: np.ndarray) -> dict[str, np.ndarray]:
    """x_0 of a recorded flight in the (time, 22) layout: x_w, vec(R) row-major, v_b, omega_b."""
    return {"position": flight[0, :3], "rotation": flight[0, 3:12].reshape(3, 3),
            "velocity_body": flight[0, 12:15], "omega_body": flight[0, 15:18]}


def fly(arguments) -> None:
    from . import report_controller as rc
    flights, settings = load_dataset(arguments.dataset, "heldout" if arguments.split == "heldout" else arguments.split)
    label, spec = parse_models([arguments.model_spec])[0]
    entry = Entry(label, spec if spec in ANALYTICAL_KINDS else spec, settings)
    vehicle = evaluation.vehicle_constants(settings)
    wind = wind_of(settings)
    gust = None
    if wind is not None:
        gust = {"model": "white_state_dependent", "linear_sigma": wind[0], "speed_gain": wind[1], "angular_sigma": wind[2],
                "rate_gain": wind[3], "hold_seconds": wind_hold_seconds(settings), "seed": int(arguments.seed)}
    # A recorded test flight as the reference: absolute position and heading as flown, and the plant starts at its x_0.
    initial_state = None
    index = recorded_flight_index(arguments.reference)
    if arguments.reference not in rc.REFERENCES and index is not None:
        flight = flights[index]
        rc.REFERENCES[arguments.reference] = rc.make_recorded_reference(flight, float(settings.get("sample_dt", STEP)),
                                                                        relative_yaw=False)
        initial_state = recorded_initial_state(flight)
    del flights
    run = entry.run if entry.run is not None else stub_run(arguments.dataset)
    rc.run_controller(run, entry.mean, arguments.output, model_label=label, training_dataset=arguments.dataset,
                      vehicle=vehicle, duration_seconds=arguments.seconds, seed=int(arguments.seed),
                      use_dissipation=arguments.dissipation == "on", reference=arguments.reference, gust=gust,
                      provider_description=f"{label}: {entry.describe()}", initial_state=initial_state)


def closed_loop_references(arguments) -> list[str]:
    """--reference names plus, with --recorded N, the first N recorded flights of --split (``test-flight00`` ...)."""
    names = list(arguments.reference or [])
    names += [f"{arguments.split}-flight{k:02d}" for k in range(int(arguments.recorded or 0))]
    if not names:
        raise SystemExit("closed-loop: give --reference and/or --recorded N")
    return names


def closed_loop(arguments) -> None:
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    models = parse_models(arguments.model) + [("Analytical-SDE", "truth")]
    references = closed_loop_references(arguments)
    jobs = []
    for label, spec in models:
        for reference in references:
            for seed in range(arguments.seeds):
                folder = output / "flights" / reference / label / f"seed{seed}"
                if (folder / "controller_rollout.npz").is_file() and not arguments.refly:
                    continue
                jobs.append((label, spec, reference, seed, folder))
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print(f"[closed-loop] {len(jobs)} flights to fly with {arguments.workers} workers", flush=True)

    def launch(job):
        label, spec, reference, seed, folder = job
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "src.models.SE3_Quadrotor.comparision.four_model_comparison", "fly",
                   "--dataset", str(arguments.dataset), "--split", arguments.split, "--model-spec", f"{label}={spec}",
                   "--reference", reference, "--seed", str(seed), "--seconds", str(arguments.seconds),
                   "--dissipation", arguments.dissipation, "--output", str(folder)]
        with open(folder / "fly.log", "w") as log:
            code = subprocess.call(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"[closed-loop] {reference:18s} {label:7s} seed{seed} exit={code}", flush=True)
        return code

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        codes = list(pool.map(launch, jobs))
    if any(codes):
        print(f"[closed-loop] {sum(1 for c in codes if c)} flights exited non-zero (see fly.log); plotting what exists", flush=True)
    closed_loop_report(arguments, [label for label, _ in models])


# Booklet rows: (label, metric key, format, best). best: "low" / "high" = smallest / largest wins, "gt" = closest to the
# Analytical-SDE column, "count" = number of failed flights (not ranked). Analytical-SDE is the reference, never ranked.
CLOSED_LOOP_ROWS = (
    ("position RMSE (m)", "position_rmse_m", "{:.3f}", "low"),
    ("position max error (m)", "position_max_m", "{:.3f}", "low"),
    ("position final error (m)", "position_final_m", "{:.3f}", "low"),
    ("velocity RMSE (m/s)", "velocity_rmse_m_per_s", "{:.3f}", "low"),
    ("yaw RMSE (deg)", "yaw_rmse_deg", "{:.2f}", "low"),
    ("attitude RMSE vs commanded R_d (deg)", "attitude_vs_command_deg", "{:.2f}", "low"),
    ("valid tracking time vs Analytical-SDE (s)", "valid_tracking_seconds_vs_truth", "{:.2f}", "high"),
    ("RMS thrust / hover", "thrust_rms_hover", "{:.3f}", "gt"),
    ("control chattering (1/s)", "control_chatter_per_s", "{:.2f}", "low"),
    ("max tilt (deg)", "max_tilt_deg", "{:.1f}", "low"),
    ("motor saturation fraction", "shared_motor_saturation_fraction", "{:.3f}", "low"),
    ("completion fraction", "completion_fraction", "{:.2f}", "high"),
    ("controller failures (flights)", "controller_failure", None, "count"),
)
TRUTH_LABEL = "Analytical-SDE"


def load_flight(folder: Path) -> dict[str, Any] | None:
    """One flown seed: trajectories and the booklet metrics (report_controller.controller_comparison + attitude vs R_d)."""
    from . import report_controller as rc
    path = folder / "controller_rollout.npz"
    if not path.is_file():
        return None
    data = dict(np.load(path))
    metadata = json.loads((folder / "controller_metadata.json").read_text())
    metrics = rc.controller_comparison({"data": data, "metadata": metadata})
    if "desired_rotation" in data and len(data["desired_rotation"]):
        trace = np.einsum("nij,nij->n", data["desired_rotation"], data["current_rotation"])
        angle = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
        metrics["attitude_vs_command_deg"] = float(np.sqrt(np.mean(angle**2)))
    return {"s_traj": data["s_traj"], "s_plan": data["s_plan"], "s_plan_full": data["s_plan_full"],
            "metrics": metrics, "status": metadata["status"]}


def add_valid_tracking(flights: dict[str, list], labels: list[str]) -> None:
    """Valid tracking time of each learned flight against the Analytical-SDE flight with the SAME wind seed: first t
    where |p_model - p_truth| > 0.158 x the reference's RMS excursion (report_controller.VALID_TRACKING_FRACTION)."""
    from . import report_controller as rc
    for label in labels:
        if label == TRUTH_LABEL:
            continue
        for flight, truth in zip(flights[label], flights.get(TRUTH_LABEL, [])):
            if flight is None or truth is None:
                continue
            count = min(len(flight["s_traj"]), len(truth["s_traj"]))
            plan = flight["s_plan"]
            excursion = float(np.sqrt(np.mean(np.sum((plan[:, :3] - plan[0, :3]) ** 2, axis=1))))
            distance = np.linalg.norm(flight["s_traj"][:count, :3] - truth["s_traj"][:count, :3], axis=1)
            flight["metrics"]["valid_tracking_seconds_vs_truth"] = rc.valid_tracking_seconds(
                flight["s_traj"][:count, -1], distance, rc.VALID_TRACKING_FRACTION * excursion)


def seed_values(flights: list, key: str) -> np.ndarray:
    return np.array([float(f["metrics"][key]) for f in flights if f is not None and f["metrics"].get(key) is not None],
                    dtype=np.float64)


def closed_loop_stats(groups: list[list], key: str, direction: str, statistic: str = "mean"):
    """groups: one list of seeds per flight. One flight -> mean +- std over its seeds; several -> each flight's seed mean,
    then mean +- std over flights ('mean') or (median, 25th, 75th percentile) over flights ('median').
    'count' -> (failed, flown)."""
    if direction == "count":
        flown = [f for group in groups for f in group if f is not None]
        return (sum(bool(f["metrics"].get(key)) for f in flown), len(flown))
    per_flight = [seed_values(group, key) for group in groups]
    values = per_flight[0] if len(groups) == 1 else np.array([v.mean() for v in per_flight if v.size])
    if values.size == 0:
        return None
    if statistic == "median":
        return tuple(float(q) for q in np.percentile(values, (50, 25, 75)))
    return (float(values.mean()), float(values.std()))


def closed_loop_table_page(title: str, subtitle: str, groups_by_label: dict[str, list[list]], labels: list[str],
                           note: str, statistic: str = "mean") -> tuple[plt.Figure, dict]:
    """Metrics as rows, models as columns, mean +- std (or median [25th, 75th percentile]); best learned model per row
    green bold (Table A style), ranked by the mean (or the median)."""
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(0.03, 0.95, title, fontsize=15, weight="bold")
    figure.text(0.03, 0.90, subtitle, fontsize=7.5, va="top")
    learned = [i for i, label in enumerate(labels) if label != TRUTH_LABEL]
    truth_index = labels.index(TRUTH_LABEL) if TRUTH_LABEL in labels else None
    cells, best_cells, values_out = [], [], {}
    for row_index, (name, key, fmt, direction) in enumerate(CLOSED_LOOP_ROWS):
        stats = [closed_loop_stats(groups_by_label[label], key, direction, statistic) for label in labels]
        values_out[key] = dict(zip(labels, stats))
        row = [name]
        for value in stats:
            if value is None:
                row.append("-")
            elif direction == "count":
                row.append(f"{value[0]} / {value[1]}")
            elif len(value) == 3:
                row.append(f"{fmt.format(value[0])} [{fmt.format(value[1])}, {fmt.format(value[2])}]")
            else:
                row.append(f"{fmt.format(value[0])} +- {fmt.format(value[1])}")
        cells.append(row)
        ranked = [i for i in learned if stats[i] is not None]
        if direction == "count" or len(ranked) < 2:
            continue
        if direction == "gt" and truth_index is not None and stats[truth_index] is not None:
            score = lambda i: abs(stats[i][0] - stats[truth_index][0])
        elif direction == "high":
            score = lambda i: -stats[i][0]
        else:
            score = lambda i: stats[i][0]
        best = min(ranked, key=score)
        # ties (same printed value) share the highlight
        best_cells += [(row_index + 1, 1 + i) for i in ranked if fmt.format(stats[i][0]) == fmt.format(stats[best][0])]
    axis = figure.add_axes([0.02, 0.07, 0.96, 0.78]); axis.axis("off")
    header = ["metric"] + [label + ("\n(reference, not ranked)" if label == TRUTH_LABEL else "") for label in labels]
    widths = [0.28] + [0.72 / len(labels)] * len(labels)
    table = axis.table(cellText=cells, colLabels=header, colWidths=widths, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1.0, 1.75)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#d0d8e8"); cell.set_height(cell.get_height() * 1.5)
        elif c == 0:
            cell.set_facecolor("#f2f2f2"); cell.set_text_props(ha="left")
    for r, c in best_cells:
        table[r, c].set_facecolor("#c8ecc8"); table[r, c].set_text_props(weight="bold")
    figure.text(0.03, 0.015, note, fontsize=6.8)
    return figure, values_out


CLOSED_LOOP_NOTE = ("Green bold = best learned model per row (lowest; highest for valid tracking time and completion; closest to "
                    "Analytical-SDE for RMS thrust). Analytical-SDE = the simulator's exact operators under the same controller: the\n"
                    "controller's ceiling, never ranked. Valid tracking time = first t with |p_model - p_Analytical| > 0.158 x the "
                    "reference's RMS excursion, against the Analytical-SDE flight with the SAME wind seed.\n"
                    "Attitude vs R_d = geodesic angle between the flown R and the controller's commanded R_d (inner-loop tracking). "
                    "Failures: flights that ended early (controller exception / non-finite wrench / PyBullet) out of those flown.")


def seed_band(flights: list, column: int, unwrap: bool = False, scale: float = 1.0):
    """Seed mean and std of one s_traj column on the common time grid (NaN after a flight ends)."""
    flown = [f for f in flights if f is not None]
    length = max(len(f["s_traj"]) for f in flown)
    stack = np.full((len(flown), length), np.nan)
    for k, f in enumerate(flown):
        series = f["s_traj"][:, column]
        stack[k, :len(series)] = (np.unwrap(series) if unwrap else series) * scale
    time = next(f for f in flown if len(f["s_traj"]) == length)["s_traj"][:, -1]
    with np.errstate(invalid="ignore"):
        return time, np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


def closed_loop_tracking_page(flights: dict[str, list], labels: list[str], reference: str, duration: float) -> plt.Figure:
    """Booklet tracking grid: one row per model, x y z yaw vx vy vz omega; line = seed mean, band = +- 1 std over seeds."""
    states = (("x (m)", 0, 1.0), ("y (m)", 1, 1.0), ("z (m)", 2, 1.0), ("yaw (deg)", 9, 180.0 / np.pi),
              ("vx (m/s)", 3, 1.0), ("vy (m/s)", 4, 1.0), ("vz (m/s)", 5, 1.0), (r"$\omega_b$ (rad/s)", None, 1.0))
    plan = next(f for label in labels for f in flights[label] if f is not None)["s_plan_full"]
    seeds = max(len(flights[label]) for label in labels)
    figure, axes = plt.subplots(len(labels), len(states), figsize=(2.45 * len(states), 1.9 * len(labels) + 1.2), squeeze=False)
    for row, label in enumerate(labels):
        flown = [f for f in flights[label] if f is not None]
        for column, (name, index, scale) in enumerate(states):
            axis = axes[row][column]
            axis.grid(True, alpha=0.3)
            if not flown:
                axis.text(0.5, 0.5, "no flight", ha="center", transform=axis.transAxes)
                continue
            if index is None:
                for state_column, colour in ((10, "#1f77b4"), (11, "#2ca02c"), (12, "#9467bd")):
                    time, mean, std = seed_band(flown, state_column)
                    axis.plot(time, mean, color=colour, lw=0.8)
                    axis.fill_between(time, mean - std, mean + std, color=colour, alpha=0.2, lw=0)
            else:
                time, mean, std = seed_band(flown, index, unwrap=index == 9, scale=scale)
                axis.fill_between(time, mean - std, mean + std, color="b", alpha=0.25, lw=0,
                                  label="+- 1 std over seeds" if (row == 0 and column == 0) else None)
                axis.plot(time, mean, color="b", lw=1.1, label="flown (seed mean)" if (row == 0 and column == 0) else None)
                axis.plot(plan[:, -1], plan[:, index] * scale, "--", color="chocolate", lw=1.1,
                          label="reference" if (row == 0 and column == 0) else None)
            axis.tick_params(labelsize=6)
            if row == 0:
                axis.set_title(name, fontsize=9, fontweight="bold")
            if row == len(labels) - 1:
                axis.set_xlabel("Time (s)", fontsize=7)
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(label, fontsize=8, fontweight="bold", color=COLORS.get(label, "k"))
    handles, names = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=3, prop={"size": 10})
    figure.suptitle(f"{reference} — closed-loop tracking, {duration:g} s — every model\n"
                    f"line = mean over {seeds} wind seeds, band = +- 1 std over seeds; dashed = recorded test flight (reference); "
                    r"$\omega_b$ column: $\omega_x$ blue, $\omega_y$ green, $\omega_z$ purple", fontsize=11, fontweight="bold")
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    return figure


def closed_loop_trajectory_page(flights: dict[str, list], labels: list[str], reference: str, duration: float) -> plt.Figure:
    """Booklet 3-D grid: one panel per model; thin = each wind seed, thick = seed mean, dashed = reference."""
    plan = next(f for label in labels for f in flights[label] if f is not None)["s_plan_full"]
    figure = plt.figure(figsize=(4.4 * len(labels), 4.8))
    for index, label in enumerate(labels, start=1):
        axis = figure.add_subplot(1, len(labels), index, projection="3d")
        flown = [f for f in flights[label] if f is not None]
        for k, f in enumerate(flown):
            axis.plot3D(*f["s_traj"][:, :3].T, color="b", lw=0.5, alpha=0.35, label="each wind seed" if index == 1 and k == 0 else None)
        if flown:
            mean = np.stack([seed_band(flown, c)[1] for c in range(3)], axis=1)
            axis.plot3D(*mean.T, color="b", lw=1.5, label="seed mean" if index == 1 else None)
        axis.plot3D(*plan[:, :3].T, "--", color="chocolate", lw=1.5, label="reference" if index == 1 else None)
        positions = np.concatenate([plan[:, :3]] + [f["s_traj"][:, :3] for f in flown], axis=0)
        positions = positions[np.all(np.isfinite(positions), axis=1)]
        lower, upper = positions.min(axis=0), positions.max(axis=0)
        span = max(float(np.max(upper - lower)) * 1.1, 1.0)
        centre = (lower + upper) / 2.0
        axis.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
        axis.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
        axis.set_zlim(centre[2] - span / 2, centre[2] + span / 2)
        axis.view_init(elev=25.0, azim=35.0)
        axis.set_xlabel("x (m)", fontsize=7); axis.set_ylabel("y (m)", fontsize=7); axis.set_zlabel("z (m)", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.set_title(f"{label}\naxis span {span:.1f} m", fontsize=10, fontweight="bold", color=COLORS.get(label, "k"))
    handles, names = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=3, prop={"size": 10})
    figure.suptitle(f"{reference} — closed-loop trajectory, {duration:g} s — every model\n"
                    "same recorded start state x_0, reference, controller gains and wind seeds in every panel", fontsize=11,
                    fontweight="bold")
    figure.tight_layout(rect=(0, 0.06, 1, 0.9))
    return figure


def flight_segments(settings: dict[str, Any], split: str, index: int) -> str:
    """The generator's segment chain of a recorded flight, e.g. 'circle -> coast -> circle'."""
    import ast
    audits = settings.get(f"{'test' if split == 'test' else split}_flight_audits")
    if isinstance(audits, str):
        try:
            audits = ast.literal_eval(audits)
        except (ValueError, SyntaxError):
            return ""
    try:
        return " -> ".join(s["name"] for s in audits[index]["segments"] if float(s.get("duration", 1.0)) > 0.0)
    except (TypeError, KeyError, IndexError):
        return ""


def closed_loop_report(arguments, labels: list[str]) -> None:
    output = arguments.output
    references = closed_loop_references(arguments)
    flights = {ref: {label: [load_flight(output / "flights" / ref / label / f"seed{s}") for s in range(arguments.seeds)]
                     for label in labels} for ref in references}
    for ref in references:
        add_valid_tracking(flights[ref], labels)
    _, settings = load_dataset(arguments.dataset, arguments.split)
    median = getattr(arguments, "statistic", "mean") == "median"
    # the median variant only changes Table C and is written next to the mean booklet, never over it
    pdf_path = output / ("closed_loop_comparison_median.pdf" if median else "closed-loop-comparison.pdf")
    summary: dict[str, Any] = {"table_c_statistic": "median [25th, 75th percentile] over flights" if median else "mean +- std over flights"}
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(closed_loop_title(arguments, labels, references)); plt.close("all")
        figure, summary["table_c"] = closed_loop_table_page(
            f"Table C{' (median)' if median else ''} - closed-loop tracking over the {len(references)} {arguments.split} flights",
            f"{arguments.dataset.name}: each flight's value is the mean over its {arguments.seeds} wind seeds; cells = "
            + (f"MEDIAN [25th, 75th percentile] over the {len(references)} flights (best = by the median)" if median
               else f"mean +- std over the {len(references)} flights")
            + f".\n{arguments.seconds:g} s per flight, plant started at the recorded x_0.",
            {label: [flights[ref][label] for ref in references] for label in labels}, labels, CLOSED_LOOP_NOTE,
            statistic="median" if median else "mean")
        pdf.savefig(figure); plt.close("all")
        summary["flights"] = {}
        for ref in references:
            index = recorded_flight_index(ref)
            chain = flight_segments(settings, arguments.split, index) if index is not None else ""
            figure, summary["flights"][ref] = closed_loop_table_page(
                f"{ref} - closed-loop tracking, {arguments.seconds:g} s",
                f"{'segments: ' + chain + chr(10) if chain else ''}cells = mean +- std over {arguments.seeds} "
                "wind seeds (seed k = the same wind draw sequence for every model); same controller and gains for every model.",
                {label: [flights[ref][label]] for label in labels}, labels, CLOSED_LOOP_NOTE)
            pdf.savefig(figure); plt.close("all")
            if any(f is not None for label in labels for f in flights[ref][label]):
                pdf.savefig(closed_loop_tracking_page(flights[ref], labels, ref, arguments.seconds)); plt.close("all")
                pdf.savefig(closed_loop_trajectory_page(flights[ref], labels, ref, arguments.seconds)); plt.close("all")
    (output / ("closed-loop-summary-median.json" if median else "closed-loop-summary.json")).write_text(
        json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[closed-loop] wrote {pdf_path}", flush=True)


def closed_loop_title(arguments, labels, references) -> plt.Figure:
    from . import report_controller as rc
    gains = ", ".join(f"{name} = {np.array2string(value, separator=' ')}" for name, value in rc.GAINS.items())
    lines = [f"dataset (wind + vehicle)   {arguments.dataset.name}",
             f"references  the first {len(references)} recorded {arguments.split} flights as flown (position, world velocity,",
             "            absolute heading; feed-forward acceleration / yaw rate = 0.05 s-smoothed finite differences)",
             "start       the plant is reset to each flight's RECORDED x_0 (position, attitude, body velocity, body rate)",
             "plant       gym-pybullet-drones CF2P, Physics.PYB, 240 Hz (the dataset was simulated at 2000 Hz), contact-free,",
             "            linear damping c = 0.5 (as the dataset)",
             "wind        the dataset's white wind on the plant, unobserved by the controller;",
             f"            {arguments.seeds} wind seeds per model and flight (seed k = the SAME wind draw sequence for every model)",
             "controller  the SAME energy-based SE(3) law for every model (report_controller.EnergyController) on the model's",
             f"            posterior-mean operators; allocation u = pinv(M^-1 g)(M^-1 w); dissipation feed-forward = {arguments.dissipation}",
             f"            {gains}; tilt limit {np.degrees(rc.MAXIMUM_TILT):.0f} deg",
             "            Only the DRIFT enters the law: the diffusion does not, so the -wind-same variants are not flown.",
             f"duration    {arguments.seconds:g} s per flight", "",
             "pages       Table C (all flights), then per flight: metrics table, tracking grid, 3-D trajectory grid", "", "models"]
    for spec in arguments.model:
        lines.append(f"  {spec}")
    lines.append("  Analytical-SDE=the simulator's exact operators (the controller's ceiling, not ranked)")
    return text_page("Closed-loop tracking on the recorded test flights", lines)


# ---------------------------------------------------------------------------
# Nonstop horizons: one open-loop rollout per flight from t = 0, scored over 0-1, 0-3, 0-5, 0-10 s
# ---------------------------------------------------------------------------
HORIZON_MARKS = (1.0, 3.0, 5.0, 10.0)
BLOCK_NAMES = {"p": "position error |p - p_true| (m)", "R": "attitude error (deg)",
               "v": "body velocity error |v - v_true| (m/s)", "w": "body angular-velocity error (rad/s)"}


def error_series(truth: np.ndarray, mean: np.ndarray) -> dict[str, np.ndarray]:
    """(T, B) error of the mean path per block; a non-finite state counts as an infinite error (never dropped)."""
    out = {"p": np.linalg.norm(mean[..., :3] - truth[..., :3], axis=-1),
           "R": np.degrees(geodesic(truth[..., 3:12], project(np.nan_to_num(mean[..., 3:12], nan=0.0, posinf=0.0, neginf=0.0)))),
           "v": np.linalg.norm(mean[..., 12:15] - truth[..., 12:15], axis=-1),
           "w": np.linalg.norm(mean[..., 15:18] - truth[..., 15:18], axis=-1)}
    bad = ~np.isfinite(mean).all(axis=-1)
    return {k: np.where(bad, np.inf, np.nan_to_num(v, nan=np.inf)) for k, v in out.items()}


def horizons(arguments) -> None:
    flights, settings = load_dataset(arguments.dataset, arguments.split)
    entries = build_entries(arguments, settings)
    wind = load_wind(arguments.dataset, arguments.split, settings)
    steps = min(int(round(arguments.seconds / STEP)), flights.shape[1] - 1)
    truth = np.transpose(flights[:, :steps + 1], (1, 0, 2))                    # (T, B, 22): every flight from t = 0
    t = np.arange(steps + 1) * STEP
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for entry in entries:
        print(f"[horizons] {entry.label} ({entry.kind}): {truth.shape[1]} flights x {steps} steps nonstop", flush=True)
        mean = entry.mean_path(truth, None if wind is None else np.transpose(wind[:, 1:steps + 1], (1, 0, 2)))
        paths = entry.sample_paths(truth, arguments.samples, arguments.seed)
        centre = centre_line(mean, paths)
        results[entry.label] = {"mean": centre, "paths": paths, "errors": error_series(truth, centre),
                                "calib": calibration_stats(truth, paths, centre),
                                "so3": so3_violations(mean if paths is None else paths),
                                "calib_flight": calibration_per_flight(truth, paths, centre)}

    marks = [m for m in HORIZON_MARKS if m <= t[-1] + 1e-9]
    table = {}
    for entry in entries:
        errors = results[entry.label]["errors"]
        table[entry.label] = {}
        for block, series in errors.items():
            table[entry.label][block] = {}
            for mark in marks:
                count = int(round(mark / STEP))
                per_flight = series[1:count + 1].mean(axis=0)            # time-average over (0, mark]
                table[entry.label][block][f"0-{mark:g}s"] = {"median_over_flights": float(np.median(per_flight)),
                                                              "at_end_median": float(np.median(series[count]))}
    calib = {e.label: results[e.label]["calib"] for e in entries}
    table_a = {e.label: table_a_values(results[e.label], marks) for e in entries}

    pdf_path = output / "open-loop-horizons-comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        gauge_states = flights[:, :steps + 1].reshape(-1, 22)[::5]
        starts = np.arange(0, steps, max(1, steps // 26))
        gauge_pairs = np.transpose(np.stack([flights[:, starts], flights[:, starts + 1]], axis=0), (0, 1, 2, 3)).reshape(2, -1, 22)
        rows = gauge_table(entries, settings, gauge_states, STEP, gauge_pairs)
        pdf.savefig(gauge_page(rows, f"Physics recovery vs ground truth - {arguments.dataset.name} ({arguments.split} split)")); plt.close("all")
        for mark in marks:
            pdf.savefig(table_a_page(table_a, entries, mark, truth.shape[1], arguments)); plt.close("all")
        pdf.savefig(horizons_title(arguments, entries, truth.shape[1], steps)); plt.close("all")
        pdf.savefig(horizon_error_page(t, results, entries, marks)); plt.close("all")
        pdf.savefig(horizon_bar_page(table, entries, marks)); plt.close("all")
        pdf.savefig(calibration_figure(t, calib, entries, marks[:-1], f"  ({truth.shape[1]} flights, nonstop from t = 0)")); plt.close("all")
        for f in range(truth.shape[1]):
            per_model = []
            for entry in entries:
                r = results[entry.label]
                per_model.append([{"t": t, "mean": r["mean"][:, f],
                                   "paths": None if r["paths"] is None else r["paths"][:, :, f]}])
            flight = flights[f, :steps + 1]
            pdf.savefig(path3d_page(flight, per_model, entries, f)); plt.close("all")
            for block in ("p", "euler", "v", "w"):
                pdf.savefig(block_page(flight, per_model, entries, f, block, [0], marks=marks[:-1])); plt.close("all")
    summary = {"generated_at": datetime.now().astimezone().isoformat(), "dataset": str(arguments.dataset),
               "split": arguments.split, "flights": int(truth.shape[1]), "seconds": float(t[-1]), "samples": arguments.samples,
               "protocol": "one nonstop open-loop rollout per flight from t = 0, recorded wrench, no restarts, no filtering",
               "models": {e.label: {"kind": e.kind, "source": e.describe(), "band": BAND_TEXT[e.kind]} for e in entries},
               "error_table": table,
               "table_a": table_a,
               "gauge_table": rows,
               "calibration_at_marks": {k: {b: {name: {f"{m:g}s": float(x[int(round(m / STEP))]) for m in marks}
                                                 for name, x in v[b].items()} for b in v} for k, v in calib.items()}}
    (output / "open-loop-horizons-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[horizons] wrote {pdf_path}", flush=True)


GAUGE_COLUMNS = (("thrust", "M1^-1 g_f  z", lambda o: o[:, 2]), ("torque_xx", "M2^-1 g_tau xx", lambda o: o[:, 0, 0]),
                 ("torque_zz", "M2^-1 g_tau zz", lambda o: o[:, 2, 2]), ("gravity", "M1^-1 grad V  z", lambda o: o[:, 2]),
                 ("damping_v", "M1^-1 D_v  diag", lambda o: np.trace(o, axis1=1, axis2=2) / 3.0),
                 ("damping_w", "M2^-1 D_w  diag", lambda o: np.trace(o, axis1=1, axis2=2) / 3.0))
ERROR_KEYS = {"thrust": "thrust", "torque_xx": "torque", "torque_zz": "torque", "gravity": "gravity",
              "damping_v": "damping_v", "damping_w": "damping_w"}


def model_dtype(model) -> Any:
    """Float dtype of a model's array leaves (float64 when it has none, e.g. the analytical model)."""
    for leaf in jax.tree_util.tree_leaves(model):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.dtype
    return jnp.float64


def gauge_products(model, states: np.ndarray) -> dict[str, np.ndarray]:
    """Gauge-invariant products of one model on (N, 22) states: thrust gain M1^-1 g_f, torque gain M2^-1 g_tau,
    gravity M1^-1 grad V, and the damping products M1^-1 D_v, M2^-1 D_w (all identifiable from the data)."""
    dtype = model_dtype(model)

    def one(x):
        m1, m2 = model.inverse_mass_1(x[:3]), model.inverse_mass_2(x[3:12])
        control = model.control_matrix(x[:12])
        grad_v = jax.grad(model.potential)(x[:12])[:3]
        return (m1 @ control[:3, 0], m2 @ control[3:, 1:], m1 @ grad_v,
                m1 @ model.dissipation_v(x[12:15], x[:3]), m2 @ model.dissipation_w(x[15:18], x[3:12]))

    names = ("thrust", "torque", "gravity", "damping_v", "damping_w")
    return dict(zip(names, (np.asarray(v, dtype=np.float64) for v in jax.vmap(one)(jnp.asarray(states, dtype=dtype)))))


def true_products(settings: dict[str, Any], count: int) -> dict[str, np.ndarray]:
    vehicle = settings["vehicle_parameters"]
    mass, inertia = float(vehicle["mass"]), np.asarray(vehicle["inertia"], dtype=np.float64)
    c_v = float(settings.get("linear_damping_coefficient", 0.5))
    c_w = float(settings.get("angular_damping_coefficient", c_v))
    return {"thrust": np.tile([0.0, 0.0, 1.0 / mass], (count, 1)), "torque": np.tile(np.linalg.inv(inertia), (count, 1, 1)),
            "gravity": np.tile([0.0, 0.0, float(vehicle["gravity_acceleration"])], (count, 1)),
            "damping_v": np.tile(c_v * np.eye(3), (count, 1, 1)), "damping_w": np.tile(c_w * np.eye(3), (count, 1, 1))}


def gauge_table(entries, settings: dict[str, Any], states: np.ndarray, step_size: float, pairs: np.ndarray) -> dict[str, Any]:
    """Model value and relative error ||O_hat - O|| / ||O|| (mean over the states) of every gauge-invariant product,
    plus the twist diffusion sqrt(diag(J J^T) / h) of the model's own step (0 for ODE models)."""
    truth = true_products(settings, len(states))
    wind = wind_of(settings)
    rows = {"Ground truth": {"values": {key: float(np.mean(pick(truth[ERROR_KEYS[key]]))) for key, _, pick in GAUGE_COLUMNS},
                             "errors": None,
                             "sigma": None if wind is None or wind[1] or wind[3] else (wind[0], wind[2])}}
    time_major = jnp.asarray(pairs)                       # (2, P, 22): consecutive samples, control of row 1 drives the step
    for e in entries:
        ops = gauge_products(e.mean, states)
        errors = {}
        for key, _, _ in GAUGE_COLUMNS:
            learned, target = ops[ERROR_KEYS[key]], truth[ERROR_KEYS[key]]
            axes = tuple(range(1, learned.ndim))
            errors[key] = float(np.mean(np.sqrt(np.sum((learned - target) ** 2, axis=axes)) / np.sqrt(np.sum(target ** 2, axis=axes))))
        step_fn = nn_sde_step if e.kind == "nn_sde" else gp_sde_step if e.kind in ("gp_sde",) + ANALYTICAL_KINDS else None
        sigma = None
        if step_fn is not None:
            # the recorded-wind analytical model passes its noise through unchanged; its diffusion is the true one
            source = truth_model(settings) if e.kind == "analytical_same" else e.mean
            diffusion = np.asarray(transition_diffusion(source, time_major.astype(model_dtype(source)), jnp.asarray(step_size), step_fn))
            sigma = (float(np.mean(diffusion[:3])), float(np.mean(diffusion[3:])))
        rows[e.label + (" (recorded wind)" if getattr(e, "wind_same", False) else "")] = {
            "values": {key: float(np.mean(pick(ops[ERROR_KEYS[key]]))) for key, _, pick in GAUGE_COLUMNS},
            "errors": errors, "sigma": sigma}
    return rows


def gauge_page(rows: dict[str, Any], title: str) -> plt.Figure:
    """Page 1: every model's gauge-invariant products and diffusion next to the ground truth (value / relative error)."""
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(0.03, 0.95, title, fontsize=15, weight="bold")
    figure.text(0.03, 0.91, "Gauge-invariant products (mean over test states) and relative error ||O_hat - O|| / ||O|| per state, averaged; "
                "diffusion = sqrt(diag(J J^T)/h) of the model's own step, in twist units per sqrt(s).", fontsize=8.5)
    header = ["model"] + [label for _, label, _ in GAUGE_COLUMNS] + ["sigma_v (force)", "sigma_w (torque)"]
    truth = rows["Ground truth"]
    cells, colours = [], []
    for name, row in rows.items():
        values = [f"{row['values'][key]:.4g}" for key, _, _ in GAUGE_COLUMNS]
        sigma = row["sigma"]
        sig = ["-", "-"] if sigma is None else [f"{sigma[0]:.4g}", f"{sigma[1]:.4g}"]
        if row["errors"] is not None:
            values = [f"{v}\n({row['errors'][key] * 100:.2f} %)" for v, (key, _, _) in zip(values, GAUGE_COLUMNS)]
            if sigma is not None and truth["sigma"] is not None:
                sig = [f"{sig[i]}\n({abs(sigma[i] / truth['sigma'][i] - 1) * 100:.2f} %)" for i in range(2)]
        cells.append([name] + values + sig)
        colours.append(["#e8e8e8" if name == "Ground truth" else "white"] * len(header))
    axis = figure.add_axes([0.02, 0.05, 0.96, 0.82]); axis.axis("off")
    widths = [0.2] + [0.8 / (len(header) - 1)] * (len(header) - 1)
    table = axis.table(cellText=cells, colLabels=header, cellColours=colours, colWidths=widths, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8)
    table.scale(1.0, 2.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#d0d8e8")
    return figure


TABLE_A_ROWS = (("p", "mean", "Position mean error (m)"), ("p", "final", "Position final error (m)"),
                ("R", "mean", "Attitude mean error (deg)"), ("R", "final", "Attitude final error (deg)"),
                ("v", "mean", "Velocity mean error (m/s)"), ("v", "final", "Velocity final error (m/s)"),
                ("w", "mean", "Angular-velocity mean error (rad/s)"), ("w", "final", "Angular-velocity final error (rad/s)"),
                ("det", "max", "Determinant max error"), ("orth", "max", "Orthogonality max error"))
UQ_BLOCK_NAMES = {"p": "Position", "R": "Attitude", "v": "Velocity", "w": "Angular-velocity"}
UQ_UNITS = {"p": "m", "R": "deg", "v": "m/s", "w": "rad/s"}
TABLE_A_ROWS = TABLE_A_ROWS + tuple(
    row for block in ("p", "R", "v", "w") for row in (
        (block, "crps", f"{UQ_BLOCK_NAMES[block]} CRPS ({UQ_UNITS[block]})"),
        (block, "cover1", f"{UQ_BLOCK_NAMES[block]} coverage mean +- 1 sigma"),
        (block, "cover2", f"{UQ_BLOCK_NAMES[block]} coverage mean +- 2 sigma"),
        (block, "spread", f"{UQ_BLOCK_NAMES[block]} spread-skill RMSE / sigma")))
# how each kind of row is ranked for the highlight: lowest value, or closest to the ideal
ROW_TARGETS = {"cover1": 0.683, "cover2": 0.954, "spread": 1.0}


def calibration_per_flight(truth: np.ndarray, paths: np.ndarray | None, centre: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Per time step and per flight (T, B), averaged over each block's 3 axes: CRPS, coverage of mean +- 1 and 2 sigma,
    squared error of the sample mean and sample variance (for the spread-skill ratio). Attitude uses the rotation
    vector Log(R_bar^T R) about the projected mean rotation, in degrees. A model without samples only gets its CRPS,
    which for a single path is its absolute error per axis."""
    def tangent(rotation_flat, reference_flat):
        a = np.asarray(reference_flat, dtype=np.float64).reshape(-1, 3, 3)
        b = np.asarray(rotation_flat, dtype=np.float64).reshape(-1, 3, 3)
        b = np.broadcast_to(b, (max(len(a), len(b)), 3, 3)) if len(b) != len(a) else b
        with np.errstate(invalid="ignore", over="ignore"):
            relative = project_matrices(np.swapaxes(np.broadcast_to(a, b.shape), -1, -2) @ b)
        vectors = np.full(relative.shape[:-1], np.nan)            # a diverged path keeps a NaN tangent
        ok = np.all(np.isfinite(relative), axis=(-2, -1))
        if np.any(ok):
            vectors[ok] = np.degrees(Rotation.from_matrix(relative[ok]).as_rotvec())
        return vectors

    out = {}
    reference_R = centre[..., 3:12]                                               # (T, B, 9) projected mean attitude
    for block, cols in (("p", slice(0, 3)), ("R", None), ("v", slice(12, 15)), ("w", slice(15, 18))):
        if block == "R":
            y = tangent(truth[..., 3:12].reshape(-1, 9), reference_R.reshape(-1, 9)).reshape(*truth.shape[:-1], 3)
            x = None if paths is None else tangent(paths[..., 3:12].reshape(-1, 9),
                                                   np.broadcast_to(reference_R, paths[..., 3:12].shape).reshape(-1, 9)
                                                   ).reshape(*paths.shape[:-1], 3)
            centre_block = np.zeros_like(y)
        else:
            y, x = truth[..., cols], None if paths is None else paths[..., cols]
            centre_block = centre[..., cols]
        if x is None:
            out[block] = {"crps": np.mean(np.abs(centre_block - y), axis=-1)}
            continue
        with np.errstate(invalid="ignore"):
            mu, sigma = np.nanmean(x, axis=0), np.nanstd(x, axis=0, ddof=1)
            z = np.abs(y - mu) / np.maximum(sigma, 1e-12)
            out[block] = {"crps": np.mean(crps_ensemble(np.nan_to_num(x, nan=1e6), y), axis=-1),
                          "cover1": np.mean(z <= 1.0, axis=-1), "cover2": np.mean(z <= 2.0, axis=-1),
                          "sq_err": np.mean((y - mu) ** 2, axis=-1), "var": np.mean(sigma ** 2, axis=-1)}
    return out


def so3_violations(states: np.ndarray) -> dict[str, np.ndarray]:
    """|det R - 1| and ||R^T R - I||_F of the RAW integrator output (never the projected mean), per time step.
    states (T, B, 22) for one path per flight or (S, T, B, 22) for sample paths; returns (..., T, B) arrays."""
    rotation = np.asarray(states[..., 3:12], dtype=np.float64).reshape(*states.shape[:-1], 3, 3)
    determinant = np.abs(np.linalg.det(rotation) - 1.0)
    gram = np.swapaxes(rotation, -1, -2) @ rotation - np.eye(3)
    orthogonality = np.sqrt(np.sum(gram * gram, axis=(-2, -1)))
    return {"det": determinant, "orth": orthogonality}


def table_a_values(result: dict[str, Any], marks) -> dict[str, dict[str, list[float]]]:
    """Paper-style Table A (as in the SO(3) paper's Tables 1-2): per horizon, mean +- std over the flights of
    the time-averaged error over (0, H] ("mean error"), the error at H ("final error"), and the maximum SO(3)
    violation over (0, H] (averaged over the sample paths of a stochastic model)."""
    values: dict[str, dict[str, list[float]]] = {}
    for mark in marks:
        count = int(round(mark / STEP))
        row: dict[str, list[float]] = {}
        for block, series in result["errors"].items():                       # (T, B)
            for kind, per_flight in (("mean", series[1:count + 1].mean(axis=0)), ("final", series[count])):
                row[f"{block}_{kind}"] = [float(np.mean(per_flight)), float(np.std(per_flight))]
        for key, series in result["so3"].items():                            # (T, B) or (S, T, B)
            window_max = series[..., 1:count + 1, :].max(axis=-2)             # (B,) or (S, B)
            per_flight = window_max.mean(axis=0) if window_max.ndim == 2 else window_max
            row[f"{key}_max"] = [float(np.mean(per_flight)), float(np.std(per_flight))]
        for block, stats in result.get("calib_flight", {}).items():            # (T, B) per metric
            window = slice(1, count + 1)
            row[f"{block}_crps"] = [float(np.mean(stats["crps"][window].mean(axis=0))), float(np.std(stats["crps"][window].mean(axis=0)))]
            if "cover1" in stats:
                for name in ("cover1", "cover2"):
                    per_flight = stats[name][window].mean(axis=0)
                    row[f"{block}_{name}"] = [float(np.mean(per_flight)), float(np.std(per_flight))]
                per_flight = np.sqrt(stats["sq_err"][window].mean(axis=0) / np.maximum(stats["var"][window].mean(axis=0), 1e-24))
                row[f"{block}_spread"] = [float(np.mean(per_flight)), float(np.std(per_flight))]
            else:
                for name in ("cover1", "cover2", "spread"):
                    row[f"{block}_{name}"] = None
        values[f"0-{mark:g}s"] = row
    return values


def table_a_page(table_a, entries, mark, flights: int, arguments) -> plt.Figure:
    """One Table A page for the horizon 0-mark s: metrics as rows, models as columns, mean +- std over flights."""
    key = f"0-{mark:g}s"
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(0.03, 0.95, f"Table A - open-loop prediction error, 0-{mark:g} s (nonstop from t = 0)", fontsize=15, weight="bold")
    figure.text(0.03, 0.91, f"{arguments.dataset.name}, {arguments.split} split: mean +- std over {flights} flights. Mean error = time "
                "average over (0, H]; final error = at H. Prediction = mean of the sample paths (single path for -wind-same). "
                "SO(3) rows: max over (0, H] of |det R - 1| and ||R^T R - I||_F on the RAW rollout.", fontsize=7.5, wrap=True)
    header = ["metric"] + [e.label.replace("-wind-", "-\nwind-") for e in entries]
    # best value per row, highlighted separately among the LEARNED models driven by a new wind draw ("wind-different")
    # and among the learned models fed the recorded wind ("wind-same"); Analytical-SDE columns are the reference, not ranked
    same_group = [i for i, e in enumerate(entries) if getattr(e, "wind_same", False)]
    different_group = [i for i, e in enumerate(entries) if not getattr(e, "wind_same", False)
                       and getattr(e, "kind", "") not in ANALYTICAL_KINDS]
    best_cells = []
    cells = []
    for row_index, (block, kind, label) in enumerate(TABLE_A_ROWS):
        entries_values = [table_a[e.label][key].get(f"{block}_{kind}") for e in entries]
        target = ROW_TARGETS.get(kind)
        score = lambda i: (abs(entries_values[i][0] - target) if target is not None else entries_values[i][0])
        for group in (different_group, same_group):
            ranked = [i for i in group if entries_values[i] is not None]
            if len(ranked) > 1:
                best_cells.append((row_index + 1, 1 + min(ranked, key=score)))
        row = [label]
        for value in entries_values:
            if value is None:
                row.append("-")
            else:
                mean, std = value
                row.append(f"{mean:.3e} +- {std:.1e}" if block in ("det", "orth") else f"{mean:.4g} +- {std:.2g}")
        cells.append(row)
    axis = figure.add_axes([0.02, 0.05, 0.96, 0.82]); axis.axis("off")
    widths = [0.19] + [0.81 / len(entries)] * len(entries)
    table = axis.table(cellText=cells, colLabels=header, colWidths=widths, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(6.3); table.scale(1.0, 1.3)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#d0d8e8"); cell.set_height(cell.get_height() * 1.6)
        elif c == 0:
            cell.set_facecolor("#f2f2f2")
    for r, c in best_cells:
        cell = table[r, c]
        cell.set_facecolor("#c8ecc8"); cell.set_text_props(weight="bold")
    figure.text(0.03, 0.015, "Green bold = best per row among the learned models with a NEW wind draw, and separately among the learned models fed the RECORDED "
                "wind (-wind-same); Analytical-SDE columns are\nreferences, not ranked. Best = lowest, except coverage (closest to 0.683 / 0.954) and "
                "spread-skill (closest to 1). CRPS of a single path = its absolute error; '-' = no samples. Uncertainty rows: time average over (0, H].",
                fontsize=6.5)
    return figure


def horizons_title(arguments, entries, flights, steps) -> plt.Figure:
    lines = [f"dataset   {arguments.dataset.name}  split={arguments.split}  ({flights} flights)",
             f"protocol  ONE nonstop open-loop Lie-IMEX rollout per flight, from the true state at t = 0 to t = {steps * STEP:g} s",
             f"          (h = {STEP} s, driven by the recorded wrench, no restarts, no feedback). Scored over 0-1, 0-3, 0-5, 0-10 s.",
             "          Nothing is filtered: a diverged rollout keeps its (large or infinite) error, and medians over flights are shown.",
             f"line/band {arguments.samples} sample paths per model: line = their mean at every t, dark = mean +- 1 sigma,",
             "          light = mean +- 2 sigma (sigma = sample std at every t). Models without samples: their single path.", "", "models"]
    for e in entries:
        lines.append(f"  {e.label:7s} {e.kind:7s} {e.describe()}")
        lines.append(f"          samples: {BAND_TEXT[e.kind]}")
    lines += ["", "how to read",
              "  Analytical-SDE = every subnetwork set to the simulator's value. -wind-same is fed the data's own recorded wind and",
              "  must lie on the data; -wind-different uses another wind draw, so like any model it drifts away. Its path also",
              "  drifts away from them; its error curve is the irreducible open-loop error of the wind, and its band is the",
              "  spread a perfect SDE must show. Open loop in this wind the vehicle's attitude random-walks and the tilted",
              "  thrust drives position error roughly quadratically in time, so beyond ~1-2 s nobody can track the data."]
    return text_page("Open-loop horizons: NN-ODE / NN-SDE / GP-ODE / GP-SDE, nonstop from t = 0", lines)


def horizon_error_page(t, results, entries, marks) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    for axis, key in zip(axes.flat, ("p", "R", "v", "w")):
        for e in entries:
            series = results[e.label]["errors"][key]
            median = plottable(np.median(series, axis=1))
            axis.plot(t[1:], median[1:], color=COLORS[e.label], lw=2 if e.kind not in ANALYTICAL_KINDS else 1.5,
                      ls="--" if e.kind in ANALYTICAL_KINDS else "-", label=e.label)
            if e.kind not in ANALYTICAL_KINDS:
                axis.fill_between(t[1:], plottable(np.quantile(series, 0.25, axis=1)[1:]), plottable(np.quantile(series, 0.75, axis=1)[1:]),
                                  color=COLORS[e.label], alpha=0.12, lw=0)
        for mark in marks[:-1]:
            axis.axvline(mark, color="0.5", lw=0.8, ls="--")
        axis.set_yscale("log"); axis.set_title(BLOCK_NAMES[key]); axis.set_xlabel("t (s)"); axis.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Nonstop open-loop error of the prediction (mean of sample paths) vs time (median over flights, shade = inter-quartile; "
                    "dashed = Analytical-SDE variants)")
    figure.tight_layout()
    return figure


def horizon_bar_page(table, entries, marks) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    width = 0.8 / len(entries)
    for axis, key in zip(axes.flat, ("p", "R", "v", "w")):
        for k, e in enumerate(entries):
            values = plottable([table[e.label][key][f"0-{m:g}s"]["median_over_flights"] for m in marks])
            axis.bar(np.arange(len(marks)) + (k - (len(entries) - 1) / 2) * width, values, width, color=COLORS[e.label],
                     label=e.label, edgecolor="k", lw=0.3, hatch="//" if e.kind in ANALYTICAL_KINDS else None)
        axis.set_xticks(range(len(marks))); axis.set_xticklabels([f"0-{m:g} s" for m in marks])
        axis.set_yscale("log"); axis.set_title("mean " + BLOCK_NAMES[key] + " over the window"); axis.grid(alpha=0.3, axis="y", which="both")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Time-averaged error over 0-1 / 0-3 / 0-5 / 0-10 s (median over flights; hatched = Analytical-SDE variants)")
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", type=Path, required=True)
    common.add_argument("--split", default="heldout")
    common.add_argument("--output", type=Path, required=True)
    o = sub.add_parser("open-loop", parents=[common])
    o.add_argument("--model", action="append", default=[], help="LABEL=RUN_DIR@STEP")
    o.add_argument("--analytical", action="append", choices=sorted(ANALYTICAL_SPECS), default=None,
                   help="analytical reference models (repeatable): plain (zero-wind line + band), same (recorded wind), different")
    o.add_argument("--horizon", type=int, default=100)
    o.add_argument("--stride", type=int, default=50)
    o.add_argument("--samples", type=int, default=32)
    o.add_argument("--flights", type=int, default=4)
    o.add_argument("--seed", type=int, default=0)
    z = sub.add_parser("horizons", parents=[common])
    z.add_argument("--model", action="append", default=[], help="LABEL=RUN_DIR@STEP")
    z.add_argument("--analytical", action="append", choices=sorted(ANALYTICAL_SPECS), default=None)
    z.add_argument("--seconds", type=float, default=10.0)
    z.add_argument("--samples", type=int, default=32)
    z.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("closed-loop", parents=[common])
    c.add_argument("--model", action="append", required=True, help="LABEL=RUN_DIR@STEP")
    c.add_argument("--reference", action="append", default=[])
    c.add_argument("--recorded", type=int, default=0, help="also fly the first N recorded flights of --split")
    c.add_argument("--seeds", type=int, default=5)
    c.add_argument("--seconds", type=float, default=20.0)
    c.add_argument("--dissipation", choices=("on", "off"), default="on")
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--refly", action="store_true")
    c.add_argument("--report-only", action="store_true")
    c.add_argument("--statistic", choices=("mean", "median"), default="mean",
                   help="Table C over flights: mean +- std, or median [25th, 75th] -> closed_loop_comparison_median.pdf")
    f = sub.add_parser("fly", parents=[common])
    f.add_argument("--model-spec", required=True)
    f.add_argument("--reference", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--seconds", type=float, default=20.0)
    f.add_argument("--dissipation", choices=("on", "off"), default="on")
    arguments = parser.parse_args()
    global SPLIT_LABEL
    SPLIT_LABEL = {"heldout": "Held-out", "test": "Test", "train": "Train"}.get(getattr(arguments, "split", ""), "Held-out")
    if arguments.command == "horizons":
        horizons(arguments)
    elif arguments.command == "open-loop":
        open_loop(arguments)
    elif arguments.command == "fly":
        fly(arguments)
    elif arguments.report_only:
        closed_loop_report(arguments, [label for label, _ in parse_models(arguments.model)] + ["Analytical-SDE"])
    else:
        closed_loop(arguments)


if __name__ == "__main__":
    main()

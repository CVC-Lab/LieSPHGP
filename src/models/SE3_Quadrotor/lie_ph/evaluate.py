"""Ground-truth evaluation of a trained SE(3) quadrotor PH-GP-SDE, and the package self-test.

This is the ONLY file of the package that reads the simulator's true values (the pickle's settings and the clean
test flights). Training never imports it.

    python src/models/SE3_Quadrotor/lie_ph/evaluate.py --run experiments/quadrotor/train_runs/<run>
    python src/models/SE3_Quadrotor/lie_ph/evaluate.py --self-test --config <yaml>

Compared quantities (gauge-invariant, controls converted back to SI), true value in brackets:
    sigma_obs [noise level], Sigma per body axis [linear and angular wind sigma], M1^-1 D_v and M2^-1 D_w [c I],
    force map M1^-1 g_F [1/m on (z, thrust), 0 elsewhere], torque map M2^-1 g_tau [J^-1 on the torque columns],
    gravity acceleration -M1^-1 R^T grad V [-R^T (0, 0, g)], clean-test open-loop error of the posterior-mean model,
    2-sigma coverage of its stochastic band. The "analytical" model is this package's model class with every
    subnetwork set to the true operator (the floor of the model class).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lie_ph.config import load_config, resolve_project_path  # noqa: E402
from lie_ph.data import apply_control_scale, cut_windows, load_noisy_windows  # noqa: E402
from lie_ph.integrator import interval_step, log_so3, rollout  # noqa: E402
from lie_ph.losses import ekf_consistency, ekf_nll  # noqa: E402
from lie_ph.network import (GP_SPECS, SE3Model, build_gp_setup, initialize_parameters,  # noqa: E402
                                   model_from_params, sample_weights)


def is_real_dataset(dataset_path: Path) -> bool:
    """Real flights (e.g. IDSIA): no simulator truth of damping and diffusion in the settings."""
    with dataset_path.open("rb") as handle:
        settings = pickle.load(handle)["settings"]
    return settings.get("damping_law") not in ("linear", "nonlinear")


def real_operators(model, states: np.ndarray, settings: dict, scale: np.ndarray) -> dict:
    """Learned gauge-invariant operators on real flights, against the PUBLISHED vehicle constants where they exist
    (thrust gain 1/m, gravity g, torque gain J^-1; reference only, never given to training). Damping has no truth."""
    vehicle = settings["vehicle_parameters"]
    mass, gravity = float(vehicle["mass"]), float(vehicle["gravity_acceleration"])
    inertia = np.diag(np.asarray(vehicle["inertia"], float))
    states = jnp.asarray(states)
    linear, angular = jax.vmap(model.effective_damping)(states)

    def control(s):
        mapping = model.control_matrix(s[3:12], s[:3])
        return jnp.concatenate([model.inverse_mass_1(s[:3]) @ mapping[:3], model.inverse_mass_2(s[3:12]) @ mapping[3:]])

    learned_map = np.asarray(jax.vmap(control)(states)) / scale[None, None, :]
    learned_gravity = np.asarray(jax.vmap(lambda s: -model.inverse_mass_1(s[:3]) @ s[3:12].reshape(3, 3).T
                                          @ jax.grad(model.potential)(s[:3]))(states))
    published_gravity = -np.einsum("nji,j->ni", np.asarray(states[:, 3:12]).reshape(-1, 3, 3), np.asarray([0.0, 0.0, gravity]))
    torque_diag = np.mean(np.diagonal(learned_map[:, 3:, 1:], axis1=1, axis2=2), 0)
    return {
        "damping_linear_M1inv_Dv": {"mean_learned": np.mean(np.asarray(linear), 0).round(4).tolist(), "true": None},
        "damping_angular_M2inv_Dw": {"mean_learned": np.mean(np.asarray(angular), 0).round(4).tolist(), "true": None},
        "thrust_gain": {"relative_error_vs_published": float(np.mean(np.abs(learned_map[:, 2, 0] - 1.0 / mass)) * mass),
                        "mean_learned": float(np.mean(learned_map[:, 2, 0])), "published": 1.0 / mass},
        "gravity_acceleration": {"relative_error_vs_published": float(np.mean(np.linalg.norm(learned_gravity - published_gravity, axis=-1)) / gravity)},
        "torque_gain_diag": {"mean_learned": torque_diag.round(1).tolist(), "published": (1.0 / inertia).round(1).tolist(),
                             "ratio": (torque_diag * inertia).round(4).tolist(),
                             "note": "published J_zz 3.2347e-6 is probably a typo for 3.2347e-5 (benchmark paper)"},
    }


def evaluate_real_run(run: Path, which: str, paths: int, horizon_points: int, config, setup, params, payload,
                      build_model=model_from_params, rollout_fn=rollout) -> dict:
    """evaluate_run for real flights: no analytical model; test EKF NLL / NIS on the measured test flights, open loop
    against the measurements, operators against the published constants."""
    dataset = resolve_project_path(config["data"]["dataset_path"])
    substeps, data = int(config["data"]["substeps"]), config["data"]
    _, test_windows, interval, _ = load_noisy_windows(dataset, int(data["window_points"]), int(data["window_stride"]),
                                                      data["control_scaling"])
    scale = np.asarray(payload["control_scale"], np.float64)
    with dataset.open("rb") as handle:
        raw = pickle.load(handle)
    flights = apply_control_scale(np.asarray(raw["test_trajectories"], np.float32), scale.astype(np.float32))
    windows = cut_windows(flights, horizon_points, horizon_points // 2)
    states = flights.reshape(-1, 22)[::5]
    pick = test_windows[:, np.linspace(0, test_windows.shape[1] - 1, min(256, test_windows.shape[1])).astype(int)]
    chunks = [jnp.asarray(pick[:, i:i + 32]) for i in range(0, pick.shape[1], 32)]
    weights = np.asarray([c.shape[1] for c in chunks], float)
    factory = lambda p, key: build_model(p, setup, key, config["model"])
    entry = {"operators": real_operators(factory(params, None), states, raw["settings"], scale),
             "open_loop_test": open_loop(params, factory, windows, interval, substeps, paths, 0, rollout_fn)}
    if "likelihood" in params:
        nll = jax.jit(lambda p, w: ekf_nll(factory(p, None), w, p["likelihood"]["log_sigma"], interval, substeps))
        nis = jax.jit(lambda p, w: ekf_consistency(factory(p, None), w, p["likelihood"]["log_sigma"], interval, substeps))
        entry["sigma_obs"] = np.exp(np.asarray(params["likelihood"]["log_sigma"])).round(5).tolist()
        entry["test_ekf_nll_noisy"] = float(np.average([float(nll(params, c)) for c in chunks], weights=weights))
        entry["test_nis_per_dimension"] = float(np.average([float(nis(params, c)) for c in chunks], weights=weights))
    if "process" in params:
        entry["diffusion"] = np.exp(np.asarray(params["process"]["log_sigma"])).round(5).tolist()
    return {"run": str(run), "checkpoint": which, "step": int(payload["step"]), "dataset": str(dataset), "real_data": True,
            "control_scale": scale.tolist(),
            "model": {k: config["model"].get(k) for k in ("family", "wind", "integrator")} | {"loss": config["training"].get("loss", "ekf")},
            "learned": entry}


def load_truth(dataset_path: Path) -> dict:
    with dataset_path.open("rb") as handle:
        payload = pickle.load(handle)
    s = payload["settings"]
    vehicle = s["vehicle_parameters"]
    diffusion = s["diffusion_ground_truth"]
    if s["damping_law"] not in ("linear", "nonlinear") or float(diffusion.get("speed_gain", 0.0)) \
            or float(diffusion.get("rate_gain", 0.0)):
        raise NotImplementedError("the analytical model covers linear / rate-dependent damping and constant diffusion")
    inertia = np.diag(np.asarray(vehicle["inertia"], float))
    return {
        "settings": s, "damping_law": s["damping_law"], "mass": float(vehicle["mass"]), "inertia": inertia, "gravity": float(vehicle["gravity_acceleration"]),
        "c_linear": float(s["linear_damping_coefficient"]), "c_angular": float(s["angular_damping_coefficient"]),
        "diffusion": np.asarray([diffusion["linear_sigma"]] * 3 + [diffusion["angular_sigma"]] * 3, float),
        "obs_noise": float(s["observation_noise"].get("level", 0.0)) if s["observation_noise"].get("enabled") else 0.0,
        "clean_test_flights": np.asarray(payload["test_trajectories"], np.float32),
    }


class TrueSE3Model(SE3Model):
    """The GP model class with zero GP weights and the true levels; for rate-dependent ("nonlinear") damping the level
    gives c and the damping is multiplied by the true (1 + |v_b|) and (1 + |omega_b|)."""

    def __init__(self, weights, setup, law: str):
        super().__init__(weights, setup)
        self.law = law

    def dissipation_v(self, velocity):
        base = super().dissipation_v(velocity)
        return base * (1.0 + jnp.linalg.norm(velocity)) if self.law == "nonlinear" else base

    def dissipation_w(self, omega):
        base = super().dissipation_w(omega)
        return base * (1.0 + jnp.linalg.norm(omega)) if self.law == "nonlinear" else base


def analytical_model(params, setup, truth: dict) -> SE3Model:
    return TrueSE3Model(sample_weights(params, None), setup, truth["damping_law"])


def true_damping(states: np.ndarray, truth: dict) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3, 3) true M1^-1 D_v and M2^-1 D_w per state: c I, times (1 + |.|) for rate-dependent damping."""
    count = len(states)
    linear = np.broadcast_to(truth["c_linear"] * np.eye(3), (count, 3, 3)).copy()
    angular = np.broadcast_to(truth["c_angular"] * np.eye(3), (count, 3, 3)).copy()
    if truth["damping_law"] == "nonlinear":
        linear *= (1.0 + np.linalg.norm(states[:, 12:15], axis=-1))[:, None, None]
        angular *= (1.0 + np.linalg.norm(states[:, 15:18], axis=-1))[:, None, None]
    return linear, angular


def true_control_map(truth: dict) -> np.ndarray:
    """(6, 4) acceleration per SI control: force rows M1^-1 [e3 0], torque rows J^-1 [0 I]."""
    mapping = np.zeros((6, 4))
    mapping[2, 0] = 1.0 / truth["mass"]
    mapping[3:, 1:] = np.diag(1.0 / truth["inertia"])
    return mapping


def analytical_params(truth: dict, setup, model_cfg: dict, scale: np.ndarray) -> dict:
    """True operators in the GP model class (whatever the evaluated family); Sigma only when the data has wind."""
    model_cfg = {**model_cfg, "family": "gp", "wind": bool(np.any(truth["diffusion"] > 0)), "initial_randomness": 0.0}
    params = initialize_parameters(model_cfg, setup, jax.random.PRNGKey(0))
    for name in GP_SPECS:
        params[name]["mean"] = jnp.zeros_like(params[name]["mean"])
    m, inertia = truth["mass"], truth["inertia"]

    def spd(diagonal):
        level = np.zeros(6)
        level[:3] = 0.5 * np.log(diagonal)
        return jnp.asarray(level, jnp.float32)

    params["M1"]["level"] = jnp.asarray(np.log(1.0 / m), jnp.float32)
    params["M2"]["level"] = spd(1.0 / inertia)
    params["Dv"]["level"] = spd(np.full(3, truth["c_linear"] * m))                    # force -c m v
    params["Dw"]["level"] = spd(truth["c_angular"] * inertia)                           # torque -c J omega
    params["V"]["level"] = jnp.asarray([0.0, 0.0, m * truth["gravity"]], jnp.float32)  # V = m g z
    selection = np.zeros((6, 4))
    selection[2, 0] = 1.0
    selection[3:, 1:] = np.eye(3)
    params["g"]["level"] = jnp.asarray(selection * scale[None, :], jnp.float32)        # scaled controls
    if "process" in params:
        params["process"]["log_sigma"] = jnp.asarray(np.log(np.maximum(truth["diffusion"], 1e-6)), jnp.float32)
    params["likelihood"]["log_sigma"] = jnp.full((4,), np.log(max(truth["obs_noise"], 1e-6)), jnp.float32)
    return params


def operator_errors(model, states: np.ndarray, truth: dict, scale: np.ndarray) -> dict:
    states = jnp.asarray(states)
    linear, angular = jax.vmap(model.effective_damping)(states)

    def control(s):
        mapping = model.control_matrix(s[3:12], s[:3])
        return jnp.concatenate([model.inverse_mass_1(s[:3]) @ mapping[:3], model.inverse_mass_2(s[3:12]) @ mapping[3:]])

    learned_map = np.asarray(jax.vmap(control)(states)) / scale[None, None, :]          # back to SI controls
    gravity = np.asarray(jax.vmap(lambda s: -model.inverse_mass_1(s[:3]) @ s[3:12].reshape(3, 3).T
                                  @ jax.grad(model.potential)(s[:3]))(states))
    true_gravity = -np.einsum("nji,j->ni", np.asarray(states[:, 3:12]).reshape(-1, 3, 3),
                              np.asarray([0.0, 0.0, truth["gravity"]]))
    true_map = true_control_map(truth)
    true_linear, true_angular = true_damping(np.asarray(states), truth)

    def relative(learned, reference):
        reference = np.broadcast_to(reference, learned.shape)
        return float(np.mean(np.linalg.norm(learned - reference, axis=(-2, -1)))
                     / np.mean(np.linalg.norm(reference, axis=(-2, -1))))

    force, torque = learned_map[:, :3], learned_map[:, 3:]
    return {
        "damping_linear_M1inv_Dv": {"relative_error": relative(np.asarray(linear), true_linear),
                                    "mean_learned": np.mean(np.asarray(linear), 0).round(4).tolist(),
                                    "true": np.mean(true_linear, 0).round(4).tolist()},
        "damping_angular_M2inv_Dw": {"relative_error": relative(np.asarray(angular), true_angular),
                                     "mean_learned": np.mean(np.asarray(angular), 0).round(4).tolist(),
                                     "true": np.mean(true_angular, 0).round(4).tolist()},
        "thrust_gain": {"relative_error": float(np.mean(np.abs(force[:, 2, 0] - true_map[2, 0])) / true_map[2, 0]),
                        "mean_learned": float(np.mean(force[:, 2, 0])), "true": true_map[2, 0],
                        "force_leakage_rms": float(np.sqrt(np.mean(np.square(np.delete(force.reshape(-1, 12), 8, 1)))))},
        "torque_map": {"relative_error": relative(torque[:, :, 1:], true_map[3:, 1:]),
                       "mean_learned_diag": np.mean(np.diagonal(torque[:, :, 1:], axis1=1, axis2=2), 0).round(1).tolist(),
                       "true_diag": np.diag(true_map[3:, 1:]).round(1).tolist(),
                       "thrust_to_torque_leakage_rms": float(np.sqrt(np.mean(np.square(torque[:, :, 0]))))},
        "gravity_acceleration": {"relative_error": float(np.mean(np.linalg.norm(gravity - true_gravity, axis=-1))
                                                         / truth["gravity"])},
    }


def open_loop(params, factory, windows: np.ndarray, interval: float, substeps: int, paths: int, seed: int,
              rollout_fn=rollout) -> dict:
    """``factory(params, key)`` builds the model (key = GP weight sample; ignored by the NN family)."""
    windows = jnp.asarray(windows)
    controls = jnp.swapaxes(windows[1:, :, 18:22], 0, 1)
    steps = windows.shape[0] - 1
    mean_model = factory(params, None)
    zero = jnp.zeros((steps, substeps, 6), windows.dtype)
    mean_path = np.asarray(jax.jit(jax.vmap(lambda x0, u: rollout_fn(mean_model, x0, u, interval, zero)))(windows[0], controls))

    @jax.jit
    def sampled(key):
        weight_key, noise_key = jax.random.split(key)
        model = factory(params, weight_key)
        noise = jax.random.normal(noise_key, (windows.shape[1], steps, substeps, 6), windows.dtype)
        return jax.vmap(lambda x0, u, z: rollout_fn(model, x0, u, interval, z))(windows[0], controls, noise)

    samples = np.stack([np.asarray(sampled(jax.random.PRNGKey(seed + i))) for i in range(paths)])
    clean = np.swapaxes(np.asarray(windows), 0, 1)
    finite = np.all(np.isfinite(mean_path), axis=(1, 2))
    position_error = np.linalg.norm(mean_path[..., :3] - clean[..., :3], axis=-1)
    angle = np.asarray(jax.vmap(jax.vmap(lambda a, b: jnp.linalg.norm(log_so3(a.reshape(3, 3).T @ b.reshape(3, 3)))))(
        jnp.asarray(np.nan_to_num(mean_path[..., 3:12])), jnp.asarray(clean[..., 3:12])))
    band_mean, band_std = samples[..., :3].mean(0), samples[..., :3].std(0)
    inside = np.all(np.abs(clean[..., :3] - band_mean) <= 2.0 * band_std + 1e-9, axis=-1)
    result = {"diverged_windows": int(np.sum(~finite))}
    for h in sorted({min(h, steps) for h in (10, 50, 100, steps)}):
        result[f"t={h * interval:.2f}s"] = {
            "position_rmse_m": float(np.sqrt(np.nanmean(position_error[finite, h] ** 2))),
            "attitude_rmse_rad": float(np.sqrt(np.nanmean(angle[finite, h] ** 2))),
            "position_2sigma_coverage": float(np.mean(inside[finite, h]))}
    return result


def load_run(run: Path, which: str):
    checkpoint = run / f"checkpoint_{which}.pkl" if run.is_dir() else run
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    config = payload["config"]
    key_setup, _, _ = jax.random.split(jax.random.PRNGKey(int(config["training"]["seed"])), 3)
    setup = build_gp_setup(config["model"], key_setup)
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    return config, setup, params, payload


def evaluate_run(run: Path, which: str, paths: int, horizon_points: int, build_model=model_from_params,
                 rollout_fn=rollout) -> dict:
    """``build_model(params, setup, key, model_config)`` and ``rollout_fn`` of the learned model (default lie_ph's;
    ph_node/evaluate.py passes its own). The analytical reference always uses lie_ph's model and Lie-IMEX."""
    config, setup, params, payload = load_run(run, which)
    if config.get("runtime", {}).get("matmul_precision"):       # evaluate with the precision the run trained with
        jax.config.update("jax_default_matmul_precision", str(config["runtime"]["matmul_precision"]))
    dataset = resolve_project_path(config["data"]["dataset_path"])
    if is_real_dataset(dataset):                       # real flights (6 Oct 2026): no simulator truth
        return evaluate_real_run(run, which, paths, horizon_points, config, setup, params, payload, build_model, rollout_fn)
    substeps = int(config["data"]["substeps"])
    data = config["data"]
    _, noisy_test, interval, scale = load_noisy_windows(dataset, int(data["window_points"]), int(data["window_stride"]),
                                                        data["control_scaling"])
    scale = np.asarray(payload["control_scale"], np.float64)
    truth = load_truth(dataset)
    clean_flights = apply_control_scale(truth["clean_test_flights"], scale.astype(np.float32))
    clean_windows = cut_windows(clean_flights, horizon_points, horizon_points // 2)
    states = clean_flights.reshape(-1, 22)[::5]
    noisy_eval = noisy_test[:, np.linspace(0, noisy_test.shape[1] - 1, min(256, noisy_test.shape[1])).astype(int)]
    chunks = [jnp.asarray(noisy_eval[:, i:i + 32]) for i in range(0, noisy_eval.shape[1], 32)]
    weights = np.asarray([c.shape[1] for c in chunks], float)
    analytical = analytical_params(truth, setup, config["model"], scale)
    factories = {"learned": lambda p, key: build_model(p, setup, key, config["model"]),
                 "analytical": lambda p, key: analytical_model(p, setup, truth)}
    report = {"run": str(run), "checkpoint": which, "step": int(payload["step"]), "dataset": str(dataset),
              "control_scale": scale.tolist(),
              "model": {k: config["model"].get(k) for k in ("family", "wind", "integrator")}
                       | {"loss": config["training"].get("loss", "ekf")}}
    for label, candidate in (("learned", params), ("analytical", analytical)):
        factory = factories[label]
        entry = {"operators": operator_errors(factory(candidate, None), states, truth, scale),
                 "open_loop_clean_test": open_loop(candidate, factory, clean_windows, interval, substeps, paths, 0,
                                                   rollout_fn if label == "learned" else rollout)}
        if "likelihood" in candidate:          # EKF-trained models (and the analytical model)
            nll = jax.jit(lambda p, w, f=factory: ekf_nll(f(p, None), w, p["likelihood"]["log_sigma"], interval, substeps))
            nis = jax.jit(lambda p, w, f=factory: ekf_consistency(f(p, None), w, p["likelihood"]["log_sigma"], interval, substeps))
            entry["sigma_obs"] = np.exp(np.asarray(candidate["likelihood"]["log_sigma"])).round(4).tolist()
            entry["test_ekf_nll_noisy"] = float(np.average([float(nll(candidate, c)) for c in chunks], weights=weights))
            # filter consistency on the noisy test windows: mean r^T S^-1 r / d, 1 = consistent
            entry["test_nis_per_dimension"] = float(np.average([float(nis(candidate, c)) for c in chunks], weights=weights))
        if "process" in candidate:
            entry["diffusion"] = np.exp(np.asarray(candidate["process"]["log_sigma"])).round(4).tolist()
        report[label] = entry
    report["truth"] = {"sigma_obs": truth["obs_noise"], "diffusion": truth["diffusion"].tolist(),
                       "damping_rate": [truth["c_linear"], truth["c_angular"]], "damping_law": truth["damping_law"], "thrust_gain": 1.0 / truth["mass"],
                       "torque_gain_diag": (1.0 / truth["inertia"]).round(1).tolist(), "gravity": truth["gravity"]}
    return report


def self_test(config_path: Path) -> dict:
    jax.config.update("jax_enable_x64", True)
    config = load_config(config_path)
    data = config["data"]
    dataset = resolve_project_path(data["dataset_path"])
    substeps = int(data["substeps"])
    noisy_train, _, interval, scale = load_noisy_windows(dataset, int(data["window_points"]), int(data["window_stride"]),
                                                         data["control_scaling"])
    truth = load_truth(dataset)
    setup = build_gp_setup(config["model"], jax.random.PRNGKey(0))
    params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float64),
                                    analytical_params(truth, setup, config["model"], np.asarray(scale, float)))
    model = analytical_model(params, setup, truth)
    clean = apply_control_scale(truth["clean_test_flights"], scale).astype(np.float64)       # (F, T, 22)
    step = jax.jit(jax.vmap(lambda s, u: interval_step(model, s, u, interval, jnp.zeros((substeps, 6)))))
    result: dict = {}

    def residual_std(row_offset):
        states = jnp.asarray(clean[:, :-1].reshape(-1, 22))
        controls = jnp.asarray(clean[:, row_offset:clean.shape[1] - 1 + row_offset, 18:22].reshape(-1, 4))
        predicted = np.asarray(step(states, controls))
        target = clean[:, 1:].reshape(-1, 22)
        return (predicted[:, 12:18] - target[:, 12:18]).std(0) / np.sqrt(interval)

    result["twist_residual_per_sqrt_s_control_row_k_plus_1"] = residual_std(1).round(4).tolist()
    result["twist_residual_per_sqrt_s_control_row_k"] = residual_std(0).round(4).tolist()
    result["true_diffusion"] = truth["diffusion"].tolist()

    windows = jnp.asarray(noisy_train[:, np.linspace(0, noisy_train.shape[1] - 1, 128).astype(int)].astype(np.float64))
    nll = jax.jit(lambda p: ekf_nll(analytical_model(p, setup, truth), windows, p["likelihood"]["log_sigma"],
                                    interval, substeps))

    def changed(group, leaf, new_value):
        candidate = jax.tree_util.tree_map(lambda v: v, params)
        candidate[group] = dict(candidate[group])
        candidate[group][leaf] = jnp.asarray(new_value, jnp.float64)
        return candidate

    four = np.asarray([0.5 * np.log(4.0)] * 3 + [0.0] * 3)
    variants = {
        "analytical": params,
        "linear_damping_x4": changed("Dv", "level", params["Dv"]["level"] + four),
        "angular_damping_x4": changed("Dw", "level", params["Dw"]["level"] + four),
        "control_x0.9": changed("g", "level", params["g"]["level"] * 0.9),
        "gravity_x0.9": changed("V", "level", params["V"]["level"] * 0.9),
        "diffusion_x2": changed("process", "log_sigma", params["process"]["log_sigma"] + np.log(2.0)),
        "sigma_obs_x0.5": changed("likelihood", "log_sigma", params["likelihood"]["log_sigma"] + np.log(0.5)),
        "untrained_start": jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float64), initialize_parameters(
            {**config["model"], "family": "gp", "wind": True}, setup, jax.random.PRNGKey(1))),
    }
    result["ekf_nll_noisy_train"] = {name: float(nll(candidate)) for name, candidate in variants.items()}
    return result


def main(build_model=model_from_params, rollout_fn=rollout) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--checkpoint", default="final", choices=("final", "best"),
                        help="final = the fixed-budget result; best exists only in runs made before 3 Oct 2026")
    parser.add_argument("--paths", type=int, default=16)
    parser.add_argument("--horizon-points", type=int, default=201, help="open-loop window length (samples)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.self_test:
        report = self_test(args.config)
    else:
        report = evaluate_run(args.run, args.checkpoint, args.paths, args.horizon_points, build_model, rollout_fn)
        out = (args.run if args.run.is_dir() else args.run.parent) / f"evaluation_{args.checkpoint}.json"
        out.write_text(json.dumps(report, indent=1))
        print(f"wrote {out}")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

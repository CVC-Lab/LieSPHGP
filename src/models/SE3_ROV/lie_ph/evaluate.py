"""Evaluation of a trained BlueROV2 PH-GP-SDE, and the package self-test.

This is the ONLY file of the package that reads a dataset's settings (the simulator's ground truth). Training never
imports it. On real data (no ground truth) it reports the learned gauge-invariant products only.

    python src/models/SE3_ROV/lie_ph/evaluate.py --run experiments/rov_se3/<...>/<run>
    python src/models/SE3_ROV/lie_ph/evaluate.py --self-test --config <yaml on a simulator dataset>

Products (controls converted back to the dataset's units):
    control map     M^-1 G                     (6 x n_u): per-axis acceleration per unit input
    damping         M1^-1 D_v, M2^-1 D_w       (3 x 3, mean over the test states)
    restoring       -M^-1 [R^T dV/dx ; sum_i r_i x dV/dr_i]   (6,) generalised hydrostatic acceleration
    anisotropy      eigenvalues of M1^-1 divided by their mean (1, 1, 1 = isotropic)
    noise           Sigma (twist diffusion), sigma_obs; test EKF NLL / d and NIS / d on the test windows
Self-test truth (simulator only): the model class with zero GP weights and the published operators (M from the added
mass, D = D_L + D_Q |nu| per axis, V = (B - W) z + B e3^T R r_b, G = E), see envs/rov_se3_marinarium/bluerov2.py.
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
from lie_ph.data import apply_control_scale, load_noisy_windows  # noqa: E402
from lie_ph.integrator import interval_step  # noqa: E402
from lie_ph.losses import ekf_consistency, ekf_nll  # noqa: E402
from lie_ph.network import SUBNETWORKS, SE3Model, build_gp_setup, initialize_parameters, model_from_params, sample_weights  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# simulator ground truth
# ---------------------------------------------------------------------------------------------------------------
def load_truth(dataset_path: Path) -> dict | None:
    """The simulator's operators, or None for real data (no settings['ground_truth'])."""
    with dataset_path.open("rb") as handle:
        payload = pickle.load(handle)
    s = payload["settings"]
    if "ground_truth" not in s:
        return None
    gt = s["ground_truth"]
    mode = s.get("input_mode", "thrusters")
    if mode not in ("thrusters", "wrench"):
        raise NotImplementedError("the analytical model needs a linear input map (thrusters or wrench)")
    diffusion = gt["diffusion"]
    if any(diffusion[c]["type"] not in ("none", "constant") for c in ("linear", "angular")):
        raise NotImplementedError("the analytical model covers constant (or no) diffusion only")
    sigma = [0.0 if diffusion[c]["type"] == "none" else float(diffusion[c]["sigma"]) for c in ("linear", "angular")]
    noise = s.get("observation_noise", {})
    return {"M": np.asarray(gt["M_diag"], float), "D_L": np.asarray(gt["damping"]["D_linear_diag"], float),
            "D_Q": np.asarray(gt["damping"]["D_quadratic_diag"], float), "weight": float(gt["weight"]),
            "buoyancy": float(gt["buoyancy"]), "r_b": np.asarray(gt["center_of_buoyancy"], float),
            "G": np.asarray(gt["G"], float) if mode == "thrusters" else np.eye(6),
            "diffusion": np.asarray([sigma[0]] * 3 + [sigma[1]] * 3), "obs_noise": float(noise.get("level", 0.0) or 0.0),
            "clean_test": np.asarray(payload["test_trajectories"], np.float32)}


class TrueROVModel(SE3Model):
    """The model class with zero GP weights and the true levels; the damping D_L + D_Q |nu| (per axis, force units)
    replaces the learned SPD dissipation, since the quadratic part has no constant level."""

    def __init__(self, weights, setup, truth):
        super().__init__(weights, setup)
        self.d_linear, self.d_quadratic = jnp.asarray(truth["D_L"]), jnp.asarray(truth["D_Q"])

    def dissipation_v(self, velocity):
        return jnp.diag(self.d_linear[:3] + self.d_quadratic[:3] * jnp.abs(velocity)).astype(velocity.dtype)

    def dissipation_w(self, omega):
        return jnp.diag(self.d_linear[3:] + self.d_quadratic[3:] * jnp.abs(omega)).astype(omega.dtype)


def analytical_params(truth: dict, setup, model_cfg: dict, scale: np.ndarray) -> dict:
    model_cfg = {**model_cfg, "family": "gp", "wind": bool(np.any(truth["diffusion"] > 0)), "initial_randomness": 0.0}
    if not (model_cfg["M1_form"] == "spd" and bool(model_cfg["V_rotation"])):
        raise ValueError("the BlueROV2 truth needs model.M1_form spd and model.V_rotation true")
    params = initialize_parameters(model_cfg, setup, jax.random.PRNGKey(0))
    for name in SUBNETWORKS:
        params[name]["mean"] = jnp.zeros_like(params[name]["mean"])

    def spd(diagonal):
        return jnp.asarray(np.r_[0.5 * np.log(diagonal), np.zeros(3)], jnp.float32)

    params["M1"]["level"] = spd(1.0 / truth["M"][:3])
    params["M2"]["level"] = spd(1.0 / truth["M"][3:])
    level = np.zeros(12)
    level[2] = truth["buoyancy"] - truth["weight"]                     # (B - W) z, z down
    level[3 + 6:3 + 9] = truth["buoyancy"] * truth["r_b"]              # B e3^T R r_b = B sum_j R[2, j] r_b[j]
    params["V"]["level"] = jnp.asarray(level, jnp.float32)
    params["g"]["level"] = jnp.asarray(truth["G"] * scale[None, :], jnp.float32)    # scaled controls
    if "process" in params:
        params["process"]["log_sigma"] = jnp.asarray(np.log(np.maximum(truth["diffusion"], 1e-6)), jnp.float32)
    params["likelihood"]["log_sigma"] = jnp.full((4,), np.log(max(truth["obs_noise"], 1e-6)), jnp.float32)
    return params


def analytical_model(params, setup, truth: dict) -> SE3Model:
    return TrueROVModel(sample_weights(params, None), setup, truth)


# ---------------------------------------------------------------------------------------------------------------
# learned products
# ---------------------------------------------------------------------------------------------------------------
def products(model, states: np.ndarray, scale: np.ndarray) -> dict:
    """Per-state gauge-invariant products, controls in the dataset's units."""
    states = jnp.asarray(states)

    def one(s):
        inverse = jax.scipy.linalg.block_diag(model.inverse_mass_1(s[:3]), model.inverse_mass_2(s[3:12]))
        rotation = s[3:12].reshape(3, 3)
        d_dx = jax.grad(model.potential, argnums=0)(s[:3], s[3:12])
        d_dr = jax.grad(model.potential, argnums=1)(s[:3], s[3:12]).reshape(3, 3)
        restoring = jnp.concatenate([-rotation.T @ d_dx, jnp.cross(rotation, d_dr, axis=-1).sum(axis=0)])
        linear, angular = model.effective_damping(s)
        m1 = model.inverse_mass_1(s[:3])
        return (inverse @ model.control_matrix(s[3:12], s[:3]), linear, angular, inverse @ restoring,
                jnp.linalg.eigvalsh(m1) / jnp.mean(jnp.linalg.eigvalsh(m1)))

    control, linear, angular, restoring, anisotropy = (np.asarray(v, np.float64) for v in jax.jit(jax.vmap(one))(states))
    return {"control": control / scale[None, None, :], "damping_linear": linear, "damping_angular": angular,
            "restoring": restoring, "anisotropy": anisotropy}


def true_products(truth: dict, states: np.ndarray) -> dict:
    m_inv = 1.0 / truth["M"]
    nu = states[:, 12:18]
    damping = (truth["D_L"] + truth["D_Q"] * np.abs(nu)) * m_inv
    rotation = states[:, 3:12].reshape(-1, 3, 3)
    down = np.einsum("nji,j->ni", rotation, np.array([0.0, 0.0, 1.0]))                 # R^T e3 (body frame)
    force = (truth["weight"] - truth["buoyancy"]) * down
    torque = np.cross(truth["r_b"], -truth["buoyancy"] * down)
    return {"control": np.broadcast_to(m_inv[:, None] * truth["G"], (len(states),) + truth["G"].shape),
            "damping_linear": np.stack([np.diag(d) for d in damping[:, :3]]),
            "damping_angular": np.stack([np.diag(d) for d in damping[:, 3:]]),
            "restoring": m_inv * np.c_[force, torque],
            "anisotropy": np.broadcast_to(m_inv[:3] / np.mean(m_inv[:3]), (len(states), 3))}


def relative_error(learned: np.ndarray, reference: np.ndarray) -> float:
    axes = tuple(range(1, learned.ndim))
    return float(np.mean(np.sqrt(np.sum((learned - reference) ** 2, axis=axes)))
                 / max(np.mean(np.sqrt(np.sum(reference ** 2, axis=axes))), 1e-12))


def summarize(values: dict) -> dict:
    return {"control_mean": np.mean(values["control"], 0).round(5).tolist(),
            "control_row_norm": np.mean(np.linalg.norm(values["control"], axis=-1), 0).round(5).tolist(),
            "damping_linear_mean": np.mean(values["damping_linear"], 0).round(4).tolist(),
            "damping_angular_mean": np.mean(values["damping_angular"], 0).round(4).tolist(),
            "restoring_mean": np.mean(values["restoring"], 0).round(5).tolist(),
            "restoring_rms": np.sqrt(np.mean(values["restoring"] ** 2, 0)).round(5).tolist(),
            "anisotropy_mean": np.mean(values["anisotropy"], 0).round(4).tolist()}


def load_run(run: Path, which: str):
    checkpoint = run / f"checkpoint_{which}.pkl" if run.is_dir() else run
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    config = payload["config"]
    key_setup, _, _ = jax.random.split(jax.random.PRNGKey(int(config["training"]["seed"])), 3)
    setup = build_gp_setup(config["model"], key_setup)
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    return config, setup, params, payload


def evaluate_run(run: Path, which: str, build_model=model_from_params) -> dict:
    """``build_model(params, setup, key, model_config)``: the learned model's builder (ph_node/evaluate.py passes its own)."""
    config, setup, params, payload = load_run(run, which)
    data = config["data"]
    dataset = resolve_project_path(data["dataset_path"])
    substeps = int(data["substeps"])
    _, test_windows, interval, _ = load_noisy_windows(dataset, int(data["window_points"]), int(data["window_stride"]),
                                                      data["control_scaling"], data["train_key"], data["test_key"])
    scale = np.asarray(payload["control_scale"], np.float64)
    states = test_windows.reshape(-1, test_windows.shape[-1])[::7].astype(np.float32)
    model = build_model(params, setup, None, config["model"])
    learned = products(model, states, scale)
    report = {"run": str(run), "checkpoint": which, "step": int(payload["step"]), "dataset": str(dataset),
              "control_scale": scale.tolist(), "model": {k: config["model"].get(k) for k in
                                                       ("family", "wind", "integrator", "M1_form", "V_rotation", "control_dim")}
              | {"loss": config["training"].get("loss", "ekf")}, "learned": summarize(learned)}
    if "likelihood" in params:
        chunks = [jnp.asarray(test_windows[:, i:i + 32]) for i in range(0, test_windows.shape[1], 32)]
        weights = np.asarray([c.shape[1] for c in chunks], float)
        nll = jax.jit(lambda w: ekf_nll(model, w, params["likelihood"]["log_sigma"], interval, substeps))
        nis = jax.jit(lambda w: ekf_consistency(model, w, params["likelihood"]["log_sigma"], interval, substeps))
        report["learned"]["sigma_obs"] = np.exp(np.asarray(params["likelihood"]["log_sigma"])).round(5).tolist()
        report["learned"]["test_ekf_nll"] = float(np.average([float(nll(c)) for c in chunks], weights=weights))
        report["learned"]["test_nis_per_dimension"] = float(np.average([float(nis(c)) for c in chunks], weights=weights))
    if "process" in params:
        report["learned"]["diffusion"] = np.exp(np.asarray(params["process"]["log_sigma"])).round(5).tolist()
    truth = load_truth(dataset)
    if truth is not None:
        expected = true_products(truth, states.astype(np.float64))
        report["truth"] = summarize(expected) | {"diffusion": truth["diffusion"].tolist(), "obs_noise": truth["obs_noise"]}
        report["relative_error"] = {name: relative_error(learned[name], expected[name]) for name in expected}
    return report


# ---------------------------------------------------------------------------------------------------------------
# self-test (simulator datasets)
# ---------------------------------------------------------------------------------------------------------------
def self_test(config_path: Path) -> dict:
    jax.config.update("jax_enable_x64", True)
    config = load_config(config_path)
    data = config["data"]
    dataset = resolve_project_path(data["dataset_path"])
    substeps = int(data["substeps"])
    noisy_train, _, interval, scale = load_noisy_windows(dataset, int(data["window_points"]), int(data["window_stride"]),
                                                         data["control_scaling"], data["train_key"], data["test_key"])
    truth = load_truth(dataset)
    if truth is None:
        raise ValueError("the self-test needs a simulator dataset (settings['ground_truth'])")
    setup = build_gp_setup(config["model"], jax.random.PRNGKey(0))
    params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float64),
                                    analytical_params(truth, setup, config["model"], np.asarray(scale, float)))
    model = analytical_model(params, setup, truth)
    clean = apply_control_scale(truth["clean_test"], scale).astype(np.float64)
    width = clean.shape[-1]
    step = jax.jit(jax.vmap(lambda s, u: interval_step(model, s, u, interval, jnp.zeros((substeps, 6)))))
    result: dict = {}

    def residual_std(row_offset):
        states = jnp.asarray(clean[:, :-1].reshape(-1, width))
        controls = jnp.asarray(clean[:, row_offset:clean.shape[1] - 1 + row_offset, 18:].reshape(-1, width - 18))
        predicted = np.asarray(step(states, controls))
        target = clean[:, 1:].reshape(-1, width)
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

    two = np.asarray([0.5 * np.log(2.0)] * 3 + [0.0] * 3)
    variants = {
        "analytical": params,
        "inverse_mass_v_x2": changed("M1", "level", params["M1"]["level"] + two),
        "inverse_mass_w_x2": changed("M2", "level", params["M2"]["level"] + two),
        "control_x0.9": changed("g", "level", params["g"]["level"] * 0.9),
        "restoring_x0.5": changed("V", "level", params["V"]["level"] * 0.5),
        "sigma_obs_x0.5": changed("likelihood", "log_sigma", params["likelihood"]["log_sigma"] + np.log(0.5)),
        "untrained_start": jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float64), initialize_parameters(
            {**config["model"], "family": "gp", "wind": "process" in params}, setup, jax.random.PRNGKey(1))),
    }
    if "process" in params:
        variants["diffusion_x2"] = changed("process", "log_sigma", params["process"]["log_sigma"] + np.log(2.0))
    result["ekf_nll_noisy_train"] = {name: float(nll(candidate)) for name, candidate in variants.items()}
    return result


def main(build_model=model_from_params) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--checkpoint", default="final", choices=("final",))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.self_test:
        report = self_test(args.config)
    else:
        report = evaluate_run(args.run, args.checkpoint, build_model)
        out = (args.run if args.run.is_dir() else args.run.parent) / f"evaluation_{args.checkpoint}.json"
        out.write_text(json.dumps(report, indent=1))
        print(f"wrote {out}")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

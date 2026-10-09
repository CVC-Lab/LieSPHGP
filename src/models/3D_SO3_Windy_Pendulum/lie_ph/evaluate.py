"""Ground-truth evaluation of a trained SO(3) pendulum Lie-PH model, and the package self-test.

This is the ONLY file of the package that reads the simulator's true values (the pickle's settings, the clean
test split, the env). Training never imports it.

    # after training (best checkpoint by default)
    python src/models/3D_SO3_Windy_Pendulum/lie_ph/evaluate.py --run experiments/pendulum_so3/train_runs/<run>
    # self-test of the model class, integrator, control alignment and EKF against the env (no run needed)
    python src/models/3D_SO3_Windy_Pendulum/lie_ph/evaluate.py --self-test --config <yaml>

Compared quantities (gauge-invariant, so the learned M^-1 scale does not matter), true value in brackets:
    sigma_obs [obs_noise_std], Sigma per body axis [sigma_wind / (m l) on x, y; 0 on z, the rod axis],
    M^-1 D [c / (m l^2) I], M^-1 g [diag(g_diag) / (m l^2)], gravity acceleration M^-1 sum r_i x dV/dr_i [env],
    clean-test open-loop error of the posterior-mean model and 2-sigma coverage of its stochastic band.
The "analytical" model is this package's model class with every subnetwork set to the true operator.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent, PROJECT_ROOT / "envs" / "pendulum_so3"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lie_ph.config import load_config, resolve_project_path  # noqa: E402
from lie_ph.data import _time_major, load_noisy_windows  # noqa: E402
from lie_ph.integrator import interval_step, log_so3, rollout  # noqa: E402
from lie_ph.losses import ekf_consistency, ekf_nll  # noqa: E402
from lie_ph.network import GP_SPECS, SO3Model, build_gp_setup, initialize_parameters, model_from_params, sample_weights  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------------------------------------------
def load_truth(dataset_path: Path) -> dict:
    with dataset_path.open("rb") as handle:
        payload = pickle.load(handle)
    s = payload["settings"]
    law = s.get("friction_law") or ("varying" if s.get("varying_friction") else "constant")
    if float(s.get("external_force_std", 0.0)) != 0.0:
        raise NotImplementedError("the analytical model covers no external force")
    m, l = float(s["m"]), float(s["l"])
    inertia = m * l * l
    return {
        "settings": s, "m": m, "l": l, "g": float(s["g"]), "inertia": inertia, "friction_law": law,
        "friction": np.broadcast_to(np.asarray(s["friction_coeff"], float), (3,)).copy(),
        "g_diag": np.asarray(s["g_diag"], float), "wind": float(s["wind_force_std"]),
        "obs_noise": float(s["obs_noise_std"]),
        "diffusion": np.asarray([s["wind_force_std"] / (m * l)] * 2 + [0.0]),
        "clean_test": _time_major(payload["test_x"]),
    }


def varying_factor(rotation_flat, omega, xp=jnp):
    """The env's varying friction factor 1 + 0.5 h + 0.5 tanh|omega| with the height term h = (1 - R_33) / 2."""
    return 1.0 + 0.25 * (1.0 - rotation_flat[..., 8]) + 0.5 * xp.tanh(xp.linalg.norm(omega, axis=-1))


class TrueSO3Model(SO3Model):
    """The GP model class with zero GP weights and the true levels; the level gives c and the damping is multiplied
    by the true factor of the law: 1 (constant), 1 + |omega| (rate_dependent), varying_factor (varying)."""

    def __init__(self, weights, setup, law: str):
        super().__init__(weights, setup)
        self.law = law

    def dissipation(self, omega, rotation_flat=None):
        base = super().dissipation(omega, rotation_flat)
        if self.law == "rate_dependent":
            return base * (1.0 + jnp.linalg.norm(omega))
        if self.law == "varying":
            return base * varying_factor(rotation_flat, omega)
        return base


def analytical_model(params, setup, truth: dict, key=None) -> SO3Model:
    return TrueSO3Model(sample_weights(params, key), setup, truth["friction_law"])


def analytical_params(truth: dict, setup, model_cfg: dict) -> dict:
    """True operators in the GP model class: GP weights zero, levels = truth (diffusion 0 on z -> 1e-6). Built as a GP
    model whatever the evaluated family; Sigma only when the data has wind."""
    model_cfg = {**model_cfg, "family": "gp", "wind": truth["wind"] > 0, "initial_randomness": 0.0}
    params = initialize_parameters(model_cfg, setup, jax.random.PRNGKey(0))
    for name in GP_SPECS:
        params[name]["mean"] = jnp.zeros_like(params[name]["mean"])
    m_level = np.zeros(6)
    m_level[:3] = 0.5 * np.log(1.0 / truth["inertia"])
    d_level = np.zeros(6)
    d_level[:3] = 0.5 * np.log(truth["friction"])
    v_level = np.zeros(9)
    v_level[8] = truth["m"] * truth["g"] * truth["l"]                # V = m g l R[2, 2] (bob height l (R e_z).e_z)
    params["M"]["level"] = jnp.asarray(m_level, jnp.float32)
    params["D"]["level"] = jnp.asarray(d_level, jnp.float32)
    params["V"]["level"] = jnp.asarray(v_level, jnp.float32)
    params["g"]["level"] = jnp.asarray(np.diag(truth["g_diag"]), jnp.float32)
    if "process" in params:
        params["process"]["log_sigma"] = jnp.asarray(np.log(np.maximum(truth["diffusion"], 1e-6)), jnp.float32)
    params["likelihood"]["log_sigma"] = jnp.full((2,), np.log(max(truth["obs_noise"], 1e-6)), jnp.float32)
    return params


def make_env(truth: dict):
    from windy_pendulum_3d import windy_pendulum_3d
    s = truth["settings"]
    return windy_pendulum_3d(g=truth["g"], m=truth["m"], l=truth["l"], dt=float(s["dt"]),
                             friction_coeff=tuple(truth["friction"]), varying_friction=False, external_force_std=0.0,
                             friction_law=truth["friction_law"],
                             wind_force_std=0.0, g_diag=tuple(truth["g_diag"]), integrator=s["integrator"],
                             substeps=int(s["substeps"]))


def true_gravity_acceleration(env, rotation: np.ndarray) -> np.ndarray:
    """Body angular acceleration from gravity alone (omega = 0, u = 0, no friction at omega = 0)."""
    rate, _ = env._compute_omega_rates(rotation, np.zeros(3), 0.0, np.zeros(3), np.zeros(3), 0.0)
    return rate


# ---------------------------------------------------------------------------------------------------------------
# learned operators
# ---------------------------------------------------------------------------------------------------------------
def operator_errors(model, states: np.ndarray, truth: dict, env) -> dict:
    states = jnp.asarray(states)
    damping = jax.vmap(model.effective_damping)(states)
    control = jax.vmap(lambda s: model.inverse_mass(s[:9]) @ model.control_matrix(s[:9]))(states)

    def gravity(s):
        zero = s.at[9:15].set(0.0)
        return model.vector_field(zero)[1] + model.effective_damping(zero) @ zero[9:12]   # damping vanishes at omega 0

    learned_gravity = np.asarray(jax.vmap(gravity)(states))
    true_gravity = np.stack([true_gravity_acceleration(env, np.asarray(s[:9]).reshape(3, 3)) for s in np.asarray(states)])
    true_damping = np.diag(truth["friction"] / truth["inertia"])
    if truth["friction_law"] == "rate_dependent":                 # c (1 + |omega|) / J, state by state
        speed = np.linalg.norm(np.asarray(states)[:, 9:12], axis=-1)
        true_damping = true_damping[None] * (1.0 + speed)[:, None, None]
    elif truth["friction_law"] == "varying":                      # c (1 + 0.5 h + 0.5 tanh|omega|) / J, state by state
        true_damping = true_damping[None] * varying_factor(np.asarray(states)[:, :9], np.asarray(states)[:, 9:12], np)[:, None, None]
    true_control = np.diag(truth["g_diag"] / truth["inertia"])

    def relative(learned, reference):
        reference = np.broadcast_to(reference, learned.shape)
        return float(np.mean(np.linalg.norm(learned - reference, axis=(-2, -1)))
                     / np.mean(np.linalg.norm(reference, axis=(-2, -1))))

    return {
        "damping_M_inv_D": {"relative_error": relative(np.asarray(damping), true_damping),
                            "mean_learned": np.mean(np.asarray(damping), axis=0).round(4).tolist(),
                            "true": np.mean(np.broadcast_to(true_damping, np.asarray(damping).shape), axis=0).round(4).tolist()},
        "control_M_inv_g": {"relative_error": relative(np.asarray(control), true_control),
                            "mean_learned": np.mean(np.asarray(control), axis=0).round(4).tolist(),
                            "true": true_control.tolist()},
        "gravity_acceleration": {"relative_error": float(np.mean(np.linalg.norm(learned_gravity - true_gravity, axis=-1))
                                                         / np.mean(np.linalg.norm(true_gravity, axis=-1)))},
    }


def open_loop(params, factory, windows: np.ndarray, interval: float, substeps: int, paths: int, seed: int,
              rollout_fn=rollout) -> dict:
    """Mean-model (no noise) rollouts from the clean x0 with the recorded controls, and a sampled band.
    ``factory(params, key)`` builds the model (key = GP weight sample; ignored by the NN family); ``rollout_fn`` is the
    integrator's rollout (default: Lie-IMEX)."""
    windows = jnp.asarray(windows)
    controls = jnp.swapaxes(windows[1:, :, 12:15], 0, 1)                          # (B, T-1, 3)
    steps = windows.shape[0] - 1
    mean_model = factory(params, None)
    zero = jnp.zeros((steps, substeps, 3), windows.dtype)
    mean_path = jax.jit(jax.vmap(lambda x0, u: rollout_fn(mean_model, x0, u, interval, zero)))(windows[0], controls)

    @jax.jit
    def sampled(key):
        weight_key, noise_key = jax.random.split(key)
        model = factory(params, weight_key)
        noise = jax.random.normal(noise_key, (windows.shape[1], steps, substeps, 3), windows.dtype)
        return jax.vmap(lambda x0, u, z: rollout_fn(model, x0, u, interval, z))(windows[0], controls, noise)

    samples = np.stack([np.asarray(sampled(jax.random.PRNGKey(seed + i))) for i in range(paths)])   # (P, B, T, 15)
    clean = np.swapaxes(np.asarray(windows), 0, 1)                                                 # (B, T, 15)
    mean_path = np.asarray(mean_path)
    omega_error = np.linalg.norm(mean_path[..., 9:12] - clean[..., 9:12], axis=-1)
    angle = np.asarray(jax.vmap(jax.vmap(lambda a, b: jnp.linalg.norm(log_so3(a.reshape(3, 3).T @ b.reshape(3, 3)))))(
        jnp.asarray(mean_path[..., :9]), jnp.asarray(clean[..., :9])))
    band_mean, band_std = samples[..., 9:12].mean(0), samples[..., 9:12].std(0)
    inside = np.all(np.abs(clean[..., 9:12] - band_mean) <= 2.0 * band_std + 1e-9, axis=-1)
    horizons = sorted({min(h, steps) for h in (1, 5, 10, 20, 40, steps)})
    return {f"t={h * interval:.2f}s": {"omega_rmse": float(np.sqrt(np.mean(omega_error[:, h] ** 2))),
                                        "attitude_rmse_rad": float(np.sqrt(np.mean(angle[:, h] ** 2))),
                                        "omega_2sigma_coverage": float(np.mean(inside[:, h]))} for h in horizons}


def load_run(run: Path, which: str):
    checkpoint = run / f"checkpoint_{which}.pkl" if run.is_dir() else run
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    config = payload["config"]
    key_setup, _, _ = jax.random.split(jax.random.PRNGKey(int(config["training"]["seed"])), 3)
    setup = build_gp_setup(config["model"], key_setup)
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    return config, setup, params, payload


def evaluate_run(run: Path, which: str, paths: int) -> dict:
    config, setup, params, payload = load_run(run, which)
    dataset = resolve_project_path(config["data"]["dataset_path"])
    substeps = int(config["data"]["substeps"])
    _, noisy_test, interval = load_noisy_windows(dataset, config["data"].get("window_points"),
                                                 config["data"].get("window_stride"))
    truth = load_truth(dataset)
    env = make_env(truth)
    clean = truth["clean_test"]
    states = clean[:, :, :].reshape(-1, 15)[:: max(1, clean.shape[0] * clean.shape[1] // 2000)]
    analytical = analytical_params(truth, setup, config["model"])
    factories = {"learned": lambda p, key: model_from_params(p, setup, key, config["model"]),
                 "analytical": lambda p, key: analytical_model(p, setup, truth, key)}
    nlls = {label: jax.jit(lambda p, w, f=f: ekf_nll(f(p, None), w, p["likelihood"]["log_sigma"], interval, substeps))
            for label, f in factories.items()}
    niss = {label: jax.jit(lambda p, w, f=f: ekf_consistency(f(p, None), w, p["likelihood"]["log_sigma"], interval, substeps))
            for label, f in factories.items()}
    chunks = [jnp.asarray(noisy_test[:, i:i + 128]) for i in range(0, noisy_test.shape[1], 128)]
    weights = np.asarray([c.shape[1] for c in chunks], float)
    report = {"run": str(run), "checkpoint": which, "step": int(payload["step"]), "dataset": str(dataset)}
    report["model"] = {k: config["model"].get(k) for k in ("family", "wind")} | {"integrator": "lie_imex",
                                                                                 "loss": config["training"].get("loss", "ekf")}
    for label, candidate in (("learned", params), ("analytical", analytical)):
        factory = factories[label]
        entry = {"operators": operator_errors(factory(candidate, None), states, truth, env),
                 # the band is only meaningful for models with wind; for the ODE variants it has zero width
                 "open_loop_clean_test": open_loop(candidate, factory, clean, interval, substeps, paths, 0)}
        if "likelihood" in candidate:          # EKF-trained models (and the analytical model)
            entry["sigma_obs"] = np.exp(np.asarray(candidate["likelihood"]["log_sigma"])).round(4).tolist()
            entry["test_ekf_nll_noisy"] = float(np.average([float(nlls[label](candidate, c)) for c in chunks], weights=weights))
            # filter consistency on the noisy test windows: mean r^T S^-1 r / d, 1 = consistent
            entry["test_nis_per_dimension"] = float(np.average([float(niss[label](candidate, c)) for c in chunks], weights=weights))
        if "process" in candidate:
            entry["diffusion"] = np.exp(np.asarray(candidate["process"]["log_sigma"])).round(4).tolist()
        report[label] = entry
    report["truth"] = {"sigma_obs": truth["obs_noise"], "diffusion": truth["diffusion"].tolist(),
                       "damping_rate": (truth["friction"] / truth["inertia"]).tolist(), "friction_law": truth["friction_law"],
                       "control_gain": (truth["g_diag"] / truth["inertia"]).tolist()}
    return report


# ---------------------------------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------------------------------
def self_test(config_path: Path) -> dict:
    jax.config.update("jax_enable_x64", True)
    config = load_config(config_path)
    dataset = resolve_project_path(config["data"]["dataset_path"])
    substeps = int(config["data"]["substeps"])
    noisy_train, _, interval = load_noisy_windows(dataset, config["data"]["window_points"], config["data"]["window_stride"])
    truth = load_truth(dataset)
    env = make_env(truth)
    env_substeps = int(truth["settings"]["substeps"])            # the simulator's own substeps (the data's accuracy)
    setup = build_gp_setup(config["model"], jax.random.PRNGKey(0))
    params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float64), analytical_params(truth, setup, config["model"]))
    model = analytical_model(params, setup, truth)
    clean = truth["clean_test"].astype(np.float64)                                    # (T, N, 15)
    result: dict = {}

    # 1. deterministic interval step: model class with true operators (config substeps) vs the env's own integrator
    rng = np.random.default_rng(0)
    picks = [(int(rng.integers(0, clean.shape[0] - 1)), int(rng.integers(0, clean.shape[1]))) for _ in range(200)]
    step = jax.jit(lambda s, u: interval_step(model, s, u, interval, jnp.zeros((substeps, 3))))
    worst = 0.0
    for k, n in picks:
        state, control = clean[k, n], clean[k + 1, n, 12:15]
        env.R, env.omega = state[:9].reshape(3, 3).copy(), state[9:12].copy()
        for _ in range(env_substeps):
            env.R, env.omega = env._lie_imex_step(env.R, env.omega, 0.0, control, interval / env_substeps, 0.0, np.zeros(3))
        mine = np.asarray(step(jnp.asarray(state), jnp.asarray(control)))
        worst = max(worst, float(np.max(np.abs(mine[:12] - np.concatenate([env.R.reshape(9), env.omega])))))
    result[f"max_abs_diff_model_{substeps}_substeps_vs_env_{env_substeps}_substeps"] = worst

    # 2. control alignment and diffusion: one-interval residual of the clean data under the true drift,
    #    using the control of row k+1 (dataset convention) or of row k (the old trainer's choice)
    def residual_std(row_offset):
        states = jnp.asarray(clean[:-1].reshape(-1, 15))
        controls = jnp.asarray(clean[row_offset:clean.shape[0] - 1 + row_offset, :, 12:15].reshape(-1, 3))
        predicted = jax.vmap(step)(states, controls)
        return np.asarray(predicted[:, 9:12] - clean[1:].reshape(-1, 15)[:, 9:12]).std(0) / np.sqrt(interval)

    result["omega_residual_per_sqrt_s_control_row_k_plus_1"] = residual_std(1).round(4).tolist()
    result["omega_residual_per_sqrt_s_control_row_k"] = residual_std(0).round(4).tolist()
    result["true_diffusion"] = truth["diffusion"].tolist()

    # 3. EKF NLL on noisy training windows: true operators vs perturbed ones vs the untrained start
    windows = jnp.asarray(noisy_train[:, :256].astype(np.float64))
    nll = jax.jit(lambda p: ekf_nll(analytical_model(p, setup, truth), windows, p["likelihood"]["log_sigma"],
                                    interval, substeps))
    variants = {"analytical": params}

    def changed(path, factor):
        candidate = jax.tree_util.tree_map(lambda v: v, params)
        group, leaf = path
        candidate[group] = dict(candidate[group])
        candidate[group][leaf] = candidate[group][leaf] + factor
        return candidate

    variants["damping_x4"] = changed(("D", "level"), jnp.asarray([0.5 * np.log(4.0)] * 3 + [0.0] * 3))
    variants["control_x0.7"] = changed(("g", "level"), params["g"]["level"] * (0.7 - 1.0))
    if "process" in params:
        variants["diffusion_x2"] = changed(("process", "log_sigma"), np.log(2.0))
    variants["sigma_obs_x0.5"] = changed(("likelihood", "log_sigma"), np.log(0.5))
    variants["gravity_x0.8"] = changed(("V", "level"), params["V"]["level"] * (0.8 - 1.0))
    gp_start = {**config["model"], "family": "gp", "wind": truth["wind"] > 0}       # a GP-class start, far from the truth
    start = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float64),
                                   initialize_parameters(gp_start, setup, jax.random.PRNGKey(1)))
    variants["untrained_start"] = start
    result["ekf_nll_noisy_train"] = {name: float(nll(candidate)) for name, candidate in variants.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, nargs="+", help="run directories (or checkpoint files), evaluated one after another")
    parser.add_argument("--checkpoint", default="final", choices=("final", "best"),
                        help="final = the fixed-budget result; best exists only in runs made before 3 Oct 2026")
    parser.add_argument("--paths", type=int, default=32, help="sampled paths for the band coverage")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path, help="config for --self-test")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.config), indent=1))
        return
    for run in args.run:                       # several runs in one process: the data and env are loaded per run
        try:
            report = evaluate_run(run, args.checkpoint, args.paths)
        except Exception as error:             # one broken run (e.g. non-finite parameters) must not stop the rest
            print(f"FAILED {run}: {type(error).__name__}: {error}", flush=True)
            continue
        out = (run if run.is_dir() else run.parent) / f"evaluation_{args.checkpoint}.json"
        out.write_text(json.dumps(report, indent=1))
        print(f"wrote {out}", flush=True)



if __name__ == "__main__":
    main()

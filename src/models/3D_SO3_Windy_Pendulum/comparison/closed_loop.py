"""Closed-loop report of the pendulum campaign (4 Oct 2026): one PDF per dataset, written next to the open-loop
report in <campaign>/reports/<setting>_noise<lvl>/.

    # swing-up and stabilisation to UPRIGHT from 5 starts (the user's choice, 4 Oct 2026): closed-loop-upright.pdf
    JAX_PLATFORMS=cpu python src/models/3D_SO3_Windy_Pendulum/comparison/closed_loop.py --task upright \
        --setting DampRate-Wind --noise 0.75
    # tracking of the 10 recorded evaluation trajectories: closed-loop-comparison.pdf
    ... closed_loop.py --task track ...

Task upright: R_d = I (bob straight up, l R e_z = l e_z, yaw 0), w_d = 0, from the x_0 of the first 5 long_test
trajectories (the data's own random attitudes and rates); e_R = Log(R_d^T R) (geodesic, non-zero up to pi rad; the
vee error below vanishes at pi rad), |u_i| <= 25 so that G u can exceed m g l on every axis.

Design follows the quadrotor closed-loop report (src/models/SE3_Quadrotor/comparison/closed_loop.py): every learned model drives the SAME energy-shaping tracking law through its
gauge-invariant operators, on the TRUE plant (the env with the dataset's settings and its own wind seeds).

Controller (geometric SO(3) tracking with full matching, i.e. IDA-PBC to the target energy
H_d = 1/2 |e_w|^2 + K_R Psi(R, R_d) with damping injection K_w):
    f(x) = domega/dt of the model at u = 0 (gravity, damping, gyroscopic terms), B(x) = M^-1(R) g(R)
    e_R = 1/2 vee(R_d^T R - R^T R_d),  e_w = w - R^T R_d w_d
    a   = -K_R e_R - K_w e_w - w x (R^T R_d w_d) + R^T R_d dw_d/dt          (then de_w/dt = -K_R e_R - K_w e_w)
    u   = clip(lstsq(B, a - f), -U_MAX, U_MAX)
Only the identifiable products M^-1 g and the full drift enter, so the learned gauge c cannot change the gains.
The learned diffusion is not used (a controller only uses the drift), hence no -wind-same models here.

Experiment: references = the dataset's 10 clean long_test trajectories (R_d, w_d over 0-10 s; dw_d/dt = central
difference of the 20 Hz samples smoothed over 3 samples), interpolated to the control rate (geodesic for R, linear for
w); each flight starts from the reference's x_0. Plant: windy_pendulum_3d with the dataset's m, l, g, G, damping
law and wind, Lie-IMEX, control at 100 Hz (dt 0.01 s, 2 substeps = the data's internal step 0.005 s), zero-order
hold. SEEDS wind seeds per trajectory (seed = SEED_BASE + 100 trajectory + k), identical for every model; a
wind-free dataset needs one. A flight fails when its state or control becomes non-finite.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
for _path in (PROJECT_ROOT, THIS_DIR.parent, PROJECT_ROOT / "envs" / "pendulum_so3"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from comparison import report as rp  # noqa: E402
from comparison.campaign import BASE_CONFIG, MODELS, RESULT_DIR, SETTINGS, run_name  # noqa: E402
from comparison.models import load_run, model_factory  # noqa: E402
from lie_ph.evaluate import analytical_model, analytical_params  # noqa: E402
from lie_ph.integrator import log_so3  # noqa: E402
from lie_ph.network import build_gp_setup  # noqa: E402

TASK_TITLE = {"track": "tracking", "upright": "swing-up to upright"}
TASKS = {  # name: settings read by the controller, the metrics and the pages
    "track": {"u_max": 10.0, "trajectories": 10, "error": "vee", "file": "closed-loop-comparison"},
    "upright": {"u_max": 25.0, "trajectories": 5, "error": "log", "file": "closed-loop-upright"},
}
UPRIGHT_TOLERANCE = math.radians(10.0)   # rad (= 10 deg): "at upright"; every angle in the reports is in rad (user, 6 Oct 2026)
SUCCESS_SHARE = 0.2                # success = within the tolerance over the last 20 % of the window (2 s of 0-10 s)
CONTROL_DT = 0.01
PLANT_SUBSTEPS = 2
DURATION = 10.0
K_R, K_W = 25.0, 10.0
U_MAX = 10.0
SEEDS = 5
SEED_BASE = 70000
SMOOTH_SAMPLES = 3
VALID_FRACTION = 0.158             # as the quadrotor: first time |bob - bob_Analytical| > 0.158 x RMS reference excursion
HORIZON_MARKS = (1.0, 3.0, 5.0, 10.0)
ANALYTICAL = "Analytical"
LEARNED = ("Lie-PH-GP-SDE", "Lie-PH-NN-SDE", "Lie-PH-GP-ODE", "Lie-PH-NN-ODE", "PH-NODE")
COLORS = {**rp.COLORS, ANALYTICAL: "tab:green"}
REF_COLOR = "chocolate"
TRACK_ROWS = (("att_rmse", "Attitude RMSE (rad)", "low"), ("att_max", "Attitude max error (rad)", "low"),
        ("att_final", "Attitude final error (rad)", "low"), ("bob_rmse", "Bob position RMSE (m)", "low"),
        ("bob_max", "Bob position max error (m)", "low"), ("bob_final", "Bob position final error (m)", "low"),
        ("w_rmse", "Angular-velocity error RMSE |e_w| (rad/s)", "low"), ("H_rmse", "Energy error RMSE (J, true H)", "low"),
        ("valid", "Valid tracking time vs Analytical (s)", "high"), ("u_rms", "RMS control |u|", "gt"),
        ("chatter", "Control chattering (1/s)", "low"), ("sat", "Saturation fraction (|u_i| > U_max)", "low"),
        ("fail", "Failures (failed / flown)", "none"))
UPRIGHT_ROWS = (("att_rmse", "Angle to upright RMSE (rad)", "low"), ("att_final", "Angle to upright at H (rad)", "low"),
                ("t_up", f"Time to upright: stays within {UPRIGHT_TOLERANCE:.3f} rad (s; = H if never)", "low"),
                ("success", f"Success: within {UPRIGHT_TOLERANCE:.3f} rad over the last {SUCCESS_SHARE:.0%} of the window", "high"),
                ("bob_rmse", "Bob distance from the top RMSE (m)", "low"), ("w_rmse", "Angular-velocity RMSE |w| (rad/s)", "low"),
                ("H_rmse", "Energy error RMSE |H - m g l| (J)", "low"), ("u_rms", "RMS control |u|", "gt"),
                ("chatter", "Control chattering (1/s)", "low"), ("sat", "Saturation fraction (|u_i| > U_max)", "low"),
                ("fail", "Failures (failed / flown)", "none"))
ROWS = TRACK_ROWS
TASK = TASKS["track"]


def set_task(name: str) -> None:
    """Select the task: rows, control limit, trajectory count and error map used by everything below."""
    global ROWS, TASK, U_MAX
    TASK = {**TASKS[name], "name": name}
    ROWS = UPRIGHT_ROWS if name == "upright" else TRACK_ROWS
    U_MAX = TASK["u_max"]


# ---------------------------------------------------------------------------------------------------------------
# models and controller
# ---------------------------------------------------------------------------------------------------------------
def load_sources(data: rp.Dataset, records: dict, learned: tuple = LEARNED) -> dict:
    """{source: context or None}: posterior-mean model of every learned model trained on this dataset + the analytical
    model (same structure as report.predict_entries, so report.physics_page can reuse it)."""
    sources = {}
    for model in learned:
        if MODELS[model][1] and not SETTINGS[data.setting][1]:
            continue
        record = records.get(run_name(model, data.setting, data.level))
        run_dir = Path(record["run_dir"]) if record and record.get("run_dir") else None
        if record is None or record.get("status") != "done" or not (run_dir / "checkpoint_final.pkl").exists():
            sources[model] = None
            continue
        config, setup, params, _ = load_run(run_dir, "final")
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(params)):
            sources[model] = None
            continue
        sources[model] = {"params": params, "config": config, "run_dir": run_dir,
                          "mean_model": model_factory(config, setup)(params, None)}
    config = yaml.safe_load(BASE_CONFIG.read_text())
    setup = build_gp_setup(config["model"], jax.random.PRNGKey(0))
    params = analytical_params(data.truth, setup, config["model"])
    sources["analytical"] = {"params": params, "config": config, "run_dir": None,
                             "mean_model": analytical_model(params, setup, data.truth, None)}
    return sources


def make_controller(model):
    """Batched controller: (F, 12) states, (F, 3, 3) R_d, (F, 3) w_d, (F, 3) dw_d -> (clipped u, requested u)."""
    def one(state, rd, wd, wdd):
        rotation, omega = state[:9].reshape(3, 3), state[9:12]
        drift = model.vector_field(jnp.concatenate([state, jnp.zeros(3, state.dtype)]))[1]       # u = 0
        gain = model.inverse_mass(state[:9]) @ model.control_matrix(state[:9])                   # M^-1 g
        error = rd.T @ rotation
        if TASK["error"] == "log":
            e_r = log_so3(error)                                   # geodesic: non-zero up to pi rad
        else:
            e_r = 0.5 * jnp.stack([error[2, 1] - error[1, 2], error[0, 2] - error[2, 0], error[1, 0] - error[0, 1]])
        relative = rotation.T @ rd
        target = relative @ wd
        e_w = omega - target
        accel = -K_R * e_r - K_W * e_w - jnp.cross(omega, target) + relative @ wdd
        u = jnp.linalg.lstsq(gain, accel - drift, rcond=1e-6)[0]
        return jnp.clip(u, -U_MAX, U_MAX), u
    return jax.jit(jax.vmap(one))


# ---------------------------------------------------------------------------------------------------------------
# references and flights
# ---------------------------------------------------------------------------------------------------------------
def upright_references(data: rp.Dataset) -> dict:
    """Constant upright target R_d = I, w_d = 0 for the first TASK trajectories; starts = their recorded x_0."""
    count = TASK["trajectories"]
    steps = int(round(DURATION / CONTROL_DT))
    start = data.clean[0, :count].astype(np.float64)
    return {"R": np.broadcast_to(np.eye(3), (steps + 1, count, 3, 3)).copy(), "w": np.zeros((steps + 1, count, 3)),
            "dw": np.zeros((steps + 1, count, 3)), "t": np.arange(steps + 1) * CONTROL_DT,
            "R0": start[:, :9].reshape(count, 3, 3), "w0": start[:, 9:12]}


def references(data: rp.Dataset) -> dict:
    """Reference at the control times for every evaluation trajectory: R_d (N+1, B, 3, 3), w_d, dw_d (N+1, B, 3)."""
    sample_dt = data.interval
    count = int(round(DURATION / sample_dt))                                   # 200 intervals
    rotation = data.clean[:count + 1, :, :9].astype(np.float64).reshape(count + 1, -1, 3, 3)
    omega = data.clean[:count + 1, :, 9:12].astype(np.float64)
    domega = uniform_filter1d(np.gradient(omega, sample_dt, axis=0), SMOOTH_SAMPLES, axis=0, mode="nearest")
    steps = int(round(DURATION / CONTROL_DT))
    t = np.arange(steps + 1) * CONTROL_DT
    index = np.minimum((t / sample_dt + 1e-9).astype(int), count - 1)
    alpha = (t - index * sample_dt) / sample_dt
    out = {"R": np.zeros((steps + 1, rotation.shape[1], 3, 3)), "w": np.zeros((steps + 1, rotation.shape[1], 3)),
           "dw": np.zeros((steps + 1, rotation.shape[1], 3)), "t": t}
    for b in range(rotation.shape[1]):
        step_vec = Rotation.from_matrix(np.swapaxes(rotation[:-1, b], -1, -2) @ rotation[1:, b]).as_rotvec()   # (count, 3)
        out["R"][:, b] = rotation[index, b] @ Rotation.from_rotvec(alpha[:, None] * step_vec[index]).as_matrix()
        out["w"][:, b] = (1 - alpha)[:, None] * omega[index, b] + alpha[:, None] * omega[index + 1, b]
        out["dw"][:, b] = (1 - alpha)[:, None] * domega[index, b] + alpha[:, None] * domega[index + 1, b]
    out["R0"], out["w0"] = out["R"][0], out["w"][0]                     # tracking: start on the reference
    return out


def make_plant(data: rp.Dataset, seed: int, rotation0: np.ndarray, omega0: np.ndarray):
    from windy_pendulum_3d import windy_pendulum_3d
    truth, s = data.truth, data.settings
    env = windy_pendulum_3d(g=truth["g"], m=truth["m"], l=truth["l"], dt=CONTROL_DT, friction_coeff=tuple(truth["friction"]),
                            varying_friction=False, external_force_std=0.0, friction_law=truth["friction_law"],
                            wind_force_std=truth["wind"], g_diag=tuple(truth["g_diag"]), integrator=s["integrator"],
                            substeps=PLANT_SUBSTEPS)
    env.reset(seed=seed, options={"R_init": rotation0, "omega_init": omega0})
    return env


def fly(data: rp.Dataset, controller, ref: dict, seeds: int) -> dict:
    """All flights of one model: flight f = (trajectory f // seeds, seed f % seeds). Returns states (N+1, F, 12),
    applied and requested controls (N, F, 3) and the step of failure (N = none); NaN after a failure."""
    trajectories = ref["R"].shape[1]
    flights = [(b, k) for b in range(trajectories) for k in range(seeds)]
    plants = [make_plant(data, SEED_BASE + 100 * b + k, ref["R0"][b], ref["w0"][b]) for b, k in flights]
    steps = ref["R"].shape[0] - 1
    states = np.full((steps + 1, len(flights), 12), np.nan)
    applied = np.full((steps, len(flights), 3), np.nan)
    requested = np.full((steps, len(flights), 3), np.nan)
    failed_at = np.full(len(flights), steps)
    alive = np.ones(len(flights), bool)
    pick = np.asarray([b for b, _ in flights])
    for k in range(steps + 1):
        for f, plant in enumerate(plants):
            if alive[f]:
                states[k, f] = np.concatenate([plant.R.reshape(9), plant.omega])
                if not np.all(np.isfinite(states[k, f])):
                    alive[f], failed_at[f] = False, k
                    states[k, f] = np.nan
        if k == steps or not alive.any():
            break
        u, u_raw = controller(jnp.asarray(np.nan_to_num(states[k]), jnp.float32), jnp.asarray(ref["R"][k, pick], jnp.float32),
                              jnp.asarray(ref["w"][k, pick], jnp.float32), jnp.asarray(ref["dw"][k, pick], jnp.float32))
        u, u_raw = np.asarray(u, np.float64), np.asarray(u_raw, np.float64)
        for f, plant in enumerate(plants):
            if not alive[f]:
                continue
            if not np.all(np.isfinite(u[f])):
                alive[f], failed_at[f] = False, k
                continue
            applied[k, f], requested[k, f] = u[f], u_raw[f]
            plant.step(u[f])
    return {"states": states, "applied": applied, "requested": requested, "failed_at": failed_at, "flights": flights}


# ---------------------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------------------
def error_series(data: rp.Dataset, ref: dict, flight: dict, seeds: int) -> dict:
    """(N+1, F) tracking errors; non-finite (after a failure) = inf."""
    states = flight["states"]
    pick = np.asarray([b for b, _ in flight["flights"]])
    rd, wd = ref["R"][:, pick], ref["w"][:, pick]
    rotation = states[..., :9].reshape(*states.shape[:2], 3, 3)
    length = data.truth["l"]
    with np.errstate(invalid="ignore"):
        target = np.einsum("tfji,tfjk,tfk->tfi", rotation, rd, wd)                # R^T R_d w_d
        reference_state = np.concatenate([rd.reshape(*rd.shape[:2], 9), wd], axis=-1)
        out = {"att": np.radians(rp.geodesic_deg(rd.reshape(*rd.shape[:2], 9), states[..., :9])),      # rad
               "bob": np.linalg.norm(length * (rotation[..., :, 2] - rd[..., :, 2]), axis=-1),
               "w": np.linalg.norm(states[..., 9:12] - target, axis=-1),
               "H": np.abs(rp.true_energy(states, data.truth) - rp.true_energy(reference_state, data.truth))}
    return {k: np.nan_to_num(v, nan=np.inf) for k, v in out.items()}


def flight_metrics(data: rp.Dataset, ref: dict, flight: dict, analytical_flight: dict | None, mark: float) -> dict:
    """(F,) value of every table row over (0, mark] for every flight."""
    n = int(round(mark / CONTROL_DT))
    errors = error_series(data, ref, flight, SEEDS)
    window = slice(1, n + 1)
    out = {}
    for key, name in (("att", "att"), ("bob", "bob"), ("w", "w"), ("H", "H")):
        series = errors[key][window]
        with np.errstate(over="ignore", invalid="ignore"):
            out[f"{name}_rmse"] = np.sqrt(np.mean(series ** 2, axis=0))
        if key in ("att", "bob"):
            out[f"{name}_max"] = np.max(series, axis=0)
            out[f"{name}_final"] = errors[key][n]
    u = flight["applied"][:n]
    with np.errstate(invalid="ignore"):
        out["u_rms"] = np.nan_to_num(np.sqrt(np.mean(np.sum(u ** 2, axis=-1), axis=0)), nan=np.inf)
        out["chatter"] = np.nan_to_num(np.sqrt(np.mean(np.sum(np.diff(u, axis=0) ** 2, axis=-1), axis=0)) / CONTROL_DT, nan=np.inf)
        out["sat"] = np.nan_to_num(np.mean(np.abs(flight["requested"][:n]) > U_MAX, axis=(0, 2)), nan=1.0)
    if TASK.get("name") == "upright":
        angle = errors["att"][: n + 1]                                      # (n+1, F), from t = 0
        inside = angle < UPRIGHT_TOLERANCE
        outside = ~inside
        last_out = np.where(outside.any(axis=0), n - np.argmax(outside[::-1], axis=0), -1)
        out["t_up"] = np.where(inside[n], (last_out + 1) * CONTROL_DT, n * CONTROL_DT)
        tail = max(1, int(round(SUCCESS_SHARE * n)))
        out["success"] = inside[n - tail + 1: n + 1].all(axis=0).astype(float)
    never = flight["applied"].shape[0]                                     # failed_at == N: the flight never failed
    out["fail"] = ((flight["failed_at"] < never) & (flight["failed_at"] <= n)).astype(float)
    if analytical_flight is not None:
        pick = np.asarray([b for b, _ in flight["flights"]])
        length = data.truth["l"]
        bob_ref = length * ref["R"][:, pick][..., :, 2]                                  # (T, F, 3)
        excursion = np.sqrt(np.mean(np.sum((bob_ref - bob_ref[:1]) ** 2, axis=-1), axis=0))        # (F,)
        mine = length * flight["states"][..., :9].reshape(*flight["states"].shape[:2], 3, 3)[..., :, 2]
        theirs = length * analytical_flight["states"][..., :9].reshape(*analytical_flight["states"].shape[:2], 3, 3)[..., :, 2]
        with np.errstate(invalid="ignore"):
            gap = np.nan_to_num(np.linalg.norm(mine - theirs, axis=-1), nan=np.inf)[: n + 1]
        lost = gap > VALID_FRACTION * excursion[None]
        first = np.where(lost.any(axis=0), lost.argmax(axis=0), n)
        out["valid"] = first * CONTROL_DT
    return out


def table_cells(metrics: dict, labels: list[str], failed: set, trajectory: int | None, trajectories: int):
    """{row key: {label: (mean, std) or None}}. trajectory None: per trajectory mean over seeds, then mean +- std over
    trajectories; else mean +- std over that trajectory's seeds. Failures: (failed, flown)."""
    out = {}
    for key, _, _ in ROWS:
        out[key] = {}
        for label in labels:
            values = None if label in failed else metrics[label].get(key)
            if values is None:
                out[key][label] = None
                continue
            per = np.asarray(values, float).reshape(trajectories, SEEDS)
            if key == "fail":
                chosen = per if trajectory is None else per[trajectory]
                out[key][label] = (float(chosen.sum()), float(chosen.size))
            elif trajectory is None:
                seed_mean = per.mean(axis=1)
                out[key][label] = (float(np.mean(seed_mean)), float(np.std(seed_mean)))
            else:
                out[key][label] = (float(np.mean(per[trajectory])), float(np.std(per[trajectory])))
    return out


# ---------------------------------------------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------------------------------------------
def near_upright_share(data: rp.Dataset, angle: float = math.radians(30.0)) -> float:
    """% of the (clean) training states within ``angle`` (rad) of upright (bob height R_zz > cos)."""
    import pickle
    with data.path.with_name(data.path.name.rsplit("_", 1)[0] + "_clean.pkl").open("rb") as handle:
        heights = np.asarray(pickle.load(handle)["x"])[..., 8].ravel()
    return float(np.mean(heights > np.cos(angle)) * 100.0)


def setup_page(data: rp.Dataset, labels, failed, seeds, notes: dict | None = None) -> plt.Figure:
    truth = data.truth
    if TASK.get("name") == "upright":
        heights = data.clean[0, :TASK["trajectories"], 8]
        angles = ", ".join(f"{a:.2f}" for a in np.arccos(np.clip(heights, -1, 1)))
        task_lines = [f"task       SWING-UP AND STABILISATION TO UPRIGHT: R_d = I (bob straight up, l R e_z = l e_z, yaw 0), w_d = 0,",
                      f"           {DURATION:g} s. Starts = x_0 of the first {TASK['trajectories']} long_test trajectories (the data's own random",
                      f"           attitudes and rates): {angles} rad from upright. {near_upright_share(data):.2f} % of the training states lie",
                      "           within 0.524 rad of upright: the learned V and g near the top are mostly extrapolation."]
    else:
        task_lines = [f"references the {data.clean.shape[1]} clean long_test trajectories (own seeds), {DURATION:g} s: R_d(t), w_d(t) from the data,",
                      f"           dw_d/dt = central difference smoothed over {SMOOTH_SAMPLES} samples; interpolated to the control rate",
                      "           (geodesic for R, linear for w). Every flight starts from the reference's x_0."]
    error_text = ("e_R = Log(R_d^T R)  (geodesic, non-zero up to pi rad)" if TASK.get("error") == "log"
                  else "e_R = 1/2 vee(R_d^T R - R^T R_d)")
    lines = [f"dataset    {data.path.name}"] + task_lines + [
             f"plant      windy_pendulum_3d with the dataset's physics: m = {truth['m']:g}, l = {truth['l']:g}, g = {truth['g']:g},"
             f" G = diag({', '.join(f'{v:g}' for v in truth['g_diag'])}),",
             f"           damping {truth['friction_law']} c = {truth['friction'][0]:g}, wind force std {truth['wind']:g}; Lie-IMEX,"
             f" control at {1 / CONTROL_DT:g} Hz (dt {CONTROL_DT:g} s, {PLANT_SUBSTEPS} substeps), zero-order hold.",
             f"seeds      {seeds} wind seed(s) per trajectory (seed = {SEED_BASE} + 100 trajectory + k), the SAME for every model;",
             "           independent of the data's recorded wind.",
             "controller energy-shaping tracking on SO(3) (IDA-PBC with full matching), the same law and gains for every model:",
             "             f = model domega/dt at u = 0,  B = M^-1 g  (gauge-invariant: the learned gauge c cannot change the gains)",
             f"             {error_text},  e_w = w - R^T R_d w_d",
             "             a = -K_R e_R - K_w e_w - w x (R^T R_d w_d) + R^T R_d dw_d/dt   (closed loop: de_w/dt = -K_R e_R - K_w e_w)",
             f"             u = clip(lstsq(B, a - f), +-{U_MAX:g}),  K_R = {K_R:g}, K_w = {K_W:g}  (w_n = {math.sqrt(K_R):g} rad/s, critically damped)",
             "           Only the drift is used (no diffusion), so there are no -wind-same models.",
             "metrics    over (0, H] for H = 1, 3, 5, 10 s, at the control rate; tables = per trajectory the mean over seeds, then",
             "           mean +- std over trajectories (per-trajectory pages: mean +- std over seeds). Energy = TRUE Hamiltonian",
             "           H = 1/2 J |w|^2 + m g l R_zz on the flown and the reference state (upright at rest: H = m g l)."]
    if TASK.get("name") == "upright":
        lines += [f"           Time to upright = first t after which the angle stays below {UPRIGHT_TOLERANCE:.3f} rad up to H (= H if it is not",
                  f"           below at H). Success = below {UPRIGHT_TOLERANCE:.3f} rad throughout the last {SUCCESS_SHARE:.0%} of the window."]
    else:
        lines += ["           Valid tracking time = first t where the bob leaves the Analytical flight (same seed) by more than",
                  f"           {VALID_FRACTION:g} x the RMS reference excursion."]
    lines += ["           Chattering = RMS |du| / dt. Saturation = share of requested |u_i| > U_max. A failure = non-finite state/control.",
             "", "models"]
    for label in labels:
        lines.append(f"  {label:16s} {'TRAINING FAILED -> -' if label in failed else ('true operators (ceiling, not ranked)' if label == ANALYTICAL else 'posterior-mean drift')}")
        if label in (notes or {}):
            lines += textwrap.wrap(notes[label], width=110, initial_indent=" " * 19, subsequent_indent=" " * 21)
    figure = plt.figure(figsize=rp.PAGE)
    figure.text(0.04, 0.95, f"Closed-loop {TASK_TITLE[TASK.get('name', 'track')]}: setup", fontsize=16, weight="bold")
    figure.text(0.04, 0.90, "\n".join(lines), fontsize=7.8, va="top", family="monospace")
    return figure


def fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if key == "fail":
        return f"{value[0]:.0f} / {value[1]:.0f}"
    mean, std = value
    return f"{mean:.4g} +- {std:.2g}"


def table_page(cells: dict, labels, failed, title: str, subtitle: str) -> plt.Figure:
    figure = plt.figure(figsize=rp.PAGE)
    figure.text(0.03, 0.955, title, fontsize=14, weight="bold")
    figure.text(0.03, 0.925, subtitle, fontsize=7.5)
    header = ["metric"] + [label.replace("Lie-PH-", "Lie-PH-\n") for label in labels]
    rows, best = [], []
    learned = [i for i, label in enumerate(labels) if label != ANALYTICAL]
    for r, (key, name, rule) in enumerate(ROWS):
        values = [cells[key][label] for label in labels]
        rows.append([name] + [fmt(key, v) for v in values])
        if rule == "none":
            continue
        reference = cells[key].get(ANALYTICAL)
        candidates = []
        for i in learned:
            if values[i] is None or not np.isfinite(values[i][0]):
                continue
            if rule == "low":
                score = values[i][0]
            elif rule == "high":
                score = -values[i][0]
            else:
                if reference is None:
                    continue
                score = abs(values[i][0] - reference[0])
            candidates.append((round(float(score), 9), i))                    # rounded: equal values share the highlight
        if len(candidates) > 1:
            top = min(candidates)[0]
            best += [(r + 1, 1 + i) for score, i in candidates if score == top]
    axis = figure.add_axes([0.01, 0.07, 0.98, 0.83])
    axis.axis("off")
    widths = [0.24] + [0.76 / len(labels)] * len(labels)
    rp.styled_table(axis, rows, header, widths, 7.2, 1.6, best, failed_cols=[1 + i for i, l in enumerate(labels) if l in failed])
    figure.text(0.03, 0.015, "Green bold = best LEARNED model per row (lowest; highest for valid tracking time and success; RMS |u| closest to "
                "Analytical; ties share the highlight).\nAnalytical = true operators in the same controller (ceiling, not ranked). "
                "'-' = training failed (red column) or not defined. Failures are not ranked.", fontsize=6.5)
    return figure


def error_page(t, errors: dict, labels, failed, trajectories) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=rp.PAGE)
    names = {"att": "attitude error (rad)", "bob": "bob position error (m)", "w": "angular-velocity error |e_w| (rad/s)",
             "H": "energy error |H(x) - H(x_d)| (J)"}
    if TASK.get("name") == "upright":
        names = {"att": "angle to upright (rad)", "bob": "bob distance from the top (m)", "w": "|w| (rad/s)",
                 "H": "energy error |H - m g l| (J)"}
    for axis, key in zip(axes.ravel(), ("att", "bob", "w", "H")):
        for label in labels:
            if label in failed:
                continue
            series = errors[label][key].reshape(len(t), trajectories, SEEDS).mean(axis=2)          # seed mean
            style = dict(color=COLORS[label], lw=1.5 if label == ANALYTICAL else 1.8, ls="--" if label == ANALYTICAL else "-")
            axis.plot(t[1:], rp.plottable(np.median(series, axis=1))[1:], label=label, **style)
            if label != ANALYTICAL:
                axis.fill_between(t[1:], rp.plottable(np.quantile(series, 0.25, axis=1))[1:],
                                  rp.plottable(np.quantile(series, 0.75, axis=1))[1:], color=COLORS[label], alpha=0.10, lw=0)
        for mark in HORIZON_MARKS[:-1]:
            axis.axvline(mark, color="0.5", lw=0.8, ls="--")
        axis.set_yscale("log")
        axis.set_title(names[key], fontsize=9)
        axis.set_xlabel("t (s)")
        axis.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=7)
    figure.suptitle(f"Closed-loop {TASK_TITLE[TASK.get('name', 'track')]}: error vs time, median over trajectories of the seed mean, shade = inter-quartile, "
                    "dashed = Analytical", fontsize=10)
    figure.tight_layout()
    return figure


def bar_page(summary: dict, labels, failed) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=rp.PAGE)
    alive = [l for l in labels if l not in failed]
    width = 0.8 / max(len(alive), 1)
    for axis, (key, name) in zip(axes.ravel(), (("att_rmse", "attitude RMSE (rad)"), ("bob_rmse", "bob position RMSE (m)"),
                                                ("w_rmse", "angular-velocity error RMSE (rad/s)"), ("H_rmse", "energy error RMSE (J)"))):
        for k, label in enumerate(alive):
            values = rp.plottable([summary[f"0-{m:g}s"][key][label][0] for m in HORIZON_MARKS])
            axis.bar(np.arange(len(HORIZON_MARKS)) + (k - (len(alive) - 1) / 2) * width, values, width, color=COLORS[label],
                     label=label, edgecolor="k", lw=0.3, hatch="//" if label == ANALYTICAL else None)
        axis.set_xticks(range(len(HORIZON_MARKS)))
        axis.set_xticklabels([f"0-{m:g} s" for m in HORIZON_MARKS])
        axis.set_yscale("log")
        axis.set_title(name, fontsize=9)
        axis.grid(alpha=0.3, axis="y", which="both")
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Closed-loop RMSE over 0-1 / 0-3 / 0-5 / 0-10 s (mean over trajectories of the seed mean; hatched = Analytical)",
                    fontsize=10)
    figure.tight_layout()
    return figure


def seed_band(values: np.ndarray):
    """(S, T, ...) -> seed mean and std (NaN-aware)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def flight_values(flights: dict, label: str, index: int) -> np.ndarray:
    """(S, N+1, 12) states of trajectory ``index`` for every seed."""
    return np.swapaxes(flights[label]["states"][:, index * SEEDS:(index + 1) * SEEDS], 0, 1)


def path3d_page(data, ref, flights, labels, failed, index) -> plt.Figure:
    length = data.truth["l"]
    columns = 3
    figure = plt.figure(figsize=rp.PAGE)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
    reference = length * ref["R"][:, index, :, 2]
    for k, label in enumerate(labels):
        axis = figure.add_subplot(math.ceil(len(labels) / columns), columns, k + 1, projection="3d")
        axis.plot_wireframe(length * np.cos(u) * np.sin(v), length * np.sin(u) * np.sin(v), length * np.cos(v), color="0.85", lw=0.3)
        axis.plot(*reference.T, color=REF_COLOR, lw=1.0, ls="--")
        axis.scatter(*reference[-1], color=REF_COLOR, marker="*", s=60, zorder=5)
        if label in failed:
            axis.set_title(f"{label}\ntraining failed", fontsize=8, color="0.4")
        else:
            states = flight_values(flights, label, index)
            bobs = length * states[..., :9].reshape(*states.shape[:2], 3, 3)[..., :, 2]
            for path in bobs:
                axis.plot(*rp.plottable(path).T, color=COLORS[label], lw=0.4, alpha=0.4)
            mean, _ = seed_band(bobs)
            axis.plot(*rp.plottable(mean).T, color=COLORS[label], lw=1.5)
            axis.set_title(label, fontsize=8, color=COLORS[label], weight="bold")
        lim = 1.1 * length
        axis.set_xlim(-lim, lim)
        axis.set_ylim(-lim, lim)
        axis.set_zlim(-lim, lim)
        axis.tick_params(labelsize=5)
    figure.suptitle(f"Trajectory {index}: bob path l R e_z (dashed = reference, thin = {SEEDS} wind seeds, bold = seed mean)",
                    fontsize=10)
    figure.tight_layout()
    return figure


def component_page(data, ref, flights, labels, failed, index, block: str) -> plt.Figure:
    """Rows = 3 components, columns = models: seed mean +- 1 std band against the dashed reference."""
    length = data.truth["l"]
    names = {"bob": (["bob x (m)", "bob y (m)", "bob z (m)"], "Bob position l R e_z"),
             "euler": (["roll (rad)", "pitch (rad)", "yaw (rad)"], "Attitude (Euler xyz of R, unwrapped)"),
             "w": (["omega_x (rad/s)", "omega_y (rad/s)", "omega_z (rad/s)"], "Body angular velocity"),
             "u": (["u_x", "u_y", "u_z"], f"Control u (clipped at +-{U_MAX:g})")}
    ylabels, title = names[block]
    t = ref["t"]

    def values(states):
        if block == "bob":
            return length * states[..., :9].reshape(*states.shape[:-1], 3, 3)[..., :, 2]
        if block == "euler":
            return np.radians(rp.euler_deg(states[..., :9]))
        return states[..., 9:12]

    reference_state = np.concatenate([ref["R"][:, index].reshape(-1, 9), ref["w"][:, index]], axis=-1)
    reference = None if block == "u" else values(reference_state)
    figure, axes = plt.subplots(3, len(labels), figsize=(max(rp.PAGE[0], 2.6 * len(labels)), rp.PAGE[1]), sharex=True, squeeze=False)
    for col, label in enumerate(labels):
        for row in range(3):
            axis = axes[row, col]
            if reference is not None:
                axis.plot(t, reference[:, row], color=REF_COLOR, lw=1.0, ls="--")
            if label in failed:
                if row == 1:
                    axis.text(0.5, 0.5, "training failed", transform=axis.transAxes, ha="center", color="0.4")
            else:
                if block == "u":
                    data_rows = np.swapaxes(flights[label]["applied"][:, index * SEEDS:(index + 1) * SEEDS], 0, 1)[..., row]
                    tt = t[:-1]
                else:
                    data_rows = values(flight_values(flights, label, index))[..., row]
                    if block == "euler":                                   # each seed on the reference's 2 pi branch
                        data_rows = data_rows - 2 * np.pi * np.round((data_rows[:, :1] - reference[0, row]) / (2 * np.pi))
                    tt = t
                mean, std = seed_band(data_rows)
                axis.fill_between(tt, rp.plottable(mean - std), rp.plottable(mean + std), color=COLORS[label], alpha=0.3, lw=0)
                axis.plot(tt, rp.plottable(mean), color=COLORS[label], lw=1.1)
            if reference is not None and np.ptp(reference[:, row]) > 1e-6:      # a moving reference: cap at its range +- 1x
                low, high = np.nanmin(reference[:, row]), np.nanmax(reference[:, row])
                reach = 1.0 * max(high - low, 1e-3)
                current = axis.get_ylim()
                axis.set_ylim(max(current[0], low - reach), min(current[1], high + reach))
            elif reference is None:
                axis.set_ylim(-1.1 * U_MAX, 1.1 * U_MAX)
            for mark in HORIZON_MARKS[:-1]:
                axis.axvline(mark, color="0.5", lw=0.5, ls="--")
            axis.grid(alpha=0.25)
            axis.tick_params(labelsize=6)
            if col == 0:
                axis.set_ylabel(ylabels[row], fontsize=7)
            if row == 0:
                axis.set_title(label.replace("Lie-PH-", "Lie-PH-\n"), color=COLORS[label], weight="bold", fontsize=7)
            if row == 2:
                axis.set_xlabel("t (s)", fontsize=7)
    note = ("no reference (the data's own u was an open-loop random input)" if block == "u"
            else "dashed = upright target" if TASK.get("name") == "upright"
            else "dashed = reference; y-axis capped at the reference range +- 1x its width")
    figure.suptitle(f"Trajectory {index}: {title}. Line = mean over {SEEDS} wind seeds, band = +- 1 std over seeds; {note}",
                    fontsize=9)
    figure.tight_layout()
    return figure


def energy_page(data, ref, flights, labels, failed, index) -> plt.Figure:
    columns = 3
    rows = math.ceil(len(labels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=rp.PAGE, sharex=True, sharey=True, squeeze=False)
    t = ref["t"]
    reference = rp.true_energy(np.concatenate([ref["R"][:, index].reshape(-1, 9), ref["w"][:, index]], axis=-1), data.truth)
    low, high = np.nanmin(reference), np.nanmax(reference)
    reach = 1.0 * max(high - low, 1e-3)
    for k, axis in enumerate(axes.ravel()):
        if k >= len(labels):
            axis.axis("off")
            continue
        label = labels[k]
        axis.plot(t, reference, color=REF_COLOR, lw=1.0, ls="--")
        if label in failed:
            axis.text(0.5, 0.5, "training failed", transform=axis.transAxes, ha="center", color="0.4")
        else:
            mean, std = seed_band(rp.true_energy(flight_values(flights, label, index), data.truth))
            axis.fill_between(t, rp.plottable(mean - std), rp.plottable(mean + std), color=COLORS[label], alpha=0.3, lw=0)
            axis.plot(t, rp.plottable(mean), color=COLORS[label], lw=1.1)
        axis.set_title(label, color=COLORS[label], weight="bold", fontsize=8)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=6)
        if k % columns == 0:
            axis.set_ylabel("H (J)", fontsize=7)
    if high - low > 1e-6:
        axes[0, 0].set_ylim(low - reach, high + reach)
    else:                                                        # constant target (upright): data range, capped at +- 3 m g l
        bound = 3.0 * data.truth["m"] * data.truth["g"] * data.truth["l"]
        current = axes[0, 0].get_ylim()
        axes[0, 0].set_ylim(max(current[0], low - bound), min(current[1], high + bound))
    figure.suptitle(f"Trajectory {index}: true Hamiltonian H = 1/2 J |w|^2 + m g l R_zz of the flown state (seed mean +- 1 std) "
                    "and of the reference (dashed)", fontsize=9)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------
# Shown under a model renamed with --label (setup page, summary json).
RENAME_NOTES = {"PH-NODE-ref-pretrain": "= run PH-NODE-ref-pretrain: Duong & Atanasov's DissipativeSO3HamNODE (3 782 parameters, "
                                        "orthogonal init gain 0.5, M^-1 pretrained to I), their 0.2 s chunks / full batch / Adam 1e-3 "
                                        "constant, weight decay 1e-4, RK4, 5000 steps"}


def closed_loop_report(setting: str, level: float, output_root: Path | None = None, refly: bool = False,
                       task: str = "track", models: tuple | None = None, rename: dict | None = None) -> Path:
    """``models``: campaign model names to fly (default LEARNED); the Analytical ceiling is always flown.
    ``rename``: {campaign model: label in the PDF}; flights are cached under the campaign name, never the label."""
    global SEEDS
    set_task(task)
    data = rp.Dataset(setting, level)
    result_dir = RESULT_DIR
    records = rp.campaign_records(result_dir)
    learned = tuple(models) if models else LEARNED
    unknown = [m for m in learned if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {list(MODELS)}")
    rename = rename or {}
    sources = load_sources(data, records, learned)
    shown = {m: rename.get(m, m) for m in learned if m in sources}          # campaign name -> label
    source_of = {label: m for m, label in shown.items()}
    if len(source_of) != len(shown):
        raise ValueError(f"two models share a label: {shown}")
    labels = list(shown.values()) + [ANALYTICAL]
    failed = {shown[m] for m in shown if sources[m] is None}
    notes = {shown[m]: RENAME_NOTES.get(m, f"= run {m}") for m in shown if m in rename}
    # Wind-free data: every flight is deterministic, so one seed per start (the plant has no noise to draw).
    seeds = 5 if data.truth["wind"] > 0 else 1
    SEEDS = seeds
    ref = upright_references(data) if task == "upright" else references(data)
    trajectories = ref["R"].shape[1]
    print(f"[closed-loop] {setting} noise {level:g}: {trajectories} trajectories x {seeds} seeds, "
          f"{len(ref['t']) - 1} steps at {1 / CONTROL_DT:g} Hz", flush=True)
    out_dir = (output_root or result_dir / "reports") / f"{setting}_noise{level:g}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = TASK["file"].replace("-comparison", "")                 # closed-loop | closed-loop-upright
    cache = out_dir / f"{stem}-flights.npz"                       # every flight's states and controls (reused unless --refly)
    stored = dict(np.load(cache, allow_pickle=True)) if cache.exists() and not refly else {}
    flights = {}
    cache_key = {label: (ANALYTICAL if label == ANALYTICAL else source_of[label]) for label in labels}
    for label in [ANALYTICAL] + [l for l in labels if l != ANALYTICAL]:
        if label in failed:
            continue
        if f"{cache_key[label]}/states" in stored:
            flights[label] = {key: stored[f"{cache_key[label]}/{key}"] for key in ("states", "applied", "requested", "failed_at")}
            flights[label]["flights"] = [(b, k) for b in range(trajectories) for k in range(seeds)]
            print(f"[closed-loop] {label:16s} loaded from {cache.name}", flush=True)
            continue
        source = "analytical" if label == ANALYTICAL else source_of[label]
        tic = datetime.now()
        flights[label] = fly(data, make_controller(sources[source]["mean_model"]), ref, seeds)
        print(f"[closed-loop] {label:16s} flown in {(datetime.now() - tic).total_seconds():.0f} s, "
              f"failures {int(np.sum(flights[label]['failed_at'] < len(ref['t']) - 1))}", flush=True)
    np.savez_compressed(cache, **{f"{cache_key[label]}/{key}": value[key] for label, value in flights.items()
                                  for key in ("states", "applied", "requested", "failed_at")})
    metrics = {f"0-{m:g}s": {label: flight_metrics(data, ref, flights[label],
                                                   None if label == ANALYTICAL or task == "upright" else flights[ANALYTICAL], m)
                             for label in flights} for m in HORIZON_MARKS}
    summary = {key: table_cells(metrics[key], labels, failed, None, trajectories) for key in metrics}
    errors = {label: error_series(data, ref, flights[label], seeds) for label in flights}
    pdf_path = out_dir / f"{TASK['file']}.pdf"
    with PdfPages(pdf_path) as pdf:
        entries = [e for e in rp.entries_for(setting) if e.source == "analytical" or e.source in shown]
        training_note = None
        if notes:
            training_note = textwrap.fill("training    " + "; ".join(f"{label} {text}" for label, text in notes.items())
                                          + ". Other models: 2 s windows, 25 % overlap, 5000 steps, final model, float32, no priors, "
                                          "no pretraining", width=150, subsequent_indent=" " * 12)
        pdf.savefig(rp.physics_page(data, entries, sources, names=shown, training_note=training_note)); plt.close("all")
        pdf.savefig(setup_page(data, labels, failed, seeds, notes)); plt.close("all")
        for key in metrics:
            pdf.savefig(table_page(summary[key], labels, failed, f"Table C - closed-loop {TASK_TITLE[task]}, {key.replace('s', ' s')}",
                                   f"{data.path.name}: per trajectory the mean over {seeds} wind seeds, then mean +- std over "
                                   f"{trajectories} trajectories.")); plt.close("all")
        pdf.savefig(error_page(ref["t"], errors, labels, failed, trajectories)); plt.close("all")
        pdf.savefig(bar_page(summary, labels, failed)); plt.close("all")
        for index in range(trajectories):
            for key in metrics:
                cells = table_cells(metrics[key], labels, failed, index, trajectories)
                pdf.savefig(table_page(cells, labels, failed, f"Trajectory {index} - closed-loop {TASK_TITLE[task]}, {key.replace('s', ' s')}",
                                       f"{data.path.name}, evaluation trajectory {index}: mean +- std over {seeds} wind seeds.")); plt.close("all")
            pdf.savefig(path3d_page(data, ref, flights, labels, failed, index)); plt.close("all")
            for block in ("bob", "euler", "w"):
                pdf.savefig(component_page(data, ref, flights, labels, failed, index, block)); plt.close("all")
            pdf.savefig(energy_page(data, ref, flights, labels, failed, index)); plt.close("all")
            pdf.savefig(component_page(data, ref, flights, labels, failed, index, "u")); plt.close("all")
    record = {"generated_at": datetime.now().astimezone().isoformat(), "dataset": str(data.path), "setting": setting, "noise": level,
              "task": task, "controller": {"K_R": K_R, "K_w": K_W, "U_max": U_MAX, "control_dt": CONTROL_DT,
                                                 "plant_substeps": PLANT_SUBSTEPS, "seeds": seeds, "seed_base": SEED_BASE},
              "models": {label: ("failed" if label in failed else str(sources["analytical" if label == ANALYTICAL else source_of[label]]["run_dir"]))
                         for label in labels},
              "renamed": {label: {"campaign_model": source_of[label], "note": text} for label, text in notes.items()},
              "table": summary}
    (out_dir / f"{stem}-summary.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(f"[closed-loop] wrote {pdf_path}", flush=True)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", required=True, choices=tuple(SETTINGS))
    parser.add_argument("--noise", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, help="default: <campaign folder>/reports")
    parser.add_argument("--refly", action="store_true", help="fly again even if the cached flights exist")
    parser.add_argument("--task", default="upright", choices=tuple(TASKS), help="upright: swing-up to upright from 5 starts; "
                        "track: track the 10 recorded evaluation trajectories")
    parser.add_argument("--models", nargs="+", help="campaign models to fly (default: the five campaign models); "
                        "Analytical is always included")
    parser.add_argument("--label", nargs="+", default=[], metavar="MODEL=NAME", help="print MODEL as NAME in the PDF")
    args = parser.parse_args()
    rename = dict(item.split("=", 1) for item in args.label)
    for level in args.noise:
        closed_loop_report(args.setting, level, args.output, args.refly, args.task, args.models, rename)


if __name__ == "__main__":
    main()

"""One SO(3) windy-pendulum dataset generator, driven by one config file (config.yaml next to this script).

    python envs/pendulum_so3/datagen/generate_dataset.py --config envs/pendulum_so3/datagen/config.yaml

Writes datasets/PENDULUM-DATASET-<name>/<name>_<variant>.pkl, one pickle per observation-noise level plus a clean one:
    clean            x = clean train,          test_x = clean test,  test_x_noisy = clean test
    obs-noise<s>     x = train + noise s,      test_x = clean test,  test_x_noisy = test + noise s
Every pickle has the layout the pendulum trainers read (x, test_x, test_x_noisy: (batches, T, N, 15); t: (T,); settings).
Row k of a trajectory is [vec(R_k) (9, row-major), w_k (3, body frame), u (3)]. The u in row k+1 is the control applied
during k -> k+1 (held constant over that interval); row 0 repeats u_0, the control of the first interval.
Sampling and noise come from windy_pendulum_3d_datagen.py.

Plant (envs/pendulum_so3/windy_pendulum_3d.py), state (R, w) on SO(3) x R^3, J = m l^2 I:
    dR = R [w]x dt
    J dw = (tau_gravity + tau_external + G u - f(R, w) * w - w x J w) dt + R^T (r x sigma dW)
integrated with Lie-IMEX (friction implicit, the rest explicit), `substeps` steps per sample dt.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]  # envs/pendulum_so3/datagen -> project root
for path in (PROJECT_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from windy_pendulum_3d_datagen import add_proper_noise_3d, sample_windy_pendulum_3d, to_pickle  # noqa: E402

INTEGRATORS = ("lie_imex",)
DISSIPATION_TYPES = ("none", "constant", "varying", "rate_dependent")
DIFFUSION_TYPES = ("none", "constant")
EXTERNAL_FORCE_TYPES = ("sine", "square", "random", "constant")


def validate(cfg: dict) -> None:
    checks = (("integrator", cfg["integrator"], INTEGRATORS), ("dissipation.type", cfg["dissipation"]["type"], DISSIPATION_TYPES),
              ("diffusion.type", cfg["diffusion"]["type"], DIFFUSION_TYPES),
              ("external_force.type", cfg["external_force"]["type"], EXTERNAL_FORCE_TYPES))
    for key, value, allowed in checks:
        if value not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    if any(float(s) <= 0 for s in cfg["observation_noise"]["levels"]):
        raise ValueError("observation_noise.levels must be positive (the clean pickle is always written)")


def env_arguments(cfg: dict) -> dict:
    """Config -> the keyword arguments of windy_pendulum_3d_datagen.sample_windy_pendulum_3d / the env."""
    d, w, f, c, p = cfg["dissipation"], cfg["diffusion"], cfg["external_force"], cfg["control"], cfg["plant"]
    friction = 0.0 if d["type"] == "none" else d["c"]
    return {
        "friction_coeff": tuple(friction) if isinstance(friction, list) else float(friction),
        "varying_friction": d["type"] == "varying",
        "friction_law": "constant" if d["type"] in ("none", "constant") else d["type"],
        "wind_force_std": float(w["sigma"]) if w["type"] == "constant" else 0.0,
        "external_force_type": f["type"], "external_force_std": float(f["std"]),
        "external_force_direction": tuple(float(x) for x in f["direction"]),
        "random_u": bool(c["random_u"]), "random_u_scale": float(c["random_u_scale"]),
        "g_diag": tuple(float(x) for x in c["g_diag"]),
        "g": float(p["g"]), "m": float(p["m"]), "l": float(p["l"]),
        "dt": float(cfg["dt"]), "substeps": int(cfg["substeps"]), "integrator": cfg["integrator"],
        "max_speed": float(cfg["validity"]["max_speed"]),
    }


def simulate(cfg: dict, timesteps: int | None = None, samples: int | None = None,
             seed_offset: int = 0) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """All control batches -> clean (batches, T, samples, 15) array, time vector, batch list, and the recorded wind
    (batches, T, samples, 3): row k+1 = body angular-velocity increment of the wind during k -> k+1."""
    batches = [tuple(float(x) for x in u) for u in cfg["control"]["batches"]]
    kwargs = env_arguments(cfg)
    trajs, winds, tspan = [], [], None
    for i, u in enumerate(batches):
        tic = time.perf_counter()
        traj, tspan, wind = sample_windy_pendulum_3d(seed=int(cfg["seed"]) + seed_offset + i * 10000,
                                                     timesteps=int(timesteps or cfg["timesteps"]),
                                                     trials=int(samples or cfg["samples"]), u=u, ori_rep="rotmat",
                                                     return_wind=True, **kwargs)
        trajs.append(traj); winds.append(wind)
        print(f"batch {i} u={u}: {traj.shape[1]} trajectories in {time.perf_counter() - tic:.1f} s", flush=True)
    return np.stack(trajs, axis=0), tspan, batches, np.stack(winds, axis=0)


def build(cfg: dict) -> dict[str, dict]:
    clean, tspan, batches, wind = simulate(cfg)
    samples, test_split, seed = int(cfg["samples"]), float(cfg["test_split"]), int(cfg["seed"])
    split = int(samples * 0.5) if test_split >= 0.5 else int(samples * (1.0 - test_split))
    train, test = clean[:, :, :split, :], clean[:, :, split:, :]
    shared = {"x_wind": wind[:, :, :split], "test_wind": wind[:, :, split:]}       # recorded wind, same in every variant
    # Optional long evaluation set (long_test): separate seeds (seed + seed_offset), so the train/test data above are
    # unchanged; used only to score long open-loop horizons.
    long = cfg.get("long_test")
    if long:
        long_clean, long_t, _, long_wind = simulate(cfg, timesteps=int(long["timesteps"]),
                                                    samples=int(long["samples_per_batch"]), seed_offset=int(long["seed_offset"]))
        shared.update({"long_t": long_t, "long_test_x": long_clean, "long_test_wind": long_wind})
    kwargs = env_arguments(cfg)
    settings = {                                                  # the get_dataset() settings keys, plus the generator's own
        "seed": seed, "samples": samples, "test_split": test_split, "us": tuple(batches), "ori_rep": "rotmat",
        "friction_coeff": kwargs["friction_coeff"], "varying_friction": kwargs["varying_friction"],
        "friction_law": kwargs["friction_law"],
        "external_force_type": kwargs["external_force_type"], "external_force_std": kwargs["external_force_std"],
        "external_force_direction": kwargs["external_force_direction"], "g": kwargs["g"],
        "obs_noise_std": 0.0, "wind_force_std": kwargs["wind_force_std"], "timesteps": int(cfg["timesteps"]),
        "random_u": kwargs["random_u"], "random_u_scale": kwargs["random_u_scale"], "g_diag": kwargs["g_diag"],
        "name": cfg["name"], "integrator": cfg["integrator"], "dt": kwargs["dt"], "substeps": kwargs["substeps"],
        "m": kwargs["m"], "l": kwargs["l"], "dissipation": cfg["dissipation"]["type"], "diffusion": cfg["diffusion"]["type"],
        "max_speed": kwargs["max_speed"], "config": copy.deepcopy(cfg),
        "recorded_wind": "x_wind / test_wind / long_test_wind: body angular-velocity increment of the wind per sample "
                         "(sum over substeps of the 0.5 (s1 + s2) noise term), row k+1 = interval k -> k+1",
        "long_test": copy.deepcopy(long) if long else None,
    }
    variants = {"clean": {"t": tspan, "x": train, "test_x": test, "test_x_noisy": test, "settings": settings, **shared}}
    if long:
        variants["clean"]["long_test_x_noisy"] = shared["long_test_x"]
    for level in cfg["observation_noise"]["levels"]:
        level = float(level)
        s = dict(settings, obs_noise_std=level)
        variants[f"obs-noise{level:g}".replace(".", "p")] = {
            "t": tspan, "settings": s,
            "x": add_proper_noise_3d(train, level, np.random.default_rng(seed + 999)),           # noisy input
            "test_x": test,                                                                      # clean target
            "test_x_noisy": add_proper_noise_3d(test, level, np.random.default_rng(seed + 1999)),
            **shared,
        }
        if long:
            variants[f"obs-noise{level:g}".replace(".", "p")]["long_test_x_noisy"] = add_proper_noise_3d(
                shared["long_test_x"], level, np.random.default_rng(seed + 2999))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=THIS_DIR / "config.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "datasets",
                        help="parent folder; the dataset goes to <output-root>/PENDULUM-DATASET-<name>/")
    parser.add_argument("--force", action="store_true", help="overwrite existing files of the same dataset")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate(cfg)
    name = str(cfg["name"])
    out_dir = (args.output_root / f"PENDULUM-DATASET-{name}").resolve()
    if not args.force and list(out_dir.glob(f"{name}_*")):
        raise FileExistsError(f"{out_dir} already holds {name}_* files; pass --force to overwrite or change `name` in the config")
    print(f"dataset {name} -> {out_dir}  (integrator {cfg['integrator']}, {cfg['substeps']} substeps per dt = {cfg['dt']} s)")
    variants = build(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, payload in variants.items():
        to_pickle(payload, str(out_dir / f"{name}_{variant}.pkl"))
    (out_dir / f"{name}_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    summary = {k: v for k, v in variants["clean"]["settings"].items() if k != "config"}
    summary["shapes"] = {k: list(v.shape) for k, v in variants["clean"].items() if isinstance(v, np.ndarray)}
    summary["variants"] = list(variants)
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()

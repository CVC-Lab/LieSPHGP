"""Step 2: Marinarium npz (bag_to_npz.py) -> SE(3) dataset for the models, driven by config.yaml.

    python envs/rov_se3_marinarium/datagen/generate_dataset.py --config envs/rov_se3_marinarium/datagen/config.yaml

Writes datasets/ROV-MARINARIUM-DATASET-<name>/<name>_clean.pkl (+ <name>_audits.json, <name>_config_used.yaml).
Real data has no noise variants: the recording already contains the measurement noise.

Row layout: [p (3), vec(R) (9), v_b (3), w_b (3), u (n_u)], world z DOWN, body forward-right-down.
    p, R          motion capture (position, quaternion slerp) at the sample times
    v_b, w_b      motion-capture body twist (or the PX4 gyro for w_b, `angular_rate: gyro`)
    u in row k    the input applied over (t_{k-1}, t_k]: mean over the ~100 Hz motor messages in that interval of
                  commands (n_u = 8), T200 thrust at the measured battery voltage (8, N), or E @ thrust (6, N / N m)
Trajectories are cut only inside clean stretches of the motion capture: no sample is interpolated across a dropout
longer than max_gap_seconds, and no trajectory contains a pose glitch (an attitude jump the gyro does not see). `streams` also stores each split as one continuous series on a uniform clock
(gaps interpolated, flagged in <split>_stream_valid), the layout of the Marinarium paper's H-step evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]  # envs/rov_se3_marinarium/datagen -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.rov_se3_marinarium.bluerov2 import PUBLISHED, THRUSTERS, T200, allocation_matrix  # noqa: E402

INPUTS = ("commands", "thrust", "wrench")
CONTROL_LAYOUT = {
    "commands": "u1..u8 PX4 normalised motor commands in [-1, 1] (0 = not commanded); thrust = T200(1500 + 400 u, V), nonlinear",
    "thrust": "T1..T8 thruster forces (N) = T200(1500 + 400 u_i, battery V) along the thruster axes; wrench = E @ T",
    "wrench": "[Fx, Fy, Fz, Mx, My, Mz] (N, N m) body frame = E @ T200 thrust",
}
TRUE_MAP = {"commands": "nonlinear: E @ T200(u, V)", "thrust": "E (6 x 8 allocation, published geometry, PX4 order)",
            "wrench": "identity"}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate(cfg: dict) -> None:
    if cfg["input"] not in INPUTS:
        raise ValueError(f"input must be one of {INPUTS}")
    if cfg["angular_rate"] not in ("mocap", "gyro"):
        raise ValueError("angular_rate must be mocap or gyro")
    if cfg["split"]["type"] not in ("chronological", "recordings"):
        raise ValueError("split.type must be chronological or recordings")
    if cfg["disarmed"] not in ("zero", "drop"):
        raise ValueError("disarmed must be zero or drop")
    if cfg["mocap_time"] not in ("header", "receive"):
        raise ValueError("mocap_time must be header or receive")
    if cfg["voltage"] != "measured" and not isinstance(cfg["voltage"], (int, float)):
        raise ValueError("voltage must be measured or a number (V)")


# --------------------------------------------------------------------------- one recording
class Recording:
    def __init__(self, path: Path, cfg: dict):
        self.name, d = path.stem, np.load(path)
        self.t = d["mocap_t_header"] if cfg["mocap_time"] == "header" else d["mocap_t"]
        self.position, self.quaternion = d["mocap_position"], d["mocap_quaternion_xyzw"]
        self.v_body, self.omega_body = d["mocap_v_body"], d["mocap_omega_body"]
        self.imu_t, self.gyro = d["imu_t"], d["imu_gyro"]
        self.motor_t, self.command = d["motor_t"], d["motor_command"]
        self.battery_t, self.voltage = d["battery_t"], d["battery_voltage"]
        self.cfg, self.E = cfg, allocation_matrix()
        gap = np.diff(self.t) > float(cfg["max_gap_seconds"])
        intervals = list(zip(self.t[:-1][gap], self.t[1:][gap]))
        self.glitches = 0
        if cfg["glitch_threshold_deg"] is not None:
            # one-step attitude jumps the gyro does not see (marker swaps, flips): excluded like dropouts
            rot = Rotation.from_quat(self.quaternion)
            step = np.degrees((rot[:-1].inv() * rot[1:]).magnitude())
            gyro_step = np.degrees(np.linalg.norm(np.stack([np.interp(self.t[1:], self.imu_t, self.gyro[:, j]) for j in range(3)], 1),
                                                  axis=1) * np.diff(self.t))
            jump = (step > float(cfg["glitch_threshold_deg"])) & (step > 5.0 * np.maximum(gyro_step, 0.2))
            m = float(cfg["glitch_margin_seconds"])
            intervals += [(a - m, b + m) for a, b in zip(self.t[:-1][jump], self.t[1:][jump])]
            self.glitches = int(jump.sum())
        merged = []
        for a, b in sorted(intervals):                           # union of dropouts and glitch windows
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.gap_start = np.array([a for a, _ in merged]); self.gap_end = np.array([b for _, b in merged])
        self._controls = self._motor_controls()
        self.disarmed = np.isnan(self.command).all(axis=1)

    def _motor_controls(self) -> np.ndarray:
        """Per motor message: the chosen input (commands, thrust or wrench), NaN commands -> 0."""
        command = np.nan_to_num(self.command, nan=0.0)
        if self.cfg["input"] == "commands":
            return command
        volts = (np.interp(self.motor_t, self.battery_t, self.voltage) if self.cfg["voltage"] == "measured"
                 else np.full(len(self.motor_t), float(self.cfg["voltage"])))
        thrust = T200().thrust_from_command(command, volts)
        return thrust if self.cfg["input"] == "thrust" else thrust @ self.E.T

    def segments(self, t_lo: float, t_hi: float) -> list[tuple[float, float]]:
        """Dropout-free stretches of the motion capture inside [t_lo, t_hi]."""
        out, start = [], t_lo
        for gs, ge in zip(self.gap_start, self.gap_end):         # gaps are sorted and disjoint
            if ge <= t_lo or gs >= t_hi:
                continue
            if gs > start:
                out.append((start, gs))
            start = max(start, ge)
        if t_hi > start:
            out.append((start, t_hi))
        return out

    def rows(self, times: np.ndarray) -> np.ndarray:
        """State + input rows at the sample times (times strictly increasing, step h)."""
        h = 1.0 / float(self.cfg["sample_hz"])
        rot = Slerp(self.t, Rotation.from_quat(self.quaternion))(np.clip(times, self.t[0], self.t[-1])).as_matrix()
        lin = lambda arr: np.stack([np.interp(times, self.t, arr[:, j]) for j in range(arr.shape[1])], 1)
        omega = (lin(self.omega_body) if self.cfg["angular_rate"] == "mocap"
                 else np.stack([np.interp(times, self.imu_t, self.gyro[:, j]) for j in range(3)], 1))
        # mean of the per-message input over (t_k - h, t_k]; an interval without any motor message (the ~100 Hz stream
        # has gaps up to ~40 ms) gives u = 0. Kept for reproducibility of the stored datasets; counted per piece in the
        # audits as `input_intervals_without_motor_message` (see empty_input_intervals).
        csum = np.vstack([np.zeros(self._controls.shape[1]), np.cumsum(self._controls, 0)])
        hi = np.searchsorted(self.motor_t, times, side="right"); lo = np.searchsorted(self.motor_t, times - h, side="right")
        count = np.maximum(hi - lo, 1)[:, None]
        u = (csum[hi] - csum[lo]) / count
        return np.hstack([lin(self.position), rot.reshape(len(times), 9), lin(self.v_body), omega, u])

    def empty_input_intervals(self, times: np.ndarray) -> int:
        """Number of sample intervals (t_k - h, t_k] that contain no motor message (their u is 0 in `rows`)."""
        h = 1.0 / float(self.cfg["sample_hz"])
        hi = np.searchsorted(self.motor_t, times, side="right"); lo = np.searchsorted(self.motor_t, times - h, side="right")
        return int(np.sum(hi == lo))

    def disarmed_fraction(self, times: np.ndarray) -> np.ndarray:
        h = 1.0 / float(self.cfg["sample_hz"])
        c = np.r_[0, np.cumsum(self.disarmed)]
        hi = np.searchsorted(self.motor_t, times, side="right"); lo = np.searchsorted(self.motor_t, times - h, side="right")
        return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)

    def trajectories(self, t_lo: float, t_hi: float) -> tuple[list[np.ndarray], dict]:
        h, cfg = 1.0 / float(self.cfg["sample_hz"]), self.cfg
        length = int(round(cfg["trajectory_seconds"] * cfg["sample_hz"])) + 1
        stride = int(round(cfg["trajectory_stride_seconds"] * cfg["sample_hz"]))
        out, used, dropped_disarmed, empty_input = [], 0.0, 0, 0
        for a, b in self.segments(t_lo, t_hi):
            times = np.arange(a + h, b + 1e-9, h)                    # first row needs one full input interval
            for s in range(0, len(times) - length + 1, stride):
                chunk = times[s:s + length]
                if cfg["disarmed"] == "drop" and np.any(self.disarmed_fraction(chunk[1:]) > 0):
                    dropped_disarmed += 1; continue
                out.append(self.rows(chunk)); used += chunk[-1] - chunk[0]
                empty_input += self.empty_input_intervals(chunk[1:])   # row 0's input is never used
        span = t_hi - t_lo
        return out, {"recording": self.name, "seconds": span, "seconds_in_trajectories": used,
                     "excluded_intervals": int(np.sum((self.gap_start > t_lo) & (self.gap_start < t_hi))),
                     "pose_glitches_in_recording": self.glitches,
                     "trajectories": len(out), "dropped_for_disarmed_motors": dropped_disarmed,
                     "input_intervals_without_motor_message": empty_input}

    def stream(self, t_lo: float, t_hi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = 1.0 / float(self.cfg["sample_hz"])
        times = np.arange(t_lo + h, t_hi + 1e-9, h)
        inside_gap = np.zeros(len(times), bool)
        for gs, ge in zip(self.gap_start, self.gap_end):
            inside_gap |= (times > gs) & (times < ge)
        return times - times[0], self.rows(times), ~inside_gap


# --------------------------------------------------------------------------- assembly
def windows(trajectories: np.ndarray, points: int, stride: int) -> np.ndarray:
    if trajectories.shape[0] == 0:
        return np.zeros((points, 0, trajectories.shape[-1]))
    chunks = [f[s:s + points] for f in trajectories for s in range(0, f.shape[0] - points + 1, stride)]
    return np.transpose(np.stack(chunks), (1, 0, 2))


def coverage(flights: np.ndarray) -> dict:
    flat = flights.reshape(-1, flights.shape[-1])
    tilt = np.degrees(np.arccos(np.clip(flat[:, 11], -1, 1)))
    pct = lambda x: [float(np.percentile(x, q)) for q in (50, 90, 100)]
    return {"position_min": flat[:, :3].min(0).tolist(), "position_max": flat[:, :3].max(0).tolist(),
            "tilt_deg_50_90_max": pct(tilt), "speed_50_90_max": pct(np.linalg.norm(flat[:, 12:15], axis=1)),
            "angular_rate_50_90_max": pct(np.linalg.norm(flat[:, 15:18], axis=1)),
            "input_std": flat[:, 18:].std(0).tolist(), "input_mean": flat[:, 18:].mean(0).tolist()}


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build(cfg: dict) -> dict:
    src = resolve(cfg["sources"])
    load = lambda name: Recording(src / f"{name}.npz", cfg)
    split, parts = cfg["split"], {"train": [], "test": []}
    if split["type"] == "chronological":
        rec = load(split["recording"])
        cut = rec.t[0] + (1.0 - float(split["test_fraction"])) * (rec.t[-1] - rec.t[0])
        parts["train"].append((rec, rec.t[0], cut)); parts["test"].append((rec, cut, rec.t[-1]))
    else:
        for part in ("train", "test"):
            for name in split[part]:
                rec = load(name); parts[part].append((rec, rec.t[0], rec.t[-1]))
    data, audits, streams = {}, {}, {}
    for part, pieces in parts.items():
        flights, audit = [], []
        for rec, lo, hi in pieces:
            f, a = rec.trajectories(lo, hi); flights += f; audit.append(a)
            if cfg["streams"]:
                t, rows, valid = rec.stream(lo, hi)
                streams.setdefault(part, []).append((rec.name, t, rows, valid))
        n_u = 8 if cfg["input"] in ("commands", "thrust") else 6
        data[part] = np.stack(flights) if flights else np.zeros((0, 0, 18 + n_u))
        audits[part] = audit
    n_u = data["train"].shape[-1] - 18
    h = 1.0 / float(cfg["sample_hz"])
    out = cfg["output"]
    payload = {"t": np.arange(out["window_points"]) * h,
               "train_trajectories": data["train"], "test_trajectories": data["test"],
               "x": windows(data["train"], out["window_points"], out["window_stride"]),
               "test_x": windows(data["test"], out["window_points"], out["window_stride"])}
    for part, items in streams.items():
        if len(items) == 1:
            name, t, rows, valid = items[0]
            payload[f"{part}_stream"], payload[f"{part}_stream_time"], payload[f"{part}_stream_valid"] = rows, t, valid
        else:
            payload[f"{part}_streams"] = {name: {"rows": rows, "time": t, "valid": valid} for name, t, rows, valid in items}
    payload["settings"] = {
        "dataset_name": cfg["name"], "config": cfg, "system": "BlueROV2 Heavy, real tank recordings (KTH Marinarium)",
        "source": {"paper": "Torroba et al., Marinarium, arXiv:2602.23053v2 (2026), Sec. IV",
                   "repository": "https://github.com/ViktorNfa/bluerov2_dynamics (MIT), commit 5843178",
                   "recordings": {p: [a["recording"] for a in audits[p]] for p in audits}},
        "mocap_time": cfg["mocap_time"], "glitch_threshold_deg": cfg["glitch_threshold_deg"],
        "frames": "world: motion-capture frame, z DOWN (gravity +e3, checked against the accelerometer); body: forward-right-down",
        "state_layout": "p_w(3), vec(R)(9) body->world row-major, v_b(3), omega_b(3), u(n_u)",
        "control_layout": CONTROL_LAYOUT[cfg["input"]], "control_dim": n_u, "input_mode": cfg["input"],
        "true_control_map": TRUE_MAP[cfg["input"]], "allocation_matrix": allocation_matrix().tolist(),
        "thrusters_px4_order": [{"axis": list(a), "position": list(p)} for a, p in THRUSTERS],
        "input_timing": "u in row k = mean input over (t_{k-1}, t_k]; drives row k-1 -> row k",
        "angular_rate_source": cfg["angular_rate"], "sample_dt": h, "trajectory_points": int(data["train"].shape[1]),
        "vehicle_parameters_nominal": {**PUBLISHED, "note": "published nominal values (von Benzon 2022); NOT ground truth for "
                                       "this vehicle - the Fossen baseline with these values drifts (Marinarium Table 2)"},
        "splits": {p: {"key": f"{p}_trajectories", "trajectories": int(data[p].shape[0]), "pieces": audits[p]} for p in data},
        "audits_train": coverage(data["train"]) if data["train"].size else {}, "audits_test": coverage(data["test"]) if data["test"].size else {},
        "clean_training_state_sha256": sha256(data["train"]), "clean_test_state_sha256": sha256(data["test"]),
        "observation_noise": {"enabled": False, "note": "real measurements; no synthetic noise added"},
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=THIS_DIR / "config.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "datasets",
                        help="parent folder; the dataset goes to <output-root>/ROV-MARINARIUM-DATASET-<name>/")
    parser.add_argument("--force", action="store_true", help="overwrite existing files of the same dataset")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate(cfg)
    name = cfg["name"]
    out_dir = (args.output_root / f"ROV-MARINARIUM-DATASET-{name}").resolve()
    if not args.force and list(out_dir.glob(f"{name}_*")):
        raise FileExistsError(f"{out_dir} already holds {name}_* files; pass --force to overwrite or change `name`")
    payload = build(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}_clean.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    s = payload["settings"]
    (out_dir / f"{name}_audits.json").write_text(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=1, default=float))
    (out_dir / f"{name}_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir}/{name}_clean.pkl")
    for part in ("train", "test"):
        arr = payload[f"{part}_trajectories"]
        pieces = s["splits"][part]["pieces"]
        print(f"  {part:5s} {arr.shape}  from {[p['recording'] for p in pieces]}  "
              f"{sum(p['seconds_in_trajectories'] for p in pieces):.0f} of {sum(p['seconds'] for p in pieces):.0f} s used, "
              f"{sum(p['excluded_intervals'] for p in pieces)} dropout / glitch intervals avoided")
        empty = sum(p["input_intervals_without_motor_message"] for p in pieces)
        if empty:
            print(f"        WARNING: {empty} input intervals without a motor message (u = 0 there), "
                  f"{empty / max(arr.shape[0] * (arr.shape[1] - 1), 1):.3%} of the input rows")
        if f"{part}_stream" in payload:
            print(f"        stream {payload[f'{part}_stream'].shape}, {payload[f'{part}_stream_valid'].mean():.1%} valid")


if __name__ == "__main__":
    main()

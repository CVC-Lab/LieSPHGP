"""Score our model with the IDSIA benchmark's OWN protocol, so the numbers sit next to their Table 7.

Their protocol (Busetto et al., Control Engineering Practice 172 (2026) 106871, section 5.3):
  * held-out Melon trajectory, every admissible start time t
  * H = 50 steps (0.5 s at 100 Hz); the model gets the true state at t and the input sequence, then predicts
    50 steps open loop with no state correction
  * MAE_(.),h = mean over start times of the Euclidean error at horizon h, for position (m), world-frame linear
    velocity (m/s) and body angular velocity (rad/s); orientation uses the geodesic angle in radians
  * MAE_(.),1:H = sum over h = 1..50 of MAE_(.),h        (their italic "cumulative simulation error" column)

Their Table 7, for reference:

  model      MAE_p (h=1 / 10 / 50 / 1:50)        MAE_R (h=1 / 10 / 50 / 1:50)
  Naive      0.0143 0.1430 0.6797 17.7878        0.0071 0.0692 0.3041 8.2138
  Physics    0.0013 0.0126 0.1269  2.3223        0.0011 0.0205 0.2544 5.1013
  Res-MLP    0.0032 0.0305 0.1712  4.0519        0.0022 0.0331 0.2268 6.0591
  Hybrid     0.0016 0.0166 0.1119  2.3625        0.0027 0.0376 0.2306 6.1534
  Res-LSTM   0.0079 0.0390 0.2572  5.9711        0.0066 0.0372 0.2325 5.6525

Usage:
  python benchmark_protocol.py --run <training run dir> [--stride 1] [--horizon 50]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[4]  # comparision/idsia -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.SE3_Quadrotor.comparision import report_evaluation as evaluation  # noqa: E402

KT, KC, ARM = 3.72e-8, 7.74e-12, 0.0353
MOTOR_COLUMNS = [f"m{i}_rads" for i in (1, 2, 3, 4)]
REFERENCE = {                       # their Table 7, h = 1 / 10 / 50 / cumulative 1:50
    "Naive":    {"p": (0.0143, 0.1430, 0.6797, 17.7878), "v": (0.0329, 0.3182, 1.4749, 38.9241),
                 "R": (0.0071, 0.0692, 0.3041, 8.2138),  "w": (0.0796, 0.3596, 0.8837, 29.0866)},
    "Physics":  {"p": (0.0013, 0.0126, 0.1269, 2.3223),  "v": (0.0080, 0.0570, 0.5781, 10.6232),
                 "R": (0.0011, 0.0205, 0.2544, 5.1013),  "w": (0.0796, 0.3596, 0.8837, 29.0866)},
    "Res-MLP":  {"p": (0.0032, 0.0305, 0.1712, 4.0519),  "v": (0.0116, 0.0890, 0.5720, 12.5809),
                 "R": (0.0022, 0.0331, 0.2268, 6.0591),  "w": (0.1138, 0.5949, 0.9005, 35.5735)},
    "Hybrid":   {"p": (0.0016, 0.0166, 0.1119, 2.3625),  "v": (0.0092, 0.0613, 0.5556, 10.4033),
                 "R": (0.0027, 0.0376, 0.2306, 6.1534),  "w": (0.0912, 0.4880, 0.5979, 28.9873)},
    "Res-LSTM": {"p": (0.0079, 0.0390, 0.2572, 5.9711),  "v": (0.0247, 0.1175, 0.7407, 16.7207),
                 "R": (0.0066, 0.0372, 0.2325, 5.6525),  "w": (0.1021, 0.4292, 1.2407, 35.8353)},
}


def input_mode_of_run(run: dict) -> str:
    """'wrench' or 'rotor2', read from the settings of the pickle the run was trained on."""
    import pickle
    path = PROJECT_ROOT / run["config"]["data"]["dataset_path"]
    with path.open("rb") as handle:
        return pickle.load(handle)["settings"].get("input_mode", "wrench")


def controls(speeds: np.ndarray, input_mode: str) -> np.ndarray:
    squared = speeds ** 2
    if input_mode == "rotor2":
        return squared * (4.0 * KT / (0.045 * 9.81))            # convert_idsia.ROTOR2_SCALE
    return np.column_stack([
        KT * squared.sum(axis=1),
        KT * ARM * ((squared[:, 2] + squared[:, 3]) - (squared[:, 0] + squared[:, 1])),
        KT * ARM * ((squared[:, 1] + squared[:, 2]) - (squared[:, 0] + squared[:, 3])),
        KC * ((squared[:, 0] + squared[:, 2]) - (squared[:, 1] + squared[:, 3]))])


def load_melon(source: Path, input_mode: str = "wrench", pattern: str = "melon*.csv") -> list[np.ndarray]:
    """Flights matching ``pattern`` under ``source``.  The held-out set is melon*, but the same protocol
    also runs on the training shapes by pointing --source at data/train with --pattern "*.csv".
    """
    flights = []
    for path in sorted(source.glob(pattern)):
        frame = pd.read_csv(path)
        rotation = Rotation.from_quat(frame[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
        control = controls(frame[MOTOR_COLUMNS].to_numpy(dtype=np.float64), input_mode)
        velocity_world = frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
        flights.append(np.concatenate([
            frame[["x", "y", "z"]].to_numpy(dtype=np.float64), rotation.reshape(-1, 9),
            np.einsum("nji,nj->ni", rotation, velocity_world),
            frame[["wx", "wy", "wz"]].to_numpy(dtype=np.float64), control], axis=1))
    return flights


def rolling_windows(flights: list[np.ndarray], horizon: int, stride: int) -> np.ndarray:
    pieces = [flight[start:start + horizon + 1]
              for flight in flights
              for start in range(0, flight.shape[0] - horizon, stride)]
    return np.stack(pieces)


def per_horizon(truth: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    """MAE at each horizon h = 1..H, the benchmark's definition (mean Euclidean norm, not RMS)."""
    position = np.linalg.norm(prediction[:, 1:, :3] - truth[:, 1:, :3], axis=-1).mean(axis=0)
    rotations_t = truth[:, 1:, 3:12].reshape(truth.shape[0], -1, 3, 3)
    rotations_p = prediction[:, 1:, 3:12].reshape(truth.shape[0], -1, 3, 3)
    velocity_t = np.einsum("fhij,fhj->fhi", rotations_t, truth[:, 1:, 12:15])
    velocity_p = np.einsum("fhij,fhj->fhi", rotations_p, prediction[:, 1:, 12:15])
    velocity = np.linalg.norm(velocity_p - velocity_t, axis=-1).mean(axis=0)
    omega = np.linalg.norm(prediction[:, 1:, 15:18] - truth[:, 1:, 15:18], axis=-1).mean(axis=0)
    relative = np.einsum("fhji,fhjk->fhik", rotations_t, rotations_p)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    orientation = np.arccos(cosine).mean(axis=0)
    return {"p": position, "v": velocity, "R": orientation, "w": omega}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--selected-step", type=int, default=None)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "tmp/idsia_raw/data/test")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--label", default="PH-GP-LieIMEX (ours)")
    parser.add_argument("--pattern", default="melon*.csv", help="glob for the flights to score")
    arguments = parser.parse_args()

    run = evaluation.load_run(Path(arguments.run), arguments.selected_step)
    input_mode = input_mode_of_run(run)
    flights = load_melon(arguments.source, input_mode, arguments.pattern)
    if not flights:
        raise SystemExit(f"no flights matching {arguments.pattern!r} under {arguments.source}")
    windows = rolling_windows(flights, arguments.horizon, arguments.stride)
    print(f"\nMelon test trajectory: {len(flights)} runs, {sum(f.shape[0] for f in flights)} samples, "
          f"input = {input_mode}")
    print(f"rolling windows: {windows.shape[0]} starts (stride {arguments.stride}), H = {arguments.horizon} steps\n")

    model, _params, _setup = evaluation.build_model(run)
    curves = []
    for begin in range(0, windows.shape[0], arguments.batch):
        chunk = windows[begin:begin + arguments.batch]
        try:
            prediction = np.asarray(evaluation.rollout(model, chunk, 0.01))
        except FloatingPointError:
            # the shared rollout raises on any non-finite window; redo this chunk window by window so the
            # finite ones are kept and the diverged ones are counted instead of aborting the whole score
            prediction = np.full((chunk.shape[0], chunk.shape[1], 18), np.nan)   # the rollout returns 18 state columns
            for j in range(chunk.shape[0]):
                try:
                    prediction[j] = np.asarray(evaluation.rollout(model, chunk[j:j + 1], 0.01))[0]
                except FloatingPointError:
                    pass
        curves.append((chunk, prediction))
        print(f"  rolled {begin + chunk.shape[0]}/{windows.shape[0]}", end="\r", flush=True)
    truth = np.concatenate([c for c, _ in curves])
    prediction = np.concatenate([p for _, p in curves])
    # A window is DIVERGED when its rollout is non-finite or physically absurd (position error > 10 m, |omega| > 100
    # rad/s, |v| > 50 m/s; the data maxima are 3.2 m, 8.4 rad/s, 3.2 m/s). Such windows are excluded from the MAEs
    # and their share is printed: the published baselines have no such exclusion, so a non-zero share weakens the row.
    with np.errstate(invalid="ignore", over="ignore"):
        deviation = np.nan_to_num(np.linalg.norm(prediction[..., :3] - truth[..., :3], axis=-1), nan=np.inf).max(axis=1)
        omega = np.nan_to_num(np.linalg.norm(prediction[..., 15:18], axis=-1), nan=np.inf).max(axis=1)
        speed = np.nan_to_num(np.linalg.norm(prediction[..., 12:15], axis=-1), nan=np.inf).max(axis=1)
    finite = np.isfinite(prediction).all(axis=(1, 2)) & (deviation <= 10.0) & (omega <= 100.0) & (speed <= 50.0)
    if not finite.all():
        print(f"  WARNING: {int((~finite).sum())} of {finite.size} windows ({100 * (~finite).mean():.2f} %) DIVERGED "
              f"(non-finite: {int((~np.isfinite(prediction).all(axis=(1, 2))).sum())}) and are EXCLUDED from the MAEs below")
        truth, prediction = truth[finite], prediction[finite]
    ours = per_horizon(truth, prediction)
    print(" " * 40, end="\r")

    names = {"p": "MAE_p [m]", "v": "MAE_v [m/s]", "R": "MAE_R [rad]", "w": "MAE_w [rad/s]"}
    for key, title in names.items():
        print(f"{title}")
        print(f"   {'model':<26}{'h=1':>10}{'h=10':>10}{'h=50':>10}{'h=1:50':>12}")
        for name, values in REFERENCE.items():
            a, b, c, total = values[key]
            print(f"   {name:<26}{a:>10.4f}{b:>10.4f}{c:>10.4f}{total:>12.4f}")
        curve = ours[key]
        print(f"   {arguments.label:<26}{curve[0]:>10.4f}{curve[9]:>10.4f}{curve[-1]:>10.4f}{curve.sum():>12.4f}")
        print()


if __name__ == "__main__":
    main()

"""Step 1: Marinarium rosbag2 recording -> one npz of raw, unresampled time series (BlueROV2 Heavy, KTH tank).

    python envs/rov_se3_marinarium/datagen/bag_to_npz.py --bag <rosbag2 folder> --out <file.npz>
    python envs/rov_se3_marinarium/datagen/bag_to_npz.py --all        # the three published recordings -> marinarium_raw/npz/

Every stream keeps its own clock (bag receive time, seconds since the first motion-capture message) and its own rate;
step 2 (generate_dataset.py) resamples. Needs `rosbags` (pip install rosbags).

Streams written (frames checked on the manual recording, see README.md):
    mocap_*   /mocap/itrl_rov_1/odom, ~100 Hz: position p (world, z DOWN), quaternion q (x, y, z, w, body -> world),
              body-frame linear and angular velocity; mocap_t = bag receive time, mocap_t_header = the system's own
              stamp (steadier) shifted onto the receive clock by its median latency. The odom has dropouts (manual: 191 gaps > 30 ms, longest 6.45 s).
    imu_*     /itrl_rov_1/fmu/out/sensor_combined, 100 Hz: PX4 gyro (rad/s) and accelerometer (m/s^2), same body axes.
    motor_*   /itrl_rov_1/fmu/out/actuator_motors, 100 Hz: 8 normalised commands in [-1, 1]; NaN = motor not commanded.
    battery_* /itrl_rov_1/fmu/out/battery_status_v1, 1 Hz: voltage (V) and current (A). The bag has no type definition
              for this message, so the two floats are read from the raw CDR bytes (offsets 16 and 20).
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ENV_DIR = THIS_DIR.parent
RAW_DIR = ENV_DIR / "marinarium_raw"
TYPES_DIR = RAW_DIR / "rosbags" / "types" / "px4_msgs" / "msg"

RECORDINGS = {                                    # name -> rosbag2 folder inside marinarium_raw/rosbags
    "oct30": "rosbag2_2025_10_30/rosbag2_2025_10_30-16_31_20",
    "manual": "rosbag2_2025_11_06/rosbag2_2025_11_06-manual",
    "stabilized": "rosbag2_2025_11_06/rosbag2_2025_11_06-stabilized",
}
TOPIC_ODOM = "/mocap/itrl_rov_1/odom"
TOPIC_IMU = "/itrl_rov_1/fmu/out/sensor_combined"
TOPIC_MOTORS = "/itrl_rov_1/fmu/out/actuator_motors"
TOPIC_BATTERY = "/itrl_rov_1/fmu/out/battery_status_v1"
MOTORS = 8


def typestore():
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    store = get_typestore(Stores.ROS2_HUMBLE)
    extra = {}
    for f in TYPES_DIR.glob("*.msg"):
        extra.update(get_types_from_msg(f.read_text(), f"px4_msgs/msg/{f.stem}"))
    store.register(extra)
    return store


def read_battery(bag: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bag time ns, voltage V, current A) from the raw CDR of px4_msgs/BatteryStatus (no type definition in the bag):
    4-byte CDR header, uint64 timestamp at 4, bool connected at 12, float32 voltage_v at 16, float32 current_a at 20."""
    times, volts, amps = [], [], []
    for db in sorted(bag.glob("*.db3")):
        con = sqlite3.connect(db)
        row = con.execute("select id from topics where name=?", (TOPIC_BATTERY,)).fetchone()
        if row is not None:
            for stamp, data in con.execute("select timestamp, data from messages where topic_id=? order by timestamp", row):
                times.append(stamp); volts.append(struct.unpack_from("<f", data, 16)[0]); amps.append(struct.unpack_from("<f", data, 20)[0])
        con.close()
    return np.asarray(times, np.int64), np.asarray(volts, float), np.asarray(amps, float)


def convert(bag: Path) -> dict[str, np.ndarray]:
    from rosbags.highlevel import AnyReader
    odom, imu, motors = [], [], []
    with AnyReader([bag], default_typestore=typestore()) as reader:
        conns = {c.topic: c for c in reader.connections}
        missing = [t for t in (TOPIC_ODOM, TOPIC_IMU, TOPIC_MOTORS) if t not in conns]
        if missing:
            raise RuntimeError(f"{bag}: missing topics {missing}")
        for conn, stamp, raw in reader.messages(connections=[conns[TOPIC_ODOM], conns[TOPIC_IMU], conns[TOPIC_MOTORS]]):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == TOPIC_ODOM:
                p, q, tw = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
                header = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                odom.append((stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w, tw.linear.x, tw.linear.y, tw.linear.z,
                             tw.angular.x, tw.angular.y, tw.angular.z, header))
            elif conn.topic == TOPIC_IMU:
                imu.append((stamp, *msg.gyro_rad, *msg.accelerometer_m_s2))
            else:
                motors.append((stamp, *np.asarray(msg.control[:MOTORS], float)))
    odom, imu, motors = (np.asarray(sorted(a), float) for a in (odom, imu, motors))
    t_bat, volt, amp = read_battery(bag)
    t0 = odom[0, 0]
    seconds = lambda ns: (np.asarray(ns, float) - t0) * 1e-9
    return {
        "mocap_t": seconds(odom[:, 0]), "mocap_position": odom[:, 1:4], "mocap_quaternion_xyzw": odom[:, 4:8],
        "mocap_v_body": odom[:, 8:11], "mocap_omega_body": odom[:, 11:14],
        # header stamp of the motion-capture system, shifted so its median offset to the receive clock is zero:
        # steadier spacing (1-99 % of steps 7.5-12.5 ms vs 6-14 ms), same clock origin as the PX4 streams
        "mocap_t_header": seconds(odom[:, 14] + np.median(odom[:, 0] - odom[:, 14])),
        "imu_t": seconds(imu[:, 0]), "imu_gyro": imu[:, 1:4], "imu_accel": imu[:, 4:7],
        "motor_t": seconds(motors[:, 0]), "motor_command": motors[:, 1:],
        "battery_t": seconds(t_bat), "battery_voltage": volt, "battery_current": amp,
        "t0_unix_ns": np.asarray(odom[0, 0], np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag", type=Path, help="rosbag2 folder")
    parser.add_argument("--out", type=Path, help="output npz")
    parser.add_argument("--all", action="store_true", help="convert the three published recordings into marinarium_raw/npz/")
    parser.add_argument("--force", action="store_true", help="overwrite existing npz files")
    args = parser.parse_args()
    jobs = ([(RAW_DIR / "rosbags" / folder, RAW_DIR / "npz" / f"{name}.npz") for name, folder in RECORDINGS.items()]
            if args.all else [(args.bag, args.out)])
    if any(b is None or o is None for b, o in jobs):
        parser.error("give --bag and --out, or --all")
    for bag, out in jobs:
        if out.exists() and not args.force:
            print(f"skip {out} (exists; --force to overwrite)"); continue
        data = convert(bag)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **data)
        gaps = np.diff(data["mocap_t"])
        print(f"{out.name}: mocap {len(data['mocap_t'])} ({1 / np.median(gaps):.0f} Hz, {int(np.sum(gaps > 0.03))} gaps > 30 ms, "
              f"longest {gaps.max():.2f} s) | imu {len(data['imu_t'])} | motors {len(data['motor_t'])} | battery "
              f"{len(data['battery_t'])} ({data['battery_voltage'].min():.2f}-{data['battery_voltage'].max():.2f} V) | "
              f"{data['mocap_t'][-1]:.0f} s")


if __name__ == "__main__":
    main()

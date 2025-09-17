#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from typing import Dict, List

import numpy as np


# Ensure source tree import works
sys.path.append("/home/hls/lerobot/src")

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig


DEFAULT_TRAJ = "/home/hls/lerobot/tmp/traj_x.csv"
DEFAULT_URDF = "/home/hls/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
DEFAULT_EE = "gripper_frame_link"
IK_PREVIEW_OUT = "/home/hls/lerobot/tmp/joints_ik_preview.csv"

JOINT_NAMES: List[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
CSV_JOINT_COLS: List[str] = [
    "shoulder_pan_deg",
    "shoulder_lift_deg",
    "elbow_flex_deg",
    "wrist_flex_deg",
    "wrist_roll_deg",
    "gripper_deg",
]


def autodetect_urdf() -> str:
    # Prefer so101 new, then old, then so100
    candidates = [
        "/home/hls/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        "/home/hls/lerobot/SO-ARM100/Simulation/SO101/so101_old_calib.urdf",
        "/home/hls/lerobot/SO-ARM100/Simulation/SO100/so100.urdf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback: search any .urdf under SO-ARM100
    for root, _, files in os.walk("/home/hls/lerobot/SO-ARM100"):
        for f in files:
            if f.endswith(".urdf"):
                return os.path.join(root, f)
    return DEFAULT_URDF


def read_traj_xyz(traj_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(traj_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise RuntimeError(f"Empty CSV: {traj_path}")
    hdr = {n: i for i, n in enumerate(rows[0])}
    for k in ("t", "x", "y", "z"):
        if k not in hdr:
            raise RuntimeError(f"Missing column '{k}' in {traj_path}")
    t_list, xyz_list = [], []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        t_list.append(float(r[hdr["t"]]))
        xyz_list.append([float(r[hdr["x"]]), float(r[hdr["y"]]), float(r[hdr["z"]])])
    t = np.asarray(t_list, dtype=float)
    xyz = np.asarray(xyz_list, dtype=float)
    if len(t) != len(xyz):
        raise RuntimeError("t and xyz length mismatch")
    return t, xyz


def get_present_joint_degrees(robot: SO100Follower) -> np.ndarray:
    obs = robot.get_observation()
    return np.asarray([float(obs[f"{n}.pos"]) for n in JOINT_NAMES], dtype=float)


def build_action_from_degrees(q_deg: np.ndarray) -> Dict[str, float]:
    return {f"{n}.pos": float(q_deg[i]) for i, n in enumerate(JOINT_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive SO101 EE along a Cartesian trajectory via IK")
    parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ, help="CSV with t,x,y,z")
    parser.add_argument("--urdf", type=str, default=None, help="URDF path (auto-detect if omitted)")
    parser.add_argument("--ee", type=str, default=DEFAULT_EE, help="End-effector frame name in URDF")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port for robot")
    parser.add_argument("--robot-id", type=str, default="so101", help="Calibration/profile id to load")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Time scaling (>1 faster)")
    parser.add_argument("--preserve-gripper", action="store_true", help="Hold current gripper during IK")
    parser.add_argument("--dry-run", action="store_true", help="Compute IK only; do not move robot")
    parser.add_argument(
        "--record-out",
        type=str,
        default="/home/hls/lerobot/tmp/joints_recorded.csv",
        help="CSV file to record executed joint positions during the run",
    )
    args = parser.parse_args()

    traj_path = args.traj
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    urdf_path = args.urdf if args.urdf else autodetect_urdf()
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    print(f"Using URDF: {urdf_path}\nEE frame: {args.ee}")

    t, xyz = read_traj_xyz(traj_path)
    kin = RobotKinematics(urdf_path=urdf_path, target_frame_name=args.ee, joint_names=JOINT_NAMES)

    # Precompute IK
    T = np.eye(4)
    q_list: List[np.ndarray] = []
    # Seed with zeros; when running live we will overwrite with present pose
    q_curr = np.zeros(len(JOINT_NAMES), dtype=float)
    preserved_gripper = float(q_curr[-1])
    for p in xyz:
        T[:3, 3] = p
        q_next = kin.inverse_kinematics(q_curr, T, position_weight=1.0, orientation_weight=0.0)
        if args.preserve_gripper:
            q_next[-1] = preserved_gripper
        q_list.append(q_next.copy())
        q_curr = q_next

    # Always emit a preview CSV
    with open(IK_PREVIEW_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", *CSV_JOINT_COLS])
        for tt, q in zip(t, q_list):
            w.writerow([tt, *[f"{v:.6f}" for v in q]])
    print(f"Saved IK preview to {IK_PREVIEW_OUT}")

    if args.dry_run:
        print("Dry-run finished (no motion).")
        return

    # Execute on robot
    cfg = SO100FollowerConfig(id=args.robot_id, port=args.port, cameras={}, use_degrees=True)
    robot = SO100Follower(cfg)
    # Enable calibration flow if the motors don't have calibration yet.
    # If a calibration file exists for --robot-id, it will be written to the motors; otherwise an
    # interactive calibration will be started by the driver.
    robot.connect(calibrate=True)
    try:
        q_curr = get_present_joint_degrees(robot)
        if args.preserve_gripper:
            preserved_gripper = float(q_curr[-1])

        t0 = time.perf_counter()
        # Prepare recorder
        rec_f = open(args.record_out, "w", newline="")
        rec_w = csv.writer(rec_f)
        rec_w.writerow(["t", *CSV_JOINT_COLS])

        for i, q_goal in enumerate(q_list):
            if args.preserve_gripper:
                q_goal[-1] = preserved_gripper
            action = build_action_from_degrees(q_goal)
            robot.send_action(action)

            # Read back present positions and record
            present = get_present_joint_degrees(robot)
            rec_t = time.perf_counter() - t0
            rec_w.writerow([f"{rec_t:.6f}", *[f"{v:.6f}" for v in present]])
            rec_f.flush()

            if i + 1 < len(t):
                step = max((t[i + 1] - t[i]) / max(args.speed_scale, 1e-6), 0.0)
                deadline = t0 + t[i + 1] / max(args.speed_scale, 1e-6)
                now = time.perf_counter()
                sleep_s = max(deadline - now, 0.0) if step > 0 else 0.0
                if sleep_s > 0:
                    time.sleep(sleep_s)
        print("Trajectory execution completed.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

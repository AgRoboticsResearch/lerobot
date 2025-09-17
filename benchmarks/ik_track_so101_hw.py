#!/usr/bin/env python

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project src on path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def generate_traj(center_T: np.ndarray, num_points: int, radius_m: float, z_amplitude_m: float, cycles: float) -> np.ndarray:
    N = num_points
    poses = np.repeat(center_T[None, :, :], N, axis=0)
    center_pos = center_T[:3, 3].copy()
    theta = np.linspace(0.0, 2.0 * math.pi * cycles, N, endpoint=False)
    x = center_pos[0] + radius_m * np.cos(theta)
    y = center_pos[1] + radius_m * np.sin(theta)
    z = center_pos[2] + z_amplitude_m * np.sin(2.0 * theta)
    poses[:, 0, 3] = x
    poses[:, 1, 3] = y
    poses[:, 2, 3] = z
    return poses


def main():
    parser = argparse.ArgumentParser(description="Track a planned EE trajectory on real SO101 using IK.")
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--target_frame_name", type=str, default="gripper_frame_link")
    parser.add_argument("--joint_names", type=str, default="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--num_points", type=int, default=200)
    parser.add_argument("--radius_m", type=float, default=0.03)
    parser.add_argument("--z_amplitude_m", type=float, default=0.02)
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--position_weight", type=float, default=1.0)
    parser.add_argument("--orientation_weight", type=float, default=0.0)
    parser.add_argument("--max_relative_target_deg", type=float, default=5.0, help="Safety cap per joint per step")
    parser.add_argument("--move_to_mid", action="store_true", help="Move to mid-range joint configuration before starting trajectory")
    parser.add_argument("--move_to_mid_timeout_s", type=float, default=20.0)
    parser.add_argument("--move_to_mid_tolerance_deg", type=float, default=2.0)
    parser.add_argument("--out_dir", type=str, default="./ik_track_hw_out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_names = [s for s in args.joint_names.split(",") if s]

    # Initialize hardware robot (degrees for IK compatibility) with safety cap
    robot_cfg = SO101FollowerConfig(
        port=args.port,
        id="so101_hw",
        use_degrees=True,
        max_relative_target=args.max_relative_target_deg,
        cameras={},
    )
    robot = SO101Follower(robot_cfg)

    # Initialize kinematics
    kin = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame_name,
        joint_names=joint_names,
    )

    robot.connect()

    # Read initial joints
    present = robot.bus.sync_read("Present_Position")
    q_meas = np.array([present[n] for n in joint_names], dtype=np.float64)
    gripper_pos = float(present["gripper"])  # keep gripper constant

    # Optionally move to mid-range configuration using calibration ranges
    if args.move_to_mid:
        if getattr(robot, "calibration", None):
            mid_targets = np.array(
                [
                    (robot.calibration[name].range_min + robot.calibration[name].range_max) / 2.0
                    for name in joint_names
                ],
                dtype=np.float64,
            )
        else:
            # Fallback: keep current if no calibration info
            mid_targets = q_meas.copy()

        action = {f"{name}.pos": float(val) for name, val in zip(joint_names, mid_targets)}
        action["gripper.pos"] = gripper_pos
        robot.send_action(action)

        # Wait until close or timeout
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.move_to_mid_timeout_s:
            present = robot.bus.sync_read("Present_Position")
            q_now = np.array([present[n] for n in joint_names], dtype=np.float64)
            if np.max(np.abs(q_now - mid_targets)) <= args.move_to_mid_tolerance_deg:
                q_meas = q_now
                break
            time.sleep(1.0 / max(args.fps, 1.0))

    # Record start (mid) joints and EE pose
    start_mid_joints_deg = q_meas.copy()
    start_mid_T = kin.forward_kinematics(start_mid_joints_deg)
    start_mid_ee_xyz = start_mid_T[:3, 3].copy()
    # Save a small CSV for the starting pose
    with open(out_dir / "start_mid_pose.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*joint_names, "ee_x", "ee_y", "ee_z"]) 
        writer.writerow([*start_mid_joints_deg.tolist(), *start_mid_ee_xyz.tolist()])

    center_T = kin.forward_kinematics(q_meas)
    desired_poses = generate_traj(center_T, args.num_points, args.radius_m, args.z_amplitude_m, args.cycles)

    achieved_xyz = []
    desired_xyz = []
    commanded_joints = []
    measured_joints = []

    dt = 1.0 / args.fps
    try:
        for i, T_des in enumerate(desired_poses):
            step_start = time.perf_counter()

            # Read current joints for IK seed and logging
            present = robot.bus.sync_read("Present_Position")
            q_meas = np.array([present[n] for n in joint_names], dtype=np.float64)
            measured_joints.append(q_meas.copy())

            # Log current achieved position (before issuing new command)
            T_meas = kin.forward_kinematics(q_meas)
            achieved_xyz.append(T_meas[:3, 3].copy())
            desired_xyz.append(T_des[:3, 3].copy())

            # IK to next desired pose (position-only by default)
            q_cmd = kin.inverse_kinematics(
                current_joint_pos=q_meas,
                desired_ee_pose=T_des,
                position_weight=args.position_weight,
                orientation_weight=args.orientation_weight,
            )
            commanded_joints.append(q_cmd.copy())

            action = {f"{name}.pos": float(val) for name, val in zip(joint_names, q_cmd)}
            action["gripper.pos"] = gripper_pos

            robot.send_action(action)

            # Timing
            remaining = dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        robot.disconnect()

    achieved_xyz = np.stack(achieved_xyz, axis=0)
    desired_xyz = np.stack(desired_xyz, axis=0)
    commanded_joints = np.stack(commanded_joints, axis=0)
    measured_joints = np.stack(measured_joints, axis=0)

    pos_err = achieved_xyz - desired_xyz
    pos_err_norm = np.linalg.norm(pos_err, axis=1)
    stats = {
        "mean_pos_err_m": float(pos_err_norm.mean()),
        "median_pos_err_m": float(np.median(pos_err_norm)),
        "max_pos_err_m": float(pos_err_norm.max()),
        "min_pos_err_m": float(pos_err_norm.min()),
    }

    # Save CSV
    csv_path = out_dir / "hw_trajectory_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "idx",
            "des_x","des_y","des_z",
            "ach_x","ach_y","ach_z",
            "err_x","err_y","err_z","err_norm",
        ] + [f"q{i}_cmd_deg" for i in range(commanded_joints.shape[1])] + [f"q{i}_meas_deg" for i in range(measured_joints.shape[1])]
        writer.writerow(header)
        for i in range(desired_xyz.shape[0]):
            row = [
                i,
                desired_xyz[i, 0], desired_xyz[i, 1], desired_xyz[i, 2],
                achieved_xyz[i, 0], achieved_xyz[i, 1], achieved_xyz[i, 2],
                pos_err[i, 0], pos_err[i, 1], pos_err[i, 2], pos_err_norm[i],
            ] + list(commanded_joints[i]) + list(measured_joints[i])
            writer.writerow(row)

    # Save NPZ
    np.savez(
        out_dir / "hw_ik_track_results.npz",
        desired_xyz=desired_xyz,
        achieved_xyz=achieved_xyz,
        commanded_joints_deg=commanded_joints,
        measured_joints_deg=measured_joints,
        start_mid_joints_deg=start_mid_joints_deg,
        start_mid_ee_xyz=start_mid_ee_xyz,
        position_error=pos_err,
        position_error_norm=pos_err_norm,
        config=json.dumps({
            "urdf_path": args.urdf_path,
            "port": args.port,
            "target_frame_name": args.target_frame_name,
            "joint_names": joint_names,
            "fps": args.fps,
            "num_points": args.num_points,
            "radius_m": args.radius_m,
            "z_amplitude_m": args.z_amplitude_m,
            "cycles": args.cycles,
            "position_weight": args.position_weight,
            "orientation_weight": args.orientation_weight,
            "max_relative_target_deg": args.max_relative_target_deg,
        }),
        allow_pickle=True,
    )

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(desired_xyz[:, 0], desired_xyz[:, 1], desired_xyz[:, 2], label="desired", c="C0")
        ax.plot(achieved_xyz[:, 0], achieved_xyz[:, 1], achieved_xyz[:, 2], label="achieved", c="C1")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "hw_trajectory_plot.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    print("Hardware IK tracking done. Output in:", out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()



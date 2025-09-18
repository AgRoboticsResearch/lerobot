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


def generate_circle_traj(center_T: np.ndarray, num_points: int, radius_m: float, z_amplitude_m: float, cycles: float) -> np.ndarray:
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


def generate_line_traj(center_T: np.ndarray, num_points: int, axis: str, amplitude_m: float, cycles: float) -> np.ndarray:
    """Generate a straight-line EE path along one axis with smooth reversals.

    The EE moves along a line segment of length 2*amplitude_m centered at center_T position,
    following a sinusoidal progression (smooth turnarounds), so it always stays on a straight line.
    """
    N = num_points
    poses = np.repeat(center_T[None, :, :], N, axis=0)
    center_pos = center_T[:3, 3].copy()
    t = np.linspace(0.0, 2.0 * math.pi * cycles, N, endpoint=False)
    delta = amplitude_m * np.sin(t)
    xyz = np.tile(center_pos, (N, 1))
    if axis == "x":
        xyz[:, 0] += delta
    elif axis == "y":
        xyz[:, 1] += delta
    else:
        xyz[:, 2] += delta
    poses[:, 0, 3] = xyz[:, 0]
    poses[:, 1, 3] = xyz[:, 1]
    poses[:, 2, 3] = xyz[:, 2]
    return poses


def generate_line_xyz_traj(center_T: np.ndarray, num_points_total: int, amplitude_m: float, cycles: float, per_axis_points: int | None = None) -> np.ndarray:
    """Concatenate straight-line segments along X, Y, and Z (in that order).

    - Each segment is a sinusoidal sweep around the center along a single axis,
      keeping orientation fixed at center_T.
    - If per_axis_points is None, points are split evenly among axes.
    """
    if per_axis_points is None:
        base = max(1, num_points_total // 3)
        remainder = max(0, num_points_total - base * 3)
        counts = [base, base, base]
        for i in range(remainder):
            counts[i] += 1
    else:
        counts = [per_axis_points, per_axis_points, per_axis_points]

    segs = []
    for axis, n in zip(["x", "y", "z"], counts, strict=False):
        segs.append(generate_line_traj(center_T, n, axis, amplitude_m, cycles))
    return np.concatenate(segs, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Track a planned EE trajectory on real SO101 using IK.")
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--target_frame_name", type=str, default="gripper_frame_link")
    parser.add_argument("--joint_names", type=str, default="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--num_points", type=int, default=200)
    parser.add_argument("--traj_type", type=str, choices=["circle", "line", "line_xyz"], default="circle")
    parser.add_argument("--line_axis", type=str, choices=["x", "y", "z"], default="x")
    parser.add_argument("--line_amplitude_m", type=float, default=0.05)
    parser.add_argument("--num_points_per_axis", type=int, default=None, help="Override points per axis for line_xyz")
    parser.add_argument("--two_phase", action="store_true", help="Phase 1: line_xyz with orientation_weight=0, then Phase 2: curve using provided orientation_weight")
    parser.add_argument("--radius_m", type=float, default=0.03)
    parser.add_argument("--z_amplitude_m", type=float, default=0.02)
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--position_weight", type=float, default=1.0)
    parser.add_argument("--orientation_weight", type=float, default=0.0)
    parser.add_argument("--max_relative_target_deg", type=float, default=5.0, help="Safety cap per joint per step")
    parser.add_argument("--ramp_in_s", type=float, default=1.0, help="Seconds to ramp from current pose to first planned point before main trajectory")
    parser.add_argument("--ramp_orientation", action="store_true", help="If set, ramp orientation weight from 0 to --orientation_weight during ramp-in")
    parser.add_argument("--out_dir", type=str, default="./ik_track_hw_out")
    parser.add_argument("--joint_traj_npz", type=str, default=None, help="Path to NPZ with precomputed joint trajectory (keys: ik_joints_deg or commanded_joints_deg)")
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

    # Start pose is whatever is measured now

    center_T = kin.forward_kinematics(q_meas)
    # If a precomputed joint trajectory is provided, skip pose generation and IK
    precomputed_joint_traj = None
    if args.joint_traj_npz is not None:
        data = np.load(args.joint_traj_npz, allow_pickle=True)
        if "ik_joints_deg" in data:
            precomputed_joint_traj = data["ik_joints_deg"]
        elif "commanded_joints_deg" in data:
            precomputed_joint_traj = data["commanded_joints_deg"]
        else:
            raise ValueError("NPZ must contain 'ik_joints_deg' or 'commanded_joints_deg'")
        if precomputed_joint_traj.ndim != 2:
            raise ValueError("Joint trajectory must be 2D (steps x joints)")
        if precomputed_joint_traj.shape[1] != len(joint_names):
            raise ValueError("Joint count mismatch: npz joints vs joint_names")
    
    # Build desired poses and per-step orientation weights (only if not precomputed joints)
    if precomputed_joint_traj is None and args.two_phase:
        n1 = max(1, args.num_points // 2)
        n2 = max(1, args.num_points - n1)
        # For phase 1, ensure total points equals n1 irrespective of per-axis override
        phase1 = generate_line_xyz_traj(center_T, n1, args.line_amplitude_m, args.cycles, per_axis_points=None)
        # Phase 2 uses the selected traj_type (default circle)
        if args.traj_type == "circle":
            phase2 = generate_circle_traj(center_T, n2, args.radius_m, args.z_amplitude_m, args.cycles)
        elif args.traj_type == "line":
            phase2 = generate_line_traj(center_T, n2, args.line_axis, args.line_amplitude_m, args.cycles)
        else:
            phase2 = generate_line_xyz_traj(center_T, n2, args.line_amplitude_m, args.cycles, args.num_points_per_axis)
        desired_poses = np.concatenate([phase1, phase2], axis=0)
        ow_sequence = [0.0] * phase1.shape[0] + [args.orientation_weight] * phase2.shape[0]
    elif precomputed_joint_traj is None:
        if args.traj_type == "circle":
            desired_poses = generate_circle_traj(center_T, args.num_points, args.radius_m, args.z_amplitude_m, args.cycles)
        elif args.traj_type == "line":
            desired_poses = generate_line_traj(center_T, args.num_points, args.line_axis, args.line_amplitude_m, args.cycles)
        else:
            desired_poses = generate_line_xyz_traj(center_T, args.num_points, args.line_amplitude_m, args.cycles, args.num_points_per_axis)
        ow_sequence = [args.orientation_weight] * desired_poses.shape[0]
    else:
        desired_poses = None
        ow_sequence = None

    achieved_xyz = []
    desired_xyz = []
    expected_xyz = []
    commanded_joints = []
    measured_joints = []

    dt = 1.0 / args.fps
    try:
        # Optional ramp-in to avoid large first step (only for pose-based mode)
        ramp_steps = int(max(0.0, args.ramp_in_s) * args.fps)
        if ramp_steps > 0 and (desired_poses is not None) and desired_poses.shape[0] > 0:
            T_start = center_T.copy()
            T_first = desired_poses[0]
            for k in range(ramp_steps):
                step_start = time.perf_counter()
                alpha = (k + 1) / ramp_steps

                # Blend position; keep orientation equal to start (trajectory already uses start orientation)
                T_blend = np.eye(4)
                T_blend[:3, :3] = T_start[:3, :3]
                T_blend[:3, 3] = (1 - alpha) * T_start[:3, 3] + alpha * T_first[:3, 3]

                present_seed = robot.bus.sync_read("Present_Position")
                q_seed = np.array([present_seed[n] for n in joint_names], dtype=np.float64)

                ow = args.orientation_weight * alpha if args.ramp_orientation else 0.0
                q_cmd = kin.inverse_kinematics(
                    current_joint_pos=q_seed,
                    desired_ee_pose=T_blend,
                    position_weight=args.position_weight,
                    orientation_weight=ow,
                )

                T_cmd = kin.forward_kinematics(q_cmd)
                action = {f"{name}.pos": float(val) for name, val in zip(joint_names, q_cmd)}
                action["gripper.pos"] = gripper_pos
                robot.send_action(action)

                remaining = dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

                present_meas = robot.bus.sync_read("Present_Position")
                q_meas = np.array([present_meas[n] for n in joint_names], dtype=np.float64)
                T_meas = kin.forward_kinematics(q_meas)

                desired_xyz.append(T_blend[:3, 3].copy())
                achieved_xyz.append(T_meas[:3, 3].copy())
                expected_xyz.append(T_cmd[:3, 3].copy())
                commanded_joints.append(q_cmd.copy())
                measured_joints.append(q_meas.copy())

        if precomputed_joint_traj is not None:
            # Execute precomputed joint trajectory directly (no IK, no encoder seeding)
            for i, q_cmd in enumerate(precomputed_joint_traj):
                step_start = time.perf_counter()

                # Predicted EE pose from joints (planned)
                T_cmd = kin.forward_kinematics(q_cmd)

                action = {f"{name}.pos": float(val) for name, val in zip(joint_names, q_cmd)}
                action["gripper.pos"] = gripper_pos
                robot.send_action(action)

                remaining = dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

                present_meas = robot.bus.sync_read("Present_Position")
                q_meas = np.array([present_meas[n] for n in joint_names], dtype=np.float64)
                T_meas = kin.forward_kinematics(q_meas)

                desired_xyz.append(T_cmd[:3, 3].copy())
                achieved_xyz.append(T_meas[:3, 3].copy())
                expected_xyz.append(T_cmd[:3, 3].copy())
                commanded_joints.append(q_cmd.copy())
                measured_joints.append(q_meas.copy())
        else:
            for i, T_des in enumerate(desired_poses):
                step_start = time.perf_counter()

            # Read current joints for IK seed
            present_seed = robot.bus.sync_read("Present_Position")
            q_seed = np.array([present_seed[n] for n in joint_names], dtype=np.float64)

            # IK to next desired pose (position-only by default)
            q_cmd = kin.inverse_kinematics(
                current_joint_pos=q_seed,
                desired_ee_pose=T_des,
                position_weight=args.position_weight,
                orientation_weight=ow_sequence[i],
            )

            # Predicted EE pose from IK command
            T_cmd = kin.forward_kinematics(q_cmd)

            action = {f"{name}.pos": float(val) for name, val in zip(joint_names, q_cmd)}
            action["gripper.pos"] = gripper_pos
            robot.send_action(action)

            # Wait for motion to execute
            remaining = dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

            # Record achieved state after the command
            present_meas = robot.bus.sync_read("Present_Position")
            q_meas = np.array([present_meas[n] for n in joint_names], dtype=np.float64)
            T_meas = kin.forward_kinematics(q_meas)

            desired_xyz.append(T_des[:3, 3].copy())
            achieved_xyz.append(T_meas[:3, 3].copy())
            expected_xyz.append(T_cmd[:3, 3].copy())
            commanded_joints.append(q_cmd.copy())
            measured_joints.append(q_meas.copy())

    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        robot.disconnect()

    achieved_xyz = np.stack(achieved_xyz, axis=0)
    desired_xyz = np.stack(desired_xyz, axis=0)
    expected_xyz = np.stack(expected_xyz, axis=0)
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
            "exp_x","exp_y","exp_z",
            "err_x","err_y","err_z","err_norm",
        ] + [f"q{i}_cmd_deg" for i in range(commanded_joints.shape[1])] + [f"q{i}_meas_deg" for i in range(measured_joints.shape[1])]
        writer.writerow(header)
        for i in range(desired_xyz.shape[0]):
            row = [
                i,
                desired_xyz[i, 0], desired_xyz[i, 1], desired_xyz[i, 2],
                achieved_xyz[i, 0], achieved_xyz[i, 1], achieved_xyz[i, 2],
                expected_xyz[i, 0], expected_xyz[i, 1], expected_xyz[i, 2],
                pos_err[i, 0], pos_err[i, 1], pos_err[i, 2], pos_err_norm[i],
            ] + list(commanded_joints[i]) + list(measured_joints[i])
            writer.writerow(row)

    # Save NPZ
    np.savez(
        out_dir / "hw_ik_track_results.npz",
        desired_xyz=desired_xyz,
        achieved_xyz=achieved_xyz,
        expected_xyz=expected_xyz,
        commanded_joints_deg=commanded_joints,
        measured_joints_deg=measured_joints,
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

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(desired_xyz[:, 0], desired_xyz[:, 1], desired_xyz[:, 2], label="planned", c="C0")
        ax.plot(achieved_xyz[:, 0], achieved_xyz[:, 1], achieved_xyz[:, 2], label="achieved", c="C1")
        if 'expected_xyz' in locals() or 'expected_xyz' in globals():
            ax.plot(expected_xyz[:, 0], expected_xyz[:, 1], expected_xyz[:, 2], label="ik_pred", c="C2")
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



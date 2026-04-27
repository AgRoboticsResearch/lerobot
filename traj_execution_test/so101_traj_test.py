#!/usr/bin/env python
"""Execute a trajectory loaded from CSV on a real SO101 robot.

Loads a relative EE trajectory from CSV, composes with the initial EE pose
via SE(3), solves IK using Placo, and drives the SO101 robot.

Two modes:
  1. Execute (default): run trajectory on real robot, save plots + CSV.
  2. Plot only (--plot-tcp-traj): no robot, just Placo FK to plot TCP positions.

Usage:
    python so101_traj_test.py --traj-csv test_x_axis.csv --robot-port /dev/ttyACM0
    python so101_traj_test.py --traj-csv test_x_axis.csv --steps 10 --step-time 0.1
    python so101_traj_test.py --traj-csv traj.csv --plot-tcp-traj
    python so101_traj_test.py --traj-csv traj.csv --tcp-frame camera_link
"""

import signal
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from traj_test_util import (
    load_trajectory,
    plot,
    plot_joints,
    plot_tcp_trajectory,
    save_result_to_csv,
)

# ============================================================
# Constants
# ============================================================

PREFIX = "real_so101"
JOINT_LABELS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
NUM_ARM_JOINTS = 5
NUM_JOINTS = 6  # 5 arm + 1 gripper

URDF_PATH = Path(__file__).parent.parent / "urdf" / "Simulation" / "SO101" / "so101_sroi.urdf"

# Gripper range in degrees
GRIPPER_CLOSED_DEG = 0.0
GRIPPER_OPEN_DEG = 100.0

# Rest position (degrees) — folded safe pose
REST_JOINTS_DEG = np.array([0, -90, 90, 0, 0, 0])
# Default reset/home pose (starting position for trajectory execution)
RESET_POSE_DEG = np.array([0, -73, 74, 0, 0, 0])


# ============================================================
# Helpers
# ============================================================

def read_joints_deg(robot, motor_names):
    """Read current joint positions from robot (in degrees)."""
    obs_dict = robot.get_observation()
    return np.array([obs_dict[f"{name}.pos"] for name in motor_names])


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Execute trajectory on real SO101 robot")
    p.add_argument("--traj-csv", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--step-time", type=float, default=0.02)
    p.add_argument("--robot-port", default="/dev/ttyACM0", help="Serial port for SO101")
    p.add_argument("--robot-id", default="so101", help="Robot ID for calibration files")
    p.add_argument("--tcp-frame", default="camera_link", help="URDF frame to use as the planning TCP")
    p.add_argument("--home-joints", default=None, help="Comma-separated home joint positions in degrees")
    p.add_argument("--plot-tcp-traj", action="store_true", help="Plot TCP trajectory without launching robot")
    p.add_argument("--save_result_to_csv", action="store_true", help="Save execution result CSV")
    return p.parse_args()


# ============================================================
# Robot setup
# ============================================================

def go_to_rest(robot, motor_names):
    """Move robot to rest position."""
    print(f"Moving to rest: {REST_JOINTS_DEG} deg")
    rest_action = {f"{name}.pos": val for name, val in zip(motor_names, REST_JOINTS_DEG)}
    robot.send_action(rest_action)
    time.sleep(2.0)
    q_rest = read_joints_deg(robot, motor_names)
    print(f"Reached rest position: {np.round(q_rest, 2)} deg")


def create_robot(robot_port, robot_id, home_deg, tcp_frame):
    """Connect to SO101 robot and move to home position."""
    motor_names = JOINT_LABELS

    print(f"Connecting to SO101 on {robot_port}...")
    robot_config = SO101FollowerConfig(
        id=robot_id,
        port=robot_port,
        use_degrees=True,
    )
    robot = SO101Follower(robot_config)
    robot.connect(calibrate=True)
    print("Robot connected")

    def on_sigint(sig, frame):
        print("\nSIGINT: going to rest...")
        try:
            go_to_rest(robot, motor_names)
        except Exception:
            pass
        print("Disconnecting robot...")
        try:
            robot.disconnect()
        except Exception:
            pass
        sys.exit(1)
    signal.signal(signal.SIGINT, on_sigint)

    input("Press Enter to move to rest position...")
    go_to_rest(robot, motor_names)

    input("Press Enter to move to home position...")
    print(f"Moving to home: {np.round(home_deg, 2)} deg")
    home_action = {f"{name}.pos": val for name, val in zip(motor_names, home_deg)}
    robot.send_action(home_action)
    time.sleep(2.0)

    q_home = read_joints_deg(robot, motor_names)
    print(f"Home joints (deg): {np.round(q_home, 2)}")

    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=tcp_frame,
        joint_names=SO101_ARM_JOINTS,
    )

    T_base = kinematics.forward_kinematics(q_home[:NUM_ARM_JOINTS])
    print(f"Home {tcp_frame} pos: {np.round(T_base[:3, 3], 6)}")

    return robot, T_base, kinematics, motor_names


# ============================================================
# Execution
# ============================================================

def exec_step(robot, step, kinematics):
    """Execute a single trajectory step."""
    gripper = step["gripper"]
    T = step["T_target"]

    q_cur_deg = read_joints_deg(robot, SO101_ARM_JOINTS)
    q_target_deg = kinematics.inverse_kinematics(
        current_joint_pos=q_cur_deg,
        desired_ee_pose=T,
        position_weight=1.0,
        orientation_weight=0.1,
    )

    # Send arm joint commands
    action = {}
    for j, name in enumerate(SO101_ARM_JOINTS):
        action[f"{name}.pos"] = q_target_deg[j]

    # Send gripper command (convert [0,1] to degrees)
    grip_deg = GRIPPER_CLOSED_DEG + gripper * (GRIPPER_OPEN_DEG - GRIPPER_CLOSED_DEG)
    action["gripper.pos"] = grip_deg

    robot.send_action(action)

    # Return arm joints + gripper position as 6-element array
    return np.append(q_target_deg, grip_deg)


def read_actual(robot, step, kinematics):
    """Read actual robot state after a step."""
    obs_dict = robot.get_observation()
    q_actual_arm = read_joints_deg(robot, SO101_ARM_JOINTS)
    q_actual_gripper = obs_dict["gripper.pos"]

    T_actual = kinematics.forward_kinematics(q_actual_arm)
    T_target = step["T_target"]
    pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])

    # Return arm joints + gripper position as 6-element array
    q_actual = np.append(q_actual_arm, q_actual_gripper)
    return T_target, T_actual, pos_err, q_actual


def run_trajectory(robot, traj, step_time, tcp_frame, kinematics):
    """Run the full trajectory on the robot."""
    print(f"\n{'='*60}")
    print(f"Trajectory: {len(traj)} steps, tcp_frame={tcp_frame}, dt={step_time}s")
    print(f"First target ({tcp_frame}): pos={np.round(traj[0]['T_target'][:3,3]*1000, 1)}mm")
    print(f"Last target ({tcp_frame}):  pos={np.round(traj[-1]['T_target'][:3,3]*1000, 1)}mm")
    print(f"{'='*60}")
    input("Press Enter to start trajectory (Ctrl+C to abort)...")

    from scipy.spatial.transform import Rotation as R
    sent_p, act_p = [], []
    sent_r, act_r = [], []
    g_sent, g_act = [], []
    cmd_joints_list, obs_joints_list = [], []
    errors = []

    try:
        for i, step in enumerate(traj):
            cmd_q = exec_step(robot, step, kinematics)
            cmd_joints_list.append(cmd_q.copy())
            time.sleep(step_time)

            T_sent, T_actual, pos_err, q_actual = read_actual(robot, step, kinematics)
            obs_joints_list.append(q_actual.copy())

            sent_p.append(T_sent[:3, 3].copy())
            act_p.append(T_actual[:3, 3].copy())
            sent_r.append(R.from_matrix(T_sent[:3, :3]).as_rotvec())
            act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
            print(f"  Step {i:3d}: pos_err={pos_err*1000:.2f}mm")
            print(f"    Sent ({tcp_frame}): pos={np.round(T_sent[:3,3]*1000,2).tolist()}mm grip={step['gripper']:.2f}")
            print(f"    True ({tcp_frame}): pos={np.round(T_actual[:3,3]*1000,2).tolist()}mm")
            errors.append(pos_err * 1000)

            g_sent.append(step["gripper"])
            g_act.append((q_actual[5] - GRIPPER_CLOSED_DEG) / (GRIPPER_OPEN_DEG - GRIPPER_CLOSED_DEG))

    except KeyboardInterrupt:
        print("\nTrajectory interrupted by user!")

    if errors:
        print(f"\nTrajectory done. {len(errors)}/{len(traj)} steps executed.")
        print(f"  Mean error: {np.mean(errors):.3f}mm")
        print(f"  Max  error: {np.max(errors):.3f}mm")

    return dict(
        sent_pos=np.array(sent_p), act_pos=np.array(act_p),
        sent_rv=np.array(sent_r), act_rv=np.array(act_r),
        g_sent=np.array(g_sent), g_act=np.array(g_act),
        cmd_joints=np.array(cmd_joints_list), obs_joints=np.array(obs_joints_list),
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    if args.home_joints:
        home_deg = np.array([float(x) for x in args.home_joints.split(",")])
        assert len(home_deg) == NUM_JOINTS
    else:
        home_deg = RESET_POSE_DEG

    # --- Plot-only mode ---
    if args.plot_tcp_traj:
        kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name=args.tcp_frame,
            joint_names=SO101_ARM_JOINTS,
        )
        T_base = kinematics.forward_kinematics(home_deg[:NUM_ARM_JOINTS])
        print(f"Home {args.tcp_frame} pos: {np.round(T_base[:3, 3], 6)}")

        traj = load_trajectory(args.traj_csv, T_base, args.steps)
        positions = np.array([s["T_target"][:3, 3] for s in traj])
        print(f"TCP trajectory: {len(positions)} points")

        csv_name = Path(args.traj_csv).stem
        plot_tcp_trajectory(positions, out / f"{PREFIX}_{csv_name}_tcp.png", title=f"TCP Trajectory ({args.tcp_frame})")
        return

    # --- Normal mode ---
    robot, T_base, kinematics, motor_names = create_robot(
        args.robot_port, args.robot_id, home_deg, args.tcp_frame
    )
    traj = load_trajectory(args.traj_csv, T_base, args.steps)
    result = run_trajectory(robot, traj, args.step_time, args.tcp_frame, kinematics)
    plot(result, out, args.tcp_frame, "SO101 Real", PREFIX)
    plot_joints(result, out, f"{PREFIX}_joints.jpg", JOINT_LABELS)

    csv_name = Path(args.traj_csv).stem
    if args.save_result_to_csv:
        save_result_to_csv(result, out, f"{PREFIX}_{csv_name}_result.csv")

    go_to_rest(robot, motor_names)
    print("Disconnecting robot...")
    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()

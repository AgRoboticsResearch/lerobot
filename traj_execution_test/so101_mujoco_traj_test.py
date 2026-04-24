#!/usr/bin/env python
"""Execute a trajectory loaded from CSV on an SO101 MuJoCo simulation.

Loads a relative EE trajectory from CSV, composes with the initial EE pose
via SE(3), solves IK using Placo, and drives MuJoCo position actuators.

Three modes:
  1. Headless (default): runs trajectory, saves video + plots, no viewer.
  2. Viewer (--viewer): interactive MuJoCo viewer, step-by-step with delays.
  3. Plot only (--plot-tcp-traj): no MuJoCo at all, just Placo FK to compute
     TCP positions from CSV, then plots XY/XZ/YZ + 3D + xyz vs steps.

Usage:
    python so101_mujoco_traj_test.py --traj-csv traj.csv
    python so101_mujoco_traj_test.py --traj-csv traj.csv --viewer --step-time 0.1
    python so101_mujoco_traj_test.py --traj-csv traj.csv --plot-tcp-traj
    python so101_mujoco_traj_test.py --traj-csv traj.csv --tcp-frame gripper_frame_link
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from traj_test_util import (
    deg2rad,
    load_trajectory,
    plot,
    plot_joints,
    plot_tcp_trajectory,
    rad2deg,
    save_result_to_csv,
    save_video,
)

# ============================================================
# Constants
# ============================================================

PREFIX = "mujoco_so101"
JOINT_LABELS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SO101_ARM_JOINTS = JOINT_LABELS
NUM_ARM_JOINTS = 5
NUM_JOINTS = 6  # 5 arm + 1 gripper

MJCF_PATH = Path(__file__).parent.parent / "urdf" / "Simulation" / "SO101" / "so101_new_calib.xml"
URDF_PATH = Path(__file__).parent.parent / "urdf" / "Simulation" / "SO101" / "so101_sroi.urdf"

# Gripper joint range in MuJoCo (radians)
GRIPPER_CLOSED_RAD = -0.17453
GRIPPER_OPEN_RAD = 1.74533

# Rest position (degrees) — folded safe pose
REST_JOINTS_DEG = np.array([0, -90, 90, 0, 0, 0])
# Default reset pose (starting position for trajectory execution)
RESET_POSE_DEG = np.array([0, -73, 74, 0, 0, 0])


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-csv", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--model-path", default=str(MJCF_PATH))
    p.add_argument("--viewer", action="store_true", help="Show interactive MuJoCo viewer")
    p.add_argument("--step-time", type=float, default=0.1, help="Delay (seconds) between steps in viewer mode")
    p.add_argument("--tcp-frame", default="camera_link", help="URDF frame to use as the planning TCP")
    p.add_argument("--plot-tcp-traj", action="store_true", help="Plot TCP trajectory without launching MuJoCo")
    return p.parse_args()


# ============================================================
# MuJoCo + Placo environment
# ============================================================

def create_env(model_path, tcp_frame):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=tcp_frame,
        joint_names=SO101_ARM_JOINTS,
    )

    rest_rad = deg2rad(REST_JOINTS_DEG)
    for j in range(NUM_JOINTS):
        data.qpos[j] = rest_rad[j]
        data.ctrl[j] = rest_rad[j]
    mujoco.mj_forward(model, data)
    print(f"Rest joints (deg): {REST_JOINTS_DEG}")

    return model, data, kinematics


def move_to_home(model, data, kinematics, tcp_frame):
    home_rad = deg2rad(RESET_POSE_DEG)
    for j in range(NUM_JOINTS):
        data.qpos[j] = home_rad[j]
        data.ctrl[j] = home_rad[j]
    mujoco.mj_forward(model, data)

    q_home_deg = RESET_POSE_DEG[:NUM_ARM_JOINTS]
    T_base = kinematics.forward_kinematics(q_home_deg)
    print(f"Home joints (deg): {np.round(q_home_deg, 2)}")
    print(f"Home {tcp_frame} pos: {np.round(T_base[:3, 3], 6)}")
    return T_base


def go_to_rest(model, data):
    rest_rad = deg2rad(REST_JOINTS_DEG)
    for j in range(NUM_JOINTS):
        data.qpos[j] = rest_rad[j]
        data.ctrl[j] = rest_rad[j]
    mujoco.mj_forward(model, data)
    print(f"Moved to rest position: {REST_JOINTS_DEG}")


# ============================================================
# Execution
# ============================================================

def exec_step(model, data, kinematics, step, i):
    T_target = step["T_target"]
    gripper = step["gripper"]

    q_cur_deg = rad2deg([data.qpos[j] for j in range(NUM_ARM_JOINTS)])
    q_target_deg = kinematics.inverse_kinematics(
        current_joint_pos=q_cur_deg, desired_ee_pose=T_target,
        position_weight=1.0, orientation_weight=0.1,
    )
    q_target_rad = deg2rad(q_target_deg)

    for j in range(NUM_ARM_JOINTS):
        data.qpos[j] = q_target_rad[j]
        data.ctrl[j] = q_target_rad[j]

    gripper_rad = GRIPPER_CLOSED_RAD + gripper * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
    data.qpos[NUM_ARM_JOINTS] = gripper_rad
    data.ctrl[NUM_ARM_JOINTS] = gripper_rad

    mujoco.mj_forward(model, data)
    T_actual = kinematics.forward_kinematics(q_target_deg)

    pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
    print(f"  Step {i:3d}: pos_err={pos_err:.6f}m")
    print(f"    Sent: pos={T_target[:3, 3].tolist()} grip={gripper:.2f}")
    print(f"    True: pos={T_actual[:3, 3].tolist()} grip={data.qpos[NUM_ARM_JOINTS]:.5f}")

    return T_target, T_actual, q_target_deg


def run_trajectory(model, data, kinematics, traj, render=True):
    renderer = None
    frames = []
    if render:
        try:
            renderer = mujoco.Renderer(model, height=720, width=960)
        except Exception as e:
            print(f"Warning: could not create renderer ({e}), skipping video")

    sent_p, sent_r, act_p, act_r = [], [], [], []
    g_sent, g_act = [], []
    cmd_joints, obs_joints = [], []

    for i, step in enumerate(traj):
        T_target, T_actual, q_cmd_deg = exec_step(model, data, kinematics, step, i)

        cmd_joints.append(q_cmd_deg.copy())
        obs_joints.append(rad2deg([data.qpos[j] for j in range(NUM_ARM_JOINTS)]).copy())

        from scipy.spatial.transform import Rotation as R
        sent_p.append(T_target[:3, 3].copy())
        sent_r.append(R.from_matrix(T_target[:3, :3]).as_rotvec())
        act_p.append(T_actual[:3, 3].copy())
        act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
        g_sent.append(step["gripper"])
        g_act.append((data.qpos[NUM_ARM_JOINTS] - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD))

        if renderer is not None:
            renderer.update_scene(data)
            frames.append(renderer.render())

    result = dict(
        sent_pos=np.array(sent_p), sent_rv=np.array(sent_r), g_sent=np.array(g_sent),
        act_pos=np.array(act_p), act_rv=np.array(act_r), g_act=np.array(g_act),
        cmd_joints=np.array(cmd_joints), obs_joints=np.array(obs_joints),
    )
    return result, frames


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    # --- Plot-only mode ---
    if args.plot_tcp_traj:
        kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH), target_frame_name=args.tcp_frame, joint_names=SO101_ARM_JOINTS,
        )
        T_base = kinematics.forward_kinematics(RESET_POSE_DEG[:NUM_ARM_JOINTS])
        print(f"Home {args.tcp_frame} pos: {np.round(T_base[:3, 3], 6)}")

        traj = load_trajectory(args.traj_csv, T_base, args.steps)
        positions = np.array([s["T_target"][:3, 3] for s in traj])
        print(f"TCP trajectory: {len(positions)} points")

        csv_name = Path(args.traj_csv).stem
        plot_tcp_trajectory(positions, out / f"{PREFIX}_{csv_name}_tcp.png", title=f"TCP Trajectory ({args.tcp_frame})")
        return

    # --- Normal mode ---
    model, data, kinematics = create_env(args.model_path, args.tcp_frame)

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.flags[mujoco.mjtFrame.mjFRAME_BODY] = True
            viewer.sync()

            input("Press Enter to move to home position...")
            T_base = move_to_home(model, data, kinematics, args.tcp_frame)
            viewer.sync()

            traj = load_trajectory(args.traj_csv, T_base, args.steps)
            input("Press Enter to start trajectory (Ctrl+C to abort)...")

            from scipy.spatial.transform import Rotation as R
            sent_p, sent_r, act_p, act_r = [], [], [], []
            g_sent, g_act = [], []
            cmd_joints, obs_joints = [], []

            for i, step in enumerate(traj):
                T_target, T_actual, q_cmd_deg = exec_step(model, data, kinematics, step, i)
                cmd_joints.append(q_cmd_deg.copy())
                obs_joints.append(rad2deg([data.qpos[j] for j in range(NUM_ARM_JOINTS)]).copy())
                sent_p.append(T_target[:3, 3].copy())
                sent_r.append(R.from_matrix(T_target[:3, :3]).as_rotvec())
                act_p.append(T_actual[:3, 3].copy())
                act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
                g_sent.append(step["gripper"])
                g_act.append((data.qpos[NUM_ARM_JOINTS] - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD))
                viewer.sync()
                if args.step_time > 0:
                    time.sleep(args.step_time)

            result = dict(
                sent_pos=np.array(sent_p), sent_rv=np.array(sent_r), g_sent=np.array(g_sent),
                act_pos=np.array(act_p), act_rv=np.array(act_r), g_act=np.array(g_act),
                cmd_joints=np.array(cmd_joints), obs_joints=np.array(obs_joints),
            )
            plot(result, out, args.tcp_frame, "SO101 MuJoCo", PREFIX)
            plot_joints(result, out, f"{PREFIX}_joints.jpg", JOINT_LABELS)
            csv_name = Path(args.traj_csv).stem
            save_result_to_csv(result, out, f"{PREFIX}_{csv_name}_result.csv")

            input("Press Enter to return to rest position...")
            go_to_rest(model, data)
            viewer.sync()
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    else:
        T_base = move_to_home(model, data, kinematics, args.tcp_frame)
        traj = load_trajectory(args.traj_csv, T_base, args.steps)
        result, frames = run_trajectory(model, data, kinematics, traj, render=True)
        if frames:
            save_video(frames, out / f"{PREFIX}_output.mp4")
        plot(result, out, args.tcp_frame, "SO101 MuJoCo", PREFIX)
        plot_joints(result, out, f"{PREFIX}_joints.jpg", JOINT_LABELS)
        csv_name = Path(args.traj_csv).stem
        save_result_to_csv(result, out, f"{PREFIX}_{csv_name}_result.csv")
        go_to_rest(model, data)


if __name__ == "__main__":
    main()

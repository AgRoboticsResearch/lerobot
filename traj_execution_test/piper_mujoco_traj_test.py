#!/usr/bin/env python
"""Execute a trajectory loaded from CSV on a Piper MuJoCo simulation.

Three modes:
  1. Headless (default): runs trajectory, saves video + plots, no viewer.
  2. Viewer (--viewer): interactive MuJoCo viewer, step-by-step with delays.
  3. Plot only (--plot-tcp-traj): no MuJoCo, just Placo FK to plot TCP positions.

Usage:
    python piper_mujoco_traj_test.py --traj-csv traj.csv
    python piper_mujoco_traj_test.py --traj-csv traj.csv --viewer --step-time 0.1
    python piper_mujoco_traj_test.py --traj-csv traj.csv --plot-tcp-traj
    python piper_mujoco_traj_test.py --traj-csv traj.csv --tcp-frame camera_link
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from frame_utils import SDK_NATIVE_FRAME, pose_from_native, resolve_tcp_frame
from lerobot.model.kinematics import RobotKinematics
from traj_test_util import (
    deg2rad, load_trajectory, plot, plot_joints, plot_tcp_trajectory,
    rad2deg, save_result_to_csv, save_video,
)

# ============================================================
# Constants
# ============================================================

PREFIX = "mujoco_piper"
JOINT_LABELS = [f"J{i+1}" for i in range(6)]
PIPER_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
NUM_ARM_JOINTS = 6
SUBSTEPS = 50

MJCF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description.xml"
URDF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description.urdf"

REST_JOINTS_DEG = np.array([-0.59, -3.11, -3.56, -2.55, 23.30, 1.17])
HOME_JOINTS_DEG = np.array([0, 73.77, -43.43, 0, -24, 0])


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-csv", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--model-path", default=str(MJCF_PATH))
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--step-time", type=float, default=0.1)
    p.add_argument("--tcp-frame", default="ee_link")
    p.add_argument("--plot-tcp-traj", action="store_true")
    return p.parse_args()


# ============================================================
# MuJoCo + Placo environment
# ============================================================

def create_env(model_path, tcp_frame):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    target_site_id = model.site("traj_target").id if "traj_target" in [model.site(i).name for i in range(model.nsite)] else -1

    kinematics = RobotKinematics(urdf_path=str(URDF_PATH), target_frame_name=tcp_frame, joint_names=PIPER_ARM_JOINTS)
    native_kinematics = RobotKinematics(urdf_path=str(URDF_PATH), target_frame_name=SDK_NATIVE_FRAME, joint_names=PIPER_ARM_JOINTS)

    rest_rad = deg2rad(REST_JOINTS_DEG)
    for j in range(NUM_ARM_JOINTS):
        data.qpos[j] = rest_rad[j]
        data.ctrl[j] = rest_rad[j]
    data.qpos[6] = 0.0; data.qpos[7] = 0.0
    data.ctrl[6] = 0.0; data.ctrl[7] = 0.0
    mujoco.mj_forward(model, data)
    print(f"Rest joints (deg): {REST_JOINTS_DEG}")

    return model, data, kinematics, native_kinematics, target_site_id


def move_to_home(model, data, kinematics, native_kinematics, frame_spec):
    home_rad = deg2rad(HOME_JOINTS_DEG)
    for j in range(NUM_ARM_JOINTS):
        data.qpos[j] = home_rad[j]; data.ctrl[j] = home_rad[j]
    data.qpos[6] = 0.0; data.qpos[7] = 0.0
    data.ctrl[6] = 0.0; data.ctrl[7] = 0.0
    mujoco.mj_forward(model, data)

    q_home_deg = rad2deg([data.qpos[i] for i in range(NUM_ARM_JOINTS)])
    T_base_native = native_kinematics.forward_kinematics(q_home_deg)
    T_base = pose_from_native(T_base_native, frame_spec)
    tcp_pos = T_base[:3, 3]

    print(f"Home joints (deg): {np.round(q_home_deg, 2)}")
    print(f"Home {frame_spec.tcp_frame} pos: {np.round(tcp_pos, 6)}")
    return T_base


def go_to_rest(model, data):
    rest_rad = deg2rad(REST_JOINTS_DEG)
    for j in range(NUM_ARM_JOINTS):
        data.qpos[j] = rest_rad[j]; data.ctrl[j] = rest_rad[j]
    data.qpos[6] = 0.0; data.qpos[7] = 0.0
    data.ctrl[6] = 0.0; data.ctrl[7] = 0.0
    mujoco.mj_forward(model, data)
    print(f"Moved to rest position: {REST_JOINTS_DEG}")


# ============================================================
# Execution
# ============================================================

def exec_step(model, data, kinematics, step, i, target_site_id=-1):
    T_target = step["T_target"]
    gripper = step["gripper"]

    q_cur_deg = rad2deg([data.qpos[j] for j in range(NUM_ARM_JOINTS)])
    q_target_deg = kinematics.inverse_kinematics(current_joint_pos=q_cur_deg, desired_ee_pose=T_target, position_weight=1.0, orientation_weight=0.1)
    q_target_rad = deg2rad(q_target_deg)

    for j in range(NUM_ARM_JOINTS):
        data.qpos[j] = q_target_rad[j]
    data.qpos[6] = gripper * 0.04
    data.qpos[7] = -gripper * 0.04
    for j in range(NUM_ARM_JOINTS):
        data.ctrl[j] = q_target_rad[j]
    data.ctrl[6] = gripper * 0.04
    data.ctrl[7] = -gripper * 0.04

    mujoco.mj_forward(model, data)
    T_actual = kinematics.forward_kinematics(q_target_deg)

    if target_site_id >= 0:
        data.site_xpos[target_site_id] = T_target[:3, 3]

    pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
    print(f"  Step {i:3d}: pos_err={pos_err:.6f}m")
    print(f"    Sent: pos={T_target[:3, 3].tolist()} grip={gripper:.2f}")
    print(f"    True: pos={T_actual[:3, 3].tolist()} grip={data.qpos[6]:.5f}")

    return T_target, T_actual, q_target_deg


def run_trajectory(model, data, kinematics, traj, render=True, target_site_id=-1):
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
        T_target, T_actual, q_cmd_deg = exec_step(model, data, kinematics, step, i, target_site_id)
        cmd_joints.append(q_cmd_deg.copy())
        obs_joints.append(rad2deg([data.qpos[j] for j in range(NUM_ARM_JOINTS)]).copy())
        sent_p.append(T_target[:3, 3].copy())
        sent_r.append(R.from_matrix(T_target[:3, :3]).as_rotvec())
        act_p.append(T_actual[:3, 3].copy())
        act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
        g_sent.append(step["gripper"])
        g_act.append(data.qpos[6] / 0.04)
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
    frame_spec = resolve_tcp_frame(URDF_PATH, args.tcp_frame, native_frame=SDK_NATIVE_FRAME)

    # --- Plot-only mode ---
    if args.plot_tcp_traj:
        native_kinematics = RobotKinematics(urdf_path=str(URDF_PATH), target_frame_name=SDK_NATIVE_FRAME, joint_names=PIPER_ARM_JOINTS)
        T_base_native = native_kinematics.forward_kinematics(HOME_JOINTS_DEG)
        T_base = pose_from_native(T_base_native, frame_spec)
        print(f"Home {frame_spec.tcp_frame} pos: {np.round(T_base[:3, 3], 6)}")

        traj = load_trajectory(args.traj_csv, T_base, args.steps)
        positions = np.array([s["T_target"][:3, 3] for s in traj])
        print(f"TCP trajectory: {len(positions)} points")

        csv_name = Path(args.traj_csv).stem
        plot_tcp_trajectory(positions, out / f"{PREFIX}_{csv_name}_tcp.png", title=f"TCP Trajectory ({frame_spec.tcp_frame})")
        return

    # --- Normal mode ---
    model, data, kinematics, native_kinematics, target_site_id = create_env(args.model_path, args.tcp_frame)

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.flags[mujoco.mjtFrame.mjFRAME_BODY] = True
            viewer.sync()

            input("Press Enter to move to home position...")
            T_base = move_to_home(model, data, kinematics, native_kinematics, frame_spec)
            viewer.sync()

            traj = load_trajectory(args.traj_csv, T_base, args.steps)
            input("Press Enter to start trajectory (Ctrl+C to abort)...")

            sent_p, sent_r, act_p, act_r = [], [], [], []
            g_sent, g_act = [], []
            cmd_joints, obs_joints = [], []

            for i, step in enumerate(traj):
                T_target, T_actual, q_cmd_deg = exec_step(model, data, kinematics, step, i, target_site_id)
                cmd_joints.append(q_cmd_deg.copy())
                obs_joints.append(rad2deg([data.qpos[j] for j in range(NUM_ARM_JOINTS)]).copy())
                sent_p.append(T_target[:3, 3].copy())
                sent_r.append(R.from_matrix(T_target[:3, :3]).as_rotvec())
                act_p.append(T_actual[:3, 3].copy())
                act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
                g_sent.append(step["gripper"])
                g_act.append(data.qpos[6] / 0.04)
                viewer.sync()
                if args.step_time > 0:
                    time.sleep(args.step_time)

            result = dict(
                sent_pos=np.array(sent_p), sent_rv=np.array(sent_r), g_sent=np.array(g_sent),
                act_pos=np.array(act_p), act_rv=np.array(act_r), g_act=np.array(g_act),
                cmd_joints=np.array(cmd_joints), obs_joints=np.array(obs_joints),
            )
            plot(result, out, frame_spec.tcp_frame, "Piper MuJoCo", PREFIX)
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
        T_base = move_to_home(model, data, kinematics, native_kinematics, frame_spec)
        traj = load_trajectory(args.traj_csv, T_base, args.steps)
        result, frames = run_trajectory(model, data, kinematics, traj, render=True, target_site_id=target_site_id)
        if frames:
            save_video(frames, out / f"{PREFIX}_output.mp4")
        plot(result, out, frame_spec.tcp_frame, "Piper MuJoCo", PREFIX)
        plot_joints(result, out, f"{PREFIX}_joints.jpg", JOINT_LABELS)
        csv_name = Path(args.traj_csv).stem
        save_result_to_csv(result, out, f"{PREFIX}_{csv_name}_result.csv")
        go_to_rest(model, data)


if __name__ == "__main__":
    main()

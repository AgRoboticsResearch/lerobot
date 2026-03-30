#!/usr/bin/env python
"""Execute a trajectory loaded from CSV on a Piper MuJoCo simulation.

Loads a relative EE trajectory from CSV, composes with the initial EE pose
via SE(3), solves IK using Placo, and drives MuJoCo position actuators.

Usage:
    python piper_mujoco_traj_test.py --traj-csv test_x_axis.csv
    python piper_mujoco_traj_test.py --traj-csv test_x_axis.csv --steps 50
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from lerobot.model.kinematics import RobotKinematics

# ============================================================
# Constants
# ============================================================

MJCF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description.xml"
URDF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description.urdf"
PIPER_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
NUM_ARM_JOINTS = 6
SUBSTEPS = 50  # MuJoCo steps per trajectory step (let actuators converge)


# ============================================================
# Helpers
# ============================================================

def make_transform(pos, rotvec):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = pos
    return T


def rad2deg(rad):
    return np.array(rad) * 180.0 / np.pi


def deg2rad(deg):
    return np.array(deg) * np.pi / 180.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj-csv", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--model-path", default=str(MJCF_PATH))
    return p.parse_args()


# ============================================================
# CSV -> absolute EE poses
# ============================================================

def load_trajectory(csv_path, T_base, max_steps=None):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} steps from {csv_path}")

    traj = []
    for _, row in df.iterrows():
        if "action.ee.x" in df.columns:
            rel_pos = [row["action.ee.x"], row["action.ee.y"], row["action.ee.z"]]
            rel_rv = [row.get("action.ee.wx", 0), row.get("action.ee.wy", 0), row.get("action.ee.wz", 0)]
            gripper = row["action.ee.gripper_pos"]
        else:
            rel_pos = [row["state.ee.x"], row["state.ee.y"], row["state.ee.z"]]
            rel_rv = [row.get("state.ee.wx", 0), row.get("state.ee.wy", 0), row.get("state.ee.wz", 0)]
            gripper = row["state.ee.gripper_pos"]

        T_t = T_base @ make_transform(rel_pos, rel_rv)
        traj.append({
            "T_target": T_t.copy(),
            "gripper": gripper,  # 1=open, 0=closed
        })

    if max_steps:
        traj = traj[:max_steps]
    print(f"Composed {len(traj)} absolute EE targets")
    return traj


# ============================================================
# MuJoCo + Placo environment
# ============================================================

def create_env(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Initialize kinematics solver
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="link6",
        joint_names=PIPER_ARM_JOINTS,
    )

    # Get initial EE pose via FK
    q_rad = np.array([data.qpos[i] for i in range(NUM_ARM_JOINTS)])
    q_deg = rad2deg(q_rad)
    T_base = kinematics.forward_kinematics(q_deg)

    print(f"Initial joints (deg): {np.round(q_deg, 2)}")
    print(f"Initial EE pos: {T_base[:3, 3]}")

    return model, data, kinematics, T_base


# ============================================================
# Execution
# ============================================================

def run_trajectory(model, data, kinematics, traj, render=True):
    # Try offscreen rendering, gracefully handle headless environments
    renderer = None
    frames = []
    if render:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
        except Exception as e:
            print(f"Warning: could not create renderer ({e}), skipping video")

    sent_p, sent_r, act_p, act_r = [], [], [], []
    g_sent, g_act = [], []

    for i, step in enumerate(traj):
        T_target = step["T_target"]
        gripper = step["gripper"]

        # Current joints (seed for IK)
        q_cur_rad = np.array([data.qpos[j] for j in range(NUM_ARM_JOINTS)])
        q_cur_deg = rad2deg(q_cur_rad)

        # Solve IK
        q_target_deg = kinematics.inverse_kinematics(
            current_joint_pos=q_cur_deg,
            desired_ee_pose=T_target,
            position_weight=1.0,
            orientation_weight=0.1,
        )
        q_target_rad = deg2rad(q_target_deg)

        # Send to MuJoCo position actuators
        for j in range(NUM_ARM_JOINTS):
            data.ctrl[j] = q_target_rad[j]

        # Gripper: 1=open -> ctrl=0, 0=closed -> ctrl=max
        grip_open = gripper  # 1=open, 0=closed
        data.ctrl[6] = (1 - grip_open) * 0.04   # joint7: [0, 0.04]
        data.ctrl[7] = -(1 - grip_open) * 0.04  # joint8: [-0.04, 0]

        # Step simulation
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        # Record actual state
        q_actual_rad = np.array([data.qpos[j] for j in range(NUM_ARM_JOINTS)])
        q_actual_deg = rad2deg(q_actual_rad)
        T_actual = kinematics.forward_kinematics(q_actual_deg)

        sent_p.append(T_target[:3, 3].copy())
        sent_r.append(R.from_matrix(T_target[:3, :3]).as_rotvec())
        act_p.append(T_actual[:3, 3].copy())
        act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
        g_sent.append(gripper)
        g_act.append(data.qpos[6] / 0.04)  # normalize to [0,1]

        # Render if available
        if renderer is not None:
            renderer.update_scene(data)
            frames.append(renderer.render())

        # Print
        pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
        print(f"  Step {i:3d}: pos_err={pos_err:.6f}m")
        print(f"    Sent: pos={T_target[:3, 3].tolist()} grip={gripper:.2f}")
        print(f"    True: pos={T_actual[:3, 3].tolist()} grip={data.qpos[6]:.5f}")

    result = dict(
        sent_pos=np.array(sent_p), sent_rv=np.array(sent_r), g_sent=np.array(g_sent),
        act_pos=np.array(act_p), act_rv=np.array(act_r), g_act=np.array(g_act),
    )
    return result, frames


# ============================================================
# Save & Plot
# ============================================================

def save_video(frames, out_dir, fps=20):
    import imageio
    p = out_dir / "piper_mujoco_traj_test_output.mp4"
    imageio.mimsave(str(p), frames, fps=fps)
    print(f"Saved {len(frames)} frames -> {p}")


def plot(result, out_dir):
    sp, ap = result["sent_pos"], result["act_pos"]
    sr, ar = result["sent_rv"], result["act_rv"]
    gs, ga = result["g_sent"], result["g_act"]
    steps = np.arange(len(sp))

    se = R.from_rotvec(sr.reshape(-1, 3)).as_euler("xyz")
    ae = R.from_rotvec(ar.reshape(-1, 3)).as_euler("xyz")

    # 3D — equal axis ranges
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sp[:, 0], sp[:, 1], sp[:, 2], "b-o", label="Sent", ms=3)
    ax.plot(ap[:, 0], ap[:, 1], ap[:, 2], "r-s", label="Actual", ms=3)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    all_pts = np.vstack([sp, ap])
    mid = all_pts.mean(axis=0)
    half = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 0.05)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_title("3D EE Trajectory [Piper MuJoCo]"); ax.legend()
    plt.tight_layout(); fig.savefig(str(out_dir / "traj_3d.png")); plt.close()

    # 2D
    labels = ["X(m)", "Y(m)", "Z(m)", "Roll", "Pitch", "Yaw", "Grip"]
    sd = np.column_stack([sp, se, gs])
    ad = np.column_stack([ap, ae, ga])
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    for a, lb, s, v in zip(axes, labels, sd.T, ad.T):
        a.plot(steps, s, "b-o", label="Sent", ms=3)
        a.plot(steps, v, "r-s", label="Actual", ms=3)
        a.set_ylabel(lb); a.legend()
        if a.get_ylim()[1] - a.get_ylim()[0] < 0.1:
            mid_a = (a.get_ylim()[0] + a.get_ylim()[1]) / 2
            a.set_ylim(mid_a - 0.05, mid_a + 0.05)
    axes[-1].set_xlabel("Step"); axes[0].set_title("EE State [Piper MuJoCo]")
    plt.tight_layout(); fig.savefig(str(out_dir / "traj_2d_states.png")); plt.close()
    print(f"Saved plots to {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    model, data, kinematics, T_base = create_env(args.model_path)
    traj = load_trajectory(args.traj_csv, T_base, args.steps)

    result, frames = run_trajectory(model, data, kinematics, traj)
    if frames:
        save_video(frames, out)
    plot(result, out)


if __name__ == "__main__":
    main()

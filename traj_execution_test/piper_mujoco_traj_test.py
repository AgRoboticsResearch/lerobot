#!/usr/bin/env python
"""Execute a trajectory loaded from CSV on a Piper MuJoCo simulation.

Loads a relative EE trajectory from CSV, composes with the initial EE pose
via SE(3), solves IK using Placo, and drives MuJoCo position actuators.

Three modes:
  1. Headless (default): runs trajectory, saves video + plots, no viewer.
  2. Viewer (--viewer): interactive MuJoCo viewer, step-by-step with delays.
  3. Plot only (--plot-tcp-traj): no MuJoCo at all, just Placo FK to compute
     TCP positions from CSV, then plots XY/XZ/YZ + 3D + xyz vs steps.

Usage:
    # Headless: run trajectory and save plots/video
    python piper_mujoco_traj_test.py --traj-csv traj.csv

    # Headless with step limit
    python piper_mujoco_traj_test.py --traj-csv traj.csv --steps 50

    # Interactive viewer with step delay
    python piper_mujoco_traj_test.py --traj-csv traj.csv --viewer --step-time 0.1

    # Plot TCP trajectory without MuJoCo
    python piper_mujoco_traj_test.py --traj-csv traj.csv --plot-tcp-traj

    # Change TCP frame (default: ee_link)
    python piper_mujoco_traj_test.py --traj-csv traj.csv --tcp-frame camera_link
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from frame_utils import SDK_NATIVE_FRAME, pose_from_native, resolve_tcp_frame
from lerobot.model.kinematics import RobotKinematics

# ============================================================
# Constants
# ============================================================

MJCF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description.xml"
URDF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description.urdf"
PIPER_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
NUM_ARM_JOINTS = 6
SUBSTEPS = 50  # MuJoCo steps per trajectory step (let actuators converge)

# Rest position (degrees) — folded safe pose, used at start/end
REST_JOINTS_DEG = np.array([-0.59, -3.11, -3.56, -2.55, 23.30, 1.17])
# Home position (degrees) — a sensible upright pose
# HOME_JOINTS_DEG = np.array([-1.40, 30.24, -58.64, -1.05, 32.45, 0.00])
HOME_JOINTS_DEG = np.array([0, 73.77, -43.43, 0, -24, 0])


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
    p.add_argument("--viewer", action="store_true", help="Show interactive MuJoCo viewer")
    p.add_argument("--step-time", type=float, default=0.1, help="Delay (seconds) between steps in viewer mode")
    p.add_argument("--tcp-frame", default="ee_link", help="URDF frame to use as the planning TCP")
    p.add_argument("--plot-tcp-traj", action="store_true", help="Plot TCP trajectory without launching MuJoCo")
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

def create_env(model_path, tcp_frame):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get visualization site indices
    target_site_id = model.site("traj_target").id if "traj_target" in [model.site(i).name for i in range(model.nsite)] else -1

    # Initialize kinematics solvers for the requested TCP frame and the SDK-native frame.
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=tcp_frame,
        joint_names=PIPER_ARM_JOINTS,
    )
    native_kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=SDK_NATIVE_FRAME,
        joint_names=PIPER_ARM_JOINTS,
    )

    # Start from rest position
    rest_rad = deg2rad(REST_JOINTS_DEG)
    for j in range(NUM_ARM_JOINTS):
        data.qpos[j] = rest_rad[j]
        data.ctrl[j] = rest_rad[j]
    data.qpos[6] = 0.0
    data.qpos[7] = 0.0
    data.ctrl[6] = 0.0
    data.ctrl[7] = 0.0
    mujoco.mj_forward(model, data)
    print(f"Rest joints (deg): {REST_JOINTS_DEG}")

    return model, data, kinematics, native_kinematics, target_site_id


def move_to_home(model, data, kinematics, native_kinematics, frame_spec):
    """Move from rest to home and return T_base."""
    home_rad = deg2rad(HOME_JOINTS_DEG)
    for j in range(NUM_ARM_JOINTS):
        data.qpos[j] = home_rad[j]
        data.ctrl[j] = home_rad[j]
    data.qpos[6] = 0.0
    data.qpos[7] = 0.0
    data.ctrl[6] = 0.0
    data.ctrl[7] = 0.0
    mujoco.mj_forward(model, data)

    q_home_rad = np.array([data.qpos[i] for i in range(NUM_ARM_JOINTS)])
    q_home_deg = rad2deg(q_home_rad)
    T_base_native = native_kinematics.forward_kinematics(q_home_deg)
    T_base = pose_from_native(T_base_native, frame_spec)
    native_pos = T_base_native[:3, 3]
    tcp_pos = T_base[:3, 3]
    offset_pos = frame_spec.T_native_to_tcp[:3, 3]
    offset_euler = R.from_matrix(frame_spec.T_native_to_tcp[:3, :3]).as_euler("xyz", degrees=True)

    print(f"Home joints (deg): {np.round(q_home_deg, 2)}")
    print(f"Home {frame_spec.native_frame} pos: {np.round(native_pos, 6)}")
    print(f"Home {frame_spec.tcp_frame} pos: {np.round(tcp_pos, 6)}")
    print(f"Offset {frame_spec.native_frame} -> {frame_spec.tcp_frame} pos (m): {np.round(offset_pos, 6)}")
    print(f"Offset {frame_spec.native_frame} -> {frame_spec.tcp_frame} rot (deg): {np.round(offset_euler, 3)}")

    return T_base


def go_to_rest(model, data):
    """Move arm to rest position (direct qpos set)."""
    rest_rad = deg2rad(REST_JOINTS_DEG)
    for j in range(NUM_ARM_JOINTS):
        data.qpos[j] = rest_rad[j]
        data.ctrl[j] = rest_rad[j]
    data.qpos[6] = 0.0
    data.qpos[7] = 0.0
    data.ctrl[6] = 0.0
    data.ctrl[7] = 0.0
    mujoco.mj_forward(model, data)
    print(f"Moved to rest position: {REST_JOINTS_DEG}")


# ============================================================
# Execution
# ============================================================

def exec_step(model, data, kinematics, step, i, target_site_id=-1):
    """Execute a single trajectory step: IK -> set qpos -> mj_forward -> print + visualize."""
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

    # Directly set joint positions (kinematic mode — bypasses actuator dynamics)
    for j in range(NUM_ARM_JOINTS):
        data.qpos[j] = q_target_rad[j]

    # Gripper: 1=open -> max spread, 0=closed -> fingers together
    grip_open = gripper  # 1=open, 0=closed
    data.qpos[6] = grip_open * 0.04       # joint7: [0, 0.04]
    data.qpos[7] = -grip_open * 0.04      # joint8: [-0.04, 0]

    # Also set ctrl so viewer sliders show correct targets
    for j in range(NUM_ARM_JOINTS):
        data.ctrl[j] = q_target_rad[j]
    data.ctrl[6] = grip_open * 0.04
    data.ctrl[7] = -grip_open * 0.04

    # Update kinematics (no dynamics step)
    mujoco.mj_forward(model, data)

    # Get actual EE pose
    T_actual = kinematics.forward_kinematics(q_target_deg)

    # Update visualization sites (if available)
    if target_site_id >= 0:
        data.site_xpos[target_site_id] = T_target[:3, 3]

    # Print
    pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
    print(f"  Step {i:3d}: pos_err={pos_err:.6f}m")
    print(f"    Sent: pos={T_target[:3, 3].tolist()} grip={gripper:.2f}")
    print(f"    True: pos={T_actual[:3, 3].tolist()} grip={data.qpos[6]:.5f}")

    return T_target, T_actual, q_target_deg


def run_trajectory(model, data, kinematics, traj, render=True, target_site_id=-1):
    # Offscreen renderer for video
    renderer = None
    frames = []
    if render:
        try:
            # Larger resolution for better view
            renderer = mujoco.Renderer(model, height=720, width=960)
        except Exception as e:
            print(f"Warning: could not create renderer ({e}), skipping video")

    sent_p, sent_r, act_p, act_r = [], [], [], []
    g_sent, g_act = [], []
    cmd_joints, obs_joints = [], []

    for i, step in enumerate(traj):
        T_target, T_actual, q_cmd_deg = exec_step(model, data, kinematics, step, i, target_site_id)

        cmd_joints.append(q_cmd_deg.copy())
        q_obs_rad = np.array([data.qpos[j] for j in range(NUM_ARM_JOINTS)])
        obs_joints.append(rad2deg(q_obs_rad).copy())

        sent_p.append(T_target[:3, 3].copy())
        sent_r.append(R.from_matrix(T_target[:3, :3]).as_rotvec())
        act_p.append(T_actual[:3, 3].copy())
        act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
        g_sent.append(step["gripper"])
        g_act.append(data.qpos[6] / 0.04)  # normalize to [0,1]

        # Render if available
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
# Save & Plot
# ============================================================

def save_video(frames, out_dir, fps=20):
    import imageio
    p = out_dir / "piper_mujoco_traj_test_output.mp4"
    imageio.mimsave(str(p), frames, fps=fps)
    print(f"Saved {len(frames)} frames -> {p}")


def plot_tcp_trajectory(positions: np.ndarray, output_path: Path, title: str = "TCP Trajectory"):
    """Plot TCP trajectory in same style as visualize_orb_traj.py (XY/XZ/YZ + 3D + xyz vs steps)."""
    fig = plt.figure(figsize=(16, 12))

    projections = [
        (0, 1, 'X', 'Y'),  # XY
        (0, 2, 'X', 'Z'),  # XZ
        (1, 2, 'Y', 'Z'),  # YZ
    ]

    for idx, (x_idx, y_idx, xlabel, ylabel) in enumerate(projections):
        ax = fig.add_subplot(3, 3, idx + 1)
        ax.plot(positions[:, x_idx], positions[:, y_idx], 'b-', linewidth=1.5, alpha=0.7)
        ax.scatter([positions[0, x_idx]], [positions[0, y_idx]], c='green', s=80, marker='o', label='Start', zorder=10)
        ax.scatter([positions[-1, x_idx]], [positions[-1, y_idx]], c='red', s=80, marker='x', label='End', zorder=10)
        ax.set_xlabel(f'{xlabel} (m)'); ax.set_ylabel(f'{ylabel} (m)')
        ax.set_title(f'{xlabel}-{ylabel} Projection'); ax.legend(); ax.grid(True, alpha=0.3); ax.axis('equal')

    ax3d = fig.add_subplot(3, 3, (4, 6), projection='3d')
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
    ax3d.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], c='green', s=80, marker='o', label='Start', zorder=10)
    ax3d.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], c='red', s=80, marker='x', label='End', zorder=10)
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory'); ax3d.legend(); ax3d.grid(True, alpha=0.3)

    max_range = max(
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min(),
        0.1,
    )
    xc = (positions[:, 0].max() + positions[:, 0].min()) / 2
    yc = (positions[:, 1].max() + positions[:, 1].min()) / 2
    zc = (positions[:, 2].max() + positions[:, 2].min()) / 2
    ax3d.set_xlim(xc - max_range/2, xc + max_range/2)
    ax3d.set_ylim(yc - max_range/2, yc + max_range/2)
    ax3d.set_zlim(zc - max_range/2, zc + max_range/2)
    ax3d.view_init(elev=20, azim=45)

    ax_steps = fig.add_subplot(3, 3, (7, 9))
    steps = np.arange(len(positions))
    ax_steps.plot(steps, positions[:, 0], 'r-', linewidth=1.5, label='X')
    ax_steps.plot(steps, positions[:, 1], 'g-', linewidth=1.5, label='Y')
    ax_steps.plot(steps, positions[:, 2], 'b-', linewidth=1.5, label='Z')
    ax_steps.set_xlabel('Step'); ax_steps.set_ylabel('Position (m)')
    ax_steps.set_title('X, Y, Z vs Step'); ax_steps.legend(); ax_steps.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Saved TCP trajectory plot to {output_path}")
    plt.close()


def plot(result, out_dir, tcp_frame):
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
    ax.set_title(f"3D {tcp_frame} Trajectory [Piper MuJoCo]"); ax.legend()
    plt.tight_layout(); fig.savefig(str(out_dir / "mujoco_piper_traj_3d.png")); plt.close()

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
    axes[-1].set_xlabel("Step"); axes[0].set_title(f"{tcp_frame} State [Piper MuJoCo]")
    plt.tight_layout(); fig.savefig(str(out_dir / "mujoco_piper_traj_2d_states.png")); plt.close()
    print(f"Saved plots to {out_dir}")


def plot_joints(result, out_dir, filename):
    cmd = result["cmd_joints"]
    obs = result["obs_joints"]
    steps = np.arange(len(cmd))
    fig, axes = plt.subplots(NUM_ARM_JOINTS, 1, figsize=(12, 12), sharex=True)
    for j in range(NUM_ARM_JOINTS):
        ax = axes[j]
        ax.plot(steps, cmd[:, j], "b-o", label="Command", ms=3)
        ax.plot(steps, obs[:, j], "r-s", label="Observed", ms=3)
        ax.set_ylabel(f"J{j+1} (°)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    axes[0].set_title("Joint Commands vs Observations")
    plt.tight_layout()
    fig.savefig(str(out_dir / filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved joints plot to {out_dir / filename}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    frame_spec = resolve_tcp_frame(URDF_PATH, args.tcp_frame, native_frame=SDK_NATIVE_FRAME)

    # --- Plot-only mode: no MuJoCo, just FK + plot ---
    if args.plot_tcp_traj:
        native_kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name=SDK_NATIVE_FRAME,
            joint_names=PIPER_ARM_JOINTS,
        )
        T_base_native = native_kinematics.forward_kinematics(HOME_JOINTS_DEG)
        T_base = pose_from_native(T_base_native, frame_spec)
        print(f"Home {frame_spec.tcp_frame} pos: {np.round(T_base[:3, 3], 6)}")

        traj = load_trajectory(args.traj_csv, T_base, args.steps)
        positions = np.array([s["T_target"][:3, 3] for s in traj])

        print(f"TCP trajectory: {len(positions)} points")
        print(f"  X range: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}] m")
        print(f"  Y range: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}] m")
        print(f"  Z range: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}] m")

        csv_name = Path(args.traj_csv).stem
        plot_tcp_trajectory(positions, out / f"mujoco_piper_{csv_name}_tcp.png", title=f"TCP Trajectory ({frame_spec.tcp_frame})")
        return

    # --- Normal mode: MuJoCo simulation ---
    model, data, kinematics, native_kinematics, target_site_id = create_env(args.model_path, args.tcp_frame)

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Show RGB axes on body frames
            viewer.opt.flags[mujoco.mjtFrame.mjFRAME_BODY] = True
            viewer.sync()

            input("Press Enter to move to home position...")
            T_base = move_to_home(model, data, kinematics, native_kinematics, frame_spec)
            viewer.sync()

            traj = load_trajectory(args.traj_csv, T_base, args.steps)
            input("Press Enter to start trajectory (Ctrl+C to abort)...")

            # Collect trajectory data for plotting
            sent_p, sent_r, act_p, act_r = [], [], [], []
            g_sent, g_act = [], []
            cmd_joints, obs_joints = [], []

            for i, step in enumerate(traj):
                T_target, T_actual, q_cmd_deg = exec_step(model, data, kinematics, step, i, target_site_id)

                cmd_joints.append(q_cmd_deg.copy())
                q_obs_rad = np.array([data.qpos[j] for j in range(NUM_ARM_JOINTS)])
                obs_joints.append(rad2deg(q_obs_rad).copy())

                sent_p.append(T_target[:3, 3].copy())
                sent_r.append(R.from_matrix(T_target[:3, :3]).as_rotvec())
                act_p.append(T_actual[:3, 3].copy())
                act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
                g_sent.append(step["gripper"])
                g_act.append(data.qpos[6] / 0.04)

                viewer.sync()
                if args.step_time > 0:
                    time.sleep(args.step_time)

            # Generate plots
            result = dict(
                sent_pos=np.array(sent_p), sent_rv=np.array(sent_r), g_sent=np.array(g_sent),
                act_pos=np.array(act_p), act_rv=np.array(act_r), g_act=np.array(g_act),
                cmd_joints=np.array(cmd_joints), obs_joints=np.array(obs_joints),
            )
            plot(result, out, frame_spec.tcp_frame)
            plot_joints(result, out, "mujoco_piper_joints.jpg")

            input("Press Enter to return to rest position...")
            go_to_rest(model, data)
            viewer.sync()
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    else:
        # Headless mode: no input prompts
        T_base = move_to_home(model, data, kinematics, native_kinematics, frame_spec)
        traj = load_trajectory(args.traj_csv, T_base, args.steps)

        result, frames = run_trajectory(model, data, kinematics, traj, render=True, target_site_id=target_site_id)
        if frames:
            save_video(frames, out)
        plot(result, out, frame_spec.tcp_frame)
        plot_joints(result, out, "mujoco_piper_joints.jpg")

        go_to_rest(model, data)


if __name__ == "__main__":
    main()

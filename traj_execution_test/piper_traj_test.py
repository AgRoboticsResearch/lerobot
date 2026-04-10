#!/usr/bin/env python
"""Execute a trajectory loaded from CSV on a real Piper robot via SDK.

Supports three modes:
  1. EE mode (default): CSV EE poses → SE(3) compose → EndPoseCtrl (robot IK)
  2. Joint mode (--mode joint): CSV joint angles → Placo IK → JointCtrl
  3. Plot only (--plot-tcp-traj): no robot, just Placo FK to compute
     TCP positions from CSV, then plots XY/XZ/YZ + 3D + xyz vs steps.

Usage:
    # EE mode (default)
    python piper_traj_test.py --traj-csv test_x_axis.csv
    python piper_traj_test.py --traj-csv test_x_axis.csv --steps 10 --step-time 0.1

    # Joint mode
    python piper_traj_test.py --traj-csv joints.csv --mode joint

    # Plot TCP trajectory without robot
    python piper_traj_test.py --traj-csv traj.csv --plot-tcp-traj

    # Change TCP frame (default: ee_link)
    python piper_traj_test.py --traj-csv traj.csv --tcp-frame camera_link
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from frame_utils import SDK_NATIVE_FRAME, pose_from_native, pose_to_native, resolve_tcp_frame
from lerobot.model.kinematics import RobotKinematics

# ============================================================
# Constants
# ============================================================

URDF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description.urdf"
PIPER_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
NUM_ARM_JOINTS = 6

# Rest position (degrees) — folded safe pose, used at start/end
REST_JOINTS_DEG = np.array([0, 0, 0, -2.95, 18.63, -2.92])
# Home position (degrees)
HOME_JOINTS_DEG = np.array([0, 73.77, -43.43, 0, -24, 0])

# Joint limits (degrees): (min, max) per joint
JOINT_LIMITS_DEG = [
    (-150, 150), (0, 180), (-170, 0),
    (-100, 100), (-70, 70), (-120, 120),
]

# Gripper: max opening 50mm
GRIPPER_MAX_M = 0.05


# ============================================================
# Helpers
# ============================================================

def make_transform(pos, rotvec):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = pos
    return T


def read_joints_deg(piper):
    """Read current joint angles from robot (degrees)."""
    msg = piper.GetArmJointMsgs()
    return np.array([
        msg.joint_state.joint_1 / 1000.0,
        msg.joint_state.joint_2 / 1000.0,
        msg.joint_state.joint_3 / 1000.0,
        msg.joint_state.joint_4 / 1000.0,
        msg.joint_state.joint_5 / 1000.0,
        msg.joint_state.joint_6 / 1000.0,
    ])


def read_ee_pose(piper):
    """Read current EE pose from robot → (pos_m, euler_deg)."""
    pose = piper.GetArmEndPoseMsgs().end_pose
    pos_m = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis]) / 1e6  # 0.001mm → m
    euler_deg = np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis]) / 1000.0  # 0.001° → °
    return pos_m, euler_deg


def ee_pose_to_T(pos_m, euler_deg):
    """Convert (pos_meters, euler_deg XYZ convention) → 4x4 matrix."""
    rotvec = R.from_euler("XYZ", euler_deg, degrees=True).as_rotvec()
    return make_transform(pos_m, rotvec)


def T_to_ee_pose(T):
    """Convert 4x4 matrix → (pos_m, euler_deg XYZ convention)."""
    pos_m = T[:3, 3]
    euler_deg = R.from_matrix(T[:3, :3]).as_euler("XYZ", degrees=True)
    return pos_m, euler_deg


def clamp_joints(q_deg):
    """Clamp joint angles to limits, return (clamped, was_clamped)."""
    clamped = q_deg.copy()
    clamped_any = False
    for j in range(NUM_ARM_JOINTS):
        lo, hi = JOINT_LIMITS_DEG[j]
        if clamped[j] < lo:
            clamped[j] = lo
            clamped_any = True
        elif clamped[j] > hi:
            clamped[j] = hi
            clamped_any = True
    return clamped, clamped_any


def parse_args():
    p = argparse.ArgumentParser(description="Execute trajectory on real Piper robot")
    p.add_argument("--traj-csv", required=True, help="Path to trajectory CSV")
    p.add_argument("--mode", choices=["ee", "joint"], default="ee",
                   help="Control mode: ee=EndPoseCtrl, joint=JointCtrl (default: ee)")
    p.add_argument("--steps", type=int, default=None, help="Limit number of steps")
    p.add_argument("--step-time", type=float, default=0.1, help="Delay between steps (seconds)")
    p.add_argument("--can-name", default="can0", help="CAN interface name")
    p.add_argument("--speed", type=int, default=100, help="Speed rate 0-100 (default: 100)")
    p.add_argument("--home-joints", default=None,
                   help="Comma-separated 6 joint values in degrees (default: built-in home)")
    p.add_argument("--tcp-frame", default="ee_link",
                   help="URDF frame to use as the planning TCP (default: ee_link)")
    p.add_argument("--plot-tcp-traj", action="store_true",
                   help="Plot TCP trajectory without connecting to robot")
    return p.parse_args()


# ============================================================
# Robot setup
# ============================================================

def go_to_rest(piper, speed):
    """Move arm to rest position and wait for convergence."""
    print(f"Moving to rest: {REST_JOINTS_DEG} deg")
    rest_mdeg = [round(d * 1000) for d in REST_JOINTS_DEG]
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*rest_mdeg)
    for i in range(100):
        q = read_joints_deg(piper)
        err = np.max(np.abs(q - REST_JOINTS_DEG))
        if err < 1.0:
            print(f"Reached rest position with max joint error {err:.2f}°")
            break
        time.sleep(0.01)
    time.sleep(1.0)  # wait for motion to start
    print("Joint State at rest: ", np.round(read_joints_deg(piper), 2))

def create_robot(can_name, home_deg, speed, frame_spec):
    """Connect, enable, move to rest, then home, read T_base."""
    from piper_sdk import C_PiperInterface_V2

    print(f"Connecting to Piper on {can_name}...")
    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()

    print("Enabling robot...")
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("Robot enabled.")

    # Install SIGINT handler early so Ctrl+C always goes to rest then disables
    def on_sigint(sig, frame):
        print("\nSIGINT: going to rest...")
        try:
            go_to_rest(piper, speed)
        except Exception:
            pass
        print("Disabling robot...")
        try:
            piper.DisablePiper()
        except Exception:
            pass
        sys.exit(1)
    signal.signal(signal.SIGINT, on_sigint)

    # Enable gripper
    piper.GripperCtrl(0, 1000, 0x02, 0)  # reset
    piper.GripperCtrl(0, 1000, 0x01, 0)  # enable
    time.sleep(0.1)

    # Go to rest first
    input("Press Enter to move to rest position...")
    go_to_rest(piper, speed)

    # Move to home (MOVE J)
    input("Press Enter to move to home position...")
    print(f"Moving to home: {np.round(home_deg, 2)} deg")
    home_mdeg = [round(d * 1000) for d in home_deg]
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*home_mdeg)

    # Wait for convergence
    err = 999.0
    start = time.time()
    while time.time() - start < 15:
        q = read_joints_deg(piper)
        err = np.max(np.abs(q - home_deg))
        if err < 1.0:
            break
        time.sleep(0.05)
    else:
        print(f"WARNING: home convergence timeout, max error={err:.2f}°")

    q_home = read_joints_deg(piper)
    pos_m, euler_deg = read_ee_pose(piper)
    T_base_native = ee_pose_to_T(pos_m, euler_deg)
    T_base = pose_from_native(T_base_native, frame_spec)
    tcp_pos_m, tcp_euler_deg = T_to_ee_pose(T_base)
    offset_pos_m, offset_euler_deg = T_to_ee_pose(frame_spec.T_native_to_tcp)

    print(f"Home joints (deg): {np.round(q_home, 2)}")
    print(f"Home {frame_spec.native_frame} pos (mm):  {np.round(pos_m * 1000, 2)}")
    print(f"Home {frame_spec.native_frame} rot (deg): {np.round(euler_deg, 2)}")
    print(f"Home {frame_spec.tcp_frame} pos (mm):  {np.round(tcp_pos_m * 1000, 2)}")
    print(f"Home {frame_spec.tcp_frame} rot (deg): {np.round(tcp_euler_deg, 2)}")
    print(f"Offset {frame_spec.native_frame} -> {frame_spec.tcp_frame} pos (mm): {np.round(offset_pos_m * 1000, 2)}")
    print(f"Offset {frame_spec.native_frame} -> {frame_spec.tcp_frame} rot (deg): {np.round(offset_euler_deg, 2)}")

    # Initialize Placo kinematics solver
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=frame_spec.tcp_frame,
        joint_names=PIPER_ARM_JOINTS,
    )

    return piper, T_base, kinematics


# ============================================================
# CSV loading
# ============================================================

def load_trajectory(csv_path, T_base, max_steps=None):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} steps from {csv_path}")

    traj = []
    for _, row in df.iterrows():
        rel_pos = [row["action.ee.x"], row["action.ee.y"], row["action.ee.z"]]
        rel_rv = [row.get("action.ee.wx", 0), row.get("action.ee.wy", 0), row.get("action.ee.wz", 0)]
        gripper = row["action.ee.gripper_pos"]
        T_t = T_base @ make_transform(rel_pos, rel_rv)
        traj.append({"T_target": T_t.copy(), "gripper": gripper})

    if max_steps:
        traj = traj[:max_steps]
    print(f"Prepared {len(traj)} trajectory steps")
    return traj


# ============================================================
# Execution
# ============================================================

def exec_step(piper, step, i, speed, mode, frame_spec, kinematics=None):
    """Send one trajectory step to the robot. Returns commanded joints (None for ee mode)."""
    gripper = step["gripper"]
    T = step["T_target"]
    cmd_joints = None

    if mode == "ee":
        # Convert planning TCP pose into the SDK-native frame before EndPoseCtrl.
        T_native = pose_to_native(T, frame_spec)
        pos_m, euler_deg = T_to_ee_pose(T_native)
        piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)  # MOVE P
        piper.EndPoseCtrl(
            round(pos_m[0] * 1e6),      # m → 0.001mm
            round(pos_m[1] * 1e6),
            round(pos_m[2] * 1e6),
            round(euler_deg[0] * 1000),  # deg → 0.001°
            round(euler_deg[1] * 1000),
            round(euler_deg[2] * 1000),
        )
    else:  # joint mode: Placo IK → JointCtrl
        q_cur_deg = read_joints_deg(piper)
        q_target_deg = kinematics.inverse_kinematics(
            current_joint_pos=q_cur_deg,
            desired_ee_pose=T,
            position_weight=1.0,
            orientation_weight=0.1,
        )
        q_clamped, was_clamped = clamp_joints(q_target_deg)
        if was_clamped:
            print(f"  WARNING: joints clamped {np.round(q_target_deg, 2)} → {np.round(q_clamped, 2)}")

        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)  # MOVE J
        piper.JointCtrl(*[round(d * 1000) for d in q_clamped])
        cmd_joints = q_clamped

    # Gripper: proportional [0,1] → [0, GRIPPER_MAX_M]
    grip_m = gripper * GRIPPER_MAX_M
    piper.GripperCtrl(abs(round(grip_m * 1e6)), 1000, 0x01, 0)
    return cmd_joints


def read_actual(piper, step, frame_spec):
    """Read actual SDK pose from robot and convert it into the planning TCP frame."""
    pos_m, euler_deg = read_ee_pose(piper)
    T_actual_native = ee_pose_to_T(pos_m, euler_deg)
    T_actual = pose_from_native(T_actual_native, frame_spec)
    T_target = step["T_target"]
    pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
    return T_target, T_actual, pos_err


# ============================================================
# Main loop
# ============================================================

def run_trajectory(piper, traj, step_time, speed, mode, frame_spec, kinematics=None):
    # Pre-flight safety check
    print(f"\n{'='*60}")
    print(
        f"Trajectory: {len(traj)} steps, mode={mode}, tcp_frame={frame_spec.tcp_frame}, "
        f"sdk_frame={frame_spec.native_frame}, dt={step_time}s, speed={speed}%"
    )

    first_pos, first_euler = T_to_ee_pose(traj[0]["T_target"])
    last_pos, last_euler = T_to_ee_pose(traj[-1]["T_target"])
    print(
        f"First target ({frame_spec.tcp_frame}): pos={np.round(first_pos*1000, 1)}mm  "
        f"rot={np.round(first_euler, 1)}°"
    )
    print(
        f"Last target ({frame_spec.tcp_frame}):  pos={np.round(last_pos*1000, 1)}mm  "
        f"rot={np.round(last_euler, 1)}°"
    )
    print(f"{'='*60}")

    input("Press Enter to start trajectory (Ctrl+C to abort)...")

    sent_p, act_p = [], []
    sent_r, act_r = [], []
    g_sent, g_act = [], []
    cmd_joints_list, obs_joints_list = [], []
    errors = []

    try:
        for i, step in enumerate(traj):
            cmd_q = exec_step(piper, step, i, speed, mode, frame_spec, kinematics)
            time.sleep(step_time)

            # Read actual joints
            q_actual = read_joints_deg(piper)
            obs_joints_list.append(q_actual.copy())

            # Commanded joints: from exec_step (joint mode) or Placo IK (ee mode)
            if cmd_q is not None:
                cmd_joints_list.append(cmd_q.copy())
            elif kinematics is not None:
                q_ik = kinematics.inverse_kinematics(
                    current_joint_pos=q_actual,
                    desired_ee_pose=step["T_target"],
                    position_weight=1.0,
                    orientation_weight=0.1,
                )
                cmd_joints_list.append(q_ik.copy())

            # Read actual state in the selected TCP frame.
            T_sent, T_actual, pos_err = read_actual(piper, step, frame_spec)
            sent_p.append(T_sent[:3, 3].copy())
            act_p.append(T_actual[:3, 3].copy())
            sent_r.append(R.from_matrix(T_sent[:3, :3]).as_rotvec())
            act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
            print(f"  Step {i:3d}: pos_err={pos_err*1000:.2f}mm")
            print(
                f"    Sent ({frame_spec.tcp_frame}): pos={np.round(T_sent[:3,3]*1000,2).tolist()}mm "
                f"grip={step['gripper']:.2f}"
            )
            print(f"    True ({frame_spec.tcp_frame}): pos={np.round(T_actual[:3,3]*1000,2).tolist()}mm")
            errors.append(pos_err * 1000)

            # Read gripper
            gmsg = piper.GetArmGripperMsgs()
            grip_actual_m = gmsg.gripper_state.grippers_angle / 1e6  # 0.001mm → m
            g_sent.append(step["gripper"])
            g_act.append(grip_actual_m / GRIPPER_MAX_M)  # normalize to [0,1]

    except KeyboardInterrupt:
        print("\nTrajectory interrupted by user!")

    # Summary
    if errors:
        print(f"\nTrajectory done. {len(errors)}/{len(traj)} steps executed.")
        print(f"  Mean error: {np.mean(errors):.3f}mm")
        print(f"  Max  error: {np.max(errors):.3f}mm")

    result = dict(
        sent_pos=np.array(sent_p), act_pos=np.array(act_p),
        sent_rv=np.array(sent_r), act_rv=np.array(act_r),
        g_sent=np.array(g_sent), g_act=np.array(g_act),
        cmd_joints=np.array(cmd_joints_list), obs_joints=np.array(obs_joints_list),
    )
    return result


# ============================================================
# Plot
# ============================================================

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


def plot(result, out_dir, mode, tcp_frame):
    sp, ap = result["sent_pos"], result["act_pos"]
    sr, ar = result["sent_rv"], result["act_rv"]
    gs, ga = result["g_sent"], result["g_act"]
    steps = np.arange(len(sp))

    se = R.from_rotvec(sr.reshape(-1, 3)).as_euler("xyz")
    ae = R.from_rotvec(ar.reshape(-1, 3)).as_euler("xyz")

    # 3D plot
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
    ax.set_title(f"3D {tcp_frame} Trajectory [Piper Real, {mode} mode]"); ax.legend()
    plt.tight_layout(); fig.savefig(str(out_dir / "real_piper_traj_3d.png")); plt.close()

    # 2D plots
    labels = ["X(m)", "Y(m)", "Z(m)", "Roll", "Pitch", "Yaw", "Grip"]
    sd = np.column_stack([sp, se, gs])
    ad = np.column_stack([ap, ae, ga])
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    for a, lb, s, v in zip(axes, labels, sd.T, ad.T):
        a.plot(steps, s, "b-o", label="Sent", ms=3)
        a.plot(steps, v, "r-s", label="Actual", ms=3)
        a.set_ylabel(lb); a.legend()
    axes[-1].set_xlabel("Step")
    axes[0].set_title(f"{tcp_frame} State [Piper Real, {mode} mode]")
    plt.tight_layout(); fig.savefig(str(out_dir / "real_piper_traj_2d_states.png")); plt.close()
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

    # Parse home joints
    if args.home_joints:
        home_deg = np.array([float(x) for x in args.home_joints.split(",")])
        assert len(home_deg) == NUM_ARM_JOINTS, f"Need {NUM_ARM_JOINTS} joint values, got {len(home_deg)}"
    else:
        home_deg = HOME_JOINTS_DEG

    # --- Plot-only mode: no robot, just FK + plot ---
    if args.plot_tcp_traj:
        native_kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name=SDK_NATIVE_FRAME,
            joint_names=PIPER_ARM_JOINTS,
        )
        T_base_native = native_kinematics.forward_kinematics(home_deg)
        T_base = pose_from_native(T_base_native, frame_spec)
        print(f"Home {frame_spec.tcp_frame} pos: {np.round(T_base[:3, 3], 6)}")

        traj = load_trajectory(args.traj_csv, T_base, args.steps)
        positions = np.array([s["T_target"][:3, 3] for s in traj])

        print(f"TCP trajectory: {len(positions)} points")
        print(f"  X range: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}] m")
        print(f"  Y range: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}] m")
        print(f"  Z range: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}] m")

        csv_name = Path(args.traj_csv).stem
        plot_tcp_trajectory(positions, out / f"real_piper_{csv_name}_tcp.png", title=f"TCP Trajectory ({frame_spec.tcp_frame})")
        return

    # --- Normal mode: connect to real robot ---
    piper, T_base, kinematics = create_robot(args.can_name, home_deg, args.speed, frame_spec)

    # Load and execute trajectory
    traj = load_trajectory(args.traj_csv, T_base, args.steps)
    result = run_trajectory(piper, traj, args.step_time, args.speed, args.mode, frame_spec, kinematics)
    plot(result, out, args.mode, frame_spec.tcp_frame)
    plot_joints(result, out, "real_piper_joints.jpg")

    # Go to rest, then disable
    go_to_rest(piper, args.speed)
    print("Disabling robot...")
    piper.DisablePiper()
    print("Done.")


if __name__ == "__main__":
    main()

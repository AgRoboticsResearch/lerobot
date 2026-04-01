#!/usr/bin/env python
"""Execute a trajectory loaded from CSV on a real Piper robot via SDK.

Supports two modes:
  EE mode (default):    CSV EE poses → SE(3) compose → EndPoseCtrl (robot IK)
  Joint mode (--mode joint): CSV joint angles → JointCtrl directly

Usage:
    # EE mode (default)
    python piper_traj_test.py --traj-csv test_x_axis.csv
    python piper_traj_test.py --traj-csv test_x_axis.csv --steps 10 --step-time 0.1

    # Joint mode
    python piper_traj_test.py --traj-csv joints.csv --mode joint
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from piper_sdk import C_PiperInterface_V2
from scipy.spatial.transform import Rotation as R

from lerobot.model.kinematics import RobotKinematics

# ============================================================
# Constants
# ============================================================

URDF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description_old.urdf"
PIPER_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
NUM_ARM_JOINTS = 6

# Rest position (degrees) — folded safe pose, used at start/end
REST_JOINTS_DEG = np.array([0, 0, 0, -2.95, 18.63, -2.92])
# Home position (degrees)
HOME_JOINTS_DEG = np.array([-1.40, 30.24, -58.64, -1.05, 32.45, 0.00])

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

def create_robot(can_name, home_deg, speed):
    """Connect, enable, move to rest, then home, read T_base."""
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
    T_base = ee_pose_to_T(pos_m, euler_deg)

    print(f"Home joints (deg): {np.round(q_home, 2)}")
    print(f"Home EE pos (mm):  {np.round(pos_m * 1000, 2)}")
    print(f"Home EE rot (deg): {np.round(euler_deg, 2)}")

    # Initialize Placo kinematics solver
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="link6",
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

def exec_step(piper, step, i, speed, mode, kinematics=None):
    """Send one trajectory step to the robot."""
    gripper = step["gripper"]
    T = step["T_target"]

    if mode == "ee":
        # Let robot firmware handle IK
        pos_m, euler_deg = T_to_ee_pose(T)
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

    # Gripper: proportional [0,1] → [0, GRIPPER_MAX_M]
    grip_m = gripper * GRIPPER_MAX_M
    piper.GripperCtrl(abs(round(grip_m * 1e6)), 1000, 0x01, 0)


def read_actual(piper, step):
    """Read actual EE pose from robot for logging."""
    pos_m, euler_deg = read_ee_pose(piper)
    T_actual = ee_pose_to_T(pos_m, euler_deg)
    T_target = step["T_target"]
    pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
    return T_target, T_actual, pos_err


# ============================================================
# Main loop
# ============================================================

def run_trajectory(piper, traj, step_time, speed, mode, kinematics=None):
    # Pre-flight safety check
    print(f"\n{'='*60}")
    print(f"Trajectory: {len(traj)} steps, mode={mode}, dt={step_time}s, speed={speed}%")

    first_pos, first_euler = T_to_ee_pose(traj[0]["T_target"])
    last_pos, last_euler = T_to_ee_pose(traj[-1]["T_target"])
    print(f"First target: pos={np.round(first_pos*1000, 1)}mm  rot={np.round(first_euler, 1)}°")
    print(f"Last target:  pos={np.round(last_pos*1000, 1)}mm  rot={np.round(last_euler, 1)}°")
    print(f"{'='*60}")

    input("Press Enter to start trajectory (Ctrl+C to abort)...")

    sent_p, act_p = [], []
    sent_r, act_r = [], []
    g_sent, g_act = [], []
    errors = []

    try:
        for i, step in enumerate(traj):
            exec_step(piper, step, i, speed, mode, kinematics)
            time.sleep(step_time)

            # Read actual state (always EE)
            T_sent, T_actual, pos_err = read_actual(piper, step)
            sent_p.append(T_sent[:3, 3].copy())
            act_p.append(T_actual[:3, 3].copy())
            sent_r.append(R.from_matrix(T_sent[:3, :3]).as_rotvec())
            act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
            print(f"  Step {i:3d}: pos_err={pos_err*1000:.2f}mm")
            print(f"    Sent: pos={np.round(T_sent[:3,3]*1000,2).tolist()}mm grip={step['gripper']:.2f}")
            print(f"    True: pos={np.round(T_actual[:3,3]*1000,2).tolist()}mm")
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
    )
    return result


# ============================================================
# Plot
# ============================================================

def plot(result, out_dir, mode):
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
    ax.set_title(f"3D EE Trajectory [Piper Real, {mode} mode]"); ax.legend()
    plt.tight_layout(); fig.savefig(str(out_dir / "traj_3d.png")); plt.close()

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
    axes[0].set_title(f"EE State [Piper Real, {mode} mode]")
    plt.tight_layout(); fig.savefig(str(out_dir / "traj_2d_states.png")); plt.close()
    print(f"Saved plots to {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    # Parse home joints
    if args.home_joints:
        home_deg = np.array([float(x) for x in args.home_joints.split(",")])
        assert len(home_deg) == NUM_ARM_JOINTS, f"Need {NUM_ARM_JOINTS} joint values, got {len(home_deg)}"
    else:
        home_deg = HOME_JOINTS_DEG

    # Connect to robot
    piper, T_base, kinematics = create_robot(args.can_name, home_deg, args.speed)

    # Load and execute trajectory
    traj = load_trajectory(args.traj_csv, T_base, args.steps)
    result = run_trajectory(piper, traj, args.step_time, args.speed, args.mode, kinematics)
    plot(result, out, args.mode)

    # Go to rest, then disable
    go_to_rest(piper, args.speed)
    print("Disabling robot...")
    piper.DisablePiper()
    print("Done.")


if __name__ == "__main__":
    main()

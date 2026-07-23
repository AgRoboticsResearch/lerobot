#!/usr/bin/env python
"""Execute a trajectory loaded from CSV on a real Piper robot via SDK.

Two modes:
  1. Execute (default): run trajectory on real robot, save plots + CSV.
  2. Plot only (--plot-tcp-traj): no robot, just Placo FK to plot TCP positions.

Usage:
    python piper_traj_test.py --traj-csv test_x_axis.csv
    python piper_traj_test.py --traj-csv test_x_axis.csv --steps 10 --step-time 0.1
    python piper_traj_test.py --traj-csv traj.csv --plot-tcp-traj
    python piper_traj_test.py --traj-csv traj.csv --tcp-frame camera_link
"""

import signal
import sys
import time
from pathlib import Path

import numpy as np

from frame_utils import SDK_NATIVE_FRAME, pose_from_native, resolve_tcp_frame
from lerobot.model.kinematics import RobotKinematics
from traj_test_util import (
    load_trajectory, plot, plot_joints, plot_tcp_trajectory,
    save_result_to_csv,
)

# ============================================================
# Constants
# ============================================================

PREFIX = "real_piper"
JOINT_LABELS = [f"J{i+1}" for i in range(6)]
PIPER_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
NUM_ARM_JOINTS = 6

URDF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description.urdf"

REST_JOINTS_DEG = np.array([0, 0, 0, -2.95, 18.63, -2.92])
HOME_JOINTS_DEG = np.array([0, 73.77, -43.43, 0, -24, 0])

JOINT_LIMITS_DEG = [
    (-150, 150), (0, 180), (-170, 0),
    (-100, 100), (-70, 70), (-120, 120),
]

GRIPPER_MAX_M = 0.07


# ============================================================
# Helpers
# ============================================================

def read_joints_deg(piper):
    msg = piper.GetArmJointMsgs()
    return np.array([
        msg.joint_state.joint_1 / 1000.0, msg.joint_state.joint_2 / 1000.0,
        msg.joint_state.joint_3 / 1000.0, msg.joint_state.joint_4 / 1000.0,
        msg.joint_state.joint_5 / 1000.0, msg.joint_state.joint_6 / 1000.0,
    ])


def clamp_joints(q_deg):
    clamped = q_deg.copy()
    clamped_any = False
    for j in range(NUM_ARM_JOINTS):
        lo, hi = JOINT_LIMITS_DEG[j]
        if clamped[j] < lo: clamped[j] = lo; clamped_any = True
        elif clamped[j] > hi: clamped[j] = hi; clamped_any = True
    return clamped, clamped_any


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Execute trajectory on real Piper robot (joint mode)")
    p.add_argument("--traj-csv", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--step-time", type=float, default=0.02)
    p.add_argument("--can-name", default="can0")
    p.add_argument("--speed", type=int, default=100)
    p.add_argument("--home-joints", default=None)
    p.add_argument("--tcp-frame", default="ee_link")
    p.add_argument("--plot-tcp-traj", action="store_true")
    return p.parse_args()


# ============================================================
# Robot setup
# ============================================================

def go_to_rest(piper, speed):
    print(f"Moving to rest: {REST_JOINTS_DEG} deg")
    rest_mdeg = [round(d * 1000) for d in REST_JOINTS_DEG]
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*rest_mdeg)
    for _ in range(100):
        q = read_joints_deg(piper)
        err = np.max(np.abs(q - REST_JOINTS_DEG))
        if err < 1.0:
            print(f"Reached rest position with max joint error {err:.2f} deg")
            break
        time.sleep(0.01)
    time.sleep(1.0)
    print("Joint State at rest: ", np.round(read_joints_deg(piper), 2))


def create_robot(can_name, home_deg, speed, frame_spec):
    from piper_sdk import C_PiperInterface_V2

    print(f"Connecting to Piper on {can_name}...")
    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()

    print("Enabling robot...")
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("Robot enabled.")

    def on_sigint(sig, frame):
        print("\nSIGINT: going to rest...")
        try: go_to_rest(piper, speed)
        except Exception: pass
        print("Disabling robot...")
        try: piper.DisablePiper()
        except Exception: pass
        sys.exit(1)
    signal.signal(signal.SIGINT, on_sigint)

    piper.GripperCtrl(0, 1000, 0x02)
    piper.GripperCtrl(0, 1000, 0x01)
    time.sleep(0.1)

    input("Press Enter to move to rest position...")
    go_to_rest(piper, speed)

    input("Press Enter to move to home position...")
    print(f"Moving to home: {np.round(home_deg, 2)} deg")
    home_mdeg = [round(d * 1000) for d in home_deg]
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*home_mdeg)

    err = 999.0
    start = time.time()
    while time.time() - start < 15:
        q = read_joints_deg(piper)
        err = np.max(np.abs(q - home_deg))
        if err < 1.0: break
        time.sleep(0.05)
    else:
        print(f"WARNING: home convergence timeout, max error={err:.2f} deg")

    q_home = read_joints_deg(piper)
    print(f"Home joints (deg): {np.round(q_home, 2)}")

    native_kinematics = RobotKinematics(urdf_path=str(URDF_PATH), target_frame_name=SDK_NATIVE_FRAME, joint_names=PIPER_ARM_JOINTS)
    kinematics = RobotKinematics(urdf_path=str(URDF_PATH), target_frame_name=frame_spec.tcp_frame, joint_names=PIPER_ARM_JOINTS)

    T_base_native = native_kinematics.forward_kinematics(q_home)
    T_base = pose_from_native(T_base_native, frame_spec)
    print(f"Home {frame_spec.native_frame} pos: {np.round(T_base_native[:3, 3], 6)}")
    print(f"Home {frame_spec.tcp_frame} pos: {np.round(T_base[:3, 3], 6)}")

    return piper, T_base, kinematics


# ============================================================
# Execution
# ============================================================

def exec_step(piper, step, speed, kinematics):
    gripper = step["gripper"]
    T = step["T_target"]

    q_cur_deg = read_joints_deg(piper)
    q_target_deg = kinematics.inverse_kinematics(current_joint_pos=q_cur_deg, desired_ee_pose=T, position_weight=1.0, orientation_weight=0.1)
    q_clamped, was_clamped = clamp_joints(q_target_deg)
    if was_clamped:
        print(f"  WARNING: joints clamped {np.round(q_target_deg, 2)} -> {np.round(q_clamped, 2)}")

    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*[round(d * 1000) for d in q_clamped])

    grip_m = gripper * GRIPPER_MAX_M
    piper.GripperCtrl(abs(round(grip_m * 1e6)), 1000, 0x01)
    return q_clamped


def read_actual(step, q_actual_deg, kinematics):
    T_actual = kinematics.forward_kinematics(q_actual_deg)
    T_target = step["T_target"]
    pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
    return T_target, T_actual, pos_err


def run_trajectory(piper, traj, step_time, speed, frame_spec, kinematics):
    print(f"\n{'='*60}")
    print(f"Trajectory: {len(traj)} steps, tcp_frame={frame_spec.tcp_frame}, dt={step_time}s, speed={speed}%")
    print(f"First target ({frame_spec.tcp_frame}): pos={np.round(traj[0]['T_target'][:3,3]*1000, 1)}mm")
    print(f"Last target ({frame_spec.tcp_frame}):  pos={np.round(traj[-1]['T_target'][:3,3]*1000, 1)}mm")
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
            cmd_q = exec_step(piper, step, speed, kinematics)
            cmd_joints_list.append(cmd_q.copy())
            time.sleep(step_time)

            q_actual = read_joints_deg(piper)
            obs_joints_list.append(q_actual.copy())

            T_sent, T_actual, pos_err = read_actual(step, q_actual, kinematics)
            sent_p.append(T_sent[:3, 3].copy())
            act_p.append(T_actual[:3, 3].copy())
            sent_r.append(R.from_matrix(T_sent[:3, :3]).as_rotvec())
            act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
            print(f"  Step {i:3d}: pos_err={pos_err*1000:.2f}mm")
            print(f"    Sent ({frame_spec.tcp_frame}): pos={np.round(T_sent[:3,3]*1000,2).tolist()}mm grip={step['gripper']:.2f}")
            print(f"    True ({frame_spec.tcp_frame}): pos={np.round(T_actual[:3,3]*1000,2).tolist()}mm")
            errors.append(pos_err * 1000)

            gmsg = piper.GetArmGripperMsgs()
            grip_actual_m = gmsg.gripper_state.grippers_angle / 1e6
            g_sent.append(step["gripper"])
            g_act.append(grip_actual_m / GRIPPER_MAX_M)

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
    frame_spec = resolve_tcp_frame(URDF_PATH, args.tcp_frame, native_frame=SDK_NATIVE_FRAME)

    if args.home_joints:
        home_deg = np.array([float(x) for x in args.home_joints.split(",")])
        assert len(home_deg) == NUM_ARM_JOINTS
    else:
        home_deg = HOME_JOINTS_DEG

    # --- Plot-only mode ---
    if args.plot_tcp_traj:
        native_kinematics = RobotKinematics(urdf_path=str(URDF_PATH), target_frame_name=SDK_NATIVE_FRAME, joint_names=PIPER_ARM_JOINTS)
        T_base_native = native_kinematics.forward_kinematics(home_deg)
        T_base = pose_from_native(T_base_native, frame_spec)
        print(f"Home {frame_spec.tcp_frame} pos: {np.round(T_base[:3, 3], 6)}")

        traj = load_trajectory(args.traj_csv, T_base, args.steps)
        positions = np.array([s["T_target"][:3, 3] for s in traj])
        print(f"TCP trajectory: {len(positions)} points")

        csv_name = Path(args.traj_csv).stem
        plot_tcp_trajectory(positions, out / f"{PREFIX}_{csv_name}_tcp.png", title=f"TCP Trajectory ({frame_spec.tcp_frame})")
        return

    # --- Normal mode ---
    piper, T_base, kinematics = create_robot(args.can_name, home_deg, args.speed, frame_spec)
    traj = load_trajectory(args.traj_csv, T_base, args.steps)
    result = run_trajectory(piper, traj, args.step_time, args.speed, frame_spec, kinematics)
    plot(result, out, frame_spec.tcp_frame, "Piper Real", PREFIX)
    plot_joints(result, out, f"{PREFIX}_joints.jpg", JOINT_LABELS)

    csv_name = Path(args.traj_csv).stem
    save_result_to_csv(result, out, f"{PREFIX}_{csv_name}_result.csv")

    go_to_rest(piper, args.speed)
    print("Disabling robot...")
    piper.DisablePiper()
    print("Done.")


if __name__ == "__main__":
    main()

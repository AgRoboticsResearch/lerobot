#!/usr/bin/env python
"""Execute a trajectory on real SO101 using chunked relative actions (simulating a policy).

Simulates how deploy_relative_ee_so101.py works:
  1. Load absolute EE trajectory from CSV.
  2. Segment it into chunks (simulating policy predictions).
  3. At each chunk boundary, record current EE as chunk_base_pose.
  4. Convert each absolute target within the chunk to a relative action:
       T_rel = chunk_base_pose^{-1} @ T_target_absolute
  5. At execution time, recover absolute target:
       T_target = chunk_base_pose @ T_rel
     (all actions in chunk share the same base — no accumulation)
  6. Read fresh robot state, solve IK, send joints.

This tests the SAME control path as the deploy script (relative → absolute → IK)
without needing a trained policy.

Usage:
    python so101_chunked_traj_test.py --traj-csv test_x_axis.csv --robot-port /dev/ttyACM0
    python so101_chunked_traj_test.py --traj-csv test_x_axis.csv --chunk-size 10 --n-action-steps 8
    python so101_chunked_traj_test.py --traj-csv test_x_axis.csv --steps 50 --step-time 0.033 --csv
"""

import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

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

PREFIX = "chunked_so101"
JOINT_LABELS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
SO101_ARM_JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll",
]
NUM_ARM_JOINTS = 5
NUM_JOINTS = 6

URDF_PATH = Path(__file__).parent.parent / "urdf" / "Simulation" / "SO101" / "so101_sroi.urdf"

GRIPPER_CLOSED_DEG = 0.0
GRIPPER_OPEN_DEG = 100.0

REST_JOINTS_DEG = np.array([0, -90, 90, 0, 0, 0])
RESET_POSE_DEG = np.array([0, -73, 74, 0, 0, 0])


# ============================================================
# SE(3) helpers
# ============================================================

def invert_pose(T):
    """Invert a 4x4 homogeneous transformation."""
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


# ============================================================
# Robot helpers
# ============================================================

def read_robot_state(robot, motor_names):
    """Read full observation dict + joint array from robot."""
    obs_dict = robot.get_observation()
    joints = np.array([obs_dict[f"{name}.pos"] for name in motor_names])
    return obs_dict, joints


def read_arm_joints(robot):
    """Read only arm joint positions (degrees)."""
    obs_dict = robot.get_observation()
    return np.array([obs_dict[f"{name}.pos"] for name in SO101_ARM_JOINTS])


def go_to_rest(robot, motor_names):
    print(f"Moving to rest: {REST_JOINTS_DEG} deg")
    robot.send_action({f"{n}.pos": v for n, v in zip(motor_names, REST_JOINTS_DEG)})
    time.sleep(2.0)
    _, q = read_robot_state(robot, motor_names)
    print(f"Rest reached: {np.round(q, 2)} deg")


def create_robot(robot_port, robot_id, home_deg, tcp_frame):
    motor_names = JOINT_LABELS

    print(f"Connecting to SO101 on {robot_port}...")
    robot_config = SO101FollowerConfig(id=robot_id, port=robot_port, use_degrees=True)
    robot = SO101Follower(robot_config)
    robot.connect(calibrate=True)
    print("Robot connected")

    def on_sigint(_sig, _frame):
        print("\nSIGINT: going to rest...")
        try:
            go_to_rest(robot, motor_names)
        except Exception:
            pass
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
    robot.send_action({f"{n}.pos": v for n, v in zip(motor_names, home_deg)})
    time.sleep(2.0)

    _, q_home = read_robot_state(robot, motor_names)
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
# Chunk segmentation — simulate policy behavior
# ============================================================

def segment_into_chunks(traj, chunk_size, n_action_steps):
    """Segment absolute trajectory into chunks of relative actions.

    For each chunk:
      - Record chunk_base_pose from the first step's T_target.
      - Convert every step in the chunk to relative:
          T_rel = chunk_base_pose^{-1} @ T_target

    Returns list of chunks, where each chunk is a list of dicts:
        {"T_rel": 4x4, "T_abs": 4x4, "gripper": float}
    """
    chunks = []
    i = 0
    while i < len(traj):
        chunk = []
        # chunk_base_pose = the absolute EE pose at chunk start
        chunk_base_pose = traj[i]["T_target"].copy()

        for j in range(chunk_size):
            if i + j >= len(traj):
                break
            step = traj[i + j]
            T_abs = step["T_target"]
            T_rel = invert_pose(chunk_base_pose) @ T_abs
            chunk.append({
                "T_rel": T_rel,
                "T_abs": T_abs,
                "gripper": step["gripper"],
            })

        chunks.append(chunk)
        # Advance by n_action_steps (simulate discarding remaining actions)
        i += n_action_steps

    return chunks


# ============================================================
# Execution — mirrors deploy_relative_ee_so101.py control path
# ============================================================

def exec_relative_step(robot, rel_step, chunk_base_pose, kinematics):
    """Execute one relative action step (same path as deploy script).

    1. Convert relative → absolute:  T_target = chunk_base_pose @ T_rel
    2. Read FRESH robot state for IK seed
    3. Solve IK
    4. Send joints
    """
    # Recover absolute target (same as deploy: chunk_base_pose @ rel_T)
    T_target = chunk_base_pose @ rel_step["T_rel"]
    gripper = rel_step["gripper"]

    # Read FRESH robot state right before IK (same fix as deploy script)
    q_cur = read_arm_joints(robot)

    q_target = kinematics.inverse_kinematics(
        current_joint_pos=q_cur,
        desired_ee_pose=T_target,
        position_weight=1.0,
        orientation_weight=0.1,
    )

    action = {}
    for j, name in enumerate(SO101_ARM_JOINTS):
        action[f"{name}.pos"] = q_target[j]
    grip_deg = GRIPPER_CLOSED_DEG + gripper * (GRIPPER_OPEN_DEG - GRIPPER_CLOSED_DEG)
    action["gripper.pos"] = grip_deg

    robot.send_action(action)

    # Return 6-element array: 5 arm joints + gripper
    return np.append(q_target, grip_deg)


# ============================================================
# Main loop — chunk-based execution
# ============================================================

def run_chunked_trajectory(robot, traj, chunk_size, n_action_steps, step_time,
                           tcp_frame, kinematics, chunk_base_ideal=False):
    """Execute trajectory using chunked relative actions (simulating policy)."""
    chunks = segment_into_chunks(traj, chunk_size, n_action_steps)
    total_steps = sum(min(len(c), n_action_steps) for c in chunks)

    print(f"\n{'='*60}")
    print(f"Chunked trajectory execution")
    print(f"  Total absolute steps: {len(traj)}")
    print(f"  Chunk size:           {chunk_size}")
    print(f"  Action steps/chunk:   {n_action_steps}")
    print(f"  Number of chunks:     {len(chunks)}")
    print(f"  Steps to execute:     ~{total_steps}")
    print(f"  Step time:            {step_time}s ({1/step_time:.1f} Hz)")
    print(f"  TCP frame:            {tcp_frame}")
    print(f"  First target:         {np.round(traj[0]['T_target'][:3,3]*1000, 1)}mm")
    print(f"  Last target:          {np.round(traj[-1]['T_target'][:3,3]*1000, 1)}mm")
    print(f"{'='*60}")
    input("Press Enter to start (Ctrl+C to abort)...")

    sent_p, act_p = [], []
    sent_r, act_r = [], []
    g_sent, g_act = [], []
    cmd_joints_list, obs_joints_list = [], []
    errors = []
    global_step = 0

    last_sent_pose = None  # Track last commanded target across chunks

    try:
        for ci, chunk in enumerate(chunks):
            # --- Chunk boundary: compute chunk_base_pose ---
            if ci == 0 or not chunk_base_ideal:
                # First chunk always from actual; others too unless --chunk-base-ideal
                q_actual = read_arm_joints(robot)
                chunk_base_pose = kinematics.forward_kinematics(q_actual)
                base_src = "actual"
            else:
                # Use last chunk's last sent target (what the "policy" predicted last)
                chunk_base_pose = last_sent_pose.copy()
                base_src = "last_sent"
            print(f"\n--- Chunk {ci} [{base_src}]: base_pos={np.round(chunk_base_pose[:3,3]*1000, 1)}mm, "
                  f"{min(len(chunk), n_action_steps)} actions ---")

            # Execute up to n_action_steps from this chunk
            for ai in range(min(len(chunk), n_action_steps)):
                step = chunk[ai]

                cmd_q = exec_relative_step(robot, step, chunk_base_pose, kinematics)
                cmd_joints_list.append(cmd_q.copy())
                time.sleep(step_time)

                # Read actual state
                _, q_all = read_robot_state(robot, JOINT_LABELS)
                obs_joints_list.append(q_all.copy())

                # FK from actual arm joints for error tracking
                q_arm_actual = q_all[:NUM_ARM_JOINTS]
                T_actual = kinematics.forward_kinematics(q_arm_actual)
                T_target = chunk_base_pose @ step["T_rel"]
                pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
                last_sent_pose = T_target.copy()

                sent_p.append(T_target[:3, 3].copy())
                act_p.append(T_actual[:3, 3].copy())
                sent_r.append(R.from_matrix(T_target[:3, :3]).as_rotvec())
                act_r.append(R.from_matrix(T_actual[:3, :3]).as_rotvec())
                errors.append(pos_err * 1000)

                g_sent.append(step["gripper"])
                g_act.append(
                    (q_all[5] - GRIPPER_CLOSED_DEG)
                    / (GRIPPER_OPEN_DEG - GRIPPER_CLOSED_DEG)
                )

                print(f"  Step {global_step:3d} (chunk {ci}:{ai}): "
                      f"pos_err={pos_err*1000:.2f}mm  "
                      f"target={np.round(T_target[:3,3]*1000,1).tolist()}")

                global_step += 1

    except KeyboardInterrupt:
        print("\nInterrupted!")

    if errors:
        print(f"\nDone. {len(errors)} steps executed across {len(chunks)} chunks.")
        print(f"  Mean error: {np.mean(errors):.3f}mm")
        print(f"  Max  error: {np.max(errors):.3f}mm")

    return dict(
        sent_pos=np.array(sent_p), act_pos=np.array(act_p),
        sent_rv=np.array(sent_r), act_rv=np.array(act_r),
        g_sent=np.array(g_sent), g_act=np.array(g_act),
        cmd_joints=np.array(cmd_joints_list),
        obs_joints=np.array(obs_joints_list),
    )


# ============================================================
# Main
# ============================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Execute trajectory on SO101 using chunked relative actions"
    )
    p.add_argument("--traj-csv", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--step-time", type=float, default=0.033)
    p.add_argument("--robot-port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="so101")
    p.add_argument("--tcp-frame", default="camera_link")
    p.add_argument("--home-joints", default=None)
    p.add_argument("--chunk-size", type=int, default=10,
                   help="Simulated policy chunk size (steps per prediction)")
    p.add_argument("--n-action-steps", type=int, default=8,
                   help="Steps to execute per chunk before predicting next")
    p.add_argument("--plot-tcp-traj", action="store_true",
                   help="Plot TCP trajectory without robot")
    p.add_argument("--save_result_to_csv", action="store_true",
                   help="Save execution result CSV")
    p.add_argument("--chunk-base-ideal", action="store_true",
                   help="Use ideal trajectory position as chunk base (not actual robot pose). "
                        "First chunk still uses actual. Prevents error accumulation.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    if args.home_joints:
        home_deg = np.array([float(x) for x in args.home_joints.split(",")])
        assert len(home_deg) == NUM_JOINTS
    else:
        home_deg = RESET_POSE_DEG

    # --- Plot-only mode (no robot needed) ---
    if args.plot_tcp_traj:
        kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name=args.tcp_frame,
            joint_names=SO101_ARM_JOINTS,
        )
        T_base = kinematics.forward_kinematics(home_deg[:NUM_ARM_JOINTS])
        traj = load_trajectory(args.traj_csv, T_base, args.steps)

        positions = np.array([s["T_target"][:3, 3] for s in traj])
        csv_name = Path(args.traj_csv).stem
        plot_tcp_trajectory(
            positions, out / f"{PREFIX}_{csv_name}_tcp.png",
            title=f"TCP Trajectory ({args.tcp_frame})",
        )

        chunks = segment_into_chunks(traj, args.chunk_size, args.n_action_steps)
        print(f"Chunks: {len(chunks)}, total steps: {len(traj)}")
        for ci, ch in enumerate(chunks):
            base_T = ch[0]["T_abs"]
            print(f"  Chunk {ci}: base={np.round(base_T[:3,3]*1000,1).tolist()}mm, "
                  f"{len(ch)} steps")
        return

    # --- Execution mode ---
    robot, T_base, kinematics, motor_names = create_robot(
        args.robot_port, args.robot_id, home_deg, args.tcp_frame,
    )
    traj = load_trajectory(args.traj_csv, T_base, args.steps)
    result = run_chunked_trajectory(
        robot, traj, args.chunk_size, args.n_action_steps,
        args.step_time, args.tcp_frame, kinematics,
        chunk_base_ideal=args.chunk_base_ideal,
    )

    csv_name = Path(args.traj_csv).stem
    plot(result, out, args.tcp_frame, f"SO101 Chunked (cs={args.chunk_size})", PREFIX)
    plot_joints(result, out, f"{PREFIX}_joints.jpg", JOINT_LABELS)
    if args.save_result_to_csv:
        save_result_to_csv(result, out, f"{PREFIX}_{csv_name}_result.csv")

    go_to_rest(robot, motor_names)
    print("Disconnecting robot...")
    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()

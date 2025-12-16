#!/usr/bin/env python

"""
Batch Debug script to replay VIO trajectory on SO101 robot using LeRobot Processor Pipeline.
Iterates over found CameraTrajectory.txt files.
"""

import os
import time
import numpy as np
import argparse
import csv
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from lerobot.model.kinematics_bac import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.utils import log_say
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# --- Configuration ---
URDF_PATH = "./SO-ARM100/Simulation/SO101/so101_new_calib.urdf" # Relative to workspace root
CONTROL_FREQ = 5  # Hz
FPS = 30

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

IK_MOTOR_NAMES = [n for n in MOTOR_NAMES if n != "gripper"]

# Reset pose (Safe starting position)
RESET_POSE = {
    'shoulder_pan.pos': 0.0,
    'shoulder_lift.pos': -80.0,
    'elbow_flex.pos': 50.0,
    'wrist_flex.pos': 40.0,
    'wrist_roll.pos': 0.0,
    'gripper.pos': 0.0,
}

def busy_wait(duration):
    """Spin-wait for high precision timing."""
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        pass

def parse_kitti_trajectory(file_path):
    waypoints = []
    print(f"Loading VIO waypoints from: {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) == 12: # KITTI format
                vals = [float(x) for x in parts]
                T = np.eye(4)
                T[0, 0:3] = vals[0:3]; T[0, 3] = vals[3]
                T[1, 0:3] = vals[4:7]; T[1, 3] = vals[7]
                T[2, 0:3] = vals[8:11]; T[2, 3] = vals[11]
                waypoints.append(T)
            elif len(parts) >= 3: # XYZ fallback
                 x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                 T = np.eye(4); T[:3, 3] = [x, y, z]
                 waypoints.append(T)
    return waypoints

def compute_raw_deltas(waypoints, scale=1.0):
    if not waypoints: return []
    T0 = waypoints[0]
    T0_inv = np.linalg.inv(T0)
    raw_deltas = []
    for T in waypoints:
        T_delta = T0_inv @ T
        # T_delta[:3, 3] *= scale # Removed spatial scaling as requested
        raw_deltas.append(T_delta)
    return raw_deltas

def prepare_action(pose: dict, action_features: dict) -> dict:
    action = {name: float(0.0) for name in action_features}
    for k, v in pose.items():
        if k in action: action[k] = float(v)
    return action

def replay_segment(robot, kinematics_solver, ee_to_joints_processor, robot_joints_to_ee_pose_processor, vio_file, args):
    print(f"\n--- Processing Segment: {os.path.basename(os.path.dirname(vio_file))} ---")
    
    # 3. Load VIO Trajectory
    vio_waypoints = parse_kitti_trajectory(vio_file)
    
    # Truncate based on scale (Percentage)
    if args.scale < 1.0 and args.scale > 0.0:
        cutoff = int(len(vio_waypoints) * args.scale)
        print(f"Truncating trajectory to {args.scale*100:.0f}%: {cutoff} / {len(vio_waypoints)} steps")
        vio_waypoints = vio_waypoints[:cutoff]

    vio_deltas = compute_raw_deltas(vio_waypoints)
    print(f"Loaded {len(vio_deltas)} VIO steps.")

    if not vio_deltas:
        print("No waypoints loaded. Skipping.")
        return

    # 4. Reset Robot & Get Initial Pose
    print("Resetting robot...")
    reset_action = prepare_action(RESET_POSE, robot.action_features)
    robot.send_action(reset_action)
    time.sleep(3.0)

    # Get Initial T_curr (World Frame)
    obs = robot.get_observation()
    initial_ee_pose = robot_joints_to_ee_pose_processor(obs)
    T_curr = np.eye(4)
    T_curr[:3, 3] = [initial_ee_pose["ee.x"], initial_ee_pose["ee.y"], initial_ee_pose["ee.z"]]
    r = R.from_rotvec([initial_ee_pose["ee.wx"], initial_ee_pose["ee.wy"], initial_ee_pose["ee.wz"]])
    T_curr[:3, :3] = r.as_matrix()
    print(f"Initial Robot Pose:\n{T_curr}")

    # 5. Compute VIO Target Poses (in Robot Frame)
    vio_targets = []
    for delta in vio_deltas:
        vio_targets.append(T_curr @ delta)

    print(f"Ready to replay {len(vio_targets)} poses.")
    
    # 6. Control Loop
    logs = []
    start_time = time.time()
    target_freq = CONTROL_FREQ * args.speed
    period = 1.0 / target_freq
    print(f"Replaying at {args.speed}x speed (dt={period*1000:.1f}ms)")

    try:
        for i, T_target in enumerate(vio_targets):
            loop_start = time.perf_counter()
            
            # A. Get Observation
            obs = robot.get_observation()
            
            # B. Prepare Target Action
            target_pos = T_target[:3, 3]
            target_rot = R.from_matrix(T_target[:3, :3]).as_rotvec()
            
            ee_action = {
                "ee.x": target_pos[0], "ee.y": target_pos[1], "ee.z": target_pos[2],
                "ee.wx": target_rot[0], "ee.wy": target_rot[1], "ee.wz": target_rot[2],
                "ee.gripper_pos": 0.0 
            }
            
            # C. Solve IK
            try:
                joint_action = ee_to_joints_processor((ee_action, obs))
                joint_action["gripper.pos"] = 0.0
                
                # IK Theoretical
                cmd_obs = {k: v for k, v in joint_action.items()}
                ik_ee = robot_joints_to_ee_pose_processor(cmd_obs)
                ik_pos = np.array([ik_ee["ee.x"], ik_ee["ee.y"], ik_ee["ee.z"]])
                
                # D. Send to Robot
                robot.send_action(joint_action)
                
                # E. Log Data (Actual Pose via FK)
                obs_after = robot.get_observation()
                actual_ee = robot_joints_to_ee_pose_processor(obs_after)
                
                # Compute Error
                pos_error = np.linalg.norm(target_pos - np.array([actual_ee["ee.x"], actual_ee["ee.y"], actual_ee["ee.z"]]))
                ik_error = np.linalg.norm(target_pos - ik_pos)
                
                log_entry = {
                    'step': i,
                    'time': time.time() - start_time,
                    'target_x': target_pos[0], 'target_y': target_pos[1], 'target_z': target_pos[2],
                    'ik_x': ik_pos[0], 'ik_y': ik_pos[1], 'ik_z': ik_pos[2],
                    'actual_x': actual_ee["ee.x"], 'actual_y': actual_ee["ee.y"], 'actual_z': actual_ee["ee.z"],
                    'pos_error': pos_error,
                    'ik_error': ik_error
                }
                
                # Log Joint Values
                for name in MOTOR_NAMES:
                    log_entry[f"ik_{name}"] = joint_action[f"{name}.pos"]
                    log_entry[f"actual_{name}"] = obs_after[f"{name}.pos"]
                
                logs.append(log_entry)
                
                if i % 30 == 0:
                    print(f"Step {i}/{len(vio_targets)}: TrackErr={pos_error*1000:.1f}mm, IKErr={ik_error*1000:.1f}mm")
                    
            except Exception as e:
                print(f"IK Failed at step {i}: {e}")

            # Maintain Frequency
            dt = time.perf_counter() - loop_start
            if dt < period:
                busy_wait(period - dt)

    except KeyboardInterrupt:
        print("Interrupted inside segment.")
        raise # Re-raise to stop batch
        
    return logs

def main():
    parser = argparse.ArgumentParser(description="Batch Debug Replay VIO on SO101")
    parser.add_argument("--root_dir", default="sroi_vio", help="Root directory containing segments")
    parser.add_argument("--limit", type=int, default=1, help="Number of segments to replay")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--scale", type=float, default=1.0, help="Percentage of trajectory points per segment (0.0 - 1.0)")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed factor")
    parser.add_argument("--output_dir", default="batch_logs", help="Directory to save logs and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_say("Starting SO101 Batch VIO Replay...")

    # Find Files
    search_pattern = os.path.join(args.root_dir, "**", "CameraTrajectory.txt")
    files = glob.glob(search_pattern, recursive=True)
    files.sort()
    
    if not files:
        print(f"No trajectories found in {args.root_dir}")
        return

    print(f"Found {len(files)} trajectories. Limiting to {args.limit}.")
    files = files[:args.limit]

    # 1. Connect to Robot (Once)
    print(f"Connecting to robot on {args.port}...")
    camera_config = {"camera": OpenCVCameraConfig(index_or_path=0, width=320, height=240, fps=FPS)}
    robot_config = SO101FollowerConfig(port=args.port, id="so101_batch_vio", use_degrees=True, cameras=camera_config)
    robot = SO101Follower(robot_config)
    try:
        robot.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    try:
        # 2. Initialize Kinematics (Once)
        print(f"Initializing kinematics from {URDF_PATH}...")
        kinematics_solver = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name="gripper_frame_link",
            joint_names=IK_MOTOR_NAMES,
        )

        ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[InverseKinematicsEEToJoints(kinematics=kinematics_solver, motor_names=IK_MOTOR_NAMES, initial_guess_current_joints=True)],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        robot_joints_to_ee_pose_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
            steps=[ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=IK_MOTOR_NAMES)],
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )

        # Loop
        all_results = {}
        
        for i, vio_file in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] {vio_file}")
            
            logs = replay_segment(robot, kinematics_solver, ee_to_joints_processor, robot_joints_to_ee_pose_processor, vio_file, args)
            
            if logs:
                segment_name = os.path.basename(os.path.dirname(vio_file))
                all_results[segment_name] = logs
                
                # Save CSV immediately
                filename_base = os.path.join(args.output_dir, f"{segment_name}_log")
                csv_file = f"{filename_base}.csv"
                print(f"Saving logs to {csv_file}")
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                    writer.writeheader()
                    writer.writerows(logs)

        # --- Consolidated Plotting ---
        if all_results:
            n_plots = len(all_results)
            print(f"Plotting {n_plots} segments...")
            
            # 1. Trajectories Plot (Grid of 3D plots)
            import math
            n_cols = 3
            n_rows = math.ceil(n_plots / n_cols)
            fig_traj = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
            
            for idx, (name, logs) in enumerate(all_results.items()):
                ax = fig_traj.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
                
                tgt_x = [l['target_x'] for l in logs]; tgt_y = [l['target_y'] for l in logs]; tgt_z = [l['target_z'] for l in logs]
                ik_x = [l['ik_x'] for l in logs]; ik_y = [l['ik_y'] for l in logs]; ik_z = [l['ik_z'] for l in logs]
                act_x = [l['actual_x'] for l in logs]; act_y = [l['actual_y'] for l in logs]; act_z = [l['actual_z'] for l in logs]
                
                # Plot
                ax.plot(tgt_x, tgt_y, tgt_z, linestyle='--', color='green', label='Target')
                ax.plot(ik_x, ik_y, ik_z, linestyle='-', color='blue', alpha=0.6, label='IK')
                ax.plot(act_x, act_y, act_z, linestyle='-', color='red', label='Actual')
                
                # Errors title
                mean_err = np.mean([l['pos_error'] for l in logs]) * 1000
                max_err = np.max([l['pos_error'] for l in logs]) * 1000
                ax.set_title(f"{name}\nMean: {mean_err:.1f}mm | Max: {max_err:.1f}mm", fontsize=8)
                
                if idx == 0: ax.legend()

            plt.tight_layout()
            traj_plot_path = os.path.join(args.output_dir, "batch_all_trajectories.png")
            plt.savefig(traj_plot_path)
            print(f"Saved {traj_plot_path}")

            # 2. Joints Plot (Grid: Rows=Joints, Cols=Segments)
            # This matches the style of debug_ee_so101 (separated joints) but aggregates all segments.
            
            # If too many segments, the plot will be very wide.
            fig_joints = plt.figure(figsize=(4 * n_plots, 12)) 
            # Height 12 is for 6 rows roughly. Width depends on N segments.
            
            # We want Rows = 6 (Motors), Cols = N (Segments)
            # Subplot index counts row-wise usually.
            
            # Motor Names as Row Headers?
            
            for seg_idx, (name, logs) in enumerate(all_results.items()):
                steps = [l['step'] for l in logs]
                
                for motor_idx, m_name in enumerate(MOTOR_NAMES):
                    # Subplot index: (total_rows, total_cols, current_index)
                    # current_index is 1-based.
                    # We want Motor 0 in Row 0.
                    # Index = motor_idx * n_plots + seg_idx + 1
                    
                    ax = fig_joints.add_subplot(len(MOTOR_NAMES), n_plots, motor_idx * n_plots + seg_idx + 1)
                    
                    ik_vals = [l[f"ik_{m_name}"] for l in logs]
                    act_vals = [l[f"actual_{m_name}"] for l in logs]
                    
                    ax.plot(steps, ik_vals, 'b--', label='IK' if (seg_idx==0 and motor_idx==0) else "")
                    ax.plot(steps, act_vals, 'r', label='Actual' if (seg_idx==0 and motor_idx==0) else "")
                    
                    ax.grid(True)
                    
                    # Only label Y axis on the first column
                    if seg_idx == 0:
                        ax.set_ylabel(f"{m_name}\n(deg)")
                    
                    # Only label X axis on the last row
                    if motor_idx == len(MOTOR_NAMES) - 1:
                        ax.set_xlabel("Step")
                        
                    # Title only on top row
                    if motor_idx == 0:
                        ax.set_title(name, fontsize=8)
                        
            # Legend (Global)
            handles, labels = fig_joints.axes[0].get_legend_handles_labels()
            fig_joints.legend(handles, labels, loc='upper right')

            plt.tight_layout()
            joints_plot_path = os.path.join(args.output_dir, "batch_all_joints.png")
            plt.savefig(joints_plot_path)
            print(f"Saved {joints_plot_path}")

    except KeyboardInterrupt:
        print("\nBatch interrupted.")
    finally:
        print("Stopping robot...")
        robot.disconnect()

if __name__ == "__main__":
    main()

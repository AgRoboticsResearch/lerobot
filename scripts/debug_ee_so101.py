#!/usr/bin/env python

"""
Debug script to replay VIO trajectory on SO101 robot using LeRobot Processor Pipeline.
Replaces ACT policy logic with direct VIO replay.
"""

import os
import time
import numpy as np
import argparse
import csv
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
# from lerobot.utils.robot_utils import busy_wait # Removed as it caused ImportError
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# --- Configuration ---
URDF_PATH = "./SO-ARM100/Simulation/SO101/so101_new_calib.urdf" # Relative to workspace root
CONTROL_FREQ = 10  # Hz
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
        T_delta[:3, 3] *= scale # Apply scaling
        raw_deltas.append(T_delta)
    return raw_deltas

def prepare_action(pose: dict, action_features: dict) -> dict:
    action = {name: float(0.0) for name in action_features}
    for k, v in pose.items():
        if k in action: action[k] = float(v)
    return action

def main():
    parser = argparse.ArgumentParser(description="Replay VIO on SO101 using Processor Pipeline")
    parser.add_argument("vio_file", help="Path to VIO CameraTrajectory.txt")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--scale", type=float, default=1.0, help="Percentage of trajectory points to replay (0.0 - 1.0)")
    parser.add_argument("--output", default="vio_replay_log.csv", help="Output CSV log")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed factor")
    args = parser.parse_args()

    log_say("Starting SO101 VIO Replay...")

    # 1. Connect to Robot
    print(f"Connecting to robot on {args.port}...")
    # Mock camera config since we aren't using policy
    camera_config = {"camera": OpenCVCameraConfig(index_or_path=2, width=320, height=240, fps=FPS)}
    
    robot_config = SO101FollowerConfig(port=args.port, id="so101_vio_replay", use_degrees=True, cameras=camera_config)
    robot = SO101Follower(robot_config)
    try:
        robot.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 2. Initialize Kinematics
    print(f"Initializing kinematics from {URDF_PATH}...")
    kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=IK_MOTOR_NAMES,
    )

    # FK/IK Processors
    # IK: EE Pose + Current Observation -> Joint Positions
    ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[InverseKinematicsEEToJoints(kinematics=kinematics_solver, motor_names=IK_MOTOR_NAMES, initial_guess_current_joints=True)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    # FK: Observation (Joints) -> EE Pose
    robot_joints_to_ee_pose_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=IK_MOTOR_NAMES)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # 3. Load VIO Trajectory
    vio_waypoints = parse_kitti_trajectory(args.vio_file)
    
    # Truncate based on scale
    if args.scale < 1.0 and args.scale > 0.0:
        cutoff = int(len(vio_waypoints) * args.scale)
        print(f"Truncating trajectory to {args.scale*100:.0f}%: {cutoff} / {len(vio_waypoints)} steps")
        vio_waypoints = vio_waypoints[:cutoff]

    vio_deltas = compute_raw_deltas(vio_waypoints, scale=1.0) # Reset delta scale to 1.0 (or add a separate arg if needed later)
    print(f"Loaded {len(vio_deltas)} VIO steps.")

    if not vio_deltas:
        print("No waypoints loaded. Exiting.")
        robot.disconnect()
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
    input("Press Enter to start... (Ctrl+C to stop)")

    # 6. Control Loop
    print("Starting control loop...")
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
                
                # --- NEW: Calculate IK Theoretical Pose (FK on commanded joints) ---
                # Construct a mock observation from the commanded joints to feed into FK
                cmd_obs = {k: v for k, v in joint_action.items()}
                # We need to ensure all motor names are present, though joint_action should have them
                ik_ee = robot_joints_to_ee_pose_processor(cmd_obs)
                ik_pos = np.array([ik_ee["ee.x"], ik_ee["ee.y"], ik_ee["ee.z"]])
                # -----------------------------------------------------------------
                
                # D. Send to Robot
                robot.send_action(joint_action)
                
                # E. Log Data (Actual Pose via FK)
                obs_after = robot.get_observation()
                actual_ee = robot_joints_to_ee_pose_processor(obs_after)
                
                # Compute Error
                pos_error = np.linalg.norm(target_pos - np.array([actual_ee["ee.x"], actual_ee["ee.y"], actual_ee["ee.z"]]))
                ik_error = np.linalg.norm(target_pos - ik_pos) # Error between Target and IK Solution
                
                log_entry = {
                    'step': i,
                    'time': time.time() - start_time,
                    'target_x': target_pos[0], 'target_y': target_pos[1], 'target_z': target_pos[2],
                    'ik_x': ik_pos[0], 'ik_y': ik_pos[1], 'ik_z': ik_pos[2], # Log IK
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
        print("Interrupted.")
    finally:
        print("Stopping robot...")
        robot.disconnect()
        
        # Save Logs
        if logs:
            print(f"Saving logs to {args.output}")
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
            
            # Plot
            print("Plotting...")
            tgt_x = [l['target_x'] for l in logs]
            tgt_y = [l['target_y'] for l in logs]
            tgt_z = [l['target_z'] for l in logs]
            
            ik_x = [l['ik_x'] for l in logs] # Extract IK
            ik_y = [l['ik_y'] for l in logs]
            ik_z = [l['ik_z'] for l in logs]

            act_x = [l['actual_x'] for l in logs]
            act_y = [l['actual_y'] for l in logs]
            act_z = [l['actual_z'] for l in logs]
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(tgt_x, tgt_y, tgt_z, label='VIO Target', color='green', linestyle='--', linewidth=2)
            ax.plot(ik_x, ik_y, ik_z, label='IK Solution', color='blue', linestyle='-', linewidth=1, alpha=0.7) # Plot IK
            ax.plot(act_x, act_y, act_z, label='Actual Robot', color='red', linestyle='-', linewidth=2)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title(f"VIO Replay (Scale={args.scale})")
            ax.legend()
            plt.savefig(args.output.replace('.csv', '_traj.png'))
            
            # Plot Joints
            print("Plotting Joints...")
            fig_joints, axes = plt.subplots(len(MOTOR_NAMES), 1, figsize=(10, 15), sharex=True)
            steps = [l['step'] for l in logs]
            
            for idx, name in enumerate(MOTOR_NAMES):
                ik_vals = [l[f"ik_{name}"] for l in logs]
                act_vals = [l[f"actual_{name}"] for l in logs]
                
                axes[idx].plot(steps, ik_vals, label='IK Command', color='blue', linestyle='--')
                axes[idx].plot(steps, act_vals, label='Actual', color='red')
                axes[idx].set_ylabel(f"{name} (deg)")
                axes[idx].legend(loc='upper right')
                axes[idx].grid(True)
                
            axes[-1].set_xlabel('Step')
            fig_joints.suptitle(f"Joint Tracking (Scale={args.scale})")
            plt.savefig(args.output.replace('.csv', '_joints.png'))
            
            print("Done.")

if __name__ == "__main__":
    main()
# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Initialize the robot and teleoperator config
follower_config = SO100FollowerConfig(
    port="/dev/ttyACM0", id="so101_follower", use_degrees=True
)
leader_config = SO100LeaderConfig(port="/dev/ttyACM1", id="so101_leader")

# Initialize the robot and teleoperator
follower = SO100Follower(follower_config)
leader = SO100Leader(leader_config)

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
follower_kinematics_solver = RobotKinematics(
    urdf_path="./SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(follower.bus.motors.keys()),
)

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
leader_kinematics_solver = RobotKinematics(
    urdf_path="./SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(leader.bus.motors.keys()),
)

# Build pipeline part 1: delta action -> EE pose (with safety checks)
# This pipeline uses EEReferenceAndDelta to apply deltas relative to follower's current pose
ee_delta_to_ee_pose = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        EEReferenceAndDelta(
            kinematics=follower_kinematics_solver,
            end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},  # Scale factors for deltas
            motor_names=list(follower.bus.motors.keys()),
            use_latched_reference=False,  # Use current pose as reference each step
        ),
        EEBoundsAndSafety(
            end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=0.10,
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Build pipeline part 2: EE pose -> follower joints
ee_pose_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        InverseKinematicsEEToJoints(
            kinematics=follower_kinematics_solver,
            motor_names=list(follower.bus.motors.keys()),
            initial_guess_current_joints=False,
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Connect to the robot and teleoperator
follower.connect()
leader.connect()

# Init rerun viewer
init_rerun(session_name="so100_so100_EE_delta_teleop")

print("Starting teleop loop with delta-based control...")
print("Leader movements will be converted to deltas and applied to follower's current pose")

# Initialize previous leader EE pose (will be set on first iteration)
leader_prev_ee_pose: np.ndarray | None = None

# Data recording for accuracy analysis
recorded_data = {
    "timestamps": [],
    "leader_positions": [],  # [x, y, z]
    "follower_positions": [],  # [x, y, z]
    "leader_rotations": [],  # rotation vectors [wx, wy, wz]
    "follower_rotations": [],  # rotation vectors [wx, wy, wz]
    "position_errors": [],  # Euclidean distance
    "rotation_errors": [],  # Angular error in radians
}
start_time = time.time()

try:
    while True:
        t0 = time.perf_counter()

        # Get robot observation (follower's current state)
        robot_obs = follower.get_observation()

        # Get teleop observation (leader's current joint positions)
        leader_joints_obs = leader.get_action()

        # Extract leader joint positions
        leader_joints = np.array(
            [
                float(leader_joints_obs[f"{name}.pos"])
                for name in leader.bus.motors.keys()
                if f"{name}.pos" in leader_joints_obs
            ]
        )

        # Extract leader gripper position (copy directly to follower)
        leader_gripper_pos = None
        if "gripper.pos" in leader_joints_obs:
            leader_gripper_pos = float(leader_joints_obs["gripper.pos"])

        # Extract follower joint positions
        follower_joints = np.array(
            [
                float(robot_obs[f"{name}.pos"])
                for name in follower.bus.motors.keys()
                if f"{name}.pos" in robot_obs
            ]
        )

        # Compute forward kinematics for leader robot
        leader_ee_pose = leader_kinematics_solver.forward_kinematics(leader_joints)

        # Compute forward kinematics for follower robot (current state)
        follower_ee_pose = follower_kinematics_solver.forward_kinematics(follower_joints)

        # On first iteration, initialize previous leader pose
        if leader_prev_ee_pose is None:
            leader_prev_ee_pose = leader_ee_pose.copy()
            # Skip this iteration to have a valid previous pose for next iteration
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            continue

        # Extract positions
        leader_pos_current = leader_ee_pose[:3, 3]
        leader_pos_previous = leader_prev_ee_pose[:3, 3]

        # Compute position delta: leader's movement (current - previous)
        position_delta = leader_pos_current - leader_pos_previous

        # Extract rotations
        leader_rot_current = Rotation.from_matrix(leader_ee_pose[:3, :3])
        leader_rot_previous = Rotation.from_matrix(leader_prev_ee_pose[:3, :3])

        # Compute rotation delta: leader's rotation change
        # This is: leader_rot_current = leader_rot_previous @ delta_rot
        # So: delta_rot = leader_rot_previous.inv() @ leader_rot_current
        rotation_delta = leader_rot_previous.inv() * leader_rot_current
        rotation_delta_vec = rotation_delta.as_rotvec()

        # Update previous leader pose for next iteration
        leader_prev_ee_pose = leader_ee_pose.copy()

        # Create delta action for EEReferenceAndDelta
        # EEReferenceAndDelta expects: target_x, target_y, target_z, target_wx, target_wy, target_wz
        delta_action: RobotAction = {
            "enabled": True,
            "target_x": float(position_delta[0]),
            "target_y": float(position_delta[1]),
            "target_z": float(position_delta[2]),
            "target_wx": float(rotation_delta_vec[0]),
            "target_wy": float(rotation_delta_vec[1]),
            "target_wz": float(rotation_delta_vec[2]),
            "gripper_vel": 0.0,  # Not used anymore, but kept for EEReferenceAndDelta compatibility
        }

        # Step 1: Apply delta to follower's current pose (EEReferenceAndDelta + EEBoundsAndSafety)
        ee_pose_action = ee_delta_to_ee_pose((delta_action, robot_obs))

        # Step 2: Manually set gripper position from leader (copy directly, no delta)
        if leader_gripper_pos is not None:
            ee_pose_action["ee.gripper_pos"] = leader_gripper_pos
        else:
            # Fallback: use current follower gripper position if leader doesn't have it
            ee_pose_action["ee.gripper_pos"] = robot_obs.get("gripper.pos", 0.0)
            print("WARNING: Leader gripper position not found, using follower's current position")

        # Step 3: Convert EE pose to joint angles (InverseKinematicsEEToJoints)
        follower_joints_act = ee_pose_to_follower_joints((ee_pose_action, robot_obs))

        print(f"Position delta: [{position_delta[0]:.4f}, {position_delta[1]:.4f}, {position_delta[2]:.4f}]")
        print(f"Rotation delta: [{rotation_delta_vec[0]:.4f}, {rotation_delta_vec[1]:.4f}, {rotation_delta_vec[2]:.4f}]")

        # Send action to robot
        _ = follower.send_action(follower_joints_act)

        # Get follower EE pose after IK (actual achieved pose)
        follower_ee_after_ik = follower_kinematics_solver.forward_kinematics(
            np.array([follower_joints_act[f"{name}.pos"] for name in follower.bus.motors.keys()])
        )

        # Record data for accuracy analysis
        current_time = time.time() - start_time
        leader_pos = leader_ee_pose[:3, 3]
        follower_pos = follower_ee_after_ik[:3, 3]
        leader_rot = Rotation.from_matrix(leader_ee_pose[:3, :3]).as_rotvec()
        follower_rot = Rotation.from_matrix(follower_ee_after_ik[:3, :3]).as_rotvec()

        # Calculate errors
        position_error = np.linalg.norm(leader_pos - follower_pos)
        rotation_error = np.linalg.norm(leader_rot - follower_rot)  # Angular error in radians

        recorded_data["timestamps"].append(current_time)
        recorded_data["leader_positions"].append(leader_pos.copy())
        recorded_data["follower_positions"].append(follower_pos.copy())
        recorded_data["leader_rotations"].append(leader_rot.copy())
        recorded_data["follower_rotations"].append(follower_rot.copy())
        recorded_data["position_errors"].append(position_error)
        recorded_data["rotation_errors"].append(rotation_error)

        # Visualize (convert follower joints to EE pose for visualization)
        follower_ee_vis = {
            "ee.x": float(follower_ee_after_ik[0, 3]),
            "ee.y": float(follower_ee_after_ik[1, 3]),
            "ee.z": float(follower_ee_after_ik[2, 3]),
        }
        log_rerun_data(observation=follower_ee_vis, action=follower_joints_act)

        # Print error statistics every 30 frames (~1 second at 30 FPS)
        if len(recorded_data["timestamps"]) % 30 == 0:
            avg_pos_error = np.mean(recorded_data["position_errors"][-30:])
            avg_rot_error = np.mean(recorded_data["rotation_errors"][-30:])
            print(f"Avg position error (last 1s): {avg_pos_error*1000:.2f}mm, Avg rotation error: {np.degrees(avg_rot_error):.2f}°")

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

# After loop ends (Ctrl+C), visualize and save results
except KeyboardInterrupt:
    print("\nTeleoperation stopped. Generating accuracy analysis...")
finally:
    if len(recorded_data["timestamps"]) > 0:
        # Convert to numpy arrays
        timestamps = np.array(recorded_data["timestamps"])
        leader_positions = np.array(recorded_data["leader_positions"])
        follower_positions = np.array(recorded_data["follower_positions"])
        leader_rotations = np.array(recorded_data["leader_rotations"])
        follower_rotations = np.array(recorded_data["follower_rotations"])
        position_errors = np.array(recorded_data["position_errors"])
        rotation_errors = np.array(recorded_data["rotation_errors"])

        # Create output directory
        output_dir = Path("teleop_accuracy_analysis")
        output_dir.mkdir(exist_ok=True)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))

        # Position comparison (X, Y, Z)
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(timestamps, leader_positions[:, 0], "b-", label="Leader X", linewidth=2)
        ax1.plot(timestamps, follower_positions[:, 0], "r--", label="Follower X", linewidth=2)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position X (m)")
        ax1.set_title("Position X Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(timestamps, leader_positions[:, 1], "b-", label="Leader Y", linewidth=2)
        ax2.plot(timestamps, follower_positions[:, 1], "r--", label="Follower Y", linewidth=2)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position Y (m)")
        ax2.set_title("Position Y Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(timestamps, leader_positions[:, 2], "b-", label="Leader Z", linewidth=2)
        ax3.plot(timestamps, follower_positions[:, 2], "r--", label="Follower Z", linewidth=2)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Position Z (m)")
        ax3.set_title("Position Z Comparison")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Rotation comparison (wx, wy, wz)
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(timestamps, leader_rotations[:, 0], "b-", label="Leader wx", linewidth=2)
        ax4.plot(timestamps, follower_rotations[:, 0], "r--", label="Follower wx", linewidth=2)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Rotation wx (rad)")
        ax4.set_title("Rotation wx Comparison")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(timestamps, leader_rotations[:, 1], "b-", label="Leader wy", linewidth=2)
        ax5.plot(timestamps, follower_rotations[:, 1], "r--", label="Follower wy", linewidth=2)
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Rotation wy (rad)")
        ax5.set_title("Rotation wy Comparison")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(timestamps, leader_rotations[:, 2], "b-", label="Leader wz", linewidth=2)
        ax6.plot(timestamps, follower_rotations[:, 2], "r--", label="Follower wz", linewidth=2)
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Rotation wz (rad)")
        ax6.set_title("Rotation wz Comparison")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Error plots
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(timestamps, position_errors * 1000, "g-", linewidth=2)  # Convert to mm
        ax7.set_xlabel("Time (s)")
        ax7.set_ylabel("Position Error (mm)")
        ax7.set_title("Position Error Over Time")
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=np.mean(position_errors) * 1000, color="r", linestyle="--", label=f"Mean: {np.mean(position_errors)*1000:.2f}mm")
        ax7.legend()

        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(timestamps, np.degrees(rotation_errors), "m-", linewidth=2)  # Convert to degrees
        ax8.set_xlabel("Time (s)")
        ax8.set_ylabel("Rotation Error (degrees)")
        ax8.set_title("Rotation Error Over Time")
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=np.degrees(np.mean(rotation_errors)), color="r", linestyle="--", label=f"Mean: {np.degrees(np.mean(rotation_errors)):.2f}°")
        ax8.legend()

        # Error statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis("off")
        stats_text = f"""
Accuracy Statistics:

Position Error:
  Mean: {np.mean(position_errors)*1000:.2f} mm
  Std:  {np.std(position_errors)*1000:.2f} mm
  Max:  {np.max(position_errors)*1000:.2f} mm
  Min:  {np.min(position_errors)*1000:.2f} mm

Rotation Error:
  Mean: {np.degrees(np.mean(rotation_errors)):.2f}°
  Std:  {np.degrees(np.std(rotation_errors)):.2f}°
  Max:  {np.degrees(np.max(rotation_errors)):.2f}°
  Min:  {np.degrees(np.min(rotation_errors)):.2f}°

Total Samples: {len(timestamps)}
Duration: {timestamps[-1]:.2f} s
        """
        ax9.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center", family="monospace")

        plt.tight_layout()
        plt.savefig(output_dir / "delta_control_accuracy_analysis.png", dpi=150, bbox_inches="tight")
        print(f"\nAccuracy analysis saved to: {output_dir / 'delta_control_accuracy_analysis.png'}")

        # Save raw data
        np.savez(
            output_dir / "recorded_data.npz",
            timestamps=timestamps,
            leader_positions=leader_positions,
            follower_positions=follower_positions,
            leader_rotations=leader_rotations,
            follower_rotations=follower_rotations,
            position_errors=position_errors,
            rotation_errors=rotation_errors,
        )
        print(f"Raw data saved to: {output_dir / 'recorded_data.npz'}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("Accuracy Summary")
        print("=" * 60)
        print(f"Position Error - Mean: {np.mean(position_errors)*1000:.2f}mm, Std: {np.std(position_errors)*1000:.2f}mm")
        print(f"Rotation Error - Mean: {np.degrees(np.mean(rotation_errors)):.2f}°, Std: {np.degrees(np.std(rotation_errors)):.2f}°")
        print("=" * 60)

        plt.show()
    else:
        print("\nNo data recorded. Exiting...")


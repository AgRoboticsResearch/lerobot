#!/usr/bin/env python

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

r"""
Deploy a trained ACT policy with RelativeEEDataset (10D relative EE actions) on SO101 robot
and visualize predicted vs actual trajectory.

This script:
1. Loads a policy trained with RelativeEEDataset
2. Moves robot to reset pose
3. Makes ONE prediction from the reset pose
4. Executes the entire action chunk while recording joint states
5. Computes EE trajectories via FK and plots predicted vs actual

Usage:
    python deploy_relative_ee_so101_visualize.py \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --urdf_path /path/to/so101.urdf \
        --robot_port /dev/ttyUSB0 \
        --num_actions 50
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, TransitionKey
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so101_follower.relative_ee_processor import (
    Relative10DAccumulatedToAbsoluteEE,
    pose10d_to_mat,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.utils import init_logging
from lerobot.utils.robot_utils import precise_sleep

# Motor names for SO101
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Default RESET pose (starting position)
RESET_POSE_DEG = np.array([
    -5.54,    # shoulder_pan
    -114.59,  # shoulder_lift
    104.44,   # elbow_flex
    7.84,    # wrist_flex
    -5.19,    # wrist_roll
    35.13,    # gripper
])

FPS = 10  # Control loop frequency


def parse_cameras_config(cameras_str: str | None) -> dict[str, Any]:
    """
    Parse cameras configuration from YAML string.

    Args:
        cameras_str: YAML-formatted string like "{ wrist: {type: intelrealsense, ...} }"

    Returns:
        Dictionary mapping camera names to CameraConfig objects
    """
    if not cameras_str or cameras_str.strip() == "":
        return {}

    # Parse YAML string to dict
    cameras_dict = yaml.safe_load(cameras_str)

    if not isinstance(cameras_dict, dict):
        raise ValueError(f"Cameras config should be a dict, got {type(cameras_dict)}")

    # Convert each camera config to CameraConfig object
    cameras: dict[str, CameraConfig] = {}
    for name, config in cameras_dict.items():
        if not isinstance(config, dict):
            raise ValueError(f"Camera '{name}' config should be a dict, got {type(config)}")

        camera_type = config.pop("type", None)
        if camera_type is None:
            raise ValueError(f"Camera '{name}' missing 'type' field")

        # Get the CameraConfig subclass for this type
        camera_config_class = CameraConfig.get_choice_class(camera_type)

        # Create the config object
        cameras[name] = camera_config_class(**config)

    return cameras


def create_relative_observation(
    current_ee_T: np.ndarray,
    gripper_pos: float,
    obs_state_horizon: int = 2,
    history_buffer: list | None = None,
) -> tuple[np.ndarray, list]:
    """
    Create the observation state for the RelativeEEDataset policy.

    For RelativeEEDataset with UMI-style:
    - Current timestep observation is always identity: [0,0,0, 1,0,0,0,1,0, gripper]
    - Historical observations can be stored in a buffer

    Args:
        current_ee_T: Current end-effector pose as 4x4 matrix (not used, always identity)
        gripper_pos: Current gripper position in [0,1]
        obs_state_horizon: Number of historical timesteps
        history_buffer: Buffer of past relative observations

    Returns:
        (observation_state tensor, updated_history_buffer)
    """
    # Current observation is always identity (relative to itself)
    # [dx, dy, dz, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]
    current_obs = np.array([
        0.0, 0.0, 0.0,           # position (identity)
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # rot6d (identity rotation)
        gripper_pos,
    ], dtype=np.float32)

    if history_buffer is None:
        history_buffer = [current_obs.copy() for _ in range(obs_state_horizon)]
    else:
        history_buffer.append(current_obs.copy())
        if len(history_buffer) > obs_state_horizon:
            history_buffer.pop(0)

    # Stack into (obs_state_horizon, 10)
    obs_state = np.stack(history_buffer, axis=0)

    return obs_state, history_buffer


def plot_trajectories(
    predicted_positions: np.ndarray,
    real_positions: np.ndarray,
    output_path: str,
    chunk_base_pose: np.ndarray | None = None,
):
    """
    Plot predicted vs real EE trajectory in 3D.

    Args:
        predicted_positions: (N, 3) array of predicted EE positions
        real_positions: (N, 3) array of actual EE positions from FK
        output_path: Path to save the plot
        chunk_base_pose: Optional starting pose for reference
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot predicted (red dashed)
    ax.plot(
        predicted_positions[:, 0],
        predicted_positions[:, 1],
        predicted_positions[:, 2],
        "r--", linewidth=2, label="Predicted (Policy)"
    )
    ax.scatter(
        predicted_positions[0, 0],
        predicted_positions[0, 1],
        predicted_positions[0, 2],
        c="red", s=100, marker="^", edgecolors="black", label="Predicted Start"
    )
    ax.scatter(
        predicted_positions[-1, 0],
        predicted_positions[-1, 1],
        predicted_positions[-1, 2],
        c="red", s=100, marker="*", edgecolors="black", label="Predicted End"
    )

    # Plot real (blue solid)
    ax.plot(
        real_positions[:, 0],
        real_positions[:, 1],
        real_positions[:, 2],
        "b-", linewidth=2, label="Real (FK from joints)"
    )
    ax.scatter(
        real_positions[0, 0],
        real_positions[0, 1],
        real_positions[0, 2],
        c="green", s=100, marker="o", edgecolors="black", label="Real Start"
    )
    ax.scatter(
        real_positions[-1, 0],
        real_positions[-1, 1],
        real_positions[-1, 2],
        c="blue", s=100, marker="s", edgecolors="black", label="Real End"
    )

    # Add connection lines at intervals to show deviation
    step = max(1, len(predicted_positions) // 10)
    for i in range(0, len(predicted_positions), step):
        ax.plot(
            [predicted_positions[i, 0], real_positions[i, 0]],
            [predicted_positions[i, 1], real_positions[i, 1]],
            [predicted_positions[i, 2], real_positions[i, 2]],
            "k:", alpha=0.3, linewidth=1
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Predicted vs Real EE Trajectory")
    ax.legend(loc="upper right")

    # Equal aspect ratio
    all_pos = np.vstack([predicted_positions, real_positions])
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
    z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()

    # Add padding
    x_pad = (x_max - x_min) * 0.1 if x_max != x_min else 0.01
    y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
    z_pad = (z_max - z_min) * 0.1 if z_max != z_min else 0.01

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")


def compute_statistics(
    predicted_positions: np.ndarray,
    real_positions: np.ndarray,
) -> dict:
    """
    Compute error statistics between predicted and real trajectories.

    Returns:
        Dictionary with error metrics
    """
    min_len = min(len(predicted_positions), len(real_positions))
    pred_trunc = predicted_positions[:min_len]
    real_trunc = real_positions[:min_len]

    # Per-point errors
    errors = np.linalg.norm(pred_trunc - real_trunc, axis=1)

    stats = {
        "mean_error_m": float(np.mean(errors)),
        "max_error_m": float(np.max(errors)),
        "min_error_m": float(np.min(errors)),
        "std_error_m": float(np.std(errors)),
        "final_error_m": float(errors[-1]),
        "num_points": min_len,
    }

    return stats


def main():
    init_logging()
    logger = logging.getLogger("deploy_relative_ee_so101_visualize")

    import argparse
    parser = argparse.ArgumentParser(
        description="Deploy ACT policy with RelativeEEDataset on SO101 and visualize trajectory"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory or training output",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to SO101 URDF file for IK",
    )
    parser.add_argument(
        "--robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO101 robot connection",
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default="so101",
        help="Robot ID for loading/saving calibration files",
    )
    parser.add_argument(
        "--reset_pose",
        type=float,
        nargs=6,
        default=list(RESET_POSE_DEG),
        help="Reset joint positions in degrees (6 values)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Control loop frequency (Hz)",
    )
    parser.add_argument(
        "--num_actions",
        type=int,
        default=50,
        help="Number of actions to execute from the predicted chunk",
    )
    parser.add_argument(
        "--obs_state_horizon",
        type=int,
        default=2,
        help="Observation state horizon (must match training)",
    )
    parser.add_argument(
        "--ee_bounds_min",
        type=float,
        nargs=3,
        default=[-0.5, -0.5, 0.0],
        help="EE position bounds minimum (x, y, z) in meters",
    )
    parser.add_argument(
        "--ee_bounds_max",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.4],
        help="EE position bounds maximum (x, y, z) in meters",
    )
    parser.add_argument(
        "--max_ee_step_m",
        type=float,
        default=0.05,
        help="Maximum EE step size in meters (safety)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./deploy_viz_output",
        help="Output directory for plots and data",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='Camera configuration in YAML format, e.g. \'{ wrist: {type: intelrealsense, serial_number_or_name: "031522070877", width: 640, height: 480, fps: 30} }\'',
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load Policy
    # ========================================================================
    logger.info("Loading trained policy...")

    model_path = args.pretrained_path
    logger.info(f"Loading from: {model_path}")

    policy = ACTPolicy.from_pretrained(model_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)

    # Create processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={},
    )

    logger.info("Policy loaded successfully")
    logger.info(f"  Chunk size: {policy.config.chunk_size}")

    # ========================================================================
    # Initialize Kinematics
    # ========================================================================
    logger.info("Initializing kinematics solver...")

    urdf_path = Path(args.urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found at {urdf_path}")

    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=MOTOR_NAMES,
    )
    logger.info(f"URDF loaded: {urdf_path}")

    # ========================================================================
    # Build RobotProcessorPipeline for converting EE actions to joints
    # ========================================================================
    logger.info("Building processor pipeline...")

    # Pipeline: relative EE action -> absolute EE -> bounds check -> IK -> joints
    ee_to_joints_pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            Relative10DAccumulatedToAbsoluteEE(gripper_scale=100.0),
            EEBoundsAndSafety(
                end_effector_bounds={"min": args.ee_bounds_min, "max": args.ee_bounds_max},
                max_ee_step_m=args.max_ee_step_m,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=MOTOR_NAMES,
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    logger.info("Pipeline built")

    # ========================================================================
    # Connect to Robot
    # ========================================================================
    logger.info("Connecting to SO101 robot...")

    # Parse cameras configuration
    cameras = parse_cameras_config(args.cameras)
    if cameras:
        logger.info(f"Configured cameras: {list(cameras.keys())}")

    robot_config = SO101FollowerConfig(
        id=args.robot_id,
        port=args.robot_port,
        use_degrees=True,
        cameras=cameras,
    )

    robot = SO101Follower(robot_config)
    robot.connect(calibrate=True)
    logger.info("Robot connected")

    # Move to reset pose
    reset_pose = np.array(args.reset_pose, dtype=np.float64)
    logger.info(f"Moving to reset pose: {reset_pose}")
    reset_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, reset_pose)}
    robot.send_action(reset_action)
    time.sleep(2.0)

    # Get current joint positions
    obs_dict = robot.get_observation()
    current_joints = np.array([obs_dict[f"{name}.pos"] for name in MOTOR_NAMES])
    current_gripper = obs_dict["gripper.pos"] / 100.0  # Convert to [0,1]

    logger.info(f"Current joints: {current_joints}")
    logger.info(f"Current gripper: {current_gripper}")

    # Get chunk base pose via FK
    chunk_base_pose = kinematics.forward_kinematics(current_joints)
    logger.info(f"Chunk base EE position: {chunk_base_pose[:3, 3]}")

    # Reset policy and processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # ========================================================================
    # Make ONE prediction from reset pose
    # ========================================================================
    logger.info("Making prediction from reset pose...")

    # Create relative observation (identity at current)
    history_buffer = None
    obs_state, history_buffer = create_relative_observation(
        current_ee_T=chunk_base_pose,
        gripper_pos=current_gripper,
        obs_state_horizon=policy.config.n_obs_steps,
        history_buffer=history_buffer,
    )

    # Prepare batch for policy
    if obs_state.shape[0] == 1:
        state_tensor = torch.from_numpy(obs_state).squeeze(0).unsqueeze(0).to(device)  # (1, 10)
    else:
        state_tensor = torch.from_numpy(obs_state[0]).unsqueeze(0).to(device)  # (1, 10)

    batch = {
        "observation.state": state_tensor,
    }

    # Add camera images if available
    camera_provided = False
    for cam_name in cameras.keys():
        img = obs_dict.get(cam_name)
        if img is not None:
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            batch[f"observation.images.{cam_name}"] = torch.from_numpy(img).unsqueeze(0).to(device)
            camera_provided = True

    # If policy expects images but no camera provided, add dummy image
    # Get expected image shape from policy config
    if not camera_provided and hasattr(policy.config, "input_features"):
        for key in policy.config.input_features:
            if key.startswith("observation.images."):
                # Get expected shape: (C, H, W)
                shape = policy.config.input_features[key]["shape"]
                logger.warning(f"No camera provided for {key}, using dummy zeros with shape {shape}")
                # Create dummy image (zeros) with batch dimension: (1, C, H, W)
                dummy_img = torch.zeros((1, *shape), device=device)
                batch[key] = dummy_img

    # Predict action chunk
    with torch.no_grad():
        processed_batch = preprocessor(batch)
        action_chunk = policy.predict_action_chunk(processed_batch)
        action_chunk = postprocessor(action_chunk)

    # action_chunk shape: (1, chunk_size, 10)
    chunk_size = action_chunk.shape[1]
    logger.info(f"Predicted action chunk with {chunk_size} actions")

    # ========================================================================
    # Compute predicted trajectory from chunk
    # ========================================================================
    logger.info("Computing predicted trajectory...")

    # Start with the base pose (position before any action)
    predicted_positions = [chunk_base_pose[:3, 3].copy()]

    for i in range(chunk_size):
        action_10d = action_chunk[0, i].cpu().numpy()
        rel_T = pose10d_to_mat(action_10d[:9])
        target_ee = chunk_base_pose @ rel_T
        predicted_positions.append(target_ee[:3, 3].copy())

    predicted_positions = np.array(predicted_positions)
    logger.info(f"Predicted trajectory: {len(predicted_positions)} points")

    # ========================================================================
    # Execute chunk and record joint states
    # ========================================================================
    logger.info(f"Executing {args.num_actions} actions...")

    recorded_joints = []
    recorded_targets = []

    # Record initial position (before any action) - this aligns with predicted start
    recorded_joints.append(current_joints.copy())

    num_actions_to_execute = min(args.num_actions, chunk_size)

    for i in range(num_actions_to_execute):
        logger.info(f"  Action {i+1}/{num_actions_to_execute}")

        # Get relative action
        action_10d = action_chunk[0, i].cpu().numpy()

        # Compute target EE pose for reference
        rel_T = pose10d_to_mat(action_10d[:9])
        target_ee_pose = chunk_base_pose @ rel_T
        recorded_targets.append(target_ee_pose[:3, 3].copy())

        # Prepare action dict for pipeline
        action_dict: RobotAction = {"rel_pose": action_10d}

        # Prepare observation dict for pipeline
        robot_obs: RobotObservation = {
            f"{name}.pos": obs_dict[f"{name}.pos"] for name in MOTOR_NAMES
        }

        # Create transition with complementary data (chunk_base_pose)
        transition = robot_action_observation_to_transition((action_dict, robot_obs))
        transition[TransitionKey.COMPLEMENTARY_DATA] = {
            "chunk_base_pose": chunk_base_pose.copy()
        }

        # Run pipeline: relative -> absolute -> bounds -> IK -> joints
        processed_transition = ee_to_joints_pipeline._forward(transition)
        joints_action = transition_to_robot_action(processed_transition)

        # Send action to robot
        robot.send_action(joints_action)

        # Wait for action to complete
        sleep_time = 1.0 / args.fps
        time.sleep(sleep_time)

        # Record actual joint states
        obs_dict = robot.get_observation()
        joints = np.array([obs_dict[f"{name}.pos"] for name in MOTOR_NAMES])
        recorded_joints.append(joints)

    logger.info(f"Recorded {len(recorded_joints)} joint states")

    # ========================================================================
    # Compute real trajectory from recorded joints via FK
    # ========================================================================
    logger.info("Computing real trajectory from FK...")

    real_positions = []
    for joints in recorded_joints:
        ee_T = kinematics.forward_kinematics(joints)
        real_positions.append(ee_T[:3, 3].copy())

    real_positions = np.array(real_positions)
    logger.info(f"Real trajectory: {len(real_positions)} points")

    # ========================================================================
    # Save data
    # ========================================================================
    logger.info("Saving data...")

    np.save(output_dir / "predicted_positions.npy", predicted_positions)
    np.save(output_dir / "real_positions.npy", real_positions)
    np.save(output_dir / "recorded_joints.npy", np.array(recorded_joints))
    np.save(output_dir / "chunk_base_pose.npy", chunk_base_pose)

    logger.info(f"Data saved to: {output_dir}")

    # ========================================================================
    # Compute statistics
    # ========================================================================
    stats = compute_statistics(predicted_positions[:len(real_positions)], real_positions)

    logger.info("=" * 60)
    logger.info("Trajectory Statistics")
    logger.info("=" * 60)
    logger.info(f"  Mean error:    {stats['mean_error_m']*1000:.2f} mm")
    logger.info(f"  Max error:     {stats['max_error_m']*1000:.2f} mm")
    logger.info(f"  Min error:     {stats['min_error_m']*1000:.2f} mm")
    logger.info(f"  Std error:     {stats['std_error_m']*1000:.2f} mm")
    logger.info(f"  Final error:   {stats['final_error_m']*1000:.2f} mm")
    logger.info(f"  Num points:    {stats['num_points']}")
    logger.info("=" * 60)

    # Save statistics to file
    stats_path = output_dir / "statistics.txt"
    with open(stats_path, "w") as f:
        f.write("Trajectory Statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Mean error:    {stats['mean_error_m']*1000:.2f} mm\n")
        f.write(f"  Max error:     {stats['max_error_m']*1000:.2f} mm\n")
        f.write(f"  Min error:     {stats['min_error_m']*1000:.2f} mm\n")
        f.write(f"  Std error:     {stats['std_error_m']*1000:.2f} mm\n")
        f.write(f"  Final error:   {stats['final_error_m']*1000:.2f} mm\n")
        f.write(f"  Num points:    {stats['num_points']}\n")
        f.write("=" * 60 + "\n")
    logger.info(f"Statistics saved to: {stats_path}")

    # ========================================================================
    # Plot trajectories
    # ========================================================================
    logger.info("Plotting trajectories...")

    plot_path = output_dir / "trajectory_comparison.png"
    plot_trajectories(
        predicted_positions[:len(real_positions)],
        real_positions,
        str(plot_path),
        chunk_base_pose=chunk_base_pose,
    )

    # ========================================================================
    # Return to safe pose
    # ========================================================================
    logger.info("Returning to reset pose...")
    robot.send_action(reset_action)
    time.sleep(1.0)

    # Disconnect robot
    logger.info("Disconnecting robot...")
    robot.disconnect()

    logger.info("Done!")


if __name__ == "__main__":
    main()

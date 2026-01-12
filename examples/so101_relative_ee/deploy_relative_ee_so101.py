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
Deploy a trained ACT policy with RelativeEEDataset (10D relative EE actions) on SO101 robot.

This script loads a policy trained with RelativeEEDataset, which outputs 10D relative
end-effector actions (position + 6D rotation + gripper). It uses RobotProcessorPipeline
with custom processors to convert to joint positions for the SO101 robot.

Usage:
    python deploy_relative_ee_so101.py \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --urdf_path /path/to/so101.urdf \
        --robot_port /dev/ttyUSB0

The observation format for this policy is:
    - observation.state: (obs_state_horizon, 10) - relative poses (identity at current timestep)
    - action: (action_horizon, 10) - relative future poses

Action chaining (UMI-style):
    - Each chunk is predicted from the CURRENT actual EE pose
    - Within a chunk, actions are chained cumulatively
    - When chunk exhausted, predict NEW chunk from actual robot pose
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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
    0.0,    # shoulder_pan
    -80.0,  # shoulder_lift
    50.0,   # elbow_flex
    40.0,   # wrist_flex
    0.0,    # wrist_roll
    0.0,    # gripper
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


def find_latest_checkpoint(model_path: str) -> str:
    """Find the latest checkpoint directory in the model path."""
    model_path = Path(model_path)
    checkpoints_dir = model_path / "checkpoints"

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found at {checkpoints_dir}")

    checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {checkpoints_dir}")

    latest = max(checkpoint_dirs, key=lambda x: int(x.name))
    pretrained_path = latest / "pretrained_model"

    if not pretrained_path.exists():
        raise FileNotFoundError(f"No pretrained_model found at {pretrained_path}")

    return str(pretrained_path)


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


def main():
    init_logging()
    logger = logging.getLogger("deploy_relative_ee_so101")

    import argparse
    parser = argparse.ArgumentParser(
        description="Deploy ACT policy with RelativeEEDataset on SO101 robot"
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
        "--num_steps",
        type=int,
        default=0,
        help="Number of control steps to run (0 for infinite)",
    )
    parser.add_argument(
        "--obs_state_horizon",
        type=int,
        default=2,
        help="Observation state horizon (must match training)",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=10,
        help="Number of actions to execute from each chunk prediction",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="Move to reset pose before starting",
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
        "--cameras",
        type=str,
        default=None,
        help='Camera configuration in YAML format, e.g. \'{ wrist: {type: intelrealsense, serial_number_or_name: "031522070877", width: 640, height: 480, fps: 30} }\'',
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load Policy
    # ========================================================================
    logger.info("Loading trained policy...")

    pretrained_path = args.pretrained_path
    if (Path(pretrained_path) / "config.json").exists():
        model_path = pretrained_path
    else:
        model_path = find_latest_checkpoint(pretrained_path)

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
    logger.info(f"  obs_state_horizon: {args.obs_state_horizon}")
    logger.info(f"  n_action_steps: {args.n_action_steps}")

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
        port=args.robot_port,
        use_degrees=True,
        cameras=cameras,
    )

    robot = SO101Follower(robot_config)
    robot.connect(calibrate=True)
    logger.info("Robot connected")

    # Move to reset pose if requested
    if args.warm_start:
        logger.info(f"Moving to reset pose: {args.reset_pose}")
        reset_pose = np.array(args.reset_pose, dtype=np.float64)

        reset_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, reset_pose)}
        robot.send_action(reset_action)
        time.sleep(2.0)

    # Use fixed safe joint positions for error recovery
    initial_safe_joints = np.array([
        -7.91,    # shoulder_pan
        -106.51,  # shoulder_lift
        87.91,    # elbow_flex
        70.74,    # wrist_flex
        -0.53,    # wrist_roll
        1.18,     # gripper
    ], dtype=np.float64)
    logger.info(f"Using fixed safe pose: {initial_safe_joints}")
    print(f"Using fixed safe pose: {initial_safe_joints}")

    current_joints = initial_safe_joints.copy()

    # Get initial EE pose via FK
    current_ee_T = kinematics.forward_kinematics(current_joints)
    logger.info(f"Initial EE position: {current_ee_T[:3, 3]}")

    # Initialize history buffer for observations
    history_buffer = None

    # ========================================================================
    # Main Control Loop
    # ========================================================================
    logger.info("Starting control loop...")
    logger.info(f"Control frequency: {args.fps} Hz")
    logger.info(f"Actions per prediction: {args.n_action_steps}")

    step_count = 0
    action_chunk = None
    action_idx_in_chunk = 0
    accumulated_ee_pose = None

    try:
        while args.num_steps == 0 or step_count < args.num_steps:
            t0 = time.perf_counter()

            # -------------------------------------------------------------------
            # Get current observation from robot
            # -------------------------------------------------------------------
            obs_dict = robot.get_observation()
            current_joints = np.array([obs_dict[f"{name}.pos"] for name in MOTOR_NAMES])
            current_gripper = obs_dict["gripper.pos"] / 100.0  # Convert to [0,1]

            # -------------------------------------------------------------------
            # Predict new action chunk if needed
            # -------------------------------------------------------------------
            if action_chunk is None or action_idx_in_chunk >= args.n_action_steps:
                # Get current EE pose via FK (base for new prediction)
                current_ee_T = kinematics.forward_kinematics(current_joints)

                # Create relative observation (UMI-style: identity at current)
                obs_state, history_buffer = create_relative_observation(
                    current_ee_T=current_ee_T,
                    gripper_pos=current_gripper,
                    obs_state_horizon=args.obs_state_horizon,
                    history_buffer=history_buffer,
                )

                # Prepare batch for policy
                batch = {
                    "observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device),
                }

                # Add camera images if available
                for cam_name in cameras.keys():
                    img = obs_dict.get(cam_name)
                    if img is not None:
                        # Convert from uint8 [0, 255] to float32 [0, 1]
                        img = img.astype(np.float32) / 255.0
                        # Convert from HWC to CHW format expected by policy
                        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                        batch[f"observation.images.{cam_name}"] = torch.from_numpy(img).unsqueeze(0).to(device)

                # Predict action chunk
                with torch.no_grad():
                    processed_batch = preprocessor(batch)
                    action_chunk = policy.select_action(processed_batch)
                    action_chunk = postprocessor(action_chunk)

                # action_chunk shape: (1, 1) or (1, chunk_size, 10)
                if action_chunk.ndim == 2:
                    action_chunk = action_chunk.unsqueeze(1)
                action_chunk = action_chunk[0].cpu().numpy()  # (chunk_size or 1, 10)

                action_idx_in_chunk = 0

                # Reset accumulated pose to current EE pose for new chunk
                accumulated_ee_pose = current_ee_T.copy()

                logger.debug(f"Predicted new chunk with {action_chunk.shape[0]} actions")

            # -------------------------------------------------------------------
            # Get current action from chunk and prepare for execution
            # -------------------------------------------------------------------
            rel_action_10d = action_chunk[action_idx_in_chunk]  # (10,)

            # Update accumulated pose (UMI-style chaining within chunk)
            if accumulated_ee_pose is None:
                accumulated_ee_pose = kinematics.forward_kinematics(current_joints)

            rel_T = pose10d_to_mat(rel_action_10d[:9])
            accumulated_ee_pose = accumulated_ee_pose @ rel_T

            # -------------------------------------------------------------------
            # Convert to joint actions via pipeline
            # -------------------------------------------------------------------
            # Prepare action dict for pipeline
            action_dict: RobotAction = {"rel_pose": rel_action_10d}

            # Prepare observation dict for pipeline
            robot_obs: RobotObservation = {
                f"{name}.pos": obs_dict[f"{name}.pos"] for name in MOTOR_NAMES
            }

            # Create transition with complementary data (accumulated pose)
            transition = robot_action_observation_to_transition((action_dict, robot_obs))
            transition[TransitionKey.COMPLEMENTARY_DATA] = {
                "accumulated_ee_pose": accumulated_ee_pose.copy()
            }

            # Run pipeline: relative -> absolute -> bounds -> IK -> joints
            transition = ee_to_joints_pipeline(transition)
            joints_action = transition_to_robot_action(transition)

            # -------------------------------------------------------------------
            # Send action to robot
            # -------------------------------------------------------------------
            robot.send_action(joints_action)

            action_idx_in_chunk += 1
            step_count += 1

            # -------------------------------------------------------------------
            # Maintain timing
            # -------------------------------------------------------------------
            elapsed = time.perf_counter() - t0
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                precise_sleep(sleep_time)

            if step_count % 100 == 0:
                pos = accumulated_ee_pose[:3, 3]
                logger.info(f"Step {step_count}: EE pos {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        # Return to safe pose
        logger.info("Returning to safe initial pose...")
        safe_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, initial_safe_joints)}
        robot.send_action(safe_action)
        time.sleep(1.0)

    except Exception as e:
        logger.error(f"Error during control loop: {e}")
        import traceback
        traceback.print_exc()
        # Return to safe pose
        logger.info("Returning to safe initial pose...")
        try:
            safe_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, initial_safe_joints)}
            robot.send_action(safe_action)
            time.sleep(1.0)
        except Exception as e2:
            logger.error(f"Error returning to safe pose: {e2}")

    finally:
        # Disconnect robot
        logger.info("Disconnecting robot...")
        robot.disconnect()
        logger.info("Done!")


if __name__ == "__main__":
    main()

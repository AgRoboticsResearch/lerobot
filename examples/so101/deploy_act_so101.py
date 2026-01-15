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
Deploy a trained ACT policy on SO101 robot.

This script loads a standard ACT policy (with joint actions) and runs inference
on the SO101 robot. It's equivalent to `lerobot-record` with a policy but as
a single standalone script without dataset recording overhead.

Usage:
    python deploy_act_so101.py \
        --pretrained_path outputs/train/my_model/checkpoints/last/pretrained_model \
        --robot_port /dev/ttyACM0 \
        --cameras "{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: 031522070877, width: 640, height: 480, fps: 30} }"

The observation format for standard ACT policy:
    - observation.images.front: (3, 480, 640) - front camera image
    - observation.images.wrist: (3, 480, 640) - wrist camera image
    - observation.state: (6,) - joint positions

The action format:
    - action: (6,) - joint position targets
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
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

# Default RESET pose (starting position) in degrees
RESET_POSE_DEG = np.array([
    -5.54,    # shoulder_pan
    -114.59,  # shoulder_lift
    80.44,    # elbow_flex
    15.84,     # wrist_flex
    -5.19,    # wrist_roll
    35.13,    # gripper
])

# Default SAFE pose for error recovery in degrees
SAFE_POSE_DEG = np.array([
    -7.91,    # shoulder_pan
    -106.51,  # shoulder_lift
    87.91,    # elbow_flex
    15.74,    # wrist_flex
    -0.53,    # wrist_roll
    1.18,     # gripper
])

FPS = 30  # Control loop frequency (Hz)


def parse_cameras_config(cameras_str: str | None) -> dict[str, Any]:
    """
    Parse cameras configuration from YAML string.

    Args:
        cameras_str: YAML-formatted string like "{ front: {type: opencv, ...}, wrist: {...} }"

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


def main():
    init_logging()
    logger = logging.getLogger("deploy_act_so101")

    import argparse
    parser = argparse.ArgumentParser(
        description="Deploy standard ACT policy on SO101 robot"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory or training output",
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
        "--fps",
        type=int,
        default=FPS,
        help="Control loop frequency (Hz)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=0,
        help="Number of control steps to run (0 for infinite, press Ctrl+C to stop)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Task description for policies that support task conditioning",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='Camera configuration in YAML format, e.g. \'{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: "031522070877", width: 640, height: 480, fps: 30} }\'',
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision for faster inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run policy on (cuda/cpu, default: auto-detect)",
    )
    parser.add_argument(
        "--reset_pose",
        type=float,
        nargs=6,
        default=list(RESET_POSE_DEG),
        help="Reset joint positions in degrees (6 values): shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper",
    )
    parser.add_argument(
        "--safe_pose",
        type=float,
        nargs=6,
        default=list(SAFE_POSE_DEG),
        help="Safe joint positions in degrees (6 values) for error recovery: shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="Move to reset pose before starting the control loop",
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load Policy
    # ========================================================================
    logger.info("Loading trained policy...")

    model_path = args.pretrained_path
    logger.info(f"Loading from: {model_path}")

    # Load policy directly using from_pretrained (same as relative EE deploy script)
    policy = ACTPolicy.from_pretrained(model_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)
    if args.use_amp:
        policy.config.use_amp = True

    # Create pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    logger.info("Policy loaded successfully")
    logger.info(f"  Policy type: {policy.config.type}")
    logger.info(f"  Chunk size (n_action_steps): {policy.config.n_action_steps}")
    logger.info(f"  Observation horizon (n_obs_steps): {policy.config.n_obs_steps}")

    # ========================================================================
    # Connect to Robot
    # ========================================================================
    logger.info("Connecting to SO101 robot...")

    # Parse cameras configuration
    cameras = parse_cameras_config(args.cameras)
    if cameras:
        logger.info(f"Configured cameras: {list(cameras.keys())}")
    else:
        logger.warning("No cameras configured - policy may expect camera observations")

    robot_config = SO101FollowerConfig(
        id=args.robot_id,
        port=args.robot_port,
        use_degrees=True,
        cameras=cameras,
    )

    robot = SO101Follower(robot_config)
    robot.connect(calibrate=True)
    logger.info("Robot connected")

    # Parse poses from arguments
    reset_pose = np.array(args.reset_pose, dtype=np.float64)
    safe_pose = np.array(args.safe_pose, dtype=np.float64)

    # Move to reset pose if warm start is requested
    if args.warm_start:
        logger.info(f"Moving to reset pose: {reset_pose}")
        reset_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, reset_pose)}
        robot.send_action(reset_action)
        time.sleep(2.0)

    logger.info(f"Using safe pose for error recovery: {safe_pose}")

    # Get current joint positions as initial state
    initial_obs = robot.get_observation()
    initial_joints = np.array([initial_obs[f"{name}.pos"] for name in MOTOR_NAMES])
    logger.info(f"Initial joint positions: {initial_joints}")

    # Reset policy and processors to ensure clean state
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # ========================================================================
    # Main Control Loop
    # ========================================================================
    logger.info("Starting control loop...")
    logger.info(f"Control frequency: {args.fps} Hz")
    logger.info(f"Actions per prediction: {policy.config.n_action_steps}")
    logger.info("Press Ctrl+C to stop")

    step_count = 0

    # Use autocast if AMP is enabled
    autocast_context = torch.autocast(device_type=device.type) if args.use_amp else torch.no_grad()

    try:
        while args.num_steps == 0 or step_count < args.num_steps:
            t0 = time.perf_counter()

            # -------------------------------------------------------------------
            # Get current observation from robot
            # -------------------------------------------------------------------
            obs_dict = robot.get_observation()

            # Build observation batch for policy
            # State: (batch, state_dim) = (1, 6)
            state = np.array([obs_dict[f"{name}.pos"] for name in MOTOR_NAMES], dtype=np.float32)
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, 6)

            batch = {
                "observation.state": state_tensor,
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

            # Add task if specified (for task-conditioned policies)
            if args.task:
                batch["task"] = [args.task]

            # -------------------------------------------------------------------
            # Get action from policy
            # -------------------------------------------------------------------
            # ACT policy's select_action manages an internal action queue and
            # returns a single action for each call
            with torch.inference_mode(), autocast_context:
                processed_batch = preprocessor(batch)
                # select_action returns a single action: (batch, action_dim)
                action_output = policy.select_action(processed_batch)
                # postprocessor converts PolicyAction to dict format
                action_output = postprocessor(action_output)

            # Extract the action from postprocessor output
            # Output is dict with "action" key containing (batch, action_dim) tensor
            if isinstance(action_output, dict):
                action = action_output["action"][0].cpu().numpy()  # (action_dim,)
            else:
                action = action_output[0].cpu().numpy()  # (action_dim,)

            # -------------------------------------------------------------------
            # Send action to robot
            # -------------------------------------------------------------------
            # Convert action to robot format (joint position targets)
            action_dict = {f"{name}.pos": float(action[i]) for i, name in enumerate(MOTOR_NAMES)}
            robot.send_action(action_dict)

            step_count += 1

            # -------------------------------------------------------------------
            # Maintain timing
            # -------------------------------------------------------------------
            elapsed = time.perf_counter() - t0
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                precise_sleep(sleep_time)

            # Log status every 100 steps
            if step_count % 100 == 0:
                logger.info(f"Step {step_count}: joints = {action}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        # Return to safe pose
        logger.info("Returning to safe pose...")
        safe_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, safe_pose)}
        robot.send_action(safe_action)
        time.sleep(1.0)

    except Exception as e:
        logger.error(f"Error during control loop: {e}")
        import traceback
        traceback.print_exc()
        # Return to safe pose
        logger.info("Returning to safe pose...")
        try:
            safe_action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, safe_pose)}
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

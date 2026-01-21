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
Visualize ACT policy predictions in real-time with robot teleoperation.

This script reads sensors from the real robot (no control), runs policy inference,
and visualizes the predicted trajectory in a placo simulation. Move the real robot
by hand and see what the policy would predict!

Usage:
    python visualize_predictions.py \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --urdf_path /path/to/so101.urdf \
        --robot_port /dev/ttyACM0 \
        --cameras "{ camera: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG} }"
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import placo
import torch
import yaml
from placo_utils.visualization import robot_frame_viz, points_viz, robot_viz

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
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

# Motor names for SO101
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

FPS = 10  # Visualization frequency


class SimulatedSO101Robot:
    """Simulated SO101 robot using placo for visualization."""

    def __init__(self, urdf_path: str, motor_names: list[str]):
        self.urdf_path = urdf_path
        self.motor_names = motor_names

        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.joints_task = self.solver.add_joints_task()
        self.viz = robot_viz(self.robot)

        self.current_joints = np.zeros(len(motor_names))
        self._update_robot_from_joints()

    def set_joints(self, joints: np.ndarray):
        """Set joint positions (in degrees)."""
        self.current_joints = joints
        self._update_robot_from_joints()

    def _update_robot_from_joints(self):
        """Update placo robot from current joint state."""
        joints_rad = np.deg2rad(self.current_joints)
        for i, name in enumerate(self.motor_names):
            self.robot.set_joint(name, joints_rad[i])
        self.robot.update_kinematics()
        self.viz.display(self.robot.state.q)


def parse_cameras_config(cameras_str: str | None) -> dict[str, Any]:
    """Parse cameras configuration from YAML string."""
    if not cameras_str or cameras_str.strip() == "":
        return {}

    cameras_dict = yaml.safe_load(cameras_str)
    if not isinstance(cameras_dict, dict):
        raise ValueError(f"Cameras config should be a dict, got {type(cameras_dict)}")

    cameras: dict[str, CameraConfig] = {}
    for name, config in cameras_dict.items():
        if not isinstance(config, dict):
            raise ValueError(f"Camera '{name}' config should be a dict, got {type(config)}")

        camera_type = config.pop("type", None)
        if camera_type is None:
            raise ValueError(f"Camera '{name}' missing 'type' field")

        camera_config_class = CameraConfig.get_choice_class(camera_type)
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
    """
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

    obs_state = np.stack(history_buffer, axis=0)
    return obs_state, history_buffer


def main():
    init_logging()
    logger = logging.getLogger("visualize_predictions")

    import argparse
    parser = argparse.ArgumentParser(
        description="Visualize ACT predictions in real-time with robot teleoperation"
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
        "--fps",
        type=int,
        default=FPS,
        help="Visualization frequency (Hz)",
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
        default=0.3,
        help="Maximum EE step size in meters (safety)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='Camera configuration in YAML format',
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load Policy
    # ========================================================================
    logger.info("Loading trained policy...")

    policy = ACTPolicy.from_pretrained(args.pretrained_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={},
    )

    logger.info("Policy loaded successfully")
    logger.info(f"  Chunk size: {policy.config.chunk_size}")
    logger.info(f"  obs_state_horizon: {args.obs_state_horizon}")

    # ========================================================================
    # Get gripper bounds from dataset metadata
    # ========================================================================
    gripper_lower = 0.0
    gripper_upper = 100.0

    try:
        dataset_repo_id = getattr(policy.config, 'dataset_repo_id', None)
        if dataset_repo_id:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            ds_meta = LeRobotDatasetMetadata(repo_id=dataset_repo_id)
            gripper_lower = ds_meta.info.get('gripper_lower_deg', 0.0)
            gripper_upper = ds_meta.info.get('gripper_upper_deg', 100.0)
            logger.info(f"Loaded gripper bounds: [{gripper_lower}°, {gripper_upper}°]")
    except Exception as e:
        logger.warning(f"Could not load gripper bounds: {e}")
        logger.info(f"Using default gripper bounds: [{gripper_lower}°, {gripper_upper}°]")

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
    # Build Processor Pipeline
    # ========================================================================
    ee_to_joints_pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            Relative10DAccumulatedToAbsoluteEE(
                gripper_lower_deg=gripper_lower,
                gripper_upper_deg=gripper_upper,
            ),
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
    # Parse cameras configuration
    # ========================================================================
    cameras = parse_cameras_config(args.cameras)
    if cameras:
        logger.info(f"Configured cameras: {list(cameras.keys())}")
    else:
        logger.warning("No cameras configured - policy may expect camera observations")

    # ========================================================================
    # Connect to Real Robot (READ ONLY)
    # ========================================================================
    logger.info("Connecting to SO101 robot for sensor reading...")

    robot_config = SO101FollowerConfig(
        id=args.robot_id,
        port=args.robot_port,
        use_degrees=True,
        cameras=cameras,
    )

    robot = SO101Follower(robot_config)
    robot.connect(calibrate=True)
    logger.info("Robot connected (sensors only, no control)")

    # Disable torque to allow free movement (like SO100Leader)
    robot.bus.disable_torque()
    logger.info("Torque disabled - you can now move the robot by hand!")

    # Flush camera buffer
    _ = robot.get_observation()

    # ========================================================================
    # Initialize Digital Twin (Placo Visualization)
    # ========================================================================
    logger.info("Initializing digital twin...")

    placo_urdf_path = str(Path(args.urdf_path).resolve())
    sim_robot = SimulatedSO101Robot(placo_urdf_path, MOTOR_NAMES)

    logger.info("Digital twin initialized")
    logger.info("Open http://127.0.0.1:7000/static/ to see visualization")
    logger.info("\nMove the real robot by hand to see predicted trajectories!")
    logger.info("Press Ctrl+C to stop\n")

    # ========================================================================
    # Main Loop
    # ========================================================================
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    history_buffer = None
    step_count = 0

    try:
        while True:
            t0 = time.perf_counter()

            # -------------------------------------------------------------------
            # Get observation from real robot
            # -------------------------------------------------------------------
            obs_dict = robot.get_observation()
            real_joints = np.array([obs_dict[f"{name}.pos"] for name in MOTOR_NAMES])
            real_gripper = obs_dict["gripper.pos"] / 100.0
            real_ee_T = kinematics.forward_kinematics(real_joints)
            chunk_base_pose = real_ee_T.copy()

            # -------------------------------------------------------------------
            # Update simulation to follow real robot
            # -------------------------------------------------------------------
            sim_robot.set_joints(real_joints)
            robot_frame_viz(sim_robot.robot, "gripper_frame_link")

            # -------------------------------------------------------------------
            # Prepare observation for policy
            # -------------------------------------------------------------------
            obs_state, history_buffer = create_relative_observation(
                current_ee_T=chunk_base_pose,
                gripper_pos=real_gripper,
                obs_state_horizon=policy.config.n_obs_steps,
                history_buffer=history_buffer,
            )

            if obs_state.shape[0] == 1:
                state_tensor = torch.from_numpy(obs_state).squeeze(0).unsqueeze(0).to(device)
            else:
                state_tensor = torch.from_numpy(obs_state[0]).unsqueeze(0).to(device)

            batch = {"observation.state": state_tensor}

            # Add camera images
            for cam_name in cameras.keys():
                img = obs_dict.get(cam_name)
                if img is not None:
                    img = img.astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))
                    batch[f"observation.images.{cam_name}"] = torch.from_numpy(img).unsqueeze(0).to(device)

            # -------------------------------------------------------------------
            # Get predicted trajectory from policy
            # -------------------------------------------------------------------
            with torch.no_grad():
                processed_batch = preprocessor(batch)

            # Get the full chunk of actions
            temp_obs = processed_batch.copy()
            chunk_actions_list = []
            for _ in range(policy.config.chunk_size):
                with torch.no_grad():
                    pred = policy.select_action(temp_obs)
                    pred = postprocessor(pred)
                    if isinstance(pred, dict):
                        action = pred["action"][0].cpu().numpy()
                    else:
                        action = pred[0].cpu().numpy()
                    chunk_actions_list.append(action.copy())

            chunk_actions = np.array(chunk_actions_list)

            # -------------------------------------------------------------------
            # Compute predicted trajectory (FK) and visualize
            # -------------------------------------------------------------------
            pred_positions = []
            pred_joints = real_joints.copy()
            for rel_action in chunk_actions:
                # Convert to joints via pipeline
                act_dict = {"rel_pose": rel_action}
                rob_obs = {f"{name}.pos": pred_joints[i] for i, name in enumerate(MOTOR_NAMES)}
                trans = robot_action_observation_to_transition((act_dict, rob_obs))
                trans[TransitionKey.COMPLEMENTARY_DATA] = {"chunk_base_pose": chunk_base_pose.copy()}
                proc = ee_to_joints_pipeline._forward(trans)
                joints = transition_to_robot_action(proc)
                pred_joints = np.array([joints[f"{name}.pos"] for name in MOTOR_NAMES])

                # FK to get EE position
                ee_T = kinematics.forward_kinematics(pred_joints)
                pred_positions.append(ee_T[:3, 3].copy())

            # Visualize predicted trajectory
            if pred_positions:
                points_viz("predicted_trajectory", np.array(pred_positions), color=0xff0000)

            # Print status every 10 steps
            if step_count % 10 == 0:
                pos = chunk_base_pose[:3, 3]
                logger.info(f"Step {step_count}: EE pos {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")

            step_count += 1

            # Maintain timing
            elapsed = time.perf_counter() - t0
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Re-enable torque before disconnect for safety
        logger.info("Re-enabling torque...")
        robot.bus.enable_torque()
        time.sleep(0.5)

        logger.info("Disconnecting robot...")
        robot.disconnect()
        logger.info("Done!")


if __name__ == "__main__":
    main()

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

This matches the data input, inference, and frame transformation logic from
deploy_relative_ee_so101.py, but without robot control - just visualization.

Usage:
    python visualize_predictions.py \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --urdf_path /path/to/so101.urdf \
        --robot_port /dev/ttyACM0 \
        --cameras "{ camera: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG} }"

=== Visualization Colors ===

- GREEN:  Predicted EE trajectory (from model output)
- YELLOW: IK -> FK trajectory (what joints can actually achieve)
          Shows discrepancy between ideal trajectory and reachable poses

The observation format for this policy is:
    - observation.state: (obs_state_horizon, 10) - relative poses (identity at current timestep)
    - action: (action_horizon, 10) - relative future poses

Action chaining (UMI-style):
    - Each chunk is predicted from the CURRENT actual EE pose
    - Within a chunk, actions are relative to chunk_base_pose
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import placo
import torch
import yaml
from placo_utils.visualization import robot_frame_viz, points_viz, robot_viz

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.temporal_wrapper import TemporalACTWrapper
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so101_follower.relative_ee_processor import pose10d_to_mat
from lerobot.datasets.relative_ee_dataset import mat_to_pose10d
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

FPS = 30  # Control loop frequency (Hz) - must match training fps


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


def make_act_pre_post_processors_with_temporal(
    config: ACTConfig,
    dataset_stats: dict | None = None,
    obs_state_horizon: int = 2,
) -> tuple:
    """Create ACT processors with temporal observation handling (UMI-style).

    This MUST match the processors used during training, otherwise
    normalization will be inconsistent and predictions will be wrong.

    Args:
        config: ACT policy configuration
        dataset_stats: Normalization statistics from dataset
        obs_state_horizon: Number of temporal steps in observations

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    from lerobot.processor import (
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        TemporalNormalizeProcessor,
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    # Create custom preprocessor with temporal normalization
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        TemporalNormalizeProcessor(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
            obs_state_horizon=obs_state_horizon,
        ),
    ]

    # Standard postprocessor
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    preprocessor = PolicyProcessorPipeline[dict, dict](
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[dict, dict](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
    )

    logging.info(f"Created temporal ACT processors with obs_state_horizon={obs_state_horizon}")

    return preprocessor, postprocessor


def build_relative_observation_from_history(
    current_T: np.ndarray,
    current_gripper: float,
    obs_state_horizon: int,
    history_buffer: list | None = None,
) -> tuple[np.ndarray, list]:
    """Match RelativeEEDataset temporal observation logic for real robot.

    - Keep a history of absolute EE poses (4x4) and gripper values.
    - Convert each historical pose to the frame of the *current* pose
      (T_rel = T_curr^{-1} @ T_hist) just like RelativeEEDataset.__getitem__.
    - Pad with the current pose when there is not enough history, so velocity
      info is zeroed at startup (same as dataset padding logic).

    Returns an observation tensor of shape (T, 10) for T>1, or (10,) for T==1
    to stay compatible with the non-temporal ACT preprocessor path.
    """

    if history_buffer is None:
        history_buffer = []

    # Append newest absolute pose + gripper
    history_buffer.append((current_T.copy(), float(current_gripper)))

    # Keep only the most recent obs_state_horizon entries
    history_buffer = history_buffer[-obs_state_horizon:]

    # If buffer is shorter (startup), pad with current pose so behavior matches dataset padding
    if len(history_buffer) < obs_state_horizon:
        pad_needed = obs_state_horizon - len(history_buffer)
        history_buffer = [(current_T.copy(), float(current_gripper))] * pad_needed + history_buffer

    # Compute relative observations w.r.t. current pose (last element)
    T_current = current_T
    T_current_inv = np.linalg.inv(T_current)

    obs_list = []
    for T_hist, grip_hist in history_buffer:
        T_rel = T_current_inv @ T_hist
        pose_9d = mat_to_pose10d(T_rel)
        obs_list.append(np.concatenate([pose_9d, [grip_hist]], dtype=np.float32))

    obs_state = np.stack(obs_list, axis=0)

    # For obs_state_horizon == 1, squeeze to (10,) to match standard ACT expectations
    if obs_state_horizon == 1:
        obs_state = obs_state.squeeze(0)

    return obs_state, history_buffer


def build_temporal_image_observation(
    current_img: np.ndarray,
    obs_state_horizon: int,
    image_history: dict | None = None,
    camera_name: str = "camera",
) -> tuple[np.ndarray, dict]:
    """Build temporal image observation matching RelativeEEDataset format.

    For obs_state_horizon > 1, the model expects (T, C, H, W) format.
    This function maintains a history buffer and pads with current frame
    when there is not enough history (matching dataset behavior).

    Args:
        current_img: Current camera image (H, W, C) uint8 [0, 255]
        obs_state_horizon: Number of temporal frames needed
        image_history: Dict of camera_name -> list of previous images (C, H, W) float32
        camera_name: Name of this camera for the history dict

    Returns:
        Tuple of (temporal_images (T, C, H, W) or (C, H, W), updated_image_history)
    """
    if image_history is None:
        image_history = {}

    # Convert to float32 [0, 1] and transpose to (C, H, W)
    img_float = current_img.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))

    # Initialize history for this camera
    if camera_name not in image_history:
        image_history[camera_name] = []

    # Append current frame
    image_history[camera_name].append(img_chw.copy())

    # Keep only the most recent obs_state_horizon frames
    image_history[camera_name] = image_history[camera_name][-obs_state_horizon:]

    # Pad with current frame if not enough history
    if len(image_history[camera_name]) < obs_state_horizon:
        pad_needed = obs_state_horizon - len(image_history[camera_name])
        image_history[camera_name] = [img_chw.copy()] * pad_needed + image_history[camera_name]

    # Stack into (T, C, H, W) format
    temporal_images = np.stack(image_history[camera_name], axis=0)

    # For obs_state_horizon == 1, squeeze to (C, H, W) to match standard ACT
    if obs_state_horizon == 1:
        temporal_images = temporal_images.squeeze(0)

    return temporal_images, image_history


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
        default=None,
        help="Observation state horizon (auto-detected from policy if not specified)",
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
    parser.add_argument(
        "--cameraview",
        action="store_true",
        help="Show camera feed in cv2 window (image fed into model)",
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

    # Get obs_state_horizon from policy config (or use default)
    policy_obs_state_horizon = getattr(policy.config, 'obs_state_horizon', 1)
    obs_state_horizon = args.obs_state_horizon if args.obs_state_horizon is not None else policy_obs_state_horizon

    # Only wrap with TemporalACTWrapper if obs_state_horizon > 1
    if obs_state_horizon > 1:
        original_model = policy.model
        policy.model = TemporalACTWrapper(original_model, policy.config)
        policy.model = policy.model.to(device)
        logger.info(f"Wrapped ACT model with TemporalACTWrapper (obs_state_horizon={obs_state_horizon})")

        # Load wrapper parameters if temporal_wrapper.pt exists
        wrapper_path = Path(args.pretrained_path).parent / "temporal_wrapper.pt"
        if not wrapper_path.exists():
            wrapper_path = Path(args.pretrained_path) / "temporal_wrapper.pt"
        if not wrapper_path.exists():
            wrapper_path = Path(args.pretrained_path).parent.parent / "temporal_wrapper.pt"
        if wrapper_path.exists():
            wrapper_state_dict = torch.load(wrapper_path, map_location=device)
            model_state = policy.model.state_dict()
            filtered_state = {k: v for k, v in wrapper_state_dict.items() if k in model_state}
            if filtered_state:
                policy.model.load_state_dict(filtered_state, strict=False)
                policy.model = policy.model.to(device)
                logger.info(f"Loaded TemporalACTWrapper parameters from {wrapper_path}")
            else:
                logger.warning(f"No matching parameters in temporal_wrapper.pt, using initialized params")
        else:
            logger.warning(f"No temporal_wrapper.pt found at {wrapper_path}, using initialized params")

        # Load temporal processors
        temp_preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy,
            pretrained_path=args.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
            postprocessor_overrides={},
        )

        # Extract stats from loaded preprocessor
        dataset_stats = None
        for step in temp_preprocessor.steps:
            if hasattr(step, 'stats') and step.stats is not None:
                dataset_stats = step.stats
                break

        if dataset_stats is None:
            raise ValueError("Could not extract normalization stats from pretrained model!")

        # Create temporal processors with the loaded stats
        preprocessor, postprocessor = make_act_pre_post_processors_with_temporal(
            config=policy.config,
            dataset_stats=dataset_stats,
            obs_state_horizon=obs_state_horizon,
        )
    else:
        # Use standard processors for obs_state_horizon=1
        logger.info(f"obs_state_horizon=1, TemporalACTWrapper not needed")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy,
            pretrained_path=args.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
            postprocessor_overrides={},
        )

    logger.info("Policy loaded successfully")
    logger.info(f"  Chunk size: {policy.config.chunk_size}")
    logger.info(f"  obs_state_horizon: {obs_state_horizon}")

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
    # Parse cameras configuration
    # ========================================================================
    cameras = parse_cameras_config(args.cameras)
    if cameras:
        logger.info(f"Configured cameras: {list(cameras.keys())}")
    else:
        logger.warning("No cameras configured - policy may expect camera observations")

    # ========================================================================
    # Check camera view availability
    # ========================================================================
    if args.cameraview:
        try:
            # Test if cv2 can display windows
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("test", test_img)
            cv2.destroyWindow("test")
            logger.info("Camera view enabled")
        except Exception as e:
            logger.warning(f"Cannot show camera view (headless mode?): {e}")
            logger.warning("Disabling --cameraview")
            args.cameraview = False

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

    current_joints = initial_safe_joints.copy()

    # Get initial EE pose via FK
    current_ee_T = kinematics.forward_kinematics(current_joints)
    logger.info(f"Initial EE position: {current_ee_T[:3, 3]}")

    # ========================================================================
    # Initialize Digital Twin (Placo Visualization)
    # ========================================================================
    logger.info("Initializing digital twin...")

    placo_urdf_path = str(Path(args.urdf_path).resolve())
    sim_robot = SimulatedSO101Robot(placo_urdf_path, MOTOR_NAMES)

    logger.info("Digital twin initialized")
    logger.info("Open http://127.0.0.1:7000/static/ to see visualization")
    logger.info("\nMove the real robot by hand to see predicted EE poses!")
    logger.info("Press Ctrl+C to stop\n")

    # ========================================================================
    # Main Control Loop
    # ========================================================================
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    history_buffer = None
    image_history = None  # For temporal camera observations
    step_count = 0

    try:
        while args.num_steps == 0 or step_count < args.num_steps:
            t0 = time.perf_counter()

            # -------------------------------------------------------------------
            # Get current observation from robot
            # -------------------------------------------------------------------
            obs_dict = robot.get_observation()
            current_joints = np.array([obs_dict[f"{name}.pos"] for name in MOTOR_NAMES])
            current_gripper = obs_dict["gripper.pos"] / 100.0  # Convert to [0,1]

            # Debug: print joint values every 30 steps
            # if step_count % 30 == 0:
            #     print(f"Real robot joints: {current_joints}")

            # -------------------------------------------------------------------
            # Get current EE pose as chunk base (predict new chunk every time)
            # -------------------------------------------------------------------
            current_ee_T = kinematics.forward_kinematics(current_joints)
            chunk_base_pose = current_ee_T.copy()

            # -------------------------------------------------------------------
            # Prepare observation for policy
            # -------------------------------------------------------------------
            # Build temporal relative observation matching RelativeEEDataset
            obs_state, history_buffer = build_relative_observation_from_history(
                current_T=current_ee_T,
                current_gripper=current_gripper,
                obs_state_horizon=obs_state_horizon,
                history_buffer=history_buffer,
            )

            # Prepare batch for policy - shape (1, T, 10) for temporal obs
            state_tensor = torch.from_numpy(obs_state).unsqueeze(0).to(device)
            batch = {"observation.state": state_tensor}

            # Add camera images if available
            # Build temporal observations matching RelativeEEDataset format
            for cam_name in cameras.keys():
                img = obs_dict.get(cam_name)
                if img is not None:
                    temporal_img, image_history = build_temporal_image_observation(
                        current_img=img,
                        obs_state_horizon=obs_state_horizon,
                        image_history=image_history,
                        camera_name=cam_name,
                    )
                    batch[f"observation.images.{cam_name}"] = torch.from_numpy(temporal_img).unsqueeze(0).to(device)

                    # Show camera view if enabled - display the actual captured image
                    if args.cameraview:
                        # img is the raw captured image from robot (H, W, C) uint8
                        img_display = img.copy()
                        if img_display.ndim == 3 and img_display.shape[2] == 3:
                            # Already HWC RGB, convert to BGR for cv2
                            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"Camera: {cam_name}", img_display)
                        cv2.waitKey(1)

            # -------------------------------------------------------------------
            # Get full chunk for visualization - predict new chunk every time
            # -------------------------------------------------------------------
            t_inference = time.perf_counter()
            with torch.no_grad():
                processed_batch = preprocessor(batch)
                pred_actions = policy.predict_action_chunk(processed_batch)

                # Call postprocessor differently based on processor type
                if obs_state_horizon > 1:
                    # Temporal processors: wrap in dict
                    pred_actions = postprocessor({"action": pred_actions})
                else:
                    # Standard processors: call directly
                    pred_actions = postprocessor(pred_actions)

                if isinstance(pred_actions, dict) and "action" in pred_actions:
                    pred_actions = pred_actions["action"]
                pred_actions = pred_actions[0].cpu().numpy()  # (chunk_size, 10)

            chunk_actions_list = [pred_actions[i] for i in range(pred_actions.shape[0])]
            inference_time = time.perf_counter() - t_inference
            print(f"Inference time: {inference_time*1000:.1f}ms ({pred_actions.shape[0]} actions)")

            # -------------------------------------------------------------------
            # Compute full trajectory for visualization
            # -------------------------------------------------------------------
            pred_positions = []
            ik_fk_positions = []  # Yellow: IK -> FK results

            # Track IK joints for smooth trajectory (use previous result as initial guess)
            ik_joints_tracking = current_joints.copy()

            for rel_action in chunk_actions_list:
                rel_T = pose10d_to_mat(rel_action[:9])
                target_ee = chunk_base_pose @ rel_T
                pred_positions.append(target_ee[:3, 3].copy())

                # Compute IK -> FK for yellow trajectory
                try:
                    # Solve IK using previous IK result as initial guess (tracking mode)
                    ik_joints_tracking = kinematics.inverse_kinematics(
                        ik_joints_tracking,  # Use previous IK result as initial guess
                        target_ee,
                        position_weight=1.0,
                        orientation_weight=0.01,
                    )
                    # Compute FK from IK joints to get actual EE pose
                    ik_fk_ee_T = kinematics.forward_kinematics(ik_joints_tracking)
                    ik_fk_positions.append(ik_fk_ee_T[:3, 3].copy())
                except Exception as e:
                    # If IK fails, add NaN to break the line and reset tracking
                    ik_fk_positions.append(np.array([np.nan, np.nan, np.nan]))
                    ik_joints_tracking = current_joints.copy()  # Reset to current robot state

            # Visualize predicted trajectory (GREEN)
            if pred_positions:
                points_viz("predicted_trajectory", np.array(pred_positions), color=0x00ff00)

            # Visualize IK -> FK trajectory (YELLOW) - shows what joints can actually achieve
            if ik_fk_positions:
                points_viz("ik_fk_trajectory", np.array(ik_fk_positions), color=0xffff00)

            # -------------------------------------------------------------------
            # Update simulation to follow real robot
            # -------------------------------------------------------------------
            sim_robot.set_joints(current_joints)
            robot_frame_viz(sim_robot.robot, "gripper_frame_link")

            # Print status
            # if step_count % 10 == 0:
            #     print(f"Step {step_count}: EE pos {chunk_base_pose[:3, 3]}")

            step_count += 1

            # -------------------------------------------------------------------
            # Maintain timing
            # -------------------------------------------------------------------
            elapsed = time.perf_counter() - t0
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                precise_sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Error during control loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close camera view windows
        if args.cameraview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass  # Ignore cv2 errors in headless mode

        # Re-enable torque before disconnect for safety
        logger.info("Re-enabling torque...")
        robot.bus.enable_torque()
        time.sleep(0.5)

        logger.info("Disconnecting robot...")
        robot.disconnect()
        logger.info("Done!")


if __name__ == "__main__":
    main()

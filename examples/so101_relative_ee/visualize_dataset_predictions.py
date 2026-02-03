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
Visualize policy predictions on dataset episodes in placo simulation.

This script loads a dataset episode, moves the robot along the GT trajectory,
and at each step uses the corresponding image to get policy predictions.
The predicted trajectories are visualized in real-time.

Usage:
    python visualize_dataset_predictions.py \
        --dataset_root /mnt/ldata/sroi_lerobot/red_strawberry_picking_260119_merged_ee \
        --episode_index 0 \
        --pretrained_path ./outputs/train/red_strawberry_picking_260119_merged_obs1_ds_1/checkpoints/100000/pretrained_model \
        --urdf_path ./urdf/Simulation/SO101/so101_new_calib.urdf

=== Visualization Colors ===

- BLUE:   Ground truth trajectory (from dataset)
- GREEN:  Predicted EE trajectory (from model output)
- YELLOW: IK -> FK trajectory (what joints can actually achieve)

=== What the script does ===

1. Loads the dataset episode
2. Extracts GT trajectory and images
3. For each timestep in the episode:
   - Moves robot to GT pose
   - Gets the corresponding image from dataset
   - Runs policy inference with that image
   - Visualizes predicted trajectory (GREEN) and IK->FK trajectory (YELLOW)
4. Shows full GT trajectory in BLUE for reference
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import placo
import torch
from placo_utils.visualization import robot_frame_viz, points_viz, robot_viz

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.relative_ee_dataset import RelativeEEDataset, pose_to_mat
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.temporal_wrapper import TemporalACTWrapper
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so101_follower.relative_ee_processor import (
    pose10d_to_mat,
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

FPS = 30  # Visualization frequency (Hz)


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


def main():
    init_logging()
    logger = logging.getLogger("visualize_dataset_predictions")

    import argparse
    parser = argparse.ArgumentParser(
        description="Visualize policy predictions on dataset episodes in placo simulation"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Dataset repo ID (optional, for loading metadata)",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="Episode index to visualize",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to SO101 URDF file for IK",
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
        default=1,
        help="Observation state horizon (must match training)",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="Start from this step in the episode",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=0,
        help="Number of steps to visualize (0 = all remaining steps)",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="camera",
        help="Camera observation name in dataset",
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
    else:
        logger.info(f"obs_state_horizon=1, TemporalACTWrapper not needed")

    # Load processors (use standard or temporal based on obs_state_horizon)
    if obs_state_horizon > 1:
        logger.info("Loading temporal processors from pretrained model...")
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
    # Load Dataset
    # ========================================================================
    logger.info("Loading dataset...")

    # repo_id is required, use provided value or infer from path
    if args.repo_id:
        repo_id = args.repo_id
    else:
        # Infer repo_id from path (last directory name)
        repo_id = Path(args.dataset_root).name

    # Use RelativeEEDataset to get temporal images (T frames instead of 1)
    fps = getattr(policy.config, 'fps', 30)
    action_delta_timestamps = [i / fps for i in range(policy.config.chunk_size)]
    delta_timestamps = {"action": action_delta_timestamps}

    dataset = RelativeEEDataset(
        repo_id=repo_id,
        root=args.dataset_root,
        obs_state_horizon=obs_state_horizon,
        delta_timestamps=delta_timestamps,
        compute_stats=False,
    )

    logger.info(f"Dataset loaded: {len(dataset)} frames")
    logger.info(f"Using RelativeEEDataset for temporal images (shape: T={obs_state_horizon}, C, H, W)")

    # Find the requested episode
    episode_start_idx = 0
    episode_end_idx = len(dataset)

    for i in range(len(dataset)):
        frame = dataset[i]
        ep_idx = frame['episode_index'].item()
        if ep_idx == args.episode_index:
            episode_start_idx = i
            # Find end of episode
            episode_end_idx = len(dataset)  # Default to end of dataset
            for j in range(i, len(dataset)):
                frame = dataset[j]
                if frame['episode_index'].item() != args.episode_index:
                    episode_end_idx = j
                    break
            break

    episode_length = episode_end_idx - episode_start_idx
    logger.info(f"Episode {args.episode_index}: frames {episode_start_idx} to {episode_end_idx-1} ({episode_length} frames)")

    # Determine step range
    start_step = args.start_step
    if args.num_steps > 0:
        end_step = min(start_step + args.num_steps, episode_length)
    else:
        end_step = episode_length

    logger.info(f"Visualizing steps {start_step} to {end_step-1}")

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
    # Get camera info from dataset
    # ========================================================================
    sample_frame = dataset[episode_start_idx]
    camera_key = f"observation.images.{args.camera_name}"
    if camera_key in sample_frame:
        image_shape = sample_frame[camera_key].shape
        logger.info(f"Camera '{args.camera_name}': {image_shape}")
    else:
        # Try to find available cameras
        available_cameras = [k.replace("observation.images.", "") for k in sample_frame.keys() if "observation.images." in k]
        logger.warning(f"Camera '{args.camera_name}' not found. Available: {available_cameras}")
        if available_cameras:
            args.camera_name = available_cameras[0]
            camera_key = f"observation.images.{args.camera_name}"
            logger.info(f"Using camera '{args.camera_name}' instead")
        else:
            raise ValueError(f"No camera found in dataset")

    # ========================================================================
    # Extract GT trajectory for visualization
    # ========================================================================
    logger.info("Extracting ground truth trajectory...")

    # RelativeEEDataset.action contains relative poses, not absolute EE poses
    # Need to get absolute EE poses from parent LeRobotDataset
    gt_positions = []
    for i in range(episode_start_idx, episode_end_idx):
        # Get absolute EE pose from parent dataset
        parent_frame = LeRobotDataset.__getitem__(dataset, i)
        parent_action = parent_frame['action'].numpy()
        if parent_action.ndim == 1:
            ee_pos = parent_action[:3]  # [x, y, z]
        else:
            ee_pos = parent_action[0, :3]
        gt_positions.append(ee_pos.copy())

    gt_positions = np.array(gt_positions)
    logger.info(f"GT trajectory: {len(gt_positions)} points")
    logger.info(f"  GT start: {gt_positions[0]}")
    logger.info(f"  GT end: {gt_positions[-1]}")

    # ========================================================================
    # Initialize Digital Twin (Placo Visualization)
    # ========================================================================
    logger.info("Initializing digital twin...")

    placo_urdf_path = str(Path(args.urdf_path).resolve())
    sim_robot = SimulatedSO101Robot(placo_urdf_path, MOTOR_NAMES)

    # Visualize full GT trajectory in blue
    points_viz("gt_trajectory", gt_positions, color=0x0000ff)
    logger.info("GT trajectory visualized in BLUE")

    logger.info("Digital twin initialized")
    logger.info("Open http://127.0.0.1:7000/static/ to see visualization")
    logger.info("\nVisualization colors:")
    logger.info("  BLUE:   Ground truth trajectory")
    logger.info("  GREEN:  Predicted EE trajectory")
    logger.info("  YELLOW: IK -> FK trajectory")
    logger.info("Press Ctrl+C to stop\n")

    # ========================================================================
    # Main Visualization Loop
    # ========================================================================
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    current_joints = np.array([0, -90, 90, 45, 0, 50])  # Initial joint values

    for step_idx in range(start_step, end_step):
        t0 = time.perf_counter()

        frame_idx = episode_start_idx + step_idx
        frame = dataset[frame_idx]

        # RelativeEEDataset provides:
        # - observation.state: (T, 10) - temporal relative observations (identity format)
        # - observation.images.camera: (T, C, H, W) - temporal images
        # - action: (chunk_size, 10) - relative actions (NOT absolute EE poses!)

        # To get absolute EE pose for IK and visualization, we need the parent dataset
        parent_frame = LeRobotDataset.__getitem__(dataset, frame_idx)
        parent_action = parent_frame['action'].numpy()
        if parent_action.ndim == 1:
            gt_ee_pose_6d = parent_action[:6]  # [x, y, z, rx, ry, rz]
            gt_gripper = parent_action[6]
        else:
            gt_ee_pose_6d = parent_action[0, :6]
            gt_gripper = parent_action[0, 6]

        # Get GT EE position
        gt_ee_pos = gt_ee_pose_6d[:3]

        # For relative EE dataset, we need the current robot joints
        # Since we're following GT, use FK to get joints from EE pose
        T_ee = pose_to_mat(gt_ee_pose_6d)

        if step_idx == start_step:
            # Initialize: use IK to get joints from first GT pose
            try:
                current_joints = kinematics.inverse_kinematics(
                    np.array([0, -90, 90, 45, 0, 50]),  # Initial guess
                    T_ee,
                    position_weight=1.0,
                    orientation_weight=0.01,
                )
            except Exception as e:
                logger.warning(f"Failed to get initial IK: {e}")
                current_joints = np.array([0, -90, 90, 45, 0, 50])
        else:
            # Use IK to get to current GT pose from previous joints
            try:
                current_joints = kinematics.inverse_kinematics(
                    current_joints,
                    T_ee,
                    position_weight=1.0,
                    orientation_weight=0.01,
                )
            except Exception as e:
                logger.warning(f"Step {step_idx}: IK failed: {e}")
                # Keep previous joints on failure

        # Set gripper
        current_joints = current_joints.copy()
        current_joints[-1] = gt_gripper * 100  # Convert to degrees

        # Update simulation
        sim_robot.set_joints(current_joints)
        chunk_base_pose = kinematics.forward_kinematics(current_joints)

        # -------------------------------------------------------------------
        # Prepare observation for policy (using RelativeEEDataset format)
        # -------------------------------------------------------------------
        # RelativeEEDataset already provides correctly formatted observations:
        # - observation.state: (T, 10) temporal relative observations
        # - observation.images.camera: (T, C, H, W) temporal images

        # Prepare batch for policy
        batch = {}
        for key, value in frame.items():
            if isinstance(value, torch.Tensor) and key.startswith('observation.'):
                # Add batch dimension
                batch[key] = value.unsqueeze(0).to(device)

        # The observation.state from RelativeEEDataset is already in the correct format:
        # - All timesteps are identity-like relative observations
        # - Shape is (T, 10) where T = obs_state_horizon

        # -------------------------------------------------------------------
        # Get full chunk for visualization
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
        inference_time = time.perf_counter() - t_inference

        chunk_actions_list = [pred_actions[i] for i in range(pred_actions.shape[0])]

        # -------------------------------------------------------------------
        # Compute trajectories for visualization
        # -------------------------------------------------------------------
        pred_positions = []
        ik_fk_positions = []

        ik_joints_tracking = current_joints.copy()

        for rel_action in chunk_actions_list:
            rel_T = pose10d_to_mat(rel_action[:9])
            target_ee = chunk_base_pose @ rel_T
            pred_positions.append(target_ee[:3, 3].copy())

            # Compute IK -> FK for yellow trajectory
            try:
                ik_joints_tracking = kinematics.inverse_kinematics(
                    ik_joints_tracking,
                    target_ee,
                    position_weight=1.0,
                    orientation_weight=0.01,
                )
                ik_fk_ee_T = kinematics.forward_kinematics(ik_joints_tracking)
                ik_fk_positions.append(ik_fk_ee_T[:3, 3].copy())
            except Exception as e:
                ik_fk_positions.append(np.array([np.nan, np.nan, np.nan]))
                ik_joints_tracking = current_joints.copy()

        # Visualize predicted trajectory (GREEN)
        if pred_positions:
            points_viz("predicted_trajectory", np.array(pred_positions), color=0x00ff00)

        # Visualize IK -> FK trajectory (YELLOW)
        if ik_fk_positions:
            points_viz("ik_fk_trajectory", np.array(ik_fk_positions), color=0xffff00)

        # Show robot frame
        robot_frame_viz(sim_robot.robot, "gripper_frame_link")

        # Print status every step
        logger.info(f"Step {step_idx}/{end_step-1}: GT pos {gt_ee_pos}, Inference: {inference_time*1000:.1f}ms")

        # Maintain timing
        elapsed = time.perf_counter() - t0
        sleep_time = max(0, 1.0 / args.fps - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info("\nVisualization complete!")
    logger.info("Press Ctrl+C to exit...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Done!")


if __name__ == "__main__":
    main()

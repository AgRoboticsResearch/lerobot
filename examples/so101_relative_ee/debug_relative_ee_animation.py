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
Debug script for RelativeEE ACT policy animation visualization.

This script loads a trained RelativeEE policy and generates frame-by-frame animations
showing the observation image alongside 3D end-effector trajectory visualization.

Features:
- Loads specific episodes from a dataset
- Runs inference on every frame sequentially
- Generates 2-subplot figures per frame:
  - Left: Current observation image
  - Right: 3D visualization of world frame, EE trajectory history, prediction, ground truth, and IK
- Optionally compiles frames to MP4 video
- Optionally shows IK-constrained trajectory (what robot can actually achieve vs policy intent)

Usage:
    python debug_relative_ee_animation.py \
        --dataset_repo_id red_strawberry_picking_260119_merged_ee \
        --dataset_root /path/to/dataset \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --episode_indices 0 \
        --save_video

    # Enable IK trajectory visualization (magenta line shows what robot can achieve)
    python debug_relative_ee_animation.py \
        --dataset_repo_id red_strawberry_picking_260119_merged_ee \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --episode_indices 0 \
        --enable_ik_trajectory \
        --n_action_steps 10  # Match deployment script
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lerobot.policies.act.modeling_act import ACTPolicy, ACT
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.temporal_wrapper import TemporalACTWrapper
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.relative_ee_dataset import (
    RelativeEEDataset,
    pose10d_to_mat,
    pose_to_mat,
    rot6d_to_mat,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.utils import init_logging
from lerobot.processor import (
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TemporalNormalizeProcessor,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_act_pre_post_processors_with_temporal(
    config: ACTConfig,
    dataset_stats: dict | None = None,
    obs_state_horizon: int = 2,
) -> tuple[
    PolicyProcessorPipeline[dict, dict],
    PolicyProcessorPipeline,
]:
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


def create_observation_batch(
    dataset: RelativeEEDataset,
    idx: int,
    n_obs_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Create observation batch for policy inference.

    For RelativeEE with temporal observations:
    - observation.state shape: (T, 10) where T=obs_state_horizon
    - observation.images.camera shape: (T, C, H, W)

    This function adds the batch dimension, so output shapes are:
    - observation.state: (1, T, 10)
    - observation.images.camera: (1, T, C, H, W)

    Args:
        dataset: RelativeEEDataset instance
        idx: Current sample index
        n_obs_steps: Number of observation steps
        device: Torch device

    Returns:
        Batch dictionary with observation data
    """
    sample = dataset[idx]

    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)

    return batch


def compute_predicted_ee_trajectory(
    relative_actions: np.ndarray,
    current_ee_pose: np.ndarray,
) -> np.ndarray:
    """
    Compute absolute EE positions from relative actions.

    In RelativeEE, each action in the chunk is a transform from the BASE (current) pose
    to a future pose. Actions are NOT sequentially chained.

    Args:
        relative_actions: (T, 10) relative actions (pos 3D + rot 6D + gripper 1D)
        current_ee_pose: (4, 4) current EE transformation matrix

    Returns:
        (T, 3) array of absolute EE positions for each predicted timestep
    """
    T = relative_actions.shape[0]
    ee_positions = []

    for i in range(T):
        # Get relative transform for this timestep
        rel_T = pose10d_to_mat(relative_actions[i, :9])

        # Apply relative transform from current pose: T_abs = T_current @ T_rel
        abs_T = current_ee_pose @ rel_T
        ee_positions.append(abs_T[:3, 3].copy())

    return np.array(ee_positions)


def compute_ik_constrained_trajectory(
    relative_actions: np.ndarray,
    chunk_base_pose: np.ndarray,
    kinematics: RobotKinematics,
    reset_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute IK-constrained EE positions from relative actions.

    This matches the approach in visualize_dataset_predictions.py:
    - First, solve IK to find joints that achieve chunk_base_pose from reset_pose
    - Each action in the chunk is a relative transform from chunk_base_pose
    - IK is solved for each target, chaining within the chunk

    Args:
        relative_actions: (T, 10) relative actions (pos 3D + rot 6D + gripper 1D)
        chunk_base_pose: (4, 4) chunk base EE transformation matrix
        kinematics: RobotKinematics instance
        reset_pose: Starting joint positions (in degrees)

    Returns:
        Tuple of:
        - (T, 3) array of actual EE positions achieved via IK
        - (6,) final joint position (in degrees) after processing all actions
    """
    T = relative_actions.shape[0]
    ee_positions = []

    # Step 1: Find joints that achieve chunk_base_pose from reset_pose
    # This is the "chunk_start_joints" equivalent
    try:
        chunk_start_joints = kinematics.inverse_kinematics(
            reset_pose,
            chunk_base_pose,
            position_weight=1.0,
            orientation_weight=0.01,
        )
    except Exception as e:
        # If IK fails, use reset_pose as fallback
        chunk_start_joints = reset_pose.copy()

    # Step 2: Track IK joint positions within the chunk
    ik_joints_tracking = chunk_start_joints.copy()

    for i in range(T):
        # Get desired EE pose from relative action
        # Each action is relative to the SAME chunk_base_pose (UMI-style)
        rel_T = pose10d_to_mat(relative_actions[i, :9])
        target_ee = chunk_base_pose @ rel_T

        # Solve IK using previous IK result as initial guess
        # This chains the IK within the chunk
        try:
            ik_joints_tracking = kinematics.inverse_kinematics(
                ik_joints_tracking,  # Use previous IK result as initial guess
                target_ee,
                position_weight=1.0,
                orientation_weight=0.01,
            )
            # Compute FK to get actual EE position achieved
            actual_ee_T = kinematics.forward_kinematics(ik_joints_tracking)
            ee_positions.append(actual_ee_T[:3, 3].copy())
        except Exception as e:
            # IK failed - add NaN to indicate discontinuity
            ee_positions.append(np.array([np.nan, np.nan, np.nan]))
            # Reset to chunk start joints
            ik_joints_tracking = chunk_start_joints.copy()

    return np.array(ee_positions), ik_joints_tracking


def plot_frame_with_ee_trajectory(
    obs_image: np.ndarray,
    ee_history: list[np.ndarray],
    pred_ee_positions: np.ndarray,
    current_ee_position: np.ndarray,
    output_path: str | Path,
    episode_idx: int,
    frame_idx: int,
    axis_limits: dict[str, tuple[float, float]] | None = None,
    gt_ee_positions: np.ndarray = None,
    ik_ee_positions: np.ndarray = None,
    use_display_frame_for_limits: bool = False,
):
    """
    Plot a single frame with observation image and EE trajectory.

    Creates a 2-subplot figure:
    - Left: Current observation image
    - Right: 3D EE trajectory with cumulative history, prediction, ground truth, and IK

    Args:
        obs_image: Observation image (C,H,W or H,W,C or T,C,H,W)
        ee_history: List of historical EE positions from episode start
        pred_ee_positions: Predicted EE positions for chunk (chunk_size, 3)
        current_ee_position: Current EE position (3,)
        output_path: Path to save the plot
        episode_idx: Episode index for title
        frame_idx: Frame index for title
        axis_limits: Optional dict with 'x', 'y', 'z' tuples for fixed axis limits
        gt_ee_positions: Ground truth future EE positions (chunk_size, 3)
        ik_ee_positions: IK-constrained EE positions (chunk_size, 3) - shows what robot can achieve
        use_display_frame_for_limits: If True, use current_ee_position for axis limits
                                    (useful when trajectories are in same frame as display)
    """
    fig = plt.figure(figsize=(16, 8))

    # ========================================================================
    # Left subplot: Observation image
    # ========================================================================
    ax1 = fig.add_subplot(1, 2, 1)

    # Handle different image shapes
    if obs_image.ndim == 4:  # (T, C, H, W)
        img = obs_image[-1]  # Use last timestep (current)
    else:
        img = obs_image

    # Convert from (C, H, W) to (H, W, C) if needed
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

    ax1.imshow(img)
    ax1.set_title(f"Episode {episode_idx}, Frame {frame_idx}", fontsize=14)
    ax1.axis("off")

    # ========================================================================
    # Right subplot: 3D EE trajectory
    # ========================================================================
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Convert history to array
    if ee_history:
        ee_hist_array = np.array(ee_history)
    else:
        ee_hist_array = np.array([[0, 0, 0]])

    # Plot world frame origin
    ax2.scatter([0], [0], [0], c="black", marker="s", s=100, label="World Origin", zorder=10)

    # Plot historical EE trajectory (from frame 0 to current) as solid blue line
    ax2.plot(
        ee_hist_array[:, 0],
        ee_hist_array[:, 1],
        ee_hist_array[:, 2],
        "b-", linewidth=2, label="History", alpha=0.8
    )

    # Plot current EE position as blue circle marker
    ax2.scatter(
        [current_ee_position[0]],
        [current_ee_position[1]],
        [current_ee_position[2]],
        c="blue", marker="o", s=150, edgecolors="black",
        label="Current EE", zorder=10
    )

    # Plot predicted trajectory as red dashed line
    if pred_ee_positions is not None and len(pred_ee_positions) > 0:
        # Start prediction from current EE position
        pred_full = np.vstack([current_ee_position.reshape(1, 3), pred_ee_positions])
        ax2.plot(
            pred_full[:, 0],
            pred_full[:, 1],
            pred_full[:, 2],
            "r--", linewidth=2, label="Prediction", alpha=0.8
        )

        # Plot predicted EE positions as red x markers
        ax2.scatter(
            pred_ee_positions[:, 0],
            pred_ee_positions[:, 1],
            pred_ee_positions[:, 2],
            c="red", marker="x", s=50, label="Pred Steps", alpha=0.8
        )

    # Plot ground truth future trajectory as green dotted line
    if gt_ee_positions is not None and len(gt_ee_positions) > 0:
        # Start GT from current EE position
        gt_full = np.vstack([current_ee_position.reshape(1, 3), gt_ee_positions])
        ax2.plot(
            gt_full[:, 0],
            gt_full[:, 1],
            gt_full[:, 2],
            "g:", linewidth=2, label="Ground Truth", alpha=0.8
        )

        # Plot GT EE positions as green circles
        ax2.scatter(
            gt_ee_positions[:, 0],
            gt_ee_positions[:, 1],
            gt_ee_positions[:, 2],
            c="green", marker="o", s=30, label="GT Steps", alpha=0.6
        )

    # Plot IK-constrained trajectory as magenta dash-dot line
    if ik_ee_positions is not None and len(ik_ee_positions) > 0:
        # Start IK from current EE position
        ik_full = np.vstack([current_ee_position.reshape(1, 3), ik_ee_positions])
        ax2.plot(
            ik_full[:, 0],
            ik_full[:, 1],
            ik_full[:, 2],
            color="magenta", linestyle="-.", linewidth=2, label="IK (Robot)", alpha=0.9
        )

        # Plot IK EE positions as magenta triangles (larger markers)
        ax2.scatter(
            ik_ee_positions[:, 0],
            ik_ee_positions[:, 1],
            ik_ee_positions[:, 2],
            c="magenta", marker="^", s=100, label="IK Steps", alpha=0.9, edgecolors="black", linewidth=1.5
        )

    ax2.set_xlabel("X (m)", fontsize=10)
    ax2.set_ylabel("Y (m)", fontsize=10)
    ax2.set_zlabel("Z (m)", fontsize=10)
    ax2.set_title(f"End-Effector Trajectory (History: {len(ee_history)} frames)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Set axis limits - use provided limits or compute from data
    if axis_limits is not None:
        ax2.set_xlim(axis_limits['x'])
        ax2.set_ylim(axis_limits['y'])
        ax2.set_zlim(axis_limits['z'])
    else:
        # Compute limits from all data
        # When ee_history is empty, start from current_ee_position (avoid origin point from wrong frame)
        if ee_history:
            all_pos = np.vstack([ee_hist_array, current_ee_position.reshape(1, 3)])
        else:
            all_pos = current_ee_position.reshape(1, 3)

        if pred_ee_positions is not None:
            all_pos = np.vstack([all_pos, pred_ee_positions])
        if gt_ee_positions is not None:
            all_pos = np.vstack([all_pos, gt_ee_positions])
        if ik_ee_positions is not None:
            all_pos = np.vstack([all_pos, ik_ee_positions])

        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
        z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()

        x_pad = max((x_max - x_min) * 0.15, 0.02)
        y_pad = max((y_max - y_min) * 0.15, 0.02)
        z_pad = max((z_max - z_min) * 0.15, 0.02)

        ax2.set_xlim(x_min - x_pad, x_max + x_pad)
        ax2.set_ylim(y_min - y_pad, y_max + y_pad)
        ax2.set_zlim(z_min - z_pad, z_max + z_pad)

        # Set equal aspect ratio for XYZ axes
        # Make all axes have the same scale by extending the shorter ranges
        x_range = (x_max + x_pad) - (x_min - x_pad)
        y_range = (y_max + y_pad) - (y_min - y_pad)
        z_range = (z_max + z_pad) - (z_min - z_pad)
        max_range = max(x_range, y_range, z_range)

        # Extend shorter ranges to match the maximum, centered on original range
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        ax2.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax2.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax2.set_zlim(z_center - max_range/2, z_center + max_range/2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_episode_axis_limits(
    dataset: RelativeEEDataset,
    episode_idx: int,
) -> dict[str, tuple[float, float]]:
    """
    Compute axis limits for an episode by processing all ground truth actions.

    Args:
        dataset: RelativeEEDataset instance
        episode_idx: Episode index

    Returns:
        Dict with 'x', 'y', 'z' tuples for axis limits
    """
    # Get episode info from dataset
    ep_info = dataset.meta.episodes[episode_idx]
    ep_length = ep_info["length"]

    # Find start index in dataset (cumulative sum of previous episode lengths)
    start_idx = 0
    for i in range(episode_idx):
        start_idx += dataset.meta.episodes[i]["length"]

    all_ee_positions = []

    # Process each frame to get EE positions
    for i in range(ep_length):
        idx = start_idx + i

        # Get ABSOLUTE EE position from parent LeRobotDataset (not the relative one)
        abs_sample = LeRobotDataset.__getitem__(dataset, idx)
        abs_state = abs_sample['observation.state'].cpu().numpy()  # (7,) - absolute pose

        # Convert absolute state to 4x4 matrix
        abs_pose_6d = abs_state[:6]  # position (3) + rotation axis-angle (3)
        current_ee_pose = pose_to_mat(abs_pose_6d)
        all_ee_positions.append(current_ee_pose[:3, 3].copy())

    all_ee = np.array(all_ee_positions)

    x_min, x_max = all_ee[:, 0].min(), all_ee[:, 0].max()
    y_min, y_max = all_ee[:, 1].min(), all_ee[:, 1].max()
    z_min, z_max = all_ee[:, 2].min(), all_ee[:, 2].max()

    # Add padding
    x_pad = max((x_max - x_min) * 0.15, 0.02)
    y_pad = max((y_max - y_min) * 0.15, 0.02)
    z_pad = max((z_max - z_min) * 0.15, 0.02)

    return {
        'x': (x_min - x_pad, x_max + x_pad),
        'y': (y_min - y_pad, y_max + y_pad),
        'z': (z_min - z_pad, z_max + z_pad),
    }


def write_actions_file(
    output_path: Path,
    frame_idx: int,
    episode_idx: int,
    current_ee_position: np.ndarray,
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    chunk_size: int,
    ik_ee_positions: np.ndarray | None = None,
    n_action_steps: int | None = None,
):
    """
    Write action predictions to a structured text file.

    Args:
        output_path: Path to save the actions file
        frame_idx: Current frame index
        episode_idx: Current episode index
        current_ee_position: Current EE position (3,)
        pred_actions: Predicted actions (chunk_size, 10)
        gt_actions: Ground truth actions (chunk_size, 10)
        chunk_size: Size of the action chunk
        ik_ee_positions: Optional IK-constrained EE positions (n_action_steps, 3)
        n_action_steps: Number of action steps executed
    """
    with open(output_path, "w") as f:
        f.write(f"# Frame: {frame_idx}, Episode: {episode_idx}\n")
        f.write(f"# Format: RelativeEE action = [dx, dy, dz, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]\n\n")

        # Header section
        f.write("[header]\n")
        f.write(f"frame_idx = {frame_idx}\n")
        f.write(f"episode_idx = {episode_idx}\n")
        f.write(f"chunk_size = {chunk_size}\n")
        if n_action_steps is not None:
            f.write(f"n_action_steps = {n_action_steps}\n")
        f.write(f"current_ee_position = [{current_ee_position[0]:.6f}, {current_ee_position[1]:.6f}, {current_ee_position[2]:.6f}]\n")
        f.write("\n")

        # Predicted actions section
        f.write("[predicted_actions]\n")
        f.write("# Format: step_idx: dx, dy, dz, r00, r01, r02, r10, r11, r12, gripper\n")
        for i, action in enumerate(pred_actions):
            vals = ", ".join([f"{v:.6f}" for v in action])
            f.write(f"{i}: {vals}\n")
        f.write("\n")

        # Ground truth actions section
        f.write("[ground_truth_actions]\n")
        f.write("# Format: step_idx: dx, dy, dz, r00, r01, r02, r10, r11, r12, gripper\n")
        for i, action in enumerate(gt_actions):
            vals = ", ".join([f"{v:.6f}" for v in action])
            f.write(f"{i}: {vals}\n")
        f.write("\n")

        # IK-constrained EE positions section (optional)
        if ik_ee_positions is not None:
            f.write("[ik_ee_positions]\n")
            f.write("# Format: step_idx: x, y, z (meters)\n")
            for i, pos in enumerate(ik_ee_positions):
                f.write(f"{i}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]\n")


def read_actions_file(input_path: Path) -> dict:
    """
    Read action predictions from a structured text file.

    Args:
        input_path: Path to the actions file

    Returns:
        Dict with keys: header, predicted_actions, ground_truth_actions, ik_ee_positions
    """
    result = {
        "header": {},
        "predicted_actions": None,
        "ground_truth_actions": None,
        "ik_ee_positions": None,
    }

    with open(input_path, "r") as f:
        lines = f.readlines()

    current_section = None
    action_data = []

    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue

        # Section headers
        if line.startswith("[") and line.endswith("]"):
            # Save previous section data
            if current_section == "predicted_actions":
                result["predicted_actions"] = np.array(action_data)
            elif current_section == "ground_truth_actions":
                result["ground_truth_actions"] = np.array(action_data)
            elif current_section == "ik_ee_positions":
                result["ik_ee_positions"] = np.array(action_data)

            # Start new section
            current_section = line[1:-1]
            action_data = []
            continue

        # Parse header lines (key = value)
        if current_section == "header":
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Parse different value types
                if key in ["frame_idx", "episode_idx", "chunk_size", "n_action_steps"]:
                    result["header"][key] = int(value)
                elif key == "current_ee_position":
                    # Parse [x, y, z]
                    nums = value.strip("[]").split(",")
                    result["header"][key] = np.array([float(n) for n in nums])
                else:
                    result["header"][key] = value

        # Parse action lines (step_idx: val1, val2, ...)
        elif current_section in ["predicted_actions", "ground_truth_actions"]:
            if ":" in line:
                parts = line.split(":", 1)
                values = [float(v.strip()) for v in parts[1].split(",")]
                action_data.append(values)

        # Parse IK position lines (step_idx: [x, y, z])
        elif current_section == "ik_ee_positions":
            if ":" in line:
                parts = line.split(":", 1)
                nums = parts[1].strip("[]").split(",")
                values = [float(n.strip()) for n in nums]
                action_data.append(values)

    # Save last section
    if current_section == "predicted_actions":
        result["predicted_actions"] = np.array(action_data)
    elif current_section == "ground_truth_actions":
        result["ground_truth_actions"] = np.array(action_data)
    elif current_section == "ik_ee_positions":
        result["ik_ee_positions"] = np.array(action_data)

    return result


def create_video_from_frames(
    frames_dir: Path,
    output_path: Path,
    fps: int = 30,
):
    """
    Create MP4 video from frames.

    Args:
        frames_dir: Directory containing frame_*.jpg files
        output_path: Output path for MP4 video
        fps: Frames per second for video
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError(
            "imageio is required for video generation. "
            "Install with: pip install imageio imageio-ffmpeg"
        )

    # Get all frame files sorted
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    # Read frames
    frames = []
    for frame_file in frame_files:
        frames.append(imageio.imread(frame_file))

    # Ensure all frames have the same size by using the first frame's size
    # and resizing any frames that don't match
    from PIL import Image
    target_size = frames[0].shape[:2][::-1]  # (width, height)
    uniform_frames = []
    for frame in frames:
        if frame.shape[:2][::-1] != target_size:
            img = Image.fromarray(frame)
            img = img.resize(target_size, Image.LANCZOS)
            uniform_frames.append(np.array(img))
        else:
            uniform_frames.append(frame)

    # Write video
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
    for frame in uniform_frames:
        writer.append_data(frame)
    writer.close()

    print(f"Video saved to: {output_path}")


def main():
    init_logging()
    logger = logging.getLogger("debug_relative_ee_animation")

    import argparse
    parser = argparse.ArgumentParser(
        description="Debug script for RelativeEE ACT policy animation visualization"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., red_strawberry_picking_260119_merged_ee)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Local path to dataset (if not provided, uses HF hub)",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory or training output",
    )
    parser.add_argument(
        "--episode_indices",
        type=int,
        nargs="+",
        default=[0],
        help="Episode indices to visualize (default: [0])",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames per episode to process (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/debug/animation_relative_ee",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Compile frames to MP4 video",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="FPS for output video (default: 30)",
    )
    parser.add_argument(
        "--fixed_axis_limits",
        action="store_true",
        help="Use fixed axis limits computed from full episode (recommended for animation)",
    )
    parser.add_argument(
        "--obs_state_horizon",
        type=int,
        default=None,
        help="Observation state horizon (auto-detected from policy if not specified)",
    )
    parser.add_argument(
        "--enable_ik_trajectory",
        action="store_true",
        help="Enable IK-constrained trajectory visualization (shows what robot can actually achieve)",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="urdf/Simulation/SO101/so101_new_calib.urdf",
        help="Path to robot URDF file for IK (default: urdf/Simulation/SO101/so101_new_calib.urdf)",
    )
    parser.add_argument(
        "--target_frame",
        type=str,
        default="gripper_frame_link",
        help="Name of end-effector frame in URDF (default: gripper_frame_link)",
    )
    parser.add_argument(
        "--reset_pose",
        type=float,
        nargs=6,
        default=[-8.00, -62.73, 65.05, 0.86, -2.55, 88.91],
        metavar=("PAN", "LIFT", "ELBOW", "FLEX", "ROLL", "GRIPPER"),
        help="Reset pose in degrees for IK (default: -8.00 -62.73 65.05 0.86 -2.55 88.91)",
    )
    parser.add_argument(
        "--pos_tolerance",
        type=float,
        default=0.02,
        help="Position tolerance for IK verification in meters (default: 0.02 = 20mm)",
    )
    parser.add_argument(
        "--rot_tolerance",
        type=float,
        default=0.3,
        help="Rotation tolerance for IK verification in radians (default: 0.3 = ~17 deg)",
    )
    parser.add_argument(
        "--joint_names",
        type=str,
        nargs=6,
        default=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Joint names for IK (default: shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper)",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=10,
        help="Number of actions to execute from each chunk prediction (default: 10, must match deployment)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Extract job name from pretrained_path
    job_name = "unknown_job"
    pretrained_path_obj = Path(args.pretrained_path)
    if "outputs" in pretrained_path_obj.parts:
        outputs_idx = pretrained_path_obj.parts.index("outputs")
        if outputs_idx + 2 < len(pretrained_path_obj.parts):
            job_name = pretrained_path_obj.parts[outputs_idx + 2]
            logger.info(f"Job name: {job_name}")

    # Extract model step
    model_step = None
    if "checkpoints" in pretrained_path_obj.parts:
        checkpoints_idx = pretrained_path_obj.parts.index("checkpoints")
        if checkpoints_idx + 1 < len(pretrained_path_obj.parts):
            model_step = pretrained_path_obj.parts[checkpoints_idx + 1]
            logger.info(f"Model step: {model_step}")

    # Create output directory
    output_dir = Path(args.output_dir) / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ========================================================================
    # Load Policy
    # ========================================================================
    logger.info("Loading trained policy...")
    logger.info(f"  From: {args.pretrained_path}")

    policy = ACTPolicy.from_pretrained(args.pretrained_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)

    # Wrap with TemporalACTWrapper
    policy_obs_state_horizon = getattr(policy.config, 'obs_state_horizon', 1)
    obs_state_horizon = args.obs_state_horizon if args.obs_state_horizon is not None else policy_obs_state_horizon

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

    logger.info(f"Using obs_state_horizon={obs_state_horizon}")

    # Load temporal processors
    logger.info("Loading processors from pretrained model...")
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

    chunk_size = policy.config.chunk_size

    logger.info("Policy loaded successfully")
    logger.info(f"  Chunk size: {chunk_size}")
    logger.info(f"  n_obs_steps: {policy.config.n_obs_steps}")
    logger.info(f"  n_action_steps: {policy.config.n_action_steps}")

    # ========================================================================
    # Load Dataset
    # ========================================================================
    logger.info("Loading dataset...")
    logger.info(f"  Repo ID: {args.dataset_repo_id}")
    logger.info(f"  Local root: {args.dataset_root}")

    fps = getattr(policy.config, 'fps', 30)
    action_delta_timestamps = [i / fps for i in range(chunk_size)]
    delta_timestamps = {"action": action_delta_timestamps}

    # Load full dataset without episodes filter (schema mismatch issue)
    dataset = RelativeEEDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        obs_state_horizon=obs_state_horizon,
        delta_timestamps=delta_timestamps,
        compute_stats=False,
    )

    logger.info(f"Dataset loaded successfully!")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Total episodes: {len(dataset.meta.episodes)}")
    logger.info(f"  Processing episodes: {args.episode_indices}")

    # ========================================================================
    # Initialize IK if enabled
    # ========================================================================
    kinematics = None
    reset_pose = np.array(args.reset_pose, dtype=np.float64)

    if args.enable_ik_trajectory:
        logger.info("Initializing IK solver...")
        logger.info(f"  URDF: {args.urdf_path}")
        logger.info(f"  Target frame: {args.target_frame}")
        logger.info(f"  Reset pose: {reset_pose}")
        logger.info(f"  Position tolerance: {args.pos_tolerance * 1000:.1f} mm")
        logger.info(f"  Rotation tolerance: {np.rad2deg(args.rot_tolerance):.1f} deg")

        kinematics = RobotKinematics(
            urdf_path=args.urdf_path,
            target_frame_name=args.target_frame,
            joint_names=args.joint_names,
        )
        logger.info("IK solver initialized successfully!")
    else:
        logger.info("IK trajectory visualization disabled (use --enable_ik_trajectory to enable)")

    # ========================================================================
    # Process each episode
    # ========================================================================
    for ep_idx in args.episode_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Episode {ep_idx}")
        logger.info(f"{'='*60}")

        # Get episode bounds
        ep_info = dataset.meta.episodes[ep_idx]
        ep_length = ep_info["length"]

        # Find start index in dataset
        start_idx = 0
        for i in range(ep_idx):
            start_idx += dataset.meta.episodes[i]["length"]
        end_idx = start_idx + ep_length

        logger.info(f"  Episode length: {ep_length} frames")
        logger.info(f"  Dataset range: [{start_idx}, {end_idx})")

        # Apply max_frames limit
        if args.max_frames is not None:
            ep_end = min(end_idx, start_idx + args.max_frames)
            logger.info(f"  Limited to {ep_end - start_idx} frames")
        else:
            ep_end = end_idx

        # Compute axis limits for this episode if requested
        axis_limits = None
        if args.fixed_axis_limits:
            logger.info("  Computing axis limits from full episode...")
            axis_limits = compute_episode_axis_limits(dataset, ep_idx)
            logger.info(f"  Axis limits: X={axis_limits['x']}, Y={axis_limits['y']}, Z={axis_limits['z']}")

        # Create episode output directory
        ep_output_dir = output_dir / f"episode_{ep_idx}"
        ep_output_dir.mkdir(parents=True, exist_ok=True)

        # Run inference on each frame
        ee_history = []  # Cumulative EE positions

        # Compute transform from world frame to reset_pose_ee frame
        # This is done ONCE per episode to maintain consistency
        if kinematics is not None:
            reset_pose_ee = kinematics.forward_kinematics(reset_pose)

            # Get the first frame's EE pose to compute the fixed offset
            first_abs_sample = LeRobotDataset.__getitem__(dataset, start_idx)
            first_abs_state = first_abs_sample['observation.state'].cpu().numpy()
            first_pose_6d = first_abs_state[:6]
            first_ee_pose = pose_to_mat(first_pose_6d)

            # Fixed offset from world frame to reset_pose_ee frame
            # T_offset = reset_pose_ee @ inv(first_ee_pose)
            T_offset_fixed = reset_pose_ee @ np.linalg.inv(first_ee_pose)
        else:
            reset_pose_ee = None
            T_offset_fixed = None

        for frame_idx in range(start_idx, ep_end):
            sample = dataset[frame_idx]

            # Get observation image
            obs_img = None
            for key in sample.keys():
                if key.startswith("observation.images"):
                    obs_img = sample[key].cpu().numpy()
                    break

            if obs_img is None:
                logger.warning(f"No observation image found for frame {frame_idx}")
                continue

            # Get ABSOLUTE EE position from parent LeRobotDataset (not the relative one)
            # The RelativeEEDataset.__getitem__ returns relative poses, but the parent stores absolute
            abs_sample = LeRobotDataset.__getitem__(dataset, frame_idx)
            abs_state = abs_sample['observation.state'].cpu().numpy()  # (7,) - absolute pose

            # Convert absolute state to 4x4 matrix
            # The state is [x, y, z, rx, ry, rz, gripper] in axis-angle format
            abs_pose_6d = abs_state[:6]  # position (3) + rotation axis-angle (3)
            current_ee_pose = pose_to_mat(abs_pose_6d)
            current_ee_position = current_ee_pose[:3, 3].copy()

            # Transform to reset_pose_ee frame if IK is enabled
            if T_offset_fixed is not None:
                current_ee_position_aligned = (T_offset_fixed @ np.array([current_ee_position[0], current_ee_position[1], current_ee_position[2], 1.0]))[:3]
            else:
                current_ee_position_aligned = current_ee_position

            # Add to history (in reset_pose_ee frame if applicable)
            ee_history.append(current_ee_position_aligned.copy())

            # Get ground truth action (for reference)
            gt_actions = sample['action'].cpu().numpy()  # (chunk_size, 10)

            # Create observation batch and run inference
            batch = create_observation_batch(dataset, frame_idx, policy.config.n_obs_steps, device)

            with torch.no_grad():
                processed_batch = preprocessor(batch)
                pred_actions = policy.predict_action_chunk(processed_batch)
                pred_actions = postprocessor({"action": pred_actions})

            if isinstance(pred_actions, dict) and "action" in pred_actions:
                pred_actions = pred_actions["action"]
            pred_actions = pred_actions[0].cpu().numpy()  # (chunk_size, 10)

            # Compute predicted EE trajectory
            # IMPORTANT: Use reset_pose_ee as the base, NOT current_ee_pose!
            # This ensures predicted trajectory aligns with IK trajectory (both from reset_pose_ee)
            pred_ee_positions = compute_predicted_ee_trajectory(pred_actions, reset_pose_ee)

            # Compute ground truth EE trajectory (also from reset_pose_ee for comparison)
            gt_ee_positions = compute_predicted_ee_trajectory(gt_actions, reset_pose_ee)

            # Compute IK-constrained trajectory if enabled
            # IMPORTANT: Only process n_action_steps from chunk to match deployment behavior!
            # In deployment, we execute n_action_steps actions, then predict a new chunk.
            ik_ee_positions = None
            if kinematics is not None:
                # All trajectories (pred, GT, IK) are now in reset_pose_ee frame - no transformation needed!
                ik_ee_positions, _ = compute_ik_constrained_trajectory(
                    relative_actions=pred_actions[:args.n_action_steps],  # Only first N actions
                    chunk_base_pose=reset_pose_ee,  # Use reset_pose_ee as chunk base
                    kinematics=kinematics,
                    reset_pose=reset_pose,
                )
            else:
                # No IK - trajectories are in world frame, use original current_ee_position
                pass

            # Save action predictions to text file
            actions_output = ep_output_dir / f"actions_{frame_idx:04d}.txt"
            write_actions_file(
                output_path=actions_output,
                frame_idx=frame_idx,
                episode_idx=ep_idx,
                current_ee_position=current_ee_position,
                pred_actions=pred_actions,
                gt_actions=gt_actions,
                chunk_size=pred_actions.shape[0],
                ik_ee_positions=ik_ee_positions,
                n_action_steps=args.n_action_steps if kinematics is not None else None,
            )

            # Generate and save plot
            frame_output = ep_output_dir / f"frame_{frame_idx:04d}.jpg"
            # For visualization, show reset_pose_ee position as "current EE" since all trajectories
            # are relative to reset_pose_ee (simulating robot starting from reset_pose)
            # When IK is enabled, don't show history since it's in a different frame
            current_ee_for_viz = reset_pose_ee[:3, 3].copy() if kinematics is not None else current_ee_position
            ee_history_for_viz = [] if kinematics is not None else ee_history
            plot_frame_with_ee_trajectory(
                obs_image=obs_img,
                ee_history=ee_history_for_viz,
                pred_ee_positions=pred_ee_positions,
                current_ee_position=current_ee_for_viz,
                output_path=frame_output,
                episode_idx=ep_idx,
                frame_idx=frame_idx,
                axis_limits=axis_limits,
                gt_ee_positions=gt_ee_positions,
                ik_ee_positions=ik_ee_positions,
            )

            if (frame_idx - start_idx) % 10 == 0:
                logger.info(f"  Processed {frame_idx - start_idx}/{ep_end - start_idx} frames")

        logger.info(f"  Episode {ep_idx} complete: {ep_end - start_idx} frames saved to {ep_output_dir}")

        # Create video if requested
        if args.save_video:
            logger.info("  Creating video...")
            video_output = output_dir / f"episode_{ep_idx}.mp4"
            create_video_from_frames(ep_output_dir, video_output, fps=args.video_fps)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

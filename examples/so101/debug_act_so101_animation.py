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
Debug script for ACT policy animation visualization.

This script loads a trained ACT policy and generates frame-by-frame animations
showing the observation image alongside 3D end-effector trajectory visualization.

Features:
- Loads specific episodes from a dataset
- Runs inference on every frame sequentially
- Generates 2-subplot figures per frame:
  - Left: Current observation image
  - Right: 3D visualization of world frame, EE trajectory history, and prediction
- Optionally compiles frames to MP4 video

Usage:
    python debug_act_so101_animation.py \
        --dataset_repo_id red_strawberry_picking_260119_merged \
        --dataset_root /path/to/dataset \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --episode_indices 0 \
        --save_video
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.utils import init_logging


# Joint names for SO101
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def create_observation_batch(
    dataset: LeRobotDataset,
    idx: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Create observation batch for policy inference.

    Args:
        dataset: LeRobotDataset instance
        idx: Current sample index
        device: Torch device

    Returns:
        Batch dictionary with observation data
    """
    sample = dataset[idx]

    # State is (6,) - joint positions
    obs_state = sample["observation.state"]

    batch = {
        "observation.state": obs_state.unsqueeze(0).to(device),  # (1, 6)
    }

    # Add camera images if available
    for key in list(sample.keys()):
        if key.startswith("observation.images"):
            img = sample[key]
            if isinstance(img, torch.Tensor):
                batch[key] = img.unsqueeze(0).to(device)  # (1, C, H, W)

    return batch


def compute_ee_trajectory(
    joint_actions: np.ndarray,
    kinematics: RobotKinematics,
) -> np.ndarray:
    """
    Compute end-effector positions from joint actions using forward kinematics.

    Args:
        joint_actions: (T, 6) joint positions in degrees
        kinematics: RobotKinematics instance

    Returns:
        (T, 3) array of EE positions
    """
    T = joint_actions.shape[0]
    ee_positions = []

    for i in range(T):
        joints = joint_actions[i].astype(np.float64)
        ee_T = kinematics.forward_kinematics(joints)
        ee_positions.append(ee_T[:3, 3].copy())

    return np.array(ee_positions)


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
):
    """
    Plot a single frame with observation image and EE trajectory.

    Creates a 2-subplot figure:
    - Left: Current observation image
    - Right: 3D EE trajectory with cumulative history, prediction, and ground truth

    Args:
        obs_image: Observation image (C,H,W or H,W,C)
        ee_history: List of historical EE positions from episode start
        pred_ee_positions: Predicted EE positions for chunk (chunk_size, 3)
        current_ee_position: Current EE position (3,)
        output_path: Path to save the plot
        episode_idx: Episode index for title
        frame_idx: Frame index for title
        axis_limits: Optional dict with 'x', 'y', 'z' tuples for fixed axis limits
        gt_ee_positions: Ground truth future EE positions (chunk_size, 3)
    """
    fig = plt.figure(figsize=(16, 8))

    # ========================================================================
    # Left subplot: Observation image
    # ========================================================================
    ax1 = fig.add_subplot(1, 2, 1)

    # Convert from (C, H, W) to (H, W, C) if needed
    if obs_image.ndim == 3 and obs_image.shape[0] in [1, 3]:
        img = obs_image.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
    else:
        img = obs_image

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
        all_pos = np.vstack([ee_hist_array, current_ee_position.reshape(1, 3)])
        if pred_ee_positions is not None:
            all_pos = np.vstack([all_pos, pred_ee_positions])
        if gt_ee_positions is not None:
            all_pos = np.vstack([all_pos, gt_ee_positions])

        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
        z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()

        x_pad = max((x_max - x_min) * 0.15, 0.02)
        y_pad = max((y_max - y_min) * 0.15, 0.02)
        z_pad = max((z_max - z_min) * 0.15, 0.02)

        ax2.set_xlim(x_min - x_pad, x_max + x_pad)
        ax2.set_ylim(y_min - y_pad, y_max + y_pad)
        ax2.set_zlim(z_min - z_pad, z_max + z_pad)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_episode_axis_limits(
    dataset: LeRobotDataset,
    episode_idx: int,
    kinematics: RobotKinematics,
) -> dict[str, tuple[float, float]]:
    """
    Compute axis limits for an episode by running FK on all ground truth actions.

    Args:
        dataset: LeRobotDataset instance
        episode_idx: Episode index
        kinematics: RobotKinematics instance

    Returns:
        Dict with 'x', 'y', 'z' tuples for axis limits
    """
    ep_info = dataset.meta.episodes[episode_idx]
    ep_start = ep_info["from_index"]
    ep_end = ep_info["to_index"]

    all_ee_positions = []

    for idx in range(ep_start, ep_end):
        sample = dataset[idx]
        gt_actions = sample['action'].cpu().numpy()

        # Get first action only (current state)
        current_joints = gt_actions[0]
        ee_T = kinematics.forward_kinematics(current_joints.astype(np.float64))
        all_ee_positions.append(ee_T[:3, 3].copy())

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
    logger = logging.getLogger("debug_act_so101_animation")

    import argparse
    parser = argparse.ArgumentParser(
        description="Debug script for ACT policy animation visualization"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., red_strawberry_picking_260119_merged)",
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
        default="./outputs/debug/animation",
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
        "--urdf_path",
        type=str,
        default="./urdf/Simulation/SO101/so101_new_calib.urdf",
        help="Path to URDF file for FK (default: ./urdf/Simulation/SO101/so101_new_calib.urdf)",
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

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={},
    )

    logger.info("Policy loaded successfully")
    logger.info(f"  Chunk size: {policy.config.chunk_size}")
    logger.info(f"  n_obs_steps: {policy.config.n_obs_steps}")
    logger.info(f"  n_action_steps: {policy.config.n_action_steps}")

    chunk_size = policy.config.chunk_size

    # ========================================================================
    # Load Dataset
    # ========================================================================
    logger.info("Loading dataset...")
    logger.info(f"  Repo ID: {args.dataset_repo_id}")
    logger.info(f"  Local root: {args.dataset_root}")

    fps = 30  # Control frequency
    action_delta_timestamps = [i / fps for i in range(chunk_size)]
    delta_timestamps = {"action": action_delta_timestamps}

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        episodes=args.episode_indices,
    )

    logger.info(f"Dataset loaded successfully!")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Episodes: {len(dataset.meta.episodes)}")

    # ========================================================================
    # Load Kinematics
    # ========================================================================
    urdf_path = Path(args.urdf_path)
    if urdf_path.exists():
        logger.info(f"Loading kinematics from: {urdf_path}")
        kinematics = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=JOINT_NAMES,
        )
        logger.info("Kinematics loaded successfully")
    else:
        logger.error(f"URDF not found at {urdf_path}")
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # ========================================================================
    # Process each episode
    # ========================================================================
    for ep_idx in args.episode_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Episode {ep_idx}")
        logger.info(f"{'='*60}")

        # Get episode bounds from metadata
        # Note: For episodes-loaded dataset, we need to map to original indices
        ep_info = dataset.meta.episodes[ep_idx]
        ep_length = ep_info["length"]
        # The dataset with episodes= contains only the requested episodes
        # So we need to find the start index in the loaded dataset
        cumulative_length = 0
        dataset_start_idx = 0
        for i in range(ep_idx):
            cumulative_length += dataset.meta.episodes[i]["length"]
        dataset_start_idx = cumulative_length
        dataset_end_idx = dataset_start_idx + ep_length

        logger.info(f"  Episode length: {ep_length} frames")
        logger.info(f"  Dataset range: [{dataset_start_idx}, {dataset_end_idx})")

        # Apply max_frames limit
        if args.max_frames is not None:
            ep_end = min(dataset_end_idx, dataset_start_idx + args.max_frames)
            logger.info(f"  Limited to {ep_end - dataset_start_idx} frames")
        else:
            ep_end = dataset_end_idx

        # Compute axis limits for this episode if requested
        axis_limits = None
        if args.fixed_axis_limits:
            logger.info("  Computing axis limits from full episode...")
            # Load full dataset (without episode filter) to compute limits
            full_dataset = LeRobotDataset(
                repo_id=args.dataset_repo_id,
                root=args.dataset_root,
                delta_timestamps=delta_timestamps,
            )
            axis_limits = compute_episode_axis_limits(full_dataset, ep_idx, kinematics)
            logger.info(f"  Axis limits: X={axis_limits['x']}, Y={axis_limits['y']}, Z={axis_limits['z']}")

        # Create episode output directory
        ep_output_dir = output_dir / f"episode_{ep_idx}"
        ep_output_dir.mkdir(parents=True, exist_ok=True)

        # Run inference on each frame
        ee_history = []  # Cumulative EE positions

        for frame_idx in range(dataset_start_idx, ep_end):
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

            # Get current state and ground truth action
            obs_state = sample["observation.state"].cpu().numpy()  # (6,)
            gt_actions = sample['action'].cpu().numpy()  # (chunk_size, 6)

            # Current EE position (from first action, which is current state)
            current_joints = gt_actions[0].astype(np.float64)
            current_ee_T = kinematics.forward_kinematics(current_joints)
            current_ee_position = current_ee_T[:3, 3].copy()

            # Add to history
            ee_history.append(current_ee_position.copy())

            # Create observation batch and run inference
            batch = create_observation_batch(dataset, frame_idx, device)

            with torch.no_grad():
                processed_batch = preprocessor(batch)
                pred_actions = policy.predict_action_chunk(processed_batch)
                pred_actions = postprocessor(pred_actions)

            if isinstance(pred_actions, dict) and "action" in pred_actions:
                pred_actions = pred_actions["action"]
            pred_actions = pred_actions[0].cpu().numpy()  # (chunk_size, 6)

            # Compute predicted EE trajectory
            pred_ee_positions = compute_ee_trajectory(pred_actions, kinematics)

            # Compute ground truth EE trajectory
            gt_ee_positions = compute_ee_trajectory(gt_actions, kinematics)

            # Generate and save plot
            frame_output = ep_output_dir / f"frame_{frame_idx:04d}.jpg"
            plot_frame_with_ee_trajectory(
                obs_image=obs_img,
                ee_history=ee_history,
                pred_ee_positions=pred_ee_positions,
                current_ee_position=current_ee_position,
                output_path=frame_output,
                episode_idx=ep_idx,
                frame_idx=frame_idx,
                axis_limits=axis_limits,
                gt_ee_positions=gt_ee_positions,
            )

            if (frame_idx - dataset_start_idx) % 10 == 0:
                logger.info(f"  Processed {frame_idx - dataset_start_idx}/{ep_end - dataset_start_idx} frames")

        logger.info(f"  Episode {ep_idx} complete: {ep_end - dataset_start_idx} frames saved to {ep_output_dir}")

        # Create video if requested
        if args.save_video:
            logger.info("  Creating video...")
            video_output = output_dir / f"episode_{ep_idx}.mp4"
            create_video_from_frames(ep_output_dir, video_output, fps=args.video_fps)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

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
Debug script for RelativeEE ACT training data loading.

This script visualizes the original trajectory chunk from the dataset
and the relative actions transformed back to absolute poses.
This helps verify that the RelativeEE dataset transformation is correct.

Features:
- Loads dataset with RelativeEEDataset wrapper
- Samples random frames from the dataset
- Extracts the original absolute trajectory chunk
- Transforms relative actions back to absolute poses
- Plots both trajectories together for comparison

Usage:
    python debug_relative_ee_dataloader.py \
        --dataset_repo_id red_strawberry_picking_260114_ee \
        --dataset_root /path/to/dataset \
        --num_samples 5 \
        --output_dir ./debug_dataloader_output
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.relative_ee_dataset import (
    RelativeEEDataset,
    pose_to_mat,
    pose10d_to_mat,
    mat_to_pose10d,
    convert_pose_mat_rep,
)
from lerobot.datasets.utils import get_delta_indices
from lerobot.utils.utils import init_logging


def axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation to 3x3 rotation matrix.

    Args:
        axis_angle: (3,) axis-angle rotation vector

    Returns:
        (3, 3) rotation matrix
    """
    from lerobot.utils.rotation import Rotation
    rot = Rotation.from_rotvec(axis_angle)
    return rot.as_matrix()


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Args:
        rot6d: (6,) 6D rotation array

    Returns:
        (3, 3) rotation matrix
    """
    a1 = rot6d[:3]
    a2 = rot6d[3:]

    # Normalize first column
    b1 = a1 / np.linalg.norm(a1)

    # Make second column orthogonal to first
    b2 = a2 - np.sum(b1 * a2) * b1
    b2 = b2 / np.linalg.norm(b2)

    # Third column via cross product
    b3 = np.cross(b1, b2)

    # Stack into rotation matrix
    rotmat = np.stack([b1, b2, b3], axis=-1)
    return rotmat


def get_original_trajectory_chunk(
    base_dataset: LeRobotDataset,
    idx: int,
    action_horizon: int,
) -> dict[str, np.ndarray]:
    """
    Get the original absolute trajectory chunk from the base dataset.

    This extracts the actual poses that would be used to compute
    the relative actions.

    When using delta_timestamps, the base dataset's 'action' key already
    contains the chunked future poses (shape: action_horizon x 7).
    These are NOT actions but the future observation states.

    Args:
        base_dataset: Original LeRobotDataset (not wrapped)
        idx: Current sample index
        action_horizon: Number of future timesteps to extract

    Returns:
        Dictionary with:
            - current_position: (3,) current position
            - current_rotmat: (3, 3) current rotation matrix
            - current_gripper: (1,) gripper state
            - future_positions: (action_horizon, 3) future positions
            - future_rotmats: (action_horizon, 3, 3) future rotations
            - future_grippers: (action_horizon,) future gripper states
    """
    # Get current item
    current_item = base_dataset[idx]

    # Get episode info
    current_episode_index = base_dataset.hf_dataset[idx]['episode_index']

    # Current state from observation
    current_state = current_item['observation.state'].cpu().numpy()
    current_position = current_state[:3]
    current_axis_angle = current_state[3:6]
    current_gripper = current_state[6]

    # Convert current rotation to matrix
    current_rotmat = axis_angle_to_rotmat(current_axis_angle)

    # Get future poses from the 'action' key
    # When using delta_timestamps, action contains future observation states
    actions = current_item['action']  # (action_horizon, 7) or (7,)
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    actions = actions.cpu().numpy()

    # DEBUG: Check if action[0] matches current observation (it should for delta=0)
    # With delta_timestamps[0]=0, action[0] should be same as observation.state
    action_0_matches_obs = np.allclose(actions[0], current_state, atol=1e-5)
    if not action_0_matches_obs:
        # In this dataset, action is the NEXT frame's observation, not current!
        # Let's check if action[0] matches observation at idx+1
        if idx + 1 < len(base_dataset):
            next_item = base_dataset[idx + 1]
            next_state = next_item['observation.state'].cpu().numpy()
            action_0_matches_next = np.allclose(actions[0], next_state, atol=1e-5)
            import sys
            print(f"\n⚠️  Dataset structure: action[i] = observation at frame i+1")
            print(f"  observation.state at idx:   {current_state[:3]}")
            print(f"  action[0] at idx:            {actions[0][:3]}")
            print(f"  observation.state at idx+1: {next_state[:3]}")
            print(f"  action[0] == next_state: {action_0_matches_next}")
            print(f"  Distance current->action[0]: {np.linalg.norm(actions[0][:3] - current_state[:3])*1000:.2f} mm")
            print(f"  Distance current->next_obs:  {np.linalg.norm(next_state[:3] - current_state[:3])*1000:.2f} mm")
            sys.stdout.flush()

    # Check for episode boundary crossings in the action horizon
    episode_indices_in_action = []
    for delta_idx in range(min(10, actions.shape[0])):  # Check first 10
        target_idx = idx + delta_idx
        if target_idx < len(base_dataset):
            target_episode = base_dataset.hf_dataset[target_idx]['episode_index']
            episode_indices_in_action.append(target_episode)

    # Extract future poses from actions
    future_positions = actions[:, :3]  # (action_horizon, 3)
    future_axis_angles = actions[:, 3:6]  # (action_horizon, 3)
    future_grippers = actions[:, 6]  # (action_horizon,)

    # Convert axis-angle to rotation matrices
    future_rotmats = np.array([
        axis_angle_to_rotmat(aa) for aa in future_axis_angles
    ])  # (action_horizon, 3, 3)

    return {
        'current_position': current_position,
        'current_axis_angle': current_axis_angle,  # Store for debugging
        'current_rotmat': current_rotmat,
        'current_gripper': current_gripper,
        'future_positions': future_positions,
        'future_axis_angles': future_axis_angles,  # Store for debugging
        'future_rotmats': future_rotmats,
        'future_grippers': future_grippers,
        'current_episode': current_episode_index,
        'episode_indices_in_action': episode_indices_in_action,
    }


def relative_actions_to_absolute(
    relative_actions: np.ndarray,
    current_position: np.ndarray,
    current_rotmat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform relative actions back to absolute poses.

    The relative actions are computed as: T_rel = T_curr^{-1} @ T_future
    To reverse: T_abs = T_curr @ T_rel

    Args:
        relative_actions: (action_horizon, 10) relative actions
            [dx, dy, dz, rot6d_0, ..., rot6d_5, gripper]
        current_position: (3,) current position
        current_rotmat: (3, 3) current rotation matrix

    Returns:
        positions: (action_horizon, 3) absolute positions
        rotmats: (action_horizon, 3, 3) absolute rotation matrices
        grippers: (action_horizon,) gripper states
    """
    action_horizon = relative_actions.shape[0]

    # Current pose as 4x4 matrix
    T_current = np.eye(4)
    T_current[:3, :3] = current_rotmat
    T_current[:3, 3] = current_position

    positions = []
    rotmats = []
    grippers = []

    for i in range(action_horizon):
        # Convert relative action to 4x4 matrix
        T_rel = pose10d_to_mat(relative_actions[i, :9])

        # Transform back to absolute: T_abs = T_current @ T_rel
        T_abs = T_current @ T_rel

        # Extract position and rotation
        positions.append(T_abs[:3, 3])
        rotmats.append(T_abs[:3, :3])
        grippers.append(relative_actions[i, 9])

    return (
        np.array(positions),
        np.array(rotmats),
        np.array(grippers),
    )


def plot_trajectory_comparison(
    original_traj: dict[str, np.ndarray],
    reconstructed_positions: np.ndarray,
    reconstructed_rotmats: np.ndarray,
    relative_actions: np.ndarray,
    output_path: str,
    sample_idx: int = 0,
):
    """
    Plot original vs reconstructed trajectory from relative actions.

    Args:
        original_traj: Dictionary with original trajectory data
        reconstructed_positions: (action_horizon, 3) positions from reversing relative actions
        reconstructed_rotmats: (action_horizon, 3, 3) rotations from reversing relative actions
        relative_actions: (action_horizon, 10) the relative actions
        output_path: Path to save the plot
        sample_idx: Sample index for title
    """
    action_horizon = len(original_traj['future_positions'])

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # -----------------------------------------------------------------------
    # 1. 3D Trajectory Plot
    # -----------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")

    # Original trajectory (from base dataset)
    # Include current position as start point
    orig_x = [original_traj['current_position'][0]] + list(original_traj['future_positions'][:, 0])
    orig_y = [original_traj['current_position'][1]] + list(original_traj['future_positions'][:, 1])
    orig_z = [original_traj['current_position'][2]] + list(original_traj['future_positions'][:, 2])

    ax1.plot(
        orig_x, orig_y, orig_z,
        "b-", linewidth=2, label="Original (from dataset)", marker="o", markersize=4
    )

    # Reconstructed trajectory (from relative actions reversed)
    # The relative actions start at t+1, so current position is not included
    # But first relative action should give position at t+1
    recon_x = list(reconstructed_positions[:, 0])
    recon_y = list(reconstructed_positions[:, 1])
    recon_z = list(reconstructed_positions[:, 2])

    ax1.plot(
        recon_x, recon_y, recon_z,
        "r--", linewidth=2, label="Reconstructed (from relative)", marker="x", markersize=4
    )
    # Mark current recon position from recon_xyz
    ax1.scatter(
        [reconstructed_positions[0, 0]],
        [reconstructed_positions[0, 1]],
        [reconstructed_positions[0, 2]],
        c="orange", s=100, marker="^", label="Reconstructed t=1", zorder=10
    )



    # Mark current position
    ax1.scatter(
        [original_traj['current_position'][0]],
        [original_traj['current_position'][1]],
        [original_traj['current_position'][2]],
        c="green", s=100, marker="*", label="Current (t=0)", zorder=10
    )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(f"Sample {sample_idx}: 3D Trajectory")
    ax1.legend()

    # Set equal aspect ratio
    all_x = np.array(orig_x + recon_x)
    all_y = np.array(orig_y + recon_y)
    all_z = np.array(orig_z + recon_z)

    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    z_range = all_z.max() - all_z.min()
    max_range = max(x_range, y_range, z_range)

    # Add 50% padding on each side for better visualization
    padding = max_range * 0.2 if max_range > 0 else 0.1

    ax1.set_xlim(all_x.mean() - x_range/2 - padding, all_x.mean() + x_range/2 + padding)
    ax1.set_ylim(all_y.mean() - y_range/2 - padding, all_y.mean() + y_range/2 + padding)
    ax1.set_zlim(all_z.mean() - z_range/2 - padding, all_z.mean() + z_range/2 + padding)

    # -----------------------------------------------------------------------
    # 2. Position Error (Original vs Reconstructed)
    # -----------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 3, 2)

    pos_errors = np.linalg.norm(
        reconstructed_positions - original_traj['future_positions'],
        axis=1
    ) * 1000  # Convert to mm

    ax2.plot(pos_errors, "r-", linewidth=2, marker="o", markersize=4)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Timestep (relative to current)")
    ax2.set_ylabel("Position Error (mm)")
    ax2.set_title("Position Error: Reconstructed vs Original")
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    mean_err = np.mean(pos_errors)
    max_err = np.max(pos_errors)
    ax2.text(
        0.02, 0.98,
        f"Mean: {mean_err:.2f} mm\nMax: {max_err:.2f} mm",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
    )

    # -----------------------------------------------------------------------
    # 3. Rotation Error (geodesic distance)
    # -----------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 3)

    rot_errors = []
    for i in range(action_horizon):
        R_orig = original_traj['future_rotmats'][i]
        R_recon = reconstructed_rotmats[i]

        # Compute relative rotation: R_diff = R_recon^T @ R_orig
        R_diff = R_recon.T @ R_orig

        # Geodesic distance on SO(3)
        trace = np.trace(R_diff)
        trace = np.clip(trace, -1.0, 3.0)
        geodesic_dist = np.arccos((trace - 1) / 2)
        rot_errors.append(np.degrees(geodesic_dist))

    rot_errors = np.array(rot_errors)

    ax3.plot(rot_errors, "g-", linewidth=2, marker="s", markersize=4)
    ax3.set_xlabel("Timestep (relative to current)")
    ax3.set_ylabel("Rotation Error (degrees)")
    ax3.set_title("Rotation Error: Reconstructed vs Original")
    ax3.grid(True, alpha=0.3)

    # Add statistics text
    mean_rot_err = np.mean(rot_errors)
    max_rot_err = np.max(rot_errors)
    ax3.text(
        0.02, 0.98,
        f"Mean: {mean_rot_err:.2f}°\nMax: {max_rot_err:.2f}°",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.5}
    )

    # -----------------------------------------------------------------------
    # 4. Per-axis position comparison
    # -----------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4)

    axes = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    for i, (axis, color) in enumerate(zip(axes, colors)):
        ax4.plot(
            original_traj['future_positions'][:, i],
            color=color, linestyle="-", marker="o", markersize=3,
            label=f"Original {axis}"
        )
        ax4.plot(
            reconstructed_positions[:, i],
            color=color, linestyle="--", marker="x", markersize=4,
            label=f"Reconstructed {axis}"
        )

    ax4.set_xlabel("Timestep (relative to current)")
    ax4.set_ylabel("Position (m)")
    ax4.set_title("Position Components: Original vs Reconstructed")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # -----------------------------------------------------------------------
    # 5. Relative action visualization (what the model actually predicts)
    # -----------------------------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 5)

    # Plot relative position components (delta from current)
    ax5.plot(
        relative_actions[:, 0] * 1000,
        "r-", linewidth=2, marker="o", markersize=3, label="dx"
    )
    ax5.plot(
        relative_actions[:, 1] * 1000,
        "g-", linewidth=2, marker="s", markersize=3, label="dy"
    )
    ax5.plot(
        relative_actions[:, 2] * 1000,
        "b-", linewidth=2, marker="^", markersize=3, label="dz"
    )

    ax5.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax5.set_xlabel("Timestep (relative to current)")
    ax5.set_ylabel("Delta Position (mm)")
    ax5.set_title("Relative Actions (Delta from Current Pose)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # -----------------------------------------------------------------------
    # 6. Gripper state
    # -----------------------------------------------------------------------
    ax6 = fig.add_subplot(2, 3, 6)

    ax6.plot(
        original_traj['future_grippers'],
        "b-o", linewidth=2, markersize=4, label="Original"
    )
    ax6.plot(
        relative_actions[:, 9],
        "r--x", linewidth=2, markersize=4, label="From Relative"
    )

    ax6.set_xlabel("Timestep (relative to current)")
    ax6.set_ylabel("Gripper State")
    ax6.set_title("Gripper: Original vs Relative")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved to: {output_path}")

    return {
        "mean_position_error": float(np.mean(pos_errors)),
        "max_position_error": float(np.max(pos_errors)),
        "mean_rotation_error": float(np.mean(rot_errors)),
        "max_rotation_error": float(np.max(rot_errors)),
    }


def main():
    init_logging()
    logger = logging.getLogger("debug_relative_ee_dataloader")

    import argparse
    parser = argparse.ArgumentParser(
        description="Debug script for RelativeEE training data loading"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., red_strawberry_picking_260114_ee)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Local path to dataset (if not provided, uses HF hub)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random samples to visualize (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./debug_relative_ee_dataloader_output",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--obs_state_horizon",
        type=int,
        default=2,
        help="Observation state horizon (must match training)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for action timestamps",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Action chunk size (action_horizon)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--specific_idx",
        type=int,
        default=None,
        help="Test a specific sample index instead of random sampling",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ========================================================================
    # Load Datasets
    # ========================================================================
    logger.info("Loading datasets...")

    # Create delta_timestamps for action
    action_delta_timestamps = [i / args.fps for i in range(args.chunk_size)]
    delta_timestamps = {"action": action_delta_timestamps}
    logger.info(f"  Action delta_timestamps: {action_delta_timestamps[:5]}... (first 5 of {len(action_delta_timestamps)})")

    # Load base dataset (original absolute poses)
    logger.info(f"  Loading base dataset: {args.dataset_repo_id}")
    base_dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
    )
    logger.info(f"  Base dataset size: {len(base_dataset)}")

    # Load RelativeEE dataset (wrapped, relative poses)
    logger.info(f"  Loading RelativeEE dataset: {args.dataset_repo_id}")
    relative_dataset = RelativeEEDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        obs_state_horizon=args.obs_state_horizon,
        delta_timestamps=delta_timestamps,
    )
    logger.info(f"  RelativeEE dataset size: {len(relative_dataset)}")

    # Get action horizon from sample
    sample = relative_dataset[0]
    action_horizon = sample['action'].shape[0]
    logger.info(f"  Action horizon: {action_horizon}")

    # ========================================================================
    # Sample indices and visualize
    # ========================================================================
    if args.specific_idx is not None:
        logger.info(f"Testing specific sample index: {args.specific_idx}")
        valid_indices = [args.specific_idx]
    else:
        logger.info(f"Generating visualizations for {args.num_samples} random samples...")
        max_idx = len(base_dataset) - action_horizon - args.obs_state_horizon
        valid_indices = np.random.choice(max_idx, args.num_samples, replace=False)

    all_errors = []

    for sample_idx, idx in enumerate(valid_indices):
        total = len(valid_indices)
        logger.info(f"\n[{sample_idx + 1}/{total}] Processing sample {idx}")

        # Get original trajectory from base dataset
        original_traj = get_original_trajectory_chunk(
            base_dataset, idx, action_horizon
        )

        # Get relative actions from RelativeEE dataset
        relative_sample = relative_dataset[idx]
        relative_actions = relative_sample['action'].cpu().numpy()  # (action_horizon, 10)

        # Reverse transform: relative -> absolute
        reconstructed_positions, reconstructed_rotmats, reconstructed_grippers = relative_actions_to_absolute(
            relative_actions,
            original_traj['current_position'],
            original_traj['current_rotmat'],
        )

        # Log some info
        logger.info(f"  Current position: {original_traj['current_position']}")
        logger.info(f"  Relative action[0] (dx,dy,dz): {relative_actions[0, :3]*1000} mm")
        logger.info(f"  Original future[0] position: {original_traj['future_positions'][0]}")
        logger.info(f"  Reconstructed future[0]: {reconstructed_positions[0]}")

        # Debug: compute distances between consecutive points
        dist_0_to_1 = np.linalg.norm(original_traj['future_positions'][0] - original_traj['current_position']) * 1000
        dist_1_to_2 = np.linalg.norm(original_traj['future_positions'][1] - original_traj['future_positions'][0]) * 1000
        dist_2_to_3 = np.linalg.norm(original_traj['future_positions'][2] - original_traj['future_positions'][1]) * 1000
        logger.info(f"  Current episode: {original_traj['current_episode']}")
        logger.info(f"  Episode indices for first 10 steps: {original_traj['episode_indices_in_action']}")
        # Check if episode boundary is crossed
        unique_episodes = set(int(e.item()) if hasattr(e, 'item') else int(e) for e in original_traj['episode_indices_in_action'])
        if len(unique_episodes) > 1:
            logger.warning(f"  ⚠️  EPISODE BOUNDARY CROSSED! Multiple episodes in action horizon: {unique_episodes}")
        else:
            logger.info(f"  All steps within same episode: {list(unique_episodes)}")
        logger.info(f"  Distance t=0 to t=1: {dist_0_to_1:.2f} mm")
        logger.info(f"  Distance t=1 to t=2: {dist_1_to_2:.2f} mm")
        logger.info(f"  Distance t=2 to t=3: {dist_2_to_3:.2f} mm")

        # Show first few positions
        for i in range(min(5, len(original_traj['future_positions']))):
            logger.info(f"  future_positions[{i}]: {original_traj['future_positions'][i]}")

        # Debug: compare rotations in detail
        logger.info(f"  Current axis-angle: {original_traj['current_axis_angle']}")
        logger.info(f"  Future[0] axis-angle (original): {original_traj['future_axis_angles'][0]}")

        # Check: convert reconstructed rotmat back to axis-angle for comparison
        from lerobot.utils.rotation import Rotation
        reconstructed_rotvec = Rotation.from_matrix(reconstructed_rotmats[0]).as_rotvec()
        logger.info(f"  Future[0] axis-angle (reconstructed): {reconstructed_rotvec}")

        # Also show the relative rotation 6D for debugging
        logger.info(f"  Relative action[0] rot6d: {relative_actions[0, 3:9]}")

        # Create comparison plot
        plot_path = output_dir / f"sample_{idx}_trajectory_comparison.png"
        errors = plot_trajectory_comparison(
            original_traj,
            reconstructed_positions,
            reconstructed_rotmats,
            relative_actions,
            str(plot_path),
            idx,
        )
        all_errors.append(errors)

        logger.info(f"  Mean position error: {errors['mean_position_error']*1000:.4f} mm")
        logger.info(f"  Mean rotation error: {errors['mean_rotation_error']:.4f} deg")

    # ========================================================================
    # Summary Statistics
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)

    all_pos_errors = [e['mean_position_error'] for e in all_errors]
    all_rot_errors = [e['mean_rotation_error'] for e in all_errors]

    logger.info(f"Average mean position error: {np.mean(all_pos_errors)*1000:.4f} mm")
    logger.info(f"Average mean rotation error: {np.mean(all_rot_errors):.4f} deg")
    logger.info(f"Std position error: {np.std(all_pos_errors)*1000:.4f} mm")
    logger.info(f"Std rotation error: {np.std(all_rot_errors):.4f} deg")

    # Save summary to file
    summary_path = output_dir / "dataloader_summary.txt"
    with open(summary_path, "w") as f:
        f.write("RelativeEE Dataloader Debug Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset_repo_id}\n")
        f.write(f"Samples tested: {args.num_samples}\n")
        f.write(f"Action horizon: {action_horizon}\n")
        f.write(f"FPS: {args.fps}\n\n")

        for i, idx in enumerate(valid_indices):
            f.write(f"\nSample {i+1} (idx={idx}):\n")
            e = all_errors[i]
            f.write(f"  Mean position error: {e['mean_position_error']*1000:.4f} mm\n")
            f.write(f"  Max position error: {e['max_position_error']*1000:.4f} mm\n")
            f.write(f"  Mean rotation error: {e['mean_rotation_error']:.4f} deg\n")
            f.write(f"  Max rotation error: {e['max_rotation_error']:.4f} deg\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Aggregate Statistics:\n")
        f.write(f"  Average mean position error: {np.mean(all_pos_errors)*1000:.4f} mm\n")
        f.write(f"  Std position error: {np.std(all_pos_errors)*1000:.4f} mm\n")
        f.write(f"  Average mean rotation error: {np.mean(all_rot_errors):.4f} deg\n")
        f.write(f"  Std rotation error: {np.std(all_rot_errors):.4f} deg\n")

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()

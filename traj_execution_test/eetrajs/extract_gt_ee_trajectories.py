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
Extract ground truth EE trajectories from a dataset.

This script loads a dataset and extracts full episode EE poses in absolute form,
then transforms so the first pose is at origin. Saved as CSV for testing robot EE control.

With --plot, generates trajectory visualizations (XY, XZ, YZ projections).

The CSV contains absolute poses [x, y, z, qx, qy, qz, qw, gripper] where
the first pose is always at origin [0, 0, 0, 0, 0, 0, 1, gripper].

Usage:
    python extract_gt_ee_trajectories.py \
        --dataset_root /mnt/ldata/sroi/sroi_lab_picking \
        --episode_indices 0 1 2 \
        --plot
"""

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def pose7d_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert 7D pose [x, y, z, roll, pitch, yaw, gripper] to 4x4 transformation matrix.

    Args:
        pose: 7D array [x, y, z, roll, pitch, yaw, gripper]

    Returns:
        (4, 4) transformation matrix
    """
    x, y, z, roll, pitch, yaw, _ = pose

    # Rotation from Euler angles (XYZ convention)
    R = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def mat_to_pose7d(T: np.ndarray, gripper: float) -> np.ndarray:
    """Convert 4x4 transformation matrix to 7D pose [x, y, z, roll, pitch, yaw, gripper].

    Args:
        T: (4, 4) transformation matrix
        gripper: Gripper state value

    Returns:
        7D array [x, y, z, roll, pitch, yaw, gripper]
    """
    position = T[:3, 3]
    R = T[:3, :3]
    rot = Rotation.from_matrix(R)
    euler = rot.as_euler('xyz')

    return np.concatenate([position, euler, [gripper]])


def mat_to_pose7d_with_quat(T: np.ndarray, gripper: float) -> np.ndarray:
    """Convert 4x4 transformation matrix to pose with quaternion [x, y, z, qx, qy, qz, qw, gripper].

    Args:
        T: (4, 4) transformation matrix
        gripper: Gripper state value

    Returns:
        8D array [x, y, z, qx, qy, qz, qw, gripper]
    """
    position = T[:3, 3]
    R = T[:3, :3]
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # [x, y, z, w]

    return np.concatenate([position, quat, [gripper]])


def extract_episode_poses(
    dataset: LeRobotDataset,
    ep_idx: int,
) -> tuple[list, np.ndarray]:
    """Extract all absolute poses for an episode.

    Args:
        dataset: LeRobotDataset instance (not RelativeEEDataset!)
        ep_idx: Episode index

    Returns:
        Tuple of (poses list, T_inv_first) where:
        - poses: list of [x, y, z, qx, qy, qz, qw, gripper] with first pose at origin
        - T_inv_first: inverse of first pose's transformation matrix
    """
    ep_info = dataset.meta.episodes[ep_idx]
    ep_length = ep_info["length"]

    # Find start index
    start_idx = 0
    for i in range(ep_idx):
        start_idx += dataset.meta.episodes[i]["length"]

    # First, collect all original absolute poses
    original_poses = []
    for frame_offset in range(ep_length):
        idx = start_idx + frame_offset
        sample = dataset[idx]

        # Get action (absolute EE pose in base frame)
        action = sample["action"]
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        original_poses.append(action)

    original_poses = np.array(original_poses)

    # Convert first pose to transformation matrix
    T_first = pose7d_to_mat(original_poses[0])
    T_inv_first = np.linalg.inv(T_first)

    # Transform all poses to first pose's frame
    transformed_poses = []
    for i, pose in enumerate(original_poses):
        T = pose7d_to_mat(pose)
        T_transformed = T_inv_first @ T
        pose_transformed = mat_to_pose7d_with_quat(T_transformed, pose[6])
        transformed_poses.append(pose_transformed)

    return transformed_poses, T_inv_first


def plot_trajectory_from_poses(poses: list, ep_idx: int, save_path: Path | None = None):
    """Plot trajectory projections from pose list.

    Args:
        poses: List of [x, y, z, qx, qy, qz, qw, gripper]
        ep_idx: Episode index for title
        save_path: If provided, save plot instead of showing
    """
    poses = np.array(poses)
    x, y, z = poses[:, 0], poses[:, 1], poses[:, 2]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XY plot (top view)
    axes[0].plot(x, y, 'b-', linewidth=1.5, label='Trajectory')
    axes[0].plot(x[0], y[0], 'go', markersize=8, label='Start (origin)')
    axes[0].plot(x[-1], y[-1], 'ro', markersize=8, label='End')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title(f'Episode {ep_idx}: XY Plane (Top View)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # XZ plot (side view)
    axes[1].plot(x, z, 'b-', linewidth=1.5, label='Trajectory')
    axes[1].plot(x[0], z[0], 'go', markersize=8, label='Start (origin)')
    axes[1].plot(x[-1], z[-1], 'ro', markersize=8, label='End')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    axes[1].set_title(f'Episode {ep_idx}: XZ Plane (Side View)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    # YZ plot (front view)
    axes[2].plot(y, z, 'b-', linewidth=1.5, label='Trajectory')
    axes[2].plot(y[0], z[0], 'go', markersize=8, label='Start (origin)')
    axes[2].plot(y[-1], z[-1], 'ro', markersize=8, label='End')
    axes[2].set_xlabel('Y (m)')
    axes[2].set_ylabel('Z (m)')
    axes[2].set_title(f'Episode {ep_idx}: YZ Plane (Front View)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')

    # Log trajectory statistics
    logger.info(f"  Trajectory range:")
    logger.info(f"    X: [{x.min():.4f}, {x.max():.4f}] m (span: {x.max() - x.min():.4f} m)")
    logger.info(f"    Y: [{y.min():.4f}, {y.max():.4f}] m (span: {y.max() - y.min():.4f} m)")
    logger.info(f"    Z: [{z.min():.4f}, {z.max():.4f}] m (span: {z.max() - z.min():.4f} m)")
    path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    logger.info(f"  Total path length: {path_length:.4f} m")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    init_logging()

    import argparse
    parser = argparse.ArgumentParser(
        description="Extract ground truth EE trajectories from dataset (absolute poses)"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--episode_indices",
        type=int,
        nargs='+',
        default=None,
        help="Episode indices to extract (default: all episodes)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/eetrajs",
        help="Base output directory",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot trajectory projections (XY, XZ, YZ planes) and save as PNG",
    )

    args = parser.parse_args()

    # Setup
    dataset_root = Path(args.dataset_root)
    dataset_name = dataset_root.name
    repo_id = dataset_name

    output_base = Path(args.output_dir) / dataset_name
    output_base.mkdir(parents=True, exist_ok=True)

    # Load dataset (use LeRobotDataset directly, not RelativeEEDataset!)
    logger.info(f"Loading dataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))

    num_episodes = len(dataset.meta.episodes)
    logger.info(f"Dataset loaded: {len(dataset)} frames, {num_episodes} episodes")

    # Determine which episodes to process
    if args.episode_indices is None:
        episode_indices = list(range(num_episodes))
        logger.info(f"Processing all {num_episodes} episodes")
    else:
        episode_indices = args.episode_indices
        logger.info(f"Processing {len(episode_indices)} episodes: {episode_indices}")

    # CSV header
    header = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']

    # Extract trajectories
    for ep_idx in episode_indices:
        logger.info(f"\nExtracting episode {ep_idx}...")

        poses, _ = extract_episode_poses(dataset, ep_idx)

        # Save as CSV
        csv_path = output_base / f"episode_{ep_idx:04d}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(poses)

        logger.info(f"  Saved to {csv_path}")
        logger.info(f"    Timesteps: {len(poses)}")
        logger.info(f"    First pose (origin): x={poses[0][0]:.6f}, y={poses[0][1]:.6f}, z={poses[0][2]:.6f}")

        # Plot trajectory if requested
        if args.plot:
            plot_path = output_base / f"episode_{ep_idx:04d}.png"
            plot_trajectory_from_poses(poses, ep_idx, save_path=plot_path)

    logger.info(f"\nDone! Trajectories saved to {output_base}/")


if __name__ == "__main__":
    main()

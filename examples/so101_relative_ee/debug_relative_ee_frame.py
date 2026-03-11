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
Debug script for visualizing RelativeEE dataset frames and GT trajectory.

This script loads a RelativeEE dataset and generates a single static plot showing:
- Base frame (RGB axes at origin)
- EE frame (RGB axes at current position)
- Full episode GT trajectory (correctly framed)

No inference, no prediction, no IK - just visualization of the dataset structure.

Usage:
    python debug_relative_ee_frame.py \
        --dataset_repo_id sroi_lab_picking \
        --dataset_root /mnt/data0/data/sroi/sroi_lab_picking \
        --episode_idx 0 \
        --frame_idx 0

    Output will be saved to:
    outputs/debug/animation_relative_ee/sroi_lab_picking/episode_0/frame_0000.png
"""

import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v3 as iio

from lerobot.datasets.relative_ee_dataset import (
    RelativeEEDataset,
    pose_to_mat,
    pose10d_to_mat,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def compute_gt_trajectory_in_first_frame(
    dataset: RelativeEEDataset,
    episode_idx: int,
) -> np.ndarray:
    """
    Compute GT trajectory expressed in first GT frame (for debugging).

    This computes: T_in_first_frame = inv(T_first_gt) @ T_gt
    The result is the SE(3) transform from first GT pose to each GT point,
    expressed in the first GT frame's coordinate system.

    Args:
        dataset: RelativeEEDataset instance
        episode_idx: Episode index

    Returns:
        (N, 3) array of EE positions in first GT frame
    """
    # Get episode info from dataset
    ep_info = dataset.meta.episodes[episode_idx]
    ep_length = ep_info["length"]

    # Find start index in dataset
    start_idx = 0
    for i in range(episode_idx):
        start_idx += dataset.meta.episodes[i]["length"]

    # Get the first frame's EE pose (in baselink frame)
    first_abs_sample = LeRobotDataset.__getitem__(dataset, start_idx)
    first_abs_state = first_abs_sample['observation.state'].cpu().numpy()
    first_pose_6d = first_abs_state[:6]
    first_ee_pose = pose_to_mat(first_pose_6d)

    all_ee_positions = []

    # Process each frame to get EE positions
    for i in range(ep_length):
        idx = start_idx + i

        # Get ABSOLUTE EE pose from parent LeRobotDataset (in baselink frame)
        abs_sample = LeRobotDataset.__getitem__(dataset, idx)
        abs_state = abs_sample['observation.state'].cpu().numpy()

        # Convert absolute state to 4x4 matrix
        abs_pose_6d = abs_state[:6]
        T_gt_in_baselink = pose_to_mat(abs_pose_6d)

        # Compute SE(3) transform in first GT frame
        T_gt_in_first_frame = np.linalg.inv(first_ee_pose) @ T_gt_in_baselink

        # Extract position (this is the translation in first GT frame)
        current_ee_position = T_gt_in_first_frame[:3, 3].copy()
        all_ee_positions.append(current_ee_position)

    # Stack all positions into (N, 3) array
    return np.array(all_ee_positions)


def compute_full_episode_gt_trajectory(
    dataset: RelativeEEDataset,
    episode_idx: int,
    reset_pose_ee: np.ndarray,
) -> np.ndarray:
    """
    Compute full episode GT trajectory reparented to reset_pose_ee frame.

    This performs SE(3) transform-based reparenting:
    1. T_in_first_frame = inv(T_first_gt) @ T_gt (relative SE(3) transform)
    2. T_in_ee = T_in_first_frame (same SE(3) transform, different frame reference)
    3. T_in_baselink = T_baselink_to_ee @ T_in_ee

    Args:
        dataset: RelativeEEDataset instance
        episode_idx: Episode index
        reset_pose_ee: (4, 4) reset pose EE transformation matrix in baselink frame

    Returns:
        (N, 3) array of EE positions after reparenting to reset_pose_ee frame
    """
    # Get episode info from dataset
    ep_info = dataset.meta.episodes[episode_idx]
    ep_length = ep_info["length"]

    # Find start index in dataset
    start_idx = 0
    for i in range(episode_idx):
        start_idx += dataset.meta.episodes[i]["length"]

    # Get the first frame's EE pose (in baselink frame)
    first_abs_sample = LeRobotDataset.__getitem__(dataset, start_idx)
    first_abs_state = first_abs_sample['observation.state'].cpu().numpy()
    first_pose_6d = first_abs_state[:6]
    first_ee_pose = pose_to_mat(first_pose_6d)

    all_ee_positions = []

    # Process each frame to get EE positions
    for i in range(ep_length):
        idx = start_idx + i

        # Get ABSOLUTE EE pose from parent LeRobotDataset (in baselink frame)
        abs_sample = LeRobotDataset.__getitem__(dataset, idx)
        abs_state = abs_sample['observation.state'].cpu().numpy()

        # Convert absolute state to 4x4 matrix
        abs_pose_6d = abs_state[:6]
        T_gt_in_baselink = pose_to_mat(abs_pose_6d)

        # Step 1: T_in_first_frame = inv(T_first_gt) @ T_gt (relative SE(3) transform)
        T_gt_in_first_frame = np.linalg.inv(first_ee_pose) @ T_gt_in_baselink

        # Step 2 & 3: T_in_baselink = T_baselink_to_ee @ T_in_ee
        # where T_in_ee = T_in_first_frame
        T_result_in_baselink = reset_pose_ee @ T_gt_in_first_frame

        # Extract position
        current_ee_position = T_result_in_baselink[:3, 3].copy()
        all_ee_positions.append(current_ee_position)

    # Stack all positions into (N, 3) array
    return np.array(all_ee_positions)


def compute_raw_gt_trajectory_with_poses(
    dataset: RelativeEEDataset,
    episode_idx: int,
) -> tuple:
    """
    Compute raw GT trajectory with full SE(3) poses (no transformation).

    This extracts the absolute EE poses (position + rotation) directly from the dataset,
    staying in the original baselink frame (no reset_pose_ee transformation).

    Args:
        dataset: RelativeEEDataset instance
        episode_idx: Episode index

    Returns:
        Tuple of (positions, poses):
        - positions: (N, 3) array of EE positions in baselink frame
        - poses: (N, 4, 4) array of EE pose matrices in baselink frame
    """
    # Get episode info from dataset
    ep_info = dataset.meta.episodes[episode_idx]
    ep_length = ep_info["length"]

    # Find start index in dataset
    start_idx = 0
    for i in range(episode_idx):
        start_idx += dataset.meta.episodes[i]["length"]

    all_ee_positions = []
    all_ee_poses = []

    # Process each frame to get absolute EE poses
    for i in range(ep_length):
        idx = start_idx + i

        # Get absolute EE pose from parent LeRobotDataset
        abs_sample = LeRobotDataset.__getitem__(dataset, idx)
        abs_state = abs_sample['observation.state'].cpu().numpy()

        # Convert absolute state to 4x4 matrix
        abs_pose_6d = abs_state[:6]
        current_ee_pose = pose_to_mat(abs_pose_6d)
        current_ee_position = current_ee_pose[:3, 3].copy()

        all_ee_positions.append(current_ee_position)
        all_ee_poses.append(current_ee_pose.copy())

    return np.array(all_ee_positions), np.array(all_ee_poses)


def compute_raw_action_trajectory(
    dataset: RelativeEEDataset,
    episode_idx: int,
) -> np.ndarray:
    """
    Compute raw action trajectory from dataset (no frame transformation).

    This extracts the absolute EE positions directly from the dataset's observation.state,
    staying in the original world frame (no reset_pose_ee transformation, no action chaining).

    Args:
        dataset: RelativeEEDataset instance
        episode_idx: Episode index

    Returns:
        (N, 3) array of EE positions in original world frame
    """
    # Get episode info from dataset
    ep_info = dataset.meta.episodes[episode_idx]
    ep_length = ep_info["length"]

    # Find start index in dataset
    start_idx = 0
    for i in range(episode_idx):
        start_idx += dataset.meta.episodes[i]["length"]

    all_ee_positions = []

    # Process each frame to get absolute EE positions
    for i in range(ep_length):
        idx = start_idx + i

        # Get absolute EE position from parent LeRobotDataset
        abs_sample = LeRobotDataset.__getitem__(dataset, idx)
        abs_state = abs_sample['observation.state'].cpu().numpy()

        # Convert absolute state to 4x4 matrix and extract position
        abs_pose_6d = abs_state[:6]
        current_ee_pose = pose_to_mat(abs_pose_6d)
        current_ee_position = current_ee_pose[:3, 3].copy()

        all_ee_positions.append(current_ee_position)

    return np.array(all_ee_positions)


def plot_frame_with_gt_trajectory(
    base_T: np.ndarray,
    ee_frame_T: np.ndarray,
    gt_trajectory: np.ndarray,
    output_path: Path,
    episode_idx: int,
    frame_idx: int,
):
    """
    Create a 3D visualization showing base frame, EE frame, and GT trajectory with RGB axes.

    Args:
        base_T: Base frame transform (4x4)
        ee_frame_T: EE frame transform (4x4)
        gt_trajectory: GT trajectory array, shape (N, 3)
        output_path: Path to save the plot
        episode_idx: Episode index for title
        frame_idx: Frame index for title
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Helper function to draw RGB frame axes
    def draw_frame_axes(origin, R, length=0.05, label_prefix=""):
        """Draw RGB axes for a coordinate frame."""
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']

        for i in range(3):
            direction = R[:, i]
            end_point = origin + direction * length

            ax.plot(
                [origin[0], end_point[0]],
                [origin[1], end_point[1]],
                [origin[2], end_point[2]],
                color=colors[i], linewidth=3, alpha=0.9
            )

            ax.text(
                end_point[0], end_point[1], end_point[2],
                f'{label_prefix}{labels[i]}',
                color=colors[i], fontsize=10, fontweight='bold'
            )

    # Draw base_link frame at origin (RGB axes)
    base_origin = base_T[:3, 3]
    base_R = base_T[:3, :3]
    draw_frame_axes(base_origin, base_R, length=0.1, label_prefix="Base ")

    # Draw EE frame (RGB axes)
    ee_origin = ee_frame_T[:3, 3]
    ee_R = ee_frame_T[:3, :3]
    draw_frame_axes(ee_origin, ee_R, length=0.08, label_prefix="EE ")

    # Add a line from base origin to EE origin
    ax.plot(
        [base_origin[0], ee_origin[0]],
        [base_origin[1], ee_origin[1]],
        [base_origin[2], ee_origin[2]],
        'gray', linestyle='--', linewidth=1, alpha=0.5
    )

    # Plot GT trajectory
    if len(gt_trajectory) > 0:
        ax.plot(
            gt_trajectory[:, 0],
            gt_trajectory[:, 1],
            gt_trajectory[:, 2],
            color='blue', linewidth=2, alpha=0.6, label='GT trajectory'
        )
        # Mark start and end
        ax.scatter(
            gt_trajectory[0, 0], gt_trajectory[0, 1], gt_trajectory[0, 2],
            c='green', s=100, marker='o', label='Start', zorder=10
        )
        ax.scatter(
            gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2],
            c='red', s=100, marker='x', label='End', zorder=10
        )
        # Mark current EE position on trajectory
        ax.scatter(
            ee_origin[0], ee_origin[1], ee_origin[2],
            c='orange', s=150, marker='*', label='Current EE', zorder=10
        )

    # Add sphere at origin
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    sphere_x = 0.02 * np.cos(u) * np.sin(v)
    sphere_y = 0.02 * np.sin(u) * np.sin(v)
    sphere_z = 0.02 * np.cos(v)
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.2)

    # Labels and formatting
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(
        f'Episode {episode_idx}, Frame {frame_idx} - Dataset Frames & GT Trajectory\n'
        f'Base Frame (RGB at origin), EE Frame (RGB offset)',
        fontsize=12
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Auto-compute axis limits from GT trajectory
    all_points = []
    all_points.append(base_origin)
    all_points.append(ee_origin)
    all_points.extend(gt_trajectory)
    all_points = np.array(all_points)

    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def plot_raw_action_trajectory(
    raw_action_trajectory: np.ndarray,
    output_path: Path,
    episode_idx: int,
    title_suffix: str = "Raw Action Trajectory (No Frame Transform)\nAbsolute positions from observation.state in original world frame",
):
    """
    Create a 3D visualization showing raw action trajectory with RGB frame at origin.

    Args:
        raw_action_trajectory: Raw action trajectory array, shape (N, 3) in world frame
        output_path: Path to save the plot
        episode_idx: Episode index for title
        title_suffix: Additional text for title
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Helper function to draw RGB frame axes
    def draw_frame_axes(origin, R, length=0.05, label_prefix=""):
        """Draw RGB axes for a coordinate frame."""
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']

        for i in range(3):
            direction = R[:, i]
            end_point = origin + direction * length

            ax.plot(
                [origin[0], end_point[0]],
                [origin[1], end_point[1]],
                [origin[2], end_point[2]],
                color=colors[i], linewidth=3, alpha=0.9
            )

            ax.text(
                end_point[0], end_point[1], end_point[2],
                f'{label_prefix}{labels[i]}',
                color=colors[i], fontsize=10, fontweight='bold'
            )

    # Draw RGB frame at origin (identity frame)
    origin = np.array([0.0, 0.0, 0.0])
    R = np.eye(3)
    draw_frame_axes(origin, R, length=0.1, label_prefix="")

    # Plot raw action trajectory
    if len(raw_action_trajectory) > 0:
        ax.plot(
            raw_action_trajectory[:, 0],
            raw_action_trajectory[:, 1],
            raw_action_trajectory[:, 2],
            color='green', linewidth=2, alpha=0.6, label='Raw action trajectory'
        )
        # Mark start and end
        ax.scatter(
            raw_action_trajectory[0, 0], raw_action_trajectory[0, 1], raw_action_trajectory[0, 2],
            c='blue', s=100, marker='o', label='Start', zorder=10
        )
        ax.scatter(
            raw_action_trajectory[-1, 0], raw_action_trajectory[-1, 1], raw_action_trajectory[-1, 2],
            c='red', s=100, marker='x', label='End', zorder=10
        )

    # Labels and formatting
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(
        f'Episode {episode_idx} - {title_suffix}',
        fontsize=12
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Auto-compute axis limits from trajectory
    x_min, x_max = raw_action_trajectory[:, 0].min(), raw_action_trajectory[:, 0].max()
    y_min, y_max = raw_action_trajectory[:, 1].min(), raw_action_trajectory[:, 1].max()
    z_min, z_max = raw_action_trajectory[:, 2].min(), raw_action_trajectory[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved raw action plot to {output_path}")
    plt.close()


def plot_raw_gt_with_frames(
    positions: np.ndarray,
    poses: np.ndarray,
    output_path: Path,
    episode_idx: int,
    frame_stride: int = 10,
):
    """
    Create a 3D visualization showing raw GT trajectory with RGB frames at regular intervals.

    Args:
        positions: Raw GT trajectory positions, shape (N, 3) in baselink frame
        poses: Raw GT trajectory poses, shape (N, 4, 4) in baselink frame
        output_path: Path to save the plot
        episode_idx: Episode index for title
        frame_stride: Draw RGB frame every N poses (default: 10)
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Helper function to draw RGB frame axes
    def draw_frame_axes(origin, R, length=0.03, label_prefix=""):
        """Draw RGB axes for a coordinate frame."""
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']

        for i in range(3):
            direction = R[:, i]
            end_point = origin + direction * length

            ax.plot(
                [origin[0], end_point[0]],
                [origin[1], end_point[1]],
                [origin[2], end_point[2]],
                color=colors[i], linewidth=2, alpha=0.7
            )

    # Draw RGB frame at regular intervals along the trajectory
    # Start from frame_stride to avoid duplicate at index 0
    for i in range(frame_stride, len(poses), frame_stride):
        origin = poses[i][:3, 3]
        R = poses[i][:3, :3]
        draw_frame_axes(origin, R, length=0.04)

    # Always draw the first and last frames
    if len(poses) > 0:
        origin = poses[0][:3, 3]
        R = poses[0][:3, :3]
        draw_frame_axes(origin, R, length=0.05)

        origin = poses[-1][:3, 3]
        R = poses[-1][:3, :3]
        draw_frame_axes(origin, R, length=0.05)

    # Plot raw GT trajectory
    if len(positions) > 0:
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color='green', linewidth=2, alpha=0.6, label='Raw GT trajectory'
        )
        # Mark start and end
        ax.scatter(
            positions[0, 0], positions[0, 1], positions[0, 2],
            c='blue', s=100, marker='o', label='Start', zorder=10
        )
        ax.scatter(
            positions[-1, 0], positions[-1, 1], positions[-1, 2],
            c='red', s=100, marker='x', label='End', zorder=10
        )

    # Draw a base frame at origin
    base_R = np.eye(3)
    draw_frame_axes(np.array([0.0, 0.0, 0.0]), base_R, length=0.08, label_prefix="Base ")

    # Labels and formatting
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(
        f'Episode {episode_idx} - Raw GT Trajectory with EE Frames\n'
        f'RGB frames shown every {frame_stride} poses (baselink coordinates)',
        fontsize=12
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Auto-compute axis limits from trajectory
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved raw GT with frames plot to {output_path}")
    plt.close()


def plot_raw_action_trajectory_2d(
    raw_action_trajectory: np.ndarray,
    output_path: Path,
    episode_idx: int,
):
    """
    Create a 2-subplot figure showing trajectory projections (x-y, x-z, y-z views).

    Args:
        raw_action_trajectory: Raw action trajectory array, shape (N, 3) in world frame
        output_path: Path to save the plot
        episode_idx: Episode index for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Define projections: (x_idx, y_idx, xlabel, ylabel)
    projections = [
        (0, 1, 'X', 'Y'),      # x-y view
        (0, 2, 'X', 'Z'),      # x-z view
        (1, 2, 'Y', 'Z'),      # y-z view
    ]

    for ax_idx, (x_idx, y_idx, xlabel, ylabel) in enumerate(projections):
        ax = axes[ax_idx]

        # Plot raw action trajectory
        ax.plot(
            raw_action_trajectory[:, x_idx],
            raw_action_trajectory[:, y_idx],
            'g-', linewidth=2, alpha=0.6, label='Raw action trajectory'
        )

        # Mark start and end
        ax.scatter(
            raw_action_trajectory[0, x_idx],
            raw_action_trajectory[0, y_idx],
            c='blue', s=100, marker='o', label='Start', zorder=10
        )
        ax.scatter(
            raw_action_trajectory[-1, x_idx],
            raw_action_trajectory[-1, y_idx],
            c='red', s=100, marker='x', label='End', zorder=10
        )

        ax.set_xlabel(f'{xlabel} (m)')
        ax.set_ylabel(f'{ylabel} (m)')
        ax.set_title(f'{xlabel}-{ylabel} Projection')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.suptitle(
        f'Episode {episode_idx} - Raw Action Trajectory (2D Projections)\n'
        f'Absolute positions from observation.state in original world frame',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved raw action 2D plot to {output_path}")
    plt.close()


def create_observation_video(
    dataset: RelativeEEDataset,
    episode_idx: int,
    output_path: Path,
    fps: int = 30,
):
    """
    Create an MP4 video from episode observation images.

    Args:
        dataset: RelativeEEDataset instance
        episode_idx: Episode index
        output_path: Path to save the video
        fps: Frames per second for output video
    """
    # Get episode info from dataset
    ep_info = dataset.meta.episodes[episode_idx]
    ep_length = ep_info["length"]

    # Find start index in dataset
    start_idx = 0
    for i in range(episode_idx):
        start_idx += dataset.meta.episodes[i]["length"]

    logger.info(f"  Creating observation video from {ep_length} frames...")
    frames = []

    # Collect all observation frames
    for i in range(ep_length):
        idx = start_idx + i
        sample = dataset[idx]

        # Get observation image (key is observation.images.camera)
        obs = sample['observation.images.camera']  # Shape depends on dataset format

        # Handle different image formats
        if obs.ndim == 4:  # (T, C, H, W) - temporal dimension
            img = obs[-1]  # Use last timestep
        elif obs.ndim == 3:  # (C, H, W)
            img = obs
        else:  # (H, W, C) or (H, W)
            img = obs

        # Convert to numpy and transpose if needed: (C, H, W) -> (H, W, C)
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W) format
            img = img.transpose(1, 2, 0)

        # Ensure uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Remove channel dimension if grayscale
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        frames.append(img)

    # Stack frames and write video
    frames_array = np.stack(frames)
    # Use 'auto' codec to let imageio choose the best available codec
    iio.imwrite(output_path, frames_array, fps=fps)
    logger.info(f"  Saved observation video to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RelativeEE dataset frames and GT trajectory"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Repository ID of the dataset (e.g., 'sroi_lab_picking')",
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
        required=True,
        help="Episode indices to visualize (e.g., 0 1 2 3)",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index within episode to visualize (EE frame position)",
    )
    parser.add_argument(
        "--target_frame",
        type=str,
        default="gripper_frame_link",
        help="Name of end-effector frame in URDF (default: gripper_frame_link)",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="urdf/Simulation/SO101/so101_new_calib.urdf",
        help="Path to robot URDF file for IK (default: urdf/Simulation/SO101/so101_new_calib.urdf)",
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
        "--joint_names",
        type=str,
        nargs=6,
        default=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Joint names for IK (default: shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper)",
    )
    parser.add_argument(
        "--obs_state_horizon",
        type=int,
        default=2,
        help="Number of temporal steps in observations (default: 2)",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Save episode observations as MP4 video",
    )

    args = parser.parse_args()
    init_logging()

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_root}/{args.dataset_repo_id}")

    # Create delta_timestamps for dataset
    fps = 30  # Default fps for action timestamps
    chunk_size = 100  # Default chunk size
    action_delta_timestamps = [i / fps for i in range(chunk_size)]
    delta_timestamps = {"action": action_delta_timestamps}

    dataset = RelativeEEDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        obs_state_horizon=args.obs_state_horizon,
        delta_timestamps=delta_timestamps,
        compute_stats=False,
    )

    logger.info(f"Dataset loaded successfully!")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Total episodes: {len(dataset.meta.episodes)}")
    logger.info(f"  Processing episodes: {args.episode_indices}")

    # Initialize kinematics
    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame,
        joint_names=args.joint_names,
    )
    logger.info("IK solver initialized successfully!")
    logger.info(f"  URDF: {args.urdf_path}")
    logger.info(f"  Target frame: {args.target_frame}")

    # Get reset pose from args
    reset_pose = np.array(args.reset_pose, dtype=np.float64)
    logger.info(f"Reset pose: {reset_pose}")

    # Compute reset_pose_ee (forward kinematics of reset pose)
    reset_pose_ee = kinematics.forward_kinematics(reset_pose)
    logger.info(f"Reset pose EE: {reset_pose_ee[:3, 3]}")

    # Process each episode
    for ep_idx in args.episode_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing episode {ep_idx}")
        logger.info(f"{'='*60}")

        # Get episode info
        ep_info = dataset.meta.episodes[ep_idx]
        ep_length = ep_info["length"]
        logger.info(f"  Episode length: {ep_length} frames")

        # Find start index in dataset
        start_idx = 0
        for i in range(ep_idx):
            start_idx += dataset.meta.episodes[i]["length"]

        # Compute full episode GT trajectory
        logger.info("  Computing full episode GT trajectory...")
        gt_trajectory = compute_full_episode_gt_trajectory(
            dataset, ep_idx, reset_pose_ee
        )
        logger.info(f"  GT trajectory shape: {gt_trajectory.shape}")

        # Compute GT trajectory in first GT frame (for debugging)
        logger.info("  Computing GT trajectory in first GT frame...")
        gt_trajectory_in_first_frame = compute_gt_trajectory_in_first_frame(
            dataset, ep_idx
        )
        logger.info(f"  GT trajectory in first frame shape: {gt_trajectory_in_first_frame.shape}")

        # Get current EE position for the specified frame
        frame_idx_global = start_idx + args.frame_idx
        abs_sample = LeRobotDataset.__getitem__(dataset, frame_idx_global)
        abs_state = abs_sample['observation.state'].cpu().numpy()
        current_ee_pose = pose_to_mat(abs_state[:6])

        # Get the first frame's EE pose
        first_abs_sample = LeRobotDataset.__getitem__(dataset, start_idx)
        first_abs_state = first_abs_sample['observation.state'].cpu().numpy()
        first_ee_pose = pose_to_mat(first_abs_state[:6])

        # Use SE(3) transform-based reparenting (same as GT trajectory computation)
        # Step 1: T_in_first_frame = inv(T_first_gt) @ T_current_ee
        T_current_ee_in_first_frame = np.linalg.inv(first_ee_pose) @ current_ee_pose

        # Step 2 & 3: T_in_baselink = T_baselink_to_ee @ T_in_ee
        T_current_ee_in_baselink = reset_pose_ee @ T_current_ee_in_first_frame
        current_ee_position_aligned = T_current_ee_in_baselink[:3, 3].copy()

        # Build EE frame transform (use reset orientation, current position)
        ee_frame_T = reset_pose_ee.copy()
        ee_frame_T[:3, 3] = current_ee_position_aligned

        # Create base frame transform (identity at origin)
        base_T = np.eye(4)

        # Create output directory
        output_dir = Path("outputs/debug/animation_relative_ee") / args.dataset_repo_id / f"episode_{ep_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"frame_{args.frame_idx:04d}.png"

        # Plot
        plot_frame_with_gt_trajectory(
            base_T=base_T,
            ee_frame_T=ee_frame_T,
            gt_trajectory=gt_trajectory,
            output_path=output_path,
            episode_idx=ep_idx,
            frame_idx=args.frame_idx,
        )

        # Compute and plot raw GT trajectory with poses (no frame transformation)
        logger.info("  Computing raw GT trajectory with poses...")
        raw_gt_positions, raw_gt_poses = compute_raw_gt_trajectory_with_poses(dataset, ep_idx)
        logger.info(f"  Raw GT positions shape: {raw_gt_positions.shape}")
        logger.info(f"  Raw GT poses shape: {raw_gt_poses.shape}")

        # Plot raw GT trajectory with EE frames
        raw_gt_with_frames_output_path = output_dir / "raw_gt_with_frames.png"
        plot_raw_gt_with_frames(
            positions=raw_gt_positions,
            poses=raw_gt_poses,
            output_path=raw_gt_with_frames_output_path,
            episode_idx=ep_idx,
            frame_stride=10,
        )

        # Plot raw action trajectory (3D with RGB frame)
        raw_action_output_path = output_dir / "raw_action.png"
        plot_raw_action_trajectory(
            raw_action_trajectory=raw_gt_positions,
            output_path=raw_action_output_path,
            episode_idx=ep_idx,
        )

        # Plot raw action trajectory (2D projections)
        raw_action_2d_output_path = output_dir / "raw_action_2d.png"
        plot_raw_action_trajectory_2d(
            raw_action_trajectory=raw_gt_positions,
            output_path=raw_action_2d_output_path,
            episode_idx=ep_idx,
        )

        # Plot GT trajectory in first GT frame (debug visualization)
        gt_in_first_frame_output_path = output_dir / "gt_in_first_frame.png"
        plot_raw_action_trajectory(
            raw_action_trajectory=gt_trajectory_in_first_frame,
            output_path=gt_in_first_frame_output_path,
            episode_idx=ep_idx,
            title_suffix="GT Trajectory in First GT Frame\nT_in_first_frame = inv(T_first_gt) @ T_gt (relative SE(3) transform)",
        )
        logger.info(f"  Saved GT trajectory in first frame to {gt_in_first_frame_output_path}")

        # Create observation video if requested
        if args.mp4:
            obs_video_output_path = output_dir / "observations.mp4"
            create_observation_video(
                dataset=dataset,
                episode_idx=ep_idx,
                output_path=obs_video_output_path,
                fps=30,
            )

        logger.info(f"  Episode {ep_idx} complete!")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

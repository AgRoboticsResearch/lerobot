#!/usr/bin/env python

"""
Visualize camera trajectory from SLAM output in 3D and 2D.

Reads a CameraTrajectory.txt file (format: 12 values per line = 3x4 transformation matrix)
and creates:
- 3D visualization with RGB frames every 10 steps
- 2D projections (xy, xz, yz views)

Usage:
    python visualize_camera_trajectory.py <trajectory_file>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_camera_trajectory(file_path: str) -> np.ndarray:
    """
    Load camera trajectory from file.

    Supports two formats:
    1. Original format: 12 floats per line (3x4 transformation matrix in row-major order)
    2. Transformed format: timestamp + 9 floats (3x3 rotation + translation, or 3x4 without last row)

    Args:
        file_path: Path to trajectory file

    Returns:
        (N, 4, 4) array of 4x4 homogeneous transformation matrices
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            values = [float(x) for x in line.split()]

            # Skip empty lines
            if len(values) == 0:
                continue

            if len(values) == 12:
                # Original format: 3x4 transformation matrix (row-major)
                mat_3x4 = np.array(values).reshape(3, 4)
                mat_4x4 = np.eye(4)
                mat_4x4[:3, :3] = mat_3x4[:, :3]
                mat_4x4[:3, 3] = mat_3x4[:, 3]
                poses.append(mat_4x4)
            elif len(values) == 16:
                # 4x4 homogeneous transformation matrix
                poses.append(np.array(values).reshape(4, 4))
            elif len(values) == 13:
                # Transformed format: timestamp + 12 values (3x4 matrix in row-major order)
                # values[0] is timestamp, values[1:13] are the 3x4 transformation matrix
                mat_3x4 = np.array(values[1:13]).reshape(3, 4)
                mat_4x4 = np.eye(4)
                mat_4x4[:3, :3] = mat_3x4[:, :3]
                mat_4x4[:3, 3] = mat_3x4[:, 3]
                poses.append(mat_4x4)
            else:
                logger.warning(f"Skipping line with {len(values)} values (expected 12, 13, or 16)")

    if len(poses) == 0:
        raise ValueError(f"No valid poses found in {file_path}")

    return np.array(poses)


def extract_positions(poses: np.ndarray) -> np.ndarray:
    """Extract translation components from 4x4 poses."""
    return poses[:, :3, 3]


def plot_trajectory_3d(
    positions: np.ndarray,
    poses: np.ndarray,
    output_path: Path,
    title: str = "Camera Trajectory",
    frame_step: int = 10,
):
    """
    Plot 3D trajectory with RGB coordinate frames every N steps.

    Args:
        positions: (N, 3) array of positions
        poses: (N, 4, 4) array of transformation matrices
        output_path: Path to save the plot
        title: Plot title
        frame_step: Show RGB frame every N steps
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Plot full trajectory
    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        'b-', linewidth=2, alpha=0.6, label='Trajectory'
    )

    # Mark start and end
    ax.scatter(
        [positions[0, 0]], [positions[0, 1]], [positions[0, 2]],
        c='green', s=100, marker='o', label='Start', zorder=10
    )
    ax.scatter(
        [positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
        c='red', s=100, marker='x', label='End', zorder=10
    )

    # Draw RGB frames every N steps
    def draw_frame_axes(origin, R, length=0.05):
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
                color=colors[i], linewidth=2, alpha=0.8
            )

    # Draw frames at regular intervals
    for i in range(0, len(poses), frame_step):
        origin = poses[i, :3, 3]
        R = poses[i, :3, :3]
        draw_frame_axes(origin, R, length=0.1)

    # Draw initial frame larger
    origin = poses[0, :3, 3]
    R = poses[0, :3, :3]
    draw_frame_axes(origin, R, length=0.15)

    # Labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'{title}\nRGB frames shown every {frame_step} steps', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    z_range = positions[:, 2].max() - positions[:, 2].min()
    max_range = max(x_range, y_range, z_range)

    x_center = (positions[:, 0].max() + positions[:, 0].min()) / 2
    y_center = (positions[:, 1].max() + positions[:, 1].min()) / 2
    z_center = (positions[:, 2].max() + positions[:, 2].min()) / 2

    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)

    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved 3D plot to {output_path}")
    plt.close()


def plot_trajectory_2d(
    positions: np.ndarray,
    output_path: Path,
    title: str = "Camera Trajectory",
):
    """
    Plot 2D projections (xy, xz, yz views).

    Args:
        positions: (N, 3) array of positions
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Define projections: (x_idx, y_idx, xlabel, ylabel)
    projections = [
        (0, 1, 'X', 'Y'),
        (0, 2, 'X', 'Z'),
        (1, 2, 'Y', 'Z'),
    ]

    for ax_idx, (x_idx, y_idx, xlabel, ylabel) in enumerate(projections):
        ax = axes[ax_idx]

        # Plot trajectory
        ax.plot(
            positions[:, x_idx],
            positions[:, y_idx],
            'b-', linewidth=2, alpha=0.6
        )

        # Mark start and end
        ax.scatter(
            [positions[0, x_idx]], [positions[0, y_idx]],
            c='green', s=100, marker='o', label='Start', zorder=10
        )
        ax.scatter(
            [positions[-1, x_idx]], [positions[-1, y_idx]],
            c='red', s=100, marker='x', label='End', zorder=10
        )

        # Mark every 10th point
        for i in range(0, len(positions), 10):
            ax.scatter(
                [positions[i, x_idx]], [positions[i, y_idx]],
                c='orange', s=20, marker='.', alpha=0.5
            )

        ax.set_xlabel(f'{xlabel} (m)')
        ax.set_ylabel(f'{ylabel} (m)')
        ax.set_title(f'{xlabel}-{ylabel} Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved 2D plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize camera trajectory from SLAM")
    parser.add_argument(
        "trajectory_file",
        type=str,
        help="Path to CameraTrajectory.txt file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Show RGB frame every N steps (default: 10)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Camera Trajectory",
        help="Plot title",
    )

    args = parser.parse_args()

    # Determine output directory
    input_path = Path(args.trajectory_file)
    if args.output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading trajectory from {args.trajectory_file}")

    # Load trajectory
    poses = load_camera_trajectory(args.trajectory_file)
    positions = extract_positions(poses)

    logger.info(f"Loaded {len(poses)} poses")
    logger.info(f"  X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    logger.info(f"  Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    logger.info(f"  Z range: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    # Generate output filename
    output_name = input_path.stem
    output_3d = output_dir / f"{output_name}_3d.png"
    output_2d = output_dir / f"{output_name}_2d.png"

    # Plot 3D
    plot_trajectory_3d(
        positions=positions,
        poses=poses,
        output_path=output_3d,
        title=args.title,
        frame_step=args.frame_step,
    )

    # Plot 2D
    plot_trajectory_2d(
        positions=positions,
        output_path=output_2d,
        title=args.title,
    )

    logger.info(f"Done! Outputs:")
    logger.info(f"  {output_3d}")
    logger.info(f"  {output_2d}")


if __name__ == "__main__":
    main()

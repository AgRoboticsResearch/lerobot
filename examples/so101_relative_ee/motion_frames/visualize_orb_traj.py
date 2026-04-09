#!/usr/bin/env python
"""Visualize ORB-SLAM camera trajectory with 2D projections and 3D plot.

Supports CameraTrajectory.txt and CameraTrajectoryTransformed.txt formats.
Creates a single figure with 4 subplots: XY, XZ, YZ projections + 3D trajectory.

Usage:
    python visualize_orb_traj.py <path_to_trajectory_file>
    python visualize_orb_traj.py /path/to/cameratraj.txt --video --fps 30
"""

import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Load trajectory
# ============================================================

def load_camera_trajectory(file_path: str) -> np.ndarray:
    """
    Load camera trajectory from file.

    Supports two formats:
    1. Original format: 12 floats per line (3x4 transformation matrix in row-major order)
    2. Transformed format: timestamp + 12 values (3x4 matrix in row-major order)

    Args:
        file_path: Path to trajectory file

    Returns:
        (N, 4, 4) array of 4x4 homogeneous transformation matrices
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            values = [float(x) for x in line.split()]
            if len(values) == 0:
                continue

            if len(values) == 12:
                # 3x4 transformation matrix (row-major)
                mat_3x4 = np.array(values).reshape(3, 4)
                mat_4x4 = np.eye(4)
                mat_4x4[:3, :3] = mat_3x4[:, :3]
                mat_4x4[:3, 3] = mat_3x4[:, 3]
                poses.append(mat_4x4)
            elif len(values) == 16:
                # 4x4 homogeneous transformation matrix
                poses.append(np.array(values).reshape(4, 4))
            elif len(values) == 13:
                # Timestamp + 12 values (3x4 matrix in row-major order)
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


# ============================================================
# Plot
# ============================================================

def plot_combined_trajectory(
    positions: np.ndarray,
    output_path: Path,
    title: str = "Camera Trajectory",
):
    """
    Create single figure with 5 subplots: XY, XZ, YZ projections + 3D trajectory + xyz vs steps.

    Args:
        positions: (N, 3) array of positions
        output_path: Path to save the plot
        title: Plot title
    """
    fig = plt.figure(figsize=(16, 12))

    # 2D projections (top row)
    projections = [
        (0, 1, 'X', 'Y'),  # XY
        (0, 2, 'X', 'Z'),  # XZ
        (1, 2, 'Y', 'Z'),  # YZ
    ]

    for idx, (x_idx, y_idx, xlabel, ylabel) in enumerate(projections):
        ax = fig.add_subplot(3, 3, idx + 1)

        # Plot trajectory
        ax.plot(
            positions[:, x_idx],
            positions[:, y_idx],
            'b-', linewidth=1.5, alpha=0.7
        )

        # Mark start and end
        ax.scatter(
            [positions[0, x_idx]], [positions[0, y_idx]],
            c='green', s=80, marker='o', label='Start', zorder=10
        )
        ax.scatter(
            [positions[-1, x_idx]], [positions[-1, y_idx]],
            c='red', s=80, marker='x', label='End', zorder=10
        )

        ax.set_xlabel(f'{xlabel} (m)')
        ax.set_ylabel(f'{ylabel} (m)')
        ax.set_title(f'{xlabel}-{ylabel} Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    # 3D trajectory (middle row, spans all 3 columns)
    ax3d = fig.add_subplot(3, 3, (4, 6), projection='3d')

    ax3d.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        'b-', linewidth=2, alpha=0.6, label='Trajectory'
    )
    ax3d.scatter(
        [positions[0, 0]], [positions[0, 1]], [positions[0, 2]],
        c='green', s=80, marker='o', label='Start', zorder=10
    )
    ax3d.scatter(
        [positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
        c='red', s=80, marker='x', label='End', zorder=10
    )

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory')
    ax3d.legend()
    ax3d.grid(True, alpha=0.3)

    # Equal aspect ratio for 3D
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    z_range = positions[:, 2].max() - positions[:, 2].min()
    max_range = max(x_range, y_range, z_range, 0.1)  # min 0.1m

    x_center = (positions[:, 0].max() + positions[:, 0].min()) / 2
    y_center = (positions[:, 1].max() + positions[:, 1].min()) / 2
    z_center = (positions[:, 2].max() + positions[:, 2].min()) / 2

    ax3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax3d.set_zlim(z_center - max_range/2, z_center + max_range/2)

    ax3d.view_init(elev=20, azim=45)

    # X, Y, Z values vs steps (bottom row, spans all 3 columns)
    ax_steps = fig.add_subplot(3, 3, (7, 9))
    steps = np.arange(len(positions))
    ax_steps.plot(steps, positions[:, 0], 'r-', linewidth=1.5, label='X')
    ax_steps.plot(steps, positions[:, 1], 'g-', linewidth=1.5, label='Y')
    ax_steps.plot(steps, positions[:, 2], 'b-', linewidth=1.5, label='Z')
    ax_steps.set_xlabel('Step')
    ax_steps.set_ylabel('Position (m)')
    ax_steps.set_title('X, Y, Z vs Step')
    ax_steps.legend()
    ax_steps.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


# ============================================================
# Video creation
# ============================================================

def create_color_video(image_dir: Path, output_path: Path, fps: int = 30):
    """
    Create h265 mp4 video from color*.png files in the directory.

    Args:
        image_dir: Directory containing color*.png files
        output_path: Path to save the video (color.mp4)
        fps: Frames per second
    """
    # Find all color*.png files, sorted naturally
    color_files = sorted(image_dir.glob("color*.png"), key=lambda p: p.name)

    if not color_files:
        logger.warning(f"No color*.png files found in {image_dir}")
        return

    logger.info(f"Found {len(color_files)} color*.png files")

    # Auto-detect filename pattern from first file (e.g. color_000000.png → color_%06d.png)
    import re
    first_name = color_files[0].name
    match = re.match(r"(color\D*)(\d+)(\.png)", first_name)
    if not match:
        logger.error(f"Cannot parse color filename pattern from {first_name}")
        return
    prefix, digits, suffix = match.group(1), match.group(2), match.group(3)
    n_digits = len(digits)
    start_idx = int(digits)
    pattern = f"{prefix}%0{n_digits}d{suffix}"

    # Use system ffmpeg (has libx265) over conda ffmpeg
    ffmpeg_bin = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"

    # Create video using ffmpeg with libx265
    cmd = [
        ffmpeg_bin,
        "-y",  # Overwrite output file
        "-framerate", str(fps),
        "-start_number", str(start_idx),
        "-i", str(image_dir / pattern),
        "-c:v", "libx265",  # h265 codec
        "-crf", "28",  # Quality (lower = better, 18-28 is typical)
        "-pix_fmt", "yuv420p",  # Compatibility
        str(output_path),
    ]

    logger.info(f"Creating video with: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Saved video to {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg: sudo apt install ffmpeg")
        raise


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ORB-SLAM camera trajectory")
    parser.add_argument(
        "trajectory_file",
        type=str,
        help="Path to CameraTrajectory.txt or CameraTrajectoryTransformed.txt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/zfei/code/lerobot/outputs/debug/motion_frames",
        help="Output directory (default: outputs/debug/motion_frames/)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Camera Trajectory",
        help="Plot title",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Create video from color*.png files in the trajectory directory",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video fps (default: 30)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Extract folder name from input path
    input_path = Path(args.trajectory_file)
    folder_name = input_path.parent.name

    # Create output directory
    output_dir = Path(args.output) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.png"

    logger.info(f"Loading trajectory from {args.trajectory_file}")

    # Load trajectory
    poses = load_camera_trajectory(args.trajectory_file)
    positions = extract_positions(poses)

    logger.info(f"Loaded {len(poses)} poses")
    logger.info(f"  X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m")
    logger.info(f"  Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m")
    logger.info(f"  Z range: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m")
    logger.info(f"Output folder: {folder_name}")

    # Plot
    plot_combined_trajectory(positions, output_path, args.title)

    # Create video from color*.png if requested
    if args.video:
        video_output = output_dir / "color.mp4"
        create_color_video(input_path.parent, video_output, args.fps)

    logger.info(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    main()

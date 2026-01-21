#!/usr/bin/env python3
"""
SO101 relative motion trajectory visualization using placo.

Loads predicted or ground truth trajectories from debug_relative_ee_inference.py
and visualizes them as point clouds in the robot workspace.

The trajectories are relative EE poses. We compute the absolute EE positions by:
1. Getting joint positions from dataset (or using default)
2. Computing FK to get base EE pose
3. Applying: T_absolute = T_base @ T_relative

Usage:
    python so101_rel_motion_debug.py --file_prefix /path/to/sample_0 --trajectory pred
    python so101_rel_motion_debug.py --file_prefix /path/to/sample_0 --trajectory gt
    python so101_rel_motion_debug.py --file_prefix /path/to/sample_0 --trajectory both
"""

import pinocchio
import placo
import numpy as np
import argparse
from pathlib import Path
from placo_utils.visualization import robot_viz, points_viz

# Import pose conversion from lerobot
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from lerobot.robots.so101_follower.relative_ee_processor import pose10d_to_mat

# Motor names for SO101
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Default home pose (in radians) - approximate from deploy script
HOME_POSE_RAD = np.array([
    -0.097,   # shoulder_pan (~ -5.54 deg)
    -2.0,     # shoulder_lift (~ -114.59 deg)
    1.4,      # elbow_flex (~ 80.44 deg)
    0.28,     # wrist_flex (~ 15.84 deg)
    -0.09,    # wrist_roll (~ -5.19 deg)
    0.61,     # gripper (~ 35.13 deg)
])

# Path to SO101 model directory (placo expects a directory with robot.urdf inside)
URDF_PATH = "../urdf/Simulation/SO101"


def load_trajectory(file_path: str) -> np.ndarray:
    """Load trajectory from txt file. Returns (T, 10) array."""
    data = np.loadtxt(file_path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_base_ee_pose(file_path: str) -> np.ndarray:
    """Load base EE pose from txt file. Returns (4, 4) matrix."""
    data = np.loadtxt(file_path)
    return data.reshape(4, 4)


def trajectory_to_positions(
    trajectory: np.ndarray,
    base_ee_T: np.ndarray,
) -> np.ndarray:
    """
    Convert relative EE trajectory to absolute positions.

    Following deploy_relative_ee_so101.py logic:
    - Each action is a relative transform from base EE pose
    - T_absolute = T_base @ T_relative

    Args:
        trajectory: (T, 10) array of [dx, dy, dz, r0, r1, r2, r3, r4, r5, gripper]
        base_ee_T: (4, 4) base EE pose at prediction time

    Returns:
        (T, 3) array of absolute EE positions
    """
    positions = []
    for i in range(trajectory.shape[0]):
        # Convert relative 10D pose to 4x4 matrix
        rel_T = pose10d_to_mat(trajectory[i, :9])
        # Apply from base pose: T_absolute = T_base @ T_relative
        abs_T = base_ee_T @ rel_T
        positions.append(abs_T[:3, 3])  # Extract translation
    return np.array(positions)


def get_base_ee_pose_from_joints(joint_positions: np.ndarray, robot: placo.RobotWrapper) -> np.ndarray:
    """
    Compute base EE pose using FK from joint positions.

    Args:
        joint_positions: (6,) array of joint positions in radians
        robot: Placo RobotWrapper

    Returns:
        (4, 4) EE pose transformation matrix
    """
    # Set joint positions directly on robot
    for i, name in enumerate(MOTOR_NAMES):
        robot.set_joint(name, joint_positions[i])

    # Update kinematics
    robot.update_kinematics()

    # Get gripper_frame_link pose
    T = robot.get_T_world_frame("gripper_frame_link")

    return T


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SO101 relative EE trajectories in placo"
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        required=True,
        help="File path prefix (e.g., /path/to/sample_0)",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="both",
        choices=["pred", "gt", "both"],
        help="Which trajectory to visualize: pred, gt, or both",
    )
    parser.add_argument(
        "--home_pose",
        type=float,
        nargs=6,
        default=list(HOME_POSE_RAD),
        help="Home joint positions in radians (6 values)",
    )

    args = parser.parse_args()

    # Load trajectories
    pred_traj = None
    gt_traj = None

    if args.trajectory in ["pred", "both"]:
        pred_path = f"{args.file_prefix}_pred.txt"
        if Path(pred_path).exists():
            pred_traj = load_trajectory(pred_path)
            print(f"Loaded predicted trajectory: {pred_path}")
            print(f"  Shape: {pred_traj.shape}")
        else:
            print(f"Warning: {pred_path} not found")

    if args.trajectory in ["gt", "both"]:
        gt_path = f"{args.file_prefix}_gt.txt"
        if Path(gt_path).exists():
            gt_traj = load_trajectory(gt_path)
            print(f"Loaded ground truth trajectory: {gt_path}")
            print(f"  Shape: {gt_traj.shape}")
        else:
            print(f"Warning: {gt_path} not found")

    if pred_traj is None and gt_traj is None:
        print("Error: No valid trajectories found!")
        return

    # ========================================================================
    # Setup placo robot and compute base EE pose
    # ========================================================================
    robot = placo.RobotWrapper(URDF_PATH, placo.Flags.ignore_collisions)
    solver = placo.KinematicsSolver(robot)
    solver.mask_fbase(True)

    # Check if base_ee.txt file exists
    base_ee_path = f"{args.file_prefix}_base_ee.txt"
    if Path(base_ee_path).exists():
        base_ee_T = load_base_ee_pose(base_ee_path)
        print(f"Loaded base EE pose from: {base_ee_path}")
        print(f"  Base EE position: {base_ee_T[:3, 3]}")
    else:
        # Compute base EE pose from home joint positions using FK
        home_joints = np.array(args.home_pose)
        print(f"Computing base EE pose from home joints...")
        print(f"  Home joints (rad): {home_joints}")

        base_ee_T = get_base_ee_pose_from_joints(home_joints, robot)
        print(f"  Computed base EE position: {base_ee_T[:3, 3]}")

    # ========================================================================
    # Convert relative trajectories to absolute positions
    # ========================================================================
    pred_positions = None
    gt_positions = None

    if pred_traj is not None:
        pred_positions = trajectory_to_positions(pred_traj, base_ee_T)
        print(f"Predicted positions: {pred_positions.shape}")

    if gt_traj is not None:
        gt_positions = trajectory_to_positions(gt_traj, base_ee_T)
        print(f"Ground truth positions: {gt_positions.shape}")

    # ========================================================================
    # Setup placo visualization
    # ========================================================================
    # Set robot to home pose
    joints_task = solver.add_joints_task()
    joints_task.set_joints({
        "shoulder_pan": args.home_pose[0],
        "shoulder_lift": args.home_pose[1],
        "elbow_flex": args.home_pose[2],
        "wrist_flex": args.home_pose[3],
        "wrist_roll": args.home_pose[4],
        "gripper": args.home_pose[5],
    })

    viz = robot_viz(robot)

    # Solve and display
    solver.solve(True)
    robot.update_kinematics()
    viz.display(robot.state.q)

    # ========================================================================
    # Static visualization: show trajectories as point clouds
    # ========================================================================
    print("\nDisplaying trajectories as point clouds...")
    print("Blue = Ground Truth, Red = Predicted")
    print("Press Ctrl+C to exit\n")

    # Show ground truth trajectory as blue points
    if gt_positions is not None:
        points_viz("gt_trajectory", gt_positions, color=0x0000ff)
        print(f"  Ground truth: {len(gt_positions)} points (blue)")

    # Show predicted trajectory as red points
    if pred_positions is not None:
        points_viz("pred_trajectory", pred_positions, color=0xff0000)
        print(f"  Predicted: {len(pred_positions)} points (red)")

    # Show base EE position
    base_pos = base_ee_T[:3, 3]
    print(f"  Base EE position: {base_pos} (green star)")

    # Keep the visualization alive
    print("\nVisualization ready. Open http://127.0.0.1:7000/static/ in your browser.")
    print("Press Ctrl+C to exit...\n")

    # Simple keep-alive loop
    try:
        from ischedule import schedule, run_loop

        @schedule(interval=1.0)
        def keep_alive():
            pass  # Just keep the visualization alive

        run_loop()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()

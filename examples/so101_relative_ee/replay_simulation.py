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
Replay saved action predictions in placo simulation.

This script loads action predictions saved by debug_relative_ee_animation.py
and executes them in a placo simulation with IK-based joint conversion.

=== Step 1: Generate action files ===

First, run debug_relative_ee_animation.py to generate the action files:

    python examples/so101_relative_ee/debug_relative_ee_animation.py \
        --dataset_repo_id red_strawberry_picking_260119_merged_ee \
        --dataset_root /mnt/ldata/sroi_lerobot/red_strawberry_picking_260119_merged_ee \
        --pretrained_path ./outputs/train/red_strawberry_picking_260119_merged_obs1_act/checkpoints/020000/pretrained_model \
        --episode_indices 0 \
        --save_video \
        --enable_ik_trajectory

This creates:
    outputs/debug/animation_relative_ee/<job_name>/episode_0/
    ├── actions_0000.txt
    ├── actions_0001.txt
    ├── ...
    └── episode_0.mp4

=== Step 2: Replay in simulation ===

    python examples/so101_relative_ee/replay_simulation.py \
        --actions_dir outputs/debug/animation_relative_ee/<job_name>/episode_0 \
        --n_action_steps 10 \
        --pause_between_chunks 1.0 \
        --step_delay 0.05

Open http://127.0.0.1:7000/static/ to see the visualization.

=== Key Arguments ===

    --actions_dir           Directory containing actions_XXXX.txt files
    --n_action_steps        Number of actions to execute from each chunk (default: 10)
    --pause_between_chunks  Pause duration between chunks in seconds (default: 1.0)
    --step_delay            Delay between action steps for visualization (default: 0.05)
    --start_chunk N         Start from chunk N (0-indexed)
    --num_chunks N          Process only N chunks (default: all)
    --follow_gt             Use GT trajectory from dataset instead of predictions (debug frame mismatch)
    --dataset_path          Path to dataset for GT trajectory (default: auto-detect)
    --no_viz                Disable placo visualization

=== What the script does ===

1. Loads all actions_XXXX.txt files from the directory
2. Pre-computes and visualizes the full ground truth trajectory (BLUE)
3. Starts the simulation at reset pose
4. For each chunk:
   - Gets predicted actions from the file
   - Visualizes predicted trajectory (RED)
   - Executes n_action_steps using IK to convert EE -> joints
   - Visualizes executed trajectory (GREEN)
   - Pauses for pause_between_chunks seconds
5. Repeats until all actions are executed

=== Visualization Colors ===

- BLUE:   Full ground truth trajectory (from dataset)
- RED:    Predicted trajectory (current chunk, default mode)
- YELLOW: GT chunk trajectory (follow_gt mode) - should match BLUE if frames aligned
- GREEN:  Executed trajectory (cumulative, from IK)

=== Action Chaining ===

The script follows the same action chaining logic as deploy_relative_ee_so101.py:
- Each action in a chunk is relative to chunk_base_pose (NOT chained sequentially)
- target_ee_pose = chunk_base_pose @ rel_T
- After n_action_steps, get new chunk_base_pose from simulation

=== Follow GT Mode (--follow_gt) ===

When --follow_gt is enabled, the script uses ground truth trajectory from the dataset
instead of model predictions. This is useful for debugging:

PURPOSE:
- Debug whether issues are from MODEL TRAINING or COORDINATE FRAME MISMATCH

EXPECTED OUTCOMES:
- If GREEN trajectory follows BLUE GT trajectory: Coordinate frames are aligned,
  problem is in MODEL TRAINING
- If GREEN trajectory goes wrong direction: Coordinate frame mismatch between
  dataset and simulation

HOW IT WORKS:
1. Load full GT absolute poses (position + rotation + gripper) from dataset
2. Convert to relative chunks using same SE(3) transformation as RelativeEE training:
   T_rel = inv(T_current) @ T_future
3. Use GT chunks as "predictions" instead of model outputs
4. Apply same DATASET_TO_PLaco_ROT transformation
5. Execute via IK and visualize

VISUALIZATION:
- BLUE: Full GT trajectory (from dataset, absolute)
- YELLOW: GT chunk trajectory (relative, transformed) - should align with BLUE
- GREEN: Executed trajectory (from IK) - should align with YELLOW if frames correct
"""

import logging
import time
from pathlib import Path

import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, points_viz

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower.relative_ee_processor import pose10d_to_mat
from lerobot.utils.utils import init_logging
from lerobot.datasets.relative_ee_dataset import pose_to_mat, mat_to_pose10d, rot6d_to_mat

# Motor names for SO101
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Default RESET pose (starting position)
# Matches red_strawberry_picking_260119_merged episode 0 initial pose
RESET_POSE_DEG = np.array([
    4.29,     # shoulder_pan
    -70.76,   # shoulder_lift
    100.00,   # elbow_flex
    35.85,    # wrist_flex
    -1.54,    # wrist_roll
    43.18,    # gripper
])

# Transformation matrix from dataset EE frame to placo gripper_frame_link
#
# Based on user analysis:
# - Dataset EE frame: X points DOWN, Z points FORWARD
# - Placo EE frame: Z points FORWARD (matches!), but need to verify other axes
# - Real robot moves UP and FORWARD, but simulation showed UP and BACKWARD
#
# The fix: negate Z axis in dataset actions to flip forward/backward direction
# This suggests the dataset Z axis might actually point BACKWARD (not forward as expected)
#
# Simple transformation: keep X and Y the same, negate Z to flip forward/backward
DATASET_TO_PLaco_ROT = np.array([
    [1.0,  0.0,  0.0],   # dataset X → placo X
    [0.0,  1.0,  0.0],   # dataset Y → placo Y
    [0.0,  0.0, -1.0],   # dataset Z → -placo Z (flip forward/backward)
], dtype=np.float32)

print("=" * 60)
print("COORDINATE FRAME TRANSFORMATION")
print("=" * 60)
print("Dataset EE frame: X points DOWN, Z points BACKWARD")
print("Placo EE frame:  Z points FORWARD")
print("Applying Z-axis flip to align forward direction...")
print("=" * 60)


def convert_gt_to_relative_chunks(
    gt_poses: np.ndarray,
    chunk_size: int = 100,
    stride: int = None,
) -> list[np.ndarray]:
    """
    Convert absolute GT trajectory to relative chunks matching RelativeEE training format.

    This function follows the same transformation as RelativeEEDataset:
    - For each action timestep in chunk: T_rel = inv(T_current) @ T_future
    - Convert to 10D pose using mat_to_pose10d
    - Append gripper value

    Args:
        gt_poses: (N, 7) array of absolute EE poses [x,y,z, rx,ry,rz, gripper]
            where rx,ry,rz are axis-angle rotation
        chunk_size: Size of each chunk (default: 100 from training config)
        stride: Step between chunk starts (default: chunk_size for training,
                but should match n_action_steps for debugging execution)

    Returns:
        List of (chunk_size, 10) arrays - relative actions for each chunk
        Each chunk has shape (chunk_size, 10) where:
        - [:3] is relative position [dx, dy, dz]
        - [3:9] is 6D rotation (first two rows of rotation matrix)
        - [9] is gripper value
    """
    chunks = []
    n_frames = len(gt_poses)

    if stride is None:
        stride = chunk_size  # Default: training mode (stride = chunk_size)

    # Create chunks with specified stride
    start_idx = 0
    while start_idx < n_frames:
        # Get current pose as reference
        current_pose_6d = gt_poses[start_idx, :6]  # [x, y, z, rx, ry, rz]
        T_current = pose_to_mat(current_pose_6d)

        # Collect relative actions for this chunk
        relative_actions = []
        for t in range(chunk_size):
            target_idx = start_idx + t

            if target_idx >= n_frames:
                # Pad with last pose
                target_idx = n_frames - 1

            # Get target pose
            target_pose_6d = gt_poses[target_idx, :6]
            gripper = gt_poses[target_idx, 6]

            # Convert to transformation matrix
            T_target = pose_to_mat(target_pose_6d)

            # Compute relative: T_rel = inv(T_current) @ T_target
            T_rel = np.linalg.inv(T_current) @ T_target

            # Convert to 10D pose (9D + gripper)
            pose_9d = mat_to_pose10d(T_rel)
            action_10d = np.concatenate([pose_9d, [gripper]])

            relative_actions.append(action_10d)

        # Stack into (chunk_size, 10) array
        chunk = np.stack(relative_actions, axis=0)
        chunks.append((start_idx, chunk))  # Store start_idx with chunk

        # Move to next chunk start
        start_idx += stride

    return chunks


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
                    # Parse [x, y, z] - handle both with and without brackets
                    value_clean = value.strip().strip("[]")
                    nums = value_clean.split(",")
                    result["header"][key] = np.array([float(n.strip()) for n in nums])
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
                # Handle both "x, y, z" and "[x, y, z]" formats
                value_clean = parts[1].strip().strip("[]")
                values = [float(n.strip()) for n in value_clean.split(",")]
                action_data.append(values)

    # Save last section
    if current_section == "predicted_actions":
        result["predicted_actions"] = np.array(action_data)
    elif current_section == "ground_truth_actions":
        result["ground_truth_actions"] = np.array(action_data)
    elif current_section == "ik_ee_positions":
        result["ik_ee_positions"] = np.array(action_data)

    return result


def transform_action_dataset_to_placo(rel_action_10d: np.ndarray) -> np.ndarray:
    """
    Transform a 10D relative action from dataset EE frame to placo EE frame.

    IMPORTANT: For relative SE(3) transformations, we use CONJUGATION on the full 4x4 matrix:
    - T_rel_in_new_frame = T_frame @ T_rel @ inv(T_frame)

    This is because relative actions describe transformations between frames,
    and conjugation preserves the geometric relationship when changing coordinates.

    Args:
        rel_action_10d: (10,) array [dx, dy, dz, rot6d_0..5, gripper] in dataset frame

    Returns:
        (10,) array in placo frame
    """
    # Extract position and rotation
    rel_pos = rel_action_10d[:3]  # [dx, dy, dz]
    rel_rot6d = rel_action_10d[3:9]  # [r00, r01, r02, r10, r11, r12]
    gripper = rel_action_10d[9]

    # Convert 6D rotation to proper 3x3 rotation matrix using rot6d_to_mat
    R_rel = rot6d_to_mat(rel_rot6d)

    # Build full 4x4 relative transformation matrix
    T_rel = np.eye(4)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3] = rel_pos

    # Build frame transformation matrix (rotation only, no translation)
    T_frame = np.eye(4)
    T_frame[:3, :3] = DATASET_TO_PLaco_ROT[:3, :3]

    # Conjugate: T_rel_in_new_frame = T_frame @ T_rel @ inv(T_frame)
    # Note: inv(T_frame) = T_frame.T when T_frame is pure rotation
    T_rel_placo = T_frame @ T_rel @ T_frame.T

    # Extract transformed position and 6D rotation
    rel_pos_placo = T_rel_placo[:3, 3]
    R_placo = T_rel_placo[:3, :3]

    # Extract 6D rotation from first two rows (UMI convention)
    r00_new = R_placo[0, 0]
    r01_new = R_placo[0, 1]
    r02_new = R_placo[0, 2]
    r10_new = R_placo[1, 0]
    r11_new = R_placo[1, 1]
    r12_new = R_placo[1, 2]

    return np.concatenate([
        rel_pos_placo,
        [r00_new, r01_new, r02_new, r10_new, r11_new, r12_new],
        [gripper],
    ])


class SimulatedSO101Robot:
    """Simulated SO101 robot using placo for kinematics and visualization."""

    def __init__(self, urdf_path: str, motor_names: list[str], enable_viz: bool = True):
        self.urdf_path = urdf_path
        self.motor_names = motor_names
        self.enable_viz = enable_viz

        # Setup placo robot
        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)

        # Setup visualization if enabled
        self.viz = None
        if enable_viz:
            self.viz = robot_viz(self.robot)

        # Current joint state (in degrees)
        self.current_joints = RESET_POSE_DEG.copy()

        # Initialize robot state
        self._update_robot_from_joints()

    def connect(self, calibrate: bool = False):
        """Simulated connection - just initialize."""
        print(f"Simulated robot connected")
        print(f"  URDF: {self.urdf_path}")
        print(f"  Motors: {self.motor_names}")
        if self.enable_viz:
            print(f"  Visualization: ENABLED")
            print(f"\nOpen http://127.0.0.1:7000/static/ in your browser to see the visualization!")

    def get_joints(self) -> np.ndarray:
        """Get current joint positions (in degrees)."""
        return self.current_joints.copy()

    def send_action(self, joints: np.ndarray):
        """
        Send joint positions to robot (simulated).

        Args:
            joints: Joint positions in degrees (array of length 6)
        """
        self.current_joints = joints.copy()
        self._update_robot_from_joints()

    def _update_robot_from_joints(self):
        """Update placo robot from current joint state."""
        # Convert degrees to radians for placo
        joints_rad = np.deg2rad(self.current_joints)

        for i, name in enumerate(self.motor_names):
            self.robot.set_joint(name, joints_rad[i])

        self.robot.update_kinematics()

        # Update visualization if enabled
        if self.viz is not None:
            self.viz.display(self.robot.state.q)

    def display_ee_frame(self, frame_name: str = "gripper_frame_link"):
        """Display the end-effector frame in visualization."""
        if self.viz is not None:
            robot_frame_viz(self.robot, frame_name)

    def visualize_trajectory(self, name: str, points: np.ndarray, color: int = 0xff0000):
        """
        Visualize a trajectory as a series of points.

        Args:
            name: Name for this trajectory (used as identifier)
            points: (N, 3) array of points
            color: RGB color as hex int (default: red)
        """
        if self.viz is not None and len(points) > 0:
            points_viz(name, points, color=color)

    def clear_trajectory(self, name: str):
        """Clear a trajectory visualization by name."""
        if self.viz is not None:
            # In placo, we can clear by adding empty points or using specific method
            points_viz(name, np.zeros((0, 3)), color=0x000000)

    def disconnect(self):
        """Simulated disconnection."""
        print("Simulated robot disconnected")


def main():
    init_logging()
    logger = logging.getLogger("replay_simulation")

    import argparse
    parser = argparse.ArgumentParser(
        description="Replay saved action predictions in placo simulation"
    )
    parser.add_argument(
        "--actions_dir",
        type=str,
        required=True,
        help="Directory containing actions_XXXX.txt files",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="urdf/Simulation/SO101/so101_new_calib.urdf",
        help="Path to SO101 URDF file for IK",
    )
    parser.add_argument(
        "--target_frame",
        type=str,
        default="gripper_frame_link",
        help="Name of end-effector frame in URDF",
    )
    parser.add_argument(
        "--reset_pose",
        type=float,
        nargs=6,
        default=list(RESET_POSE_DEG),
        metavar=("PAN", "LIFT", "ELBOW", "FLEX", "ROLL", "GRIPPER"),
        help="Reset pose in degrees",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=10,
        help="Number of actions to execute from each chunk",
    )
    parser.add_argument(
        "--pause_between_chunks",
        type=float,
        default=1.0,
        help="Pause duration between chunks (seconds)",
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.05,
        help="Delay between action steps (seconds) for visualization",
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
        "--gripper_lower",
        type=float,
        default=0.0,
        help="Gripper lower bound in degrees",
    )
    parser.add_argument(
        "--gripper_upper",
        type=float,
        default=100.0,
        help="Gripper upper bound in degrees",
    )
    parser.add_argument(
        "--start_chunk",
        type=int,
        default=0,
        help="Starting chunk index (0-indexed)",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="Number of chunks to execute (None = all)",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable placo visualization",
    )
    parser.add_argument(
        "--follow_gt",
        action="store_true",
        help="Use GT trajectory from dataset instead of predictions (debug frame/coordinate mismatch)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/ldata/sroi_lerobot/red_strawberry_picking_260119_merged_ee",
        help="Path to dataset for GT trajectory (used with --follow_gt)",
    )

    args = parser.parse_args()

    actions_dir = Path(args.actions_dir)
    if not actions_dir.exists():
        raise FileNotFoundError(f"Actions directory not found: {actions_dir}")

    # Find all actions_XXXX.txt files
    action_files = sorted(actions_dir.glob("actions_*.txt"))
    if not action_files:
        raise FileNotFoundError(f"No actions_*.txt files found in {actions_dir}")

    logger.info(f"Found {len(action_files)} action files in {actions_dir}")

    # Apply start_chunk and num_chunks filters
    start_chunk = args.start_chunk
    num_chunks = args.num_chunks if args.num_chunks is not None else len(action_files) - start_chunk
    end_chunk = min(start_chunk + num_chunks, len(action_files))

    action_files = action_files[start_chunk:end_chunk]
    logger.info(f"Processing chunks {start_chunk} to {end_chunk-1} ({len(action_files)} chunks)")

    # ========================================================================
    # Initialize Kinematics
    # ========================================================================
    logger.info("Initializing kinematics solver...")

    urdf_path = Path(args.urdf_path)
    if not urdf_path.exists():
        # Try to find URDF in parent directories
        for parent in [Path.cwd(), Path.cwd().parent]:
            candidate = parent / args.urdf_path
            if candidate.exists():
                urdf_path = candidate
                break

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found at {args.urdf_path}")

    # For RobotKinematics, we need the actual URDF file path
    if urdf_path.is_dir():
        urdf_file = urdf_path / "robot.urdf"
    else:
        urdf_file = urdf_path

    kinematics = RobotKinematics(
        urdf_path=str(urdf_file),
        target_frame_name=args.target_frame,
        joint_names=MOTOR_NAMES,
    )
    logger.info(f"URDF loaded: {urdf_file}")

    # ========================================================================
    # Initialize Simulation Robot
    # ========================================================================
    logger.info("Initializing simulated SO101 robot...")

    reset_pose = np.array(args.reset_pose, dtype=np.float64)

    robot = SimulatedSO101Robot(str(urdf_path), MOTOR_NAMES, enable_viz=not args.no_viz)
    robot.connect(calibrate=False)

    # Set robot to reset pose
    robot.send_action(reset_pose)
    sim_joints = reset_pose.copy()

    # Get initial EE pose from simulation
    chunk_base_pose = kinematics.forward_kinematics(sim_joints)
    logger.info(f"Robot initialized at reset pose: {reset_pose}")
    logger.info(f"Initial EE position: {chunk_base_pose[:3, 3]}")

    # Gripper bounds for action conversion
    gripper_lower = args.gripper_lower
    gripper_upper = args.gripper_upper

    # Track all executed positions (cumulative across chunks)
    all_executed_positions = []

    # Add initial position
    initial_ee = kinematics.forward_kinematics(sim_joints)[:3, 3]
    all_executed_positions.append(initial_ee.copy())

    # ========================================================================
    # Load GT trajectory directly from dataset
    # ========================================================================
    logger.info("Loading ground truth trajectory from dataset...")

    # Load the original dataset to get GT EE positions
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Use the dataset path from args or default
    dataset_path = args.dataset_path

    if not Path(dataset_path).exists():
        logger.warning(f"Could not find dataset at {dataset_path}, skipping GT trajectory visualization")
        all_gt_positions = np.array([])
        all_gt_poses_with_rot = np.array([])
    else:
        dataset = LeRobotDataset(repo_id=dataset_path, root=dataset_path)

        # Collect all EE positions AND full poses from episode 0
        all_gt_positions = []
        all_gt_poses_with_rot = []  # Full 7D: [x, y, z, rx, ry, rz, gripper]
        current_ep = None
        for i in range(len(dataset)):
            frame = dataset[i]
            ep_idx = frame['episode_index'].item()

            # Stop if we've moved past episode 0
            if current_ep is None:
                current_ep = ep_idx
            elif ep_idx != current_ep:
                break

            # Only process episode 0
            if ep_idx == 0:
                # Action format: [x, y, z, rx, ry, rz, gripper] - EE poses!
                action = frame['action'].numpy()
                if action.ndim == 1:
                    ee_pos = action[:3]
                    full_pose = action.copy()
                else:
                    ee_pos = action[0, :3]
                    full_pose = action[0].copy()
                all_gt_positions.append(ee_pos.copy())
                all_gt_poses_with_rot.append(full_pose.copy())

        all_gt_positions = np.array(all_gt_positions)
        all_gt_poses_with_rot = np.array(all_gt_poses_with_rot)
        logger.info(f"  GT trajectory: {len(all_gt_positions)} points from dataset")
        logger.info(f"  GT start: {all_gt_positions[0]}")
        logger.info(f"  GT end: {all_gt_positions[-1]}")

        # Visualize full GT trajectory in blue
        # Note: GT trajectory from _ee dataset is already in placo/URDF frame
        # (computed via FK from the same robot, so no transformation needed)
        if not args.no_viz and len(all_gt_positions) > 0:
            robot.visualize_trajectory("ground_truth", all_gt_positions, color=0x0000ff)
            logger.info("  GT trajectory visualized in BLUE")

    # ========================================================================
    # Convert GT to relative chunks if --follow_gt is enabled
    # ========================================================================
    gt_chunks = []
    if args.follow_gt:
        if len(all_gt_poses_with_rot) == 0:
            logger.error("--follow_gt enabled but no GT poses loaded. Check --dataset_path.")
            return

        logger.info("=" * 60)
        logger.info("FOLLOW_GT MODE: Using GT trajectory as predictions")
        logger.info("=" * 60)
        logger.info("Converting GT trajectory to relative chunks...")

        # Get chunk_size from first action file (or use default 100)
        chunk_size = 100  # Default from training config
        if action_files:
            # Try to read chunk size from first action file
            try:
                first_data = read_actions_file(action_files[0])
                if first_data["predicted_actions"] is not None:
                    chunk_size = first_data["predicted_actions"].shape[0]
            except Exception:
                pass

        # Important: stride = n_action_steps - 1 because action[0] is identity (no movement)
        # After executing n_action_steps, robot moves n_action_steps - 1 frames
        stride_for_chunks = args.n_action_steps - 1 if args.n_action_steps > 1 else 1
        gt_chunks = convert_gt_to_relative_chunks(
            all_gt_poses_with_rot,
            chunk_size=chunk_size,
            stride=stride_for_chunks
        )
        # gt_chunks is now list of (start_idx, chunk) tuples
        logger.info(f"  Created {len(gt_chunks)} GT chunks (chunk_size={chunk_size}, stride={stride_for_chunks})")
        logger.info(f"  GT poses shape: {all_gt_poses_with_rot.shape}")
        if len(gt_chunks) > 0:
            logger.info(f"  First chunk: start_idx={gt_chunks[0][0]}, shape={gt_chunks[0][1].shape}")
        logger.info("  GT chunks will be used instead of model predictions")
        logger.info("=" * 60)

    # ========================================================================
    # Load and execute actions
    # ========================================================================
    logger.info("Starting action replay...")
    logger.info(f"  Actions per chunk: {args.n_action_steps}")
    logger.info(f"  Pause between chunks: {args.pause_between_chunks}s")
    logger.info(f"  Step delay: {args.step_delay}s")
    if args.follow_gt:
        logger.info(f"  Mode: FOLLOW_GT (using GT trajectory as predictions)")

    total_steps_executed = 0

    for chunk_loop_idx, action_file in enumerate(action_files):
        chunk_idx = start_chunk + chunk_loop_idx

        logger.info(f"\n{'='*60}")
        if args.follow_gt:
            # Use GT chunk instead of loading from file
            if chunk_idx >= len(gt_chunks):
                logger.info(f"No more GT chunks available (chunk {chunk_idx} >= {len(gt_chunks)})")
                break

            start_idx_gt, pred_actions = gt_chunks[chunk_idx]
            logger.info(f"Processing GT chunk {chunk_idx} (start_idx={start_idx_gt})")
        else:
            # Load actions from file
            logger.info(f"Processing chunk {chunk_idx}: {action_file.name}")
            data = read_actions_file(action_file)
            pred_actions = data["predicted_actions"]

            if pred_actions is None:
                logger.warning(f"No predicted actions found in {action_file.name}")
                continue

        logger.info(f"{'='*60}")

        chunk_size = pred_actions.shape[0]
        n_steps = min(args.n_action_steps, chunk_size)

        logger.info(f"  Chunk size: {chunk_size}")
        logger.info(f"  Executing: {n_steps} steps")

        # Get current EE pose as chunk base (in placo frame from FK)
        chunk_base_pose = kinematics.forward_kinematics(sim_joints)
        logger.info(f"  Chunk base EE: {chunk_base_pose[:3, 3]}")

        # Compute predicted trajectory by applying relative actions
        # Note: Both GT chunks and model predictions are in placo frame
        # (computed via FK from the same robot, no transformation needed)
        pred_ee_positions = []
        for rel_action in pred_actions:
            rel_T = pose10d_to_mat(rel_action[:9])
            target_ee_pose = chunk_base_pose @ rel_T
            pred_ee_positions.append(target_ee_pose[:3, 3].copy())
        pred_ee_positions = np.array(pred_ee_positions)

        # Visualize predicted trajectory
        # - RED: Model predictions (default mode)
        # - YELLOW: GT chunks (follow_gt mode) - should match BLUE GT trajectory
        if not args.no_viz:
            if args.follow_gt:
                robot.visualize_trajectory("gt_chunk", pred_ee_positions, color=0xffff00)
            else:
                robot.visualize_trajectory("predicted", pred_ee_positions, color=0xff0000)

        # Execute n_action_steps
        for step_idx in range(n_steps):
            rel_action = pred_actions[step_idx]

            # Extract gripper and compute joint value
            gripper_value = rel_action[9]  # Value in [0, 1]
            gripper_deg = gripper_lower + gripper_value * (gripper_upper - gripper_lower)

            # Convert relative action to transformation matrix and apply
            rel_T = pose10d_to_mat(rel_action[:9])
            target_ee_pose = chunk_base_pose @ rel_T
            target_ee_pos = target_ee_pose[:3, 3]

            # Check bounds
            ee_bounds_min = np.array(args.ee_bounds_min)
            ee_bounds_max = np.array(args.ee_bounds_max)

            if np.any(target_ee_pos < ee_bounds_min) or np.any(target_ee_pos > ee_bounds_max):
                logger.warning(f"    Step {step_idx}: EE position out of bounds: {target_ee_pos}")
                logger.warning(f"    Bounds: min={ee_bounds_min}, max={ee_bounds_max}")
                # Skip this step or clamp to bounds?
                # For now, continue with IK (it will fail gracefully)

            # Solve IK
            try:
                joints = kinematics.inverse_kinematics(
                    sim_joints,
                    target_ee_pose,
                    position_weight=1.0,
                    orientation_weight=0.01,
                )
            except Exception as e:
                logger.warning(f"    Step {step_idx}: IK failed: {e}")
                continue

            # Set gripper
            joints[-1] = gripper_deg

            # Send to simulation
            robot.send_action(joints)
            sim_joints = joints

            # Update EE position for visualization
            current_ee_pos = kinematics.forward_kinematics(sim_joints)[:3, 3]
            all_executed_positions.append(current_ee_pos.copy())

            # Display EE frame
            if not args.no_viz:
                robot.display_ee_frame(args.target_frame)

            # Log progress
            if step_idx == 0 or step_idx == n_steps - 1:
                logger.info(f"    Step {step_idx}: EE pos {current_ee_pos}")

            total_steps_executed += 1

            # Delay between steps
            if args.step_delay > 0:
                time.sleep(args.step_delay)

        # Visualize cumulative executed trajectory (green) - updates each chunk
        if not args.no_viz and len(all_executed_positions) > 0:
            robot.visualize_trajectory("executed", np.array(all_executed_positions), color=0x00ff00)

        # Pause between chunks
        if args.pause_between_chunks > 0:
            logger.info(f"  Pausing for {args.pause_between_chunks}s...")
            time.sleep(args.pause_between_chunks)

    # ========================================================================
    # Done
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Replay complete!")
    logger.info(f"  Total steps executed: {total_steps_executed}")
    logger.info(f"  Total chunks processed: {len(action_files)}")
    logger.info(f"{'='*60}")

    robot.disconnect()
    logger.info("Done!")


if __name__ == "__main__":
    main()

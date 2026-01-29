# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Converter for transforming EE-only datasets to joint-based datasets using IK."""

import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from datasets import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics

# Default reset pose for SO101 in degrees
DEFAULT_RESET_POSE_DEG = np.array(
    [
        -8.00,  # shoulder_pan
        -62.73,  # shoulder_lift
        65.05,  # elbow_flex
        0.86,  # wrist_flex
        -2.55,  # wrist_roll
        88.91,  # gripper
    ],
    dtype=np.float64,
)

DEFAULT_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]




def write_parquet_with_fixed_size_arrays(
    data: list[dict],
    output_path,
    state_dim: int = 6,
) -> None:
    """
    Write data to parquet using datasets.Dataset for compatibility with LeRobotDataset.

    LeRobot datasets use variable-length lists (not fixed_size_list) with float32.
    We specify features explicitly to ensure correct dtypes.

    Note: 'task' string column is NOT included in parquet - only task_index.
    Task names are stored in meta/tasks.parquet separately.
    """
    from datasets import Dataset, Features, Sequence, Value

    # Convert data - using tolist() creates variable-length lists
    # Note: 'task' is NOT included in parquet - only task_index
    data_dict = {
        "observation.state": [frame["observation.state"].tolist() for frame in data],
        "action": [frame["action"].tolist() for frame in data],
        "episode_index": [frame["episode_index"] for frame in data],
        "frame_index": [frame["frame_index"] for frame in data],
        "timestamp": [frame["timestamp"] for frame in data],
        "task_index": [frame["task_index"] for frame in data],
        "index": [frame["index"] for frame in data],
    }

    # Specify features explicitly to ensure float32 dtype
    # This matches the schema expected by LeRobotDataset
    features = Features({
        "observation.state": Sequence(Value("float32")),
        "action": Sequence(Value("float32")),
        "episode_index": Value("int64"),
        "frame_index": Value("int64"),
        "timestamp": Value("float32"),
        "task_index": Value("int64"),
        "index": Value("int64"),
    })

    dataset = Dataset.from_dict(data_dict, features=features)
    dataset.to_parquet(str(output_path))


def ee_pose_to_matrix(ee_pose_7d: np.ndarray) -> np.ndarray:
    """
    Convert 7D EE pose [x, y, z, wx, wy, wz, gripper] to 4x4 transformation matrix.

    Args:
        ee_pose_7d: 7D array where:
            - [0:3] = position [x, y, z] in meters
            - [3:6] = axis-angle rotation [wx, wy, wz]
            - [6] = gripper position (not used for transform)

    Returns:
        4x4 homogeneous transformation matrix
    """
    pos = ee_pose_7d[:3]  # [x, y, z]
    rotvec = ee_pose_7d[3:6]  # [wx, wy, wz] - axis-angle

    # Convert axis-angle to 3x3 rotation matrix
    rotmat = Rotation.from_rotvec(rotvec).as_matrix()

    # Build 4x4 homogeneous matrix
    tf_matrix = np.eye(4)
    tf_matrix[:3, :3] = rotmat
    tf_matrix[:3, 3] = pos

    return tf_matrix


def verify_ik_solution(
    kinematics: RobotKinematics,
    joint_pos: np.ndarray,
    desired_ee_pose: np.ndarray,
    pos_tolerance: float = 0.005,  # 5mm position tolerance
    rot_tolerance: float = 0.1,  # ~5.7 degrees rotation tolerance
) -> tuple[bool, float, float]:
    """
    Verify IK solution by computing FK and comparing to desired EE pose.

    Args:
        kinematics: RobotKinematics instance
        joint_pos: Joint positions to verify (in degrees)
        desired_ee_pose: Desired 4x4 transformation matrix
        pos_tolerance: Maximum acceptable position error in meters
        rot_tolerance: Maximum acceptable rotation error in radians

    Returns:
        (is_valid, position_error, rotation_error)
    """
    # Compute FK on the IK result
    actual_ee_pose = kinematics.forward_kinematics(joint_pos)

    # Position error (L2 norm)
    pos_error = np.linalg.norm(actual_ee_pose[:3, 3] - desired_ee_pose[:3, 3])

    # Rotation error (using geodesic distance on SO(3))
    # Error = ||log(r_desired.T @ r_actual)||
    r_desired = desired_ee_pose[:3, :3]
    r_actual = actual_ee_pose[:3, :3]
    r_rel = r_desired.T @ r_actual
    rot_error = np.linalg.norm(Rotation.from_matrix(r_rel).as_rotvec())

    is_valid = (pos_error < pos_tolerance) and (rot_error < rot_tolerance)
    return is_valid, pos_error, rot_error


def solve_ik_with_fallback(
    kinematics: RobotKinematics,
    ee_pose_7d: np.ndarray,
    initial_guess: np.ndarray,
    previous_valid_joint: np.ndarray | None,
    position_weight: float = 1.0,
    orientation_weight: float = 0.01,
    pos_tolerance: float = 0.005,
    rot_tolerance: float = 0.1,
) -> tuple[np.ndarray, bool, float, float]:
    """
    Solve IK with verification and fallback.

    Strategy:
    1. Convert EE pose to transformation matrix
    2. Try IK with initial guess
    3. Verify solution with FK
    4. If verification fails, retry with previous valid joint as guess
    5. If still fails, return previous valid joint

    Args:
        kinematics: RobotKinematics instance
        ee_pose_7d: 7D end-effector pose [x, y, z, wx, wy, wz, gripper]
        initial_guess: Initial joint position guess for IK (in degrees)
        previous_valid_joint: Previous valid joint position for fallback (in degrees)
        position_weight: Weight for position constraint in IK
        orientation_weight: Weight for orientation constraint in IK
        pos_tolerance: Position tolerance for verification (meters)
        rot_tolerance: Rotation tolerance for verification (radians)

    Returns:
        (joint_pos, success, pos_error, rot_error)
    """
    # Step 1: Convert to transformation matrix
    tf_desired = ee_pose_to_matrix(ee_pose_7d)

    # Step 2: First IK attempt (with initial guess)
    joint_pos = kinematics.inverse_kinematics(
        current_joint_pos=initial_guess,
        desired_ee_pose=tf_desired,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
    )

    # Step 3: Verify solution
    is_valid, pos_err, rot_err = verify_ik_solution(
        kinematics, joint_pos, tf_desired, pos_tolerance, rot_tolerance
    )

    if is_valid:
        return joint_pos, True, pos_err, rot_err

    # Step 4: Retry with previous valid joint as guess (if available)
    if previous_valid_joint is not None:
        joint_pos_retry = kinematics.inverse_kinematics(
            current_joint_pos=previous_valid_joint,
            desired_ee_pose=tf_desired,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )
        is_valid_retry, pos_err_retry, rot_err_retry = verify_ik_solution(
            kinematics, joint_pos_retry, tf_desired, pos_tolerance, rot_tolerance
        )
        if is_valid_retry:
            return joint_pos_retry, True, pos_err_retry, rot_err_retry

    # Step 5: All attempts failed - return previous valid joint (or initial guess)
    fallback = previous_valid_joint if previous_valid_joint is not None else initial_guess
    return fallback, False, pos_err, rot_err


def convert_episode(
    ee_dataset: LeRobotDataset,
    episode_idx: int,
    kinematics: RobotKinematics,
    reset_pose: np.ndarray,
    pos_tolerance: float = 0.005,
    rot_tolerance: float = 0.1,
    position_weight: float = 1.0,
    orientation_weight: float = 0.01,
) -> dict:
    """
    Convert a single episode from EE space to joint space.

    Algorithm:
    1. Get episode start/end indices
    2. Get first EE pose and compute reset pose FK
    3. Compute transformation to align trajectory to reset pose
    4. For each frame:
       a. Extract EE pose (7D) from observation.state
       b. Align EE pose to reset pose reference frame
       c. Solve IK with tracking (use previous valid joint as initial guess)
       d. Verify solution
       e. If verification fails, use previous valid joint
       f. Store joint position + copy gripper from EE data
    5. Return converted data

    Args:
        ee_dataset: Input EE-only dataset
        episode_idx: Index of episode to convert
        kinematics: RobotKinematics instance
        reset_pose: Reset pose in degrees (used as IK initial guess and reference)
        pos_tolerance: Position tolerance for IK verification (meters)
        rot_tolerance: Rotation tolerance for IK verification (radians)
        position_weight: Weight for position constraint in IK
        orientation_weight: Weight for orientation constraint in IK

    Returns:
        Dictionary with converted joint positions and statistics
    """
    # Get episode boundaries
    episode = ee_dataset.meta.episodes[episode_idx]
    start_idx = episode["dataset_from_index"]
    end_idx = episode["dataset_to_index"]

    joint_positions = []
    ik_successes = []
    pos_errors = []
    rot_errors = []

    # Get first EE pose to compute alignment transform
    first_frame = ee_dataset[start_idx]
    first_ee_pose = first_frame["observation.state"].cpu().numpy()  # (7,) [x, y, z, wx, wy, wz, gripper]
    first_ee_tf = ee_pose_to_matrix(first_ee_pose[:6])  # Exclude gripper

    # Get FK of reset pose
    reset_fk = kinematics.forward_kinematics(reset_pose)

    # Compute alignment transform: T_align = T_reset @ T_first_ee^(-1)
    # This transforms the EE trajectory to start from reset pose
    first_ee_tf_inv = np.linalg.inv(first_ee_tf)
    alignment_tf = reset_fk @ first_ee_tf_inv

    # Track last valid joint for tracking (start with reset_pose)
    last_valid_joint = reset_pose

    # Note: end_idx is inclusive in LeRobot, so we use end_idx + 1 for range
    for frame_idx in range(start_idx, end_idx + 1):
        # Get EE pose from dataset
        frame = ee_dataset[frame_idx]
        ee_pose = frame["observation.state"].cpu().numpy()  # (7,) [x, y, z, wx, wy, wz, gripper]

        # Align EE pose to reset pose reference frame
        ee_tf = ee_pose_to_matrix(ee_pose[:6])
        aligned_ee_tf = alignment_tf @ ee_tf

        # Convert back to 7D pose for IK
        aligned_pos = aligned_ee_tf[:3, 3]
        aligned_rotvec = Rotation.from_matrix(aligned_ee_tf[:3, :3]).as_rotvec()
        aligned_ee_pose_7d = np.concatenate([aligned_pos, aligned_rotvec, [ee_pose[6]]])  # Keep original gripper

        # Solve IK using previous valid joint as initial guess (tracking)
        # Note: previous_valid_joint=None means fallback to initial_guess (last_valid_joint)
        joint_pos, success, pos_err, rot_err = solve_ik_with_fallback(
            kinematics=kinematics,
            ee_pose_7d=aligned_ee_pose_7d,
            initial_guess=last_valid_joint,
            previous_valid_joint=None,  # Fallback to initial_guess (last_valid_joint) on failure
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            pos_tolerance=pos_tolerance,
            rot_tolerance=rot_tolerance,
        )

        # Store result
        joint_positions.append(joint_pos)
        ik_successes.append(success)
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)

        # Update last valid joint for tracking
        if success:
            last_valid_joint = joint_pos

    # Compute statistics - only average successful frames for meaningful error
    success_rate = np.mean(ik_successes)
    # For errors, only consider successful frames
    successful_pos_errors = [pe for pe, s in zip(pos_errors, ik_successes) if s]
    successful_rot_errors = [re for re, s in zip(rot_errors, ik_successes) if s]
    avg_pos_error = np.mean(successful_pos_errors) if successful_pos_errors else 999.0
    avg_rot_error = np.mean(successful_rot_errors) if successful_rot_errors else 999.0

    return {
        "joint_positions": np.stack(joint_positions),  # (num_frames, 6)
        "ik_successes": ik_successes,
        "success_rate": success_rate,
        "avg_pos_error": avg_pos_error,
        "avg_rot_error": avg_rot_error,
    }


class EEToJointDatasetConverter:
    """
    Converts EE-only datasets to joint-based datasets using inverse kinematics.

    This converter takes a dataset with end-effector poses (7D: position + axis-angle rotation + gripper)
    and transforms it into a dataset with joint positions (6D) using robot kinematics.

    The IK process:
    1. Converts EE pose to 4x4 transformation matrix
    2. Solves IK for each frame (using RESET pose as initial guess)
    3. Verifies solution with forward kinematics
    4. Falls back to previous valid joint if verification fails
    5. Copies video frames unchanged

    Example:
        converter = EEToJointDatasetConverter(
            ee_dataset_path="/path/to/ee_dataset",
            output_repo_id="output_joint_dataset",
        )
        output_dataset = converter.convert_all()
    """

    def __init__(
        self,
        ee_dataset_path: str,
        output_repo_id: str,
        root: str | Path | None = None,
        urdf_path: str = "urdf/Simulation/SO101/so101_new_calib.urdf",
        reset_pose_deg: np.ndarray = DEFAULT_RESET_POSE_DEG,
        target_frame: str = "gripper_frame_link",
        joint_names: list[str] = DEFAULT_JOINT_NAMES,
        pos_tolerance: float = 0.02,  # 20mm - more reasonable for "always from reset" IK
        rot_tolerance: float = 0.3,  # ~17 deg
        position_weight: float = 1.0,  # Balanced weights for better IK convergence
        orientation_weight: float = 1.0,
    ):
        """
        Initialize the converter.

        Args:
            ee_dataset_path: Path to input EE-only dataset
            output_repo_id: Repo ID for output joint-based dataset
            root: Root directory for output dataset (default: HF_LEROBOT_HOME/output_repo_id)
            urdf_path: Path to robot URDF file for kinematics
            reset_pose_deg: Reset joint pose in degrees, used as IK initial guess
            target_frame: Name of end-effector frame in URDF
            joint_names: Names of joints in output dataset
            pos_tolerance: Position tolerance for IK verification (meters)
            rot_tolerance: Rotation tolerance for IK verification (radians)
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK
        """
        self.ee_dataset = LeRobotDataset(ee_dataset_path)
        self.output_repo_id = output_repo_id
        # Use float64 for IK compatibility with placo
        self.reset_pose = np.array(reset_pose_deg, dtype=np.float64)
        self.pos_tol = pos_tolerance
        self.rot_tol = rot_tolerance
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

        # Initialize kinematics
        self.kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )

        # Determine video/image keys from input dataset
        self.video_keys = self.ee_dataset.meta.video_keys
        self.image_keys = self.ee_dataset.meta.image_keys

        # Get input camera feature to copy
        camera_feature = None
        for key in self.ee_dataset.meta.features:
            if key.startswith("observation.images") or key.startswith("observation.video"):
                camera_feature = self.ee_dataset.meta.features[key]
                self.camera_key = key
                break

        if camera_feature is None:
            raise ValueError("No camera key found in input dataset")

        # Output features: 6D joint positions + metadata columns
        # Note: LeRobot requires ALL columns in info.json features, including metadata
        self.output_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": joint_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": joint_names,
            },
            "timestamp": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            self.camera_key: camera_feature,
        }

        self.root = Path(root) if root is not None else self.ee_dataset.meta.root.parent / output_repo_id

    def convert_all(
        self,
        video_backend: str | None = None,
        copy_videos: bool = True,
        save_debug_viz: bool = False,
        debug_output_dir: str | Path = "./outputs/debug/ee_to_joint_converter",
        num_episodes_to_viz: int = 10,
        episode_indices: list[int] | None = None,
    ) -> LeRobotDataset:
        """
        Convert all episodes and save as new LeRobotDataset.

        Args:
            video_backend: Video backend to use for decoding (not used if copy_videos=True)
            copy_videos: If True, copy video files directly instead of re-encoding (much faster)
            save_debug_viz: If True, save trajectory comparison visualizations for debugging
            debug_output_dir: Directory to save debug visualizations (default: ./outputs/debug/ee_to_joint_converter)
            num_episodes_to_viz: Number of episodes to visualize (default: 10, use -1 for all)
            episode_indices: List of episode indices to convert (default: None = convert all episodes)

        Returns:
            LeRobotDataset with joint-based observations and actions
        """
        # Create output dataset
        output_root = Path(self.root)
        output_root.mkdir(parents=True, exist_ok=True)

        # Create the directory structure
        (output_root / "data").mkdir(exist_ok=True)
        (output_root / "videos" / self.camera_key).mkdir(parents=True, exist_ok=True)
        (output_root / "meta").mkdir(exist_ok=True)

        # Copy videos directly if requested
        input_videos_path = self.ee_dataset.meta.root / "videos" / self.camera_key
        output_videos_path = output_root / "videos" / self.camera_key

        if copy_videos and input_videos_path.exists():
            print("Copying video files directly (no re-encoding)...")
            # Copy all chunks
            for chunk_dir in input_videos_path.iterdir():
                if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
                    output_chunk_dir = output_videos_path / chunk_dir.name
                    output_chunk_dir.mkdir(exist_ok=True)
                    for video_file in chunk_dir.glob("*.mp4"):
                        shutil.copy2(video_file, output_chunk_dir / video_file.name)
            print("Video files copied.")

        total_episodes = self.ee_dataset.meta.total_episodes
        # Determine which episodes to convert
        if episode_indices is not None:
            # Validate episode indices
            invalid_indices = [i for i in episode_indices if i < 0 or i >= total_episodes]
            if invalid_indices:
                raise ValueError(f"Invalid episode indices: {invalid_indices}. Valid range: 0-{total_episodes-1}")
            episodes_to_convert = sorted(set(episode_indices))  # Deduplicate and sort
            print(f"Converting {len(episodes_to_convert)} specific episodes: {episodes_to_convert}")
        else:
            episodes_to_convert = list(range(total_episodes))

        chunks_size = self.ee_dataset.meta.info.get("chunks_size", 1000)

        # Track current chunk and global frame counter for index column
        current_chunk = 0
        current_file = 0
        frames_in_current_file = 0
        frames_written_total = 0  # Global counter for index computation
        chunk_data = []

        # Collect all joint positions for stats computation
        all_joint_positions = []
        # Collect per-episode joint positions for debug visualization
        all_episode_joint_positions = []

        # Create chunk directory
        chunk_dir = output_root / "data" / f"chunk-{current_chunk:03d}"
        chunk_dir.mkdir(exist_ok=True)

        for ep_idx in episodes_to_convert:
            # Convert episode
            result = convert_episode(
                ee_dataset=self.ee_dataset,
                episode_idx=ep_idx,
                kinematics=self.kinematics,
                reset_pose=self.reset_pose,
                pos_tolerance=self.pos_tol,
                rot_tolerance=self.rot_tol,
                position_weight=self.position_weight,
                orientation_weight=self.orientation_weight,
            )

            # Collect joint positions for stats
            all_joint_positions.append(result["joint_positions"].astype(np.float32))
            # Collect per-episode joint positions for visualization (keep as float64 for kinematics)
            all_episode_joint_positions.append(result["joint_positions"])

            # Get episode data from input
            episode = self.ee_dataset.meta.episodes[ep_idx]
            start_idx = episode["dataset_from_index"]
            end_idx = episode["dataset_to_index"]

            # Note: end_idx is inclusive in LeRobot, so we use end_idx + 1 for range
            for frame_idx in range(start_idx, end_idx + 1):
                rel_idx = frame_idx - start_idx
                input_frame = self.ee_dataset[frame_idx]

                # Get task index (task names are stored in meta/tasks.parquet)
                task_idx = input_frame["task_index"].item()

                # Convert to float32 for output dataset
                joint_pos = result["joint_positions"][rel_idx].astype(np.float32)

                # Get timestamp - convert tensor to scalar if present
                timestamp_val = frame_idx / self.ee_dataset.fps
                if "timestamp" in input_frame:
                    timestamp_val = float(input_frame["timestamp"].item())

                # Create frame data (without video - we copied it directly)
                # Note: 'task' string is NOT stored in parquet, only task_index
                frame_data = {
                    "observation.state": joint_pos,
                    "action": joint_pos,
                    "episode_index": ep_idx,
                    "frame_index": frame_idx,
                    "timestamp": timestamp_val,
                    "task_index": task_idx,
                    "index": frames_written_total,  # Use global frame counter
                }
                chunk_data.append(frame_data)
                frames_in_current_file += 1
                frames_written_total += 1  # Increment global counter

                # Check if we need to start a new file
                if frames_in_current_file >= chunks_size:
                    # Write chunk data with fixed-size arrays
                    write_parquet_with_fixed_size_arrays(chunk_data, chunk_dir / f"file-{current_file:03d}.parquet")
                    chunk_data = []
                    frames_in_current_file = 0
                    current_file += 1

                    # Check if we need a new chunk
                    if current_file >= 1:  # For simplicity, create new chunk per file
                        current_chunk += 1
                        current_file = 0
                        chunk_dir = output_root / "data" / f"chunk-{current_chunk:03d}"
                        chunk_dir.mkdir(exist_ok=True)

            print(
                f"Episode {ep_idx + 1}/{total_episodes}: "
                f"IK success rate={result['success_rate']:.1%}, "
                f"avg pos error={result['avg_pos_error'] * 1000:.1f}mm, "
                f"avg rot error={np.rad2deg(result['avg_rot_error']):.1f}deg"
            )

        # Write remaining data
        if chunk_data:
            write_parquet_with_fixed_size_arrays(chunk_data, chunk_dir / f"file-{current_file:03d}.parquet")

        # Compute stats from all joint positions
        all_joint_positions = np.concatenate(all_joint_positions, axis=0)  # (total_frames, 6)
        stats = self._compute_stats(all_joint_positions)

        # Save stats to JSON file
        self._save_stats(stats, output_root / "meta" / "stats.json")

        # Calculate actual converted episode/frame counts
        actual_total_episodes = len(episodes_to_convert)
        actual_total_frames = frames_written_total

        # Create meta info
        info = {
            "codebase_version": "v3.0",
            "robot_type": "so101",
            "total_episodes": actual_total_episodes,
            "total_frames": actual_total_frames,
            "total_tasks": self.ee_dataset.meta.total_tasks,
            "chunks_size": chunks_size,
            "fps": self.ee_dataset.fps,
            "splits": {"train": f"0:{actual_total_episodes}"},
            "video_path": f"videos/{self.camera_key}/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.mp4",
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": self.output_features,
        }

        with open(output_root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # Copy episodes and tasks from input (only for converted episodes)
        input_meta = self.ee_dataset.meta.root / "meta"
        output_meta = output_root / "meta"

        # Load all episodes from the input dataset and filter them
        from lerobot.datasets.utils import load_episodes

        # Load all episodes from input (load_episodes expects the dataset root, not meta dir)
        all_input_episodes = load_episodes(self.ee_dataset.meta.root)

        # Filter to only converted episodes
        filtered_episodes = all_input_episodes.select(episodes_to_convert)

        # Re-index episodes to 0..N-1 for output and update frame indices
        cumulative_frames = 0
        filtered_data = []
        for i in range(len(filtered_episodes)):
            ep = filtered_episodes[i]
            old_from = ep['dataset_from_index']
            old_to = ep['dataset_to_index']
            num_frames = old_to - old_from + 1

            new_ep = {
                'episode_index': i,
                'tasks': ep['tasks'],
                'length': num_frames,
                'data/chunk_index': 0,  # All data goes to chunk-0
                'data/file_index': 0,   # All data goes to file-0
                'dataset_from_index': cumulative_frames,
                'dataset_to_index': cumulative_frames + num_frames - 1,
                'videos/observation.images.camera/chunk_index': 0,
                'videos/observation.images.camera/file_index': 0,
                'videos/observation.images.camera/from_timestamp': ep['videos/observation.images.camera/from_timestamp'],
                'videos/observation.images.camera/to_timestamp': ep['videos/observation.images.camera/to_timestamp'],
                'meta/episodes/chunk_index': 0,
                'meta/episodes/file_index': 0,
            }
            filtered_data.append(new_ep)
            cumulative_frames += num_frames

        # Save filtered episodes using the same chunked structure
        shutil.rmtree(output_meta / "episodes", ignore_errors=True)
        (output_meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

        # Write episodes to parquet using original features
        from datasets import Dataset

        ep_dataset = Dataset.from_list(filtered_data, features=all_input_episodes.features)
        ep_dataset.to_parquet(str(output_meta / "episodes" / "chunk-000" / "file-000.parquet"))

        # Copy tasks.parquet
        if (input_meta / "tasks.parquet").exists():
            shutil.copy2(input_meta / "tasks.parquet", output_meta / "tasks.parquet")

        print(f"\nConversion complete! Output saved to: {output_root}")
        print(f"Total episodes: {actual_total_episodes}")
        print(f"Total frames: {actual_total_frames}")

        # Save debug visualizations if requested
        if save_debug_viz:
            # Handle -1 meaning "visualize all episodes"
            num_to_viz = num_episodes_to_viz if num_episodes_to_viz != -1 else len(episodes_to_convert)
            save_trajectory_debug_viz(
                ee_dataset=self.ee_dataset,
                joint_positions=all_episode_joint_positions,
                episode_indices=episodes_to_convert,  # Pass original episode indices
                reset_pose=self.reset_pose,
                kinematics=self.kinematics,
                output_dir=Path(debug_output_dir),
                num_episodes_to_viz=num_to_viz,
            )

        # Return None (user can load the dataset from the output path)
        return None

    def _compute_stats(self, data: np.ndarray) -> dict:
        """
        Compute statistics for the dataset.

        Args:
            data: Array of shape (num_frames, state_dim) containing the data

        Returns:
            Dictionary with statistics for 'observation.state', 'action', and camera keys
        """
        # Compute statistics along the time axis (axis=0)
        stats_dict = {
            "observation.state": {
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0),
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
                "q01": np.quantile(data, 0.01, axis=0),
                "q10": np.quantile(data, 0.10, axis=0),
                "q50": np.quantile(data, 0.50, axis=0),
                "q90": np.quantile(data, 0.90, axis=0),
                "q99": np.quantile(data, 0.99, axis=0),
            },
            "action": {
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0),
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
                "q01": np.quantile(data, 0.01, axis=0),
                "q10": np.quantile(data, 0.10, axis=0),
                "q50": np.quantile(data, 0.50, axis=0),
                "q90": np.quantile(data, 0.90, axis=0),
                "q99": np.quantile(data, 0.99, axis=0),
            },
        }

        # Add dummy stats for camera keys (will be filled with ImageNet stats during training)
        # Note: We need to add non-empty stats so cast_stats_to_numpy doesn't filter them out
        for camera_key in [self.camera_key]:
            stats_dict[camera_key] = {
                "mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),  # RGB mean placeholder
                "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),   # RGB std placeholder
            }

        return stats_dict

    def _save_stats(self, stats: dict, output_path: Path) -> None:
        """
        Save statistics to a JSON file.

        Args:
            stats: Dictionary of statistics
            output_path: Path to save the stats.json file
        """
        # Convert numpy arrays to lists for JSON serialization
        def serialize_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: serialize_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_numpy(v) for v in obj]
            return obj

        serialized_stats = serialize_numpy(stats)

        with open(output_path, "w") as f:
            json.dump(serialized_stats, f, indent=2)


def plot_trajectory_comparison_3d(
    original_ee_poses: np.ndarray,
    joint_poses: np.ndarray,
    fk_ee_poses: np.ndarray,
    reset_pose: np.ndarray,
    output_path: Path,
    episode_idx: int = 0,
    kinematics: RobotKinematics | None = None,
):
    """
    Create 3D visualization comparing original EE trajectory vs FK-recovered trajectory.

    Args:
        original_ee_poses: (T, 3) original EE positions from dataset (meters)
        joint_poses: (T, 6) converted joint positions (degrees)
        fk_ee_poses: (T, 3) EE positions computed from joint poses via FK (meters)
        reset_pose: (6,) reset joint pose (degrees)
        output_path: Path to save visualization
        episode_idx: Episode index for title
        kinematics: RobotKinematics instance (optional, for reset pose FK)
    """
    T = original_ee_poses.shape[0]

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 8))

    # Extract gripper states
    original_gripper = original_ee_poses[:, -1] if original_ee_poses.shape[1] >= 7 else None

    # Plot 1: Full trajectories (top-down view)
    ax1 = fig.add_subplot(121)

    # Plot original EE trajectory (blue)
    ax1.plot(
        original_ee_poses[:, 0], original_ee_poses[:, 1],
        "b-", linewidth=2, label="Original EE", alpha=0.8
    )
    ax1.scatter(
        original_ee_poses[0, 0], original_ee_poses[0, 1],
        c="blue", s=100, marker="o", edgecolors="black", label="Start", zorder=10
    )
    ax1.scatter(
        original_ee_poses[-1, 0], original_ee_poses[-1, 1],
        c="blue", s=100, marker="s", edgecolors="black", label="End", zorder=10
    )

    # Plot FK-recovered EE trajectory (red)
    ax1.plot(
        fk_ee_poses[:, 0], fk_ee_poses[:, 1],
        "r--", linewidth=2, label="FK from Joint", alpha=0.8
    )
    ax1.scatter(
        fk_ee_poses[0, 0], fk_ee_poses[0, 1],
        c="red", s=100, marker="^", edgecolors="black", label="FK Start", zorder=10
    )
    ax1.scatter(
        fk_ee_poses[-1, 0], fk_ee_poses[-1, 1],
        c="red", s=100, marker="*", edgecolors="black", label="FK End", zorder=10
    )

    # Add gripper state visualization with point size
    if original_gripper is not None:
        for i in range(0, T, max(1, T // 20)):  # Show ~20 points
            size = 50 + original_gripper[i] * 100  # Scale by gripper state
            ax1.scatter(
                original_ee_poses[i, 0], original_ee_poses[i, 1],
                c="blue", s=size, alpha=0.3
            )
            ax1.scatter(
                fk_ee_poses[i, 0], fk_ee_poses[i, 1],
                c="red", s=size, alpha=0.3
            )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(f"Episode {episode_idx}: EE Trajectory Comparison (Top-Down View)")
    ax1.legend()
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)

    # Plot 2: 3D view
    ax2 = fig.add_subplot(122, projection="3d")

    ax2.plot(
        original_ee_poses[:, 0], original_ee_poses[:, 1], original_ee_poses[:, 2],
        "b-", linewidth=2, label="Original EE", alpha=0.8
    )
    ax2.scatter(
        [original_ee_poses[0, 0]], [original_ee_poses[0, 1]], [original_ee_poses[0, 2]],
        c="blue", s=100, marker="o", edgecolors="black", label="EE Start", zorder=10
    )
    ax2.scatter(
        [original_ee_poses[-1, 0]], [original_ee_poses[-1, 1]], [original_ee_poses[-1, 2]],
        c="blue", s=100, marker="s", edgecolors="black", label="EE End", zorder=10
    )

    ax2.plot(
        fk_ee_poses[:, 0], fk_ee_poses[:, 1], fk_ee_poses[:, 2],
        "r--", linewidth=2, label="FK from Joint", alpha=0.8
    )
    ax2.scatter(
        [fk_ee_poses[0, 0]], [fk_ee_poses[0, 1]], [fk_ee_poses[0, 2]],
        c="red", s=100, marker="^", edgecolors="black", label="FK Start", zorder=10
    )
    ax2.scatter(
        [fk_ee_poses[-1, 0]], [fk_ee_poses[-1, 1]], [fk_ee_poses[-1, 2]],
        c="red", s=100, marker="*", edgecolors="black", label="FK End", zorder=10
    )

    # Add connection lines at intervals
    step = max(1, T // 10)
    for i in range(0, T, step):
        ax2.plot(
            [original_ee_poses[i, 0], fk_ee_poses[i, 0]],
            [original_ee_poses[i, 1], fk_ee_poses[i, 1]],
            [original_ee_poses[i, 2], fk_ee_poses[i, 2]],
            "k:", alpha=0.3, linewidth=1
        )

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_title(f"Episode {episode_idx}: 3D Trajectory Comparison")
    ax2.legend()

    # Set equal aspect ratio
    all_pos = np.vstack([original_ee_poses[:, :3], fk_ee_poses[:, :3]])
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
    z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()

    x_pad = max((x_max - x_min) * 0.1, 0.01)
    y_pad = max((y_max - y_min) * 0.1, 0.01)
    z_pad = max((z_max - z_min) * 0.1, 0.01)

    ax2.set_xlim(x_min - x_pad, x_max + x_pad)
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)
    ax2.set_zlim(z_min - z_pad, z_max + z_pad)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def save_trajectory_debug_viz(
    ee_dataset: LeRobotDataset,
    joint_positions: list[np.ndarray],
    episode_indices: list[int],
    reset_pose: np.ndarray,
    kinematics: RobotKinematics,
    output_dir: Path,
    num_episodes_to_viz: int = 10,
):
    """
    Save trajectory comparison visualizations for debugging IK conversion.

    Args:
        ee_dataset: Input EE-only dataset
        joint_positions: List of (T_i, 6) joint position arrays for each converted episode
        episode_indices: Original episode indices in the dataset corresponding to joint_positions
        reset_pose: Reset joint pose used for alignment
        kinematics: RobotKinematics instance
        output_dir: Output directory for visualizations
        num_episodes_to_viz: Number of episodes to visualize
    """
    viz_dir = output_dir / "trajectory_comparisons"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving trajectory visualizations to: {viz_dir}")

    # Sample episodes to visualize
    total_converted = len(joint_positions)
    num_to_viz = min(num_episodes_to_viz, total_converted)

    # Select evenly spaced episodes from the converted ones
    local_indices = np.linspace(0, total_converted - 1, num_to_viz, dtype=int)

    for local_idx in local_indices:
        original_ep_idx = episode_indices[local_idx]  # Get original episode index

        # Get episode boundaries from original dataset
        episode = ee_dataset.meta.episodes[original_ep_idx]
        start_idx = episode["dataset_from_index"]
        end_idx = episode["dataset_to_index"] + 1  # +1 because end_idx is inclusive

        # Collect EE poses for this episode
        ee_poses = []
        for frame_idx in range(start_idx, end_idx):
            frame = ee_dataset[frame_idx]
            ee_pose = frame["observation.state"].cpu().numpy()  # (7,) [x, y, z, wx, wy, wz, gripper]
            ee_poses.append(ee_pose[:6])  # (6,) position + rotation
        ee_poses = np.array(ee_poses)  # (T, 6)

        # Get FK of reset pose
        reset_fk = kinematics.forward_kinematics(reset_pose)

        # IMPORTANT: Use the ORIGINAL first EE pose for alignment (same as in convert_episode)
        # This is critical - we must use the same alignment that was used during conversion
        first_ee_pose = ee_poses[0]  # (6,) [x, y, z, wx, wy, wz]
        first_ee_tf = np.eye(4)
        first_ee_tf[:3, :3] = Rotation.from_rotvec(first_ee_pose[3:6]).as_matrix()
        first_ee_tf[:3, 3] = first_ee_pose[:3]

        # Compute alignment transform: T_align = T_reset @ T_first_ee^(-1)
        # This transforms the EE trajectory to start from reset pose
        first_ee_tf_inv = np.linalg.inv(first_ee_tf)
        alignment_tf = reset_fk @ first_ee_tf_inv

        # Compute FK-recovered EE poses (from converted joint positions)
        # These should already be close to reset pose at the start since IK was solved on aligned EE
        fk_ee_poses = []
        for joint_pos in joint_positions[local_idx]:
            ee_T = kinematics.forward_kinematics(joint_pos.astype(np.float64))
            fk_ee_poses.append(ee_T[:3, 3].copy())
        fk_ee_poses = np.array(fk_ee_poses)  # (T, 3)

        # Compute aligned original EE trajectory (using the SAME alignment as in convert_episode)
        original_aligned_ee = []
        for ee_pose in ee_poses:
            ee_tf = np.eye(4)
            ee_tf[:3, :3] = Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
            ee_tf[:3, 3] = ee_pose[:3]
            aligned_tf = alignment_tf @ ee_tf
            original_aligned_ee.append(aligned_tf[:3, 3])
        original_aligned_ee = np.array(original_aligned_ee)  # (T, 3)

        # Compute error metrics
        pos_errors = np.linalg.norm(fk_ee_poses - original_aligned_ee, axis=1) * 1000  # mm
        mean_err = np.mean(pos_errors)
        max_err = np.max(pos_errors)

        # Save plot
        output_path = viz_dir / f"episode_{original_ep_idx:03d}_trajectory_comparison.png"
        plot_trajectory_comparison_3d(
            original_ee_poses=original_aligned_ee,
            joint_poses=joint_positions[local_idx],
            fk_ee_poses=fk_ee_poses,
            reset_pose=reset_pose,
            output_path=str(output_path),
            episode_idx=original_ep_idx,
            kinematics=kinematics,
        )

        print(f"  Episode {original_ep_idx}: saved to {output_path.name}")
        print(f"    Mean position error: {mean_err:.2f} mm, Max error: {max_err:.2f} mm")

    print(f"\nVisualizations saved for {len(local_indices)} episodes")

    # Create summary statistics
    print(f"\n=== IK Conversion Summary ===")
    print(f"Reset pose (degrees): {reset_pose}")
    print(f"Episodes visualized: {len(local_indices)}")


def convert_ee_to_joint_dataset(
    ee_dataset_path: str,
    output_repo_id: str,
    root: str | Path | None = None,
    urdf_path: str = "urdf/Simulation/SO101/so101_new_calib.urdf",
    reset_pose_deg: np.ndarray | list = DEFAULT_RESET_POSE_DEG,
    target_frame: str = "gripper_frame_link",
    joint_names: list[str] = DEFAULT_JOINT_NAMES,
    pos_tolerance: float = 0.005,
    rot_tolerance: float = 0.1,
    position_weight: float = 1.0,
    orientation_weight: float = 0.01,
    video_backend: str | None = None,
    save_debug_viz: bool = False,
    debug_output_dir: str | Path = "./outputs/debug/ee_to_joint_converter",
    num_episodes_to_viz: int = 10,
    episode_indices: list[int] | None = None,
) -> LeRobotDataset:
    """
    Convenience function to convert an EE-only dataset to a joint-based dataset.

    Args:
        ee_dataset_path: Path to input EE-only dataset
        output_repo_id: Repo ID for output dataset
        root: Root directory for output dataset
        urdf_path: Path to robot URDF
        reset_pose_deg: Reset pose in degrees
        target_frame: End-effector frame name
        joint_names: Names of joints
        pos_tolerance: Position tolerance for IK (meters)
        rot_tolerance: Rotation tolerance for IK (radians)
        position_weight: Position constraint weight
        orientation_weight: Orientation constraint weight
        video_backend: Video backend for output
        save_debug_viz: If True, save trajectory comparison visualizations for debugging
        debug_output_dir: Directory to save debug visualizations
        num_episodes_to_viz: Number of episodes to visualize (default: 10, use -1 for all)
        episode_indices: List of episode indices to convert (default: None = convert all)

    Returns:
        LeRobotDataset with joint positions

    Example:
        from lerobot.datasets.ee_to_joint_converter import convert_ee_to_joint_dataset

        output_ds = convert_ee_to_joint_dataset(
            ee_dataset_path="/path/to/ee_dataset",
            output_repo_id="output_joint_dataset",
        )
    """
    if isinstance(reset_pose_deg, list):
        reset_pose_deg = np.array(reset_pose_deg, dtype=np.float64)

    converter = EEToJointDatasetConverter(
        ee_dataset_path=ee_dataset_path,
        output_repo_id=output_repo_id,
        root=root,
        urdf_path=urdf_path,
        reset_pose_deg=reset_pose_deg,
        target_frame=target_frame,
        joint_names=joint_names,
        pos_tolerance=pos_tolerance,
        rot_tolerance=rot_tolerance,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
    )
    return converter.convert_all(
        video_backend=video_backend,
        save_debug_viz=save_debug_viz,
        debug_output_dir=debug_output_dir,
        num_episodes_to_viz=num_episodes_to_viz,
        episode_indices=episode_indices,
    )

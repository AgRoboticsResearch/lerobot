#!/usr/bin/env python
"""
Convert a LeRobot joint-space dataset to EE-space for relative EE training.

This script reads a LeRobot dataset with joint positions (6 DOF),
computes end-effector poses using forward kinematics, and writes
a new dataset that:
- keeps `observation.state` as 6D joints (unchanged)
- adds `observation.ee` as 7D EE pose at the current frame (for T_current)
- replaces `action` with 7D EE pose at the next frame (for T_future)
- copies all `observation.images.*` camera streams from the source

The output dataset is intended for use with RelativeEEDataset only.
Mode 1 (lerobot-train) should use the original source dataset.

Usage:
    python convert_joint_to_ee_dataset.py <source> <target> [--urdf URDF_PATH]
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import sys

from lerobot.model.kinematics import RobotKinematics


# ---------------- DEFAULT CONFIGURATION ----------------
DEFAULT_URDF_PATH = "urdf/Simulation/SO101/so101_sroi.urdf"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EE_LINK_NAME = "camera_link"

# Gripper limits from URDF (in degrees)
GRIPPER_LOWER_DEG = -10.0
GRIPPER_UPPER_DEG = 100.0
# -----------------------------------------------


JOINT_OBSERVATION_KEY = "observation.joint_state"


def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to axis-angle rotation vector."""
    trace = np.trace(R)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)

    skew = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    )
    axis = skew / (2.0 * np.sin(theta))
    return axis * theta


def transform_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract position and rotation vector from 4x4 homogeneous transformation matrix.
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        Tuple of (position [x, y, z], rotvec [wx, wy, wz])
    """
    position = T[:3, 3]
    rotvec = rotation_matrix_to_rotvec(T[:3, :3])
    return position, rotvec


def load_dataset_info(dataset_path: Path) -> dict:
    """Load dataset info from info.json."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def load_episodes_metadata(dataset_path: Path) -> pd.DataFrame:
    """Load episodes metadata from parquet file."""
    episodes_dir = dataset_path / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("**/*.parquet"))
    episodes = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        episodes.append(df)
    return pd.concat(episodes, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def load_data(dataset_path: Path) -> pd.DataFrame:
    """Load main data from parquet files."""
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    data = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        data.append(df)
    return pd.concat(data, ignore_index=True).sort_values("index").reset_index(drop=True)


def joint_state_to_ee_state(joint_state_deg: np.ndarray, kinematics: RobotKinematics) -> np.ndarray:
    """
    Convert 6D joint state to 7D EE state.

    Args:
        joint_state_deg: [shoulder_pan, shoulder_lift, elbow_flex,
                         wrist_flex, wrist_roll, gripper] in degrees

    Returns:
        ee_state: [ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos]
    """
    # Convert to float64 for placo compatibility
    joint_state_deg = np.asarray(joint_state_deg, dtype=np.float64)

    # Compute FK: degrees -> 4x4 transform matrix
    T_ee = kinematics.forward_kinematics(joint_state_deg)

    # Extract position and rotation vector
    position, rotvec = transform_to_pose(T_ee)

    # Normalize gripper from degrees to [0, 1]
    gripper_deg = joint_state_deg[5]
    gripper_norm = (gripper_deg - GRIPPER_LOWER_DEG) / (GRIPPER_UPPER_DEG - GRIPPER_LOWER_DEG)
    gripper_norm = np.clip(gripper_norm, 0.0, 1.0)

    # Combine: [x, y, z, wx, wy, wz, gripper]
    return np.concatenate([position, rotvec, [gripper_norm]])


def extract_video_segment(
    source_video_path: Path,
    output_video_path: Path,
    start_frame: int,
    num_frames: int,
    fps: int = 30
):
    """Extract a segment of frames from video using ffmpeg."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg with -c:v copy for direct stream copy (fast, no re-encoding)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_frame / fps),  # Start time in seconds
        "-i", str(source_video_path),
        "-t", str(num_frames / fps),    # Duration in seconds
        "-c:v", "copy",                 # Copy without re-encoding
        "-an",                          # No audio
        str(output_video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract video segment: {result.stderr}")


def copy_video_direct(
    source_video_path: Path,
    output_video_path: Path,
):
    """Copy entire video file directly."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_video_path, output_video_path)


def concatenate_videos_with_ffmpeg(
    source_videos: list[Path],
    output_video_path: Path,
):
    """
    Concatenate multiple video files into a single video using ffmpeg.

    Creates a temporary concat list file and uses ffmpeg's concat demuxer
    for fast concatenation without re-encoding.

    Args:
        source_videos: List of video file paths to concatenate (in order)
        output_video_path: Path for the output concatenated video
    """
    import tempfile
    import os

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(source_videos) == 0:
        raise FileNotFoundError("No source videos provided")

    if len(source_videos) == 1:
        shutil.copy(source_videos[0], output_video_path)
        return

    # Create a temporary concat list file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for video_path in source_videos:
            # Use absolute paths and escape single quotes
            abs_path = str(video_path.resolve()).replace("'", "'\\''")
            f.write(f"file '{abs_path}'\n")

    try:
        # Use ffmpeg concat demuxer with stream copy (no re-encoding)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-an",  # No audio
            str(output_video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to concatenate videos: {result.stderr}")

    finally:
        # Clean up temporary file
        os.unlink(concat_file)


def compute_stats_for_feature(data: np.ndarray) -> dict:
    """Compute statistics for a feature array."""
    return {
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q10": np.quantile(data, 0.10, axis=0).tolist(),
        "q50": np.quantile(data, 0.50, axis=0).tolist(),
        "q90": np.quantile(data, 0.90, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def get_camera_keys(features: dict) -> list[str]:
    """Return all observation image/video keys to preserve."""
    return [key for key in features if key.startswith("observation.images.")]


def normalize_camera_feature(feature: dict, fps: int) -> dict:
    """Copy camera feature metadata and normalize image shape naming to CHW."""
    video_info = feature.copy()
    if "shape" in video_info and len(video_info["shape"]) == 3:
        if video_info["shape"] == [480, 640, 3]:
            video_info["shape"] = [3, 480, 640]
            video_info["names"] = ["channels", "height", "width"]

    video_info.setdefault("dtype", "video")
    video_info.setdefault("shape", [3, 480, 640])
    video_info.setdefault("names", ["channels", "height", "width"])
    video_info.setdefault(
        "info",
        {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    )
    return video_info


def list_camera_videos(source_path: Path, camera_key: str) -> list[Path]:
    """List all video files for a camera key in episode order."""
    source_video_dir = source_path / "videos" / camera_key
    return sorted(source_video_dir.glob("**/*.mp4"))


def filter_camera_videos(source_path: Path, camera_key: str, episodes_to_convert: list[int] | None) -> list[Path]:
    """Filter source videos to the requested episode indices."""
    source_video_dir = source_path / "videos" / camera_key
    source_videos = list_camera_videos(source_path, camera_key)
    if episodes_to_convert is None:
        return source_videos

    filtered_videos = []
    for ep_idx in sorted(episodes_to_convert):
        video_name = f"file-{ep_idx:03d}.mp4"
        video_path = source_video_dir / video_name
        if video_path.exists():
            filtered_videos.append(video_path)
            continue

        found = False
        for chunk_dir in source_video_dir.iterdir():
            if not chunk_dir.is_dir():
                continue
            chunk_video = chunk_dir / video_name
            if chunk_video.exists():
                filtered_videos.append(chunk_video)
                found = True
                break

        if not found:
            print(f"  Warning: Video not found for camera {camera_key}, episode {ep_idx}: {video_name}")

    return filtered_videos


def update_episode_video_timestamps(new_episodes: pd.DataFrame, camera_keys: list[str], fps: int) -> None:
    """Shift per-episode timestamps to match the concatenated per-camera video files."""
    for camera_key in camera_keys:
        from_col = f"videos/{camera_key}/from_timestamp"
        to_col = f"videos/{camera_key}/to_timestamp"
        chunk_col = f"videos/{camera_key}/chunk_index"
        file_col = f"videos/{camera_key}/file_index"

        if from_col not in new_episodes.columns or to_col not in new_episodes.columns:
            continue

        cumulative_duration = 0.0
        for ep_idx in range(len(new_episodes)):
            ep_length = int(new_episodes.at[ep_idx, "length"])
            ep_duration = ep_length / fps if ep_length > 0 else 0.0

            new_episodes.at[ep_idx, from_col] = cumulative_duration
            new_episodes.at[ep_idx, to_col] = cumulative_duration + ep_duration
            if chunk_col in new_episodes.columns:
                new_episodes.at[ep_idx, chunk_col] = int(0)
            if file_col in new_episodes.columns:
                new_episodes.at[ep_idx, file_col] = int(0)

            cumulative_duration += ep_duration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot dataset from joint-space to end-effector space."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the source dataset (joint-space)",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to the target dataset (end-effector space)",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help=f"Path to the URDF file (default: {DEFAULT_URDF_PATH})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Episode indices to convert (default: all episodes)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = args.source
    target_path = args.target
    urdf_path = args.urdf
    episodes_to_convert = args.episodes

    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    print(f"Converting dataset from joint-space to end-effector space")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    if episodes_to_convert is not None:
        print(f"Episodes: {episodes_to_convert}")

    # 1. Load source dataset info
    print("\nLoading source dataset...")
    info = load_dataset_info(source_path)
    episodes_df = load_episodes_metadata(source_path)
    data_df = load_data(source_path)

    # Filter episodes if specified
    if episodes_to_convert is not None:
        # Validate episode indices
        max_episode = len(episodes_df) - 1
        invalid_episodes = [ep for ep in episodes_to_convert if ep < 0 or ep > max_episode]
        if invalid_episodes:
            raise ValueError(f"Invalid episode indices: {invalid_episodes}. Valid range: 0-{max_episode}")

        # Filter episodes_df and data_df
        episodes_df = episodes_df.iloc[episodes_to_convert].reset_index(drop=True)
        data_df = data_df[data_df['episode_index'].isin(episodes_to_convert)].copy()

        # Remap episode indices to 0..n-1
        old_to_new_ep_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episodes_to_convert))}
        data_df['episode_index'] = data_df['episode_index'].replace(old_to_new_ep_idx)

    print(f"  Episodes: {len(episodes_df)}")
    print(f"  Total frames: {len(data_df)}")
    print(f"  FPS: {info.get('fps', 30)}")

    # 2. Initialize kinematics solver
    print("\nInitializing kinematics solver...")
    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name=EE_LINK_NAME,
        joint_names=JOINT_NAMES,
    )
    print(f"  URDF: {urdf_path}")
    print(f"  EE link: {EE_LINK_NAME}")

    # 3. Create target directory structure
    print("\nCreating target directory structure...")
    target_path.mkdir(parents=True, exist_ok=True)
    (target_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (target_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    camera_keys = get_camera_keys(info["features"])

    # Copy entire videos directory structure from source (preserve exact structure)
    source_videos_dir = source_path / "videos"
    target_videos_dir = target_path / "videos"
    if source_videos_dir.exists():
        shutil.copytree(source_videos_dir, target_videos_dir)
        print(f"  Copied videos directory: {source_videos_dir} -> {target_videos_dir}")

    # 4. Convert data for each episode
    print("\nConverting episodes...")
    all_converted_data = []
    fps = info.get("fps", 30)

    if camera_keys:
        print(f"  Preserving camera keys: {camera_keys}")
    else:
        print("  No observation.images.* camera keys found")

    for ep_idx in tqdm(range(len(episodes_df))):
        episode_info = episodes_df.iloc[ep_idx]
        from_idx = int(episode_info["dataset_from_index"])
        to_idx = int(episode_info["dataset_to_index"])

        # Get episode data
        episode_data = data_df[
            (data_df["episode_index"] == ep_idx) &
            (data_df["index"] >= from_idx) &
            (data_df["index"] <= to_idx)
        ].copy()

        # Pre-convert all joint states to EE states for this episode
        ee_states = []
        for _, row in episode_data.iterrows():
            joint_state = np.array(row['observation.state'])
            ee_state = joint_state_to_ee_state(joint_state, kinematics)
            ee_states.append(ee_state)

        # observation.state: joints unchanged
        # observation.ee: EE at current frame (for T_current in relative EE computation)
        # action: EE at next frame (for T_future in relative EE computation)
        converted_rows = []
        num_frames = len(episode_data)
        for i, (_, row) in enumerate(episode_data.iterrows()):
            joint_state = np.array(row['observation.state']).astype(np.float32)

            # observation.ee: EE at current frame
            obs_ee = ee_states[i].astype(np.float32)

            # action: EE at next frame, or current for last frame
            if i < num_frames - 1:
                action_ee = ee_states[i + 1].astype(np.float32)
            else:
                action_ee = ee_states[i].astype(np.float32)

            converted_rows.append({
                'observation.state': joint_state,
                'observation.ee': obs_ee,
                'action': action_ee,
                'timestamp': row['timestamp'],
                'frame_index': row['frame_index'],
                'episode_index': row['episode_index'],
                'index': row['index'],
                'task_index': row['task_index'],
            })

        all_converted_data.extend(converted_rows)

    # Videos already copied directly in step 3

    # 5. Combine and save data
    print("\nSaving converted data...")
    combined_data = pd.DataFrame(all_converted_data)
    combined_data = combined_data.sort_values("index").reset_index(drop=True)

    # Re-number indices sequentially
    combined_data["index"] = range(len(combined_data))
    # Keep frame_index as is (within episode)
    # Keep episode_index as is

    # Build PyArrow arrays with correct types
    obs_state_list = combined_data['observation.state'].tolist()
    obs_ee_list = combined_data['observation.ee'].tolist()
    action_list = combined_data['action'].tolist()

    obs_state_array = pa.array(obs_state_list, type=pa.list_(pa.float32()))
    obs_ee_array = pa.array(obs_ee_list, type=pa.list_(pa.float32()))
    action_array = pa.array(action_list, type=pa.list_(pa.float32()))
    timestamp_array = pa.array(combined_data['timestamp'], type=pa.float32())
    frame_index_array = pa.array(combined_data['frame_index'], type=pa.int64())
    episode_index_array = pa.array(combined_data['episode_index'], type=pa.int64())
    index_array = pa.array(combined_data['index'], type=pa.int64())
    task_index_array = pa.array(combined_data['task_index'], type=pa.int64())

    schema = pa.schema([
        ('observation.state', pa.list_(pa.float32())),
        ('observation.ee', pa.list_(pa.float32())),
        ('action', pa.list_(pa.float32())),
        ('timestamp', pa.float32()),
        ('frame_index', pa.int64()),
        ('episode_index', pa.int64()),
        ('index', pa.int64()),
        ('task_index', pa.int64()),
    ])

    table = pa.Table.from_arrays([
        obs_state_array,
        obs_ee_array,
        action_array,
        timestamp_array,
        frame_index_array,
        episode_index_array,
        index_array,
        task_index_array,
    ], schema=schema)

    # Copy HuggingFace metadata from original
    original_data = load_data_with_arrow(source_path)
    if original_data.schema.metadata and b'huggingface' in original_data.schema.metadata:
        table = table.replace_schema_metadata(original_data.schema.metadata)

    data_output_path = target_path / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(table, data_output_path)
    print(f"  Saved data: {data_output_path}")

    # 6. Create episodes metadata
    print("\nCreating episodes metadata...")
    new_episodes = episodes_df.copy()

    for ep_idx in range(len(new_episodes)):
        new_episodes.at[ep_idx, "episode_index"] = ep_idx
        # Update frame counts
        ep_data = combined_data[combined_data["episode_index"] == ep_idx]
        if len(ep_data) > 0:
            new_episodes.at[ep_idx, "dataset_from_index"] = ep_data["index"].min()
            new_episodes.at[ep_idx, "dataset_to_index"] = ep_data["index"].max()
            new_episodes.at[ep_idx, "length"] = len(ep_data)

    # Keep video timestamps from source unchanged

    episodes_output_path = target_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    # Ensure integer dtypes for index columns before saving
    for col in new_episodes.columns:
        if col.endswith("/chunk_index") or col.endswith("/file_index"):
            new_episodes[col] = new_episodes[col].astype("int64")

    new_episodes.to_parquet(episodes_output_path, index=False)
    print(f"  Saved episodes metadata: {episodes_output_path}")

    # 7. Create info.json
    print("\nCreating info.json...")
    new_info = info.copy()

    # Update feature shapes and names
    new_info["total_episodes"] = len(episodes_df)
    new_info["total_frames"] = len(combined_data)
    new_info["robot_type"] = "so101"  # Update robot type

    # observation.state stays as [6] joints (unchanged from source)
    # action becomes [7] EE at next frame, observation.ee is [7] EE at current frame
    ee_feature = {
        "dtype": "float32",
        "names": ["x", "y", "z", "wx", "wy", "wz", "gripper"],
        "shape": [7],
    }
    new_info["features"]["action"] = ee_feature.copy()
    new_info["features"]["observation.ee"] = ee_feature.copy()

    # Keep camera features exactly as source
    for camera_key in camera_keys:
        new_info["features"][camera_key] = info["features"][camera_key]

    # Update splits
    new_info["splits"] = {"train": f"0:{len(episodes_df)}"}

    # Record EE frame info for deployment frame handling
    new_info["ee_target_frame"] = EE_LINK_NAME
    new_info["ee_urdf_path"] = str(urdf_path)

    info_output_path = target_path / "meta" / "info.json"
    with open(info_output_path, "w") as f:
        json.dump(new_info, f, indent=2)
    print(f"  Saved info.json: {info_output_path}")

    # 8. Copy source stats and add action.ee stats
    print("\nComputing statistics...")
    stats_path = source_path / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"  Copied stats from source: {list(stats.keys())}")
    else:
        stats = {}

    # Add stats for new fields
    obs_ee_data = np.stack(combined_data['observation.ee'].values)
    stats["observation.ee"] = compute_stats_for_feature(obs_ee_data)
    action_data = np.stack(combined_data['action'].values)
    stats["action"] = compute_stats_for_feature(action_data)

    stats_output_path = target_path / "meta" / "stats.json"
    with open(stats_output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats.json: {stats_output_path}")

    # 9. Copy tasks.parquet if exists
    tasks_source = source_path / "meta" / "tasks.parquet"
    if tasks_source.exists():
        shutil.copy(tasks_source, target_path / "meta" / "tasks.parquet")
        print(f"  Copied tasks.parquet")

    print(f"\nConversion complete!")
    print(f"  Output: {target_path}")
    print(f"  Episodes: {len(episodes_df)}")
    print(f"  Frames: {len(combined_data)}")


def load_data_with_arrow(dataset_path: Path) -> pa.Table:
    """Load main data using PyArrow to preserve schema."""
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    tables = []
    for f in parquet_files:
        table = pq.read_table(f)
        tables.append(table)
    combined = pa.concat_tables(tables)
    return combined.sort_by("index")


if __name__ == "__main__":
    main()

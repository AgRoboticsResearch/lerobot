#!/usr/bin/env python
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

"""Convert LeRobot EE datasets to UMI Zarr format."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import av
import zarr
import numcodecs
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def extract_frames_from_video(video_path: Path, num_frames: int) -> np.ndarray:
    """Extract frames from MP4 video.

    Args:
        video_path: Path to MP4 video file
        num_frames: Number of frames to extract

    Returns:
        Array of shape (num_frames, H, W, 3) with uint8 RGB images
    """
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i >= num_frames:
                break
            # Convert to RGB numpy array
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)

    if len(frames) != num_frames:
        raise ValueError(f"Expected {num_frames} frames, got {len(frames)} from {video_path}")

    return np.stack(frames, axis=0)


def convert_lerobot_to_umi(
    input_dataset_path: str,
    output_zarr_path: str,
    image_size: tuple = (224, 224),
    camera_key: str = "observation.images.camera",
) -> None:
    """
    Convert a LeRobot EE dataset to UMI Zarr format.

    The LeRobot EE dataset has:
    - observation.state: 7D [x, y, z, wx, wy, wz, gripper]
    - action: 7D [x, y, z, wx, wy, wz, gripper]

    UMI format expects:
    - robot0_eef_pos: [T, 3] - EE position
    - robot0_eef_rot_axis_angle: [T, 3] - EE rotation (axis-angle)
    - robot0_gripper_width: [T, 1] - gripper state
    - camera0_rgb: [T, H, W, 3] - camera images
    - action: [T, 7] - future EE pose (position + rotation + gripper)
    - robot0_demo_start_pose: [T, 6] - demo start pose (for pose repr)
    - robot0_demo_end_pose: [T, 6] - demo end pose (for pose repr)

    Args:
        input_dataset_path: Path to LeRobot EE dataset
        output_zarr_path: Path to output .zarr.zip file
        image_size: Size to resize images to (H, W)
        camera_key: Key for camera observations in LeRobot dataset
    """
    # Load input dataset
    print(f"Loading LeRobot dataset from: {input_dataset_path}")
    lerobot_dataset = LeRobotDataset(input_dataset_path)

    print(f"Episodes: {lerobot_dataset.num_episodes}")
    print(f"Total frames: {len(lerobot_dataset)}")
    print(f"FPS: {lerobot_dataset.fps}")

    # Check that it's an EE dataset (using info from metadata)
    state_shape = lerobot_dataset.meta.info['features']['observation.state']['shape']
    if state_shape[0] != 7:
        raise ValueError(f"Expected 7D EE state, got shape {state_shape}")

    # Create UMI replay buffer in memory
    root = zarr.group()
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    # Get image size from dataset if available, otherwise use default
    camera_info = lerobot_dataset.meta.info['features'][camera_key]
    orig_height, orig_width = camera_info['shape'][1], camera_info['shape'][2]
    out_h, out_w = image_size

    print(f"Original image size: ({orig_height}, {orig_width})")
    print(f"Output image size: ({out_h}, {out_w})")

    # Initialize episode tracking
    all_episode_ends = []
    current_frame_idx = 0

    # Temporary storage for data
    all_eef_pos = []
    all_eef_rot = []
    all_gripper = []
    all_action = []
    all_images = []

    # Process each episode
    for ep_idx in tqdm(range(lerobot_dataset.num_episodes), desc="Converting episodes"):
        episode = lerobot_dataset.meta.episodes[ep_idx]
        start_idx = episode["dataset_from_index"]
        end_idx = episode["dataset_to_index"]
        episode_length = end_idx - start_idx

        episode_eef_pos = []
        episode_eef_rot = []
        episode_gripper = []
        episode_action = []
        episode_images = []

        for frame_idx in range(start_idx, end_idx):
            frame = lerobot_dataset[frame_idx]
            state = frame["observation.state"].cpu().numpy()
            action = frame["action"].cpu().numpy()

            # Extract EE pose components
            # state and action are both 7D: [x, y, z, wx, wy, wz, gripper]
            eef_pos = state[:3]  # [x, y, z]
            eef_rot = state[3:6]  # [wx, wy, wz] - axis-angle
            gripper = state[6:7]  # [gripper]

            # For UMI, action is the future EE pose (next timestep's observation)
            # We use the action from current frame as the target
            # action is already the target EE pose
            action_pose = action[:6]  # [x, y, z, wx, wy, wz]
            action_gripper = action[6:7]

            episode_eef_pos.append(eef_pos)
            episode_eef_rot.append(eef_rot)
            episode_gripper.append(gripper)
            episode_action.append(np.concatenate([action_pose, action_gripper]))

            # Get image
            img = frame[camera_key].cpu().numpy()
            # Image is (C, H, W), convert to (H, W, C) and resize
            img = np.transpose(img, (1, 2, 0))
            if img.shape[:2] != (out_h, out_w):
                # Use cv2 for fast resize
                import cv2
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            episode_images.append(img)

        # Stack episode data
        all_eef_pos.append(np.stack(episode_eef_pos, axis=0))
        all_eef_rot.append(np.stack(episode_eef_rot, axis=0))
        all_gripper.append(np.stack(episode_gripper, axis=0))
        all_action.append(np.stack(episode_action, axis=0))
        all_images.append(np.stack(episode_images, axis=0))

        current_frame_idx += episode_length
        all_episode_ends.append(current_frame_idx)

    # Concatenate all episodes
    all_eef_pos = np.concatenate(all_eef_pos, axis=0).astype(np.float32)
    all_eef_rot = np.concatenate(all_eef_rot, axis=0).astype(np.float32)
    all_gripper = np.concatenate(all_gripper, axis=0).astype(np.float32)
    all_action = np.concatenate(all_action, axis=0).astype(np.float32)
    all_images = np.concatenate(all_images, axis=0).astype(np.uint8)

    print(f"Total frames after conversion: {all_eef_pos.shape[0]}")

    # Compute demo start and end poses
    # For UMI, these are the first and last poses of each episode
    demo_start_poses = []
    demo_end_poses = []
    frame_idx = 0
    for ep_idx in range(lerobot_dataset.num_episodes):
        episode = lerobot_dataset.meta.episodes[ep_idx]
        episode_length = episode["dataset_to_index"] - episode["dataset_from_index"]

        # Start pose: first frame's EE pose
        start_pose = np.concatenate([all_eef_pos[frame_idx], all_eef_rot[frame_idx]])
        # End pose: last frame's EE pose
        end_pose = np.concatenate([
            all_eef_pos[frame_idx + episode_length - 1],
            all_eef_rot[frame_idx + episode_length - 1]
        ])

        # Broadcast to all frames in episode
        demo_start_poses.append(np.tile(start_pose, (episode_length, 1)))
        demo_end_poses.append(np.tile(end_pose, (episode_length, 1)))

        frame_idx += episode_length

    all_demo_start = np.concatenate(demo_start_poses, axis=0).astype(np.float32)
    all_demo_end = np.concatenate(demo_end_poses, axis=0).astype(np.float32)

    # Create zarr arrays
    print("Creating zarr arrays...")

    # Compression settings
    compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE)

    # Low-dim data
    data_group.create_dataset(
        'robot0_eef_pos',
        data=all_eef_pos,
        chunks=(100, 3),
        compressor=compressor,
        dtype=np.float32
    )
    data_group.create_dataset(
        'robot0_eef_rot_axis_angle',
        data=all_eef_rot,
        chunks=(100, 3),
        compressor=compressor,
        dtype=np.float32
    )
    data_group.create_dataset(
        'robot0_gripper_width',
        data=all_gripper,
        chunks=(100, 1),
        compressor=compressor,
        dtype=np.float32
    )
    data_group.create_dataset(
        'action',
        data=all_action,
        chunks=(100, 7),
        compressor=compressor,
        dtype=np.float32
    )
    data_group.create_dataset(
        'robot0_demo_start_pose',
        data=all_demo_start,
        chunks=(100, 6),
        compressor=compressor,
        dtype=np.float32
    )
    data_group.create_dataset(
        'robot0_demo_end_pose',
        data=all_demo_end,
        chunks=(100, 6),
        compressor=compressor,
        dtype=np.float32
    )

    # Images with JPEG-XL compression (if available) or JPEG
    try:
        from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
        register_codecs()
        img_compressor = JpegXl(level=85, numthreads=1)
        print("Using JpegXl compression for images")
    except ImportError:
        # Fallback to standard compression
        img_compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
        print("JpegXl not available, using Blosc compression for images")

    data_group.create_dataset(
        'camera0_rgb',
        data=all_images,
        chunks=(1, out_h, out_w, 3),
        compressor=img_compressor,
        dtype=np.uint8
    )

    # Meta data
    meta_group.create_dataset(
        'episode_ends',
        data=np.array(all_episode_ends, dtype=np.int64),
        chunks=(len(all_episode_ends),),
        compressor=None,
        dtype=np.int64
    )

    # Save to ZipStore
    print(f"Saving to: {output_zarr_path}")
    output_path = Path(output_zarr_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use zarr.copy_store to save to zip
    with zarr.ZipStore(str(output_path), mode='w') as zip_store:
        zarr.copy_store(source=root.store, dest=zip_store)

    print(f"Conversion complete!")
    print(f"Output saved to: {output_path}")
    print(f"  Episodes: {lerobot_dataset.num_episodes}")
    print(f"  Frames: {all_eef_pos.shape[0]}")
    print(f"  Image size: {all_images.shape[1:4]}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot EE dataset to UMI Zarr format"
    )
    parser.add_argument(
        "--input-dataset",
        required=True,
        help="Path to input LeRobot EE dataset",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output .zarr.zip file",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("HEIGHT", "WIDTH"),
        help="Output image size (default: 224 224)",
    )
    parser.add_argument(
        "--camera-key",
        default="observation.images.camera",
        help="Key for camera observations in LeRobot dataset",
    )

    args = parser.parse_args()

    convert_lerobot_to_umi(
        input_dataset_path=args.input_dataset,
        output_zarr_path=args.output,
        image_size=tuple(args.image_size),
        camera_key=args.camera_key,
    )


if __name__ == "__main__":
    main()

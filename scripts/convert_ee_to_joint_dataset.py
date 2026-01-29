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

"""Convert EE-only datasets to joint-based datasets using inverse kinematics."""

import argparse

import numpy as np

from lerobot.datasets.ee_to_joint_converter import (
    DEFAULT_JOINT_NAMES,
    EEToJointDatasetConverter,
)

DEFAULT_RESET_POSE = [
    -8.00,
    -62.73,
    65.05,
    0.86,
    -2.55,
    88.91,
]


def main():
    parser = argparse.ArgumentParser(
        description="Convert end-effector dataset to joint-based dataset using IK"
    )
    parser.add_argument(
        "--input-dataset",
        required=True,
        help="Path to input EE-only dataset",
    )
    parser.add_argument(
        "--output-repo-id",
        required=True,
        help="Repo ID for output joint-based dataset",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory for output dataset (default: HF_LEROBOT_HOME/output_repo_id)",
    )
    parser.add_argument(
        "--urdf",
        default="urdf/Simulation/SO101/so101_new_calib.urdf",
        help="Path to robot URDF file",
    )
    parser.add_argument(
        "--reset-pose",
        nargs=6,
        type=float,
        default=DEFAULT_RESET_POSE,
        metavar=("PAN", "LIFT", "ELBOW", "FLEX", "ROLL", "GRIPPER"),
        help="Reset pose in degrees (shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper)",
    )
    parser.add_argument(
        "--pos-tolerance",
        type=float,
        default=0.02,
        help="Position tolerance for IK verification in meters (default: 0.02 = 20mm)",
    )
    parser.add_argument(
        "--rot-tolerance",
        type=float,
        default=0.3,
        help="Rotation tolerance for IK verification in radians (default: 0.3 = ~17 deg)",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=1.0,
        help="Weight for position constraint in IK (default: 1.0)",
    )
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=1.0,
        help="Weight for orientation constraint in IK (default: 1.0)",
    )
    parser.add_argument(
        "--target-frame",
        default="gripper_frame_link",
        help="Name of end-effector frame in URDF (default: gripper_frame_link)",
    )
    parser.add_argument(
        "--video-backend",
        default=None,
        choices=["torchcodec", "pyav", "video_reader"],
        help="Video backend to use for output dataset",
    )
    parser.add_argument(
        "--save-debug-viz",
        action="store_true",
        help="Save trajectory comparison visualizations for debugging IK conversion",
    )
    parser.add_argument(
        "--debug-output-dir",
        default="./outputs/debug/ee_to_joint_converter",
        help="Directory to save debug visualizations (default: ./outputs/debug/ee_to_joint_converter)",
    )
    parser.add_argument(
        "--num-episodes-to-viz",
        type=int,
        default=10,
        help="Number of episodes to visualize for debugging (default: 10, use -1 for all)",
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific episode indices to convert (default: None = convert all episodes). "
             "Example: --episode-indices 0 1 2",
    )

    args = parser.parse_args()

    print(f"Converting EE dataset: {args.input_dataset}")
    print(f"Output repo ID: {args.output_repo_id}")
    print(f"URDF: {args.urdf}")
    print(f"Reset pose (degrees): {args.reset_pose}")
    print(f"Position tolerance: {args.pos_tolerance * 1000:.1f} mm")
    print(f"Rotation tolerance: {np.rad2deg(args.rot_tolerance):.1f} deg")
    print()

    converter = EEToJointDatasetConverter(
        ee_dataset_path=args.input_dataset,
        output_repo_id=args.output_repo_id,
        root=args.root,
        urdf_path=args.urdf,
        reset_pose_deg=np.array(args.reset_pose, dtype=np.float64),
        target_frame=args.target_frame,
        joint_names=DEFAULT_JOINT_NAMES,
        pos_tolerance=args.pos_tolerance,
        rot_tolerance=args.rot_tolerance,
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
    )

    output_ds = converter.convert_all(
        video_backend=args.video_backend,
        save_debug_viz=args.save_debug_viz,
        debug_output_dir=args.debug_output_dir,
        num_episodes_to_viz=args.num_episodes_to_viz,
        episode_indices=args.episode_indices,
    )

    # Conversion complete - stats are already printed by convert_all()


if __name__ == "__main__":
    main()

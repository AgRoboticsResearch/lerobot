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

"""
Processor for converting 10D relative EE actions to absolute EE poses.

This processor is used for deploying policies trained with RelativeEEDataset (UMI-style).
The policy outputs 10D relative poses: [dx, dy, dz, rot6d_0..5, gripper]
This processor converts them to absolute EE poses for downstream IK processing.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Following UMI's row-based convention (axis=-2 for stacking).
    The Zhou et al. 2019 paper describes columns, but UMI's code uses rows.

    Args:
        rot6d: 6D rotation array of shape (..., 6)

    Returns:
        3x3 rotation matrix, shape (..., 3, 3)
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]

    # Normalize first row
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)

    # Make second row orthogonal to first
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)

    # Third row via cross product
    b3 = np.cross(b1, b2, axis=-1)

    # Stack into rotation matrix (axis=-2 = rows, following UMI's convention)
    rotmat = np.stack([b1, b2, b3], axis=-2)
    return rotmat


def pose10d_to_mat(pose10d: np.ndarray) -> np.ndarray:
    """Convert 10D pose representation to 4x4 transformation matrix.

    Args:
        pose10d: 10D pose array where:
            - pose10d[:3] is position
            - pose10d[3:9] is 6D rotation
            - pose10d[9] is gripper (not used for matrix)

    Returns:
        4x4 transformation matrix
    """
    pos = pose10d[:3]
    rot6d = pose10d[3:9]
    rotmat = rot6d_to_mat(rot6d)

    mat = np.eye(4, dtype=pose10d.dtype)
    mat[:3, :3] = rotmat
    mat[:3, 3] = pos
    return mat


@ProcessorStepRegistry.register("relative_10d_to_absolute_ee")
@dataclass
class Relative10DToAbsoluteEE(RobotActionProcessorStep):
    """
    Converts 10D relative EE actions to absolute EE poses using forward kinematics.

    This processor is designed for policies trained with RelativeEEDataset (UMI-style).
    The policy outputs 10D relative poses: [dx, dy, dz, rot6d_0..5, gripper]
    These are relative to the current end-effector pose.

    The processor:
    1. Gets current joint positions from the observation
    2. Computes current EE pose via forward kinematics
    3. Converts the 10D relative action to a transformation matrix
    4. Chains: T_target = T_current @ T_rel (UMI-style)
    5. Outputs absolute EE pose with keys: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        motor_names: A list of motor names for getting joint positions.
        gripper_scale: Scale factor for gripper (default 100 for [0,1] -> [0,100]).
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    gripper_scale: float = 100.0

    def action(self, action: RobotAction) -> RobotAction:
        """Convert 10D relative action to absolute EE pose."""
        # Get current observation for FK
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for computing current EE pose")

        # Extract current joint positions
        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )

        if len(q_raw) == 0:
            raise ValueError("No joint positions found in observation")

        # Forward kinematics to get current EE pose
        t_curr = self.kinematics.forward_kinematics(q_raw)

        # Get 10D relative action from policy
        # The action can be provided as:
        # - "rel_pose" key with numpy array of shape (10,)
        # - Individual keys: rel_dx, rel_dy, rel_dz, rel_rot6d_0..5, rel_gripper
        if "rel_pose" in action:
            rel_pose_10d = action.pop("rel_pose")
            if isinstance(rel_pose_10d, np.ndarray):
                rel_pose_10d = rel_pose_10d.flatten()
            else:
                rel_pose_10d = np.array(rel_pose_10d, dtype=float)
        else:
            # Build from individual keys
            rel_pose_10d = np.zeros(10, dtype=float)
            rel_pose_10d[0] = action.pop("rel_dx", 0.0)
            rel_pose_10d[1] = action.pop("rel_dy", 0.0)
            rel_pose_10d[2] = action.pop("rel_dz", 0.0)
            for i in range(6):
                rel_pose_10d[3 + i] = action.pop(f"rel_rot6d_{i}", 0.0)
            rel_pose_10d[9] = action.pop("rel_gripper", 0.0)

        # Convert relative 10D to transformation matrix
        t_rel = pose10d_to_mat(rel_pose_10d)

        # Chain: T_target = T_current @ T_rel (UMI-style)
        t_target = t_curr @ t_rel

        # Extract position and rotation (as rotation vector)
        target_pos = t_target[:3, 3]
        target_rotmat = t_target[:3, :3]
        target_rotvec = Rotation.from_matrix(target_rotmat).as_rotvec()

        # Scale gripper from [0,1] to [0,100]
        gripper_pos = np.clip(rel_pose_10d[9] * self.gripper_scale, 0.0, 100.0)

        # Output absolute EE pose for downstream processors
        action["ee.x"] = float(target_pos[0])
        action["ee.y"] = float(target_pos[1])
        action["ee.z"] = float(target_pos[2])
        action["ee.wx"] = float(target_rotvec[0])
        action["ee.wy"] = float(target_rotvec[1])
        action["ee.wz"] = float(target_rotvec[2])
        action["ee.gripper_pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Remove relative pose features and add absolute EE pose features."""
        # Remove input features
        for key in ["rel_pose", "rel_dx", "rel_dy", "rel_dz", "rel_gripper"]:
            features[PipelineFeatureType.ACTION].pop(key, None)
        for i in range(6):
            features[PipelineFeatureType.ACTION].pop(f"rel_rot6d_{i}", None)

        # Add output features
        for feat in ["x", "y", "z", "wx", "wy", "wz"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features


@ProcessorStepRegistry.register("relative_10d_accumulated_to_absolute_ee")
@dataclass
class Relative10DAccumulatedToAbsoluteEE(RobotActionProcessorStep):
    """
    Converts 10D relative EE actions to absolute EE poses using chunk base pose.

    This variant is designed for UMI-style action chunking where:
    - Each action in a chunk is relative to the SAME base pose (chunk start)
    - Actions are NOT accumulated/chained sequentially
    - A "chunk_base_pose" is maintained in complementary_data (set at chunk start)

    The processor:
    1. Gets the chunk base pose from complementary_data
    2. Converts the 10D relative action to a transformation matrix
    3. Applies: T_target = T_base @ T_rel (where T_base is the chunk start pose)
    4. Outputs absolute EE pose with keys: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos

    Note: Despite the class name containing "Accumulated", this does NOT accumulate.
    For RelativeEEDataset format, all actions in a chunk are relative to the base.

    Attributes:
        gripper_scale: Scale factor for gripper (default 100 for [0,1] -> [0,100]).
    """

    gripper_scale: float = 100.0

    def action(self, action: RobotAction) -> RobotAction:
        """Convert 10D relative action to absolute EE pose using chunk base."""
        # Get chunk base pose from complementary data
        complementary_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        t_base = complementary_data.get("chunk_base_pose")

        # Fallback to old key name for backward compatibility
        if t_base is None:
            t_base = complementary_data.get("accumulated_ee_pose")

        if t_base is None:
            raise ValueError(
                "chunk_base_pose not found in complementary_data. "
                "Make sure to set it before calling this processor."
            )

        # Get 10D relative action from policy
        if "rel_pose" in action:
            rel_pose_10d = action.pop("rel_pose")
            if isinstance(rel_pose_10d, np.ndarray):
                rel_pose_10d = rel_pose_10d.flatten()
            else:
                rel_pose_10d = np.array(rel_pose_10d, dtype=float)
        else:
            # Build from individual keys
            rel_pose_10d = np.zeros(10, dtype=float)
            rel_pose_10d[0] = action.pop("rel_dx", 0.0)
            rel_pose_10d[1] = action.pop("rel_dy", 0.0)
            rel_pose_10d[2] = action.pop("rel_dz", 0.0)
            for i in range(6):
                rel_pose_10d[3 + i] = action.pop(f"rel_rot6d_{i}", 0.0)
            rel_pose_10d[9] = action.pop("rel_gripper", 0.0)

        # Convert relative 10D to transformation matrix
        t_rel = pose10d_to_mat(rel_pose_10d)

        # Apply from base: T_target = T_base @ T_rel
        # All actions in the chunk are relative to T_base, NOT chained sequentially
        t_target = t_base @ t_rel

        # Extract position and rotation (as rotation vector)
        target_pos = t_target[:3, 3]
        target_rotmat = t_target[:3, :3]
        target_rotvec = Rotation.from_matrix(target_rotmat).as_rotvec()

        # Scale gripper from [0,1] to [0,100]
        gripper_pos = np.clip(rel_pose_10d[9] * self.gripper_scale, 0.0, 100.0)

        # Output absolute EE pose for downstream processors
        action["ee.x"] = float(target_pos[0])
        action["ee.y"] = float(target_pos[1])
        action["ee.z"] = float(target_pos[2])
        action["ee.wx"] = float(target_rotvec[0])
        action["ee.wy"] = float(target_rotvec[1])
        action["ee.wz"] = float(target_rotvec[2])
        action["ee.gripper_pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Remove relative pose features and add absolute EE pose features."""
        # Remove input features
        for key in ["rel_pose", "rel_dx", "rel_dy", "rel_dz", "rel_gripper"]:
            features[PipelineFeatureType.ACTION].pop(key, None)
        for i in range(6):
            features[PipelineFeatureType.ACTION].pop(f"rel_rot6d_{i}", None)

        # Add output features
        for feat in ["x", "y", "z", "wx", "wy", "wz"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features

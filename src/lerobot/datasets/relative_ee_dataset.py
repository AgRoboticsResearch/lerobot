#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. All rights reserved.
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

"""Dataset wrapper for transforming absolute end-effector poses to relative with SE(3) transformation.

This implementation follows UMI's approach for relative pose calculation:
- Uses 4x4 homogeneous transformation matrices
- Computes relative poses via: T_rel = T_curr^{-1} @ T_future
- Uses 6D rotation representation (first two columns of rotation matrix)
- Both observations AND actions are relative to current pose (UMI-style)

Reference: https://github.com/real-stanford/universal_manipulation_interface
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.rotation import Rotation
import torch
import numpy as np


# ============================================================================
# Pose Utility Functions (following UMI's umi/common/pose_util.py)
# ============================================================================

def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert 6D pose (position + axis-angle rotation) to 4x4 transformation matrix.

    Args:
        pose: Array of shape (6,) or (..., 6) where:
            - pose[..., :3] is position [x, y, z]
            - pose[..., 3:] is axis-angle rotation [rx, ry, rz]

    Returns:
        4x4 homogeneous transformation matrix, shape (..., 4, 4)
    """
    pos = pose[..., :3]
    rotvec = pose[..., 3:]

    # Convert rotation vector to rotation matrix
    rot = Rotation.from_rotvec(rotvec).as_matrix()

    # Build 4x4 homogeneous matrix
    mat = np.eye(4, dtype=pose.dtype)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat


def mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    """Convert 4x4 transformation matrix to 10D pose representation.

    The 10D representation consists of:
    - 3D position
    - 6D rotation (first two ROWS of rotation matrix, following UMI's convention)

    The third row of the rotation matrix can be reconstructed via cross product.

    Note: The Zhou et al. 2019 paper describes using columns, but UMI's code
    uses rows. We follow UMI's code convention here for compatibility.

    Args:
        mat: 4x4 transformation matrix, shape (..., 4, 4)

    Returns:
        10D pose array, shape (..., 10)
    """
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    # Take first two ROWS to match UMI's rot6d convention
    # rot6d[:3] = first row [R[0,0], R[0,1], R[0,2]], rot6d[3:] = second row [R[1,0], R[1,1], R[1,2]]
    batch_dim = rotmat.shape[:-2]
    rot6d = rotmat[..., :2, :].copy().reshape(batch_dim + (6,))
    return np.concatenate([pos, rot6d], axis=-1)


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    This reconstructs the third row via cross product and orthonormalization,
    following UMI's actual implementation (row-based convention).

    Note: The Zhou et al. 2019 paper describes using columns, but UMI's code
    uses rows. We follow UMI's code convention here for compatibility.

    Args:
        rot6d: 6D rotation array, shape (..., 6)

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
        pose10d: 10D pose array, shape (..., 10) where:
            - pose10d[..., :3] is position
            - pose10d[..., 3:] is 6D rotation

    Returns:
        4x4 transformation matrix, shape (..., 4, 4)
    """
    pos = pose10d[..., :3]
    rot6d = pose10d[..., 3:]
    rotmat = rot6d_to_mat(rot6d)

    mat = np.eye(4, dtype=pose10d.dtype)
    mat[:3, :3] = rotmat
    mat[:3, 3] = pos
    return mat


def convert_pose_mat_rep(pose_mat: np.ndarray, base_pose_mat: np.ndarray,
                         pose_rep: str = 'relative', backward: bool = False) -> np.ndarray:
    """Transform pose between different representations.

    Following UMI's diffusion_policy/common/pose_repr_util.py implementation.

    Args:
        pose_mat: Target pose as 4x4 transformation matrix
        base_pose_mat: Reference (base) pose as 4x4 transformation matrix
        pose_rep: Type of representation:
            - 'abs': Return absolute pose (no transformation)
            - 'relative': Compute T_rel = T_base^{-1} @ T_pose
        backward: If True, compute inverse transformation (for inference)

    Returns:
        Transformed pose as 4x4 transformation matrix
    """
    if not backward:
        # Training: absolute -> relative
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'relative':
            # T_rel = T_base^{-1} @ T_pose
            return np.linalg.inv(base_pose_mat) @ pose_mat
        else:
            raise ValueError(f"Unsupported pose_rep: {pose_rep}")
    else:
        # Inference: relative -> absolute
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'relative':
            # T_abs = T_base @ T_rel
            return base_pose_mat @ pose_mat
        else:
            raise ValueError(f"Unsupported pose_rep: {pose_rep}")


class RelativeEEDataset(LeRobotDataset):
    """
    Wraps LeRobotDataset to transform absolute end-effector poses to relative.

    This dataset performs proper SE(3) transformation following UMI's approach:
    - Observations: either 6D joint positions (use_joint_obs=True) or 10D identity
      (use_joint_obs=False) where current pose is the reference frame
    - Actions: future EE poses relative to current pose, represented as 10D
      [delta.xyz(3), rot6d(6), gripper(1)]
    - All actions in a chunk are relative to the same base pose (chunk start),
      NOT chained sequentially

    Expected dataset columns (produced by convert_joint_to_ee_dataset.py):
    - ``observation.state``: 6D joints at current frame (unchanged from source)
    - ``observation.ee``: 7D EE pose at current frame [x,y,z, wx,wy,wz, gripper]
    - ``action``: 7D EE pose at next frame [x,y,z, wx,wy,wz, gripper]
    - ``observation.images.*``: camera images

    T_current reads directly from ``observation.ee[idx]`` (no idx tricks).
    T_future reads from ``action`` column via standard delta_timestamps.

    Metadata shapes (for policy architecture):
    - observation.state: (6,) with use_joint_obs=True, (10,) with use_joint_obs=False
    - action: (10,) - relative EE representation
    - observation.images.*: (C, H, W) - single frame dimension

    Args:
        obs_state_horizon: Number of historical timesteps to include in observation state.
            Default is 2, matching UMI's low_dim_obs_horizon.
        obs_down_sample_steps: Downsampling factor for observation history (UMI-style).
            Default is 1 (consecutive frames). Use 3 to match UMI's default.
        gripper_lower_deg: Lower bound for gripper in degrees (default: 0.0).
            Stored in metadata for deployment denormalization.
        gripper_upper_deg: Upper bound for gripper in degrees (default: 100.0).
            Stored in metadata for deployment denormalization.
        use_joint_obs: If True, use 6D joints as observation input.
            If False, use 10D identity (current = reference frame).
    """

    def __init__(self, *args, obs_state_horizon: int = 2,
                 obs_down_sample_steps: int = 1,
                 gripper_lower_deg: float = 0.0,
                 gripper_upper_deg: float = 100.0,
                 use_joint_obs: bool = False,
                 **kwargs):
        """
        Initialize the dataset wrapper and update metadata for the new shapes.

        Following UMI's design:
        - observation.state shape changes from (7,) to (10,) when use_joint_obs=False
          Current pose becomes identity: [0,0,0, 1,0,0,0,1,0, gripper]
        - When use_joint_obs=True, observation.state stays as [6] joints (unchanged)
        - action shape always changes from (7,) to (10,) - relative EE representation
        - action.ee is removed from features/stats so the policy doesn't predict it

        Args:
            obs_state_horizon: Number of historical timesteps to include in observation state.
                Default is 2, matching UMI's low_dim_obs_horizon.
            obs_down_sample_steps: Downsampling factor for observation history (UMI-style).
                - 1 (default): Consecutive frames [t-1, t]
                - 3: Skip frames [t-3, t] - provides ~50ms delta at 60Hz (UMI default)
                - Higher values increase temporal receptive field but reduce frame count
        """
        import torch.utils.data

        self.obs_state_horizon = obs_state_horizon
        self.obs_down_sample_steps = obs_down_sample_steps
        self.gripper_lower_deg = gripper_lower_deg
        self.gripper_upper_deg = gripper_upper_deg
        self.use_joint_obs = use_joint_obs

        # Extract custom kwargs before passing to parent
        compute_stats = kwargs.pop('compute_stats', True)
        # num_stat_samples=0 means use all samples (UMI's approach, default)
        # >0 uses random subset for faster stats computation
        num_stat_samples = kwargs.pop('num_stat_samples', 0)

        super().__init__(*args, **kwargs)

        # NOTE: Metadata shapes stay as 1D/3D - NOT modified to include temporal dimension
        # The temporal dimension is handled in __getitem__ data loading, not in metadata
        # This is because the policy factory expects:
        #   - state: (D,) not (T, D)
        #   - images: (C, H, W) not (T, C, H, W)
        #
        # The actual data returned by __getitem__ has temporal dimension, which is then
        # flattened by TemporalFlattenProcessor during preprocessing.

        # Set metadata shapes — these are fixed by this class, not derived from the parquet.
        if hasattr(self, 'meta') and hasattr(self.meta, 'info'):
            features = self.meta.info.setdefault('features', {})

            if self.use_joint_obs:
                # Observation: 15D = 6D joints + 9D EE pose (pos3 + rot6d, matching action rotation format)
                features['observation.state'] = {
                    'dtype': 'float32',
                    'shape': [15],
                    'names': [
                        'shoulder_pan', 'shoulder_lift', 'elbow_flex',
                        'wrist_flex', 'wrist_roll', 'gripper',
                        'ee_x', 'ee_y', 'ee_z',
                        'ee_rot6d.0', 'ee_rot6d.1', 'ee_rot6d.2',
                        'ee_rot6d.3', 'ee_rot6d.4', 'ee_rot6d.5',
                    ],
                }
            else:
                # Observation: 10D identity (current pose = reference frame)
                features['observation.state'] = {
                    'dtype': 'float32',
                    'shape': [10],
                    'names': [
                        'delta.x', 'delta.y', 'delta.z',
                        'rot6d.0', 'rot6d.1', 'rot6d.2', 'rot6d.3', 'rot6d.4', 'rot6d.5',
                        'gripper'
                    ],
                }

            # Action: always 10D relative EE regardless of source format
            features['action'] = {
                'dtype': 'float32',
                'shape': [10],
                'names': [
                    'delta.x', 'delta.y', 'delta.z',
                    'rot6d.0', 'rot6d.1', 'rot6d.2', 'rot6d.3', 'rot6d.4', 'rot6d.5',
                    'gripper'
                ],
            }

        # Store gripper bounds in metadata (following LeRobot's stats pattern)
        # These are used during deployment to denormalize gripper values correctly
        if hasattr(self, 'meta') and hasattr(self.meta, 'info'):
            self.meta.info['gripper_lower_deg'] = self.gripper_lower_deg
            self.meta.info['gripper_upper_deg'] = self.gripper_upper_deg

        # Expose EE target frame from dataset metadata for propagation to policy config
        self.ee_target_frame = self.meta.info.get('ee_target_frame', '')

        # Update normalization stats for the new shapes
        # Following UMI's approach: compute stats from actual transformed data
        if hasattr(self, 'meta') and hasattr(self.meta, 'stats'):
            worker_info = torch.utils.data.get_worker_info()

            if compute_stats and worker_info is None:
                # Only compute stats in main process (worker_info is None)
                print("Computing normalization stats from transformed data...")
                # num_stat_samples=0 means use all samples (UMI's approach)
                num_frames = len(self) - self.obs_state_horizon
                if num_stat_samples == 0 or num_stat_samples >= num_frames:
                    print(f"  Processing ALL {num_frames} training frames...")
                else:
                    print(f"  Processing {num_stat_samples}/{num_frames} training frames...")
                stats = self._compute_normalization_stats(num_stat_samples)

                # Update observation state stats
                if 'observation.state' in stats:
                    if 'observation.state' in self.meta.stats:
                        self.meta.stats['observation.state'] = stats['observation.state']

                # Update action stats
                if 'action' in stats:
                    if 'action' in self.meta.stats:
                        self.meta.stats['action'] = stats['action']

                print("Stats computed successfully!")
            elif compute_stats and worker_info is not None:
                # In worker process - stats should already be computed by main process
                pass
            else:
                # If not computing stats, remove to avoid shape mismatch
                if 'observation.state' in self.meta.stats:
                    del self.meta.stats['observation.state']
                if 'action' in self.meta.stats:
                    del self.meta.stats['action']

    def _compute_normalization_stats(self, num_samples: int = 1000) -> dict:
        """
        Compute normalization statistics from the actual transformed data.

        Reads EE data from observation.ee (current frame) and action (future frame)
        columns, computes relative SE(3) transforms, and collects statistics.

        Args:
            num_samples: Number of samples to collect statistics from.
                0 means use all samples.

        Returns:
            Dictionary with 'observation.state' and/or 'action' stats
        """
        import torch
        from lerobot.datasets.utils import get_delta_indices

        if self.delta_timestamps is not None:
            delta_indices = get_delta_indices(self.delta_timestamps, self.fps)
        else:
            delta_indices = {}

        action_delta_indices = delta_indices.get("action", [0])

        available_samples = max(1, len(self.hf_dataset) - self.obs_state_horizon)

        if num_samples == 0 or num_samples >= available_samples:
            num_samples = available_samples
            indices = range(available_samples)
        else:
            indices = np.random.choice(available_samples, num_samples, replace=False)

        obs_list = []
        actions_list = []
        is_pad_list = []

        print_interval = max(1, num_samples // 10)
        next_print = print_interval

        for i, idx in enumerate(indices):
            if i >= next_print or i == num_samples - 1:
                progress = (i + 1) / num_samples * 100
                print(f"  Progress: {i + 1}/{num_samples} ({progress:.1f}%)")
                next_print = min(i + print_interval, num_samples)

            # T_current from observation.ee at current frame — direct read, no idx tricks
            current_ee = torch.tensor(self.hf_dataset[idx]['observation.ee'], dtype=torch.float32)
            current_pose_6d = torch.cat([current_ee[:3], current_ee[3:6]]).numpy()
            T_current = pose_to_mat(current_pose_6d)

            # Episode boundary info
            current_episode = self.hf_dataset[idx]['episode_index']
            ep = self.meta.episodes[current_episode.item()]
            ep_start = ep["dataset_from_index"]
            ep_end = ep["dataset_to_index"]

            # Get future EE actions — pad with last frame if needed
            actions_for_idx = []
            is_pad_for_idx = []
            last_valid_action = None
            for delta_idx in action_delta_indices:
                target_idx = idx + delta_idx
                if target_idx >= ep_start and target_idx < ep_end:
                    target_action = self.hf_dataset[target_idx]['action']
                    action_tensor = torch.as_tensor(target_action, dtype=torch.float32)
                    actions_for_idx.append(action_tensor)
                    last_valid_action = action_tensor
                    is_pad_for_idx.append(False)
                else:
                    if last_valid_action is None:
                        last_valid_action = torch.tensor(
                            self.hf_dataset[min(ep_end - 1, len(self.hf_dataset) - 1)]['action'],
                            dtype=torch.float32
                        )
                    actions_for_idx.append(last_valid_action)
                    is_pad_for_idx.append(True)

            actions = torch.stack(actions_for_idx, dim=0)

            # Observation based on mode
            if self.use_joint_obs:
                # 15D: 6D joints + 9D EE pose (pos3 + rot6d, matching action representation)
                joints = np.array(self.hf_dataset[idx]['observation.state'], dtype=np.float32)
                T_ee = pose_to_mat(current_pose_6d)
                ee_9d = mat_to_pose10d(T_ee)
                obs_list.append(np.concatenate([joints, ee_9d]))
            else:
                identity_obs = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, current_ee[6].item()], dtype=np.float32)
                obs_list.append(identity_obs)

            # Compute relative actions
            relative_actions_list = []
            for t in range(actions.shape[0]):
                future_ee = actions[t]
                future_pose_6d = torch.cat([future_ee[:3], future_ee[3:6]]).numpy()
                T_future = pose_to_mat(future_pose_6d)
                T_rel = convert_pose_mat_rep(T_future, T_current, pose_rep='relative')
                pose_9d = mat_to_pose10d(T_rel)
                relative_actions_list.append(np.concatenate([
                    pose_9d,
                    [future_ee[6].item()]
                ]))

            relative_actions = np.stack(relative_actions_list, axis=0)
            actions_list.append(relative_actions)
            is_pad_list.append(is_pad_for_idx)

        all_obs = np.stack(obs_list, axis=0)
        all_actions = np.concatenate(actions_list, axis=0)
        all_is_pad = np.concatenate(is_pad_list, axis=0)
        valid_actions = all_actions[~all_is_pad]

        def compute_stats(all_data, override_rot6d=True):
            mean = all_data.mean(axis=0)
            std = all_data.std(axis=0)
            min_val = all_data.min(axis=0)
            max_val = all_data.max(axis=0)

            # Override rotation 6D stats for identity/rot6d normalization (10D obs/action)
            if override_rot6d and all_data.shape[-1] == 10:
                mean[3:9] = 0.0
                std[3:9] = 1.0
                min_val[3:9] = -1.0
                max_val[3:9] = 1.0

            return {
                'mean': torch.tensor(mean, dtype=torch.float32),
                'std': torch.tensor(std, dtype=torch.float32),
                'min': torch.tensor(min_val, dtype=torch.float32),
                'max': torch.tensor(max_val, dtype=torch.float32),
            }

        stats = {
            'observation.state': compute_stats(all_obs, override_rot6d=not self.use_joint_obs),
            'action': compute_stats(valid_actions),
        }

        return stats

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample with relative transformation applied.

        T_current is read directly from observation.ee[idx] (EE at current frame).
        T_future comes from item['action'] (EE at future frames, fetched via delta_timestamps).

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
            - observation.state:
              - use_joint_obs=True: 15D (6D joints + 9D EE pose in rot6d format)
              - use_joint_obs=False: 10D identity (current = reference frame)
            - observation.images.*: Image history of shape (obs_state_horizon, C, H, W)
            - action: Relative actions of shape (action_horizon, 10)
            - action_is_pad: Boolean tensor indicating padded timesteps
            - All other original keys (timestamps, etc.) unchanged
        """
        # Get original item with temporal batching (includes action_is_pad from parent)
        item = LeRobotDataset.__getitem__(self, idx)

        # Collect and stack image history
        if hasattr(self.meta, 'camera_keys'):
            for cam_key in self.meta.camera_keys:
                if cam_key in item:
                    current_img = item[cam_key]
                    imgs_to_stack = []
                    for t_offset in range(self.obs_state_horizon):
                        hist_idx = idx - (self.obs_state_horizon - 1 - t_offset) * self.obs_down_sample_steps
                        if hist_idx >= 0:
                            hist_item = LeRobotDataset.__getitem__(self, hist_idx)
                            imgs_to_stack.append(hist_item[cam_key])
                        else:
                            imgs_to_stack.append(current_img)
                    item[cam_key] = torch.stack(imgs_to_stack, dim=0)
                    if self.obs_state_horizon == 1:
                        item[cam_key] = item[cam_key].squeeze(0)

        # T_current from observation.ee at current frame — direct read, no idx tricks
        current_ee = torch.tensor(self.hf_dataset[idx]['observation.ee'], dtype=torch.float32)
        current_pose_6d = torch.cat([current_ee[:3], current_ee[3:6]]).numpy()
        T_current = pose_to_mat(current_pose_6d)

        # Future EE from action column (fetched by parent via delta_timestamps)
        ee_actions = item['action']
        if ee_actions.ndim == 1:
            ee_actions = ee_actions.unsqueeze(0)

        # Compute relative actions
        relative_actions_list = []
        for t in range(ee_actions.shape[0]):
            future_ee = ee_actions[t]
            future_pose_6d = torch.cat([future_ee[:3], future_ee[3:6]]).numpy()
            T_future = pose_to_mat(future_pose_6d)
            T_rel = convert_pose_mat_rep(T_future, T_current, pose_rep='relative')
            pose_9d = mat_to_pose10d(T_rel)
            relative_actions_list.append(np.concatenate([
                pose_9d,
                [future_ee[6].item()]
            ]))

        relative_actions = torch.tensor(
            np.stack(relative_actions_list, axis=0),
            dtype=torch.float32,
        )

        # Set observation based on mode
        if self.use_joint_obs:
            # 15D: 6D joints + 9D EE pose (pos3 + rot6d, matching action rotation representation)
            joints = item['observation.state']  # (T, 6) or (6,)
            # Convert EE from axis-angle to rot6d (same representation as actions)
            ee_pose_6d = current_pose_6d  # [x, y, z, wx, wy, wz] (numpy, from line 515)
            T_ee = pose_to_mat(ee_pose_6d)  # 4x4 matrix
            ee_9d = mat_to_pose10d(T_ee)  # [x, y, z, rot6d_0..5] (numpy, 9D)
            ee_9d_tensor = torch.tensor(ee_9d, dtype=torch.float32)
            if joints.ndim == 1:
                combined = torch.cat([joints, ee_9d_tensor])
            else:
                # Temporal: (T, 6) + (9,) -> (T, 15), EE pose is the same for all timesteps
                ee_expanded = ee_9d_tensor.unsqueeze(0).expand(joints.shape[0], -1)
                combined = torch.cat([joints, ee_expanded], dim=-1)
            item['observation.state'] = combined
        else:
            identity_obs = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, current_ee[6].item()], dtype=np.float32)
            item['observation.state'] = torch.tensor(identity_obs, dtype=torch.float32)

        item['action'] = relative_actions

        return item

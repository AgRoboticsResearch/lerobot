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
    - Both observations AND actions are relative to current pose
    - Observations: historical poses relative to current pose (provides velocity info)
    - Actions: future poses relative to current pose
    - Uses 6D rotation representation for both obs and action
    - Temporal observations: image history and proprio history are provided

    Metadata shapes (for policy architecture):
    - observation.state: (10,) - single timestep dimension
    - observation.images.*: (C, H, W) - single frame dimension

    Actual data shapes (returned by __getitem__):
    - observation.state: (obs_state_horizon, 10) - temporal, each timestep is relative to current
    - observation.images.*: (obs_state_horizon, C, H, W) - temporal image history
    - action: (action_horizon, 10) - relative future poses

    After collation into batch and TemporalFlattenProcessor:
    - observation.state: (B, T, 10) -> (B*T, 10)
    - observation.images.*: (B, T, C, H, W) -> (B*T, C, H, W)

    Args:
        obs_state_horizon: Number of historical timesteps to include in observation state.
            Default is 2, matching UMI's low_dim_obs_horizon.
        gripper_lower_deg: Lower bound for gripper in degrees (default: 0.0).
            Stored in metadata for deployment denormalization.
        gripper_upper_deg: Upper bound for gripper in degrees (default: 100.0).
            Stored in metadata for deployment denormalization.
    """

    def __init__(self, *args, obs_state_horizon: int = 2,
                 gripper_lower_deg: float = 0.0,
                 gripper_upper_deg: float = 100.0, **kwargs):
        """
        Initialize the dataset wrapper and update metadata for the new shapes.

        Following UMI's design:
        - observation.state shape changes from (7,) to (10,) - relative pose at current timestep
          Current pose becomes identity: [0,0,0, 1,0,0,0,1,0, gripper]
        - action shape changes from (7,) to (action_horizon, 10) - relative poses for future timesteps
          action_horizon is determined by delta_timestamps['action'], e.g., chunk_size=100 for ACT
        """
        import torch.utils.data

        self.obs_state_horizon = obs_state_horizon
        self.gripper_lower_deg = gripper_lower_deg
        self.gripper_upper_deg = gripper_upper_deg

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

        # Update state metadata to 10D (from 7D) - keep 1D shape
        if hasattr(self, 'meta') and hasattr(self.meta, 'info') and 'observation.state' in self.meta.info.get('features', {}):
            self.meta.info['features']['observation.state']['shape'] = [10]
            self.meta.info['features']['observation.state']['names'] = [
                'delta.x', 'delta.y', 'delta.z',
                'rot6d.0', 'rot6d.1', 'rot6d.2', 'rot6d.3', 'rot6d.4', 'rot6d.5',
                'gripper'
            ]

        # Update action shape in metadata
        # IMPORTANT: Keep as [10,] (1D) even though actual data will be (action_horizon, 10) when using delta_timestamps
        # This matches the original dataset behavior where metadata is [7,] but data is (100, 7) with action_delta_indices
        # The policy uses shape[0] to determine the input dimension, so it MUST be 10, not action_horizon
        if hasattr(self, 'meta') and hasattr(self.meta, 'info') and 'action' in self.meta.info.get('features', {}):
            self.meta.info['features']['action']['shape'] = [10]
            self.meta.info['features']['action']['names'] = [
                'delta.x', 'delta.y', 'delta.z',
                'rot6d.0', 'rot6d.1', 'rot6d.2', 'rot6d.3', 'rot6d.4', 'rot6d.5',
                'gripper'
            ]

        # Store gripper bounds in metadata (following LeRobot's stats pattern)
        # These are used during deployment to denormalize gripper values correctly
        if hasattr(self, 'meta') and hasattr(self.meta, 'info'):
            self.meta.info['gripper_lower_deg'] = self.gripper_lower_deg
            self.meta.info['gripper_upper_deg'] = self.gripper_upper_deg

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
                if 'observation.state' in self.meta.stats:
                    self.meta.stats['observation.state'] = stats['observation.state']

                # Update action stats
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

        Following UMI's approach (diffusion_policy/dataset/umi_dataset.py:214-216):
            B, T, D = data_cache[key].shape
            data_cache[key] = data_cache[key].reshape(B*T, D)  # All timesteps!

        This samples from the dataset to get real statistics of the relative SE(3) transforms.

        IMPORTANT: Access data directly from self.hf_dataset to avoid video decoding.
        Only 'observation.state' and 'action' are needed - videos are NOT accessed.

        Args:
            num_samples: Number of samples to collect statistics from

        Returns:
            Dictionary with 'observation.state' and 'action' stats
        """
        import torch
        from lerobot.datasets.utils import get_delta_indices

        # Get delta_indices for temporal batching
        if self.delta_timestamps is not None:
            delta_indices = get_delta_indices(self.delta_timestamps, self.fps)
        else:
            delta_indices = {}

        # Get action delta indices
        action_delta_indices = delta_indices.get("action", [0])
        action_horizon = len(action_delta_indices)

        # Following UMI's approach: iterate through ALL training samples
        # UMI's get_normalizer reshapes B*T data, using all available training data
        # See: diffusion_policy/dataset/umi_dataset.py:214-216
        available_samples = max(1, len(self.hf_dataset) - self.obs_state_horizon)

        # num_stat_samples=0 means use all samples (UMI's approach)
        # Otherwise use the specified number of samples
        if num_samples == 0 or num_samples >= available_samples:
            num_samples = available_samples
            indices = range(available_samples)  # Iterate all, not random choice
        else:
            # Use random sampling for faster stats computation
            indices = np.random.choice(available_samples, num_samples, replace=False)

        obs_list = []
        # For temporal observations, collect all timesteps across all samples
        obs_temporal_list = []  # Will flatten: (N * obs_state_horizon, 10)
        actions_list = []
        is_pad_list = []  # Track which action timesteps are padded

        # Progress printing
        print_interval = max(1, num_samples // 10)  # Print 10 times total
        next_print = print_interval

        for i, idx in enumerate(indices):
            # Progress printing - use i (position in indices) not idx (actual data index)
            if i >= next_print or i == num_samples - 1:
                progress = (i + 1) / num_samples * 100
                print(f"  Progress: {i + 1}/{num_samples} ({progress:.1f}%)")
                next_print = min(i + print_interval, num_samples)
            # Collect historical observation states
            obs_states = []
            for t_offset in range(self.obs_state_horizon):
                hist_idx = idx - (self.obs_state_horizon - 1 - t_offset)
                if hist_idx >= 0:
                    state_data = self.hf_dataset[hist_idx]['observation.state']
                    obs_states.append(torch.tensor(state_data, dtype=torch.float32))
                else:
                    # Pad with zeros if before episode start
                    state_data = self.hf_dataset[0]['observation.state']
                    obs_states.append(torch.zeros_like(torch.tensor(state_data, dtype=torch.float32)))

            if len(obs_states) < self.obs_state_horizon:
                continue

            # Stack observation states: (obs_state_horizon, 7)
            obs_states = torch.stack(obs_states, dim=0)

            # Get current state (most recent observation)
            current_state = obs_states[-1]  # (7,)

            # Get current episode info for boundary checking
            current_episode = self.hf_dataset[idx]['episode_index']
            ep = self.meta.episodes[current_episode.item()]
            ep_start = ep["dataset_from_index"]
            ep_end = ep["dataset_to_index"]

            # Get actions across horizon - pad with last frame if needed
            # Respect episode boundaries like parent class does
            actions_for_idx = []
            is_pad_for_idx = []
            last_valid_action = None
            for delta_idx in action_delta_indices:
                target_idx = idx + delta_idx
                # Check if target is within the same episode
                if target_idx >= ep_start and target_idx < ep_end:
                    target_action = self.hf_dataset[target_idx]['action']
                    action_tensor = torch.tensor(target_action, dtype=torch.float32)
                    actions_for_idx.append(action_tensor)
                    last_valid_action = action_tensor
                    is_pad_for_idx.append(False)
                else:
                    # Pad with last valid action (clamp to episode end)
                    # Same behavior as parent class's _get_query_indices
                    if last_valid_action is None:
                        # Should not happen if delta_indices includes 0
                        last_valid_action = torch.tensor(
                            self.hf_dataset[min(ep_end - 1, len(self.hf_dataset) - 1)]['action'],
                            dtype=torch.float32
                        )
                    actions_for_idx.append(last_valid_action)
                    is_pad_for_idx.append(True)

            # Stack actions: (action_horizon, 7)
            actions = torch.stack(actions_for_idx, dim=0)

            # Transform ALL historical observations to relative (current as reference)
            # This provides velocity information: where the EE was relative to where it is now
            # Following UMI's approach where ALL timesteps contribute to statistics
            current_pose_6d = torch.cat([
                current_state[:3],   # position
                current_state[3:6]   # rotation (axis-angle)
            ]).numpy()  # (6,)

            T_current = pose_to_mat(current_pose_6d)  # (4, 4)

            # Transform ALL historical observations to relative
            relative_obs_list = []
            for t in range(self.obs_state_horizon):
                hist_state = obs_states[t]  # Already collected above

                hist_pose_6d = torch.cat([
                    hist_state[:3],   # position
                    hist_state[3:6]   # rotation (axis-angle)
                ]).numpy()

                T_hist = pose_to_mat(hist_pose_6d)

                # Compute relative: T_rel = T_curr^{-1} @ T_hist
                T_rel = convert_pose_mat_rep(T_hist, T_current, pose_rep='relative')

                pose_9d = mat_to_pose10d(T_rel)
                relative_obs_list.append(np.concatenate([
                    pose_9d,
                    [hist_state[6].item()]  # gripper
                ]))

            # Stack: (obs_state_horizon, 10) - keep temporal dimension for stats
            relative_obs_temporal = np.stack(relative_obs_list, axis=0)
            # Flatten for stats: add to list, will stack later
            for t in range(self.obs_state_horizon):
                obs_temporal_list.append(relative_obs_temporal[t])

            # Keep single observation for backward compatibility (current timestep only)
            # Current timestep (t=-1) is always identity
            obs_list.append(relative_obs_list[-1])  # Last one is current timestep (identity)

            # Transform all actions to relative (future -> current)
            actions_pos = actions[:, :3].numpy()
            actions_rot = actions[:, 3:6].numpy()
            actions_gripper = actions[:, 6:7].numpy()

            # Get action horizon
            action_horizon = actions.shape[0]

            relative_actions_list = []
            for t in range(action_horizon):
                future_pose_6d = np.concatenate([actions_pos[t], actions_rot[t]])
                T_future = pose_to_mat(future_pose_6d)

                # Compute relative transform: T_rel = T_curr^{-1} @ T_future
                T_rel = convert_pose_mat_rep(T_future, T_current, pose_rep='relative')

                pose_9d = mat_to_pose10d(T_rel)
                relative_action = np.concatenate([
                    pose_9d,
                    [actions_gripper[t].item()]
                ])  # (10,)

                relative_actions_list.append(relative_action)

            # Stack: (action_horizon, 10)
            relative_actions = np.stack(relative_actions_list, axis=0)
            actions_list.append(relative_actions)
            is_pad_list.append(is_pad_for_idx)

        # Stack all samples
        # For observations: use ALL temporal timesteps (UMI's approach)
        # This matches UMI's B*T flattening for statistics
        all_obs = np.stack(obs_temporal_list, axis=0)  # (N * obs_state_horizon, 10)
        # Flatten all actions: (N * action_horizon, 10)
        all_actions = np.concatenate(actions_list, axis=0)  # (N * action_horizon, 10)
        # Flatten is_pad: (N * action_horizon,)
        all_is_pad = np.concatenate(is_pad_list, axis=0)  # (N * action_horizon,)

        # Filter out padded frames for accurate statistics on real data only
        # Padded frames have repeated data which would skew statistics
        valid_actions = all_actions[~all_is_pad]  # Only non-padded frames

        # Note: obs_temporal_list already contains all timesteps, no need to filter
        # Historical obs_states are computed from actual data (or zero-padded at episode start)
        # The zero-padding at episode start is intentional and provides valid statistics

        def compute_stats(all_data):
            mean = all_data.mean(axis=0)
            std = all_data.std(axis=0)
            min_val = all_data.min(axis=0)
            max_val = all_data.max(axis=0)

            # Override rotation 6D stats for identity normalization (following UMI)
            # Indices [3:9] are the 6D rotation representation
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
            'observation.state': compute_stats(all_obs),
            'action': compute_stats(valid_actions),  # Use only non-padded frames for accurate stats
        }

        return stats

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample with relative transformation applied (UMI-style).

        Following UMI's diffusion_policy/dataset/umi_dataset.py __getitem__:
        - Both observations AND actions are transformed relative to current pose
        - Observations show "where the gripper WAS relative to where it IS NOW" (velocity info)
        - Actions show "where the gripper WILL BE relative to where it IS NOW"

        The current pose becomes the identity frame [0,0,0, 1,0,0,0,1,0, gripper].

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
            - observation.state: Relative observations of shape (obs_state_horizon, 10)
              Current timestep (last index) is identity [0,0,0, 1,0,0,0,1,0, gripper]
              Historical timesteps show where the EE was relative to current (velocity)
            - observation.images.*: Image history of shape (obs_state_horizon, C, H, W)
              Each timestep contains the image at that point in history
            - action: Relative actions of shape (action_horizon, 10)
              action_horizon is determined by delta_timestamps['action'] (e.g., 100 for ACT)
              Each action is a future timestep relative to current
            - action_is_pad: Boolean tensor indicating padded timesteps
            - All other original keys (timestamps, etc.) unchanged
        """
        # Get original item with temporal batching (includes action_is_pad from parent)
        item = LeRobotDataset.__getitem__(self, idx)

        # Get current state (most recent observation)
        current_state = item['observation.state']

        # Collect historical observation states
        obs_states = []
        for t_offset in range(self.obs_state_horizon):
            hist_idx = idx - (self.obs_state_horizon - 1 - t_offset)
            if hist_idx >= 0:
                hist_item = LeRobotDataset.__getitem__(self, hist_idx)
                obs_states.append(hist_item['observation.state'])
            else:
                # Pad with current state if before episode start
                obs_states.append(current_state.clone())

        # Stack observation states: (obs_state_horizon, 7)
        obs_states = torch.stack(obs_states, dim=0)

        # Collect and stack image history (UMI-style)
        # Following UMI's approach where image history is provided for temporal observations
        if hasattr(self.meta, 'camera_keys'):
            for cam_key in self.meta.camera_keys:
                if cam_key in item:
                    current_img = item[cam_key]  # Shape: (C, H, W)
                    imgs_to_stack = []

                    # Collect historical images (oldest to newest)
                    for t_offset in range(self.obs_state_horizon):
                        hist_idx = idx - (self.obs_state_horizon - 1 - t_offset)
                        if hist_idx >= 0:
                            hist_item = LeRobotDataset.__getitem__(self, hist_idx)
                            imgs_to_stack.append(hist_item[cam_key])
                        else:
                            # Pad with current image if before episode start
                            imgs_to_stack.append(current_img)

                    # Stack: (obs_state_horizon, C, H, W)
                    item[cam_key] = torch.stack(imgs_to_stack, dim=0)

        # Get actions across horizon
        actions = item['action']
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        action_horizon = actions.shape[0]

        # Get action_is_pad if provided by parent class
        # Note: action_is_pad is already in the item from LeRobotDataset
        # and will be preserved automatically when we return the item

        # Current pose (most recent observation)
        current_pose_6d = torch.cat([
            current_state[:3],   # position
            current_state[3:6]   # rotation (axis-angle)
        ]).numpy()  # (6,)

        T_current = pose_to_mat(current_pose_6d)  # (4, 4)

        # Transform ALL historical observations to relative (current as reference)
        # This provides velocity information: where the EE was relative to where it is now
        # Following UMI's approach where all timesteps are processed
        relative_obs_list = []
        for t in range(self.obs_state_horizon):
            hist_state = obs_states[t]  # Already collected above

            hist_pose_6d = torch.cat([
                hist_state[:3],   # position
                hist_state[3:6]   # rotation (axis-angle)
            ]).numpy()

            T_hist = pose_to_mat(hist_pose_6d)

            # Compute relative: T_rel = T_curr^{-1} @ T_hist
            T_rel = convert_pose_mat_rep(T_hist, T_current, pose_rep='relative')

            pose_9d = mat_to_pose10d(T_rel)
            relative_obs_list.append(np.concatenate([
                pose_9d,
                [hist_state[6].item()]  # gripper
            ]))

        # Stack: (obs_state_horizon, 10) - keep temporal dimension
        relative_obs = torch.tensor(
            np.stack(relative_obs_list, axis=0),
            dtype=current_state.dtype,
            device=current_state.device
        )  # shape: (obs_state_horizon, 10)

        # Transform actions to relative (future -> current)
        # Handle multiple action timesteps (chunk_size)
        actions_pos = actions[:, :3].numpy()
        actions_rot = actions[:, 3:6].numpy()
        actions_gripper = actions[:, 6:7]

        # Get action horizon from the data
        action_horizon = actions.shape[0]

        relative_actions_list = []
        for t in range(action_horizon):
            future_pose_6d = np.concatenate([actions_pos[t], actions_rot[t]])
            T_future = pose_to_mat(future_pose_6d)

            # Compute relative transform: T_rel = T_curr^{-1} @ T_future
            T_rel = convert_pose_mat_rep(T_future, T_current, pose_rep='relative')

            pose_9d = mat_to_pose10d(T_rel)
            pose_with_gripper = np.concatenate([
                pose_9d,
                [actions_gripper[t].item()]
            ])  # (10,)

            relative_actions_list.append(pose_with_gripper)

        # Stack into tensor: (action_horizon, 10)
        # When action_delta_indices=[0,1,...,99], action_horizon=100 for chunk_size
        relative_actions = torch.tensor(
            np.stack(relative_actions_list, axis=0),
            dtype=actions.dtype,
            device=actions.device
        )  # shape: (action_horizon, 10)

        # Update item with transformed data
        item['observation.state'] = relative_obs
        item['action'] = relative_actions

        # Note: action_is_pad is already in the item from parent class
        # It indicates which timesteps were padded (clamped to episode boundary)

        return item

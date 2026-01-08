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
    - 6D rotation (first two columns of rotation matrix)

    The third column of the rotation matrix can be reconstructed via cross product.

    Args:
        mat: 4x4 transformation matrix, shape (..., 4, 4)

    Returns:
        10D pose array, shape (..., 10)
    """
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    rot6d = rotmat[..., :2, :].reshape(rotmat.shape[:-2] + (6,))
    return np.concatenate([pos, rot6d], axis=-1)


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    This reconstructs the third column via cross product and orthonormalization,
    following the approach in Zhou et al. 2019.

    Args:
        rot6d: 6D rotation array, shape (..., 6)

    Returns:
        3x3 rotation matrix, shape (..., 3, 3)
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]

    # Normalize first column
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)

    # Make second column orthogonal to first
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)

    # Third column via cross product
    b3 = np.cross(b1, b2, axis=-1)

    # Stack into rotation matrix
    rotmat = np.stack([b1, b2, b3], axis=-1)
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

    Transforms:
    - observation.state: Relative SE(3) pose - shape (10,)
      Current timestep observation is always identity: [0,0,0, 1,0,0,0,1,0, gripper]
      Contains: [delta.x, delta.y, delta.z, rot6d.0, ..., rot6d.5, gripper]
    - action: Relative SE(3) poses (future -> current) - shape (action_horizon, 10)
      action_horizon is determined by delta_timestamps['action'] (e.g., 100 for ACT's chunk_size)
      Contains: [delta.x, delta.y, delta.z, rot6d.0, ..., rot6d.5, gripper] for each timestep

    The 6D rotation representation (Zhou et al. 2019) takes the first two columns of the
    rotation matrix. The third column can be reconstructed via cross product and normalization,
    ensuring the matrix is orthonormal.

    Args:
        obs_state_horizon: Number of historical timesteps to include in observation state.
            Default is 2, matching UMI's low_dim_obs_horizon.
    """

    def __init__(self, *args, obs_state_horizon: int = 2, **kwargs):
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

        # Extract custom kwargs before passing to parent
        compute_stats = kwargs.pop('compute_stats', True)
        num_stat_samples = kwargs.pop('num_stat_samples', 1000)

        super().__init__(*args, **kwargs)

        # Update observation.state shape in metadata
        # Original: (7,) -> New: (10,) - same 1D fashion as original
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

        # Update normalization stats for the new shapes
        # Following UMI's approach: compute stats from actual transformed data
        if hasattr(self, 'meta') and hasattr(self.meta, 'stats'):
            worker_info = torch.utils.data.get_worker_info()

            if compute_stats and worker_info is None:
                # Only compute stats in main process (worker_info is None)
                print("Computing normalization stats from transformed data...")
                print(f"  Sampling {min(num_stat_samples, len(self))} frames...")
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

        # Sample random indices
        # Adjust num_samples based on available data (accounting for action horizon)
        available_samples = max(1, len(self.hf_dataset) - max(action_delta_indices))
        num_samples = min(num_samples, available_samples)
        indices = np.random.choice(available_samples, num_samples, replace=False)

        obs_list = []
        actions_list = []

        for idx in indices:
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

            # Get actions across horizon
            actions_for_idx = []
            for delta_idx in action_delta_indices:
                target_idx = idx + delta_idx
                if target_idx < len(self.hf_dataset):
                    target_action = self.hf_dataset[target_idx]['action']
                    actions_for_idx.append(torch.tensor(target_action, dtype=torch.float32))

            if len(actions_for_idx) < action_horizon:
                continue

            # Stack actions: (action_horizon, 7)
            actions = torch.stack(actions_for_idx, dim=0)

            # Transform observations to relative
            # Current pose as reference
            current_pose_6d = torch.cat([
                current_state[:3],   # position
                current_state[3:6]   # rotation (axis-angle)
            ]).numpy()  # (6,)

            T_current = pose_to_mat(current_pose_6d)  # (4, 4)

            # Observation: current -> current = identity
            # [0,0,0, 1,0,0,0,1,0, gripper]
            relative_obs = np.array([
                0.0, 0.0, 0.0,  # position (identity)
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # rot6d (identity rotation)
                current_state[6].item()  # gripper
            ])  # (10,)
            obs_list.append(relative_obs)

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

        # Stack all samples
        all_obs = np.stack(obs_list, axis=0)  # (N, 10)
        # Flatten all actions: (N * action_horizon, 10)
        all_actions = np.concatenate(actions_list, axis=0)  # (N * action_horizon, 10)

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
            'action': compute_stats(all_actions),
        }

        return stats

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample with relative transformation applied (UMI-style).

        Following UMI's diffusion_policy/dataset/umi_dataset.py __getitem__:
        - Both observations AND actions are transformed relative to current pose
        - Observations show "where the gripper WAS relative to where it IS NOW"
        - Actions show "where the gripper WILL BE relative to where it IS NOW"

        The current pose becomes the identity frame [0,0,0, 0,0,0,0,0,0, gripper].

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
            - observation.state: Relative observation of shape (10,)
              Current timestep is identity [0,0,0, 1,0,0,0,1,0, gripper]
            - action: Relative actions of shape (action_horizon, 10)
              action_horizon is determined by delta_timestamps['action'] (e.g., 100 for ACT)
              Each action is a future timestep relative to current
            - All other original keys (images, timestamps, etc.) unchanged
        """
        # Get original item with temporal batching
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

        # Get actions across horizon
        actions = item['action']
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        action_horizon = actions.shape[0]

        # Current pose (most recent observation)
        current_pose_6d = torch.cat([
            current_state[:3],   # position
            current_state[3:6]   # rotation (axis-angle)
        ]).numpy()  # (6,)

        T_current = pose_to_mat(current_pose_6d)  # (4, 4)

        # Transform observation to relative (current -> current = identity)
        # Current pose becomes identity: [0,0,0, 0,0,0,0,0,0, gripper]
        relative_obs = torch.tensor(
            np.array([0.0, 0.0, 0.0,  # position (identity)
                      1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # rot6d (identity rotation)
                      current_state[6].item()]),  # gripper
            dtype=current_state.dtype,
            device=current_state.device
        )  # shape: (10,)

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

        return item

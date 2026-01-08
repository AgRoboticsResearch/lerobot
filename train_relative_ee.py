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

"""
Training script for ACT policy with relative end-effector pose dataset (UMI-style).

This script uses RelativeEEDataset which transforms absolute ee poses to wrist-relative
poses using proper SE(3) transformation following UMI's approach:

Observations (UMI-style):
- Historical poses relative to current pose: T_rel = T_curr^{-1} @ T_hist
- Provides velocity/trajectory information
- Shape: (obs_state_horizon, 10) where 10 = [dx, dy, dz, rot6d_0, ..., rot6d_5, gripper]
- Current timestep is identity [0,0,0, 0,0,0,0,0,0, gripper]

Actions:
- Future poses relative to current pose: T_rel = T_curr^{-1} @ T_future
- Shape: (action_horizon, 10)

Reference: https://github.com/real-stanford/universal_manipulation_interface
"""

import logging
import sys

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.relative_ee_dataset import RelativeEEDataset
from lerobot.scripts.lerobot_train import train


def make_relative_ee_dataset(cfg: TrainPipelineConfig, obs_state_horizon: int = 2):
    """
    Create RelativeEEDataset with proper SE(3) wrist-relative transformation (UMI-style).

    This replaces the standard make_dataset function to use RelativeEEDataset instead
    of LeRobotDataset. The dataset automatically updates its metadata to reflect the
    new action and observation shapes (10D instead of 7D).

    Args:
        cfg: Training pipeline configuration
        obs_state_horizon: Number of historical timesteps in observation state.
            Default is 2, matching UMI's low_dim_obs_horizon.

    Returns:
        RelativeEEDataset instance
    """
    # Get dataset metadata
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision
    )

    # Resolve delta timestamps from policy config
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    # Create RelativeEEDataset with wrist-relative SE(3) transformation
    dataset = RelativeEEDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        obs_state_horizon=obs_state_horizon,  # UMI-style: historical observations
    )

    logging.info(f"Created RelativeEEDataset (UMI-style)")
    logging.info(f"  obs_state_horizon: {obs_state_horizon}")
    logging.info(f"  observation.state shape: {dataset.meta.info['features']['observation.state']['shape']}")
    logging.info(f"  action shape: {dataset.meta.info['features']['action']['shape']}")
    logging.info(f"  Action names: {dataset.meta.info['features']['action'].get('names', 'N/A')}")

    return dataset


# Override the make_dataset function in the train function's scope
# This is a bit of a hack, but it allows us to use the existing training infrastructure
import lerobot.scripts.lerobot_train as train_module
original_make_dataset = train_module.make_dataset


def _make_relative_ee_dataset_wrapper(cfg: TrainPipelineConfig):
    """Wrapper that creates RelativeEEDataset with configurable obs_state_horizon."""
    # Get obs_state_horizon from policy config or use default
    obs_state_horizon = getattr(cfg.policy, 'obs_state_horizon', 2)
    return make_relative_ee_dataset(cfg, obs_state_horizon=obs_state_horizon)


train_module.make_dataset = _make_relative_ee_dataset_wrapper

# Also need to override in the factory module since it might be imported elsewhere
import lerobot.datasets.factory as factory_module
factory_module.make_dataset = _make_relative_ee_dataset_wrapper


@parser.wrap()
def train_with_relative_ee(cfg: TrainPipelineConfig):
    """
    Train ACT policy using RelativeEEDataset (UMI-style).

    This function wraps the standard training pipeline but uses RelativeEEDataset
    instead of LeRobotDataset. The rest of the training process remains unchanged.

    Following UMI's design:
    - Observations: Historical poses relative to current (provides velocity info)
    - Actions: Future poses relative to current
    - Both use 6D rotation representation

    Args:
        cfg: Training pipeline configuration containing dataset and policy settings.
    """
    # Get obs_state_horizon from policy config or use default
    obs_state_horizon = getattr(cfg.policy, 'obs_state_horizon', 2)

    # Log that we're using relative EE dataset
    logging.info("=" * 80)
    logging.info("Training with RelativeEEDataset (UMI-style: relative obs + actions)")
    logging.info("=" * 80)
    logging.info(f"Dataset: {cfg.dataset.repo_id}")
    logging.info(f"Policy: {cfg.policy.type}")
    logging.info(f"obs_state_horizon: {obs_state_horizon}")
    logging.info("Observation shape: (obs_state_horizon, 10) = relative historical poses")
    logging.info("Action shape: (action_horizon, 10) = relative future poses")
    logging.info("  Format: [dx, dy, dz, rot6d_0, ..., rot6d_5, gripper]")
    logging.info("=" * 80)

    # IMPORTANT: Following UMI's normalization approach
    # UMI uses min/max normalization (map to [-1, 1]) for position and gripper,
    # and identity normalization for rotation 6D.
    # See: diffusion_policy/dataset/umi_dataset.py:223-225
    cfg.policy.normalization_mapping["ACTION"] = NormalizationMode.MIN_MAX
    cfg.policy.normalization_mapping["STATE"] = NormalizationMode.MIN_MAX
    logging.info("Normalization: MIN_MAX (UMI-style: pos/gripper to [-1,1], rot6d identity)")

    # Call the standard training function
    # It will use our overridden make_dataset function
    train(cfg)


if __name__ == "__main__":
    train_with_relative_ee()

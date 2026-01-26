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

"""Processor to handle temporal observations without dimension concatenation.

This processor passes through temporal observations, allowing the model to handle
temporal batching (UMI-style) where each timestep is encoded independently.
"""

from dataclasses import dataclass

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="temporal_flatten_processor")
class TemporalFlattenProcessor(ProcessorStep):
    """Pass-through processor for temporal observations (UMI-style batching).

    This processor keeps temporal dimensions intact, allowing the model to handle
    temporal batching by flattening (B, T, ...) -> (B*T, ...) for encoding,
    then aggregating back to (B, T*feature_dim).

    The key difference from channel concatenation:
    - Channel concat: (B, T, C, H, W) -> (B, T*C, H, W) - modifies backbone
    - UMI batch flatten: (B, T, C, H, W) -> (B*T, C, H, W) - standard backbone

    With UMI's approach, each timestep is encoded independently through the
    standard 3-channel pretrained backbone, then features are concatenated.

    Note: This processor should be inserted early in the preprocessing pipeline,
    before normalization and other transformations.

    Args:
        obs_state_horizon: Number of temporal steps in the observation dimension.
            Default is 2, matching UMI's low_dim_obs_horizon.
    """

    obs_state_horizon: int = 2

    def __post_init__(self):
        """Validate the obs_state_horizon parameter."""
        if self.obs_state_horizon < 1:
            raise ValueError(f"obs_state_horizon must be >= 1, got {self.obs_state_horizon}")

    def __call__(self, transition):
        """Pass-through temporal observations, add action_is_pad for ACT.

        Args:
            transition: The input transition containing observations and possibly actions.

        Returns:
            The modified transition with temporal dimensions preserved and action_is_pad added.
        """
        from .core import TransitionKey

        new_transition = transition.copy()

        # Get observation from transition
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        # Keep observation as-is - temporal dimensions are preserved
        # The model wrapper will handle temporal batching

        # Add action_is_pad for ACT compatibility
        import torch
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            if action.ndim == 3:  # (B, action_horizon, D)
                action_horizon = action.shape[1]
                batch_size = action.shape[0]
                action_is_pad = torch.zeros(
                    batch_size, action_horizon,
                    dtype=torch.bool, device=action.device
                )
                comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
                comp_data["action_is_pad"] = action_is_pad
                new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data
            elif action.ndim == 2:  # (B, D)
                batch_size = action.shape[0]
                action_is_pad = torch.zeros(
                    batch_size,
                    dtype=torch.bool, device=action.device
                )
                comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
                comp_data["action_is_pad"] = action_is_pad
                new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Update feature shapes to preserve temporal dimensions.

        Args:
            features: Input feature description.

        Returns:
            Updated feature description with temporal dimensions preserved.
        """
        # Keep features as-is - temporal dimensions are preserved
        # The model wrapper handles the transformation
        return features

    def get_config(self) -> dict:
        """Return the processor configuration for serialization.

        Returns:
            Dictionary containing the processor configuration.
        """
        return {
            "obs_state_horizon": self.obs_state_horizon,
        }

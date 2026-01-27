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

"""Normalization processor that handles temporal observations."""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor.normalize_processor import _NormalizationMixin
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from .core import TransitionKey


@dataclass
@ProcessorStepRegistry.register(name="temporal_normalize_processor")
class TemporalNormalizeProcessor(_NormalizationMixin, ProcessorStep):
    """Normalization processor for temporal observations (UMI-style batching).

    This processor handles normalization of temporal observations by:
    1. Reshaping (B, T, D) -> (B*T, D) for normalization
    2. Applying standard normalization
    3. Reshaping back to (B, T, D)

    This allows each timestep to be normalized consistently while preserving
    the temporal dimension for later batching. The temporal dimension is always
    preserved (even for T=1) to ensure unified processing through TemporalACTWrapper.

    Attributes:
        features: A dictionary mapping feature names to PolicyFeature objects.
        norm_map: A dictionary mapping FeatureType to NormalizationMode.
        stats: A dictionary containing normalization statistics.
        device: The PyTorch device for tensor operations.
        obs_state_horizon: Number of temporal steps in observations.
    """

    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    eps: float = 1e-8
    normalize_observation_keys: set[str] | None = None
    obs_state_horizon: int = 2

    # Additional fields for mixin compatibility
    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)
    _stats_explicitly_provided: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Initialize the processor after dataclass construction."""
        # Call the parent's __post_init__ from _NormalizationMixin
        _NormalizationMixin.__post_init__(self)

    def __call__(self, transition):
        """Normalize temporal observations by reshaping, normalizing, and reshaping back.

        The temporal dimension is always preserved for unified processing through
        TemporalACTWrapper, regardless of obs_state_horizon value.
        """
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        new_observation = dict(observation)

        # Normalize each feature in the observation
        for key in list(new_observation.keys()):
            if key in self._tensor_stats and key in self.features:
                data = new_observation[key]
                feature_type = self.features[key].type

                # Handle temporal dimension: (B, T, D) -> (B*T, D) -> normalize -> (B, T, D)
                # Always preserve temporal dimension - T=1 will be handled by TemporalACTWrapper
                if data.ndim == 3:  # State: (B, T, D)
                    B, T, D = data.shape
                    data_flat = data.reshape(B * T, D)
                    normalized = self._apply_transform(data_flat, key, feature_type, inverse=False)
                    new_observation[key] = normalized.reshape(B, T, D)  # (B, T, D)
                elif data.ndim == 5:  # Images: (B, T, C, H, W)
                    B, T, C, H, W = data.shape
                    data_flat = data.reshape(B * T, C, H, W)
                    normalized = self._apply_transform(data_flat, key, feature_type, inverse=False)
                    new_observation[key] = normalized.reshape(B, T, C, H, W)  # (B, T, C, H, W)
                else:
                    # Standard normalization for non-temporal data
                    new_observation[key] = self._apply_transform(data, key, feature_type, inverse=False)

        new_transition[TransitionKey.OBSERVATION] = new_observation

        # Also normalize actions if needed
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None and ACTION in self._tensor_stats:
            if action.ndim == 3:  # (B, action_horizon, D)
                B, T, D = action.shape
                action_flat = action.reshape(B * T, D)
                normalized = self._apply_transform(action_flat, ACTION, FeatureType.ACTION, inverse=False)
                new_transition[TransitionKey.ACTION] = normalized.reshape(B, T, D)
            else:
                new_transition[TransitionKey.ACTION] = self._apply_transform(action, ACTION, FeatureType.ACTION, inverse=False)

        return new_transition

    def to(self, device=None, dtype=None):
        """Move the processor's normalization stats to the specified device."""
        # Use the parent class's to method from _NormalizationMixin
        if hasattr(_NormalizationMixin, 'to'):
            return _NormalizationMixin.to(self, device=device, dtype=dtype)
        # Fallback implementation
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        return self

    def state_dict(self):
        """Return the normalization statistics as a flat state dictionary."""
        if hasattr(_NormalizationMixin, 'state_dict'):
            return _NormalizationMixin.state_dict(self)
        # Fallback implementation
        flat = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()
        return flat

    def load_state_dict(self, state):
        """Load normalization statistics from a state dictionary."""
        if hasattr(_NormalizationMixin, 'load_state_dict'):
            return _NormalizationMixin.load_state_dict(self, state)
        # Fallback implementation
        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
                dtype=torch.float32, device=self.device
            )

        # Reconstruct stats dict
        self.stats = {}
        for key, tensor_dict in self._tensor_stats.items():
            self.stats[key] = {}
            for stat_name, tensor in tensor_dict.items():
                self.stats[key][stat_name] = from_tensor_to_numpy(tensor)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Return features unchanged - temporal dimensions are preserved."""
        # For temporal normalization, we keep features as-is
        # The normalization handles reshaping internally
        return features

    def get_config(self) -> dict:
        """Return the processor configuration for serialization."""
        return {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
            "obs_state_horizon": self.obs_state_horizon,
        }

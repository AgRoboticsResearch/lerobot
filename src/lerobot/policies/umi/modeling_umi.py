# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
UMI Policy Modeling for LeRobot.

This module provides the UmiPolicy class that integrates UMI diffusion policies
with LeRobot's policy system, allowing seamless use of UMI-trained models.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ...utils.logging_utils import get_logger
from ...utils.config import BaseConfig
from ..base_policy import BasePolicy
from .configuration_umi import UmiConfig

logger = get_logger(__name__)

try:
    from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
    from diffusion_policy.policy.base_image_policy import BaseImagePolicy
    from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
    from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy
    from diffusion_policy.policy.ibc_dfo_lowdim_policy import IbcDfoLowdimPolicy
    from diffusion_policy.policy.bet_lowdim_policy import BetLowdimPolicy
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.common.normalize_util import get_identity_normalizer_from_stat
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    UMI_AVAILABLE = True
except ImportError:
    UMI_AVAILABLE = False
    logger.warning("UMI dependencies not available. UMI policy support will be limited.")


class UmiPolicy(BasePolicy):
    """
    UMI Policy for LeRobot.
    
    This class provides integration with UMI diffusion policies, supporting:
    - Multiple UMI policy types (UNet, Transformer, IBC, BET)
    - Vision-based and low-dimensional policies
    - Real-time inference
    - Multi-robot support
    """
    
    def __init__(
        self,
        config: UmiConfig,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize UMI policy.
        
        Args:
            config: UMI configuration
            device: Device to run policy on
            **kwargs: Additional arguments
        """
        if not UMI_AVAILABLE:
            raise ImportError("UMI dependencies not available. Please install UMI first.")
        
        super().__init__(config, device, **kwargs)
        
        self.config = config
        self.device = device or config.device
        self.dtype = getattr(torch, config.dtype)
        
        # Initialize UMI policy
        self._init_umi_policy()
        
        # Initialize normalizer
        self._init_normalizer()
        
        logger.info(f"Initialized UMI policy: {config.policy.policy_type}")
    
    def _init_umi_policy(self):
        """Initialize the underlying UMI policy."""
        policy_config = self.config.to_umi_config()
        
        # Create policy based on type
        policy_type = self.config.policy.policy_type
        
        if policy_type == "diffusion_unet_timm":
            self.umi_policy = DiffusionUnetTimmPolicy(
                shape_meta=policy_config["shape_meta"],
                noise_scheduler=policy_config["policy"]["noise_scheduler"],
                obs_encoder=policy_config["policy"]["obs_encoder"],
                num_inference_steps=policy_config["policy"]["num_inference_steps"],
                obs_as_global_cond=policy_config["policy"]["obs_as_global_cond"],
                diffusion_step_embed_dim=policy_config["policy"]["diffusion_step_embed_dim"],
                down_dims=policy_config["policy"]["down_dims"],
                up_dims=policy_config["policy"]["up_dims"],
            )
        elif policy_type == "diffusion_transformer_timm":
            self.umi_policy = DiffusionTransformerTimmPolicy(
                shape_meta=policy_config["shape_meta"],
                noise_scheduler=policy_config["policy"]["noise_scheduler"],
                obs_encoder=policy_config["policy"]["obs_encoder"],
                num_inference_steps=policy_config["policy"]["num_inference_steps"],
                obs_as_global_cond=policy_config["policy"]["obs_as_global_cond"],
                diffusion_step_embed_dim=policy_config["policy"]["diffusion_step_embed_dim"],
                down_dims=policy_config["policy"]["down_dims"],
                up_dims=policy_config["policy"]["up_dims"],
            )
        elif policy_type == "ibc_dfo":
            self.umi_policy = IbcDfoLowdimPolicy(
                shape_meta=policy_config["shape_meta"],
                # Add IBC-specific parameters
            )
        elif policy_type == "bet":
            self.umi_policy = BetLowdimPolicy(
                shape_meta=policy_config["shape_meta"],
                # Add BET-specific parameters
            )
        else:
            raise ValueError(f"Unsupported UMI policy type: {policy_type}")
        
        # Move to device
        self.umi_policy.to(device=self.device, dtype=self.dtype)
    
    def _init_normalizer(self):
        """Initialize normalizer for observations and actions."""
        # Create identity normalizer for now
        # In practice, this should be loaded from training data
        self.normalizer = get_identity_normalizer_from_stat(
            shape_meta=self.config.shape_meta
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load policy checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load policy weights
        if "policy" in checkpoint:
            self.umi_policy.load_state_dict(checkpoint["policy"])
        else:
            self.umi_policy.load_state_dict(checkpoint)
        
        # Load normalizer if available
        if "normalizer" in checkpoint:
            self.normalizer = checkpoint["normalizer"]
        
        logger.info(f"Loaded UMI policy checkpoint: {checkpoint_path}")
    
    def predict_action(
        self,
        obs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Predict action from observation.
        
        Args:
            obs: Observation dictionary
            **kwargs: Additional arguments
            
        Returns:
            Action dictionary
        """
        # Ensure observations are on correct device and dtype
        obs = dict_apply(obs, lambda x: x.to(device=self.device, dtype=self.dtype))
        
        # Normalize observations
        obs_norm = self.normalizer.normalize(obs)
        
        # Get action from UMI policy
        with torch.no_grad():
            action_norm = self.umi_policy.predict_action(obs_norm)
        
        # Denormalize actions
        action = self.normalizer.denormalize(action_norm)
        
        return action
    
    def predict_actions(
        self,
        obs: Dict[str, torch.Tensor],
        num_actions: int = 1,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predict multiple actions from observation.
        
        Args:
            obs: Observation dictionary
            num_actions: Number of actions to predict
            **kwargs: Additional arguments
            
        Returns:
            List of action dictionaries
        """
        actions = []
        for _ in range(num_actions):
            action = self.predict_action(obs, **kwargs)
            actions.append(action)
        
        return actions
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get action statistics."""
        return {
            "policy_type": self.config.policy.policy_type,
            "device": self.device,
            "dtype": self.dtype,
            "shape_meta": self.config.shape_meta,
            "num_inference_steps": self.config.policy.num_inference_steps,
        }
    
    def to(self, device: str):
        """Move policy to device."""
        self.device = device
        self.umi_policy.to(device=device)
        return self
    
    def eval(self):
        """Set policy to evaluation mode."""
        self.umi_policy.eval()
        return self
    
    def train(self):
        """Set policy to training mode."""
        self.umi_policy.train()
        return self


def create_umi_policy(
    config: UmiConfig,
    device: Optional[str] = None,
    **kwargs
) -> UmiPolicy:
    """
    Factory function to create UMI policy.
    
    Args:
        config: UMI configuration
        device: Device to run policy on
        **kwargs: Additional arguments
        
    Returns:
        UmiPolicy instance
    """
    return UmiPolicy(
        config=config,
        device=device,
        **kwargs
    ) 
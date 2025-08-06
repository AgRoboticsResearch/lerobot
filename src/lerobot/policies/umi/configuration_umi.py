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
UMI Policy Configuration for LeRobot.

This module provides configuration classes for UMI policies, supporting
various UMI policy types and parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import torch

from ...utils.config import BaseConfig


@dataclass
class UmiNoiseSchedulerConfig:
    """Configuration for UMI noise scheduler."""
    
    num_train_timesteps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    prediction_type: str = "epsilon"  # "epsilon" or "sample"


@dataclass
class UmiObsEncoderConfig:
    """Configuration for UMI observation encoder."""
    
    model_name: str = "vit_base_patch16_clip_224.openai"
    pretrained: bool = True
    frozen: bool = False
    global_pool: str = ""
    feature_aggregation: str = "attention_pool_2d"
    position_encoding: str = "sinusoidal"  # "learnable" or "sinusoidal"
    downsample_ratio: int = 32
    use_group_norm: bool = True
    share_rgb_model: bool = False
    imagenet_norm: bool = True
    
    # Transforms
    transforms: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"type": "RandomCrop", "ratio": 0.95},
        {
            "_target_": "torchvision.transforms.ColorJitter",
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08
        }
    ])


@dataclass
class UmiPolicyConfig:
    """Configuration for UMI policy."""
    
    policy_type: str = "diffusion_unet_timm"  # "diffusion_unet", "diffusion_transformer", "ibc", "bet"
    num_inference_steps: int = 16
    obs_as_global_cond: bool = True
    diffusion_step_embed_dim: int = 128
    down_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    up_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    noise_scheduler: UmiNoiseSchedulerConfig = field(default_factory=UmiNoiseSchedulerConfig)
    obs_encoder: UmiObsEncoderConfig = field(default_factory=UmiObsEncoderConfig)


@dataclass
class UmiConfig(BaseConfig):
    """
    Configuration for UMI policy integration.
    
    This configuration class supports various UMI policy types and parameters,
    allowing seamless integration with LeRobot's policy system.
    """
    
    # Policy configuration
    policy: UmiPolicyConfig = field(default_factory=UmiPolicyConfig)
    
    # Training configuration
    n_action_steps: int = 8
    shape_meta: Optional[Dict[str, Any]] = None
    task_name: str = "umi"
    
    # Model configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    
    # Inference configuration
    steps_per_inference: int = 6
    frequency: float = 10.0
    command_latency: float = 0.01
    
    # Pose representation
    pose_repr: Dict[str, str] = field(default_factory=lambda: {
        "obs_pose_repr": "rel",
        "action_pose_repr": "rel"
    })
    
    # Multi-robot support
    num_robots: int = 1
    bimanual: bool = False
    
    # Camera configuration
    camera_config: Dict[str, Any] = field(default_factory=dict)
    
    # Safety configuration
    collision_avoidance: bool = True
    height_threshold: float = 0.0
    
    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        
        # Set default shape_meta if not provided
        if self.shape_meta is None:
            self.shape_meta = self._get_default_shape_meta()
    
    def _get_default_shape_meta(self) -> Dict[str, Any]:
        """Get default shape metadata for UMI."""
        return {
            "obs": {
                "rgb": {"type": "rgb", "shape": (3, 224, 224)},
                "robot_eef_pos": {"type": "low_dim", "shape": (3,)},
                "robot_eef_rot": {"type": "low_dim", "shape": (3,)},
                "robot_gripper": {"type": "low_dim", "shape": (1,)},
            },
            "action": {
                "robot_eef_pos": {"type": "low_dim", "shape": (3,)},
                "robot_eef_rot": {"type": "low_dim", "shape": (3,)},
                "robot_gripper": {"type": "low_dim", "shape": (1,)},
            }
        }
    
    @classmethod
    def from_umi_config(cls, umi_config_path: str) -> "UmiConfig":
        """
        Create UmiConfig from UMI configuration file.
        
        Args:
            umi_config_path: Path to UMI configuration file
            
        Returns:
            UmiConfig instance
        """
        # This would load from UMI's YAML configuration format
        # For now, return default config
        return cls()
    
    def to_umi_config(self) -> Dict[str, Any]:
        """
        Convert to UMI configuration format.
        
        Returns:
            Dictionary in UMI configuration format
        """
        return {
            "policy": {
                "_target_": f"diffusion_policy.policy.{self.policy.policy_type}_policy.{self.policy.policy_type.title()}Policy",
                "shape_meta": self.shape_meta,
                "noise_scheduler": {
                    "_target_": "diffusers.DDIMScheduler",
                    **self.policy.noise_scheduler.__dict__
                },
                "obs_encoder": {
                    "_target_": "diffusion_policy.model.vision.timm_obs_encoder.TimmObsEncoder",
                    "shape_meta": self.shape_meta,
                    **self.policy.obs_encoder.__dict__
                },
                "num_inference_steps": self.policy.num_inference_steps,
                "obs_as_global_cond": self.policy.obs_as_global_cond,
                "diffusion_step_embed_dim": self.policy.diffusion_step_embed_dim,
                "down_dims": self.policy.down_dims,
                "up_dims": self.policy.up_dims,
            },
            "n_action_steps": self.n_action_steps,
            "shape_meta": self.shape_meta,
            "task_name": self.task_name,
        } 
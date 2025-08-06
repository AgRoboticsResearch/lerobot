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
UMI Integration Example for LeRobot.

This example demonstrates how to use the UMI integration with LeRobot,
including dataset loading, policy creation, and robot control.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.umi_dataset import create_umi_dataset
from lerobot.policies.umi import UmiConfig, UmiPolicy
from lerobot.robots.umi import UmiRobotConfig
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_umi_dataset_example():
    """Example of loading a UMI dataset."""
    logger.info("=== UMI Dataset Loading Example ===")
    
    # Example dataset path (you would replace this with your actual dataset)
    dataset_path = "path/to/your/umi_dataset.zarr.zip"
    
    try:
        # Create UMI dataset
        dataset = create_umi_dataset(
            dataset_path=dataset_path,
            split="train",
            cache_dir="./cache",
            pose_repr={
                "obs_pose_repr": "rel",
                "action_pose_repr": "rel"
            }
        )
        
        # Get dataset statistics
        stats = dataset.get_dataset_stats()
        logger.info(f"Dataset stats: {stats}")
        
        # Get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            logger.info(f"Observation keys: {list(sample['obs'].keys())}")
            logger.info(f"Action keys: {list(sample['action'].keys())}")
        
        return dataset
        
    except FileNotFoundError:
        logger.warning(f"Dataset not found: {dataset_path}")
        logger.info("This is expected if you don't have a UMI dataset yet.")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None


def create_umi_policy_example():
    """Example of creating a UMI policy."""
    logger.info("=== UMI Policy Creation Example ===")
    
    # Create UMI configuration
    config = UmiConfig(
        policy=UmiConfig.UmiPolicyConfig(
            policy_type="diffusion_unet_timm",
            num_inference_steps=16,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            up_dims=[1024, 512, 256],
            noise_scheduler=UmiConfig.UmiNoiseSchedulerConfig(
                num_train_timesteps=50,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon"
            ),
            obs_encoder=UmiConfig.UmiObsEncoderConfig(
                model_name="vit_base_patch16_clip_224.openai",
                pretrained=True,
                frozen=False,
                feature_aggregation="attention_pool_2d",
                position_encoding="sinusoidal"
            )
        ),
        n_action_steps=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        steps_per_inference=6,
        frequency=10.0,
        num_robots=1,
        bimanual=False
    )
    
    try:
        # Create UMI policy
        policy = UmiPolicy(config=config)
        logger.info(f"Created UMI policy: {config.policy.policy_type}")
        
        # Load checkpoint if available
        checkpoint_path = "path/to/your/umi_checkpoint.ckpt"
        if os.path.exists(checkpoint_path):
            policy.load_checkpoint(checkpoint_path)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return policy
        
    except Exception as e:
        logger.error(f"Error creating policy: {e}")
        return None


def create_umi_robot_config_example():
    """Example of creating a UMI robot configuration."""
    logger.info("=== UMI Robot Configuration Example ===")
    
    # Create UMI robot configuration
    config = UmiRobotConfig(
        robots=[
            UmiRobotConfig.UmiRobotArmConfig(
                robot_type="ur5",
                robot_ip="192.168.1.100",  # Replace with your robot IP
                payload_mass=1.81,
                payload_cog=(0.002, -0.006, 0.037),
                control_frequency=10.0,
                command_latency=0.01
            )
        ],
        grippers=[
            UmiRobotConfig.UmiGripperConfig(
                gripper_type="wsg50",
                gripper_ip="192.168.1.101",  # Replace with your gripper IP
                wsg50_width_range=(0.0, 0.11),
                wsg50_speed=0.2,
                wsg50_force=40.0
            )
        ],
        cameras=[
            UmiRobotConfig.UmiCameraConfig(
                camera_type="gopro",
                camera_serial="C3441328164125",  # Replace with your camera serial
                resolution=(1920, 1080),
                fps=30,
                fisheye=True
            )
        ],
        teleop=UmiRobotConfig.UmiTeleopConfig(
            teleop_type="spacemouse",
            spacemouse_sensitivity=1.0,
            spacemouse_deadzone=0.05
        ),
        bimanual=False,
        collision_avoidance=True,
        emergency_stop_enabled=True
    )
    
    # Validate configuration
    config.validate_config()
    
    # Save configuration to YAML
    config.to_yaml("umi_robot_config.yaml")
    logger.info("Saved UMI robot configuration to umi_robot_config.yaml")
    
    return config


def policy_inference_example(policy):
    """Example of running policy inference."""
    if policy is None:
        logger.warning("No policy available for inference example")
        return
    
    logger.info("=== UMI Policy Inference Example ===")
    
    # Create dummy observation (replace with real observation)
    obs = {
        "rgb": torch.randn(1, 3, 224, 224),  # RGB image
        "robot_eef_pos": torch.randn(1, 3),  # End-effector position
        "robot_eef_rot": torch.randn(1, 3),  # End-effector rotation
        "robot_gripper": torch.randn(1, 1),  # Gripper state
    }
    
    try:
        # Run inference
        with torch.no_grad():
            action = policy.predict_action(obs)
        
        logger.info(f"Predicted action keys: {list(action.keys())}")
        for key, value in action.items():
            logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")
        
        return action
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return None


def main():
    """Main example function."""
    logger.info("Starting UMI Integration Example")
    
    # Example 1: Load UMI dataset
    dataset = load_umi_dataset_example()
    
    # Example 2: Create UMI policy
    policy = create_umi_policy_example()
    
    # Example 3: Create UMI robot configuration
    robot_config = create_umi_robot_config_example()
    
    # Example 4: Run policy inference
    action = policy_inference_example(policy)
    
    logger.info("UMI Integration Example completed!")
    
    # Print summary
    logger.info("\n=== Summary ===")
    logger.info(f"Dataset loaded: {dataset is not None}")
    logger.info(f"Policy created: {policy is not None}")
    logger.info(f"Robot config created: {robot_config is not None}")
    logger.info(f"Inference successful: {action is not None}")


if __name__ == "__main__":
    main() 
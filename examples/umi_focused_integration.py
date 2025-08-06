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
Focused UMI Integration Example for LeRobot.

This example demonstrates the unique UMI features:
1. SLAM-based pose estimation (UMI's core innovation)
2. UMI-style teleoperation with IK calculations
3. Integration with LeRobot's existing diffusion policies

The focus is on UMI's distinctive capabilities without duplicating
LeRobot's mature diffusion policy and dataset infrastructure.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lerobot.utils.umi_slam import create_umi_slam_processor
from lerobot.teleoperators.umi import UmiTeleoperatorConfig, create_umi_teleoperator
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def umi_slam_example():
    """Example of UMI SLAM processing - UMI's core innovation."""
    logger.info("=== UMI SLAM Processing Example ===")
    
    # Create UMI SLAM processor
    slam_processor = create_umi_slam_processor(
        umi_root_path="universal_manipulation_interface",
        calibration_dir="universal_manipulation_interface/example/calibration"
    )
    
    # Example session directory (replace with your actual session)
    session_dir = "path/to/your/umi_session"
    
    try:
        # Run full UMI SLAM pipeline
        logger.info("Running UMI SLAM pipeline...")
        success = slam_processor.run_slam_pipeline(session_dir)
        
        if success:
            logger.info("UMI SLAM pipeline completed successfully!")
            
            # Generate dataset for training
            output_path = f"{session_dir}/dataset.zarr.zip"
            dataset_success = slam_processor.generate_dataset(session_dir, output_path)
            
            if dataset_success:
                logger.info(f"Dataset generated: {output_path}")
                return output_path
            else:
                logger.error("Failed to generate dataset")
        else:
            logger.error("UMI SLAM pipeline failed")
            
    except FileNotFoundError:
        logger.warning(f"Session directory not found: {session_dir}")
        logger.info("This is expected if you don't have UMI session data yet.")
    except Exception as e:
        logger.error(f"Error in SLAM processing: {e}")
    
    return None


def umi_teleoperation_example():
    """Example of UMI teleoperation with IK - UMI's unique approach."""
    logger.info("=== UMI Teleoperation Example ===")
    
    # Create UMI teleoperator configuration
    config = UmiTeleoperatorConfig(
        spacemouse=UmiTeleoperatorConfig.UmiSpaceMouseConfig(
            sensitivity_translation=1.0,
            sensitivity_rotation=1.0,
            deadzone=0.05,
            lock_z_axis=False,
            lock_rotation=False,
            max_velocity=0.5,
            max_angular_velocity=1.0
        ),
        ik=UmiTeleoperatorConfig.UmiIkConfig(
            robot_type="ur5",
            ik_solver="ikfast",
            max_iterations=100,
            tolerance_position=0.001,
            tolerance_orientation=0.01,
            collision_avoidance=True,
            collision_margin=0.05,
            workspace_limits={
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 1.0)
            },
            height_threshold=0.0,
            table_collision_avoidance=True
        ),
        control_frequency=10.0,
        command_latency=0.01,
        num_robots=1,
        bimanual=False,
        emergency_stop_enabled=True,
        workspace_limits_enabled=True,
        collision_avoidance_enabled=True
    )
    
    try:
        # Create UMI teleoperator
        teleoperator = create_umi_teleoperator(config)
        
        # Start teleoperation
        logger.info("Starting UMI teleoperation...")
        teleoperator.start()
        
        # Run for a few seconds
        time.sleep(5.0)
        
        # Get current state
        state = teleoperator.get_current_state()
        logger.info(f"Current pose: {state['current_pose'][:3, 3]}")
        logger.info(f"Joint angles: {state['current_joint_angles']}")
        
        # Stop teleoperation
        teleoperator.stop()
        logger.info("UMI teleoperation completed")
        
        return teleoperator
        
    except Exception as e:
        logger.error(f"Error in teleoperation: {e}")
        return None


def lerobot_diffusion_policy_example():
    """Example of using LeRobot's existing diffusion policy with UMI data."""
    logger.info("=== LeRobot Diffusion Policy Example ===")
    
    # This demonstrates how to use LeRobot's mature diffusion policy
    # with UMI-processed data, without duplicating functionality
    
    try:
        # Create LeRobot diffusion policy configuration
        config = DiffusionConfig(
            input_features={
                "observation.images": (3, 224, 224),
                "observation.state": (7,),  # 3 pos + 3 rot + 1 gripper
            },
            output_features={
                "action": (7,),  # 3 pos + 3 rot + 1 gripper
            },
            n_obs_steps=2,
            n_action_steps=8,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            up_dims=[1024, 512, 256],
            noise_scheduler="ddim",
            num_inference_steps=16,
        )
        
        # Create LeRobot diffusion policy
        policy = DiffusionPolicy(config=config)
        logger.info("Created LeRobot diffusion policy")
        
        # Load UMI-processed dataset (if available)
        dataset_path = "path/to/umi_processed_dataset.zarr.zip"
        if os.path.exists(dataset_path):
            logger.info(f"Loading UMI-processed dataset: {dataset_path}")
            # Here you would load the dataset using LeRobot's dataset loader
            # dataset = load_dataset(dataset_path)
            # policy.load_state_dict(checkpoint)
        
        return policy
        
    except Exception as e:
        logger.error(f"Error creating diffusion policy: {e}")
        return None


def integrated_umi_workflow():
    """Integrated UMI workflow example."""
    logger.info("=== Integrated UMI Workflow Example ===")
    
    # Step 1: Process UMI data with SLAM
    logger.info("Step 1: Processing UMI data with SLAM...")
    dataset_path = umi_slam_example()
    
    # Step 2: Demonstrate UMI teleoperation
    logger.info("Step 2: Demonstrating UMI teleoperation...")
    teleoperator = umi_teleoperation_example()
    
    # Step 3: Use LeRobot's diffusion policy with UMI data
    logger.info("Step 3: Using LeRobot diffusion policy with UMI data...")
    policy = lerobot_diffusion_policy_example()
    
    # Summary
    logger.info("\n=== Integration Summary ===")
    logger.info(f"UMI SLAM processing: {'✓' if dataset_path else '✗'}")
    logger.info(f"UMI teleoperation: {'✓' if teleoperator else '✗'}")
    logger.info(f"LeRobot diffusion policy: {'✓' if policy else '✗'}")
    
    logger.info("\n=== Key UMI Features Demonstrated ===")
    logger.info("1. SLAM-based pose estimation (UMI's core innovation)")
    logger.info("2. Real-time IK calculations for teleoperation")
    logger.info("3. Integration with LeRobot's mature policy system")
    logger.info("4. No duplication of existing LeRobot capabilities")


def main():
    """Main example function."""
    logger.info("Starting Focused UMI Integration Example")
    logger.info("This example focuses on UMI's unique features:")
    logger.info("- SLAM-based pose estimation")
    logger.info("- UMI-style teleoperation with IK")
    logger.info("- Integration with LeRobot's existing capabilities")
    
    # Run integrated workflow
    integrated_umi_workflow()
    
    logger.info("\nUMI Integration Example completed!")
    logger.info("The integration focuses on UMI's distinctive capabilities")
    logger.info("while leveraging LeRobot's mature infrastructure.")


if __name__ == "__main__":
    main() 
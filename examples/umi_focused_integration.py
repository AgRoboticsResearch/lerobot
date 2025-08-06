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
2. UMI-style teleoperation with IK calculations using LeRobot's placo-based IK
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


def umi_teleoperation_with_lerobot_ik_example():
    """Example of UMI teleoperation using LeRobot's placo-based IK solver."""
    logger.info("=== UMI Teleoperation with LeRobot IK Example ===")
    
    # Create UMI teleoperator configuration with LeRobot's IK integration
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
            # Robot configuration for SO101
            robot_type="so101",  # Using SO101 robot
            urdf_path="path/to/so101_new_calib.urdf",  # SO101 URDF from SO-ARM100 repo
            target_frame_name="gripper_frame_link",  # End-effector frame
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", 
                        "wrist_flex", "wrist_roll", "gripper"],
            
            # LeRobot's placo-based IK parameters
            position_weight=1.0,  # Weight for position constraint
            orientation_weight=0.01,  # Weight for orientation constraint
            
            # Safety and workspace parameters
            collision_avoidance=True,
            collision_margin=0.05,
            workspace_limits={
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.8)
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
        # Create UMI teleoperator (uses LeRobot's IK internally)
        teleoperator = create_umi_teleoperator(config)
        
        # Start teleoperation
        logger.info("Starting UMI teleoperation with LeRobot IK for SO101...")
        teleoperator.start()
        
        # Run for a few seconds
        time.sleep(5.0)
        
        # Get current state
        state = teleoperator.get_current_state()
        logger.info(f"Current pose: {state['current_pose'][:3, 3]}")
        logger.info(f"Joint angles: {state['current_joint_angles']}")
        logger.info(f"Using LeRobot's placo-based IK solver for SO101")
        
        # Stop teleoperation
        teleoperator.stop()
        logger.info("UMI teleoperation with LeRobot IK for SO101 completed")
        
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


def so101_specific_example():
    """Specific example for SO101 robot with UMI teleoperation and LeRobot IK."""
    logger.info("=== SO101-Specific UMI Teleoperation Example ===")
    
    # SO101-specific configuration
    config = UmiTeleoperatorConfig(
        spacemouse=UmiTeleoperatorConfig.UmiSpaceMouseConfig(
            sensitivity_translation=0.8,  # Slightly lower for SO101 precision
            sensitivity_rotation=0.6,     # Lower rotation sensitivity
            deadzone=0.08,                # Larger deadzone for stability
            lock_z_axis=False,
            lock_rotation=False,
            max_velocity=0.3,             # Conservative velocity for SO101
            max_angular_velocity=0.8      # Conservative angular velocity
        ),
        ik=UmiTeleoperatorConfig.UmiIkConfig(
            # SO101-specific configuration
            robot_type="so101",
            urdf_path="path/to/so101_new_calib.urdf",  # SO101 URDF from SO-ARM100 repo
            target_frame_name="gripper_frame_link",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", 
                        "wrist_flex", "wrist_roll", "gripper"],
            
            # SO101-specific IK parameters
            position_weight=1.0,
            orientation_weight=0.005,  # Lower orientation weight for SO101
            
            # SO101-specific workspace limits (based on SO101 specs)
            workspace_limits={
                "x": (-0.4, 0.4),    # SO101 workspace limits
                "y": (-0.4, 0.4),
                "z": (0.0, 0.6)
            },
            
            # SO101-specific joint limits (in degrees)
            joint_limits=[
                (-180, 180),  # shoulder_pan
                (-180, 180),  # shoulder_lift
                (-180, 180),  # elbow_flex
                (-180, 180),  # wrist_flex
                (-180, 180),  # wrist_roll
                (0, 100),     # gripper
            ],
            
            # SO101-specific velocity limits (in deg/s)
            velocity_limits=[90.0, 90.0, 90.0, 90.0, 90.0, 50.0],
            
            collision_avoidance=True,
            collision_margin=0.08,  # Larger margin for SO101
            height_threshold=0.05,  # Slightly above table
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
        logger.info("Creating SO101-specific UMI teleoperator...")
        logger.info(f"Robot type: {config.ik.robot_type}")
        logger.info(f"URDF path: {config.ik.urdf_path}")
        logger.info(f"Target frame: {config.ik.target_frame_name}")
        logger.info(f"Joint names: {config.ik.joint_names}")
        logger.info(f"Workspace limits: {config.ik.workspace_limits}")
        
        # Create teleoperator
        teleoperator = create_umi_teleoperator(config)
        
        logger.info("SO101 UMI teleoperator created successfully!")
        logger.info("This configuration uses LeRobot's placo-based IK solver")
        logger.info("with SO101-specific parameters and workspace limits.")
        
        return teleoperator
        
    except Exception as e:
        logger.error(f"Error creating SO101 teleoperator: {e}")
        return None


def demonstrate_lerobot_ik_integration():
    """Demonstrate the integration between UMI teleoperation and LeRobot's IK."""
    logger.info("=== LeRobot IK Integration Demonstration ===")
    
    # Show how UMI teleoperation leverages LeRobot's mature IK pipeline
    logger.info("Key benefits of using LeRobot's IK pipeline:")
    logger.info("1. Placo-based IK solver (robust and fast)")
    logger.info("2. URDF-based robot models (standard format)")
    logger.info("3. Configurable position/orientation weights")
    logger.info("4. Automatic joint limit handling")
    logger.info("5. Gripper state preservation")
    
    # Example configuration showing the integration
    config = UmiTeleoperatorConfig(
        ik=UmiTeleoperatorConfig.UmiIkConfig(
            robot_type="so101",  # Using SO101 as example
            urdf_path="path/to/so101_new_calib.urdf",
            target_frame_name="gripper_frame_link",  # SO101 end-effector frame
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                        "wrist_flex", "wrist_roll", "gripper"],
            position_weight=1.0,
            orientation_weight=0.01,
        )
    )
    
    logger.info(f"UMI teleoperator configured to use LeRobot's IK for {config.ik.robot_type}")
    logger.info(f"URDF: {config.ik.urdf_path}")
    logger.info(f"Target frame: {config.ik.target_frame_name}")
    logger.info(f"Joint names: {config.ik.joint_names}")
    logger.info(f"Position weight: {config.ik.position_weight}")
    logger.info(f"Orientation weight: {config.ik.orientation_weight}")
    
    # Show SO101-specific benefits
    logger.info("\nSO101-specific benefits:")
    logger.info("1. Uses SO101 URDF from SO-ARM100 repository")
    logger.info("2. Proper joint limits and velocity constraints")
    logger.info("3. Workspace limits based on SO101 specifications")
    logger.info("4. Compatible with LeRobot's existing SO101 support")


def integrated_umi_workflow():
    """Integrated UMI workflow example."""
    logger.info("=== Integrated UMI Workflow Example ===")
    
    # Step 1: Process UMI data with SLAM
    logger.info("Step 1: Processing UMI data with SLAM...")
    dataset_path = umi_slam_example()
    
    # Step 2: Demonstrate UMI teleoperation with LeRobot's IK
    logger.info("Step 2: Demonstrating UMI teleoperation with LeRobot IK...")
    teleoperator = umi_teleoperation_with_lerobot_ik_example()
    
    # Step 3: Demonstrate SO101-specific configuration
    logger.info("Step 3: Demonstrating SO101-specific UMI teleoperation...")
    so101_teleoperator = so101_specific_example()
    
    # Step 4: Use LeRobot's diffusion policy with UMI data
    logger.info("Step 4: Using LeRobot diffusion policy with UMI data...")
    policy = lerobot_diffusion_policy_example()
    
    # Step 5: Demonstrate IK integration
    logger.info("Step 5: Demonstrating LeRobot IK integration...")
    demonstrate_lerobot_ik_integration()
    
    # Summary
    logger.info("\n=== Integration Summary ===")
    logger.info(f"UMI SLAM processing: {'✓' if dataset_path else '✗'}")
    logger.info(f"UMI teleoperation with LeRobot IK: {'✓' if teleoperator else '✗'}")
    logger.info(f"SO101-specific teleoperation: {'✓' if so101_teleoperator else '✗'}")
    logger.info(f"LeRobot diffusion policy: {'✓' if policy else '✗'}")
    
    logger.info("\n=== Key UMI Features Demonstrated ===")
    logger.info("1. SLAM-based pose estimation (UMI's core innovation)")
    logger.info("2. Real-time IK calculations using LeRobot's placo-based solver")
    logger.info("3. Integration with LeRobot's mature policy system")
    logger.info("4. No duplication of existing LeRobot capabilities")
    logger.info("5. Leveraging LeRobot's proven IK infrastructure")
    logger.info("6. SO101-specific configuration and optimization")


def main():
    """Main example function."""
    logger.info("Starting Focused UMI Integration Example")
    logger.info("This example focuses on UMI's unique features:")
    logger.info("- SLAM-based pose estimation")
    logger.info("- UMI-style teleoperation with LeRobot's IK")
    logger.info("- Integration with LeRobot's existing capabilities")
    
    # Run integrated workflow
    integrated_umi_workflow()
    
    logger.info("\nUMI Integration Example completed!")
    logger.info("The integration focuses on UMI's distinctive capabilities")
    logger.info("while leveraging LeRobot's mature infrastructure.")
    logger.info("Key highlight: UMI teleoperation now uses LeRobot's placo-based IK solver!")


if __name__ == "__main__":
    main() 
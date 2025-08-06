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
ORB-SLAM Integration Example

This example demonstrates the ONLY component we need to integrate from UMI:
ORB-SLAM visual odometry that works with LeRobot's existing camera infrastructure.

Everything else (cameras, IK, teleoperation) is already available in LeRobot!
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lerobot.cameras.realsense import RealSenseCamera, RealSenseConfig
from lerobot.utils.orb_slam_integration import create_orb_slam_processor, OrbSlamConfig, get_orb_slam_status
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def demonstrate_orb_slam_integration():
    """Demonstrate ORB-SLAM integration with LeRobot cameras."""
    logger.info("=== ORB-SLAM Integration Demonstration ===")
    logger.info("This shows the ONLY component we need from UMI!")
    
    # Step 1: Check ORB-SLAM availability
    logger.info("Step 1: Checking ORB-SLAM integration status...")
    status = get_orb_slam_status()
    logger.info(f"ORB-SLAM Status: {status}")
    
    # Step 2: Configure LeRobot RealSense cameras (already available!)
    logger.info("Step 2: Configuring LeRobot RealSense cameras...")
    camera_configs = {
        "camera_left": RealSenseConfig(
            device_id="left_camera_serial",
            width=640,
            height=480,
            fps=30
        ),
        "camera_right": RealSenseConfig(
            device_id="right_camera_serial", 
            width=640,
            height=480,
            fps=30
        )
    }
    
    # Step 3: Initialize ORB-SLAM processor (the ONLY UMI component we need)
    logger.info("Step 3: Initializing ORB-SLAM processor...")
    orb_slam_config = OrbSlamConfig(
        camera_config_path="universal_manipulation_interface/example/calibration/camera_config.yaml",
        vocabulary_path="universal_manipulation_interface/example/calibration/ORBvoc.txt",
        settings_path="universal_manipulation_interface/example/calibration/settings.yaml",
        max_features=2000,
        output_frequency=10.0
    )
    
    orb_slam_processor = create_orb_slam_processor(orb_slam_config)
    
    # Step 4: Demonstrate the integration
    logger.info("Step 4: Demonstrating ORB-SLAM integration...")
    
    # Simulate camera frames (in practice, these would come from LeRobot cameras)
    dummy_frames = {
        "camera_left": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "camera_right": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    }
    
    # Process frames with ORB-SLAM
    estimated_pose = orb_slam_processor.process_camera_frames(dummy_frames)
    
    if estimated_pose is not None:
        logger.info("✅ ORB-SLAM pose estimation successful!")
        logger.info(f"Estimated pose:\n{estimated_pose}")
    else:
        logger.info("⚠️ ORB-SLAM pose estimation failed (expected with dummy data)")
    
    # Step 5: Show what we DON'T need to integrate
    logger.info("\nStep 5: What we DON'T need from UMI (already in LeRobot):")
    logger.info("✅ LeRobot RealSense cameras - Mature infrastructure")
    logger.info("✅ LeRobot IK solver - Placo-based for SO101")
    logger.info("✅ LeRobot teleoperation - SpaceMouse integration")
    logger.info("✅ LeRobot dataset handling - Proven data format")
    logger.info("✅ LeRobot policy training - Diffusion policies")
    
    logger.info("\n🎯 ORB-SLAM is the ONLY component we need from UMI!")


def show_integration_benefits():
    """Show the benefits of this focused integration approach."""
    logger.info("\n=== Integration Benefits ===")
    
    logger.info("🎯 Focused Integration:")
    logger.info("   - Only integrate ORB-SLAM from UMI")
    logger.info("   - Use LeRobot's mature infrastructure for everything else")
    logger.info("   - No duplication of existing capabilities")
    
    logger.info("\n🔧 Technical Benefits:")
    logger.info("   - LeRobot cameras provide reliable dual camera input")
    logger.info("   - ORB-SLAM provides robust visual odometry")
    logger.info("   - LeRobot IK provides proven inverse kinematics")
    logger.info("   - Clean separation of concerns")
    
    logger.info("\n🚀 Development Benefits:")
    logger.info("   - Faster integration (only one component)")
    logger.info("   - Lower maintenance burden")
    logger.info("   - Leverages existing LeRobot expertise")
    logger.info("   - Minimal code changes required")


def demonstrate_usage_with_real_cameras():
    """Demonstrate how to use ORB-SLAM with real LeRobot cameras."""
    logger.info("\n=== Usage with Real LeRobot Cameras ===")
    
    logger.info("To use with real cameras:")
    logger.info("1. Connect Intel RealSense cameras")
    logger.info("2. Update device IDs in camera config")
    logger.info("3. Initialize LeRobot RealSense cameras")
    logger.info("4. Get frames from cameras")
    logger.info("5. Process with ORB-SLAM")
    logger.info("6. Use pose for IK and teleoperation")
    
    logger.info("\nExample code:")
    logger.info("""
# Initialize cameras
cameras = {}
for camera_name, config in camera_configs.items():
    camera = RealSenseCamera(config)
    camera.connect()
    cameras[camera_name] = camera

# Get frames
frames = {}
for camera_name, camera in cameras.items():
    frame = camera.async_read()
    if frame is not None:
        frames[camera_name] = frame

# Process with ORB-SLAM
estimated_pose = orb_slam_processor.process_camera_frames(frames)

# Use pose for IK (LeRobot's existing IK)
if estimated_pose is not None:
    joint_angles = ik_solver.inverse_kinematics(estimated_pose)
    # Send to robot...
    """)


def main():
    """Main function demonstrating ORB-SLAM integration."""
    logger.info("Starting ORB-SLAM Integration Example")
    logger.info("This demonstrates the ONLY component we need from UMI!")
    
    # Demonstrate the integration
    demonstrate_orb_slam_integration()
    
    # Show benefits
    show_integration_benefits()
    
    # Show usage with real cameras
    demonstrate_usage_with_real_cameras()
    
    logger.info("\n=== Summary ===")
    logger.info("✅ ORB-SLAM integration: The ONLY component from UMI")
    logger.info("✅ LeRobot cameras: Already available and mature")
    logger.info("✅ LeRobot IK: Already available and proven")
    logger.info("✅ LeRobot teleoperation: Already available and tested")
    logger.info("✅ Complete pipeline: Camera → ORB-SLAM → IK → Teleoperation")
    
    logger.info("\n🎯 Perfect integration strategy: Focus on UMI's unique value!")


if __name__ == "__main__":
    main() 
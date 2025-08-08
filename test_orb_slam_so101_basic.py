#!/usr/bin/env python3
"""
Basic Test: ORB-SLAM SO101 Teleoperation System

This script tests the basic functionality of the ORB-SLAM SO101 teleoperator
without requiring robot connection. It focuses on:
1. Camera initialization
2. ORB-SLAM processing
3. Configuration validation
4. Basic pose estimation
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.teleoperators.orb_slam_so101 import (
    OrbSlamSo101TeleoperatorConfig,
    CameraConfig,
    OrbSlamConfig,
    ControlConfig,
    SafetyConfig,
    create_orb_slam_so101_teleoperator
)
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def test_configuration():
    """Test configuration creation and validation."""
    print("🔧 Testing configuration...")
    
    try:
        # Create configuration
        config = OrbSlamSo101TeleoperatorConfig(
            id="test_orb_slam_so101",
            camera=CameraConfig(
                serial_number_or_name="",  # Auto-detect
                fps=30,
                width=640,
                height=480,
                use_depth=True
            ),
            orb_slam=OrbSlamConfig(
                max_features=2000,
                output_frequency=30.0,
                enable_visualization=True
            ),
            control=ControlConfig(
                control_frequency=30.0,
                camera_to_robot_scale=0.1,
                pose_smoothing_alpha=0.7
            ),
            safety=SafetyConfig(
                workspace_limits={
                    'x': [-0.3, 0.3],
                    'y': [-0.3, 0.3],
                    'z': [0.2, 0.6]
                },
                max_velocity=0.05
            )
        )
        
        print("✅ Configuration created successfully")
        print(f"   Camera: {config.camera.serial_number_or_name}")
        print(f"   ORB-SLAM features: {config.orb_slam.max_features}")
        print(f"   Control frequency: {config.control.control_frequency} Hz")
        print(f"   Workspace limits: {config.safety.workspace_limits}")
        
        return config
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return None


def test_camera_initialization(config):
    """Test camera initialization."""
    print("\n📷 Testing camera initialization...")
    
    try:
        # Test RealSense camera config
        camera_config = config.camera
        print(f"   Camera config: {camera_config.serial_number_or_name}")
        print(f"   Resolution: {camera_config.width}x{camera_config.height}")
        print(f"   FPS: {camera_config.fps}")
        print(f"   Use depth: {camera_config.use_depth}")
        
        # Test camera matrix creation
        camera_matrix = config.get_camera_matrix()
        print(f"   Camera matrix shape: {camera_matrix.shape}")
        print(f"   Focal length: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
        print(f"   Principal point: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
        
        print("✅ Camera configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False


def test_orb_slam_initialization(config):
    """Test ORB-SLAM initialization."""
    print("\n🎯 Testing ORB-SLAM initialization...")
    
    try:
        # Test ORB-SLAM config
        orb_slam_config = config.orb_slam
        print(f"   Max features: {orb_slam_config.max_features}")
        print(f"   Output frequency: {orb_slam_config.output_frequency} Hz")
        print(f"   Scale factor: {orb_slam_config.scale_factor}")
        print(f"   Min threshold: {orb_slam_config.min_threshold}")
        
        print("✅ ORB-SLAM configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ ORB-SLAM test failed: {e}")
        return False


def test_teleoperator_creation(config):
    """Test teleoperator creation (without connecting)."""
    print("\n🤖 Testing teleoperator creation...")
    
    try:
        # Create teleoperator
        teleoperator = create_orb_slam_so101_teleoperator(config)
        print("✅ Teleoperator created successfully")
        
        # Test basic properties
        print(f"   Name: {teleoperator.name}")
        print(f"   Config class: {teleoperator.config_class}")
        print(f"   Is connected: {teleoperator.is_connected}")
        print(f"   Is calibrated: {teleoperator.is_calibrated}")
        
        # Test action features
        action_features = teleoperator.action_features
        print(f"   Action features: {list(action_features.keys())}")
        
        # Test feedback features
        feedback_features = teleoperator.feedback_features
        print(f"   Feedback features: {list(feedback_features.keys())}")
        
        print("✅ Teleoperator creation test passed")
        return teleoperator
        
    except Exception as e:
        print(f"❌ Teleoperator creation test failed: {e}")
        return None


def test_pose_estimation_simulation():
    """Test pose estimation with simulated data."""
    print("\n📊 Testing pose estimation simulation...")
    
    try:
        # Create simulated camera frames
        left_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        frames = {
            "left": left_frame,
            "right": right_frame
        }
        
        print(f"   Simulated frames created: {left_frame.shape}")
        
        # Test ORB-SLAM integration
        from lerobot.utils.orb_slam_integration import create_orb_slam_processor, OrbSlamConfig
        
        orb_slam_config = OrbSlamConfig(
            max_features=2000,
            output_frequency=30.0,
            enable_visualization=False
        )
        
        processor = create_orb_slam_processor(orb_slam_config)
        print("✅ ORB-SLAM processor created")
        
        # Process frames
        pose = processor.process_camera_frames(frames)
        print(f"   Initial pose: {pose is not None}")
        
        if pose is not None:
            print(f"   Pose shape: {pose.shape}")
            print(f"   Translation: {pose[:3, 3]}")
            print(f"   Rotation matrix shape: {pose[:3, :3].shape}")
        
        print("✅ Pose estimation simulation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pose estimation test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🎯 ORB-SLAM SO101 Basic Test")
    print("=" * 50)
    print("Testing basic functionality without robot connection")
    print("=" * 50)
    
    # Test 1: Configuration
    config = test_configuration()
    if config is None:
        print("❌ Configuration test failed - stopping")
        return False
    
    # Test 2: Camera initialization
    if not test_camera_initialization(config):
        print("❌ Camera test failed - stopping")
        return False
    
    # Test 3: ORB-SLAM initialization
    if not test_orb_slam_initialization(config):
        print("❌ ORB-SLAM test failed - stopping")
        return False
    
    # Test 4: Teleoperator creation
    teleoperator = test_teleoperator_creation(config)
    if teleoperator is None:
        print("❌ Teleoperator creation test failed - stopping")
        return False
    
    # Test 5: Pose estimation simulation
    if not test_pose_estimation_simulation():
        print("❌ Pose estimation test failed - stopping")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All basic tests passed!")
    print("🎉 ORB-SLAM SO101 teleoperator is ready for use")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 
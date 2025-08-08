#!/usr/bin/env python3
"""
Camera Test: ORB-SLAM SO101 Teleoperation System

This script tests the camera connection and ORB-SLAM tracking
without requiring robot connection.
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


def test_camera_connection():
    """Test camera connection and basic functionality."""
    print("🎯 ORB-SLAM SO101 Camera Test")
    print("=" * 50)
    print("Testing camera connection and ORB-SLAM tracking")
    print("Move your RealSense camera to see pose tracking!")
    print("=" * 50)
    
    # Create configuration
    config = OrbSlamSo101TeleoperatorConfig(
        id="test_orb_slam_so101_camera",
        camera=CameraConfig(
            serial_number_or_name="031522070877",  # Your camera serial number
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
    
    # Create teleoperator
    teleoperator = create_orb_slam_so101_teleoperator(config)
    
    try:
        print("🔌 Connecting to camera...")
        teleoperator.connect(calibrate=True)
        
        if not teleoperator.is_connected:
            print("❌ Failed to connect to camera")
            return False
        
        print("✅ Connected to camera successfully!")
        print("🚀 Starting camera tracking...")
        
        # Test camera tracking for 10 seconds
        start_time = time.time()
        frame_count = 0
        pose_count = 0
        
        while time.time() - start_time < 10.0:  # 10 seconds
            try:
                # Get camera frames
                frames = teleoperator._get_camera_frames()
                if frames:
                    frame_count += 1
                    
                    # Process with ORB-SLAM
                    camera_pose = teleoperator._process_orb_slam(frames)
                    if camera_pose is not None:
                        pose_count += 1
                        
                        # Extract position and orientation
                        position = camera_pose[:3, 3]
                        rotation = camera_pose[:3, :3]
                        
                        # Print status every 2 seconds
                        if int(time.time() - start_time) % 2 == 0 and int(time.time() - start_time) != int(time.time() - start_time - 0.1):
                            print(f"📷 Frame: {frame_count}, Pose: {pose_count}")
                            print(f"   Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
                            print(f"   Distance from origin: {np.linalg.norm(position):.3f} m")
                            print("-" * 30)
                
                time.sleep(0.033)  # ~30 FPS
                
            except KeyboardInterrupt:
                print("\n🛑 User interrupted test")
                break
            except Exception as e:
                print(f"❌ Error in tracking loop: {e}")
                break
        
        # Print final statistics
        elapsed_time = time.time() - start_time
        print(f"\n📊 Tracking Statistics:")
        print(f"   Duration: {elapsed_time:.1f} seconds")
        print(f"   Frames processed: {frame_count}")
        print(f"   Poses estimated: {pose_count}")
        print(f"   Average FPS: {frame_count / elapsed_time:.1f}")
        print(f"   Success rate: {pose_count / max(frame_count, 1) * 100:.1f}%")
        
        if pose_count > 0:
            print("✅ Camera tracking test successful!")
            return True
        else:
            print("❌ No poses estimated - check camera movement and lighting")
            return False
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            teleoperator.disconnect()
            print("🔌 Disconnected from camera")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")


if __name__ == "__main__":
    success = test_camera_connection()
    if not success:
        sys.exit(1) 
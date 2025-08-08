#!/usr/bin/env python3
"""
Demo: ORB-SLAM SO101 Teleoperation

This is a simple demo showing how to use the ORB-SLAM SO101 teleoperator
for camera-based robot control. Move the RealSense camera and watch
the SO101 robot arm follow the movement!

Usage:
    python demo_orb_slam_so101_teleoperation.py
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.teleoperators.orb_slam_so101 import (
    OrbSlamSo101TeleoperatorConfig,
    create_orb_slam_so101_teleoperator
)
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """Main demo function."""
    print("🎯 ORB-SLAM SO101 Teleoperation Demo")
    print("=" * 50)
    print("📷 Move your RealSense camera to control the SO101 robot arm!")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Create a simple configuration
    from lerobot.teleoperators.orb_slam_so101 import (
        CameraConfig, ControlConfig, SafetyConfig
    )
    
    config = OrbSlamSo101TeleoperatorConfig(
        id="demo_orb_slam_so101",
        camera=CameraConfig(
            serial_number_or_name="031522070877",  # Your camera serial number
            fps=30,
            width=640,
            height=480,
            use_depth=True
        ),
        control=ControlConfig(
            control_frequency=30.0,
            camera_to_robot_scale=0.1,  # Scale camera movement to robot movement
            pose_smoothing_alpha=0.7
        ),
        safety=SafetyConfig(
            workspace_limits={
                'x': [-0.3, 0.3],  # Safe workspace limits
                'y': [-0.3, 0.3],
                'z': [0.2, 0.6]
            },
            max_velocity=0.05  # Slow and safe
        )
    )
    
    # Create teleoperator
    teleoperator = create_orb_slam_so101_teleoperator(config)
    
    try:
        # Connect to systems
        print("🔌 Connecting to camera and robot...")
        teleoperator.connect(calibrate=True)
        
        if not teleoperator.is_connected:
            print("❌ Failed to connect!")
            return
        
        print("✅ Connected successfully!")
        print("🚀 Starting teleoperation...")
        
        # Start teleoperation
        teleoperator.start()
        
        # Main loop
        start_time = time.time()
        last_print_time = start_time
        
        while True:
            try:
                # Get current action
                action = teleoperator.get_action()
                
                # Extract data
                camera_pose = np.array(action['camera_pose'])
                target_pose = np.array(action['target_pose'])
                velocity = action['velocity']
                
                # Print status every 2 seconds
                current_time = time.time()
                if current_time - last_print_time > 2.0:
                    camera_pos = camera_pose[:3, 3]
                    robot_pos = target_pose[:3, 3]
                    
                    # Format velocity display
                    if velocity < 0.001:
                        velocity_str = "0.000 m/s (stationary)"
                    else:
                        velocity_str = f"{velocity:.3f} m/s"
                    
                    print(f"📷 Camera: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
                    print(f"🤖 Robot:  [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
                    print(f"⚡ Velocity: {velocity_str}")
                    print("-" * 30)
                    
                    last_print_time = current_time
                
                # Brief pause
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("\n🛑 Demo stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        # Stop teleoperation
        print("🛑 Stopping teleoperation...")
        teleoperator.stop()
        
        print("✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        try:
            teleoperator.disconnect()
            print("🔌 Disconnected")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test Calibrated ORB-SLAM3 Docker Integration Only

This script tests only the calibrated ORB-SLAM3 Docker integration without robot connection.
"""

import time
import sys
import numpy as np
import subprocess
import threading
import re
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orb_slam_docker_calibrated import CalibratedOrbSlamDockerIntegration


def test_calibrated_orb_slam_docker_only():
    """Test only the calibrated ORB-SLAM3 Docker integration."""
    print("🧪 Testing Calibrated ORB-SLAM3 Docker Integration Only")
    print("=" * 50)
    
    # Create integration
    integration = CalibratedOrbSlamDockerIntegration()
    
    try:
        # Test ORB-SLAM3 startup
        print("🐳 Testing calibrated ORB-SLAM3 startup...")
        if not integration.start_orb_slam3():
            print("❌ Calibrated ORB-SLAM3 startup failed")
            return False
        
        print("✅ Calibrated ORB-SLAM3 started successfully")
        
        # Test pose retrieval for a few seconds
        print("📊 Testing pose retrieval...")
        for i in range(30):  # Test for 30 seconds
            pose = integration.get_current_pose()
            print(f"   Frame {i+1}: Pose shape {pose.shape}")
            print(f"   Pose: {pose[:3, 3]}")  # Show translation part
            time.sleep(1)
        
        print("✅ Pose retrieval working")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        integration.stop_orb_slam3()


def check_camera_devices():
    """Check available camera devices."""
    print("🔍 Checking Camera Devices")
    print("=" * 30)
    
    try:
        # List video devices
        result = subprocess.run(["ls", "-la", "/dev/video*"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("📹 Available video devices:")
            print(result.stdout)
        else:
            print("📹 No video devices found or permission issue")
        
        # Check RealSense devices
        result = subprocess.run(["lsusb"], capture_output=True, text=True, check=True)
        print("🔌 USB devices:")
        print(result.stdout)
        
        # Check if RealSense is detected
        if "Intel" in result.stdout:
            print("✅ Intel RealSense detected")
        else:
            print("❌ Intel RealSense not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False


def main():
    """Main test function."""
    print("🎯 Calibrated ORB-SLAM3 Docker Integration Test (No Robot)")
    print("=" * 50)
    
    # Check camera devices first
    if not check_camera_devices():
        print("❌ Camera device check failed")
        return
    
    # Test integration
    if test_calibrated_orb_slam_docker_only():
        print("✅ All tests passed! Calibrated ORB-SLAM3 Docker integration is working.")
        print("🎯 You can now run the full teleoperator with robot control.")
    else:
        print("❌ Tests failed. Check the Docker setup and ORB-SLAM3 configuration.")


if __name__ == "__main__":
    main() 
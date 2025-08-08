#!/usr/bin/env python3
"""
Test ORB-SLAM3 Docker Integration Only

This script tests only the ORB-SLAM3 Docker integration without robot connection.
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

from orb_slam_docker_simple import SimpleOrbSlamDockerIntegration


def test_orb_slam_docker_only():
    """Test only the ORB-SLAM3 Docker integration."""
    print("🧪 Testing ORB-SLAM3 Docker Integration Only")
    print("=" * 50)
    
    # Create integration
    integration = SimpleOrbSlamDockerIntegration()
    
    try:
        # Test ORB-SLAM3 startup
        print("🐳 Testing ORB-SLAM3 startup...")
        if not integration.start_orb_slam3():
            print("❌ ORB-SLAM3 startup failed")
            return False
        
        print("✅ ORB-SLAM3 started successfully")
        
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


def main():
    """Main test function."""
    print("🎯 ORB-SLAM3 Docker Integration Test (No Robot)")
    print("=" * 50)
    
    # Test integration
    if test_orb_slam_docker_only():
        print("✅ All tests passed! ORB-SLAM3 Docker integration is working.")
        print("🎯 You can now run the full teleoperator with robot control.")
    else:
        print("❌ Tests failed. Check the Docker setup and ORB-SLAM3 configuration.")


if __name__ == "__main__":
    main() 
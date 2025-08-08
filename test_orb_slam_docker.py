#!/usr/bin/env python3
"""
Test ORB-SLAM3 Docker Integration

This script tests the basic ORB-SLAM3 Docker integration before running the full teleoperator.
"""

import time
import sys
import subprocess
import json
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orb_slam_docker_integration import OrbSlamDockerIntegration


def test_docker_integration():
    """Test the ORB-SLAM3 Docker integration."""
    print("🧪 Testing ORB-SLAM3 Docker Integration")
    print("=" * 50)
    
    # Create integration
    integration = OrbSlamDockerIntegration()
    
    try:
        # Test container startup
        print("🐳 Testing container startup...")
        if not integration.start_container():
            print("❌ Container startup failed")
            return False
        
        print("✅ Container started successfully")
        
        # Wait a bit for ORB-SLAM3 to initialize
        print("⏳ Waiting for ORB-SLAM3 to initialize...")
        time.sleep(5)
        
        # Test pose retrieval
        print("📊 Testing pose retrieval...")
        for i in range(10):
            pose = integration.get_current_pose()
            print(f"   Frame {i+1}: Pose shape {pose.shape}")
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
        integration.stop_container()


def check_docker_setup():
    """Check the Docker setup and available images."""
    print("🔍 Checking Docker Setup")
    print("=" * 30)
    
    try:
        # Check Docker is running
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, check=True)
        print("✅ Docker is running")
        
        # List ORB-SLAM images
        result = subprocess.run(["docker", "images"], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        
        orb_images = [line for line in lines if 'orb' in line.lower()]
        print(f"📋 Found {len(orb_images)} ORB-SLAM related images:")
        for img in orb_images:
            print(f"   {img}")
        
        # Check running containers
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, check=True)
        print(f"🐳 Running containers: {len(result.stdout.splitlines()) - 1}")
        
        return True
        
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False


def main():
    """Main test function."""
    print("🎯 ORB-SLAM3 Docker Integration Test")
    print("=" * 50)
    
    # Check Docker setup first
    if not check_docker_setup():
        print("❌ Docker setup check failed")
        return
    
    # Test integration
    if test_docker_integration():
        print("✅ All tests passed! ORB-SLAM3 Docker integration is working.")
    else:
        print("❌ Tests failed. Check the Docker setup and ORB-SLAM3 configuration.")


if __name__ == "__main__":
    main() 
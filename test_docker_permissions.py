#!/usr/bin/env python3
"""
Quick test for Docker permissions with RealSense camera
"""

import subprocess
import time
import sys

def test_docker_permissions():
    """Test if Docker can access RealSense camera with new permissions."""
    print("🧪 Testing Docker permissions for RealSense camera...")
    
    # Test basic Docker access
    try:
        result = subprocess.run(["docker", "run", "--rm", "--privileged", 
                               "--device", "/dev/video0:/dev/video0:rw",
                               "--device", "/dev/video1:/dev/video1:rw",
                               "--device", "/dev/video2:/dev/video2:rw",
                               "--device", "/dev/video3:/dev/video3:rw",
                               "--device", "/dev/video4:/dev/video4:rw",
                               "--device", "/dev/video5:/dev/video5:rw",
                               "--device", "/dev/video6:/dev/video6:rw",
                               "--device", "/dev/video7:/dev/video7:rw",
                               "--device", "/dev/bus/usb:/dev/bus/usb",
                               "--cap-add", "SYS_RAWIO",
                               "--cap-add", "SYS_ADMIN",
                               "lmwafer/orb-slam-3-ready:1.1-ubuntu18.04",
                               "/bin/bash", "-c", 
                               "ls -la /dev/video* && echo 'Camera devices accessible!'"],
                               capture_output=True, text=True, timeout=30)
        
        print("✅ Docker container started successfully")
        print("📹 Camera devices in container:")
        print(result.stdout)
        
        if "Camera devices accessible!" in result.stdout:
            print("✅ Camera devices are accessible in Docker!")
            return True
        else:
            print("❌ Camera devices not accessible in Docker")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Docker container timed out")
        return False
    except Exception as e:
        print(f"❌ Docker test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🎯 Docker Permissions Test for RealSense Camera")
    print("=" * 50)
    
    if test_docker_permissions():
        print("✅ Docker permissions are working!")
        print("🎯 You can now run the ORB-SLAM3 Docker teleoperator.")
    else:
        print("❌ Docker permissions need to be fixed.")
        print("🔄 Try logging out and back in, then run this test again.")

if __name__ == "__main__":
    main() 
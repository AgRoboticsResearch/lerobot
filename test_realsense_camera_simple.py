#!/usr/bin/env python3
"""
Simple RealSense Camera Test

This script tests basic RealSense camera connectivity and frame reading.
"""

import time
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def test_realsense_camera():
    """Test RealSense camera connection and frame reading."""
    print("📷 Simple RealSense Camera Test")
    print("=" * 50)
    
    try:
        # Create camera config
        camera_config = RealSenseCameraConfig(
            serial_number_or_name="Intel RealSense D435I",  # Use the actual camera name
            fps=30,
            width=640,
            height=480,
            use_depth=True
        )
        
        print("🔧 Camera config created")
        print(f"   Serial/Name: '{camera_config.serial_number_or_name}'")
        print(f"   Resolution: {camera_config.width}x{camera_config.height}")
        print(f"   FPS: {camera_config.fps}")
        print(f"   Use depth: {camera_config.use_depth}")
        
        # Create camera
        print("\n📷 Creating RealSense camera...")
        camera = RealSenseCamera(camera_config)
        print("✅ Camera object created")
        
        # Connect to camera
        print("\n🔌 Connecting to camera...")
        camera.connect()
        print("✅ Connected to camera")
        
        # Test frame reading
        print("\n📸 Testing frame reading...")
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 10:  # Read 10 frames
            try:
                # Read frame
                frame = camera.async_read()
                if frame is not None:
                    frame_count += 1
                    print(f"   Frame {frame_count}: {frame.shape}")
                    
                    # Brief pause
                    time.sleep(0.1)
                else:
                    print("   No frame received")
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n🛑 User interrupted")
                break
            except Exception as e:
                print(f"❌ Error reading frame: {e}")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 Frame reading test completed:")
        print(f"   Frames read: {frame_count}")
        print(f"   Time elapsed: {elapsed_time:.2f} seconds")
        print(f"   Average FPS: {frame_count / elapsed_time:.1f}")
        
        if frame_count > 0:
            print("✅ Camera test successful!")
            return True
        else:
            print("❌ No frames were read")
            return False
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            if 'camera' in locals():
                camera.disconnect()
                print("🔌 Camera disconnected")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")


if __name__ == "__main__":
    success = test_realsense_camera()
    if not success:
        sys.exit(1) 
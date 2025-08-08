#!/usr/bin/env python3
"""
Direct RealSense Test using pyrealsense2

This script uses pyrealsense2 directly to test camera connectivity
and get the serial number.
"""

import time
import sys
import numpy as np

try:
    import pyrealsense2 as rs
    print("✅ pyrealsense2 imported successfully")
except ImportError as e:
    print(f"❌ pyrealsense2 not available: {e}")
    sys.exit(1)


def test_realsense_direct():
    """Test RealSense camera using pyrealsense2 directly."""
    print("📷 Direct RealSense Test using pyrealsense2")
    print("=" * 50)
    
    try:
        # Create a context
        ctx = rs.context()
        print("✅ RealSense context created")
        
        # Get device list
        devices = ctx.query_devices()
        print(f"📋 Found {len(devices)} RealSense device(s)")
        
        if len(devices) == 0:
            print("❌ No RealSense devices found")
            return False
        
        # List all devices
        for i, device in enumerate(devices):
            print(f"\n📷 Device {i}:")
            print(f"   Name: {device.get_info(rs.camera_info.name)}")
            print(f"   Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"   Product ID: {device.get_info(rs.camera_info.product_id)}")
            print(f"   Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Use the first device
        device = devices[0]
        serial_number = device.get_info(rs.camera_info.serial_number)
        device_name = device.get_info(rs.camera_info.name)
        
        print(f"\n🎯 Using device: {device_name} (Serial: {serial_number})")
        
        # Create pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        print("🔧 Pipeline configured")
        
        # Start pipeline
        print("🚀 Starting pipeline...")
        profile = pipeline.start(config)
        print("✅ Pipeline started")
        
        # Get device from profile
        device = profile.get_device()
        
        # Get depth sensor
        depth_sensor = device.query_sensors()[0]
        
        # Test frame reading
        print("\n📸 Testing frame reading...")
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 10:  # Read 10 frames
            try:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                
                # Get color and depth frames
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    frame_count += 1
                    
                    # Convert to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    print(f"   Frame {frame_count}: Color {color_image.shape}, Depth {depth_image.shape}")
                    
                    # Brief pause
                    time.sleep(0.1)
                else:
                    print("   No frames received")
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n🛑 User interrupted")
                break
            except Exception as e:
                print(f"❌ Error reading frame: {e}")
                break
        
        # Stop pipeline
        pipeline.stop()
        print("🛑 Pipeline stopped")
        
        elapsed_time = time.time() - start_time
        print(f"\n📊 Frame reading test completed:")
        print(f"   Frames read: {frame_count}")
        print(f"   Time elapsed: {elapsed_time:.2f} seconds")
        print(f"   Average FPS: {frame_count / elapsed_time:.1f}")
        
        if frame_count > 0:
            print("✅ Direct RealSense test successful!")
            print(f"📋 Serial number: {serial_number}")
            print(f"📋 Device name: {device_name}")
            return True
        else:
            print("❌ No frames were read")
            return False
        
    except Exception as e:
        print(f"❌ Direct RealSense test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_realsense_direct()
    if not success:
        sys.exit(1) 
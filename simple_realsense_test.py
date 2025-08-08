#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import time

print("=== Simple RealSense Test ===")

# Create pipeline
pipeline = rs.pipeline()
config = rs.config()

# Get device
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("No RealSense devices found!")
    exit(1)

device = devices[0]
print(f"Found device: {device.get_info(rs.camera_info.name)}")
print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")

# Try to enable color stream with lower resolution
try:
    config.enable_stream(rs.stream.color, 0, 424, 240, rs.format.bgr8, 30)
    print("✅ Color stream configured (424x240)")
except Exception as e:
    print(f"❌ Error configuring color stream: {e}")
    # Try even lower resolution
    try:
        config = rs.config()
        config.enable_stream(rs.stream.color, 0, 320, 240, rs.format.bgr8, 15)
        print("✅ Color stream configured (320x240)")
    except Exception as e2:
        print(f"❌ Error configuring color stream (fallback): {e2}")

# Try to start pipeline
try:
    profile = pipeline.start(config)
    print("✅ Pipeline started successfully")
except Exception as e:
    print(f"❌ Error starting pipeline: {e}")
    exit(1)

# Try to get frames
print("Trying to get frames...")
for i in range(5):
    try:
        frames = pipeline.wait_for_frames(timeout_ms=2000)
        color_frame = frames.get_color_frame()
        if color_frame:
            print(f"✅ Frame {i+1} received successfully")
            image = np.asanyarray(color_frame.get_data())
            print(f"   Image shape: {image.shape}")
            cv2.imwrite(f"test_frame_{i+1}.jpg", image)
        else:
            print(f"❌ Frame {i+1}: No color frame")
    except Exception as e:
        print(f"❌ Frame {i+1}: Error - {e}")
    
    time.sleep(1)

pipeline.stop()
print("✅ Test completed") 
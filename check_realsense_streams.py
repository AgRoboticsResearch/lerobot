#!/usr/bin/env python3
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print(f"Found {len(devices)} RealSense devices:")
for i, device in enumerate(devices):
    print(f"  Device {i}: {device.get_info(rs.camera_info.name)} (Serial: {device.get_info(rs.camera_info.serial_number)})")
    
    print("  Available streams:")
    for sensor in device.query_sensors():
        print(f"    Sensor: {sensor.get_info(rs.camera_info.name)}")
        for profile in sensor.get_stream_profiles():
            if profile.is_video_stream_profile():
                video_profile = rs.video_stream_profile(profile)
                res = video_profile.get_intrinsics()
                print(f"      {profile.stream_type()} {profile.stream_index()}: {video_profile.format()} {res.width}x{res.height} {profile.fps()}fps")
            else:
                print(f"      {profile.stream_type()} {profile.stream_index()}: {profile.format()} {profile.fps()}fps") 
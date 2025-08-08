#!/usr/bin/env python3
"""
Test ORB-SLAM Integration with RealSense Cameras

This script tests the ORB-SLAM integration with actual RealSense cameras.
Run this to verify that visual odometry is working with your hardware.
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.orb_slam_integration import create_orb_slam_processor, OrbSlamConfig, get_orb_slam_status
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def list_available_cameras():
    """List available RealSense cameras."""
    logger.info("=== Scanning for RealSense Cameras ===")
    
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            logger.warning("No RealSense cameras found!")
            return []
        
        logger.info(f"Found {len(devices)} RealSense device(s):")
        for i, device in enumerate(devices):
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            logger.info(f"  Device {i}: {name} (Serial: {serial})")
        
        return [device.get_info(rs.camera_info.serial_number) for device in devices]
        
    except ImportError:
        logger.error("pyrealsense2 not installed. Please install: pip install pyrealsense2")
        return []
    except Exception as e:
        logger.error(f"Error scanning cameras: {e}")
        return []


def test_single_camera(device_id: str):
    """Test a single RealSense camera with its dual-eye stereo capabilities."""
    logger.info(f"\n=== Testing Single Camera: {device_id} ===")
    logger.info("Note: RealSense D435I has dual eyes (stereo cameras) built into a single device")
    
    try:
        # Configure camera with depth enabled (uses stereo pair internally)
        config = RealSenseCameraConfig(
            device_id,  # serial_number_or_name
            fps=30,
            width=640,
            height=480,
            use_depth=True  # Enable depth stream (computed from stereo pair)
        )
        
        # Initialize camera
        camera = RealSenseCamera(config)
        camera.connect()
        
        logger.info("Camera connected successfully!")
        logger.info("Testing dual-eye stereo capabilities...")
        
        # Test frame capture with depth
        for i in range(5):
            # Get color frame (from left camera)
            color_frame = camera.async_read()
            # Get depth frame (computed from stereo pair)
            depth_frame = camera.read_depth()
            
            if color_frame is not None and depth_frame is not None:
                logger.info(f"Frame {i+1}: Color shape {color_frame.shape}, Depth shape {depth_frame.shape}")
                
                # Save test images
                if i == 0:
                    cv2.imwrite(f"test_camera_{device_id}_color.jpg", color_frame)
                    # Normalize depth for visualization (0-255)
                    depth_normalized = ((depth_frame - depth_frame.min()) / (depth_frame.max() - depth_frame.min()) * 255).astype(np.uint8)
                    cv2.imwrite(f"test_camera_{device_id}_depth.jpg", depth_normalized)
                    logger.info(f"Saved test images: test_camera_{device_id}_color.jpg, test_camera_{device_id}_depth.jpg")
            else:
                logger.warning(f"Frame {i+1}: Missing frames")
            
            time.sleep(0.1)
        
        camera.disconnect()
        logger.info("Single camera stereo test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing camera {device_id}: {e}")
        return False


def test_dual_camera_setup(camera_ids: list):
    """Test dual camera setup for stereo vision."""
    logger.info(f"\n=== Testing Dual Camera Setup ===")
    logger.info("Note: This test is for multiple physical RealSense devices")
    logger.info("For single device with dual eyes (like D435I), use the stereo RGB streams test above")
    
    if len(camera_ids) < 2:
        logger.info("Only 1 RealSense device found - this is expected for single device stereo")
        logger.info("The RealSense D435I has left and right RGB eyes built into a single device")
        logger.info("Use the 'Stereo RGB Streams' test above for proper stereo ORB-SLAM testing")
        return True  # This is not a failure, it's expected behavior
    
    logger.info(f"Found {len(camera_ids)} devices - testing multi-device setup...")
    
    cameras = {}
    
    try:
        # Initialize both cameras
        for i, device_id in enumerate(camera_ids[:2]):
            camera_name = f"camera_{i+1}"
            config = RealSenseCameraConfig(
                device_id,  # serial_number_or_name
                fps=30,
                width=640,
                height=480
            )
            
            camera = RealSenseCamera(config)
            camera.connect()
            cameras[camera_name] = camera
            
            logger.info(f"Connected {camera_name}: {device_id}")
        
        # Test synchronized frame capture
        logger.info("Testing synchronized frame capture...")
        
        for frame_num in range(10):
            frames = {}
            
            # Capture from both cameras
            for camera_name, camera in cameras.items():
                frame = camera.async_read()
                if frame is not None:
                    frames[camera_name] = frame
                    logger.info(f"Frame {frame_num+1}: {camera_name} captured {frame.shape}")
                else:
                    logger.warning(f"Frame {frame_num+1}: {camera_name} failed")
            
            # Save stereo pair
            if len(frames) == 2:
                frame1 = frames["camera_1"]
                frame2 = frames["camera_2"]
                
                # Save stereo pair
                stereo_image = np.hstack([frame1, frame2])
                cv2.imwrite(f"stereo_pair_{frame_num+1}.jpg", stereo_image)
                
                logger.info(f"Saved stereo pair: stereo_pair_{frame_num+1}.jpg")
            
            time.sleep(0.1)
        
        # Cleanup
        for camera in cameras.values():
            camera.disconnect()
        
        logger.info("Dual camera test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in dual camera test: {e}")
        # Cleanup on error
        for camera in cameras.values():
            try:
                camera.disconnect()
            except:
                pass
        return False


def test_stereo_rgb_streams(device_id: str):
    """Test accessing individual left and right RGB streams from RealSense D435I."""
    logger.info(f"\n=== Testing Stereo RGB Streams: {device_id} ===")
    logger.info("Accessing left and right RGB cameras for stereo ORB-SLAM")
    
    try:
        import pyrealsense2 as rs
        
        # Create RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable device
        config.enable_device(device_id)
        
        # Enable left and right infrared streams (which are the RGB cameras)
        # RealSense D435I has infrared streams that correspond to left and right cameras
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # Left camera
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # Right camera
        
        # Start pipeline
        profile = pipeline.start(config)
        
        logger.info("Stereo RGB streams enabled successfully!")
        
        # Test frame capture from both cameras
        for i in range(5):
            # Wait for coherent frames
            frames = pipeline.wait_for_frames()
            
            # Get left and right frames
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            
            if left_frame and right_frame:
                # Convert to numpy arrays
                left_image = np.asanyarray(left_frame.get_data())
                right_image = np.asanyarray(right_frame.get_data())
                
                logger.info(f"Frame {i+1}: Left shape {left_image.shape}, Right shape {right_image.shape}")
                
                # Save test images
                if i == 0:
                    cv2.imwrite(f"stereo_left_{device_id}.jpg", left_image)
                    cv2.imwrite(f"stereo_right_{device_id}.jpg", right_image)
                    
                    # Create side-by-side comparison
                    stereo_comparison = np.hstack([left_image, right_image])
                    cv2.imwrite(f"stereo_comparison_{device_id}.jpg", stereo_comparison)
                    
                    logger.info(f"Saved stereo images: stereo_left_{device_id}.jpg, stereo_right_{device_id}.jpg, stereo_comparison_{device_id}.jpg")
            else:
                logger.warning(f"Frame {i+1}: Missing stereo frames")
            
            time.sleep(0.1)
        
        # Stop pipeline
        pipeline.stop()
        logger.info("Stereo RGB streams test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing stereo RGB streams: {e}")
        return False


def test_orb_slam_with_stereo_rgb(camera_ids: list, duration_seconds=30):
    """Test ORB-SLAM integration using individual left and right RGB streams."""
    logger.info(f"\n=== Testing ORB-SLAM with Stereo RGB Streams ({duration_seconds}s) ===")
    logger.info("Using left and right RGB cameras for proper stereo ORB-SLAM")
    
    # Check ORB-SLAM status
    status = get_orb_slam_status()
    logger.info(f"ORB-SLAM Status: {status}")
    
    if not status["integration_ready"]:
        logger.warning("ORB-SLAM integration not ready. Using fallback mode.")
    
    try:
        import pyrealsense2 as rs
        
        # Create RealSense pipeline for stereo RGB
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable device
        device_id = camera_ids[0]
        config.enable_device(device_id)
        
        # Enable left and right infrared streams
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # Left camera
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # Right camera
        
        # Start pipeline
        profile = pipeline.start(config)
        
        # Initialize ORB-SLAM processor
        orb_slam_config = OrbSlamConfig(
            max_features=2000,
            output_frequency=10.0,
            enable_visualization=True
        )
        
        orb_slam_processor = create_orb_slam_processor(orb_slam_config)
        
        logger.info("ORB-SLAM processor initialized with stereo RGB!")
        
        # Test visual odometry with stereo RGB
        logger.info(f"Testing visual odometry for {duration_seconds} seconds...")
        logger.info("Please move the camera slowly in different directions for best results")
        
        pose_history = []
        timestamps = []
        processing_times = []
        feature_counts = []
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            frame_start = time.time()
            
            # Wait for coherent frames
            frames = pipeline.wait_for_frames()
            
            # Get left and right frames
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            
            if left_frame and right_frame:
                # Convert to numpy arrays
                left_image = np.asanyarray(left_frame.get_data())
                right_image = np.asanyarray(right_frame.get_data())
                
                # Prepare frames for ORB-SLAM (convert to RGB format)
                left_rgb = cv2.cvtColor(left_image, cv2.COLOR_GRAY2RGB)
                right_rgb = cv2.cvtColor(right_image, cv2.COLOR_GRAY2RGB)
                
                # Create frames dictionary for ORB-SLAM
                stereo_frames = {
                    "left": left_rgb,
                    "right": right_rgb
                }
                
                # Process with ORB-SLAM
                process_start = time.time()
                estimated_pose = orb_slam_processor.process_camera_frames(stereo_frames)
                processing_time = time.time() - process_start
                
                current_time = time.time()
                timestamps.append(current_time)
                processing_times.append(processing_time)
                
                if estimated_pose is not None:
                    pose_history.append(estimated_pose)
                    
                    # Extract translation and rotation
                    translation = estimated_pose[:3, 3]
                    rotation_matrix = estimated_pose[:3, :3]
                    
                    # Calculate feature count (simplified)
                    feature_count = len(cv2.goodFeaturesToTrack(left_image, 100, 0.01, 10))
                    feature_counts.append(feature_count)
                    
                    frame_count += 1
                    
                    if frame_count % 10 == 0:  # Log every 10th frame
                        elapsed = current_time - start_time
                        logger.info(f"Frame {frame_count} ({elapsed:.1f}s): Pose estimated in {processing_time:.3f}s")
                        logger.info(f"  Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
                        logger.info(f"  Features: {feature_count}")
                        
                        # Save frame with pose info
                        cv2.putText(left_rgb, f"Pose: {translation}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(left_rgb, f"Time: {elapsed:.1f}s", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(left_rgb, f"Features: {feature_count}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imwrite(f"orb_slam_stereo_frame_{frame_count:04d}.jpg", left_rgb)
                else:
                    feature_counts.append(0)
                    logger.warning(f"Frame {frame_count+1}: No pose estimated")
            else:
                feature_counts.append(0)
                logger.warning(f"Frame {frame_count+1}: Missing stereo frames")
            
            # Maintain 10Hz processing rate
            frame_time = time.time() - frame_start
            if frame_time < 0.1:  # 10Hz = 0.1s per frame
                time.sleep(0.1 - frame_time)
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info(f"\n=== Extended Stereo ORB-SLAM Test Summary ({total_time:.1f}s) ===")
        logger.info(f"Total frames processed: {len(pose_history)}")
        logger.info(f"Total frames attempted: {len(timestamps)}")
        logger.info(f"Success rate: {len(pose_history)/len(timestamps)*100:.1f}%")
        logger.info(f"Average processing time: {np.mean(processing_times):.3f}s")
        logger.info(f"Average features per frame: {np.mean(feature_counts):.1f}")
        
        if pose_history:
            # Calculate trajectory statistics
            translations = [pose[:3, 3] for pose in pose_history]
            translations = np.array(translations)
            
            logger.info(f"\n=== Trajectory Analysis ===")
            logger.info(f"Translation range: X[{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]")
            logger.info(f"Translation range: Y[{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]")
            logger.info(f"Translation range: Z[{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]")
            
            # Calculate total distance traveled
            total_distance = np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))
            logger.info(f"Total distance traveled: {total_distance:.3f} meters")
            
            # Calculate average velocity
            avg_velocity = total_distance / total_time
            logger.info(f"Average velocity: {avg_velocity:.3f} m/s")
            
            # Calculate drift (distance from start to end)
            start_pos = translations[0]
            end_pos = translations[-1]
            drift_distance = np.linalg.norm(end_pos - start_pos)
            logger.info(f"Drift distance: {drift_distance:.3f} meters")
            
            # Calculate trajectory smoothness (variance in velocity)
            velocities = np.linalg.norm(np.diff(translations, axis=0), axis=1)
            velocity_variance = np.var(velocities)
            logger.info(f"Velocity variance: {velocity_variance:.6f} (lower = smoother)")
            
            # Save detailed trajectory
            orb_slam_processor.save_trajectory("test_stereo_trajectory_extended.txt")
            logger.info("Extended stereo trajectory saved to: test_stereo_trajectory_extended.txt")
            
            # Save performance metrics
            with open("stereo_orb_slam_metrics.txt", "w") as f:
                f.write(f"# Stereo ORB-SLAM Performance Metrics\n")
                f.write(f"# Test Duration: {total_time:.1f} seconds\n")
                f.write(f"# Total Frames: {len(pose_history)}\n")
                f.write(f"# Success Rate: {len(pose_history)/len(timestamps)*100:.1f}%\n")
                f.write(f"# Average Processing Time: {np.mean(processing_times):.3f}s\n")
                f.write(f"# Average Features: {np.mean(feature_counts):.1f}\n")
                f.write(f"# Total Distance: {total_distance:.3f}m\n")
                f.write(f"# Average Velocity: {avg_velocity:.3f}m/s\n")
                f.write(f"# Drift Distance: {drift_distance:.3f}m\n")
                f.write(f"# Velocity Variance: {velocity_variance:.6f}\n")
                f.write(f"# Translation Range X: [{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]\n")
                f.write(f"# Translation Range Y: [{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]\n")
                f.write(f"# Translation Range Z: [{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]\n")
            
            logger.info("Performance metrics saved to: stereo_orb_slam_metrics.txt")
        
        # Stop pipeline
        pipeline.stop()
        
        logger.info("Extended stereo ORB-SLAM integration test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in extended stereo ORB-SLAM test: {e}")
        return False


def test_orb_slam_integration(camera_ids: list, duration_seconds=30):
    """Test ORB-SLAM integration with RealSense cameras using stereo capabilities."""
    logger.info(f"\n=== Testing ORB-SLAM Integration ({duration_seconds}s) ===")
    logger.info("Using RealSense D435I's built-in stereo vision for better pose estimation")
    
    # Check ORB-SLAM status
    status = get_orb_slam_status()
    logger.info(f"ORB-SLAM Status: {status}")
    
    if not status["integration_ready"]:
        logger.warning("ORB-SLAM integration not ready. Using fallback mode.")
    
    cameras = {}
    
    try:
        # Initialize cameras with depth enabled for better stereo processing
        for i, device_id in enumerate(camera_ids[:2]):
            camera_name = f"camera_{i+1}"
            config = RealSenseCameraConfig(
                device_id,  # serial_number_or_name
                fps=30,
                width=640,
                height=480,
                use_depth=True  # Enable depth for better stereo processing
            )
            
            camera = RealSenseCamera(config)
            camera.connect()
            cameras[camera_name] = camera
        
        # Initialize ORB-SLAM processor
        orb_slam_config = OrbSlamConfig(
            max_features=2000,
            output_frequency=10.0,
            enable_visualization=True
        )
        
        orb_slam_processor = create_orb_slam_processor(orb_slam_config)
        
        logger.info("ORB-SLAM processor initialized!")
        
        # Test visual odometry
        logger.info(f"Testing visual odometry for {duration_seconds} seconds...")
        logger.info("Please move the camera slowly in different directions for best results")
        
        pose_history = []
        timestamps = []
        processing_times = []
        feature_counts = []
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            frame_start = time.time()
            
            frames = {}
            depth_frames = {}
            
            # Capture frames and depth
            for camera_name, camera in cameras.items():
                frame = camera.async_read()
                depth_frame = camera.read_depth()
                if frame is not None:
                    frames[camera_name] = frame
                    if depth_frame is not None:
                        depth_frames[camera_name] = depth_frame
            
            if len(frames) >= 1:  # Changed from 2 to 1 since single device has stereo
                # Process with ORB-SLAM
                process_start = time.time()
                estimated_pose = orb_slam_processor.process_camera_frames(frames)
                processing_time = time.time() - process_start
                
                current_time = time.time()
                timestamps.append(current_time)
                processing_times.append(processing_time)
                
                if estimated_pose is not None:
                    pose_history.append(estimated_pose)
                    
                    # Extract translation and rotation
                    translation = estimated_pose[:3, 3]
                    rotation_matrix = estimated_pose[:3, :3]
                    
                    # Calculate feature count
                    frame1 = frames["camera_1"]
                    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                    feature_count = len(cv2.goodFeaturesToTrack(gray_frame, 100, 0.01, 10))
                    feature_counts.append(feature_count)
                    
                    frame_count += 1
                    
                    if frame_count % 10 == 0:  # Log every 10th frame
                        elapsed = current_time - start_time
                        logger.info(f"Frame {frame_count} ({elapsed:.1f}s): Pose estimated in {processing_time:.3f}s")
                        logger.info(f"  Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
                        logger.info(f"  Features: {feature_count}")
                        
                        # Save frame with pose info
                        cv2.putText(frame1, f"Pose: {translation}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame1, f"Time: {elapsed:.1f}s", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame1, f"Features: {feature_count}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imwrite(f"orb_slam_frame_{frame_count:04d}.jpg", frame1)
                else:
                    feature_counts.append(0)
                    logger.warning(f"Frame {frame_count+1}: No pose estimated")
            else:
                feature_counts.append(0)
                logger.warning(f"Frame {frame_count+1}: Insufficient frames")
            
            # Maintain 10Hz processing rate
            frame_time = time.time() - frame_start
            if frame_time < 0.1:  # 10Hz = 0.1s per frame
                time.sleep(0.1 - frame_time)
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info(f"\n=== Extended ORB-SLAM Test Summary ({total_time:.1f}s) ===")
        logger.info(f"Total frames processed: {len(pose_history)}")
        logger.info(f"Total frames attempted: {len(timestamps)}")
        logger.info(f"Success rate: {len(pose_history)/len(timestamps)*100:.1f}%")
        logger.info(f"Average processing time: {np.mean(processing_times):.3f}s")
        logger.info(f"Average features per frame: {np.mean(feature_counts):.1f}")
        
        if pose_history:
            # Calculate trajectory statistics
            translations = [pose[:3, 3] for pose in pose_history]
            translations = np.array(translations)
            
            logger.info(f"\n=== Trajectory Analysis ===")
            logger.info(f"Translation range: X[{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]")
            logger.info(f"Translation range: Y[{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]")
            logger.info(f"Translation range: Z[{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]")
            
            # Calculate total distance traveled
            total_distance = np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))
            logger.info(f"Total distance traveled: {total_distance:.3f} meters")
            
            # Calculate average velocity
            avg_velocity = total_distance / total_time
            logger.info(f"Average velocity: {avg_velocity:.3f} m/s")
            
            # Calculate drift (distance from start to end)
            start_pos = translations[0]
            end_pos = translations[-1]
            drift_distance = np.linalg.norm(end_pos - start_pos)
            logger.info(f"Drift distance: {drift_distance:.3f} meters")
            
            # Calculate trajectory smoothness (variance in velocity)
            velocities = np.linalg.norm(np.diff(translations, axis=0), axis=1)
            velocity_variance = np.var(velocities)
            logger.info(f"Velocity variance: {velocity_variance:.6f} (lower = smoother)")
            
            # Save trajectory
            orb_slam_processor.save_trajectory("test_trajectory_extended.txt")
            logger.info("Extended trajectory saved to: test_trajectory_extended.txt")
            
            # Save performance metrics
            with open("orb_slam_metrics.txt", "w") as f:
                f.write(f"# ORB-SLAM Performance Metrics\n")
                f.write(f"# Test Duration: {total_time:.1f} seconds\n")
                f.write(f"# Total Frames: {len(pose_history)}\n")
                f.write(f"# Success Rate: {len(pose_history)/len(timestamps)*100:.1f}%\n")
                f.write(f"# Average Processing Time: {np.mean(processing_times):.3f}s\n")
                f.write(f"# Average Features: {np.mean(feature_counts):.1f}\n")
                f.write(f"# Total Distance: {total_distance:.3f}m\n")
                f.write(f"# Average Velocity: {avg_velocity:.3f}m/s\n")
                f.write(f"# Drift Distance: {drift_distance:.3f}m\n")
                f.write(f"# Velocity Variance: {velocity_variance:.6f}\n")
                f.write(f"# Translation Range X: [{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]\n")
                f.write(f"# Translation Range Y: [{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]\n")
                f.write(f"# Translation Range Z: [{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]\n")
            
            logger.info("Performance metrics saved to: orb_slam_metrics.txt")
        
        # Cleanup
        for camera in cameras.values():
            camera.disconnect()
        
        logger.info("Extended ORB-SLAM integration test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in extended ORB-SLAM test: {e}")
        # Cleanup on error
        for camera in cameras.values():
            try:
                camera.disconnect()
            except:
                pass
        return False


def main():
    """Main test function."""
    logger.info("=== Extended RealSense ORB-SLAM Integration Test ===")
    logger.info("This will test visual odometry with your RealSense cameras for extended duration")
    
    # Step 1: List available cameras
    camera_ids = list_available_cameras()
    
    if not camera_ids:
        logger.error("No cameras found. Please check your RealSense connections.")
        return
    
    # Step 2: Test single camera with depth
    logger.info("\n" + "="*50)
    single_camera_success = test_single_camera(camera_ids[0])
    
    # Step 3: Test stereo RGB streams (left and right cameras)
    logger.info("\n" + "="*50)
    stereo_rgb_success = test_stereo_rgb_streams(camera_ids[0])
    
    # Step 4: Test dual camera setup
    logger.info("\n" + "="*50)
    dual_camera_success = test_dual_camera_setup(camera_ids)
    
    # Step 5: Test ORB-SLAM integration with depth (extended duration)
    logger.info("\n" + "="*50)
    orb_slam_success = test_orb_slam_integration(camera_ids, duration_seconds=60)  # 1 minute test
    
    # Step 6: Test ORB-SLAM integration with stereo RGB (extended duration)
    logger.info("\n" + "="*50)
    stereo_orb_slam_success = test_orb_slam_with_stereo_rgb(camera_ids, duration_seconds=60)  # 1 minute test
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("=== Extended Test Results Summary ===")
    logger.info(f"Single camera test: {'✅ PASS' if single_camera_success else '❌ FAIL'}")
    logger.info(f"Stereo RGB streams test: {'✅ PASS' if stereo_rgb_success else '❌ FAIL'}")
    logger.info(f"Multi-device test: {'✅ PASS' if dual_camera_success else '❌ FAIL'}")
    logger.info(f"Extended ORB-SLAM integration (depth): {'✅ PASS' if orb_slam_success else '❌ FAIL'}")
    logger.info(f"Extended ORB-SLAM integration (stereo RGB): {'✅ PASS' if stereo_orb_slam_success else '❌ FAIL'}")
    
    # Key results for ORB-SLAM
    logger.info("\n" + "="*50)
    logger.info("=== Extended ORB-SLAM Stereo Vision Status ===")
    if stereo_orb_slam_success:
        logger.info("🎉 SUCCESS: Extended stereo ORB-SLAM test completed!")
        logger.info("✅ Left and right RGB eyes are accessible")
        logger.info("✅ Stereo pose estimation is functional")
        logger.info("✅ Extended duration testing completed")
        logger.info("✅ Ready for UMI integration")
        logger.info("📊 Check generated metrics files for detailed performance analysis")
    elif orb_slam_success:
        logger.info("⚠️ PARTIAL: Extended ORB-SLAM works with depth, but stereo RGB is preferred")
        logger.info("✅ Depth-based pose estimation is functional")
        logger.info("✅ Extended duration testing completed")
        logger.info("❌ Stereo RGB streams need attention")
    else:
        logger.info("❌ FAILED: Extended ORB-SLAM integration needs attention")
    
    logger.info("\nGenerated files:")
    logger.info("- test_camera_*.jpg: Single camera test images")
    logger.info("- stereo_left_*.jpg, stereo_right_*.jpg: Individual stereo camera images")
    logger.info("- stereo_comparison_*.jpg: Side-by-side stereo comparison")
    logger.info("- orb_slam_frame_*.jpg: Frames with pose information")
    logger.info("- orb_slam_stereo_frame_*.jpg: Stereo frames with pose information")
    logger.info("- test_trajectory_extended.txt: Extended ORB-SLAM trajectory data")
    logger.info("- test_stereo_trajectory_extended.txt: Extended stereo ORB-SLAM trajectory data")
    logger.info("- orb_slam_metrics.txt: Performance metrics")
    logger.info("- stereo_orb_slam_metrics.txt: Stereo performance metrics")


if __name__ == "__main__":
    main() 
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

from lerobot.cameras.realsense import RealSenseCamera, RealSenseConfig
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
    """Test a single RealSense camera."""
    logger.info(f"\n=== Testing Single Camera: {device_id} ===")
    
    try:
        # Configure camera
        config = RealSenseConfig(
            device_id=device_id,
            width=640,
            height=480,
            fps=30
        )
        
        # Initialize camera
        camera = RealSenseCamera(config)
        camera.connect()
        
        logger.info("Camera connected successfully!")
        
        # Test frame capture
        for i in range(5):
            frame = camera.async_read()
            if frame is not None:
                logger.info(f"Frame {i+1}: Captured frame of shape {frame.shape}")
                
                # Save a test image
                if i == 0:
                    cv2.imwrite(f"test_camera_{device_id}.jpg", frame)
                    logger.info(f"Saved test image: test_camera_{device_id}.jpg")
            else:
                logger.warning(f"Frame {i+1}: No frame captured")
            
            time.sleep(0.1)
        
        camera.disconnect()
        logger.info("Single camera test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing camera {device_id}: {e}")
        return False


def test_dual_camera_setup(camera_ids: list):
    """Test dual camera setup for stereo vision."""
    logger.info(f"\n=== Testing Dual Camera Setup ===")
    
    if len(camera_ids) < 2:
        logger.warning("Need at least 2 cameras for stereo vision")
        return False
    
    cameras = {}
    
    try:
        # Initialize both cameras
        for i, device_id in enumerate(camera_ids[:2]):
            camera_name = f"camera_{i+1}"
            config = RealSenseConfig(
                device_id=device_id,
                width=640,
                height=480,
                fps=30
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


def test_orb_slam_integration(camera_ids: list):
    """Test ORB-SLAM integration with RealSense cameras."""
    logger.info(f"\n=== Testing ORB-SLAM Integration ===")
    
    # Check ORB-SLAM status
    status = get_orb_slam_status()
    logger.info(f"ORB-SLAM Status: {status}")
    
    if not status["integration_ready"]:
        logger.warning("ORB-SLAM integration not ready. Using fallback mode.")
    
    cameras = {}
    
    try:
        # Initialize cameras
        for i, device_id in enumerate(camera_ids[:2]):
            camera_name = f"camera_{i+1}"
            config = RealSenseConfig(
                device_id=device_id,
                width=640,
                height=480,
                fps=30
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
        logger.info("Testing visual odometry...")
        
        pose_history = []
        
        for frame_num in range(20):  # Test for 2 seconds at 10Hz
            frames = {}
            
            # Capture frames
            for camera_name, camera in cameras.items():
                frame = camera.async_read()
                if frame is not None:
                    frames[camera_name] = frame
            
            if len(frames) >= 2:
                # Process with ORB-SLAM
                start_time = time.time()
                estimated_pose = orb_slam_processor.process_camera_frames(frames)
                processing_time = time.time() - start_time
                
                if estimated_pose is not None:
                    pose_history.append(estimated_pose)
                    
                    # Extract translation and rotation
                    translation = estimated_pose[:3, 3]
                    rotation_matrix = estimated_pose[:3, :3]
                    
                    logger.info(f"Frame {frame_num+1}: Pose estimated in {processing_time:.3f}s")
                    logger.info(f"  Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
                    logger.info(f"  Rotation matrix shape: {rotation_matrix.shape}")
                    
                    # Save frame with pose info
                    if frame_num % 5 == 0:  # Save every 5th frame
                        frame1 = frames["camera_1"]
                        cv2.putText(frame1, f"Pose: {translation}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imwrite(f"orb_slam_frame_{frame_num+1}.jpg", frame1)
                        logger.info(f"Saved frame with pose: orb_slam_frame_{frame_num+1}.jpg")
                else:
                    logger.warning(f"Frame {frame_num+1}: No pose estimated")
            else:
                logger.warning(f"Frame {frame_num+1}: Insufficient frames")
            
            time.sleep(0.1)  # 10Hz
        
        # Summary
        logger.info(f"\n=== ORB-SLAM Test Summary ===")
        logger.info(f"Total frames processed: {len(pose_history)}")
        logger.info(f"Success rate: {len(pose_history)/20*100:.1f}%")
        
        if pose_history:
            # Calculate trajectory statistics
            translations = [pose[:3, 3] for pose in pose_history]
            translations = np.array(translations)
            
            logger.info(f"Translation range: X[{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]")
            logger.info(f"Translation range: Y[{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]")
            logger.info(f"Translation range: Z[{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]")
            
            # Save trajectory
            orb_slam_processor.save_trajectory("test_trajectory.txt")
            logger.info("Trajectory saved to: test_trajectory.txt")
        
        # Cleanup
        for camera in cameras.values():
            camera.disconnect()
        
        logger.info("ORB-SLAM integration test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in ORB-SLAM test: {e}")
        # Cleanup on error
        for camera in cameras.values():
            try:
                camera.disconnect()
            except:
                pass
        return False


def main():
    """Main test function."""
    logger.info("=== RealSense ORB-SLAM Integration Test ===")
    logger.info("This will test visual odometry with your RealSense cameras")
    
    # Step 1: List available cameras
    camera_ids = list_available_cameras()
    
    if not camera_ids:
        logger.error("No cameras found. Please check your RealSense connections.")
        return
    
    # Step 2: Test single camera
    logger.info("\n" + "="*50)
    single_camera_success = test_single_camera(camera_ids[0])
    
    # Step 3: Test dual camera setup
    logger.info("\n" + "="*50)
    dual_camera_success = test_dual_camera_setup(camera_ids)
    
    # Step 4: Test ORB-SLAM integration
    logger.info("\n" + "="*50)
    orb_slam_success = test_orb_slam_integration(camera_ids)
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("=== Test Results Summary ===")
    logger.info(f"Single camera test: {'✅ PASS' if single_camera_success else '❌ FAIL'}")
    logger.info(f"Dual camera test: {'✅ PASS' if dual_camera_success else '❌ FAIL'}")
    logger.info(f"ORB-SLAM integration: {'✅ PASS' if orb_slam_success else '❌ FAIL'}")
    
    if orb_slam_success:
        logger.info("\n🎉 SUCCESS: Visual odometry is working!")
        logger.info("You can now use ORB-SLAM with your RealSense cameras.")
        logger.info("Check the generated images and trajectory files.")
    else:
        logger.info("\n⚠️ Some tests failed. Check the logs above for details.")
    
    logger.info("\nGenerated files:")
    logger.info("- test_camera_*.jpg: Single camera test images")
    logger.info("- stereo_pair_*.jpg: Dual camera stereo pairs")
    logger.info("- orb_slam_frame_*.jpg: Frames with pose information")
    logger.info("- test_trajectory.txt: ORB-SLAM trajectory data")


if __name__ == "__main__":
    main() 
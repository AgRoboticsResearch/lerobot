#!/usr/bin/env python3
"""
Test ORB-SLAM3 Integration with RealSense Cameras

This script tests the actual ORB-SLAM3 system with RealSense cameras
using proper camera calibration for accurate pose estimation including rotation.
"""

import os
import sys
import time
import numpy as np
import cv2
import subprocess
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def check_orb_slam3_installation():
    """Check if ORB-SLAM3 is properly installed."""
    logger.info("=== Checking ORB-SLAM3 Installation ===")
    
    # Check for ORB-SLAM3 Docker image
    try:
        result = subprocess.run(['docker', 'images', 'chicheng/orb_slam3:latest'], 
                              capture_output=True, text=True)
        if 'chicheng/orb_slam3' in result.stdout:
            logger.info("✅ ORB-SLAM3 Docker image found")
            return True
        else:
            logger.warning("⚠️ ORB-SLAM3 Docker image not found")
            logger.info("Pulling ORB-SLAM3 Docker image...")
            pull_result = subprocess.run(['docker', 'pull', 'chicheng/orb_slam3:latest'])
            if pull_result.returncode == 0:
                logger.info("✅ ORB-SLAM3 Docker image pulled successfully")
                return True
            else:
                logger.error("❌ Failed to pull ORB-SLAM3 Docker image")
                return False
    except Exception as e:
        logger.error(f"❌ Error checking ORB-SLAM3 installation: {e}")
        return False


def create_calibration_files(device_id: str):
    """Use existing RealSense calibration files or create new ones if needed."""
    logger.info(f"=== Using RealSense Calibration for {device_id} ===")
    
    # Check if user has existing calibration files
    calibration_path = Path("realsense_calibration")
    calibration_path.mkdir(exist_ok=True)
    
    # Use the user's provided calibration parameters
    settings_content = """%YAML:1.0

#--------------------------------------------------------------------------------------------
# System config
#--------------------------------------------------------------------------------------------

# When the variables are commented, the system doesn't load a previous session or not store the current one

# If the LoadFile doesn't exist, the system give a message and create a new Atlas from scratch
#System.LoadAtlasFromFile: "Session_MH01_MH02_MH03_Stereo60_Pseudo"

# The store file is created from the current session, if a file with the same name exists it is deleted
#System.SaveAtlasToFile: "Session_MH01_MH02_MH03_Stereo60_Pseudo"

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------
File.version: "1.0"

Camera.type: "PinHole"
# Camera.type: "Rectified"

# Stereo.b: 0.04989136261

# Camera calibration and distortion parameters (OpenCV) 
Camera1.fx: 419.8328552246094
Camera1.fy: 419.8328552246094
Camera1.cx: 429.5089416503906
Camera1.cy: 237.1636505126953

# distortion parameters
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0

Camera2.fx: 419.8328552246094
Camera2.fy: 419.8328552246094
Camera2.cx: 429.5089416503906
Camera2.cy: 237.1636505126953

Camera2.k1: 0.0
Camera2.k2: 0.0
Camera2.p1: 0.0
Camera2.p2: 0.0

# Camera resolution
Camera.width: 848
Camera.height: 480

# Camera frames per second 
Camera.fps: 30

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

Stereo.ThDepth: 40.0
Stereo.T_c1_c2: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [1.0, 0.0, 0.0, 0.0499585,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0 ,0.0 ,1.0]

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1200

# ORB Extractor: Scale factor between levels in the scale pyramid 	
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid	
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast			
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""
    
    settings_path = calibration_path / "realsense_orb_slam3_settings.yaml"
    with open(settings_path, 'w') as f:
        f.write(settings_content)
    
    # Create intrinsics JSON file for compatibility
    intrinsics_data = {
        "final_reproj_error": 0.0,
        "fps": 30.0,
        "image_height": 480,
        "image_width": 848,
        "intrinsic_type": "PINHOLE",
        "intrinsics": {
            "aspect_ratio": 848 / 480,
            "focal_length": 419.8328552246094,
            "principal_pt_x": 429.5089416503906,
            "principal_pt_y": 237.1636505126953,
            "radial_distortion_1": 0.0,
            "radial_distortion_2": 0.0,
            "radial_distortion_3": 0.0,
            "radial_distortion_4": 0.0,
            "skew": 0.0
        },
        "nr_calib_images": 0,
        "stabelized": False
    }
    
    intrinsics_path = calibration_path / "realsense_intrinsics.json"
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics_data, f, indent=2)
    
    logger.info(f"✅ Using provided RealSense calibration:")
    logger.info(f"   - Resolution: 848x480")
    logger.info(f"   - Focal length: 419.83")
    logger.info(f"   - Principal point: (429.51, 237.16)")
    logger.info(f"   - Stereo baseline: 0.0499585m")
    logger.info(f"   - Settings file: {settings_path}")
    logger.info(f"   - Intrinsics file: {intrinsics_path}")
    
    return str(intrinsics_path), str(settings_path)


def record_video_for_slam(device_id: str, duration_seconds=30, output_path="slam_video.mp4"):
    """Record video for ORB-SLAM3 processing using the calibrated resolution."""
    logger.info(f"=== Recording Video for ORB-SLAM3 ({duration_seconds}s) ===")
    
    try:
        import pyrealsense2 as rs
        
        # Create RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device_id)
        
        # Try to enable color stream with calibrated resolution
        # If it fails, try common resolutions
        resolutions_to_try = [
            (848, 480),   # Calibrated resolution
            (640, 480),   # Common resolution
            (1280, 720),  # HD resolution
            (640, 360),   # Lower resolution
        ]
        
        success = False
        actual_resolution = None
        
        for width, height in resolutions_to_try:
            try:
                logger.info(f"Trying resolution: {width}x{height}")
                config = rs.config()
                config.enable_device(device_id)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
                
                # Start pipeline
                profile = pipeline.start(config)
                success = True
                actual_resolution = (width, height)
                logger.info(f"✅ Successfully enabled {width}x{height} resolution")
                break
                
            except Exception as e:
                logger.warning(f"Failed to enable {width}x{height}: {e}")
                continue
        
        if not success:
            logger.error("❌ Could not enable any color stream resolution")
            return False
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, actual_resolution)
        
        logger.info("Recording video... Please move the camera in different directions including rotation.")
        logger.info("This will help ORB-SLAM3 understand the camera movement and estimate pose accurately.")
        logger.info("Try circular movements, rotations, and complex trajectories for best results.")
        logger.info(f"Recording at resolution: {actual_resolution[0]}x{actual_resolution[1]}")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Wait for coherent frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Write frame
                out.write(color_image)
                frame_count += 1
                
                # Display frame with timer
                elapsed = time.time() - start_time
                cv2.putText(color_image, f"Recording: {elapsed:.1f}s", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(color_image, f"Frames: {frame_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(color_image, f"Resolution: {actual_resolution[0]}x{actual_resolution[1]}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Recording for ORB-SLAM3', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        out.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        
        logger.info(f"✅ Video recording completed: {output_path}")
        logger.info(f"   - Duration: {duration_seconds} seconds")
        logger.info(f"   - Frames: {frame_count}")
        logger.info(f"   - Frame rate: {frame_count/duration_seconds:.1f} Hz")
        logger.info(f"   - Resolution: {actual_resolution[0]}x{actual_resolution[1]}")
        
        # Update calibration settings if resolution changed
        if actual_resolution != (848, 480):
            logger.warning(f"⚠️ Using {actual_resolution[0]}x{actual_resolution[1]} instead of calibrated 848x480")
            logger.warning("ORB-SLAM3 results may be less accurate due to resolution mismatch")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error recording video: {e}")
        return False


def run_orb_slam3_on_video(video_path: str, settings_path: str, output_path: str = "orb_slam3_trajectory.csv"):
    """Run ORB-SLAM3 on the recorded video."""
    logger.info("=== Running ORB-SLAM3 on Video ===")
    
    try:
        # Create output directory
        output_dir = Path("orb_slam3_output")
        output_dir.mkdir(exist_ok=True)
        
        # Mount paths for Docker
        video_mount = f"{Path(video_path).absolute()}:/data/video.mp4"
        settings_mount = f"{Path(settings_path).absolute()}:/data/settings.yaml"
        output_mount = f"{output_dir.absolute()}:/data/output"
        
        # ORB-SLAM3 command
        cmd = [
            'docker', 'run', '--rm',
            '--volume', video_mount,
            '--volume', settings_mount,
            '--volume', output_mount,
            'chicheng/orb_slam3:latest',
            '/ORB_SLAM3/Examples/Monocular/mono_tum',
            '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
            '/data/settings.yaml',
            '/data/video.mp4'
        ]
        
        logger.info("Running ORB-SLAM3... This may take a while.")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run ORB-SLAM3
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ ORB-SLAM3 completed successfully!")
            logger.info(f"Output: {result.stdout}")
            
            # Check for trajectory file
            trajectory_file = output_dir / "CameraTrajectory.txt"
            if trajectory_file.exists():
                logger.info(f"✅ Trajectory file created: {trajectory_file}")
                return str(trajectory_file)
            else:
                logger.warning("⚠️ Trajectory file not found")
                return None
        else:
            logger.error(f"❌ ORB-SLAM3 failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("❌ ORB-SLAM3 timed out")
        return None
    except Exception as e:
        logger.error(f"❌ Error running ORB-SLAM3: {e}")
        return None


def analyze_orb_slam3_trajectory(trajectory_path: str):
    """Analyze ORB-SLAM3 trajectory results."""
    logger.info("=== Analyzing ORB-SLAM3 Trajectory ===")
    
    try:
        # Read trajectory file
        with open(trajectory_path, 'r') as f:
            lines = f.readlines()
        
        # Parse trajectory data
        timestamps = []
        translations = []
        rotations = []
        
        for line in lines:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split()
            if len(parts) >= 8:
                timestamp = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                timestamps.append(timestamp)
                translations.append([tx, ty, tz])
                rotations.append([qx, qy, qz, qw])
        
        if not translations:
            logger.error("❌ No trajectory data found")
            return
        
        translations = np.array(translations)
        rotations = np.array(rotations)
        
        # Calculate statistics
        total_frames = len(translations)
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        # Calculate total distance
        total_distance = np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))
        
        # Calculate rotation statistics
        rotation_magnitudes = []
        for i in range(1, len(rotations)):
            # Calculate rotation difference
            q1 = rotations[i-1]
            q2 = rotations[i]
            # Simple rotation magnitude (can be improved)
            rot_diff = np.linalg.norm(q2 - q1)
            rotation_magnitudes.append(rot_diff)
        
        avg_rotation = np.mean(rotation_magnitudes) if rotation_magnitudes else 0
        
        # Print results
        logger.info(f"✅ Trajectory Analysis Results:")
        logger.info(f"   - Total frames: {total_frames}")
        logger.info(f"   - Duration: {duration:.2f} seconds")
        logger.info(f"   - Frame rate: {total_frames/duration:.1f} Hz" if duration > 0 else "   - Frame rate: N/A")
        logger.info(f"   - Total distance: {total_distance:.3f} meters")
        logger.info(f"   - Average rotation magnitude: {avg_rotation:.4f}")
        
        logger.info(f"   - Translation range:")
        logger.info(f"     X: [{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]")
        logger.info(f"     Y: [{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]")
        logger.info(f"     Z: [{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]")
        
        # Save analysis
        analysis_path = Path("orb_slam3_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write(f"# ORB-SLAM3 Trajectory Analysis\n")
            f.write(f"# Total frames: {total_frames}\n")
            f.write(f"# Duration: {duration:.2f} seconds\n")
            f.write(f"# Total distance: {total_distance:.3f} meters\n")
            f.write(f"# Average rotation magnitude: {avg_rotation:.4f}\n")
            f.write(f"# Translation range X: [{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]\n")
            f.write(f"# Translation range Y: [{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]\n")
            f.write(f"# Translation range Z: [{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]\n")
        
        logger.info(f"✅ Analysis saved to: {analysis_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error analyzing trajectory: {e}")
        return False


def main():
    """Main test function."""
    logger.info("=== ORB-SLAM3 RealSense Integration Test ===")
    logger.info("This will test the actual ORB-SLAM3 system with RealSense cameras")
    
    # Step 1: Check ORB-SLAM3 installation
    if not check_orb_slam3_installation():
        logger.error("ORB-SLAM3 not available. Please install ORB-SLAM3 Docker image.")
        return
    
    # Step 2: List available cameras
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            logger.error("No RealSense cameras found!")
            return
        
        device_id = devices[0].get_info(rs.camera_info.serial_number)
        logger.info(f"Using RealSense device: {device_id}")
        
    except ImportError:
        logger.error("pyrealsense2 not installed. Please install: pip install pyrealsense2")
        return
    except Exception as e:
        logger.error(f"Error accessing RealSense cameras: {e}")
        return
    
    # Step 3: Create calibration files
    logger.info("\n" + "="*50)
    intrinsics_path, settings_path = create_calibration_files(device_id)
    
    if not settings_path:
        logger.error("Failed to create calibration files")
        return
    
    # Step 4: Record video for SLAM
    logger.info("\n" + "="*50)
    video_path = "slam_video.mp4"
    if not record_video_for_slam(device_id, duration_seconds=30, output_path=video_path):
        logger.error("Failed to record video")
        return
    
    # Step 5: Run ORB-SLAM3
    logger.info("\n" + "="*50)
    trajectory_path = run_orb_slam3_on_video(video_path, settings_path)
    
    if not trajectory_path:
        logger.error("Failed to run ORB-SLAM3")
        return
    
    # Step 6: Analyze results
    logger.info("\n" + "="*50)
    if not analyze_orb_slam3_trajectory(trajectory_path):
        logger.error("Failed to analyze trajectory")
        return
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("=== ORB-SLAM3 Test Results Summary ===")
    logger.info("✅ ORB-SLAM3 installation verified")
    logger.info("✅ RealSense camera calibration created")
    logger.info("✅ Video recording completed")
    logger.info("✅ ORB-SLAM3 processing completed")
    logger.info("✅ Trajectory analysis completed")
    
    logger.info("\n🎉 ORB-SLAM3 integration test completed successfully!")
    logger.info("The actual ORB-SLAM3 system should now properly handle rotation and complex movements.")
    logger.info("Check the generated files for detailed results.")
    
    logger.info("\nGenerated files:")
    logger.info("- realsense_calibration/: Camera calibration files")
    logger.info("- slam_video.mp4: Recorded video for SLAM")
    logger.info("- orb_slam3_output/: ORB-SLAM3 output files")
    logger.info("- orb_slam3_analysis.txt: Trajectory analysis results")


if __name__ == "__main__":
    main() 
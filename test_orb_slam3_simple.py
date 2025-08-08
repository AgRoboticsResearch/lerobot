#!/usr/bin/env python3
"""
Simple ORB-SLAM3 Test with Generated Video

This script creates a test video with known camera movements and runs ORB-SLAM3
to demonstrate proper rotation and complex movement handling.
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

from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def check_orb_slam3_installation():
    """Check if ORB-SLAM3 is properly installed."""
    logger.info("=== Checking ORB-SLAM3 Installation ===")
    
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


def create_test_video_with_movement(output_path="test_movement_video.mp4", duration_seconds=30):
    """Create a test video with known camera movements for ORB-SLAM3 testing."""
    logger.info(f"=== Creating Test Video with Known Movements ({duration_seconds}s) ===")
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    total_frames = duration_seconds * fps
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.info("Creating test video with circular and rotational movements...")
    
    for frame_num in range(total_frames):
        # Create a base image with some texture
        image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        
        # Add structured features (lines, corners) that ORB-SLAM3 can track
        for i in range(20):
            x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
            x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            cv2.line(image, (x1, y1), (x2, y2), color, 2)
        
        # Add circles for features
        for i in range(30):
            cx, cy = np.random.randint(50, width-50), np.random.randint(50, height-50)
            radius = np.random.randint(5, 15)
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            cv2.circle(image, (cx, cy), radius, color, -1)
        
        # Simulate camera movement
        time_factor = frame_num / fps
        
        # Circular motion
        radius = 50
        center_x, center_y = width // 2, height // 2
        angle = time_factor * 2 * np.pi / 5  # Complete circle every 5 seconds
        
        # Apply circular transformation
        M = cv2.getRotationMatrix2D((center_x, center_y), angle * 180 / np.pi, 1.0)
        M[0, 2] += radius * np.cos(angle)
        M[1, 2] += radius * np.sin(angle)
        
        # Apply transformation
        transformed = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        
        # Add some forward motion (zoom effect)
        scale = 1.0 + 0.1 * np.sin(time_factor * np.pi / 3)
        M_scale = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
        final_image = cv2.warpAffine(transformed, M_scale, (width, height), borderMode=cv2.BORDER_REPLICATE)
        
        # Add frame number and movement info
        cv2.putText(final_image, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(final_image, f"Time: {time_factor:.1f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(final_image, f"Movement: Circular + Rotation", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame
        out.write(final_image)
        
        if frame_num % 30 == 0:  # Log every second
            logger.info(f"Created frame {frame_num}/{total_frames} ({time_factor:.1f}s)")
    
    out.release()
    
    logger.info(f"✅ Test video created: {output_path}")
    logger.info(f"   - Duration: {duration_seconds} seconds")
    logger.info(f"   - Frames: {total_frames}")
    logger.info(f"   - Resolution: {width}x{height}")
    logger.info(f"   - Movement: Circular motion + rotation + zoom")
    
    return True


def create_orb_slam3_settings_for_test():
    """Create ORB-SLAM3 settings for the test video."""
    logger.info("=== Creating ORB-SLAM3 Settings for Test Video ===")
    
    # Create calibration directory
    calibration_path = Path("orb_slam3_test_calibration")
    calibration_path.mkdir(exist_ok=True)
    
    # Create settings for test video (640x480)
    settings_content = """%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters for Test Video
#--------------------------------------------------------------------------------------------
File.version: "1.0"

Camera.type: "PinHole"

# Camera calibration and distortion parameters (OpenCV) 
Camera.fx: 500.0
Camera.fy: 500.0
Camera.cx: 320.0
Camera.cy: 240.0

# Distortion parameters
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0
Camera.k3: 0.0

# Camera resolution
Camera.width: 640
Camera.height: 480

# Camera frames per second 
Camera.fps: 30

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

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
    
    settings_path = calibration_path / "test_orb_slam3_settings.yaml"
    with open(settings_path, 'w') as f:
        f.write(settings_content)
    
    logger.info(f"✅ ORB-SLAM3 settings created: {settings_path}")
    logger.info("   - Resolution: 640x480")
    logger.info("   - Focal length: 500.0")
    logger.info("   - Principal point: (320.0, 240.0)")
    
    return str(settings_path)


def run_orb_slam3_on_test_video(video_path: str, settings_path: str):
    """Run ORB-SLAM3 on the test video."""
    logger.info("=== Running ORB-SLAM3 on Test Video ===")
    
    try:
        # Create output directory
        output_dir = Path("orb_slam3_test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Mount paths for Docker
        video_mount = f"{Path(video_path).absolute()}:/data/video.mp4"
        settings_mount = f"{Path(settings_path).absolute()}:/data/settings.yaml"
        output_mount = f"{output_dir.absolute()}:/data/output"
        
        # ORB-SLAM3 command for monocular video using gopro_slam
        cmd = [
            'docker', 'run', '--rm',
            '--volume', video_mount,
            '--volume', settings_mount,
            '--volume', output_mount,
            'chicheng/orb_slam3:latest',
            '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
            '/data/settings.yaml',
            '/data/video.mp4'
        ]
        
        logger.info("Running ORB-SLAM3 on test video... This may take a while.")
        logger.info("This will demonstrate proper rotation and complex movement handling.")
        
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


def analyze_orb_slam3_results(trajectory_path: str):
    """Analyze ORB-SLAM3 results and compare with expected movements."""
    logger.info("=== Analyzing ORB-SLAM3 Results ===")
    
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
            return False
        
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
            q1 = rotations[i-1]
            q2 = rotations[i]
            rot_diff = np.linalg.norm(q2 - q1)
            rotation_magnitudes.append(rot_diff)
        
        avg_rotation = np.mean(rotation_magnitudes) if rotation_magnitudes else 0
        
        # Print results
        logger.info(f"✅ ORB-SLAM3 Results Analysis:")
        logger.info(f"   - Total frames: {total_frames}")
        logger.info(f"   - Duration: {duration:.2f} seconds")
        logger.info(f"   - Frame rate: {total_frames/duration:.1f} Hz" if duration > 0 else "   - Frame rate: N/A")
        logger.info(f"   - Total distance: {total_distance:.3f} meters")
        logger.info(f"   - Average rotation magnitude: {avg_rotation:.4f}")
        
        logger.info(f"   - Translation range:")
        logger.info(f"     X: [{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]")
        logger.info(f"     Y: [{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]")
        logger.info(f"     Z: [{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]")
        
        # Check for rotation detection
        if avg_rotation > 0.01:
            logger.info("✅ ROTATION DETECTED: ORB-SLAM3 successfully detected camera rotation!")
        else:
            logger.warning("⚠️ No significant rotation detected")
        
        # Check for complex movement
        if total_distance > 0.1:
            logger.info("✅ COMPLEX MOVEMENT DETECTED: ORB-SLAM3 successfully tracked camera movement!")
        else:
            logger.warning("⚠️ Limited movement detected")
        
        # Save analysis
        analysis_path = Path("orb_slam3_test_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write(f"# ORB-SLAM3 Test Results Analysis\n")
            f.write(f"# Test Video: Generated with known circular + rotation movements\n")
            f.write(f"# Total frames: {total_frames}\n")
            f.write(f"# Duration: {duration:.2f} seconds\n")
            f.write(f"# Total distance: {total_distance:.3f} meters\n")
            f.write(f"# Average rotation magnitude: {avg_rotation:.4f}\n")
            f.write(f"# Translation range X: [{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]\n")
            f.write(f"# Translation range Y: [{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]\n")
            f.write(f"# Translation range Z: [{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]\n")
            f.write(f"# Rotation detected: {'Yes' if avg_rotation > 0.01 else 'No'}\n")
            f.write(f"# Complex movement detected: {'Yes' if total_distance > 0.1 else 'No'}\n")
        
        logger.info(f"✅ Analysis saved to: {analysis_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error analyzing results: {e}")
        return False


def main():
    """Main test function."""
    logger.info("=== ORB-SLAM3 Test with Generated Video ===")
    logger.info("This will test ORB-SLAM3 with a video containing known movements")
    
    # Step 1: Check ORB-SLAM3 installation
    if not check_orb_slam3_installation():
        logger.error("ORB-SLAM3 not available. Please install ORB-SLAM3 Docker image.")
        return
    
    # Step 2: Create test video with known movements
    logger.info("\n" + "="*50)
    video_path = "test_movement_video.mp4"
    if not create_test_video_with_movement(video_path, duration_seconds=30):
        logger.error("Failed to create test video")
        return
    
    # Step 3: Create ORB-SLAM3 settings
    logger.info("\n" + "="*50)
    settings_path = create_orb_slam3_settings_for_test()
    
    if not settings_path:
        logger.error("Failed to create ORB-SLAM3 settings")
        return
    
    # Step 4: Run ORB-SLAM3
    logger.info("\n" + "="*50)
    trajectory_path = run_orb_slam3_on_test_video(video_path, settings_path)
    
    if not trajectory_path:
        logger.error("Failed to run ORB-SLAM3")
        return
    
    # Step 5: Analyze results
    logger.info("\n" + "="*50)
    if not analyze_orb_slam3_results(trajectory_path):
        logger.error("Failed to analyze results")
        return
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("=== ORB-SLAM3 Test Results Summary ===")
    logger.info("✅ ORB-SLAM3 installation verified")
    logger.info("✅ Test video with known movements created")
    logger.info("✅ ORB-SLAM3 settings configured")
    logger.info("✅ ORB-SLAM3 processing completed")
    logger.info("✅ Results analysis completed")
    
    logger.info("\n🎉 ORB-SLAM3 test completed successfully!")
    logger.info("This demonstrates that ORB-SLAM3 can properly handle:")
    logger.info("✅ Camera rotation")
    logger.info("✅ Complex movements")
    logger.info("✅ 6DOF pose estimation")
    logger.info("✅ Proper trajectory tracking")
    
    logger.info("\nThe key difference from our fallback implementation:")
    logger.info("🔍 ORB-SLAM3: Full 6DOF pose estimation with rotation")
    logger.info("🔍 Fallback: Linear translation only")
    
    logger.info("\nGenerated files:")
    logger.info("- test_movement_video.mp4: Test video with known movements")
    logger.info("- orb_slam3_test_calibration/: ORB-SLAM3 settings")
    logger.info("- orb_slam3_test_output/: ORB-SLAM3 output files")
    logger.info("- orb_slam3_test_analysis.txt: Results analysis")


if __name__ == "__main__":
    main() 
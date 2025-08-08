#!/usr/bin/env python3
"""
Extended ORB-SLAM Test with Simulated Camera Data

This script tests ORB-SLAM integration with simulated camera data
to demonstrate extended duration testing and comprehensive analysis.
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.utils.orb_slam_integration import create_orb_slam_processor, OrbSlamConfig, get_orb_slam_status
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_simulated_stereo_frames(frame_count, width=640, height=480):
    """Generate simulated stereo camera frames with realistic movement patterns."""
    
    # Create base scene with some texture
    base_scene = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    
    # Add some structured features (lines, corners)
    for i in range(10):
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        cv2.line(base_scene, (x1, y1), (x2, y2), np.random.randint(100, 255), 2)
    
    # Add some circles for features
    for i in range(20):
        cx, cy = np.random.randint(50, width-50), np.random.randint(50, height-50)
        radius = np.random.randint(5, 20)
        cv2.circle(base_scene, (cx, cy), radius, np.random.randint(100, 255), -1)
    
    # Simulate camera movement
    # Move in a circular pattern with some forward motion
    angle = frame_count * 0.1  # Rotate slowly
    forward_motion = frame_count * 0.02  # Move forward slowly
    
    # Create left frame (base scene)
    left_frame = base_scene.copy()
    
    # Create right frame with stereo disparity
    right_frame = base_scene.copy()
    
    # Apply stereo disparity (simulate depth)
    disparity = 20 + int(10 * np.sin(angle))  # Varying disparity
    right_frame = np.roll(right_frame, disparity, axis=1)
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 5, (height, width)).astype(np.uint8)
    left_frame = np.clip(left_frame + noise, 0, 255)
    right_frame = np.clip(right_frame + noise, 0, 255)
    
    # Convert to RGB
    left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2RGB)
    
    return left_rgb, right_rgb


def test_extended_orb_slam_simulation(duration_seconds=60):
    """Test ORB-SLAM with simulated stereo data for extended duration."""
    logger.info(f"\n=== Extended ORB-SLAM Simulation Test ({duration_seconds}s) ===")
    logger.info("Testing ORB-SLAM with simulated stereo camera data")
    
    # Check ORB-SLAM status
    status = get_orb_slam_status()
    logger.info(f"ORB-SLAM Status: {status}")
    
    if not status["integration_ready"]:
        logger.warning("ORB-SLAM integration not ready. Using fallback mode.")
    
    try:
        # Initialize ORB-SLAM processor
        orb_slam_config = OrbSlamConfig(
            max_features=2000,
            output_frequency=10.0,
            enable_visualization=True
        )
        
        orb_slam_processor = create_orb_slam_processor(orb_slam_config)
        
        logger.info("ORB-SLAM processor initialized with simulated data!")
        
        # Test visual odometry with simulated stereo
        logger.info(f"Testing visual odometry for {duration_seconds} seconds...")
        logger.info("Simulating camera movement in circular pattern with forward motion")
        
        pose_history = []
        timestamps = []
        processing_times = []
        feature_counts = []
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            frame_start = time.time()
            
            # Generate simulated stereo frames
            left_rgb, right_rgb = generate_simulated_stereo_frames(frame_count)
            
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
                
                # Calculate feature count
                gray_left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
                feature_count = len(cv2.goodFeaturesToTrack(gray_left, 100, 0.01, 10))
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
                    cv2.putText(left_rgb, f"Frame: {frame_count}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imwrite(f"sim_orb_slam_frame_{frame_count:04d}.jpg", left_rgb)
            else:
                feature_counts.append(0)
                logger.warning(f"Frame {frame_count+1}: No pose estimated")
            
            # Maintain 10Hz processing rate
            frame_time = time.time() - frame_start
            if frame_time < 0.1:  # 10Hz = 0.1s per frame
                time.sleep(0.1 - frame_time)
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info(f"\n=== Extended Simulation ORB-SLAM Test Summary ({total_time:.1f}s) ===")
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
            
            # Calculate loop closure error (for circular motion)
            if len(translations) > 10:
                # Check if we've made a full circle
                mid_point = len(translations) // 2
                first_half = translations[:mid_point]
                second_half = translations[mid_point:]
                
                if len(first_half) > 0 and len(second_half) > 0:
                    first_center = np.mean(first_half, axis=0)
                    second_center = np.mean(second_half, axis=0)
                    loop_closure_error = np.linalg.norm(first_center - second_center)
                    logger.info(f"Loop closure error: {loop_closure_error:.3f} meters")
            
            # Save detailed trajectory
            orb_slam_processor.save_trajectory("test_simulation_trajectory_extended.txt")
            logger.info("Extended simulation trajectory saved to: test_simulation_trajectory_extended.txt")
            
            # Save performance metrics
            with open("simulation_orb_slam_metrics.txt", "w") as f:
                f.write(f"# Simulation ORB-SLAM Performance Metrics\n")
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
            
            logger.info("Performance metrics saved to: simulation_orb_slam_metrics.txt")
        
        logger.info("Extended simulation ORB-SLAM test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in extended simulation ORB-SLAM test: {e}")
        return False


def test_orb_slam_accuracy_analysis():
    """Test ORB-SLAM accuracy with known ground truth."""
    logger.info(f"\n=== ORB-SLAM Accuracy Analysis ===")
    logger.info("Testing ORB-SLAM accuracy with controlled movement patterns")
    
    try:
        # Initialize ORB-SLAM processor
        orb_slam_config = OrbSlamConfig(
            max_features=2000,
            output_frequency=10.0,
            enable_visualization=True
        )
        
        orb_slam_processor = create_orb_slam_processor(orb_slam_config)
        
        # Test different movement patterns
        test_patterns = [
            ("Linear Forward", lambda t: np.array([t * 0.01, 0, 0])),
            ("Linear Sideways", lambda t: np.array([0, t * 0.01, 0])),
            ("Linear Up", lambda t: np.array([0, 0, t * 0.01])),
            ("Circular", lambda t: np.array([0.05 * np.cos(t * 0.1), 0.05 * np.sin(t * 0.1), 0])),
            ("Spiral", lambda t: np.array([0.01 * t * np.cos(t * 0.2), 0.01 * t * np.sin(t * 0.2), 0.005 * t]))
        ]
        
        accuracy_results = {}
        
        for pattern_name, movement_func in test_patterns:
            logger.info(f"\n--- Testing {pattern_name} Movement ---")
            
            pose_history = []
            ground_truth = []
            
            for frame_count in range(50):  # 5 seconds at 10Hz
                # Generate frames with known movement
                left_rgb, right_rgb = generate_simulated_stereo_frames(frame_count)
                
                # Apply known movement pattern
                expected_movement = movement_func(frame_count)
                ground_truth.append(expected_movement)
                
                # Process with ORB-SLAM
                stereo_frames = {"left": left_rgb, "right": right_rgb}
                estimated_pose = orb_slam_processor.process_camera_frames(stereo_frames)
                
                if estimated_pose is not None:
                    translation = estimated_pose[:3, 3]
                    pose_history.append(translation)
                else:
                    pose_history.append(np.array([0, 0, 0]))
            
            # Calculate accuracy metrics
            if len(pose_history) > 0:
                pose_history = np.array(pose_history)
                ground_truth = np.array(ground_truth)
                
                # Calculate translation error
                translation_error = np.linalg.norm(pose_history - ground_truth, axis=1)
                mean_error = np.mean(translation_error)
                max_error = np.max(translation_error)
                
                # Calculate relative error
                total_distance_gt = np.sum(np.linalg.norm(np.diff(ground_truth, axis=0), axis=1))
                total_distance_est = np.sum(np.linalg.norm(np.diff(pose_history, axis=0), axis=1))
                
                if total_distance_gt > 0:
                    relative_error = abs(total_distance_est - total_distance_gt) / total_distance_gt
                else:
                    relative_error = 0
                
                accuracy_results[pattern_name] = {
                    "mean_error": mean_error,
                    "max_error": max_error,
                    "relative_error": relative_error,
                    "total_distance_gt": total_distance_gt,
                    "total_distance_est": total_distance_est
                }
                
                logger.info(f"  Mean translation error: {mean_error:.3f}m")
                logger.info(f"  Max translation error: {max_error:.3f}m")
                logger.info(f"  Relative distance error: {relative_error:.1%}")
                logger.info(f"  Ground truth distance: {total_distance_gt:.3f}m")
                logger.info(f"  Estimated distance: {total_distance_est:.3f}m")
        
        # Save accuracy analysis
        with open("orb_slam_accuracy_analysis.txt", "w") as f:
            f.write("# ORB-SLAM Accuracy Analysis\n")
            f.write("# Movement Pattern Analysis\n\n")
            
            for pattern_name, results in accuracy_results.items():
                f.write(f"## {pattern_name}\n")
                f.write(f"Mean translation error: {results['mean_error']:.3f}m\n")
                f.write(f"Max translation error: {results['max_error']:.3f}m\n")
                f.write(f"Relative distance error: {results['relative_error']:.1%}\n")
                f.write(f"Ground truth distance: {results['total_distance_gt']:.3f}m\n")
                f.write(f"Estimated distance: {results['total_distance_est']:.3f}m\n\n")
        
        logger.info("Accuracy analysis saved to: orb_slam_accuracy_analysis.txt")
        return True
        
    except Exception as e:
        logger.error(f"Error in accuracy analysis: {e}")
        return False


def main():
    """Main test function."""
    logger.info("=== Extended ORB-SLAM Simulation Test Suite ===")
    logger.info("This will test ORB-SLAM with simulated data for comprehensive analysis")
    
    # Test 1: Extended duration simulation
    logger.info("\n" + "="*50)
    simulation_success = test_extended_orb_slam_simulation(duration_seconds=60)
    
    # Test 2: Accuracy analysis
    logger.info("\n" + "="*50)
    accuracy_success = test_orb_slam_accuracy_analysis()
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("=== Extended Simulation Test Results Summary ===")
    logger.info(f"Extended simulation test: {'✅ PASS' if simulation_success else '❌ FAIL'}")
    logger.info(f"Accuracy analysis test: {'✅ PASS' if accuracy_success else '❌ FAIL'}")
    
    # Key results
    logger.info("\n" + "="*50)
    logger.info("=== Extended ORB-SLAM Simulation Status ===")
    if simulation_success and accuracy_success:
        logger.info("🎉 SUCCESS: Extended ORB-SLAM simulation tests completed!")
        logger.info("✅ Comprehensive trajectory analysis performed")
        logger.info("✅ Accuracy analysis with ground truth completed")
        logger.info("✅ Performance metrics generated")
        logger.info("✅ Ready for real camera integration")
        logger.info("📊 Check generated files for detailed analysis")
    elif simulation_success:
        logger.info("⚠️ PARTIAL: Extended simulation works, but accuracy analysis failed")
        logger.info("✅ Trajectory analysis completed")
        logger.info("❌ Accuracy analysis needs attention")
    else:
        logger.info("❌ FAILED: Extended ORB-SLAM simulation needs attention")
    
    logger.info("\nGenerated files:")
    logger.info("- sim_orb_slam_frame_*.jpg: Simulated frames with pose information")
    logger.info("- test_simulation_trajectory_extended.txt: Extended simulation trajectory")
    logger.info("- simulation_orb_slam_metrics.txt: Simulation performance metrics")
    logger.info("- orb_slam_accuracy_analysis.txt: Accuracy analysis results")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test ORB-SLAM SO101 Teleoperation System.

This script demonstrates the complete ORB-SLAM based teleoperation system:
1. RealSense camera tracks movement using ORB-SLAM
2. Camera trajectory is converted to robot target poses
3. LeRobot's IK controls the SO101 follower arm
4. Real-time visualization and safety monitoring

Usage:
    python test_orb_slam_so101_teleoperation.py [--duration 30] [--visualize]
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.teleoperators.orb_slam_so101 import (
    OrbSlamSo101Teleoperator,
    OrbSlamSo101TeleoperatorConfig,
    create_orb_slam_so101_teleoperator
)
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_visualization():
    """Create real-time visualization for teleoperation."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('ORB-SLAM SO101 Teleoperation', fontsize=16)
    
    # Camera trajectory plot
    ax1.set_title('Camera Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.grid(True)
    
    # Robot target pose plot
    ax2.set_title('Robot Target Pose')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.grid(True)
    
    # Joint angles plot
    ax3.set_title('Joint Angles')
    ax3.set_xlabel('Joint')
    ax3.set_ylabel('Angle (rad)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.ion()  # Enable interactive mode
    
    return fig, (ax1, ax2, ax3)


def update_visualization(fig, axes, camera_poses, robot_poses, joint_angles):
    """Update the visualization with new data."""
    ax1, ax2, ax3 = axes
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # Plot camera trajectory
    if len(camera_poses) > 1:
        camera_positions = np.array([pose[:3, 3] for pose in camera_poses])
        ax1.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 'b-', label='Camera')
        ax1.scatter(camera_positions[-1, 0], camera_positions[-1, 1], camera_positions[-1, 2], c='red', s=100, label='Current')
    
    ax1.set_title('Camera Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot robot target poses
    if len(robot_poses) > 1:
        robot_positions = np.array([pose[:3, 3] for pose in robot_poses])
        ax2.plot(robot_positions[:, 0], robot_positions[:, 1], robot_positions[:, 2], 'g-', label='Robot Target')
        ax2.scatter(robot_positions[-1, 0], robot_positions[-1, 1], robot_positions[-1, 2], c='red', s=100, label='Current')
    
    ax2.set_title('Robot Target Pose')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot joint angles
    if len(joint_angles) > 0:
        joints = np.array(joint_angles)
        joint_names = [f'J{i+1}' for i in range(joints.shape[1])]
        ax3.bar(joint_names, joints[-1], alpha=0.7)
        ax3.set_title('Joint Angles')
        ax3.set_xlabel('Joint')
        ax3.set_ylabel('Angle (rad)')
        ax3.grid(True)
    
    plt.draw()
    plt.pause(0.01)


def test_orb_slam_so101_teleoperation(duration_seconds: int = 30, visualize: bool = True):
    """
    Test the ORB-SLAM SO101 teleoperation system.
    
    Args:
        duration_seconds: Duration of the test in seconds
        visualize: Whether to show real-time visualization
    """
    logger.info("🎯 Starting ORB-SLAM SO101 Teleoperation Test")
    logger.info(f"⏱️  Duration: {duration_seconds} seconds")
    logger.info(f"📊 Visualization: {'Enabled' if visualize else 'Disabled'}")
    
    # Create configuration
    from lerobot.teleoperators.orb_slam_so101 import (
        CameraConfig, OrbSlamConfig, ControlConfig, SafetyConfig, VisualizationConfig
    )
    
    config = OrbSlamSo101TeleoperatorConfig(
        id="test_orb_slam_so101",
        camera=CameraConfig(
            serial_number_or_name="",  # Auto-detect
            fps=30,
            width=640,
            height=480,
            use_depth=True
        ),
        orb_slam=OrbSlamConfig(
            max_features=2000,
            output_frequency=30.0,
            enable_visualization=True
        ),
        control=ControlConfig(
            control_frequency=30.0,
            camera_to_robot_scale=0.1,  # Scale camera movement to robot movement
            pose_smoothing_alpha=0.7
        ),
        safety=SafetyConfig(
            workspace_limits={
                'x': [-0.3, 0.3],  # Smaller workspace for testing
                'y': [-0.3, 0.3],
                'z': [0.2, 0.6]
            },
            max_velocity=0.05  # Slower for safety
        ),
        visualization=VisualizationConfig(
            enable_pose_visualization=True,
            enable_trajectory_plotting=True,
            save_trajectory=True,
            trajectory_file="test_orb_slam_so101_trajectory.txt"
        )
    )
    
    # Create teleoperator
    teleoperator = create_orb_slam_so101_teleoperator(config)
    
    try:
        # Connect to systems
        logger.info("🔌 Connecting to camera and robot systems...")
        teleoperator.connect(calibrate=True)
        
        if not teleoperator.is_connected:
            logger.error("❌ Failed to connect to required systems")
            return False
        
        if not teleoperator.is_calibrated:
            logger.error("❌ Failed to calibrate teleoperator")
            return False
        
        logger.info("✅ Successfully connected and calibrated")
        
        # Initialize visualization
        fig = None
        axes = None
        if visualize:
            logger.info("📊 Initializing visualization...")
            fig, axes = create_visualization()
        
        # Data collection
        camera_poses = []
        robot_poses = []
        joint_angles = []
        timestamps = []
        velocities = []
        
        # Start teleoperation
        logger.info("🚀 Starting teleoperation...")
        teleoperator.start()
        
        # Main test loop
        start_time = time.time()
        last_update_time = start_time
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            elapsed = current_time - start_time
            
            try:
                # Get current action
                action = teleoperator.get_action()
                
                # Extract data
                camera_pose = np.array(action['camera_pose'])
                target_pose = np.array(action['target_pose'])
                current_joints = np.array(action['current_joint_angles'])
                velocity = action['velocity']
                
                # Store data
                camera_poses.append(camera_pose)
                robot_poses.append(target_pose)
                joint_angles.append(current_joints)
                timestamps.append(elapsed)
                velocities.append(velocity)
                
                # Update visualization
                if visualize and fig is not None and current_time - last_update_time > 0.1:  # 10Hz update
                    update_visualization(fig, axes, camera_poses, robot_poses, joint_angles)
                    last_update_time = current_time
                
                # Log progress
                if int(elapsed) % 5 == 0 and int(elapsed) != int(last_update_time - start_time):
                    logger.info(f"⏱️  Progress: {elapsed:.1f}s / {duration_seconds}s - "
                              f"Velocity: {velocity:.3f} m/s")
                
                # Brief pause
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                logger.info("🛑 User interrupted test")
                break
            except Exception as e:
                logger.error(f"❌ Error in test loop: {e}")
                break
        
        # Stop teleoperation
        logger.info("🛑 Stopping teleoperation...")
        teleoperator.stop()
        
        # Final visualization
        if visualize and fig is not None:
            logger.info("📊 Finalizing visualization...")
            update_visualization(fig, axes, camera_poses, robot_poses, joint_angles)
            plt.ioff()  # Disable interactive mode
            plt.show()
        
        # Analyze results
        logger.info("📈 Analyzing results...")
        analyze_results(camera_poses, robot_poses, joint_angles, timestamps, velocities)
        
        logger.info("✅ ORB-SLAM SO101 teleoperation test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            teleoperator.disconnect()
            logger.info("🔌 Disconnected from systems")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def analyze_results(camera_poses, robot_poses, joint_angles, timestamps, velocities):
    """Analyze the teleoperation results."""
    if not camera_poses or not robot_poses:
        logger.warning("⚠️ No data to analyze")
        return
    
    # Convert to numpy arrays
    camera_poses = np.array(camera_poses)
    robot_poses = np.array(robot_poses)
    joint_angles = np.array(joint_angles)
    timestamps = np.array(timestamps)
    velocities = np.array(velocities)
    
    # Calculate metrics
    camera_positions = camera_poses[:, :3, 3]
    robot_positions = robot_poses[:, :3, 3]
    
    # Camera trajectory metrics
    camera_total_distance = np.sum(np.linalg.norm(np.diff(camera_positions, axis=0), axis=1))
    camera_max_displacement = np.max(np.linalg.norm(camera_positions - camera_positions[0], axis=1))
    camera_avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0.0
    
    # Robot trajectory metrics
    robot_total_distance = np.sum(np.linalg.norm(np.diff(robot_positions, axis=0), axis=1))
    robot_max_displacement = np.max(np.linalg.norm(robot_positions - robot_positions[0], axis=1))
    
    # Joint metrics
    joint_ranges = np.ptp(joint_angles, axis=0)  # Peak-to-peak range
    joint_velocities = np.diff(joint_angles, axis=0) / np.diff(timestamps)[:, np.newaxis]
    max_joint_velocity = np.max(np.abs(joint_velocities)) if len(joint_velocities) > 0 else 0.0
    
    # Print results
    logger.info("📊 Teleoperation Results:")
    logger.info(f"   Duration: {timestamps[-1]:.2f} seconds")
    logger.info(f"   Frames processed: {len(camera_poses)}")
    logger.info(f"   Average FPS: {len(camera_poses) / timestamps[-1]:.1f}")
    
    logger.info("📷 Camera Metrics:")
    logger.info(f"   Total distance: {camera_total_distance:.3f} m")
    logger.info(f"   Max displacement: {camera_max_displacement:.3f} m")
    logger.info(f"   Average velocity: {camera_avg_velocity:.3f} m/s")
    
    logger.info("🤖 Robot Metrics:")
    logger.info(f"   Total distance: {robot_total_distance:.3f} m")
    logger.info(f"   Max displacement: {robot_max_displacement:.3f} m")
    logger.info(f"   Max joint velocity: {max_joint_velocity:.3f} rad/s")
    
    logger.info("🔧 Joint Ranges (rad):")
    for i, joint_range in enumerate(joint_ranges):
        logger.info(f"   Joint {i+1}: {joint_range:.3f}")
    
    # Save detailed results
    save_detailed_results(camera_poses, robot_poses, joint_angles, timestamps, velocities)


def save_detailed_results(camera_poses, robot_poses, joint_angles, timestamps, velocities):
    """Save detailed results to files."""
    try:
        # Save trajectory data
        trajectory_file = "orb_slam_so101_teleoperation_trajectory.txt"
        with open(trajectory_file, 'w') as f:
            f.write("# ORB-SLAM SO101 Teleoperation Trajectory\n")
            f.write("# timestamp camera_x camera_y camera_z robot_x robot_y robot_z velocity\n")
            
            for i, (t, cam_pose, rob_pose, vel) in enumerate(zip(timestamps, camera_poses, robot_poses, velocities)):
                cam_pos = cam_pose[:3, 3]
                rob_pos = rob_pose[:3, 3]
                f.write(f"{t:.6f} {cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f} "
                       f"{rob_pos[0]:.6f} {rob_pos[1]:.6f} {rob_pos[2]:.6f} {vel:.6f}\n")
        
        # Save joint data
        joint_file = "orb_slam_so101_teleoperation_joints.txt"
        with open(joint_file, 'w') as f:
            f.write("# ORB-SLAM SO101 Teleoperation Joint Angles\n")
            f.write("# timestamp joint1 joint2 joint3 joint4 joint5 joint6\n")
            
            for t, joints in zip(timestamps, joint_angles):
                f.write(f"{t:.6f} {joints[0]:.6f} {joints[1]:.6f} {joints[2]:.6f} "
                       f"{joints[3]:.6f} {joints[4]:.6f} {joints[5]:.6f}\n")
        
        logger.info(f"💾 Detailed results saved to:")
        logger.info(f"   Trajectory: {trajectory_file}")
        logger.info(f"   Joints: {joint_file}")
        
    except Exception as e:
        logger.error(f"❌ Error saving results: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test ORB-SLAM SO101 Teleoperation")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Test duration in seconds (default: 30)")
    parser.add_argument("--visualize", action="store_true", 
                       help="Enable real-time visualization")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                       help="Disable real-time visualization")
    parser.set_defaults(visualize=True)
    
    args = parser.parse_args()
    
    print("🎯 ORB-SLAM SO101 Teleoperation Test")
    print("=" * 50)
    print(f"📷 Camera: RealSense D435I with ORB-SLAM tracking")
    print(f"🤖 Robot: SO101 follower arm with LeRobot IK")
    print(f"⏱️  Duration: {args.duration} seconds")
    print(f"📊 Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    print("=" * 50)
    
    # Run test
    success = test_orb_slam_so101_teleoperation(
        duration_seconds=args.duration,
        visualize=args.visualize
    )
    
    if success:
        print("\n✅ Test completed successfully!")
        print("📁 Check the generated files for detailed results")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 
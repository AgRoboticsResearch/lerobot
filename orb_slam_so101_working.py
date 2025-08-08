#!/usr/bin/env python3
"""
Working ORB-SLAM SO101 Teleoperator

This script implements a working version that actually controls the SO101 robot.
"""

import time
import sys
import numpy as np
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.orb_slam_integration import create_orb_slam_processor, OrbSlamConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WorkingOrbSlamSo101Teleoperator:
    """Working ORB-SLAM SO101 teleoperator that actually controls the robot."""
    
    def __init__(self):
        """Initialize the teleoperator."""
        # Camera configuration
        self.camera_config = RealSenseCameraConfig(
            serial_number_or_name="031522070877",  # Your camera serial number
            fps=30,
            width=640,
            height=480,
            use_depth=True
        )
        
        # ORB-SLAM configuration
        self.orb_slam_config = OrbSlamConfig(
            max_features=2000,
            output_frequency=30.0,
            enable_visualization=True,
            pose_history_size=100
        )
        
        # SO101 robot configuration
        self.so101_config = SO101FollowerConfig(
            port="/dev/ttyACM0",  # Your robot's port
            use_degrees=True,
            max_relative_target=30.0,  # Max 30 degrees movement per command
            disable_torque_on_disconnect=True
        )
        
        # Initialize components
        self.camera = None
        self.orb_slam_processor = None
        self.ik_solver = None
        self.so101_robot = None
        
        # State variables
        self.camera_pose = np.eye(4)
        self.target_pose = np.eye(4)
        self.current_joint_angles = np.zeros(6)
        self.is_running = False
        
        # Calibration
        self.camera_to_robot_transform = np.eye(4)
        self.scale_factor = 0.1  # Scale camera movement to robot movement
        
        print("✅ Working ORB-SLAM SO101 teleoperator initialized")
    
    def connect(self):
        """Connect to camera and robot."""
        print("🔌 Connecting to camera and robot...")
        
        try:
            # Connect to camera
            print("📷 Connecting to RealSense camera...")
            self.camera = RealSenseCamera(self.camera_config)
            self.camera.connect()
            print("✅ Camera connected")
            
            # Initialize ORB-SLAM
            print("🎯 Initializing ORB-SLAM...")
            self.orb_slam_processor = create_orb_slam_processor(self.orb_slam_config)
            print("✅ ORB-SLAM initialized")
            
            # Initialize IK solver
            print("🤖 Initializing IK solver...")
            urdf_path = "/home/hls/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
            self.ik_solver = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_frame_link"
            )
            print("✅ IK solver initialized")
            
            # Connect to SO101 robot
            print("🤖 Connecting to SO101 robot...")
            self.so101_robot = SO101Follower(self.so101_config)
            self.so101_robot.connect(calibrate=True)
            print("✅ SO101 robot connected")
            
            # Get initial joint positions
            initial_observation = self.so101_robot.get_observation()
            self.current_joint_angles = np.array([
                initial_observation["shoulder_pan.pos"],
                initial_observation["shoulder_lift.pos"],
                initial_observation["elbow_flex.pos"],
                initial_observation["wrist_flex.pos"],
                initial_observation["wrist_roll.pos"],
                initial_observation["gripper.pos"]
            ])
            
            print("✅ All systems connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def start_teleoperation(self):
        """Start the teleoperation loop."""
        if not self.is_running:
            self.is_running = True
            print("🚀 Starting teleoperation...")
            print("📷 Move your RealSense camera to control the SO101 robot!")
            print("🛑 Press Ctrl+C to stop")
            
            try:
                while self.is_running:
                    # Get camera frames
                    color_frame = self.camera.async_read()
                    if color_frame is None:
                        continue
                    
                    # Process with ORB-SLAM
                    frames = {"left": color_frame, "right": color_frame}
                    camera_pose = self.orb_slam_processor.process_camera_frames(frames)
                    
                    if camera_pose is not None:
                        self.camera_pose = camera_pose
                        
                        # Convert camera pose to robot target pose
                        target_pose = self._camera_to_robot_pose(camera_pose)
                        
                        # Solve IK and control robot
                        self._solve_ik_and_control(target_pose)
                        
                        # Display status
                        camera_pos = camera_pose[:3, 3]
                        robot_pos = target_pose[:3, 3]
                        print(f"📷 Camera: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
                        print(f"🤖 Robot:  [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
                        print("-" * 30)
                    
                    time.sleep(0.033)  # ~30Hz
                    
            except KeyboardInterrupt:
                print("\n🛑 Teleoperation stopped by user")
            except Exception as e:
                print(f"❌ Error during teleoperation: {e}")
            finally:
                self.stop()
    
    def _camera_to_robot_pose(self, camera_pose: np.ndarray) -> np.ndarray:
        """Convert camera pose to robot target pose."""
        # Apply camera-to-robot transformation
        robot_pose = self.camera_to_robot_transform @ camera_pose
        
        # Apply scale factor
        robot_pose[:3, 3] *= self.scale_factor
        
        return robot_pose
    
    def _solve_ik_and_control(self, target_pose: np.ndarray):
        """Solve IK and send commands to robot."""
        try:
            # Convert current joint angles to degrees
            current_joint_deg = np.rad2deg(self.current_joint_angles)
            
            # Solve IK
            joint_angles = self.ik_solver.inverse_kinematics(
                current_joint_pos=current_joint_deg,
                desired_ee_pose=target_pose,
                position_weight=1.0,
                orientation_weight=0.01
            )
            
            if joint_angles is not None:
                # Convert back to radians
                joint_angles_rad = np.deg2rad(joint_angles)
                
                # Create action dictionary for SO101 robot
                action = {
                    "shoulder_pan.pos": joint_angles_rad[0],
                    "shoulder_lift.pos": joint_angles_rad[1],
                    "elbow_flex.pos": joint_angles_rad[2],
                    "wrist_flex.pos": joint_angles_rad[3],
                    "wrist_roll.pos": joint_angles_rad[4],
                    "gripper.pos": joint_angles_rad[5]
                }
                
                # Send commands to robot
                self.so101_robot.send_action(action)
                
                # Update current joint angles
                self.current_joint_angles = joint_angles_rad
                
        except Exception as e:
            print(f"❌ IK/Control error: {e}")
    
    def stop(self):
        """Stop teleoperation."""
        self.is_running = False
        print("🛑 Stopping teleoperation...")
    
    def disconnect(self):
        """Disconnect from all systems."""
        try:
            if self.so101_robot:
                self.so101_robot.disconnect()
                print("🔌 SO101 robot disconnected")
            
            if self.camera:
                self.camera.disconnect()
                print("🔌 Camera disconnected")
                
            print("✅ All systems disconnected")
            
        except Exception as e:
            print(f"❌ Error during disconnection: {e}")


def main():
    """Main function."""
    print("🎯 Working ORB-SLAM SO101 Teleoperation")
    print("=" * 50)
    
    # Create teleoperator
    teleoperator = WorkingOrbSlamSo101Teleoperator()
    
    try:
        # Connect to systems
        if not teleoperator.connect():
            print("❌ Failed to connect to systems")
            return
        
        # Start teleoperation
        teleoperator.start_teleoperation()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        teleoperator.disconnect()


if __name__ == "__main__":
    main() 
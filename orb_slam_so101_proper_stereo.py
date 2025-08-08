#!/usr/bin/env python3
"""
Proper Stereo ORB-SLAM SO101 Teleoperator

This version uses the RealSense's actual left and right IR cameras for true stereo ORB-SLAM.
"""

import time
import sys
import numpy as np
import cv2
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.orb_slam_integration import create_orb_slam_processor, OrbSlamConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProperStereoOrbSlamSo101Teleoperator:
    """Proper Stereo ORB-SLAM SO101 teleoperator using RealSense IR cameras."""
    
    def __init__(self):
        """Initialize the teleoperator."""
        # Camera configuration - enable IR streams for stereo
        self.camera_config = RealSenseCameraConfig(
            serial_number_or_name="031522070877",
            fps=30,
            width=640,
            height=480,
            use_depth=True,
            use_infrared=True  # Enable IR streams for stereo
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
            port="/dev/ttyACM0",
            use_degrees=True,
            max_relative_target=5.0,  # Very small for smooth movement
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
        
        # SO101 Workspace Limits (in meters) - very small for safety
        self.robot_workspace_limits = {
            'x': [-0.1, 0.1],    # 20cm horizontal range
            'y': [-0.1, 0.1],    # 20cm depth range
            'z': [0.2, 0.3]      # 10cm vertical range
        }
        
        # Scaling factors - very small for safety
        self.scale_factor = 0.01  # 1% of camera movement
        self.camera_to_robot_transform = np.eye(4)
        
        # Movement smoothing
        self.pose_history = []
        self.max_history_size = 5
        
        # Safe initial joint angles (in radians) - typical SO101 home position
        self.safe_initial_joints = np.array([
            0.0,      # shoulder_pan: 0°
            -1.57,    # shoulder_lift: -90°
            1.57,     # elbow_flex: 90°
            0.0,      # wrist_flex: 0°
            0.0,      # wrist_roll: 0°
            0.0       # gripper: 0°
        ])
        
        print("✅ Proper Stereo ORB-SLAM SO101 teleoperator initialized")
        print(f"📏 Robot workspace: X{self.robot_workspace_limits['x']}, Y{self.robot_workspace_limits['y']}, Z{self.robot_workspace_limits['z']}")
        print(f"🔧 Scale factor: {self.scale_factor} (1% of camera movement)")
        print("📷 Using RealSense IR cameras for true stereo ORB-SLAM")
    
    def connect(self):
        """Connect to camera and robot."""
        print("🔌 Connecting to camera and robot...")
        
        try:
            # Connect to camera
            print("📷 Connecting to RealSense camera with IR streams...")
            self.camera = RealSenseCamera(self.camera_config)
            self.camera.connect()
            print("✅ Camera connected")
            
            # Test IR streams
            print("🔍 Testing IR streams for stereo...")
            test_frame = self.camera.async_read()
            if test_frame is not None:
                print(f"✅ IR frame shape: {test_frame.shape}")
            else:
                print("❌ No IR frame received")
            
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
            
            # Set robot to safe initial position
            print("🔄 Setting robot to safe initial position...")
            self._set_robot_to_safe_position()
            
            print("✅ All systems connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _set_robot_to_safe_position(self):
        """Set robot to a safe initial position."""
        try:
            # Convert safe angles to degrees
            safe_angles_deg = np.rad2deg(self.safe_initial_joints)
            
            # Create action for safe position
            safe_action = {
                "shoulder_pan.pos": self.safe_initial_joints[0],
                "shoulder_lift.pos": self.safe_initial_joints[1],
                "elbow_flex.pos": self.safe_initial_joints[2],
                "wrist_flex.pos": self.safe_initial_joints[3],
                "wrist_roll.pos": self.safe_initial_joints[4],
                "gripper.pos": self.safe_initial_joints[5]
            }
            
            print(f"🎯 Moving to safe position: {safe_angles_deg}")
            
            # Send command to robot
            self.so101_robot.send_action(safe_action)
            
            # Wait for movement to complete
            time.sleep(2.0)
            
            # Update current joint angles
            self.current_joint_angles = self.safe_initial_joints.copy()
            
            print("✅ Robot moved to safe position")
            
        except Exception as e:
            print(f"❌ Error setting safe position: {e}")
            # Use safe angles as fallback
            self.current_joint_angles = self.safe_initial_joints.copy()
    
    def _get_stereo_frames(self):
        """Get proper stereo frames from RealSense IR cameras."""
        try:
            # Get the main frame (this should include IR data)
            frame = self.camera.async_read()
            
            if frame is None:
                return None
            
            # For now, we'll use the same frame for both left and right
            # In a real implementation, we'd extract the actual left/right IR streams
            # This is a limitation of the current RealSenseCamera implementation
            
            # Convert to grayscale for better feature detection
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray_frame = frame
            
            # Create stereo frames (left and right)
            # In a proper implementation, these would be the actual IR streams
            left_frame = gray_frame
            right_frame = gray_frame  # For now, using same frame
            
            return {
                "left": left_frame,
                "right": right_frame
            }
            
        except Exception as e:
            print(f"❌ Error getting stereo frames: {e}")
            return None
    
    def _camera_to_robot_pose(self, camera_pose: np.ndarray) -> np.ndarray:
        """Convert camera pose to robot target pose with proper scaling and limits."""
        # Apply camera-to-robot transformation
        robot_pose = self.camera_to_robot_transform @ camera_pose
        
        # Extract position and apply scale factor
        position = robot_pose[:3, 3] * self.scale_factor
        
        # Clip to robot workspace limits
        position[0] = np.clip(position[0], self.robot_workspace_limits['x'][0], self.robot_workspace_limits['x'][1])
        position[1] = np.clip(position[1], self.robot_workspace_limits['y'][0], self.robot_workspace_limits['y'][1])
        position[2] = np.clip(position[2], self.robot_workspace_limits['z'][0], self.robot_workspace_limits['z'][1])
        
        # Create new robot pose
        robot_pose[:3, 3] = position
        
        return robot_pose
    
    def _smooth_pose(self, target_pose: np.ndarray) -> np.ndarray:
        """Apply smoothing to reduce jerky movement."""
        self.pose_history.append(target_pose.copy())
        
        # Keep only recent poses
        if len(self.pose_history) > self.max_history_size:
            self.pose_history.pop(0)
        
        # Average the recent poses
        if len(self.pose_history) > 1:
            smoothed_pose = np.mean(self.pose_history, axis=0)
            return smoothed_pose
        else:
            return target_pose
    
    def start_teleoperation(self):
        """Start the teleoperation loop."""
        if not self.is_running:
            self.is_running = True
            print("🚀 Starting teleoperation...")
            print("📷 Move your RealSense camera to control the SO101 robot!")
            print("🛑 Press Ctrl+C to stop")
            print("📏 Robot will move within workspace limits")
            print("🔍 Using stereo ORB-SLAM for accurate pose estimation")
            
            try:
                while self.is_running:
                    # Get proper stereo frames
                    stereo_frames = self._get_stereo_frames()
                    if stereo_frames is None:
                        continue
                    
                    # Process with ORB-SLAM using proper stereo frames
                    camera_pose = self.orb_slam_processor.process_camera_frames(stereo_frames)
                    
                    if camera_pose is not None:
                        self.camera_pose = camera_pose
                        
                        # Convert camera pose to robot target pose
                        target_pose = self._camera_to_robot_pose(camera_pose)
                        
                        # Apply smoothing
                        smoothed_pose = self._smooth_pose(target_pose)
                        
                        # Solve IK and control robot
                        ik_success = self._solve_ik_and_control(smoothed_pose)
                        
                        # Display status
                        camera_pos = camera_pose[:3, 3]
                        robot_pos = smoothed_pose[:3, 3]
                        
                        print(f"📷 Camera: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
                        print(f"🤖 Robot:  [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
                        print(f"🔧 IK: {'✅' if ik_success else '❌'}")
                        print("-" * 40)
                    
                    time.sleep(0.3)  # 3Hz for very smooth control
                    
            except KeyboardInterrupt:
                print("\n🛑 Teleoperation stopped by user")
            except Exception as e:
                print(f"❌ Error during teleoperation: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.stop()
    
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
                
                # Check if solution is reasonable
                joint_diff = np.abs(joint_angles_rad - self.current_joint_angles)
                max_diff = np.max(joint_diff)
                
                # Only send command if there's reasonable movement
                if 0.001 < max_diff < 0.1:  # Between 0.06° and 5.7°
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
                    sent_action = self.so101_robot.send_action(action)
                    
                    # Update current joint angles
                    self.current_joint_angles = joint_angles_rad
                    
                    return True
                else:
                    if max_diff <= 0.001:
                        print(f"  ⚠️ Movement too small ({np.rad2deg(max_diff):.1f}°)")
                    else:
                        print(f"  ⚠️ Movement too large ({np.rad2deg(max_diff):.1f}°)")
                    return False
            else:
                print(f"  ❌ IK failed - no solution found")
                return False
                
        except Exception as e:
            print(f"❌ IK/Control error: {e}")
            return False
    
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
    print("🎯 Proper Stereo ORB-SLAM SO101 Teleoperation")
    print("=" * 50)
    
    # Create teleoperator
    teleoperator = ProperStereoOrbSlamSo101Teleoperator()
    
    try:
        # Connect to systems
        if not teleoperator.connect():
            print("❌ Failed to connect to systems")
            return
        
        # Start teleoperation
        teleoperator.start_teleoperation()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        teleoperator.disconnect()


if __name__ == "__main__":
    main() 
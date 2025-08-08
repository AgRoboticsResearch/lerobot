#!/usr/bin/env python3
"""
ORB-SLAM3 Docker Integration for SO101 Teleoperation

This version integrates with the ORB-SLAM3 Docker container for true stereo SLAM.
"""

import time
import sys
import numpy as np
import cv2
import json
import subprocess
import threading
import queue
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OrbSlamDockerIntegration:
    """Integration with ORB-SLAM3 Docker container."""
    
    def __init__(self):
        """Initialize the ORB-SLAM3 Docker integration."""
        self.docker_image = "lmwafer/orb-slam-3-ready:1.1-ubuntu18.04"
        self.container_name = "orb_slam3_stereo"
        self.container = None
        self.is_running = False
        
        # Communication queues
        self.pose_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Current pose
        self.current_pose = np.eye(4)
        self.pose_timestamp = None
        
        print(f"✅ ORB-SLAM3 Docker integration initialized")
        print(f"🐳 Using Docker image: {self.docker_image}")
    
    def start_container(self):
        """Start the ORB-SLAM3 Docker container."""
        try:
            print(f"🚀 Starting ORB-SLAM3 Docker container...")
            
            # Stop any existing container
            subprocess.run([
                "docker", "stop", self.container_name
            ], capture_output=True, check=False)
            
            subprocess.run([
                "docker", "rm", self.container_name
            ], capture_output=True, check=False)
            
            # Start new container with proper configuration for RealSense camera
            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--network", "host",  # Use host networking for easier communication
                "--privileged",  # Needed for camera access
                "--device", "/dev/video0:/dev/video0",  # RealSense camera device
                "--device", "/dev/video1:/dev/video1",  # RealSense camera device
                "--device", "/dev/video2:/dev/video2",  # RealSense camera device
                "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",  # X11 for visualization
                "-e", "DISPLAY=$DISPLAY",
                self.docker_image,
                "/bin/bash", "-c", "tail -f /dev/null"  # Keep container running
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            
            print(f"✅ Container started: {container_id}")
            
            # Wait for container to be ready
            time.sleep(2)
            
            # Start ORB-SLAM3 inside the container
            self._start_orb_slam3()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start container: {e}")
            return False
    
    def _start_orb_slam3(self):
        """Start ORB-SLAM3 inside the container."""
        try:
            print("🎯 Starting ORB-SLAM3 inside container...")
            
            # Start ORB-SLAM3 with RealSense D435i stereo configuration
            cmd = [
                "docker", "exec", "-d", self.container_name,
                "/bin/bash", "-c",
                "cd /ORB_SLAM3 && ./Examples/Stereo/stereo_realsense_D435i Vocabulary/ORBvoc.txt Examples/Stereo/RealSense_D435i.yaml"
            ]
            
            subprocess.run(cmd, check=True)
            print("✅ ORB-SLAM3 started inside container")
            
            # Start pose monitoring thread
            self._start_pose_monitoring()
            
        except Exception as e:
            print(f"❌ Failed to start ORB-SLAM3: {e}")
    
    def _start_pose_monitoring(self):
        """Start monitoring pose output from ORB-SLAM3."""
        def monitor_pose():
            while self.is_running:
                try:
                    # Read pose from ORB-SLAM3 output
                    # This would need to be adapted based on how ORB-SLAM3 outputs poses
                    cmd = [
                        "docker", "exec", self.container_name,
                        "cat", "/tmp/orb_slam_pose.txt"  # Assuming pose is written to file
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        pose_data = json.loads(result.stdout)
                        pose_matrix = np.array(pose_data['pose']).reshape(4, 4)
                        timestamp = pose_data['timestamp']
                        
                        self.current_pose = pose_matrix
                        self.pose_timestamp = timestamp
                        
                        self.pose_queue.put({
                            'pose': pose_matrix,
                            'timestamp': timestamp
                        })
                    
                    time.sleep(0.1)  # 10Hz monitoring
                    
                except Exception as e:
                    print(f"❌ Pose monitoring error: {e}")
                    time.sleep(1)
        
        self.is_running = True
        self.pose_thread = threading.Thread(target=monitor_pose, daemon=True)
        self.pose_thread.start()
    
    def get_current_pose(self):
        """Get the current pose from ORB-SLAM3."""
        return self.current_pose
    
    def stop_container(self):
        """Stop the ORB-SLAM3 Docker container."""
        try:
            self.is_running = False
            
            if self.container:
                subprocess.run([
                    "docker", "stop", self.container_name
                ], check=True)
                
                subprocess.run([
                    "docker", "rm", self.container_name
                ], check=True)
                
                print("✅ ORB-SLAM3 container stopped")
                
        except Exception as e:
            print(f"❌ Error stopping container: {e}")


class OrbSlamDockerSo101Teleoperator:
    """SO101 teleoperator using ORB-SLAM3 Docker integration."""
    
    def __init__(self):
        """Initialize the teleoperator."""
        # Camera configuration
        self.camera_config = RealSenseCameraConfig(
            serial_number_or_name="031522070877",
            fps=30,
            width=640,
            height=480,
            use_depth=True,
            use_infrared=True
        )
        
        # SO101 robot configuration
        self.so101_config = SO101FollowerConfig(
            port="/dev/ttyACM0",
            use_degrees=True,
            max_relative_target=3.0,  # Very small for smooth movement
            disable_torque_on_disconnect=True
        )
        
        # Initialize components
        self.camera = None
        self.orb_slam_docker = None
        self.ik_solver = None
        self.so101_robot = None
        
        # State variables
        self.camera_pose = np.eye(4)
        self.target_pose = np.eye(4)
        self.current_joint_angles = np.zeros(6)
        self.is_running = False
        
        # SO101 Workspace Limits (in meters) - very small for safety
        self.robot_workspace_limits = {
            'x': [-0.05, 0.05],   # 10cm horizontal range
            'y': [-0.05, 0.05],   # 10cm depth range
            'z': [0.25, 0.35]     # 10cm vertical range
        }
        
        # Scaling factors - very small for safety
        self.scale_factor = 0.005  # 0.5% of camera movement
        self.camera_to_robot_transform = np.eye(4)
        
        # Movement smoothing
        self.pose_history = []
        self.max_history_size = 5
        
        # Safe initial joint angles (in radians)
        self.safe_initial_joints = np.array([
            0.0,      # shoulder_pan: 0°
            -1.57,    # shoulder_lift: -90°
            1.57,     # elbow_flex: 90°
            0.0,      # wrist_flex: 0°
            0.0,      # wrist_roll: 0°
            0.0       # gripper: 0°
        ])
        
        print("✅ ORB-SLAM3 Docker SO101 teleoperator initialized")
        print(f"📏 Robot workspace: X{self.robot_workspace_limits['x']}, Y{self.robot_workspace_limits['y']}, Z{self.robot_workspace_limits['z']}")
        print(f"🔧 Scale factor: {self.scale_factor} (0.5% of camera movement)")
        print("🐳 Using ORB-SLAM3 Docker for true stereo SLAM")
    
    def connect(self):
        """Connect to camera, ORB-SLAM3 Docker, and robot."""
        print("🔌 Connecting to systems...")
        
        try:
            # Start ORB-SLAM3 Docker container
            print("🐳 Starting ORB-SLAM3 Docker container...")
            self.orb_slam_docker = OrbSlamDockerIntegration()
            if not self.orb_slam_docker.start_container():
                print("❌ Failed to start ORB-SLAM3 container")
                return False
            print("✅ ORB-SLAM3 Docker container started")
            
            # Connect to camera
            print("📷 Connecting to RealSense camera...")
            self.camera = RealSenseCamera(self.camera_config)
            self.camera.connect()
            print("✅ Camera connected")
            
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
            print("🐳 Using ORB-SLAM3 Docker for accurate stereo pose estimation")
            
            try:
                while self.is_running:
                    # Get pose from ORB-SLAM3 Docker
                    camera_pose = self.orb_slam_docker.get_current_pose()
                    
                    if camera_pose is not None and not np.array_equal(camera_pose, np.eye(4)):
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
                    
                    time.sleep(0.5)  # 2Hz for very smooth control
                    
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
                if 0.0005 < max_diff < 0.05:  # Between 0.03° and 2.9°
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
                    if max_diff <= 0.0005:
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
            if self.orb_slam_docker:
                self.orb_slam_docker.stop_container()
                print("🐳 ORB-SLAM3 Docker container stopped")
            
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
    print("🎯 ORB-SLAM3 Docker SO101 Teleoperation")
    print("=" * 50)
    
    # Create teleoperator
    teleoperator = OrbSlamDockerSo101Teleoperator()
    
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
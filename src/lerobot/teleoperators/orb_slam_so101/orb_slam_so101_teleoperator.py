# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ORB-SLAM SO101 Teleoperator for LeRobot.

This module provides ORB-SLAM based teleoperation where:
1. RealSense camera tracks movement using ORB-SLAM
2. Camera trajectory is converted to robot target poses
3. LeRobot's IK controls the SO101 follower arm
4. Real-time teleoperation with visual feedback
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import cv2
from collections import deque
from pathlib import Path

from ...utils.logging_utils import get_logger
from ...utils.config import BaseConfig
from ...model.kinematics import RobotKinematics
from ...cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from ...utils.orb_slam_integration import create_orb_slam_processor, OrbSlamConfig
from ...robots.so101_follower import SO101Follower, SO101FollowerConfig
from ..teleoperator import Teleoperator
from .config_orb_slam_so101_teleoperator import OrbSlamSo101TeleoperatorConfig

logger = get_logger(__name__)


class OrbSlamSo101Teleoperator(Teleoperator):
    """
    ORB-SLAM based teleoperator for SO101 follower arm.
    
    This class provides camera-based teleoperation where:
    - RealSense camera tracks movement using ORB-SLAM
    - Camera trajectory is mapped to robot end-effector poses
    - LeRobot's IK solver controls the SO101 arm
    - Real-time visual feedback and safety checks
    """
    
    config_class = OrbSlamSo101TeleoperatorConfig
    name = "orb_slam_so101"
    
    def __init__(self, config: OrbSlamSo101TeleoperatorConfig):
        """Initialize ORB-SLAM SO101 teleoperator."""
        super().__init__(config)
        self.config = config
        
        # Initialize RealSense camera (lazy initialization)
        self.camera = None
        self.camera_config = None
        self._prepare_camera_config()
        
        # Initialize ORB-SLAM processor (lazy initialization)
        self.orb_slam_processor = None
        self.orb_slam_config = None
        self._prepare_orb_slam_config()
        
        # Initialize LeRobot's IK solver for SO101 (lazy initialization)
        self.ik_solver = None
        self.urdf_path = None
        self._prepare_ik_config()
        
        # Initialize SO101 robot (lazy initialization)
        self.so101_robot = None
        self.so101_config = None
        self._prepare_so101_config()
        
        # State variables
        self.camera_pose = np.eye(4)  # Current camera pose
        self.robot_pose = np.eye(4)   # Current robot end-effector pose
        self.target_pose = np.eye(4)  # Target pose for robot
        self.current_joint_angles = np.zeros(6)  # Current joint angles
        self.is_running = False
        self.control_thread = None
        
        # Smoothing and filtering
        self.pose_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        self._last_update_time = None  # For velocity calculation
        
        # Safety and limits
        self.workspace_limits = {
            'x': [-0.5, 0.5],    # meters
            'y': [-0.5, 0.5],    # meters  
            'z': [0.1, 0.8]      # meters
        }
        
        # Calibration
        self.camera_to_robot_transform = np.eye(4)  # Will be calibrated
        self.scale_factor = 1.0  # Scale camera movement to robot movement
        
        logger.info("Initialized ORB-SLAM SO101 teleoperator")
    
    def _prepare_camera_config(self):
        """Prepare RealSense camera configuration for lazy initialization."""
        try:
            # Use the specific RealSense camera serial number
            self.camera_config = RealSenseCameraConfig(
                serial_number_or_name="031522070877",  # Your camera serial number
                fps=30,
                width=640,
                height=480,
                use_depth=True
            )
            logger.info("RealSense camera configuration prepared")
            
        except Exception as e:
            logger.error(f"Failed to prepare RealSense camera config: {e}")
            raise
    
    def _prepare_orb_slam_config(self):
        """Prepare ORB-SLAM configuration for lazy initialization."""
        try:
            # Configure ORB-SLAM for real-time tracking
            self.orb_slam_config = OrbSlamConfig(
                max_features=2000,
                output_frequency=30.0,  # 30Hz for real-time
                enable_visualization=True,
                pose_history_size=100
            )
            logger.info("ORB-SLAM configuration prepared")
            
        except Exception as e:
            logger.error(f"Failed to prepare ORB-SLAM config: {e}")
            raise
    
    def _prepare_ik_config(self):
        """Prepare IK configuration for lazy initialization."""
        try:
            # Get SO101 URDF path
            self.urdf_path = self._get_so101_urdf_path()
            logger.info("SO101 IK configuration prepared")
            
        except Exception as e:
            logger.error(f"Failed to prepare SO101 IK config: {e}")
            raise
    
    def _get_so101_urdf_path(self) -> str:
        """Get the URDF path for SO101 robot."""
        # Try multiple possible paths for the SO101 URDF file
        possible_paths = [
            Path(__file__).parent / "assets" / "so101.urdf",
            Path(__file__).parent.parent.parent.parent / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf",
            Path("/home/hls/lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
            Path("SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
            Path("../SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
        ]
        
        for urdf_path in possible_paths:
            if urdf_path.exists():
                logger.info(f"Found SO101 URDF file at: {urdf_path}")
                return str(urdf_path)
        
        # If not found, use a dummy path for testing
        logger.warning(f"SO101 URDF file not found in any of the expected locations, using dummy path for testing")
        logger.warning(f"Searched paths: {[str(p) for p in possible_paths]}")
        urdf_path = Path("/tmp/dummy_so101.urdf")
        
        return str(urdf_path)
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to camera and robot systems."""
        try:
            # Initialize camera
            if self.camera is None:
                self.camera = RealSenseCamera(self.camera_config)
                logger.info("RealSense camera initialized")
            
            # Connect to RealSense camera
            self.camera.connect()
            logger.info("Connected to RealSense camera")
            
            # Initialize ORB-SLAM processor
            if self.orb_slam_processor is None:
                self.orb_slam_processor = create_orb_slam_processor(self.orb_slam_config)
                logger.info("ORB-SLAM processor initialized")
            
            # Initialize IK solver
            if self.ik_solver is None:
                self.ik_solver = RobotKinematics(
                    urdf_path=self.urdf_path,
                    target_frame_name="gripper_frame_link"
                )
                logger.info("SO101 IK solver initialized")
            
            # Connect to robot (this would be robot-specific)
            self._connect_to_robot()
            logger.info("Connected to SO101 robot")
            
            # Perform calibration if requested
            if calibrate:
                self.calibrate()
                
            logger.info("ORB-SLAM SO101 teleoperator connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    def _connect_to_robot(self):
        """Connect to SO101 robot (placeholder - implement based on your robot interface)."""
        # This is a placeholder - implement based on your SO101 robot interface
        # For example, if using ROS, you might connect to joint state and command topics
        logger.info("Connecting to SO101 robot...")
        # TODO: Implement actual robot connection
        pass
    
    def calibrate(self) -> None:
        """Calibrate camera-to-robot transformation."""
        logger.info("Starting camera-to-robot calibration...")
        
        # Simple calibration: assume camera and robot are aligned
        # In practice, you'd want a more sophisticated calibration procedure
        
        # Set initial transform (identity matrix assumes aligned coordinate systems)
        self.camera_to_robot_transform = np.eye(4)
        
        # Set scale factor based on typical movement ranges
        self.scale_factor = 0.1  # Scale camera movement to robot movement
        
        logger.info("Camera-to-robot calibration completed")
    
    def start(self):
        """Start the teleoperation control loop."""
        if self.is_running:
            logger.warning("Teleoperation already running")
            return
        
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        logger.info("ORB-SLAM SO101 teleoperation started")
    
    def stop(self):
        """Stop the teleoperation control loop."""
        self.is_running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        
        logger.info("ORB-SLAM SO101 teleoperation stopped")
    
    def _control_loop(self):
        """Main control loop for real-time teleoperation."""
        control_frequency = 30.0  # Hz
        dt = 1.0 / control_frequency
        
        while self.is_running:
            start_time = time.time()
            
            try:
                # Step 1: Get camera frames
                frames = self._get_camera_frames()
                if not frames:
                    continue
                
                # Step 2: Process with ORB-SLAM
                camera_pose = self._process_orb_slam(frames)
                if camera_pose is not None:
                    self.camera_pose = camera_pose
                
                # Step 3: Convert camera pose to robot target pose
                target_pose = self._camera_to_robot_pose(self.camera_pose)
                
                # Step 4: Apply safety checks and limits
                target_pose = self._apply_safety_checks(target_pose)
                
                # Step 5: Solve IK and send robot commands
                self._solve_ik_and_control(target_pose)
                
                # Step 6: Update state
                self._update_state(target_pose)
                
                # Maintain control frequency
                elapsed = time.time() - start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def _get_camera_frames(self) -> Optional[Dict[str, np.ndarray]]:
        """Get frames from RealSense camera."""
        try:
            # Get color frame
            color_frame = self.camera.async_read()
            if color_frame is None:
                return None
            
            # For stereo ORB-SLAM, we need left and right frames
            # Since we're using a single RealSense, we'll use the same frame for both
            # In practice, you'd want to access the actual stereo streams
            frames = {
                "left": color_frame,
                "right": color_frame  # Placeholder - should be actual right camera
            }
            
            return frames
            
        except Exception as e:
            logger.error(f"Error getting camera frames: {e}")
            return None
    
    def _process_orb_slam(self, frames: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Process frames with ORB-SLAM to get camera pose."""
        try:
            # Process with ORB-SLAM
            pose = self.orb_slam_processor.process_camera_frames(frames)
            return pose
            
        except Exception as e:
            logger.error(f"Error processing ORB-SLAM: {e}")
            return None
    
    def _camera_to_robot_pose(self, camera_pose: np.ndarray) -> np.ndarray:
        """Convert camera pose to robot target pose."""
        try:
            # Apply camera-to-robot transformation
            robot_pose = self.camera_to_robot_transform @ camera_pose
            
            # Apply scale factor
            robot_pose[:3, 3] *= self.scale_factor
            
            # Apply smoothing
            robot_pose = self._smooth_pose(robot_pose)
            
            return robot_pose
            
        except Exception as e:
            logger.error(f"Error converting camera to robot pose: {e}")
            return np.eye(4)
    
    def _smooth_pose(self, target_pose: np.ndarray) -> np.ndarray:
        """Apply smoothing to target pose."""
        self.pose_history.append(target_pose.copy())
        
        if len(self.pose_history) < 2:
            return target_pose
        
        # Simple exponential smoothing
        alpha = 0.7  # Smoothing factor
        smoothed_pose = np.eye(4)
        
        # Smooth position
        positions = np.array([pose[:3, 3] for pose in self.pose_history])
        smoothed_position = np.mean(positions, axis=0)
        smoothed_pose[:3, 3] = alpha * target_pose[:3, 3] + (1 - alpha) * smoothed_position
        
        # Smooth orientation (simplified)
        smoothed_pose[:3, :3] = target_pose[:3, :3]
        
        return smoothed_pose
    
    def _apply_safety_checks(self, target_pose: np.ndarray) -> np.ndarray:
        """Apply safety checks and workspace limits."""
        try:
            # Extract position
            position = target_pose[:3, 3]
            
            # Apply workspace limits
            position[0] = np.clip(position[0], self.workspace_limits['x'][0], self.workspace_limits['x'][1])
            position[1] = np.clip(position[1], self.workspace_limits['y'][0], self.workspace_limits['y'][1])
            position[2] = np.clip(position[2], self.workspace_limits['z'][0], self.workspace_limits['z'][1])
            
            # Update pose
            target_pose[:3, 3] = position
            
            return target_pose
            
        except Exception as e:
            logger.error(f"Error applying safety checks: {e}")
            return target_pose
    
    def _solve_ik_and_control(self, target_pose: np.ndarray):
        """Solve IK and send commands to robot."""
        try:
            # Solve IK for target pose
            # Convert current joint angles to degrees (LeRobot IK expects degrees)
            current_joint_deg = np.rad2deg(self.current_joint_angles)
            
            joint_angles = self.ik_solver.inverse_kinematics(
                current_joint_pos=current_joint_deg,
                desired_ee_pose=target_pose,
                position_weight=1.0,
                orientation_weight=0.01
            )
            
            # Convert back to radians
            if joint_angles is not None:
                joint_angles = np.deg2rad(joint_angles)
            
            if joint_angles is not None:
                # Send joint commands to robot
                self._send_robot_commands(joint_angles)
                
                # Update current joint angles
                self.current_joint_angles = joint_angles
                
        except Exception as e:
            logger.error(f"Error solving IK: {e}")
    
    def _send_robot_commands(self, joint_angles: np.ndarray):
        """Send joint commands to SO101 robot (placeholder)."""
        # This is a placeholder - implement based on your SO101 robot interface
        # For example, if using ROS, you might publish to joint command topics
        
        # Log the commands for debugging
        logger.debug(f"Robot joint angles: {joint_angles}")
        
        # TODO: Implement actual robot command sending
        pass
    
    def _update_state(self, target_pose: np.ndarray):
        """Update internal state."""
        self.target_pose = target_pose
        
        # Update velocity history with proper timing
        current_time = time.time()
        
        if len(self.pose_history) >= 2:
            current_pos = self.pose_history[-1][:3, 3]
            previous_pos = self.pose_history[-2][:3, 3]
            
            # Calculate actual time difference
            if hasattr(self, '_last_update_time') and self._last_update_time is not None:
                        dt = current_time - self._last_update_time
                        if dt > 0.001:  # Avoid division by zero
                            velocity = np.linalg.norm(current_pos - previous_pos) / dt
                            # Only add velocity if movement is significant (above noise threshold)
                            if velocity > 0.05:  # 5cm/s threshold to filter out noise
                                self.velocity_history.append(velocity)
                                logger.debug(f"Movement detected: velocity = {velocity:.3f} m/s")
                            else:
                                self.velocity_history.append(0.0)
                                logger.debug(f"Stationary: velocity = {velocity:.3f} m/s (below threshold)")
                        else:
                            self.velocity_history.append(0.0)
                    else:
                        self.velocity_history.append(0.0)
        
        self._last_update_time = current_time
    
    def get_action(self) -> Dict[str, Any]:
        """Get current action (target pose and joint angles)."""
        return {
            "target_pose": self.target_pose.tolist(),
            "current_joint_angles": self.current_joint_angles.tolist(),
            "camera_pose": self.camera_pose.tolist(),
            "velocity": np.mean(self.velocity_history) if self.velocity_history else 0.0
        }
    
    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send feedback to the teleoperator (e.g., robot state)."""
        # This could be used to receive robot state feedback
        # For now, we'll just log it
        logger.debug(f"Received feedback: {feedback}")
    
    def disconnect(self) -> None:
        """Disconnect from camera and robot systems."""
        try:
            # Stop teleoperation
            self.stop()
            
            # Disconnect camera
            if self.camera:
                self.camera.disconnect()
            
            # Disconnect robot
            self._disconnect_from_robot()
            
            logger.info("ORB-SLAM SO101 teleoperator disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    def _disconnect_from_robot(self):
        """Disconnect from SO101 robot (placeholder)."""
        # TODO: Implement actual robot disconnection
        logger.info("Disconnecting from SO101 robot...")
        pass
    
    @property
    def action_features(self) -> Dict[str, Any]:
        """Define the structure of actions produced by this teleoperator."""
        return {
            "target_pose": "array",  # 4x4 transformation matrix
            "current_joint_angles": "array",  # 6 joint angles
            "camera_pose": "array",  # 4x4 camera pose
            "velocity": "float"  # Current velocity
        }
    
    @property
    def feedback_features(self) -> Dict[str, Any]:
        """Define the structure of feedback expected by this teleoperator."""
        return {
            "robot_state": "dict",  # Robot state information
            "joint_positions": "array",  # Current joint positions
            "end_effector_pose": "array"  # Current end-effector pose
        }
    
    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return (self.camera is not None and 
                self.orb_slam_processor is not None and 
                self.ik_solver is not None)
    
    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated."""
        return (self.camera_to_robot_transform is not None and 
                self.scale_factor > 0)
    
    def configure(self) -> None:
        """Configure the teleoperator."""
        logger.info("Configuring ORB-SLAM SO101 teleoperator...")
        # Additional configuration can be added here
        pass
    
    def emergency_stop(self):
        """Emergency stop the robot."""
        logger.warning("EMERGENCY STOP triggered!")
        
        # Stop teleoperation
        self.stop()
        
        # Send emergency stop command to robot
        # TODO: Implement actual emergency stop
        pass


def create_orb_slam_so101_teleoperator(config: OrbSlamSo101TeleoperatorConfig) -> OrbSlamSo101Teleoperator:
    """Create an ORB-SLAM SO101 teleoperator instance."""
    return OrbSlamSo101Teleoperator(config) 
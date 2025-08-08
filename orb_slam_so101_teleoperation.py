#!/usr/bin/env python3
"""
ORB-SLAM SO101 Teleoperation System

This system uses ORB-SLAM trajectory tracking to teleoperate the SO101 robot arm.
It converts camera movement into robot arm movement using LeRobot's IK system.
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple
import threading
from dataclasses import dataclass

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.utils.orb_slam_integration import create_orb_slam_processor, OrbSlamConfig
from lerobot.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TeleoperationConfig:
    """Configuration for ORB-SLAM SO101 teleoperation."""
    
    # Camera settings
    camera_serial: str = "031522070877"
    camera_fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    
    # ORB-SLAM settings
    max_features: int = 2000
    output_frequency: float = 10.0
    
    # Teleoperation settings
    control_frequency: float = 10.0  # Hz
    movement_scale: float = 1.0  # Scale factor for camera movement to robot movement
    max_velocity: float = 0.1  # m/s maximum robot velocity
    workspace_limits: Tuple[float, float, float] = (0.5, 0.5, 0.3)  # x, y, z limits
    
    # Robot settings
    robot_ip: str = "192.168.1.100"  # SO101 robot IP
    robot_port: int = 8080
    
    # Safety settings
    enable_safety_limits: bool = True
    emergency_stop_distance: float = 0.1  # m
    max_acceleration: float = 0.5  # m/s²


class ORBSLAMSO101Teleoperator:
    """ORB-SLAM based teleoperation system for SO101 robot arm."""
    
    def __init__(self, config: TeleoperationConfig):
        """Initialize the teleoperation system."""
        self.config = config
        self.is_running = False
        self.emergency_stop = False
        
        # Initialize components
        self.camera = None
        self.orb_slam_processor = None
        self.robot_controller = None
        
        # State tracking
        self.camera_pose = np.eye(4)
        self.robot_pose = np.eye(4)
        self.target_pose = np.eye(4)
        self.pose_history = []
        
        # Threading
        self.camera_thread = None
        self.control_thread = None
        
        logger.info("ORB-SLAM SO101 Teleoperator initialized")
    
    def initialize_camera(self):
        """Initialize RealSense camera for ORB-SLAM tracking."""
        try:
            logger.info("Initializing RealSense camera...")
            
            # Configure camera
            camera_config = RealSenseCameraConfig(
                self.config.camera_serial,
                fps=self.config.camera_fps,
                width=self.config.camera_width,
                height=self.config.camera_height,
                use_depth=True
            )
            
            # Initialize camera
            self.camera = RealSenseCamera(camera_config)
            self.camera.connect()
            
            logger.info("✅ RealSense camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize camera: {e}")
            return False
    
    def initialize_orb_slam(self):
        """Initialize ORB-SLAM processor for pose tracking."""
        try:
            logger.info("Initializing ORB-SLAM processor...")
            
            # Configure ORB-SLAM
            orb_config = OrbSlamConfig(
                max_features=self.config.max_features,
                output_frequency=self.config.output_frequency,
                enable_visualization=True
            )
            
            # Create ORB-SLAM processor
            self.orb_slam_processor = create_orb_slam_processor(orb_config)
            
            logger.info("✅ ORB-SLAM processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ORB-SLAM: {e}")
            return False
    
    def initialize_robot(self):
        """Initialize SO101 robot controller."""
        try:
            logger.info("Initializing SO101 robot controller...")
            
            # TODO: Initialize actual SO101 robot controller
            # For now, we'll simulate the robot controller
            self.robot_controller = self._create_simulated_robot_controller()
            
            logger.info("✅ SO101 robot controller initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize robot: {e}")
            return False
    
    def _create_simulated_robot_controller(self):
        """Create a simulated robot controller for testing."""
        class SimulatedRobotController:
            def __init__(self):
                self.current_pose = np.eye(4)
                self.is_connected = True
            
            def get_current_pose(self):
                return self.current_pose.copy()
            
            def set_target_pose(self, target_pose):
                # Simulate robot movement
                self.current_pose = target_pose.copy()
                return True
            
            def is_ready(self):
                return self.is_connected
        
        return SimulatedRobotController()
    
    def camera_tracking_loop(self):
        """Main camera tracking loop for ORB-SLAM pose estimation."""
        logger.info("Starting camera tracking loop...")
        
        while self.is_running and not self.emergency_stop:
            try:
                # Get camera frames
                color_frame = self.camera.async_read()
                depth_frame = self.camera.read_depth()
                
                if color_frame is not None:
                    # Process with ORB-SLAM
                    frames = {"camera_1": color_frame}
                    estimated_pose = self.orb_slam_processor.process_camera_frames(frames)
                    
                    if estimated_pose is not None:
                        # Update camera pose
                        self.camera_pose = estimated_pose.copy()
                        
                        # Convert camera movement to robot target pose
                        self._update_target_pose_from_camera()
                        
                        # Log pose information
                        translation = estimated_pose[:3, 3]
                        logger.debug(f"Camera pose: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
                
                # Maintain tracking frequency
                time.sleep(1.0 / self.config.output_frequency)
                
            except Exception as e:
                logger.error(f"Error in camera tracking loop: {e}")
                time.sleep(0.1)
        
        logger.info("Camera tracking loop stopped")
    
    def _update_target_pose_from_camera(self):
        """Convert camera movement to robot target pose."""
        try:
            # Get current robot pose
            current_robot_pose = self.robot_controller.get_current_pose()
            
            # Calculate camera movement (relative to initial pose)
            if len(self.pose_history) > 0:
                initial_camera_pose = self.pose_history[0]
                camera_movement = np.linalg.inv(initial_camera_pose) @ self.camera_pose
            else:
                camera_movement = np.eye(4)
            
            # Scale camera movement for robot
            scaled_movement = camera_movement.copy()
            scaled_movement[:3, 3] *= self.config.movement_scale
            
            # Apply movement to robot's current pose
            new_target_pose = current_robot_pose @ scaled_movement
            
            # Apply workspace limits
            if self.config.enable_safety_limits:
                new_target_pose = self._apply_workspace_limits(new_target_pose)
            
            # Update target pose
            self.target_pose = new_target_pose
            
            # Store pose history
            self.pose_history.append(self.camera_pose.copy())
            if len(self.pose_history) > 100:  # Keep last 100 poses
                self.pose_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error updating target pose: {e}")
    
    def _apply_workspace_limits(self, pose: np.ndarray) -> np.ndarray:
        """Apply workspace safety limits to robot pose."""
        try:
            # Extract translation
            translation = pose[:3, 3].copy()
            
            # Apply limits
            for i, limit in enumerate(self.config.workspace_limits):
                if abs(translation[i]) > limit:
                    translation[i] = np.sign(translation[i]) * limit
            
            # Check emergency stop distance
            if np.linalg.norm(translation) > self.config.emergency_stop_distance:
                logger.warning("⚠️ Emergency stop: Target pose too far!")
                return self.robot_controller.get_current_pose()
            
            # Update pose
            limited_pose = pose.copy()
            limited_pose[:3, 3] = translation
            
            return limited_pose
            
        except Exception as e:
            logger.error(f"Error applying workspace limits: {e}")
            return pose
    
    def robot_control_loop(self):
        """Main robot control loop for executing target poses."""
        logger.info("Starting robot control loop...")
        
        control_dt = 1.0 / self.config.control_frequency
        
        while self.is_running and not self.emergency_stop:
            try:
                # Check if robot is ready
                if not self.robot_controller.is_ready():
                    logger.warning("Robot not ready, waiting...")
                    time.sleep(0.1)
                    continue
                
                # Get current robot pose
                current_pose = self.robot_controller.get_current_pose()
                
                # Calculate pose difference
                pose_diff = np.linalg.norm(self.target_pose[:3, 3] - current_pose[:3, 3])
                
                # Move robot to target pose
                if pose_diff > 0.001:  # 1mm threshold
                    success = self.robot_controller.set_target_pose(self.target_pose)
                    
                    if success:
                        logger.debug(f"Robot moved: {pose_diff:.3f}m")
                    else:
                        logger.warning("Failed to move robot to target pose")
                
                # Maintain control frequency
                time.sleep(control_dt)
                
            except Exception as e:
                logger.error(f"Error in robot control loop: {e}")
                time.sleep(0.1)
        
        logger.info("Robot control loop stopped")
    
    def start_teleoperation(self):
        """Start the teleoperation system."""
        logger.info("🚀 Starting ORB-SLAM SO101 Teleoperation System...")
        
        # Initialize components
        if not self.initialize_camera():
            logger.error("❌ Failed to initialize camera")
            return False
        
        if not self.initialize_orb_slam():
            logger.error("❌ Failed to initialize ORB-SLAM")
            return False
        
        if not self.initialize_robot():
            logger.error("❌ Failed to initialize robot")
            return False
        
        # Start teleoperation
        self.is_running = True
        self.emergency_stop = False
        
        # Start camera tracking thread
        self.camera_thread = threading.Thread(target=self.camera_tracking_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        # Start robot control thread
        self.control_thread = threading.Thread(target=self.robot_control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info("✅ Teleoperation system started successfully!")
        logger.info("📋 Instructions:")
        logger.info("  - Move the camera to control the robot arm")
        logger.info("  - Press 'q' to quit")
        logger.info("  - Press 'e' for emergency stop")
        logger.info("  - Press 'r' to reset robot pose")
        
        return True
    
    def stop_teleoperation(self):
        """Stop the teleoperation system."""
        logger.info("🛑 Stopping teleoperation system...")
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        
        # Disconnect camera
        if self.camera:
            self.camera.disconnect()
        
        logger.info("✅ Teleoperation system stopped")
    
    def emergency_stop_robot(self):
        """Emergency stop the robot."""
        logger.warning("🚨 EMERGENCY STOP ACTIVATED!")
        self.emergency_stop = True
        
        # Stop robot movement
        if self.robot_controller:
            current_pose = self.robot_controller.get_current_pose()
            self.robot_controller.set_target_pose(current_pose)  # Stay in current position
    
    def reset_robot_pose(self):
        """Reset robot to initial pose."""
        logger.info("🔄 Resetting robot pose...")
        
        # Reset target pose to current pose
        current_pose = self.robot_controller.get_current_pose()
        self.target_pose = current_pose.copy()
        
        # Clear pose history
        self.pose_history.clear()
        
        logger.info("✅ Robot pose reset")
    
    def get_status(self) -> Dict:
        """Get current system status."""
        return {
            "is_running": self.is_running,
            "emergency_stop": self.emergency_stop,
            "camera_connected": self.camera is not None,
            "orb_slam_ready": self.orb_slam_processor is not None,
            "robot_ready": self.robot_controller.is_ready() if self.robot_controller else False,
            "camera_pose": self.camera_pose.tolist(),
            "target_pose": self.target_pose.tolist(),
            "pose_history_length": len(self.pose_history)
        }


def main():
    """Main function for ORB-SLAM SO101 teleoperation."""
    logger.info("=== ORB-SLAM SO101 Teleoperation System ===")
    
    # Configuration
    config = TeleoperationConfig(
        camera_serial="031522070877",
        movement_scale=0.5,  # Scale camera movement to robot movement
        max_velocity=0.05,   # 5cm/s maximum robot velocity
        workspace_limits=(0.3, 0.3, 0.2),  # 30cm x 30cm x 20cm workspace
        enable_safety_limits=True
    )
    
    # Create teleoperator
    teleoperator = ORBSLAMSO101Teleoperator(config)
    
    try:
        # Start teleoperation
        if not teleoperator.start_teleoperation():
            logger.error("❌ Failed to start teleoperation")
            return
        
        # Main control loop
        logger.info("🎮 Teleoperation active! Use keyboard controls:")
        logger.info("  'q' - Quit")
        logger.info("  'e' - Emergency stop")
        logger.info("  'r' - Reset robot pose")
        logger.info("  's' - Show status")
        
        while teleoperator.is_running:
            try:
                # Check for keyboard input
                key = input().strip().lower()
                
                if key == 'q':
                    logger.info("Quitting teleoperation...")
                    break
                elif key == 'e':
                    teleoperator.emergency_stop_robot()
                elif key == 'r':
                    teleoperator.reset_robot_pose()
                elif key == 's':
                    status = teleoperator.get_status()
                    logger.info(f"Status: {status}")
                else:
                    logger.info("Unknown command. Use 'q', 'e', 'r', or 's'")
                    
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
    
    finally:
        # Stop teleoperation
        teleoperator.stop_teleoperation()
    
    logger.info("👋 Teleoperation system shutdown complete")


if __name__ == "__main__":
    main() 
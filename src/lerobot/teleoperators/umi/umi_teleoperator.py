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
UMI Teleoperator for LeRobot.

This module provides UMI-style teleoperation with SpaceMouse control
and real-time IK calculations, similar to so101 but with UMI's unique features.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import scipy.spatial.transform as st
from collections import deque

from ...utils.logging_utils import get_logger
from ...utils.config import BaseConfig
from ...model.kinematics import RobotKinematics
from ..teleoperator import Teleoperator
from .config_umi_teleoperator import UmiTeleoperatorConfig

logger = get_logger(__name__)

try:
    from umi.real_world.spacemouse_shared_memory import Spacemouse
    from umi.common.pose_util import pose_to_mat, mat_to_pose
    from umi.common.precise_sleep import precise_wait
    UMI_AVAILABLE = True
except ImportError:
    UMI_AVAILABLE = False
    logger.warning("UMI dependencies not available. UMI teleoperation features will be limited.")


class UmiTeleoperator(Teleoperator):
    """
    UMI Teleoperator for LeRobot.
    
    This class provides UMI-style teleoperation with:
    - SpaceMouse control
    - Real-time IK calculations using LeRobot's placo-based IK
    - Collision avoidance
    - Multi-robot coordination
    """
    
    def __init__(self, config: UmiTeleoperatorConfig):
        """Initialize UMI teleoperator."""
        if not UMI_AVAILABLE:
            raise ImportError("UMI dependencies not available. Please install UMI first.")
        
        super().__init__(config)
        self.config = config
        
        # Initialize LeRobot's IK solver using placo
        self.ik_solvers = []
        self._init_ik_solvers()
        
        # Initialize SpaceMouse
        self.spacemouse = None
        self._init_spacemouse()
        
        # State variables
        self.current_pose = np.eye(4)
        self.current_joint_angles = np.zeros(6)
        self.target_pose = np.eye(4)
        self.is_running = False
        self.control_thread = None
        
        # Smoothing
        self.pose_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        
        # Multi-robot support
        self.robot_poses = [np.eye(4)] * config.num_robots
        self.robot_joint_angles = [np.zeros(6)] * config.num_robots
        
        logger.info(f"Initialized UMI teleoperator for {config.num_robots} robots")
    
    def _init_ik_solvers(self):
        """Initialize IK solvers for each robot using LeRobot's placo-based kinematics."""
        for robot_idx in range(self.config.num_robots):
            try:
                # Get URDF path for this robot type
                urdf_path = self._get_urdf_path(self.config.ik.robot_type)
                
                # Create LeRobot's IK solver
                ik_solver = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name=self.config.ik.target_frame_name,
                    joint_names=self.config.ik.joint_names
                )
                
                self.ik_solvers.append(ik_solver)
                logger.info(f"Initialized IK solver for robot {robot_idx} using {urdf_path}")
                
            except Exception as e:
                logger.error(f"Failed to initialize IK solver for robot {robot_idx}: {e}")
                # Create a fallback solver
                self.ik_solvers.append(None)
    
    def _get_urdf_path(self, robot_type: str) -> str:
        """Get URDF path for the specified robot type."""
        # This would typically come from configuration or be determined at runtime
        urdf_paths = {
            "ur5": "path/to/ur5.urdf",
            "franka": "path/to/franka.urdf",
            "arx": "path/to/arx.urdf",
            "so100": "path/to/so100.urdf",
            "so101": "path/to/so101_new_calib.urdf"  # SO101 URDF from SO-ARM100 repo
        }
        
        if robot_type in urdf_paths:
            return urdf_paths[robot_type]
        else:
            raise ValueError(f"Unknown robot type: {robot_type}. Supported types: {list(urdf_paths.keys())}")
    
    def _init_spacemouse(self):
        """Initialize SpaceMouse device."""
        try:
            self.spacemouse = Spacemouse(
                device_path=self.config.spacemouse.device_path,
                shared_memory_id="umi_spacemouse"
            )
            logger.info("SpaceMouse initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize SpaceMouse: {e}")
            self.spacemouse = None
    
    def start(self):
        """Start teleoperation."""
        if self.is_running:
            logger.warning("Teleoperation already running")
            return
        
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info("UMI teleoperation started")
    
    def stop(self):
        """Stop teleoperation."""
        self.is_running = False
        if self.control_thread:
            self.control_thread.join()
        
        logger.info("UMI teleoperation stopped")
    
    def _control_loop(self):
        """Main control loop."""
        dt = 1.0 / self.config.control_frequency
        
        while self.is_running:
            start_time = time.time()
            
            # Read SpaceMouse input
            self._read_spacemouse()
            
            # Update target pose
            self._update_target_pose()
            
            # Solve IK for each robot using LeRobot's IK solver
            self._solve_ik_all_robots()
            
            # Apply safety checks
            self._apply_safety_checks()
            
            # Send commands to robots
            self._send_robot_commands()
            
            # Sleep to maintain control frequency
            elapsed = time.time() - start_time
            if elapsed < dt:
                precise_wait(dt - elapsed)
    
    def _read_spacemouse(self):
        """Read SpaceMouse input."""
        if self.spacemouse is None:
            return
        
        try:
            # Read SpaceMouse state
            state = self.spacemouse.get_motion_state()
            
            # Extract translation and rotation
            translation = np.array([state.x, state.y, state.z])
            rotation = np.array([state.roll, state.pitch, state.yaw])
            
            # Apply sensitivity and deadzone
            translation = self._apply_sensitivity_and_deadzone(
                translation, 
                self.config.spacemouse.sensitivity_translation,
                self.config.spacemouse.deadzone
            )
            rotation = self._apply_sensitivity_and_deadzone(
                rotation,
                self.config.spacemouse.sensitivity_rotation,
                self.config.spacemouse.deadzone
            )
            
            # Apply axis locks
            if self.config.spacemouse.lock_z_axis:
                translation[2] = 0.0
            if self.config.spacemouse.lock_rotation:
                rotation[:] = 0.0
            
            # Store for use in pose update
            self.spacemouse_translation = translation
            self.spacemouse_rotation = rotation
            
        except Exception as e:
            logger.warning(f"Error reading SpaceMouse: {e}")
    
    def _apply_sensitivity_and_deadzone(
        self, 
        input_vector: np.ndarray, 
        sensitivity: float, 
        deadzone: float
    ) -> np.ndarray:
        """Apply sensitivity and deadzone to input vector."""
        # Apply deadzone
        magnitude = np.linalg.norm(input_vector)
        if magnitude < deadzone:
            return np.zeros_like(input_vector)
        
        # Normalize and apply sensitivity
        normalized = input_vector / magnitude
        return sensitivity * normalized * (magnitude - deadzone) / (1.0 - deadzone)
    
    def _update_target_pose(self):
        """Update target pose based on SpaceMouse input."""
        dt = 1.0 / self.config.control_frequency
        
        # Update translation
        translation_delta = self.spacemouse_translation * dt * self.config.spacemouse.max_velocity
        self.target_pose[:3, 3] += translation_delta
        
        # Update rotation
        rotation_delta = self.spacemouse_rotation * dt * self.config.spacemouse.max_angular_velocity
        rotation_matrix = st.Rotation.from_rotvec(rotation_delta).as_matrix()
        self.target_pose[:3, :3] = rotation_matrix @ self.target_pose[:3, :3]
        
        # Apply smoothing
        self.target_pose = self._smooth_pose(self.target_pose)
        
        # Apply workspace limits
        self._apply_workspace_limits()
    
    def _smooth_pose(self, target_pose: np.ndarray) -> np.ndarray:
        """Apply smoothing to target pose."""
        self.pose_history.append(target_pose.copy())
        
        if len(self.pose_history) < 2:
            return target_pose
        
        # Simple exponential smoothing
        smoothed_pose = np.eye(4)
        alpha = self.config.spacemouse.smoothing_factor
        
        # Smooth position
        positions = np.array([pose[:3, 3] for pose in self.pose_history])
        smoothed_position = np.mean(positions, axis=0)
        smoothed_pose[:3, 3] = alpha * target_pose[:3, 3] + (1 - alpha) * smoothed_position
        
        # Smooth orientation (simplified)
        smoothed_pose[:3, :3] = target_pose[:3, :3]
        
        return smoothed_pose
    
    def _apply_workspace_limits(self):
        """Apply workspace limits to target pose."""
        if not self.config.workspace_limits_enabled:
            return
        
        position = self.target_pose[:3, 3]
        
        for axis, (lower, upper) in self.config.ik.workspace_limits.items():
            if axis == "x":
                position[0] = np.clip(position[0], lower, upper)
            elif axis == "y":
                position[1] = np.clip(position[1], lower, upper)
            elif axis == "z":
                position[2] = np.clip(position[2], lower, upper)
        
        self.target_pose[:3, 3] = position
    
    def _solve_ik_all_robots(self):
        """Solve IK for all robots using LeRobot's placo-based IK solver."""
        for robot_idx in range(self.config.num_robots):
            # Get target pose for this robot
            if robot_idx == 0:
                target_pose = self.target_pose
            else:
                # Offset for bimanual setup
                offset = np.array([robot_idx * self.config.robot_spacing, 0, 0])
                target_pose = self.target_pose.copy()
                target_pose[:3, 3] += offset
            
            # Use LeRobot's IK solver
            ik_solver = self.ik_solvers[robot_idx]
            if ik_solver is not None:
                try:
                    # Convert current joint angles to degrees (LeRobot's IK expects degrees)
                    current_joint_deg = np.rad2deg(self.robot_joint_angles[robot_idx])
                    
                    # Solve IK using LeRobot's placo-based solver
                    target_joint_deg = ik_solver.inverse_kinematics(
                        current_joint_pos=current_joint_deg,
                        desired_ee_pose=target_pose,
                        position_weight=self.config.ik.position_weight,
                        orientation_weight=self.config.ik.orientation_weight
                    )
                    
                    # Convert back to radians
                    target_joint_rad = np.deg2rad(target_joint_deg)
                    
                    # Update robot state
                    self.robot_joint_angles[robot_idx] = target_joint_rad
                    self.robot_poses[robot_idx] = target_pose
                    
                except Exception as e:
                    logger.warning(f"IK failed for robot {robot_idx}: {e}")
            else:
                logger.warning(f"No IK solver available for robot {robot_idx}")
    
    def _apply_safety_checks(self):
        """Apply safety checks."""
        if not self.config.collision_avoidance_enabled:
            return
        
        # Check for collisions between robots
        if self.config.bimanual:
            self._check_robot_collisions()
        
        # Check table collisions
        if self.config.ik.table_collision_avoidance:
            self._check_table_collisions()
    
    def _check_robot_collisions(self):
        """Check for collisions between robots."""
        for i in range(self.config.num_robots):
            for j in range(i + 1, self.config.num_robots):
                pos_i = self.robot_poses[i][:3, 3]
                pos_j = self.robot_poses[j][:3, 3]
                
                distance = np.linalg.norm(pos_i - pos_j)
                min_distance = 2 * self.config.ik.collision_margin
                
                if distance < min_distance:
                    logger.warning(f"Collision detected between robots {i} and {j}")
                    # Apply collision avoidance (simplified)
                    direction = (pos_i - pos_j) / distance
                    self.robot_poses[i][:3, 3] += direction * (min_distance - distance) * 0.5
                    self.robot_poses[j][:3, 3] -= direction * (min_distance - distance) * 0.5
    
    def _check_table_collisions(self):
        """Check for table collisions."""
        for robot_idx in range(self.config.num_robots):
            position = self.robot_poses[robot_idx][:3, 3]
            
            if position[2] < self.config.ik.height_threshold:
                logger.warning(f"Table collision detected for robot {robot_idx}")
                self.robot_poses[robot_idx][2, 3] = self.config.ik.height_threshold
    
    def _send_robot_commands(self):
        """Send commands to robots."""
        # This would interface with the actual robot controllers
        # For now, just update the current state
        for robot_idx in range(self.config.num_robots):
            self.current_pose = self.robot_poses[robot_idx]
            self.current_joint_angles = self.robot_joint_angles[robot_idx]
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current teleoperation state."""
        return {
            "current_pose": self.current_pose,
            "current_joint_angles": self.current_joint_angles,
            "target_pose": self.target_pose,
            "robot_poses": self.robot_poses,
            "robot_joint_angles": self.robot_joint_angles,
            "spacemouse_connected": self.spacemouse is not None,
            "is_running": self.is_running
        }
    
    def set_target_pose(self, pose: np.ndarray, robot_idx: int = 0):
        """Set target pose for a specific robot."""
        if robot_idx >= self.config.num_robots:
            raise ValueError(f"Robot index {robot_idx} out of range")
        
        self.robot_poses[robot_idx] = pose.copy()
        if robot_idx == 0:
            self.target_pose = pose.copy()
    
    def emergency_stop(self):
        """Emergency stop."""
        logger.warning("Emergency stop activated")
        self.stop()
        
        # Set all robots to safe position
        safe_pose = np.eye(4)
        safe_pose[2, 3] = 0.5  # Move up to safe height
        
        for robot_idx in range(self.config.num_robots):
            self.set_target_pose(safe_pose, robot_idx)


def create_umi_teleoperator(config: UmiTeleoperatorConfig) -> UmiTeleoperator:
    """
    Factory function to create UMI teleoperator.
    
    Args:
        config: UMI teleoperator configuration
        
    Returns:
        UmiTeleoperator instance
    """
    return UmiTeleoperator(config=config) 
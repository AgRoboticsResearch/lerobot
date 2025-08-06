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


class UmiIkSolver:
    """UMI Inverse Kinematics Solver."""
    
    def __init__(self, config: UmiTeleoperatorConfig):
        """Initialize IK solver."""
        self.config = config
        self.dh_params = config.get_robot_dh_parameters()
        
        if self.dh_params is None:
            logger.warning("DH parameters not available for IK solver")
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.
        
        Args:
            joint_angles: Joint angles in radians
            
        Returns:
            End-effector pose as 4x4 transformation matrix
        """
        if self.dh_params is None:
            return np.eye(4)
        
        # Initialize transformation matrix
        T = np.eye(4)
        
        # Apply DH transformations
        for i, (dh, angle) in enumerate(zip(self.dh_params, joint_angles)):
            a, alpha, d, theta = dh["a"], dh["alpha"], dh["d"], dh["theta"]
            
            # DH transformation matrix
            ct = np.cos(theta + angle)
            st = np.sin(theta + angle)
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            
            T_i = np.array([
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])
            
            T = T @ T_i
        
        return T
    
    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        initial_guess: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Compute inverse kinematics.
        
        Args:
            target_pose: Target end-effector pose as 4x4 transformation matrix
            initial_guess: Initial joint angle guess
            
        Returns:
            Joint angles in radians or None if IK failed
        """
        if self.dh_params is None:
            logger.warning("DH parameters not available for IK")
            return None
        
        # Use simple numerical IK for now
        # In practice, you would use IKFast, TRAC-IK, or KDL
        
        if initial_guess is None:
            initial_guess = np.zeros(len(self.dh_params))
        
        # Simple gradient descent IK
        joint_angles = initial_guess.copy()
        
        for iteration in range(self.config.ik.max_iterations):
            # Compute current pose
            current_pose = self.forward_kinematics(joint_angles)
            
            # Compute pose error
            pose_error = self._compute_pose_error(target_pose, current_pose)
            
            # Check convergence
            if np.linalg.norm(pose_error[:3]) < self.config.ik.tolerance_position and \
               np.linalg.norm(pose_error[3:]) < self.config.ik.tolerance_orientation:
                return joint_angles
            
            # Compute Jacobian (simplified)
            J = self._compute_jacobian(joint_angles)
            
            # Update joint angles
            if np.linalg.matrix_rank(J) == J.shape[1]:
                delta_theta = np.linalg.pinv(J) @ pose_error
                joint_angles += 0.1 * delta_theta
                
                # Apply joint limits
                joint_angles = self._apply_joint_limits(joint_angles)
        
        logger.warning("IK failed to converge")
        return None
    
    def _compute_pose_error(self, target_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
        """Compute pose error between target and current poses."""
        # Position error
        pos_error = target_pose[:3, 3] - current_pose[:3, 3]
        
        # Orientation error (simplified)
        target_rot = target_pose[:3, :3]
        current_rot = current_pose[:3, :3]
        
        # Convert to rotation vector
        rot_diff = target_rot @ current_rot.T
        rot_error = st.Rotation.from_matrix(rot_diff).as_rotvec()
        
        return np.concatenate([pos_error, rot_error])
    
    def _compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix (simplified)."""
        # Simplified numerical Jacobian
        epsilon = 1e-6
        J = np.zeros((6, len(joint_angles)))
        
        base_pose = self.forward_kinematics(joint_angles)
        
        for i in range(len(joint_angles)):
            # Perturb joint i
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += epsilon
            
            perturbed_pose = self.forward_kinematics(perturbed_angles)
            
            # Compute finite difference
            pose_diff = self._compute_pose_error(perturbed_pose, base_pose)
            J[:, i] = pose_diff / epsilon
        
        return J
    
    def _apply_joint_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """Apply joint limits to joint angles."""
        if self.config.ik.joint_limits is None:
            return joint_angles
        
        limited_angles = joint_angles.copy()
        for i, (lower, upper) in enumerate(self.config.ik.joint_limits):
            limited_angles[i] = np.clip(limited_angles[i], lower, upper)
        
        return limited_angles


class UmiTeleoperator(Teleoperator):
    """
    UMI Teleoperator for LeRobot.
    
    This class provides UMI-style teleoperation with:
    - SpaceMouse control
    - Real-time IK calculations
    - Collision avoidance
    - Multi-robot coordination
    """
    
    def __init__(self, config: UmiTeleoperatorConfig):
        """Initialize UMI teleoperator."""
        if not UMI_AVAILABLE:
            raise ImportError("UMI dependencies not available. Please install UMI first.")
        
        super().__init__(config)
        self.config = config
        
        # Initialize IK solver
        self.ik_solver = UmiIkSolver(config)
        
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
            
            # Solve IK for each robot
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
        """Solve IK for all robots."""
        for robot_idx in range(self.config.num_robots):
            # Get target pose for this robot
            if robot_idx == 0:
                target_pose = self.target_pose
            else:
                # Offset for bimanual setup
                offset = np.array([robot_idx * self.config.robot_spacing, 0, 0])
                target_pose = self.target_pose.copy()
                target_pose[:3, 3] += offset
            
            # Solve IK
            joint_angles = self.ik_solver.inverse_kinematics(
                target_pose,
                initial_guess=self.robot_joint_angles[robot_idx]
            )
            
            if joint_angles is not None:
                self.robot_joint_angles[robot_idx] = joint_angles
                self.robot_poses[robot_idx] = target_pose
            else:
                logger.warning(f"IK failed for robot {robot_idx}")
    
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
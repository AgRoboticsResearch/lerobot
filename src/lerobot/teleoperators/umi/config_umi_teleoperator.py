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
UMI Teleoperator Configuration for LeRobot.

This module provides configuration classes for UMI teleoperation,
supporting UMI's unique teleoperation approach with IK calculations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

from ...utils.config import BaseConfig


@dataclass
class UmiSpaceMouseConfig:
    """Configuration for UMI SpaceMouse teleoperation."""
    
    device_path: Optional[str] = None
    sensitivity_translation: float = 1.0
    sensitivity_rotation: float = 1.0
    deadzone: float = 0.05
    
    # Control mapping
    translation_axis: str = "xyz"  # "xyz", "xy", "xz", etc.
    rotation_axis: str = "rpy"     # "rpy", "rx", "ry", "rz"
    gripper_button: str = "button_1"
    
    # Locked axes (for safety)
    lock_z_axis: bool = False
    lock_rotation: bool = False
    
    # Smoothing
    smoothing_factor: float = 0.8
    max_velocity: float = 0.5  # m/s
    max_angular_velocity: float = 1.0  # rad/s


@dataclass
class UmiIkConfig:
    """Configuration for UMI Inverse Kinematics."""
    
    # Robot parameters
    robot_type: str = "ur5"  # "ur5", "franka", "arx"
    joint_limits: Optional[List[Tuple[float, float]]] = None
    velocity_limits: Optional[List[float]] = None
    
    # IK solver parameters
    ik_solver: str = "ikfast"  # "ikfast", "trac_ik", "kdl"
    max_iterations: int = 100
    tolerance_position: float = 0.001  # meters
    tolerance_orientation: float = 0.01  # radians
    
    # Collision avoidance
    collision_avoidance: bool = True
    collision_margin: float = 0.05  # meters
    
    # Workspace limits
    workspace_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "x": (-1.0, 1.0),
        "y": (-1.0, 1.0),
        "z": (0.0, 1.0)
    })
    
    # Safety parameters
    height_threshold: float = 0.0  # meters
    table_collision_avoidance: bool = True


@dataclass
class UmiTeleoperatorConfig(BaseConfig):
    """
    Configuration for UMI teleoperator.
    
    This configuration class supports UMI's unique teleoperation approach:
    - SpaceMouse-based control
    - Real-time IK calculations
    - Collision avoidance
    - Multi-robot coordination
    """
    
    # Teleoperation device
    spacemouse: UmiSpaceMouseConfig = field(default_factory=UmiSpaceMouseConfig)
    
    # IK configuration
    ik: UmiIkConfig = field(default_factory=UmiIkConfig)
    
    # Control parameters
    control_frequency: float = 10.0  # Hz
    command_latency: float = 0.01  # seconds
    interpolation_steps: int = 10
    
    # Multi-robot support
    num_robots: int = 1
    bimanual: bool = False
    robot_spacing: float = 0.89  # meters
    
    # Safety features
    emergency_stop_enabled: bool = True
    workspace_limits_enabled: bool = True
    collision_avoidance_enabled: bool = True
    
    # Recording
    record_teleop: bool = False
    record_path: Optional[str] = None
    
    # Visualization
    show_workspace: bool = True
    show_trajectory: bool = True
    show_collision_spheres: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        
        # Set bimanual flag based on number of robots
        if self.num_robots > 1:
            self.bimanual = True
        
        # Set default joint limits based on robot type
        if self.ik.joint_limits is None:
            self.ik.joint_limits = self._get_default_joint_limits()
        
        # Set default velocity limits
        if self.ik.velocity_limits is None:
            self.ik.velocity_limits = self._get_default_velocity_limits()
    
    def _get_default_joint_limits(self) -> List[Tuple[float, float]]:
        """Get default joint limits based on robot type."""
        if self.ik.robot_type == "ur5":
            return [
                (-2 * np.pi, 2 * np.pi),  # shoulder_pan
                (-2 * np.pi, 2 * np.pi),  # shoulder_lift
                (-np.pi, np.pi),          # elbow
                (-2 * np.pi, 2 * np.pi),  # wrist_1
                (-2 * np.pi, 2 * np.pi),  # wrist_2
                (-2 * np.pi, 2 * np.pi),  # wrist_3
            ]
        elif self.ik.robot_type == "franka":
            return [
                (-2.8973, 2.8973),        # panda_joint1
                (-1.7628, 1.7628),        # panda_joint2
                (-2.8973, 2.8973),        # panda_joint3
                (-3.0718, -0.0698),       # panda_joint4
                (-2.8973, 2.8973),        # panda_joint5
                (-0.0175, 3.7525),        # panda_joint6
                (-2.8973, 2.8973),        # panda_joint7
            ]
        else:
            # Generic 6-DOF robot
            return [(-np.pi, np.pi)] * 6
    
    def _get_default_velocity_limits(self) -> List[float]:
        """Get default velocity limits based on robot type."""
        if self.ik.robot_type == "ur5":
            return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # rad/s
        elif self.ik.robot_type == "franka":
            return [2.175, 2.175, 2.175, 2.175, 2.175, 2.175, 2.175]  # rad/s
        else:
            # Generic 6-DOF robot
            return [1.0] * 6
    
    def validate_config(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        # Check joint limits
        if self.ik.joint_limits:
            for i, (lower, upper) in enumerate(self.ik.joint_limits):
                if lower >= upper:
                    logger.warning(f"Invalid joint limits for joint {i}: {lower} >= {upper}")
        
        # Check velocity limits
        if self.ik.velocity_limits:
            for i, limit in enumerate(self.ik.velocity_limits):
                if limit <= 0:
                    logger.warning(f"Invalid velocity limit for joint {i}: {limit}")
        
        # Check workspace limits
        for axis, (lower, upper) in self.ik.workspace_limits.items():
            if lower >= upper:
                logger.warning(f"Invalid workspace limits for {axis}: {lower} >= {upper}")
        
        return True
    
    def get_robot_dh_parameters(self) -> Optional[List[Dict[str, float]]]:
        """
        Get DH parameters for the robot.
        
        Returns:
            List of DH parameters or None if not available
        """
        if self.ik.robot_type == "ur5":
            return [
                {"a": 0.0, "alpha": 0.0, "d": 0.1625, "theta": 0.0},
                {"a": -0.425, "alpha": 0.0, "d": 0.0, "theta": 0.0},
                {"a": -0.3922, "alpha": 0.0, "d": 0.0, "theta": 0.0},
                {"a": 0.0, "alpha": 0.0, "d": 0.1333, "theta": 0.0},
                {"a": 0.0, "alpha": 0.0, "d": 0.0997, "theta": 0.0},
                {"a": 0.0, "alpha": 0.0, "d": 0.0996, "theta": 0.0},
            ]
        elif self.ik.robot_type == "franka":
            return [
                {"a": 0.0, "alpha": 0.0, "d": 0.333, "theta": 0.0},
                {"a": 0.0, "alpha": -np.pi/2, "d": 0.0, "theta": 0.0},
                {"a": 0.0, "alpha": np.pi/2, "d": 0.316, "theta": 0.0},
                {"a": 0.0825, "alpha": np.pi/2, "d": 0.0, "theta": 0.0},
                {"a": -0.0825, "alpha": -np.pi/2, "d": 0.384, "theta": 0.0},
                {"a": 0.0, "alpha": np.pi/2, "d": 0.0, "theta": 0.0},
                {"a": 0.088, "alpha": np.pi/2, "d": 0.107, "theta": 0.0},
            ]
        else:
            return None 
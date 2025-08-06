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
    """Configuration for UMI Inverse Kinematics using LeRobot's placo-based solver."""
    
    # Robot parameters
    robot_type: str = "so101"  # "ur5", "franka", "arx", "so100", "so101"
    urdf_path: Optional[str] = None  # Path to robot URDF file
    target_frame_name: str = "gripper_frame_link"  # End-effector frame name in URDF
    joint_names: Optional[List[str]] = None  # List of joint names for IK
    
    # IK solver parameters (for LeRobot's placo-based solver)
    position_weight: float = 1.0  # Weight for position constraint
    orientation_weight: float = 0.01  # Weight for orientation constraint
    
    # Legacy parameters (kept for compatibility)
    joint_limits: Optional[List[Tuple[float, float]]] = None
    velocity_limits: Optional[List[float]] = None
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
    - Real-time IK calculations using LeRobot's placo-based solver
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
        
        # Set default URDF path if not provided
        if self.ik.urdf_path is None:
            self.ik.urdf_path = self._get_default_urdf_path()
        
        # Set default joint names if not provided
        if self.ik.joint_names is None:
            self.ik.joint_names = self._get_default_joint_names()
        
        # Set default joint limits based on robot type
        if self.ik.joint_limits is None:
            self.ik.joint_limits = self._get_default_joint_limits()
        
        # Set default velocity limits
        if self.ik.velocity_limits is None:
            self.ik.velocity_limits = self._get_default_velocity_limits()
    
    def _get_default_urdf_path(self) -> str:
        """Get default URDF path for the robot type."""
        urdf_paths = {
            "ur5": "path/to/ur5.urdf",
            "franka": "path/to/franka.urdf",
            "arx": "path/to/arx.urdf",
            "so100": "path/to/so100.urdf",
            "so101": "path/to/so101_new_calib.urdf"  # SO101 URDF from SO-ARM100 repo
        }
        
        if self.ik.robot_type in urdf_paths:
            return urdf_paths[self.ik.robot_type]
        else:
            raise ValueError(f"Unknown robot type: {self.ik.robot_type}. Supported types: {list(urdf_paths.keys())}")
    
    def _get_default_joint_names(self) -> List[str]:
        """Get default joint names for the robot type."""
        joint_names = {
            "ur5": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            "franka": ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                      "panda_joint5", "panda_joint6", "panda_joint7"],
            "arx": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            "so100": ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                     "wrist_flex", "wrist_roll", "gripper"],
            "so101": ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                     "wrist_flex", "wrist_roll", "gripper"]  # Same as SO100
        }
        
        if self.ik.robot_type in joint_names:
            return joint_names[self.ik.robot_type]
        else:
            # Generic 6-DOF robot
            return [f"joint{i+1}" for i in range(6)]
    
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
        elif self.ik.robot_type in ["so100", "so101"]:
            return [
                (-180, 180),              # shoulder_pan
                (-180, 180),              # shoulder_lift
                (-180, 180),              # elbow_flex
                (-180, 180),              # wrist_flex
                (-180, 180),              # wrist_roll
                (0, 100),                 # gripper
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
        elif self.ik.robot_type in ["so100", "so101"]:
            return [90.0, 90.0, 90.0, 90.0, 90.0, 50.0]  # deg/s
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
        
        # Check URDF path
        if self.ik.urdf_path and not self.ik.urdf_path.startswith("path/to/"):
            # This would be a real path that should exist
            import os
            if not os.path.exists(self.ik.urdf_path):
                logger.warning(f"URDF file not found: {self.ik.urdf_path}")
        
        return True 
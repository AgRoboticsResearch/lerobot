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
UMI Robot Configuration for LeRobot.

This module provides configuration classes for UMI robot setups, supporting
various UMI hardware configurations and robot types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np

from ...utils.config import BaseConfig


@dataclass
class UmiGripperConfig:
    """Configuration for UMI gripper."""
    
    gripper_type: str = "wsg50"  # "wsg50", "franka", "custom"
    gripper_ip: Optional[str] = None
    gripper_port: int = 1000
    gripper_hw_idx: int = 0
    
    # WSG50 specific
    wsg50_width_range: tuple = (0.0, 0.11)  # meters
    wsg50_speed: float = 0.2  # m/s
    wsg50_force: float = 40.0  # N
    
    # Franka specific
    franka_gripper_width_range: tuple = (0.0, 0.08)  # meters
    franka_gripper_speed: float = 0.1  # m/s
    franka_gripper_force: float = 20.0  # N


@dataclass
class UmiCameraConfig:
    """Configuration for UMI camera."""
    
    camera_type: str = "gopro"  # "gopro", "realsense", "uvc"
    camera_serial: Optional[str] = None
    camera_idx: int = 0
    camera_ip: Optional[str] = None
    
    # Camera parameters
    resolution: tuple = (1920, 1080)
    fps: int = 30
    exposure: Optional[float] = None
    gain: Optional[float] = None
    
    # Intrinsics (if known)
    intrinsics: Optional[Dict[str, float]] = None
    
    # Fisheye parameters
    fisheye: bool = True
    fisheye_params: Optional[Dict[str, float]] = None


@dataclass
class UmiRobotArmConfig:
    """Configuration for UMI robot arm."""
    
    robot_type: str = "ur5"  # "ur5", "franka", "arx"
    robot_ip: Optional[str] = None
    robot_port: int = 30002
    
    # Robot parameters
    payload_mass: float = 1.81  # kg
    payload_cog: tuple = (0.002, -0.006, 0.037)  # meters (x, y, z)
    
    # Safety parameters
    sphere_center: tuple = (0.0, 0.0, 0.0)  # meters
    sphere_radius: float = 0.1  # meters
    height_threshold: float = 0.0  # meters
    
    # Control parameters
    control_frequency: float = 10.0  # Hz
    command_latency: float = 0.01  # seconds
    interpolation_steps: int = 10
    
    # UR5 specific
    ur5_tcp_offset: tuple = (0.0, 0.0, 0.0)  # meters
    ur5_payload_mass: float = 1.81  # kg
    ur5_payload_cog: tuple = (0.002, -0.006, 0.037)  # meters
    
    # Franka specific
    franka_tcp_offset: tuple = (0.0, 0.0, 0.0)  # meters
    franka_payload_mass: float = 1.0  # kg
    franka_payload_cog: tuple = (0.0, 0.0, 0.0)  # meters


@dataclass
class UmiTeleopConfig:
    """Configuration for UMI teleoperation."""
    
    teleop_type: str = "spacemouse"  # "spacemouse", "keyboard", "gamepad"
    
    # SpaceMouse specific
    spacemouse_device: Optional[str] = None
    spacemouse_sensitivity: float = 1.0
    spacemouse_deadzone: float = 0.05
    
    # Control mapping
    control_mapping: Dict[str, str] = field(default_factory=lambda: {
        "translation": "xyz",
        "rotation": "rpy",
        "gripper": "button_1"
    })


@dataclass
class UmiRobotConfig(BaseConfig):
    """
    Configuration for UMI robot setup.
    
    This configuration class supports various UMI hardware setups including:
    - Single and bimanual robot arms (UR5, Franka, ARX)
    - Multiple camera setups (GoPro, RealSense, UVC)
    - Gripper configurations (WSG50, Franka gripper)
    - Teleoperation interfaces (SpaceMouse, keyboard, gamepad)
    """
    
    # Robot arms
    robots: List[UmiRobotArmConfig] = field(default_factory=lambda: [UmiRobotArmConfig()])
    
    # Grippers
    grippers: List[UmiGripperConfig] = field(default_factory=lambda: [UmiGripperConfig()])
    
    # Cameras
    cameras: List[UmiCameraConfig] = field(default_factory=lambda: [UmiCameraConfig()])
    
    # Teleoperation
    teleop: UmiTeleopConfig = field(default_factory=UmiTeleopConfig)
    
    # Multi-robot setup
    bimanual: bool = False
    robot_spacing: float = 0.89  # meters (distance between robots in bimanual setup)
    
    # Safety configuration
    collision_avoidance: bool = True
    emergency_stop_enabled: bool = True
    workspace_limits: Dict[str, tuple] = field(default_factory=lambda: {
        "x": (-1.0, 1.0),
        "y": (-1.0, 1.0),
        "z": (0.0, 1.0)
    })
    
    # Calibration
    calibration_dir: Optional[str] = None
    camera_intrinsics_path: Optional[str] = None
    aruco_config_path: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        
        # Ensure we have at least one robot, gripper, and camera
        if not self.robots:
            self.robots = [UmiRobotArmConfig()]
        if not self.grippers:
            self.grippers = [UmiGripperConfig()]
        if not self.cameras:
            self.cameras = [UmiCameraConfig()]
        
        # Set bimanual flag based on number of robots
        if len(self.robots) > 1:
            self.bimanual = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "UmiRobotConfig":
        """
        Create UmiRobotConfig from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            UmiRobotConfig instance
        """
        import yaml
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclass instances
        if 'robots' in config_dict:
            config_dict['robots'] = [UmiRobotArmConfig(**robot) for robot in config_dict['robots']]
        if 'grippers' in config_dict:
            config_dict['grippers'] = [UmiGripperConfig(**gripper) for gripper in config_dict['grippers']]
        if 'cameras' in config_dict:
            config_dict['cameras'] = [UmiCameraConfig(**camera) for camera in config_dict['cameras']]
        if 'teleop' in config_dict:
            config_dict['teleop'] = UmiTeleopConfig(**config_dict['teleop'])
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        import yaml
        
        config_dict = {
            'robots': [robot.__dict__ for robot in self.robots],
            'grippers': [gripper.__dict__ for gripper in self.grippers],
            'cameras': [camera.__dict__ for camera in self.cameras],
            'teleop': self.teleop.__dict__,
            'bimanual': self.bimanual,
            'robot_spacing': self.robot_spacing,
            'collision_avoidance': self.collision_avoidance,
            'emergency_stop_enabled': self.emergency_stop_enabled,
            'workspace_limits': self.workspace_limits,
            'calibration_dir': self.calibration_dir,
            'camera_intrinsics_path': self.camera_intrinsics_path,
            'aruco_config_path': self.aruco_config_path,
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_robot_config(self, robot_idx: int = 0) -> UmiRobotArmConfig:
        """Get robot configuration by index."""
        if robot_idx >= len(self.robots):
            raise IndexError(f"Robot index {robot_idx} out of range")
        return self.robots[robot_idx]
    
    def get_gripper_config(self, gripper_idx: int = 0) -> UmiGripperConfig:
        """Get gripper configuration by index."""
        if gripper_idx >= len(self.grippers):
            raise IndexError(f"Gripper index {gripper_idx} out of range")
        return self.grippers[gripper_idx]
    
    def get_camera_config(self, camera_idx: int = 0) -> UmiCameraConfig:
        """Get camera configuration by index."""
        if camera_idx >= len(self.cameras):
            raise IndexError(f"Camera index {camera_idx} out of range")
        return self.cameras[camera_idx]
    
    def validate_config(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        # Check that we have matching numbers of robots and grippers
        if len(self.robots) != len(self.grippers):
            logger.warning(f"Number of robots ({len(self.robots)}) doesn't match number of grippers ({len(self.grippers)})")
        
        # Check robot IPs
        for i, robot in enumerate(self.robots):
            if not robot.robot_ip:
                logger.warning(f"Robot {i} has no IP address configured")
        
        # Check gripper IPs
        for i, gripper in enumerate(self.grippers):
            if gripper.gripper_type == "wsg50" and not gripper.gripper_ip:
                logger.warning(f"WSG50 gripper {i} has no IP address configured")
        
        return True 
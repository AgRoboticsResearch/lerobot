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
Configuration for ORB-SLAM SO101 Teleoperator.

This module defines the configuration parameters for the ORB-SLAM based
teleoperation system that controls the SO101 follower arm.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

from ...utils.config import BaseConfig
from ...teleoperators.config import TeleoperatorConfig


@dataclass
class CameraConfig(BaseConfig):
    """Configuration for RealSense camera setup."""
    
    # Camera parameters
    serial_number_or_name: str = ""  # Auto-detect if empty
    fps: int = 30
    width: int = 640
    height: int = 480
    use_depth: bool = True
    
    # Camera calibration (from your YAML file)
    fx: float = 419.8328552246094
    fy: float = 419.8328552246094
    cx: float = 429.5089416503906
    cy: float = 237.1636505126953
    baseline: float = 0.0499585  # Stereo baseline in meters


@dataclass
class OrbSlamConfig(BaseConfig):
    """Configuration for ORB-SLAM processing."""
    
    # ORB-SLAM parameters
    max_features: int = 2000
    output_frequency: float = 30.0  # Hz
    enable_visualization: bool = True
    pose_history_size: int = 100
    
    # Feature detection parameters
    scale_factor: float = 1.2
    n_levels: int = 8
    min_threshold: int = 7
    min_parallax: float = 1.0
    
    # Processing parameters
    ransac_threshold: float = 0.5
    min_matches: int = 8
    max_distance: float = 50.0


@dataclass
class RobotConfig(BaseConfig):
    """Configuration for SO101 robot arm."""
    
    # Robot type
    robot_type: str = "so101"
    
    # URDF paths
    urdf_path: Optional[str] = None  # Will auto-detect if None
    
    # Joint limits (in radians)
    joint_limits: Dict[str, List[float]] = field(default_factory=lambda: {
        "lower": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
        "upper": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
    })
    
    # Velocity limits (rad/s)
    velocity_limits: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # IK parameters
    ik_max_iterations: int = 100
    ik_tolerance: float = 1e-3
    ik_alpha: float = 0.1  # Damping factor


@dataclass
class SafetyConfig(BaseConfig):
    """Configuration for safety and limits."""
    
    # Workspace limits (meters)
    workspace_limits: Dict[str, List[float]] = field(default_factory=lambda: {
        "x": [-0.5, 0.5],
        "y": [-0.5, 0.5],
        "z": [0.1, 0.8]
    })
    
    # Velocity limits
    max_velocity: float = 0.1  # m/s
    max_angular_velocity: float = 0.5  # rad/s
    
    # Safety checks
    enable_collision_detection: bool = True
    enable_workspace_limits: bool = True
    enable_velocity_limits: bool = True
    
    # Emergency stop
    emergency_stop_timeout: float = 0.1  # seconds


@dataclass
class ControlConfig(BaseConfig):
    """Configuration for control loop and smoothing."""
    
    # Control loop
    control_frequency: float = 30.0  # Hz
    control_timeout: float = 0.1  # seconds
    
    # Smoothing parameters
    pose_smoothing_alpha: float = 0.7
    velocity_smoothing_alpha: float = 0.8
    pose_history_size: int = 10
    velocity_history_size: int = 5
    
    # Scale factors
    camera_to_robot_scale: float = 0.1  # Scale camera movement to robot movement
    position_scale: float = 1.0
    orientation_scale: float = 1.0
    
    # Deadzone
    position_deadzone: float = 0.01  # meters
    orientation_deadzone: float = 0.01  # radians


@dataclass
class VisualizationConfig(BaseConfig):
    """Configuration for visualization and debugging."""
    
    # Visualization options
    enable_pose_visualization: bool = True
    enable_trajectory_plotting: bool = True
    enable_debug_info: bool = True
    
    # Plotting parameters
    plot_update_frequency: float = 5.0  # Hz
    trajectory_history_size: int = 1000
    
    # Debug output
    log_level: str = "INFO"
    save_trajectory: bool = True
    trajectory_file: str = "orb_slam_so101_trajectory.txt"


@dataclass
class OrbSlamSo101TeleoperatorConfig(TeleoperatorConfig):
    """
    Configuration for ORB-SLAM SO101 Teleoperator.
    
    This configuration combines all the parameters needed for:
    - RealSense camera setup and calibration
    - ORB-SLAM visual odometry processing
    - SO101 robot arm control
    - Safety and workspace limits
    - Control loop and smoothing
    - Visualization and debugging
    """
    
    # Base teleoperator config
    id: str = "orb_slam_so101_teleoperator"
    calibration_dir: Optional[str] = None
    
    # Component configurations
    camera: CameraConfig = field(default_factory=CameraConfig)
    orb_slam: OrbSlamConfig = field(default_factory=OrbSlamConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Advanced settings
    enable_auto_calibration: bool = True
    enable_emergency_stop: bool = True
    enable_safety_monitoring: bool = True
    
    # Performance settings
    use_multithreading: bool = True
    buffer_size: int = 10
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate camera parameters
        if self.camera.fps <= 0:
            raise ValueError("Camera FPS must be positive")
        
        if self.camera.width <= 0 or self.camera.height <= 0:
            raise ValueError("Camera dimensions must be positive")
        
        # Validate ORB-SLAM parameters
        if self.orb_slam.output_frequency <= 0:
            raise ValueError("ORB-SLAM output frequency must be positive")
        
        if self.orb_slam.max_features <= 0:
            raise ValueError("ORB-SLAM max features must be positive")
        
        # Validate control parameters
        if self.control.control_frequency <= 0:
            raise ValueError("Control frequency must be positive")
        
        if not (0 <= self.control.pose_smoothing_alpha <= 1):
            raise ValueError("Pose smoothing alpha must be between 0 and 1")
        
        # Validate safety parameters
        for axis, limits in self.safety.workspace_limits.items():
            if len(limits) != 2 or limits[0] >= limits[1]:
                raise ValueError(f"Invalid workspace limits for {axis}: {limits}")
        
        if self.safety.max_velocity <= 0:
            raise ValueError("Max velocity must be positive")
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get camera intrinsic matrix from configuration."""
        return np.array([
            [self.camera.fx, 0, self.camera.cx],
            [0, self.camera.fy, self.camera.cy],
            [0, 0, 1]
        ])
    
    def get_workspace_limits_array(self) -> np.ndarray:
        """Get workspace limits as numpy array."""
        limits = self.safety.workspace_limits
        return np.array([
            [limits['x'][0], limits['x'][1]],
            [limits['y'][0], limits['y'][1]],
            [limits['z'][0], limits['z'][1]]
        ])
    
    def get_joint_limits_array(self) -> np.ndarray:
        """Get joint limits as numpy array."""
        limits = self.robot.joint_limits
        return np.array([
            limits['lower'],
            limits['upper']
        ]).T 
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
ORB-SLAM Integration for LeRobot

This module provides ORB-SLAM visual odometry integration that works with
LeRobot's existing camera infrastructure. It focuses specifically on:
1. Visual odometry from dual camera setup
2. Pose estimation and tracking
3. Integration with LeRobot's camera classes

This is the ONLY component we need to integrate from UMI - everything else
is already available in LeRobot's mature infrastructure.
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Check if UMI is available for ORB-SLAM
try:
    # Add UMI to path if available
    umi_path = Path("universal_manipulation_interface")
    if umi_path.exists():
        sys.path.append(str(umi_path))
        from umi.visual_odometry import ORBSLAMProcessor
        ORB_SLAM_AVAILABLE = True
        logger.info("UMI ORB-SLAM integration available")
    else:
        ORB_SLAM_AVAILABLE = False
        logger.warning("UMI not found - ORB-SLAM features will be limited")
except ImportError:
    ORB_SLAM_AVAILABLE = False
    logger.warning("UMI ORB-SLAM not available - using fallback implementation")


@dataclass
class OrbSlamConfig:
    """Configuration for ORB-SLAM integration."""
    
    # Camera configuration
    camera_config_path: str = "universal_manipulation_interface/example/calibration/camera_config.yaml"
    
    # ORB-SLAM parameters
    vocabulary_path: str = "universal_manipulation_interface/example/calibration/ORBvoc.txt"
    settings_path: str = "universal_manipulation_interface/example/calibration/settings.yaml"
    
    # Processing parameters
    max_features: int = 2000
    scale_factor: float = 1.2
    n_levels: int = 8
    min_threshold: int = 7
    min_parallax: float = 1.0
    
    # Output parameters
    output_frequency: float = 10.0  # Hz
    pose_history_size: int = 100
    
    # Debug parameters
    enable_visualization: bool = False
    save_trajectory: bool = False
    trajectory_path: str = "orb_slam_trajectory.txt"


class OrbSlamProcessor:
    """
    ORB-SLAM processor for visual odometry.
    
    This class integrates ORB-SLAM with LeRobot's camera infrastructure
    to provide real-time pose estimation and tracking.
    """
    
    def __init__(self, config: OrbSlamConfig):
        """
        Initialize ORB-SLAM processor.
        
        Args:
            config: ORB-SLAM configuration
        """
        self.config = config
        self.is_initialized = False
        self.current_pose = np.eye(4)
        self.pose_history = []
        self.frame_count = 0
        
        # Initialize ORB-SLAM if available
        if ORB_SLAM_AVAILABLE:
            self._init_orb_slam()
        else:
            self._init_fallback()
    
    def _init_orb_slam(self):
        """Initialize UMI's ORB-SLAM processor."""
        try:
            logger.info("Initializing UMI ORB-SLAM processor...")
            
            # Initialize UMI's ORB-SLAM processor
            self.orb_slam = ORBSLAMProcessor(
                vocabulary_path=self.config.vocabulary_path,
                settings_path=self.config.settings_path,
                camera_config_path=self.config.camera_config_path
            )
            
            self.is_initialized = True
            logger.info("UMI ORB-SLAM processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize UMI ORB-SLAM: {e}")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback pose estimation (when UMI is not available)."""
        logger.info("Initializing fallback pose estimation...")
        
        # Simple feature-based pose estimation as fallback
        self.feature_detector = cv2.ORB_create(
            nfeatures=self.config.max_features,
            scaleFactor=self.config.scale_factor,
            nlevels=self.config.n_levels,
            edgeThreshold=self.config.min_threshold
        )
        
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        
        self.is_initialized = True
        logger.info("Fallback pose estimation initialized")
    
    def process_camera_frames(self, frames: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Process camera frames to estimate pose using ORB-SLAM.
        
        Args:
            frames: Dictionary of camera frames from LeRobot cameras
            
        Returns:
            Estimated pose as 4x4 transformation matrix or None
        """
        if not self.is_initialized:
            logger.error("ORB-SLAM processor not initialized")
            return None
        
        if ORB_SLAM_AVAILABLE:
            return self._process_with_umi_orb_slam(frames)
        else:
            return self._process_with_fallback(frames)
    
    def _process_with_umi_orb_slam(self, frames: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Process frames using UMI's ORB-SLAM implementation."""
        try:
            # Convert LeRobot camera frames to UMI format
            umi_frames = self._convert_frames_to_umi_format(frames)
            
            # Process with UMI's ORB-SLAM
            pose = self.orb_slam.process_frames(umi_frames)
            
            if pose is not None:
                self.current_pose = pose
                self._update_pose_history(pose)
                self.frame_count += 1
                
                logger.debug(f"ORB-SLAM pose estimated: frame {self.frame_count}")
                return pose
            else:
                logger.debug("ORB-SLAM failed to estimate pose")
                return None
                
        except Exception as e:
            logger.error(f"Error in UMI ORB-SLAM processing: {e}")
            return None
    
    def _process_with_fallback(self, frames: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Process frames using fallback feature-based pose estimation."""
        try:
            # For fallback, we'll use a simple approach
            # In practice, this would be more sophisticated
            
            if len(frames) < 2:
                logger.warning("Need at least 2 camera frames for stereo processing")
                return None
            
            # Get the first two frames (assuming stereo setup)
            frame_names = list(frames.keys())
            frame1 = frames[frame_names[0]]
            frame2 = frames[frame_names[1]]
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Detect features
            keypoints1, descriptors1 = self.feature_detector.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = self.feature_detector.detectAndCompute(gray2, None)
            
            if descriptors1 is None or descriptors2 is None:
                logger.debug("No features detected")
                return None
            
            # Match features
            matches = self.feature_matcher.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < 10:
                logger.debug("Insufficient feature matches")
                return None
            
            # Simple pose estimation (placeholder)
            # In practice, this would use proper stereo geometry
            pose = self._estimate_pose_from_matches(keypoints1, keypoints2, matches)
            
            if pose is not None:
                self.current_pose = pose
                self._update_pose_history(pose)
                self.frame_count += 1
                
                logger.debug(f"Fallback pose estimated: frame {self.frame_count}")
                return pose
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fallback processing: {e}")
            return None
    
    def _convert_frames_to_umi_format(self, frames: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Convert LeRobot camera frames to UMI format."""
        umi_frames = {}
        
        for camera_name, frame in frames.items():
            # Convert frame to UMI format
            # This would depend on UMI's expected format
            umi_frames[camera_name] = {
                "image": frame,
                "timestamp": time.time(),
                "camera_id": camera_name
            }
        
        return umi_frames
    
    def _estimate_pose_from_matches(self, kp1, kp2, matches):
        """Estimate pose from feature matches (simplified)."""
        # This is a simplified implementation
        # In practice, this would use proper stereo geometry and RANSAC
        
        if len(matches) < 8:
            return None
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find fundamental matrix
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
        
        if F is None:
            return None
        
        # For now, return identity matrix (placeholder)
        # In practice, this would recover pose from fundamental matrix
        return np.eye(4)
    
    def _update_pose_history(self, pose: np.ndarray):
        """Update pose history for trajectory tracking."""
        self.pose_history.append({
            "timestamp": time.time(),
            "pose": pose.copy(),
            "frame_count": self.frame_count
        })
        
        # Keep only recent poses
        if len(self.pose_history) > self.config.pose_history_size:
            self.pose_history.pop(0)
    
    def get_current_pose(self) -> np.ndarray:
        """Get the current estimated pose."""
        return self.current_pose.copy()
    
    def get_pose_history(self) -> List[Dict[str, Any]]:
        """Get the pose history."""
        return self.pose_history.copy()
    
    def get_trajectory(self) -> np.ndarray:
        """Get the trajectory as a sequence of poses."""
        if not self.pose_history:
            return np.array([])
        
        poses = [entry["pose"] for entry in self.pose_history]
        return np.array(poses)
    
    def save_trajectory(self, filepath: Optional[str] = None):
        """Save trajectory to file."""
        if filepath is None:
            filepath = self.config.trajectory_path
        
        if not self.pose_history:
            logger.warning("No trajectory to save")
            return
        
        try:
            with open(filepath, 'w') as f:
                f.write("# ORB-SLAM Trajectory\n")
                f.write("# timestamp frame_count tx ty tz qx qy qz qw\n")
                
                for entry in self.pose_history:
                    timestamp = entry["timestamp"]
                    frame_count = entry["frame_count"]
                    pose = entry["pose"]
                    
                    # Extract translation
                    tx, ty, tz = pose[:3, 3]
                    
                    # Extract rotation (convert to quaternion)
                    rotation_matrix = pose[:3, :3]
                    # This is simplified - in practice, use proper quaternion conversion
                    qx, qy, qz, qw = 0, 0, 0, 1  # Placeholder
                    
                    f.write(f"{timestamp:.6f} {frame_count} {tx:.6f} {ty:.6f} {tz:.6f} "
                           f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
            logger.info(f"Trajectory saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")
    
    def reset(self):
        """Reset the ORB-SLAM processor."""
        self.current_pose = np.eye(4)
        self.pose_history.clear()
        self.frame_count = 0
        
        if ORB_SLAM_AVAILABLE and hasattr(self, 'orb_slam'):
            self.orb_slam.reset()
        
        logger.info("ORB-SLAM processor reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the ORB-SLAM processor."""
        return {
            "is_initialized": self.is_initialized,
            "orb_slam_available": ORB_SLAM_AVAILABLE,
            "frame_count": self.frame_count,
            "pose_history_size": len(self.pose_history),
            "current_pose": self.current_pose.tolist(),
            "config": {
                "max_features": self.config.max_features,
                "output_frequency": self.config.output_frequency,
                "enable_visualization": self.config.enable_visualization
            }
        }


def create_orb_slam_processor(config: Optional[OrbSlamConfig] = None) -> OrbSlamProcessor:
    """
    Create an ORB-SLAM processor instance.
    
    Args:
        config: ORB-SLAM configuration (optional)
        
    Returns:
        Initialized ORB-SLAM processor
    """
    if config is None:
        config = OrbSlamConfig()
    
    return OrbSlamProcessor(config)


# Convenience functions for easy integration
def estimate_pose_from_cameras(
    frames: Dict[str, np.ndarray],
    config: Optional[OrbSlamConfig] = None
) -> Optional[np.ndarray]:
    """
    Convenience function to estimate pose from camera frames.
    
    Args:
        frames: Dictionary of camera frames
        config: ORB-SLAM configuration (optional)
        
    Returns:
        Estimated pose as 4x4 transformation matrix or None
    """
    processor = create_orb_slam_processor(config)
    return processor.process_camera_frames(frames)


def get_orb_slam_status() -> Dict[str, Any]:
    """Get the status of ORB-SLAM integration."""
    return {
        "orb_slam_available": ORB_SLAM_AVAILABLE,
        "umi_path_exists": Path("universal_manipulation_interface").exists(),
        "integration_ready": ORB_SLAM_AVAILABLE
    } 
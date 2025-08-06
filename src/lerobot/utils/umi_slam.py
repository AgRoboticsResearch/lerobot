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
UMI SLAM Integration for LeRobot.

This module provides integration with UMI's SLAM-based pose estimation,
which is the core innovation of UMI for "in-the-wild" robot teaching.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    from umi.common.pose_util import pose_to_mat, mat_to_pose
    from umi.common.cv_util import parse_fisheye_intrinsics, FisheyeRectConverter
    UMI_AVAILABLE = True
except ImportError:
    UMI_AVAILABLE = False
    logger.warning("UMI dependencies not available. UMI SLAM features will be limited.")


class UmiSlamProcessor:
    """
    UMI SLAM Processor for pose estimation.
    
    This class provides integration with UMI's SLAM pipeline for:
    - Multi-camera pose estimation
    - Fisheye lens handling
    - Real-time pose tracking
    - Data synchronization
    """
    
    def __init__(
        self,
        umi_root_path: Optional[str] = None,
        calibration_dir: Optional[str] = None,
        camera_intrinsics_path: Optional[str] = None,
        aruco_config_path: Optional[str] = None
    ):
        """
        Initialize UMI SLAM processor.
        
        Args:
            umi_root_path: Path to UMI repository root
            calibration_dir: Directory containing calibration files
            camera_intrinsics_path: Path to camera intrinsics file
            aruco_config_path: Path to Aruco configuration file
        """
        if not UMI_AVAILABLE:
            raise ImportError("UMI dependencies not available. Please install UMI first.")
        
        self.umi_root_path = Path(umi_root_path) if umi_root_path else Path("universal_manipulation_interface")
        self.calibration_dir = Path(calibration_dir) if calibration_dir else self.umi_root_path / "example" / "calibration"
        self.camera_intrinsics_path = Path(camera_intrinsics_path) if camera_intrinsics_path else self.calibration_dir / "gopro_intrinsics_2_7k.json"
        self.aruco_config_path = Path(aruco_config_path) if aruco_config_path else self.calibration_dir / "aruco_config.yaml"
        
        # Validate paths
        self._validate_paths()
        
        # Initialize fisheye converter if intrinsics available
        self.fisheye_converter = None
        if self.camera_intrinsics_path.exists():
            self._init_fisheye_converter()
    
    def _validate_paths(self):
        """Validate required paths exist."""
        if not self.umi_root_path.exists():
            raise FileNotFoundError(f"UMI root path not found: {self.umi_root_path}")
        
        if not self.calibration_dir.exists():
            logger.warning(f"Calibration directory not found: {self.calibration_dir}")
        
        if not self.camera_intrinsics_path.exists():
            logger.warning(f"Camera intrinsics not found: {self.camera_intrinsics_path}")
        
        if not self.aruco_config_path.exists():
            logger.warning(f"Aruco config not found: {self.aruco_config_path}")
    
    def _init_fisheye_converter(self):
        """Initialize fisheye lens converter."""
        try:
            intrinsics = parse_fisheye_intrinsics(str(self.camera_intrinsics_path))
            self.fisheye_converter = FisheyeRectConverter(intrinsics)
            logger.info("Initialized fisheye converter")
        except Exception as e:
            logger.warning(f"Failed to initialize fisheye converter: {e}")
    
    def run_slam_pipeline(self, session_dir: str) -> bool:
        """
        Run UMI SLAM pipeline on a session directory.
        
        Args:
            session_dir: Path to session directory containing video data
            
        Returns:
            True if SLAM pipeline completed successfully
        """
        session_path = Path(session_dir)
        if not session_path.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        logger.info(f"Running UMI SLAM pipeline on: {session_dir}")
        
        try:
            # Run UMI SLAM pipeline
            cmd = [
                "python", "run_slam_pipeline.py",
                str(session_path)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.umi_root_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("UMI SLAM pipeline completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"UMI SLAM pipeline failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error running UMI SLAM pipeline: {e}")
            return False
    
    def process_videos(self, session_dir: str) -> bool:
        """
        Process videos for SLAM pipeline.
        
        Args:
            session_dir: Path to session directory
            
        Returns:
            True if video processing completed successfully
        """
        session_path = Path(session_dir)
        script_path = self.umi_root_path / "scripts_slam_pipeline" / "00_process_videos.py"
        
        if not script_path.exists():
            logger.error(f"Video processing script not found: {script_path}")
            return False
        
        try:
            cmd = ["python", str(script_path), str(session_path)]
            result = subprocess.run(cmd, cwd=self.umi_root_path, check=True)
            logger.info("Video processing completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Video processing failed: {e}")
            return False
    
    def extract_imu_data(self, session_dir: str) -> bool:
        """
        Extract IMU data from GoPro videos.
        
        Args:
            session_dir: Path to session directory
            
        Returns:
            True if IMU extraction completed successfully
        """
        session_path = Path(session_dir)
        script_path = self.umi_root_path / "scripts_slam_pipeline" / "01_extract_gopro_imu.py"
        
        if not script_path.exists():
            logger.error(f"IMU extraction script not found: {script_path}")
            return False
        
        try:
            cmd = ["python", str(script_path), str(session_path)]
            result = subprocess.run(cmd, cwd=self.umi_root_path, check=True)
            logger.info("IMU extraction completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"IMU extraction failed: {e}")
            return False
    
    def create_slam_map(self, session_dir: str) -> bool:
        """
        Create SLAM map from mapping data.
        
        Args:
            session_dir: Path to session directory
            
        Returns:
            True if map creation completed successfully
        """
        session_path = Path(session_dir)
        demo_dir = session_path / "demos"
        mapping_dir = demo_dir / "mapping"
        map_path = mapping_dir / "map_atlas.osa"
        
        if map_path.exists():
            logger.info("SLAM map already exists")
            return True
        
        script_path = self.umi_root_path / "scripts_slam_pipeline" / "02_create_map.py"
        
        if not script_path.exists():
            logger.error(f"Map creation script not found: {script_path}")
            return False
        
        try:
            cmd = [
                "python", str(script_path),
                "--input_dir", str(mapping_dir),
                "--map_path", str(map_path)
            ]
            result = subprocess.run(cmd, cwd=self.umi_root_path, check=True)
            logger.info("SLAM map creation completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"SLAM map creation failed: {e}")
            return False
    
    def run_batch_slam(self, session_dir: str) -> bool:
        """
        Run batch SLAM on all demonstrations.
        
        Args:
            session_dir: Path to session directory
            
        Returns:
            True if batch SLAM completed successfully
        """
        session_path = Path(session_dir)
        demo_dir = session_path / "demos"
        mapping_dir = demo_dir / "mapping"
        map_path = mapping_dir / "map_atlas.osa"
        
        script_path = self.umi_root_path / "scripts_slam_pipeline" / "03_batch_slam.py"
        
        if not script_path.exists():
            logger.error(f"Batch SLAM script not found: {script_path}")
            return False
        
        try:
            cmd = [
                "python", str(script_path),
                "--input_dir", str(demo_dir),
                "--map_path", str(map_path)
            ]
            result = subprocess.run(cmd, cwd=self.umi_root_path, check=True)
            logger.info("Batch SLAM completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Batch SLAM failed: {e}")
            return False
    
    def detect_aruco_markers(self, session_dir: str) -> bool:
        """
        Detect Aruco markers in videos.
        
        Args:
            session_dir: Path to session directory
            
        Returns:
            True if Aruco detection completed successfully
        """
        session_path = Path(session_dir)
        demo_dir = session_path / "demos"
        
        script_path = self.umi_root_path / "scripts_slam_pipeline" / "04_detect_aruco.py"
        
        if not script_path.exists():
            logger.error(f"Aruco detection script not found: {script_path}")
            return False
        
        try:
            cmd = [
                "python", str(script_path),
                "--input_dir", str(demo_dir),
                "--camera_intrinsics", str(self.camera_intrinsics_path),
                "--aruco_yaml", str(self.aruco_config_path)
            ]
            result = subprocess.run(cmd, cwd=self.umi_root_path, check=True)
            logger.info("Aruco detection completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Aruco detection failed: {e}")
            return False
    
    def run_calibrations(self, session_dir: str) -> bool:
        """
        Run camera and robot calibrations.
        
        Args:
            session_dir: Path to session directory
            
        Returns:
            True if calibration completed successfully
        """
        session_path = Path(session_dir)
        demo_dir = session_path / "demos"
        
        script_path = self.umi_root_path / "scripts_slam_pipeline" / "05_run_calibrations.py"
        
        if not script_path.exists():
            logger.error(f"Calibration script not found: {script_path}")
            return False
        
        try:
            cmd = [
                "python", str(script_path),
                "--input_dir", str(demo_dir),
                "--camera_intrinsics", str(self.camera_intrinsics_path),
                "--aruco_yaml", str(self.aruco_config_path)
            ]
            result = subprocess.run(cmd, cwd=self.umi_root_path, check=True)
            logger.info("Calibration completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def generate_dataset(self, session_dir: str, output_path: str) -> bool:
        """
        Generate training dataset from processed session.
        
        Args:
            session_dir: Path to session directory
            output_path: Path to output dataset file
            
        Returns:
            True if dataset generation completed successfully
        """
        session_path = Path(session_dir)
        script_path = self.umi_root_path / "scripts_slam_pipeline" / "07_generate_replay_buffer.py"
        
        if not script_path.exists():
            logger.error(f"Dataset generation script not found: {script_path}")
            return False
        
        try:
            cmd = [
                "python", str(script_path),
                "-o", output_path,
                str(session_path)
            ]
            result = subprocess.run(cmd, cwd=self.umi_root_path, check=True)
            logger.info(f"Dataset generation completed successfully: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Dataset generation failed: {e}")
            return False
    
    def get_pose_from_image(self, image: np.ndarray, camera_idx: int = 0) -> Optional[np.ndarray]:
        """
        Get pose from image using UMI's pose estimation.
        
        Args:
            image: Input image
            camera_idx: Camera index
            
        Returns:
            Pose as 6D vector [x, y, z, rx, ry, rz] or None if failed
        """
        if self.fisheye_converter is None:
            logger.warning("Fisheye converter not initialized")
            return None
        
        try:
            # Convert fisheye image to rectified image
            rectified = self.fisheye_converter.fisheye_to_rect(image)
            
            # Here you would integrate with UMI's pose estimation
            # For now, return None as this requires full UMI integration
            logger.info("Pose estimation from image (placeholder)")
            return None
            
        except Exception as e:
            logger.error(f"Error estimating pose from image: {e}")
            return None


def create_umi_slam_processor(
    umi_root_path: Optional[str] = None,
    calibration_dir: Optional[str] = None,
    **kwargs
) -> UmiSlamProcessor:
    """
    Factory function to create UMI SLAM processor.
    
    Args:
        umi_root_path: Path to UMI repository root
        calibration_dir: Directory containing calibration files
        **kwargs: Additional arguments
        
    Returns:
        UmiSlamProcessor instance
    """
    return UmiSlamProcessor(
        umi_root_path=umi_root_path,
        calibration_dir=calibration_dir,
        **kwargs
    ) 
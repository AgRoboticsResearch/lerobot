#!/usr/bin/env python

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

r"""
Run ACT policy inference with camera input only (no robot).

This script runs policy inference using only camera observations, without
connecting to any physical robot or running a digital twin simulation.
It's useful for:
- Testing camera setup and capture
- Debugging model inference pipeline
- Measuring inference timing
- Developing camera-based features

Usage:
    python visualize_camera_prediction.py \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --cameras "{ front: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: intelrealsense, serial_number_or_name: 031522070877, width: 640, height: 480, fps: 30} }"

The observation format for relative EE policies:
    - observation.state: (obs_state_horizon, 10) - relative poses (identity at current timestep)
    - action: (action_horizon, 10) - relative future poses

Since there's no robot, the state observation uses identity pose (zero relative motion),
matching the behavior of RelativeEEDataset when there's no motion history.
"""

import logging
import time
from pathlib import Path
from typing import Any
from threading import Thread

import numpy as np
import cv2
import torch
import yaml
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for real-time updates
import matplotlib.pyplot as plt

# Try to import pyrealsense2 for getting camera intrinsics
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so101_follower.relative_ee_processor import pose10d_to_mat
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.temporal_wrapper import TemporalACTWrapper
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import init_logging

FPS = 30  # Control loop frequency (Hz) - should match training fps


def project_points_to_image(points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D image coordinates using camera intrinsics.

    Args:
        points_3d: (N, 3) array of 3D points in camera optical frame (x forward, y left, z up)
        camera_matrix: (3, 3) camera intrinsics matrix K

    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    # Transform from optical frame (x forward, y left, z up) to camera frame (x right, y down, z forward)
    # x_cam = -y, y_cam = -z, z_cam = x
    points_cam = points_3d[:, [1, 2, 0]] * np.array([[-1, -1, 1]])  # (N, 3)

    # Project using camera matrix: [u, v, 1]^T = K * [X, Y, Z]^T / Z
    points_homogeneous = points_cam  # (N, 3)
    z = points_homogeneous[:, 2:3]  # (N, 1)

    # Avoid division by zero
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)

    points_2d_homogeneous = (camera_matrix @ points_homogeneous.T).T  # (N, 3)
    points_2d = points_2d_homogeneous[:, :2] / z  # (N, 2)

    return points_2d


def draw_trajectory_on_image(img: np.ndarray, points_2d: np.ndarray, colors: list = None) -> np.ndarray:
    """Draw trajectory on image.

    Args:
        img: (H, W, 3) RGB image
        points_2d: (N, 2) array of 2D pixel coordinates
        colors: list of BGR colors for each segment

    Returns:
        Image with trajectory drawn
    """
    img_draw = img.copy()

    if colors is None:
        # Gradient from blue to red
        n = len(points_2d)
        colors = [
            (int(255 * i / n), int(255 * (1 - i / n)), 0)  # BGR
            for i in range(n - 1)
        ]

    # Draw line segments
    for i in range(len(points_2d) - 1):
        pt1 = tuple(points_2d[i].astype(int))
        pt2 = tuple(points_2d[i + 1].astype(int))

        # Check if points are within image bounds
        h, w = img.shape[:2]
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(img_draw, pt1, pt2, colors[i % len(colors)], 2)

    # Draw start point (green) and end point (red)
    if len(points_2d) > 0:
        start_pt = tuple(points_2d[0].astype(int))
        end_pt = tuple(points_2d[-1].astype(int))
        h, w = img.shape[:2]

        if 0 <= start_pt[0] < w and 0 <= start_pt[1] < h:
            cv2.circle(img_draw, start_pt, 5, (0, 255, 0), -1)  # Green
        if 0 <= end_pt[0] < w and 0 <= end_pt[1] < h:
            cv2.circle(img_draw, end_pt, 5, (0, 0, 255), -1)  # Red

    return img_draw


class TrajectoryPlotter:
    """Real-time 3D trajectory plotter using matplotlib."""

    def __init__(self):
        """Initialize the trajectory plotter."""
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []

        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initialize plot line and markers
        self.line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Predicted Trajectory')
        self.start, = self.ax.plot([], [], [], 'go', markersize=5, label='Start')
        self.head, = self.ax.plot([], [], [], 'ro', markersize=5, label='End Position')

        # Set labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Predicted End-Effector Trajectory (from origin)')
        self.ax.legend()

        # Set fixed axis limits: all from -0.25 to +0.25
        self.ax.set_xlim(-0.25, 0.25)
        self.ax.set_ylim(-0.25, 0.25)
        self.ax.set_zlim(-0.25, 0.25)

        # Start matplotlib in non-blocking mode
        plt.ion()
        self.fig.show()
        self.fig.canvas.flush_events()

    def set_trajectory(self, actions: np.ndarray):
        """Set the trajectory from relative 10D poses (from origin).

        Each action is a 10D relative pose [dx,dy,dz,rot6d_0..5,gripper] that needs
        to be converted to a 4x4 transformation matrix and applied to the base pose.

        Args:
            actions: (N, 10) array of relative actions
        """
        import numpy as np

        # Start from origin (identity matrix)
        base_pose = np.eye(4)

        # Start at origin
        self.trajectory_x = [0.0]
        self.trajectory_y = [0.0]
        self.trajectory_z = [0.0]

        # For each action in the chunk, apply the relative transformation
        for action in actions:
            # Convert 10D pose to 4x4 transformation matrix
            rel_pose = pose10d_to_mat(action[:9])

            # Apply to base pose: T_abs = T_base @ T_rel
            # All actions in chunk are relative to the SAME base pose (UMI-style)
            abs_pose = base_pose @ rel_pose

            # Extract position
            self.trajectory_x.append(abs_pose[0, 3])
            self.trajectory_y.append(abs_pose[1, 3])
            self.trajectory_z.append(abs_pose[2, 3])

    def update(self):
        """Update the plot with current trajectory."""
        # Update line data
        self.line.set_data(self.trajectory_x, self.trajectory_y)
        self.line.set_3d_properties(self.trajectory_z)

        # Update start marker (origin - first point)
        self.start.set_data([self.trajectory_x[0]], [self.trajectory_y[0]])
        self.start.set_3d_properties([self.trajectory_z[0]])

        # Update head (end position)
        self.head.set_data([self.trajectory_x[-1]], [self.trajectory_y[-1]])
        self.head.set_3d_properties([self.trajectory_z[-1]])

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the plot."""
        plt.ioff()
        plt.close(self.fig)


def parse_cameras_config(cameras_str: str | None) -> dict[str, Any]:
    """Parse cameras configuration from YAML string."""
    if not cameras_str or cameras_str.strip() == "":
        return {}

    cameras_dict = yaml.safe_load(cameras_str)
    if not isinstance(cameras_dict, dict):
        raise ValueError(f"Cameras config should be a dict, got {type(cameras_dict)}")

    cameras: dict[str, CameraConfig] = {}
    for name, config in cameras_dict.items():
        if not isinstance(config, dict):
            raise ValueError(f"Camera '{name}' config should be a dict, got {type(config)}")

        camera_type = config.pop("type", None)
        if camera_type is None:
            raise ValueError(f"Camera '{name}' missing 'type' field")

        # Map camera type to config class directly
        if camera_type == "opencv":
            camera_config_class = OpenCVCameraConfig
            # Convert index_or_path to string if it's an int (for device indices)
            if "index_or_path" in config and isinstance(config["index_or_path"], int):
                config["index_or_path"] = str(config["index_or_path"])
        elif camera_type == "intelrealsense":
            camera_config_class = RealSenseCameraConfig
            # Keep serial_number_or_name as string - YAML might parse it as int
            if "serial_number_or_name" in config and not isinstance(config["serial_number_or_name"], str):
                # If it's a number, convert to string and zero-pad if it looks like a serial number
                serial = str(config["serial_number_or_name"])
                # Re-add leading zeros if YAML removed them (common with serial numbers starting with 0)
                if len(serial) == 12 and cameras_str.count(f"0{serial}") > 0:
                    serial = "0" + serial
                config["serial_number_or_name"] = serial
        else:
            # Fallback to registry
            camera_config_class = CameraConfig.get_choice_class(camera_type)

        cameras[name] = camera_config_class(**config)

    return cameras


def make_act_pre_post_processors_with_temporal(
    config: ACTConfig,
    dataset_stats: dict | None = None,
    obs_state_horizon: int = 2,
) -> tuple:
    """Create ACT processors with temporal observation handling (UMI-style).

    This MUST match the processors used during training, otherwise
    normalization will be inconsistent and predictions will be wrong.

    Args:
        config: ACT policy configuration
        dataset_stats: Normalization statistics from dataset
        obs_state_horizon: Number of temporal steps in observations

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    from lerobot.processor import (
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        TemporalNormalizeProcessor,
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    # Create custom preprocessor with temporal normalization
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        TemporalNormalizeProcessor(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
            obs_state_horizon=obs_state_horizon,
        ),
    ]

    # Standard postprocessor
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    preprocessor = PolicyProcessorPipeline[dict, dict](
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[dict, dict](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
    )

    logging.info(f"Created temporal ACT processors with obs_state_horizon={obs_state_horizon}")

    return preprocessor, postprocessor


def build_temporal_image_observation(
    current_img: np.ndarray,
    obs_state_horizon: int,
    image_history: dict | None = None,
    camera_name: str = "camera",
) -> tuple[np.ndarray, dict]:
    """Build temporal image observation matching RelativeEEDataset format.

    For obs_state_horizon > 1, the model expects (T, C, H, W) format.
    This function maintains a history buffer and pads with current frame
    when there is not enough history (matching dataset behavior).

    Args:
        current_img: Current camera image (H, W, C) uint8 [0, 255]
        obs_state_horizon: Number of temporal frames needed
        image_history: Dict of camera_name -> list of previous images (C, H, W) float32
        camera_name: Name of this camera for the history dict

    Returns:
        Tuple of (temporal_images (T, C, H, W) or (C, H, W), updated_image_history)
    """
    if image_history is None:
        image_history = {}

    # Convert to float32 [0, 1] and transpose to (C, H, W)
    img_float = current_img.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))

    # Initialize history for this camera
    if camera_name not in image_history:
        image_history[camera_name] = []

    # Append current frame
    image_history[camera_name].append(img_chw.copy())

    # Keep only the most recent obs_state_horizon frames
    image_history[camera_name] = image_history[camera_name][-obs_state_horizon:]

    # Pad with current frame if not enough history
    if len(image_history[camera_name]) < obs_state_horizon:
        pad_needed = obs_state_horizon - len(image_history[camera_name])
        image_history[camera_name] = [img_chw.copy()] * pad_needed + image_history[camera_name]

    # Stack into (T, C, H, W) format
    temporal_images = np.stack(image_history[camera_name], axis=0)

    # For obs_state_horizon == 1, squeeze to (C, H, W) to match standard ACT
    if obs_state_horizon == 1:
        temporal_images = temporal_images.squeeze(0)

    return temporal_images, image_history


def main():
    init_logging()
    logger = logging.getLogger("visualize_camera_prediction")

    import argparse
    parser = argparse.ArgumentParser(
        description="Run ACT policy inference with camera input only (no robot)"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory or training output",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Control loop frequency (Hz)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=0,
        help="Number of control steps to run (0 for infinite)",
    )
    parser.add_argument(
        "--obs_state_horizon",
        type=int,
        default=None,
        help="Observation state horizon (auto-detected from policy if not specified)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='Camera configuration in YAML format',
    )
    parser.add_argument(
        "--cameraview",
        action="store_true",
        help="Show camera feed in cv2 window (image fed into model)",
    )
    parser.add_argument(
        "--plot-traj",
        action="store_true",
        help="Show real-time 3D trajectory plot",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load Policy
    # ========================================================================
    logger.info("Loading trained policy...")

    policy = ACTPolicy.from_pretrained(args.pretrained_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)

    # Get obs_state_horizon from policy config (or use default)
    policy_obs_state_horizon = getattr(policy.config, 'obs_state_horizon', 1)
    obs_state_horizon = args.obs_state_horizon if args.obs_state_horizon is not None else policy_obs_state_horizon

    # Only wrap with TemporalACTWrapper if obs_state_horizon > 1
    if obs_state_horizon > 1:
        original_model = policy.model
        policy.model = TemporalACTWrapper(original_model, policy.config)
        policy.model = policy.model.to(device)
        logger.info(f"Wrapped ACT model with TemporalACTWrapper (obs_state_horizon={obs_state_horizon})")

        # Load wrapper parameters if temporal_wrapper.pt exists
        wrapper_path = Path(args.pretrained_path).parent / "temporal_wrapper.pt"
        if not wrapper_path.exists():
            wrapper_path = Path(args.pretrained_path) / "temporal_wrapper.pt"
        if not wrapper_path.exists():
            wrapper_path = Path(args.pretrained_path).parent.parent / "temporal_wrapper.pt"
        if wrapper_path.exists():
            wrapper_state_dict = torch.load(wrapper_path, map_location=device)
            model_state = policy.model.state_dict()
            filtered_state = {k: v for k, v in wrapper_state_dict.items() if k in model_state}
            if filtered_state:
                policy.model.load_state_dict(filtered_state, strict=False)
                policy.model = policy.model.to(device)
                logger.info(f"Loaded TemporalACTWrapper parameters from {wrapper_path}")
            else:
                logger.warning(f"No matching parameters in temporal_wrapper.pt, using initialized params")
        else:
            logger.warning(f"No temporal_wrapper.pt found at {wrapper_path}, using initialized params")

        # Load temporal processors
        temp_preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy,
            pretrained_path=args.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
            postprocessor_overrides={},
        )

        # Extract stats from loaded preprocessor
        dataset_stats = None
        for step in temp_preprocessor.steps:
            if hasattr(step, 'stats') and step.stats is not None:
                dataset_stats = step.stats
                break

        if dataset_stats is None:
            raise ValueError("Could not extract normalization stats from pretrained model!")

        # Create temporal processors with the loaded stats
        preprocessor, postprocessor = make_act_pre_post_processors_with_temporal(
            config=policy.config,
            dataset_stats=dataset_stats,
            obs_state_horizon=obs_state_horizon,
        )
    else:
        # Use standard processors for obs_state_horizon=1
        logger.info(f"obs_state_horizon=1, TemporalACTWrapper not needed")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy,
            pretrained_path=args.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
            postprocessor_overrides={},
        )

    logger.info("Policy loaded successfully")
    logger.info(f"  Chunk size: {policy.config.chunk_size}")
    logger.info(f"  obs_state_horizon: {obs_state_horizon}")

    # ========================================================================
    # Parse cameras configuration
    # ========================================================================
    cameras_config = parse_cameras_config(args.cameras)
    if not cameras_config:
        raise ValueError("No cameras configured. Please provide --cameras argument.")

    logger.info(f"Configured cameras: {list(cameras_config.keys())}")

    # Create camera instances
    cameras = make_cameras_from_configs(cameras_config)

    # Connect all cameras
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info(f"Connected camera: {cam_name}")

        # Print camera intrinsics for RealSense cameras
        if hasattr(camera, 'intrinsics'):
            intrinsics = camera.intrinsics
            logger.info(f"  Camera intrinsics ({camera.width}x{camera.height}):")
            logger.info(f"    fx: {intrinsics.fx:.2f}")
            logger.info(f"    fy: {intrinsics.fy:.2f}")
            logger.info(f"    cx: {intrinsics.ppx:.2f}")
            logger.info(f"    cy: {intrinsics.ppy:.2f}")
            logger.info(f"    Camera matrix K:")
            logger.info(f"      [[{intrinsics.fx:.2f}, 0.00, {intrinsics.ppx:.2f}],")
            logger.info(f"       [0.00, {intrinsics.fy:.2f}, {intrinsics.ppy:.2f}],")
            logger.info(f"       [0.00, 0.00, 1.00]]")

    # ========================================================================
    # Initialize trajectory plotter if enabled
    # ========================================================================
    plotter = None
    if args.plot_traj:
        plotter = TrajectoryPlotter()
        logger.info("Trajectory plot enabled")

    # ========================================================================
    # Check camera view availability
    # ========================================================================
    if args.cameraview:
        try:
            # Test if cv2 can display windows
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("test", test_img)
            cv2.destroyWindow("test")
            logger.info("Camera view enabled")
        except Exception as e:
            logger.warning(f"Cannot show camera view (headless mode?): {e}")
            logger.warning("Disabling --cameraview")
            args.cameraview = False

    # ========================================================================
    # Prepare state observation (identity pose for relative EE)
    # ========================================================================
    # For relative EE policies, the current observation is identity pose
    # because T_rel = T_curr^{-1} @ T_curr = I
    # 10D identity pose: [dx,dy,dz, rot6d_0..5, gripper]
    # For identity: [0,0,0, 1,0,0,0,1,0, 0.5]
    identity_pose = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5], dtype=np.float32)

    if obs_state_horizon > 1:
        # All timesteps are identity (zero velocity, matching dataset padding behavior)
        obs_state = np.stack([identity_pose] * obs_state_horizon, axis=0)
    else:
        obs_state = identity_pose

    logger.info(f"Using identity pose for state observation: {obs_state.shape}")

    # ========================================================================
    # Main Inference Loop
    # ========================================================================
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    image_history = None
    step_count = 0
    trajectory_3d = None  # Store 3D trajectory points for projection
    camera_matrix = None  # Store camera intrinsics

    logger.info("Starting inference loop...")
    logger.info("Press Ctrl+C to stop\n")

    try:
        while args.num_steps == 0 or step_count < args.num_steps:
            t0 = time.perf_counter()

            # -------------------------------------------------------------------
            # Prepare batch with state observation
            # -------------------------------------------------------------------
            batch = {"observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device)}

            # -------------------------------------------------------------------
            # Get images from all cameras
            # -------------------------------------------------------------------
            camera_images = {}  # Store images for display after inference
            for cam_name, camera in cameras.items():
                img = camera.read()
                camera_images[cam_name] = img  # Store for later display

                temporal_img, image_history = build_temporal_image_observation(
                    current_img=img,
                    obs_state_horizon=obs_state_horizon,
                    image_history=image_history,
                    camera_name=cam_name,
                )
                batch[f"observation.images.{cam_name}"] = torch.from_numpy(temporal_img).unsqueeze(0).to(device)

            # -------------------------------------------------------------------
            # Run inference
            # -------------------------------------------------------------------
            t_inference = time.perf_counter()
            with torch.no_grad():
                processed_batch = preprocessor(batch)
                pred_actions = policy.predict_action_chunk(processed_batch)

                # Call postprocessor differently based on processor type
                if obs_state_horizon > 1:
                    # Temporal processors: wrap in dict
                    pred_actions = postprocessor({"action": pred_actions})
                else:
                    # Standard processors: call directly
                    pred_actions = postprocessor(pred_actions)

                if isinstance(pred_actions, dict) and "action" in pred_actions:
                    pred_actions = pred_actions["action"]
                pred_actions = pred_actions[0].cpu().numpy()  # (chunk_size, 10)

            inference_time = time.perf_counter() - t_inference
            print(f"Step {step_count}: Inference {inference_time*1000:.1f}ms | Actions shape: {pred_actions.shape}")

            # -------------------------------------------------------------------
            # Update trajectory plot if enabled
            # -------------------------------------------------------------------
            if plotter is not None:
                plotter.set_trajectory(pred_actions)
                plotter.update()

                # Store 3D trajectory points for camera projection
                trajectory_3d = np.column_stack([
                    plotter.trajectory_x,
                    plotter.trajectory_y,
                    plotter.trajectory_z
                ])

                # Get camera matrix from first RealSense camera
                if camera_matrix is None:
                    for cam in cameras.values():
                        # Try to get intrinsics from RealSense camera
                        if rs is not None and hasattr(cam, 'rs_pipeline') and cam.rs_pipeline is not None:
                            try:
                                # Get active streams and extract intrinsics
                                profile = cam.rs_pipeline.get_active_profile()
                                color_stream = profile.get_stream(rs.stream.color)
                                intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                                camera_matrix = np.array([
                                    [intrinsics.fx, 0, intrinsics.ppx],
                                    [0, intrinsics.fy, intrinsics.ppy],
                                    [0, 0, 1]
                                ])
                                print(f"Debug: Got camera matrix from RealSense: fx={intrinsics.fx}, fy={intrinsics.fy}")
                                break
                            except Exception as e:
                                print(f"Debug: Failed to get intrinsics from RealSense: {e}")
                        elif hasattr(cam, 'intrinsics'):
                            # Fallback for cameras with intrinsics attribute
                            intrinsics = cam.intrinsics
                            camera_matrix = np.array([
                                [intrinsics.fx, 0, intrinsics.ppx],
                                [0, intrinsics.fy, intrinsics.ppy],
                                [0, 0, 1]
                            ])
                            print(f"Debug: Got camera matrix from intrinsics: fx={intrinsics.fx}, fy={intrinsics.fy}")
                            break

            # -------------------------------------------------------------------
            # Show camera view with trajectory overlay (after trajectory is computed)
            # -------------------------------------------------------------------
            if args.cameraview:
                for cam_name, img in camera_images.items():
                    img_display = img.copy()
                    if img_display.ndim == 3 and img_display.shape[2] == 3:
                        # Already HWC RGB, convert to BGR for cv2
                        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

                    # Draw trajectory overlay if available
                    if trajectory_3d is not None and camera_matrix is not None:
                        points_2d = project_points_to_image(trajectory_3d, camera_matrix)
                        print(f"Debug: trajectory_3d shape={trajectory_3d.shape}, points_2d shape={points_2d.shape}")
                        print(f"Debug: First 3D point: {trajectory_3d[0]}, First 2D point: {points_2d[0]}")
                        print(f"Debug: Last 3D point: {trajectory_3d[-1]}, Last 2D point: {points_2d[-1]}")
                        img_display = draw_trajectory_on_image(img_display, points_2d)
                    else:
                        if trajectory_3d is None:
                            print("Debug: trajectory_3d is None")
                        if camera_matrix is None:
                            print("Debug: camera_matrix is None")

                    cv2.imshow(f"Camera: {cam_name}", img_display)
                cv2.waitKey(1)

            step_count += 1

            # -------------------------------------------------------------------
            # Maintain timing
            # -------------------------------------------------------------------
            elapsed = time.perf_counter() - t0
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Error during control loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close trajectory plot
        if plotter is not None:
            plotter.close()

        # Close camera view windows
        if args.cameraview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass  # Ignore cv2 errors in headless mode

        # Disconnect cameras
        logger.info("Disconnecting cameras...")
        for cam_name, camera in cameras.items():
            camera.disconnect()
        logger.info("Done!")


if __name__ == "__main__":
    main()

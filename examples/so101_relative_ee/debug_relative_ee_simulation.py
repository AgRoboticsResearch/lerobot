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
Debug script for RelativeEE ACT policy with placo simulation.

This script loads a trained RelativeEE policy and simulates robot execution
using placo. It follows the same action chaining logic as deploy_relative_ee_so101.py:
- Each action in a chunk is relative to chunk_base_pose (NOT chained sequentially)
- After n_action_steps, get new observation and predict new chunk
- Uses dataset images + simulation joint observations (closed-loop)

Features:
- Loads specific episodes from a dataset
- Runs chunk-based inference with action execution
- Generates frame-by-frame visualizations showing:
  - Observation image from dataset
  - 3D EE trajectory with history, prediction, and simulation EE position
- Optionally compiles frames to MP4 video

Usage:
    python debug_relative_ee_simulation.py \
        --dataset_repo_id red_strawberry_picking_260119_merged_ee \
        --dataset_root /path/to/dataset \
        --pretrained_path outputs/train/my_model/checkpoints/020000/pretrained_model \
        --episode_indices 0 \
        --n_action_steps 10 \
        --save_video

The key difference from debug_relative_ee_animation.py:
- Actions are EXECUTED in simulation (not just visualized)
- Joint observations come from simulation (closed-loop)
- Shows where robot actually is vs where policy wants to go
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.temporal_wrapper import TemporalACTWrapper
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.relative_ee_dataset import (
    RelativeEEDataset,
    pose_to_mat,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.utils import init_logging
from lerobot.processor import (
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TemporalNormalizeProcessor,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.robots.so101_follower.relative_ee_processor import pose10d_to_mat
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, TransitionKey
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.relative_ee_processor import (
    Relative10DAccumulatedToAbsoluteEE,
)

# Motor names for SO101
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Default RESET pose (starting position)
RESET_POSE_DEG = np.array([
    -8.00,    # shoulder_pan
    -62.73,   # shoulder_lift
    65.05,    # elbow_flex
    0.86,     # wrist_flex
    -2.55,    # wrist_roll
    88.91,    # gripper
])


def make_act_pre_post_processors_with_temporal(
    config: ACTConfig,
    dataset_stats: dict | None = None,
    obs_state_horizon: int = 2,
) -> tuple[
    PolicyProcessorPipeline[dict, dict],
    PolicyProcessorPipeline,
]:
    """Create ACT processors with temporal observation handling (UMI-style).

    This MUST match the processors used during training, otherwise
    normalization will be inconsistent and predictions will be wrong.
    """
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


class SimulatedSO101Robot:
    """Simulated SO101 robot using placo for kinematics and visualization."""

    def __init__(self, urdf_path: str, motor_names: list[str], enable_viz: bool = False):
        try:
            import placo
        except ImportError:
            raise ImportError(
                "placo is required for simulation. "
                "Install with: pip install placo"
            )

        self.urdf_path = urdf_path
        self.motor_names = motor_names
        self.enable_viz = enable_viz

        # Setup placo robot
        self.robot = placo.RobotWrapper(urdf_path, placo.Flags.ignore_collisions)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)

        # Setup visualization if enabled
        self.viz = None
        if enable_viz:
            from placo_utils.visualization import robot_viz
            self.viz = robot_viz(self.robot)

        # Current joint state (in degrees)
        self.current_joints = RESET_POSE_DEG.copy()

        # Initialize robot state
        self._update_robot_from_joints()

    def connect(self, calibrate: bool = False):
        """Simulated connection - just initialize."""
        print(f"Simulated robot connected")
        print(f"  URDF: {self.urdf_path}")
        print(f"  Motors: {self.motor_names}")
        if self.enable_viz:
            print(f"  Visualization: ENABLED")
            print(f"\nOpen http://127.0.0.1:7000/static/ in your browser to see the visualization!")

    def get_observation(self) -> dict:
        """
        Get current observation (simulated).

        Returns dict with keys like "shoulder_pan.pos", etc.
        Values are in degrees.
        """
        return {f"{name}.pos": self.current_joints[i] for i, name in enumerate(self.motor_names)}

    def send_action(self, action: dict):
        """
        Send action to robot (simulated).

        Args:
            action: Dict with keys like "shoulder_pan.pos", etc.
                    Values are in degrees.
        """
        # Update current joint state
        for i, name in enumerate(self.motor_names):
            if f"{name}.pos" in action:
                self.current_joints[i] = float(action[f"{name}.pos"])

        # Update robot state
        self._update_robot_from_joints()

    def _update_robot_from_joints(self):
        """Update placo robot from current joint state."""
        # Convert degrees to radians for placo
        joints_rad = np.deg2rad(self.current_joints)

        for i, name in enumerate(self.motor_names):
            self.robot.set_joint(name, joints_rad[i])

        self.robot.update_kinematics()

        # Update visualization if enabled
        if self.viz is not None:
            self.viz.display(self.robot.state.q)

    def display_ee_frame(self, frame_name: str = "gripper_frame_link"):
        """Display the end-effector frame in visualization."""
        if self.viz is not None:
            from placo_utils.visualization import robot_frame_viz
            robot_frame_viz(self.robot, frame_name)

    def disconnect(self):
        """Simulated disconnection."""
        print("Simulated robot disconnected")


def create_observation_batch_with_sim_state(
    dataset: RelativeEEDataset,
    idx: int,
    n_obs_steps: int,
    device: torch.device,
    sim_joints: np.ndarray,
    kinematics: RobotKinematics,
    obs_state_horizon: int = 2,
) -> dict[str, torch.Tensor]:
    """
    Create observation batch for policy inference using simulation joint state.

    For RelativeEE with temporal observations:
    - observation.state shape: (T, 10) where T=obs_state_horizon
      - State is computed from SIMULATION joints (closed-loop)
    - observation.images.camera shape: (T, C, H, W)
      - Images come from dataset

    Args:
        dataset: RelativeEEDataset instance
        idx: Current sample index
        n_obs_steps: Number of observation steps
        device: Torch device
        sim_joints: Current joint positions from simulation (degrees)
        kinematics: RobotKinematics instance for FK
        obs_state_horizon: Temporal observation horizon

    Returns:
        Batch dictionary with observation data
    """
    # Get image observation from dataset
    sample = dataset[idx]

    batch = {}

    # Compute EE pose from simulation joints (closed-loop state)
    # Current observation is identity (relative to itself)
    current_obs = np.array([
        0.0, 0.0, 0.0,           # position (identity)
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # rot6d (identity rotation)
        sim_joints[-1] / 100.0,  # gripper (convert to [0,1])
    ], dtype=np.float32)

    # Create temporal observations - fill with current obs
    # (In closed-loop deployment, we'd use actual history buffer)
    obs_state = np.stack([current_obs] * obs_state_horizon, axis=0)  # (obs_state_horizon, 10)

    batch["observation.state"] = torch.from_numpy(obs_state).unsqueeze(0).to(device)

    # Add images from dataset
    for key, value in sample.items():
        if key.startswith("observation.images") and isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)

    return batch


def compute_predicted_ee_trajectory(
    relative_actions: np.ndarray,
    chunk_base_pose: np.ndarray,
) -> np.ndarray:
    """
    Compute absolute EE positions from relative actions.

    In RelativeEE with UMI-style action chaining:
    - Each action in the chunk is a transform from chunk_base_pose
    - Actions are NOT sequentially chained

    Args:
        relative_actions: (T, 10) relative actions (pos 3D + rot 6D + gripper 1D)
        chunk_base_pose: (4, 4) base transformation matrix for this chunk

    Returns:
        (T, 3) array of absolute EE positions for each predicted timestep
    """
    T = relative_actions.shape[0]
    ee_positions = []

    for i in range(T):
        # Get relative transform for this timestep
        rel_T = pose10d_to_mat(relative_actions[i, :9])

        # Apply relative transform from base pose: T_abs = T_base @ T_rel
        abs_T = chunk_base_pose @ rel_T
        ee_positions.append(abs_T[:3, 3].copy())

    return np.array(ee_positions)


def plot_frame_with_ee_trajectory(
    obs_image: np.ndarray,
    ee_history: list[np.ndarray],
    pred_ee_positions: np.ndarray,
    current_ee_position: np.ndarray,
    sim_ee_position: np.ndarray,
    output_path: str | Path,
    episode_idx: int,
    frame_idx: int,
    axis_limits: dict[str, tuple[float, float]] | None = None,
    gt_ee_positions: np.ndarray = None,
    chunk_base_pose: np.ndarray = None,
):
    """
    Plot a single frame with observation image and EE trajectory.

    Creates a 2-subplot figure:
    - Left: Current observation image
    - Right: 3D EE trajectory with history, prediction, ground truth, and simulation EE

    Args:
        obs_image: Observation image (C,H,W or H,W,C or T,C,H,W)
        ee_history: List of historical EE positions from episode start
        pred_ee_positions: Predicted EE positions for chunk (chunk_size, 3)
        current_ee_position: Current EE position from dataset (3,)
        sim_ee_position: Current EE position from simulation (3,)
        output_path: Path to save the plot
        episode_idx: Episode index for title
        frame_idx: Frame index for title
        axis_limits: Optional dict with 'x', 'y', 'z' tuples for fixed axis limits
        gt_ee_positions: Ground truth future EE positions (chunk_size, 3)
        chunk_base_pose: Chunk base pose position (3,) for visualization
    """
    fig = plt.figure(figsize=(16, 8))

    # ========================================================================
    # Left subplot: Observation image
    # ========================================================================
    ax1 = fig.add_subplot(1, 2, 1)

    # Handle different image shapes
    if obs_image.ndim == 4:  # (T, C, H, W)
        img = obs_image[-1]  # Use last timestep (current)
    else:
        img = obs_image

    # Convert from (C, H, W) to (H, W, C) if needed
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

    ax1.imshow(img)
    ax1.set_title(f"Episode {episode_idx}, Frame {frame_idx}", fontsize=14)
    ax1.axis("off")

    # ========================================================================
    # Right subplot: 3D EE trajectory
    # ========================================================================
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Convert history to array
    if ee_history:
        ee_hist_array = np.array(ee_history)
    else:
        ee_hist_array = np.array([[0, 0, 0]])

    # Plot world frame origin
    ax2.scatter([0], [0], [0], c="black", marker="s", s=100, label="World Origin", zorder=10)

    # Plot historical EE trajectory (from frame 0 to current) as solid blue line
    ax2.plot(
        ee_hist_array[:, 0],
        ee_hist_array[:, 1],
        ee_hist_array[:, 2],
        "b-", linewidth=2, label="History", alpha=0.8
    )

    # Plot current EE position (from dataset) as blue circle marker
    ax2.scatter(
        [current_ee_position[0]],
        [current_ee_position[1]],
        [current_ee_position[2]],
        c="blue", marker="o", s=150, edgecolors="black",
        label="Dataset EE", zorder=10
    )

    # Plot simulation EE position (where robot actually is) as cyan star
    ax2.scatter(
        [sim_ee_position[0]],
        [sim_ee_position[1]],
        [sim_ee_position[2]],
        c="cyan", marker="*", s=200, edgecolors="black",
        label="Simulation EE", zorder=11
    )

    # Plot chunk base pose if provided
    if chunk_base_pose is not None:
        ax2.scatter(
            [chunk_base_pose[0]],
            [chunk_base_pose[1]],
            [chunk_base_pose[2]],
            c="orange", marker="s", s=100,
            label="Chunk Base", zorder=9
        )

    # Plot predicted trajectory as red dashed line
    if pred_ee_positions is not None and len(pred_ee_positions) > 0:
        # Start prediction from chunk base pose (or current if not provided)
        start_pos = chunk_base_pose if chunk_base_pose is not None else current_ee_position
        pred_full = np.vstack([start_pos.reshape(1, 3), pred_ee_positions])
        ax2.plot(
            pred_full[:, 0],
            pred_full[:, 1],
            pred_full[:, 2],
            "r--", linewidth=2, label="Prediction", alpha=0.8
        )

        # Plot predicted EE positions as red x markers
        ax2.scatter(
            pred_ee_positions[:, 0],
            pred_ee_positions[:, 1],
            pred_ee_positions[:, 2],
            c="red", marker="x", s=50, label="Pred Steps", alpha=0.8
        )

    # Plot ground truth future trajectory as green dotted line
    if gt_ee_positions is not None and len(gt_ee_positions) > 0:
        # Start GT from current EE position
        gt_full = np.vstack([current_ee_position.reshape(1, 3), gt_ee_positions])
        ax2.plot(
            gt_full[:, 0],
            gt_full[:, 1],
            gt_full[:, 2],
            "g:", linewidth=2, label="Ground Truth", alpha=0.8
        )

        # Plot GT EE positions as green circles
        ax2.scatter(
            gt_ee_positions[:, 0],
            gt_ee_positions[:, 1],
            gt_ee_positions[:, 2],
            c="green", marker="o", s=30, label="GT Steps", alpha=0.6
        )

    ax2.set_xlabel("X (m)", fontsize=10)
    ax2.set_ylabel("Y (m)", fontsize=10)
    ax2.set_zlabel("Z (m)", fontsize=10)
    ax2.set_title(f"End-Effector Trajectory (History: {len(ee_history)} frames)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Set axis limits - use provided limits or compute from data
    if axis_limits is not None:
        ax2.set_xlim(axis_limits['x'])
        ax2.set_ylim(axis_limits['y'])
        ax2.set_zlim(axis_limits['z'])
    else:
        # Compute limits from all data
        all_pos = np.vstack([ee_hist_array, current_ee_position.reshape(1, 3), sim_ee_position.reshape(1, 3)])
        if pred_ee_positions is not None:
            all_pos = np.vstack([all_pos, pred_ee_positions])
        if gt_ee_positions is not None:
            all_pos = np.vstack([all_pos, gt_ee_positions])
        if chunk_base_pose is not None:
            all_pos = np.vstack([all_pos, chunk_base_pose.reshape(1, 3)])

        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
        z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()

        x_pad = max((x_max - x_min) * 0.15, 0.02)
        y_pad = max((y_max - y_min) * 0.15, 0.02)
        z_pad = max((z_max - z_min) * 0.15, 0.02)

        ax2.set_xlim(x_min - x_pad, x_max + x_pad)
        ax2.set_ylim(y_min - y_pad, y_max + y_pad)
        ax2.set_zlim(z_min - z_pad, z_max + z_pad)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_episode_axis_limits(
    dataset: RelativeEEDataset,
    episode_idx: int,
) -> dict[str, tuple[float, float]]:
    """
    Compute axis limits for an episode by processing all ground truth actions.

    Args:
        dataset: RelativeEEDataset instance
        episode_idx: Episode index

    Returns:
        Dict with 'x', 'y', 'z' tuples for axis limits
    """
    # Get episode info from dataset
    ep_info = dataset.meta.episodes[episode_idx]
    ep_length = ep_info["length"]

    # Find start index in dataset
    start_idx = 0
    for i in range(episode_idx):
        start_idx += dataset.meta.episodes[i]["length"]

    all_ee_positions = []

    # Process each frame to get EE positions
    for i in range(ep_length):
        idx = start_idx + i

        # Get ABSOLUTE EE position from parent LeRobotDataset
        abs_sample = LeRobotDataset.__getitem__(dataset, idx)
        abs_state = abs_sample['observation.state'].cpu().numpy()  # (7,) - absolute pose

        # Convert absolute state to 4x4 matrix
        abs_pose_6d = abs_state[:6]  # position (3) + rotation axis-angle (3)
        current_ee_pose = pose_to_mat(abs_pose_6d)
        all_ee_positions.append(current_ee_pose[:3, 3].copy())

    all_ee = np.array(all_ee_positions)

    x_min, x_max = all_ee[:, 0].min(), all_ee[:, 0].max()
    y_min, y_max = all_ee[:, 1].min(), all_ee[:, 1].max()
    z_min, z_max = all_ee[:, 2].min(), all_ee[:, 2].max()

    # Add padding
    x_pad = max((x_max - x_min) * 0.15, 0.02)
    y_pad = max((y_max - y_min) * 0.15, 0.02)
    z_pad = max((z_max - z_min) * 0.15, 0.02)

    return {
        'x': (x_min - x_pad, x_max + x_pad),
        'y': (y_min - y_pad, y_max + y_pad),
        'z': (z_min - z_pad, z_max + z_pad),
    }


def create_video_from_frames(
    frames_dir: Path,
    output_path: Path,
    fps: int = 30,
):
    """Create MP4 video from frames."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError(
            "imageio is required for video generation. "
            "Install with: pip install imageio imageio-ffmpeg"
        )

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    frames = []
    for frame_file in frame_files:
        frames.append(imageio.imread(frame_file))

    # Ensure uniform frame sizes
    from PIL import Image
    target_size = frames[0].shape[:2][::-1]
    uniform_frames = []
    for frame in frames:
        if frame.shape[:2][::-1] != target_size:
            img = Image.fromarray(frame)
            img = img.resize(target_size, Image.LANCZOS)
            uniform_frames.append(np.array(img))
        else:
            uniform_frames.append(frame)

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
    for frame in uniform_frames:
        writer.append_data(frame)
    writer.close()

    print(f"Video saved to: {output_path}")


def main():
    init_logging()
    logger = logging.getLogger("debug_relative_ee_simulation")

    import argparse
    parser = argparse.ArgumentParser(
        description="Debug script for RelativeEE ACT policy with placo simulation"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., red_strawberry_picking_260119_merged_ee)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Local path to dataset (if not provided, uses HF hub)",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory or training output",
    )
    parser.add_argument(
        "--episode_indices",
        type=int,
        nargs="+",
        default=[0],
        help="Episode indices to visualize (default: [0])",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames per episode to process (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/debug/simulation_relative_ee",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Compile frames to MP4 video",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="FPS for output video (default: 30)",
    )
    parser.add_argument(
        "--fixed_axis_limits",
        action="store_true",
        help="Use fixed axis limits computed from full episode",
    )
    parser.add_argument(
        "--obs_state_horizon",
        type=int,
        default=None,
        help="Observation state horizon (auto-detected from policy if not specified)",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=10,
        help="Number of actions to execute from each chunk prediction (default: 10)",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="urdf/Simulation/SO101/so101_new_calib.urdf",
        help="Path to robot URDF file for IK (default: urdf/Simulation/SO101/so101_new_calib.urdf)",
    )
    parser.add_argument(
        "--target_frame",
        type=str,
        default="gripper_frame_link",
        help="Name of end-effector frame in URDF (default: gripper_frame_link)",
    )
    parser.add_argument(
        "--reset_pose",
        type=float,
        nargs=6,
        default=list(RESET_POSE_DEG),
        metavar=("PAN", "LIFT", "ELBOW", "FLEX", "ROLL", "GRIPPER"),
        help="Reset pose in degrees for simulation (default: -8.00 -62.73 65.05 0.86 -2.55 88.91)",
    )
    parser.add_argument(
        "--ee_bounds_min",
        type=float,
        nargs=3,
        default=[-0.5, -0.5, 0.0],
        help="EE position bounds minimum (x, y, z) in meters",
    )
    parser.add_argument(
        "--ee_bounds_max",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.4],
        help="EE position bounds maximum (x, y, z) in meters",
    )
    parser.add_argument(
        "--max_ee_step_m",
        type=float,
        default=0.05,
        help="Maximum EE step size in meters (safety)",
    )
    parser.add_argument(
        "--gripper_lower",
        type=float,
        default=0.0,
        help="Gripper lower bound in degrees",
    )
    parser.add_argument(
        "--gripper_upper",
        type=float,
        default=100.0,
        help="Gripper upper bound in degrees",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable placo 3D visualization at http://127.0.0.1:7000/static/",
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.0,
        help="Delay in seconds between action steps (for visualization, default: 0.0 = no delay)",
    )
    parser.add_argument(
        "--keep_viz_alive",
        action="store_true",
        help="Keep script running after processing to view visualization (press Ctrl+C to exit)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Extract job name from pretrained_path
    job_name = "unknown_job"
    pretrained_path_obj = Path(args.pretrained_path)
    if "outputs" in pretrained_path_obj.parts:
        outputs_idx = pretrained_path_obj.parts.index("outputs")
        if outputs_idx + 2 < len(pretrained_path_obj.parts):
            job_name = pretrained_path_obj.parts[outputs_idx + 2]
            logger.info(f"Job name: {job_name}")

    # Create output directory
    output_dir = Path(args.output_dir) / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    reset_pose = np.array(args.reset_pose, dtype=np.float64)

    # ========================================================================
    # Load Policy
    # ========================================================================
    logger.info("Loading trained policy...")
    logger.info(f"  From: {args.pretrained_path}")

    policy = ACTPolicy.from_pretrained(args.pretrained_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)

    # Wrap with TemporalACTWrapper
    policy_obs_state_horizon = getattr(policy.config, 'obs_state_horizon', 1)
    obs_state_horizon = args.obs_state_horizon if args.obs_state_horizon is not None else policy_obs_state_horizon

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

    logger.info(f"Using obs_state_horizon={obs_state_horizon}")

    # Load temporal processors
    logger.info("Loading processors from pretrained model...")
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

    chunk_size = policy.config.chunk_size

    logger.info("Policy loaded successfully")
    logger.info(f"  Chunk size: {chunk_size}")
    logger.info(f"  n_obs_steps: {policy.config.n_obs_steps}")
    logger.info(f"  n_action_steps: {args.n_action_steps}")

    # ========================================================================
    # Load Dataset
    # ========================================================================
    logger.info("Loading dataset...")
    logger.info(f"  Repo ID: {args.dataset_repo_id}")
    logger.info(f"  Local root: {args.dataset_root}")

    fps = getattr(policy.config, 'fps', 30)
    action_delta_timestamps = [i / fps for i in range(chunk_size)]
    delta_timestamps = {"action": action_delta_timestamps}

    dataset = RelativeEEDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        obs_state_horizon=obs_state_horizon,
        delta_timestamps=delta_timestamps,
        compute_stats=False,
    )

    logger.info(f"Dataset loaded successfully!")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Total episodes: {len(dataset.meta.episodes)}")
    logger.info(f"  Processing episodes: {args.episode_indices}")

    # ========================================================================
    # Initialize Kinematics
    # ========================================================================
    logger.info("Initializing kinematics solver...")

    urdf_path = Path(args.urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found at {urdf_path}")

    # For RobotKinematics, we need the actual URDF file path
    if urdf_path.is_dir():
        urdf_file = urdf_path / "robot.urdf"
    else:
        urdf_file = urdf_path

    kinematics = RobotKinematics(
        urdf_path=str(urdf_file),
        target_frame_name=args.target_frame,
        joint_names=MOTOR_NAMES,
    )
    logger.info(f"URDF loaded: {urdf_file}")

    # ========================================================================
    # Build Processor Pipeline
    # ========================================================================
    logger.info("Building processor pipeline...")

    ee_to_joints_pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            Relative10DAccumulatedToAbsoluteEE(
                gripper_lower_deg=args.gripper_lower,
                gripper_upper_deg=args.gripper_upper,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": args.ee_bounds_min, "max": args.ee_bounds_max},
                max_ee_step_m=args.max_ee_step_m,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=MOTOR_NAMES,
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    logger.info("Pipeline built")

    # ========================================================================
    # Initialize Simulation Robot
    # ========================================================================
    logger.info("Initializing simulated SO101 robot...")

    robot = SimulatedSO101Robot(str(urdf_path), MOTOR_NAMES, enable_viz=args.visualize)
    robot.connect(calibrate=False)

    # Set robot to reset pose
    robot.current_joints = reset_pose.copy()
    robot.send_action({f"{name}.pos": val for name, val in zip(MOTOR_NAMES, reset_pose)})

    sim_joints = reset_pose.copy()
    logger.info(f"Robot initialized at reset pose: {reset_pose}")

    # ========================================================================
    # Process each episode
    # ========================================================================
    for ep_idx in args.episode_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Episode {ep_idx}")
        logger.info(f"{'='*60}")

        # Get episode bounds
        ep_info = dataset.meta.episodes[ep_idx]
        ep_length = ep_info["length"]

        # Find start index in dataset
        start_idx = 0
        for i in range(ep_idx):
            start_idx += dataset.meta.episodes[i]["length"]
        end_idx = start_idx + ep_length

        logger.info(f"  Episode length: {ep_length} frames")
        logger.info(f"  Dataset range: [{start_idx}, {end_idx})")

        # Apply max_frames limit
        if args.max_frames is not None:
            ep_end = min(end_idx, start_idx + args.max_frames)
            logger.info(f"  Limited to {ep_end - start_idx} frames")
        else:
            ep_end = end_idx

        # Compute axis limits for this episode if requested
        axis_limits = None
        if args.fixed_axis_limits:
            logger.info("  Computing axis limits from full episode...")
            axis_limits = compute_episode_axis_limits(dataset, ep_idx)
            logger.info(f"  Axis limits: X={axis_limits['x']}, Y={axis_limits['y']}, Z={axis_limits['z']}")

        # Create episode output directory
        ep_output_dir = output_dir / f"episode_{ep_idx}"
        ep_output_dir.mkdir(parents=True, exist_ok=True)

        # Reset simulation to reset pose
        sim_joints = reset_pose.copy()
        robot.current_joints = reset_pose.copy()
        robot.send_action({f"{name}.pos": val for name, val in zip(MOTOR_NAMES, reset_pose)})
        ee_to_joints_pipeline.reset()

        # Get initial EE pose from simulation
        chunk_base_pose = kinematics.forward_kinematics(sim_joints)
        sim_start_ee = chunk_base_pose[:3, 3].copy()

        # Get dataset start frame EE position
        abs_sample = LeRobotDataset.__getitem__(dataset, start_idx)
        abs_state = abs_sample['observation.state'].cpu().numpy()
        abs_pose_6d = abs_state[:6]
        dataset_start_ee_pose = pose_to_mat(abs_pose_6d)
        dataset_start_ee = dataset_start_ee_pose[:3, 3].copy()

        # Compute alignment transform: dataset -> simulation
        # T_align maps dataset positions to simulation frame
        # For position only: sim_pos = dataset_pos + offset
        ee_offset = sim_start_ee - dataset_start_ee

        logger.info(f"  Simulation start EE: {sim_start_ee}")
        logger.info(f"  Dataset start EE: {dataset_start_ee}")
        logger.info(f"  Alignment offset: {ee_offset}")

        # Run chunk-based inference with action execution
        ee_history = []  # Cumulative EE positions (from simulation)
        chunk_count = 0

        frame_idx = start_idx
        while frame_idx < ep_end:
            # Get observation image from dataset at current frame
            sample = dataset[frame_idx]

            # Get observation image for visualization
            obs_img = None
            for key in sample.keys():
                if key.startswith("observation.images"):
                    obs_img = sample[key].cpu().numpy()
                    break

            if obs_img is None:
                logger.warning(f"No observation image found for frame {frame_idx}")
                frame_idx += 1
                continue

            # Get ground truth EE position from dataset (for reference)
            abs_sample = LeRobotDataset.__getitem__(dataset, frame_idx)
            abs_state = abs_sample['observation.state'].cpu().numpy()
            abs_pose_6d = abs_state[:6]
            dataset_ee_pose = pose_to_mat(abs_pose_6d)
            dataset_ee_position = dataset_ee_pose[:3, 3].copy()

            # Apply alignment offset to map dataset frame to simulation frame
            aligned_dataset_ee_position = dataset_ee_position + ee_offset

            # Get GT action (for reference)
            gt_actions = sample['action'].cpu().numpy()  # (chunk_size, 10)

            # Set chunk base pose from SIMULATION joints (closed-loop)
            chunk_base_pose = kinematics.forward_kinematics(sim_joints)

            # Create observation batch with simulation state
            batch = create_observation_batch_with_sim_state(
                dataset=dataset,
                idx=frame_idx,
                n_obs_steps=policy.config.n_obs_steps,
                device=device,
                sim_joints=sim_joints,
                kinematics=kinematics,
                obs_state_horizon=obs_state_horizon,
            )

            # Predict action chunk
            with torch.no_grad():
                processed_batch = preprocessor(batch)
                pred_actions = policy.predict_action_chunk(processed_batch)
                pred_actions = postprocessor({"action": pred_actions})

            if isinstance(pred_actions, dict) and "action" in pred_actions:
                pred_actions = pred_actions["action"]
            pred_actions = pred_actions[0].cpu().numpy()  # (chunk_size, 10)

            # Compute predicted EE trajectory from chunk base pose
            pred_ee_positions = compute_predicted_ee_trajectory(pred_actions, chunk_base_pose)

            # Compute ground truth EE trajectory from ALIGNED dataset pose
            # Create aligned dataset EE pose (same orientation, shifted position)
            aligned_dataset_ee_pose = dataset_ee_pose.copy()
            aligned_dataset_ee_pose[:3, 3] = aligned_dataset_ee_position
            gt_ee_positions = compute_predicted_ee_trajectory(gt_actions, aligned_dataset_ee_pose)

            # Get current simulation EE position
            sim_ee_pose = kinematics.forward_kinematics(sim_joints)
            sim_ee_position = sim_ee_pose[:3, 3].copy()

            # Add simulation EE position to history
            ee_history.append(sim_ee_position.copy())

            # Generate and save plot BEFORE executing actions
            frame_output = ep_output_dir / f"frame_{frame_idx:04d}.jpg"
            plot_frame_with_ee_trajectory(
                obs_image=obs_img,
                ee_history=ee_history,
                pred_ee_positions=pred_ee_positions,
                current_ee_position=aligned_dataset_ee_position,  # Use aligned position
                sim_ee_position=sim_ee_position,
                output_path=frame_output,
                episode_idx=ep_idx,
                frame_idx=frame_idx,
                axis_limits=axis_limits,
                gt_ee_positions=gt_ee_positions,
                chunk_base_pose=chunk_base_pose[:3, 3],
            )

            # Execute n_action_steps from the predicted chunk
            logger.info(f"Chunk {chunk_count} at frame {frame_idx}: executing {args.n_action_steps} actions")
            for action_i in range(min(args.n_action_steps, len(pred_actions))):
                rel_action = pred_actions[action_i]
                rel_T = pose10d_to_mat(rel_action[:9])

                # Target EE = chunk_base_pose @ rel_T (NOT chained)
                target_ee_pose = chunk_base_pose @ rel_T

                # Solve IK
                joints = kinematics.inverse_kinematics(sim_joints, target_ee_pose)

                # Send to simulation
                action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, joints)}
                robot.send_action(action)

                # Display EE frame in visualization
                if args.visualize:
                    robot.display_ee_frame("gripper_frame_link")

                # Update sim_joints for next iteration (closed-loop)
                sim_joints = joints

                # Add delay for visualization
                if args.step_delay > 0:
                    time.sleep(args.step_delay)

            # Advance dataset index by n_action_steps
            frame_idx += args.n_action_steps
            chunk_count += 1

            if chunk_count % 5 == 0:
                logger.info(f"  Processed {chunk_count} chunks, frame {frame_idx}/{ep_end}")

        logger.info(f"  Episode {ep_idx} complete: {chunk_count} chunks, {len(ee_history)} frames saved to {ep_output_dir}")

        # Create video if requested
        if args.save_video:
            logger.info("  Creating video...")
            video_output = output_dir / f"episode_{ep_idx}.mp4"
            create_video_from_frames(ep_output_dir, video_output, fps=args.video_fps)

    # Keep visualization alive if requested
    if args.visualize and args.keep_viz_alive:
        logger.info("\nVisualization server running at http://127.0.0.1:7000/static/")
        logger.info("Press Ctrl+C to exit...")
        try:
            # Keep displaying the current robot state
            while True:
                robot.display_ee_frame("gripper_frame_link")
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("\nExiting...")

    # Disconnect robot
    robot.disconnect()
    logger.info("\nDone!")


if __name__ == "__main__":
    main()

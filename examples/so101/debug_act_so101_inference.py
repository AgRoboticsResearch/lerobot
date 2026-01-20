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
Debug script for standard ACT policy (joint control) inference.

This script loads a trained ACT policy with joint actions and compares its
predictions against ground truth from the dataset. Useful for debugging model
behavior without needing to connect to the robot.

Features:
- Loads dataset with LeRobotDataset
- Samples random frames from the dataset
- Runs inference on each sample
- Computes joint position errors
- Visualizes predicted vs ground truth joint trajectories
- Saves results for further analysis

Usage:
    python debug_act_so101_inference.py \
        --dataset_repo_id red_strawberry_picking_260119_merged \
        --dataset_root /path/to/dataset \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --num_samples 10 \
        --output_dir ./debug_output
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.utils import init_logging


# Joint names for SO101
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def compute_error_metrics(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
) -> dict[str, Any]:
    """
    Compute error metrics between predicted and ground truth actions.

    Args:
        pred_actions: (T, 6) predicted actions (joint positions)
        gt_actions: (T, 6) ground truth actions

    Returns:
        Dictionary with error metrics
    """
    T = min(pred_actions.shape[0], gt_actions.shape[0])
    pred = pred_actions[:T]
    gt = gt_actions[:T]

    # L2 error per timestep (across all joints)
    l2_errors = np.linalg.norm(pred - gt, axis=1)

    # Per-joint absolute errors
    per_joint_errors = np.abs(pred - gt)  # (T, 6)

    return {
        "l2_errors": l2_errors,
        "per_joint_errors": per_joint_errors,
        "mean_l2_error": float(np.mean(l2_errors)),
        "max_l2_error": float(np.max(l2_errors)),
        "final_l2_error": float(l2_errors[-1]),
        "mean_per_joint_errors": np.mean(per_joint_errors, axis=0),
    }


def plot_joint_trajectories(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    output_path: str,
    sample_idx: int = 0,
):
    """
    Plot predicted vs ground truth joint trajectories.

    Creates 6 subplots (one per joint) showing predicted and ground truth
    trajectories over time.

    Args:
        pred_actions: (T, 6) predicted joint positions
        gt_actions: (T, 6) ground truth joint positions
        output_path: Path to save the plot
        sample_idx: Sample index for title
    """
    T = min(pred_actions.shape[0], gt_actions.shape[0])
    pred = pred_actions[:T]
    gt = gt_actions[:T]

    # Create figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    time_steps = np.arange(T)

    for i in range(6):
        ax = axes[i]
        ax.plot(time_steps, gt[:, i], "b-", linewidth=2, label="Ground Truth", alpha=0.7)
        ax.plot(time_steps, pred[:, i], "r--", linewidth=2, label="Predicted", alpha=0.7)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Joint Position (degrees)")
        ax.set_title(f"{JOINT_NAMES[i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sample {sample_idx}: Joint Trajectories Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved to: {output_path}")


def plot_error_analysis(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    output_path: str,
    sample_idx: int = 0,
):
    """
    Plot error analysis for joint predictions.

    Creates a 2x2 plot showing:
    - L2 error over time
    - Per-joint mean absolute error
    - Per-joint error distribution
    - Cumulative error

    Args:
        pred_actions: (T, 6) predicted joint positions
        gt_actions: (T, 6) ground truth joint positions
        output_path: Path to save the plot
        sample_idx: Sample index for title
    """
    T = min(pred_actions.shape[0], gt_actions.shape[0])
    pred = pred_actions[:T]
    gt = gt_actions[:T]

    metrics = compute_error_metrics(pred, gt)

    fig = plt.figure(figsize=(14, 10))

    # L2 error over time
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(metrics["l2_errors"], "r-", linewidth=2)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("L2 Error (degrees)")
    ax1.set_title("L2 Error Over Time")
    ax1.grid(True, alpha=0.3)

    # Per-joint mean absolute error
    ax2 = fig.add_subplot(2, 2, 2)
    mean_errors = metrics["per_joint_errors"].mean(axis=0)
    bars = ax2.bar(JOINT_NAMES, mean_errors, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    ax2.set_ylabel("Mean Absolute Error (degrees)")
    ax2.set_title("Per-Joint Mean Absolute Error")
    ax2.grid(True, axis="y", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # Per-joint error distribution (box plot)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.boxplot([metrics["per_joint_errors"][:, i] for i in range(6)], labels=JOINT_NAMES)
    ax3.set_ylabel("Absolute Error (degrees)")
    ax3.set_title("Per-Joint Error Distribution")
    ax3.grid(True, axis="y", alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

    # Cumulative L2 error
    ax4 = fig.add_subplot(2, 2, 4)
    cumulative_error = np.cumsum(metrics["l2_errors"])
    ax4.plot(cumulative_error, "g-", linewidth=2)
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("Cumulative L2 Error")
    ax4.set_title("Cumulative L2 Error Over Time")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Sample {sample_idx}: Error Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Error analysis plot saved to: {output_path}")


def compute_ee_trajectories(
    joint_actions: np.ndarray,
    kinematics: RobotKinematics,
) -> np.ndarray:
    """
    Compute end-effector positions from joint actions using forward kinematics.

    Args:
        joint_actions: (T, 6) joint positions in degrees
        kinematics: RobotKinematics instance

    Returns:
        (T, 3) array of EE positions
    """
    T = joint_actions.shape[0]
    ee_positions = []

    for i in range(T):
        joints = joint_actions[i].astype(np.float64)  # Convert to float64 for kinematics
        ee_T = kinematics.forward_kinematics(joints)
        ee_positions.append(ee_T[:3, 3].copy())

    return np.array(ee_positions)


def plot_ee_trajectory_3d(
    pred_ee_positions: np.ndarray,
    gt_ee_positions: np.ndarray,
    output_path: str,
    sample_idx: int = 0,
):
    """
    Plot predicted vs ground truth end-effector trajectory in 3D.

    Args:
        pred_ee_positions: (T, 3) predicted EE positions
        gt_ee_positions: (T, 3) ground truth EE positions
        output_path: Path to save the plot
        sample_idx: Sample index for title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot ground truth (blue solid)
    ax.plot(
        gt_ee_positions[:, 0],
        gt_ee_positions[:, 1],
        gt_ee_positions[:, 2],
        "b-", linewidth=3, label="Ground Truth", marker="o", markersize=4, alpha=0.8
    )
    ax.scatter(
        gt_ee_positions[0, 0],
        gt_ee_positions[0, 1],
        gt_ee_positions[0, 2],
        c="blue", s=100, marker="o", edgecolors="black", label="GT Start"
    )
    ax.scatter(
        gt_ee_positions[-1, 0],
        gt_ee_positions[-1, 1],
        gt_ee_positions[-1, 2],
        c="blue", s=100, marker="s", edgecolors="black", label="GT End"
    )

    # Plot predicted (red dashed)
    ax.plot(
        pred_ee_positions[:, 0],
        pred_ee_positions[:, 1],
        pred_ee_positions[:, 2],
        "r--", linewidth=3, label="Predicted", marker="x", markersize=4, alpha=0.8
    )
    ax.scatter(
        pred_ee_positions[0, 0],
        pred_ee_positions[0, 1],
        pred_ee_positions[0, 2],
        c="red", s=100, marker="^", edgecolors="black", label="Pred Start"
    )
    ax.scatter(
        pred_ee_positions[-1, 0],
        pred_ee_positions[-1, 1],
        pred_ee_positions[-1, 2],
        c="red", s=100, marker="*", edgecolors="black", label="Pred End"
    )

    # Add connection lines at intervals to show deviation
    step = max(1, len(pred_ee_positions) // 10)
    for i in range(0, len(pred_ee_positions), step):
        ax.plot(
            [pred_ee_positions[i, 0], gt_ee_positions[i, 0]],
            [pred_ee_positions[i, 1], gt_ee_positions[i, 1]],
            [pred_ee_positions[i, 2], gt_ee_positions[i, 2]],
            "k:", alpha=0.3, linewidth=1
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Sample {sample_idx}: End-Effector Trajectory (via FK)")
    ax.legend(loc="upper right")

    # Set equal aspect ratio and reasonable limits
    all_pos = np.vstack([pred_ee_positions, gt_ee_positions])
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
    z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()

    # Add padding
    x_pad = max((x_max - x_min) * 0.1, 0.01)
    y_pad = max((y_max - y_min) * 0.1, 0.01)
    z_pad = max((z_max - z_min) * 0.1, 0.01)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  EE trajectory plot saved to: {output_path}")


def compute_ee_error_metrics(
    pred_ee_positions: np.ndarray,
    gt_ee_positions: np.ndarray,
) -> dict[str, float]:
    """
    Compute EE position error metrics.

    Args:
        pred_ee_positions: (T, 3) predicted EE positions
        gt_ee_positions: (T, 3) ground truth EE positions

    Returns:
        Dictionary with error metrics
    """
    # Position error (L2 norm in meters)
    pos_errors = np.linalg.norm(pred_ee_positions - gt_ee_positions, axis=1)

    return {
        "mean_ee_error_m": float(np.mean(pos_errors)),
        "max_ee_error_m": float(np.max(pos_errors)),
        "final_ee_error_m": float(pos_errors[-1]),
        "mean_ee_error_mm": float(np.mean(pos_errors) * 1000),
        "max_ee_error_mm": float(np.max(pos_errors) * 1000),
    }


def create_observation_batch(
    dataset: LeRobotDataset,
    idx: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Create observation batch for policy inference.

    Args:
        dataset: LeRobotDataset instance
        idx: Current sample index
        device: Torch device

    Returns:
        Batch dictionary with observation data
    """
    sample = dataset[idx]

    # State is (6,) - joint positions
    obs_state = sample["observation.state"]

    batch = {
        "observation.state": obs_state.unsqueeze(0).to(device),  # (1, 6)
    }

    # Add camera images if available
    for key in list(sample.keys()):
        if key.startswith("observation.images"):
            img = sample[key]
            if isinstance(img, torch.Tensor):
                batch[key] = img.unsqueeze(0).to(device)  # (1, C, H, W)

    return batch


def main():
    init_logging()
    logger = logging.getLogger("debug_act_so101_inference")

    import argparse
    parser = argparse.ArgumentParser(
        description="Debug script for ACT policy (joint control) inference"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., red_strawberry_picking_260119_merged)",
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
        "--num_samples",
        type=int,
        default=10,
        help="Number of random samples to test (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./debug_act_so101_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Save all predictions and ground truth to disk",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="./urdf/Simulation/SO101/so101_new_calib.urdf",
        help="Path to URDF file for FK (default: ./urdf/Simulation/SO101/so101_new_calib.urdf)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load Policy first (to get chunk_size for delta_timestamps)
    # ========================================================================
    logger.info("Loading trained policy first to get config...")
    logger.info(f"  From: {args.pretrained_path}")

    policy = ACTPolicy.from_pretrained(args.pretrained_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)

    # Create processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={},
    )

    logger.info("Policy loaded successfully")
    logger.info(f"  Chunk size: {policy.config.chunk_size}")
    logger.info(f"  n_obs_steps: {policy.config.n_obs_steps}")
    logger.info(f"  n_action_steps: {policy.config.n_action_steps}")

    # ========================================================================
    # Load Dataset with LeRobotDataset
    # ========================================================================
    logger.info("Loading dataset...")
    logger.info(f"  Repo ID: {args.dataset_repo_id}")
    logger.info(f"  Local root: {args.dataset_root}")

    # Compute delta_timestamps for the action horizon
    # The action horizon should match the policy's chunk_size
    chunk_size = policy.config.chunk_size
    fps = 30  # Control frequency for joint control (Hz)

    # Create delta_timestamps for action: [0, 1/fps, 2/fps, ..., (chunk_size-1)/fps]
    action_delta_timestamps = [i / fps for i in range(chunk_size)]
    delta_timestamps = {"action": action_delta_timestamps}

    logger.info(f"  Delta timestamps for action: {len(action_delta_timestamps)} timesteps")
    logger.info(f"  Range: 0 to {action_delta_timestamps[-1]:.2f} seconds at {fps} Hz")

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
    )

    logger.info(f"Dataset loaded successfully!")
    logger.info(f"  Total samples: {len(dataset)}")

    # Get dataset info
    sample = dataset[0]
    logger.info(f"  Observation state shape: {sample['observation.state'].shape}")
    logger.info(f"  Action shape: {sample['action'].shape}")
    logger.info(f"  Action horizon: {sample['action'].shape[0]}")

    # ========================================================================
    # Initialize Kinematics for EE trajectory visualization
    # ========================================================================
    urdf_path = Path(args.urdf_path)
    if urdf_path.exists():
        logger.info(f"Loading kinematics from: {urdf_path}")
        kinematics = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=JOINT_NAMES,
        )
        logger.info("Kinematics loaded successfully")
    else:
        logger.warning(f"URDF not found at {urdf_path}, skipping EE trajectory visualization")
        kinematics = None

    # ========================================================================
    # Sample random indices and run inference
    # ========================================================================
    logger.info(f"Running inference on {args.num_samples} random samples...")

    # Get valid indices (accounting for action horizon)
    action_horizon = sample['action'].shape[0]
    max_idx = len(dataset) - action_horizon
    valid_indices = np.random.choice(max_idx, args.num_samples, replace=False)

    all_results = []

    for sample_idx, idx in enumerate(valid_indices):
        logger.info(f"\n[{sample_idx + 1}/{args.num_samples}] Testing sample {idx}")

        # Get sample from dataset
        sample = dataset[idx]
        gt_actions = sample['action'].cpu().numpy()  # (action_horizon, 6)

        # Create observation batch
        batch = create_observation_batch(dataset, idx, device)

        # Run inference
        with torch.no_grad():
            processed_batch = preprocessor(batch)
            pred_actions = policy.predict_action_chunk(processed_batch)  # (1, chunk_size, 6)
            pred_actions = postprocessor(pred_actions)

        # Extract prediction
        if isinstance(pred_actions, dict) and "action" in pred_actions:
            pred_actions = pred_actions["action"]
        pred_actions = pred_actions[0].cpu().numpy()  # (chunk_size, 6)

        # Truncate to match ground truth length
        T = min(pred_actions.shape[0], gt_actions.shape[0])
        pred_trunc = pred_actions[:T]
        gt_trunc = gt_actions[:T]

        # Debug: show first action values
        logger.info(f"  GT[0] (first action): {gt_trunc[0]}")
        logger.info(f"  Pred[0] (first action): {pred_trunc[0]}")

        # Compute error metrics
        metrics = compute_error_metrics(pred_trunc, gt_trunc)

        logger.info(f"  Mean L2 error: {metrics['mean_l2_error']:.2f} degrees")
        logger.info(f"  Max L2 error: {metrics['max_l2_error']:.2f} degrees")
        logger.info(f"  Final L2 error: {metrics['final_l2_error']:.2f} degrees")

        # Plot joint trajectories
        traj_path = output_dir / f"sample_{idx}_trajectories.png"
        plot_joint_trajectories(pred_trunc, gt_trunc, str(traj_path), idx)

        # Plot error analysis
        error_path = output_dir / f"sample_{idx}_errors.png"
        plot_error_analysis(pred_trunc, gt_trunc, str(error_path), idx)

        # Compute and plot EE trajectories (via FK)
        if kinematics is not None:
            pred_ee_positions = compute_ee_trajectories(pred_trunc, kinematics)
            gt_ee_positions = compute_ee_trajectories(gt_trunc, kinematics)

            ee_traj_path = output_dir / f"sample_{idx}_ee_trajectory.png"
            plot_ee_trajectory_3d(pred_ee_positions, gt_ee_positions, str(ee_traj_path), idx)

            ee_metrics = compute_ee_error_metrics(pred_ee_positions, gt_ee_positions)
            logger.info(f"  Mean EE error: {ee_metrics['mean_ee_error_mm']:.2f} mm")
            logger.info(f"  Max EE error: {ee_metrics['max_ee_error_mm']:.2f} mm")
            logger.info(f"  Final EE error: {ee_metrics['final_ee_error_m']*1000:.2f} mm")

            # Save results with EE metrics
            all_results.append({
                "idx": idx,
                "metrics": metrics,
                "ee_metrics": ee_metrics,
                "pred_actions": pred_trunc,
                "gt_actions": gt_trunc,
                "pred_ee_positions": pred_ee_positions,
                "gt_ee_positions": gt_ee_positions,
            })
        else:
            # Save results without EE metrics
            all_results.append({
                "idx": idx,
                "metrics": metrics,
                "pred_actions": pred_trunc,
                "gt_actions": gt_trunc,
            })

    # ========================================================================
    # Summary Statistics
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)

    # Aggregate statistics
    all_l2_errors = []
    all_per_joint_errors = []
    all_ee_errors = []

    for result in all_results:
        metrics = result["metrics"]
        all_l2_errors.append(metrics["mean_l2_error"])
        all_per_joint_errors.append(metrics["per_joint_errors"])
        if "ee_metrics" in result:
            all_ee_errors.append(result["ee_metrics"]["mean_ee_error_mm"])

    all_per_joint_errors = np.array(all_per_joint_errors)  # (N, T, 6)

    logger.info(f"Average mean L2 error: {np.mean(all_l2_errors):.2f} degrees")
    logger.info(f"Std mean L2 error: {np.std(all_l2_errors):.2f} degrees")

    if all_ee_errors:
        logger.info(f"Average mean EE error: {np.mean(all_ee_errors):.2f} mm")
        logger.info(f"Std mean EE error: {np.std(all_ee_errors):.2f} mm")

    # Per-joint aggregate errors
    mean_per_joint = all_per_joint_errors.mean(axis=(0, 1))  # (6,)
    logger.info("\nPer-joint mean absolute errors:")
    for i, name in enumerate(JOINT_NAMES):
        logger.info(f"  {name}: {mean_per_joint[i]:.2f} degrees")

    # Save summary to file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("ACT Policy (Joint Control) Inference Debug Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset_repo_id}\n")
        f.write(f"Model: {args.pretrained_path}\n")
        f.write(f"Samples tested: {args.num_samples}\n\n")

        for i, result in enumerate(all_results):
            f.write(f"\nSample {i+1} (idx={result['idx']}):\n")
            m = result["metrics"]
            f.write(f"  Mean L2 error: {m['mean_l2_error']:.2f} degrees\n")
            f.write(f"  Max L2 error: {m['max_l2_error']:.2f} degrees\n")
            f.write(f"  Final L2 error: {m['final_l2_error']:.2f} degrees\n")
            if "ee_metrics" in result:
                ee = result["ee_metrics"]
                f.write(f"  Mean EE error: {ee['mean_ee_error_mm']:.2f} mm\n")
                f.write(f"  Max EE error: {ee['max_ee_error_mm']:.2f} mm\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Aggregate Statistics:\n")
        f.write(f"  Average mean L2 error: {np.mean(all_l2_errors):.2f} degrees\n")
        f.write(f"  Std mean L2 error: {np.std(all_l2_errors):.2f} degrees\n")
        if all_ee_errors:
            f.write(f"  Average mean EE error: {np.mean(all_ee_errors):.2f} mm\n")
            f.write(f"  Std mean EE error: {np.std(all_ee_errors):.2f} mm\n")
        f.write("\nPer-joint mean absolute errors:\n")
        for i, name in enumerate(JOINT_NAMES):
            f.write(f"  {name}: {mean_per_joint[i]:.2f} degrees\n")

    logger.info(f"\nSummary saved to: {summary_path}")

    # ========================================================================
    # Save all predictions if requested
    # ========================================================================
    if args.save_all:
        logger.info("\nSaving all predictions and ground truth...")

        all_pred = np.stack([r["pred_actions"] for r in all_results], axis=0)  # (N, T, 6)
        all_gt = np.stack([r["gt_actions"] for r in all_results], axis=0)  # (N, T, 6)

        np.save(output_dir / "all_predictions.npy", all_pred)
        np.save(output_dir / "all_ground_truth.npy", all_gt)

        logger.info(f"  Predictions: {output_dir / 'all_predictions.npy'}")
        logger.info(f"  Ground truth: {output_dir / 'all_ground_truth.npy'}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

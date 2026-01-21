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
Debug script for RelativeEE ACT policy inference.

This script loads a trained RelativeEE policy and compares its predictions
against ground truth from the dataset. Useful for debugging model behavior
without needing to connect to the robot.

Features:
- Loads dataset with RelativeEEDataset wrapper
- Samples random frames from the dataset
- Runs inference on each sample
- Computes position, rotation, and gripper errors
- Visualizes predicted vs ground truth trajectories
- Saves results for further analysis

Usage:
    python debug_relative_ee_inference.py \
        --dataset_repo_id red_strawberry_picking_260114_ee \
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
from lerobot.datasets.relative_ee_dataset import (
    RelativeEEDataset,
    pose10d_to_mat,
    rot6d_to_mat,
)
from lerobot.utils.utils import init_logging


def rotation_error(rot6d_pred: np.ndarray, rot6d_gt: np.ndarray) -> float:
    """
    Compute geodesic distance between two 6D rotations.

    Args:
        rot6d_pred: Predicted 6D rotation (6,)
        rot6d_gt: Ground truth 6D rotation (6,)

    Returns:
        Geodesic distance in radians
    """
    R_pred = rot6d_to_mat(rot6d_pred)
    R_gt = rot6d_to_mat(rot6d_gt)

    # Compute relative rotation: R_diff = R_pred^T @ R_gt
    R_diff = R_pred.T @ R_gt

    # Geodesic distance on SO(3)
    trace = np.trace(R_diff)
    # Clamp to valid range for arccos
    trace = np.clip(trace, -1.0, 3.0)
    return np.arccos((trace - 1) / 2)


def compute_error_metrics(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
) -> dict[str, Any]:
    """
    Compute error metrics between predicted and ground truth actions.

    Args:
        pred_actions: (T, 10) predicted actions
        gt_actions: (T, 10) ground truth actions

    Returns:
        Dictionary with error metrics
    """
    T = min(pred_actions.shape[0], gt_actions.shape[0])
    pred = pred_actions[:T]
    gt = gt_actions[:T]

    # Position error (first 3 dimensions)
    pos_errors = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)

    # Rotation error (6D rotation, dimensions 3-9)
    rot_errors = np.array([
        rotation_error(pred[i, 3:9], gt[i, 3:9])
        for i in range(T)
    ])

    # Gripper error (last dimension)
    gripper_errors = np.abs(pred[:, 9] - gt[:, 9])

    return {
        "position_errors": pos_errors,
        "rotation_errors": rot_errors,
        "gripper_errors": gripper_errors,
        "mean_position_error": float(np.mean(pos_errors)),
        "mean_rotation_error": float(np.mean(rot_errors)),
        "mean_gripper_error": float(np.mean(gripper_errors)),
        "max_position_error": float(np.max(pos_errors)),
        "max_rotation_error": float(np.max(rot_errors)),
    }


def plot_trajectory_comparison(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    output_path: str,
    sample_idx: int = 0,
):
    """
    Plot predicted vs ground truth EE trajectory.

    In RelativeEE, each action in the chunk is a transform from the BASE (current) pose
    to a future pose. Actions are NOT sequentially chained.
    - action[0] = T_base^(-1) @ T(t+1)  -> relative to base
    - action[1] = T_base^(-1) @ T(t+2)  -> relative to base
    - action[2] = T_base^(-1) @ T(t+3)  -> relative to base

    To get absolute positions: apply each action from origin (base pose at origin).

    Args:
        pred_actions: (T, 10) predicted relative actions
        gt_actions: (T, 10) ground truth relative actions
        output_path: Path to save the plot
        sample_idx: Sample index for title
    """
    T = min(pred_actions.shape[0], gt_actions.shape[0])

    # Compute trajectories by applying each action from origin (base pose)
    # In RelativeEE, each action is independent: from base to future timestep
    pred_positions = [[0.0, 0.0, 0.0]]  # Start at origin (base pose)
    gt_positions = [[0.0, 0.0, 0.0]]

    for i in range(T):
        # Predicted: apply action[i] from origin (base pose)
        pred_T = pose10d_to_mat(pred_actions[i, :9])
        pred_positions.append(pred_T[:3, 3].tolist())

        # Ground truth: apply action[i] from origin (base pose)
        gt_T = pose10d_to_mat(gt_actions[i, :9])
        gt_positions.append(gt_T[:3, 3].tolist())

    pred_positions = np.array(pred_positions)
    gt_positions = np.array(gt_positions)

    fig = plt.figure(figsize=(14, 10))

    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(
        pred_positions[:, 0],
        pred_positions[:, 1],
        pred_positions[:, 2],
        "r--", linewidth=2, label="Predicted", marker="o", markersize=3
    )
    ax1.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        gt_positions[:, 2],
        "b-", linewidth=2, label="Ground Truth", marker="x", markersize=3
    )
    ax1.scatter([0], [0], [0], c="green", s=100, marker="*", label="Start")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(f"Sample {sample_idx}: EE Trajectory")
    ax1.legend()

    # Set equal aspect ratio and reasonable limits
    all_pos = np.vstack([pred_positions, gt_positions])
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
    z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()

    # Add padding
    x_pad = max((x_max - x_min) * 0.1, 0.01)
    y_pad = max((y_max - y_min) * 0.1, 0.01)
    z_pad = max((z_max - z_min) * 0.1, 0.01)

    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)
    ax1.set_zlim(z_min - z_pad, z_max + z_pad)

    # Position error over time
    ax2 = fig.add_subplot(2, 2, 2)
    metrics = compute_error_metrics(pred_actions, gt_actions)
    ax2.plot(metrics["position_errors"] * 1000, "r-", linewidth=2)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Position Error (mm)")
    ax2.set_title("Position Error Over Time")
    ax2.grid(True, alpha=0.3)

    # Rotation error over time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(np.degrees(metrics["rotation_errors"]), "g-", linewidth=2)
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Rotation Error (degrees)")
    ax3.set_title("Rotation Error Over Time")
    ax3.grid(True, alpha=0.3)

    # Per-dimension error
    ax4 = fig.add_subplot(2, 2, 4)
    T_min = min(pred_actions.shape[0], gt_actions.shape[0])
    dims = ["dx", "dy", "dz", "r0", "r1", "r2", "r3", "r4", "r5", "gripper"]
    dim_errors = np.abs(pred_actions[:T] - gt_actions[:T]).mean(axis=0)
    bars = ax4.bar(range(len(dims)), dim_errors)
    ax4.set_xticks(range(len(dims)))
    ax4.set_xticklabels(dims, rotation=45, ha="right")
    ax4.set_ylabel("Mean Absolute Error")
    ax4.set_title("Per-Dimension Mean Absolute Error")
    ax4.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved to: {output_path}")


def create_observation_batch(
    dataset: RelativeEEDataset,
    idx: int,
    n_obs_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Create observation batch for policy inference.

    For RelativeEE, the current observation is always identity.
    For n_obs_steps > 1, we need historical context.

    Args:
        dataset: RelativeEEDataset instance
        idx: Current sample index
        n_obs_steps: Number of observation steps (horizon)
        device: Torch device

    Returns:
        Batch dictionary with observation data
    """
    # Get current sample
    sample = dataset[idx]

    # For RelativeEE, observation.state is already (10,) - identity at current
    obs_state = sample["observation.state"]  # (10,)

    # Prepare batch
    batch = {
        "observation.state": obs_state.unsqueeze(0).to(device),  # (1, 10)
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
    logger = logging.getLogger("debug_relative_ee_inference")

    import argparse
    parser = argparse.ArgumentParser(
        description="Debug script for RelativeEE ACT policy inference"
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., red_strawberry_picking_260114_ee)",
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
        default="./debug_relative_ee_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--obs_state_horizon",
        type=int,
        default=2,
        help="Observation state horizon (must match training)",
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
    # Load Policy first (to get chunk_size and fps for delta_timestamps)
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
    # Load Dataset with RelativeEEDataset wrapper
    # ========================================================================
    logger.info("Loading dataset...")
    logger.info(f"  Repo ID: {args.dataset_repo_id}")
    logger.info(f"  Local root: {args.dataset_root}")

    # Compute delta_timestamps for the action horizon
    # The action horizon should match the policy's chunk_size
    chunk_size = policy.config.chunk_size
    fps = getattr(policy.config, 'fps', 10)  # Default to 10Hz if not specified

    # Create delta_timestamps for action: [0, 1/fps, 2/fps, ..., (chunk_size-1)/fps]
    action_delta_timestamps = [i / fps for i in range(chunk_size)]
    delta_timestamps = {"action": action_delta_timestamps}

    logger.info(f"  Delta timestamps for action: {len(action_delta_timestamps)} timesteps")
    logger.info(f"  Range: 0 to {action_delta_timestamps[-1]:.2f} seconds at {fps} Hz")

    dataset = RelativeEEDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        obs_state_horizon=args.obs_state_horizon,
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
    # Sample random indices and run inference
    # ========================================================================
    logger.info(f"Running inference on {args.num_samples} random samples...")

    # Get valid indices (accounting for action horizon and obs_state_horizon)
    action_horizon = sample['action'].shape[0]
    max_idx = len(dataset) - action_horizon - args.obs_state_horizon

    # Generate consistent indices across different dataset sizes
    # Use a deterministic approach that produces the same indices for same seed
    rng = np.random.default_rng(args.seed)
    # Generate more random values than needed, then take first N unique values < max_idx
    random_values = rng.random(max_idx * 2)
    # Create an array of indices sorted by random values (shuffle by random values)
    all_indices = np.arange(max_idx)
    sorted_indices = all_indices[np.argsort(random_values[:max_idx])]
    # Take first N
    valid_indices = sorted_indices[:args.num_samples]

    all_results = []

    for sample_idx, idx in enumerate(valid_indices):
        logger.info(f"\n[{sample_idx + 1}/{args.num_samples}] Testing sample {idx}")

        # Get sample from dataset
        sample = dataset[idx]
        gt_actions = sample['action'].cpu().numpy()  # (action_horizon, 10)

        # Create observation batch
        batch = create_observation_batch(dataset, idx, policy.config.n_obs_steps, device)

        # Run inference
        with torch.no_grad():
            processed_batch = preprocessor(batch)
            pred_actions = policy.predict_action_chunk(processed_batch)  # (1, chunk_size, 10)
            pred_actions = postprocessor(pred_actions)

        # Extract prediction
        if isinstance(pred_actions, dict) and "action" in pred_actions:
            pred_actions = pred_actions["action"]
        pred_actions = pred_actions[0].cpu().numpy()  # (chunk_size, 10)

        # Truncate to match ground truth length
        T = min(pred_actions.shape[0], gt_actions.shape[0])
        pred_trunc = pred_actions[:T]
        gt_trunc = gt_actions[:T]

        # Debug: show first few action values
        logger.info(f"  GT[0] (first action): pos={gt_trunc[0, :3]}, gripper={gt_trunc[0, 9]:.3f}")
        logger.info(f"  Pred[0] (first action): pos={pred_trunc[0, :3]}, gripper={pred_trunc[0, 9]:.3f}")

        # Compute error metrics
        metrics = compute_error_metrics(pred_trunc, gt_trunc)

        logger.info(f"  Mean position error: {metrics['mean_position_error']*1000:.2f} mm")
        logger.info(f"  Mean rotation error: {np.degrees(metrics['mean_rotation_error']):.2f} deg")
        logger.info(f"  Mean gripper error: {metrics['mean_gripper_error']:.4f}")
        logger.info(f"  Max position error: {metrics['max_position_error']*1000:.2f} mm")

        # Plot trajectory comparison
        plot_path = output_dir / f"sample_{idx}_comparison.png"
        plot_trajectory_comparison(pred_trunc, gt_trunc, str(plot_path), idx)

        # Save results
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
    all_pos_errors = []
    all_rot_errors = []
    all_grip_errors = []

    for result in all_results:
        metrics = result["metrics"]
        all_pos_errors.append(metrics["mean_position_error"])
        all_rot_errors.append(metrics["mean_rotation_error"])
        all_grip_errors.append(metrics["mean_gripper_error"])

    logger.info(f"Average mean position error: {np.mean(all_pos_errors)*1000:.2f} mm")
    logger.info(f"Average mean rotation error: {np.degrees(np.mean(all_rot_errors)):.2f} deg")
    logger.info(f"Average mean gripper error: {np.mean(all_grip_errors):.4f}")
    logger.info(f"Std position error: {np.std(all_pos_errors)*1000:.2f} mm")
    logger.info(f"Std rotation error: {np.degrees(np.std(all_rot_errors)):.2f} deg")

    # Save summary to file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("RelativeEE ACT Policy Inference Debug Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset_repo_id}\n")
        f.write(f"Model: {args.pretrained_path}\n")
        f.write(f"Samples tested: {args.num_samples}\n\n")

        for i, result in enumerate(all_results):
            f.write(f"\nSample {i+1} (idx={result['idx']}):\n")
            m = result["metrics"]
            f.write(f"  Mean position error: {m['mean_position_error']*1000:.2f} mm\n")
            f.write(f"  Mean rotation error: {np.degrees(m['mean_rotation_error']):.2f} deg\n")
            f.write(f"  Mean gripper error: {m['mean_gripper_error']:.4f}\n")
            f.write(f"  Max position error: {m['max_position_error']*1000:.2f} mm\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Aggregate Statistics:\n")
        f.write(f"  Average mean position error: {np.mean(all_pos_errors)*1000:.2f} mm\n")
        f.write(f"  Average mean rotation error: {np.degrees(np.mean(all_rot_errors)):.2f} deg\n")
        f.write(f"  Average mean gripper error: {np.mean(all_grip_errors):.4f}\n")
        f.write(f"  Std position error: {np.std(all_pos_errors)*1000:.2f} mm\n")
        f.write(f"  Std rotation error: {np.degrees(np.std(all_rot_errors)):.2f} deg\n")

    logger.info(f"\nSummary saved to: {summary_path}")

    # ========================================================================
    # Save all predictions if requested
    # ========================================================================
    if args.save_all:
        logger.info("\nSaving all predictions and ground truth...")

        all_pred = np.stack([r["pred_actions"] for r in all_results], axis=0)  # (N, T, 10)
        all_gt = np.stack([r["gt_actions"] for r in all_results], axis=0)  # (N, T, 10)

        np.save(output_dir / "all_predictions.npy", all_pred)
        np.save(output_dir / "all_ground_truth.npy", all_gt)

        logger.info(f"  Predictions: {output_dir / 'all_predictions.npy'}")
        logger.info(f"  Ground truth: {output_dir / 'all_ground_truth.npy'}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

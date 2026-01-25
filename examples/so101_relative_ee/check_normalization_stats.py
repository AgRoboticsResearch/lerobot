#!/usr/bin/env python3
"""
Load and display normalization stats from a trained model checkpoint.

This script loads the normalization stats stored in the preprocessor safetensors file
and displays them for comparison with debug script output.

Usage:
    python check_normalization_stats.py --checkpoint /path/to/checkpoint
"""

import argparse
from pathlib import Path
from safetensors import safe_open


def load_stats(checkpoint_path: str) -> dict:
    """Load normalization stats from preprocessor safetensors file."""
    preprocessor_path = Path(checkpoint_path) / "pretrained_model" / "policy_preprocessor_step_3_normalizer_processor.safetensors"

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

    stats = {}
    with safe_open(preprocessor_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            stats[key] = f.get_tensor(key).numpy()

    return stats


def print_stats(stats: dict):
    """Print normalization stats in a readable format."""
    print("=" * 60)
    print("NORMALIZATION STATS FROM TRAINED MODEL")
    print("=" * 60)

    # Action stats
    print("\n--- ACTION Stats ---")
    for key in sorted(stats.keys()):
        if key.startswith("action."):
            tensor = stats[key]
            print(f"  {key}: shape={tensor.shape}")
            if tensor.ndim == 1:
                print(f"    values: {tensor}")
            else:
                print(f"    values: {tensor.flatten()[:10]}...")  # First 10 elements

    # Observation stats
    print("\n--- OBSERVATION Stats ---")
    for key in sorted(stats.keys()):
        if key.startswith("observation."):
            tensor = stats[key]
            print(f"  {key}: shape={tensor.shape}")
            if tensor.ndim == 1 and tensor.size <= 20:
                print(f"    values: {tensor}")
            elif tensor.ndim == 1:
                print(f"    values: {tensor[:5]}...{tensor[-5:]}")  # First and last 5

    # Detailed breakdown for relative EE action (10D)
    print("\n--- Relative EE Action Breakdown (10D) ---")
    print("  [0:3]   : Position (x, y, z)")
    print("  [3:9]   : Rotation 6D")
    print("  [9]     : Gripper")

    action_max = stats.get("action.max")
    action_min = stats.get("action.min")
    action_mean = stats.get("action.mean")
    action_std = stats.get("action.std")

    if action_max is not None:
        print(f"\n  Position max:   {action_max[0:3]}")
        print(f"  Position min:   {action_min[0:3]}")
        print(f"  Position mean:  {action_mean[0:3]}")
        print(f"  Position std:   {action_std[0:3]}")

        print(f"\n  Rotation6D max:  {action_max[3:9]}")
        print(f"  Rotation6D min:  {action_min[3:9]}")
        print(f"  Rotation6D mean: {action_mean[3:9]}")
        print(f"  Rotation6D std:  {action_std[3:9]}")

        print(f"\n  Gripper max:     {action_max[9]}")
        print(f"  Gripper min:     {action_min[9]}")
        print(f"  Gripper mean:    {action_mean[9]}")
        print(f"  Gripper std:     {action_std[9]}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Load and display normalization stats from trained model checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs/train/red_strawberry_picking_260119_merged_ultralong/checkpoints/200000",
        help="Path to checkpoint directory",
    )

    args = parser.parse_args()

    print(f"Loading stats from: {args.checkpoint}")
    stats = load_stats(args.checkpoint)
    print_stats(stats)


if __name__ == "__main__":
    main()

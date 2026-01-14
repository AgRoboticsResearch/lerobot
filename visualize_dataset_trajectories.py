#!/usr/bin/env python
"""
Visualize trajectories from a LeRobot dataset to understand the data.

This script loads a dataset and creates comprehensive plots of:
- 3D end-effector position trajectories
- Orientation trajectories (6D rotation)
- Gripper state over time
- Action vs Observation comparison

Usage:
    python visualize_dataset_trajectories.py --dataset-path /path/to/dataset
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import pyarrow.parquet as pq


def load_dataset_info(dataset_path: str | Path) -> dict:
    """Load dataset info from info.json."""
    path = Path(dataset_path)
    info_path = path / "meta" / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    return info


def load_episodes(dataset_path: str | Path) -> pd.DataFrame:
    """Load episode metadata from parquet files."""
    path = Path(dataset_path)
    episodes_dir = path / "meta" / "episodes"

    # Find all parquet files in episodes directory
    parquet_files = sorted(episodes_dir.glob("**/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}")

    # Load and concatenate all episode data
    episodes = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        episodes.append(df)

    episodes_df = pd.concat(episodes, ignore_index=True)
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)

    return episodes_df


def load_data_by_episode(dataset_path: str | Path, episodes_df: pd.DataFrame) -> dict:
    """
    Load data for each episode from parquet files.

    Returns a dict mapping episode_index to (data_dict, num_frames) tuples.
    """
    path = Path(dataset_path)

    episode_data = {}

    for _, row in episodes_df.iterrows():
        ep_idx = int(row["episode_index"])
        chunk_idx = int(row["data/chunk_index"])
        file_idx = int(row["data/file_index"])

        # Construct path to data file
        data_path = path / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"

        if not data_path.exists():
            print(f"Warning: Data file not found for episode {ep_idx}: {data_path}")
            continue

        # Load parquet file
        table = pq.read_table(data_path)
        df = table.to_pandas()

        # Filter rows for this episode
        ep_mask = df["episode_index"] == ep_idx
        ep_df = df[ep_mask].sort_values("index").reset_index(drop=True)

        episode_data[ep_idx] = {
            "df": ep_df,
            "num_frames": len(ep_df),
            "from_index": int(row["dataset_from_index"]),
            "to_index": int(row["dataset_to_index"]),
        }

    return episode_data


def extract_trajectories(episode_data: dict, info: dict) -> tuple[dict, list, list]:
    """Extract trajectories from episode data."""
    print("Extracting trajectories...")

    trajectories = defaultdict(lambda: defaultdict(list))

    # Get feature names
    state_names = info["features"].get("observation.state", {}).get("names", [])
    action_names = info["features"].get("action", {}).get("names", [])

    print(f"  State dimensions: {state_names}")
    print(f"  Action dimensions: {action_names}")

    for ep_idx, data in episode_data.items():
        df = data["df"]

        # Extract observation state
        if "observation.state" in df.columns:
            for state_arr in df["observation.state"]:
                trajectories["state"][ep_idx].append(state_arr)
        else:
            print("  Warning: 'observation.state' not found in dataset")

        # Extract action
        if "action" in df.columns:
            for action_arr in df["action"]:
                trajectories["action"][ep_idx].append(action_arr)
        else:
            print("  Warning: 'action' not found in dataset")

        # Extract timestamp
        for ts in df["timestamp"]:
            trajectories["timestamp"][ep_idx].append(ts)

    # Convert to numpy arrays
    for key in trajectories:
        for ep_idx in sorted(trajectories[key].keys()):
            trajectories[key][ep_idx] = np.array(trajectories[key][ep_idx])

    return trajectories, state_names, action_names


def plot_3d_trajectories(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot 3D trajectories of end-effector position."""
    state = trajectories["state"]

    # Find x, y, z indices
    x_idx = state_names.index("ee.x") if "ee.x" in state_names else 0
    y_idx = state_names.index("ee.y") if "ee.y" in state_names else 1
    z_idx = state_names.index("ee.z") if "ee.z" in state_names else 2

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(state))))

    for ep_idx, color in zip(sorted(state.keys()), colors):
        ep_state = state[ep_idx]
        x = ep_state[:, x_idx]
        y = ep_state[:, y_idx]
        z = ep_state[:, z_idx]

        # Plot trajectory
        ax.plot(x, y, z, color=color, alpha=0.7, linewidth=1.5, label=f"Ep {ep_idx}")
        # Mark start
        ax.scatter(x[0], y[0], z[0], color=color, s=50, marker="o", edgecolors="black")
        # Mark end
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=50, marker="s", edgecolors="black")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector 3D Trajectories")

    # Add legend
    if len(state) <= 20:
        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / "trajectories_3d.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path / 'trajectories_3d.png'}")
    else:
        plt.show()

    plt.close()


def plot_position_over_time(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot position dimensions over time."""
    state = trajectories["state"]

    x_idx = state_names.index("ee.x") if "ee.x" in state_names else 0
    y_idx = state_names.index("ee.y") if "ee.y" in state_names else 1
    z_idx = state_names.index("ee.z") if "ee.z" in state_names else 2

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(state))))

    for ep_idx, color in zip(sorted(state.keys()), colors):
        ep_state = state[ep_idx]
        timestamp = trajectories["timestamp"][ep_idx]

        axes[0].plot(timestamp, ep_state[:, x_idx], color=color, alpha=0.7, linewidth=1, label=f"Ep {ep_idx}")
        axes[1].plot(timestamp, ep_state[:, y_idx], color=color, alpha=0.7, linewidth=1)
        axes[2].plot(timestamp, ep_state[:, z_idx], color=color, alpha=0.7, linewidth=1)

    axes[0].set_ylabel("X (m)")
    axes[0].set_title("End-Effector Position Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    axes[1].set_ylabel("Y (m)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Z (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / "position_over_time.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path / 'position_over_time.png'}")
    else:
        plt.show()

    plt.close()


def plot_orientation_and_gripper(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot orientation and gripper over time."""
    state = trajectories["state"]

    # Find orientation indices
    wx_idx = state_names.index("ee.wx") if "ee.wx" in state_names else 3
    wy_idx = state_names.index("ee.wy") if "ee.wy" in state_names else 4
    wz_idx = state_names.index("ee.wz") if "ee.wz" in state_names else 5
    gripper_idx = state_names.index("ee.gripper_pos") if "ee.gripper_pos" in state_names else 6

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(state))))

    for ep_idx, color in zip(sorted(state.keys()), colors):
        ep_state = state[ep_idx]
        timestamp = trajectories["timestamp"][ep_idx]

        axes[0].plot(timestamp, ep_state[:, wx_idx], color=color, alpha=0.7, linewidth=1)
        axes[1].plot(timestamp, ep_state[:, wy_idx], color=color, alpha=0.7, linewidth=1)
        axes[2].plot(timestamp, ep_state[:, wz_idx], color=color, alpha=0.7, linewidth=1)
        axes[3].plot(timestamp, ep_state[:, gripper_idx], color=color, alpha=0.7, linewidth=1)

    axes[0].set_ylabel("ωx (rad/s)")
    axes[0].set_title("Orientation (Angular Velocity) and Gripper Over Time")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("ωy (rad/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("ωz (rad/s)")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("Gripper")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / "orientation_and_gripper.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path / 'orientation_and_gripper.png'}")
    else:
        plt.show()

    plt.close()


def plot_action_vs_state(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot action vs state for comparison."""
    state = trajectories["state"]
    action = trajectories["action"]

    if not action:
        print("  No action data found, skipping action vs state plot")
        return

    # Use first dimension for comparison (ee.x)
    x_idx = state_names.index("ee.x") if "ee.x" in state_names else 0

    num_eps = len(state)
    fig, axes = plt.subplots(num_eps, 1, figsize=(14, 3 * num_eps))
    if num_eps == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(state))))

    for i, ep_idx in enumerate(sorted(state.keys())):
        ep_state = state[ep_idx]
        ep_action = action.get(ep_idx)
        timestamp = trajectories["timestamp"][ep_idx]

        axes[i].plot(timestamp, ep_state[:, x_idx], "b-", alpha=0.7, linewidth=2, label="State (ee.x)")

        if ep_action is not None:
            axes[i].plot(timestamp, ep_action[:, x_idx], "r--", alpha=0.7, linewidth=1.5, label="Action (ee.x)")

        axes[i].set_ylabel(f"Episode {ep_idx}")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Action vs State Comparison (ee.x dimension)", y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / "action_vs_state.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path / 'action_vs_state.png'}")
    else:
        plt.show()

    plt.close()


def plot_trajectory_statistics(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot statistics about trajectories."""
    state = trajectories["state"]

    # Compute statistics per episode
    episode_lengths = [len(state[ep_idx]) for ep_idx in sorted(state.keys())]
    total_frames = sum(episode_lengths)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # Episode lengths
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(len(episode_lengths)), episode_lengths, color="steelblue")
    ax1.set_xlabel("Episode Index")
    ax1.set_ylabel("Number of Frames")
    ax1.set_title("Episode Lengths")
    ax1.grid(True, alpha=0.3, axis="y")

    # Position ranges
    ax2 = fig.add_subplot(gs[0, 1])

    x_idx = state_names.index("ee.x") if "ee.x" in state_names else 0
    y_idx = state_names.index("ee.y") if "ee.y" in state_names else 1
    z_idx = state_names.index("ee.z") if "ee.z" in state_names else 2

    ranges = []
    for ep_idx in sorted(state.keys()):
        ep_state = state[ep_idx]
        x_range = ep_state[:, x_idx].max() - ep_state[:, x_idx].min()
        y_range = ep_state[:, y_idx].max() - ep_state[:, y_idx].min()
        z_range = ep_state[:, z_idx].max() - ep_state[:, z_idx].min()
        ranges.append((x_range, y_range, z_range))

    ranges = np.array(ranges)
    x = np.arange(len(ranges))
    width = 0.25
    ax2.bar(x - width, ranges[:, 0], width, label="X range", color="r", alpha=0.7)
    ax2.bar(x, ranges[:, 1], width, label="Y range", color="g", alpha=0.7)
    ax2.bar(x + width, ranges[:, 2], width, label="Z range", color="b", alpha=0.7)
    ax2.set_xlabel("Episode Index")
    ax2.set_ylabel("Range (m)")
    ax2.set_title("Position Range per Episode")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Start positions
    ax3 = fig.add_subplot(gs[1, 0])
    for ep_idx in sorted(state.keys()):
        ep_state = state[ep_idx]
        ax3.scatter(ep_state[0, x_idx], ep_state[0, y_idx], s=100, alpha=0.6, edgecolors="black")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title("Start Positions (Top View)")
    ax3.grid(True, alpha=0.3)

    # Gripper usage
    ax4 = fig.add_subplot(gs[1, 1])
    gripper_idx = state_names.index("ee.gripper_pos") if "ee.gripper_pos" in state_names else 6

    gripper_means = []
    for ep_idx in sorted(state.keys()):
        ep_state = state[ep_idx]
        gripper_means.append(ep_state[:, gripper_idx].mean())

    ax4.bar(range(len(gripper_means)), gripper_means, color="coral")
    ax4.set_xlabel("Episode Index")
    ax4.set_ylabel("Average Gripper Position")
    ax4.set_title("Average Gripper State per Episode")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Dataset Statistics (Total: {len(state)} episodes, {total_frames} frames)", y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / "statistics.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path / 'statistics.png'}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset trajectories")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for saving plots. If not provided, plots will be displayed.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Specific episodes to visualize. If not provided, all episodes will be visualized.",
    )

    args = parser.parse_args()

    # Load dataset info
    print(f"Loading dataset info from {args.dataset_path}")
    info = load_dataset_info(args.dataset_path)
    print(f"  Robot type: {info.get('robot_type', 'N/A')}")
    print(f"  Total episodes: {info['total_episodes']}")
    print(f"  Total frames: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")

    # Load episodes metadata
    episodes_df = load_episodes(args.dataset_path)
    print(f"  Loaded {len(episodes_df)} episodes")

    # Filter episodes if specified
    if args.episodes:
        print(f"Filtering to episodes: {args.episodes}")
        episodes_df = episodes_df[episodes_df["episode_index"].isin(args.episodes)]

    # Load data
    episode_data = load_data_by_episode(args.dataset_path, episodes_df)
    print(f"  Loaded data for {len(episode_data)} episodes")

    # Extract trajectories
    trajectories, state_names, action_names = extract_trajectories(episode_data, info)

    # Setup output directory
    output_path = Path(args.output_dir) if args.output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving plots to: {output_path}")

    # Create plots
    print("\nCreating visualizations...")
    plot_3d_trajectories(trajectories, state_names, output_path)
    plot_position_over_time(trajectories, state_names, output_path)
    plot_orientation_and_gripper(trajectories, state_names, output_path)
    plot_action_vs_state(trajectories, state_names, output_path)
    plot_trajectory_statistics(trajectories, state_names, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()

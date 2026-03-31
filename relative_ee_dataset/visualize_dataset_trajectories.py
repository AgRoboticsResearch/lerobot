#!/usr/bin/env python
"""
Visualize trajectories from a LeRobot dataset to understand the data.

This script loads a dataset and creates per-episode plots of:
- 3D end-effector position trajectories
- 2D projections (XY, YZ, XZ)
- Position, orientation, and gripper over time
- Action vs Observation comparison
- Per-episode statistics

Usage:
    python visualize_dataset_trajectories.py --dataset-path /path/to/dataset --output-dir ./plots
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


def _get_indices(state_names: list):
    """Get feature indices from state names."""
    x_idx = state_names.index("ee.x") if "ee.x" in state_names else 0
    y_idx = state_names.index("ee.y") if "ee.y" in state_names else 1
    z_idx = state_names.index("ee.z") if "ee.z" in state_names else 2
    wx_idx = state_names.index("ee.wx") if "ee.wx" in state_names else 3
    wy_idx = state_names.index("ee.wy") if "ee.wy" in state_names else 4
    wz_idx = state_names.index("ee.wz") if "ee.wz" in state_names else 5
    gripper_idx = state_names.index("ee.gripper_pos") if "ee.gripper_pos" in state_names else 6
    return x_idx, y_idx, z_idx, wx_idx, wy_idx, wz_idx, gripper_idx


def plot_3d_trajectories(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot 3D trajectory for each episode as a separate file."""
    state = trajectories["state"]
    x_idx, y_idx, z_idx = _get_indices(state_names)[:3]

    for ep_idx in sorted(state.keys()):
        ep_state = state[ep_idx]
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        x, y, z = ep_state[:, x_idx], ep_state[:, y_idx], ep_state[:, z_idx]
        ax.plot(x, y, z, color="steelblue", alpha=0.7, linewidth=1.5)
        ax.scatter(x[0], y[0], z[0], color="green", s=80, marker="o", edgecolors="black", label="Start")
        ax.scatter(x[-1], y[-1], z[-1], color="red", s=80, marker="s", edgecolors="black", label="End")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"End-Effector 3D Trajectory (Episode {ep_idx})")
        ax.legend()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / f"trajectories_3d_ep{ep_idx:03d}.png", dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    if output_path:
        print(f"  Saved {len(state)} episode files: {output_path / 'trajectories_3d_ep*.png'}")


def plot_2d_projections(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot 2D projections (XY, YZ, XZ) for each episode as a separate file."""
    state = trajectories["state"]
    x_idx, y_idx, z_idx = _get_indices(state_names)[:3]

    for ep_idx in sorted(state.keys()):
        ep_state = state[ep_idx]
        x, y, z = ep_state[:, x_idx], ep_state[:, y_idx], ep_state[:, z_idx]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # XY
        axes[0].plot(x, y, color="steelblue", alpha=0.7, linewidth=1.5)
        axes[0].scatter(x[0], y[0], color="green", s=80, marker="o", edgecolors="black", label="Start")
        axes[0].scatter(x[-1], y[-1], color="red", s=80, marker="s", edgecolors="black", label="End")
        axes[0].set_xlabel("X (m)")
        axes[0].set_ylabel("Y (m)")
        axes[0].set_title("XY Projection")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis("equal")

        # YZ
        axes[1].plot(y, z, color="steelblue", alpha=0.7, linewidth=1.5)
        axes[1].scatter(y[0], z[0], color="green", s=80, marker="o", edgecolors="black", label="Start")
        axes[1].scatter(y[-1], z[-1], color="red", s=80, marker="s", edgecolors="black", label="End")
        axes[1].set_xlabel("Y (m)")
        axes[1].set_ylabel("Z (m)")
        axes[1].set_title("YZ Projection")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axis("equal")

        # XZ
        axes[2].plot(x, z, color="steelblue", alpha=0.7, linewidth=1.5)
        axes[2].scatter(x[0], z[0], color="green", s=80, marker="o", edgecolors="black", label="Start")
        axes[2].scatter(x[-1], z[-1], color="red", s=80, marker="s", edgecolors="black", label="End")
        axes[2].set_xlabel("X (m)")
        axes[2].set_ylabel("Z (m)")
        axes[2].set_title("XZ Projection")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].axis("equal")

        plt.suptitle(f"2D Trajectory Projections (Episode {ep_idx})", y=1.02)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / f"2d_projections_ep{ep_idx:03d}.png", dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    if output_path:
        print(f"  Saved {len(state)} episode files: {output_path / '2d_projections_ep*.png'}")


def plot_position_over_time(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot position dimensions over time for each episode as a separate file."""
    state = trajectories["state"]
    x_idx, y_idx, z_idx = _get_indices(state_names)[:3]

    for ep_idx in sorted(state.keys()):
        ep_state = state[ep_idx]
        timestamp = trajectories["timestamp"][ep_idx]
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(timestamp, ep_state[:, x_idx], color="steelblue", alpha=0.7, linewidth=1.5)
        axes[1].plot(timestamp, ep_state[:, y_idx], color="steelblue", alpha=0.7, linewidth=1.5)
        axes[2].plot(timestamp, ep_state[:, z_idx], color="steelblue", alpha=0.7, linewidth=1.5)
        axes[0].set_ylabel("X (m)")
        axes[0].set_title(f"End-Effector Position Over Time (Episode {ep_idx})")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel("Y (m)")
        axes[1].grid(True, alpha=0.3)
        axes[2].set_ylabel("Z (m)")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / f"position_over_time_ep{ep_idx:03d}.png", dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    if output_path:
        print(f"  Saved {len(state)} episode files: {output_path / 'position_over_time_ep*.png'}")


def plot_orientation_and_gripper(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot orientation and gripper over time for each episode as a separate file."""
    state = trajectories["state"]
    _, _, _, wx_idx, wy_idx, wz_idx, gripper_idx = _get_indices(state_names)

    for ep_idx in sorted(state.keys()):
        ep_state = state[ep_idx]
        timestamp = trajectories["timestamp"][ep_idx]
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        axes[0].plot(timestamp, ep_state[:, wx_idx], color="steelblue", alpha=0.7, linewidth=1.5)
        axes[1].plot(timestamp, ep_state[:, wy_idx], color="steelblue", alpha=0.7, linewidth=1.5)
        axes[2].plot(timestamp, ep_state[:, wz_idx], color="steelblue", alpha=0.7, linewidth=1.5)
        axes[3].plot(timestamp, ep_state[:, gripper_idx], color="coral", alpha=0.7, linewidth=1.5)
        axes[0].set_ylabel("wx (rad/s)")
        axes[0].set_title(f"Orientation (Angular Velocity) and Gripper Over Time (Episode {ep_idx})")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel("wy (rad/s)")
        axes[1].grid(True, alpha=0.3)
        axes[2].set_ylabel("wz (rad/s)")
        axes[2].grid(True, alpha=0.3)
        axes[3].set_ylabel("Gripper")
        axes[3].set_xlabel("Time (s)")
        axes[3].grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / f"orientation_and_gripper_ep{ep_idx:03d}.png", dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    if output_path:
        print(f"  Saved {len(state)} episode files: {output_path / 'orientation_and_gripper_ep*.png'}")


def plot_action_vs_state(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot action vs state for each episode as a separate file."""
    state = trajectories["state"]
    action = trajectories["action"]

    if not action:
        print("  No action data found, skipping action vs state plot")
        return

    x_idx = _get_indices(state_names)[0]

    for ep_idx in sorted(state.keys()):
        ep_state = state[ep_idx]
        ep_action = action.get(ep_idx)
        timestamp = trajectories["timestamp"][ep_idx]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(timestamp, ep_state[:, x_idx], "b-", alpha=0.7, linewidth=2, label="State (ee.x)")
        if ep_action is not None:
            ax.plot(timestamp, ep_action[:, x_idx], "r--", alpha=0.7, linewidth=1.5, label="Action (ee.x)")
        ax.set_title(f"Action vs State Comparison (Episode {ep_idx})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ee.x (m)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / f"action_vs_state_ep{ep_idx:03d}.png", dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    if output_path:
        print(f"  Saved {len(state)} episode files: {output_path / 'action_vs_state_ep*.png'}")


def plot_trajectory_statistics(trajectories: dict, state_names: list, output_path: Path = None):
    """Plot per-episode statistics as a separate file for each episode."""
    state = trajectories["state"]
    x_idx, y_idx, z_idx = _get_indices(state_names)[:3]
    _, _, _, _, _, _, gripper_idx = _get_indices(state_names)

    for ep_idx in sorted(state.keys()):
        ep_state = state[ep_idx]
        timestamp = trajectories["timestamp"][ep_idx]

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

        # Position over time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(timestamp, ep_state[:, x_idx], label="X", color="r", alpha=0.7)
        ax1.plot(timestamp, ep_state[:, y_idx], label="Y", color="g", alpha=0.7)
        ax1.plot(timestamp, ep_state[:, z_idx], label="Z", color="b", alpha=0.7)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_title("Position Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gripper state
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(timestamp, ep_state[:, gripper_idx], color="coral", alpha=0.7)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Gripper Position")
        ax2.set_title("Gripper State")
        ax2.grid(True, alpha=0.3)

        # XY projection
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(ep_state[:, x_idx], ep_state[:, y_idx], color="steelblue", alpha=0.7)
        ax3.scatter(ep_state[0, x_idx], ep_state[0, y_idx], color="green", s=80, marker="o", edgecolors="black", label="Start")
        ax3.scatter(ep_state[-1, x_idx], ep_state[-1, y_idx], color="red", s=80, marker="s", edgecolors="black", label="End")
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_title("XY Projection")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Position ranges
        ax4 = fig.add_subplot(gs[1, 1])
        x_range = ep_state[:, x_idx].max() - ep_state[:, x_idx].min()
        y_range = ep_state[:, y_idx].max() - ep_state[:, y_idx].min()
        z_range = ep_state[:, z_idx].max() - ep_state[:, z_idx].min()
        ax4.bar(["X", "Y", "Z"], [x_range, y_range, z_range], color=["r", "g", "b"], alpha=0.7)
        ax4.set_ylabel("Range (m)")
        ax4.set_title("Position Range")
        ax4.grid(True, alpha=0.3, axis="y")

        plt.suptitle(f"Episode {ep_idx} Statistics ({len(ep_state)} frames)", y=1.02)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / f"statistics_ep{ep_idx:03d}.png", dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    if output_path:
        print(f"  Saved {len(state)} episode files: {output_path / 'statistics_ep*.png'}")


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
        print(f"\nSaving per-episode plots to: {output_path}")

    # Save trajectories to CSV
    if output_path:
        print("\nSaving trajectory CSVs...")
        for ep_idx in sorted(trajectories["state"].keys()):
            ep_state = trajectories["state"][ep_idx]
            ep_action = trajectories["action"].get(ep_idx)
            ep_ts = trajectories["timestamp"][ep_idx]
            n_frames = len(ep_state)

            rows = []
            for i in range(n_frames):
                row = {"step": i, "timestamp": ep_ts[i]}
                for j, name in enumerate(state_names):
                    row[f"state.{name}"] = ep_state[i, j]
                if ep_action is not None:
                    for j, name in enumerate(action_names):
                        row[f"action.{name}"] = ep_action[i, j]
                rows.append(row)

            csv_path = output_path / f"trajectory_ep{ep_idx:03d}.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"  Saved {csv_path} ({n_frames} rows, columns: {list(rows[0].keys())})")

    # Create plots (one file per episode each)
    print("\nCreating visualizations...")
    plot_3d_trajectories(trajectories, state_names, output_path)
    plot_2d_projections(trajectories, state_names, output_path)
    plot_position_over_time(trajectories, state_names, output_path)
    plot_orientation_and_gripper(trajectories, state_names, output_path)
    plot_action_vs_state(trajectories, state_names, output_path)
    plot_trajectory_statistics(trajectories, state_names, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()

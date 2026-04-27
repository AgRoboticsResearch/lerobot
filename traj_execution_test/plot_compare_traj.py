#!/usr/bin/env python
"""Compare sent/actual EE trajectories from two execution result CSVs in one 3D plot.

Usage:
    python plot_compare_traj.py real_so101_test_x_axis_result.csv chunked_so101_test_x_axis_result.csv
    python plot_compare_traj.py a.csv b.csv --labels "Absolute IK" "Chunked Relative"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ee_positions(csv_path):
    df = pd.read_csv(csv_path)
    return {
        "sent": df[["sent_x", "sent_y", "sent_z"]].values,
        "act": df[["act_x", "act_y", "act_z"]].values,
        "pos_err": df["pos_err_mm"].values,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare EE trajectories from two result CSVs")
    parser.add_argument("csv_a", help="First result CSV")
    parser.add_argument("csv_b", help="Second result CSV")
    parser.add_argument("--labels", nargs=2, default=None,
                        help="Labels for the two CSVs (e.g. 'Absolute IK' 'Chunked Relative')")
    parser.add_argument("-o", "--output", default=None, help="Output image path")
    args = parser.parse_args()

    base = Path(__file__).parent / "output"

    path_a = Path(args.csv_a)
    if not path_a.is_absolute():
        path_a = base / args.csv_a
    path_b = Path(args.csv_b)
    if not path_b.is_absolute():
        path_b = base / args.csv_b

    a = load_ee_positions(path_a)
    b = load_ee_positions(path_b)

    name_a = path_a.stem.replace("_result", "")
    name_b = path_b.stem.replace("_result", "")
    if args.labels:
        name_a, name_b = args.labels

    # --- 3D plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(a["sent"][:, 0], a["sent"][:, 1], a["sent"][:, 2],
            "b-", linewidth=2, alpha=0.8, label=f"{name_a} Sent")
    ax.plot(a["act"][:, 0], a["act"][:, 1], a["act"][:, 2],
            "b--", linewidth=2, alpha=0.5, label=f"{name_a} Actual")

    ax.plot(b["sent"][:, 0], b["sent"][:, 1], b["sent"][:, 2],
            "r-", linewidth=2, alpha=0.8, label=f"{name_b} Sent")
    ax.plot(b["act"][:, 0], b["act"][:, 1], b["act"][:, 2],
            "r--", linewidth=2, alpha=0.5, label=f"{name_b} Actual")

    # Mark start/end
    for data, color in [(a, "blue"), (b, "red")]:
        ax.scatter(*data["sent"][0], c=color, s=100, marker="o", zorder=10, edgecolors="black")
        ax.scatter(*data["sent"][-1], c=color, s=100, marker="x", zorder=10, linewidths=3)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("EE Trajectory Comparison: Sent (solid) vs Actual (dashed)")
    ax.legend()

    # Equal aspect ratio
    all_pts = np.vstack([a["sent"], a["act"], b["sent"], b["act"]])
    mid = all_pts.mean(axis=0)
    half = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 0.05)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    out_path = args.output or str(base / "compare_traj_3d.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # --- Error summary ---
    print(f"\n{name_a}: mean={np.mean(a['pos_err']):.3f}mm  max={np.max(a['pos_err']):.3f}mm")
    print(f"{name_b}: mean={np.mean(b['pos_err']):.3f}mm  max={np.max(b['pos_err']):.3f}mm")

    plt.show()


if __name__ == "__main__":
    main()

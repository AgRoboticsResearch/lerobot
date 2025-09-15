#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from lerobot.scripts.ik.ik_traj_replay import read_csv_trajectory, _rpy_deg_to_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Cartesian trajectory CSV (t,x,y,z[,r,p,y])")
    parser.add_argument("--traj-csv", type=Path, required=True, help="CSV file with t,x,y,z[,roll_deg,pitch_deg,yaw_deg]")
    parser.add_argument("--save", type=Path, default=None, help="Save figure to path instead of showing")
    parser.add_argument("--title", type=str, default="Trajectory Visualization", help="Plot title")
    args = parser.parse_args()

    traj = read_csv_trajectory(args.traj_csv)
    if len(traj) == 0:
        raise SystemExit("Empty trajectory")

    t = np.array([p.t for p in traj], dtype=float)
    x = np.array([p.pos[0] for p in traj], dtype=float)
    y = np.array([p.pos[1] for p in traj], dtype=float)
    z = np.array([p.pos[2] for p in traj], dtype=float)

    have_rpy = all(p.rpy_deg is not None for p in traj)
    if have_rpy:
        r = np.array([p.rpy_deg[0] for p in traj], dtype=float)
        pch = np.array([p.rpy_deg[1] for p in traj], dtype=float)
        yv = np.array([p.rpy_deg[2] for p in traj], dtype=float)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(args.title)

    # 3D path
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    # Color by time
    c = (t - t.min()) / max(1e-9, (t.max() - t.min()))
    ax3d.plot(x, y, z, color="C0", linewidth=2, alpha=0.9)
    ax3d.scatter([x[0]], [y[0]], [z[0]], color="green", s=40, label="start")
    ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=40, label="end")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend(loc="best")
    try:
        ax3d.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    except Exception:
        pass
    ax3d.grid(True, alpha=0.3)

    # If orientation present, draw a few orientation arrows along path
    if have_rpy:
        idxs = np.linspace(0, len(traj) - 1, num=min(10, len(traj)), dtype=int)
        scale = 0.03  # arrow length
        for i in idxs:
            R = _rpy_deg_to_matrix(r[i], pch[i], yv[i])
            o = np.array([x[i], y[i], z[i]])
            # Draw x (red), y (green), z (blue) axes of the EE
            ax3d.quiver(o[0], o[1], o[2], R[0, 0], R[1, 0], R[2, 0], length=scale, color="r", alpha=0.7)
            ax3d.quiver(o[0], o[1], o[2], R[0, 1], R[1, 1], R[2, 1], length=scale, color="g", alpha=0.7)
            ax3d.quiver(o[0], o[1], o[2], R[0, 2], R[1, 2], R[2, 2], length=scale, color="b", alpha=0.7)

    # Time plots
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, x, label="x [m]")
    ax.plot(t, y, label="y [m]")
    ax.plot(t, z, label="z [m]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if have_rpy:
        ax2 = fig.add_subplot(2, 2, 4)
        ax2.plot(t, r, label="roll [deg]")
        ax2.plot(t, pch, label="pitch [deg]")
        ax2.plot(t, yv, label="yaw [deg]")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("orientation [deg]")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()



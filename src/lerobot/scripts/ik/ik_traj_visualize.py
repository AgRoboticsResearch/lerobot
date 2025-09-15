#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from lerobot.scripts.ik.ik_traj_replay import read_csv_trajectory, _rpy_deg_to_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Cartesian trajectory CSV (t,x,y,z[,r,p,y]) and optional actual overlay")
    parser.add_argument("--traj-csv", type=Path, required=True, help="Planned CSV: t,x,y,z[,roll_deg,pitch_deg,yaw_deg]")
    parser.add_argument("--actual-csv", type=Path, default=None, help="Actual CSV from executor: t,x,y,z,roll_deg,pitch_deg,yaw_deg,...")
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

    # If actual provided, load raw columns via numpy to allow flexible columns
    have_actual = False
    if args.actual_csv is not None and args.actual_csv.exists():
        try:
            import csv
            ts_a = []
            xs_a = []
            ys_a = []
            zs_a = []
            rs_a = []
            ps_a = []
            ysaw_a = []
            with args.actual_csv.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row or row[0].lower() in {"t", "time"}:
                        continue
                    vals = [float(v) for v in row if v != ""]
                    if len(vals) >= 4:
                        ts_a.append(vals[0]); xs_a.append(vals[1]); ys_a.append(vals[2]); zs_a.append(vals[3])
                    if len(vals) >= 7:
                        rs_a.append(vals[4]); ps_a.append(vals[5]); ysaw_a.append(vals[6])
            t_a = np.array(ts_a, dtype=float) if ts_a else None
            x_a = np.array(xs_a, dtype=float) if xs_a else None
            y_a = np.array(ys_a, dtype=float) if ys_a else None
            z_a = np.array(zs_a, dtype=float) if zs_a else None
            have_actual = t_a is not None and x_a is not None
        except Exception:
            have_actual = False

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(args.title)

    # 3D path
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    # Color by time
    c = (t - t.min()) / max(1e-9, (t.max() - t.min()))
    ax3d.plot(x, y, z, color="C0", linewidth=2, alpha=0.9, label="planned")
    if have_actual:
        ax3d.plot(x_a, y_a, z_a, color="C1", linewidth=2, alpha=0.9, label="actual")
    ax3d.scatter([x[0]], [y[0]], [z[0]], color="green", s=40, label="start")
    ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=40, label="end (planned)")
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
    ax.plot(t, x, label="x [m] planned")
    ax.plot(t, y, label="y [m] planned")
    ax.plot(t, z, label="z [m] planned")
    if have_actual:
        ax.plot(t_a, x_a, "--", label="x [m] actual")
        ax.plot(t_a, y_a, "--", label="y [m] actual")
        ax.plot(t_a, z_a, "--", label="z [m] actual")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = fig.add_subplot(2, 2, 4)
    if have_rpy:
        ax2.plot(t, r, label="roll [deg] planned")
        ax2.plot(t, pch, label="pitch [deg] planned")
        ax2.plot(t, yv, label="yaw [deg] planned")
    if have_actual and len(rs_a) == len(xs_a) and rs_a:
        ax2.plot(t_a, rs_a, "--", label="roll [deg] actual")
        ax2.plot(t_a, ps_a, "--", label="pitch [deg] actual")
        ax2.plot(t_a, ysaw_a, "--", label="yaw [deg] actual")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("orientation [deg]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Optional error plot if actual present: L2 position error over time
    if have_actual:
        # Resample actual to planned timestamps via nearest for quick viz
        def _nearest(x_src, y_src, x_tgt):
            idx = np.searchsorted(x_src, x_tgt)
            idx = np.clip(idx, 1, len(x_src) - 1)
            left = idx - 1
            right = idx
            choose_left = (np.abs(x_tgt - x_src[left]) <= np.abs(x_tgt - x_src[right]))
            return y_src[left] * choose_left + y_src[right] * (~choose_left)
        x_a_i = _nearest(t_a, x_a, t)
        y_a_i = _nearest(t_a, y_a, t)
        z_a_i = _nearest(t_a, z_a, t)
        err = np.sqrt((x - x_a_i) ** 2 + (y - y_a_i) ** 2 + (z - z_a_i) ** 2)
        ax_err = fig.add_subplot(2, 2, 3)
        ax_err.plot(t, err, color="C3", label="pos error [m]")
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("error [m]")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()



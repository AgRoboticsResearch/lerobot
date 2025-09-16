#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from lerobot.scripts.ik.ik_traj_replay import read_csv_trajectory, _rpy_deg_to_matrix
from lerobot.model.kinematics import RobotKinematics


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Cartesian trajectory CSV (t,x,y,z[,r,p,y]) and optional actual overlay")
    parser.add_argument("--traj-csv", type=Path, required=True, help="Planned CSV: t,x,y,z[,roll_deg,pitch_deg,yaw_deg]")
    parser.add_argument("--actual-csv", type=Path, default=None, help="Actual CSV from executor: t,x,y,z,roll_deg,pitch_deg,yaw_deg,...")
    parser.add_argument("--save", type=Path, default=None, help="Save figure to path instead of showing")
    parser.add_argument("--title", type=str, default="Trajectory Visualization", help="Plot title")
    # For joints-only CSV visualization
    parser.add_argument("--joints-csv", type=Path, default=None, help="CSV of joints over time (t,<joint_deg...>)")
    parser.add_argument("--urdf", type=Path, default=None, help="URDF needed to compute FK from joints")
    parser.add_argument("--target-frame", type=str, default="gripper_frame_link", help="End-effector frame in URDF")
    parser.add_argument("--joint-names", type=str, default=None, help="Comma-separated joint names to match CSV columns order")
    parser.add_argument("--show-frames", action="store_true", help="Draw small EE orientation frames along path (less clean)")
    parser.add_argument("--show-error", action="store_true", help="Plot position error over time (disabled by default)")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip actual/joints data to planned time range")
    parser.add_argument("--align-start", action="store_true", help="Translate actual path so its first point matches planned start (plotting only)")
    # IK-from-planned overlay
    parser.add_argument("--ik-from-traj", action="store_true", help="Compute IK path from planned CSV and overlay")
    parser.add_argument("--ik-position-weight", type=float, default=1.0, help="IK position weight for overlay")
    parser.add_argument("--ik-orientation-weight", type=float, default=0.0, help="IK orientation weight for overlay")
    parser.add_argument("--ik-initial-joints", type=float, nargs="+", default=None, help="Initial joints (deg) for IK overlay")
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
    rs_a: list[float] = []
    ps_a: list[float] = []
    ysaw_a: list[float] = []
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

    # If joints-csv is provided, compute EE path via FK
    if args.joints_csv is not None and args.joints_csv.exists():
        if args.urdf is None:
            raise SystemExit("--urdf is required when using --joints-csv")
        joint_names = [n.strip() for n in args.joint_names.split(",")] if args.joint_names else None
        kin = RobotKinematics(str(args.urdf), args.target_frame, joint_names)
        import csv as _csv
        ts_j = []
        qs_j = []
        with args.joints_csv.open("r", newline="") as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                vals = [float(v) for v in row if v != ""]
                ts_j.append(vals[0])
                qs_j.append(vals[1:])
        t_j = np.array(ts_j, dtype=float)
        qs_j = np.array(qs_j, dtype=float)
        xs_j, ys_j, zs_j = [], [], []
        for q in qs_j:
            T = kin.forward_kinematics(q)
            xs_j.append(T[0, 3]); ys_j.append(T[1, 3]); zs_j.append(T[2, 3])
        x_j = np.array(xs_j); y_j = np.array(ys_j); z_j = np.array(zs_j)
        # Treat as "actual" overlay when plotting
        have_actual = True
        t_a, x_a, y_a, z_a = t_j, x_j, y_j, z_j
        rs_a = ps_a = ysaw_a = []

    # Clip actual data to planned time window unless disabled
    if have_actual and not args.no_clip:
        t_min, t_max = float(t.min()), float(t.max())
        mask = (t_a >= t_min) & (t_a <= t_max)
        if mask.any():
            t_a = t_a[mask]
            x_a = x_a[mask]
            y_a = y_a[mask]
            z_a = z_a[mask]
            if rs_a:
                rs_a = list(np.array(rs_a)[mask])
                ps_a = list(np.array(ps_a)[mask])
                ysaw_a = list(np.array(ysaw_a)[mask])

    # Optional: IK-from-planned overlay path
    have_ik = False
    if args.ik_from_traj:
        if args.urdf is None:
            raise SystemExit("--urdf is required when using --ik-from-traj")
        joint_names = [n.strip() for n in args.joint_names.split(",")] if args.joint_names else None
        kin_ik = RobotKinematics(str(args.urdf), args.target_frame, joint_names)
        nj = len(kin_ik.joint_names)
        if args.ik_initial_joints is not None:
            if len(args.ik_initial_joints) != nj:
                raise SystemExit(f"--ik-initial-joints length {len(args.ik_initial_joints)} != joint count {nj}")
            qk_prev = np.array(args.ik_initial_joints, dtype=float)
        else:
            qk_prev = np.zeros(nj, dtype=float)

        # If planned has no orientation, keep FK orientation from initial
        keep_Rk = kin_ik.forward_kinematics(qk_prev)[:3, :3]
        x_k, y_k, z_k = [], [], []
        for p in traj:
            if p.rpy_deg is None:
                T = np.eye(4)
                T[:3, :3] = keep_Rk
                T[:3, 3] = np.array(p.pos)
            else:
                T = np.eye(4)
                T[:3, :3] = _rpy_deg_to_matrix(p.rpy_deg[0], p.rpy_deg[1], p.rpy_deg[2])
                T[:3, 3] = np.array(p.pos)
            qk = kin_ik.inverse_kinematics(
                current_joint_pos=qk_prev,
                desired_ee_pose=T,
                position_weight=args.ik_position_weight,
                orientation_weight=args.ik_orientation_weight,
            )
            qk_prev = qk
            Tk = kin_ik.forward_kinematics(qk)
            x_k.append(Tk[0, 3]); y_k.append(Tk[1, 3]); z_k.append(Tk[2, 3])
        x_k = np.array(x_k); y_k = np.array(y_k); z_k = np.array(z_k)
        have_ik = True

    # Optional: align actual start to planned start for clearer visual comparison
    if have_actual and args.align_start and len(x_a) > 0:
        dx = float(x[0] - x_a[0])
        dy = float(y[0] - y_a[0])
        dz = float(z[0] - z_a[0])
        x_a = x_a + dx
        y_a = y_a + dy
        z_a = z_a + dz

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(args.title)

    # 3D path
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    # Color by time
    c = (t - t.min()) / max(1e-9, (t.max() - t.min()))
    ax3d.plot(x, y, z, color="C0", linewidth=2.5, alpha=0.9, label="planned")
    if have_actual:
        ax3d.plot(x_a, y_a, z_a, color="C1", linewidth=2.0, alpha=0.8, linestyle="--", label="actual")
    if 'have_ik' in locals() and have_ik:
        ax3d.plot(x_k, y_k, z_k, color="C5", linewidth=2.0, alpha=0.9, linestyle=":", label="ik-from-planned")
    ax3d.scatter([x[0]], [y[0]], [z[0]], color="green", s=30, label="start")
    ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=30, label="end (planned)")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.view_init(elev=22, azim=45)
    ax3d.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=3, framealpha=0.6)

    # Ensure non-zero axis extents to avoid singular projection for straight-line or static paths
    if have_actual:
        x_all = np.concatenate([x, x_a])
        y_all = np.concatenate([y, y_a])
        z_all = np.concatenate([z, z_a])
    else:
        x_all, y_all, z_all = x, y, z

    dx = float(np.ptp(x_all))
    dy = float(np.ptp(y_all))
    dz = float(np.ptp(z_all))

    # Pad axis limits if any dimension is near zero range
    def _pad_axis(set_lim, vmin, vmax):
        if abs(vmax - vmin) < 1e-6:
            mid = 0.5 * (vmin + vmax)
            pad = 0.01
            set_lim(mid - pad, mid + pad)

    _pad_axis(ax3d.set_xlim, float(x_all.min()), float(x_all.max()))
    _pad_axis(ax3d.set_ylim, float(y_all.min()), float(y_all.max()))
    _pad_axis(ax3d.set_zlim, float(z_all.min()), float(z_all.max()))

    try:
        ax3d.set_box_aspect((max(dx, 1e-6), max(dy, 1e-6), max(dz, 1e-6)))
    except Exception:
        pass
    ax3d.grid(True, alpha=0.3)

    # If orientation present, draw a few orientation arrows along path
    if args.show_frames and have_rpy:
        idxs = np.linspace(0, len(traj) - 1, num=min(6, len(traj)), dtype=int)
        scale = 0.03
        for i in idxs:
            R = _rpy_deg_to_matrix(r[i], pch[i], yv[i])
            o = np.array([x[i], y[i], z[i]])
            ax3d.quiver(o[0], o[1], o[2], R[0, 0], R[1, 0], R[2, 0], length=scale, color="r", alpha=0.6)
            ax3d.quiver(o[0], o[1], o[2], R[0, 1], R[1, 1], R[2, 1], length=scale, color="g", alpha=0.6)
            ax3d.quiver(o[0], o[1], o[2], R[0, 2], R[1, 2], R[2, 2], length=scale, color="b", alpha=0.6)

    # Time plots
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, x, label="x planned", color="C0", linewidth=2.0)
    ax.plot(t, y, label="y planned", color="C2", linewidth=2.0)
    ax.plot(t, z, label="z planned", color="C3", linewidth=2.0)
    if have_actual:
        ax.plot(t_a, x_a, "--", label="x actual", color="C0", alpha=0.8)
        ax.plot(t_a, y_a, "--", label="y actual", color="C2", alpha=0.8)
        ax.plot(t_a, z_a, "--", label="z actual", color="C3", alpha=0.8)
    if 'have_ik' in locals() and have_ik:
        ax.plot(t, x_k, ":", label="x ik", color="C5", alpha=0.9)
        ax.plot(t, y_k, ":", label="y ik", color="C6", alpha=0.9)
        ax.plot(t, z_k, ":", label="z ik", color="C7", alpha=0.9)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=3, framealpha=0.6)

    ax2 = fig.add_subplot(2, 2, 4)
    if have_rpy:
        ax2.plot(t, r, label="roll planned", linewidth=1.8)
        ax2.plot(t, pch, label="pitch planned", linewidth=1.8)
        ax2.plot(t, yv, label="yaw planned", linewidth=1.8)
    if have_actual and rs_a:
        ax2.plot(t_a, rs_a, "--", label="roll actual", alpha=0.8)
        ax2.plot(t_a, ps_a, "--", label="pitch actual", alpha=0.8)
        ax2.plot(t_a, ysaw_a, "--", label="yaw actual", alpha=0.8)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("orientation [deg]")
    ax2.grid(True, alpha=0.3)
    handles, labels = ax2.get_legend_handles_labels()
    if labels:
        ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=3, framealpha=0.6)

    # Optional error plot if requested: L2 position error over time
    if args.show_error and have_actual:
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
        ax_err.plot(t, err, color="C4", linewidth=2.0, label="pos error [m]")
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("error [m]")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), framealpha=0.6)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()



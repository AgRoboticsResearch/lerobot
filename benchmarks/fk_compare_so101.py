#!/usr/bin/env python

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project src on path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def parse_degrees_list(s: str, expected_len: int) -> np.ndarray:
    vals = [float(x) for x in s.split(",")]
    if len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} degrees, got {len(vals)}")
    return np.asarray(vals, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Compare FK at a given joint configuration: desired q vs measured q on SO101"
    )
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--degrees", type=str, required=False, help="Comma-separated 5 joint degrees (single-point mode)")
    parser.add_argument("--target_frame_name", type=str, default="gripper_frame_link")
    parser.add_argument(
        "--joint_names",
        type=str,
        default="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll",
    )
    parser.add_argument("--snap_tolerance_deg", type=float, default=2.0)
    parser.add_argument("--snap_timeout_s", type=float, default=10.0)
    parser.add_argument("--snap_boost_max_relative_target_deg", type=float, default=45.0)
    parser.add_argument("--out_dir", type=str, default="./fk_compare_out")
    parser.add_argument("--plot_units", type=str, choices=["m", "cm"], default="cm", help="Units for plot axes (keeps JSON in meters)")
    parser.add_argument("--context_box_cm", type=float, default=20.0, help="Context cube size around point (cm) for the context plot")
    parser.add_argument("--zoom_box_cm", type=float, default=3.0, help="Zoom cube size (cm) around the points for the zoom plot")
    # Multi-point options from ik_eval_out
    parser.add_argument("--joint_traj_npz", type=str, default=None, help="Path to ik_eval_out/ik_eval_results.npz (expects key 'ik_joints_deg')")
    parser.add_argument("--sample_points", type=int, default=None, help="Randomly sample K indices from NPZ trajectory")
    parser.add_argument("--test_point_indices", type=str, default=None, help="Comma-separated indices into the NPZ joint trajectory")
    parser.add_argument("--sample_seed", type=int, default=0)
    # Joint isolation (e.g., only move wrist_roll)
    parser.add_argument("--isolate_joint_name", type=str, default=None, help="If set, only this joint moves; others fixed to base pose")
    parser.add_argument("--base_degrees", type=str, default=None, help="Comma-separated base joint degrees when isolating a joint")
    parser.add_argument("--use_measured_as_base", action="store_true", help="Use current measured joints as base when isolating a joint")
    parser.add_argument("--isolate_values_deg", type=str, default=None, help="Comma-separated explicit values for the isolated joint (deg)")
    parser.add_argument("--isolate_span_deg", type=str, default=None, help="min,max in deg for the isolated joint sweep")
    parser.add_argument("--isolate_steps", type=int, default=11, help="Number of steps if --isolate_span_deg is used")
    # Plot control
    parser.add_argument("--series_ylim_abs_deg", type=float, default=None, help="If set, apply symmetric y-limits +/- this value to all series subplots [deg]")
    parser.add_argument(
        "--series_ylim_deg",
        type=str,
        default=None,
        help=(
            "Comma-separated per-joint symmetric y-limits [deg] in joint_names order. "
            "Example for 5 DoF: '10,30,30,15,3' sets ±10, ±30, ±30, ±15, ±3."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_names = [s for s in args.joint_names.split(",") if s]
    if len(joint_names) != 5:
        raise ValueError("This script assumes 5 DoF (5 joint_names)")

    q_target = None
    if args.degrees is not None:
        q_target = parse_degrees_list(args.degrees, expected_len=len(joint_names))

    # Initialize hardware robot
    robot_cfg = SO101FollowerConfig(
        port=args.port,
        id="so101_hw",
        use_degrees=True,
        max_relative_target=args.snap_boost_max_relative_target_deg,
        cameras={},
    )
    robot = SO101Follower(robot_cfg)

    # Initialize kinematics
    kin = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame_name,
        joint_names=joint_names,
    )

    robot.connect()
    try:
        # If NPZ is provided, run multi-point comparison; otherwise single-point mode using --degrees
        if args.joint_traj_npz is not None:
            data = np.load(args.joint_traj_npz, allow_pickle=True)
            if "ik_joints_deg" not in data:
                raise ValueError("NPZ must contain 'ik_joints_deg'")
            traj = data["ik_joints_deg"]
            if traj.shape[1] != len(joint_names):
                raise ValueError("Joint count mismatch in NPZ vs joint_names")

            # Planned full curve in EE if available
            planned_full_xyz = None
            ee_path = Path(args.joint_traj_npz).parent / "ee_points.npy"
            if ee_path.exists():
                try:
                    planned_full_xyz = np.load(ee_path)
                except Exception:
                    planned_full_xyz = None
            if planned_full_xyz is None:
                pts = [kin.forward_kinematics(q)[:3, 3] for q in traj]
                planned_full_xyz = np.asarray(pts)

            # Select indices
            if args.test_point_indices:
                indices = [int(x) for x in args.test_point_indices.split(",")]
            elif args.sample_points:
                rng = np.random.default_rng(args.sample_seed)
                N = traj.shape[0]
                indices = rng.choice(N, size=min(args.sample_points, N), replace=False).tolist()
            else:
                # default: use all
                indices = list(range(traj.shape[0]))

            chosen_xyz = []
            achieved_xyz = []
            chosen_q = []
            measured_q = []

            gripper_pos = robot.bus.sync_read("Present_Position")["gripper"]

            # Isolation setup
            iso_idx = None
            base_q = None
            if args.isolate_joint_name:
                if args.isolate_joint_name not in joint_names:
                    raise ValueError(f"isolate_joint_name '{args.isolate_joint_name}' not in joint_names {joint_names}")
                iso_idx = joint_names.index(args.isolate_joint_name)
                if args.base_degrees is not None:
                    base_q = parse_degrees_list(args.base_degrees, expected_len=len(joint_names))
                elif args.use_measured_as_base:
                    present0 = robot.bus.sync_read("Present_Position")
                    base_q = np.array([present0[n] for n in joint_names], dtype=np.float64)
                else:
                    base_q = traj[indices[0]].copy()

            # Build target sequence, possibly overriding with isolated joint sweep
            target_q_list = []
            if iso_idx is not None and (args.isolate_values_deg or args.isolate_span_deg):
                if args.isolate_values_deg:
                    iso_values = np.asarray([float(x) for x in args.isolate_values_deg.split(",")], dtype=np.float64)
                else:
                    lo, hi = [float(x) for x in args.isolate_span_deg.split(",")]
                    steps = max(2, int(args.isolate_steps))
                    iso_values = np.linspace(lo, hi, steps)
                for val in iso_values:
                    q_t = base_q.copy()
                    q_t[iso_idx] = float(val)
                    target_q_list.append(q_t)
                indices = list(range(len(target_q_list)))
            else:
                # Use NPZ-selected points; if isolated joint has negligible variation, warn and create a small ±10 deg sweep
                if iso_idx is not None:
                    q_vals = traj[indices, iso_idx]
                    if float(np.ptp(q_vals)) < 1e-3:
                        print(f"[WARN] Isolated joint '{args.isolate_joint_name}' has ~no variation in NPZ selection; creating ±10 deg sweep around base")
                        iso_values = np.linspace(base_q[iso_idx] - 10.0, base_q[iso_idx] + 10.0, 11)
                        for val in iso_values:
                            q_t = base_q.copy()
                            q_t[iso_idx] = float(val)
                            target_q_list.append(q_t)
                        indices = list(range(len(target_q_list)))
                if not target_q_list:
                    for i in indices:
                        q_t_full = traj[i]
                        if iso_idx is not None:
                            q_t = base_q.copy()
                            q_t[iso_idx] = q_t_full[iso_idx]
                        else:
                            q_t = q_t_full
                        target_q_list.append(q_t)

            for q_t in target_q_list:
                action = {f"{name}.pos": float(val) for name, val in zip(joint_names, q_t)}
                action["gripper.pos"] = gripper_pos

                t0 = time.perf_counter()
                while time.perf_counter() - t0 < args.snap_timeout_s:
                    robot.send_action(action)
                    present = robot.bus.sync_read("Present_Position")
                    q_meas = np.array([present[n] for n in joint_names], dtype=np.float64)
                    if np.max(np.abs(q_meas - q_t)) <= args.snap_tolerance_deg:
                        break
                    time.sleep(0.02)

                T_t = kin.forward_kinematics(q_t)
                T_m = kin.forward_kinematics(q_meas)
                chosen_xyz.append(T_t[:3, 3].copy())
                achieved_xyz.append(T_m[:3, 3].copy())
                chosen_q.append(q_t.copy())
                measured_q.append(q_meas.copy())

            chosen_xyz = np.asarray(chosen_xyz)
            achieved_xyz = np.asarray(achieved_xyz)
            chosen_q = np.asarray(chosen_q)
            measured_q = np.asarray(measured_q)
            pos_err = achieved_xyz - chosen_xyz
            pos_err_norm = np.linalg.norm(pos_err, axis=1)
            joint_err_deg = measured_q - chosen_q
            abs_joint_err_deg = np.abs(joint_err_deg)

            # Save CSV summary
            import csv as _csv
            with open(out_dir / "fk_compare_multi.csv", "w", newline="") as f:
                w = _csv.writer(f)
                header = [
                    "idx","t_x","t_y","t_z","m_x","m_y","m_z","e_x","e_y","e_z","e_norm",
                ] + [f"t_q{i}" for i in range(chosen_q.shape[1])] + [f"m_q{i}" for i in range(measured_q.shape[1])] + [f"e_q{i}" for i in range(joint_err_deg.shape[1])]
                w.writerow(header)
                for row_i, idx in enumerate(indices):
                    w.writerow([
                        idx,
                        *chosen_xyz[row_i].tolist(),
                        *achieved_xyz[row_i].tolist(),
                        *pos_err[row_i].tolist(),
                        float(pos_err_norm[row_i]),
                        *chosen_q[row_i].tolist(),
                        *measured_q[row_i].tolist(),
                        *joint_err_deg[row_i].tolist(),
                    ])

            # Plot overlay: planned curve (blue), chosen points (red), achieved (green) + connections
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                scale = 100.0 if args.plot_units == "cm" else 1.0
                unit_label = "cm" if args.plot_units == "cm" else "m"
                curve = planned_full_xyz * scale
                ch = chosen_xyz * scale
                ac = achieved_xyz * scale

                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot(curve[:,0], curve[:,1], curve[:,2], c="C0", alpha=0.6, label="planned_curve")
                ax.scatter(ch[:,0], ch[:,1], ch[:,2], c="r", s=36, label="chosen_points")
                ax.scatter(ac[:,0], ac[:,1], ac[:,2], c="g", s=36, label="achieved")
                for k in range(ch.shape[0]):
                    ax.plot([ch[k,0], ac[k,0]],[ch[k,1], ac[k,1]],[ch[k,2], ac[k,2]], c="g", alpha=0.4)
                ax.set_xlabel(f"X [{unit_label}]")
                ax.set_ylabel(f"Y [{unit_label}]")
                ax.set_zlabel(f"Z [{unit_label}]")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "fk_compare_multi_plot.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] Plotting failed: {e}")

            # Joint-space 3D PCA plot: IK (chosen_q) vs measured_q, with connections
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                # PCA via SVD on concatenated joint data (centered)
                X = np.vstack([chosen_q, measured_q])  # shape (2N, D)
                mean = X.mean(axis=0, keepdims=True)
                Xc = X - mean
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                W = Vt[:3].T  # D x 3
                ch3 = (chosen_q - mean) @ W
                me3 = (measured_q - mean) @ W

                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(ch3[:,0], ch3[:,1], ch3[:,2], c="r", s=36, label="IK joints (chosen)")
                ax.scatter(me3[:,0], me3[:,1], me3[:,2], c="g", s=36, label="Measured joints")
                for k in range(ch3.shape[0]):
                    ax.plot([ch3[k,0], me3[k,0]],[ch3[k,1], me3[k,1]],[ch3[k,2], me3[k,2]], c="gray", alpha=0.4)
                ax.set_xlabel("PC1 [deg]")
                ax.set_ylabel("PC2 [deg]")
                ax.set_zlabel("PC3 [deg]")
                ax.set_title("Joint-space (PCA-3D): IK vs Measured")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "fk_compare_joint_space_pca.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] Joint-space PCA plotting failed: {e}")

            # Per-joint parity plots: measured vs IK (y vs x) with y=x reference
            try:
                import matplotlib.pyplot as plt

                cols = 3
                rows = 2
                fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
                axes = axes.flatten()
                for j, name in enumerate(joint_names):
                    ax = axes[j]
                    x = chosen_q[:, j]
                    y = measured_q[:, j]
                    mn = float(min(x.min(), y.min()))
                    mx = float(max(x.max(), y.max()))
                    pad = 0.05 * (mx - mn + 1e-6)
                    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], c="k", lw=1, ls="--", alpha=0.6, label="y=x")
                    ax.scatter(x, y, c="tab:green", s=18, alpha=0.9)
                    ax.set_xlabel("IK [deg]")
                    ax.set_ylabel("Measured [deg]")
                    ax.set_title(name)
                for k in range(len(joint_names), rows * cols):
                    fig.delaxes(axes[k])
                fig.suptitle("Per-joint parity: measured vs IK (deg)")
                fig.tight_layout()
                fig.savefig(out_dir / "fk_compare_joint_parity.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] Joint parity plotting failed: {e}")

            # Per-joint series plots across sampled points: IK vs measured with connections
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
                idx = np.arange(len(indices))
                per_joint_limits = None
                if args.series_ylim_deg is not None:
                    try:
                        vals = [float(x) for x in args.series_ylim_deg.split(",")]
                        if len(vals) != len(joint_names):
                            raise ValueError
                        per_joint_limits = vals
                    except Exception:
                        print("[WARN] --series_ylim_deg must have one value per joint; ignoring")
                        per_joint_limits = None
                for j, name in enumerate(joint_names):
                    ax = axes[j]
                    ax.plot(idx, chosen_q[:, j], "o-", c="r", ms=4, lw=1, label="IK")
                    ax.plot(idx, measured_q[:, j], "o-", c="g", ms=4, lw=1, label="Measured")
                    ax.set_ylabel(f"{name} [deg]")
                    if per_joint_limits is not None:
                        lim = float(per_joint_limits[j])
                        ax.set_ylim(-lim, lim)
                    elif args.series_ylim_abs_deg is not None:
                        lim = float(args.series_ylim_abs_deg)
                        ax.set_ylim(-lim, lim)
                    ax.grid(True, alpha=0.3)
                    if j == 0:
                        ax.legend(loc="upper right")
                axes[-1].set_xlabel("sample index")
                fig.tight_layout()
                fig.savefig(out_dir / "fk_compare_joint_series.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] Joint series plotting failed: {e}")

            # Save JSON aggregate
            joint_mean_abs_deg = abs_joint_err_deg.mean(axis=0).tolist()
            joint_median_abs_deg = np.median(abs_joint_err_deg, axis=0).tolist()
            joint_max_abs_deg = abs_joint_err_deg.max(axis=0).tolist()
            joint_rmse_deg = np.sqrt((joint_err_deg ** 2).mean(axis=0)).tolist()

            summary = {
                "indices": indices,
                "mean_err_m": float(pos_err_norm.mean()),
                "median_err_m": float(np.median(pos_err_norm)),
                "max_err_m": float(pos_err_norm.max()),
                "min_err_m": float(pos_err_norm.min()),
                "joint_names": joint_names,
                "joint_mean_abs_err_deg": joint_mean_abs_deg,
                "joint_median_abs_err_deg": joint_median_abs_deg,
                "joint_max_abs_err_deg": joint_max_abs_deg,
                "joint_rmse_deg": joint_rmse_deg,
            }
            (out_dir / "fk_compare_multi.json").write_text(json.dumps(summary, indent=2))

            print(json.dumps(summary, indent=2))
            print("Output dir:", out_dir)

            # Optional: per-joint error bar plot (deg)
            try:
                import matplotlib.pyplot as plt
                x = np.arange(len(joint_names))
                fig = plt.figure(figsize=(7, 4))
                ax = fig.add_subplot(111)
                ax.bar(x - 0.2, joint_mean_abs_deg, width=0.4, label="mean |err|")
                ax.bar(x + 0.2, joint_max_abs_deg, width=0.4, label="max |err|")
                ax.set_xticks(x)
                ax.set_xticklabels(joint_names, rotation=20)
                ax.set_ylabel("Error [deg]")
                ax.set_title("Per-joint absolute position error (deg)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "fk_compare_joint_error_deg.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] Joint error plotting failed: {e}")

            return
        # ---- Single-point mode below ----
        if q_target is None:
            raise ValueError("Provide either --degrees (single point) or --joint_traj_npz (multi points)")

        action = {f"{name}.pos": float(val) for name, val in zip(joint_names, q_target)}
        action["gripper.pos"] = robot.bus.sync_read("Present_Position")["gripper"]

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.snap_timeout_s:
            robot.send_action(action)
            present = robot.bus.sync_read("Present_Position")
            q_meas = np.array([present[n] for n in joint_names], dtype=np.float64)
            if np.max(np.abs(q_meas - q_target)) <= args.snap_tolerance_deg:
                break
            time.sleep(0.02)

        # Compute FK for target and measured
        T_target = kin.forward_kinematics(q_target)
        T_meas = kin.forward_kinematics(q_meas)

        p_target = T_target[:3, 3]
        p_meas = T_meas[:3, 3]
        pos_err = p_meas - p_target

        # Save summary
        stats = {
            "q_target_deg": q_target.tolist(),
            "q_meas_deg": q_meas.tolist(),
            "target_xyz_m": p_target.tolist(),
            "measured_xyz_m": p_meas.tolist(),
            "position_error_m": pos_err.tolist(),
            "position_error_norm_m": float(np.linalg.norm(pos_err)),
        }
        (out_dir / "fk_compare.json").write_text(json.dumps(stats, indent=2))

        # Plot
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            # Convert to desired units for plotting
            scale = 100.0 if args.plot_units == "cm" else 1.0
            unit_label = "cm" if args.plot_units == "cm" else "m"
            pt = p_target * scale
            pm = p_meas * scale
            delta_norm = float(np.linalg.norm(pos_err) * scale)

            def set_box(ax, center, size):
                half = size / 2.0
                for i, label in enumerate(["x", "y", "z"]):
                    ax.set(**{f"set_{label}lim": (center[i] - half, center[i] + half)})

            # Context plot with fixed large cube
            fig1 = plt.figure(figsize=(6, 6))
            ax1 = fig1.add_subplot(111, projection="3d")
            ax1.scatter([pt[0]], [pt[1]], [pt[2]], c="C0", s=60, label="target (URDF FK)")
            ax1.scatter([pm[0]], [pm[1]], [pm[2]], c="C2", s=60, label="measured (enc FK)")
            ax1.plot([pt[0], pm[0]], [pt[1], pm[1]], [pt[2], pm[2]], c="C2", alpha=0.6, label=f"delta {delta_norm:.2f} {unit_label}")
            ax1.set_xlabel(f"X [{unit_label}]")
            ax1.set_ylabel(f"Y [{unit_label}]")
            ax1.set_zlabel(f"Z [{unit_label}]")
            center = (pt + pm) / 2.0
            box_size = args.context_box_cm if args.plot_units == "cm" else args.context_box_cm / 100.0
            set_box(ax1, center, box_size)
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig(out_dir / "fk_compare_plot_context.png", dpi=150)
            plt.close(fig1)

            # Zoom plot with small cube around the points
            fig2 = plt.figure(figsize=(6, 6))
            ax2 = fig2.add_subplot(111, projection="3d")
            ax2.scatter([pt[0]], [pt[1]], [pt[2]], c="C0", s=60, label="target (URDF FK)")
            ax2.scatter([pm[0]], [pm[1]], [pm[2]], c="C2", s=60, label="measured (enc FK)")
            ax2.plot([pt[0], pm[0]], [pt[1], pm[1]], [pt[2], pm[2]], c="C2", alpha=0.8, linewidth=2.0, label=f"delta {delta_norm:.2f} {unit_label}")
            ax2.set_xlabel(f"X [{unit_label}]")
            ax2.set_ylabel(f"Y [{unit_label}]")
            ax2.set_zlabel(f"Z [{unit_label}]")
            zoom_size = args.zoom_box_cm if args.plot_units == "cm" else args.zoom_box_cm / 100.0
            set_box(ax2, center, zoom_size)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig(out_dir / "fk_compare_plot_zoom.png", dpi=150)
            plt.close(fig2)
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

        print(json.dumps(stats, indent=2))
        print("Output dir:", out_dir)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()



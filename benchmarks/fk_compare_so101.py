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
    parser.add_argument("--degrees", type=str, required=True, help="Comma-separated 5 joint degrees")
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_names = [s for s in args.joint_names.split(",") if s]
    if len(joint_names) != 5:
        raise ValueError("This script assumes 5 DoF (5 joint_names)")

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
        # Command the target and wait until within tolerance
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

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
            # Convert to desired units for plotting
            scale = 100.0 if args.plot_units == "cm" else 1.0
            unit_label = "cm" if args.plot_units == "cm" else "m"
            pt = p_target * scale
            pm = p_meas * scale
            ax.scatter([pt[0]], [pt[1]], [pt[2]], c="C0", s=60, label="target (URDF FK)")
            ax.scatter([pm[0]], [pm[1]], [pm[2]], c="C2", s=60, label="measured (enc FK)")
            ax.plot([pt[0], pm[0]], [pt[1], pm[1]], [pt[2], pm[2]], c="C2", alpha=0.6, label="delta")
            ax.set_xlabel(f"X [{unit_label}]")
            ax.set_ylabel(f"Y [{unit_label}]")
            ax.set_zlabel(f"Z [{unit_label}]")
            # Annotate error magnitude
            ax.text(pm[0], pm[1], pm[2], f"|Δ|={np.linalg.norm(pos_err)*scale:.2f} {unit_label}", color="C2")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "fk_compare_plot.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

        print(json.dumps(stats, indent=2))
        print("Output dir:", out_dir)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()



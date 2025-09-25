#!/usr/bin/env python

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Ensure project src on path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.model.kinematics import RobotKinematics


@dataclass
class EvalConfig:
    urdf_path: str
    target_frame_name: str = "gripper_frame_link"
    # Joint names used by the URDF model (exclude gripper for IK if desired)
    joint_names: list[str] = None
    # Middle joint configuration in degrees (same length as joint_names)
    mid_joints_deg: list[float] | None = None
    # Trajectory params
    traj_type: str = "circle"  # circle | line | line_xyz
    line_axis: str = "z"  # x | y | z
    line_amplitude_m: float = 0.05
    num_points_per_axis: int | None = None
    num_points: int = 200
    radius_m: float = 0.03
    z_amplitude_m: float = 0.02
    cycles: float = 1.0
    # IK weights
    position_weight: float = 1.0
    orientation_weight: float = 0.0
    # Output
    out_dir: str = "./ik_eval_out"


def generate_traj(center_T: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    """
    Generate a trajectory around/along the center pose.

    traj_type = circle:
      - XY circle with radius and Z sinusoidal modulation
    traj_type = line:
      - Straight line along chosen axis (x|y|z) with sinusoidal progression
    """
    N = cfg.num_points
    poses = np.repeat(center_T[None, :, :], N, axis=0)
    center_pos = center_T[:3, 3].copy()

    if cfg.traj_type == "line":
        t = np.linspace(0.0, 2.0 * math.pi * cfg.cycles, N, endpoint=False)
        delta = cfg.line_amplitude_m * np.sin(t)
        xyz = np.tile(center_pos, (N, 1))
        if cfg.line_axis == "x":
            xyz[:, 0] += delta
        elif cfg.line_axis == "y":
            xyz[:, 1] += delta
        else:
            xyz[:, 2] += delta
        poses[:, 0, 3] = xyz[:, 0]
        poses[:, 1, 3] = xyz[:, 1]
        poses[:, 2, 3] = xyz[:, 2]
        return poses

    if cfg.traj_type == "line_xyz":
        # Split points among X, Y, Z
        if cfg.num_points_per_axis is None:
            base = max(1, N // 3)
            remainder = max(0, N - base * 3)
            counts = [base, base, base]
            for i in range(remainder):
                counts[i] += 1
        else:
            counts = [cfg.num_points_per_axis] * 3
        segs = []
        for axis, n in zip(["x", "y", "z"], counts, strict=False):
            t = np.linspace(0.0, 2.0 * math.pi * cfg.cycles, n, endpoint=False)
            delta = cfg.line_amplitude_m * np.sin(t)
            xyz = np.tile(center_pos, (n, 1))
            if axis == "x":
                xyz[:, 0] += delta
            elif axis == "y":
                xyz[:, 1] += delta
            else:
                xyz[:, 2] += delta
            seg = np.repeat(center_T[None, :, :], n, axis=0)
            seg[:, 0, 3] = xyz[:, 0]
            seg[:, 1, 3] = xyz[:, 1]
            seg[:, 2, 3] = xyz[:, 2]
            segs.append(seg)
        return np.concatenate(segs, axis=0)

    # circle (default)
    theta = np.linspace(0.0, 2.0 * math.pi * cfg.cycles, N, endpoint=False)
    x = center_pos[0] + cfg.radius_m * np.cos(theta)
    y = center_pos[1] + cfg.radius_m * np.sin(theta)
    z = center_pos[2] + cfg.z_amplitude_m * np.sin(2.0 * theta)
    poses[:, 0, 3] = x
    poses[:, 1, 3] = y
    poses[:, 2, 3] = z
    return poses


def run_eval(cfg: EvalConfig) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default joint names matching SO101 order used in the codebase (excluding gripper)
    if cfg.joint_names is None:
        cfg.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]

    # Default mid joints if not provided: 0 deg for all
    if cfg.mid_joints_deg is None:
        cfg.mid_joints_deg = [0.0] * len(cfg.joint_names)

    kin = RobotKinematics(
        urdf_path=cfg.urdf_path,
        target_frame_name=cfg.target_frame_name,
        joint_names=cfg.joint_names,
    )

    mid_joints = np.array(cfg.mid_joints_deg, dtype=np.float64)
    center_T = kin.forward_kinematics(mid_joints)

    desired_poses = generate_traj(center_T, cfg)

    achieved_poses = []
    ik_joints = []
    seed = mid_joints.copy()
    for T_des in desired_poses:
        q_deg = kin.inverse_kinematics(
            current_joint_pos=seed,
            desired_ee_pose=T_des,
            position_weight=cfg.position_weight,
            orientation_weight=cfg.orientation_weight,
        )
        ik_joints.append(q_deg)
        T_fk = kin.forward_kinematics(q_deg)
        achieved_poses.append(T_fk)
        seed = q_deg

    ik_joints = np.stack(ik_joints, axis=0)
    achieved_poses = np.stack(achieved_poses, axis=0)

    # Compute position errors
    desired_xyz = desired_poses[:, :3, 3]
    achieved_xyz = achieved_poses[:, :3, 3]
    pos_err = achieved_xyz - desired_xyz
    pos_err_norm = np.linalg.norm(pos_err, axis=1)

    # Stats
    stats = {
        "mean_pos_err_m": float(pos_err_norm.mean()),
        "median_pos_err_m": float(np.median(pos_err_norm)),
        "max_pos_err_m": float(pos_err_norm.max()),
        "min_pos_err_m": float(pos_err_norm.min()),
    }

    # Save CSVs
    csv_path = out_dir / "trajectory_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "idx",
            "des_x","des_y","des_z",
            "ach_x","ach_y","ach_z",
            "err_x","err_y","err_z","err_norm",
        ] + [f"q{i}_deg" for i in range(ik_joints.shape[1])]
        writer.writerow(header)
        for i in range(cfg.num_points):
            row = [
                i,
                desired_xyz[i, 0], desired_xyz[i, 1], desired_xyz[i, 2],
                achieved_xyz[i, 0], achieved_xyz[i, 1], achieved_xyz[i, 2],
                pos_err[i, 0], pos_err[i, 1], pos_err[i, 2], pos_err_norm[i],
            ] + list(ik_joints[i])
            writer.writerow(row)

    # Save NPZ
    np.savez(
        out_dir / "ik_eval_results.npz",
        desired_poses=desired_poses,
        achieved_poses=achieved_poses,
        ik_joints_deg=ik_joints,
        position_error=pos_err,
        position_error_norm=pos_err_norm,
        config=json.dumps(asdict(cfg)),
        allow_pickle=True,
    )

    # Quick 3D plot
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(desired_xyz[:, 0], desired_xyz[:, 1], desired_xyz[:, 2], label="desired", c="C0")
        ax.plot(achieved_xyz[:, 0], achieved_xyz[:, 1], achieved_xyz[:, 2], label="achieved", c="C1")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.legend()
        fig.tight_layout()
        fig_path = out_dir / "trajectory_plot.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        # Plotting is optional
        print(f"[WARN] Plotting failed: {e}")

    # Print summary
    print("IK evaluation done. Output in:", out_dir)
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate IK accuracy for SO101 using a planned EE trajectory.")
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--target_frame_name", type=str, default="gripper_frame_link")
    parser.add_argument("--joint_names", type=str, default="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll")
    parser.add_argument("--mid_joints_deg", type=str, default=None, help="Comma-separated degrees. Defaults to zeros.")
    parser.add_argument("--traj_type", type=str, choices=["circle", "line", "line_xyz"], default="circle")
    parser.add_argument("--line_axis", type=str, choices=["x", "y", "z"], default="x")
    parser.add_argument("--line_amplitude_m", type=float, default=0.05)
    parser.add_argument("--num_points_per_axis", type=int, default=None)
    parser.add_argument("--num_points", type=int, default=200)
    parser.add_argument("--radius_m", type=float, default=0.03)
    parser.add_argument("--z_amplitude_m", type=float, default=0.02)
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--position_weight", type=float, default=1.0)
    parser.add_argument("--orientation_weight", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str, default="./ik_eval_out")
    args = parser.parse_args()

    joint_names = [s for s in args.joint_names.split(",") if s]
    mid_joints = None
    if args.mid_joints_deg is not None:
        mid_joints = [float(x) for x in args.mid_joints_deg.split(",")]

    cfg = EvalConfig(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame_name,
        joint_names=joint_names,
        mid_joints_deg=mid_joints,
        traj_type=args.traj_type,
        line_axis=args.line_axis,
        line_amplitude_m=args.line_amplitude_m,
        num_points_per_axis=args.num_points_per_axis,
        num_points=args.num_points,
        radius_m=args.radius_m,
        z_amplitude_m=args.z_amplitude_m,
        cycles=args.cycles,
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
        out_dir=args.out_dir,
    )

    run_eval(cfg)


if __name__ == "__main__":
    main()



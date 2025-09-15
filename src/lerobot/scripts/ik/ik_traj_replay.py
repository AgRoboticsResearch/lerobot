#!/usr/bin/env python3

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from lerobot.model.kinematics import RobotKinematics


def _rpy_deg_to_matrix(r_deg: float, p_deg: float, y_deg: float) -> np.ndarray:
    r = math.radians(r_deg)
    p = math.radians(p_deg)
    y = math.radians(y_deg)

    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _pose_from_pos_rpy(position_m: Iterable[float], rpy_deg: Optional[Iterable[float]]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    position_m = list(position_m)
    T[:3, 3] = np.array(position_m, dtype=float)
    if rpy_deg is not None:
        rpy_deg = list(rpy_deg)
        T[:3, :3] = _rpy_deg_to_matrix(rpy_deg[0], rpy_deg[1], rpy_deg[2])
    return T


@dataclass
class TrajPoint:
    t: float
    pos: Tuple[float, float, float]
    rpy_deg: Optional[Tuple[float, float, float]]


def generate_linear_sample(
    center: Tuple[float, float, float],
    offset: Tuple[float, float, float],
    duration_s: float,
    rate_hz: float,
    keep_rpy: Optional[Tuple[float, float, float]] = None,
) -> List[TrajPoint]:
    """
    Generate a simple linear trajectory from center to center+offset.
    """
    num = max(2, int(duration_s * rate_hz))
    ts = np.linspace(0.0, duration_s, num=num)
    center = np.array(center, dtype=float)
    end = center + np.array(offset, dtype=float)
    positions = (center[None, :] + (end - center)[None, :] * (ts / duration_s)[:, None]).astype(float)

    points: List[TrajPoint] = []
    for i, t in enumerate(ts):
        rpy = tuple(keep_rpy) if keep_rpy is not None else None
        points.append(TrajPoint(float(t), (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])), rpy))
    return points


def read_csv_trajectory(path: Path) -> List[TrajPoint]:
    """
    CSV columns:
      t,x,y,z[,roll_deg,pitch_deg,yaw_deg]
    Header row optional.
    """
    points: List[TrajPoint] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Skip header
            if row[0].lower() in {"t", "time"}:
                continue
            vals = [float(x) for x in row]
            if len(vals) not in (4, 7):
                raise ValueError("CSV row must have 4 or 7 columns: t,x,y,z[,roll_deg,pitch_deg,yaw_deg]")
            t, x, y, z = vals[:4]
            rpy = tuple(vals[4:7]) if len(vals) == 7 else None
            points.append(TrajPoint(t=float(t), pos=(x, y, z), rpy_deg=rpy))
    return points


def write_csv_trajectory(points: List[TrajPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"])
        for p in points:
            if p.rpy_deg is None:
                writer.writerow([p.t, p.pos[0], p.pos[1], p.pos[2], "", "", ""])
            else:
                writer.writerow([p.t, p.pos[0], p.pos[1], p.pos[2], p.rpy_deg[0], p.rpy_deg[1], p.rpy_deg[2]])


def save_joint_trajectory_csv(joint_names: List[str], times: List[float], joints_deg: List[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", *joint_names])
        for t, q in zip(times, joints_deg):
            writer.writerow([t, *[float(x) for x in q]])


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot IK trajectory replay from CSV or generated sample")

    # Kinematics
    parser.add_argument("--urdf", required=True, help="Path to robot URDF file")
    parser.add_argument("--target-frame", default="gripper_frame_link", help="End-effector frame name in URDF")
    parser.add_argument("--joint-names", default=None, help="Comma-separated joint names to control (optional)")
    parser.add_argument("--position-weight", type=float, default=1.0, help="IK position weight")
    parser.add_argument("--orientation-weight", type=float, default=0.01, help="IK orientation weight (0 for position-only)")
    parser.add_argument("--initial-joints", type=float, nargs="+", default=None, help="Initial joints in degrees")

    # Input trajectory
    parser.add_argument("--traj-csv", type=Path, default=None, help="CSV file with t,x,y,z[,r,p,y] in degrees")
    parser.add_argument("--rate", type=float, default=30.0, help="Control rate Hz for resampling if needed")

    # Sample generation
    parser.add_argument("--generate-sample", type=Path, default=None, help="If set, write a sample CSV and exit")
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"), default=[0.30, 0.0, 0.20], help="Center position (m)")
    parser.add_argument("--offset", type=float, nargs=3, metavar=("dX", "dY", "dZ"), default=[0.10, 0.0, 0.0], help="Offset from center (m)")
    parser.add_argument("--duration", type=float, default=3.0, help="Sample duration seconds")

    # Output
    parser.add_argument("--save-joints", type=Path, default=None, help="Save computed joint trajectory CSV")
    parser.add_argument("--max-joint-step-deg", type=float, default=None, help="Clamp per-step joint change (deg)")

    args = parser.parse_args()

    joint_names: Optional[List[str]] = (
        [n.strip() for n in args.joint_names.split(",")] if args.joint_names else None
    )

    kin = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name=args.target_frame,
        joint_names=joint_names,
    )
    nj = len(kin.joint_names)

    # Initial joints
    if args.initial_joints is None:
        current_joint_deg = np.zeros(nj, dtype=float)
    else:
        if len(args.initial_joints) != nj:
            raise ValueError(f"--initial-joints length {len(args.initial_joints)} != joint count {nj}")
        current_joint_deg = np.array(args.initial_joints, dtype=float)

    # Build or load trajectory
    if args.generate_sample is not None:
        # Keep current orientation from FK
        fk_T = kin.forward_kinematics(current_joint_deg)
        rpy_keep = None  # keep None to position-only by default
        traj = generate_linear_sample(tuple(args.center), tuple(args.offset), args.duration, args.rate, rpy_keep)
        write_csv_trajectory(traj, args.generate_sample)
        print(f"Wrote sample trajectory to {args.generate_sample}")
        return

    if args.traj_csv is None:
        raise SystemExit("Provide --traj-csv or use --generate-sample to create one.")

    traj = read_csv_trajectory(args.traj_csv)

    # If trajectory has no orientation, keep current orientation
    keep_R = kin.forward_kinematics(current_joint_deg)[:3, :3]

    times: List[float] = []
    joints: List[np.ndarray] = []
    q_prev = current_joint_deg.copy()

    for p in traj:
        if p.rpy_deg is None:
            T = np.eye(4)
            T[:3, :3] = keep_R
            T[:3, 3] = np.array(p.pos)
        else:
            T = _pose_from_pos_rpy(p.pos, p.rpy_deg)

        q = kin.inverse_kinematics(
            current_joint_pos=q_prev,
            desired_ee_pose=T,
            position_weight=args.position_weight,
            orientation_weight=args.orientation_weight,
        )

        # Optional per-step clamp
        if args.max_joint_step_deg is not None:
            dq = np.clip(q - q_prev, -args.max_joint_step_deg, args.max_joint_step_deg)
            q = q_prev + dq

        times.append(p.t)
        joints.append(q)
        q_prev = q

    # Final FK check
    fk_T = kin.forward_kinematics(joints[-1])
    pos_err = np.linalg.norm(fk_T[:3, 3] - (traj[-1].pos))
    print("Computed joint trajectory with", len(joints), "points")
    print("Final position:", fk_T[:3, 3], "| target:", traj[-1].pos, "| pos_err(m):", f"{pos_err:.4f}")

    if args.save_joints is not None:
        save_joint_trajectory_csv(kin.joint_names, times, joints, args.save_joints)
        print(f"Saved joint trajectory to {args.save_joints}")


if __name__ == "__main__":
    main()



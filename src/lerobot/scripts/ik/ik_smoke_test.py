#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from lerobot.model.kinematics import RobotKinematics


def rpy_deg_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix from ZYX (yaw-pitch-roll) angles in degrees.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    # Z (yaw) * Y (pitch) * X (roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

    return Rz @ Ry @ Rx


def build_target_pose(
    position_m: List[float],
    rpy_deg: Optional[List[float]],
) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.array(position_m, dtype=float)
    if rpy_deg is not None:
        T[:3, :3] = rpy_deg_to_matrix(rpy_deg[0], rpy_deg[1], rpy_deg[2])
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot IK smoke test")

    parser.add_argument("--urdf", required=True, help="Path to robot URDF file")
    parser.add_argument(
        "--target-frame",
        default="gripper_frame_link",
        help="End-effector frame name in URDF",
    )
    parser.add_argument(
        "--joint-names",
        default=None,
        help="Comma-separated joint names to control (optional). Defaults to URDF order.",
    )

    parser.add_argument(
        "--pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.3, 0.0, 0.2],
        help="Target EE position in meters (base frame)",
    )
    parser.add_argument(
        "--rpy",
        type=float,
        nargs=3,
        metavar=("R", "P", "Y"),
        default=None,
        help="Target EE orientation as roll/pitch/yaw degrees (ZYX). If omitted, keep current orientation.",
    )

    parser.add_argument(
        "--position-weight",
        type=float,
        default=1.0,
        help="IK position weight",
    )
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=0.01,
        help="IK orientation weight (use 0 for position-only)",
    )

    parser.add_argument(
        "--initial-joints",
        type=float,
        nargs="+",
        default=None,
        help="Initial joint guess in degrees (length must match joint count).",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Path to save IK result JSON (joints_deg, fk_pose).",
    )

    args = parser.parse_args()

    joint_names: Optional[List[str]] = (
        [name.strip() for name in args.joint_names.split(",")] if args.joint_names else None
    )

    kin = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name=args.target_frame,
        joint_names=joint_names,
    )

    # Determine number of controlled joints
    num_joints = len(kin.joint_names)

    if args.initial_joints is None:
        current_joint_deg = np.zeros(num_joints, dtype=float)
    else:
        if len(args.initial_joints) != num_joints:
            raise ValueError(
                f"--initial-joints length {len(args.initial_joints)} != joint count {num_joints}"
            )
        current_joint_deg = np.array(args.initial_joints, dtype=float)

    # If orientation omitted, use FK orientation from current joints
    if args.rpy is None:
        current_T = kin.forward_kinematics(current_joint_deg)
        desired_T = np.eye(4, dtype=float)
        desired_T[:3, :3] = current_T[:3, :3]
        desired_T[:3, 3] = np.array(args.pos, dtype=float)
    else:
        desired_T = build_target_pose(args.pos, args.rpy)

    # Solve IK
    joints_deg = kin.inverse_kinematics(
        current_joint_pos=current_joint_deg,
        desired_ee_pose=desired_T,
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
    )

    # Verify with FK
    fk_T = kin.forward_kinematics(joints_deg)

    pos_err = np.linalg.norm(fk_T[:3, 3] - desired_T[:3, 3])
    # Orientation error via trace (angle-axis)
    R_err = desired_T[:3, :3].T @ fk_T[:3, :3]
    ang_err = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_err) - 1.0) / 2.0))))

    print("IK result (deg):", np.array2string(joints_deg, precision=3))
    print(f"FK position: {fk_T[:3, 3]}  | desired: {desired_T[:3, 3]}  | pos_err: {pos_err:.4f} m")
    print(f"FK orientation error: {ang_err:.2f} deg")

    if args.save_json is not None:
        payload = {
            "joint_names": kin.joint_names,
            "joints_deg": joints_deg.tolist(),
            "fk_pose": fk_T.tolist(),
            "desired_pose": desired_T.tolist(),
            "position_error_m": float(pos_err),
            "orientation_error_deg": float(ang_err),
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2))
        print(f"Saved IK result to {args.save_json}")


if __name__ == "__main__":
    main()



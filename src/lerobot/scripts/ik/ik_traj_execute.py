#!/usr/bin/env python3

import argparse
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.scripts.ik.ik_traj_replay import read_csv_trajectory, _pose_from_pos_rpy


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Cartesian trajectory CSV on SO100 robot via IK")

    # Robot
    parser.add_argument("--port", required=True, help="Serial port to connect to the arm, e.g. /dev/ttyUSB0")
    parser.add_argument("--no-degrees", action="store_true", help="If set, do not use degrees mode on motors")
    parser.add_argument("--max-relative-target", type=float, default=None, help="SO100Follower safety cap per joint")

    # Kinematics
    parser.add_argument("--urdf", required=True, help="Path to robot URDF file")
    parser.add_argument("--target-frame", default="gripper_frame_link", help="End-effector frame name in URDF")
    parser.add_argument("--joint-names", default=None, help="Comma-separated joint names to control (optional)")
    parser.add_argument("--position-weight", type=float, default=1.0, help="IK position weight")
    parser.add_argument("--orientation-weight", type=float, default=0.0, help="IK orientation weight (0 for position-only)")
    parser.add_argument("--initial-joints", type=float, nargs="+", default=None, help="Initial joints in degrees")
    parser.add_argument("--max-joint-step-deg", type=float, default=2.0, help="Clamp per-step joint change (deg)")

    # Trajectory
    parser.add_argument("--traj-csv", type=Path, required=True, help="CSV file with t,x,y,z[,r,p,y] (deg)")
    parser.add_argument("--preview-only", action="store_true", help="Do not move, only print first/last targets")
    parser.add_argument("--dry-run", action="store_true", help="Compute IK and timings, but do not send commands")

    args = parser.parse_args()

    # Kinematics init
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
        q_prev = np.zeros(nj, dtype=float)
    else:
        if len(args.initial_joints) != nj:
            raise ValueError(f"--initial-joints length {len(args.initial_joints)} != joint count {nj}")
        q_prev = np.array(args.initial_joints, dtype=float)

    traj = read_csv_trajectory(args.traj_csv)
    if len(traj) < 2:
        raise SystemExit("Trajectory must contain at least two points")

    # Orientation for points that have none
    keep_R = kin.forward_kinematics(q_prev)[:3, :3]

    print(f"Trajectory points: {len(traj)} | First t={traj[0].t:.3f}s, Last t={traj[-1].t:.3f}s")
    print(f"Initial joints (deg): {np.array2string(q_prev, precision=2)}")
    if args.preview_only:
        first = traj[0]
        last = traj[-1]
        print("First pose:", first.pos, first.rpy_deg)
        print("Last pose:", last.pos, last.rpy_deg)
        return

    # Robot init
    robot_cfg = SO100FollowerConfig(
        id="so100",
        port=args.port,
        cameras={},
        use_degrees=(not args.no_degrees),
        max_relative_target=(args.max_relative_target if args.max_relative_target is not None else None),
    )
    robot = SO100Follower(robot_cfg)
    robot.connect(calibrate=False)

    try:
        print("Ready. Press ENTER to start execution...")
        input()

        t0 = time.monotonic()
        traj_t0 = traj[0].t

        for i, p in enumerate(traj):
            # Time sync: wait until wall-clock matches trajectory time offset
            target_elapsed = (p.t - traj_t0)
            while not args.dry_run and (time.monotonic() - t0) < target_elapsed:
                time.sleep(0.001)

            # Build target pose
            if p.rpy_deg is None:
                T = np.eye(4)
                T[:3, :3] = keep_R
                T[:3, 3] = np.array(p.pos)
            else:
                T = _pose_from_pos_rpy(p.pos, p.rpy_deg)

            # IK
            q = kin.inverse_kinematics(
                current_joint_pos=q_prev,
                desired_ee_pose=T,
                position_weight=args.position_weight,
                orientation_weight=args.orientation_weight,
            )

            # Clamp per-step change
            if args.max_joint_step_deg is not None:
                dq = np.clip(q - q_prev, -args.max_joint_step_deg, args.max_joint_step_deg)
                q = q_prev + dq

            action = {
                "shoulder_pan.pos": float(q[0]),
                "shoulder_lift.pos": float(q[1]),
                "elbow_flex.pos": float(q[2]),
                "wrist_flex.pos": float(q[3]),
                "wrist_roll.pos": float(q[4]),
                # Optional gripper: if present, clamp to [5, 50] like end-effector follower
            }
            if len(q) >= 6:
                # Use a mild clamp for safety
                action["gripper.pos"] = float(np.clip(q[5], 5.0, 50.0))

            if args.dry_run:
                if i in (0, len(traj) - 1):
                    print(f"t={p.t:.2f}s -> action: {action}")
            else:
                robot.send_action(action)
                q_prev = q

        if not args.dry_run:
            print("Trajectory execution finished.")

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()



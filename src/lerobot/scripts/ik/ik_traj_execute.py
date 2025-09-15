#!/usr/bin/env python3

import argparse
import math
import time
import csv
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
    parser.add_argument("--robot-id", default="ik_so101", help="Calibration/profile id for the robot")

    # Kinematics
    parser.add_argument("--urdf", required=True, help="Path to robot URDF file")
    parser.add_argument("--target-frame", default="gripper_frame_link", help="End-effector frame name in URDF")
    parser.add_argument("--joint-names", default=None, help="Comma-separated joint names to control (optional)")
    parser.add_argument("--position-weight", type=float, default=1.0, help="IK position weight")
    parser.add_argument("--orientation-weight", type=float, default=0.0, help="IK orientation weight (0 for position-only)")
    parser.add_argument("--initial-joints", type=float, nargs="+", default=None, help="Initial joints in degrees")
    parser.add_argument("--seed-from-robot", action="store_true", help="Read current robot joints as initial seed")
    parser.add_argument("--max-joint-step-deg", type=float, default=2.0, help="Clamp per-step joint change (deg)")

    # Trajectory
    parser.add_argument("--traj-csv", type=Path, required=True, help="CSV file with t,x,y,z[,r,p,y] (deg)")
    parser.add_argument("--preview-only", action="store_true", help="Do not move, only print first/last targets")
    parser.add_argument("--dry-run", action="store_true", help="Compute IK and timings, but do not send commands")
    parser.add_argument("--record-actual", type=Path, default=None, help="Record measured/commanded EE path to CSV (t,x,y,z[,r,p,y])")
    parser.add_argument("--record-rate-hz", type=float, default=10.0, help="Max recording rate when --record-actual is set")
    parser.add_argument("--record-commanded", action="store_true", help="Record commanded joints instead of reading bus")

    # Pre/post moves
    parser.add_argument("--pre-mid", action="store_true", help="Before trajectory, move all joints to mid position")
    parser.add_argument("--post-mid", action="store_true", help="After trajectory, move all joints back to mid position")
    parser.add_argument("--return-to-seed", action="store_true", help="After trajectory, return to initial seed posture")
    parser.add_argument("--step-dt", type=float, default=0.02, help="Sleep between incremental joint steps (s)")
    parser.add_argument("--hold-seconds", type=float, default=0.0, help="Hold at the end before disconnecting (s)")
    parser.add_argument("--no-wait-to-exit", action="store_true", help="Exit without waiting for ENTER at the end")

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

    # Optionally connect early to read seed from robot
    robot = None
    if args.seed_from_robot:
        robot_cfg = SO100FollowerConfig(
            id=args.robot_id,
            port=args.port,
            cameras={},
            use_degrees=(not args.no_degrees),
            max_relative_target=(args.max_relative_target if args.max_relative_target is not None else None),
        )
        robot = SO100Follower(robot_cfg)
        robot.connect(calibrate=False)

    # Initial joints
    if args.initial_joints is not None:
        if len(args.initial_joints) != nj:
            raise ValueError(f"--initial-joints length {len(args.initial_joints)} != joint count {nj}")
        q_prev = np.array(args.initial_joints, dtype=float)
    elif robot is not None:
        present = robot.bus.sync_read("Present_Position")
        try:
            q_prev = np.array([present[name] for name in kin.joint_names], dtype=float)
        except KeyError as e:
            missing = str(e)
            raise SystemExit(f"Missing joint '{missing}' in Present_Position; available: {list(present.keys())}")
    else:
        q_prev = np.zeros(nj, dtype=float)

    q_seed = q_prev.copy()

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
        # Disconnect early-created robot if any
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        return

    # Robot init (if not already created for seeding)
    if robot is None:
        robot_cfg = SO100FollowerConfig(
            id=args.robot_id,
            port=args.port,
            cameras={},
            use_degrees=(not args.no_degrees),
            max_relative_target=(args.max_relative_target if args.max_relative_target is not None else None),
        )
        robot = SO100Follower(robot_cfg)
        robot.connect(calibrate=False)

    # Small helper to convert rotation matrix to ZYX rpy degrees
    def _matrix_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
        # ZYX convention
        sy = -R[2, 0]
        sy = float(max(-1.0, min(1.0, sy)))
        pitch = math.asin(sy)
        if abs(sy) < 0.9999:
            roll = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock
            roll = math.atan2(-R[0, 2], R[1, 1])
            yaw = 0.0
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    # Prepare recorder
    actual_rows: list[list[float]] = []
    if args.record_actual is not None:
        # Header: t,x,y,z,roll_deg,pitch_deg,yaw_deg plus joints (deg)
        pass
    next_record_time = 0.0

    # Helper: incremental move to joint target with per-step clamp
    def move_to_joint_target(q_target: np.ndarray, label: str) -> None:
        nonlocal q_prev
        if args.dry_run:
            print(f"[dry-run] Would move to {label}: {np.array2string(q_target, precision=2)}")
            return
        max_step = float(args.max_joint_step_deg) if args.max_joint_step_deg is not None else 5.0
        for _ in range(2000):  # safety cap
            dq = np.clip(q_target - q_prev, -max_step, max_step)
            if np.allclose(dq, 0.0, atol=1e-2):
                break
            q_prev = q_prev + dq
            action = {
                "shoulder_pan.pos": float(q_prev[0]),
                "shoulder_lift.pos": float(q_prev[1]),
                "elbow_flex.pos": float(q_prev[2]),
                "wrist_flex.pos": float(q_prev[3]),
                "wrist_roll.pos": float(q_prev[4]),
            }
            if len(q_prev) >= 6:
                action["gripper.pos"] = float(np.clip(q_prev[5], 5.0, 95.0))
            robot.send_action(action)
            time.sleep(max(0.0, args.step_dt))

    # Optional: move to mid pose before trajectory
    if args.pre_mid:
        q_mid = np.zeros_like(q_prev)
        if len(q_mid) >= 6:
            q_mid[5] = 50.0  # gripper midpoint in [0..100]
        print("Moving to joint mid position before trajectory...")
        move_to_joint_target(q_mid, label="mid")

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

                # Record actual measured FK at this step if requested
                if args.record_actual is not None:
                    now = time.monotonic() - t0
                    if now >= next_record_time:
                        next_record_time = now + (1.0 / max(1e-3, args.record_rate_hz))

                        if args.record_commanded:
                            q_src = q_prev.copy()
                        else:
                            present = robot.bus.sync_read("Present_Position")
                            try:
                                q_src = np.array([present[name] for name in kin.joint_names], dtype=float)
                            except KeyError:
                                q_src = q_prev.copy()

                        fk_T = kin.forward_kinematics(q_src)
                        roll_deg, pitch_deg, yaw_deg = _matrix_to_rpy_deg(fk_T[:3, :3])
                        actual_rows.append([
                            float(now),
                            float(fk_T[0, 3]),
                            float(fk_T[1, 3]),
                            float(fk_T[2, 3]),
                            float(roll_deg),
                            float(pitch_deg),
                            float(yaw_deg),
                        ] + [float(v) for v in q_src.tolist()])

        if not args.dry_run:
            print("Trajectory execution finished.")

        # Post-move behavior
        # Allow both actions in sequence
        if args.return_to_seed:
            print("Returning to initial seed posture...")
            move_to_joint_target(q_seed, label="seed")
        if args.post_mid:
            q_mid = np.zeros_like(q_prev)
            if len(q_mid) >= 6:
                q_mid[5] = 50.0
            print("Moving back to joint mid position after trajectory...")
            move_to_joint_target(q_mid, label="mid")

        if args.hold_seconds and args.hold_seconds > 0:
            time.sleep(args.hold_seconds)

        # Wait for user confirmation before disconnecting (unless explicitly disabled)
        if not args.no_wait_to_exit:
            print("Press ENTER to disconnect and exit safely...")
            try:
                input()
            except EOFError:
                pass

    finally:
        # Save actual trajectory if requested
        if args.record_actual is not None and actual_rows:
            args.record_actual.parent.mkdir(parents=True, exist_ok=True)
            with args.record_actual.open("w", newline="") as f:
                w = csv.writer(f)
                header = ["t", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"] + [f"{n}_deg" for n in kin.joint_names]
                w.writerow(header)
                for row in actual_rows:
                    w.writerow(row)
            print(f"Saved actual path to {args.record_actual}")
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()



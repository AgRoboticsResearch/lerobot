#!/usr/bin/env python

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def try_build_visualizer(robot):
    """Try to construct a viewer bound to this RobotWrapper using placo_utils.visualization."""
    try:
        from placo_utils.visualization import (
            robot_viz,
            robot_frame_viz,  # noqa: F401
            point_viz,  # noqa: F401
            points_viz,  # noqa: F401
        )

        viz = robot_viz(robot)
        return viz, robot_frame_viz
    except Exception:
        print(
            "[WARN] Could not initialize placo_utils.visualization viewer.\n"
            "       You can still run headless, or start: python -m placo_utils.view <urdf>"
        )
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Play a saved IK joint trajectory inside Placo viewer")
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--npz_path", type=str, required=True, help="NPZ file containing ik_joints_deg")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loop", action="store_true", help="Loop the animation")
    parser.add_argument("--target_frame_name", type=str, default="gripper_frame_link")
    parser.add_argument("--show_effector", action="store_true")
    parser.add_argument("--show_path", action="store_true", help="Try to visualize all EE points as a path if viewer supports it; always dumps to NPZ/CSV")
    args = parser.parse_args()

    try:
        import placo
    except Exception:
        print("[ERROR] placo is required. Install with `pip install placo placo_utils`", file=sys.stderr)
        raise

    npz = Path(args.npz_path)
    if not npz.exists():
        raise FileNotFoundError(f"NPZ not found: {npz}")
    data = np.load(npz, allow_pickle=True)
    if "ik_joints_deg" in data:
        q_traj = data["ik_joints_deg"]
    elif "commanded_joints_deg" in data:
        q_traj = data["commanded_joints_deg"]
    else:
        raise ValueError("NPZ must contain 'ik_joints_deg' or 'commanded_joints_deg'")

    # Build robot and viewer
    robot = placo.RobotWrapper(args.urdf_path)
    solver = placo.KinematicsSolver(robot)
    solver.mask_fbase(True)

    viz, frame_viz = try_build_visualizer(robot)

    # Precompute all EE points from the trajectory
    ee_points = []
    for q_deg in q_traj:
        for jn, qd in zip(["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"], q_deg):
            robot.set_joint(jn, np.deg2rad(qd))
        robot.update_kinematics()
        T = robot.get_T_world_frame(args.target_frame_name)
        ee_points.append(T[:3, 3].copy())
    ee_points = np.asarray(ee_points)

    # Save points for external plotting
    try:
        dump_dir = Path(args.npz_path).parent
        np.save(dump_dir / "ee_points.npy", ee_points)
        with open(dump_dir / "ee_points.csv", "w") as f:
            f.write("x,y,z\n")
            for p in ee_points:
                f.write(f"{p[0]},{p[1]},{p[2]}\n")
    except Exception:
        pass

    # Try to draw path in viewer if supported
    if args.show_path and viz is not None:
        try:
            # Prefer points_viz to draw all samples; fallback silently if not available
            from placo_utils.visualization import points_viz  # type: ignore
            points_viz("ee_path", [tuple(p) for p in ee_points], radius=0.01, color=0x00AAFF)
        except Exception:
            # Best effort; the viewer may not support path drawing
            pass

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    # Try to use ischedule-based loop if available (smooth GUI updates)
    try:
        from ischedule import schedule, run_loop

        idx = {"k": 0}
        dt = 1.0 / max(1, args.fps)

        @schedule(interval=dt)
        def loop():  # noqa: D401
            i = idx["k"]
            q_deg = q_traj[i]
            for jn, qd in zip(joint_names, q_deg):
                robot.set_joint(jn, np.deg2rad(qd))
            robot.update_kinematics()
            if viz is not None:
                try:
                    viz.display(robot.state.q)
                except Exception:
                    pass
            if args.show_effector and frame_viz is not None:
                try:
                    frame_viz(robot, args.target_frame_name)
                except Exception:
                    pass

            # Draw current EE point if available
            if args.show_path and viz is not None:
                try:
                    from placo_utils.visualization import point_viz  # type: ignore
                    T = robot.get_T_world_frame(args.target_frame_name)
                    point_viz("ee_current", T[:3, 3], radius=0.015, color=0x00FF00)
                except Exception:
                    pass

            i += 1
            if i >= len(q_traj):
                if args.loop:
                    i = 0
                else:
                    # Stop the schedule loop by exiting the process cleanly
                    raise SystemExit
            idx["k"] = i

        run_loop()
        return
    except Exception:
        pass

    # Fallback: simple time.sleep loop
    dt = 1.0 / max(1, args.fps)
    try:
        while True:
            for q_deg in q_traj:
                for jn, qd in zip(joint_names, q_deg):
                    robot.set_joint(jn, np.deg2rad(qd))
                robot.update_kinematics()
                if viz is not None:
                    try:
                        viz.display(robot.state.q)
                    except Exception:
                        pass
                if args.show_effector and frame_viz is not None:
                    try:
                        frame_viz(robot, args.target_frame_name)
                    except Exception:
                        pass
                time.sleep(dt)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()



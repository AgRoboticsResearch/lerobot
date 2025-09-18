#!/usr/bin/env python

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def try_start_viewer(robot):
    """
    Try to start the Placo web viewer bound to this RobotWrapper in-process.
    We attempt a few known entry points from placo_utils. If none work, we continue without a viewer.
    """
    try:
        import placo_utils.view as view_mod

        # Some versions expose a function to launch a viewer with an existing robot
        if hasattr(view_mod, "start"):
            view_mod.start(robot)
            return True
        if hasattr(view_mod, "run"):
            # non-blocking run if available
            try:
                view_mod.run(robot, blocking=False)
            except TypeError:
                # Fallback: run() may not accept blocking arg
                view_mod.run(robot)
            return True
    except Exception:
        pass

    try:
        import placo_utils.viewer as viewer_mod

        # Some versions provide a Viewer class taking a robot
        if hasattr(viewer_mod, "Viewer"):
            v = viewer_mod.Viewer(robot)
            if hasattr(v, "start"):
                v.start()
                return True
    except Exception:
        pass

    print("[WARN] Could not start Placo viewer in-process."
          " If you want visualization, run `python -m placo_utils.view <urdf>` in another terminal.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Play a saved IK joint trajectory inside Placo viewer")
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--npz_path", type=str, required=True, help="NPZ file containing ik_joints_deg")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loop", action="store_true", help="Loop the animation")
    args = parser.parse_args()

    try:
        import placo
    except Exception as e:
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

    # Start viewer bound to this robot (best effort)
    try_start_viewer(robot)

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    dt = 1.0 / max(1, args.fps)
    try:
        while True:
            for q_deg in q_traj:
                for jn, qd in zip(joint_names, q_deg):
                    robot.set_joint(jn, np.deg2rad(qd))
                robot.update_kinematics()
                time.sleep(dt)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()



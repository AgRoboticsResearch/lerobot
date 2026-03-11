#!/usr/bin/env python3
"""
SO101 joint space simulation using placo.
Controls the robot directly in joint space with sinusoidal motion patterns.
"""

import placo
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz

# Path to SO101 URDF file
URDF_PATH = "../urdf/Simulation/SO101/so101_new_calib.urdf"

# Loading the robot
robot = placo.RobotWrapper(URDF_PATH, placo.Flags.ignore_collisions)

# Creating the solver
solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)

# Create joints task for controlling joints
joints_task = solver.add_joints_task()

# Set initial joint positions
joints_task.set_joints({
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.5,
    "elbow_flex": 1.0,
    "wrist_flex": -0.5,
    "wrist_roll": 0.0,
    "gripper": 0.5,
})

viz = robot_viz(robot)

t = 0
dt = 0.01
solver.dt = dt


@schedule(interval=dt)
def loop():
    global t
    t += dt

    # Update joint targets with sinusoidal motion
    joints_task.set_joints({
        "shoulder_pan": np.sin(t * 0.3) * 0.3,
        "shoulder_lift": -0.5 + np.sin(t * 0.4) * 0.3,
        "elbow_flex": 1.0 + np.sin(t * 0.5) * 0.4,
        "wrist_flex": -0.5 + np.sin(t * 0.6) * 0.3,
        "wrist_roll": np.sin(t * 0.7) * 0.5,
        "gripper": 0.5 + np.sin(t * 0.8) * 0.5,  # Open and close gripper slowly
    })

    # Solving the IK
    solver.solve(True)
    robot.update_kinematics()

    # Display the robot
    viz.display(robot.state.q)

    # Display all frames with RGB axes
    for frame in robot.model.frames:
        robot_frame_viz(robot, frame.name)


if __name__ == "__main__":
    print("Starting SO101 joint space simulation...")
    print(f"URDF: {URDF_PATH}")
    print("Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper")

    # Display all links and their names
    print("\n--- Robot Links ---")
    model = robot.model
    print(f"Link names in model: {len(model.names)} entries")
    for i, link_name in enumerate(model.names):
        if link_name:  # Skip empty names
            try:
                frame_id = model.getFrameId(link_name)
                print(f"  [{i:2d}] '{link_name}' (frame_id={frame_id})")
            except Exception as e:
                print(f"  [{i:2d}] '{link_name}' (error: {e})")

    # Also display all frames
    print(f"\n--- Robot Frames (model.frames) ---")
    for i, frame in enumerate(model.frames):
        print(f"  [{i:2d}] '{frame.name}' (type={frame.type}, parent={frame.parentJoint})")

    print("\nPress Ctrl+C to stop")
    run_loop()

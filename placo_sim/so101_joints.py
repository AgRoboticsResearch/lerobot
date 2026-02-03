#!/usr/bin/env python3
"""
SO101 joint space simulation using placo.
Controls the robot directly in joint space with sinusoidal motion patterns.
"""

import pinocchio
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
    robot_frame_viz(robot, "gripper_frame_link")


if __name__ == "__main__":
    print("Starting SO101 joint space simulation...")
    print(f"URDF: {URDF_PATH}")
    print("Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper")
    print("Press Ctrl+C to stop")
    run_loop()

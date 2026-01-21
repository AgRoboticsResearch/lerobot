#!/usr/bin/env python3
"""
Basic SO101 motion simulation using placo.
The robot's end effector (gripper_frame_link) follows a sinusoidal trajectory.
"""

import pinocchio
import placo
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz
from placo_utils.tf import tf

# Path to SO101 model directory (placo expects a directory with robot.urdf inside)
URDF_PATH = "../urdf/Simulation/SO101"

# Loading the robot
robot = placo.RobotWrapper(URDF_PATH, placo.Flags.ignore_collisions)

# Creating the solver
solver = placo.KinematicsSolver(robot)

# The floating base is fixed (robot is not moving)
solver.mask_fbase(True)

# Creating a task for the gripper frame (end effector)
effector_task = solver.add_frame_task("gripper_frame_link", np.eye(4))
effector_task.configure("gripper_frame", "soft", 1.0, 1.0)

# Enable joints velocity limits
solver.enable_velocity_limits(True)

viz = robot_viz(robot)

t = 0
dt = 0.01
solver.dt = dt


@schedule(interval=dt)
def loop():
    global t
    t += dt

    # Define target position for the gripper
    # SO101 reaches forward with sinusoidal motion
    target_x = 0.15 + 0.05 * np.cos(t * 0.5)
    target_y = 0.0 + 0.08 * np.sin(t)
    target_z = 0.10 + 0.05 * np.sin(t * 0.7)

    # Create target transform (identity rotation, translation above)
    effector_task.T_world_frame = tf.translation_matrix([target_x, target_y, target_z])

    # Solving the IK
    solver.solve(True)
    robot.update_kinematics()

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, "gripper_frame_link")
    frame_viz("target", effector_task.T_world_frame)


if __name__ == "__main__":
    print("Starting SO101 basic motion simulation...")
    print(f"URDF: {URDF_PATH}")
    print("Press Ctrl+C to stop")
    run_loop()

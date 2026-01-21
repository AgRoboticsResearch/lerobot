#!/usr/bin/env python3
"""
SO101 trajectory simulation using placo.
The robot's end effector (gripper_frame_link) follows a figure-8 (∞) trajectory.
"""

import pinocchio
import placo
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz, points_viz
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
effector_task.configure("gripper_frame", "soft", 10.0, 1.0)

# Enable joints velocity limits
solver.enable_velocity_limits(True)

viz = robot_viz(robot)

t = 0
dt = 0.01
solver.dt = dt
last_targets = []
last_target_t = 0


@schedule(interval=dt)
def loop():
    global t, last_targets, last_target_t
    t += dt

    # Figure-8 (∞) trajectory for the gripper
    # Centered at a reachable position in front of the robot
    target_x = 0.15
    target_y = np.cos(t) * 0.06
    target_z = 0.12 + np.sin(2 * t) * 0.03

    effector_task.T_world_frame = tf.translation_matrix([target_x, target_y, target_z])

    # Solving the IK
    solver.solve(True)
    robot.update_kinematics()

    # Displaying the robot, effector and target
    viz.display(robot.state.q)
    robot_frame_viz(robot, "gripper_frame_link")
    frame_viz("target", effector_task.T_world_frame)

    # Drawing the trajectory trail (one point every 100ms)
    if t - last_target_t > 0.1:
        last_target_t = t
        last_targets.append([target_x, target_y, target_z])
        last_targets = last_targets[-50:]
        points_viz("trajectory", last_targets, color=0xaaff00)


if __name__ == "__main__":
    print("Starting SO101 trajectory simulation...")
    print(f"URDF: {URDF_PATH}")
    print("The gripper will follow a figure-8 pattern")
    print("Press Ctrl+C to stop")
    run_loop()

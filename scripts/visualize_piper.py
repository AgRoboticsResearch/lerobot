import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz
from ischedule import schedule, run_loop
import os

"""
Visualizes the Piper robot
"""

# Get the absolute path to the URDF file
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "../piper_description/urdf/piper_description.urdf")

print(f"Loading robot from: {urdf_path}")

robot = placo.RobotWrapper(urdf_path)
viz = robot_viz(robot)

t = 0
dt = 0.01

@schedule(interval=dt)
def loop():
    global t
    t += dt

    # Moving some Joints to middle position
    robot.set_joint('joint1', 0.0)
    robot.set_joint('joint2', 1.57)
    robot.set_joint('joint3', -1.4835)
    robot.set_joint('joint4', 0.0)
    robot.set_joint('joint5', 0.0)
    robot.set_joint('joint6', 0.0)

    # Updating kinematics
    robot.update_kinematics()

    # Print gripper_base pose
    # T_world_gripper = robot.get_T_world_frame("gripper_base")
    # print(f"gripper_base pose:\n{T_world_gripper}\n")

    # Showing effector frame (link6 is the last revolute link, but let's check if there is a specific effector frame)
    # Based on URDF, 'link6' or 'gripper_base' seems appropriate. 
    # Let's visualize 'link6' frame as a proxy for effector if 'effector' doesn't exist.
    # The user example used "effector", which is likely a frame name in their specific URDF.
    # In Piper URDF, we have 'link6', 'gripper_base'.
    robot_frame_viz(robot, "gripper_base")

    # Updating the viewer
    viz.display(robot.state.q)

print("Starting loop...")
run_loop()

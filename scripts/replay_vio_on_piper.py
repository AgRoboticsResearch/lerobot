import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, point_viz, points_viz, frame_viz
from ischedule import schedule, run_loop
import os
import csv
from scipy.spatial.transform import Rotation as R

"""
Replays the relative VIO trajectory on the Piper robot using IK.
"""

# 1. Load Robot
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "../piper_description/urdf/piper_description.urdf")
print(f"Loading robot from: {urdf_path}")
robot = placo.RobotWrapper(urdf_path)

# Set initial configuration (middle position)
robot.set_joint('joint1', 0.0)
robot.set_joint('joint2', 1.0)
robot.set_joint('joint3', -1.0)
robot.set_joint('joint4', 0.0)
robot.set_joint('joint5', -0.7)
robot.set_joint('joint6', 0.0)
robot.update_kinematics()

# 2. Get Initial EE Pose (T_curr)
effector_frame = "gripper_base"
T_curr = robot.get_T_world_frame(effector_frame)
print(f"Initial {effector_frame} pose:\n{T_curr}\n")

# 3. Load Relative VIO Trajectory
csv_path = "example_demo_session/demos/demo_C3441328164125_2024.01.10_10.57.34.882133/vio_trajectory_relative.csv"
abs_csv_path = os.path.abspath(csv_path)
print(f"Loading relative trajectory from: {abs_csv_path}")

relative_poses = []
with open(abs_csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x = float(row['x'])
        y = float(row['y'])
        z = float(row['z'])
        qx = float(row['q_x'])
        qy = float(row['q_y'])
        qz = float(row['q_z'])
        qw = float(row['q_w'])
        
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        r = R.from_quat([qx, qy, qz, qw])
        T[:3, :3] = r.as_matrix()
        relative_poses.append(T)

print(f"Loaded {len(relative_poses)} poses")

# 4. Compute Target Poses (T_new_i = T_curr * {^P0}T_{Pi})
target_poses = []
target_points = []
for T_rel in relative_poses:
    T_target = T_curr @ T_rel
    target_poses.append(T_target)
    target_points.append(T_target[:3, 3])

# 5. Visualization Loop
viz = robot_viz(robot)
solver = placo.KinematicsSolver(robot)

# The floating base is fixed (robot is not moving)
solver.mask_fbase(True)

# Task: End Effector Pose
task_ee = solver.add_frame_task(effector_frame, np.eye(4))
task_ee.configure("ee_pose", "soft", 10.0, 1.0)

# Task: Regularization (keep close to default/current posture)
task_reg = solver.add_regularization_task(1e-5)

# Enable joints velocity limits
solver.enable_velocity_limits(True)

t = 0
dt = 0.01
solver.dt = dt
current_idx = 0

@schedule(interval=dt)
def loop():
    global t, current_idx
    t += dt
    
    # Update index
    current_idx = int((t * 30) % len(target_poses))
    T_target = target_poses[current_idx]
    
    # Update Task Target
    task_ee.T_world_frame = T_target
    
    # Solve
    solver.solve(True) # True = apply solution to robot
    robot.update_kinematics()
    
    # Debugging: Print distance to target
    T_ee_current = robot.get_T_world_frame(effector_frame)
    error_pos = np.linalg.norm(T_target[:3, 3] - T_ee_current[:3, 3])
    # print(f"Target: {T_target[:3, 3]}, Current: {T_ee_current[:3, 3]}, Error: {error_pos:.4f}")
    
    # Update Visualization
    viz.display(robot.state.q)
    
    # Visualize Target Frame
    robot_frame_viz(robot, effector_frame)
    frame_viz("target", T_target)
    
    # Visualize Trajectory
    points_viz("trajectory", target_points, radius=0.005, color=0x0000FF)
    point_viz("target_point", T_target[:3, 3], radius=0.01, color=0x00FF00)

print("Starting replay loop...")
run_loop()

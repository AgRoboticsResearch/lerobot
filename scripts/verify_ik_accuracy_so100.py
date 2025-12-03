import numpy as np
import placo
import os
import csv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

"""
Verifies IK accuracy for SO-ARM100 by comparing FK results with VIO targets.
"""

# 1. Load Robot
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "../SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
print(f"Loading robot from: {urdf_path}")
robot = placo.RobotWrapper(urdf_path)

# Set initial configuration (middle range)
robot.set_joint('shoulder_pan', 0.0)
robot.set_joint('shoulder_lift', 0.0)
robot.set_joint('elbow_flex', 0.0)
robot.set_joint('wrist_flex', 0.0)
robot.set_joint('wrist_roll', 0.0)
robot.set_joint('gripper', 0.0)
robot.update_kinematics()

# 2. Get Initial EE Pose (T_curr)
effector_frame = "gripper_link"
T_curr = robot.get_T_world_frame(effector_frame)

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
        T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [x, y, z]
        relative_poses.append(T)

print(f"Loaded {len(relative_poses)} poses")

# 4. Compute Target Poses
target_poses = []
for T_rel in relative_poses:
    T_target = T_curr @ T_rel
    target_poses.append(T_target)

# 5. Run IK and Record Results
solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)

# Task: End Effector Pose
task_ee = solver.add_frame_task(effector_frame, np.eye(4))
task_ee.configure("ee_pose", "soft", 10.0, 1.0)

# Task: Regularization
task_reg = solver.add_regularization_task(1e-5)

# Enable joints velocity limits
solver.enable_velocity_limits(True)
solver.dt = 0.01 # Assume same dt

actual_positions = []
target_positions = []
position_errors = []
rotation_errors = [] # Angle difference in degrees

print("Running IK...")
for i, T_target in enumerate(target_poses):
    # Update Task Target
    task_ee.T_world_frame = T_target
    
    # Solve
    solver.solve(True)
    robot.update_kinematics()
    
    # Get Actual Pose via FK
    T_actual = robot.get_T_world_frame(effector_frame)
    
    # Record Data
    target_pos = T_target[:3, 3]
    actual_pos = T_actual[:3, 3]
    
    actual_positions.append(actual_pos)
    target_positions.append(target_pos)
    
    # Compute Errors
    pos_err = np.linalg.norm(target_pos - actual_pos)
    position_errors.append(pos_err)
    
    # Rotation Error (Geodesic distance / Angle of difference rotation)
    R_target = T_target[:3, :3]
    R_actual = T_actual[:3, :3]
    R_diff = R_target.T @ R_actual
    angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    rotation_errors.append(np.degrees(angle_diff))

actual_positions = np.array(actual_positions)
target_positions = np.array(target_positions)
position_errors = np.array(position_errors)
rotation_errors = np.array(rotation_errors)

print(f"Mean Position Error: {np.mean(position_errors):.6f} m")
print(f"Max Position Error: {np.max(position_errors):.6f} m")
print(f"Mean Rotation Error: {np.mean(rotation_errors):.6f} deg")
print(f"Max Rotation Error: {np.max(rotation_errors):.6f} deg")

# 6. Plot Results
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: 3D Trajectory (2D projection for simplicity or subplots for XYZ)
time = np.arange(len(target_poses)) * solver.dt

axs[0].plot(time, target_positions[:, 0], 'r--', label='Target X')
axs[0].plot(time, actual_positions[:, 0], 'r-', label='Actual X')
axs[0].plot(time, target_positions[:, 1], 'g--', label='Target Y')
axs[0].plot(time, actual_positions[:, 1], 'g-', label='Actual Y')
axs[0].plot(time, target_positions[:, 2], 'b--', label='Target Z')
axs[0].plot(time, actual_positions[:, 2], 'b-', label='Actual Z')
axs[0].set_title('SO-ARM100 Trajectory Tracking (XYZ)')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Position (m)')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Position Error
axs[1].plot(time, position_errors, 'k-')
axs[1].set_title('Position Error')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Error (m)')
axs[1].grid(True)

# Plot 3: Rotation Error
axs[2].plot(time, rotation_errors, 'm-')
axs[2].set_title('Rotation Error')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Error (deg)')
axs[2].grid(True)

plt.tight_layout()
output_plot = "ik_accuracy_plot_so100.png"
plt.savefig(output_plot)
print(f"Saved plot to {output_plot}")

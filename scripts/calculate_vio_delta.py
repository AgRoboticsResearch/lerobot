import numpy as np
import csv
import os
from scipy.spatial.transform import Rotation as R

"""
Calculates the Raw Delta {^P0}T_{Pi} = P0^{-1} * Pi for the VIO trajectory.
"""

# Path to the CSV file
input_csv_path = "example_demo_session/demos/demo_C3441328164125_2024.01.10_10.57.34.882133/camera_trajectory.csv"
output_csv_path = "example_demo_session/demos/demo_C3441328164125_2024.01.10_10.57.34.882133/camera_trajectory_relative.csv"

abs_input_path = os.path.abspath(input_csv_path)
abs_output_path = os.path.abspath(output_csv_path)

print(f"Reading from: {abs_input_path}")

def get_transform_matrix(x, y, z, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    r = R.from_quat([qx, qy, qz, qw])
    T[:3, :3] = r.as_matrix()
    return T

def get_pose_from_matrix(T):
    x, y, z = T[:3, 3]
    r = R.from_matrix(T[:3, :3])
    qx, qy, qz, qw = r.as_quat()
    return x, y, z, qx, qy, qz, qw

poses = []
headers = []
rows = []

with open(abs_input_path, 'r') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    for row in reader:
        rows.append(row)
        x = float(row['x'])
        y = float(row['y'])
        z = float(row['z'])
        qx = float(row['q_x'])
        qy = float(row['q_y'])
        qz = float(row['q_z'])
        qw = float(row['q_w'])
        poses.append(get_transform_matrix(x, y, z, qx, qy, qz, qw))

if not poses:
    print("No poses found!")
    exit(1)

P0 = poses[0]
P0_inv = np.linalg.inv(P0)

relative_rows = []

print(f"Computing relative poses for {len(poses)} frames...")

for i, Pi in enumerate(poses):
    # {^P0}T_{Pi} = P0^{-1} * Pi
    T_relative = P0_inv @ Pi
    
    rx, ry, rz, rqx, rqy, rqz, rqw = get_pose_from_matrix(T_relative)
    
    # Create new row with relative pose
    new_row = rows[i].copy()
    new_row['x'] = rx
    new_row['y'] = ry
    new_row['z'] = rz
    new_row['q_x'] = rqx
    new_row['q_y'] = rqy
    new_row['q_z'] = rqz
    new_row['q_w'] = rqw
    
    relative_rows.append(new_row)

print(f"Writing to: {abs_output_path}")

with open(abs_output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(relative_rows)

print("Done.")

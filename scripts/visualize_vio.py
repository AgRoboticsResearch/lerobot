import numpy as np
from placo_utils.visualization import point_viz, points_viz
from ischedule import schedule, run_loop
import os
import csv

"""
Visualizes VIO waypoints from camera_trajectory.csv
"""

# Path to the CSV file
csv_path = "example_demo_session/demos/demo_C3441328164125_2024.01.10_10.57.34.882133/camera_trajectory_relative.csv"
abs_csv_path = os.path.abspath(csv_path)

print(f"Loading waypoints from: {abs_csv_path}")

waypoints = []
with open(abs_csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x = float(row['x'])
        y = float(row['y'])
        z = float(row['z'])
        waypoints.append([x, y, z])

waypoints = np.array(waypoints)
print(f"Loaded {len(waypoints)} waypoints")

t = 0
dt = 0.01
current_idx = 0

@schedule(interval=dt)
def loop():
    global t, current_idx
    t += dt
    
    # Animate a point moving along the trajectory
    # We can map time to index, or just cycle through indices
    current_idx = int((t * 30) % len(waypoints)) # Assume 30Hz playback speed roughly
    
    current_point = waypoints[current_idx]
    
    # Visualize the current point (Green)
    point_viz("current_point", current_point, radius=0.02, color=0x00FF00)

    # Visualize the entire trajectory (Blue)
    # We pass the list of lists/arrays directly
    points_viz("trajectory", waypoints, radius=0.005, color=0x0000FF)

print("Starting loop...")
run_loop()

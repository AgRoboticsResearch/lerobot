import numpy as np
from placo_utils.visualization import point_viz, points_viz
from ischedule import schedule, run_loop
import os
import csv
import argparse

"""
Visualizes VIO waypoints from a trajectory file.
Supports:
- CSV with headers 'x', 'y', 'z'
- KITTI format (space separated, 12 columns): r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
"""

def parse_trajectory(file_path):
    waypoints = []
    print(f"Loading waypoints from: {file_path}")
    
    with open(file_path, 'r') as f:
        # Check first line to determine format
        first_line = f.readline().strip()
        f.seek(0) # Reset pointer
        
        if ',' in first_line and 'x' in first_line:
            # Assume CSV with headers
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                waypoints.append([x, y, z])
        else:
            # Assume space separated (KITTI or simple xyz)
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                if len(parts) == 12: # KITTI format
                    # r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
                    # tx is index 3, ty is index 7, tz is index 11
                    x = float(parts[3])
                    y = float(parts[7])
                    z = float(parts[11])
                    waypoints.append([x, y, z])
                elif len(parts) >= 3:
                    # Try reading first 3 as x, y, z
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        waypoints.append([x, y, z])
                    except ValueError:
                        pass
                        
    return np.array(waypoints)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize VIO trajectory")
    parser.add_argument("file_path", nargs='?', help="Path to the trajectory file")
    args = parser.parse_args()

    if not args.file_path:
        print("Please provide a file path.")
        exit(1)

    abs_csv_path = os.path.abspath(args.file_path)
    waypoints = parse_trajectory(abs_csv_path)
    
    print(f"Loaded {len(waypoints)} waypoints")

    if len(waypoints) == 0:
        print("No waypoints loaded. Exiting.")
        exit(1)

    t = 0
    dt = 0.01
    current_idx = 0

    @schedule(interval=dt)
    def loop():
        global t, current_idx
        t += dt
        
        # Animate a point moving along the trajectory
        current_idx = int((t * 30) % len(waypoints)) # Assume 30Hz playback speed roughly
        
        current_point = waypoints[current_idx]
        
        # Visualize the current point (Green)
        point_viz("current_point", current_point, radius=0.02, color=0x00FF00)

        # Visualize the entire trajectory (Blue)
        points_viz("trajectory", waypoints, radius=0.005, color=0x0000FF)

    print("Starting loop...")
    run_loop()

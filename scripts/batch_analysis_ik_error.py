import numpy as np
import placo
import os
import glob
import argparse
import csv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

"""
Batch analysis of IK error for SROI VIO trajectories.
"""

def load_robot(robot_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if robot_name == 'piper':
        urdf_path = os.path.join(current_dir, "../piper_description/urdf/piper_description.urdf")
        effector_frame = "gripper_base"
        initial_joints = {
            'joint1': 0.0,
            'joint2': 1.0,
            'joint3': -0.5,
            'joint4': 0.0,
            'joint5': -0.7,
            'joint6': 0.0
        }
    elif robot_name == 'so101':
        urdf_path = os.path.join(current_dir, "../SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
        effector_frame = "gripper_frame_link"
        initial_joints = {
            'shoulder_pan': 0.0,
            'shoulder_lift': -0.5,
            'elbow_flex': -0.5,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
    else:
        raise ValueError(f"Unknown robot: {robot_name}")

    # print(f"Loading robot from: {urdf_path}")
    robot = placo.RobotWrapper(urdf_path)
    
    # Set initial configuration
    for joint, value in initial_joints.items():
        try:
            robot.set_joint(joint, value)
        except KeyError:
            pass
            
    robot.update_kinematics()
    return robot, effector_frame

def parse_kitti_trajectory(file_path):
    waypoints = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if len(parts) == 12: # KITTI format
                vals = [float(x) for x in parts]
                T = np.eye(4)
                T[0, 0:3] = vals[0:3]
                T[0, 3] = vals[3]
                T[1, 0:3] = vals[4:7]
                T[1, 3] = vals[7]
                T[2, 0:3] = vals[8:11]
                T[2, 3] = vals[11]
                waypoints.append(T)
            elif len(parts) >= 3:
                 x = float(parts[0])
                 y = float(parts[1])
                 z = float(parts[2])
                 T = np.eye(4)
                 T[:3, 3] = [x, y, z]
                 waypoints.append(T)
    return waypoints

def compute_raw_deltas(waypoints):
    if not waypoints:
        return []
    
    T0 = waypoints[0]
    T0_inv = np.linalg.inv(T0)
    
    raw_deltas = []
    for T in waypoints:
        T_delta = T0_inv @ T
        raw_deltas.append(T_delta)
        
    return raw_deltas

def analyze_trajectory(file_path, robot_name):
    robot, effector_frame = load_robot(robot_name)
    solver = placo.KinematicsSolver(robot)
    solver.mask_fbase(True)
    task_ee = solver.add_frame_task(effector_frame, np.eye(4))
    task_ee.configure("ee_pose", "soft", 10.0, 1.0)
    solver.add_regularization_task(1e-5)
    solver.enable_velocity_limits(True)
    solver.dt = 0.01 # Assume 100Hz simulation step for IK
    
    waypoints = parse_kitti_trajectory(file_path)
    if not waypoints:
        return None

    raw_deltas = compute_raw_deltas(waypoints)
    T_curr = robot.get_T_world_frame(effector_frame)
    
    pos_errors = []
    rot_errors = []
    
    for T_delta in raw_deltas:
        T_target = T_curr @ T_delta
        
        # Update Task Target
        task_ee.T_world_frame = T_target
        
        # Solve
        solver.solve(True)
        robot.update_kinematics()
        
        # Calculate Error
        T_actual = robot.get_T_world_frame(effector_frame)
        
        # Position Error
        pos_err = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
        pos_errors.append(pos_err)
        
        # Rotation Error (angle)
        R_diff = T_target[:3, :3].T @ T_actual[:3, :3]
        r = R.from_matrix(R_diff)
        rot_err = r.magnitude() # Returns angle in radians
        rot_errors.append(rot_err)
        
    return {
        'file': os.path.basename(os.path.dirname(file_path)), # Use segment name
        'count': len(raw_deltas),
        'pos_errors': pos_errors,
        'rot_errors': rot_errors,
        'mean_pos_err': np.mean(pos_errors),
        'max_pos_err': np.max(pos_errors),
        'mean_rot_err': np.mean(rot_errors),
        'max_rot_err': np.max(rot_errors)
    }

def plot_results(results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    segments = [r['file'] for r in results]
    x = np.arange(len(segments))
    width = 0.35

    mean_pos_errs = [r['mean_pos_err'] for r in results]
    max_pos_errs = [r['max_pos_err'] for r in results]
    mean_rot_errs = [r['mean_rot_err'] for r in results]
    max_rot_errs = [r['max_rot_err'] for r in results]

    # Create a single figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Bar chart of Mean/Max Position Errors (Top Left)
    ax = axs[0, 0]
    rects1 = ax.bar(x - width/2, mean_pos_errs, width, label='Mean Pos Err')
    rects2 = ax.bar(x + width/2, max_pos_errs, width, label='Max Pos Err')
    ax.set_ylabel('Error (m)')
    ax.set_title('IK Position Errors by Segment')
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=45, ha='right')
    ax.legend()

    # 2. Bar chart of Mean/Max Rotation Errors (Top Right)
    ax = axs[0, 1]
    rects1 = ax.bar(x - width/2, mean_rot_errs, width, label='Mean Rot Err')
    rects2 = ax.bar(x + width/2, max_rot_errs, width, label='Max Rot Err')
    ax.set_ylabel('Error (rad)')
    ax.set_title('IK Rotation Errors by Segment')
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=45, ha='right')
    ax.legend()

    # 3. Plot Position Error over Time (Bottom Left)
    ax = axs[1, 0]
    for r in results:
        ax.plot(r['pos_errors'], label=r['file'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error over Time')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Legend might be too big here

    # 4. Plot Rotation Error over Time (Bottom Right)
    ax = axs[1, 1]
    for r in results:
        ax.plot(r['rot_errors'], label=r['file'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Rotation Error (rad)')
    ax.set_title('Rotation Error over Time')
    # Place legend outside the plot for the last one, or maybe just one legend for both line plots?
    # Let's put one legend on the side of the whole figure or just on the last plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ik_analysis_summary.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch analysis of IK error")
    parser.add_argument("--root", default="sroi_vio", help="Root directory to search for trajectories")
    parser.add_argument("--robot", choices=['piper', 'so101'], default='piper', help="Robot to use")
    parser.add_argument("--output", default="ik_error_analysis.csv", help="Output CSV file")
    parser.add_argument("--plot-dir", default="analysis_plots", help="Directory to save plots")
    args = parser.parse_args()

    search_pattern = os.path.join(args.root, "**", "CameraTrajectory.txt")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} trajectory files in {args.root}")
    
    results = []
    for i, file_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] Analyzing {file_path}...")
        try:
            res = analyze_trajectory(file_path, args.robot)
            if res:
                results.append(res)
                print(f"  Mean Pos Err: {res['mean_pos_err']:.6f}, Max Pos Err: {res['max_pos_err']:.6f}")
                print(f"  Mean Rot Err: {res['mean_rot_err']:.6f}, Max Rot Err: {res['max_rot_err']:.6f}")
        except Exception as e:
            print(f"  Error analyzing {file_path}: {e}")
            
    # Save to CSV
    if results:
        # Filter out array data for CSV
        csv_results = [{k: v for k, v in r.items() if k not in ['pos_errors', 'rot_errors']} for r in results]
        
        keys = csv_results[0].keys()
        with open(args.output, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(csv_results)
        print(f"\nAnalysis complete. Results saved to {args.output}")
        
        # Generate Plots
        print(f"Generating plots in {args.plot_dir}...")
        plot_results(results, args.plot_dir)
        
        # Print Summary
        print("\nSummary:")
        print(f"{'Segment':<40} | {'Mean Pos Err':<15} | {'Max Pos Err':<15} | {'Mean Rot Err':<15} | {'Max Rot Err':<15}")
        print("-" * 110)
        for res in results:
            print(f"{res['file']:<40} | {res['mean_pos_err']:.6f}          | {res['max_pos_err']:.6f}          | {res['mean_rot_err']:.6f}          | {res['max_rot_err']:.6f}")
    else:
        print("No results to save.")

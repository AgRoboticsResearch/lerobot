import numpy as np
import time
import argparse
import os
import glob
import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from lerobot.model.kinematics import RobotKinematics

# Initial joints configuration (in radians)
PIPER_INITIAL_JOINTS = {
    'joint1': 0.0,
    'joint2': 1.0,
    'joint3': -0.5,
    'joint4': 0.0,
    'joint5': -0.7,
    'joint6': 0.0
}

# Import Piper classes
try:
    from lerobot_robot_piper.piper import Piper
    from lerobot_robot_piper.config_piper import PiperConfig
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../lerobot_robot_piper"))
    from lerobot_robot_piper.piper import Piper
    from lerobot_robot_piper.config_piper import PiperConfig

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
                T[0, 0:3] = vals[0:3]; T[0, 3] = vals[3]
                T[1, 0:3] = vals[4:7]; T[1, 3] = vals[7]
                T[2, 0:3] = vals[8:11]; T[2, 3] = vals[11]
                waypoints.append(T)
            elif len(parts) >= 3:
                 x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
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

def compute_pose_error(T_target, T_actual):
    pos_error = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
    R_target = R.from_matrix(T_target[:3, :3])
    R_actual = R.from_matrix(T_actual[:3, :3])
    R_diff = R_target * R_actual.inv()
    rot_error = np.linalg.norm(R_diff.as_rotvec())
    return pos_error, rot_error

def move_to_initial_configuration(robot, joint_names):
    print("Moving to initial configuration...")
    action = {}
    for name in joint_names:
        # PIPER_INITIAL_JOINTS values are in radians, convert to degrees for the robot
        target_rad = PIPER_INITIAL_JOINTS.get(name, 0.0)
        target_deg = np.rad2deg(target_rad)
        action[f"{name}.pos"] = target_deg
    
    robot.send_action(action)
    # Wait for the robot to reach the position
    time.sleep(4.0)

def replay_trajectory(robot, kinematics, file_path, speed=0.5, joint_names=None, effector_frame="gripper_base"):
    print(f"--- Replaying: {os.path.basename(os.path.dirname(file_path))} ---")
    waypoints = parse_kitti_trajectory(file_path)
    if not waypoints:
        print("Empty trajectory, skipping.")
        return None

    raw_deltas = compute_raw_deltas(waypoints)

    # Get Initial Pose
    obs = robot.get_observation()
    current_joints_deg = [obs.get(f"{name}.pos", 0.0) for name in joint_names]
    
    current_joints_rad = np.deg2rad(current_joints_deg)
    for i, name in enumerate(joint_names):
        kinematics.robot.set_joint(name, current_joints_rad[i])
    kinematics.robot.update_kinematics()
    T_curr = kinematics.robot.get_T_world_frame(effector_frame)

    # Compute Targets
    target_poses = [T_curr @ delta for delta in raw_deltas]
    
    # Control Loop
    trajectory_data = {
        'target_x': [], 'target_y': [], 'target_z': [],
        'ik_x': [], 'ik_y': [], 'ik_z': [],
        'actual_x': [], 'actual_y': [], 'actual_z': [],
        'track_errors': [],
        'ik_errors': []
    }
    
    target_freq = 30.0
    target_dt = (1.0 / target_freq) / speed
    
    print(f"Steps: {len(target_poses)}, Speed: {speed}x")
    
    try:
        for i, T_target in enumerate(target_poses):
            loop_start = time.time()
            
            # IK
            obs = robot.get_observation()
            current_joints_deg_read = [obs.get(f"{name}.pos", 0.0) for name in joint_names]
            
            computed_joints_deg = kinematics.inverse_kinematics(
                current_joint_pos=current_joints_deg_read,
                desired_ee_pose=T_target,
                position_weight=1.0,
                orientation_weight=1.0
            )

            # --- Compute IK Pose (Theoretical) ---
            computed_joints_rad = np.deg2rad(computed_joints_deg)
            for k, name in enumerate(joint_names):
                kinematics.robot.set_joint(name, computed_joints_rad[k])
            kinematics.robot.update_kinematics()
            T_ik = kinematics.robot.get_T_world_frame(effector_frame)
            ik_pos_err, _ = compute_pose_error(T_target, T_ik)
            # -------------------------------------
            
            # Send Action
            action = {f"{name}.pos": computed_joints_deg[j] for j, name in enumerate(joint_names)}
            robot.send_action(action)
            
            # Wait
            elapsed = time.time() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            
            # Record Data
            obs_after = robot.get_observation()
            actual_joints_deg = [obs_after.get(f"{name}.pos", 0.0) for name in joint_names]
            
            actual_joints_rad = np.deg2rad(actual_joints_deg)
            for k, name in enumerate(joint_names):
                kinematics.robot.set_joint(name, actual_joints_rad[k])
            kinematics.robot.update_kinematics()
            T_actual = kinematics.robot.get_T_world_frame(effector_frame)
            
            track_pos_err, _ = compute_pose_error(T_target, T_actual)
            
            trajectory_data['target_x'].append(T_target[0, 3])
            trajectory_data['target_y'].append(T_target[1, 3])
            trajectory_data['target_z'].append(T_target[2, 3])
            
            trajectory_data['ik_x'].append(T_ik[0, 3])
            trajectory_data['ik_y'].append(T_ik[1, 3])
            trajectory_data['ik_z'].append(T_ik[2, 3])

            trajectory_data['actual_x'].append(T_actual[0, 3])
            trajectory_data['actual_y'].append(T_actual[1, 3])
            trajectory_data['actual_z'].append(T_actual[2, 3])
            
            trajectory_data['track_errors'].append(track_pos_err)
            trajectory_data['ik_errors'].append(ik_pos_err)

    except KeyboardInterrupt:
        print("Interrupted!")
        raise

    return trajectory_data

def main():
    parser = argparse.ArgumentParser(description="Batch Replay SROI VIO on Piper")
    parser.add_argument("--root_dir", default="sroi_vio", help="Root directory to search for CameraTrajectory.txt")
    parser.add_argument("--can_interface", default="can0", help="CAN interface")
    parser.add_argument("--speed", type=float, default=0.5, help="Replay speed factor")
    parser.add_argument("--output_plot", default="batch_replay_plot.png", help="Output plot filename")
    args = parser.parse_args()

    # Find files
    search_pattern = os.path.join(args.root_dir, "**", "CameraTrajectory.txt")
    files = glob.glob(search_pattern, recursive=True)
    files.sort()
    
    if not files:
        print(f"No CameraTrajectory.txt files found in {args.root_dir}")
        return

    print(f"Found {len(files)} trajectories.")

    # Initialize Robot
    joint_names = [f"joint{i+1}" for i in range(6)]
    robot_config = PiperConfig(
        can_interface=args.can_interface,
        use_degrees=True,
        include_gripper=True,
        cameras={},
        joint_names=joint_names
    )
    robot = Piper(robot_config)
    
    try:
        robot.connect()
        print("Robot connected.")
        
        # Initialize Kinematics
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "../piper_description/urdf/piper_description.urdf")
        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_base",
            joint_names=joint_names
        )

        all_results = {}

        for file_path in files:
            print(f"\n--- Preparing for: {os.path.basename(os.path.dirname(file_path))} ---")
            
            # Move to initial configuration before starting replay
            move_to_initial_configuration(robot, joint_names)
            
            # Short pause to ensure stability
            time.sleep(1.0)
            
            try:
                data = replay_trajectory(robot, kinematics, file_path, speed=args.speed, joint_names=joint_names)
                if data:
                    name = os.path.basename(os.path.dirname(file_path)) # Use parent folder name
                    all_results[name] = data
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error replaying {file_path}: {e}")

    finally:
        robot.disconnect()
        print("Robot disconnected.")

    # Plotting
    if all_results:
        print(f"Plotting {len(all_results)} trajectories...")
        
        n_plots = len(all_results)
        n_cols = 3
        n_rows = math.ceil(n_plots / n_cols)
        
        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
        
        for idx, (name, data) in enumerate(all_results.items()):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
            
            # Plot Target (Green Dashed)
            ax.plot(data['target_x'], data['target_y'], data['target_z'], 
                    linestyle='--', linewidth=1, color='green', label='Target (VIO)')
            
            # Plot IK (Blue Dotted)
            ax.plot(data['ik_x'], data['ik_y'], data['ik_z'], 
                    linestyle=':', linewidth=1, color='blue', alpha=0.7, label='IK Sol')

            # Plot Actual (Red Solid)
            ax.plot(data['actual_x'], data['actual_y'], data['actual_z'], 
                    linestyle='-', linewidth=2, color='red', label='Actual (FK)')
            
            # Mark Start
            ax.scatter(data['actual_x'][0], data['actual_y'][0], data['actual_z'][0], color='black', marker='o', s=20)

            # Stats
            mean_track_err = np.mean(data['track_errors']) * 1000
            mean_ik_err = np.mean(data['ik_errors']) * 1000
            max_track_err = np.max(data['track_errors']) * 1000
            
            ax.set_title(f"{name}\nTrackErr: {mean_track_err:.1f}mm (Max {max_track_err:.0f}) | IKErr: {mean_ik_err:.1f}mm", fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Add legend to the first plot only to avoid clutter, or all if preferred
            if idx == 0:
                ax.legend(fontsize='x-small')

        plt.tight_layout()
        plt.savefig(args.output_plot)
        print(f"Plots saved to {args.output_plot}")

if __name__ == "__main__":
    main()
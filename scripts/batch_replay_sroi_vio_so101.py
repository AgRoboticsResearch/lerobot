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

# LeRobot Imports
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_robot_action,
    transition_to_observation,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)

# Initial joints configuration (in radians) for SO101
SO101_INITIAL_JOINTS = {
    'shoulder_pan': 0.0,
    'shoulder_lift': -1.8,
    'elbow_flex': 1.0,
    'wrist_flex': 0.5,
    'wrist_roll': 0.0,
    'gripper': 0.0
}

# Import SO101 classes
try:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
except ImportError:
    import sys
    # Adjust path if necessary, assuming standard lerobot structure
    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

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
        # SO101_INITIAL_JOINTS values are in radians, convert to degrees for the robot
        target_rad = SO101_INITIAL_JOINTS.get(name, 0.0)
        target_deg = np.rad2deg(target_rad)
        action[f"{name}.pos"] = target_deg
    
    # Ensure gripper is included if not in joint_names but present in robot
    if "gripper" not in joint_names:
         action["gripper.pos"] = 0.0

    robot.send_action(action)
    # Wait for the robot to reach the position
    time.sleep(4.0)

def pose_dict_to_matrix(pose_dict):
    T = np.eye(4)
    T[:3, 3] = [pose_dict["ee.x"], pose_dict["ee.y"], pose_dict["ee.z"]]
    r = R.from_rotvec([pose_dict["ee.wx"], pose_dict["ee.wy"], pose_dict["ee.wz"]])
    T[:3, :3] = r.as_matrix()
    return T

def replay_trajectory(robot, ik_processor, fk_processor, file_path, speed=0.5, joint_names=None):
    print(f"--- Replaying: {os.path.basename(os.path.dirname(file_path))} ---")
    waypoints = parse_kitti_trajectory(file_path)
    if not waypoints:
        print("Empty trajectory, skipping.")
        return None

    # Truncate trajectory to first 1/3
    original_len = len(waypoints)
    limit = original_len // 3
    waypoints = waypoints[:limit]
    print(f"Truncating trajectory: {original_len} -> {len(waypoints)} steps (1/3)")

    raw_deltas = compute_raw_deltas(waypoints)

    # Get Initial Pose using FK Processor
    obs = robot.get_observation()
    initial_ee_pose = fk_processor(obs)
    T_curr = pose_dict_to_matrix(initial_ee_pose)

    # Compute Targets
    target_poses = [T_curr @ delta for delta in raw_deltas]
    
    # Control Loop
    trajectory_data = {
        'target_x': [], 'target_y': [], 'target_z': [],
        'ik_x': [], 'ik_y': [], 'ik_z': [],
        'actual_x': [], 'actual_y': [], 'actual_z': [],
        'track_errors': [],
        'ik_errors': [],
        'ik_joints': [],
        'actual_joints': []
    }
    
    target_freq = 30.0
    target_dt = (1.0 / target_freq) / speed
    
    print(f"Steps: {len(target_poses)}, Speed: {speed}x")
    
    try:
        for i, T_target in enumerate(target_poses):
            loop_start = time.time()
            
            # A. Get Observation
            obs = robot.get_observation()
            
            # B. Prepare Target Action
            target_pos = T_target[:3, 3]
            target_rot = R.from_matrix(T_target[:3, :3]).as_rotvec()
            
            ee_action = {
                "ee.x": target_pos[0], "ee.y": target_pos[1], "ee.z": target_pos[2],
                "ee.wx": target_rot[0], "ee.wy": target_rot[1], "ee.wz": target_rot[2],
                "ee.gripper_pos": 0.0 
            }
            
            # C. Solve IK
            joint_action = ik_processor((ee_action, obs))
            joint_action["gripper.pos"] = 0.0
            
            # --- Compute IK Pose (Theoretical) using FK Processor ---
            cmd_obs = {k: v for k, v in joint_action.items()}
            ik_ee_pose = fk_processor(cmd_obs)
            T_ik = pose_dict_to_matrix(ik_ee_pose)
            ik_pos_err, _ = compute_pose_error(T_target, T_ik)
            # ------------------------------------------------------
            
            # D. Send Action
            robot.send_action(joint_action)
            
            # Wait
            elapsed = time.time() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            
            # E. Record Data (Actual Pose via FK Processor)
            obs_after = robot.get_observation()
            actual_ee_pose = fk_processor(obs_after)
            T_actual = pose_dict_to_matrix(actual_ee_pose)
            
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

            # Record Joints
            ik_vals = [joint_action[f"{name}.pos"] for name in joint_names]
            act_vals = [obs_after.get(f"{name}.pos", 0.0) for name in joint_names]
            trajectory_data['ik_joints'].append(ik_vals)
            trajectory_data['actual_joints'].append(act_vals)

    except KeyboardInterrupt:
        print("Interrupted!")
        raise

    return trajectory_data

def main():
    parser = argparse.ArgumentParser(description="Batch Replay SROI VIO on SO101")
    parser.add_argument("--root_dir", default="sroi_vio", help="Root directory to search for CameraTrajectory.txt")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--speed", type=float, default=0.5, help="Replay speed factor")
    parser.add_argument("--output_plot", default="batch_replay_so101_plot.png", help="Output plot filename")
    args = parser.parse_args()

    # Find files
    search_pattern = os.path.join(args.root_dir, "**", "CameraTrajectory.txt")
    files = glob.glob(search_pattern, recursive=True)
    files.sort()
    files = files[:5]
    
    if not files:
        print(f"No CameraTrajectory.txt files found in {args.root_dir}")
        return

    print(f"Found {len(files)} trajectories.")

    # Initialize Robot
    # SO101 typically uses 5 joints for IK + gripper
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    
    robot_config = SO101FollowerConfig(
        port=args.port,
        id="so101_batch_replay",
        use_degrees=True,
        cameras={}
    )
    robot = SO101Follower(robot_config)
    
    try:
        robot.connect()
        print("Robot connected.")
        
        # Initialize Kinematics
        # Assuming standard path relative to script or workspace
        urdf_path = "./SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
        if not os.path.exists(urdf_path):
             # Fallback or try to find it
             print(f"Warning: URDF not found at {urdf_path}, trying absolute path...")
             urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../SO-ARM100/Simulation/SO101/so101_new_calib.urdf"))
        
        print(f"Loading URDF: {urdf_path}")
        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=joint_names
        )

        # Initialize Processors
        ik_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=joint_names, initial_guess_current_joints=True)],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
        
        fk_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
            steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=joint_names)],
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )

        all_results = {}

        for file_path in files:
            print(f"\n--- Preparing for: {os.path.basename(os.path.dirname(file_path))} ---")
            
            # Move to initial configuration before starting replay
            move_to_initial_configuration(robot, joint_names)
            
            # Short pause to ensure stability
            time.sleep(1.0)
            
            try:
                data = replay_trajectory(robot, ik_processor, fk_processor, file_path, speed=args.speed, joint_names=joint_names)
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

        # --- Joints Plot ---
        print("Plotting joint comparisons...")
        fig_joints = plt.figure(figsize=(12, 4 * n_plots))
        
        for idx, (name, data) in enumerate(all_results.items()):
            ax = fig_joints.add_subplot(n_plots, 1, idx + 1)
            
            ik_joints = np.array(data['ik_joints']) # (N, n_joints)
            actual_joints = np.array(data['actual_joints']) # (N, n_joints)
            steps = np.arange(len(ik_joints))
            
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            for j_idx, j_name in enumerate(joint_names):
                color = colors[j_idx % len(colors)]
                # Plot IK
                ax.plot(steps, ik_joints[:, j_idx], linestyle='--', color=color, alpha=0.6)
                # Plot Actual
                ax.plot(steps, actual_joints[:, j_idx], linestyle='-', color=color, label=j_name)
            
            ax.set_title(f"{name} - Joints (Solid=Actual, Dashed=IK)")
            ax.set_ylabel("Angle (deg)")
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                # Legend only has actual labels (to save space), but explanation in title covers it
                ax.legend(loc='upper right', ncol=len(joint_names), fontsize='small')

        joints_plot_file = args.output_plot.replace(".png", "_joints.png")
        plt.tight_layout()
        plt.savefig(joints_plot_file)
        print(f"Joint plots saved to {joints_plot_file}")

if __name__ == "__main__":
    main()
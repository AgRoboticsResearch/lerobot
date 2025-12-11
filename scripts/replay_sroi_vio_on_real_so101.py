import numpy as np
import time
import argparse
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from lerobot.model.kinematics_bac import RobotKinematics
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

"""
Replays the relative VIO trajectory on a REAL SO-101 robot and computes the error.
"""

def parse_kitti_trajectory(file_path):
    waypoints = []
    print(f"Loading waypoints from: {file_path}")
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if len(parts) == 12: # KITTI format
                # r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
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
                 # Fallback for simple XYZ, assuming identity rotation
                 x = float(parts[0])
                 y = float(parts[1])
                 z = float(parts[2])
                 T = np.eye(4)
                 T[:3, 3] = [x, y, z]
                 waypoints.append(T)
                 
    return waypoints

def compute_raw_deltas(waypoints):
    """
    Calculates the Raw Delta {^P0}T_{Pi} = P0^{-1} * Pi for the VIO.
    """
    if not waypoints:
        return []
    
    T0 = waypoints[0]
    T0_inv = np.linalg.inv(T0)
    
    raw_deltas = []
    for i, T in enumerate(waypoints):
        # Calculate relative pose in the initial frame
        # {^P0}T_{Pi} = P0^{-1} * Pi
        T_delta = T0_inv @ T
        raw_deltas.append(T_delta)
        
    return raw_deltas

def compute_pose_error(T_target, T_actual):
    pos_error = np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3])
    
    R_target = R.from_matrix(T_target[:3, :3])
    R_actual = R.from_matrix(T_actual[:3, :3])
    
    # Orientation error: angle of R_diff = R_target * R_actual.inv()
    R_diff = R_target * R_actual.inv()
    rot_vec = R_diff.as_rotvec()
    rot_error = np.linalg.norm(rot_vec) # radians
    
    return pos_error, rot_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay VIO trajectory on REAL SO-101 robot and compute error")
    parser.add_argument("file_path", help="Path to the trajectory file")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for the robot")
    parser.add_argument("--output", default="error_log.csv", help="Output file for error log")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed factor (e.g., 0.5 for half speed)")
    args = parser.parse_args()

    # 1. Initialize Real Robot
    print(f"Initializing SO-101 on port {args.port}...")
    robot_config = SO101FollowerConfig(
        id="so101_test",
        port=args.port,
        use_degrees=True, # Use degrees to match IK script
        max_relative_target=None 
    )
    real_robot = SO101Follower(robot_config)
    
    try:
        real_robot.connect()
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        exit(1)

    print("Robot connected.")

    # 2. Initialize Kinematics (for IK and FK verification)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "../SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
    effector_frame = "gripper_frame_link"
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    
    print(f"Loading kinematics from: {urdf_path}")
    kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=effector_frame,
        joint_names=joint_names
    )

    # 3. Load Trajectory
    abs_path = os.path.abspath(args.file_path)
    waypoints = parse_kitti_trajectory(abs_path)
    print(f"Loaded {len(waypoints)} waypoints")

    if not waypoints:
        print("No waypoints loaded. Exiting.")
        real_robot.disconnect()
        exit(1)

    # 4. Calculate Raw Delta
    raw_deltas = compute_raw_deltas(waypoints)

    # 5. Get Initial Pose from Real Robot
    # Read current joints
    obs = real_robot.get_observation()
    current_joints_deg = []
    for name in joint_names:
        key = f"{name}.pos"
        if key in obs:
            current_joints_deg.append(obs[key])
        else:
            print(f"Warning: Joint {name} not found in observation")
            current_joints_deg.append(0.0)
    
    # Update kinematics with current real robot state
    current_joints_rad = np.deg2rad(current_joints_deg)
    for i, name in enumerate(joint_names):
        kinematics.robot.set_joint(name, current_joints_rad[i])
    kinematics.robot.update_kinematics()
    
    T_curr = kinematics.robot.get_T_world_frame(effector_frame)
    print(f"Initial Real Robot Pose:\n{T_curr}\n")

    # 6. Compute Target Poses
    target_poses = []
    for T_delta in raw_deltas:
        T_target = T_curr @ T_delta
        target_poses.append(T_target)

    print(f"Ready to replay {len(target_poses)} poses.")
    input("Press Enter to start moving the robot... (Ctrl+C to stop)")

    # 7. Control Loop
    errors = []
    start_time = time.time()
    target_freq = 30.0
    target_dt = (1.0 / target_freq) / args.speed
    
    print(f"Replaying at {args.speed}x speed (dt={target_dt*1000:.1f}ms)")
    
    try:
        for i, T_target in enumerate(target_poses):
            loop_start = time.time()
            
            # a. Get current state for IK warm start
            obs = real_robot.get_observation()
            current_joints_deg_read = []
            for name in joint_names:
                current_joints_deg_read.append(obs.get(f"{name}.pos", 0.0))
            
            # b. Solve IK
            computed_joints_deg = kinematics.inverse_kinematics(
                current_joint_pos=current_joints_deg_read,
                desired_ee_pose=T_target,
                position_weight=1.0,
                orientation_weight=1.0
            )

            # Check IK Error (Theoretical limit)
            computed_joints_rad = np.deg2rad(computed_joints_deg)
            for k, name in enumerate(joint_names):
                kinematics.robot.set_joint(name, computed_joints_rad[k])
            kinematics.robot.update_kinematics()
            T_ik = kinematics.robot.get_T_world_frame(effector_frame)
            ik_pos_err, ik_rot_err = compute_pose_error(T_target, T_ik)
            
            # c. Send Action
            action = {}
            for j, name in enumerate(joint_names):
                action[f"{name}.pos"] = computed_joints_deg[j]
            
            real_robot.send_action(action)
            
            # d. Wait to maintain target frequency
            elapsed = time.time() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            
            # e. Compute Tracking Error (Real)
            obs_after = real_robot.get_observation()
            actual_joints_deg = []
            for name in joint_names:
                actual_joints_deg.append(obs_after.get(f"{name}.pos", 0.0))
            
            # Compute FK for actual joints
            actual_joints_rad = np.deg2rad(actual_joints_deg)
            for k, name in enumerate(joint_names):
                kinematics.robot.set_joint(name, actual_joints_rad[k])
            kinematics.robot.update_kinematics()
            T_actual = kinematics.robot.get_T_world_frame(effector_frame)
            
            pos_err, rot_err = compute_pose_error(T_target, T_actual)
            errors.append({
                'step': i,
                'time': time.time() - start_time,
                'pos_error': pos_err,
                'rot_error': rot_err,
                'ik_pos_error': ik_pos_err,
                'ik_rot_error': ik_rot_err,
                'target_x': T_target[0,3],
                'target_y': T_target[1,3],
                'target_z': T_target[2,3],
                'ik_x': T_ik[0,3],
                'ik_y': T_ik[1,3],
                'ik_z': T_ik[2,3],
                'actual_x': T_actual[0,3],
                'actual_y': T_actual[1,3],
                'actual_z': T_actual[2,3]
            })
            
            if i % 10 == 0:
                actual_dt = time.time() - loop_start
                freq = 1.0 / actual_dt if actual_dt > 0 else 0.0
                print(f"Step {i}/{len(target_poses)}: TrackErr={pos_err*1000:.1f}mm, IKErr={ik_pos_err*1000:.1f}mm, Freq={freq:.1f}Hz")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Disconnecting...")
        real_robot.disconnect()
        
        # Save errors
        if errors:
            print(f"Saving errors to {args.output}")
            keys = errors[0].keys()
            with open(args.output, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(errors)
            
            # Print stats
            pos_errors = [e['pos_error'] for e in errors]
            rot_errors = [e['rot_error'] for e in errors]
            ik_pos_errors = [e['ik_pos_error'] for e in errors]
            print(f"Mean Track Pos Error: {np.mean(pos_errors)*1000:.2f} mm")
            print(f"Max Track Pos Error: {np.max(pos_errors)*1000:.2f} mm")
            print(f"Mean IK Pos Error: {np.mean(ik_pos_errors)*1000:.2f} mm")
            print(f"Mean Rot Error: {np.rad2deg(np.mean(rot_errors)):.2f} deg")

            # Plot Trajectories
            print("Plotting trajectories...")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract data
            target_x = [e['target_x'] for e in errors]
            target_y = [e['target_y'] for e in errors]
            target_z = [e['target_z'] for e in errors]
            
            ik_x = [e['ik_x'] for e in errors]
            ik_y = [e['ik_y'] for e in errors]
            ik_z = [e['ik_z'] for e in errors]
            
            actual_x = [e['actual_x'] for e in errors]
            actual_y = [e['actual_y'] for e in errors]
            actual_z = [e['actual_z'] for e in errors]
            
            ax.plot(target_x, target_y, target_z, label='Ground Truth (VIO)', color='green', linestyle='--', linewidth=2)
            ax.plot(ik_x, ik_y, ik_z, label='IK Solution', color='blue', linestyle='-', linewidth=1, alpha=0.7)
            ax.plot(actual_x, actual_y, actual_z, label='Real Robot (FK)', color='red', linestyle='-', linewidth=2)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Trajectory Comparison: GT vs IK vs Real')
            ax.legend()
            
            plot_file = args.output.replace('.csv', '.png')
            plt.savefig(plot_file)
            print(f"Plot saved to {plot_file}")


import numpy as np
import placo
import argparse
import os
from scipy.spatial.transform import Rotation as R
from lerobot.model.kinematics_bac import RobotKinematics

def load_robot_config(robot_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if robot_name == 'piper':
        urdf_path = os.path.join(current_dir, "../piper_description/urdf/piper_description.urdf")
        effector_frame = "gripper_base"
        initial_joints = {
            'joint1': 0.0, 'joint2': 1.0, 'joint3': -0.5,
            'joint4': 0.0, 'joint5': -0.7, 'joint6': 0.0
        }
        joint_names = None
    elif robot_name == 'so101':
        urdf_path = os.path.join(current_dir, "../SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
        effector_frame = "gripper_frame_link"
        initial_joints = {
            'shoulder_pan': 0.0, 'shoulder_lift': -1.8, 'elbow_flex': 1.0,
            'wrist_flex': 0.5, 'wrist_roll': 0.0, 'gripper': 0.0
        }
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    else:
        raise ValueError(f"Unknown robot: {robot_name}")
    return urdf_path, effector_frame, initial_joints, joint_names

def parse_trajectory(file_path):
    waypoints = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) == 12:
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

def compute_error(T_target, T_achieved):
    pos_error = np.linalg.norm(T_target[:3, 3] - T_achieved[:3, 3])
    rot_target = R.from_matrix(T_target[:3, :3])
    rot_achieved = R.from_matrix(T_achieved[:3, :3])
    rot_error_rad = np.linalg.norm((rot_target.inv() * rot_achieved).as_rotvec())
    return pos_error, np.degrees(rot_error_rad)

def main():
    parser = argparse.ArgumentParser(description="Compare IK Solvers")
    parser.add_argument("file_path", help="Path to trajectory file")
    parser.add_argument("--robot", choices=['piper', 'so101'], default='piper')
    args = parser.parse_args()

    urdf_path, effector_frame, initial_joints, joint_names = load_robot_config(args.robot)
    
    # Initialize Solvers
    # 1. Direct Placo
    robot_placo = placo.RobotWrapper(urdf_path)
    solver_placo = placo.KinematicsSolver(robot_placo)
    solver_placo.mask_fbase(True)
    task_ee_placo = solver_placo.add_frame_task(effector_frame, np.eye(4))
    task_ee_placo.configure("ee_pose", "soft", 10.0, 1.0) # Matches replay script
    solver_placo.add_regularization_task(1e-5)
    
    # 2. LeRobot Kinematics
    kinematics_lerobot = RobotKinematics(urdf_path, effector_frame, joint_names)

    # Set Initial State
    for j, v in initial_joints.items():
        try:
            robot_placo.set_joint(j, v)
            kinematics_lerobot.robot.set_joint(j, v)
        except KeyError: pass
    
    robot_placo.update_kinematics()
    kinematics_lerobot.robot.update_kinematics()
    
    T_curr = robot_placo.get_T_world_frame(effector_frame)
    waypoints = parse_trajectory(args.file_path)
    
    if not waypoints:
        print("No waypoints found.")
        return

    # Compute Target Poses (Relative to Start)
    T0_inv = np.linalg.inv(waypoints[0])
    target_poses = []
    for T in waypoints:
        target_poses.append(T_curr @ (T0_inv @ T))

    # Storage for errors
    errors_placo = {'pos': [], 'rot': []}
    errors_lerobot = {'pos': [], 'rot': []}

    print(f"{'Idx':<5} | {'Placo Pos (mm)':<15} | {'Placo Rot (deg)':<15} | {'LeRobot Pos (mm)':<15} | {'LeRobot Rot (deg)':<15}")
    print("-" * 80)

    # Simulation Loop
    # We maintain separate robot states for each solver to avoid interference
    # For Placo solver, we update robot_placo state incrementally
    # For LeRobot solver, we update kinematics_lerobot.robot state incrementally
    
    for i, T_target in enumerate(target_poses):
        # --- Solver 1: Direct Placo ---
        task_ee_placo.T_world_frame = T_target
        solver_placo.solve(True)
        robot_placo.update_kinematics()
        T_achieved_placo = robot_placo.get_T_world_frame(effector_frame)
        p_err_placo, r_err_placo = compute_error(T_target, T_achieved_placo)
        errors_placo['pos'].append(p_err_placo * 1000)
        errors_placo['rot'].append(r_err_placo)

        # --- Solver 2: LeRobot Kinematics ---
        # Get current joints for warm start
        current_joints_rad = [kinematics_lerobot.robot.get_joint(n) for n in kinematics_lerobot.joint_names]
        current_joints_deg = np.rad2deg(current_joints_rad)
        
        computed_joints_deg = kinematics_lerobot.inverse_kinematics(
            current_joints_deg, T_target, position_weight=1.0, orientation_weight=1.0
        )
        
        # Update state
        computed_joints_rad = np.deg2rad(computed_joints_deg)
        for idx, name in enumerate(kinematics_lerobot.joint_names):
            kinematics_lerobot.robot.set_joint(name, computed_joints_rad[idx])
        kinematics_lerobot.robot.update_kinematics()
        
        T_achieved_lerobot = kinematics_lerobot.robot.get_T_world_frame(effector_frame)
        p_err_lerobot, r_err_lerobot = compute_error(T_target, T_achieved_lerobot)
        errors_lerobot['pos'].append(p_err_lerobot * 1000)
        errors_lerobot['rot'].append(r_err_lerobot)

        print(f"{i:<5} | {p_err_placo*1000:<15.4f} | {r_err_placo:<15.4f} | {p_err_lerobot*1000:<15.4f} | {r_err_lerobot:<15.4f}")

    # Summary
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"Direct Placo:")
    print(f"  Mean Pos Error: {np.mean(errors_placo['pos']):.4f} mm")
    print(f"  Max Pos Error:  {np.max(errors_placo['pos']):.4f} mm")
    print(f"  Mean Rot Error: {np.mean(errors_placo['rot']):.4f} deg")
    print(f"  Max Rot Error:  {np.max(errors_placo['rot']):.4f} deg")
    print("-" * 40)
    print(f"LeRobot Kinematics:")
    print(f"  Mean Pos Error: {np.mean(errors_lerobot['pos']):.4f} mm")
    print(f"  Max Pos Error:  {np.max(errors_lerobot['pos']):.4f} mm")
    print(f"  Mean Rot Error: {np.mean(errors_lerobot['rot']):.4f} deg")
    print(f"  Max Rot Error:  {np.max(errors_lerobot['rot']):.4f} deg")

if __name__ == "__main__":
    main()

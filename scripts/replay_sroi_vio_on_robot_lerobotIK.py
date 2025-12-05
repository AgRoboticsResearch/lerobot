import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, point_viz, points_viz, frame_viz
from ischedule import schedule, run_loop
import os
import argparse
from scipy.spatial.transform import Rotation as R
from lerobot.model.kinematics import RobotKinematics

"""
Replays the relative VIO trajectory on a robot (Piper or SO-101) using LeRobot Kinematics.
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
        joint_names = None # Use default from URDF
    elif robot_name == 'so101':
        urdf_path = os.path.join(current_dir, "../SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
        effector_frame = "gripper_frame_link"
        initial_joints = {
            'shoulder_pan': 0.0,
            'shoulder_lift': -1.8,
            'elbow_flex': 1.0,
            'wrist_flex': 0.5,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        # Explicitly list joint names if needed, or let RobotKinematics infer
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    else:
        raise ValueError(f"Unknown robot: {robot_name}")

    print(f"Loading robot from: {urdf_path}")
    
    kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=effector_frame,
        joint_names=joint_names
    )
    
    # Set initial configuration using the underlying placo robot wrapper
    # RobotKinematics doesn't expose a direct set_joint method for initialization, 
    # but we can access .robot
    for joint, value in initial_joints.items():
        try:
            kinematics.robot.set_joint(joint, value)
        except KeyError:
            print(f"Warning: Joint {joint} not found in robot")
            
    kinematics.robot.update_kinematics()
    return kinematics, effector_frame, initial_joints

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay VIO trajectory on robot using LeRobot Kinematics")
    parser.add_argument("file_path", help="Path to the trajectory file")
    parser.add_argument("--robot", choices=['piper', 'so101'], default='piper', help="Robot to use (piper or so101)")
    args = parser.parse_args()

    # 1. Load Robot Kinematics
    kinematics, effector_frame, initial_joints = load_robot(args.robot)
    
    # 2. Get Initial EE Pose (T_curr)
    # Use the underlying robot wrapper to get the initial pose
    T_curr = kinematics.robot.get_T_world_frame(effector_frame)
    print(f"Initial {effector_frame} pose:\n{T_curr}\n")
    
    # 3. Load Trajectory
    abs_path = os.path.abspath(args.file_path)
    waypoints = parse_kitti_trajectory(abs_path)
    print(f"Loaded {len(waypoints)} waypoints")

    if not waypoints:
        print("No waypoints loaded. Exiting.")
        exit(1)

    # 4. Calculate Raw Delta {^P0}T_{Pi} = P0^{-1} * Pi
    print("Calculating Raw Delta {^P0}T_{Pi} = P0^{-1} * Pi...")
    raw_deltas = compute_raw_deltas(waypoints)
    
    # Optional: Print first few deltas to verify
    print(f"Computed {len(raw_deltas)} raw deltas.")
    if len(raw_deltas) > 1:
        print(f"First delta (should be Identity):\n{raw_deltas[0]}")
        print(f"Second delta:\n{raw_deltas[1]}")

    # 5. Compute Target Poses
    # We apply the raw delta directly to the robot's end effector.
    target_poses = []
    target_points = []
    for T_delta in raw_deltas:
        # T_target = T_curr * T_delta
        T_target = T_curr @ T_delta
        target_poses.append(T_target)
        target_points.append(T_target[:3, 3])

    # 5. Visualization Loop
    # We still use placo visualization utils, passing the underlying robot wrapper
    viz = robot_viz(kinematics.robot)
    
    t = 0
    dt = 0.01
    current_idx = 0
    
    # Initial joint positions (degrees) for IK warm start
    current_joints_deg = np.zeros(len(kinematics.joint_names))
    # Fill in initial values
    for i, name in enumerate(kinematics.joint_names):
        if name in initial_joints:
            current_joints_deg[i] = np.rad2deg(initial_joints[name])

    @schedule(interval=dt)
    def loop():
        global t, current_idx, current_joints_deg
        t += dt
        
        # Update index
        current_idx = int((t * 30) % len(target_poses))
        T_target = target_poses[current_idx]
        
        # Solve IK using LeRobot Kinematics
        # Note: inverse_kinematics takes degrees and returns degrees
        
        # Get current joint positions in degrees for warm start
        # We can read from the robot state, but since we are updating it, 
        # we can also just use the result from the previous step.
        # Let's read from robot to be safe and consistent.
        
        # Construct current_joints_deg from robot state
        current_joints_rad = []
        for name in kinematics.joint_names:
            current_joints_rad.append(kinematics.robot.get_joint(name))
        current_joints_deg_read = np.rad2deg(current_joints_rad)

        computed_joints_deg = kinematics.inverse_kinematics(
            current_joint_pos=current_joints_deg_read,
            desired_ee_pose=T_target,
            position_weight=1.0,
            orientation_weight=1.0 # Enforce orientation as well
        )
        
        # Update robot state for visualization
        # RobotKinematics.inverse_kinematics doesn't automatically update the robot state
        # It returns the joint values. We need to set them.
        computed_joints_rad = np.deg2rad(computed_joints_deg)
        for i, name in enumerate(kinematics.joint_names):
            kinematics.robot.set_joint(name, computed_joints_rad[i])
            
        kinematics.robot.update_kinematics()
        
        # Update Visualization
        viz.display(kinematics.robot.state.q)
        
        # Visualize Target Frame
        robot_frame_viz(kinematics.robot, effector_frame)
        frame_viz("target", T_target)
        
        # Visualize Trajectory
        points_viz("trajectory", target_points, radius=0.005, color=0x0000FF)
        point_viz("target_point", T_target[:3, 3], radius=0.01, color=0x00FF00)

    print("Starting replay loop...")
    run_loop()

import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, point_viz, points_viz, frame_viz
from ischedule import schedule, run_loop
import os
import argparse
from scipy.spatial.transform import Rotation as R

"""
Replays the relative VIO trajectory on a robot (Piper or SO-101) using IK.
"""

def load_robot(robot_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if robot_name == 'piper':
        urdf_path = os.path.join(current_dir, "../piper_description/urdf/piper_description.urdf")
        effector_frame = "gripper_base"
        initial_joints = {
            'joint1': 0.0,
            'joint2': 1.0,
            'joint3': -1.0,
            'joint4': 0.0,
            'joint5': -0.2,
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

    print(f"Loading robot from: {urdf_path}")
    robot = placo.RobotWrapper(urdf_path)
    
    # Set initial configuration
    for joint, value in initial_joints.items():
        try:
            robot.set_joint(joint, value)
        except KeyError:
            print(f"Warning: Joint {joint} not found in robot")
            
    robot.update_kinematics()
    return robot, effector_frame

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
    parser = argparse.ArgumentParser(description="Replay VIO trajectory on robot")
    parser.add_argument("file_path", help="Path to the trajectory file")
    parser.add_argument("--robot", choices=['piper', 'so101'], default='piper', help="Robot to use (piper or so101)")
    args = parser.parse_args()

    # 1. Load Robot
    robot, effector_frame = load_robot(args.robot)
    
    # 2. Get Initial EE Pose (T_curr)
    T_curr = robot.get_T_world_frame(effector_frame)
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
    viz = robot_viz(robot)
    solver = placo.KinematicsSolver(robot)
    
    solver.mask_fbase(True)
    
    # Task: End Effector Pose
    task_ee = solver.add_frame_task(effector_frame, np.eye(4))
    task_ee.configure("ee_pose", "soft", 10.0, 1.0)
    
    # Task: Regularization
    task_reg = solver.add_regularization_task(1e-5)
    
    # Enable joints velocity limits
    solver.enable_velocity_limits(True)
    
    t = 0
    dt = 0.01
    solver.dt = dt
    current_idx = 0
    
    @schedule(interval=dt)
    def loop():
        global t, current_idx
        t += dt
        
        # Update index
        current_idx = int((t * 30) % len(target_poses))
        T_target = target_poses[current_idx]
        
        # Update Task Target
        task_ee.T_world_frame = T_target
        
        # Solve
        solver.solve(True)
        robot.update_kinematics()
        
        # Update Visualization
        viz.display(robot.state.q)
        
        # Visualize Target Frame
        robot_frame_viz(robot, effector_frame)
        frame_viz("target", T_target)
        
        # Visualize Trajectory
        points_viz("trajectory", target_points, radius=0.005, color=0x0000FF)
        point_viz("target_point", T_target[:3, 3], radius=0.01, color=0x00FF00)

    print("Starting replay loop...")
    run_loop()

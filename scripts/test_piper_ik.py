import sys
import os
import numpy as np

# Add src to path to import lerobot modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from lerobot.model.kinematics import RobotKinematics

def main():
    # Define paths
    urdf_path = "/home/hls/codes/lerobot/piper_description/urdf/piper_description.urdf"
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        return

    try:
        # Initialize kinematics with gripper_base as target
        kinematics = RobotKinematics(urdf_path, target_frame_name="gripper_base")
        print(f"Loaded URDF from {urdf_path}")
        print(f"Joint names: {kinematics.joint_names}")
    except Exception:
        import traceback
        traceback.print_exc()
        return

    # 1. Set initial joint positions
    num_joints = len(kinematics.joint_names)
    initial_joints = np.zeros(num_joints)
    print(f"\nInitial joints (deg): {initial_joints}")

    # 2. Get FK for these joints to get start pose
    try:
        start_pose = kinematics.forward_kinematics(initial_joints)
        print("Start Pose (FK of initial joints):\n", start_pose)
    except Exception as e:
        print(f"FK failed: {e}")
        return

    # 3. Define end pose (move 5cm in X direction relative to base)
    end_pose = start_pose.copy()
    end_pose[0, 3] += 0.05 
    print("\nTarget End Pose (Start + 0.05m in X):\n", end_pose)

    # 4. Interpolate and Solve IK
    steps = 10
    print(f"\nInterpolating in {steps} steps...")
    
    current_joints = initial_joints.copy()
    
    for i in range(steps + 1):
        alpha = i / steps
        # Linear interpolation of position
        target_pos = (1 - alpha) * start_pose[:3, 3] + alpha * end_pose[:3, 3]
        
        # Keep orientation same as start
        target_rot = start_pose[:3, :3]
        
        target_pose = np.eye(4)
        target_pose[:3, :3] = target_rot
        target_pose[:3, 3] = target_pos
        
        # 5. Compute IK
        try:
            ik_joints = kinematics.inverse_kinematics(
                current_joints,
                target_pose,
                position_weight=1.0,
                orientation_weight=1.0
            )
            
            if ik_joints is None:
                print(f"Step {i}: IK failed to find a solution.")
            else:
                # Verify FK
                achieved_pose = kinematics.forward_kinematics(ik_joints)
                pos_error = np.linalg.norm(achieved_pose[:3, 3] - target_pos)
                
                print(f"Step {i}:")
                # print(f"  Target Pos: {target_pos}")
                print(f"  Joints: {np.round(ik_joints, 3)}")
                print(f"  Achieved Pos Error: {pos_error:.6f} m")
                
                current_joints = ik_joints
                
        except Exception as e:
            print(f"Step {i}: IK Exception: {e}")

if __name__ == "__main__":
    main()

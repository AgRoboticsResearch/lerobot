#!/usr/bin/env python

"""
Simple debug script to test IK/FK roundtrip at the RESET pose.
This verifies that the kinematics pipeline doesn't introduce errors
at the initial position using RobotKinematics directly.
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation
from lerobot.model.kinematics import RobotKinematics

# Configuration - use path relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(CURRENT_DIR, "../SO-ARM100/Simulation/SO101/so101_new_calib.urdf")

# Motor names for the SO101 robot
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Reset pose from so101_send_reset.py (initial joint positions in degrees)
RESET_POSE = {
    'shoulder_pan.pos': 0.0,
    'shoulder_lift.pos': -80.0,
    'elbow_flex.pos': 50.0,
    'wrist_flex.pos': 40.0,
    'wrist_roll.pos': 0.0,
    'gripper.pos': 0.0,
}


def main():
    print("=" * 60)
    print("IK/FK Roundtrip Debug at RESET Pose (Direct IK)")
    print("=" * 60)
    print(f"Using URDF: {URDF_PATH}")

    # Get original joint positions as numpy array
    joint_pos_deg = np.array([RESET_POSE[f'{name}.pos'] for name in MOTOR_NAMES])
    zero_joints = np.zeros(len(MOTOR_NAMES))

    # Step 1: Print original RESET pose
    print("\n" + "-" * 60)
    print("Step 1: Original RESET Joint Pose (degrees)")
    print("-" * 60)
    for name in MOTOR_NAMES:
        print(f"  {name}: {RESET_POSE[f'{name}.pos']:.4f}")

    # Initialize FRESH kinematics solver
    print("\nInitializing kinematics solver...")
    kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=MOTOR_NAMES,
    )

    # Step 2: FK - Convert joints to EE pose
    print("\n" + "-" * 60)
    print("Step 2: Forward Kinematics (Joints -> EE)")
    print("-" * 60)
    
    T_ee = kinematics_solver.forward_kinematics(joint_pos_deg)
    pos = T_ee[:3, 3]
    rot_matrix = T_ee[:3, :3]
    rotvec = Rotation.from_matrix(rot_matrix).as_rotvec()
    euler = Rotation.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
    
    print("EE Position (meters):")
    print(f"  x: {pos[0]:.6f}")
    print(f"  y: {pos[1]:.6f}")
    print(f"  z: {pos[2]:.6f}")
    print("EE Orientation (rotation vector):")
    print(f"  wx: {rotvec[0]:.6f}")
    print(f"  wy: {rotvec[1]:.6f}")
    print(f"  wz: {rotvec[2]:.6f}")
    print("EE Orientation (euler xyz, degrees):")
    print(f"  roll:  {euler[0]:.4f}")
    print(f"  pitch: {euler[1]:.4f}")
    print(f"  yaw:   {euler[2]:.4f}")

    # Step 3: IK - Convert EE pose back to joints
    print("\n" + "-" * 60)
    print("Step 3: Inverse Kinematics (EE -> Joints)")
    print("-" * 60)
    
    recovered_joints = kinematics_solver.inverse_kinematics(
        current_joint_pos=joint_pos_deg,
        desired_ee_pose=T_ee,
        position_weight=1.0,
        orientation_weight=0.01,
    )
    
    print("Recovered Joint Pose (degrees):")
    for i, name in enumerate(MOTOR_NAMES):
        original = RESET_POSE[f'{name}.pos']
        recovered = recovered_joints[i]
        diff = recovered - original
        print(f"  {name}: {recovered:.4f} (original: {original:.4f}, diff: {diff:+.4f})")

    # Step 4: FK again - Convert recovered joints back to EE
    print("\n" + "-" * 60)
    print("Step 4: Forward Kinematics on Recovered Joints")
    print("-" * 60)
    
    T_recovered = kinematics_solver.forward_kinematics(recovered_joints)
    pos_recovered = T_recovered[:3, 3]
    
    print("Recovered EE Position (meters):")
    print(f"  x: {pos_recovered[0]:.6f} (original: {pos[0]:.6f}, diff: {pos_recovered[0] - pos[0]:+.6f})")
    print(f"  y: {pos_recovered[1]:.6f} (original: {pos[1]:.6f}, diff: {pos_recovered[1] - pos[1]:+.6f})")
    print(f"  z: {pos_recovered[2]:.6f} (original: {pos[2]:.6f}, diff: {pos_recovered[2] - pos[2]:+.6f})")

    # Step 5: Compute errors
    print("\n" + "-" * 60)
    print("Step 5: Error Summary")
    print("-" * 60)
    
    # Joint error
    joint_errors = [abs(recovered_joints[i] - RESET_POSE[f'{name}.pos']) for i, name in enumerate(MOTOR_NAMES)]
    
    print("Joint Errors (degrees):")
    print(f"  Max:  {max(joint_errors):.6f}")
    print(f"  Mean: {np.mean(joint_errors):.6f}")
    
    # EE position error
    pos_error = pos_recovered - pos
    pos_error_norm = np.linalg.norm(pos_error)
    
    print(f"\nEE Position Error:")
    print(f"  X: {pos_error[0]*1000:.4f} mm")
    print(f"  Y: {pos_error[1]*1000:.4f} mm")
    print(f"  Z: {pos_error[2]*1000:.4f} mm")
    print(f"  Total: {pos_error_norm*1000:.4f} mm")

    # Test with zero initial guess
    print("\n" + "=" * 60)
    print("Test: IK with Zero Initial Guess")
    print("=" * 60)
    
    recovered_joints_zero = kinematics_solver.inverse_kinematics(
        current_joint_pos=zero_joints,
        desired_ee_pose=T_ee,
        position_weight=1.0,
        orientation_weight=0.01,
    )
    
    print("Recovered Joint Pose with Zero Initial Guess (degrees):")
    for i, name in enumerate(MOTOR_NAMES):
        original = RESET_POSE[f'{name}.pos']
        recovered = recovered_joints_zero[i]
        diff = recovered - original
        print(f"  {name}: {recovered:.4f} (original: {original:.4f}, diff: {diff:+.4f})")
    
    # FK on zero-guess recovered joints
    T_recovered_zero = kinematics_solver.forward_kinematics(recovered_joints_zero)
    pos_recovered_zero = T_recovered_zero[:3, 3]
    pos_error_zero = np.linalg.norm(pos_recovered_zero - pos)
    
    print(f"\nEE Position Error with Zero Initial Guess:")
    print(f"  Total: {pos_error_zero*1000:.4f} mm")

    # Test with small perturbations from RESET pose
    print("\n" + "=" * 60)
    print("Test: Small Perturbations from RESET Pose")
    print("=" * 60)
    
    perturbations = [
        (0.01, 0.0, 0.0),       # 10mm in X
        (0.0, 0.01, 0.0),       # 10mm in Y
        (0.0, 0.0, 0.01),       # 10mm in Z
        (0.01, 0.01, 0.01),     # 10mm in all directions
        (0.001, 0.001, 0.001),  # 1mm in all directions
        (0.0, 0.0, 0.0),        # No perturbation (sanity check)
    ]
    
    for dx, dy, dz in perturbations:
        # Create FRESH solver for each perturbation test
        fresh_solver = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name="gripper_frame_link",
            joint_names=MOTOR_NAMES,
        )
        
        # Get target pose with perturbation
        T_target = fresh_solver.forward_kinematics(joint_pos_deg)
        T_target[0, 3] += dx
        T_target[1, 3] += dy
        T_target[2, 3] += dz
        
        print(f"\nPerturbation: dx={dx*1000:.1f}mm, dy={dy*1000:.1f}mm, dz={dz*1000:.1f}mm")
        
        # IK with RESET joints as initial guess
        ik_result_reset = fresh_solver.inverse_kinematics(
            current_joint_pos=joint_pos_deg,
            desired_ee_pose=T_target,
            position_weight=1.0,
            orientation_weight=0.01,
        )
        T_fk_reset = fresh_solver.forward_kinematics(ik_result_reset)
        error_reset = np.linalg.norm(T_fk_reset[:3, 3] - T_target[:3, 3]) * 1000
        
        # Create another fresh solver for zero guess
        fresh_solver_zero = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name="gripper_frame_link",
            joint_names=MOTOR_NAMES,
        )
        
        # IK with zero initial guess
        ik_result_zero = fresh_solver_zero.inverse_kinematics(
            current_joint_pos=zero_joints,
            desired_ee_pose=T_target,
            position_weight=1.0,
            orientation_weight=0.01,
        )
        T_fk_zero = fresh_solver_zero.forward_kinematics(ik_result_zero)
        error_zero = np.linalg.norm(T_fk_zero[:3, 3] - T_target[:3, 3]) * 1000
        
        print(f"  IK (RESET guess, fresh solver): error = {error_reset:.4f} mm")
        print(f"  IK (zero guess, fresh solver):  error = {error_zero:.4f} mm")
        
        # Show joint differences
        print("  Joint diffs (RESET guess):", end=" ")
        for i, name in enumerate(MOTOR_NAMES[:-1]):  # Skip gripper
            orig = RESET_POSE[f'{name}.pos']
            new = ik_result_reset[i]
            print(f"{name}={new-orig:+.2f}°", end=" ")
        print()

    print("\n" + "=" * 60)
    print("Debug Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
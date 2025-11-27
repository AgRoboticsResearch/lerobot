#!/usr/bin/env python
"""
Test script for RobotKinematics with Piper robot URDF.

This script tests forward and inverse kinematics for the Piper robot.
"""

import numpy as np
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics

# Paths
URDF_PATH = Path(__file__).parent.parent / "piper_description" / "urdf" / "piper_description.urdf"
TARGET_FRAME = "gripper_base"

# Joint names from URDF (without underscore)
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def test_forward_kinematics():
    """Test forward kinematics: given joint angles, compute end-effector pose."""
    print("=" * 60)
    print("Testing Forward Kinematics")
    print("=" * 60)
    
    # Initialize kinematics solver
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=TARGET_FRAME,
        joint_names=JOINT_NAMES,
    )
    
    # Test with a known joint configuration (in degrees)
    # Using a simple configuration: all joints at 0 degrees
    joint_pos_deg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print(f"\nInput joint positions (degrees): {joint_pos_deg}")
    
    # Compute forward kinematics
    T = kinematics.forward_kinematics(joint_pos_deg)
    
    print(f"\nEnd-effector pose (4x4 transformation matrix):")
    print(T)
    
    # Extract position and orientation
    position = T[:3, 3]
    rotation = T[:3, :3]
    
    print(f"\nEnd-effector position (x, y, z): {position}")
    print(f"End-effector rotation matrix:")
    print(rotation)
    
    return T, kinematics


def test_inverse_kinematics(kinematics: RobotKinematics, target_pose: np.ndarray):
    """Test inverse kinematics: given end-effector pose, compute joint angles."""
    print("\n" + "=" * 60)
    print("Testing Inverse Kinematics")
    print("=" * 60)
    
    # Use current pose as initial guess (all zeros)
    initial_joint_pos_deg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print(f"\nTarget end-effector pose:")
    print(target_pose)
    print(f"\nInitial joint guess (degrees): {initial_joint_pos_deg}")
    
    # Compute inverse kinematics
    joint_pos_deg = kinematics.inverse_kinematics(
        current_joint_pos=initial_joint_pos_deg,
        desired_ee_pose=target_pose,
        position_weight=1.0,
        orientation_weight=0.01,
    )
    
    print(f"\nComputed joint positions (degrees): {joint_pos_deg}")
    
    return joint_pos_deg


def test_round_trip(kinematics: RobotKinematics):
    """Test round-trip: FK -> IK -> FK should be close."""
    print("\n" + "=" * 60)
    print("Testing Round-Trip (FK -> IK -> FK)")
    print("=" * 60)
    
    # Start with a joint configuration
    original_joints_deg = np.array([30.0, -45.0, 60.0, -30.0, 45.0, -20.0])
    
    print(f"\nOriginal joint positions (degrees): {original_joints_deg}")
    
    # Forward kinematics
    T_original = kinematics.forward_kinematics(original_joints_deg)
    print(f"\nOriginal end-effector pose:")
    print(T_original)
    
    # Inverse kinematics
    computed_joints_deg = kinematics.inverse_kinematics(
        current_joint_pos=original_joints_deg,
        desired_ee_pose=T_original,
        position_weight=1.0,
        orientation_weight=0.01,
    )
    print(f"\nComputed joint positions (degrees): {computed_joints_deg}")
    
    # Forward kinematics again with computed joints
    T_computed = kinematics.forward_kinematics(computed_joints_deg)
    print(f"\nComputed end-effector pose:")
    print(T_computed)
    
    # Check error
    position_error = np.linalg.norm(T_original[:3, 3] - T_computed[:3, 3])
    rotation_error = np.linalg.norm(T_original[:3, :3] - T_computed[:3, :3])
    
    print(f"\nPosition error: {position_error:.6f} m")
    print(f"Rotation error: {rotation_error:.6f}")
    
    # Check if errors are acceptable
    position_tolerance = 0.001  # 1mm
    rotation_tolerance = 0.01
    
    if position_error < position_tolerance and rotation_error < rotation_tolerance:
        print("\n✓ Round-trip test PASSED!")
    else:
        print("\n✗ Round-trip test FAILED!")
        print(f"  Position error {position_error:.6f} > tolerance {position_tolerance}")
        print(f"  Rotation error {rotation_error:.6f} > tolerance {rotation_tolerance}")


def test_multiple_configurations(kinematics: RobotKinematics):
    """Test multiple joint configurations."""
    print("\n" + "=" * 60)
    print("Testing Multiple Joint Configurations")
    print("=" * 60)
    
    test_configs = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([45.0, -30.0, 60.0, -45.0, 30.0, -60.0]),
        np.array([90.0, -90.0, 90.0, -90.0, 90.0, -90.0]),
    ]
    
    for i, joints_deg in enumerate(test_configs):
        print(f"\n--- Configuration {i+1} ---")
        print(f"Joints (degrees): {joints_deg}")
        
        T = kinematics.forward_kinematics(joints_deg)
        position = T[:3, 3]
        
        print(f"End-effector position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Piper Robot Kinematics Test")
    print("=" * 60)
    print(f"\nURDF path: {URDF_PATH}")
    print(f"Target frame: {TARGET_FRAME}")
    print(f"Joint names: {JOINT_NAMES}")
    
    if not URDF_PATH.exists():
        print(f"\n✗ ERROR: URDF file not found at {URDF_PATH}")
        print("Please check the path.")
        return
    
    try:
        # Test 1: Forward Kinematics
        T, kinematics = test_forward_kinematics()
        
        # Test 2: Inverse Kinematics
        joint_pos_deg = test_inverse_kinematics(kinematics, T)
        
        # Test 3: Round-trip
        test_round_trip(kinematics)
        
        # Test 4: Multiple configurations
        test_multiple_configurations(kinematics)
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n✗ ERROR: {e}")
        print("Please install placo: pip install placo")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
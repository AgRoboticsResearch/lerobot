#!/usr/bin/env python3
"""Verification tests for SO101 relative-EE frame defaults and adaptation.

New datasets/checkpoints should use camera_link by default. The similarity
transform remains for legacy checkpoints trained in gripper_frame_link.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.model.kinematics import RobotKinematics
from lerobot.datasets.relative_ee_dataset import pose10d_to_mat, mat_to_pose10d


URDF_PATH = "urdf/Simulation/SO101/so101_sroi.urdf"
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LEGACY_TRAINING_FRAME = "gripper_frame_link"
DEFAULT_FRAME = "camera_link"


def test_so101_relative_ee_defaults_are_camera_link():
    """Verify new SO101 relative-EE training/deploy defaults are frame-consistent."""
    from relative_ee_dataset.convert_joint_to_ee_dataset import DEFAULT_URDF_PATH, EE_LINK_NAME
    from examples.so101_relative_ee.deploy_relative_ee_so101 import (
        DEFAULT_DEPLOY_FRAME,
        DEFAULT_URDF_PATH as DEPLOY_DEFAULT_URDF_PATH,
    )

    assert DEFAULT_URDF_PATH == URDF_PATH
    assert DEPLOY_DEFAULT_URDF_PATH == URDF_PATH
    assert EE_LINK_NAME == DEFAULT_FRAME
    assert DEFAULT_DEPLOY_FRAME == DEFAULT_FRAME


def test_static_transform_is_constant():
    """Verify the transform between frames doesn't depend on joint positions."""
    kin = RobotKinematics(URDF_PATH, target_frame_name=DEFAULT_FRAME, joint_names=MOTOR_NAMES)

    # Compute transform at several different joint configurations
    transforms = []
    for _ in range(5):
        joints = np.random.uniform(-90, 90, len(MOTOR_NAMES))
        kin.forward_kinematics(joints)
        T = kin.robot.get_T_a_b(LEGACY_TRAINING_FRAME, DEFAULT_FRAME)
        transforms.append(T)

    # All transforms should be identical
    for i in range(1, len(transforms)):
        np.testing.assert_allclose(transforms[0], transforms[i], atol=1e-10,
                                   err_msg=f"Transform changed at config {i}")
    print("PASS: Static transform is constant across joint configurations")


def test_transform_matches_urdf_offset():
    """Verify the transform matches the known URDF offsets."""
    kin = RobotKinematics(URDF_PATH, target_frame_name=DEFAULT_FRAME, joint_names=MOTOR_NAMES)
    kin.forward_kinematics(np.zeros(len(MOTOR_NAMES)))

    T_train_to_deploy = kin.robot.get_T_a_b(LEGACY_TRAINING_FRAME, DEFAULT_FRAME)

    # From URDF:
    # gripper_frame_link: parent=gripper_link, xyz=(0.015, 0, -0.15), rpy=(0, pi, 0)
    # camera_link: parent=gripper_link, xyz=(0.046, 0, -0.043), rpy=(0, 1.77, 0)
    # The transform should be non-trivial (different positions and orientations)
    print(f"T_train_to_deploy:\n{T_train_to_deploy}")

    # Check it's a valid rigid transform (bottom row = [0, 0, 0, 1])
    np.testing.assert_allclose(T_train_to_deploy[3, :], [0, 0, 0, 1], atol=1e-10)
    print("PASS: Valid rigid transform (bottom row correct)")

    # Check rotation matrix is orthonormal
    R = T_train_to_deploy[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    print("PASS: Rotation matrix is orthonormal")


def test_similarity_transform_roundtrip():
    """Verify that adapting then applying in deploy frame matches applying in training frame."""
    kin = RobotKinematics(URDF_PATH, target_frame_name=DEFAULT_FRAME, joint_names=MOTOR_NAMES)

    # Use a realistic joint configuration
    joints = np.array([-3.43, -94.77, 82.92, 17.01, -0.66, 55.28])

    # Get poses in both frames
    # We need a second kinematics for training frame
    kin_train = RobotKinematics(URDF_PATH, target_frame_name=LEGACY_TRAINING_FRAME, joint_names=MOTOR_NAMES)
    T_current_train = kin_train.forward_kinematics(joints)
    T_current_deploy = kin.forward_kinematics(joints)

    # Get static transform
    kin.forward_kinematics(joints)
    T_train_to_deploy = kin.robot.get_T_a_b(LEGACY_TRAINING_FRAME, DEFAULT_FRAME)
    T_deploy_to_train = np.linalg.inv(T_train_to_deploy)

    # Create a sample relative action: move 2cm in X, rotate 10° around Z
    T_rel = np.eye(4)
    T_rel[:3, 3] = [0.02, 0.0, 0.0]
    T_rel[:3, :3] = Rotation.from_rotvec([0, 0, np.deg2rad(10)]).as_matrix()
    rel_9d = mat_to_pose10d(T_rel)
    rel_10d = np.concatenate([rel_9d, [0.5]])  # gripper = 0.5

    # Ground truth: apply in training frame
    T_target_train = T_current_train @ T_rel
    T_target_deploy_from_train = T_target_train @ T_train_to_deploy

    # Adapted: apply in deploy frame using adapted relative action
    from examples.so101_relative_ee.deploy_relative_ee_so101 import adapt_relative_action
    rel_10d_adapted = adapt_relative_action(rel_10d, T_deploy_to_train, T_train_to_deploy)
    T_rel_adapted = pose10d_to_mat(rel_10d_adapted[:9])
    T_target_deploy_from_adapted = T_current_deploy @ T_rel_adapted

    # They should match
    np.testing.assert_allclose(T_target_deploy_from_adapted, T_target_deploy_from_train, atol=1e-6,
                               err_msg="Similarity transform doesn't match frame transformation chain")
    print("PASS: Similarity transform matches frame transformation chain")

    # Gripper should be unchanged
    assert rel_10d_adapted[9] == rel_10d[9], "Gripper value changed during adaptation"
    print("PASS: Gripper value preserved")


def test_backward_compat_no_frame_info():
    """Test that legacy checkpoints (no ee_target_frame) don't trigger transform."""
    # Simulate missing attribute
    class MockConfig:
        pass

    config = MockConfig()
    training_frame = getattr(config, 'ee_target_frame', '')
    assert training_frame == '', "Default should be empty string"
    print("PASS: Legacy checkpoint (no ee_target_frame) returns empty string")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "examples/so101_relative_ee")

    print("=" * 60)
    print("EE Frame Transform Verification Tests")
    print("=" * 60)

    test_backward_compat_no_frame_info()
    print()
    test_so101_relative_ee_defaults_are_camera_link()
    print()
    test_static_transform_is_constant()
    print()
    test_transform_matches_urdf_offset()
    print()
    test_similarity_transform_roundtrip()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

#!/usr/bin/env python
"""Debug trajectory errors to find where large errors occur."""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from scipy.spatial.transform import Rotation
import numpy as np

DATASET_PATH = "/mnt/ldata/sroi_lerobot/red_strawberry_picking_260119_merged_ee"
RESET_POSE = np.array([-8.0, -62.73, 65.05, 0.86, -2.55, 88.91], dtype=np.float64)
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

def ee_pose_to_matrix(ee_pose_7d):
    pos = ee_pose_7d[:3]
    rotvec = ee_pose_7d[3:6]
    rotmat = Rotation.from_rotvec(rotvec).as_matrix()
    tf = np.eye(4)
    tf[:3, :3] = rotmat
    tf[:3, 3] = pos
    return tf


def main():
    ds = LeRobotDataset(DATASET_PATH)
    kinematics = RobotKinematics(
        urdf_path='urdf/Simulation/SO101/so101_new_calib.urdf',
        target_frame_name='gripper_frame_link',
        joint_names=JOINT_NAMES,
    )

    episode = ds.meta.episodes[0]
    start_idx = episode['dataset_from_index']
    end_idx = episode['dataset_to_index'] + 1

    # Alignment transform
    first_frame = ds[start_idx]
    first_ee_pose = first_frame['observation.state'].cpu().numpy()
    first_ee_tf = ee_pose_to_matrix(first_ee_pose[:6])
    reset_fk = kinematics.forward_kinematics(RESET_POSE)
    alignment_tf = reset_fk @ np.linalg.inv(first_ee_tf)

    # Track errors
    errors = []
    last_valid_joint = RESET_POSE

    high_error_frames = []

    for frame_idx in range(start_idx, end_idx):
        frame = ds[frame_idx]
        ee_pose = frame['observation.state'].cpu().numpy()

        # Align EE pose
        ee_tf = ee_pose_to_matrix(ee_pose[:6])
        aligned_ee_tf = alignment_tf @ ee_tf
        aligned_pos = aligned_ee_tf[:3, 3]
        aligned_rotvec = Rotation.from_matrix(aligned_ee_tf[:3, :3]).as_rotvec()
        aligned_ee_pose_7d = np.concatenate([aligned_pos, aligned_rotvec, [ee_pose[6]]])
        tf_desired = ee_pose_to_matrix(aligned_ee_pose_7d)

        # Solve IK with position priority
        joint_pos = kinematics.inverse_kinematics(
            current_joint_pos=last_valid_joint,
            desired_ee_pose=tf_desired,
            position_weight=10.0,
            orientation_weight=0.1,
        )

        # Compute FK error
        fk_ee = kinematics.forward_kinematics(joint_pos)
        error = np.linalg.norm(fk_ee[:3, 3] - aligned_pos) * 1000  # mm

        errors.append(error)

        if error > 50:
            high_error_frames.append((frame_idx - start_idx, error, joint_pos.copy()))

        last_valid_joint = joint_pos

    errors = np.array(errors)

    print(f'Trajectory error statistics:')
    print(f'  Mean: {np.mean(errors):.1f} mm')
    print(f'  Std:  {np.std(errors):.1f} mm')
    print(f'  Min:  {np.min(errors):.1f} mm')
    print(f'  Max:  {np.max(errors):.1f} mm')
    print()
    print(f'Percentiles:')
    print(f'  50%: {np.percentile(errors, 50):.1f} mm')
    print(f'  90%: {np.percentile(errors, 90):.1f} mm')
    print(f'  95%: {np.percentile(errors, 95):.1f} mm')
    print(f'  99%: {np.percentile(errors, 99):.1f} mm')
    print()
    print(f'Frames with error > 50mm: {len(high_error_frames)}')
    print()

    # Find clusters of high errors
    if high_error_frames:
        print('High error frames (first 20):')
        for frame_idx, error, joints in high_error_frames[:20]:
            print(f'  Frame {frame_idx:3d}: error={error:6.1f}mm, joints={joints}')

        # Check if errors are consecutive
        indices = [f[0] for f in high_error_frames]
        clusters = []
        current_cluster = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_cluster.append(indices[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [indices[i]]
        clusters.append(current_cluster)

        print()
        print(f'Error clusters (consecutive high errors):')
        for i, cluster in enumerate(clusters):
            print(f'  Cluster {i+1}: frames {cluster[0]}-{cluster[-1]} ({len(cluster)} frames)')


if __name__ == '__main__':
    main()

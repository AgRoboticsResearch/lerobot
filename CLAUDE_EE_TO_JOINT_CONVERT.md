# EE to Joint Dataset Converter - Design and Verification

## Overview

This document describes the design and testing methodology for converting end-effector (EE) only datasets to standard LeRobotDataset format with virtual joint states computed via inverse kinematics.

**Purpose**: Enable training on UMI-style datasets (collected without robot state) using standard `lerobot_train.py`.

## Design

### Input Format
- **Dataset**: Standard LeRobot v3.0 format with EE poses
- **State**: 7D `[x, y, z, wx, wy, wz, gripper]`
  - Position: `[x, y, z]` in meters
  - Rotation: `[wx, wy, wz]` axis-angle representation
  - Gripper: scalar position value
- **Videos**: MP4 format, copied to output unchanged

### Output Format
- **Dataset**: Standard LeRobotDataset with joint positions
- **State**: 6D `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` in degrees
- **Actions**: Absolute joint positions (same as state)
- **Videos**: Identical to input

### IK Algorithm

For each frame in each episode:

```
1. Extract EE pose: ee_pose = frame['observation.state']  # (7,)
2. Convert to 4x4 transformation matrix:
   - T[:3, :3] = Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
   - T[:3, 3] = ee_pose[0:3]
3. Solve IK: joint_pos = inverse_kinematics(RESET_pose, T)
4. Verify solution:
   - T_fk = forward_kinematics(joint_pos)
   - pos_err = ||T_fk[:3,3] - T[:3,3]||
   - rot_err = geodesic_distance(T_fk[:3,:3], T[:3,:3])
   - valid = (pos_err < 0.005) and (rot_err < 0.1)
5. If invalid: retry IK with previous valid joint as guess
6. If still invalid: use previous valid joint (fallback)
7. Store: joint_positions[frame_idx] = joint_pos
```

### Key Configuration

```python
RESET_POSE_DEG = np.array([
    -8.00,    # shoulder_pan
    -62.73,   # shoulder_lift
    65.05,    # elbow_flex
    0.86,     # wrist_flex
    -2.55,    # wrist_roll
    88.91,    # gripper
])

# IK tolerances
POS_TOLERANCE = 0.005   # 5mm
ROT_TOLERANCE = 0.1      # ~5.7 degrees

# IK weights
POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 0.01
```

## Files

### `src/lerobot/datasets/ee_to_joint_converter.py`
Main converter module containing:
- `ee_pose_to_matrix()` - Convert 7D EE pose to 4x4 transform
- `verify_ik_solution()` - FK-based verification
- `solve_ik_with_fallback()` - IK with retry logic
- `convert_episode()` - Per-frame conversion
- `EEToJointDatasetConverter` - Main converter class

### `scripts/convert_ee_to_joint_dataset.py`
CLI tool for conversion:
```bash
python scripts/convert_ee_to_joint_dataset.py \
    --input-dataset /path/to/ee_dataset \
    --output-repo-id output_name \
    --urdf urdf/Simulation/SO101/so101_new_calib.urdf \
    --reset-pose -8.00 -62.73 65.05 0.86 -2.55 88.91
```

### `scripts/train_ee_to_joint.py`
Training script that uses standard `lerobot_train.train()` with the converted dataset.

## Test and Verification

### Phase 1: Unit Tests

Create `tests/test_ee_to_joint_converter.py`:

```python
import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.datasets.ee_to_joint_converter import (
    ee_pose_to_matrix,
    verify_ik_solution,
    solve_ik_with_fallback,
)

def test_ee_pose_to_matrix():
    """Test EE pose to transformation matrix conversion."""
    # Identity rotation, zero position
    ee_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    T = ee_pose_to_matrix(ee_pose)
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [0, 0, 0])

    # Known rotation (90 deg around Z), known position
    ee_pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, np.pi/2, 0.5])
    T = ee_pose_to_matrix(ee_pose)
    # Check position
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3])
    # Check rotation (90° around Z maps X->Y, Y->-X)
    assert np.allclose(T[0, :3], [0, -1, 0], atol=1e-6)
    assert np.allclose(T[1, :3], [1, 0, 0], atol=1e-6)

def test_verify_ik_solution():
    """Test IK verification using FK."""
    kin = RobotKinematics("urdf/Simulation/SO101/so101_new_calib.urdf")
    reset_pose = np.array([-8.00, -62.73, 65.05, 0.86, -2.55, 88.91])

    # FK from reset pose
    T_reset = kin.forward_kinematics(reset_pose)

    # Verify should pass for same pose
    is_valid, pos_err, rot_err = verify_ik_solution(kin, reset_pose, T_reset)
    assert is_valid
    assert pos_err < 1e-6
    assert rot_err < 1e-6

    # Verify should fail for different pose
    different_pose = reset_pose + 10  # 10 degrees offset
    is_valid, pos_err, rot_err = verify_ik_solution(kin, different_pose, T_reset)
    assert not is_valid

def test_solve_ik_with_fallback():
    """Test IK with verification and fallback."""
    kin = RobotKinematics("urdf/Simulation/SO101/so101_new_calib.urdf")
    reset_pose = np.array([-8.00, -62.73, 65.05, 0.86, -2.55, 88.91])

    # Get target EE pose from RESET
    T_target = kin.forward_kinematics(reset_pose)
    ee_pose = np.concatenate([
        T_target[:3, 3],
        Rotation.from_matrix(T_target[:3, :3]).as_rotvec(),
        [88.91]
    ])

    # Should solve successfully
    joint_pos, success, pos_err, rot_err = solve_ik_with_fallback(
        kin, ee_pose, reset_pose, None
    )
    assert success
    assert pos_err < 0.005
    assert rot_err < 0.1
    # Check joints are close to original
    assert np.allclose(joint_pos[:5], reset_pose[:5], atol=2)  # 2 deg tolerance
```

### Phase 2: Integration Test on Real Data

Test conversion on a small dataset:

```python
def test_convert_single_episode():
    """Test full conversion of a single episode."""
    from lerobot.datasets.ee_to_joint_converter import EEToJointDatasetConverter

    converter = EEToJointDatasetConverter(
        ee_dataset_path="/mnt/ldata/sroi_lerobot/red_strawberry_picking_260119_merged_ee",
        output_repo_id="test_output_joint",
    )

    # Convert just first episode
    result = converter.convert_episode(0)

    # Check output shape
    episode = converter.ee_dataset.meta.episodes[0]
    num_frames = episode["length"]
    assert result['joint_positions'].shape == (num_frames, 6)

    # Check success rate > 90%
    assert result['success_rate'] > 0.9

    # Check joint values are within reasonable bounds
    assert np.all(np.abs(result['joint_positions'][:, :5]) < 180)  # Joint angles

    print(f"Episode 0: success={result['success_rate']:.1%}, "
          f"pos_err={result['avg_pos_error']*1000:.1f}mm, "
          f"rot_err={np.rad2deg(result['avg_rot_error']):.1f}deg")
```

### Phase 3: FK Verification

Verify that FK on computed joints matches original EE poses:

```python
def test_fk_matches_original_ee():
    """Verify FK on output joints matches input EE poses."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Load datasets
    ee_ds = LeRobotDataset("/path/to/ee_dataset")
    joint_ds = LeRobotDataset("/path/to/joint_dataset")
    kin = RobotKinematics("urdf/Simulation/SO101/so101_new_calib.urdf")

    episode_idx = 0
    episode = ee_ds.meta.episodes[episode_idx]
    start_idx = episode["dataset_from_index"]
    end_idx = episode["dataset_to_index"]

    pos_errors = []
    rot_errors = []

    for frame_idx in range(start_idx, end_idx):
        # Get original EE pose
        ee_frame = ee_ds[frame_idx]
        ee_pose = ee_frame['observation.state'].numpy()  # (7,)
        T_ee = ee_pose_to_matrix(ee_pose)

        # Get joint positions and compute FK
        joint_frame = joint_ds[frame_idx]
        joint_pos = joint_frame['observation.state'].numpy()
        T_fk = kin.forward_kinematics(joint_pos)

        # Compute errors
        pos_err = np.linalg.norm(T_fk[:3, 3] - T_ee[:3, 3])
        rot_err = np.linalg.norm(Rotation.from_matrix(
            T_ee[:3, :3].T @ T_fk[:3, :3]
        ).as_rotvec())

        pos_errors.append(pos_err)
        rot_errors.append(rot_err)

    avg_pos_err = np.mean(pos_errors) * 1000  # mm
    max_pos_err = np.max(pos_errors) * 1000
    avg_rot_err = np.rad2deg(np.mean(rot_errors))

    print(f"Average position error: {avg_pos_err:.1f}mm")
    print(f"Max position error: {max_pos_err:.1f}mm")
    print(f"Average rotation error: {avg_rot_err:.1f}deg")

    # Assertions
    assert avg_pos_err < 10, f"Average pos error too high: {avg_pos_err}mm"
    assert max_pos_err < 50, f"Max pos error too high: {max_pos_err}mm"
    assert avg_rot_err < 15, f"Average rot error too high: {avg_rot_err}deg"
```

### Phase 4: Visual Verification

Create visualization script to inspect conversion quality:

```python
# scripts/verify_ee_to_joint_conversion.py
"""Visual verification of EE to Joint conversion."""

import numpy as np
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.datasets.ee_to_joint_converter import ee_pose_to_matrix

def plot_conversion_quality(ee_dataset_path, joint_dataset_path, episode_idx=0):
    """Plot EE pose comparison and IK success statistics."""
    ee_ds = LeRobotDataset(ee_dataset_path)
    joint_ds = LeRobotDataset(joint_dataset_path)
    kin = RobotKinematics("urdf/Simulation/SO101/so101_new_calib.urdf")

    episode = ee_ds.meta.episodes[episode_idx]
    start_idx = episode["dataset_from_index"]
    end_idx = episode["dataset_to_index"]

    ee_poses = []
    fk_poses = []
    joint_poses = []

    for frame_idx in range(start_idx, end_idx):
        ee_frame = ee_ds[frame_idx]
        joint_frame = joint_ds[frame_idx]

        ee_pose = ee_frame['observation.state'].numpy()
        joint_pos = joint_frame['observation.state'].numpy()

        T_ee = ee_pose_to_matrix(ee_pose)
        T_fk = kin.forward_kinematics(joint_pos)

        ee_poses.append(T_ee[:3, 3])
        fk_poses.append(T_fk[:3, 3])
        joint_poses.append(joint_pos)

    ee_poses = np.array(ee_poses)
    fk_poses = np.array(fk_poses)
    joint_poses = np.array(joint_pos)

    # Plot 1: EE position comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    labels = ['X (m)', 'Y (m)', 'Z (m)']
    for i, ax in enumerate(axes):
        ax.plot(ee_poses[:, i], label='Original EE', alpha=0.7)
        ax.plot(fk_poses[:, i], label='FK from joints', alpha=0.7, linestyle='--')
        ax.set_ylabel(labels[i])
        ax.legend()
    axes[-1].set_xlabel('Frame index')
    fig.suptitle('EE Position: Original vs FK from Converted Joints')
    plt.tight_layout()
    plt.savefig('ee_position_comparison.png')

    # Plot 2: Joint positions
    fig, axes = plt.subplots(6, 1, figsize=(12, 10))
    joint_labels = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                    'wrist_flex', 'wrist_roll', 'gripper']
    for i, ax in enumerate(axes):
        ax.plot(np.rad2deg(joint_poses[:, i]))
        ax.set_ylabel(f'{joint_labels[i]} (deg)')
    axes[-1].set_xlabel('Frame index')
    fig.suptitle('Converted Joint Positions')
    plt.tight_layout()
    plt.savefig('joint_positions.png')

    print(f"Plots saved to ee_position_comparison.png and joint_positions.png")

if __name__ == "__main__":
    import sys
    plot_conversion_quality(sys.argv[1], sys.argv[2])
```

### Phase 5: End-to-End Training Test

Verify training works on converted dataset:

```bash
# 1. Convert dataset
python scripts/convert_ee_to_joint_dataset.py \
    --input-dataset /mnt/ldata/sroi_lerobot/red_strawberry_picking_260119_merged_ee \
    --output-repo-id red_strawberry_picking_joint_test

# 2. Quick training test (100 steps)
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=red_strawberry_picking_joint_test \
    --env.type=so101_sim \
    --steps=100 \
    --eval.freq=50

# 3. Check that loss decreases
# Loss should start high and decrease over 100 steps
```

## Success Criteria

### Conversion Quality
- [ ] IK success rate > 90% for all episodes
- [ ] Average position error < 10mm
- [ ] Average rotation error < 15 degrees
- [ ] Max position error < 50mm

### Dataset Validity
- [ ] Output dataset loads with `LeRobotDataset`
- [ ] All frames have valid joint positions (no NaN/Inf)
- [ ] Joint positions within robot limits
- [ ] Videos are correctly copied and playable

### Training
- [ ] `lerobot-train` runs without errors
- [ ] Training loss decreases
- [ ] Policy checkpoint saves and loads correctly

## Troubleshooting

### Low IK Success Rate
- **Symptom**: Success rate < 80%
- **Possible causes**:
  - EE poses outside robot workspace
  - URDF mismatch with data collection setup
  - Incorrect target frame name
- **Solutions**:
  - Check workspace: visualize EE trajectories
  - Verify URDF matches real robot kinematics
  - Try `target_frame="gripper_link"` instead of `"gripper_frame_link"`

### High Position Error
- **Symptom**: Average error > 20mm
- **Possible causes**:
  - Loose IK tolerances
  - Incorrect axis-angle conversion
  - URDF joint offsets
- **Solutions**:
  - Tighten tolerances: `--pos-tolerance 0.001`
  - Verify `ee_pose_to_matrix()` with known transforms
  - Check URDF joint calibration

### Joint Limits Violated
- **Symptom**: Joint positions > 180 degrees or NaN
- **Possible causes**:
  - Unreachable EE poses
  - IK solver divergence
- **Solutions**:
  - Add joint limit clamping in converter
  - Skip frames with unreachable poses
  - Use orientation_weight=0 to prioritize position

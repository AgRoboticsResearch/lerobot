# Deploy Relative EE Policy on SO101 Robot

## Overview

This document describes the deployment system for ACT policies trained with `RelativeEEDataset` (UMI-style relative end-effector actions) on the SO101 robot.

## Background

### What is Relative EE?

Traditional robot policies output absolute end-effector poses: "move to position (x, y, z)". Relative EE policies output **relative** transformations: "move by (dx, dy, dz) from current pose".

**UMI-style** (Universal Manipulation Interface) uses SE(3) transformations where:
- Actions are relative to the current timestep
- Uses 6D rotation representation (first two columns of rotation matrix)
- Format: `[dx, dy, dz, rot6d_0..5, gripper]` = 10 dimensions

### The Challenge

When deploying a relative EE policy on a real robot:
1. **Policy outputs**: 10D relative EE poses
2. **Robot expects**: Joint positions
3. **Need**: IK (inverse kinematics) to convert EE → joints

### RelativeEEDataset Action Format

**IMPORTANT**: In `RelativeEEDataset`, ALL actions in a chunk are relative to the SAME base observation (the EE pose at chunk start):

```
Predicted chunk: [action_0, action_1, action_2, ...]

Where:
action_0 = T_base^(-1) @ T_1  (from base to step 1)
action_1 = T_base^(-1) @ T_2  (from base to step 2)
action_t = T_base^(-1) @ T_(t+1)  (from base to step t+1)

Each action is: "move from BASE to target", NOT "move from previous step"
```

To execute action at step t: `T_target = T_base @ action_t`

## Design Logic

### Deployment Solution

1. **Each chunk is independent** - Always predict from the ACTUAL current robot pose
2. **Within a chunk: Apply from base** - Each action is applied from the chunk base pose

```python
# At prediction time (once per chunk)
current_pose = get_actual_robot_pose()  # From FK
chunk_base_pose = current_pose  # Store as base for all actions in this chunk
chunk = policy.predict(observation)      # All relative to chunk_base_pose

# During execution (for each action in chunk)
for action in chunk:
    # Apply from base: T_target = T_base @ T_action
    target_pose = chunk_base_pose @ action
    execute(target_pose)
```

This approach:
- ✅ Matches RelativeEEDataset format (all actions relative to base)
- ✅ Handles drift naturally (each chunk starts fresh from actual pose)
- ✅ Correctly interprets each action independently

## Architecture

### Components

```
┌─────────────────┐
│  ACT Policy     │  Outputs (chunk_size, 10) relative poses
│  (RelativeEED)  │  All actions relative to chunk base
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Relative10DAccumulatedToAbsoluteEE     │
│  - Gets chunk_base_pose from state     │
│  - Applies: T_target = T_base @ T_rel  │
│  - Outputs absolute ee.x/y/z/wx/wy/wz   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  EEBoundsAndSafety                      │
│  - Clips to workspace bounds            │
│  - Checks for unsafe jumps              │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  InverseKinematicsEEToJoints            │
│  - IK: absolute EE pose → joints        │
│  - Outputs shoulder_pan.pos, etc.       │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  SO101 Robot    │
└─────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `relative_ee_processor.py` | Custom processors for 10D → absolute conversion |
| `deploy_relative_ee_so101.py` | Main deployment script |
| `robot_kinematic_processor.py` | Existing IK/FK processors |

## Usage

### Basic Usage

```bash
python examples/so101_relative_ee/deploy_relative_ee_so101.py \
    --pretrained_path outputs/train/my_model \
    --urdf_path /path/to/so101.urdf \
    --robot_port /dev/ttyUSB0
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pretrained_path` | required | Path to model checkpoint or training output |
| `--urdf_path` | required | Path to SO101 URDF file for IK |
| `--robot_port` | `/dev/ttyUSB0` | Serial port for robot connection |
| `--fps` | `10` | Control loop frequency (Hz) |
| `--num_steps` | `0` | Number of steps to run (0 = infinite) |
| `--obs_state_horizon` | `2` | Must match training value |
| `--n_action_steps` | `10` | Actions executed per prediction |
| `--warm_start` | `false` | Move to reset pose before starting |
| `--ee_bounds_min` | `-0.5 -0.5 0.0` | EE position minimum (m) |
| `--ee_bounds_max` | `0.5 0.5 0.4` | EE position maximum (m) |
| `--max_ee_step_m` | `0.05` | Max step size for safety (m) |

### Training with Matching Config

Your training should use:

```bash
python train_relative_ee.py \
    --dataset.repo_id=your_dataset \
    --policy.type=act \
    --policy.obs_state_horizon=2 \
    --policy.chunk_size=100 \
    # ... other params
```

The deployment script will:
1. Load the checkpoint
2. Use `obs_state_horizon=2` for observation
3. Execute 10 actions per prediction (`n_action_steps`)
4. Re-predict when actions exhausted

## Implementation Details

### Observation Format

The policy expects observations in **relative 10D format**:
- Current timestep is always **identity**: `[0,0,0, 1,0,0,0,1,0, gripper]`
- Historical timesteps stored in buffer

```python
def create_relative_observation(current_ee_T, gripper_pos, obs_state_horizon=2):
    # Current obs is always identity relative to itself
    current_obs = np.array([
        0.0, 0.0, 0.0,              # position (identity)
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # rot6d (identity rotation)
        gripper_pos,
    ])
    # Stack with history
    return stack_with_history(current_obs, history_buffer)
```

### Action Application Logic

```python
# When new chunk is predicted
current_ee_T = kinematics.forward_kinematics(current_joints)
chunk_base_pose = current_ee_T  # All actions relative to this base

# For each action in chunk
rel_T = pose10d_to_mat(action_10d[:9])
target_ee_pose = chunk_base_pose @ rel_T  # Apply from base (NOT chaining)

# Convert to joints and execute
target_joints = kinematics.inverse_kinematics(current_joints, target_ee_pose)
robot.send_action(target_joints)
```

### Processor Pipeline

The deployment uses `RobotProcessorPipeline` like the teleoperation example:

```python
ee_to_joints_pipeline = RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
](
    steps=[
        Relative10DAccumulatedToAbsoluteEE(gripper_scale=100.0),
        EEBoundsAndSafety(
            end_effector_bounds={"min": min_bounds, "max": max_bounds},
            max_ee_step_m=0.05,
        ),
        InverseKinematicsEEToJoints(
            kinematics=kinematics,
            motor_names=MOTOR_NAMES,
            initial_guess_current_joints=False,
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)
```

### Gripper Scaling

- Policy outputs gripper in `[0, 1]` range
- Robot expects `[0, 100]` range
- Processor scales by `gripper_scale=100.0`

## Safety Features

1. **EE Bounds** - Clips target position to workspace limits
2. **Max Step Size** - Prevents sudden large movements
3. **FK Verification** - Current pose always computed from actual joints

## Troubleshooting

### "IK failed"
- Check URDF path is correct
- Verify target is within workspace bounds
- Try increasing `--max_ee_step_m`

### "Robot not connected"
- Check `--robot_port` matches your device
- Verify robot is powered on
- Try `ls /dev/tty*` to find correct port

### Policy seems random
- Verify checkpoint was trained with `RelativeEEDataset`
- Check `obs_state_horizon` matches training
- Ensure dataset has correct 10D format

### Robot trajectory doesn't match expected
- Verify action chaining uses `chunk_base_pose` not sequential accumulation
- Check that `chunk_base_pose` is set once at chunk start, not updated

## References

- `train_relative_ee.py` - Training script
- `test_relative_ee_policy.py` - Test/validation script
- `examples/so100_to_so100_EE/teleoperate.py` - Teleoperation pattern
- UMI: https://github.com/real-stanford/universal_manipulation_interface

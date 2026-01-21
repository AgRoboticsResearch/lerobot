# SO101 Placo Simulation

Basic motion simulations for the SO101 robot arm using placo inverse kinematics.

## Prerequisites

```bash
pip install placo pinocchio ischedule placo-utils
```

## Scripts

### `so101_basic.py`
Simple end-effector following a sinusoidal target in 3D space. Good starting point to understand the setup.

### `so101_trajectory.py`
End-effector follows a figure-8 (∞) trajectory with visual trail showing the path.

### `so101_joints.py`
Direct joint-space control with sinusoidal patterns on each joint. More predictable than IK-based control.

## Running

```bash
# From the lerobot root directory
cd placo_sim
python so101_basic.py
# or
python so101_trajectory.py
# or
python so101_joints.py
```

## Robot Details

- **URDF**: `urdf/Simulation/SO101/so101_new_calib.urdf`
- **Joints**: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
- **End Effector**: `gripper_frame_link`
- **Base**: Fixed (floating base masked)

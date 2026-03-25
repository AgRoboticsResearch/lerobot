# LIBERO Control Modes and OSC Controller

## Control Modes

LIBERO supports two control modes that determine how actions are interpreted:

### Relative Mode (`control_mode="relative"`)

- **Policy outputs**: Changes from current end-effector pose
  - `action[:6]` = delta position + delta orientation
  - `action[6]` = delta gripper
- **Applied as**: `new_pose = current_pose + scaled_action`
- **Training data**: Demonstrations record frame-to-frame differences
- **Use case**: Imitation learning from smooth demo trajectories
- **Current training**: ✅ **Using this mode**

### Absolute Mode (`control_mode="absolute"`)

- **Policy outputs**: Absolute target end-effector pose
  - `action[:6]` = target position + target orientation
  - `action[6]` = gripper state
- **Applied as**: `new_pose = scaled_action` (directly)
- **Training data**: Demonstrations record absolute poses
- **Use case**: Goal-conditioned policies, VLA models

**Important**: Both modes use identical action space `Box(-1, 1, shape=(7,))`. The difference is purely semantic - how those values are interpreted.

## OSC (Operational Space Control) Controller

### What is OSC?

**Operational Space Control** is a control framework for robots that operates in **task space** (end-effector coordinates) rather than joint space.

#### Core Idea
Instead of controlling individual joint angles, OSC controls the **end-effector pose** (position + orientation) directly:
- **Inputs**: Desired end-effector position/orientation (or delta)
- **Outputs**: Joint torques to achieve that pose

#### Mathematical Foundation

Based on Khatib's 1987 paper ["A Unified Approach for Motion and Force Control of Robot Manipulators"](http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf):

**Task Space Dynamics:**
```
Λ(x) * ẍ + μ(x, ẋ) + p(x) = F
```
Where:
- `Λ(x)` = Task space inertia matrix (pseudo-inertia)
- `μ(x, ẋ)` = Centrifugal/Coriolis forces in task space
- `p(x)` = Gravity forces in task space
- `F` = Control force in task space

**Control Law:**
```
F = Kp * (x_target - x_current) + Kd * (ẋ_target - ẋ_current)
```

**Joint Torque Mapping:**
```
τ = J^T * F + τ_gravity + τ_nullspace
```
Where:
- `J` = Jacobian matrix (maps joint velocities to end-effector velocities)
- `J^T * F` = Project task space force to joint torques
- `τ_gravity` = Gravity compensation
- `τ_nullspace` = Secondary task (e.g., joint pose optimization)

### Why OSC for LIBERO?

| Feature | Joint Space Control | OSC (Task Space) |
|---|---|---|
| Natural for manipulation | ❌ Need to compute IK | ✅ Direct EE control |
| Handles singularities | ⚠️ Problematic | ✅ Graceful degradation |
| Intuitive for learning | ❌ Joint angles meaningless | ✅ EE pose = task goal |
| Used in VLAs | Rare | ✅ OpenVLA, etc. |

### OSC in Code

**File**: `/home/zfei/anaconda3/envs/py310/lib/python3.10/site-packages/robosuite/controllers/osc.py`

**Implementation**:
```python
# Position error
position_error = desired_pos - self.ee_pos

# PD control law
desired_force = kp * position_error + kd * (-self.ee_pos_vel)

# Orientation error (axis-angle)
ori_error = orientation_error(desired_ori, self.ee_ori_mat)
desired_torque = kp * ori_error + kd * (-self.ee_ori_vel)

# Combine
wrench = [desired_force, desired_torque]

# Project to joint space via Jacobian transpose
lambda_matrix = (J * M^-1 * J^T)^-1  # Task space inertia
decoupled_wrench = lambda_matrix * wrench
torques = J^T * decoupled_wrench + gravity_compensation
```

**Your LIBERO Setup:**
- **Input**: 6D delta (dx, dy, dz, droll, dpitch, dyaw) in `Box(-1, 1)`
- **Scaling**: `[-1,1]` → `[±0.05m, ±0.5rad]` (configurable via `output_min/max`)
- **Output**: 7D joint torques for the Franka-style arm

## Complete Action Execution Chain

### The Final Execution Call

```python
# /home/zfei/anaconda3/envs/py310/lib/python3.10/site-packages/robosuite/robots/single_arm.py:259
self.sim.data.ctrl[self._ref_joint_actuator_indexes] = self.torques
```

### Full Execution Flow

```
Policy outputs action (7D): [-1, 1] → Box space
         ↓
LiberoEnv.step(action)
         ↓
robot.control(action, policy_step=True)
         ↓
controller.set_goal(arm_action)  # OSC controller
         │
         ├─ if use_delta=True:  # YOUR CURRENT MODE
         │   goal_pos = ee_pos + scaled_delta[:3]
         │   goal_ori = ee_ori + scaled_delta[3:6]
         │
         └─ if use_delta=False:  # Absolute mode
             goal_pos = scaled_delta[:3]
             goal_ori = scaled_delta[3:6]
         ↓
controller.run_controller()
         │
         ├─ Compute position/orientation error
         ├─ Compute desired force/torque: F = kp * error + kd * vel_error
         ├─ Convert to joint torques via OSC: τ = J^T * F + gravity_comp
         └─ Return: self.torques (7D)
         ↓
np.clip(torques, torque_limits[0], torque_limits[1])
         ↓
self.sim.data.ctrl[actuator_indexes] = torques  # ← FINAL EXECUTION
         ↓
MuJoCo simulator steps forward
```

### Key Files Reference

| Component | File Path |
|---|---|
| **OSC Controller** | `/home/zfei/anaconda3/envs/py310/lib/python3.10/site-packages/robosuite/controllers/osc.py` |
| **Robot Control** | `/home/zfei/anaconda3/envs/py310/lib/python3.10/site-packages/robosuite/robots/single_arm.py` |
| **LIBERO Env** | `/home/zfei/code/lerobot/src/lerobot/envs/libero.py` |
| **Base Env** | `/home/zfei/anaconda3/envs/py310/lib/python3.10/site-packages/libero/libero/envs/bddl_base_domain.py` |

## LeRobot Integration

When using LIBERO through LeRobot:

### Observations
- `observation.state` – proprioceptive features (agent state)
- `observation.images.image` – main camera view (agentview_image)
- `observation.images.image2` – wrist camera view (robot0_eye_in_hand_image)

### Actions
- Continuous control values in `Box(-1, 1, shape=(7,))` space
- 6D for end-effector (position + orientation) + 1D for gripper

### Environment Configuration
```python
# In LeRobot configs
env.type = "libero"
env.task = "libero_object"  # or libero_spatial, libero_goal, etc.
env.control_mode = "relative"  # or "absolute"
env.init_states = True  # Use demonstration init states
```

## References

- Khatib, O. (1987). "A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation" [PDF](http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf)
- LIBERO Documentation: https://lifelong-robotic-learning.github.io/libero-site/
- LeRobot Documentation: https://github.com/huggingface/lerobot

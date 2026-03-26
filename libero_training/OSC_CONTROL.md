# LIBERO Control Modes and Robot Control Architectures

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

---

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

---

## LIBERO Action Execution Chain

### The Final Execution Call

```python
# /home/zfei/anaconda3/envs/py310/lib/python3.10/site-packages/robosuite/robots/single_arm.py:259
self.sim.data.ctrl[self._ref_joint_actuator_indexes] = self.torques
```

### Full Execution Flow (LIBERO Simulation)

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

### Key Files Reference (LIBERO)

| Component | File Path |
|---|---|
| **OSC Controller** | `/home/zfei/anaconda3/envs/py310/lib/python3.10/site-packages/robosuite/controllers/osc.py` |
| **Robot Control** | `/home/zfei/anaconda3/envs/py310/lib/python3.10/site-packages/robosuite/robots/single_arm.py` |
| **LIBERO Env** | `/home/zfei/code/lerobot/src/lerobot/envs/libero.py` |
| **Base Env** | `/home/zfei/anaconda3/envs/py310/lib/python3.10/site-packages/libero/libero/envs/bddl_base_domain.py` |

---

## Real Robot Control Architectures

### Comparison: Simulation vs Real Robots

| Aspect | LIBERO (Simulation) | SO101 (Real Robot) | UMI-ARX (Real Robot) | UR/Franka (Real Robot) |
|---|---|---|---|---|
| **Control Method** | OSC (torque) | IK → Position | TCP Pose (ZMQ) | Built-in Cartesian |
| **Output** | Joint torques | Joint positions | EE poses | EE poses |
| **Low-level** | MuJoCo physics | Hardware PID | Robot firmware | Robot firmware |
| **IK Location** | Controller (Python) | Placo (Python) | Robot firmware | Robot firmware |
| **Communication** | In-process | Serial/USB | ZMQ (REQ-REP) | RTDE/EtherCAT |
| **Control Loop** | 20 Hz | 30 Hz | 100 Hz | Up to 500 Hz |
| **Dynamics Model** | Perfect (MuJoCo) | Unknown | Unknown | Identified |
| **Safety** | Not needed | Hardware limits | Hardware limits | Built-in safety |

---

### 1. SO101 Control (LeRobot Approach)

**Architecture**: IK → Position Control → Hardware PID

**Execution Chain**:
```
Policy (10D relative EE)
    ↓
Relative10DAccumulatedToAbsoluteEE (Processor)
    ↓
EEBoundsAndSafety (Processor)
    ↓
InverseKinematicsEEToJoints (Processor, Placo-based)
    ↓
robot.send_action(joint_positions)
    ↓
SO101 Hardware PID Motors
```

**Final Execution** (`examples/so101_relative_ee/deploy_relative_ee_so101.py:764`):
```python
robot.send_action(joints_action)  # Joint positions
```

**Key Characteristics**:
- ✅ Works on any position-controlled robot
- ✅ Safety through position limits
- ✅ No dynamics model needed
- ⚠️ Requires IK implementation
- ⚠️ Dependent on IK quality

---

### 2. UMI-ARX Control (Multi-Process + ZMQ)

**Architecture**: Multi-process controller with ZMQ communication

**Execution Chain**:
```
Policy → Action Queue → Arx5Controller (100 Hz process)
                              ↓
                    PoseTrajectoryInterpolator
                              ↓
                    ZMQ Client (REQ-REP socket)
                              ↓
                    Robot Controller Server (handles IK)
                              ↓
                    Shared Memory Ring Buffer (feedback)
```

**Final Execution** (`modules/arx5_zmq_client.py:161-162`):
```python
self.socket.send_pyobj({
    "cmd": "SET_EE_POSE",
    "data": {"ee_pose": pose_6d, "gripper_pos": gripper_pos}
})
reply_msg = self.socket.recv_pyobj()
```

**Key Characteristics**:
- ✅ Decoupled control (separate process)
- ✅ High-frequency control (100 Hz)
- ✅ Trajectory smoothing
- ✅ Efficient shared memory
- ⚠️ More complex architecture
- ⚠️ Requires robot controller server

**Files** (`/home/zfei/code/umi-arx/`):
- `modules/arx5_controller.py` - Multi-process controller
- `modules/arx5_zmq_client.py` - ZMQ communication
- `modules/pose_trajectory_interpolator.py` - Trajectory smoothing

---

### 3. UR/Franka Control (Built-in Cartesian)

**Architecture**: Use robot's built-in Cartesian/impedance control

**UR Example** (RTDE):
```python
import rtde_control
rtde_control.sendPose(pose_x, pose_y, pose_z, rx, ry, rz)
# Robot handles IK + dynamics internally
```

**Franka Example** (Cartesian Impedance):
```python
robot.setCartesianImpedance([2000, 2000, 2000, 200, 200, 200])
robot.setCartesianPose(target_pose)
# Robot handles IK + dynamics + safety
```

**Key Characteristics**:
- ✅ Manufacturer-optimized control
- ✅ Built-in safety layers
- ✅ Identified dynamics
- ✅ High performance
- ⚠️ Robot-specific APIs
- ⚠️ May require proprietary SDKs

---

### 4. Pepper Arm Control

**Specifications**:
- 5 DOF per arm (limited for manipulation)
- Position control only
- Cable-driven (high friction)
- Designed for gestures, not manipulation

**Control Method**: Direct Joint Control
```python
import qi
motion = ALProxy("ALMotion", pepper_ip, 9559)
motion.setAngles(["LShoulderPitch", "LShoulderRoll", ...], angles, speed)
```

**NOT Recommended**:
- ❌ OSC - No torque interface
- ❌ IK - Limited by 5 DOF
- ⚠️ General manipulation - Not designed for it

---

## Real Robot Control Recommendations

### By Robot Type

| Robot | Recommended Method | Why |
|---|---|---|
| **SO101** | IK + Position | No dynamics, simple hardware |
| **UMI-ARX** | ZMQ TCP Pose | Already implemented, efficient |
| **UR5/e-series** | Built-in RTDE Cartesian | Has impedance control |
| **Franka** | Built-in Cartesian Impedance | Excellent dynamics, 7 DOF |
| **Pepper** | Direct Joint (or pre-programmed) | Limited DOF, gesture-focused |

### When to Use Each Method

**Use IK → Position Control** when:
- Robot has no built-in Cartesian control
- You have a reliable IK solver
- Safety through position limits is sufficient
- Examples: SO101, hobbyist robots

**Use Built-in Cartesian Control** when:
- Robot has identified dynamics
- Manufacturer provides safety layers
- You need impedance/force control
- Examples: UR, Franka, KUKA

**Use Custom OSC** when:
- You have accurate dynamics model
- Robot provides torque control interface
- Need specialized behaviors
- Examples: Research platforms with torque sensors

**Use ZMQ/Network Architecture** when:
- Need high-frequency decoupled control
- Robot has separate controller server
- Want smooth trajectory interpolation
- Examples: UMI-ARX

---

## LeRobot Integration

### LIBERO Through LeRobot

**Observations**:
- `observation.state` – proprioceptive features (agent state)
- `observation.images.image` – main camera view (agentview_image)
- `observation.images.image2` – wrist camera view (robot0_eye_in_hand_image)

**Actions**:
- Continuous control values in `Box(-1, 1, shape=(7,))` space
- 6D for end-effector (position + orientation) + 1D for gripper

**Environment Configuration**:
```python
env.type = "libero"
env.task = "libero_object"  # or libero_spatial, libero_goal, etc.
env.control_mode = "relative"  # or "absolute"
env.init_states = True  # Use demonstration init states
```

---

## Summary

### Key Takeaways

1. **LIBERO (Simulation)**: Uses OSC with direct torque control in MuJoCo
2. **SO101 (Real)**: Uses IK → Position control via LeRobot processors
3. **UMI-ARX (Real)**: Uses ZMQ-based EE pose control with robot-side IK
4. **Industrial Arms**: Use built-in Cartesian/impedance control
5. **Pepper**: Limited to joint-level control for gestures

### Control Hierarchy

```
High Level (Policy)
        ↓
Task Space (EE Pose)
        ↓
    IK Solver?
        ↓
    ↓                    ↓
Joint Torques      Joint Positions
(OSC)              (Position Control)
        ↓                    ↓
   Direct Torque        Hardware PID
```

### Choosing the Right Approach

**For your robot deployment**:
1. Check if robot has built-in Cartesian control → Use it
2. Otherwise, use IK → Position control (LeRobot default)
3. Only implement custom OSC if you have torque control + good dynamics

---

## References

- Khatib, O. (1987). "A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation" [PDF](http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf)
- LIBERO Documentation: https://lifelong-robotic-learning.github.io/libero-site/
- LeRobot Documentation: https://github.com/huggingface/lerobot
- UMI-ARX Repository: `/home/zfei/code/umi-arx/`

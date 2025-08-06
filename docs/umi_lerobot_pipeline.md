# UMI-LeRobot Pipeline: Dual Camera → ORB-SLAM → IK → Teleoperation

This document describes the complete UMI-LeRobot pipeline that combines the strengths of both frameworks:

- **LeRobot's mature camera infrastructure** (Intel RealSense support)
- **UMI's ORB-SLAM visual odometry** (UMI's core innovation)
- **LeRobot's proven IK pipeline** (placo-based for SO101)
- **UMI's end-effector teleoperation** (SpaceMouse control)

## Pipeline Overview

```
LeRobot RealSense Cameras → UMI ORB-SLAM Visual Odometry → LeRobot IK Solver → End-Effector Teleoperation
```

### Key Components

1. **Dual RealSense Cameras** (LeRobot)
   - Intel RealSense cameras for stereo vision
   - LeRobot's mature camera abstraction layer
   - Real-time frame capture and processing

2. **ORB-SLAM Visual Odometry** (UMI)
   - Feature detection and matching between cameras
   - Visual odometry for pose estimation
   - Real-time pose tracking and refinement

3. **Inverse Kinematics** (LeRobot)
   - Placo-based IK solver for SO101
   - Joint limit handling and collision avoidance
   - Real-time joint angle computation

4. **End-Effector Teleoperation** (UMI)
   - SpaceMouse-based control interface
   - Real-time pose control
   - Safety features and workspace limits

## Architecture

### Data Flow

```
Camera Frames (RGB + Depth) → Feature Detection → ORB-SLAM → Pose Estimation → IK Solver → Joint Commands → Robot Control
```

### Control Flow

```
SpaceMouse Input → Target Pose → IK Solver → Joint Angles → Robot Motors
```

## Implementation

### 1. Camera Setup

```python
from lerobot.cameras.realsense import RealSenseCamera, RealSenseConfig

# Configure dual RealSense cameras
camera_configs = {
    "camera_left": RealSenseConfig(
        device_id="left_camera_serial",
        width=640,
        height=480,
        fps=30
    ),
    "camera_right": RealSenseConfig(
        device_id="right_camera_serial", 
        width=640,
        height=480,
        fps=30
    )
}

# Initialize cameras
cameras = {}
for camera_name, config in camera_configs.items():
    camera = RealSenseCamera(config)
    camera.connect()
    cameras[camera_name] = camera
```

### 2. ORB-SLAM Integration

```python
from lerobot.utils.umi_slam import create_umi_slam_processor

# Initialize UMI SLAM processor
slam_processor = create_umi_slam_processor(
    umi_root_path="universal_manipulation_interface",
    calibration_dir="universal_manipulation_interface/example/calibration"
)

# Get camera frames
frames = {}
for camera_name, camera in cameras.items():
    frame = camera.async_read()
    if frame is not None:
        frames[camera_name] = frame

# Estimate pose using ORB-SLAM
estimated_pose = slam_processor.estimate_pose_from_frames(frames)
```

### 3. IK Solver Setup

```python
from lerobot.model.kinematics import RobotKinematics

# Initialize LeRobot's IK solver for SO101
ik_solver = RobotKinematics(
    urdf_path="path/to/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", 
                "wrist_flex", "wrist_roll", "gripper"]
)

# Solve IK for target pose
current_joint_deg = np.rad2deg(current_joint_angles)
target_joint_deg = ik_solver.inverse_kinematics(
    current_joint_pos=current_joint_deg,
    desired_ee_pose=target_pose,
    position_weight=1.0,
    orientation_weight=0.01
)
```

### 4. Teleoperation Integration

```python
from lerobot.teleoperators.umi import UmiTeleoperatorConfig, create_umi_teleoperator

# Configure UMI teleoperator with LeRobot IK
config = UmiTeleoperatorConfig(
    spacemouse=UmiTeleoperatorConfig.UmiSpaceMouseConfig(
        sensitivity_translation=0.8,
        sensitivity_rotation=0.6,
        deadzone=0.08,
        max_velocity=0.3,
        max_angular_velocity=0.8
    ),
    ik=UmiTeleoperatorConfig.UmiIkConfig(
        robot_type="so101",
        urdf_path="path/to/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", 
                    "wrist_flex", "wrist_roll", "gripper"],
        position_weight=1.0,
        orientation_weight=0.005,
        workspace_limits={
            "x": (-0.4, 0.4),
            "y": (-0.4, 0.4),
            "z": (0.0, 0.6)
        }
    ),
    control_frequency=10.0,
    num_robots=1
)

# Create teleoperator
teleoperator = create_umi_teleoperator(config)
```

## Complete Pipeline Example

See `examples/umi_lerobot_pipeline_example.py` for a complete implementation of the pipeline.

### Key Features

1. **Real-time Processing**: 10Hz control loop for smooth teleoperation
2. **Dual Camera Support**: Stereo vision for robust pose estimation
3. **Safety Features**: Workspace limits, collision avoidance, emergency stop
4. **Modular Design**: Each component can be used independently
5. **Error Handling**: Graceful degradation when components fail

## Configuration

### Camera Configuration

```yaml
cameras:
  camera_left:
    device_id: "left_camera_serial"
    width: 640
    height: 480
    fps: 30
    depth_enabled: true
  
  camera_right:
    device_id: "right_camera_serial"
    width: 640
    height: 480
    fps: 30
    depth_enabled: true
```

### Robot Configuration

```yaml
robot:
  type: "so101"
  urdf_path: "path/to/so101_new_calib.urdf"
  target_frame_name: "gripper_frame_link"
  joint_names:
    - "shoulder_pan"
    - "shoulder_lift"
    - "elbow_flex"
    - "wrist_flex"
    - "wrist_roll"
    - "gripper"
  workspace_limits:
    x: [-0.4, 0.4]
    y: [-0.4, 0.4]
    z: [0.0, 0.6]
```

### Teleoperation Configuration

```yaml
teleoperation:
  spacemouse:
    sensitivity_translation: 0.8
    sensitivity_rotation: 0.6
    deadzone: 0.08
    max_velocity: 0.3
    max_angular_velocity: 0.8
  
  ik:
    position_weight: 1.0
    orientation_weight: 0.005
    max_iterations: 100
    tolerance_position: 0.001
    tolerance_orientation: 0.01
  
  safety:
    collision_avoidance: true
    collision_margin: 0.08
    emergency_stop_enabled: true
    workspace_limits_enabled: true
```

## Benefits

### UMI Benefits

1. **Visual Odometry**: ORB-SLAM for robust pose estimation
2. **Multi-camera Support**: Stereo vision capabilities
3. **Real-time Processing**: Low-latency pose tracking
4. **Feature-rich**: Advanced SLAM algorithms

### LeRobot Benefits

1. **Mature Infrastructure**: Proven camera and IK implementations
2. **SO101 Support**: Native support for SO101 robot
3. **Safety Features**: Built-in safety and collision avoidance
4. **Modularity**: Reusable components

### Combined Benefits

1. **Best of Both Worlds**: UMI's innovation + LeRobot's maturity
2. **No Duplication**: Leverages existing infrastructure
3. **Extensible**: Easy to add new features
4. **Robust**: Multiple layers of error handling

## Usage Examples

### Basic Pipeline

```python
from examples.umi_lerobot_pipeline_example import UmiLeRobotPipeline

# Create pipeline
pipeline = UmiLeRobotPipeline(
    camera_configs=camera_configs,
    robot_urdf_path="path/to/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link"
)

# Start teleoperation
pipeline.start_teleoperation()
```

### Advanced Configuration

```python
# Custom camera configuration
camera_configs = {
    "camera_left": RealSenseConfig(
        device_id="left_camera_serial",
        width=1280,
        height=720,
        fps=60
    ),
    "camera_right": RealSenseConfig(
        device_id="right_camera_serial",
        width=1280,
        height=720,
        fps=60
    )
}

# Custom IK parameters
ik_config = UmiTeleoperatorConfig.UmiIkConfig(
    robot_type="so101",
    position_weight=1.0,
    orientation_weight=0.01,
    workspace_limits={
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.8)
    }
)
```

## Troubleshooting

### Common Issues

1. **Camera Connection Failed**
   - Check device IDs and USB connections
   - Verify RealSense SDK installation
   - Test cameras individually

2. **SLAM Initialization Failed**
   - Verify UMI installation
   - Check calibration files
   - Ensure sufficient lighting

3. **IK Solver Failed**
   - Verify URDF file path
   - Check joint names and limits
   - Ensure target pose is reachable

4. **Teleoperation Issues**
   - Check SpaceMouse connection
   - Verify workspace limits
   - Test emergency stop functionality

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run pipeline with debug output
pipeline = UmiLeRobotPipeline(...)
status = pipeline.get_pipeline_status()
print(f"Pipeline status: {status}")
```

## Future Enhancements

1. **Multi-robot Support**: Extend to bimanual setups
2. **Advanced SLAM**: Integration with more SLAM algorithms
3. **Learning-based IK**: Neural network IK solvers
4. **Haptic Feedback**: Force feedback integration
5. **Remote Operation**: Network-based teleoperation

## References

- [UMI Paper](https://arxiv.org/abs/2401.08595)
- [UMI GitHub Repository](https://github.com/real-stanford/universal_manipulation_interface)
- [LeRobot Documentation](https://lerobot.github.io/)
- [SO101 Assembly Guide](https://github.com/TheRobotStudio/SO-ARM100)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) 
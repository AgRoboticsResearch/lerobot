# UMI Integration with LeRobot

This document describes the focused integration of the Universal Manipulation Interface (UMI) with LeRobot, emphasizing UMI's unique capabilities while leveraging LeRobot's mature infrastructure.

## Overview

UMI (Universal Manipulation Interface) is a comprehensive robot learning framework that enables "in-the-wild" robot teaching without requiring robots in the wild. The integration with LeRobot focuses on UMI's distinctive features:

- **UMI SLAM Pipeline**: SLAM-based pose estimation (UMI's core innovation)
- **UMI Teleoperation**: SpaceMouse-based control with real-time IK calculations
- **Integration with LeRobot**: Use UMI-processed data with LeRobot's mature diffusion policies

## Key Design Principles

### Focus on UMI's Unique Value

Rather than duplicating LeRobot's mature capabilities, this integration focuses on UMI's distinctive features:

1. **SLAM-based Pose Estimation**: UMI's core innovation for "in-the-wild" data collection
2. **Real-time IK Calculations**: UMI's approach to teleoperation with inverse kinematics
3. **Multi-camera Fisheye Support**: UMI's sophisticated camera handling

### Leverage LeRobot's Infrastructure

The integration uses LeRobot's existing mature components:

- **Diffusion Policies**: Use LeRobot's proven diffusion policy implementation
- **Dataset Management**: Use LeRobot's dataset infrastructure
- **Robot Control**: Use LeRobot's robot abstraction layers

## Architecture

### Core UMI Components

1. **UMI SLAM Processor** (`src/lerobot/utils/umi_slam.py`)
   - Integration with UMI's SLAM pipeline
   - Multi-camera pose estimation
   - Fisheye lens handling
   - Data synchronization

2. **UMI Teleoperator** (`src/lerobot/teleoperators/umi/`)
   - SpaceMouse-based control
   - Real-time IK calculations
   - Collision avoidance
   - Multi-robot coordination

### Integration Points

1. **Data Flow**: UMI SLAM → LeRobot Dataset → LeRobot Diffusion Policy
2. **Control Flow**: UMI Teleoperator → IK Solver → Robot Control
3. **Policy Flow**: LeRobot Diffusion Policy → UMI-processed Data

## Installation

### Prerequisites

1. **UMI Dependencies**: Install UMI and its dependencies
   ```bash
   # Clone UMI repository
   git clone https://github.com/real-stanford/universal_manipulation_interface.git
   cd universal_manipulation_interface
   
   # Install UMI dependencies
   mamba env create -f conda_environment.yaml
   conda activate umi
   ```

2. **LeRobot Dependencies**: Ensure LeRobot is properly installed
   ```bash
   pip install -e .
   ```

### Integration Setup

The UMI integration is designed to work with optional dependencies. If UMI is not available, the integration will gracefully degrade with appropriate warnings.

## Usage

### UMI SLAM Processing

```python
from lerobot.utils.umi_slam import create_umi_slam_processor

# Create UMI SLAM processor
slam_processor = create_umi_slam_processor(
    umi_root_path="universal_manipulation_interface",
    calibration_dir="universal_manipulation_interface/example/calibration"
)

# Run UMI SLAM pipeline
session_dir = "path/to/your/umi_session"
success = slam_processor.run_slam_pipeline(session_dir)

if success:
    # Generate dataset for LeRobot
    output_path = f"{session_dir}/dataset.zarr.zip"
    slam_processor.generate_dataset(session_dir, output_path)
```

### UMI Teleoperation with IK

```python
from lerobot.teleoperators.umi import UmiTeleoperatorConfig, create_umi_teleoperator

# Create UMI teleoperator configuration
config = UmiTeleoperatorConfig(
    spacemouse=UmiTeleoperatorConfig.UmiSpaceMouseConfig(
        sensitivity_translation=1.0,
        sensitivity_rotation=1.0,
        deadzone=0.05,
        max_velocity=0.5,
        max_angular_velocity=1.0
    ),
    ik=UmiTeleoperatorConfig.UmiIkConfig(
        robot_type="ur5",
        ik_solver="ikfast",
        collision_avoidance=True,
        workspace_limits={
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.0, 1.0)
        }
    ),
    control_frequency=10.0,
    num_robots=1
)

# Create and start teleoperator
teleoperator = create_umi_teleoperator(config)
teleoperator.start()

# Get current state
state = teleoperator.get_current_state()
print(f"Current pose: {state['current_pose'][:3, 3]}")
print(f"Joint angles: {state['current_joint_angles']}")

teleoperator.stop()
```

### Using LeRobot's Diffusion Policy with UMI Data

```python
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy

# Create LeRobot diffusion policy (using existing infrastructure)
config = DiffusionConfig(
    input_features={
        "observation.images": (3, 224, 224),
        "observation.state": (7,),  # 3 pos + 3 rot + 1 gripper
    },
    output_features={
        "action": (7,),  # 3 pos + 3 rot + 1 gripper
    },
    n_obs_steps=2,
    n_action_steps=8,
    obs_as_global_cond=True,
    diffusion_step_embed_dim=128,
    down_dims=[256, 512, 1024],
    up_dims=[1024, 512, 256],
    noise_scheduler="ddim",
    num_inference_steps=16,
)

# Create policy using LeRobot's mature implementation
policy = DiffusionPolicy(config=config)

# Load UMI-processed dataset
dataset_path = "path/to/umi_processed_dataset.zarr.zip"
# Use LeRobot's dataset loader to load UMI-processed data
```

## UMI's Unique Features

### SLAM-based Pose Estimation

UMI's core innovation is its SLAM pipeline for pose estimation:

1. **Multi-camera Setup**: Simultaneous recording from multiple cameras
2. **Fisheye Lens Support**: Handles fisheye distortion automatically
3. **ORB-SLAM3 Integration**: Advanced SLAM for pose estimation
4. **Aruco Marker Detection**: Marker-based calibration and tracking
5. **Data Synchronization**: Temporal alignment of all data streams

### Real-time IK Teleoperation

UMI's teleoperation approach includes:

1. **SpaceMouse Control**: 3D input device for intuitive control
2. **Real-time IK**: Inverse kinematics for joint-level control
3. **Collision Avoidance**: Automatic collision detection and prevention
4. **Multi-robot Support**: Coordination between multiple robots
5. **Safety Features**: Emergency stop and workspace limits

### Integration Benefits

1. **No Duplication**: Uses LeRobot's mature diffusion policy implementation
2. **Focused Value**: Emphasizes UMI's unique SLAM and teleoperation features
3. **Clean Architecture**: Clear separation of concerns
4. **Extensible**: Easy to add more UMI features as needed

## Data Flow

### UMI Data Collection → LeRobot Training

```
UMI Session Data
    ↓
UMI SLAM Pipeline
    ↓
UMI-processed Dataset (Zarr format)
    ↓
LeRobot Dataset Loader
    ↓
LeRobot Diffusion Policy Training
    ↓
Trained Policy
```

### UMI Teleoperation → Robot Control

```
SpaceMouse Input
    ↓
UMI Teleoperator
    ↓
IK Solver
    ↓
Joint Commands
    ↓
Robot Control
```

## Examples

See `examples/umi_focused_integration.py` for a complete example demonstrating:

1. **UMI SLAM Processing**: Process raw UMI session data
2. **UMI Teleoperation**: Demonstrate SpaceMouse control with IK
3. **LeRobot Integration**: Use UMI data with LeRobot's diffusion policy

## Comparison with Existing Approaches

### vs. Direct UMI Usage

| Feature | Direct UMI | LeRobot Integration |
|---------|------------|-------------------|
| SLAM Processing | ✓ Full UMI pipeline | ✓ UMI SLAM integration |
| Diffusion Policy | UMI's implementation | ✓ LeRobot's mature implementation |
| Dataset Management | UMI's Zarr format | ✓ LeRobot's dataset infrastructure |
| Robot Control | UMI's robot interfaces | ✓ LeRobot's robot abstraction |
| Teleoperation | ✓ UMI's SpaceMouse + IK | ✓ UMI teleoperation + LeRobot robots |

### vs. Pure LeRobot

| Feature | Pure LeRobot | With UMI Integration |
|---------|--------------|-------------------|
| Pose Estimation | Manual/External | ✓ UMI SLAM pipeline |
| Multi-camera Support | Limited | ✓ UMI fisheye handling |
| Teleoperation | Basic | ✓ UMI SpaceMouse + IK |
| "In-the-wild" Data | Challenging | ✓ UMI's core innovation |

## Future Work

### Planned Enhancements

1. **Advanced SLAM Integration**: Direct integration with UMI's SLAM pipeline
2. **Real-time Visualization**: Live policy visualization and debugging
3. **Hardware Abstraction**: Unified hardware interface for different robot types
4. **Training Integration**: Direct training of policies with UMI data

### Contributing

To contribute to the UMI integration:

1. Focus on UMI's unique features (SLAM, teleoperation, IK)
2. Leverage LeRobot's existing infrastructure
3. Maintain clean separation of concerns
4. Add comprehensive tests and documentation

## Troubleshooting

### Common Issues

1. **UMI Dependencies Not Available**:
   - Install UMI and its dependencies
   - Ensure conda environment is activated
   - Check import paths

2. **SLAM Pipeline Errors**:
   - Verify UMI repository path
   - Check calibration files
   - Ensure sufficient disk space

3. **Teleoperation Issues**:
   - Verify SpaceMouse connection
   - Check IK solver configuration
   - Ensure robot is in remote control mode

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## References

- [UMI Paper](https://arxiv.org/abs/2401.08595)
- [UMI GitHub Repository](https://github.com/real-stanford/universal_manipulation_interface)
- [UMI Project Page](https://umi-gripper.github.io/)
- [LeRobot Documentation](https://lerobot.github.io/)
- [LeRobot Diffusion Policy](https://lerobot.github.io/docs/policies/diffusion/) 
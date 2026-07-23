# SO101 Deployment Examples

This directory contains scripts for deploying trained policies on the SO101 robot.

## Deploy ACT Policy

`deploy_act_so101.py` - Deploy a standard ACT policy (joint actions) on SO101 robot.

This is equivalent to `lerobot-record` with a policy but as a single standalone script without dataset recording overhead.

### Usage

```bash
python examples/so101/deploy_act_so101.py \
    --pretrained_path ./outputs/train/act_compare_strawberry_picking/checkpoints/last/pretrained_model \
    --robot_port /dev/ttyACM0 \
    --cameras "{ front: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: intelrealsense, serial_number_or_name: 031522070877, width: 640, height: 480, fps: 30} }"
```

With warm start (move to reset pose before starting):
```bash
python examples/so101/deploy_act_so101.py \
    --pretrained_path ./outputs/train/act_compare_strawberry_picking/checkpoints/last/pretrained_model \
    --robot_port /dev/ttyACM0 \
    --cameras "{ front: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: 031522070877, width: 640, height: 480, fps: 30} }" \
    --warm_start
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_path` | Required | Path to trained model checkpoint |
| `--robot_port` | `/dev/ttyACM0` | Serial port for SO101 |
| `--robot_id` | `so101` | Robot ID for calibration files |
| `--fps` | `30` | Control loop frequency (Hz) |
| `--num_steps` | `0` | Number of steps (0 = infinite) |
| `--task` | `""` | Task description for task-conditioned policies |
| `--cameras` | `None` | Camera config in YAML format |
| `--use_amp` | `False` | Use automatic mixed precision |
| `--device` | Auto | Device (cuda/cpu) |
| `--reset_pose` | `-5.54 -114.59 80.44 7.84 -5.19 35.13` | Reset pose (deg): shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper |
| `--safe_pose` | `-7.91 -106.51 87.91 70.74 -0.53 1.18` | Safe pose (deg) for error recovery |
| `--warm_start` | `False` | Move to reset pose before starting |

### Camera Configuration

Cameras are configured as a YAML string with the format:

```yaml
{
  front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30},
  wrist: {type: intelrealsense, serial_number_or_name: "031522070877", width: 640, height: 480, fps: 30}
}
```

Supported camera types:
- `opencv` - USB cameras (via /dev/video*)
- `intelrealsense` - Intel RealSense cameras

### Policy Requirements

The policy expects observations with:
- `observation.images.front` - Front camera image (3, 480, 640)
- `observation.images.wrist` - Wrist camera image (3, 480, 640)
- `observation.state` - Joint positions (6,)

And outputs:
- `action` - Joint position targets (6,)

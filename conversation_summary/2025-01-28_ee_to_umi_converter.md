# EE to UMI Dataset Converter

## Date
2025-01-28

## Overview
Created a converter script to transform LeRobot EE-only datasets to UMI's Zarr format, enabling training with UMI's Diffusion Policy implementation.

## Motivation
- LeRobot EE datasets contain 7D end-effector poses [x, y, z, wx, wy, wz, gripper]
- UMI expects Zarr format with specific data structure
- Need to bridge the two ecosystems for comparison and experimentation

## Implementation

### Created Files
- **`scripts/convert_lerobot_to_umi.py`** - CLI tool for conversion

### Key Features
1. **Direct data mapping**:
   - EE position → `robot0_eef_pos` [T,3]
   - EE rotation (axis-angle) → `robot0_eef_rot_axis_angle` [T,3]
   - Gripper → `robot0_gripper_width` [T,1]
   - Action (future EE pose) → `action` [T,7]

2. **Image handling**:
   - Resizes from source resolution (e.g., 480x640) to target (224x224)
   - Converts from channel-first (C,H,W) to channel-last (H,W,C)
   - Uses cv2 for fast resize

3. **Metadata**:
   - Episode boundaries → `/meta/episode_ends`
   - Demo start/end poses → `robot0_demo_start_pose`, `robot0_demo_end_pose`

4. **Compression**:
   - Low-dim data: Blosc lz4 compression
   - Images: Blosc (JpegXl if available)

## Usage

### Conversion
```bash
python scripts/convert_lerobot_to_umi.py \
    --input-dataset /path/to/ee_dataset \
    --output /path/to/output.zarr.zip \
    --image-size 224 224
```

### Training with UMI
```bash
cd /home/zfei/code/universal_manipulation_interface
python train.py --config-name=train_diffusion_transformer_umi_workspace \
    task.dataset_path=/path/to/output.zarr.zip
```

## Test Results

### Dataset: `red_strawberry_picking_260119_merged_ee`
- **Input**: 54 episodes, 29,517 frames, 480x640 images
- **Output**: 43MB `.zarr.zip` file
- **Conversion time**: ~1.5 minutes

### Verification
```python
# Loads correctly with UMI's ReplayBuffer
from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr

with zarr.ZipStore('/path/to/dataset.zarr.zip', mode='r') as store:
    replay_buffer = ReplayBuffer.copy_from_store(store, zarr.MemoryStore())

# Loads correctly with UmiDataset
from diffusion_policy.dataset.umi_dataset import UmiDataset
dataset = UmiDataset(shape_meta=shape_meta, dataset_path='/path/to/dataset.zarr.zip', ...)
```

### Training Test
- Training starts successfully
- Loss decreases: 1.24 → 0.099 (first epoch)
- Normalization computed correctly
- 27,033 training samples (after temporal sampling)

## UMI Data Format Reference

### Zarr Structure
```
root/
├── data/
│   ├── robot0_eef_pos              [T, 3]   float32
│   ├── robot0_eef_rot_axis_angle   [T, 3]   float32
│   ├── robot0_gripper_width        [T, 1]   float32
│   ├── action                      [T, 7]   float32
│   ├── robot0_demo_start_pose      [T, 6]   float32
│   ├── robot0_demo_end_pose        [T, 6]   float32
│   └── camera0_rgb                 [T, H, W, 3] uint8
└── meta/
    └── episode_ends                [N]      int64
```

### Action Format
- Actions are **future EE poses** (not deltas)
- During training, UMI converts to relative pose representation
- Stored as 7D: 3 position + 3 rotation (axis-angle) + 1 gripper
- During training: converted to 10D (3 pos + 6D rotation) for the model

## Dependencies Required for UMI Training
- `imagecodecs` - for image compression
- `timm` - for vision transformer backbone
- `zarr` - for zarr format handling

## Notes
- The action in UMI is the observation at timestep t+1 (future pose)
- UMI's `UmiDataset` computes relative poses on-the-fly during training
- Images are compressed to reduce file size (43MB for ~30k frames at 224x224)

# UMI-Style End-Effector Action Training with Processor Pipeline

This document describes the UMI-style relative EE training pipeline implemented in this repo. It uses a **processor pipeline** approach where all SE(3) math happens in preprocessor/postprocessor steps (not in a custom dataset class), and the dataset stores only raw absolute EE poses.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [SE(3) Math](#se3-math)
- [Dimension Flow](#dimension-flow)
- [Dataset Format](#dataset-format)
- [Processor Steps](#processor-steps)
- [Training](#training)
- [Inference & Deployment](#inference--deployment)
- [Files](#files)
- [Comparison with Pattern A (RelativeEEDataset)](#comparison-with-pattern-a-relativeeedataset)

## Overview

The pipeline follows the UMI (Universal Manipulation Interface) convention for relative end-effector actions:

1. **Dataset** stores absolute 7D EE poses `[x, y, z, wx, wy, wz, gripper]` in axis-angle format. No `observation.state` column needed — state is derived from the action column.

2. **Preprocessor** (saved in checkpoint) converts to relative rot6d:
   - Derives 2-step state from action column
   - Converts absolute 7D aa actions → relative 10D rot6d `[dx, dy, dz, rot6d(6), gripper]`
   - Converts 2-step state → 20D relative rot6d state
   - Normalizes

3. **Model** (standard ACT) sees 20D state and 10D action — no architecture changes.

4. **Postprocessor** (saved in checkpoint) converts back:
   - Unnormalizes
   - Converts 10D rot6d relative → 7D aa absolute
   - Output goes to IK → joints → robot

## Architecture

```
TRAINING:

  Dataset on disk (no observation.state):
    action: [x, y, z, wx, wy, wz, gripper]  (7D aa absolute, per frame)
    observation.images.camera: video

  DataLoader returns batch (via action_delta_indices = [-1, 0, 1, ..., chunk_size-1]):
    action: [B, chunk+1, 7]  (extra timestep t=-1 for state derivation)
    observation.images.camera: [B, 1, C, H, W]

  Preprocessor pipeline:
    1. RenameObservations     — identity (no rename)
    2. AddBatchDimension      — adds batch dim if needed
    3. DeviceStep             — move to cuda
    4. DeriveStateFromAction  — extract [action[t-1], action[t]] as state, strip extra timestep
                               → action: [B, chunk, 7],  observation.state: [B, 2, 7]
    5. RelativeRot6dActions   — 7D aa absolute → SE(3) relative → 10D rot6d
                               → action: [B, chunk, 10]
                               → caches current state for postprocessor
    6. RelativeRot6dState     — [prev_aa, curr_aa] → relative rot6d → flatten
                               → observation.state: [B, 20]
    7. Normalizer             — MIN_MAX normalization
                               → action: [B, chunk, 10],  observation.state: [B, 20]

  ACT Model:
    Input:  observation.state [B, 20] + observation.images.camera [B, 1, C, H, W]
    Output: action [B, chunk, 10]  (normalized rot6d relative)


INFERENCE (per step):

  Robot FK → 7D aa EE pose [x, y, z, wx, wy, wz, gripper]

  Preprocessor pipeline:
    1-3. Rename + Batch + Device
    4. DeriveStateFromAction — no-op (state comes from robot, not action)
    5. RelativeRot6dActions  — caches current 7D state (no action conversion needed)
    6. RelativeRot6dState     — buffers prev state, stacks [prev, curr], converts → 20D
    7. Normalizer

  ACT Model → 10D rot6d relative action (single step from action queue)

  Postprocessor pipeline:
    1. Unnormalizer
    2. AbsoluteRot6dActions  — 10D rot6d relative → SE(3) absolute → 7D aa (uses cached state)
    3. DeviceStep (CPU)

  7D aa absolute → IK → joint positions → robot
```

## SE(3) Math

### Relative action (7D aa → 10D rot6d)

Given current state `T_curr` and future action `T_future` (both as 7D axis-angle poses):

```
# Convert to 4x4 homogeneous matrices
T_curr = [R_curr | t_curr]    R from axis-angle via Rodrigues formula
         [0    | 1     ]

T_future = [R_future | t_future]
           [0        | 1       ]

# Compute relative transform
T_rel = T_curr^{-1} @ T_future

R_rel = R_curr^T @ R_future
t_rel = R_curr^T @ (t_future - t_curr)

# Convert R_rel to rot6d (first two rows of rotation matrix, flattened)
rot6d = [R_rel[0,0], R_rel[0,1], R_rel[0,2], R_rel[1,0], R_rel[1,1], R_rel[1,2]]

# Output: 10D relative action
action_rel = [t_rel(3), rot6d(6), gripper(1)]
```

### Absolute action (10D rot6d → 7D aa) — inverse at inference

Given cached reference state `T_ref` (7D aa) and predicted relative action (10D rot6d):

```
# Reconstruct R_rel from rot6d via Gram-Schmidt
a1 = rot6d[:3],  a2 = rot6d[3:]
b1 = a1 / |a1|
b2 = a2 - (b1 · a2) * b1;  b2 = b2 / |b2|
b3 = b1 × b2
R_rel = [b1; b2; b3]   (3x3 rotation matrix)

# Compose absolute transform
T_abs = T_ref @ T_rel
R_abs = R_ref @ R_rel
t_abs = t_ref + R_ref @ t_rel

# Convert R_abs back to axis-angle
aa_abs = matrix_to_axis_angle(R_abs)

# Output: 7D absolute action
action_abs = [t_abs(3), aa_abs(3), gripper(1)]
```

### Relative state (2×7D aa → 20D rot6d)

The 2-step state `[state[t-1], state[t]]` is converted to relative rot6d by expressing each timestep relative to the **current** (last) timestep:

```
current = state[t]   # reference frame

# For each timestep i in [t-1, t]:
T_rel[i] = T_current^{-1} @ T[i]
relative[i] = [t_rel(3), rot6d(6), gripper(1)]   # 10D

# Flatten: [relative[t-1](10), relative[t](10)] = 20D
```

This means `relative[t]` (the current step) is always close to identity, and `relative[t-1]` (the previous step) encodes how much the EE moved in one frame.

## Dimension Flow

| Stage | Action Dim | State Dim | Format |
|--------|-----------|-----------|--------|
| Dataset on disk | 7 | (none) | aa `[x,y,z,wx,wy,wz,gripper]` |
| DataLoader output | chunk+1 × 7 | — | aa with extra timestep for derivation |
| After DeriveState | chunk × 7 | 2 × 7 = 14 | aa, 2 timesteps |
| After RelativeRot6d | 10 | 2 × 10 = 20 | rot6d `[dx,dy,dz,r0..r5,gripper]` |
| After Normalizer | 10 | 20 | normalized |
| Model output | 10 | — | normalized rot6d relative |
| After Unnormalizer | 10 | — | rot6d relative |
| After AbsoluteRot6d | 7 | — | aa absolute |
| Robot command | 6 (joints) | — | via IK |

## Dataset Format

### Minimal dataset (no observation.state)

The dataset only needs:
- `action`: 7D EE pose `[x, y, z, wx, wy, wz, gripper]` (absolute, axis-angle)
- `observation.images.camera`: camera video

No `observation.state` column is needed because `DeriveStateFromActionStep` derives it from the action column at training time.

### Conversion script

Use `sroi_to_lerobot.py` to convert SROI rosbag data:

```bash
cd ~/code/sroi_rosbag_utilities

python lerobot/sroi_to_lerobot.py \
  --data_path /path/to/episodes \
  --repo_id your_dataset_name \
  --fps 30 \
  --root /path/to/output \
  --task "pick the strawberry"
```

The conversion script outputs only `action` and `observation.images.camera` — no `observation.state`.

### Existing datasets with observation.state

Datasets that already have `observation.state` still work. The `DeriveStateFromActionStep` overwrites it with the derived state anyway. The training script updates the shape from `[7]` to `[20]` in metadata.

## Processor Steps

Four processor steps implement the pipeline, all registered via `@ProcessorStepRegistry.register`:

### 1. DeriveStateFromActionStep

**Registry**: `derive_state_from_action_rot6d`

Extracts 2-step observation state from the action chunk. Only active during training (state comes from robot FK during inference).

- Input: `action` `[B, chunk+1, 7]` (extra timestep from `action_delta_indices = [-1, 0, ..., chunk-1]`)
- Output: `observation.state` `[B, 2, 7]`, `action` `[B, chunk, 7]` (stripped leading timestep)

### 2. RelativeRot6dActionsProcessorStep

**Registry**: `relative_rot6d_actions_processor`

Converts 7D aa absolute actions to 10D rot6d relative actions via SE(3).

- Training: reads state from `DeriveStateFromActionStep` output, computes `T_rel = T_curr⁻¹ @ T_future`
- Inference: caches current state from observation for the postprocessor (no action conversion — model outputs rot6d directly)
- Always caches state to a module-level shared dict (survives serialization)

### 3. RelativeRot6dStateProcessorStep

**Registry**: `relative_rot6d_state_processor`

Converts 2-step 7D aa state to 20D relative rot6d state.

- Training: state is `[B, 2, 7]` from `DeriveStateFromActionStep`, converts each step relative to current
- Inference: state is `[B, 7]` (single timestep from FK), buffers previous state, stacks `[prev, curr]`, converts to 20D

### 4. AbsoluteRot6dActionsProcessorStep

**Registry**: `absolute_rot6d_actions_processor`

Converts 10D rot6d relative actions back to 7D aa absolute actions. Only used in postprocessor.

- Reads cached state from shared dict (written by `RelativeRot6dActionsProcessorStep`)
- Computes `T_abs = T_ref @ T_rel`
- Returns 7D aa absolute action

### Shared State Cache

The preprocessor and postprocessor are separate pipelines, but the postprocessor needs the cached state from the preprocessor. Since processor steps are serialized to JSON independently, direct object references don't survive checkpoint save/load. Instead, a module-level dict `_state_cache` in `relative_action_processor.py` serves as the shared cache.

## Training

### Training script

`train_relative_ee_processor.py` — uses monkey-patching to override the standard training pipeline.

### Training command

```bash
PYTHONPATH=../../src python train_relative_ee_processor.py \
  --dataset.repo_id=lerobot_sroi_v2 \
  --dataset.root=/mnt/data1/data/lerobot/lerobot_sroi_v2 \
  --policy.type=act \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.device=cuda \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true \
  --policy.push_to_hub=false \
  --policy.repo_id=your/repo \
  --steps=500000 \
  --save_freq=50000 \
  --batch_size=8
```

### Key config flags (in `ACTConfig`)

| Flag | Default | Description |
|------|---------|-------------|
| `derive_state_from_action` | `false` | Derive `observation.state` from action column (UMI-style) |
| `use_relative_actions` | `false` | Enable SE(3) relative rot6d action conversion |
| `pose_dim` | `0` | `6` = xyz + axis-angle triggers SE(3) mode |
| `use_rot6d` | `false` | `true` = 10D rot6d output instead of 7D aa |
| `relative_exclude_joints` | `["gripper"]` | Joint names excluded from SE(3) conversion (kept absolute) |

When `derive_state_from_action=true`, `action_delta_indices` returns `[-1, 0, 1, ..., chunk_size-1]` (extra leading timestep for state derivation).

### Normalization

UMI-style uses MIN_MAX normalization for both actions and state (not MEAN_STD). The training script forces this:

```python
cfg.policy.normalization_mapping["ACTION"] = NormalizationMode.MIN_MAX
cfg.policy.normalization_mapping["STATE"] = NormalizationMode.MIN_MAX
```

### What the training script does

1. **Creates dataset** via `_make_dataset_wrapper`:
   - Loads standard `LeRobotDataset`
   - Calls `recompute_stats()` to compute 10D rot6d relative action stats from 7D aa data
   - Updates dataset metadata: action shape `[7]` → `[10]`
   - Adds `observation.state` with shape `[20]` to metadata (for model creation)
   - Applies ImageNet normalization stats for cameras

2. **Creates processors** via `_make_pre_post_processors_wrapper`:
   - When `use_relative_actions=true`: creates UMI-style preprocessor/postprocessor
   - Otherwise: creates standard ACT processors

3. **Calls `train(cfg)`** — standard LeRobot training loop, no modifications

### Stats computation

`recompute_stats()` in `relative_action_stats.py` computes normalization stats on the 10D rot6d relative actions (not the raw 7D aa). It iterates all valid action chunks, converts to relative rot6d via SE(3), and computes quantile stats. State stats (20D) are computed similarly from derived 2-step state.

## Inference & Deployment

### Deployment script

`deploy_relative_ee_processor_so101.py`

### Deployment command

```bash
python deploy_relative_ee_processor_so101.py \
  --pretrained_path outputs/<run>/checkpoints/last/pretrained_model \
  --robot_port /dev/ttyACM0 \
  --cameras '{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG}}' \
  --deploy_frame camera_link \
  --n_action_steps 30 \
  --warm_start
```

### Deployment flow (per control step)

1. **Read robot state**: Get joint positions from motors
2. **FK**: Compute 7D aa EE pose `[x, y, z, wx, wy, wz, gripper]` via forward kinematics
3. **Build batch**: `{observation.state: [1,7], observation.images.camera: [1,C,H,W]}`
4. **Preprocessor**:
   - `RelativeRot6dActionsProcessorStep`: caches current 7D state
   - `RelativeRot6dStateProcessorStep`: buffers prev state, converts to 20D relative rot6d
   - `Normalizer`: normalizes
5. **Model**: `policy.select_action(processed_batch)` → dequeues from action queue, or predicts new chunk if empty → returns `[1, 10]` (single step)
6. **Postprocessor**:
   - `Unnormalizer`: unnormalizes
   - `AbsoluteRot6dActionsProcessorStep`: 10D rot6d relative → 7D aa absolute (uses cached state)
7. **IK**: 7D aa absolute → joint positions
8. **Send**: `robot.send_action(joints)`

### Key deployment details

- **Frame consistency**: Use the same URDF and target frame during training (dataset conversion) and deployment. Default is `camera_link` via `so101_sroi.urdf`.
- **Action queue**: `select_action` manages a queue of `n_action_steps` actions. When the queue empties, it predicts a new chunk. This means the model is queried every `n_action_steps` frames, not every frame.
- **Processors are saved in checkpoint**: No need to pass config flags at deployment — the preprocessor and postprocessor are loaded from `policy_preprocessor.json` and `policy_postprocessor.json` in the checkpoint directory.

## Visualization

`visualize_predictions.py` — overlays predicted or GT trajectories onto camera images.

All projection works purely with **relative** SE(3) transforms — no base frame involved:
```
P_optical = T_opt_cam @ T_rel @ T_cam_grip
```

### Camera mode (handheld camera, no robot needed)

Runs live inference with a physical camera. Uses identity 7D aa state (no FK).

```bash
PYTHONPATH=../../src python visualize_predictions.py \
  --pretrained_path outputs/<run>/checkpoints/last/pretrained_model \
  --cameras "{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG}}" \
  --camera_info_path /path/to/camera_info.json
```

Optional flags:
- `--gripper`: show gripper state on trajectory
- `--initial_state 0 0 0 0 0 0 0.5`: custom initial 7D aa state
- `--update_state`: chain predictions (last predicted action becomes next state)

### Dataset mode

Loads a `LeRobotDataset` and projects trajectories onto observation images.

GT only:
```bash
PYTHONPATH=../../src python visualize_predictions.py \
  --dataset_root /path/to/dataset \
  --episode_indices 0 \
  --camera_name camera \
  --camera_info_path /path/to/camera_info.json \
  --mp4 \
  --output_dir outputs/debug/visualization
```

With model inference (+ GT overlay):
```bash
PYTHONPATH=../../src python visualize_predictions.py \
  --dataset_root /path/to/dataset \
  --episode_indices 0 \
  --camera_name camera \
  --camera_info_path /path/to/camera_info.json \
  --pretrained_path outputs/<run>/checkpoints/last/pretrained_model \
  --inference --gt --gripper \
  --mp4 \
  --output_dir outputs/debug/visualization
```

### Projection approach

For inference: the model outputs 10D rot6d relative actions. These are unnormalized and converted to 4×4 relative transforms, then projected directly — the postprocessor's absolute conversion is skipped for visualization.

For GT: the dataset stores 7D aa absolute actions. These are converted to relative using `T_rel = inv(action[0]) @ action[k]` (relative to the current frame), then projected.

In both cases the trajectory start point is fixed at `T_opt_cam @ T_cam_grip` (current gripper position in optical frame) and does not drift across frames.

## Files

### New files (processor pipeline)

| File | Description |
|------|-------------|
| `src/lerobot/processor/relative_action_processor.py` | Core SE(3) rot6d math + 4 processor steps |
| `src/lerobot/processor/relative_action_config.py` | `ACTRelativeEEConfig` — config subclass with UMI fields |
| `src/lerobot/processor/relative_action_processor_act.py` | ACT processor factory (preprocessor/postprocessor) |
| `src/lerobot/datasets/relative_action_stats.py` | Stats computation for 10D rot6d relative actions |
| `examples/umi_relative_ee/train_relative_ee_processor.py` | Training script (monkey-patches standard pipeline) |
| `examples/umi_relative_ee/deploy_relative_ee_processor_so101.py` | SO101 deployment script |
| `examples/umi_relative_ee/test_inference_processor.py` | Inference test script |
| `examples/umi_relative_ee/verify_pipeline_correctness.py` | Pipeline verification tests |
| `examples/umi_relative_ee/visualize_predictions.py` | Trajectory visualization (camera + dataset modes) |

### Modified files (minimal)

| File | Changes |
|------|---------|
| `src/lerobot/policies/act/configuration_act.py` | Added 5 config fields + `action_delta_indices` override |
| `src/lerobot/processor/__init__.py` | Added imports for new processor steps |

### Dataset conversion (separate repo)

| File | Description |
|------|-------------|
| `~/code/sroi_rosbag_utilities/lerobot/sroi_to_lerobot.py` | Converts SROI data to LeRobot format (no observation.state) |

## Comparison with Pattern A (RelativeEEDataset)

| Aspect | Pattern A (RelativeEEDataset) | UMI Processor Pipeline |
|--------|-------------------------------|----------------------|
| **Dataset class** | Custom `RelativeEEDataset` wrapper | Standard `LeRobotDataset` |
| **State source** | `observation.ee` column in dataset | Derived from action column |
| **SE(3) math location** | Dataset `__getitem__()` | Processor steps |
| **Model wrapper** | `TemporalACTWrapper` needed | Standard `ACTPolicy` |
| **Processors saved** | No | Yes (in checkpoint) |
| **observation.state in dataset** | Required | Not needed |
| **Stats** | Computed on raw data | Recomputed on 10D rot6d relative |
| **Inference state** | Identity frame or manual | Processors cache automatically |
| **Code coupling** | Tied to dataset class | Decoupled (any policy can use) |

### When to use which

- **Processor pipeline** (this doc): Preferred for new experiments. Modular, processors are saved in checkpoints, works with standard datasets. State derived from action column.
- **Pattern A (RelativeEEDataset)**: Legacy approach. State comes from `observation.ee` column. Requires dataset to have EE poses stored. Good for existing datasets with EE columns.

> Historical runbook preserved from `fei`. For the maintained v5 command and shared `use_umi_relative_ee` interface, start with `README.md`.

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
PYTHONPATH=../../../src python ../train_relative_ee_processor.py \
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
PYTHONPATH=../../../src python ../visualize_predictions.py \
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
PYTHONPATH=../../../src python ../visualize_predictions.py \
  --dataset_root /path/to/dataset \
  --episode_indices 0 \
  --camera_name camera \
  --camera_info_path /path/to/camera_info.json \
  --mp4 \
  --output_dir outputs/debug/visualization
```

With model inference (+ GT overlay):
```bash
PYTHONPATH=../../../src python ../visualize_predictions.py \
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

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

> 从 `fei` 保留的历史运行手册。关于维护中的 v5 命令和共享的 `use_umi_relative_ee` 接口，请先阅读 `README.md`。

# 使用处理器管线的 UMI 风格末端执行器动作训练

本文档描述本仓库中实现的 UMI 风格相对 EE 训练管线。它采用**处理器管线**方法，所有 SE(3) 数学运算都发生在 preprocessor/postprocessor 步骤中（而不是在自定义数据集类中），数据集只存储原始的绝对 EE 位姿。

## 目录

- [概述](#overview)
- [架构](#architecture)
- [SE(3) 数学](#se3-math)
- [维度流](#dimension-flow)
- [数据集格式](#dataset-format)
- [处理器步骤](#processor-steps)
- [训练](#training)
- [推理与部署](#inference--deployment)
- [文件](#files)
- [与模式 A（RelativeEEDataset）的比较](#comparison-with-pattern-a-relativeeedataset)

## 概述

该管线遵循 UMI（Universal Manipulation Interface，通用操作接口）的约定来处理相对末端执行器动作：

1. **数据集**以轴角格式存储绝对 7D EE 位姿 `[x, y, z, wx, wy, wz, gripper]`。不需要 `observation.state` 列——状态由动作列推导而来。

2. **预处理器**（保存在检查点中）转换为相对 rot6d：
   - 从动作列推导 2 步状态
   - 将绝对 7D aa 动作 → 相对 10D rot6d `[dx, dy, dz, rot6d(6), gripper]`
   - 将 2 步状态 → 20D 相对 rot6d 状态
   - 归一化

3. **模型**（标准 ACT）看到 20D 状态和 10D 动作——无需架构改动。

4. **后处理器**（保存在检查点中）转换回来：
   - 反归一化
   - 将 10D rot6d 相对 → 7D aa 绝对
   - 输出进入 IK → 关节 → 机器人

## 架构

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

## SE(3) 数学

### 相对动作（7D aa → 10D rot6d）

给定当前状态 `T_curr` 和未来动作 `T_future`（二者均为 7D 轴角位姿）：

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

### 绝对动作（10D rot6d → 7D aa）——推理时的逆运算

给定缓存的参考状态 `T_ref`（7D aa）和预测的相对动作（10D rot6d）：

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

### 相对状态（2×7D aa → 20D rot6d）

2 步状态 `[state[t-1], state[t]]` 通过将每个时间步表示为相对于**当前**（最后一个）时间步来转换为相对 rot6d：

```
current = state[t]   # reference frame

# For each timestep i in [t-1, t]:
T_rel[i] = T_current^{-1} @ T[i]
relative[i] = [t_rel(3), rot6d(6), gripper(1)]   # 10D

# Flatten: [relative[t-1](10), relative[t](10)] = 20D
```

这意味着 `relative[t]`（当前步）始终接近单位矩阵，而 `relative[t-1]`（上一步）编码了 EE 在一帧内移动了多少。

## 维度流

| 阶段 | 动作维度 | 状态维度 | 格式 |
|--------|-----------|-----------|--------|
| 磁盘上的数据集 | 7 | （无） | aa `[x,y,z,wx,wy,wz,gripper]` |
| DataLoader 输出 | chunk+1 × 7 | — | aa，带用于推导的额外时间步 |
| DeriveState 之后 | chunk × 7 | 2 × 7 = 14 | aa，2 个时间步 |
| RelativeRot6d 之后 | 10 | 2 × 10 = 20 | rot6d `[dx,dy,dz,r0..r5,gripper]` |
| Normalizer 之后 | 10 | 20 | 归一化 |
| 模型输出 | 10 | — | 归一化 rot6d 相对 |
| Unnormalizer 之后 | 10 | — | rot6d 相对 |
| AbsoluteRot6d 之后 | 7 | — | aa 绝对 |
| 机器人命令 | 6（关节） | — | 通过 IK |

## 数据集格式

### 最小数据集（无 observation.state）

数据集仅需要：
- `action`：7D EE 位姿 `[x, y, z, wx, wy, wz, gripper]`（绝对，轴角）
- `observation.images.camera`：相机视频

不需要 `observation.state` 列，因为 `DeriveStateFromActionStep` 在训练时从动作列推导它。

### 转换脚本

使用 `sroi_to_lerobot.py` 转换 SROI rosbag 数据：

```bash
cd ~/code/sroi_rosbag_utilities

python lerobot/sroi_to_lerobot.py \
  --data_path /path/to/episodes \
  --repo_id your_dataset_name \
  --fps 30 \
  --root /path/to/output \
  --task "pick the strawberry"
```

转换脚本仅输出 `action` 和 `observation.images.camera`——没有 `observation.state`。

### 已带有 observation.state 的现有数据集

已经含有 `observation.state` 的数据集仍然可用。`DeriveStateFromActionStep` 无论如何都会用推导出的状态覆盖它。训练脚本会在元数据中将形状从 `[7]` 更新为 `[20]`。

## 处理器步骤

四个处理器步骤实现该管线，全部通过 `@ProcessorStepRegistry.register` 注册：

### 1. DeriveStateFromActionStep

**注册名**：`derive_state_from_action_rot6d`

从动作块中提取 2 步观测状态。仅在训练期间激活（推理期间状态来自机器人 FK）。

- 输入：`action` `[B, chunk+1, 7]`（来自 `action_delta_indices = [-1, 0, ..., chunk-1]` 的额外时间步）
- 输出：`observation.state` `[B, 2, 7]`，`action` `[B, chunk, 7]`（去掉前导时间步）

### 2. RelativeRot6dActionsProcessorStep

**注册名**：`relative_rot6d_actions_processor`

通过 SE(3) 将 7D aa 绝对动作转换为 10D rot6d 相对动作。

- 训练：从 `DeriveStateFromActionStep` 输出读取状态，计算 `T_rel = T_curr⁻¹ @ T_future`
- 推理：为后处理器缓存来自观测的当前状态（不做动作转换——模型直接输出 rot6d）
- 始终将状态缓存到模块级共享 dict（可在序列化后存活）

### 3. RelativeRot6dStateProcessorStep

**注册名**：`relative_rot6d_state_processor`

将 2 步 7D aa 状态转换为 20D 相对 rot6d 状态。

- 训练：状态为来自 `DeriveStateFromActionStep` 的 `[B, 2, 7]`，将每一步相对于当前步进行转换
- 推理：状态为来自 FK 的单时间步 `[B, 7]`，缓冲上一步状态，堆叠 `[prev, curr]`，转换为 20D

### 4. AbsoluteRot6dActionsProcessorStep

**注册名**：`absolute_rot6d_actions_processor`

将 10D rot6d 相对动作转换回 7D aa 绝对动作。仅在后处理器中使用。

- 从共享 dict 读取缓存状态（由 `RelativeRot6dActionsProcessorStep` 写入）
- 计算 `T_abs = T_ref @ T_rel`
- 返回 7D aa 绝对动作

### 共享状态缓存

预处理器和后处理器是两个独立的管线，但后处理器需要来自预处理器的缓存状态。由于处理器步骤会独立序列化为 JSON，直接的对象引用无法在检查点保存/加载后存活。因此，`relative_action_processor.py` 中的模块级 dict `_state_cache` 用作共享缓存。

## 训练

### 训练脚本

`train_relative_ee_processor.py`——使用 monkey-patching 覆盖标准训练管线。

### 训练命令

```bash
PYTHONPATH=../../../src python ../train_relative_ee_processor.py \
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

### 关键配置标志（位于 `ACTConfig` 中）

| 标志 | 默认值 | 描述 |
|------|---------|-------------|
| `derive_state_from_action` | `false` | 从动作列推导 `observation.state`（UMI 风格） |
| `use_relative_actions` | `false` | 启用 SE(3) 相对 rot6d 动作转换 |
| `pose_dim` | `0` | `6` = xyz + 轴角，触发 SE(3) 模式 |
| `use_rot6d` | `false` | `true` = 输出 10D rot6d 而非 7D aa |
| `relative_exclude_joints` | `["gripper"]` | 从 SE(3) 转换中排除的关节名（保持绝对） |

当 `derive_state_from_action=true` 时，`action_delta_indices` 返回 `[-1, 0, 1, ..., chunk_size-1]`（用于状态推导的额外前导时间步）。

### 归一化

UMI 风格对动作和状态均使用 MIN_MAX 归一化（不是 MEAN_STD）。训练脚本强制这样做：

```python
cfg.policy.normalization_mapping["ACTION"] = NormalizationMode.MIN_MAX
cfg.policy.normalization_mapping["STATE"] = NormalizationMode.MIN_MAX
```

### 训练脚本做了什么

1. **创建数据集**，通过 `_make_dataset_wrapper`：
   - 加载标准 `LeRobotDataset`
   - 调用 `recompute_stats()` 从 7D aa 数据计算 10D rot6d 相对动作统计量
   - 更新数据集元数据：动作形状 `[7]` → `[10]`
   - 向元数据添加形状为 `[20]` 的 `observation.state`（用于创建模型）
   - 对相机应用 ImageNet 归一化统计量

2. **创建处理器**，通过 `_make_pre_post_processors_wrapper`：
   - 当 `use_relative_actions=true` 时：创建 UMI 风格预处理器/后处理器
   - 否则：创建标准 ACT 处理器

3. **调用 `train(cfg)`**——标准 LeRobot 训练循环，无修改

### 统计量计算

`relative_action_stats.py` 中的 `recompute_stats()` 在 10D rot6d 相对动作（而非原始 7D aa）上计算归一化统计量。它遍历所有有效动作块，通过 SE(3) 转换为相对 rot6d，并计算分位数统计量。状态统计量（20D）类似地从推导出的 2 步状态计算。

## 推理与部署

### 部署脚本

`deploy_relative_ee_processor_so101.py`

### 部署命令

```bash
python deploy_relative_ee_processor_so101.py \
  --pretrained_path outputs/<run>/checkpoints/last/pretrained_model \
  --robot_port /dev/ttyACM0 \
  --cameras '{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG}}' \
  --deploy_frame camera_link \
  --n_action_steps 30 \
  --warm_start
```

### 部署流程（每个控制步）

1. **读取机器人状态**：从电机获取关节位置
2. **FK**：通过正运动学计算 7D aa EE 位姿 `[x, y, z, wx, wy, wz, gripper]`
3. **构建 batch**：`{observation.state: [1,7], observation.images.camera: [1,C,H,W]}`
4. **预处理器**：
   - `RelativeRot6dActionsProcessorStep`：缓存当前 7D 状态
   - `RelativeRot6dStateProcessorStep`：缓冲上一状态，转换为 20D 相对 rot6d
   - `Normalizer`：归一化
5. **模型**：`policy.select_action(processed_batch)` → 从动作队列出队，若为空则预测新块 → 返回 `[1, 10]`（单步）
6. **后处理器**：
   - `Unnormalizer`：反归一化
   - `AbsoluteRot6dActionsProcessorStep`：10D rot6d 相对 → 7D aa 绝对（使用缓存状态）
7. **IK**：7D aa 绝对 → 关节位置
8. **发送**：`robot.send_action(joints)`

### 关键部署细节

- **坐标系一致性**：在训练（数据集转换）和部署期间使用相同的 URDF 和目标坐标系。默认通过 `so101_sroi.urdf` 使用 `camera_link`。
- **动作队列**：`select_action` 管理一个包含 `n_action_steps` 个动作的队列。当队列为空时，它预测新的块。这意味着模型每 `n_action_steps` 帧查询一次，而不是每一帧。
- **处理器保存在检查点中**：部署时无需传递配置标志——预处理器和后处理器从检查点目录中的 `policy_preprocessor.json` 和 `policy_postprocessor.json` 加载。

## 可视化

`visualize_predictions.py`——将预测或 GT 轨迹叠加到相机图像上。

所有投影完全使用**相对** SE(3) 变换——不涉及基坐标系：
```
P_optical = T_opt_cam @ T_rel @ T_cam_grip
```

### 相机模式（手持相机，无需机器人）

使用物理相机运行实时推理。使用单位 7D aa 状态（无 FK）。

```bash
PYTHONPATH=../../../src python ../visualize_predictions.py \
  --pretrained_path outputs/<run>/checkpoints/last/pretrained_model \
  --cameras "{wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 25, fourcc: MJPG}}" \
  --camera_info_path /path/to/camera_info.json
```

可选标志：
- `--gripper`：在轨迹上显示夹爪状态
- `--initial_state 0 0 0 0 0 0 0.5`：自定义初始 7D aa 状态
- `--update_state`：链式预测（最后一个预测动作成为下一个状态）

### 数据集模式

加载 `LeRobotDataset` 并将轨迹投影到观测图像上。

仅 GT：
```bash
PYTHONPATH=../../../src python ../visualize_predictions.py \
  --dataset_root /path/to/dataset \
  --episode_indices 0 \
  --camera_name camera \
  --camera_info_path /path/to/camera_info.json \
  --mp4 \
  --output_dir outputs/debug/visualization
```

带模型推理（+ GT 叠加）：
```bash
PYTHONPATH=../../../src python ../visualize_predictions.py \
  --dataset_root /path/to/dataset \
  --episode_indices 0 \
  --camera_name camera \
  --camera_info_path /path/to/camera_info.json \
  --pretrained_path outputs/<run>/checkpoints/last/pretrained_model \
  --inference --gt --gripper \
  --mp4 \
  --output_dir outputs/debug/visualization
```

### 投影方法

对于推理：模型输出 10D rot6d 相对动作。它们被反归一化并转换为 4×4 相对变换，然后直接投影——可视化时跳过后处理器的绝对转换。

对于 GT：数据集存储 7D aa 绝对动作。使用 `T_rel = inv(action[0]) @ action[k]`（相对于当前帧）转换为相对，然后投影。

在两种情况下，轨迹起点固定为 `T_opt_cam @ T_cam_grip`（光坐标系中的当前夹爪位置），并且不会跨帧漂移。

## 文件

### 新文件（处理器管线）

| 文件 | 描述 |
|------|-------------|
| `src/lerobot/processor/relative_action_processor.py` | 核心 SE(3) rot6d 数学 + 4 个处理器步骤 |
| `src/lerobot/processor/relative_action_config.py` | `ACTRelativeEEConfig`——带 UMI 字段的配置子类 |
| `src/lerobot/processor/relative_action_processor_act.py` | ACT 处理器工厂（预处理器/后处理器） |
| `src/lerobot/datasets/relative_action_stats.py` | 10D rot6d 相对动作的统计量计算 |
| `examples/umi_relative_ee/train_relative_ee_processor.py` | 训练脚本（monkey-patch 标准管线） |
| `examples/umi_relative_ee/deploy_relative_ee_processor_so101.py` | SO101 部署脚本 |
| `examples/umi_relative_ee/test_inference_processor.py` | 推理测试脚本 |
| `examples/umi_relative_ee/verify_pipeline_correctness.py` | 管线正确性验证测试 |
| `examples/umi_relative_ee/visualize_predictions.py` | 轨迹可视化（相机 + 数据集模式） |

### 修改的文件（最少）

| 文件 | 变更 |
|------|---------|
| `src/lerobot/policies/act/configuration_act.py` | 添加 5 个配置字段 + `action_delta_indices` 覆盖 |
| `src/lerobot/processor/__init__.py` | 为新处理器步骤添加导入 |

### 数据集转换（独立仓库）

| 文件 | 描述 |
|------|-------------|
| `~/code/sroi_rosbag_utilities/lerobot/sroi_to_lerobot.py` | 将 SROI 数据转换为 LeRobot 格式（无 observation.state） |

## 与模式 A（RelativeEEDataset）的比较

| 方面 | 模式 A（RelativeEEDataset） | UMI 处理器管线 |
|--------|-------------------------------|----------------------|
| **数据集类** | 自定义 `RelativeEEDataset` 包装器 | 标准 `LeRobotDataset` |
| **状态来源** | 数据集中的 `observation.ee` 列 | 从动作列推导 |
| **SE(3) 数学位置** | 数据集 `__getitem__()` | 处理器步骤 |
| **模型包装器** | 需要 `TemporalACTWrapper` | 标准 `ACTPolicy` |
| **处理器保存** | 否 | 是（在检查点中） |
| **数据集中的 observation.state** | 必需 | 不需要 |
| **统计量** | 在原始数据上计算 | 在 10D rot6d 相对上重新计算 |
| **推理状态** | 单位坐标系或手动 | 处理器自动缓存 |
| **代码耦合** | 与数据集类绑定 | 解耦（任何策略均可使用） |

### 何时使用哪种

- **处理器管线**（本文档）：新实验的首选。模块化，处理器保存在检查点中，适用于标准数据集。状态从动作列推导。
- **模式 A（RelativeEEDataset）**：旧方法。状态来自 `observation.ee` 列。要求数据集存储 EE 位姿。适合已有 EE 列的数据集。

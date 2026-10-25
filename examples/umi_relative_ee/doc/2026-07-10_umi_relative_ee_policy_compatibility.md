---
conversation_id: unavailable
created: 2026-07-10
tags:
  - lerobot
  - umi
  - relative-ee
  - act
  - smolvla
  - diffusion-policy
  - robotics
---

# UMI Relative-EE Policy Compatibility Review

## Overview

This document records the review of the UMI-style relative end-effector
pipeline under examples/umi_relative_ee. It covers:

- The current ACT training and Piper deployment workflow.
- The exact model and processor shapes verified in an existing checkpoint.
- Correctness fixes applied to ACT training and deployment.
- Implemented SmolVLA training and Piper deployment support.
- Implemented Diffusion Policy training and Piper deployment support.
- Commands, architecture, and validation coverage for all three policies.

The 10D UMI relative-EE representation is now policy-generic. ACT, SmolVLA,
and Diffusion Policy share the same SE(3) processors, saved-checkpoint format,
and one-base-pose-per-chunk deployment rule.

## Implementation Status

Implemented and validated on 2026-07-10:

- ACT derived statistics now match the training timestamps.
- UMI statistics stay in memory and no longer overwrite raw 7D dataset stats.
- Piper preprocessing updates the two-frame state every control tick.
- SmolVLA adds relative-EE config fields while retaining language tokenization.
- Diffusion uses n_obs_steps=1, a 20D processor-managed state, and direct chunk prediction.
- Piper loads ACT, SmolVLA, or Diffusion dynamically from checkpoint config.
- Legacy ACT processor JSON without cache identifiers remains loadable.

## Current ACT Workflow

### Training command

The current training command is:

~~~bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --policy.type=act \
  --output_dir=/home/zfei/code/lerobots/lerobot/outputs/train/ee_vs_joints/umi_processor_ee_action_chunk_30_sroi_v2_1000_one_sb \
  --job_name=act_umi_processor_ee_action_chunk30 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=zfff/act_policy \
  --policy.push_to_hub=false \
  --save_freq=100000 \
  --steps=2500000 \
  --batch_size=8 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true
~~~

### Deployment command

The current Piper deployment command is:

~~~bash
python examples/umi_relative_ee/deploy_umi_relative_ee_piper.py \
  --pretrained_path=outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_merge/checkpoints/500000/pretrained_model \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}"
~~~

### Verified checkpoint

The following checkpoint was inspected:

~~~text
outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_merge/
  checkpoints/500000/pretrained_model
~~~

Its saved configuration confirms:

| Setting | Value |
|---|---:|
| Policy | ACT |
| Camera input | 3 x 480 x 640 |
| Model state input | 20D |
| Model action output | 10D |
| Chunk size | 30 |
| Executed actions per chunk | 30 |
| State normalization | MIN_MAX |
| Action normalization | MIN_MAX |

The saved preprocessor is:

~~~text
RenameObservations
AddBatchDimension
Device
DeriveStateFromAction
RelativeRot6dActions
RelativeRot6dState
Normalizer
~~~

The saved postprocessor is:

~~~text
Unnormalizer
AbsoluteRot6dActions
Device(CPU)
~~~

### Representation

Raw dataset actions are expected to be 7D:

~~~text
[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
~~~

The model predicts 10D relative actions:

~~~text
[dx, dy, dz, rot6d_0, ..., rot6d_5, gripper]
~~~

For one predicted chunk, every action must use the same chunk-start pose:

~~~text
T_relative[k] = inverse(T_chunk_base) @ T_target[k]
T_target[k]   = T_chunk_base @ T_relative[k]
~~~

The actions must not be chained as:

~~~text
T_target[k] = T_target[k - 1] @ T_relative[k]
~~~

The current Piper deployment predicts and postprocesses an entire chunk at
once. That part correctly preserves the single-base-per-chunk rule.

## ACT Correctness Fixes

### 1. Deployment state history is sampled once per chunk

Relevant code:

~~~text
examples/umi_relative_ee/deploy_umi_relative_ee_piper.py:839
src/lerobot/processor/relative_action_processor.py:518
~~~

Training derives the 20D state from consecutive poses:

~~~text
[action(t - 1), action(t)]
~~~

At deployment, the preprocessor currently runs only when the external action
queue is empty. With a chunk size of 30 at 30 Hz, the previous state stored by
RelativeRot6dStateProcessorStep is approximately one second old instead of one
frame old.

Current structure:

~~~python
if len(action_queue) == 0:
    processed = preprocessor(batch)
    pred_norm = policy.predict_action_chunk(processed)
    pred = postprocessor(pred_norm)
~~~

Recommended structure:

~~~python
with torch.no_grad():
    processed = preprocessor(batch)

if len(action_queue) == 0:
    with torch.no_grad():
        pred_norm = policy.predict_action_chunk(processed)
        pred = postprocessor(pred_norm)
~~~

The preprocessor should run every control tick so its state history remains
consecutive. The model and postprocessor should still run only when a new
chunk is needed.

This remains compatible with the UMI action rule because the full new chunk is
postprocessed immediately using the state cached on that chunk-boundary tick.
The external queue then stores already-converted absolute 7D targets.

The same fix should be made in:

~~~text
examples/umi_relative_ee/deploy_relative_ee_processor_so101.py
~~~

### 2. Derived relative-action statistics are shifted by one frame

Relevant code:

~~~text
src/lerobot/datasets/relative_action_stats.py:324
~~~

The training preprocessor receives:

~~~text
[action(t - 1), action(t), ..., action(t + chunk_size - 1)]
~~~

It derives state from the first two entries, strips the extra previous entry,
and therefore trains on:

~~~text
base    = action(t)
targets = action(t), ..., action(t + chunk_size - 1)
~~~

The current statistics code instead computes:

~~~text
base    = action(t)
targets = action(t + 1), ..., action(t + chunk_size)
~~~

This excludes the identity/current target at the start of each training chunk
and includes one target beyond the training horizon.

The derived statistics logic should align with the actual training sample:

~~~python
states = all_actions[batch + 1]

for start in batch:
    chunk = all_actions[start + 1 : start + 1 + chunk_size]
    state_expanded = np.broadcast_to(states_for_this_chunk, (chunk_size, 7)).copy()
    chunk_relative = _pose_se3_relative_aa_to_rot6d_np(state_expanded, chunk)
~~~

The exact vectorized implementation can differ, but it must preserve that
alignment.

Existing checkpoints must continue using their saved normalizer and
unnormalizer files. Do not recompute statistics and replace only the
processors of an existing model. Correct the statistics before a new training
run and retrain the model with those corrected statistics.

### 3. Statistics recomputation mutates the source dataset

recompute_stats writes transformed 10D action statistics back into the dataset
root even though the stored action column is still raw 7D.

This can contaminate later standard ACT, joint-action, or absolute-action
experiments using the same dataset root.

Recommended options:

1. Keep transformed statistics in memory and save them only with the policy.
2. Save UMI-transformed statistics under the training output directory.
3. Use a copied dataset root for UMI experiments until the behavior is fixed.

### 4. Several flags are ACT-only or informational

The fields derive_state_from_action, use_relative_actions, pose_dim, and
use_rot6d currently exist on ACTConfig but not on SmolVLAConfig or
DiffusionConfig.

The core processor currently performs the 7D axis-angle to 10D rot6d
conversion whenever use_relative_actions is enabled. pose_dim and use_rot6d
are largely configuration/logging signals in this path and do not make the
processor support arbitrary pose layouts.

The supported raw action layout should therefore be validated explicitly as:

~~~text
7D = xyz + axis-angle + gripper
~~~

## Policy-Type Dispatch

The training wrapper now customizes only dataset construction, transformed
metadata, and statistics. The standard LeRobot policy factory dispatches to
ACT, SmolVLA, or Diffusion processor composition based on the config type.

For pretrained SmolVLA fine-tuning, model weights load from
lerobot/smolvla_base while new relative processors are built from the current
dataset. Resume training instead reloads the relative processors saved in the
checkpoint. Piper deployment reads the checkpoint config and dynamically loads
the matching policy class.

## Policy Compatibility Summary

| Policy | Can model 10D relative EE? | Works with current example unchanged? | Main integration work |
|---|---|---|---|
| ACT | Yes | Yes | Corrected state cadence and statistics |
| SmolVLA | Yes | Yes | Requires a task string at deployment |
| Diffusion Policy | Yes | Yes | Requires n_obs_steps=1 in relative mode |

## Implemented Architecture

The SE(3) conversion steps should remain policy-independent:

~~~text
DeriveStateFromActionStep
RelativeRot6dActionsProcessorStep
RelativeRot6dStateProcessorStep
AbsoluteRot6dActionsProcessorStep
~~~

Policy-specific code should only compose those steps with the policy's normal
preprocessing:

~~~text
ACT relative factory
SmolVLA relative factory
Diffusion relative factory
~~~

The training wrapper should dispatch by the configuration type or policy type,
not through one global ACT branch.

The dataset wrapper should use a policy-specific prediction horizon:

~~~python
if policy.type in {"act", "smolvla"}:
    prediction_horizon = policy.chunk_size
elif policy.type == "diffusion":
    prediction_horizon = policy.horizon
else:
    raise NotImplementedError(...)
~~~

Image normalization must also remain policy-specific:

- ACT may use the current ImageNet statistics.
- SmolVLA uses visual IDENTITY normalization and performs its own image
  mapping internally.
- Diffusion should follow its configured visual normalization rather than
  inheriting ACT behavior automatically.

## SmolVLA Implementation Plan

SmolVLA is the easiest next integration because:

- n_obs_steps is already 1.
- It predicts an action chunk.
- max_state_dim defaults to 32, so a 20D state fits.
- max_action_dim defaults to 32, so a 10D action fits.
- Its flow-matching action model is not tied to the original raw action
  dimension.

### 1. Extend SmolVLAConfig

File:

~~~text
src/lerobot/policies/smolvla/configuration_smolvla.py
~~~

Add:

~~~python
derive_state_from_action: bool = False
use_relative_actions: bool = False
pose_dim: int = 0
use_rot6d: bool = False
relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
~~~

When derive_state_from_action is enabled:

~~~python
@property
def action_delta_indices(self) -> list:
    if self.derive_state_from_action:
        return [-1] + list(range(self.chunk_size))
    return list(range(self.chunk_size))
~~~

Add validation that:

~~~text
max_state_dim >= 20
max_action_dim >= 10
~~~

### 2. Add a SmolVLA relative-EE processor factory

Implementation file:

~~~text
src/lerobot/processor/relative_action_processor_smolvla.py
~~~

Preprocessor order:

~~~text
RenameObservations
AddBatchDimension
SmolVLANewLineProcessor
TokenizerProcessorStep
DeviceProcessorStep
DeriveStateFromActionStep
RelativeRot6dActionsProcessorStep
RelativeRot6dStateProcessorStep
NormalizerProcessorStep
~~~

Postprocessor order:

~~~text
UnnormalizerProcessorStep
AbsoluteRot6dActionsProcessorStep
DeviceProcessorStep(CPU)
~~~

The language newline and tokenizer steps must not be dropped when adding the
relative-EE processors.

### 3. Fine-tune the pretrained SmolVLA base model

Starting command:

~~~bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --policy.path=lerobot/smolvla_base \
  --output_dir=outputs/train/umi_relative_ee_smolvla \
  --job_name=smolvla_umi_relative_ee \
  --policy.device=cuda \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true \
  --batch_size=8 \
  --steps=20000 \
  --wandb.enable=true
~~~

The repository documentation recommends starting SmolVLA fine-tuning around
20,000 steps. Evaluate checkpoints and increase toward 50,000 or 100,000 only
if needed. Do not automatically reuse the 2.5-million-step ACT schedule.

### 4. Supply language during deployment

Piper deployment needs a task argument:

~~~bash
--task="pick the strawberry"
~~~

The inference batch must include:

~~~python
batch["task"] = args.task
~~~

Use the same task wording used in the training dataset when possible.

### 5. SmolVLA normalization

SmolVLA defaults to MEAN_STD for state/action and IDENTITY for images, while
the current UMI training wrapper forces MIN_MAX for state/action.

The first implementation should make this an explicit choice rather than
silently overriding it:

- Keeping MIN_MAX gives direct consistency with the current ACT UMI setup.
- Keeping SmolVLA's pretrained normalization convention may reduce
  fine-tuning distribution shift.

Whichever is selected, the saved processor statistics must remain paired with
the trained checkpoint.

## Diffusion Policy Implementation Plan

Diffusion Policy can model the same 10D relative actions, but it has stricter
temporal input and horizon assumptions.

### Recommended first configuration

Use:

~~~text
n_obs_steps = 1
processor-managed state history = 20D
horizon = 32
n_action_steps = 30
~~~

Do not begin with Diffusion's default n_obs_steps=2. The existing UMI state
processor already represents two consecutive EE poses in one flattened 20D
state. Using n_obs_steps=1 avoids creating a second, redundant temporal
history in the Diffusion observation queue.

Use horizon 32 instead of 30 because the default Diffusion U-Net has three
downsampling stages and requires:

~~~text
horizon % 8 == 0
~~~

The model can train on a horizon of 32 while executing only the first 30
actions.

### 1. Extend DiffusionConfig

File:

~~~text
src/lerobot/policies/diffusion/configuration_diffusion.py
~~~

Add the same UMI configuration fields used for ACT and SmolVLA.

For the recommended n_obs_steps=1 mode:

~~~python
@property
def action_delta_indices(self) -> list:
    if self.derive_state_from_action:
        return [-1] + list(range(self.horizon))
    return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))
~~~

Add validation requiring n_obs_steps=1 for this first relative-EE
implementation.

### 2. Add a Diffusion relative-EE processor factory

Implementation file:

~~~text
src/lerobot/processor/relative_action_processor_diffusion.py
~~~

The common derivation and SE(3) conversion steps can be reused, but training
must present the state to Diffusion as:

~~~text
[batch, n_obs_steps, state_dim] = [B, 1, 20]
~~~

During live inference the preprocessor should produce [B, 20], after which the
Diffusion observation queue adds the one-step temporal dimension. A small
Diffusion-specific processor or model adapter is therefore needed to add the
training temporal dimension without double-adding it during inference.

This behavior must have explicit unit tests for both training and inference
batches.

### 3. Horizon handling

The dataset wrapper uses chunk_size for ACT/SmolVLA and horizon for Diffusion:

~~~python
prediction_horizon = policy.horizon if policy.type == "diffusion" else policy.chunk_size
~~~

For the recommended Diffusion configuration, delta timestamps and statistics
use horizon 32 while deployment executes the first 30 actions.

### 4. Diffusion deployment queue handling

DiffusionPolicy.predict_action_chunk now inserts the current processed
observation into its history queue before generation. select_action shares an
internal queue-based generator, avoiding double insertion.

Piper runs the relative-EE preprocessor every control tick. At queue refill it
generates and immediately postprocesses the complete relative chunk, then
stores absolute 7D targets in the external execution queue.

### 5. Starting Diffusion command

Starting command:

~~~bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --policy.type=diffusion \
  --output_dir=outputs/train/umi_relative_ee_diffusion \
  --job_name=diffusion_umi_relative_ee \
  --policy.device=cuda \
  --policy.n_obs_steps=1 \
  --policy.horizon=32 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true \
  --batch_size=8 \
  --wandb.enable=true
~~~

## Generic Piper Deployment

Current hard-coded load:

~~~text
examples/umi_relative_ee/deploy_umi_relative_ee_piper.py:608
~~~

Replace the ACTPolicy-specific loader with checkpoint-driven loading:

~~~python
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

policy_config = PreTrainedConfig.from_pretrained(model_path)
policy_class = get_policy_class(policy_config.type)
policy = policy_class.from_pretrained(model_path, local_files_only=True)
policy.eval()
policy.config.device = str(device)
~~~

Load saved processors using:

~~~python
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=model_path,
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
    },
)
~~~

Then dispatch only the model-specific inference queue behavior:

- ACT: direct predict_action_chunk.
- SmolVLA: direct predict_action_chunk plus task tokens.
- Diffusion: populate observation history before predict_action_chunk.

Robot FK, relative-to-absolute postprocessing, IK, safety checks, gripper
mapping, and motor commands remain policy-independent.

## Implemented Files

| File | Implemented change |
|---|---|
| examples/umi_relative_ee/train_relative_ee_processor.py | Generic horizon and policy processor dispatch |
| examples/umi_relative_ee/deploy_umi_relative_ee_piper.py | Generic policy loader, per-frame preprocessing, task argument, Diffusion queue adapter |
| examples/umi_relative_ee/deploy_relative_ee_processor_so101.py | Per-frame preprocessing and generic loading |
| src/lerobot/datasets/relative_action_stats.py | Correct action/stat temporal alignment and avoid source-dataset mutation |
| src/lerobot/processor/relative_action_processor.py | Shared validation and any generic state-shape helpers |
| src/lerobot/processor/relative_action_processor_smolvla.py | New SmolVLA relative-EE factory |
| src/lerobot/processor/relative_action_processor_diffusion.py | New Diffusion relative-EE factory |
| src/lerobot/policies/smolvla/configuration_smolvla.py | Add relative-EE config fields and delta indices |
| src/lerobot/policies/diffusion/configuration_diffusion.py | Add relative-EE fields and n_obs_steps=1 delta indices |
| tests/processor/ | Add common, SmolVLA, and Diffusion processor tests |

## Validation Checklist

### Common processor tests

- 7D absolute to 10D relative to 7D absolute SE(3) round trip.
- Every action in a chunk uses the same base pose.
- The first target in a training chunk is identity/current pose when expected.
- Gripper remains absolute and normalized consistently.
- Preprocessor reset clears previous-state history.
- Separate processor instances do not overwrite each other's action base.

New processor pairs serialize a unique cache identifier. Legacy checkpoints
without that field use the original shared key for backward compatibility.

### Dataset/statistics tests

- A deterministic synthetic episode verifies exact t-1, t, and future indices.
- Stats inputs exactly match tensors emitted by the training preprocessor.
- Episode boundaries and padding do not mix poses from different episodes.
- Running UMI training does not replace raw 7D dataset statistics globally.

### ACT deployment tests

- Preprocessor runs every control frame.
- Model inference runs every n_action_steps frames.
- State history represents consecutive control frames.
- The entire action chunk is converted using one chunk-start base.

### SmolVLA tests

- Training batch contains language tokens and attention masks.
- State is accepted as 20D and action as 10D before internal 32D padding.
- predict_action_chunk returns [B, chunk_size, 10].
- Postprocessor returns [B, chunk_size, 7].
- Deployment task string matches the training dataset.

### Diffusion tests

- Training state shape is [B, 1, 20].
- Inference input becomes [B, 1, 20] exactly once.
- Training action shape is [B, 32, 10].
- Generated execution chunk is [B, 30, 10].
- Observation queue is refreshed at the intended control cadence.
- Postprocessing occurs before the base cache advances.

### Hardware safety tests

- Start with camera-only dry run.
- Visualize predicted relative trajectories.
- Run one paused/single chunk without motor writes.
- Verify all absolute targets are within workspace bounds.
- Use conservative max EE step and joint limits.
- Test low-speed execution before continuous deployment.

## Recommended Order of Work

1. Fix ACT deployment state-history cadence.
2. Fix relative statistics alignment.
3. Add deterministic regression tests for both fixes.
4. Refactor training and Piper loading to dispatch by policy type.
5. Add SmolVLA support and validate on a short fine-tuning run.
6. Add Diffusion support using n_obs_steps=1 and horizon=32.
7. Only after synchronous chunk deployment is correct, consider SmolVLA RTC
   or asynchronous inference.

## Decisions and Key Insights

- Keep the 10D action representation unchanged across policies.
- Keep the one-base-pose-per-chunk rule unchanged across policies.
- Postprocess a full newly generated chunk immediately at its chunk boundary.
- Update state history every control tick, even while executing a queued chunk.
- SmolVLA is the lower-risk first extension.
- Diffusion is feasible but needs explicit temporal-shape and observation-queue
  handling.
- Existing checkpoints must always use their own saved processor statistics.
- New runs should be trained after correcting the statistics alignment.

## Files Created

| File | Description |
|---|---|
| examples/umi_relative_ee/doc/2026-07-10_umi_relative_ee_policy_compatibility.md | This compatibility review and implementation plan |

## Optional Offline Validation

The generic training loop accepts a second nested dataset configuration. For
example, add these arguments to an ACT, SmolVLA, or Diffusion training command:

```bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --policy.type=act \
  --output_dir=/home/zfei/code/lerobots/lerobot/outputs/train/ee_vs_joints/act_relative_ee_with_validation \
  --job_name=act_relative_ee_with_validation \
  --policy.device=cuda \
  --wandb.enable=true \
  --steps=2500000 \
  --batch_size=8 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true \
  --val_freq=1000
```

Validation evaluates the full selected validation dataset and logs `val/loss`
to W&B. Use `--validation_dataset.episodes=[...]` to restrict the validation
split. Validation uses normalization statistics computed from the training
dataset. If `--validation_dataset.repo_id` is omitted, training behavior is
unchanged. Set `--val_freq<=0` to disable configured validation.

### Exact ACT Fresh-Restart Command

The stopped run remains at
`outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_1000_one_sb_corrected`.
The following command starts again from step 0 in a new output directory:

```bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_20260521_20260702 \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_20260521_20260702 \
  --val_freq=1000 \
  --policy.type=act \
  --output_dir=/home/zfei/code/lerobots/lerobot/outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_1000_one_sb_corrected_with_validation \
  --job_name=act_umi_relative_ee_chunk30_corrected_with_validation \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=zfff/act_policy \
  --policy.push_to_hub=false \
  --save_freq=100000 \
  --steps=2500000 \
  --batch_size=8 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true
```

The validation dataset contains 205 episodes and 27,182 frames. At batch
size 8, each complete validation pass has about 3,398 batches, so validation
every 1,000 training steps is compute-intensive. Increase `--val_freq` or
select a subset with `--validation_dataset.episodes=[...]` if needed.

## Tags

#python #pytorch #lerobot #robotics #umi #relative-ee #act #smolvla
#diffusion-policy #review #implementation-plan

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# UMI 相对 EE 策略兼容性审查

## 概述

本文档记录了针对 examples/umi_relative_ee 下 UMI 风格相对末端执行器
（relative end-effector）流程的审查。内容涵盖：

- 当前的 ACT 训练与 Piper 部署工作流。
- 在现有检查点中验证的确切模型与处理器形状。
- 应用于 ACT 训练与部署的正确性修复。
- 已实现的 SmolVLA 训练与 Piper 部署支持。
- 已实现的 Diffusion Policy 训练与 Piper 部署支持。
- 三种策略的命令、架构与验证覆盖范围。

10D UMI 相对 EE 表示现在已与具体策略无关。ACT、SmolVLA
和 Diffusion Policy 共享相同的 SE(3) 处理器、保存检查点的格式，
以及“每个 chunk 使用单个基准位姿”的部署规则。

## 实现状态

已于 2026-07-10 完成实现并验证：

- ACT 派生统计数据现在与训练时间戳一致。
- UMI 统计数据保留在内存中，不再覆盖原始 7D 数据集统计。
- Piper 预处理在每个控制 tick 更新两帧状态。
- SmolVLA 新增相对 EE 配置字段，同时保留语言分词（language tokenization）。
- Diffusion 使用 n_obs_steps=1、由处理器管理的 20D 状态，以及直接 chunk 预测。
- Piper 从检查点配置动态加载 ACT、SmolVLA 或 Diffusion。
- 不含 cache 标识符的旧版 ACT processor JSON 仍可加载。

## 当前 ACT 工作流

### 训练命令

当前训练命令为：

~~~bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --policy.type=act \
  --output_dir=/home/zfei/code/lerobots/lerobot/outputs/train/ee_vs_joints/umi_processor_ee_action_chunk_30_sroi_v2_1000_one_sb \
  --job_name=act_umi_processor_ee_action_chunk30 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=zfff/act_policy \
  --policy.push_to_hub=false \
  --save_freq=100000 \
  --steps=2500000 \
  --batch_size=8 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true
~~~

### 部署命令

当前 Piper 部署命令为：

~~~bash
python examples/umi_relative_ee/deploy_umi_relative_ee_piper.py \
  --pretrained_path=outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_merge/checkpoints/500000/pretrained_model \
  --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}"
~~~

### 已验证的检查点

已检查以下检查点：

~~~text
outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_merge/
  checkpoints/500000/pretrained_model
~~~

其保存的配置确认：

| 设置 | 值 |
|---|---:|
| 策略 | ACT |
| 相机输入 | 3 x 480 x 640 |
| 模型状态输入 | 20D |
| 模型动作输出 | 10D |
| Chunk 大小 | 30 |
| 每个 chunk 执行的动作数 | 30 |
| 状态归一化 | MIN_MAX |
| 动作归一化 | MIN_MAX |

保存的预处理器为：

~~~text
RenameObservations
AddBatchDimension
Device
DeriveStateFromAction
RelativeRot6dActions
RelativeRot6dState
Normalizer
~~~

保存的后处理器为：

~~~text
Unnormalizer
AbsoluteRot6dActions
Device(CPU)
~~~

### 表示

原始数据集动作应为 7D：

~~~text
[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
~~~

模型预测 10D 相对动作：

~~~text
[dx, dy, dz, rot6d_0, ..., rot6d_5, gripper]
~~~

对于一个预测出的 chunk，每个动作都必须使用相同的 chunk 起始位姿：

~~~text
T_relative[k] = inverse(T_chunk_base) @ T_target[k]
T_target[k]   = T_chunk_base @ T_relative[k]
~~~

动作不得按如下方式链式组合：

~~~text
T_target[k] = T_target[k - 1] @ T_relative[k]
~~~

当前 Piper 部署会一次性预测并后处理整个 chunk。该部分正确遵循了
“每个 chunk 单一基准位姿”的规则。

## ACT 正确性修复

### 1. 部署状态历史每个 chunk 仅采样一次

相关代码：

~~~text
examples/umi_relative_ee/deploy_umi_relative_ee_piper.py:839
src/lerobot/processor/relative_action_processor.py:518
~~~

训练从连续位姿派生 20D 状态：

~~~text
[action(t - 1), action(t)]
~~~

在部署时，预处理器目前仅在外部动作队列为空时运行。在 30 Hz 下
chunk 大小为 30 的情况下，RelativeRot6dStateProcessorStep 所保存的先前状态
大约旧了一秒，而不是旧一帧。

当前结构：

~~~python
if len(action_queue) == 0:
    processed = preprocessor(batch)
    pred_norm = policy.predict_action_chunk(processed)
    pred = postprocessor(pred_norm)
~~~

推荐结构：

~~~python
with torch.no_grad():
    processed = preprocessor(batch)

if len(action_queue) == 0:
    with torch.no_grad():
        pred_norm = policy.predict_action_chunk(processed)
        pred = postprocessor(pred_norm)
~~~

预处理器应在每个控制 tick 运行，使其状态历史保持连续。模型和后处理器
仍应仅在新 chunk 需要时运行。

这与 UMI 动作规则保持兼容，因为完整的新 chunk 会立即使用该 chunk 边界
tick 上缓存的状态进行后处理。随后外部队列存储的是已转换完成的
绝对 7D 目标。

同样的修复也应应用于：

~~~text
examples/umi_relative_ee/deploy_relative_ee_processor_so101.py
~~~

### 2. 派生的相对动作统计数据偏移了一帧

相关代码：

~~~text
src/lerobot/datasets/relative_action_stats.py:324
~~~

训练预处理器接收：

~~~text
[action(t - 1), action(t), ..., action(t + chunk_size - 1)]
~~~

它从前两个条目派生状态，去掉额外的前一项，
因此实际训练使用的是：

~~~text
base    = action(t)
targets = action(t), ..., action(t + chunk_size - 1)
~~~

而当前的统计代码计算的是：

~~~text
base    = action(t)
targets = action(t + 1), ..., action(t + chunk_size)
~~~

这排除了每个训练 chunk 开头的恒等/当前目标，
并包含了超出训练视野的一个目标。

派生统计逻辑应与实际训练样本对齐：

~~~python
states = all_actions[batch + 1]

for start in batch:
    chunk = all_actions[start + 1 : start + 1 + chunk_size]
    state_expanded = np.broadcast_to(states_for_this_chunk, (chunk_size, 7)).copy()
    chunk_relative = _pose_se3_relative_aa_to_rot6d_np(state_expanded, chunk)
~~~

确切的向量化实现可以不同，但必须保持该对齐关系。

现有检查点必须继续使用其保存的 normalizer 和 unnormalizer 文件。
不要重新计算统计数据并只替换现有模型的处理器。应在新的训练运行之前
修正统计数据，并使用修正后的统计数据重新训练模型。

### 3. 重新计算统计数据会修改源数据集

recompute_stats 会将转换后的 10D 动作统计写回数据集根目录，
即使存储的动作列仍然是原始 7D。

这可能污染后续使用同一数据集根目录的标准 ACT、关节动作或
绝对动作实验。

推荐选项：

1. 将转换后的统计保留在内存中，仅随策略一起保存。
2. 将 UMI 转换后的统计保存在训练输出目录下。
3. 在行为修复之前，为 UMI 实验使用数据集根目录的副本。

### 4. 若干标志仅适用于 ACT 或仅为信息性

字段 derive_state_from_action、use_relative_actions、pose_dim 和
use_rot6d 目前存在于 ACTConfig 上，但不存在于 SmolVLAConfig 或
DiffusionConfig 上。

只要启用 use_relative_actions，核心处理器当前就会执行
7D 轴角到 10D rot6d 的转换。在此路径中，pose_dim 和 use_rot6d
在很大程度上只是配置/日志信号，并不会使处理器支持任意位姿布局。

因此，应显式验证所支持的原始动作布局为：

~~~text
7D = xyz + axis-angle + gripper
~~~

## 策略类型分派

训练封装器现在只定制数据集构建、转换后的元数据和统计数据。标准
LeRobot 策略工厂会根据配置类型分派到 ACT、SmolVLA 或 Diffusion 的
处理器组合。

对于预训练 SmolVLA 微调，模型权重从 lerobot/smolvla_base 加载，
而新的相对处理器则基于当前数据集构建。恢复训练则会重新加载
检查点中保存的相对处理器。Piper 部署读取检查点配置并动态加载
匹配的策略类。

## 策略兼容性总结

| 策略 | 能否建模 10D 相对 EE？ | 能否在现有示例保持不变的情况下工作？ | 主要集成工作 |
|---|---|---|---|
| ACT | 是 | 是 | 修正状态节奏与统计数据 |
| SmolVLA | 是 | 是 | 部署时需要 task 字符串 |
| Diffusion Policy | 是 | 是 | 相对模式下需要 n_obs_steps=1 |

## 已实现的架构

SE(3) 转换步骤应保持与策略无关：

~~~text
DeriveStateFromActionStep
RelativeRot6dActionsProcessorStep
RelativeRot6dStateProcessorStep
AbsoluteRot6dActionsProcessorStep
~~~

策略特定的代码应仅将这些步骤与策略正常的预处理组合起来：

~~~text
ACT relative factory
SmolVLA relative factory
Diffusion relative factory
~~~

训练封装器应依据配置类型或策略类型进行分派，
而不是通过一个全局的 ACT 分支。

数据集封装器应使用与策略相关的预测视野：

~~~python
if policy.type in {"act", "smolvla"}:
    prediction_horizon = policy.chunk_size
elif policy.type == "diffusion":
    prediction_horizon = policy.horizon
else:
    raise NotImplementedError(...)
~~~

图像归一化也必须保持策略特定：

- ACT 可以使用当前的 ImageNet 统计。
- SmolVLA 使用视觉 IDENTITY 归一化，并在内部执行自己的图像映射。
- Diffusion 应遵循其配置的视觉归一化，而不是自动继承 ACT 行为。

## SmolVLA 实现计划

SmolVLA 是最容易的下一步集成，因为：

- n_obs_steps 已经是 1。
- 它预测一个动作 chunk。
- max_state_dim 默认是 32，因此 20D 状态可以容纳。
- max_action_dim 默认是 32，因此 10D 动作可以容纳。
- 其 flow-matching 动作模型不绑定于原始动作维度。

### 1. 扩展 SmolVLAConfig

文件：

~~~text
src/lerobot/policies/smolvla/configuration_smolvla.py
~~~

添加：

~~~python
derive_state_from_action: bool = False
use_relative_actions: bool = False
pose_dim: int = 0
use_rot6d: bool = False
relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
~~~

当启用 derive_state_from_action 时：

~~~python
@property
def action_delta_indices(self) -> list:
    if self.derive_state_from_action:
        return [-1] + list(range(self.chunk_size))
    return list(range(self.chunk_size))
~~~

添加验证：

~~~text
max_state_dim >= 20
max_action_dim >= 10
~~~

### 2. 添加 SmolVLA 相对 EE 处理器工厂

实现文件：

~~~text
src/lerobot/processor/relative_action_processor_smolvla.py
~~~

预处理器顺序：

~~~text
RenameObservations
AddBatchDimension
SmolVLANewLineProcessor
TokenizerProcessorStep
DeviceProcessorStep
DeriveStateFromActionStep
RelativeRot6dActionsProcessorStep
RelativeRot6dStateProcessorStep
NormalizerProcessorStep
~~~

后处理器顺序：

~~~text
UnnormalizerProcessorStep
AbsoluteRot6dActionsProcessorStep
DeviceProcessorStep(CPU)
~~~

在添加相对 EE 处理器时，不得丢弃语言换行和分词器步骤。

### 3. 微调预训练的 SmolVLA 基础模型

起始命令：

~~~bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --policy.path=lerobot/smolvla_base \
  --output_dir=outputs/train/umi_relative_ee_smolvla \
  --job_name=smolvla_umi_relative_ee \
  --policy.device=cuda \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true \
  --batch_size=8 \
  --steps=20000 \
  --wandb.enable=true
~~~

仓库文档建议从约 20,000 步开始 SmolVLA 微调。评估检查点，
仅在需要时增加到 50,000 或 100,000。不要自动复用 250 万步的
ACT 训练计划。

### 4. 部署时提供语言输入

Piper 部署需要一个 task 参数：

~~~bash
--task="pick the strawberry"
~~~

推理批次必须包含：

~~~python
batch["task"] = args.task
~~~

尽可能使用与训练数据集中相同的任务措辞。

### 5. SmolVLA 归一化

SmolVLA 对状态/动作默认为 MEAN_STD，对图像默认为 IDENTITY，
而当前 UMI 训练封装器强制对状态/动作使用 MIN_MAX。

第一个实现应将此作为显式选择，而不是静默覆盖：

- 保留 MIN_MAX 可与当前 ACT UMI 设置直接保持一致。
- 保留 SmolVLA 的预训练归一化约定可能减少微调分布偏移。

无论选择哪一种，保存的处理器统计都必须与训练后的检查点配对。

## Diffusion Policy 实现计划

Diffusion Policy 可以建模相同的 10D 相对动作，但它对时间输入和
视野假设更为严格。

### 推荐的首个配置

使用：

~~~text
n_obs_steps = 1
processor-managed state history = 20D
horizon = 32
n_action_steps = 30
~~~

不要从 Diffusion 默认的 n_obs_steps=2 开始。现有的 UMI 状态处理器
已经在一个展平的 20D 状态中表示两个连续的 EE 位姿。使用
n_obs_steps=1 可以避免在 Diffusion 观测队列中创建第二个冗余的
时间历史。

使用 horizon 32 而不是 30，因为默认的 Diffusion U-Net 有三个
下采样阶段，并要求：

~~~text
horizon % 8 == 0
~~~

模型可以在 horizon 32 上训练，但仅执行前 30 个动作。

### 1. 扩展 DiffusionConfig

文件：

~~~text
src/lerobot/policies/diffusion/configuration_diffusion.py
~~~

添加用于 ACT 和 SmolVLA 的相同 UMI 配置字段。

对于推荐的 n_obs_steps=1 模式：

~~~python
@property
def action_delta_indices(self) -> list:
    if self.derive_state_from_action:
        return [-1] + list(range(self.horizon))
    return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))
~~~

添加验证，要求此首个相对 EE 实现使用 n_obs_steps=1。

### 2. 添加 Diffusion 相对 EE 处理器工厂

实现文件：

~~~text
src/lerobot/processor/relative_action_processor_diffusion.py
~~~

可以复用通用的派生和 SE(3) 转换步骤，但训练时必须将状态
呈现给 Diffusion 为：

~~~text
[batch, n_obs_steps, state_dim] = [B, 1, 20]
~~~

在实时推理期间，预处理器应产生 [B, 20]，随后 Diffusion
观测队列会添加一步的时间维度。因此需要一个小的 Diffusion
专用处理器或模型适配器，以便在训练时添加时间维度，
而在推理时不会重复添加。

此行为必须对训练和推理批次都有明确的单元测试。

### 3. 视野处理

数据集封装器对 ACT/SmolVLA 使用 chunk_size，对 Diffusion 使用 horizon：

~~~python
prediction_horizon = policy.horizon if policy.type == "diffusion" else policy.chunk_size
~~~

对于推荐的 Diffusion 配置，delta 时间戳和统计使用 horizon 32，
而部署执行前 30 个动作。

### 4. Diffusion 部署队列处理

DiffusionPolicy.predict_action_chunk 现在会在生成之前将当前处理过的
观测插入其历史队列。select_action 共享一个基于队列的内部生成器，
避免重复插入。

Piper 在每个控制 tick 运行相对 EE 预处理器。在队列补充时，
它生成并立即后处理完整的相对 chunk，然后将绝对 7D 目标
存储到外部执行队列中。

### 5. Diffusion 起始命令

起始命令：

~~~bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --policy.type=diffusion \
  --output_dir=outputs/train/umi_relative_ee_diffusion \
  --job_name=diffusion_umi_relative_ee \
  --policy.device=cuda \
  --policy.n_obs_steps=1 \
  --policy.horizon=32 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true \
  --batch_size=8 \
  --wandb.enable=true
~~~

## 通用 Piper 部署

当前硬编码加载：

~~~text
examples/umi_relative_ee/deploy_umi_relative_ee_piper.py:608
~~~

将 ACTPolicy 专用加载器替换为检查点驱动的加载：

~~~python
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

policy_config = PreTrainedConfig.from_pretrained(model_path)
policy_class = get_policy_class(policy_config.type)
policy = policy_class.from_pretrained(model_path, local_files_only=True)
policy.eval()
policy.config.device = str(device)
~~~

使用以下方式加载保存的处理器：

~~~python
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=model_path,
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
    },
)
~~~

然后仅分派模型特定的推理队列行为：

- ACT：直接 predict_action_chunk。
- SmolVLA：直接 predict_action_chunk 外加 task token。
- Diffusion：在 predict_action_chunk 之前填充观测历史。

机器人 FK、相对到绝对的后处理、IK、安全检查、夹爪映射和
电机命令保持与策略无关。

## 已实现文件

| 文件 | 已实现的更改 |
|---|---|
| examples/umi_relative_ee/train_relative_ee_processor.py | 通用视野与策略处理器分派 |
| examples/umi_relative_ee/deploy_umi_relative_ee_piper.py | 通用策略加载器、逐帧预处理、task 参数、Diffusion 队列适配器 |
| examples/umi_relative_ee/deploy_relative_ee_processor_so101.py | 逐帧预处理与通用加载 |
| src/lerobot/datasets/relative_action_stats.py | 修正动作/统计的时间对齐并避免源数据集被修改 |
| src/lerobot/processor/relative_action_processor.py | 共享验证及任何通用状态形状辅助函数 |
| src/lerobot/processor/relative_action_processor_smolvla.py | 新增 SmolVLA 相对 EE 工厂 |
| src/lerobot/processor/relative_action_processor_diffusion.py | 新增 Diffusion 相对 EE 工厂 |
| src/lerobot/policies/smolvla/configuration_smolvla.py | 添加相对 EE 配置字段和 delta 索引 |
| src/lerobot/policies/diffusion/configuration_diffusion.py | 添加相对 EE 字段和 n_obs_steps=1 的 delta 索引 |
| tests/processor/ | 添加通用、SmolVLA 和 Diffusion 处理器测试 |

## 验证清单

### 通用处理器测试

- 7D 绝对到 10D 相对再到 7D 绝对的 SE(3) 往返。
- chunk 中的每个动作使用相同的基准位姿。
- 在预期情况下，训练 chunk 中的第一个目标是恒等/当前位姿。
- 夹爪保持绝对并以一致方式归一化。
- 预处理器 reset 会清除先前状态历史。
- 独立的处理器实例不会覆盖彼此的动作基准。

新的处理器对会序列化唯一的 cache 标识符。没有该字段的旧版检查点
使用原始共享键以保持向后兼容。

### 数据集/统计测试

- 确定性的合成 episode 会验证精确的 t-1、t 和未来索引。
- 统计输入与训练预处理器发出的张量完全匹配。
- episode 边界和填充不会混入来自不同 episode 的位姿。
- 运行 UMI 训练不会全局替换原始 7D 数据集统计。

### ACT 部署测试

- 预处理器每个控制帧运行。
- 模型推理每 n_action_steps 帧运行。
- 状态历史表示连续的控制帧。
- 整个动作 chunk 使用一个 chunk 起始基准进行转换。

### SmolVLA 测试

- 训练批次包含语言 token 和 attention mask。
- 在内部 32D 填充之前，状态以 20D 接受、动作以 10D 接受。
- predict_action_chunk 返回 [B, chunk_size, 10]。
- 后处理器返回 [B, chunk_size, 7]。
- 部署 task 字符串与训练数据集匹配。

### Diffusion 测试

- 训练状态形状为 [B, 1, 20]。
- 推理输入恰好一次变为 [B, 1, 20]。
- 训练动作形状为 [B, 32, 10]。
- 生成的执行 chunk 为 [B, 30, 10]。
- 观测队列按预期的控制节奏刷新。
- 后处理在基准 cache 前进之前发生。

### 硬件安全测试

- 从仅相机的 dry run 开始。
- 可视化预测的相对轨迹。
- 在不写入电机的情况下运行一个暂停/单 chunk。
- 验证所有绝对目标都在工作空间边界内。
- 使用保守的最大 EE 步长和关节限位。
- 在连续部署之前测试低速执行。

## 推荐工作顺序

1. 修复 ACT 部署状态历史节奏。
2. 修复相对统计对齐。
3. 为这两个修复添加确定性回归测试。
4. 重构训练和 Piper 加载，以按策略类型分派。
5. 添加 SmolVLA 支持，并在短微调运行中验证。
6. 使用 n_obs_steps=1 和 horizon=32 添加 Diffusion 支持。
7. 只有在同步 chunk 部署正确之后，才考虑 SmolVLA RTC
   或异步推理。

## 决策与关键洞察

- 在所有策略之间保持 10D 动作表示不变。
- 在所有策略之间保持“每个 chunk 单一基准位姿”规则不变。
- 在其 chunk 边界立即后处理完整的新生成 chunk。
- 每个控制 tick 更新状态历史，即使正在执行排队的 chunk。
- SmolVLA 是风险较低的第一个扩展。
- Diffusion 可行，但需要明确处理时间形状和观测队列。
- 现有检查点必须始终使用其自己保存的处理器统计。
- 新运行应在修正统计对齐之后进行训练。

## 创建的文件

| 文件 | 描述 |
|---|---|
| examples/umi_relative_ee/doc/2026-07-10_umi_relative_ee_policy_compatibility.md | 本兼容性审查与实现计划 |

## 可选的离线验证

通用训练循环接受第二个嵌套数据集配置。
例如，将以下参数添加到 ACT、SmolVLA 或 Diffusion 训练命令中：

```bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --policy.type=act \
  --output_dir=/home/zfei/code/lerobots/lerobot/outputs/train/ee_vs_joints/act_relative_ee_with_validation \
  --job_name=act_relative_ee_with_validation \
  --policy.device=cuda \
  --wandb.enable=true \
  --steps=2500000 \
  --batch_size=8 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true \
  --val_freq=1000
```

验证会评估完整的选定验证数据集，并将 `val/loss`
记录到 W&B。使用 `--validation_dataset.episodes=[...]` 限制验证
划分。验证使用从训练数据集计算出的归一化统计。如果省略
`--validation_dataset.repo_id`，训练行为不变。将 `--val_freq<=0`
设置为禁用已配置的验证。

### 确切的 ACT 全新重启命令

已停止的运行仍位于
`outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_1000_one_sb_corrected`。
以下命令在新的输出目录中从第 0 步重新开始：

```bash
python examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1000onesb \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_20260521_20260702 \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_20260521_20260702 \
  --val_freq=1000 \
  --policy.type=act \
  --output_dir=/home/zfei/code/lerobots/lerobot/outputs/train/ee_vs_joints/umi_processor_ee_action_chunk30_sroi_v2_1000_one_sb_corrected_with_validation \
  --job_name=act_umi_relative_ee_chunk30_corrected_with_validation \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=zfff/act_policy \
  --policy.push_to_hub=false \
  --save_freq=100000 \
  --steps=2500000 \
  --batch_size=8 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.derive_state_from_action=true \
  --policy.use_relative_actions=true \
  --policy.pose_dim=6 \
  --policy.use_rot6d=true
```

验证数据集包含 205 个 episode 和 27,182 帧。在 batch
size 为 8 的情况下，每次完整验证 pass 约有 3,398 个 batch，因此每
1,000 个训练步进行一次验证是计算密集的。如有需要，可增大 `--val_freq`，
或使用 `--validation_dataset.episodes=[...]` 选择子集。

## 标签

#python #pytorch #lerobot #robotics #umi #relative-ee #act #smolvla
#diffusion-policy #review #implementation-plan

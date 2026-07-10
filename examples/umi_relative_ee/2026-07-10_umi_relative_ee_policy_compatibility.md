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
| examples/umi_relative_ee/2026-07-10_umi_relative_ee_policy_compatibility.md | This compatibility review and implementation plan |

## Tags

#python #pytorch #lerobot #robotics #umi #relative-ee #act #smolvla
#diffusion-policy #review #implementation-plan

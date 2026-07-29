---
created: 2026-07-27
workspace: /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
branch: fei-v5.0-umi-unified
status: training guide
tags:
  - pi0.5
  - lora
  - peft
  - umi
  - relative-ee
---

# π0.5 LoRA Fine-Tuning in the Unified UMI Workspace

This guide describes the π0.5 LoRA training path implemented in:

```text
/home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
```

It is specific to the unified branch and the strawberry-picking datasets. It
explains the raw data contract, temporal query, SE(3) target, normalization,
flow-matching loss, PEFT LoRA setup, validation, checkpoint loading, and
visualization.

The main recommendation is:

```text
Raw dataset       absolute 7D axis-angle EE poses
Processor target  same-base relative 10D row-rot6d
State             derived 20D two-pose relative state
Chunk             30 actions
Normalization     q01/q99 quantiles
Model             LeRobot PI05Policy initialized from lerobot/pi05_base
Fine-tuning       PyTorch PEFT LoRA, rank 16
```

## 1. Which implementation this guide uses

This workspace contains a PyTorch port of π0.5 adapted from OpenPI, integrated
with LeRobot's policy, processor, dataset, checkpoint, validation, and PEFT
systems.

This is different from training directly in the official
[Physical Intelligence OpenPI repository](https://github.com/Physical-Intelligence/openpi):

- Use this workspace's `lerobot-train` implementation.
- Use `--policy.type=pi05`.
- Use LeRobot's `--peft.*` options for LoRA.
- Use `--policy.use_umi_relative_ee=true` for the strawberry EE target.
- Do not copy an OpenPI JAX training config into this repository.
- Do not run the ordinary `lerobot/pi05_base` processor for UMI data.

The relevant local implementation files are:

- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/pi05/modeling_pi05.py`
- `src/lerobot/policies/pi05/processor_pi05.py`
- `src/lerobot/processor/umi_relative_ee_processor.py`
- `src/lerobot/datasets/umi_relative_ee_stats.py`
- `src/lerobot/datasets/factory.py`
- `examples/umi_relative_ee/train_pi05_lora.sh`
- `examples/umi_relative_ee/visualize_predictions.py`

## 2. Always pin Python to the unified source tree

The `py312` Conda environment may have another LeRobot checkout installed in
editable mode. Running a script without pinning `PYTHONPATH` can silently import
the wrong branch.

Start commands from the unified workspace:

```bash
cd /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"
```

Confirm before a long run:

```bash
/home/zfei/anaconda3/envs/py312/bin/python -c \
  'import lerobot; print(lerobot.__file__)'
```

The printed path must contain:

```text
lerobot-fei-v5.0-umi-unified/src/lerobot
```

## 3. Raw dataset contract

The dataset stays in the standard LeRobot format. Each frame stores an absolute
end-effector action:

```text
action =
    [x, y, z,
     axis_angle_x, axis_angle_y, axis_angle_z,
     gripper]
```

Requirements:

- `action` has shape `(7,)`.
- Translation uses a consistent metric unit, normally metres.
- Axis-angle is a rotation vector in radians, not Euler angles.
- Gripper direction and range are consistent across episodes.
- Every action is finite.
- Every frame has a non-empty task string.
- Episodes used with `chunk_size=30` contain at least 31 contiguous frames.

An `observation.state` column is not required. In UMI mode, state is derived
from the action sequence.

The unified dataset factory checks the raw action shape before training and
rejects UMI data that is not 7D.

### Dataset used for the current experiment

Training:

```text
repo_id: sroi/sroiv2_strawberry_picking_lab_1000onesb_1125
root:    /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb_1125
```

Validation:

```text
repo_id: sroi/sroiv2_strawberry_picking_lab_validation
root:    /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
```

The validation dataset contains 100 episodes and is kept separate from the
training dataset.

## 4. Temporal query and target alignment

With:

```bash
--policy.use_umi_relative_ee=true
--policy.chunk_size=30
```

`PI05Config.action_delta_indices` returns:

```text
[-1, 0, 1, ..., 29]
```

At sample time `t`, the data loader therefore returns:

```text
action[t-1], action[t], action[t+1], ..., action[t+29]
```

The UMI derive-state processor performs:

```text
state = [action[t-1], action[t]]
targets = [action[t], action[t+1], ..., action[t+29]]
```

This produces exactly 30 retained targets.

Consequences:

- `action[t]` is both the current/base pose and the first target.
- The first relative target is approximately identity.
- The previous frame is used only to build the two-pose state.
- Episode-end padding is carried into `action_is_pad` after removing the leading
  state-only sample.

This indexing matches the unified ACT and SmolVLA UMI pipeline.

## 5. UMI SE(3) action target

For the current absolute pose:

```text
T_base = T_absolute[t]
```

For every action `k` in the retained chunk:

```text
T_relative[k] = inverse(T_base) @ T_absolute[t+k]
```

Equivalently:

```text
R_relative[k] = transpose(R_base) @ R_target[k]

p_relative[k] =
    transpose(R_base) @ (p_target[k] - p_base)
```

Every target in the chunk uses the same `T_base`. Targets are not chained:

```text
correct:   inverse(T_base) @ T_target[k]
incorrect: inverse(T_target[k-1]) @ T_target[k]
```

### 5.1 Row-based rot6d

The relative rotation matrix is encoded using its first two rows:

```text
rot6d =
    [R00, R01, R02,
     R10, R11, R12]
```

The model's physical action target is:

```text
[dx, dy, dz,
 R00, R01, R02,
 R10, R11, R12,
 gripper]
```

Therefore:

```text
raw action shape:      [batch, 31, 7]
retained raw targets:  [batch, 30, 7]
physical model target: [batch, 30, 10]
```

Do not use elementwise axis-angle subtraction. Relative orientation must be
calculated with rotation matrices.

### 5.2 Inference inverse

The postprocessor:

1. Unnormalizes the 10D prediction.
2. Reconstructs a valid relative rotation matrix from rot6d using
   Gram-Schmidt.
3. Uses the base pose cached by the preprocessor.
4. Computes:

   ```text
   T_absolute[k] = T_base @ T_relative[k]
   ```

5. Converts the absolute rotation back to axis-angle.
6. Returns an absolute 7D EE action for IK and robot control.

The cached `T_base` is shared by the entire predicted chunk.

## 6. Derived observation state

The raw state pair is:

```text
[action[t-1], action[t]]
```

Both poses are expressed relative to the current pose `action[t]`:

```text
previous_relative = inverse(T_current) @ T_previous
current_relative  = inverse(T_current) @ T_current
```

Each relative pose is encoded as 10D row-rot6d and the pair is flattened:

```text
observation.state shape = [batch, 20]
```

Interpretation:

- The first 10 dimensions describe previous-to-current motion.
- The final 10 dimensions describe the current pose relative to itself.
- The second block is close to identity, except for the absolute gripper value.

During real-time inference, the state processor buffers the previous absolute
7D EE state so it can reconstruct the same two-pose input.

## 7. Unified π0.5 preprocessing order

When `use_umi_relative_ee=true`, the π0.5 preprocessor runs:

```text
1. Rename observations
2. Add batch dimension
3. Derive [previous, current] state from action
4. Convert absolute action chunk to relative 10D rot6d
5. Convert two-pose state to relative 20D rot6d
6. Normalize state and action with q01/q99
7. Discretize normalized state into 256 bins
8. Insert task and state bins into the language prompt
9. Tokenize with google/paligemma-3b-pt-224
10. Move tensors to the configured device
```

The order is important:

```text
raw absolute -> relative SE(3) -> normalize -> tokenize
```

Normalizing the raw 7D axis-angle data before the relative transform would
produce the wrong target.

The postprocessor runs:

```text
unnormalize -> relative-to-absolute SE(3) -> CPU
```

Both pipelines are serialized into every checkpoint.

## 8. Images and language input

π0.5 is language-conditioned. The processor requires a task string, for
example:

```text
pick the strawberry
```

After state normalization, the processor discretizes each state value into one
of 256 bins and creates a prompt shaped like:

```text
Task: pick the strawberry, State: <20 integer bins>;
Action:
```

The state is therefore supplied to π0.5 through the language-token side of the
model, matching this workspace's π0.5 design.

Image behavior:

- Dataset images are returned as uint8.
- Model preprocessing resizes with padding to 224×224.
- Visual normalization mode is `IDENTITY`; π0.5 performs its expected image
  preprocessing internally.
- Do not apply ACT's ImageNet processor normalization to the π0.5 input.

The PaliGemma tokenizer is gated:

```text
google/paligemma-3b-pt-224
```

The Hugging Face account used by the training process must have accepted the
model license and be authenticated.

## 9. Model action dimensions

The physical UMI target is 10D, while π0.5 uses:

```python
max_action_dim = 32
```

Before the flow model:

```text
[batch, 30, 10] -> zero-pad -> [batch, 30, 32]
```

The action input/output projections retain the 32D pretrained model shape.

The policy loss then slices the loss tensor back to the actual configured
physical action dimension:

```text
[batch, 30, 32] -> first 10 loss dimensions
```

This unified implementation therefore does not average the reported training
loss over the 22 padded action dimensions.

At inference, generated actions are likewise sliced back to the physical output
dimension before unnormalization.

## 10. Flow-matching target and loss

π0.5 does not directly regress the clean action with ACT-style L1 loss.

Let:

```text
a       normalized clean action chunk
epsilon Gaussian noise with the same shape
t       sampled flow time
```

The model constructs:

```text
x_t = t * epsilon + (1 - t) * a
u_t = epsilon - a
```

The action expert predicts the velocity field:

```text
v_theta(x_t, observation, task, t)
```

The per-dimension objective is:

```text
(v_theta - u_t)^2
```

The unified policy:

- Keeps only the 10 physical action dimensions in the loss.
- Masks padded action timesteps using `action_is_pad`.
- Averages over valid timesteps and physical dimensions.
- Logs `loss_per_dim` for the 10 physical action dimensions.

There is no built-in translation-versus-rotation weighting. Normalization is
what puts those dimensions onto comparable scales.

## 11. Quantile normalization

The unified π0.5 defaults are:

```text
VISUAL: IDENTITY
STATE:  QUANTILES
ACTION: QUANTILES
```

For state and actions:

```text
x_normalized =
    2 * (x - q01) / (q99 - q01) - 1
```

This maps:

```text
q01 -> -1
q99 -> +1
```

Values outside the `[q01, q99]` interval are not clipped, so outliers can have
absolute normalized values larger than one.

When a dimension has an effectively zero `q99-q01` span, the normalizer uses a
safe denominator rather than dividing by zero.

### 11.1 Statistics are computed in transformed UMI space

For the training dataset, `make_dataset()` calls:

```text
compute_umi_relative_ee_stats(...)
```

That function:

1. Finds every valid contiguous 31-frame query.
2. Uses the second frame as the chunk base.
3. Converts all 30 retained actions to relative 10D rot6d.
4. Converts `[previous,current]` into the relative 20D state.
5. Computes running quantile statistics on those exact tensors.
6. Adds the transformed statistics to `dataset.meta.stats` in memory.

The raw dataset's on-disk actions and raw statistics are not rewritten.

This is a major difference from a generic π0.5 dataset: no offline conversion
to a second 10D dataset is required.

### 11.2 Validation uses training statistics

For validation:

- The validation dataset receives the UMI feature shapes and temporal query.
- UMI statistics are not recomputed from validation data.
- The saved/training preprocessor uses the training dataset's state/action
  statistics.

This avoids normalization leakage.

### 11.3 Rotation statistics to inspect

Small relative rotations are close to:

```text
[R00, R01, R02, R10, R11, R12]
    approximately
[1,   0,   0,   0,   1,   0]
```

`R00` and `R11` may therefore have narrow `q99-q01` spans. Before a long run,
inspect for every action and state dimension:

```text
q01
q99
q99 - q01
min(normalized)
max(normalized)
max(abs(normalized))
fraction(abs(normalized) > 3)
fraction(abs(normalized) > 5)
```

Also verify all values are finite.

For the first ACT-versus-π0.5 comparison, retain the same 10D rot6d target.
Only change to a 3D rotation-vector target after establishing a comparable
baseline.

## 12. LoRA implementation in this workspace

LoRA is applied through LeRobot's PyTorch PEFT integration:

```bash
--peft.method_type=LORA
--peft.r=16
--peft.lora_alpha=16
```

With rank 16 and alpha 16:

```text
LoRA scaling = alpha / rank = 1
```

The π0.5 policy supplies a default target-module regular expression covering:

- `q_proj` and `v_proj` self-attention projections in the Gemma action expert.
- Action input projection.
- Action output projection.
- Other named action/state projection modules when present and matched.

The base policy is initialized from:

```text
lerobot/pi05_base
```

The PEFT wrapper freezes non-targeted base weights and trains the inserted LoRA
parameters. Always inspect the printed trainable-parameter count before
starting the full run.

### 12.1 Public π0.5 fine-tuning recipes

There is not yet one community-standard π0.5 recipe. Published commands also
mix two materially different implementations:

- OpenPI's JAX LoRA inserts adapters into attention and feed-forward layers in
  both the PaliGemma VLM and the Gemma action expert.
- LeRobot PEFT, as configured by this workspace, adapts only action-expert
  `q_proj`/`v_proj` attention projections and the named action/state
  projections listed above.

Consequently, an OpenPI `rank=16/32` run is much broader than this workspace's
single-rank adapter, and its memory use and capacity are not directly
comparable.

Publicly documented configurations include:

| Source | Fine-tuning scope | Rank | Batch | Learning-rate schedule | Steps |
|---|---|---:|---:|---|---:|
| [Official OpenPI LIBERO config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) | Full model | N/A | 256 | peak `5e-5`, warmup 10k | 30k |
| [OpenPI community π0.5 LoRA config](https://github.com/Physical-Intelligence/openpi/issues/672) | VLM attention/FFN plus expert attention/FFN | VLM 16, expert 32 | 32 | peak `5e-5`, warmup 10k | 30k |
| [Real Franka jar task](https://huggingface.co/IDEAS-Lab-Northwestern/pi05-real-jar-60-droid-refined-lora) | OpenPI dual adapter, 60 demonstrations | VLM 16, expert 32 | 4 | Not reported | 20k |
| [Simulated multitask picking](https://huggingface.co/IDEAS-Lab-Northwestern/pi05-sim-pnp-multitask-3cam-libero-lora) | OpenPI dual adapter | VLM 16, expert 32 | 4 | `2.5e-5` to `2.5e-6`, warmup 1k | 50k + 30k |
| [LeRobot SO-101 sock task](https://huggingface.co/RyuRobot/pi05_sock_in_bowl_lora_July_10_2026) | Same target-module pattern as this workspace | 32 | 32 | `2.5e-4` to `2.5e-5`, warmup 1k | 15k |
| [Public LeRobot rank-16 run](https://huggingface.co/Tna001/pi05_lora_r16_lr3e4/blob/main/train_config.json) | LeRobot PEFT | 16 | 16 | `3e-4` to `2.5e-6`, warmup 500 | 20k |

Additional official reference points:

- The [LeRobot π0.5 guide](https://huggingface.co/docs/lerobot/pi05) shows full
  fine-tuning for 3k steps with batch 32, bfloat16, and gradient checkpointing.
- The [LeRobot LIBERO reproduction](https://huggingface.co/docs/lerobot/libero)
  starts from `pi05_libero` and trains for another 6k steps with global batch
  256 on eight H100 GPUs.
- OpenPI estimates more than 22.5 GB for its broader JAX LoRA recipe and more
  than 70 GB for full fine-tuning. These estimates do not apply directly to
  this workspace's narrower adapter; see the
  [OpenPI README](https://github.com/Physical-Intelligence/openpi).

The public native-LeRobot rank-32 adapter is particularly useful for
comparison. Its
[saved adapter configuration](https://huggingface.co/RyuRobot/pi05_sock_in_bowl_lora_July_10_2026/blob/main/adapter_config.json)
uses the same target regular expression as this workspace. The accompanying
model card reports about 2.5M trainable parameters. The adapter uses `r=32`,
`alpha=8`, so its LoRA scale is
`alpha / rank = 0.25`. The model card does not yet report real-robot evaluation,
so treat it as a reproducible training example rather than evidence that those
hyperparameters are optimal.

Hugging Face Discord history is login-gated and was not publicly indexed when
this comparison was prepared. Do not cite an alleged Discord consensus without
a stable public message or an independently reproducible configuration.

### 12.2 Interpretation for the UMI strawberry task

The baseline in this guide uses:

```text
rank = 16
alpha = 16
LoRA scale = 1
trainable parameters ~= 1.29M
peak learning rate = 1e-4
```

These values are inside the range of public LeRobot experiments:

- Rank 16 and rank 32 are both in active use.
- Batch 4 is normal for single-GPU LoRA and does not imply that batching is
  broken merely because VRAM changes little.
- Public peak learning rates range from roughly `2.5e-5` to `3e-4`; `1e-4` is
  not an obvious outlier.
- Published runs commonly use 15k-80k steps. The 500k-step command in this
  workspace is unusually long and must be justified by dataset size,
  effective epochs, validation, and physical evaluation rather than copied as
  a universal default.

Use a controlled capacity ablation before broadening the adapter:

```bash
--peft.r=32 \
--peft.lora_alpha=32
```

Keeping `alpha / rank = 1` isolates the effect of doubling adapter rank. Keep
the seed, batch size, dataset, optimizer, and validation schedule fixed, and
start from `lerobot/pi05_base` in a new output directory. Compare checkpoints
at 10k, 25k, 50k, and 100k before extending the run. If batch size changes,
compare examples or effective epochs rather than raw step counts.

Interpret the ablation as follows:

- Better training and validation performance suggests rank 16 was limiting
  capacity.
- Better training loss without better validation or robot success indicates
  overfitting, not a need for still higher rank.
- No meaningful improvement suggests that rank is not the bottleneck. Check
  normalization, UMI transforms, data diversity, camera/domain shift, and
  gripper timing next.
- If failures are primarily visual or semantic, raising the rank of the same
  action-side targets cannot adapt the frozen vision/VLM representation.
  Broader expert FFN or selected VLM targets are then a more meaningful
  experiment than immediately trying rank 64.

Always select the final checkpoint using held-out validation visualization and
closed-loop picking success. Training loss alone is insufficient for comparing
LoRA capacity.

### 12.3 Checkpoint contents

A PEFT checkpoint stores:

- Adapter weights.
- `adapter_config.json`, including the base-model reference.
- π0.5 policy `config.json`.
- `train_config.json`.
- Serialized UMI preprocessor and postprocessor.
- Normalization-stat tensor files.
- Optimizer, scheduler, RNG, and training-step state.

Do not copy only the adapter weights. The saved processors and their UMI
statistics are part of the trained policy contract.

## 13. Recommended 1125-episode training command

The shell launcher defaults to the older dataset name, so override it for the
1125-episode experiment:

```bash
cd /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"

DATASET_REPO_ID=sroi/sroiv2_strawberry_picking_lab_1000onesb_1125 \
DATASET_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb_1125 \
VALIDATION_DATASET_REPO_ID=sroi/sroiv2_strawberry_picking_lab_validation \
VALIDATION_DATASET_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
OUTPUT_DIR=outputs/train/pi05_lora_umi_relative_ee_1125train_100val \
POLICY_REPO_ID=zfff/pi05_lora_umi_relative_ee_1125train_100val \
bash examples/umi_relative_ee/train_pi05_lora.sh
```

The launcher currently uses:

```text
pretrained model       lerobot/pi05_base
LoRA rank              16
LoRA alpha             16
dtype                  bfloat16
gradient checkpointing true
torch.compile          false
batch size             2
workers                8
chunk size             30
action steps           30
peak learning rate     1e-4
decay learning rate    1e-5
warmup                 1,000 steps
decay                  50,000 steps
total steps            50,000
checkpoint frequency   5,000
validation frequency   10,000
```

The 24 GB starting configuration is batch size 2. If it runs out of memory,
reduce to batch size 1. The current trainer does not expose gradient
accumulation.

## 14. Validation semantics

The launcher provides:

```bash
--validation_dataset.repo_id=...
--validation_dataset.root=...
--val_freq=10000
```

At a validation event, the trainer:

1. Resets the processor state.
2. Evaluates the selected validation dataset deterministically.
3. Disables gradients.
4. Computes sample-weighted loss.
5. Logs under `val/`.
6. Restores training mode.
7. Resets processor state again.

Validation uses the same flow-matching loss implementation as training, with
fresh sampled noise and flow time. It is comparable across the same code and
settings, but it is not an ACT L1 loss and should not be numerically compared
directly with ACT's validation loss.

The full 100-episode validation set has 9,274 raw frames and is expensive for
π0.5. For a smoke test, use a small explicit episode subset. Use the full set
for checkpoint selection.

## 15. Required smoke tests

Before starting 50,000 steps, verify the following.

### 15.1 Import location

```bash
python -c 'import lerobot; print(lerobot.__file__)'
```

It must resolve to the unified workspace.

### 15.2 One processed batch

Inspect one batch after preprocessing:

```text
raw action query        [B, 31, 7]
derived state pair      [B, 2, 7]
retained action         [B, 30, 7]
relative state          [B, 20]
relative action         [B, 30, 10]
model-padded action     [B, 30, 32]
```

Confirm:

- The first retained target is near identity in pose.
- The first six rot6d values follow the row convention.
- The gripper stays absolute.
- Task tokens are present.
- Image keys match the checkpoint policy configuration.

### 15.3 SE(3) round trip

For sampled actions:

```text
absolute 7D
    -> relative 10D
    -> absolute 7D
```

Measure:

- Translation reconstruction error.
- Rotation geodesic error in degrees.
- Gripper reconstruction error.

Do not use direct axis-angle-vector difference as the main orientation metric.

### 15.4 Normalization audit

Check q01/q99 spans and normalized tails for the complete transformed training
set. Pay particular attention to the nearly constant rotation diagonal
channels and the identity half of the 20D state.

### 15.5 Short LoRA run

Run 10–100 steps and confirm:

- Training loss is finite.
- Every `loss_per_dim` value is finite.
- At least one LoRA parameter has a non-zero gradient.
- Frozen base parameters do not change.
- A checkpoint saves and reloads.
- The saved preprocessor still contains the UMI steps and transformed stats.

## 16. Resume

Resume from the saved training configuration:

```bash
cd /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"

/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/train_pi05_lora.py \
  --config_path=outputs/train/pi05_lora_umi_relative_ee_1125train_100val/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

Resume restores:

- Policy and LoRA adapters.
- UMI processor configuration.
- Normalization statistics.
- Optimizer.
- Scheduler.
- RNG state.
- Training step.

Do not resume by rerunning the initial launcher against an existing output
directory.

## 17. Visualization

Use the unified visualizer and explicitly pin the unified source:

```bash
cd /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"

/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/pi05_lora_umi_relative_ee_1125train_100val/checkpoints/050000/pretrained_model \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 3 4 \
  --task "pick the strawberry" \
  --project \
  --camera_info_path /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation/meta/camera_info/validation_20260714_160922-png__episode_040/camera_info_color.json \
  --output_dir outputs/debug/viz_pi05_lora_umi_1125_validation
```

The script automatically detects a PEFT checkpoint:

1. Reads `adapter_config.json`.
2. Loads the referenced base π0.5 policy using the fine-tuned policy config.
3. Applies the LoRA adapter.
4. Loads the saved UMI processors.
5. Runs recorded-observation open-loop inference.

The `config=policy_config` load step is important: it preserves the fine-tuned
camera feature names instead of reverting to the base model's default image
keys.

Each output video contains:

- Camera image.
- Predicted and GT on-image gripper-tip trajectories with `--project`.
- Predicted and GT 3D trajectories.
- Per-dimension action curves.
- Per-frame endpoint errors.

## 18. Deployment contract

At every robot control tick:

1. Obtain the current absolute 7D EE pose from FK.
2. Run the preprocessor so it updates the two-frame state history and caches the
   current chunk base.
3. Include the task string.
4. Predict a normalized relative action chunk.
5. Run the postprocessor on the entire chunk.
6. Receive absolute 7D EE targets.
7. Convert targets through IK.
8. Send safe joint commands to the robot.

When executing a predicted chunk, every relative target must be composed with
the same chunk-start base. Do not accumulate predicted relative transforms.

## 19. Common mistakes

- Importing a different LeRobot checkout from the `py312` environment.
- Forgetting `--policy.use_umi_relative_ee=true`.
- Using raw dataset q01/q99 statistics instead of transformed UMI statistics.
- Recomputing normalization statistics on validation data.
- Treating axis-angle subtraction as relative rotation.
- Using column-based rot6d instead of this workspace's row-based convention.
- Chaining relative actions instead of using one shared base.
- Removing `action[t]` because it is near identity; it is intentionally the
  first retained target.
- Applying ACT's visual normalization to π0.5.
- Omitting the task string.
- Copying only LoRA adapter files without the saved processors.
- Loading the LoRA base model without the fine-tuned policy configuration.
- Comparing π0.5 flow loss numerically with ACT L1/KL validation loss.
- Starting a full run before checking q01/q99 spans and normalized rotation
  tails.

## 20. Recommended experiment sequence

1. Verify the unified import path.
2. Run the existing UMI pipeline correctness tests.
3. Inspect one processed π0.5 batch.
4. Audit transformed q01/q99 statistics.
5. Run a 10–100-step LoRA smoke test.
6. Save, reload, and visualize the smoke-test checkpoint.
7. Run a few thousand steps and compare predictions with ACT.
8. Continue to 50,000 steps only if validation and visualization improve.
9. Select checkpoints using full validation plus physical-space visualization,
   not training loss alone.

## 21. Related unified documentation

- `examples/umi_relative_ee/README.md`
- `examples/umi_relative_ee/prediction_visualization.md`
- `examples/umi_relative_ee/visualize_predictions_pi05.md`
- `examples/umi_relative_ee/train_pi05_lora.sh`
- `docs/source/pi05.mdx`


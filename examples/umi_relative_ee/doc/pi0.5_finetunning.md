---
created: 2026-07-27
updated: 2026-08-04
workspace: /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
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
/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
```

It is specific to the unified branch and the strawberry-picking datasets. It
explains the raw data contract, temporal query, SE(3) target, normalization,
flow-matching loss, PEFT LoRA setup, validation, checkpoint loading, and
visualization.

This is now the authoritative combined guide. It incorporates the completed
broad-LoRA run, the OpenPI comparison, and the validation-loss versus physical
prediction investigation that were previously split across two other notes.

The demonstrated configuration is:

```text
Raw dataset       absolute 7D axis-angle EE poses
Processor target  same-base relative 10D row-rot6d
State             derived 20D two-pose relative state
Chunk             30 actions
Normalization     q01/q99 quantiles
Model             LeRobot PI05Policy initialized from lerobot/pi05_base
Fine-tuning       broad PyTorch PEFT LoRA, rank 16/alpha 16
Targets           VLM + expert q/k/v/o and FFN
Full modules      action input/output and time MLP input/output
Padding flow      masked_subspace
```

The recommended next capacity experiment changes only the expert LoRA to
rank/alpha 32/32 while retaining VLM 16/16. This matches OpenPI's split LoRA
definition more closely; details are in section 12.2.

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
- `examples/umi_relative_ee/shell_scripts/train_pi05_lora.sh`
- `examples/umi_relative_ee/visualize_predictions.py`

## 2. Always pin Python to the unified source tree

The `py312` Conda environment may have another LeRobot checkout installed in
editable mode. Running a script without pinning `PYTHONPATH` can silently import
the wrong branch.

Start commands from the unified workspace:

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
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

Training used by the completed broad run:

```text
repo_id: sroi/sroiv2_strawberry_picking_lab_1302_occlusion
root:    /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion
```

Validation:

```text
repo_id: sroi/sroiv2_strawberry_picking_lab_validation
root:    /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
```

The validation dataset contains 100 episodes and is kept separate from the
training dataset. The older 1125-episode commands are retained in section 13.3
as a legacy launcher example.

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

The action input/output projections retain the 32D pretrained model shape. The
loss and sampling domain depend on `flow_matching_padding_mode`:

| Mode | Training noise and loss | Inference |
| --- | --- | --- |
| `openpi_full_width` | 32-D Gaussian noise; mean loss over all 32 coordinates | integrate all 32, then slice to 10 |
| `masked_subspace` | padded noise is zero; loss only over the first 10 coordinates | keep padded flow state/velocity zero, then slice to 10 |

The default remains OpenPI full width for pretrained compatibility. The
completed broad-LoRA UMI run explicitly used `masked_subspace` to concentrate
adaptation on the physical 10-D action space.

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

The policy masks invalid episode-boundary timesteps with `action_is_pad` and
logs `flow_loss_real_dims` and `flow_loss_padded_dims`. In full-width mode the
scalar averages 10 real and 22 padded coordinates; in masked-subspace mode it
averages only the 10 real coordinates.

There is no built-in translation-versus-rotation weighting or smoothness term.
Normalization puts dimensions onto broadly comparable scales, but this remains
normalized velocity-field MSE—not millimetres, SO(3) degrees, endpoint error,
or task success. Section 22 gives the measured consequences.

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

LoRA is applied through LeRobot's PyTorch PEFT integration. The old local
default was narrow and trained only about 1.29M parameters. It targeted action
expert q/v and action projections, and formerly named nonexistent
`action_time_mlp_in/out` modules. The default naming bug has been corrected to
`time_mlp_in/out`, but the default is still intentionally a low-memory narrow
adapter.

The completed broad run overrides the default:

```bash
--peft.method_type=LORA
--peft.r=16
--peft.lora_alpha=16
--peft.target_modules='.*\.(paligemma|gemma_expert)\..*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
--peft.full_training_modules='["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]'
```

With rank 16 and alpha 16:

```text
LoRA scaling = alpha / rank = 1
```

This adapts q/k/v/o and gate/up/down in both the PaliGemma language model and
Gemma action expert. The four action/time projections are fully trained and
saved with the adapter. The resulting run trained 31,693,856 parameters.

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
- This workspace's **default** LeRobot PEFT target remains the narrow
  action-expert q/v pattern; the demonstrated launcher overrides it with the
  broad OpenPI-like attention+FFN scope.

Consequently, the completed local broad run is comparable in module scope to
OpenPI, but not in rank: it uses one global rank 16 rather than VLM 16 / expert
32.

Publicly documented configurations include:

| Source | Fine-tuning scope | Rank | Batch | Learning-rate schedule | Steps |
|---|---|---:|---:|---|---:|
| [Official OpenPI LIBERO config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) | Full model | N/A | 256 | peak `5e-5`, warmup 10k | 30k |
| [OpenPI community π0.5 LoRA config](https://github.com/Physical-Intelligence/openpi/issues/672) | VLM attention/FFN plus expert attention/FFN | VLM 16, expert 32 | 32 | peak `5e-5`, warmup 10k | 30k |
| [Real Franka jar task](https://huggingface.co/IDEAS-Lab-Northwestern/pi05-real-jar-60-droid-refined-lora) | OpenPI dual adapter, 60 demonstrations | VLM 16, expert 32 | 4 | Not reported | 20k |
| [Simulated multitask picking](https://huggingface.co/IDEAS-Lab-Northwestern/pi05-sim-pnp-multitask-3cam-libero-lora) | OpenPI dual adapter | VLM 16, expert 32 | 4 | `2.5e-5` to `2.5e-6`, warmup 1k | 50k + 30k |
| [LeRobot SO-101 sock task](https://huggingface.co/RyuRobot/pi05_sock_in_bowl_lora_July_10_2026) | Narrow LeRobot default-style target pattern | 32 | 32 | `2.5e-4` to `2.5e-5`, warmup 1k | 15k |
| [Public LeRobot rank-16 run](https://huggingface.co/Tna001/pi05_lora_r16_lr3e4/blob/main/train_config.json) | LeRobot PEFT | 16 | 16 | `3e-4` to `2.5e-6`, warmup 500 | 20k |

Additional official reference points:

- The [LeRobot π0.5 guide](https://huggingface.co/docs/lerobot/pi05) shows full
  fine-tuning for 3k steps with batch 32, bfloat16, and gradient checkpointing.
- The [LeRobot LIBERO reproduction](https://huggingface.co/docs/lerobot/libero)
  starts from `pi05_libero` and trains for another 6k steps with global batch
  256 on eight H100 GPUs.
- OpenPI estimates more than 22.5 GB for its JAX LoRA recipe and more
  than 70 GB for full fine-tuning. These estimates do not apply directly to
  this workspace's PyTorch PEFT implementation; see the
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

### 12.2 Recommended higher-capacity UMI configuration

OpenPI defines [`gemma_2b_lora`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py)
with attention and FFN rank/alpha 16/16, and `gemma_300m_lora` with attention
and FFN rank/alpha 32/32. The completed local
run matched the official module scope but used one global rank 16, including in
the action expert. The clearest next capacity step is therefore:

| Component | Rank | Alpha | Targets |
| --- | ---: | ---: | --- |
| PaliGemma/VLM language layers | 16 | 16 | q/k/v/o + gate/up/down |
| Gemma action expert | 32 | 32 | q/k/v/o + gate/up/down |
| Action/time projections | full | N/A | `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out` |
| Vision encoder | frozen initially | N/A | none |

Recommended training parameters on this 24 GB RTX 4090:

```text
base                         lerobot/pi05_base
padding flow                 masked_subspace
chunk / execution horizon    30 / 30
batch size                   4; fall back to 2 only after a real memory test
optimizer                    AdamW, betas 0.9/0.95, eps 1e-8
peak / final LR              5e-5 / 5e-6
weight decay                 0.01
gradient clipping            1.0
warmup                       1,000 steps at batch 4
schedule                     cosine decay
total steps                  100,000 maximum
checkpoint / physical audit  every 12,500 steps
dtype                        bfloat16
gradient checkpointing       true
compile                      false for the first verified run
seed                         1000
```

At batch 4, 75K steps processes 300K samples (about 2.47 dataset passes) and
100K processes 400K samples (about 3.30 passes). Do not assume the final step is
best: run the full 100-episode decoded audit at every checkpoint. The current
evidence favors continuing through at least 75K; 100K is an evaluation ceiling,
not an instruction to deploy the last checkpoint.

The local `PeftConfig` now exposes PEFT `rank_pattern` and `alpha_pattern`, so
the split can be expressed directly. The global 16/16 values apply to all broad
targets, while the expert pattern overrides matched Gemma expert modules:

```bash
--peft.r=16
--peft.lora_alpha=16
--peft.rank_pattern="{'.*\\.gemma_expert\\..*': 32}"
--peft.alpha_pattern="{'.*\\.gemma_expert\\..*': 32}"
```

This is preferable to global 32/32 because it increases action-expert capacity
without unnecessarily doubling the VLM adapter rank.

Do not unfreeze the full vision encoder as the first next step. If the split-rank
run still fails mainly on unseen lighting, occlusion, or strawberry appearance,
then test low-rank adaptation of the last vision blocks or unfreeze only the
last one or two blocks. Full-model fine-tuning is not a practical 24 GB recipe.

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

## 13. Training commands

### 13.1 Demonstrated broad-LoRA 1302 run

The completed run used the current 1302 occlusion dataset and batch 4:

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_pi05_broad_lora_umi.sh 4 \
  > examples/umi_relative_ee/logs/pi05_broad_lora_masked_bs4.log 2>&1 < /dev/null &
```

The launcher explicitly sets broad targets, masked-subspace flow, batch-scaled
steps, validation, and checkpoint frequencies. Batch 4 runs 75K steps; batch 2
runs 150K so both process 300K samples. Do not run both simultaneously.

### 13.2 Higher-capacity split-rank run

The OpenPI-style launcher uses PaliGemma rank/alpha 16/16 and Gemma expert
32/32:

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
bash examples/umi_relative_ee/shell_scripts/run_pi05_openpi_split_lora_umi.sh 4
```

Batch 4 runs 100K steps and saves every 12.5K. Batch 2 is only the memory
fallback; it runs 200K steps so both settings process 400K samples. The live
experiment and output paths are recorded in section 24.

### 13.3 Legacy 1125-episode narrow launcher

The shell launcher defaults to the older dataset name, so override it for the
1125-episode experiment:

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"

DATASET_REPO_ID=sroi/sroiv2_strawberry_picking_lab_1000onesb_1125 \
DATASET_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb_1125 \
VALIDATION_DATASET_REPO_ID=sroi/sroiv2_strawberry_picking_lab_validation \
VALIDATION_DATASET_ROOT=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
OUTPUT_DIR=outputs/train/pi05_lora_umi_relative_ee_1125train_100val \
POLICY_REPO_ID=zfff/pi05_lora_umi_relative_ee_1125train_100val \
bash examples/umi_relative_ee/shell_scripts/train_pi05_lora.sh
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

## 14. Validation semantics and checkpoint selection

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

Validation uses the same flow-matching loss implementation as training. It
resets the RNG to seed 0 for deterministic evaluation. Raw loss is comparable
only within the same padding objective. In full-width mode, 22 easy padded
coordinates dilute the scalar; use `flow_loss_real_dims` for a cautious
cross-mode diagnostic.

Flow loss is not sufficient for checkpoint selection even within one mode. It
scores normalized velocity MSE across all chunk timesteps, whereas deployment
integrates the field and cares about physical XYZ, SO(3), gripper, and
smoothness. Run `eval_open_loop_dataset.py` on the full validation set at every
saved checkpoint. Section 22 records the observed disagreement.

The full 100-episode validation set has 9,274 raw frames and is expensive for
π0.5. For a smoke test, use a small explicit episode subset. Use the full set
for checkpoint selection.

## 15. Required smoke tests

Before starting a long run, verify the following.

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
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
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
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
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
8. Continue based on decoded all-episode metrics, not raw flow loss alone.
9. Compare pose chunk mean, endpoint, gripper, and within-chunk acceleration.
10. Confirm the selected checkpoints in closed-loop robot trials.

## 21. Completed broad-LoRA run

The previous narrow full-width run trained 1,287,168 parameters and was stopped
after its 400K checkpoint was preserved. Its default target covered mainly
action-expert q/v and action projections; the then-broken time-MLP target names
matched no modules.

The replacement configuration was:

| Setting | Broad run |
| --- | --- |
| Dataset | `sroiv2_strawberry_picking_lab_1302_occlusion` |
| Base | `lerobot/pi05_base` |
| Real/model action width | 10 / 32 |
| Padding flow | `masked_subspace` |
| LoRA scope | q/k/v/o and gate/up/down in VLM language layers and action expert |
| Fully trained modules | action input/output and time MLP input/output |
| Vision encoder | frozen |
| Rank / alpha | global 16 / 16 |
| Peak / final LR | `5e-5` / `5e-6` |
| Batch / steps | 4 / 75K |
| Learnable / total parameters | 31,693,856 / 4,175,098,672 |
| Observed GPU memory | about 12,220 / 24,564 MiB |
| Throughput | about 1.7-1.8 steps/s |

Run artifacts:

```text
output: outputs/train/pi05_broad_lora_masked_1302_bs4
log:    examples/umi_relative_ee/logs/pi05_broad_lora_masked_bs4.log
W&B:    biorobotlab/lerobot/9erobpvd
```

The run completed 75K steps and 300K samples. Saved checkpoints are 12.5K,
25K, 37.5K, 50K, 62.5K, and 75K.

### Difference from the official LeRobot example

The LeRobot guide's example is full-model fine-tuning: vision encoder, VLM,
action expert, and projections are all trainable. It uses batch 32 and 3K steps
(96K examples), compile, and gradient checkpointing. This broad-LoRA run is a
24 GB task-adaptation recipe: frozen vision, LoRA transformer weights, full
action/time projections, batch 4, and a UMI-specific 10-D masked flow.

### Difference from official OpenPI LoRA

OpenPI's checked-in LoRA model definitions apply adapters to attention and FFN
with VLM rank/alpha 16/16 and expert 32/32. The local completed run matches the
scope and VLM rank but uses expert 16/16. OpenPI does not currently provide one
canonical checked-in π0.5 LoRA training preset; its explicit low-memory LoRA
preset is for π0, while its checked-in π0.5 LIBERO preset is a full fine-tune.

## 22. Why validation flow loss disagreed with prediction quality

The raw validation losses of full-width and masked-subspace runs are not
comparable. At old narrow 200K:

```text
(10 * 0.038027 real-dim loss + 22 * 0.000191 padded-dim loss) / 32
= 0.012015 reported loss
```

Broad 25K reports `0.037607` because it averages only the 10 real dimensions.
Thus the corrected comparison is 0.038027 versus 0.037607, not 0.012015 versus
0.037607.

| Checkpoint | Padding | Raw validation loss | Real-dim loss | Padded loss |
| --- | --- | ---: | ---: | ---: |
| narrow 200K | full width | **0.012015** | 0.038027 | 0.000191 |
| broad 25K | masked | 0.037607 | **0.037607** | 0 |
| broad 75K | masked | 0.052667 | 0.052667 | 0 |

Even the real-dimension proxy disagreed with deployed behavior. Reasons:

1. It measures normalized velocity at sampled flow points, not the action after
   ten Euler integration steps.
2. It averages raw XYZ, rot6d, and gripper coordinates rather than metres and
   SO(3) geodesic angle.
3. It averages all 30 timesteps and has no endpoint or smoothness term.
4. Full-width and masked models are conditioned on different padded flow states.
5. Grouped XYZ/rotation/gripper tradeoffs are hidden in one 10-D mean.
6. Broad LoRA has about 25 times the trainable capacity of the old adapter.

The two checkpoint processor files were diffed: transforms, normalization
modes, and statistics are identical; only the runtime cache ID differs. The
improvement is not caused by mismatched normalization.

### Full 100-episode physical audit

All 100 validation episodes were evaluated at five evenly spaced non-padded
query frames, seed 1000, using checkpoint-saved processing. Values are
episode-balanced and lower is better.

| Metric | narrow 200K | broad 25K | broad 75K |
| --- | ---: | ---: | ---: |
| Rotation chunk mean | 3.077° | 2.749° | **2.516°** |
| Rotation endpoint | 5.584° | 4.846° | **4.500°** |
| XYZ chunk mean | 16.75 mm | 14.73 mm | **13.63 mm** |
| XYZ endpoint | 31.20 mm | 25.02 mm | **23.47 mm** |
| Gripper chunk mean | 0.1060 | **0.0996** | 0.1032 |
| Gripper endpoint | 0.1717 | **0.1580** | 0.1655 |
| Rotation acceleration proxy | 0.1013° | 0.0880° | **0.0729°** |
| XYZ acceleration proxy | 0.563 mm | 0.476 mm | **0.382 mm** |

Broad 75K improves old 200K by about 19% in rotation endpoint, 25% in XYZ
endpoint, 28% in rotation acceleration, and 32% in XYZ acceleration. Broad 25K
is the gripper-favoring alternative. The earlier interpretation that 75K was
simply overfit came from flow loss and three episodes; it is not supported by
the full physical audit.

Reports:

- `outputs/debug/compare_pi05_loss_vs_physical/all100_old_fullwidth_200k/pi05_lora_openpi_fullwidth_1302_1M_0200000_open_loop_metrics.json`
- `outputs/debug/compare_pi05_loss_vs_physical/all100_broad_masked_25k/pi05_broad_lora_masked_1302_bs4_025000_open_loop_metrics.json`
- `outputs/debug/compare_pi05_loss_vs_physical/all100_broad_masked_75k/pi05_broad_lora_masked_1302_bs4_075000_open_loop_metrics.json`

For this run, test broad 75K first for pose/smoothness and broad 25K as the
gripper-favoring alternative. Real-robot picking remains the final selector.

## 23. Related unified documentation

- `examples/umi_relative_ee/doc/README.md`
- `examples/umi_relative_ee/doc/prediction_visualization.md`
- `examples/umi_relative_ee/doc/visualize_predictions_pi05.md`
- `examples/umi_relative_ee/shell_scripts/train_pi05_lora.sh`
- `examples/umi_relative_ee/shell_scripts/run_pi05_broad_lora_umi.sh`
- `examples/umi_relative_ee/shell_scripts/run_pi05_openpi_split_lora_umi.sh`
- `docs/source/pi05.mdx`

## 24. Completed higher-capacity split-rank run

Started on 2026-08-03 at 16:52 Asia/Taipei in tmux window
`0:pi05-split`:

```text
run name       pi05_openpi_split_lora_masked_1302_bs4
output         outputs/train/pi05_openpi_split_lora_masked_1302_bs4
log            examples/umi_relative_ee/logs/pi05_openpi_split_lora_masked_1302_bs4.log
W&B            https://wandb.ai/biorobotlab/lerobot/runs/pnbp5ewm
batch / steps  4 / 100,000
save / val     12,500 / 5,000 steps
trainables     38,624,288
total params   4,182,029,104
GPU memory     12,339 / 24,564 MiB after backward
throughput     about 1.8 steps/s at startup
```

The logged PEFT configuration contains global 16/16 and expert
`rank_pattern`/`alpha_pattern` 32/32. Compared with the completed broad global
rank-16 run, trainable parameters increased from 31,693,856 to 38,624,288. At
step 50 the run reported loss 0.467 and gradient norm 2.446; at step 200 it
reported loss 0.354 and gradient norm 1.814. There was no out-of-memory error.
Batch 4 therefore fits with about 11.8 GiB free after a forward/backward step.

The scheduled checkpoints are 12.5K, 25K, 37.5K, 50K, 62.5K, 75K, 87.5K,
and 100K. Apply the same 100-episode decoded audit to each useful checkpoint;
do not select 100K merely because it is last.

Progress snapshot at 2026-08-03 22:17 Asia/Taipei:

```text
step                 30,588 / 100,000
recent throughput    about 1.77 steps/s
recent train loss    about 0.03-0.04
saved checkpoints    12.5K and 25K (436 MiB each)
GPU                   11,947 MiB, 92% utilization, 86 C, 375 W
estimated completion  about 2026-08-04 10:40 Asia/Taipei
```

| Validation step | Masked real-dimension flow loss |
| ---: | ---: |
| 5K | 0.044052 |
| 10K | 0.038571 |
| 15K | 0.039730 |
| 20K | **0.038474** |
| 25K | 0.039386 |
| 30K | 0.040399 |
| 35K | 0.041474 |
| 40K | 0.043412 |
| 45K | 0.043351 |
| 50K | 0.043854 |
| 55K | 0.046869 |
| 60K | 0.046562 |
| 65K | 0.052282 |
| 70K | 0.049739 |
| 75K | 0.052020 |
| 80K | 0.054120 |
| 85K | 0.054612 |
| 90K | 0.054842 |
| 95K | 0.057460 |
| 100K | 0.059480 |

The run completed all 100,000 steps on 2026-08-04 at 10:37 Asia/Taipei
(17h43m wall, about 1.79 steps/s mean), processing 400K samples (3.30 epochs).
Best masked flow validation is at 20K (**0.038474**); from there it rises
monotonically to 0.059480 at 100K, a 55% increase — the same flow-loss
overfitting shape as the prior broad-LoRA run. As established in section 22,
rising flow loss does not by itself indicate worse decoded behavior: the
5-episode decoded check (section 28.6) shows 38M@75K only marginally better
than 38M@50K on mean endpoint error and equal on the median. Do not select
100K from this table; the section 22 physical audit and closed-loop trials
remain the selectors. All eight scheduled checkpoints (12.5K-100K) plus
`last` are saved.

## 25. Single-GPU common sense: RTX 4090 and RTX 5090

Both cards should normally be treated as π0.5 **LoRA** devices, not full-model
fine-tuning devices. The RTX 4090 has 24 GB and the RTX 5090 has 32 GB, while
OpenPI estimates more than 22.5 GB for its native LoRA path and more than 70 GB
for full fine-tuning. Exact memory differs between OpenPI JAX and this
workspace's PyTorch PEFT implementation.

| Setting | RTX 4090 starting point | RTX 5090 starting point |
| --- | --- | --- |
| Fine-tuning method | broad or mixed-rank LoRA | same mixed-rank LoRA first |
| VLM / expert rank | 16 / 16–32 | 16 / 32; increase only after an ablation |
| Vision encoder | frozen | frozen initially |
| Action/time projections | fully trained | fully trained |
| Precision | bfloat16 | bfloat16 |
| Gradient checkpointing | enabled | enabled initially |
| Initial batch | 2–4 | 4–8 |
| Full-model fine-tune | not practical | still not practical on 32 GB |

More VRAM should first buy stability, batch size, cameras, and controlled
ablations—not indiscriminate rank or full vision unfreezing. If a 5090 is used,
keep the 38.62M parameter configuration for the first hardware comparison and
increase batch only after measuring memory. Preserve the same number of seen
samples by reducing optimizer steps when batch is increased; changing rank,
batch, learning rate, and sample budget simultaneously prevents a clean
comparison.

Only adapt late vision blocks after decoded errors demonstrate a visual domain
gap, such as failures specific to lighting, occlusion, camera placement, or
object appearance. For action geometry and within-chunk behavior, prioritize
the action expert, projections, target representation, normalization, and
physical checkpoint metrics.

### Why OpenPI says more than 22.5 GB while this run uses 12.34 GiB

These measurements are not the same recipe. OpenPI's checked-in low-memory
LoRA example is currently a π0 JAX configuration and inherits global batch 32.
OpenPI documents JAX training as float32 weights and gradients with mostly
bfloat16 computation, and recommends allowing XLA to reserve 90% of GPU
memory. Its more-than-22.5-GB figure is therefore a capacity requirement for
that stack, not a minimum that every correct LoRA implementation must consume.

This run uses LeRobot PyTorch PEFT with batch 4, a bfloat16 policy path for most
language/action base weights, a frozen base and vision encoder, and gradient
checkpointing. Some numerically sensitive or adapter parameters can remain
float32, but this is still substantially different from JAX float32 base
weights. Its
measured 12,339 MiB after backward is consequently plausible even though it
trains a broad 38.62M-parameter adapter. Lower memory does not imply that only
the old narrow modules were matched: the logged trainable count increased from
31,693,856 at global rank 16 to 38,624,288 with expert rank 32, exactly the
expected 6,930,432-parameter increase.

## 26. Full-parameter batch-1 trial on the two-4090 host

On 2026-08-03, the repository and UMI assets were prepared on
`zfei@10.98.19.22:2202` to test whether full-parameter π0.5 training can pass a
real optimizer step at batch 1. This is an empirical memory trial, not a claim
that two RTX 4090 cards meet OpenPI's greater-than-70-GB recommendation.

Remote layout:

```text
repository  /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
train data  /home/zfei/data/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion
validation  /home/zfei/data/lerobot/sroiv2_strawberry_picking_lab_validation
base model  /home/zfei/.cache/huggingface/hub/models--lerobot--pi05_base
Python      /home/zfei/anaconda3/envs/py312/bin/python
```

The host has two 24,564-MiB RTX 4090 GPUs and 62 GiB system RAM. Code, both
datasets, and the local Hugging Face π0.5 cache were copied because the remote
Hugging Face connection timed out. The Python 3.12 environment was also copied
from the working machine because the remote package mirror downloaded large
training wheels too slowly to be operationally useful.

The launcher is
`examples/umi_relative_ee/shell_scripts/run_pi05_full_finetune_umi_bs1.sh`. It deliberately
omits `--peft`, sets both `freeze_vision_encoder=false` and
`train_expert_only=false`, uses bfloat16 plus gradient checkpointing, batch 1,
30-step masked-subspace actions, and the same 5e-5 to 5e-6 learning-rate
schedule as the LoRA comparison. The 100K configuration saves only at the end
because a full model plus AdamW state can consume tens of gigabytes per
checkpoint.

The required gate before launching 100K steps is:

```bash
cd /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
TRAIN_STEPS=2 WARMUP_STEPS=1 SAVE_CHECKPOINT=false VAL_FREQ=0 \
WANDB_ENABLE=false RUN_SUFFIX=smoke CUDA_DEVICE=0 \
bash examples/umi_relative_ee/shell_scripts/run_pi05_full_finetune_umi_bs1.sh
```

Batch 1 reduces activations but does not materially reduce parameter,
gradient, or optimizer-state memory. Ordinary two-GPU DDP is not a solution:
it replicates the complete model and optimizer on each GPU. Using both cards
for memory would require an explicitly tested sharded strategy such as FSDP or
ZeRO; 48 GB aggregate VRAM is still below OpenPI's full-finetuning estimate.

### Measured result

The single-GPU smoke test loaded all 812 checkpoint keys and reported
4,143,404,816 learnable parameters out of 4,143,404,816 total. Forward and
backward completed, but the first `optimizer.step()` failed while PyTorch Adam
was allocating `exp_avg_sq`:

```text
GPU 0 capacity       23.52 GiB
process usage        23.46 GiB
PyTorch allocated    22.08 GiB
PyTorch reserved      0.93 GiB
failed allocation    20.00 MiB
completed steps       0
```

This is decisive: batch 1 fits the activations and gradients but not the Adam
state. Do not launch the unsharded 100K configuration on a 24-GB card.

Two bounded sharding tests were also performed. Accelerate FSDP2 found and
wrapped the custom `_PiGemmaDecoderLayerBase` and `SiglipEncoderLayer` classes,
but failed in the vision patch convolution because the custom model mixed a
regular input tensor with a sharded DTensor. FSDP1 failed during wrapping
because each custom transformer layer contains both bfloat16 and float32
parameters, which cannot be combined into one flat parameter. Ordinary DDP
would replicate the single-GPU failure. No long full-parameter run was
launched.

Logs on the remote host:

```text
examples/umi_relative_ee/logs/pi05_full_finetune_masked_1302_bs1_smoke2.log
examples/umi_relative_ee/logs/pi05_full_finetune_masked_1302_bs1_fsdp2_smoke2.log
examples/umi_relative_ee/logs/pi05_full_finetune_masked_1302_bs1_fsdp1_smoke.log
```

### Does this follow the official LeRobot example?

It follows the official definition of full-model fine-tuning, but adapts the
task and runtime settings to this UMI experiment.

| Setting | Official documentation example | This UMI trial | Reason |
| --- | --- | --- | --- |
| Base | `lerobot/pi05_base` | same | same initialization |
| PEFT | none | none | full parameters |
| Vision frozen | false | false | train vision |
| Expert-only | false | false | train VLM and expert |
| Precision | bfloat16 | same | reduced memory |
| Gradient checkpointing | true | same | reduced activation memory |
| Compile | true | false | simpler memory diagnosis and sharding compatibility |
| Batch | 32 | 1 per GPU | hardware limit |
| Steps | 3,000 | planned 100,000 | existing 1,302-episode experiment budget |
| Action target | generic policy defaults | UMI relative EE, 10-D rot6d | embodiment-specific geometry |
| Chunk | default 50 | 30 | UMI control horizon |
| Padding loss | default full width | masked 10-D subspace | do not spend loss on 22 padded coordinates |
| Learning rate | policy default | 5e-5 to 5e-6 cosine | match this experiment family |

The official command's batch 32 is an example to adapt to available hardware;
it is not a promise that full tuning fits a consumer GPU. For this exact
LeRobot mixed-dtype implementation, the observed pre-optimizer usage plus the
remaining Adam moments suggests roughly 45–55 GiB at batch 1, but allocator,
checkpointing, and implementation details make that a planning estimate. Use
an 80-GB-class GPU for a reliable run, consistent with OpenPI's greater-than-
70-GB guidance. Treat 48 GB as experimental, not guaranteed. A 24- or 32-GB
card should use LoRA unless a compatible sharded/offloaded implementation is
added and validated.

## 27. High-capacity LoRA on kiwi's RTX 5080

A second LoRA run was launched on `kiwi` on 2026-08-04 without interrupting
the local rank-16/32 run. Kiwi has one RTX 5080 with 16,303 MiB. The selected
configuration uses global rank/alpha 96/96 and action-expert rank/alpha
192/192 at batch 4.

| Configuration | Local split-rank baseline | Kiwi high-capacity run |
| --- | ---: | ---: |
| Vision/VLM rank | 16 | 96 |
| Action-expert rank | 32 | 192 |
| Trainable parameters | 38,624,288 | 220,916,768 |
| Relative capacity | 1.00× | 5.72× |
| Batch size | 4 | 4 |
| Steady GPU memory | about 12.34 GiB on RTX 4090 | 14,994 / 16,303 MiB |
| Steady throughput | about 1.75 steps/s | about 1.27 steps/s |

The kiwi trainable-parameter breakdown is:

```text
vision LoRA          17,915,904
VLM language LoRA   117,669,888
action-expert LoRA   83,165,184
full projections      2,165,792
total               220,916,768
```

`freeze_vision_encoder=true` freezes the pretrained vision weights and keeps
the vision tower in evaluation mode. PEFT is applied afterward, so LoRA
weights on matched vision attention q/k/v projections remain trainable. The
PaliGemma language model and action expert adapt attention plus FFN modules;
the action/time input and output projections remain fully trained.

Before the long run, rank 96/192 passed a two-step optimizer smoke test and a
50-step steady-state sweep. The sweep held 14,994 MiB, trained at about 1.20–
1.29 steps/s, and ended without OOM. This uses about 92% of the card while
retaining roughly 1.28 GiB for allocator variation and validation. Rank
112/224 was not selected because its estimated margin was too small for a
reliable overnight job.

Active run:

```text
host          kiwi
run name      pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full
output        outputs/train/pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full
log           examples/umi_relative_ee/logs/pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full.log
W&B           https://wandb.ai/biorobotlab/lerobot/runs/fyatnla4
steps         100,000
save / val    12,500 / 5,000 steps
launcher      examples/umi_relative_ee/shell_scripts/run_pi05_high_capacity_lora_kiwi.sh
```

At 2026-08-04 07:17 Asia/Taipei, the run was healthy just past step 26,000.
Recent training loss was approximately 0.030--0.041. Validation at step 25,000
was 0.040136 on real action dimensions (`flow_loss_padded_dims=0`), and the
25,000 checkpoint was written. It was using about 14,940 MiB with 98% GPU
utilization and running at about 1.27 steps/s between validation/checkpoint
pauses. Approximately 16.2 compute hours remained, with a rough wall-clock
completion estimate near 2026-08-05 01:00 Asia/Taipei after scheduled pauses.

At the same snapshot, the local rank-16/32 baseline had reached step 80,350
of 100,000 at about 1.70--1.75 steps/s. Recent training loss was approximately
0.017--0.028, and validation at step 80,000 was 0.054120 on real action
dimensions (`flow_loss_padded_dims=0`). Checkpoints through step 75,000 were
present. Its compute ETA was about 3.2 hours, plus the remaining validation
and checkpoint pauses. Compare both runs at matched seen-sample checkpoints
with the same decoded 100-episode physical audit. More adapter parameters do
not guarantee better robot behavior, so do not select the kiwi final
checkpoint from training or validation flow loss alone.

Progress update at 2026-08-04 about 14:05 Asia/Taipei: the kiwi run was at
step about 54,649 / 100,000 (about 55%), running at about 1.28 steps/s,
14,950 MiB / 98% util / 66 C, newest checkpoint 50K (matched-50K decoded
comparison in section 28.6), ETA near 2026-08-05 00:00 Asia/Taipei. The local
split-rank baseline has finished (section 24), so the local 4090 is idle.

| Validation step | Masked real-dimension flow loss |
| ---: | ---: |
| 5K | 0.040779 |
| 10K | 0.042527 |
| 15K | 0.039064 |
| 20K | **0.036459** |
| 25K | 0.040136 |
| 30K | 0.041968 |
| 35K | 0.041201 |
| 40K | 0.044948 |
| 45K | 0.043402 |
| 50K | 0.049185 |
| 55K | 0.049476 |
| 60K | 0.050197 |
| 65K | 0.052390 |
| 70K | 0.053407 |
| 75K | 0.056286 |
| 80K | 0.059316 |
| 85K | 0.060923 |
| 90K | 0.064176 |
| 95K | 0.068004 |
| 100K | 0.069094 |

The run completed all 100,000 steps on 2026-08-05 at 01:10 Asia/Taipei
(24h20m wall, about 1.27 steps/s), processing 400K samples (3.30 epochs).
Final train loss about 0.02. Best masked flow validation is at 20K
(**0.036459**); from there it rises monotonically to 0.069094 at 100K — a
steeper overfit climb than the 38M split-rank run (0.038474 -> 0.059480). The
best-val checkpoint that is actually saved is 25K (0.040136), because
checkpoints are written every 12.5K and the 20K optimum is not saved.

Verdict: the 220M run is a negative capacity result. Its best flow val (20K)
is only marginally lower than the 38M's (0.0365 vs 0.0385), yet it overfit
harder by 100K (0.0691 vs 0.0595), and the matched-50K decoded comparison
(section 28.6) showed no endpoint-quality advantage at equal training. The
5.72x adapter capacity did not improve behavior; the 38M split-rank
configuration remains the recommended π0.5 recipe (section 28, README). All
eight checkpoints (12.5K-100K) plus `last` are saved on kiwi.

## 28. Open-loop prediction visualization of the two active runs

On 2026-08-04 the most recent checkpoint of each active run was visualized with
`visualize_predictions.py` (section 17) on the held-out validation set,
episodes 0-4, identical seed, with `--project` on-image gripper-tip projection.
This is a qualitative + decoded-metric snapshot, not a checkpoint-selection
audit: it samples five episodes, whereas section 22's audit runs all 100.

### 28.1 Checkpoints visualized

| Run | Most recent checkpoint | Step then training | Trainable params |
| --- | --- | ---: | ---: |
| Local split-rank `pi05_openpi_split_lora_masked_1302_bs4` | `075000` | about 83K / 100K | 38,624,288 |
| Kiwi high-capacity `pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full` | `025000` | about 28K / 100K | 220,916,768 |

The two checkpoints are at **different training progress** (75K vs 25K, i.e. the
38M run has seen about 3× the samples). Read the metrics below with that caveat;
see section 28.4.

### 28.2 How the 38M run was visualized (local, alongside training)

The local 4090 held about 12.6 GiB free while the 38M training consumed about
11.7 GiB, so visualization ran concurrently with the live run:

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"

/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/pi05_openpi_split_lora_masked_1302_bs4/checkpoints/075000/pretrained_model \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 3 4 \
  --task "pick the strawberry" \
  --project \
  --camera_info_path /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation/meta/camera_info/validation_20260714_160922-png__episode_040/camera_info_color.json \
  --output_dir outputs/debug/viz_pi05_38M_split_075000
```

### 28.3 How the 220M run was visualized (adapter copied from kiwi)

The 220M run executes on `kiwi` (zfei@10.98.19.22, port 2203), whose RTX 5080
held only about 1.4 GiB free during training (14.9 / 16.3 GiB). Running the
about-5-6 GiB visualizer there would have OOM-ed the training run, so only the
PEFT adapter directory was copied to the local host and decoded there. The base
`lerobot/pi05_base` is already in the local Hugging Face cache (the local run
loads it), so no base copy was needed.

Note kiwi's repository lives directly under `~/code/`, not under
`~/code/lerobots/` like the local tree:

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
mkdir -p outputs/train/pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full/checkpoints/025000

rsync -azh -e "ssh -p 2203" \
  zfei@10.98.19.22:code/lerobot-fei-v5.0-umi-unified/outputs/train/pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full/checkpoints/025000/pretrained_model \
  outputs/train/pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full/checkpoints/025000/
```

The copied `pretrained_model/` is about 843 MiB: `adapter_model.safetensors`
(883 MiB for the 220M adapter), `adapter_config.json`, the policy
`config.json`, `train_config.json`, and the serialized UMI preprocessor /
postprocessor plus normalization stats. With `adapter_config.json` present, the
visualizer auto-detects the PEFT adapter, loads the named base, and applies it.

The visualization command is identical to section 28.2 except for the
checkpoint path and output directory:

```bash
/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/visualize_predictions.py \
  --pretrained_path outputs/train/pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full/checkpoints/025000/pretrained_model \
  --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --episode_indices 0 1 2 3 4 \
  --task "pick the strawberry" \
  --project \
  --camera_info_path /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation/meta/camera_info/validation_20260714_160922-png__episode_040/camera_info_color.json \
  --output_dir outputs/debug/viz_pi05_220M_highcap_025000
```

The two runs were executed **sequentially**, not concurrently, so that peak GPU
use stayed at training plus one visualizer rather than training plus two.

### 28.4 Decoded open-loop metrics (episodes 0-4)

Both writes are at
`outputs/debug/viz_pi05_<tag>/sroiv2_strawberry_picking_lab_validation/`:
`pred_episode_0..4.mp4` (each about 3-4 MB) and `prediction_metrics.json`.
Values are endpoint (chunk-final) errors over 285 sampled frames; lower is
better.

| Endpoint metric | 38M @75K | 220M @25K |
| --- | ---: | ---: |
| XYZ mean | **27.1 mm** | 35.2 mm |
| XYZ median | **19.8 mm** | 28.9 mm |
| Rotation mean | **3.64 deg** | 4.93 deg |
| Gripper mean | **0.218** | 0.254 |

At these unequal training stages the 38M checkpoint decodes better on all four
endpoint metrics. **This is not a capacity result**: the 38M checkpoint has
trained three times longer. Per section 24 / section 27, at matched step 25K the
two runs' masked real-dimension flow losses are nearly tied
(38M 0.039386 vs 220M 0.040136), so the spread above is dominated by training
stage, not by adapter rank. Do not conclude that 220M capacity is worse from
these numbers alone.

### 28.5 Fair (matched-sample) comparison

The two runs also share a 50K checkpoint (matched step and matched about 200K
samples seen), so the apples-to-apples capacity view is a decode of both at
50K with the section 28.2 command form, swapping in
`.../checkpoints/050000/pretrained_model` for each run. The 220M@50K adapter
was copied from kiwi exactly as in section 28.3. The result follows.

### 28.6 Matched-50K decoded comparison (capacity result)

Decoded on the same validation episodes 0-4, seed, and `--project` protocol as
section 28.4 (285 frames each); lower is better. The fair comparison is the
**38M@50K vs 220M@50K** column pair (matched step and matched samples).

| Endpoint metric | 38M @50K | 38M @75K | 220M @25K | 220M @50K |
| --- | ---: | ---: | ---: | ---: |
| XYZ mean | 30.3 mm | **27.1 mm** | 35.2 mm | 28.9 mm |
| XYZ median | **19.8 mm** | **19.8 mm** | 28.9 mm | 24.5 mm |
| Rotation mean | 3.91 deg | **3.64 deg** | 4.93 deg | 3.99 deg |
| Gripper mean | **0.22** | **0.22** | 0.25 | 0.27 |

At matched training the 5.72x-capacity 220M adapter does **not** beat the 38M
adapter: 38M is better on XYZ median (19.8 vs 24.5 mm), gripper (0.22 vs
0.27), and roughly tied on rotation; 220M is better only on XYZ mean (28.9 vs
30.3 mm), and that is a tail effect since it loses on the median. Extra
capacity is therefore not the lever here — consistent with the section 27
caution that more parameters do not guarantee better behavior.

Two trends: the 220M run is still climbing (25K->50K: 35.2->28.9 mm mean,
4.93->3.99 deg) at only 55% of training, whereas the 38M has largely plateaued
(50K->75K: about flat on median and rotation). The 220M gripper error,
uniquely, worsened 25K->50K (0.25->0.27) while its pose metrics improved.

Caveat: this is a 5-episode (285-frame) sample, not the 100-episode decoded
audit of section 22, and the gaps are within 5-episode noise. Treat it as "no
signal that 220M helps," not a precise ranking. The 38M split-rank
configuration remains the more efficient choice so far; closed-loop robot
trials remain the final selector.

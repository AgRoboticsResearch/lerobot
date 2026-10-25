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

---

# 中文翻译

> 以下为原文的完整中文翻译（AI 生成）。代码块、命令、参数、路径、URL、公式和数值保持原样；正文、标题、表格说明和列表文字译为中文。

# 统一 UMI 工作区中的 π0.5 LoRA 微调

本指南介绍在以下仓库中实现的 π0.5 LoRA 训练路径：

```text
/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
```

它特定于统一分支和草莓采摘数据集。文中解释了原始数据契约、时序查询、SE(3) 目标、归一化、流匹配损失、PEFT LoRA 配置、验证、检查点加载以及可视化。

本文档现在是权威的合并指南。它整合了已完成的 broad-LoRA 运行、OpenPI 对比，以及此前分散在另外两篇笔记中的“验证损失与物理预测”调查。

演示的配置为：

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

推荐的下一项容量实验仅将专家 LoRA 改为 rank/alpha 32/32，同时保留 VLM 16/16。这更接近 OpenPI 的分离式 LoRA 定义；详情见第 12.2 节。

## 1. 本指南使用哪种实现

本工作区包含一个改编自 OpenPI 的 π0.5 PyTorch 移植版本，并与 LeRobot 的策略、处理器、数据集、检查点、验证和 PEFT 系统集成。

这与直接在官方 [Physical Intelligence OpenPI 仓库](https://github.com/Physical-Intelligence/openpi) 中训练不同：

- 使用本工作区的 `lerobot-train` 实现。
- 使用 `--policy.type=pi05`。
- 使用 LeRobot 的 `--peft.*` 选项进行 LoRA。
- 对草莓 EE 目标使用 `--policy.use_umi_relative_ee=true`。
- 不要将 OpenPI 的 JAX 训练配置复制到本仓库。
- 不要对 UMI 数据运行普通的 `lerobot/pi05_base` 处理器。

相关的本地实现文件有：

- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/pi05/modeling_pi05.py`
- `src/lerobot/policies/pi05/processor_pi05.py`
- `src/lerobot/processor/umi_relative_ee_processor.py`
- `src/lerobot/datasets/umi_relative_ee_stats.py`
- `src/lerobot/datasets/factory.py`
- `examples/umi_relative_ee/shell_scripts/train_pi05_lora.sh`
- `examples/umi_relative_ee/visualize_predictions.py`

## 2. 始终将 Python 固定到统一源码树

`py312` Conda 环境可能以 editable 模式安装了另一个 LeRobot 检出。若未固定 `PYTHONPATH` 就运行脚本，可能会静默导入错误的分支。

从统一工作区启动命令：

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"
```

在长时间运行前确认：

```bash
/home/zfei/anaconda3/envs/py312/bin/python -c \
  'import lerobot; print(lerobot.__file__)'
```

打印出的路径必须包含：

```text
lerobot-fei-v5.0-umi-unified/src/lerobot
```

## 3. 原始数据契约

数据集保持标准 LeRobot 格式。每一帧存储一个绝对末端执行器动作：

```text
action =
    [x, y, z,
     axis_angle_x, axis_angle_y, axis_angle_z,
     gripper]
```

要求：

- `action` 的形状为 `(7,)`。
- 平移使用一致的公制单位，通常是米。
- 轴角是以弧度表示的旋转向量，而不是欧拉角。
- 夹爪方向与范围在各 episode 之间保持一致。
- 每个动作都是有限值。
- 每一帧都有非空的任务字符串。
- 与 `chunk_size=30` 配合使用的 episode 至少包含 31 个连续帧。

不需要 `observation.state` 列。在 UMI 模式下，状态由动作序列派生。

统一数据集工厂会在训练前检查原始动作形状，并拒绝不是 7D 的 UMI 数据。

### 当前实验使用的数据集

已完成的 broad 运行所使用的训练数据：

```text
repo_id: sroi/sroiv2_strawberry_picking_lab_1302_occlusion
root:    /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion
```

验证：

```text
repo_id: sroi/sroiv2_strawberry_picking_lab_validation
root:    /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
```

验证数据集包含 100 个 episode，并与训练数据集分开保存。较旧的 1125 episode 命令作为遗留启动器示例保留在第 13.3 节。

## 4. 时序查询与目标对齐

配合：

```bash
--policy.use_umi_relative_ee=true
--policy.chunk_size=30
```

`PI05Config.action_delta_indices` 返回：

```text
[-1, 0, 1, ..., 29]
```

因此，在采样时刻 `t`，数据加载器返回：

```text
action[t-1], action[t], action[t+1], ..., action[t+29]
```

UMI 派生状态处理器执行：

```text
state = [action[t-1], action[t]]
targets = [action[t], action[t+1], ..., action[t+29]]
```

这正好产生 30 个保留目标。

影响：

- `action[t]` 既是当前/基准位姿，也是第一个目标。
- 第一个相对目标近似为单位变换。
- 前一帧仅用于构建双位姿状态。
- 移除只用于状态的前导采样后，episode 末尾的填充会带入 `action_is_pad`。

此索引方式与统一的 ACT 和 SmolVLA UMI 流水线一致。

## 5. UMI SE(3) 动作目标

对于当前绝对位姿：

```text
T_base = T_absolute[t]
```

对于保留 chunk 中的每个动作 `k`：

```text
T_relative[k] = inverse(T_base) @ T_absolute[t+k]
```

等价地：

```text
R_relative[k] = transpose(R_base) @ R_target[k]

p_relative[k] =
    transpose(R_base) @ (p_target[k] - p_base)
```

chunk 中的每个目标都使用同一个 `T_base`。目标之间不链式累积：

```text
correct:   inverse(T_base) @ T_target[k]
incorrect: inverse(T_target[k-1]) @ T_target[k]
```

### 5.1 基于行的 rot6d

相对旋转矩阵使用其前两行进行编码：

```text
rot6d =
    [R00, R01, R02,
     R10, R11, R12]
```

模型的物理动作目标为：

```text
[dx, dy, dz,
 R00, R01, R02,
 R10, R11, R12,
 gripper]
```

因此：

```text
raw action shape:      [batch, 31, 7]
retained raw targets:  [batch, 30, 7]
physical model target: [batch, 30, 10]
```

不要使用逐元素轴角相减。相对朝向必须用旋转矩阵计算。

### 5.2 推理时的逆变换

后处理器：

1. 对 10D 预测进行反归一化。
2. 使用 Gram-Schmidt 从 rot6d 重建有效的相对旋转矩阵。
3. 使用预处理器缓存的基准位姿。
4. 计算：

   ```text
   T_absolute[k] = T_base @ T_relative[k]
   ```

5. 将绝对旋转转换回轴角。
6. 返回绝对 7D EE 动作，供 IK 和机器人控制使用。

缓存的 `T_base` 由整个预测 chunk 共享。

## 6. 派生的观测状态

原始状态对为：

```text
[action[t-1], action[t]]
```

两个位姿都相对于当前位姿 `action[t]` 表示：

```text
previous_relative = inverse(T_current) @ T_previous
current_relative  = inverse(T_current) @ T_current
```

每个相对位姿编码为 10D 行式 rot6d，然后将二者展平：

```text
observation.state shape = [batch, 20]
```

解释：

- 前 10 维描述从上一帧到当前帧的运动。
- 后 10 维描述当前位姿相对于自身的位姿。
- 第二个块接近单位变换，只有夹爪的绝对值例外。

在实时推理期间，状态处理器会缓存上一个绝对 7D EE 状态，以便重建相同的双位姿输入。

## 7. 统一的 π0.5 预处理顺序

当 `use_umi_relative_ee=true` 时，π0.5 预处理器按以下顺序运行：

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

顺序很重要：

```text
raw absolute -> relative SE(3) -> normalize -> tokenize
```

如果在相对变换之前对原始 7D 轴角数据进行归一化，将产生错误的目标。

后处理器按以下顺序运行：

```text
unnormalize -> relative-to-absolute SE(3) -> CPU
```

两个流水线都会序列化到每个检查点中。

## 8. 图像与语言输入

π0.5 是语言条件化的。处理器需要一个任务字符串，例如：

```text
pick the strawberry
```

状态归一化后，处理器将每个状态值离散化为 256 个 bin 之一，并构造如下形式的提示：

```text
Task: pick the strawberry, State: <20 integer bins>;
Action:
```

因此，状态是通过模型的语言 token 一侧提供给 π0.5 的，这与本工作区的 π0.5 设计一致。

图像行为：

- 数据集图像以 uint8 返回。
- 模型预处理通过带填充的缩放调整到 224×224。
- 视觉归一化模式为 `IDENTITY`；π0.5 在内部执行其预期的图像预处理。
- 不要对 π0.5 输入应用 ACT 的 ImageNet 处理器归一化。

PaliGemma 分词器是受限访问（gated）的：

```text
google/paligemma-3b-pt-224
```

训练过程使用的 Hugging Face 账号必须已接受该模型许可证并完成认证。

## 9. 模型动作维度

物理 UMI 目标是 10D，而 π0.5 使用：

```python
max_action_dim = 32
```

在流模型之前：

```text
[batch, 30, 10] -> zero-pad -> [batch, 30, 32]
```

动作输入/输出投影保留 32D 预训练模型形状。损失和采样域取决于 `flow_matching_padding_mode`：

| 模式 | 训练噪声与损失 | 推理 |
| --- | --- | --- |
| `openpi_full_width` | 32 维高斯噪声；在所有 32 个坐标上取平均损失 | 对全部 32 维积分，然后切片到 10 |
| `masked_subspace` | 填充噪声为零；损失仅在前 10 个坐标上计算 | 保持填充的流状态/速度为零，然后切片到 10 |

默认值仍为 OpenPI 全宽模式，以保持预训练兼容性。已完成的 broad-LoRA UMI 运行显式使用了 `masked_subspace`，将适配集中在物理 10 维动作空间上。

推理时，生成的动作同样会在反归一化之前切片回物理输出维度。

## 10. 流匹配目标与损失

π0.5 并不使用 ACT 风格的 L1 损失直接回归干净动作。

设：

```text
a       normalized clean action chunk
epsilon Gaussian noise with the same shape
t       sampled flow time
```

模型构造：

```text
x_t = t * epsilon + (1 - t) * a
u_t = epsilon - a
```

动作专家预测速度场：

```text
v_theta(x_t, observation, task, t)
```

逐维目标为：

```text
(v_theta - u_t)^2
```

策略使用 `action_is_pad` 屏蔽无效的 episode 边界时间步，并记录 `flow_loss_real_dims` 与 `flow_loss_padded_dims`。在全宽模式下，标量对 10 个真实坐标和 22 个填充坐标取平均；在 masked-subspace 模式下，仅对 10 个真实坐标取平均。

没有内置的平移与旋转权重，也没有平滑项。归一化会将各维度放到大致可比较的尺度上，但这仍然是归一化速度场的 MSE——不是毫米、SO(3) 角度、端点误差或任务成功率。第 22 节给出了实测后果。

## 11. 分位数归一化

统一 π0.5 的默认值为：

```text
VISUAL: IDENTITY
STATE:  QUANTILES
ACTION: QUANTILES
```

对于状态和动作：

```text
x_normalized =
    2 * (x - q01) / (q99 - q01) - 1
```

这会映射：

```text
q01 -> -1
q99 -> +1
```

`[q01, q99]` 区间之外的值不会被裁剪，因此离群点的归一化绝对值可能大于 1。

当某个维度的 `q99-q01` 跨度实际上为零时，归一化器会使用安全分母，而不是除以零。

### 11.1 统计量在变换后的 UMI 空间中计算

对于训练数据集，`make_dataset()` 调用：

```text
compute_umi_relative_ee_stats(...)
```

该函数：

1. 查找每一个有效的连续 31 帧查询。
2. 使用第二帧作为 chunk 基准。
3. 将全部 30 个保留动作转换为相对 10D rot6d。
4. 将 `[previous,current]` 转换为相对 20D 状态。
5. 在这些精确张量上计算运行分位数统计量。
6. 将变换后的统计量添加到内存中的 `dataset.meta.stats`。

原始数据集磁盘上的动作和原始统计量不会被重写。

这与通用 π0.5 数据集有一个重大区别：不需要离线转换为第二个 10D 数据集。

### 11.2 验证使用训练统计量

对于验证：

- 验证数据集接收 UMI 特征形状和时序查询。
- 不会从验证数据重新计算 UMI 统计量。
- 保存的/训练用的预处理器使用训练数据集的状态/动作统计量。

这避免了归一化泄漏。

### 11.3 需要检查的旋转统计量

小的相对旋转接近于：

```text
[R00, R01, R02, R10, R11, R12]
    approximately
[1,   0,   0,   0,   1,   0]
```

因此 `R00` 和 `R11` 的 `q99-q01` 跨度可能很窄。在长时间运行之前，检查每个动作和状态维度：

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

还要验证所有值都是有限值。

在第一次 ACT 与 π0.5 的对比中，保留相同的 10D rot6d 目标。只有在建立可比基线之后，才改为 3D 旋转向量目标。

## 12. 本工作区的 LoRA 实现

LoRA 通过 LeRobot 的 PyTorch PEFT 集成应用。旧的本地默认配置范围很窄，仅训练约 1.29M 参数。它针对动作专家的 q/v 和动作投影，并且此前命名了不存在的 `action_time_mlp_in/out` 模块。默认命名错误已修正为 `time_mlp_in/out`，但默认配置仍有意保持为低内存的窄适配器。

已完成的 broad 运行覆盖了默认配置：

```bash
--peft.method_type=LORA
--peft.r=16
--peft.lora_alpha=16
--peft.target_modules='.*\.(paligemma|gemma_expert)\..*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
--peft.full_training_modules='["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]'
```

在 rank 16、alpha 16 下：

```text
LoRA scaling = alpha / rank = 1
```

这会在 PaliGemma 语言模型和 Gemma 动作专家中同时适配 q/k/v/o 以及 gate/up/down。四个动作/时间投影完全训练，并与适配器一起保存。由此得到的运行训练了 31,693,856 个参数。

基础策略从以下模型初始化：

```text
lerobot/pi05_base
```

PEFT 包装器会冻结非目标的基础权重，并训练插入的 LoRA 参数。在启动完整运行之前，务必检查打印出的可训练参数数量。

### 12.1 公开的 π0.5 微调配方

目前尚无一个社区标准的 π0.5 配方。已公开的命令还混合了两种实质不同的实现：

- OpenPI 的 JAX LoRA 在 PaliGemma VLM 和 Gemma 动作专家两者中，将适配器插入注意力和前馈层。
- 本工作区的**默认** LeRobot PEFT 目标仍是窄的动作专家 q/v 模式；演示的启动器用类似 OpenPI 的广度注意力+FFN 范围覆盖了它。

因此，已完成的本地 broad 运行在模块范围上与 OpenPI 可比，但在 rank 上不可比：它使用单一全局 rank 16，而不是 VLM 16 / 专家 32。

公开记录的配置包括：

| 来源 | 微调范围 | Rank | Batch | 学习率调度 | Steps |
|---|---|---:|---:|---|---:|
| [官方 OpenPI LIBERO 配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) | 完整模型 | N/A | 256 | 峰值 `5e-5`，warmup 10k | 30k |
| [OpenPI 社区 π0.5 LoRA 配置](https://github.com/Physical-Intelligence/openpi/issues/672) | VLM 注意力/FFN 加专家注意力/FFN | VLM 16，专家 32 | 32 | 峰值 `5e-5`，warmup 10k | 30k |
| [真实 Franka 罐子任务](https://huggingface.co/IDEAS-Lab-Northwestern/pi05-real-jar-60-droid-refined-lora) | OpenPI 双适配器，60 条演示 | VLM 16，专家 32 | 4 | 未报告 | 20k |
| [仿真多任务抓取](https://huggingface.co/IDEAS-Lab-Northwestern/pi05-sim-pnp-multitask-3cam-libero-lora) | OpenPI 双适配器 | VLM 16，专家 32 | 4 | `2.5e-5` 到 `2.5e-6`，warmup 1k | 50k + 30k |
| [LeRobot SO-101 袜子任务](https://huggingface.co/RyuRobot/pi05_sock_in_bowl_lora_July_10_2026) | 窄的 LeRobot 默认风格目标模式 | 32 | 32 | `2.5e-4` 到 `2.5e-5`，warmup 1k | 15k |
| [公开的 LeRobot rank-16 运行](https://huggingface.co/Tna001/pi05_lora_r16_lr3e4/blob/main/train_config.json) | LeRobot PEFT | 16 | 16 | `3e-4` 到 `2.5e-6`，warmup 500 | 20k |

其他官方参考点：

- [LeRobot π0.5 指南](https://huggingface.co/docs/lerobot/pi05) 展示了完整微调 3k 步，batch 32、bfloat16 和梯度检查点。
- [LeRobot LIBERO 复现](https://huggingface.co/docs/lerobot/libero) 从 `pi05_libero` 开始，在八张 H100 GPU 上以全局 batch 256 再训练 6k 步。
- OpenPI 估计其 JAX LoRA 配方需要超过 22.5 GB，完整微调需要超过 70 GB。这些估计并不直接适用于本工作区的 PyTorch PEFT 实现；见 [OpenPI README](https://github.com/Physical-Intelligence/openpi)。

公开的原生 LeRobot rank-32 适配器特别适合用于对比。其[保存的适配器配置](https://huggingface.co/RyuRobot/pi05_sock_in_bowl_lora_July_10_2026/blob/main/adapter_config.json)使用了与本工作区相同的目标正则表达式。随附的模型卡报告约 2.5M 可训练参数。该适配器使用 `r=32`、`alpha=8`，因此其 LoRA scale 为 `alpha / rank = 0.25`。模型卡尚未报告真实机器人评估，因此应将其视为可复现的训练示例，而不是这些超参数最优的证据。

Hugging Face Discord 历史记录需要登录才能访问，在准备本对比时未被公开索引。不要在没有稳定公开消息或可独立复现配置的情况下引用所谓的 Discord 共识。

### 12.2 推荐的更高容量 UMI 配置

OpenPI 定义了 [`gemma_2b_lora`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py)，其注意力和 FFN 的 rank/alpha 为 16/16；以及 `gemma_300m_lora`，其注意力和 FFN 的 rank/alpha 为 32/32。已完成的本地运行匹配了官方模块范围，但使用单一全局 rank 16，动作专家中也是如此。因此，最清晰的下一步容量提升是：

| 组件 | Rank | Alpha | 目标模块 |
| --- | ---: | ---: | --- |
| PaliGemma/VLM 语言层 | 16 | 16 | q/k/v/o + gate/up/down |
| Gemma 动作专家 | 32 | 32 | q/k/v/o + gate/up/down |
| 动作/时间投影 | full | N/A | `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out` |
| 视觉编码器 | 初始冻结 | N/A | 无 |

在这块 24 GB RTX 4090 上推荐的训练参数：

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

在 batch 4 下，75K 步处理 300K 个样本（约 2.47 个数据集轮次），100K 步处理 400K 个样本（约 3.30 个轮次）。不要假定最后一步就是最好的：在每个检查点都运行完整的 100 episode 解码审计。当前证据支持至少继续到 75K；100K 是评估上限，而不是部署最后检查点的指令。

本地 `PeftConfig` 现在暴露了 PEFT 的 `rank_pattern` 和 `alpha_pattern`，因此可以直接表达这种分离。全局 16/16 值应用于所有 broad 目标，而专家模式会覆盖匹配的 Gemma 专家模块：

```bash
--peft.r=16
--peft.lora_alpha=16
--peft.rank_pattern="{'.*\\.gemma_expert\\..*': 32}"
--peft.alpha_pattern="{'.*\\.gemma_expert\\..*': 32}"
```

这比全局 32/32 更可取，因为它增加了动作专家的容量，同时不必无谓地将 VLM 适配器 rank 加倍。

不要把解冻完整视觉编码器作为下一步。如果分离 rank 的运行仍主要因未见过的光照、遮挡或草莓外观而失败，再测试最后几个视觉块的低秩适配，或仅解冻最后一到两个块。完整模型微调不是实用的 24 GB 配方。

### 12.3 检查点内容

PEFT 检查点存储：

- 适配器权重。
- `adapter_config.json`，包括基础模型引用。
- π0.5 策略 `config.json`。
- `train_config.json`。
- 序列化的 UMI 预处理器和后处理器。
- 归一化统计量张量文件。
- 优化器、调度器、RNG 和训练步状态。

不要只复制适配器权重。保存的处理器及其 UMI 统计量是训练后策略契约的一部分。

## 13. 训练命令

### 13.1 已演示的 broad-LoRA 1302 运行

已完成的运行使用当前的 1302 遮挡数据集和 batch 4：

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
setsid nohup bash examples/umi_relative_ee/shell_scripts/run_pi05_broad_lora_umi.sh 4 \
  > examples/umi_relative_ee/logs/pi05_broad_lora_masked_bs4.log 2>&1 < /dev/null &
```

启动器显式设置 broad 目标、masked-subspace 流、按 batch 缩放的步数、验证和检查点频率。batch 4 运行 75K 步；batch 2 运行 150K 步，因此两者都处理 300K 个样本。不要同时运行两者。

### 13.2 更高容量的分离 rank 运行

OpenPI 风格的启动器使用 PaliGemma rank/alpha 16/16 和 Gemma 专家 32/32：

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
bash examples/umi_relative_ee/shell_scripts/run_pi05_openpi_split_lora_umi.sh 4
```

batch 4 运行 100K 步，每 12.5K 保存一次。batch 2 仅作为内存回退；它运行 200K 步，因此两种设置都处理 400K 个样本。实时实验和输出路径记录在第 24 节。

### 13.3 遗留的 1125 episode 窄启动器

该 shell 启动器默认使用较旧的数据集名称，因此需要为 1125 episode 实验覆盖它：

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

该启动器目前使用：

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

24 GB 的起始配置是 batch size 2。如果内存不足，降到 batch size 1。当前训练器没有暴露梯度累积。

## 14. 验证语义与检查点选择

启动器提供：

```bash
--validation_dataset.repo_id=...
--validation_dataset.root=...
--val_freq=10000
```

在验证事件发生时，训练器会：

1. 重置处理器状态。
2. 确定性地评估选定的验证数据集。
3. 禁用梯度。
4. 计算样本加权损失。
5. 在 `val/` 下记录。
6. 恢复训练模式。
7. 再次重置处理器状态。

验证使用与训练相同的流匹配损失实现。它将 RNG 重置为 seed 0，以便进行确定性评估。原始损失只有在相同的填充目标下才可比较。在全宽模式下，22 个简单的填充坐标会稀释标量；使用 `flow_loss_real_dims` 作为谨慎的跨模式诊断。

即使在同一种模式下，流损失也不足以用于检查点选择。它评分的是所有 chunk 时间步上的归一化速度 MSE，而部署时要对该场积分，并关心物理 XYZ、SO(3)、夹爪和平滑度。在每个保存的检查点上，对完整验证集运行 `eval_open_loop_dataset.py`。第 22 节记录了观察到的不一致。

完整的 100 episode 验证集有 9,274 个原始帧，对 π0.5 来说开销很大。对于冒烟测试，使用一个小的显式 episode 子集。检查点选择应使用完整集合。

## 15. 必需的冒烟测试

在启动长时间运行之前，验证以下内容。

### 15.1 导入位置

```bash
python -c 'import lerobot; print(lerobot.__file__)'
```

它必须解析到统一工作区。

### 15.2 一个已处理批次

在预处理后检查一个批次：

```text
raw action query        [B, 31, 7]
derived state pair      [B, 2, 7]
retained action         [B, 30, 7]
relative state          [B, 20]
relative action         [B, 30, 10]
model-padded action     [B, 30, 32]
```

确认：

- 第一个保留目标在位姿上接近单位变换。
- 前六个 rot6d 值遵循行式约定。
- 夹爪保持绝对值。
- 任务 token 存在。
- 图像键与检查点策略配置匹配。

### 15.3 SE(3) 往返

对于采样的动作：

```text
absolute 7D
    -> relative 10D
    -> absolute 7D
```

测量：

- 平移重建误差。
- 以度表示的旋转测地误差。
- 夹爪重建误差。

不要将直接的轴角向量差作为主要的朝向度量。

### 15.4 归一化审计

检查完整变换后训练集的 q01/q99 跨度和归一化尾部。特别关注几乎恒定的旋转对角线通道，以及 20D 状态中的单位变换那一半。

### 15.5 短程 LoRA 运行

运行 10–100 步并确认：

- 训练损失是有限值。
- 每个 `loss_per_dim` 值都是有限值。
- 至少有一个 LoRA 参数具有非零梯度。
- 冻结的基础参数没有变化。
- 检查点可以保存并重新加载。
- 保存的预处理器仍包含 UMI 步骤和变换后的统计量。

## 16. 恢复训练

从保存的训练配置恢复：

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
export PYTHONPATH="$PWD/src"

/home/zfei/anaconda3/envs/py312/bin/python \
  examples/umi_relative_ee/train_pi05_lora.py \
  --config_path=outputs/train/pi05_lora_umi_relative_ee_1125train_100val/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

恢复会还原：

- 策略和 LoRA 适配器。
- UMI 处理器配置。
- 归一化统计量。
- 优化器。
- 调度器。
- RNG 状态。
- 训练步。

不要通过针对现有输出目录重新运行初始启动器来恢复。

## 17. 可视化

使用统一的可视化器，并显式固定统一源码：

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

该脚本会自动检测 PEFT 检查点：

1. 读取 `adapter_config.json`。
2. 使用微调后的策略配置加载所引用的基础 π0.5 策略。
3. 应用 LoRA 适配器。
4. 加载保存的 UMI 处理器。
5. 运行录制观测的离线（open-loop）推理。

`config=policy_config` 加载步骤很重要：它会保留微调后的相机特征名称，而不是退回基础模型的默认图像键。

每个输出视频包含：

- 相机图像。
- 使用 `--project` 投影到图像上的预测与 GT 夹爪尖端轨迹。
- 预测与 GT 3D 轨迹。
- 逐维动作曲线。
- 逐帧端点误差。

## 18. 部署契约

在每个机器人控制 tick：

1. 从 FK 获取当前绝对 7D EE 位姿。
2. 运行预处理器，使其更新两帧状态历史并缓存当前 chunk 基准。
3. 包含任务字符串。
4. 预测归一化的相对动作 chunk。
5. 对整个 chunk 运行后处理器。
6. 获得绝对 7D EE 目标。
7. 通过 IK 转换目标。
8. 向机器人发送安全的关节命令。

执行预测 chunk 时，每个相对目标都必须与同一个 chunk 起始基准复合。不要累积预测的相对变换。

## 19. 常见错误

- 从 `py312` 环境导入了不同的 LeRobot 检出。
- 忘记 `--policy.use_umi_relative_ee=true`。
- 使用原始数据集的 q01/q99 统计量，而不是变换后的 UMI 统计量。
- 在验证数据上重新计算归一化统计量。
- 将轴角相减当作相对旋转。
- 使用基于列的 rot6d，而不是本工作区的基于行约定。
- 链式累积相对动作，而不是使用一个共享基准。
- 因为 `action[t]` 接近单位变换就将其移除；它有意作为第一个保留目标。
- 对 π0.5 应用 ACT 的视觉归一化。
- 省略任务字符串。
- 只复制 LoRA 适配器文件而不复制保存的处理器。
- 在没有微调策略配置的情况下加载 LoRA 基础模型。
- 将 π0.5 流损失与 ACT L1/KL 验证损失进行数值比较。
- 在检查 q01/q99 跨度和归一化旋转尾部之前启动完整运行。

## 20. 推荐的实验顺序

1. 验证统一导入路径。
2. 运行现有的 UMI 流水线正确性测试。
3. 检查一个已处理的 π0.5 批次。
4. 审计变换后的 q01/q99 统计量。
5. 运行 10–100 步 LoRA 冒烟测试。
6. 保存、重新加载并可视化冒烟测试检查点。
7. 运行几千步，并将预测与 ACT 对比。
8. 基于解码后的全 episode 指标继续，而不是仅看原始流损失。
9. 比较位姿 chunk 均值、端点、夹爪和 chunk 内加速度。
10. 在闭环机器人试验中确认选定的检查点。

## 21. 已完成的 broad-LoRA 运行

此前的窄全宽运行训练了 1,287,168 个参数，并在保留其 400K 检查点后停止。其默认目标主要覆盖动作专家的 q/v 和动作投影；当时损坏的时间 MLP 目标名称没有匹配任何模块。

替代配置为：

| 设置 | Broad 运行 |
| --- | --- |
| 数据集 | `sroiv2_strawberry_picking_lab_1302_occlusion` |
| 基础模型 | `lerobot/pi05_base` |
| 真实/模型动作宽度 | 10 / 32 |
| Padding flow | `masked_subspace` |
| LoRA 范围 | VLM 语言层和动作专家中的 q/k/v/o 与 gate/up/down |
| 完全训练的模块 | 动作输入/输出和时间 MLP 输入/输出 |
| 视觉编码器 | 冻结 |
| Rank / alpha | 全局 16 / 16 |
| 峰值 / 最终 LR | `5e-5` / `5e-6` |
| Batch / steps | 4 / 75K |
| 可学习 / 总参数 | 31,693,856 / 4,175,098,672 |
| 观察到的 GPU 内存 | 约 12,220 / 24,564 MiB |
| 吞吐量 | 约 1.7-1.8 steps/s |

运行产物：

```text
output: outputs/train/pi05_broad_lora_masked_1302_bs4
log:    examples/umi_relative_ee/logs/pi05_broad_lora_masked_bs4.log
W&B:    biorobotlab/lerobot/9erobpvd
```

该运行完成了 75K 步和 300K 个样本。保存的检查点为 12.5K、25K、37.5K、50K、62.5K 和 75K。

### 与官方 LeRobot 示例的区别

LeRobot 指南中的示例是完整模型微调：视觉编码器、VLM、动作专家和投影全部可训练。它使用 batch 32 和 3K 步（96K 个示例）、compile 以及梯度检查点。本次 broad-LoRA 运行是 24 GB 的任务适配配方：冻结视觉、LoRA 变换器权重、完整动作/时间投影、batch 4，以及 UMI 专用的 10 维 masked flow。

### 与官方 OpenPI LoRA 的区别

OpenPI 检入的 LoRA 模型定义将适配器应用于注意力和 FFN，VLM rank/alpha 为 16/16，专家为 32/32。本地已完成的运行匹配范围和 VLM rank，但使用专家 16/16。OpenPI 目前没有提供一个规范的、已检入的 π0.5 LoRA 训练预设；其显式的低内存 LoRA 预设是针对 π0 的，而已检入的 π0.5 LIBERO 预设是完整微调。

## 22. 为什么验证流损失与预测质量不一致

全宽和 masked-subspace 运行的原始验证损失不可比较。在旧的窄模型 200K 时：

```text
(10 * 0.038027 real-dim loss + 22 * 0.000191 padded-dim loss) / 32
= 0.012015 reported loss
```

Broad 25K 报告 `0.037607`，因为它仅对 10 个真实维度取平均。因此修正后的比较是 0.038027 对 0.037607，而不是 0.012015 对 0.037607。

| 检查点 | Padding | 原始验证损失 | 真实维度损失 | 填充损失 |
| --- | --- | ---: | ---: | ---: |
| narrow 200K | 全宽 | **0.012015** | 0.038027 | 0.000191 |
| broad 25K | masked | 0.037607 | **0.037607** | 0 |
| broad 75K | masked | 0.052667 | 0.052667 | 0 |

即使是真实维度代理指标也与部署行为不一致。原因：

1. 它测量的是采样流点处的归一化速度，而不是十步 Euler 积分后的动作。
2. 它对原始 XYZ、rot6d 和夹爪坐标取平均，而不是以米和 SO(3) 测地角度量。
3. 它对全部 30 个时间步取平均，且没有端点或平滑项。
4. 全宽和 masked 模型以不同的填充流状态为条件。
5. 分组的 XYZ/旋转/夹爪权衡被隐藏在一个 10 维均值中。
6. Broad LoRA 的可训练容量约为旧适配器的 25 倍。

对两个检查点的处理器文件做了 diff：变换、归一化模式和统计量完全相同；只有运行时缓存 ID 不同。改善不是由归一化不匹配造成的。

### 完整 100 episode 物理审计

全部 100 个验证 episode 在五个均匀间隔的非填充查询帧上、使用 seed 1000 和检查点保存的处理流程进行评估。数值按 episode 均衡，越低越好。

| 指标 | narrow 200K | broad 25K | broad 75K |
| --- | ---: | ---: | ---: |
| 旋转 chunk 均值 | 3.077° | 2.749° | **2.516°** |
| 旋转端点 | 5.584° | 4.846° | **4.500°** |
| XYZ chunk 均值 | 16.75 mm | 14.73 mm | **13.63 mm** |
| XYZ 端点 | 31.20 mm | 25.02 mm | **23.47 mm** |
| 夹爪 chunk 均值 | 0.1060 | **0.0996** | 0.1032 |
| 夹爪端点 | 0.1717 | **0.1580** | 0.1655 |
| 旋转加速度代理 | 0.1013° | 0.0880° | **0.0729°** |
| XYZ 加速度代理 | 0.563 mm | 0.476 mm | **0.382 mm** |

Broad 75K 相对旧的 200K 在旋转端点改善约 19%，XYZ 端点改善 25%，旋转加速度改善 28%，XYZ 加速度改善 32%。Broad 25K 是偏向夹爪的替代方案。此前认为 75K 只是过拟合的解释来自流损失和三个 episode；完整物理审计并不支持这一点。

报告：

- `outputs/debug/compare_pi05_loss_vs_physical/all100_old_fullwidth_200k/pi05_lora_openpi_fullwidth_1302_1M_0200000_open_loop_metrics.json`
- `outputs/debug/compare_pi05_loss_vs_physical/all100_broad_masked_25k/pi05_broad_lora_masked_1302_bs4_025000_open_loop_metrics.json`
- `outputs/debug/compare_pi05_loss_vs_physical/all100_broad_masked_75k/pi05_broad_lora_masked_1302_bs4_075000_open_loop_metrics.json`

对于本次运行，优先测试 broad 75K 的位姿/平滑度，并将 broad 25K 作为偏向夹爪的替代方案。真实机器人采摘仍是最终选择器。

## 23. 相关的统一文档

- `examples/umi_relative_ee/doc/README.md`
- `examples/umi_relative_ee/doc/prediction_visualization.md`
- `examples/umi_relative_ee/doc/visualize_predictions_pi05.md`
- `examples/umi_relative_ee/shell_scripts/train_pi05_lora.sh`
- `examples/umi_relative_ee/shell_scripts/run_pi05_broad_lora_umi.sh`
- `examples/umi_relative_ee/shell_scripts/run_pi05_openpi_split_lora_umi.sh`
- `docs/source/pi05.mdx`

## 24. 已完成的更高容量分离 rank 运行

于 2026-08-03 16:52（Asia/Taipei）在 tmux 窗口 `0:pi05-split` 中启动：

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

记录的 PEFT 配置包含全局 16/16 以及专家 `rank_pattern`/`alpha_pattern` 32/32。与已完成的 broad 全局 rank-16 运行相比，可训练参数从 31,693,856 增加到 38,624,288。在第 50 步，运行报告 loss 0.467、梯度范数 2.446；在第 200 步报告 loss 0.354、梯度范数 1.814。没有出现内存不足错误。因此 batch 4 在一次前向/反向传播后仍可剩余约 11.8 GiB 空闲内存。

计划的检查点为 12.5K、25K、37.5K、50K、62.5K、75K、87.5K 和 100K。对每个有用的检查点应用相同的 100 episode 解码审计；不要仅仅因为 100K 是最后一个就选择它。

2026-08-03 22:17（Asia/Taipei）的进度快照：

```text
step                 30,588 / 100,000
recent throughput    about 1.77 steps/s
recent train loss    about 0.03-0.04
saved checkpoints    12.5K and 25K (436 MiB each)
GPU                   11,947 MiB, 92% utilization, 86 C, 375 W
estimated completion  about 2026-08-04 10:40 Asia/Taipei
```

| 验证步 | Masked 真实维度流损失 |
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

该运行于 2026-08-04 10:37（Asia/Taipei）完成全部 100,000 步（墙钟时间 17h43m，平均约 1.79 steps/s），处理 400K 个样本（3.30 个 epoch）。最佳 masked flow 验证出现在 20K（**0.038474**）；此后单调上升到 100K 的 0.059480，增幅 55%——与此前 broad-LoRA 运行相同的流损失过拟合形态。正如第 22 节所确立的，流损失上升本身并不表示解码行为变差：5 episode 解码检查（第 28.6 节）显示 38M@75K 在平均端点误差上仅略优于 38M@50K，在中位数上持平。不要从该表选择 100K；第 22 节的物理审计和闭环试验仍是选择依据。全部八个计划检查点（12.5K-100K）加上 `last` 均已保存。

## 25. 单 GPU 常识：RTX 4090 与 RTX 5090

这两张卡通常都应被视为 π0.5 **LoRA** 设备，而不是完整模型微调设备。RTX 4090 有 24 GB，RTX 5090 有 32 GB，而 OpenPI 估计其原生 LoRA 路径需要超过 22.5 GB，完整微调需要超过 70 GB。OpenPI JAX 与本工作区的 PyTorch PEFT 实现之间，确切内存占用有所不同。

| 设置 | RTX 4090 起点 | RTX 5090 起点 |
| --- | --- | --- |
| 微调方法 | broad 或混合 rank LoRA | 同样先使用混合 rank LoRA |
| VLM / 专家 rank | 16 / 16–32 | 16 / 32；仅在消融后增加 |
| 视觉编码器 | 冻结 | 初始冻结 |
| 动作/时间投影 | 完全训练 | 完全训练 |
| 精度 | bfloat16 | bfloat16 |
| 梯度检查点 | 启用 | 初始启用 |
| 初始 batch | 2–4 | 4–8 |
| 完整模型微调 | 不实用 | 在 32 GB 上仍不实用 |

更多显存首先应换取稳定性、batch size、相机和受控消融——而不是盲目提高 rank 或解冻全部视觉。如果使用 5090，在首次硬件对比中保持 38.62M 参数配置，只有在测量内存后才增加 batch。增加 batch 时通过减少优化器步数来保持相同的已见样本数；同时改变 rank、batch、学习率和样本预算会妨碍干净的对比。

只有在解码误差表明存在视觉域差距后，才适配后期视觉块，例如特定于光照、遮挡、相机位置或物体外观的失败。对于动作几何和 chunk 内行为，应优先考虑动作专家、投影、目标表示、归一化和物理检查点指标。

### 为什么 OpenPI 说超过 22.5 GB，而本次运行只使用 12.34 GiB

这些测量不是同一个配方。OpenPI 检入的低内存 LoRA 示例目前是 π0 JAX 配置，并继承全局 batch 32。OpenPI 将 JAX 训练记录为 float32 权重和梯度，计算大多使用 bfloat16，并建议允许 XLA 预留 90% 的 GPU 内存。因此其超过 22.5 GB 的数字是该技术栈的容量需求，而不是每个正确的 LoRA 实现都必须消耗的最小值。

本次运行使用 LeRobot PyTorch PEFT，batch 4，大多数语言/动作基础权重采用 bfloat16 策略路径，基础模型和视觉编码器冻结，并启用梯度检查点。一些数值敏感或适配器参数可能仍为 float32，但这仍与 JAX 的 float32 基础权重有实质不同。因此，即使它训练了一个广度的 38.62M 参数适配器，反向传播后实测的 12,339 MiB 也是合理的。更低的内存并不意味着只匹配了旧的窄模块：记录的可训练数量从全局 rank 16 的 31,693,856 增加到专家 rank 32 的 38,624,288，正好是预期的 6,930,432 参数增量。

## 26. 在两块 4090 主机上的全参数 batch-1 试验

2026-08-03，在 `zfei@10.98.19.22:2202` 上准备了仓库和 UMI 资产，以测试全参数 π0.5 训练能否在 batch 1 下通过一次真实的优化器步。这是一次经验性内存试验，并不声称两块 RTX 4090 满足 OpenPI 超过 70 GB 的建议。

远程布局：

```text
repository  /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
train data  /home/zfei/data/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion
validation  /home/zfei/data/lerobot/sroiv2_strawberry_picking_lab_validation
base model  /home/zfei/.cache/huggingface/hub/models--lerobot--pi05_base
Python      /home/zfei/anaconda3/envs/py312/bin/python
```

该主机有两块 24,564-MiB 的 RTX 4090 GPU 和 62 GiB 系统内存。代码、两个数据集以及本地 Hugging Face π0.5 缓存都是复制过去的，因为远程 Hugging Face 连接超时。Python 3.12 环境也是从工作机器复制的，因为远程软件包镜像下载大型训练 wheel 太慢，无法实际使用。

启动器为 `examples/umi_relative_ee/shell_scripts/run_pi05_full_finetune_umi_bs1.sh`。它有意省略 `--peft`，同时设置 `freeze_vision_encoder=false` 和 `train_expert_only=false`，使用 bfloat16 加梯度检查点、batch 1、30 步 masked-subspace 动作，以及与 LoRA 对比相同的 5e-5 到 5e-6 学习率调度。100K 配置只在结束时保存，因为完整模型加 AdamW 状态每个检查点可能消耗几十 GB。

在启动 100K 步之前，必须通过的检查是：

```bash
cd /home/zfei/code/lerobots/lerobot-fei-v5.0-umi-unified
TRAIN_STEPS=2 WARMUP_STEPS=1 SAVE_CHECKPOINT=false VAL_FREQ=0 \
WANDB_ENABLE=false RUN_SUFFIX=smoke CUDA_DEVICE=0 \
bash examples/umi_relative_ee/shell_scripts/run_pi05_full_finetune_umi_bs1.sh
```

batch 1 减少了激活，但不会实质减少参数、梯度或优化器状态内存。普通的两 GPU DDP 不是解决方案：它会在每块 GPU 上复制完整模型和优化器。要用两块卡获得内存，需要明确测试过的分片策略，例如 FSDP 或 ZeRO；48 GB 聚合显存仍低于 OpenPI 的完整微调估计。

### 实测结果

单 GPU 冒烟测试加载了全部 812 个检查点键，并报告 4,143,404,816 个可学习参数，总共也是 4,143,404,816 个。前向和反向传播完成，但第一次 `optimizer.step()` 在 PyTorch Adam 分配 `exp_avg_sq` 时失败：

```text
GPU 0 capacity       23.52 GiB
process usage        23.46 GiB
PyTorch allocated    22.08 GiB
PyTorch reserved      0.93 GiB
failed allocation    20.00 MiB
completed steps       0
```

这是决定性的：batch 1 能容纳激活和梯度，但容纳不了 Adam 状态。不要在 24-GB 卡上启动未分片的 100K 配置。

还进行了两项有界分片测试。Accelerate FSDP2 找到并包装了自定义的 `_PiGemmaDecoderLayerBase` 和 `SiglipEncoderLayer` 类，但在视觉 patch 卷积中失败，因为自定义模型将常规输入张量与分片的 DTensor 混用。FSDP1 在包装期间失败，因为每个自定义 transformer 层同时包含 bfloat16 和 float32 参数，无法合并为一个扁平参数。普通 DDP 会复制单 GPU 的失败。没有启动长时间的全参数运行。

远程主机上的日志：

```text
examples/umi_relative_ee/logs/pi05_full_finetune_masked_1302_bs1_smoke2.log
examples/umi_relative_ee/logs/pi05_full_finetune_masked_1302_bs1_fsdp2_smoke2.log
examples/umi_relative_ee/logs/pi05_full_finetune_masked_1302_bs1_fsdp1_smoke.log
```

### 这是否遵循官方 LeRobot 示例？

它遵循完整模型微调的官方定义，但针对本次 UMI 实验调整了任务和运行时设置。

| 设置 | 官方文档示例 | 本次 UMI 试验 | 原因 |
| --- | --- | --- | --- |
| 基础模型 | `lerobot/pi05_base` | 相同 | 相同初始化 |
| PEFT | 无 | 无 | 全参数 |
| 视觉冻结 | false | false | 训练视觉 |
| 仅专家 | false | false | 训练 VLM 和专家 |
| 精度 | bfloat16 | 相同 | 减少内存 |
| 梯度检查点 | true | 相同 | 减少激活内存 |
| Compile | true | false | 简化内存诊断和分片兼容性 |
| Batch | 32 | 每 GPU 1 | 硬件限制 |
| Steps | 3,000 | 计划 100,000 | 现有 1,302 episode 实验预算 |
| 动作目标 | 通用策略默认值 | UMI 相对 EE，10 维 rot6d | 体现形态特定几何 |
| Chunk | 默认 50 | 30 | UMI 控制时域 |
| Padding 损失 | 默认全宽 | masked 10 维子空间 | 不要把损失花在 22 个填充坐标上 |
| 学习率 | 策略默认 | 5e-5 到 5e-6 cosine | 匹配此实验系列 |

官方命令中的 batch 32 是一个应根据可用硬件调整的示例；它并不承诺完整微调能装进消费级 GPU。对于这个确切的 LeRobot 混合 dtype 实现，观察到的优化器前使用量加上剩余的 Adam 动量表明，batch 1 下大约需要 45–55 GiB，但分配器、检查点和实现细节使这只是一个规划估计。使用 80 GB 级别的 GPU 才能可靠运行，这与 OpenPI 超过 70 GB 的指导一致。将 48 GB 视为实验性，而非保证。24 或 32 GB 的卡应使用 LoRA，除非添加并验证了兼容的分片/卸载实现。

## 27. 在 kiwi 的 RTX 5080 上进行高容量 LoRA

2026-08-04，在 `kiwi` 上启动了第二次 LoRA 运行，且未中断本地的 rank-16/32 运行。Kiwi 有一块 RTX 5080，显存为 16,303 MiB。所选配置在 batch 4 下使用全局 rank/alpha 96/96 和动作专家 rank/alpha 192/192。

| 配置 | 本地分离 rank 基线 | Kiwi 高容量运行 |
| --- | ---: | ---: |
| 视觉/VLM rank | 16 | 96 |
| 动作专家 rank | 32 | 192 |
| 可训练参数 | 38,624,288 | 220,916,768 |
| 相对容量 | 1.00× | 5.72× |
| Batch size | 4 | 4 |
| 稳定 GPU 内存 | RTX 4090 上约 12.34 GiB | 14,994 / 16,303 MiB |
| 稳定吞吐量 | 约 1.75 steps/s | 约 1.27 steps/s |

kiwi 的可训练参数明细为：

```text
vision LoRA          17,915,904
VLM language LoRA   117,669,888
action-expert LoRA   83,165,184
full projections      2,165,792
total               220,916,768
```

`freeze_vision_encoder=true` 会冻结预训练视觉权重，并保持视觉塔处于评估模式。PEFT 在之后应用，因此匹配的视觉注意力 q/k/v 投影上的 LoRA 权重仍可训练。PaliGemma 语言模型和动作专家适配注意力和 FFN 模块；动作/时间输入和输出投影完全训练。

在长时间运行之前，rank 96/192 通过了两步优化器冒烟测试和 50 步稳态扫描。扫描保持 14,994 MiB，以约 1.20–1.29 steps/s 训练，结束且无 OOM。这使用了约 92% 的显卡，同时保留约 1.28 GiB 用于分配器变化和验证。未选择 rank 112/224，因为其估计余量太小，不适合可靠的过夜任务。

活动运行：

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

2026-08-04 07:17（Asia/Taipei），运行状态健康，刚刚超过第 26,000 步。最近的训练损失约为 0.030--0.041。第 25,000 步的验证在真实动作维度上为 0.040136（`flow_loss_padded_dims=0`），25,000 检查点已写入。它使用约 14,940 MiB，GPU 利用率 98%，在验证/检查点暂停之间以约 1.27 steps/s 运行。剩余约 16.2 个计算小时，考虑计划暂停后的粗略墙钟完成估计接近 2026-08-05 01:00（Asia/Taipei）。

在同一快照中，本地 rank-16/32 基线已达到 100,000 步中的第 80,350 步，速度约 1.70--1.75 steps/s。最近的训练损失约为 0.017--0.028，第 80,000 步的验证在真实动作维度上为 0.054120（`flow_loss_padded_dims=0`）。截至第 75,000 步的检查点均已存在。其计算 ETA 约为 3.2 小时，再加上剩余的验证和检查点暂停。在匹配的已见样本检查点上，用相同的解码 100 episode 物理审计比较两次运行。更多适配器参数并不保证更好的机器人行为，因此不要仅根据训练或验证流损失选择 kiwi 的最终检查点。

2026-08-04 约 14:05（Asia/Taipei）的进度更新：kiwi 运行处于约第 54,649 / 100,000 步（约 55%），以约 1.28 steps/s 运行，14,950 MiB / 98% util / 66 C，最新检查点为 50K（匹配 50K 的解码对比见第 28.6 节），ETA 接近 2026-08-05 00:00（Asia/Taipei）。本地分离 rank 基线已完成（第 24 节），因此本地 4090 空闲。

| 验证步 | Masked 真实维度流损失 |
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

该运行于 2026-08-05 01:10（Asia/Taipei）完成全部 100,000 步（墙钟时间 24h20m，约 1.27 steps/s），处理 400K 个样本（3.30 个 epoch）。最终训练损失约 0.02。最佳 masked flow 验证出现在 20K（**0.036459**）；此后单调上升到 100K 的 0.069094——比 38M 分离 rank 运行（0.038474 -> 0.059480）更陡的过拟合攀升。实际保存的最佳验证检查点是 25K（0.040136），因为检查点每 12.5K 写入一次，20K 的最优点未被保存。

结论：220M 运行是一个负面的容量结果。其最佳 flow 验证（20K）仅略低于 38M 的（0.0365 对 0.0385），但到 100K 时过拟合更严重（0.0691 对 0.0595），且匹配 50K 的解码对比（第 28.6 节）显示在同等训练下没有端点质量优势。5.72 倍的适配器容量并未改善行为；38M 分离 rank 配置仍是推荐的 π0.5 配方（第 28 节，README）。全部八个检查点（12.5K-100K）加上 `last` 均保存在 kiwi 上。

## 28. 两次活动运行的离线预测可视化

2026-08-04，使用 `visualize_predictions.py`（第 17 节）在留出的验证集 episode 0-4 上可视化了每个活动运行的最新检查点，使用相同 seed，并启用 `--project` 图像夹爪尖端投影。这是定性 + 解码指标的快照，不是检查点选择审计：它只采样五个 episode，而第 22 节的审计运行全部 100 个。

### 28.1 可视化所用的检查点

| 运行 | 最新检查点 | 当时训练步 | 可训练参数 |
| --- | --- | ---: | ---: |
| 本地分离 rank `pi05_openpi_split_lora_masked_1302_bs4` | `075000` | 约 83K / 100K | 38,624,288 |
| Kiwi 高容量 `pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full` | `025000` | 约 28K / 100K | 220,916,768 |

两个检查点处于**不同的训练进度**（75K 对 25K，即 38M 运行已见过的样本约为 3 倍）。阅读下面的指标时请记住这一注意事项；见第 28.4 节。

### 28.2 38M 运行是如何可视化的（本地，与训练并行）

本地 4090 在 38M 训练消耗约 11.7 GiB 时仍有约 12.6 GiB 空闲，因此可视化与实时运行并行进行：

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

### 28.3 220M 运行是如何可视化的（从 kiwi 复制适配器）

220M 运行在 `kiwi`（zfei@10.98.19.22，端口 2203）上执行，其 RTX 5080 在训练期间仅有约 1.4 GiB 空闲（14.9 / 16.3 GiB）。在那里运行约 5-6 GiB 的可视化器会导致训练运行 OOM，因此只将 PEFT 适配器目录复制到本地主机并在本地解码。基础模型 `lerobot/pi05_base` 已在本地的 Hugging Face 缓存中（本地运行会加载它），因此无需复制基础模型。

注意 kiwi 的仓库直接位于 `~/code/` 下，而不像本地树那样位于 `~/code/lerobots/` 下：

```bash
cd /mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified
mkdir -p outputs/train/pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full/checkpoints/025000

rsync -azh -e "ssh -p 2203" \
  zfei@10.98.19.22:code/lerobot-fei-v5.0-umi-unified/outputs/train/pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full/checkpoints/025000/pretrained_model \
  outputs/train/pi05_high_capacity_lora_r96_expert_r192_masked_1302_bs4_full/checkpoints/025000/
```

复制后的 `pretrained_model/` 约为 843 MiB：`adapter_model.safetensors`（220M 适配器为 883 MiB）、`adapter_config.json`、策略 `config.json`、`train_config.json`，以及序列化的 UMI 预处理器/后处理器和归一化统计量。存在 `adapter_config.json` 时，可视化器会自动检测 PEFT 适配器、加载指定的基础模型并应用它。

可视化命令与第 28.2 节相同，只是检查点路径和输出目录不同：

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

两次运行是**顺序**执行的，而不是并发执行，因此峰值 GPU 使用保持在训练加一个可视化器，而不是训练加两个。

### 28.4 解码后的离线指标（episode 0-4）

两次写入都位于 `outputs/debug/viz_pi05_<tag>/sroiv2_strawberry_picking_lab_validation/`：`pred_episode_0..4.mp4`（每个约 3-4 MB）和 `prediction_metrics.json`。数值是 285 个采样帧上的端点（chunk 最终）误差；越低越好。

| 端点指标 | 38M @75K | 220M @25K |
| --- | ---: | ---: |
| XYZ 均值 | **27.1 mm** | 35.2 mm |
| XYZ 中位数 | **19.8 mm** | 28.9 mm |
| 旋转均值 | **3.64 deg** | 4.93 deg |
| 夹爪均值 | **0.218** | 0.254 |

在这些不相等的训练阶段，38M 检查点在全部四个端点指标上解码更好。**这不是容量结果**：38M 检查点的训练时间是其三倍。根据第 24 节 / 第 27 节，在匹配的第 25K 步，两次运行的 masked 真实维度流损失几乎持平（38M 0.039386 对 220M 0.040136），因此上述差距主要来自训练阶段，而不是适配器 rank。不要仅凭这些数字就断定 220M 容量更差。

### 28.5 公平（匹配样本）对比

两次运行还共享一个 50K 检查点（匹配步数和匹配的约 200K 已见样本），因此同类容量视角是使用第 28.2 节命令形式对两者在 50K 处解码，并为每次运行换入 `.../checkpoints/050000/pretrained_model`。220M@50K 适配器完全按照第 28.3 节的方式从 kiwi 复制。结果如下。

### 28.6 匹配 50K 的解码对比（容量结果）

在与第 28.4 节相同的验证 episode 0-4、seed 和 `--project` 协议上解码（各 285 帧）；越低越好。公平对比是 **38M@50K 对 220M@50K** 这一对列（匹配步数和匹配样本）。

| 端点指标 | 38M @50K | 38M @75K | 220M @25K | 220M @50K |
| --- | ---: | ---: | ---: | ---: |
| XYZ 均值 | 30.3 mm | **27.1 mm** | 35.2 mm | 28.9 mm |
| XYZ 中位数 | **19.8 mm** | **19.8 mm** | 28.9 mm | 24.5 mm |
| 旋转均值 | 3.91 deg | **3.64 deg** | 4.93 deg | 3.99 deg |
| 夹爪均值 | **0.22** | **0.22** | 0.25 | 0.27 |

在匹配训练下，5.72 倍容量的 220M 适配器并**没有**胜过 38M 适配器：38M 在 XYZ 中位数（19.8 对 24.5 mm）、夹爪（0.22 对 0.27）上更好，在旋转上大致持平；220M 仅在 XYZ 均值上更好（28.9 对 30.3 mm），而这是尾部效应，因为它在中位数上落后。因此额外容量不是这里的杠杆——这与第 27 节的告诫一致：更多参数并不保证更好的行为。

两个趋势：220M 运行在仅完成 55% 训练时仍在攀升（25K->50K：均值 35.2->28.9 mm，4.93->3.99 deg），而 38M 已基本趋于平稳（50K->75K：中位数和旋转大致持平）。220M 的夹爪误差独特地在 25K->50K 期间恶化（0.25->0.27），而其位姿指标却在改善。

注意事项：这是一个 5 episode（285 帧）样本，而不是第 22 节的 100 episode 解码审计，且差距在 5 episode 噪声范围内。应将其视为“没有 220M 有帮助的信号”，而不是精确排名。38M 分离 rank 配置仍是目前更高效的选择；闭环机器人试验仍是最终选择器。

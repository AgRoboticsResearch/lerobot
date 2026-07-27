---
created: 2026-07-27
status: design guide
tags:
  - openpi
  - pi0.5
  - lora
  - umi
  - relative-ee
---

# π0.5 LoRA Fine-Tuning on the Strawberry-Picking UMI Dataset

This document explains how the official Physical Intelligence OpenPI training
pipeline represents data, constructs action targets, normalizes state and
actions, and performs LoRA fine-tuning. It then maps that pipeline to the
strawberry-picking dataset used by this repository.

The intended starting point is:

- Raw actions on disk: absolute 7D end-effector poses in axis-angle format.
- Training actions: same-base relative 10D poses using row-based rot6d.
- Action horizon: 30.
- Model: official OpenPI π0.5 flow-matching model.
- Fine-tuning method: LoRA using the JAX training implementation.

This is a design guide. The custom OpenPI transforms and training configuration
described below still need to be implemented and tested in an OpenPI checkout.

## 1. Important distinction: LeRobot π0.5 versus OpenPI

This LeRobot repository contains its own π0.5 integration, but this document is
about fine-tuning the model with the official
[Physical Intelligence OpenPI repository](https://github.com/Physical-Intelligence/openpi).

The OpenPI repository currently supports three model families:

- π0: flow-matching vision-language-action model.
- π0-FAST: autoregressive model using the FAST action tokenizer.
- π0.5: upgraded π0 model; the public repository currently supports its
  flow-matching action head.

The checkpoint proposed here is:

```text
gs://openpi-assets/checkpoints/pi05_base
```

## 2. Recommended pipeline

```text
LeRobot frame at time t
    |
    |-- RGB camera observations
    |-- current absolute EE pose
    |-- absolute demonstrated EE action
    |-- task instruction
    v
OpenPI temporal query
    actions[t], actions[t+1], ..., actions[t+29]
    v
Dataset key repacking
    v
Custom Strawberry UMI input transform
    absolute 7D axis-angle -> same-base relative 10D rot6d
    v
Fresh state/action normalization statistics
    q01/q99 normalization for π0.5
    v
Image resize and prompt/state tokenization
    v
Zero-pad state and action to model action_dim=32
    v
π0.5 flow-matching loss
```

The inverse path is required at inference:

```text
π0.5 normalized [30, 32] output
    v
Unnormalize
    v
Keep the first 10 physical action dimensions
    v
rot6d -> valid rotation matrix
    v
T_absolute[k] = T_base @ T_relative[k]
    v
Absolute 7D axis-angle EE command
    v
IK -> robot joint command
```

## 3. Dataset format

OpenPI trains from LeRobot datasets. Field names on disk do not have to match
the model input names exactly because OpenPI uses a `RepackTransform` followed
by robot-specific input transforms.

The official LIBERO conversion example stores:

| Field | Type | Example shape |
|---|---|---:|
| Main image | image | `(H, W, 3)` |
| Wrist image | image | `(H, W, 3)` |
| State | float32 | `(8,)` |
| Action | float32 | `(7,)` |
| Task | string | scalar |

See the official
[LIBERO-to-LeRobot conversion example](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/libero/convert_libero_data_to_lerobot.py).

### 3.1 Current strawberry dataset

The current UMI pipeline stores:

```text
action =
    [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
```

This is an absolute 7D EE pose for each frame. The existing ACT processor asks
the data loader for:

```text
action_delta_indices = [-1, 0, 1, ..., 29]
```

It uses the `-1` and `0` samples to construct observation state, removes the
leading sample, and trains on the 30 actions at offsets `0..29`.

The stock OpenPI LeRobot loader instead constructs a sequence using:

```text
[0, 1, ..., action_horizon - 1]
```

At a horizon of 30, that produces:

```text
actions[t], actions[t+1], ..., actions[t+29]
```

See the official
[OpenPI data loader](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/data_loader.py).

Consequently:

- The 30-action target indexing can match the existing ACT target.
- Reproducing ACT's exact 20D state also requires `action[t-1]`.
- The standard OpenPI loader must be extended if the previous frame is required.
- A cleaner alternative is to store or supply the current EE pose as a distinct
  state field.

### 3.2 Recommended raw fields

For a clean OpenPI adapter, make these logical values available:

```text
observation.image             RGB exterior image
observation.wrist_image       RGB wrist image, if present
observation.ee_pose           current absolute 7D EE pose
actions                       absolute 7D demonstrated EE pose
task                          language instruction
```

The names on disk may differ. `RepackTransform` maps them to the names consumed
by the custom input transform.

## 4. UMI action target

OpenPI does not impose one universal physical action definition. The
robot-specific data transform defines the target.

OpenPI includes a generic `DeltaActions` transform, but that transform performs
elementwise subtraction:

```python
actions[..., selected_dims] -= state[..., selected_dims]
```

That is suitable for some joint-position targets. It is not the correct way to
calculate relative rotations from absolute axis-angle poses. Do not use the
generic `DeltaActions` transform for this dataset.

See the official
[`DeltaActions` implementation](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/transforms.py).

### 4.1 Correct SE(3) conversion

For sample time `t`, construct one base transform:

```text
T_base = T_absolute[t]
```

For every target `k` in the action chunk:

```text
T_relative[k] = inverse(T_base) @ T_absolute[t + k]
```

Equivalently:

```text
R_relative[k] = transpose(R_base) @ R_absolute[t + k]

t_relative[k] =
    transpose(R_base) @ (t_absolute[t + k] - t_base)
```

All actions in the chunk use the same `T_base`. They must not be accumulated or
chained from one predicted action to the next.

### 4.2 Physical model target

Encode `R_relative` using the first two rows of the relative rotation matrix:

```text
rot6d =
    [R00, R01, R02,
     R10, R11, R12]
```

The physical action target becomes:

```text
[dx, dy, dz,
 R00, R01, R02,
 R10, R11, R12,
 gripper]
```

Its shape is:

```text
[action_horizon, physical_action_dim] = [30, 10]
```

This is the same row-based rot6d convention used by
`examples/umi_relative_ee/umi_style_ee_processor_pipeline.md`.

### 4.3 Inference inverse

For every predicted 10D action:

1. Reconstruct a valid rotation matrix from rot6d using Gram-Schmidt.
2. Construct `T_relative`.
3. Compose it with the current chunk base:

```text
T_absolute[k] = T_base @ T_relative[k]
```

4. Convert `R_absolute` to axis-angle.
5. Return:

```text
[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
```

The same base must be retained for all actions executed from that prediction
chunk.

## 5. Observation state choices

OpenPI independently normalizes state and actions, so their physical dimensions
may differ before padding.

There are three reasonable state designs.

### Option A: exact ACT-compatible 20D state

Reproduce the existing processor:

```text
state =
    [
      relative_pose(action[t-1], base=action[t]),
      relative_pose(action[t],   base=action[t])
    ]
```

Each relative pose is 10D, giving a 20D state.

Advantages:

- Closest comparison with the current ACT experiment.
- Previous-to-current motion is explicitly available.

Disadvantages:

- Requires loading `action[t-1]`.
- The second pose is always approximately identity.
- Identity rotation dimensions can have extremely small statistical ranges.
- Half of the state is largely redundant.

### Option B: current absolute 7D EE pose

Use:

```text
[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
```

Advantages:

- Matches the conventional meaning of proprioceptive state.
- Does not require a previous timestep.
- Avoids an always-identity relative state block.

Disadvantages:

- It is not identical to the ACT experiment.
- Absolute pose statistics are workspace- and camera-setup-specific.

### Option C: previous-to-current 10D motion state

Use only:

```text
relative_pose(action[t-1], base=action[t])
```

Advantages:

- Preserves the useful motion information from ACT's 20D state.
- Removes the redundant identity pose.

Disadvantages:

- Still requires the previous frame.
- It is less similar to the proprioceptive states used during OpenPI
  pretraining.

### Recommendation

For the first OpenPI run, use Option B unless an exact ACT comparison is the
primary goal. For an exact comparison, use Option A but carefully audit its
quantile statistics.

## 6. OpenPI model-side shapes

The official `Pi0Config` defaults to:

```python
action_dim = 32
action_horizon = 50
```

For this experiment:

```python
action_dim = 32
action_horizon = 30
```

Do not initially change `action_dim` to 10. The pretrained action input and
output projection weights were created for 32 dimensions.

OpenPI normalizes the physical state and action first, then applies
`PadStatesAndActions`:

```text
physical action: [30, 10]
model action:    [30, 32]
```

The last 22 values are zero padding. At inference, the output transform keeps:

```python
actions[..., :10]
```

See the official
[π0 configuration](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0_config.py),
[padding transform](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/transforms.py),
and [LIBERO output adapter](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/policies/libero_policy.py).

## 7. π0.5 flow-matching training target

ACT directly predicts a normalized action chunk and optimizes an action
reconstruction loss. π0.5 uses a different objective.

Let:

```text
a       normalized clean action chunk
epsilon Gaussian noise with the same shape as a
t       sampled flow time
```

OpenPI constructs:

```text
x_t = t * epsilon + (1 - t) * a
u_t = epsilon - a
```

The action expert receives `x_t` and predicts the velocity `u_t`:

```text
loss = mean_over_action_dim((predicted_velocity - u_t)^2)
```

At inference, sampling starts from Gaussian noise at `t=1` and numerically
integrates the learned velocity toward an action sample at `t=0`.

See the official
[π0/π0.5 loss implementation](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py).

### 7.1 Implications for target weights

The upstream loss:

- Does not explicitly weight translation, rotation, and gripper differently.
- Gives each normalized model dimension equal weight.
- Relies on normalization to make physical dimensions comparable.
- Also trains the zero-padded model dimensions toward zero clean targets.

Therefore, normalization errors can appear like target-weighting problems.

## 8. Normalization

The OpenPI transform order is:

```text
repack transforms
    -> robot/data transforms
    -> Normalize
    -> model transforms
```

This order is essential. The normalization statistics must describe the final
physical target presented to the model, not the raw absolute 7D poses.

The custom absolute-to-relative SE(3) conversion must therefore be a
`data_transforms` input step. It must not be placed after normalization.

See the official
[data configuration](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/config.py)
and [transform ordering](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/data_loader.py).

### 8.1 π0 versus π0.5 defaults

OpenPI selects normalization by model family:

| Model | Default state/action normalization |
|---|---|
| π0 | mean/std z-score |
| π0.5 | q01/q99 quantile |
| π0-FAST | q01/q99 quantile |

For π0.5:

```text
x_norm =
    (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
```

This maps:

```text
q01 -> -1
q99 -> +1
```

It does not clip values outside the percentile interval. An outlier may
therefore produce `x_norm < -1` or `x_norm > 1`.

See the official
[OpenPI normalization transform](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/transforms.py).

### 8.2 Fresh statistics are required

Do not reuse Trossen, DROID, LIBERO, or another pretrained robot's normalization
statistics for this 10D UMI action space. Those robots use different action
semantics, such as joint position, joint velocity, or simulator delta commands.

OpenPI only recommends reusing pretrained statistics when the robot and action
definitions match. See the official
[normalization-statistics guide](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md).

After the custom data configuration is implemented, run:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_strawberry_umi_lora
```

The official statistics script applies repacking and data transforms before
accumulating `state` and `actions`, which is exactly what is needed here:

```text
raw absolute 7D
    -> custom relative 10D transform
    -> compute q01/q99
```

See
[`scripts/compute_norm_stats.py`](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/scripts/compute_norm_stats.py).

### 8.3 Rot6d normalization risk

For small relative rotations, a target is close to:

```text
[R00, R01, R02, R10, R11, R12]
    approximately
[1,   0,   0,   0,   1,   0]
```

The `R00` and `R11` channels may have very small `q99 - q01` spans. Because
OpenPI does not clip quantile-normalized data, a rare larger rotation can become
a very large normalized value.

For every state and action dimension, inspect:

```text
q01
q99
q99 - q01
minimum normalized value
maximum normalized value
maximum absolute normalized value
fraction where abs(x_norm) > 3
fraction where abs(x_norm) > 5
```

Also verify that all values are finite.

### 8.4 Recommended normalization policy

Start with OpenPI's standard q01/q99 normalization for π0.5, not raw min/max.
However, do not begin a long training run until the transformed dataset has
passed the per-dimension normalized-tail audit.

If rot6d produces severe tails, consider:

1. Keeping rot6d and applying a documented minimum denominator/span to
   near-constant rotation channels.
2. Keeping rot6d and clipping only extreme normalized tails, with the same
   inverse convention at inference.
3. Changing the relative rotation target to a 3D rotation vector:

   ```text
   [dx, dy, dz, rx, ry, rz, gripper]
   ```

The third option is often easier to normalize for small motions, but it no
longer provides a target identical to the current ACT experiment. For the first
model comparison, retain 10D rot6d and audit it carefully.

## 9. π0.5 state tokenization

π0.5 differs from π0 in how it handles proprioceptive state:

- π0 projects continuous state into a state token in the action suffix.
- π0.5 normally includes state in the discrete language-side tokens.

The relevant option is:

```python
discrete_state_input=True
```

This is the π0.5 default. Keep it enabled for the strawberry robot unless an
intentional image-only ablation is desired.

In the current JAX implementation, setting it to `False` prevents state from
being added to the prompt tokens, while the π0.5 model also skips π0's
continuous state token. This can effectively remove direct proprioceptive
conditioning.

See
[`Pi0Config`](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0_config.py)
and
[`TokenizePrompt`](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/transforms.py).

## 10. LoRA support

Use the official JAX training path for LoRA.

The current OpenPI README explicitly lists LoRA as unsupported by its PyTorch
training implementation. The official resource estimate for single-GPU LoRA
fine-tuning is more than 22.5 GB of GPU memory.

See the current
[OpenPI README](https://github.com/Physical-Intelligence/openpi).

### 10.1 LoRA model variants

The official π0 low-memory example enables LoRA in both language/action model
variants:

```python
paligemma_variant="gemma_2b_lora"
action_expert_variant="gemma_300m_lora"
```

It also:

- Loads base checkpoint parameters.
- Uses `get_freeze_filter()` from the matching model configuration.
- Disables EMA with `ema_decay=None`.

See the official
[LoRA training configuration](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/config.py).

### 10.2 Proposed π0.5 LoRA model configuration

OpenPI provides a ready-made π0 LoRA example and a π0.5 full-fine-tuning
example, but it does not currently provide a named π0.5 LoRA example for this
robot. The following is a proposed combination of the supported configuration
mechanisms:

```python
strawberry_model = pi0_config.Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=30,
    discrete_state_input=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
```

The matching training configuration should contain:

```python
TrainConfig(
    name="pi05_strawberry_umi_lora",
    model=strawberry_model,
    data=StrawberryUMIDataConfig(
        repo_id="sroi/sroiv2_strawberry_picking_lab_1000onesb_1125",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    freeze_filter=strawberry_model.get_freeze_filter(),
    ema_decay=None,
    num_train_steps=30_000,
)
```

Do not copy the model configuration once for `model` and recreate it differently
for `freeze_filter`. Both must use identical model variants and dimensions.

### 10.3 What to verify before training

Because this specific π0.5 LoRA combination is not a ready-made upstream
example, perform these checks:

1. The π0.5 base checkpoint loads without missing or incompatible core weights.
2. LoRA parameters are present in both configured model variants.
3. Frozen and trainable parameter counts are printed.
4. Base language-model weights are actually frozen.
5. Action input/output projections have the expected 32D shapes.
6. One forward/backward step completes with finite loss.
7. At least one LoRA parameter receives a finite, non-zero gradient.
8. A checkpoint can be saved, loaded, and used for one inference call.

The upstream freeze filter primarily freezes non-LoRA language-model
parameters. Inspect the complete trainable parameter tree rather than assuming
that only adapter matrices are trainable.

## 11. Custom OpenPI components to implement

The adapter should contain the following pieces.

### 11.1 `StrawberryUMIInputs`

Responsibilities:

- Parse camera images as uint8 HWC arrays.
- Map available cameras to:

  ```text
  base_0_rgb
  left_wrist_0_rgb
  right_wrist_0_rgb
  ```

- Provide correct image masks for missing views.
- Extract or construct observation state.
- Convert the raw absolute 7D action chunk to relative 10D rot6d.
- Forward the task prompt.
- Retain the current absolute base pose when needed by output conversion.

### 11.2 `StrawberryUMIOutputs`

Responsibilities:

- Keep the first 10 physical dimensions from the 32D model output.
- Project rot6d back onto SO(3).
- Convert each action to an absolute pose using the same chunk base.
- Return absolute 7D axis-angle actions for the existing IK/deployment stack.

### 11.3 `LeRobotStrawberryUMIDataConfig`

Responsibilities:

- Define dataset key repacking.
- Add `StrawberryUMIInputs` before normalization.
- Add `StrawberryUMIOutputs` after unnormalization.
- Enable prompt extraction from LeRobot task metadata.
- Set the action sequence key used for temporal queries.
- Load fresh normalization statistics for this dataset.

### 11.4 Temporal loader support

If using the exact ACT-compatible state, extend the loader so the transform
receives:

```text
action[t-1], action[t], ..., action[t+29]
```

The custom transform should:

```text
1. Construct state from t-1 and t.
2. Use action[t] as T_base.
3. Remove t-1 from the target sequence.
4. Produce exactly 30 target actions.
```

If using current absolute state instead, the standard `[t..t+29]` action query
is sufficient.

## 12. Pre-training validation checklist

Before launching LoRA training, dump several transformed samples and verify the
following.

### Raw data

- Images correspond to the same timestep as the state.
- Raw action shape is `[30, 7]`.
- Axis-angle values use radians.
- Gripper meaning and direction are documented.
- No chunk crosses episode boundaries incorrectly.

### Relative target

- Relative action shape is `[30, 10]`.
- `action[0]` is close to identity if its raw pose equals the base pose.
- The same base pose is used for every action in the chunk.
- Rot6d uses rows, not columns.
- Reconstructed rotation matrices have determinant close to `+1`.

### Round trip

For every test target:

```text
absolute 7D
    -> matrix
    -> relative 10D
    -> matrix
    -> absolute 7D
```

Check:

- Translation reconstruction error.
- Geodesic rotation error in degrees.
- Gripper reconstruction error.

Do not compare axis-angle vectors directly as the primary rotation check,
because different axis-angle vectors can represent the same rotation.

### Normalization

- Statistics were computed after the UMI transform.
- `norm_stats.json` contains state and actions.
- q01/q99 shapes match physical dimensions before 32D padding.
- No `q99-q01` span is unexpectedly tiny.
- No NaN or infinity appears after normalization.
- Rotation outlier magnitudes are understood.

### Model batch

- Images have the expected OpenPI image keys.
- Image masks correctly identify missing views.
- Prompt is non-empty.
- State reaches π0.5 tokenization.
- Final action tensor has shape `[batch, 30, 32]`.
- The first 10 dimensions contain normalized physical targets.
- The remaining 22 dimensions are zero before flow noise is applied.

## 13. Suggested staged experiment

### Stage 1: data-only test

- Load 100 samples.
- Visualize raw images.
- Print raw and transformed state/action shapes.
- Run SE(3) round-trip checks.
- Produce rotation and translation target histograms.

### Stage 2: normalization test

- Compute full training statistics.
- Produce a per-dimension q01/q99 table.
- Measure normalized tails.
- Compare train and validation distributions using training statistics.

Validation data must use training statistics. Do not compute an independent
normalization system for validation.

### Stage 3: LoRA smoke test

- Use one GPU.
- Train for 10 to 100 steps.
- Save a checkpoint.
- Reload it.
- Run inference on fixed validation frames.
- Confirm LoRA weights changed and frozen weights did not.

### Stage 4: short comparison

- Train for a few thousand steps.
- Keep the action representation and horizon identical to ACT.
- Visualize predicted versus ground-truth translation and rotation.
- Report physical-space errors after unnormalization.

### Stage 5: full run

Only begin the long run after:

- The data transform is validated.
- Normalized rotation tails are acceptable.
- LoRA checkpoint reload works.
- Short-run predictions improve over the initial checkpoint.

## 14. Recommended first experiment

Use:

```text
Model                 π0.5 base
Backend               JAX
Fine-tuning           LoRA
Model action_dim      32
Physical action_dim   10
Action horizon        30
Action representation same-base relative SE(3), row rot6d
Normalization         fresh q01/q99
Prompt                LeRobot task instruction
State                 current absolute 7D EE pose
```

This changes the state representation relative to the existing ACT experiment,
but it is the cleanest OpenPI-style proprioceptive input and avoids the
always-identity portion of the 20D ACT state.

For an apples-to-apples model comparison, run a second experiment using the
exact 20D ACT-compatible state after confirming that its quantile normalization
is numerically safe.

## 15. Key decisions

- Use the official OpenPI repository for OpenPI LoRA fine-tuning.
- Use JAX because upstream PyTorch training currently lacks LoRA support.
- Keep `action_dim=32` to preserve pretrained projection shapes.
- Set `action_horizon=30`.
- Convert raw absolute EE poses to same-base relative SE(3) before statistics.
- Do not use OpenPI's elementwise `DeltaActions` for axis-angle poses.
- Keep the physical 10D row-rot6d target for the first ACT comparison.
- Compute fresh q01/q99 statistics on transformed training data.
- Audit near-constant rot6d dimensions before a long run.
- Use training statistics for validation and inference.
- Verify the proposed π0.5 LoRA configuration with a short smoke test.

## 16. Related local documentation

- `examples/umi_relative_ee/umi_style_ee_processor_pipeline.md`
- `examples/umi_relative_ee/2026-07-10_umi_relative_ee_policy_compatibility.md`
- `examples/umi_relative_ee/smolvla_relative_ee_training.md`


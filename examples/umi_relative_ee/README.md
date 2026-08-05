# Unified UMI-style relative EE training

This directory is the maintained entrypoint for ACT, SmolVLA, and π0.5 on one shared dataset contract:

- on disk, `action` is absolute 7D
  `[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]`;
- `observation.state` may be absent because it is derived from consecutive
  actions `[action(t-1), action(t)]`;
- the model receives a flattened 20D two-pose relative rot6d state;
- the model predicts 10D `[dx, dy, dz, rot6d(6), gripper]` actions;
- all targets in one chunk use `action(t)` as the same base pose. They are not
  chained from the preceding predicted target.

The SE(3) transforms are saved in each checkpoint's preprocessor and
postprocessor. The raw dataset and its raw 7D statistics are not modified.

## Unified entrypoints

All three policies use `--policy.use_umi_relative_ee=true` with the standard
`lerobot-train` command. `train_umi_relative_ee.py`,
`train_relative_ee_processor.py`, and `train_pi05_lora.py` are thin compatible
wrappers around that same trainer. ACT and SmolVLA use MIN_MAX normalization;
π0.5 uses QUANTILES. Existing ACT and SmolVLA checkpoints with the legacy
serialized processor names load directly.

- ACT and shared processor design: `doc/umi_style_ee_processor_pipeline.md`
- SmolVLA runbook: `doc/smolvla_relative_ee_training.md`
- SmolVLA/π0.5 padded-noise strategies: `doc/padded_noise_strategy.md`
- π0.5 commands: the sections below and `shell_scripts/train_pi05_lora.sh`
- Unified prediction visualization: `doc/prediction_visualization.md`
- Policy-neutral ACT/SmolVLA/π0.5 dataset metrics: `eval_open_loop_dataset.py`
- Pi0.5 and SmolVLA RTC deployment: `doc/rtc.md`
- Piper async server/client deployment: `doc/ASYNC_INFERENCE.md`
- Rotation normalization analysis (UMI identity vs. our per-dim scaling; jumpiness hypothesis + A/B test): `doc/rotation_normalization.md`
- Migration manifest and checksums: `../../docs/umi_migration_manifest.md`
- Historical-tool smoke results: `../../docs/umi_legacy_tool_smoke.md`
- Historical compatibility review: `doc/2026-07-10_umi_relative_ee_policy_compatibility.md`

## Install

Use the existing Conda `py312` environment (do not create a project `.venv`):

```bash
conda activate py312
VIRTUAL_ENV="$CONDA_PREFIX" ~/.local/bin/uv sync --active --inexact --locked \
  --extra pi --extra peft --extra test --extra dev
```

The first run also needs access to Hugging Face to download
`lerobot/pi05_base` and the PaliGemma tokenizer.

> **Hugging Face access:** π0.5 preprocessing loads
> `google/paligemma-3b-pt-224`, which is gated. Accept its license and run
> `hf auth login` with an approved account before training. The current machine
> receives HTTP 403 for that repository.


## Recommended training commands

The current recommended command for each policy on the strawberry
`1302_occlusion` dataset with the separate validation set. The entry scripts
are executable, so run them directly (`./` or absolute path). Dataset roots are
host-specific: ACT and π0.5 below use the local workstation (`/mnt/...`),
SmolVLA uses `kiwi` (`/home/zfei/...`). Adjust roots to the host you run on.

### π0.5 — 38M split-rank LoRA (current recommendation)

The completed 38M split-rank run (global rank/alpha 16, action-expert 32/32,
masked-subspace flow, 38,624,288 trainable params) is the current π0.5
fine-tuning recommendation. Launch it through the batch-scaled wrapper —
batch 4 runs 100K steps and reproduces the completed run:

```bash
bash examples/umi_relative_ee/shell_scripts/run_pi05_openpi_split_lora_umi.sh 4
```

Full config, the validation-loss trajectory, and the matched-50K capacity
comparison against the 220M run are in `doc/pi0.5_finetunning.md`. The narrow
`train_pi05_lora.sh` starter in "Training baseline" below is the low-memory
default, not the recommended config.

### ACT

Identity-rot6d normalization, chunk 30, 2.5M steps, batch 8, on the local
workstation:

```bash
/mnt/data0/code/lerobots/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
  --val_freq=10000 \
  --policy.type=act \
  --policy.use_umi_relative_ee=true \
  --policy.umi_rot6d_identity_norm=true \
  --policy.device=cuda \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.repo_id=zfff/act_umi_identity_rot6d_1302 \
  --policy.push_to_hub=false \
  --seed=1000 \
  --save_freq=100000 \
  --steps=2500000 \
  --batch_size=8 \
  --num_workers=4 \
  --log_freq=200 \
  --eval_freq=0 \
  --output_dir=outputs/train/act_umi_identity_rot6d_1302 \
  --job_name=act_umi_identity_rot6d_1302 \
  --wandb.enable=true \
  --wandb.project=lerobot
```

> **Result (trained 2026-08-05 on 1459_occlusion, 1M steps):** ACT 52M params,
> 11h30m. Best validation among saved checkpoints is **800K (val 0.0338)**;
> the global optimum 750K (0.0332) is not saved (checkpoints are every 100K),
> and val is flat at about 0.0334 from ~700K on with no overfit blowup. This
> beats the prior 1302_occlusion ACT run (best ~0.0394). Checkpoints at
> `outputs/train/act_umi_identity_rot6d_1459`.

### SmolVLA

OpenPI full-width flow, chunk 30, 1M steps, batch 8, on `kiwi`:

```bash
/home/zfei/code/lerobot-fei-v5.0-umi-unified/examples/umi_relative_ee/train_relative_ee_processor.py \
  --dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_1302_occlusion \
  --dataset.root=/home/zfei/data/sroiv2_strawberry_picking_lab_1302_occlusion \
  --validation_dataset.repo_id=sroi/sroiv2_strawberry_picking_lab_validation \
  --validation_dataset.root=/home/zfei/data/sroiv2_strawberry_picking_lab_validation \
  --val_freq=50000 \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.use_umi_relative_ee=true \
  --policy.device=cuda \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=1000000 \
  --policy.scheduler_decay_lr=0.0000025 \
  --policy.repo_id=zfff/smolvla_openpi_fullwidth_1302_1M \
  --policy.push_to_hub=false \
  --seed=1000 \
  --steps=1000000 \
  --save_freq=100000 \
  --log_freq=200 \
  --eval_freq=0 \
  --batch_size=8 \
  --num_workers=4 \
  --output_dir=outputs/train/smolvla_openpi_fullwidth_1302_1M \
  --job_name=smolvla_openpi_fullwidth_1302_1M \
  --wandb.enable=true \
  --wandb.project=lerobot
```

## Training baseline

The launcher defaults to the dataset recorded in the source UMI notes:

```text
repo: sroi/sroiv2_strawberry_picking_lab_1000onesb
root: /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb
```

Run:

```bash
bash examples/umi_relative_ee/shell_scripts/train_pi05_lora.sh
```

To point it elsewhere without editing the file:

```bash
DATASET_REPO_ID=my_org/my_dataset \
DATASET_ROOT=/data/my_dataset \
VALIDATION_DATASET_REPO_ID=my_org/my_validation_dataset \
VALIDATION_DATASET_ROOT=/data/my_validation_dataset \
OUTPUT_DIR=outputs/train/my_pi05_umi_lora \
POLICY_REPO_ID=my_org/my_pi05_umi_lora \
bash examples/umi_relative_ee/shell_scripts/train_pi05_lora.sh
```

The 24 GB launcher uses LoRA rank 16, bf16, gradient
checkpointing, batch size 2, a 30-step chunk, and no `torch.compile`. If it
OOMs, return to batch size 1. This
trainer does not currently expose gradient accumulation.

The launcher uses 50,000 optimizer steps as an initial run. Prefer 5--10
dataset epochs when selecting the final value:

```text
steps_per_epoch = ceil(number_of_frames / batch_size)
total_steps     = desired_epochs * steps_per_epoch
```

Change `scheduler_decay_steps` to the same final step count.


## Offline validation

The launcher enables the separate validation dataset from the previous UMI
workflow by default:

```text
repo: sroi/sroiv2_strawberry_picking_lab_validation
root: /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation
frequency: every 10,000 optimizer steps
```

Each validation event evaluates the full selected validation dataset in a
deterministic order and logs sample-weighted `val/loss` to W&B. The
preprocessor is reset before and after validation, the policy returns to train
mode afterward, and only statistics from the training dataset are used. No
validation statistics are recomputed or written to disk.

This local validation set has 100 episodes and 9,274 frames, so a full π0.5
validation pass at batch size 1 is expensive. Increase `VAL_FREQ`, or append
`--validation_dataset.episodes=[0,1,...]` to a direct invocation for faster
iteration. Set `VAL_FREQ=0` to disable offline validation.

## Required data checks

Before a long run, confirm:

- every action is finite and has exactly seven values;
- rotation values are axis-angle rotation vectors in radians, not Euler angles;
- gripper values use one consistent range;
- each episode used for a 30-step chunk has at least 31 contiguous frames;
- each frame has a non-empty task string, because π0.5 is language-conditioned;
- camera keys and views are consistent across episodes.

The training entry point fails early on the action shape and too-short episode
case. It computes transformed quantile statistics in memory without rewriting
the dataset metadata on disk.

## Resume

Use the standard checkpoint config; do not start the launcher again with
the same output directory:

```bash
/home/zfei/anaconda3/envs/py312/bin/python examples/umi_relative_ee/train_pi05_lora.py \
  --config_path=outputs/train/pi05_lora_umi_relative_ee/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## Deployment contract

At each new action chunk, run the preprocessor on the current absolute 7D EE
pose before calling the policy. Postprocess the entire predicted chunk at once;
the saved postprocessor uses that cached chunk-start pose for all 30 targets.
During execution, continue updating the two-frame state history every control
tick. Convert the resulting absolute 7D targets through IK before sending joint
commands.

## Deploy state: where the EE pose comes from and how the two-frame state is built

The policy never consumes a raw 7D pose. It consumes a flattened **20D
two-pose relative state**, derived from a *current* and a *previous* absolute
7D EE pose. This section records where those poses come from at deploy time and
why the sync and async deploy paths build the pair differently. Full source
citations are in `doc/live_inference_input_contract_and_pose_source.md`.

### The 7D state is robot proprioception (FK), not SLAM

Each control tick, both `deploy_umi_relative_ee_piper.py` and the async Piper
client build the **current** 7D absolute EE pose of the `camera_link` frame with
`ee_pose_aa_from_fk`:

```text
[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper_norm]
 \_____ position _____/  \____ orientation ____/   normalized 0..1
```

- `x,y,z` + axis-angle come from `kinematics.forward_kinematics(joints)` — placo
  solving the URDF chain from base to `camera_link` from the six joint encoders;
- `gripper_norm` is the gripper reading normalized to `[0,1]`.

There is **no SLAM/visual-odometry tracker in the control loop.** The deploy
pose is free proprioception from the arm, exactly like any robot-arm policy.
(SLAM is offline data-collection only; see `doc/...pose_source.md` §5.) This is
the single 7D **`current`** pose; the **`previous`** pose is simply
`current(t-1)`.

### The 20D state = `[inv(T_current) @ T_prev,  identity]`

`to_umi_relative_state` (`src/lerobot/processor/umi_relative_ee_processor.py`)
turns the `[previous, current]` pair into 20D by computing
`inv(T_current) @ T_pose` for both poses and flattening:

```text
rel_prev    = inv(T_current) @ T_prev      -> [dx,dy,dz, rot6d(6), gripper]  (10D)
rel_current = inv(T_current) @ T_current   -> identity                       (10D)
=> 20D = [ rel_prev(10), identity(10) ]
```

The only non-trivial half is *how far the previous pose was from now*. **If the
arm is held still, `prev ≈ current`, the whole 20D collapses to
`[identity, identity]`, and the image alone drives the predicted chunk.** This
is why the camera-only test client and `visualize_predictions.py` camera mode
predict near-zero motion from a static identity pose, and why
`--update_state` (chaining the last prediction as the next pose) is what
synthesizes a motion cue in the no-robot visualizers.

### Sync sends `[1,7]`; the preprocessor chains `previous` internally

`deploy_umi_relative_ee_piper.py` keeps no `previous` variable. It sends only
the current pose, and `UmiRelativeStateStep` synthesizes the pair from its own
instance state (the `state.ndim == 2` branch):

```python
ee_aa = ee_pose_aa_from_fk(kinematics, current_joints, gripper_norm)   # [7]
batch = {OBS_STATE: ee_aa.unsqueeze(0)}                                # [1,7]  single pose
# inside UmiRelativeStateStep:
#   previous = current if self._previous_state is None else self._previous_state
#   state_pair = stack([previous, current])                            # [1,2,7]
#   self._previous_state = current.detach().clone()                    # "chaining"
```

First tick: `_previous_state` is `None`, so `prev = current` (identity pair).
Every later tick: `prev` is the pose from the previous call. This works because
the sync `preprocessor` is a **persistent in-process object called every tick**,
so its internal `_previous_state` stays exactly one step behind.

### Async sends an explicit `[previous, current]` pair

The async client maintains both poses itself and sends the pair
(`np.stack([previous_ee, current_ee])` → `[2,7]`). The server passes it straight
through (`_prepare_umi_observation` requires shape `(2,7)`), and
`UmiRelativeStateStep` takes the provided-pair branch (`state.ndim == 3`,
`shape[1] == 2`) instead of chaining. The "chaining" here happens in the client:
`previous_ee = current_ee.copy()` at the end of each tick.

This is **not** a stylistic choice — it is forced by the async decoupling. The
client ticks every `1/fps`, but only **sends an observation when the action
queue needs refilling** (roughly every `actions_per_chunk × chunk_size_threshold`
ticks), so the server sees a *subset* of ticks. The two-frame state needs
**consecutive** frames `current(t-1), current(t)`; if the server chained
internally, its `_previous` would be the pose from the *last inference request*
(many ticks ago) — not `t-1` — and the policy would get a stale, non-consecutive
pair. The client is the only process that reads the arm every tick, so it owns
the consecutive pair and ships it explicitly.

### Net effect

Both paths deliver identical `[1,2,7]` pairs to `to_umi_relative_state`, hence
the same 20D state and the same policy input — the only difference is *who*
remembers the previous pose (the in-process preprocessor vs the client). The
predicted chunk is postprocessed with one chunk-start base in both cases (sync:
one local `postprocessor(chunk)` call; async: the UMI server override
postprocesses the whole chunk in a single call, deliberately **not** the stock
per-action loop, which would re-base each target on a refreshed pose).


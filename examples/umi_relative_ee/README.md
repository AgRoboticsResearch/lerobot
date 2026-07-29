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

- ACT and shared processor design: `umi_style_ee_processor_pipeline.md`
- SmolVLA runbook: `smolvla_relative_ee_training.md`
- π0.5 commands: the sections below and `train_pi05_lora.sh`
- Unified prediction visualization: `prediction_visualization.md`
- Pi0.5 and SmolVLA RTC deployment: `rtc.md`
- Migration manifest and checksums: `../../docs/umi_migration_manifest.md`
- Historical-tool smoke results: `../../docs/umi_legacy_tool_smoke.md`
- Historical compatibility review: `2026-07-10_umi_relative_ee_policy_compatibility.md`

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


## Training baseline

The launcher defaults to the dataset recorded in the source UMI notes:

```text
repo: sroi/sroiv2_strawberry_picking_lab_1000onesb
root: /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_1000onesb
```

Run:

```bash
bash examples/umi_relative_ee/train_pi05_lora.sh
```

To point it elsewhere without editing the file:

```bash
DATASET_REPO_ID=my_org/my_dataset \
DATASET_ROOT=/data/my_dataset \
VALIDATION_DATASET_REPO_ID=my_org/my_validation_dataset \
VALIDATION_DATASET_ROOT=/data/my_validation_dataset \
OUTPUT_DIR=outputs/train/my_pi05_umi_lora \
POLICY_REPO_ID=my_org/my_pi05_umi_lora \
bash examples/umi_relative_ee/train_pi05_lora.sh
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

# LIBERO Training Workspace Plan

## Overview

Training an **ACT** (Action Chunking with Transformers) policy on the **LIBERO `libero_object`** task suite using LeRobot's training infrastructure.

## Folder Structure

```
libero_training/
├── PLAN.md             # This file
├── .gitignore          # Ignores checkpoints/ and training artifacts
├── train_libero.sh     # Training launch script
└── checkpoints/        # (git-ignored) checkpoint output directory
```

## Dataset

**Source:** `HuggingFaceVLA/libero` (preprocessed, LeRobot v3.0 format)

**Local path:** `/mnt/data0/data/libero/LIBERO_HF`

- Loaded directly from local disk via `--dataset.root`
- Already follows LeRobot naming conventions -- no key remapping needed

**Local raw data** at `/mnt/data0/data/libero/LIBERO` contains the original HDF5 demos and is used by the LIBERO sim environment for episode init states, not directly as the training dataset.

### Observation Keys

| LeRobot Key | Source | Shape |
|---|---|---|
| `observation.state` | Proprioceptive (eef_pos, axis_angle, gripper_qpos) | (8,) |
| `observation.images.image` | `agentview_image` (main camera) | (3, H, W) |
| `observation.images.image2` | `robot0_eye_in_hand_image` (wrist camera) | (3, H, W) |

### Actions

- 7D continuous control: 6D end-effector + 1D gripper
- Action space: `Box(-1, 1, shape=(7,))`

## Policy: ACT

- **Type:** `act` (Action Chunking with Transformers)
- **Feature mapping:** Auto-configured from dataset metadata via `make_policy()`
- **Optimizer/Scheduler:** Uses ACT's default training preset (`use_policy_training_preset=True`)

## Environment: LIBERO

- **Task suite:** `libero_object` -- object manipulation tasks
- **Max episode steps:** 280
- **Observation type:** `pixels_agent_pos`
- **Control mode:** `relative` (delta actions)
- **Env processor:** `LiberoProcessorStep` handles image flipping (180 deg) and state conversion during eval

## Training Script (`train_libero.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

# LIBERO ACT Training
# Task suite: libero_object (280 max steps, object manipulation)
# Dataset: HuggingFaceVLA/libero (Local: /mnt/data0/data/libero/LIBERO_HF)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

lerobot-train \
    --policy.type=act \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.root=/mnt/data0/data/libero/LIBERO_HF \
    --env.type=libero \
    --env.task=libero_object \
    --output_dir="${SCRIPT_DIR}/checkpoints" \
    --batch_size=8 \
    --steps=100000 \
    --eval_freq=10000 \
    --save_freq=10000 \
    --eval.batch_size=1 \
    --eval.n_episodes=3 \
    --log_freq=200
```

### Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `--policy.type` | `act` | Action Chunking with Transformers |
| `--dataset.repo_id` | `HuggingFaceVLA/libero` | Preprocessed dataset on HF Hub |
| `--env.type` | `libero` | LIBERO simulation environment |
| `--env.task` | `libero_object` | Object manipulation suite |
| `--output_dir` | `./libero_training/checkpoints` | Git-ignored output directory |
| `--batch_size` | `8` | Adjust based on GPU memory |
| `--steps` | `100000` | Total training steps |
| `--eval_freq` | `10000` | Eval in sim every 10k steps |
| `--save_freq` | `10000` | Save checkpoint every 10k steps |
| `--eval.batch_size` | `1` | Parallel eval environments |
| `--eval.n_episodes` | `3` | Episodes per eval round |
| `--log_freq` | `200` | Log metrics every 200 steps |

## Prerequisites

1. **LeRobot installed** with LIBERO extras:
   ```bash
   poetry sync --extra libero   # or: uv sync --extra libero
   ```
2. **`lerobot-train` on PATH** (provided by the install above)
3. **GPU recommended** -- ACT training is compute-intensive. Batch size 8 requires ~16GB VRAM.
4. **HF Hub access** -- The dataset downloads automatically on first run (~several GB).

## Evaluation (after training)

```bash
lerobot-eval \
    --policy.path=./libero_training/checkpoints/<checkpoint_dir> \
    --env.type=libero \
    --env.task=libero_object \
    --eval.batch_size=1 \
    --eval.n_episodes=10
```

## Optional Tuning

- **Learning rate:** Override with `--optimizer.lr=1e-4`
- **Chunk size:** Override with `--policy.chunk_size=100`
- **Mixed precision:** Add `--fp16` or wrap with `accelerate launch`
- **W&B logging:** Add `--wandb.enable=true --wandb.project=libero-act`
- **Resume training:** `--resume=true --config_path=./libero_training/checkpoints/<run>/train_config.json`
- **Different suite:** Change `--env.task` to `libero_spatial`, `libero_goal`, `libero_10`, or `libero_90`

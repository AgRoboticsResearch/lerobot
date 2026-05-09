#!/usr/bin/env python
"""Train SmolVLA in UMI EE mode on sroi_piper_strawberry_picking.

Full UMI pipeline: DeriveStateFromAction → RelativeAction → RelativeState → Normalize.
- State: 2-timestep relative (14D flattened), derived from action column
- Action: relative to current state (7D)
- Image key: observation.images.color (480x640)

50k steps training on 59 episodes / 9918 frames.
"""

import logging
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/sroi_piper_strawberry_picking"
OUTPUT_DIR = Path("/home/hls/codes/lerobot_piper_sroi/outputs/smolvla_umi_strawberry_50k")

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Compute relative state stats (14D) before training, save to disk
# ---------------------------------------------------------------------------
print("=" * 60)
print("Recomputing stats with relative state (14D) for UMI pipeline...")
print(f"Dataset: {DATASET_ROOT}")
print("=" * 60)

ds = LeRobotDataset("sroi_piper_strawberry_picking", root=DATASET_ROOT)
print(f"Before: observation.state stats dim = {ds.meta.stats['observation.state']['mean'].shape}")

ds = recompute_stats(
    ds,
    num_workers=2,
    relative_action=True,
    relative_exclude_joints=["gripper"],
    relative_state=True,
    relative_exclude_state_joints=["gripper"],
    state_obs_steps=2,
    derive_state_from_action=True,
)

state_mean = ds.meta.stats["observation.state"]["mean"]
state_std = ds.meta.stats["observation.state"]["std"]
print(f"After: observation.state stats dim = {state_mean.shape}")
print(f"  mean[:7] (prev - current): {[f'{v:.4f}' for v in state_mean[:7]]}")
print(f"  mean[7:] (current - current): {[f'{v:.4f}' for v in state_mean[7:]]}")
print(f"  std[:7]:  {[f'{v:.4f}' for v in state_std[:7]]}")
print(f"  std[7:]: {[f'{v:.4f}' for v in state_std[7:]]}")
print()

# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------
cfg = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="sroi_piper_strawberry_picking",
        root=DATASET_ROOT,
        video_backend="torchcodec",
    ),
    policy=SmolVLAConfig(
        # Full UMI EE-pose: relative state + relative actions
        derive_state_from_action=True,
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        relative_exclude_state_joints=["gripper"],
        # Training config — from scratch (no pretrained weights)
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        load_vlm_weights=False,
        push_to_hub=False,
        # Resize: 480×640 → pad to square
        resize_imgs_with_padding=(512, 512),
        # Optimizer
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-10,
        optimizer_grad_clip_norm=10,
        # Scheduler
        scheduler_warmup_steps=1000,
        scheduler_decay_steps=50000,
        scheduler_decay_lr=2.5e-6,
    ),
    output_dir=OUTPUT_DIR,
    job_name="smolvla_umi_strawberry_50k",
    seed=1000,
    batch_size=64,
    steps=50000,
    num_workers=4,
    log_freq=200,
    save_freq=5000,
    save_checkpoint=True,
    eval_freq=0,
    wandb=WandBConfig(enable=False),
)

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import train

    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Dataset: sroi_piper_strawberry_picking (59 episodes, 9918 frames)")
    print(f"UMI pipeline: DeriveState → RelativeAction → RelativeState → Normalize")
    print(f"State: 2-timestep relative (14D), Action: relative 7D")
    print(f"Steps: 50000, Batch size: 64")
    print(f"Image: 480×640 → pad to 512×512")
    print()

    train(cfg)

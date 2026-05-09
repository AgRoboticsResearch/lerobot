#!/usr/bin/env python
"""Train SmolVLA with UMI EE-pose relative actions on sroi_lab_picking_all.

20k steps smoke test to verify the full training pipeline works.
"""

import dataclasses
import sys
from pathlib import Path

from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

DATASET_ROOT = "/home/hls/codes/lerobot_piper_sroi/Datasets/sroi_lab_picking_all"
OUTPUT_DIR = Path("/home/hls/codes/lerobot_piper_sroi/outputs/smolvla_umi_ee_50k")

cfg = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="sroi_lab_picking_all",
        root=DATASET_ROOT,
        video_backend="torchcodec",
    ),
    policy=SmolVLAConfig(
        # UMI EE-pose relative actions
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        # Training config — from scratch
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        load_vlm_weights=False,
        # Don't push to hub (local training)
        push_to_hub=False,
        # Resize to 512x512 (dataset is 480x848)
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
    job_name="smolvla_umi_ee_50k",
    seed=1000,
    batch_size=16,
    steps=50000,
    num_workers=4,
    log_freq=200,
    save_freq=5000,
    save_checkpoint=True,
    eval_freq=0,  # no sim env, eval separately
    wandb=WandBConfig(enable=False),
)


if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import train

    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Policy: SmolVLA with use_relative_actions=True")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Steps: 50000, Batch size: 16")
    print()

    train(cfg)

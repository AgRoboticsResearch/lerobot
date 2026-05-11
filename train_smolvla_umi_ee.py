#!/usr/bin/env python
"""Train SmolVLA in UMI EE mode on any local dataset.

Full UMI pipeline: DeriveStateFromAction → RelativeAction → RelativeState → Normalize.
- State: 2-timestep relative (14D flattened), derived from action column
- Action: relative to current state (7D)
- Image key: observation.images.color (480x640)

Usage:
  # Smoke test (default: 1000 steps, batch=8)
  conda run -n lerobot_piper_sroi python lerobot/train_smolvla_umi_ee.py \
      --dataset_root Datasets/test_ee_dataset

  # Full training
  conda run -n lerobot_piper_sroi python lerobot/train_smolvla_umi_ee.py \
      --dataset_root Datasets/sroi_piper_strawberry_picking \
      --steps 50000 --batch_size 64 --num_workers 4

  # New dataset
  conda run -n lerobot_piper_sroi python lerobot/train_smolvla_umi_ee.py \
      --dataset_root Datasets/sroi_piper_260505 \
      --steps 50000 --batch_size 64
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

PROJECT_ROOT = Path("/home/hls/codes/lerobot_piper_sroi")

_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
logging.basicConfig(level=logging.INFO, handlers=[_handler])

logger = logging.getLogger("train_smolvla_umi_ee")


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA UMI EE mode on a local dataset")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Dataset path (relative to project root or absolute), e.g. Datasets/my_dataset")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="Dataset repo ID (default: derived from folder name)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/<repo_id>_<steps>k)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--decay_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_root)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    repo_id = args.repo_id or dataset_path.name

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        step_label = f"{args.steps // 1000}k" if args.steps >= 1000 else str(args.steps)
        output_dir = PROJECT_ROOT / "outputs" / f"{repo_id}_{step_label}"

    print("=" * 60)
    print("Recomputing stats with relative state (14D) for UMI pipeline...")
    print(f"Dataset: {dataset_path}")
    print(f"Repo ID: {repo_id}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    t0 = time.time()
    logger.info("Loading dataset from %s ...", dataset_path)
    ds = LeRobotDataset(repo_id, root=str(dataset_path))
    logger.info("Dataset loaded: %d episodes, %d frames, fps=%.1f", ds.num_episodes, ds.num_frames, ds.fps)

    logger.info("Features: %s", list(ds.meta.features.keys()))
    for feat_name, feat_info in ds.meta.features.items():
        if feat_info["dtype"] not in ("image", "video", "string"):
            logger.info("  %s: shape=%s, dtype=%s", feat_name, feat_info.get("shape"), feat_info.get("dtype"))

    logger.info("Before stats recompute:")
    for key in ("observation.state", "action"):
        if key in ds.meta.stats:
            s = ds.meta.stats[key]
            logger.info("  %s: mean.shape=%s, mean=[%s], std=[%s]",
                        key, s["mean"].shape,
                        ", ".join(f"{v:.4f}" for v in s["mean"]),
                        ", ".join(f"{v:.4f}" for v in s["std"]))

    logger.info("Calling recompute_stats(relative_action=True, relative_state=True, derive_state_from_action=True) ...")
    ds = recompute_stats(
        ds,
        num_workers=2,
        relative_action=True,
        relative_exclude_joints=["gripper"],
        relative_state=True,
        relative_exclude_state_joints=["gripper"],
        state_obs_steps=2,
        derive_state_from_action=True,
        pose_dim=6,
    )
    stats_time = time.time() - t0
    logger.info("Stats recomputation finished in %.1f seconds", stats_time)

    for key in ("observation.state", "action"):
        if key in ds.meta.stats:
            s = ds.meta.stats[key]
            logger.info("After recompute - %s: mean.shape=%s", key, s["mean"].shape)
            logger.info("  mean=[%s]", ", ".join(f"{v:.4f}" for v in s["mean"]))
            logger.info("  std=[%s]", ", ".join(f"{v:.4f}" for v in s["std"]))
            logger.info("  min=[%s]", ", ".join(f"{v:.4f}" for v in s["min"]))
            logger.info("  max=[%s]", ", ".join(f"{v:.4f}" for v in s["max"]))
            logger.info("  q01=[%s]", ", ".join(f"{v:.4f}" for v in s["q01"]))
            logger.info("  q99=[%s]", ", ".join(f"{v:.4f}" for v in s["q99"]))

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=repo_id,
            root=str(dataset_path),
            video_backend="torchcodec",
        ),
        policy=SmolVLAConfig(
            derive_state_from_action=True,
            use_relative_actions=True,
            pose_dim=6,
            relative_exclude_joints=["gripper"],
            relative_exclude_state_joints=["gripper"],
            freeze_vision_encoder=True,
            train_expert_only=True,
            train_state_proj=True,
            load_vlm_weights=False,
            push_to_hub=False,
            resize_imgs_with_padding=(512, 512),
            optimizer_lr=args.lr,
            optimizer_weight_decay=1e-10,
            optimizer_grad_clip_norm=10,
            scheduler_warmup_steps=args.warmup_steps,
            scheduler_decay_steps=args.decay_steps,
            scheduler_decay_lr=2.5e-6,
        ),
        output_dir=output_dir,
        job_name=f"{repo_id}_{args.steps}",
        seed=args.seed,
        batch_size=args.batch_size,
        steps=args.steps,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        save_checkpoint=True,
        eval_freq=0,
        wandb=WandBConfig(enable=False),
    )

    from lerobot.scripts.lerobot_train import train

    print()
    print("=" * 60)
    print("Starting training")
    print(f"  Output dir:  {output_dir}")
    print(f"  Dataset:     {dataset_path} ({ds.num_episodes} episodes, {ds.num_frames} frames)")
    print(f"  Pipeline:    DeriveState → RelativeAction → RelativeState → Normalize")
    print(f"  State:       2-timestep relative (14D), Action: relative 7D")
    print(f"  Steps:       {args.steps}, Batch size: {args.batch_size}")
    print(f"  LR:          {args.lr}, Warmup: {args.warmup_steps}, Decay: {args.decay_steps}")
    print(f"  Log freq:    {args.log_freq}, Save freq: {args.save_freq}")
    print(f"  Image:       480x640 → pad to 512x512")
    print(f"  Stats time:  {stats_time:.1f}s")
    print("=" * 60)
    print()

    train(cfg)


if __name__ == "__main__":
    main()

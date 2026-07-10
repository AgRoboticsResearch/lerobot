#!/usr/bin/env python

"""Training script for ACT policy with UMI-style relative EE actions via processor pipeline.

Uses UMI-style architecture:
- Standard LeRobotDataset (no RelativeEEDataset)
- derive_state_from_action: state derived from action column
- Processor pipeline: DeriveState → RelativeRot6dActions → RelativeRot6dState → Normalize
- 10D rot6d output (same math as Pattern A)

Usage:
    PYTHONPATH=../../src python train_relative_ee_processor.py \
      --dataset.repo_id=lerobot_sroi_v2 \
      --dataset.root=/path/to/dataset \
      --policy.type=act \
      --policy.derive_state_from_action=true \
      --policy.use_relative_actions=true \
      --policy.pose_dim=6 \
      --policy.use_rot6d=true \
      --steps=500000 --batch_size=8
"""

import logging

import torch

import lerobot.datasets.factory as factory_module
import lerobot.scripts.lerobot_train as train_module
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.relative_action_stats import recompute_stats

train = train_module.train

def _make_dataset_wrapper(cfg: TrainPipelineConfig):
    """Create dataset with relative rot6d action stats."""
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )

    if not hasattr(cfg.policy, 'fps'):
        cfg.policy.fps = ds_meta.fps
        logging.info(f"Set policy.fps = {ds_meta.fps} Hz (from dataset metadata)")

    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
    )

    logging.info(f"Created LeRobotDataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    logging.info(f"  Action shape: {dataset.meta.info['features']['action']['shape']}")

    use_relative_actions = getattr(cfg.policy, 'use_relative_actions', False)
    derive_state_from_action = getattr(cfg.policy, 'derive_state_from_action', False)
    prediction_horizon = (
        cfg.policy.horizon if cfg.policy.type == "diffusion" else cfg.policy.chunk_size
    )

    if use_relative_actions:
        logging.info("Recomputing stats for relative rot6d actions...")
        recompute_stats(
            dataset,
            relative_action=True,
            relative_exclude_joints=cfg.policy.relative_exclude_joints,
            chunk_size=prediction_horizon,
            num_workers=2,
            derive_state_from_action=derive_state_from_action,
            write_to_disk=False,
        )

        old_shape = dataset.meta.info['features']['action']['shape']
        dataset.meta.info['features']['action']['shape'] = [10]
        logging.info(f"  Updated action metadata shape: {old_shape} → [10]")

    if derive_state_from_action:
        # Add/update observation.state in metadata with derived shape.
        # This is needed so make_policy creates the model with correct 20D state input,
        # even when the dataset doesn't have an observation.state column.
        state_entry = {
            "dtype": "float32",
            "shape": [20],
            "names": [f"rel_state_{i}" for i in range(20)],
        }
        if "observation.state" in dataset.meta.info["features"]:
            old_state = dataset.meta.info["features"]["observation.state"]["shape"]
            logging.info(f"  Updated state metadata shape: {old_state} → [20]")
        else:
            logging.info("  Added derived observation.state to metadata (shape [20])")
        dataset.meta.info["features"]["observation.state"] = state_entry

        logging.info(f"  Action stats mean shape: {dataset.meta.stats['action']['mean'].shape}")

    use_imagenet_stats = getattr(cfg.dataset, "use_imagenet_stats", True)
    if use_imagenet_stats and cfg.policy.type == "act":
        for key in dataset.meta.camera_keys:
            if key not in dataset.meta.stats:
                dataset.meta.stats[key] = {}
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        logging.info("Applied ImageNet normalization stats")

    return dataset


# Monkey-patch dataset creation so transformed metadata/stats are visible to
# the otherwise standard LeRobot training loop. Policy factories dispatch by type.
train_module.make_dataset = _make_dataset_wrapper

factory_module.make_dataset = _make_dataset_wrapper


@parser.wrap()
def train_with_relative_ee_processor(cfg: TrainPipelineConfig):
    """Train ACT with UMI-style relative EE actions via processor pipeline."""
    use_relative_actions = getattr(cfg.policy, 'use_relative_actions', False)
    derive_state_from_action = getattr(cfg.policy, 'derive_state_from_action', False)
    prediction_horizon = (
        cfg.policy.horizon if cfg.policy.type == "diffusion" else cfg.policy.chunk_size
    )

    logging.info("=" * 80)
    logging.info("Training with UMI-style: Processor Pipeline + rot6d (10D)")
    logging.info("=" * 80)
    logging.info(f"Dataset: {cfg.dataset.repo_id}")
    logging.info(f"Policy: {cfg.policy.type}")
    logging.info(f"prediction_horizon: {prediction_horizon}")
    logging.info(f"derive_state_from_action: {derive_state_from_action}")
    logging.info(f"use_relative_actions: {use_relative_actions}")
    logging.info(f"use_rot6d: {getattr(cfg.policy, 'use_rot6d', False)}")
    logging.info("")
    logging.info("Pipeline: DeriveState → RelativeRot6dActions → RelativeRot6dState → Normalize")
    logging.info("  Input:  7D aa absolute from dataset")
    logging.info("  Output: 10D rot6d relative to model")
    logging.info("  State:  20D (2×10D rot6d relative, flattened)")
    logging.info("=" * 80)

    if use_relative_actions:
        cfg.policy.normalization_mapping["ACTION"] = NormalizationMode.MIN_MAX
        cfg.policy.normalization_mapping["STATE"] = NormalizationMode.MIN_MAX
        logging.info("Normalization: MIN_MAX (UMI-style)")

    train(cfg)


if __name__ == "__main__":
    train_with_relative_ee_processor()

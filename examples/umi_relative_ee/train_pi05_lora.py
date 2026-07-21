#!/usr/bin/env python
"""Fine-tune π0.5 with LoRA on absolute 7D UMI-style EE data.

This entry point adapts datasets whose action column is
``[x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]`` and which do
not need an ``observation.state`` column. It derives the model's 20D state and
10D relative rot6d action in checkpoint-serializable processor steps.
"""

import logging

import torch

import lerobot.scripts.lerobot_train as train_module
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.umi_relative_ee_stats import compute_umi_relative_ee_stats
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_STATE


def _make_umi_dataset(cfg: TrainPipelineConfig, dataset_config=None) -> LeRobotDataset:
    is_validation = dataset_config is not None
    dataset_config = dataset_config or cfg.dataset
    if cfg.policy is None or cfg.policy.type != "pi05":
        raise ValueError("This entry point only supports --policy.type=pi05")
    if not cfg.policy.use_umi_relative_ee:
        raise ValueError("Pass --policy.use_umi_relative_ee=true")
    if dataset_config.streaming:
        raise ValueError("UMI statistics currently require a non-streaming local/Hub dataset")

    metadata = LeRobotDatasetMetadata(
        dataset_config.repo_id,
        root=dataset_config.root,
        revision=dataset_config.revision,
    )
    raw_action = metadata.features.get(ACTION)
    if raw_action is None or tuple(raw_action["shape"]) != (7,):
        shape = None if raw_action is None else raw_action["shape"]
        raise ValueError(f"Expected raw absolute EE action shape [7], got {shape}")

    image_transforms = (
        ImageTransforms(dataset_config.image_transforms) if dataset_config.image_transforms.enable else None
    )
    dataset = LeRobotDataset(
        dataset_config.repo_id,
        root=dataset_config.root,
        episodes=dataset_config.episodes,
        delta_timestamps=resolve_delta_timestamps(cfg.policy, metadata),
        image_transforms=image_transforms,
        revision=dataset_config.revision,
        video_backend=dataset_config.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )

    if not is_validation:
        transformed_stats = compute_umi_relative_ee_stats(dataset.hf_dataset, cfg.policy.chunk_size)
        dataset.meta.stats.update(transformed_stats)

    # make_policy reads these in-memory features after dataset construction.
    dataset.meta.info.features[ACTION] = {
        "dtype": "float32",
        "shape": [10],
        "names": [
            "dx",
            "dy",
            "dz",
            "rot6d_0",
            "rot6d_1",
            "rot6d_2",
            "rot6d_3",
            "rot6d_4",
            "rot6d_5",
            "gripper",
        ],
    }
    dataset.meta.info.features[OBS_STATE] = {
        "dtype": "float32",
        "shape": [20],
        "names": [f"umi_relative_state_{index}" for index in range(20)],
    }

    if dataset_config.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    logging.info(
        "UMI π0.5 %s dataset ready: %d episodes, %d frames, raw action 7D -> model action 10D, "
        "derived state 20D",
        "validation" if is_validation else "training",
        dataset.num_episodes,
        dataset.num_frames,
    )
    return dataset


# The standard trainer imports this symbol directly, so replace that hook while
# retaining its policy, optimizer, checkpoint, PEFT, and distributed logic.
train_module.make_dataset = _make_umi_dataset


@parser.wrap()
def train_pi05_lora(cfg: TrainPipelineConfig) -> None:
    train_module.train(cfg)


if __name__ == "__main__":
    train_pi05_lora()

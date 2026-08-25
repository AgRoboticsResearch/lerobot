#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pprint import pformat

import torch

from lerobot.configs import DatasetConfig, PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, OBS_STATE, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset
from .umi_relative_ee_stats import compute_umi_relative_axis_angle_stats, compute_umi_relative_ee_stats


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(
    cfg: TrainPipelineConfig, dataset_config: DatasetConfig | None = None
) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.
        dataset_config: Optional dataset override used for offline validation.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    is_validation = dataset_config is not None
    dataset_config = dataset_config or cfg.dataset
    trainable_config = cfg.trainable_config
    use_umi_relative_ee = bool(getattr(trainable_config, "use_umi_relative_ee", False))
    if use_umi_relative_ee and trainable_config.type not in {
        "act",
        "diffusion",
        "umi_official_dp",
        "umi_official_transformer_dp",
        "smolvla",
        "pi05",
        "multi_task_dit",
        "lingbot_va",
    }:
        raise ValueError(
            "UMI relative-EE training is supported for policy.type=act, diffusion, umi_official_dp, "
            "umi_official_transformer_dp, smolvla, pi05, multi_task_dit, or lingbot_va."
        )
    if use_umi_relative_ee and dataset_config.streaming:
        raise ValueError("UMI relative-EE statistics require a non-streaming dataset.")

    image_transforms = (
        ImageTransforms(dataset_config.image_transforms) if dataset_config.image_transforms.enable else None
    )

    if isinstance(dataset_config.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            dataset_config.repo_id, root=dataset_config.root, revision=dataset_config.revision
        )
        if use_umi_relative_ee:
            raw_action = ds_meta.features.get(ACTION)
            raw_shape = None if raw_action is None else tuple(raw_action["shape"])
            if raw_shape != (7,):
                raise ValueError(
                    "UMI relative EE requires raw absolute action shape [7] "
                    f"[xyz, axis-angle, gripper], got {raw_shape}."
                )
        delta_timestamps = resolve_delta_timestamps(trainable_config, ds_meta)
        if not dataset_config.streaming:
            dataset = LeRobotDataset(
                dataset_config.repo_id,
                root=dataset_config.root,
                episodes=dataset_config.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_config.revision,
                video_backend=dataset_config.video_backend,
                return_uint8=True,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                dataset_config.repo_id,
                root=dataset_config.root,
                episodes=dataset_config.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_config.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            dataset_config.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=dataset_config.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if use_umi_relative_ee:
        is_lingbot = trainable_config.type == "lingbot_va"
        uses_axis_angle = is_lingbot or (
            trainable_config.type == "smolvla"
            and getattr(trainable_config, "umi_rotation_representation", "rot6d") == "axis_angle"
        )
        chunk_size = getattr(trainable_config, "chunk_size", None) or trainable_config.horizon
        if not is_validation:
            dataset.meta.stats.update(
                compute_umi_relative_axis_angle_stats(
                    dataset.hf_dataset,
                    chunk_size,
                    symmetric_rotation_scale=(trainable_config.type == "smolvla"),
                    identity_state_rot6d=bool(
                        getattr(trainable_config, "umi_rot6d_identity_norm", False)
                    ),
                )
                if uses_axis_angle
                else compute_umi_relative_ee_stats(
                    dataset.hf_dataset,
                    chunk_size,
                    identity_rot6d=bool(
                        getattr(trainable_config, "umi_rot6d_identity_norm", False)
                    ),
                )
            )
        dataset.meta.info.features[ACTION] = {
            "dtype": "float32",
            "shape": [7 if uses_axis_angle else 10],
            "names": (
                ["dx", "dy", "dz", "daxis_angle_x", "daxis_angle_y", "daxis_angle_z", "gripper"]
                if uses_axis_angle
                else [
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
                ]
            ),
        }
        if getattr(trainable_config, "use_proprioception", True):
            dataset.meta.info.features[OBS_STATE] = {
                "dtype": "float32",
                "shape": [20],
                "names": [f"umi_relative_state_{index}" for index in range(20)],
            }
        else:
            # Image-only policies (e.g. ACT `use_proprioception=False`): drop the
            # synthetic state feature so no state lands in the policy inputs.
            dataset.meta.info.features.pop(OBS_STATE, None)
        logging.info(
            "Prepared %s UMI relative-EE dataset for %s: raw action 7D -> model action %s, "
            "derived state 20D",
            "validation" if is_validation else "training",
            trainable_config.type,
            "7D axis-angle" if uses_axis_angle else "10D rot6d",
        )

    if dataset_config.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

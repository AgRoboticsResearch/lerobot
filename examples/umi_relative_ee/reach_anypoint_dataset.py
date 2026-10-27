"""Load Reach AnyPoint planning chunks for the UMI example training entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def train_with_reach_anypoint(cfg):
    """Use the example dataset factory for this training call, then restore it.

    The standard trainer imports its factory into its own module. Override that
    reference for both training and validation without changing library files.
    Datasets without a Reach AnyPoint manifest use the original factory.
    """
    import lerobot.scripts.lerobot_train as train_module

    original_make_dataset = train_module.make_dataset

    def make_example_dataset(cfg, dataset_config=None):
        config = dataset_config if dataset_config is not None else cfg.dataset
        if config.root is not None:
            manifest = Path(config.root) / "annotations" / "reach_anypoint" / "manifest.json"
            if manifest.is_file():
                return make_reach_anypoint_dataset(cfg, dataset_config)
        return original_make_dataset(cfg, dataset_config=dataset_config)

    train_module.make_dataset = make_example_dataset
    try:
        return train_module.train(cfg)
    finally:
        train_module.make_dataset = original_make_dataset


def make_action_window(actions, previous_state, chunk_size: int):
    """Build [previous, current, ...] for existing UMI derive-state processors."""
    actions = np.asarray(actions, dtype=np.float32)
    previous_state = np.asarray(previous_state, dtype=np.float32)
    if chunk_size < 1 or actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError("expected a positive chunk_size and nonempty [N,7] actions")
    if previous_state.shape != (7,) or not np.isfinite(previous_state).all():
        raise ValueError("previous_state must be finite [7] in observation camera_link")
    if not np.isfinite(actions).all():
        raise ValueError("actions must be finite")
    count = min(len(actions), chunk_size)
    targets = np.concatenate([actions[:count], np.repeat(actions[-1:], chunk_size - count, axis=0)])
    window = np.concatenate([previous_state[None], targets])
    padding = np.r_[False, np.arange(chunk_size) >= count]
    return window, padding


class ReachAnyPointDataset(Dataset):
    """Wrap a standard one-observation-per-episode LeRobot dataset.

    Raw action is [H+1,7]: previous plus H targets starting at current pose.
    UmiDeriveStateFromActionStep produces the truthful previous/current state
    and strips the prefix, leaving exactly H actions. No temporal queries cross
    independently generated paths.
    """

    def __init__(self, base_dataset, chunk_size: int):
        self.base_dataset = base_dataset
        self.chunk_size = chunk_size
        self.annotation_dir = Path(base_dataset.root) / "annotations" / "reach_anypoint"
        self.manifest = json.loads((self.annotation_dir / "manifest.json").read_text())
        if self.manifest.get("format") != "reach_anypoint_lerobot_motion_chunks":
            raise ValueError("unsupported ReachAnyPoint annotation format")
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported ReachAnyPoint annotation version")
        if self.manifest.get("action_frame") != "observation_camera_link":
            raise ValueError("expected observation_camera_link action frame")
        if self.manifest.get("fps") != base_dataset.fps:
            raise ValueError("annotation FPS differs from LeRobot FPS")
        self.records = {record["episode_index"]: record for record in self.manifest["samples"]}
        self.meta = base_dataset.meta
        self.episodes = base_dataset.episodes

    def __getattr__(self, name):
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def __len__(self):
        return len(self.base_dataset)

    def _window(self, episode_index):
        record = self.records[int(episode_index)]
        label_path = (self.annotation_dir / record["label_path"]).resolve()
        if not label_path.is_relative_to(self.annotation_dir.resolve()):
            raise ValueError("label path escapes annotation directory")
        with np.load(label_path, allow_pickle=False) as label:
            actions = label["action_camera_ref_7d"]
            times = label["trajectory_time_s"]
            if len(times) != len(actions) or not np.allclose(
                times, np.arange(len(times)) / self.manifest["fps"]
            ):
                raise ValueError(f"invalid execution timestamps: {label_path}")
            if not np.allclose(actions[0, :6], 0, atol=1e-6):
                raise ValueError("first target must be current camera-link identity")
            return make_action_window(actions, label["previous_state_camera_ref_7d"], self.chunk_size)

    def __getitem__(self, index):
        sample = dict(self.base_dataset[index])
        window, padding = self._window(int(sample["episode_index"]))
        sample["action"] = torch.from_numpy(window)
        sample["action_is_pad"] = torch.from_numpy(padding)
        sample.pop("observation.state", None)  # existing processor derives it from prefix
        return sample

    def compute_relative_stats(self, identity_rot6d=False):
        """Compute stats from exactly the unpadded targets returned by this wrapper."""
        from lerobot.datasets.compute_stats import RunningQuantileStats
        from lerobot.datasets.umi_relative_ee_stats import (
            _absolute_aa_to_relative_rot6d,
            _force_identity_rot6d_stats,
        )

        if len(self) < 2:
            raise ValueError("training statistics require at least two planning observations")

        action_stats, state_stats = RunningQuantileStats(), RunningQuantileStats()
        for index in range(len(self)):
            row = self.base_dataset.get_raw_item(index)
            window, padding = self._window(int(row["episode_index"]))
            current = window[1]
            targets = window[1:][~padding[1:]]
            relative = _absolute_aa_to_relative_rot6d(np.broadcast_to(current, targets.shape), targets)
            action_stats.update(relative)
            state = _absolute_aa_to_relative_rot6d(np.broadcast_to(current, (2, 7)), window[:2])
            state_stats.update(state.reshape(1, 20))
        stats = {"action": action_stats.get_statistics(), "observation.state": state_stats.get_statistics()}
        if identity_rot6d:
            _force_identity_rot6d_stats(stats)
        return stats


def make_reach_anypoint_dataset(cfg, dataset_config=None):
    """Example factory compatible with the standard trainer's dataset hook."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.transforms import ImageTransforms

    config = dataset_config or cfg.dataset
    policy = cfg.trainable_config
    if config.streaming or not isinstance(config.repo_id, str):
        raise ValueError("ReachAnyPoint requires a local non-streaming single dataset")
    if not getattr(policy, "use_umi_relative_ee", False) or policy.type not in {
        "act", "smolvla", "pi05", "multi_task_dit"
    }:
        raise ValueError("enable use_umi_relative_ee with an existing supported UMI policy")
    if not config.root:
        raise ValueError("dataset.root is required for planning annotations")
    chunk_size = getattr(policy, "chunk_size", None) or getattr(policy, "horizon", None)
    if not chunk_size or chunk_size < 1:
        raise ValueError("positive policy chunk_size/horizon is required")
    if getattr(policy, "drop_n_last_frames", 0):
        raise ValueError("set policy.drop_n_last_frames=0 for one-observation episodes")
    transforms = ImageTransforms(config.image_transforms) if config.image_transforms.enable else None
    base = LeRobotDataset(
        config.repo_id,
        root=config.root,
        episodes=config.episodes,
        delta_timestamps=None,
        image_transforms=transforms,
        revision=config.revision,
        video_backend=config.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )
    if tuple(base.meta.features["action"]["shape"]) != (7,):
        raise ValueError("on-disk action must be raw 7D")
    # One base row per observation is essential for sampler/index correspondence.
    lengths = np.asarray(base.meta.episodes["length"])
    if not np.all(lengths == 1):
        raise ValueError("ReachAnyPoint base episodes must each contain one observation")
    dataset = ReachAnyPointDataset(base, chunk_size)
    if dataset_config is None:
        dataset.meta.stats.update(
            dataset.compute_relative_stats(
                identity_rot6d=bool(getattr(policy, "umi_rot6d_identity_norm", False))
            )
        )
    dataset.meta.info.features["action"] = {
        "dtype": "float32",
        "shape": [10],
        "names": ["dx", "dy", "dz", *[f"rot6d_{i}" for i in range(6)], "gripper"],
    }
    dataset.meta.info.features["observation.state"] = {
        "dtype": "float32",
        "shape": [20],
        "names": [f"umi_relative_state_{i}" for i in range(20)],
    }
    if config.use_imagenet_stats:
        from lerobot.utils.constants import IMAGENET_STATS

        for key in dataset.meta.camera_keys:
            dataset.meta.stats.setdefault(key, {}).update(
                {k: torch.tensor(v, dtype=torch.float32) for k, v in IMAGENET_STATS.items()}
            )
    return dataset

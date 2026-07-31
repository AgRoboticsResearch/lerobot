#!/usr/bin/env python
"""Policy-neutral open-loop evaluation for UMI relative-EE checkpoints.

This evaluator works with ACT, SmolVLA, and π0.5. It feeds recorded observations
to the policy, decodes each predicted chunk through the checkpoint's saved
postprocessor, and compares the resulting absolute 7D poses with ground truth.

By default every episode is evaluated at 10 evenly spaced, non-padded query
frames. Use ``--episode_indices`` to select a subset explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.umi_relative_ee_processor import axis_angle_to_matrix
from lerobot.utils.collate import lerobot_collate_fn

logger = logging.getLogger(__name__)

ACTION = "action"
CAMERA_KEY = "observation.images.camera"
DEFAULT_TASK = "pick the strawberry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument(
        "--episode_indices",
        type=int,
        nargs="+",
        default=None,
        help="Episodes to evaluate. Omit to evaluate every episode in the dataset.",
    )
    parser.add_argument(
        "--samples_per_episode",
        type=int,
        default=10,
        help="Evenly spaced valid query frames per episode (default: 10).",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/debug/open_loop_eval")
    return parser.parse_args()


def choose_query_indices(
    episode_records: list[dict[str, Any]],
    episode_indices: list[int],
    chunk_size: int,
    samples_per_episode: int,
) -> list[tuple[int, int, int]]:
    """Return ``(dataset_index, episode_index, frame_index)`` query locations.

    A valid UMI query needs ``[t-1, t, ..., t+chunk_size-1]`` within one episode,
    so local frame ``t`` is restricted to ``[1, episode_length-chunk_size]``.
    """
    if samples_per_episode <= 0:
        raise ValueError("samples_per_episode must be positive")

    selected: list[tuple[int, int, int]] = []
    for episode_index in episode_indices:
        record = episode_records[episode_index]
        first_frame = 1
        last_frame = int(record["length"]) - chunk_size
        if last_frame < first_frame:
            logger.warning(
                "Skipping episode %d: length=%d is too short for chunk_size=%d",
                episode_index,
                record["length"],
                chunk_size,
            )
            continue
        count = min(samples_per_episode, last_frame - first_frame + 1)
        local_frames = np.linspace(first_frame, last_frame, num=count, dtype=np.int64)
        for frame_index in np.unique(local_frames):
            selected.append(
                (
                    int(record["dataset_from_index"]) + int(frame_index),
                    episode_index,
                    int(frame_index),
                )
            )
    return selected


def load_policy_and_processors(model_path: str, device: torch.device):
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    if not getattr(policy_config, "use_umi_relative_ee", False):
        raise ValueError(f"{model_path} is not a UMI relative-EE checkpoint")

    if (Path(model_path) / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_path)
        base_name = peft_config.base_model_name_or_path
        if not base_name:
            raise ValueError("LoRA adapter_config.json has no base_model_name_or_path")
        policy_class = get_policy_class(policy_config.type)
        policy = policy_class.from_pretrained(base_name, config=policy_config)
        policy = PeftModel.from_pretrained(policy, model_path, config=peft_config)
    else:
        policy_class = get_policy_class(policy_config.type)
        policy = policy_class.from_pretrained(model_path, local_files_only=True)

    policy.to(device).eval()
    policy_config.device = str(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor, policy_config


def rotation_error_deg(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_rotation = axis_angle_to_matrix(predicted[..., 3:6])
    target_rotation = axis_angle_to_matrix(target[..., 3:6])
    relative = predicted_rotation.transpose(-2, -1) @ target_rotation
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def summarize(samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("No samples were evaluated")

    metric_names = (
        "rotation_chunk_mean_deg",
        "rotation_end_deg",
        "xyz_chunk_mean_m",
        "xyz_end_m",
        "gripper_chunk_mean",
        "gripper_end",
    )
    episode_samples: dict[int, list[dict[str, float]]] = defaultdict(list)
    for sample in samples:
        episode_samples[int(sample["episode_index"])].append(sample)

    frame_weighted = {
        name: float(np.mean([sample[name] for sample in samples])) for name in metric_names
    }
    episode_means = {
        episode: {
            name: float(np.mean([sample[name] for sample in rows])) for name in metric_names
        }
        for episode, rows in episode_samples.items()
    }
    episode_balanced = {
        name: float(np.mean([row[name] for row in episode_means.values()])) for name in metric_names
    }
    return {
        "num_episodes": len(episode_samples),
        "num_samples": len(samples),
        "primary_metric": "episode_balanced.rotation_end_deg",
        "episode_balanced": episode_balanced,
        "sample_weighted": frame_weighted,
        "per_episode": episode_means,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but CUDA is unavailable")

    device = torch.device(args.device)
    model_path = str(Path(args.pretrained_path).resolve())
    dataset_root = Path(args.dataset_root).resolve()
    repo_id = args.repo_id or f"local/{dataset_root.name}"
    policy, preprocessor, postprocessor, policy_config = load_policy_and_processors(
        model_path, device
    )
    if args.num_steps is not None:
        step_field = (
            "num_inference_steps" if policy_config.type in {"pi0", "pi05"} else "num_steps"
        )
        setattr(policy_config, step_field, args.num_steps)
        core_policy = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        setattr(core_policy.config, step_field, args.num_steps)

    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    episode_indices = (
        list(range(metadata.total_episodes))
        if args.episode_indices is None
        else args.episode_indices
    )
    invalid = [
        episode
        for episode in episode_indices
        if episode < 0 or episode >= metadata.total_episodes
    ]
    if invalid:
        raise ValueError(
            f"Episode indices out of range for {metadata.total_episodes} episodes: {invalid}"
        )
    if metadata.episodes is None:
        raise RuntimeError("Dataset metadata does not contain episode records")
    query_indices = choose_query_indices(
        metadata.episodes,
        episode_indices,
        policy_config.chunk_size,
        args.samples_per_episode,
    )
    if not query_indices:
        raise ValueError("No valid query frames were found")

    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        delta_timestamps=resolve_delta_timestamps(policy_config, metadata),
        return_uint8=True,
    )
    logger.info(
        "Evaluating %s on %d/%d episodes and %d query frames",
        policy_config.type,
        len({episode for _, episode, _ in query_indices}),
        metadata.total_episodes,
        len(query_indices),
    )

    samples: list[dict[str, float]] = []
    for sample_index, (dataset_index, episode_index, frame_index) in enumerate(query_indices):
        batch = lerobot_collate_fn([dataset[dataset_index]])
        if CAMERA_KEY in batch and batch[CAMERA_KEY].dtype == torch.uint8:
            batch[CAMERA_KEY] = batch[CAMERA_KEY].to(torch.float32) / 255.0
        if not batch.get("task"):
            batch["task"] = [args.task]
        padding = batch.get("action_is_pad")
        if padding is not None and bool(padding.any()):
            raise RuntimeError(
                f"Internal sampling error: episode {episode_index} frame {frame_index} is padded"
            )

        preprocessor.reset()
        postprocessor.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        torch.manual_seed(args.seed + sample_index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + sample_index)

        ground_truth = batch[ACTION][0, 1:].to(torch.float32)
        with torch.no_grad():
            processed = preprocessor(batch)
            processed.pop(ACTION, None)
            predicted_model = policy.predict_action_chunk(processed)
            predicted = postprocessor(predicted_model).squeeze(0).to(torch.float32).cpu()

        steps = min(len(predicted), len(ground_truth))
        predicted = predicted[:steps]
        ground_truth = ground_truth[:steps].cpu()
        rotation_error = rotation_error_deg(predicted, ground_truth)
        xyz_error = torch.linalg.vector_norm(
            predicted[:, :3] - ground_truth[:, :3], dim=-1
        )
        gripper_error = (predicted[:, 6] - ground_truth[:, 6]).abs()
        samples.append(
            {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "rotation_chunk_mean_deg": float(rotation_error.mean()),
                "rotation_end_deg": float(rotation_error[-1]),
                "xyz_chunk_mean_m": float(xyz_error.mean()),
                "xyz_end_m": float(xyz_error[-1]),
                "gripper_chunk_mean": float(gripper_error.mean()),
                "gripper_end": float(gripper_error[-1]),
            }
        )
        if (sample_index + 1) % 50 == 0 or sample_index + 1 == len(query_indices):
            logger.info("Completed %d/%d query frames", sample_index + 1, len(query_indices))

    report = {
        "policy_type": policy_config.type,
        "checkpoint": model_path,
        "dataset_root": str(dataset_root),
        "dataset_total_episodes": metadata.total_episodes,
        "requested_episode_indices": episode_indices,
        "samples_per_episode": args.samples_per_episode,
        "seed": args.seed,
        "summary": summarize(samples),
        "samples": samples,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"{Path(model_path).parents[2].name}_{Path(model_path).parent.name}"
    report_path = output_dir / f"{checkpoint_name}_open_loop_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Saved %s", report_path)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Policy-neutral open-loop evaluation for UMI relative-EE checkpoints.

This evaluator works with ACT, Diffusion Policy, SmolVLA, and π0.5. It feeds recorded observations
to the policy, decodes each predicted chunk through the checkpoint's saved
postprocessor, and compares the resulting absolute 7D poses with ground truth.

By default every episode is evaluated at 10 evenly spaced, non-padded query
frames. Use ``--episode_indices`` to select a subset explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
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
    parser.add_argument(
        "--video_backend",
        default="pyav",
        choices=("pyav", "torchcodec"),
        help="Decoder backend. PyAV is pinned by default to match the 1459 baseline.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument(
        "--query_min_action_offset",
        type=int,
        default=None,
        help="Override the minimum action delta index used to choose valid query frames.",
    )
    parser.add_argument(
        "--query_max_action_offset",
        type=int,
        default=None,
        help="Override the maximum action delta index used to choose valid query frames.",
    )
    parser.add_argument(
        "--legacy_full_action_noise",
        action="store_true",
        help="Deprecated for SmolVLA/π0.5; disables padded masking only for legacy π0 evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to outputs/research_report/eval_<datetime> so standalone runs don't clobber.",
    )
    return parser.parse_args()


def choose_query_indices(
    episode_records: list[dict[str, Any]],
    episode_indices: list[int],
    min_action_offset: int,
    max_action_offset: int,
    samples_per_episode: int,
) -> list[tuple[int, int, int]]:
    """Return ``(dataset_index, episode_index, frame_index)`` query locations.

    Every requested action offset relative to local frame ``t`` must remain in
    the episode. Explicit bounds let a multi-policy evaluation use the common
    intersection even when policies request different action horizons.
    """
    if samples_per_episode <= 0:
        raise ValueError("samples_per_episode must be positive")
    if min_action_offset > max_action_offset:
        raise ValueError("min_action_offset must not exceed max_action_offset")

    selected: list[tuple[int, int, int]] = []
    for episode_index in episode_indices:
        record = episode_records[episode_index]
        first_frame = max(0, -min_action_offset)
        last_frame = int(record["length"]) - 1 - max_action_offset
        if last_frame < first_frame:
            logger.warning(
                "Skipping episode %d: length=%d is too short for action offsets [%d, %d]",
                episode_index,
                record["length"],
                min_action_offset,
                max_action_offset,
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


def load_policy_and_processors(
    model_path: str,
    device: torch.device,
    legacy_full_action_noise: bool = False,
):
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    if not getattr(policy_config, "use_umi_relative_ee", False):
        raise ValueError(f"{model_path} is not a UMI relative-EE checkpoint")

    if policy_config.type in {"smolvla", "pi05"}:
        policy_config.mask_padded_action_dims_at_inference = False
    elif policy_config.type == "pi0":
        policy_config.mask_padded_action_dims_at_inference = not legacy_full_action_noise

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

    core_policy = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    if policy_config.type in {"smolvla", "pi05"}:
        core_policy.config.mask_padded_action_dims_at_inference = False
    elif policy_config.type == "pi0":
        core_policy.config.mask_padded_action_dims_at_inference = not legacy_full_action_noise

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


def _so3_angle_deg(matrix: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (degrees) of a (..., 3, 3) rotation-matrix batch."""
    cosine = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def within_chunk_jerk(poses: torch.Tensor) -> dict[str, float]:
    """Within-chunk smoothness of a single predicted or GT chunk.

    This is the "jitter" metric: how much the per-step motion *changes* across the
    chunk (the second difference, a.k.a. jerk). It is computed on one prediction in
    isolation (no cross-frame/open-loop accumulation). ~0 means a smooth,
    near-constant-velocity trajectory; large values mean the predicted action
    oscillates within the chunk — the wiggle seen in the video, which the endpoint
    accuracy metrics (``rotation_end_deg`` etc.) cannot see.

    ``poses`` is ``[steps, 7]`` absolute ``[xyz, axis-angle, gripper]``.
    Returns rotation jerk (deg) and xyz jerk (m).
    """
    steps = poses.shape[0]
    if steps < 3:
        return {"rot_jerk_deg": 0.0, "xyz_jerk_m": 0.0}
    rot = axis_angle_to_matrix(poses[:, 3:6])  # [steps, 3, 3]
    step_rot = rot[:-1].transpose(-2, -1) @ rot[1:]  # inter-step rotation
    rot_jerk = _so3_angle_deg(step_rot[:-1].transpose(-2, -1) @ step_rot[1:]).mean()
    step_xyz = poses[1:, :3] - poses[:-1, :3]  # inter-step position delta
    xyz_jerk = (step_xyz[1:] - step_xyz[:-1]).norm(dim=-1).mean()
    return {"rot_jerk_deg": float(rot_jerk), "xyz_jerk_m": float(xyz_jerk)}


def bootstrap_episode_confidence_intervals(
    episode_means: dict[int, dict[str, float]],
    metric_names: tuple[str, ...],
    *,
    confidence: float = 0.95,
    num_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Bootstrap episode-balanced means, treating episodes as independent units."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")

    values = np.asarray(
        [[row[name] for name in metric_names] for row in episode_means.values()], dtype=np.float64
    )
    if len(values) == 0:
        raise ValueError("No episode means were provided")
    rng = np.random.default_rng(seed)
    resampled_indices = rng.integers(0, len(values), size=(num_resamples, len(values)))
    resampled_means = values[resampled_indices].mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(resampled_means, [tail, 1 - tail], axis=0)
    return {
        name: {"low": float(lower[index]), "high": float(upper[index])}
        for index, name in enumerate(metric_names)
    }


def summarize(samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("No samples were evaluated")

    metric_names = (
        "rotation_chunk_mean_deg",
        "rotation_chunk_rmse_deg",
        "rotation_chunk_mse_deg2",
        "rotation_end_deg",
        "xyz_chunk_mean_m",
        "xyz_chunk_rmse_m",
        "xyz_chunk_mse_m2",
        "xyz_end_m",
        "gripper_chunk_mean",
        "gripper_chunk_rmse",
        "gripper_chunk_mse",
        "gripper_end",
        "rot_jerk_deg",
        "xyz_jerk_m",
        "gt_rot_jerk_deg",
        "gt_xyz_jerk_m",
    )
    episode_samples: dict[int, list[dict[str, float]]] = defaultdict(list)
    for sample in samples:
        episode_samples[int(sample["episode_index"])].append(sample)

    frame_weighted = {name: float(np.mean([sample[name] for sample in samples])) for name in metric_names}
    episode_means = {
        episode: {name: float(np.mean([sample[name] for sample in rows])) for name in metric_names}
        for episode, rows in episode_samples.items()
    }
    episode_balanced = {
        name: float(np.mean([row[name] for row in episode_means.values()])) for name in metric_names
    }
    confidence_intervals = bootstrap_episode_confidence_intervals(episode_means, metric_names)
    return {
        "num_episodes": len(episode_samples),
        "num_samples": len(samples),
        "primary_metric": "episode_balanced.rot_jerk_deg",
        "episode_balanced": episode_balanced,
        "episode_balanced_95ci": confidence_intervals,
        "confidence_interval_method": {
            "unit": "episode",
            "resamples": 10_000,
            "confidence": 0.95,
            "seed": 0,
        },
        "sample_weighted": frame_weighted,
        "per_episode": episode_means,
    }


def summarize_inference_latency(seconds: list[float]) -> dict[str, float | int]:
    """Summarize synchronized policy-only latency, excluding one cold call."""
    if not seconds:
        raise ValueError("No inference timings were provided")
    warm = np.asarray(seconds[1:] if len(seconds) > 1 else seconds, dtype=np.float64)
    return {
        "num_warm_samples": len(warm),
        "cold_seconds": seconds[0],
        "mean_seconds": float(warm.mean()),
        "median_seconds": float(np.median(warm)),
        "p95_seconds": float(np.quantile(warm, 0.95)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        import datetime as _dt

        args.output_dir = f"outputs/research_report/eval_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but CUDA is unavailable")

    device = torch.device(args.device)
    model_path = str(Path(args.pretrained_path).resolve())
    dataset_root = Path(args.dataset_root).resolve()
    repo_id = args.repo_id or f"local/{dataset_root.name}"
    policy, preprocessor, postprocessor, policy_config = load_policy_and_processors(
        model_path, device, args.legacy_full_action_noise
    )
    if args.num_steps is not None:
        step_field = (
            "num_inference_steps" if policy_config.type in {"diffusion", "pi0", "pi05"} else "num_steps"
        )
        setattr(policy_config, step_field, args.num_steps)
        core_policy = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        setattr(core_policy.config, step_field, args.num_steps)
        if policy_config.type == "diffusion":
            core_policy.diffusion.num_inference_steps = args.num_steps

    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    episode_indices = (
        list(range(metadata.total_episodes)) if args.episode_indices is None else args.episode_indices
    )
    invalid = [episode for episode in episode_indices if episode < 0 or episode >= metadata.total_episodes]
    if invalid:
        raise ValueError(f"Episode indices out of range for {metadata.total_episodes} episodes: {invalid}")
    if metadata.episodes is None:
        raise RuntimeError("Dataset metadata does not contain episode records")
    action_delta_indices = policy_config.action_delta_indices
    if action_delta_indices is None or not action_delta_indices:
        raise ValueError("UMI relative-EE evaluation requires action delta indices")
    query_min_action_offset = (
        min(action_delta_indices)
        if args.query_min_action_offset is None
        else args.query_min_action_offset
    )
    query_max_action_offset = (
        max(action_delta_indices)
        if args.query_max_action_offset is None
        else args.query_max_action_offset
    )
    query_indices = choose_query_indices(
        metadata.episodes,
        episode_indices,
        query_min_action_offset,
        query_max_action_offset,
        args.samples_per_episode,
    )
    if not query_indices:
        raise ValueError("No valid query frames were found")

    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        delta_timestamps=resolve_delta_timestamps(policy_config, metadata),
        return_uint8=True,
        video_backend=args.video_backend,
    )
    logger.info(
        "Evaluating %s on %d/%d episodes and %d query frames with action-offset bounds [%d, %d]",
        policy_config.type,
        len({episode for _, episode, _ in query_indices}),
        metadata.total_episodes,
        len(query_indices),
        query_min_action_offset,
        query_max_action_offset,
    )
    if policy_config.type in {"smolvla", "pi0", "pi05"}:
        logger.info(
            "Flow action dimensions: active=%d, model=%d, padded masking=%s",
            policy_config.action_feature.shape[0],
            policy_config.max_action_dim,
            policy_config.mask_padded_action_dims_at_inference,
        )

    samples: list[dict[str, float]] = []
    inference_seconds: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            predicted_model = policy.predict_action_chunk(processed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds.append(time.perf_counter() - inference_start)
            predicted = postprocessor(predicted_model).squeeze(0).to(torch.float32).cpu()

        steps = min(len(predicted), len(ground_truth))
        predicted = predicted[:steps]
        ground_truth = ground_truth[:steps].cpu()
        rotation_error = rotation_error_deg(predicted, ground_truth)
        xyz_error = torch.linalg.vector_norm(predicted[:, :3] - ground_truth[:, :3], dim=-1)
        gripper_error = (predicted[:, 6] - ground_truth[:, 6]).abs()
        # Trajectory MSE/RMSE over the chunk (mean / root-mean-square of the
        # squared per-step error). RMSE is in natural units (deg, m); MSE is the
        # raw squared quantity. Kept per-modality — combining deg, m, and gripper
        # into one MSE would be meaningless.
        rotation_mse = float((rotation_error**2).mean())
        xyz_mse = float((xyz_error**2).mean())
        gripper_mse = float((gripper_error**2).mean())
        pred_jerk = within_chunk_jerk(predicted)
        gt_jerk = within_chunk_jerk(ground_truth)
        samples.append(
            {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "rotation_chunk_mean_deg": float(rotation_error.mean()),
                "rotation_chunk_rmse_deg": float(rotation_mse**0.5),
                "rotation_chunk_mse_deg2": rotation_mse,
                "rotation_end_deg": float(rotation_error[-1]),
                "xyz_chunk_mean_m": float(xyz_error.mean()),
                "xyz_chunk_rmse_m": float(xyz_mse**0.5),
                "xyz_chunk_mse_m2": xyz_mse,
                "xyz_end_m": float(xyz_error[-1]),
                "gripper_chunk_mean": float(gripper_error.mean()),
                "gripper_chunk_rmse": float(gripper_mse**0.5),
                "gripper_chunk_mse": gripper_mse,
                "gripper_end": float(gripper_error[-1]),
                "rot_jerk_deg": pred_jerk["rot_jerk_deg"],
                "xyz_jerk_m": pred_jerk["xyz_jerk_m"],
                "gt_rot_jerk_deg": gt_jerk["rot_jerk_deg"],
                "gt_xyz_jerk_m": gt_jerk["xyz_jerk_m"],
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
        "query_action_offset_bounds": {
            "min": query_min_action_offset,
            "max": query_max_action_offset,
        },
        "seed": args.seed,
        "video_backend": args.video_backend,
        "inference_latency_seconds": summarize_inference_latency(inference_seconds),
        "cuda_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "action_dimension_inference": {
            "active_action_dim": policy_config.action_feature.shape[0],
            "model_action_dim": getattr(
                policy_config, "max_action_dim", policy_config.action_feature.shape[0]
            ),
            "mask_padded_action_dims": getattr(policy_config, "mask_padded_action_dims_at_inference", None),
            "legacy_full_action_noise": args.legacy_full_action_noise,
        },
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

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Evaluate RTC with UMI-relative Pi0.5 or SmolVLA on sequential dataset frames.

Unlike ``examples/rtc/eval_dataset.py``, this evaluator preserves the coordinate
frame contract of UMI relative-EE policies. It stores the preceding prediction
as absolute 7D EE targets, advances the action queue, then re-expresses the
leftover tail relative to the current observation before RTC denoising.

Example:
    PYTHONPATH=src uv run python examples/umi_relative_ee/eval_rtc_dataset.py \
        --pretrained_path outputs/train/pi05_umi/checkpoints/500000/pretrained_model \
        --dataset_root /path/to/validation_dataset \
        --episode_indices 0 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import imageio.v3 as iio  # noqa: E402
except ImportError:
    iio = None

from visualize_predictions import (  # noqa: E402
    DEFAULT_EXTRINSICS,
    aa_pose_to_matrix,
    find_camera_info,
    load_K,
    load_tip_kin,
    project_future,
)

from lerobot.configs import RTCAttentionSchedule  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.rtc import ActionQueue, RTCConfig, reanchor_umi_rtc_prefix  # noqa: E402
from lerobot.processor import NormalizerProcessorStep, UmiRelativeActionsStep  # noqa: E402
from lerobot.processor.umi_relative_ee_processor import axis_angle_to_matrix  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402

logger = logging.getLogger(__name__)
CAMERA_KEY = "observation.images.camera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--episode_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--transitions_per_episode", type=int, default=1)
    parser.add_argument(
        "--query_stride",
        type=int,
        default=5,
        help="Actions executed from the previous prediction before the next observation.",
    )
    parser.add_argument("--inference_delay", type=int, default=4)
    parser.add_argument("--execution_horizon", type=int, default=10)
    parser.add_argument("--max_guidance_weight", type=float, default=10.0)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument(
        "--legacy_full_action_noise",
        action="store_true",
        help="Deprecated compatibility no-op; SmolVLA/π0.5 always use full-width OpenPI flow.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default="pick the strawberry")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="outputs/debug/rtc_umi_dataset")
    parser.add_argument("--video_fps", type=float, default=6.0)
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--extrinsics_config", default=DEFAULT_EXTRINSICS)
    parser.add_argument("--camera_info_path", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy_and_processors(
    model_path: str,
    device: torch.device,
    num_steps: int | None,
    legacy_full_action_noise: bool = False,
):
    """Load a full checkpoint or a PEFT adapter and its saved processors."""
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    if policy_config.type not in {"pi05", "smolvla"}:
        raise ValueError(
            "RTC dataset evaluation supports policy.type=pi05 or smolvla; "
            f"got {policy_config.type!r}. Use eval_open_loop_dataset.py for ACT."
        )
    if not getattr(policy_config, "use_umi_relative_ee", False):
        raise ValueError(f"{model_path} is not configured for UMI relative-EE actions")
    policy_config.mask_padded_action_dims_at_inference = False

    if num_steps is not None:
        step_field = "num_inference_steps" if policy_config.type in {"pi0", "pi05"} else "num_steps"
        setattr(policy_config, step_field, num_steps)
        logger.info("Overriding %s=%d", step_field, num_steps)

    policy_class = get_policy_class(policy_config.type)
    if (Path(model_path) / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_path)
        base_name = peft_config.base_model_name_or_path
        if not base_name:
            raise ValueError("LoRA adapter_config.json has no base_model_name_or_path")
        logger.info("Loading LoRA base '%s' and adapter '%s'", base_name, model_path)
        policy = policy_class.from_pretrained(base_name, config=policy_config)
        policy = PeftModel.from_pretrained(policy, model_path, config=peft_config)
    else:
        logger.info("Loading full checkpoint '%s'", model_path)
        policy = policy_class.from_pretrained(model_path, local_files_only=True)

    policy.to(device).eval()
    policy_config.device = str(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    core_policy = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    core_policy.config.mask_padded_action_dims_at_inference = False
    return policy, core_policy, preprocessor, postprocessor, policy_config


def prepare_batch(batch: dict[str, Any], task: str) -> dict[str, Any]:
    result = dict(batch)
    if CAMERA_KEY in result and result[CAMERA_KEY].dtype == torch.uint8:
        result[CAMERA_KEY] = result[CAMERA_KEY].to(torch.float32) / 255.0
    if not result.get("task"):
        result["task"] = [task]
    return result


def collect_sequential_pairs(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    query_stride: int,
    transitions_per_episode: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Collect valid (previous, current) frames from the same episode."""
    wanted = set(episode_indices)
    candidates: dict[int, list[dict[str, Any]]] = {ep: [] for ep in wanted}
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lerobot_collate_fn,
    )
    for batch in loader:
        if batch is None:
            continue
        ep = int(batch["episode_index"][0].item())
        if ep not in wanted:
            continue
        is_pad = batch.get("action_is_pad")
        if is_pad is not None and bool(is_pad[0].any()):
            continue
        candidates[ep].append(batch)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for ep in episode_indices:
        episode_batches = candidates.get(ep, [])
        start = 0
        made = 0
        while start < len(episode_batches) and made < transitions_per_episode:
            previous = episode_batches[start]
            previous_frame = int(previous["frame_index"][0].item())
            current = next(
                (
                    item
                    for item in episode_batches[start + 1 :]
                    if int(item["frame_index"][0].item()) >= previous_frame + query_stride
                ),
                None,
            )
            if current is None:
                break
            pairs.append((previous, current))
            made += 1
            current_frame = int(current["frame_index"][0].item())
            start = next(
                (
                    i
                    for i, item in enumerate(episode_batches)
                    if int(item["frame_index"][0].item()) >= current_frame
                ),
                len(episode_batches),
            )
    if not pairs:
        raise RuntimeError("No valid sequential frame pairs were found")
    return pairs


def fixed_prefix_length(actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    if len(actions) >= target_steps:
        return actions[:target_steps]
    result = torch.zeros(
        (target_steps, actions.shape[-1]),
        dtype=actions.dtype,
        device=actions.device,
    )
    result[: len(actions)] = actions
    return result


def rotation_error_rad(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_rot = axis_angle_to_matrix(first[..., 3:6])
    second_rot = axis_angle_to_matrix(second[..., 3:6])
    relative = first_rot.transpose(-2, -1) @ second_rot
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.acos(cosine)


def pose_metrics(predicted: torch.Tensor, target: torch.Tensor, prefix: str) -> dict[str, float]:
    steps = min(len(predicted), len(target))
    predicted = predicted[:steps].float().cpu()
    target = target[:steps].float().cpu()
    xyz_error = torch.linalg.vector_norm(predicted[:, :3] - target[:, :3], dim=-1)
    rot_error = rotation_error_rad(predicted, target)
    grip_error = (predicted[:, 6] - target[:, 6]).abs()
    return {
        f"{prefix}_xyz_mean_mm": float(xyz_error.mean() * 1000),
        f"{prefix}_xyz_max_mm": float(xyz_error.max() * 1000),
        f"{prefix}_rot_mean_deg": float(torch.rad2deg(rot_error).mean()),
        f"{prefix}_rot_max_deg": float(torch.rad2deg(rot_error).max()),
        f"{prefix}_gripper_mean": float(grip_error.mean()),
    }


def roughness_mm(actions: torch.Tensor) -> float:
    if len(actions) < 3:
        return 0.0
    acceleration = torch.diff(actions[:, :3].float().cpu(), n=2, dim=0)
    return float(torch.linalg.vector_norm(acceleration, dim=-1).mean() * 1000)


def camera_to_rgb(image: torch.Tensor | None) -> np.ndarray:
    if image is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    array = image[0].detach().cpu().numpy() if image.ndim == 4 else image.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array


def project_absolute_path(
    actions: torch.Tensor,
    current_pose: torch.Tensor,
    camera_matrix: np.ndarray,
    tip_kin,
) -> np.ndarray:
    poses = [aa_pose_to_matrix(current_pose.float().cpu().numpy())]
    poses.extend(aa_pose_to_matrix(action) for action in actions.float().cpu().numpy())
    px, py = project_future(np.stack(poses), 0, camera_matrix, tip_kin)
    return np.stack([px, py], axis=-1)


def draw_projected_trajectories(
    image: np.ndarray,
    current_pose: torch.Tensor,
    previous_tail: torch.Tensor,
    no_rtc: torch.Tensor,
    rtc: torch.Tensor,
    ground_truth: torch.Tensor,
    camera_matrix: np.ndarray,
    tip_kin,
) -> tuple[np.ndarray, dict[str, int]]:
    output = image.copy()
    paths = [
        ("previous tail", previous_tail, (255, 165, 0)),
        ("ground truth", ground_truth, (255, 0, 255)),
        ("no RTC", no_rtc, (30, 144, 255)),
        ("RTC", rtc, (0, 255, 0)),
    ]
    height, width = output.shape[:2]
    visible_counts = {}
    for label, actions, color in paths:
        points = project_absolute_path(actions, current_pose, camera_matrix, tip_kin)
        in_bounds = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        visible_counts[label] = int(in_bounds.sum())
        for start, end in zip(points[:-1], points[1:], strict=True):
            if np.isfinite(start).all() and np.isfinite(end).all():
                cv2.line(
                    output,
                    tuple(np.rint(start).astype(int)),
                    tuple(np.rint(end).astype(int)),
                    color,
                    3,
                    cv2.LINE_AA,
                )

    legend_x, legend_y = 12, 24
    cv2.rectangle(output, (5, 5), (205, 112), (0, 0, 0), -1)
    for index, (label, _, color) in enumerate(paths):
        y = legend_y + index * 25
        cv2.line(output, (legend_x, y - 5), (legend_x + 28, y - 5), color, 4, cv2.LINE_AA)
        cv2.putText(
            output,
            label,
            (legend_x + 38, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output, visible_counts


def plot_transition(
    previous_tail: torch.Tensor,
    no_rtc: torch.Tensor,
    rtc: torch.Tensor,
    ground_truth: torch.Tensor,
    camera_rgb: np.ndarray,
    path: Path,
    title: str,
) -> None:
    arrays = {
        "previous tail": previous_tail.float().cpu().numpy(),
        "no RTC": no_rtc.float().cpu().numpy(),
        "RTC": rtc.float().cpu().numpy(),
        "ground truth": ground_truth.float().cpu().numpy(),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    image_ax = axes[0, 0]
    image_ax.imshow(camera_rgb)
    image_ax.set_title("current dataset observation")
    image_ax.axis("off")
    curve_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for dim, ax in enumerate(curve_axes):
        for label, values in arrays.items():
            linestyle = "--" if label == "ground truth" else "-"
            ax.plot(values[:, dim], label=label, linestyle=linestyle)
        ax.set_ylabel(f"{'xyz'[dim]} (m)")
        ax.set_xlabel("action step")
        ax.grid(alpha=0.25)
    curve_axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_noise(core_policy, device: torch.device) -> torch.Tensor:
    shape = (
        1,
        core_policy.config.chunk_size,
        core_policy.config.max_action_dim,
    )
    return core_policy.model.sample_noise(shape, device)


def evaluate_transition(
    policy,
    core_policy,
    preprocessor,
    postprocessor,
    previous_batch: dict[str, Any],
    current_batch: dict[str, Any],
    rtc_config: RTCConfig,
    args: argparse.Namespace,
    transition_index: int,
    output_dir: Path,
    projection,
) -> dict[str, Any]:
    episode = int(current_batch["episode_index"][0].item())
    previous_frame = int(previous_batch["frame_index"][0].item())
    current_frame = int(current_batch["frame_index"][0].item())
    consumed = current_frame - previous_frame

    preprocessor.reset()
    postprocessor.reset()
    policy.reset()

    previous_batch = prepare_batch(previous_batch, args.task)
    with torch.no_grad():
        previous_processed = preprocessor(previous_batch)
        previous_processed.pop("action", None)
        rtc_config.enabled = False
        set_seed(args.seed + transition_index)
        previous_model = policy.predict_action_chunk(previous_processed)
        previous_absolute = postprocessor(previous_model).squeeze(0)

    queue = ActionQueue(RTCConfig(enabled=True))
    queue.merge(previous_model.squeeze(0), previous_absolute, real_delay=0)
    for _ in range(consumed):
        if queue.get() is None:
            raise RuntimeError(f"Previous action chunk exhausted after {consumed} consumed actions")
    _, _, previous_absolute_tail = queue.get_left_over_snapshot()
    if previous_absolute_tail is None or len(previous_absolute_tail) == 0:
        raise RuntimeError("No previous absolute actions remain for RTC")

    current_batch = prepare_batch(current_batch, args.task)
    camera_rgb = camera_to_rgb(current_batch.get(CAMERA_KEY))
    ground_truth = current_batch["action"][0, 1:].clone()
    with torch.no_grad():
        current_processed = preprocessor(current_batch)
        current_processed.pop("action", None)

    umi_step = next(
        (step for step in preprocessor.steps if isinstance(step, UmiRelativeActionsStep) and step.enabled),
        None,
    )
    normalizer_step = next(
        (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
        None,
    )
    if umi_step is None or umi_step.get_cached_state() is None:
        raise RuntimeError("The loaded preprocessor did not cache a UMI current EE state")

    rtc_prefix = reanchor_umi_rtc_prefix(
        prev_actions_absolute=previous_absolute_tail,
        current_state=umi_step.get_cached_state(),
        normalizer_step=normalizer_step,
        policy_device=args.device,
    )
    rtc_prefix = fixed_prefix_length(rtc_prefix, args.execution_horizon)
    decoded_prefix = postprocessor(rtc_prefix.unsqueeze(0)).squeeze(0)

    set_seed(args.seed + 10000 + transition_index)
    noise = sample_noise(core_policy, torch.device(args.device))
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        rtc_config.enabled = False
        no_rtc_model = policy.predict_action_chunk(current_processed, noise=noise.clone())
    torch.cuda.synchronize()
    no_rtc_latency = time.perf_counter() - start

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        rtc_config.enabled = True
        rtc_model = policy.predict_action_chunk(
            current_processed,
            noise=noise.clone(),
            inference_delay=args.inference_delay,
            prev_chunk_left_over=rtc_prefix,
            execution_horizon=args.execution_horizon,
        )
    torch.cuda.synchronize()
    rtc_latency = time.perf_counter() - start

    no_rtc_absolute = postprocessor(no_rtc_model).squeeze(0)
    rtc_absolute = postprocessor(rtc_model).squeeze(0)
    overlap_steps = min(args.execution_horizon, len(previous_absolute_tail))
    previous_overlap = previous_absolute_tail[:overlap_steps]

    metrics: dict[str, Any] = {
        "episode": episode,
        "previous_frame": previous_frame,
        "current_frame": current_frame,
        "consumed_actions": consumed,
        "overlap_steps": overlap_steps,
        "no_rtc_latency_s": no_rtc_latency,
        "rtc_latency_s": rtc_latency,
        "model_space_mean_abs_change": float((rtc_model - no_rtc_model).abs().mean().detach().cpu()),
        "no_rtc_roughness_mm": roughness_mm(no_rtc_absolute),
        "rtc_roughness_mm": roughness_mm(rtc_absolute),
    }
    metrics.update(pose_metrics(decoded_prefix[:overlap_steps], previous_overlap, "reanchor"))
    metrics.update(pose_metrics(no_rtc_absolute[:overlap_steps], previous_overlap, "no_rtc_overlap"))
    metrics.update(pose_metrics(rtc_absolute[:overlap_steps], previous_overlap, "rtc_overlap"))
    metrics.update(pose_metrics(no_rtc_absolute, ground_truth, "no_rtc_gt"))
    metrics.update(pose_metrics(rtc_absolute, ground_truth, "rtc_gt"))
    baseline = metrics["no_rtc_overlap_xyz_mean_mm"]
    metrics["overlap_xyz_improvement_pct"] = (
        0.0 if baseline == 0 else 100 * (baseline - metrics["rtc_overlap_xyz_mean_mm"]) / baseline
    )

    projected_points = None
    if projection is not None:
        camera_matrix, tip_kin = projection
        camera_rgb, projected_points = draw_projected_trajectories(
            camera_rgb,
            ground_truth[0],
            previous_absolute_tail,
            no_rtc_absolute,
            rtc_absolute,
            ground_truth,
            camera_matrix,
            tip_kin,
        )

    plot_path = output_dir / f"episode_{episode}_frames_{previous_frame}_{current_frame}.png"
    plot_transition(
        previous_absolute_tail,
        no_rtc_absolute,
        rtc_absolute,
        ground_truth,
        camera_rgb,
        plot_path,
        f"{core_policy.config.type}: episode {episode}, frames {previous_frame}→{current_frame}",
    )
    metrics["projected_points"] = projected_points
    metrics["plot"] = str(plot_path)
    return metrics


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    numeric_keys = [
        key
        for key, value in results[0].items()
        if isinstance(value, (int, float)) and key not in {"episode", "previous_frame", "current_frame"}
    ]
    return {key: float(np.mean([result[key] for result in results])) for key in numeric_keys}


def write_episode_videos(results: list[dict[str, Any]], output_dir: Path, fps: float) -> list[str]:
    if iio is None:
        raise ImportError("imageio is required for RTC MP4 output")
    episode_results: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        episode_results.setdefault(result["episode"], []).append(result)

    video_paths = []
    for episode, items in episode_results.items():
        items.sort(key=lambda item: item["current_frame"])
        frames = np.stack([iio.imread(item["plot"]) for item in items])
        video_path = output_dir / f"episode_{episode}_rtc_comparison.mp4"
        iio.imwrite(video_path, frames, fps=fps, macro_block_size=1, quality=8)
        video_paths.append(str(video_path))
        logger.info("Saved RTC video to %s (%d frames)", video_path, len(frames))
    return video_paths


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but CUDA is not available")
    if args.execution_horizon > 0 and args.inference_delay > args.execution_horizon:
        raise ValueError("inference_delay must not exceed execution_horizon")

    device = torch.device(args.device)
    model_path = str(Path(args.pretrained_path).resolve())
    dataset_root = Path(args.dataset_root).resolve()
    repo_id = args.repo_id or f"local/{dataset_root.name}"
    model_dir = Path(model_path)
    checkpoint_name = f"{model_dir.parents[2].name}_{model_dir.parent.name}"
    output_dir = Path(args.output_dir) / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, core_policy, preprocessor, postprocessor, policy_config = load_policy_and_processors(
        model_path, device, args.num_steps, args.legacy_full_action_noise
    )
    logger.info(
        "Loaded %s on %s (chunk=%d, steps=%s)",
        policy_config.type,
        device,
        policy_config.chunk_size,
        getattr(policy_config, "num_inference_steps", getattr(policy_config, "num_steps", None)),
    )
    logger.info(
        "Flow action dimensions: active=%d, model=%d, padded masking=%s",
        policy_config.action_feature.shape[0],
        policy_config.max_action_dim,
        policy_config.mask_padded_action_dims_at_inference,
    )
    if args.execution_horizon > policy_config.chunk_size:
        raise ValueError("execution_horizon must not exceed policy chunk_size")

    rtc_config = RTCConfig(
        enabled=False,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    core_policy.config.rtc_config = rtc_config
    core_policy.init_rtc_processor()

    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=args.episode_indices,
        delta_timestamps=resolve_delta_timestamps(policy_config, metadata),
        return_uint8=True,
    )
    projection = None
    if args.project:
        camera_info = Path(args.camera_info_path) if args.camera_info_path else find_camera_info(dataset_root)
        if camera_info is None:
            raise ValueError("--project requires camera intrinsics; none were found under dataset meta")
        camera_matrix = load_K(camera_info)
        if camera_matrix is None:
            raise ValueError(f"Could not load camera intrinsics from {camera_info}")
        projection = (camera_matrix, load_tip_kin(args.extrinsics_config))
        logger.info(
            "Projection enabled: extrinsics=%s intrinsics=%s",
            args.extrinsics_config,
            camera_info,
        )
    pairs = collect_sequential_pairs(
        dataset,
        args.episode_indices,
        args.query_stride,
        args.transitions_per_episode,
    )
    logger.info("Evaluating %d sequential transition(s) from %s", len(pairs), dataset_root)

    results = []
    for index, (previous, current) in enumerate(pairs):
        result = evaluate_transition(
            policy,
            core_policy,
            preprocessor,
            postprocessor,
            previous,
            current,
            rtc_config,
            args,
            index,
            output_dir,
            projection,
        )
        results.append(result)
        logger.info(
            "episode=%d frames=%d→%d reanchor=%.6fmm overlap xyz: %.2f→%.2fmm (%+.1f%%)",
            result["episode"],
            result["previous_frame"],
            result["current_frame"],
            result["reanchor_xyz_max_mm"],
            result["no_rtc_overlap_xyz_mean_mm"],
            result["rtc_overlap_xyz_mean_mm"],
            result["overlap_xyz_improvement_pct"],
        )

    video_paths = write_episode_videos(results, output_dir, args.video_fps)

    report = {
        "policy_type": policy_config.type,
        "checkpoint": model_path,
        "dataset_root": str(dataset_root),
        "device": str(device),
        "action_dimension_inference": {
            "active_action_dim": policy_config.action_feature.shape[0],
            "model_action_dim": policy_config.max_action_dim,
            "mask_padded_action_dims": policy_config.mask_padded_action_dims_at_inference,
            "legacy_full_action_noise": args.legacy_full_action_noise,
        },
        "videos": video_paths,
        "rtc": {
            "inference_delay": args.inference_delay,
            "execution_horizon": args.execution_horizon,
            "max_guidance_weight": args.max_guidance_weight,
            "prefix_attention_schedule": "EXP",
        },
        "results": results,
        "mean": aggregate_metrics(results),
    }
    report_path = output_dir / "rtc_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Saved metrics to %s", report_path)
    print(json.dumps(report["mean"], indent=2))


if __name__ == "__main__":
    main()

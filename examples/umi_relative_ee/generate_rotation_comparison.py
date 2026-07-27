#!/usr/bin/env python
"""Generate raw 9D and SO(3) GT-vs-prediction figures for a UMI policy."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.collate import lerobot_collate_fn


CHANNELS = ("dx", "dy", "dz", "r00", "r01", "r02", "r10", "r11", "r12")


def load_visualizer(path: Path):
    spec = importlib.util.spec_from_file_location("umi_visualize_predictions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gt_relative_raw9(vis, reference: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_inv = np.linalg.inv(vis.aa_pose_to_matrix(reference))
    transforms = np.stack([reference_inv @ vis.aa_pose_to_matrix(target) for target in targets])
    raw9 = np.concatenate([transforms[:, :3, 3], transforms[:, :2, :3].reshape(-1, 6)], axis=1)
    return raw9, transforms[:, :3, :3]


def save_raw_csv(path: Path, records: list[dict]) -> None:
    fields = ["episode", "frame", "chunk_step", "rotation_error_deg"]
    fields += [f"gt_{name}" for name in CHANNELS]
    fields += [f"pred_{name}" for name in CHANNELS]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for step, (gt, pred, error) in enumerate(
                zip(record["gt_raw9"], record["pred_raw9"], record["chunk_rotation_error_deg"])
            ):
                row = {
                    "episode": record["episode"],
                    "frame": record["frame"],
                    "chunk_step": step,
                    "rotation_error_deg": float(error),
                }
                row.update({f"gt_{name}": float(value) for name, value in zip(CHANNELS, gt)})
                row.update({f"pred_{name}": float(value) for name, value in zip(CHANNELS, pred)})
                writer.writerow(row)


def plot_raw9_representative(out_dir: Path, records: list[dict]) -> list[Path]:
    paths = []
    for episode in sorted({record["episode"] for record in records}):
        ep_records = [record for record in records if record["episode"] == episode]
        errors = np.asarray([record["end_rotation_error_deg"] for record in ep_records])
        median_error = float(np.median(errors))
        record = min(ep_records, key=lambda item: abs(item["end_rotation_error_deg"] - median_error))

        fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True, constrained_layout=True)
        x = np.arange(len(record["gt_raw9"]))
        for index, (axis, name) in enumerate(zip(axes.flat, CHANNELS)):
            axis.plot(x, record["gt_raw9"][:, index], color="#111827", linestyle="--", linewidth=2, label="GT")
            axis.plot(x, record["pred_raw9"][:, index], color="#2563eb", linewidth=2, label="Prediction")
            axis.set_title(name)
            axis.grid(alpha=0.25)
            if index < 3:
                axis.set_ylabel("meters")
            else:
                axis.set_ylabel("raw rot6d")
            if index >= 6:
                axis.set_xlabel("action-chunk step")
        axes.flat[0].legend(loc="best")
        fig.suptitle(
            f"Episode {episode}, frame {record['frame']}: raw unnormalized 9D relative pose\n"
            "xyz + first two rows of R_rel; representative frame nearest the episode median endpoint error "
            f"({record['end_rotation_error_deg']:.2f}°)",
            fontsize=13,
        )
        path = out_dir / f"raw_9d_gt_vs_prediction_episode_{episode}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_rotation_summary(out_dir: Path, records: list[dict]) -> Path:
    episodes = sorted({record["episode"] for record in records})
    fig, axes = plt.subplots(4, len(episodes), figsize=(6 * len(episodes), 12), sharex="col", constrained_layout=True)
    if len(episodes) == 1:
        axes = axes[:, None]

    for column, episode in enumerate(episodes):
        ep_records = [record for record in records if record["episode"] == episode]
        frame = np.asarray([record["frame"] for record in ep_records])
        gt_rotvec = np.asarray([record["gt_end_rotvec_deg"] for record in ep_records])
        pred_rotvec = np.asarray([record["pred_end_rotvec_deg"] for record in ep_records])
        errors = np.asarray([record["end_rotation_error_deg"] for record in ep_records])

        for row, label in enumerate(("rotvec x", "rotvec y", "rotvec z")):
            axis = axes[row, column]
            axis.plot(frame, gt_rotvec[:, row], color="#111827", linestyle="--", linewidth=1.8, label="GT")
            axis.plot(frame, pred_rotvec[:, row], color="#2563eb", linewidth=1.8, label="Prediction")
            axis.set_ylabel(f"{label} (deg)")
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_title(f"Episode {episode}")
                axis.legend(loc="best")

        error_axis = axes[3, column]
        error_axis.plot(frame, errors, color="#dc2626", linewidth=1.8)
        error_axis.fill_between(frame, 0, errors, color="#fecaca", alpha=0.6)
        error_axis.axhline(errors.mean(), color="#991b1b", linestyle="--", linewidth=1.2)
        error_axis.set_ylabel("SO(3) error (deg)")
        error_axis.set_xlabel("dataset frame")
        error_axis.set_ylim(bottom=0)
        error_axis.grid(alpha=0.25)
        error_axis.text(
            0.02,
            0.95,
            f"mean={errors.mean():.2f}°\nmedian={np.median(errors):.2f}°",
            transform=error_axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig.suptitle(
        "GT vs predicted endpoint rotation for each 30-step action chunk\n"
        "Rotation-vector components show direction; geodesic SO(3) error is coordinate-invariant",
        fontsize=14,
    )
    path = out_dir / "rotation_gt_vs_prediction_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualizer", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="pick the strawberry")
    args = parser.parse_args()

    vis = load_visualizer(args.visualizer)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1000)

    policy, preprocessor, policy_config = vis.load_policy_and_processors(args.checkpoint, device)
    action_stats = vis.extract_action_stats(preprocessor)
    action_norm_mode = policy_config.normalization_mapping["ACTION"]
    repo_id = args.dataset_root.name
    meta = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy_config, meta)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
        episodes=args.episodes,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lerobot_collate_fn)

    records = []
    for batch in loader:
        if batch is None:
            continue
        is_pad = batch.get("action_is_pad")
        if is_pad is not None and bool(is_pad[0].any()):
            continue
        if vis.CAMERA_KEY in batch and batch[vis.CAMERA_KEY].dtype == torch.uint8:
            batch[vis.CAMERA_KEY] = batch[vis.CAMERA_KEY].to(torch.float32) / 255.0
        if not batch.get("task"):
            batch["task"] = [args.task]

        gt_absolute = batch["action"][0].cpu().numpy()
        gt_reference = gt_absolute[1]
        gt_targets = gt_absolute[1:]
        gt_raw9, gt_rotations = gt_relative_raw9(vis, gt_reference, gt_targets)

        preprocessor.reset()
        with torch.inference_mode():
            processed = preprocessor(batch)
            if policy_config.type == "act":
                processed.pop("action", None)
            prediction = policy.predict_action_chunk(processed)
        pred_relative = vis.unnormalize_actions(prediction, action_stats, action_norm_mode)[0].cpu().numpy()
        pred_raw9 = pred_relative[:, :9]
        pred_rotations = np.stack([vis.rot6d_to_matrix(action)[:3, :3] for action in pred_relative])
        rotation_errors = np.degrees(
            [vis.rot_angle_from_matrix(pred.T @ gt) for pred, gt in zip(pred_rotations, gt_rotations)]
        )

        records.append(
            {
                "episode": int(batch["episode_index"][0].item()),
                "frame": int(batch["frame_index"][0].item()),
                "gt_raw9": gt_raw9,
                "pred_raw9": pred_raw9,
                "chunk_rotation_error_deg": np.asarray(rotation_errors),
                "gt_end_rotvec_deg": np.degrees(Rotation.from_matrix(gt_rotations[-1]).as_rotvec()),
                "pred_end_rotvec_deg": np.degrees(Rotation.from_matrix(pred_rotations[-1]).as_rotvec()),
                "end_rotation_error_deg": float(rotation_errors[-1]),
            }
        )

    if not records:
        raise RuntimeError("No valid dataset frames were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "raw_9d_gt_vs_prediction.csv"
    save_raw_csv(csv_path, records)
    summary_path = plot_rotation_summary(args.output_dir, records)
    raw_paths = plot_raw9_representative(args.output_dir, records)
    print(f"Saved {summary_path}")
    for path in raw_paths:
        print(f"Saved {path}")
    print(f"Saved {csv_path} ({len(records)} frames)")


if __name__ == "__main__":
    main()

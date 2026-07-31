#!/usr/bin/env python
"""Measure within-chunk action-to-action jitter (jerk) for predicted vs GT chunks.

Reuses visualize_predictions geometry/loading. For each valid frame it predicts one
chunk, converts pred + GT to 3D point trajectories in the chunk-start frame, and
reports the mean second-difference (jerk) of those trajectories -- lower = smoother
within a single prediction. Translation in meters, rotation in radians.
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from visualize_predictions import (
    CAMERA_KEY,
    DEFAULT_TASK,
    extract_action_stats,
    gt_abs_to_rel_traj,
    gt_rel_angles,
    load_policy_and_processors,
    pred_rel_angles,
    rel_actions_to_traj,
    unnormalize_actions,
)
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.utils.collate import lerobot_collate_fn


def jerk(pts: np.ndarray) -> float:
    """Mean L2 norm of the second difference (jerk) over a [T, D] trajectory."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return float("nan")
    d2 = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]
    return float(np.mean(np.linalg.norm(d2, axis=-1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--episode_indices", type=int, nargs="+", required=True)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--max_frames", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo_id = os.path.basename(args.dataset_root)
    policy, preprocessor, pcfg = load_policy_and_processors(args.pretrained_path, device)
    action_stats = extract_action_stats(preprocessor)
    action_norm_mode = pcfg.normalization_mapping["ACTION"]
    meta = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(pcfg, meta)
    eps = [e for e in args.episode_indices if e < meta.total_episodes]
    ds = LeRobotDataset(
        repo_id=repo_id, root=args.dataset_root, delta_timestamps=delta_ts,
        return_uint8=True, episodes=eps,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lerobot_collate_fn)

    tp, tg, rp, rg = [], [], [], []
    n = 0
    for batch in loader:
        if batch is None:
            continue
        is_pad = batch.get("action_is_pad")
        if is_pad is not None and bool(is_pad[0].any()):
            continue
        if args.max_frames and n >= args.max_frames:
            break
        if CAMERA_KEY in batch and batch[CAMERA_KEY].dtype == torch.uint8:
            batch[CAMERA_KEY] = batch[CAMERA_KEY].to(torch.float32) / 255.0
        if not batch.get("task"):
            batch["task"] = [args.task]

        gt_abs = batch["action"][0].cpu().numpy()
        gt_ref = gt_abs[1]
        gt_chunk = gt_abs[1:]
        gt_traj = gt_abs_to_rel_traj(gt_chunk, gt_ref)   # [chunk+1, 3] meters
        gt_ang = gt_rel_angles(gt_chunk, gt_ref)          # [chunk] rad

        preprocessor.reset()
        with torch.no_grad():
            processed = preprocessor(batch)
            if pcfg.type == "act":
                processed.pop("action", None)
            pred = policy.predict_action_chunk(processed)
        pred_rel = unnormalize_actions(pred, action_stats, action_norm_mode)[0].cpu().numpy()
        pred_traj = rel_actions_to_traj(pred_rel)         # [chunk+1, 3] meters
        pred_ang = pred_rel_angles(pred_rel)              # [chunk] rad

        tp.append(jerk(pred_traj)); tg.append(jerk(gt_traj))
        rp.append(jerk(pred_ang)); rg.append(jerk(gt_ang))
        n += 1

    def m(x):
        return st.mean(x) if x else float("nan")
    mt, mg = m(tp), m(tg)
    mr, mrg = m(rp), m(rg)
    print(f"n_frames={n}")
    print(f"TRANS jerk(m)   pred={mt:.6f}  gt={mg:.6f}  pred/gt={mt/mg if mg else float('nan'):.2f}")
    print(f"ROT   jerk(rad) pred={mr:.6f}  gt={mrg:.6f}  pred/gt={mr/mrg if mrg else float('nan'):.2f}")


if __name__ == "__main__":
    main()

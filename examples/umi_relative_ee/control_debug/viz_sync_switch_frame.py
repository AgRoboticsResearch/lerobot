#!/usr/bin/env python
"""Render the visualize_predictions overlay + a per-dim raw-prediction figure for the
SYNC mock's chunk-switch frame(s).

Reuses visualize_predictions.py's rendering (policy load, relative-trajectory + per-dim
panels) but renders ONLY the requested dataset frame(s) to PNG — no mp4/imageio. Also
emits a 7-panel absolute predicted-vs-GT figure for that frame's chunk (the "raw
prediction at each dim" view), which exposes the chunk-switch discontinuity.

Usage:
  uv run python examples/umi_relative_ee/control_debug/viz_sync_switch_frame.py \
      --pretrained_path outputs/train/act_umi_identity_rot6d_1459/checkpoints/last/pretrained_model \
      --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
      --episode 0 --frames 30 60 \
      --out outputs/research_report/low_level_control_debug/episode_debug/mock_sync_20260808_140112
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# examples/umi_relative_ee/ is the parent of this file's dir → put it on sys.path so
# `import visualize_predictions` works regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.utils.collate import lerobot_collate_fn

import visualize_predictions as vp
from visualize_predictions import (
    CAMERA_KEY, extract_action_stats, gt_abs_to_rel_traj,
    gt_rel_angles, image_to_rgb_uint8, load_policy_and_processors,
    rel_actions_to_traj, render_frame, unnormalize_actions,
)

EE_DIM_TITLES = ["x [m]", "y [m]", "z [m]", "rot_x [rad]", "rot_y [rad]", "rot_z [rad]", "gripper"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="+", required=True,
                   help="Dataset frame index(es) at chunk switches to render.")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default="pick the strawberry")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--anchor_from_log", default=None,
                   help="Control-log stem (no suffix). If set, override the policy's state "
                        "with the dead-reckoned pose from this log (see --anchor_tick), "
                        "instead of the dataset GT — for dead-reckon switch visualization.")
    p.add_argument("--anchor_tick", type=int, default=29,
                   help="Executed-tick index in --anchor_from_log whose action_agg is the "
                        "dead-reckoned anchor pose (e.g. the last cmd of the previous chunk).")
    p.add_argument("--project", action="store_true",
                   help="Draw predicted + GT gripper-tip trajectories on the camera image "
                        "(needs hand-eye extrinsics + intrinsics K).")
    p.add_argument("--extrinsics_config", default=str(Path(__file__).resolve().parents[1] / "camera_gripper_extrinsics_sroi_v2_d405.json"))
    p.add_argument("--camera_info_path", default=None,
                   help="camera_info_color.json for intrinsics K (auto-found under dataset meta/ if omitted).")
    return p.parse_args()


def make_composite_fig(policy_type, ckpt, repo_id):
    fig = plt.figure(figsize=(15, 8))
    gs_outer = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.08,
                                left=0.02, right=0.98, top=0.93, bottom=0.04)
    ax_img = fig.add_subplot(gs_outer[0, 0])
    gs_right = gs_outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.35)
    ax_3d = fig.add_subplot(gs_right[0], projection="3d")
    gs_curves = gs_right[1].subgridspec(2, 2, hspace=0.65, wspace=0.30)
    axes = (ax_img, ax_3d, fig.add_subplot(gs_curves[0, 0]), fig.add_subplot(gs_curves[0, 1]),
            fig.add_subplot(gs_curves[1, 0]), fig.add_subplot(gs_curves[1, 1]))
    fig.suptitle(f"{policy_type.upper()}  chunk-switch prediction  ·  ckpt {ckpt}  ·  {repo_id}", fontsize=12, y=0.99)
    return fig, axes


def per_dim_figure(pred_abs, gt_abs, frame, out, tag):
    """7-panel absolute predicted-vs-GT over the chunk steps (the per-dim raw view)."""
    n = min(len(pred_abs), len(gt_abs))
    xs = np.arange(n)
    fig, axes = plt.subplots(7, 1, figsize=(8, 11), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(xs, pred_abs[:n, j], "-", color="tab:blue", lw=1.6, label="predicted")
        ax.plot(xs, gt_abs[:n, j], "--", color="tab:orange", lw=1.6, label="GT")
        ax.set_ylabel(EE_DIM_TITLES[j], fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("chunk step")
    fig.suptitle(f"Chunk-switch frame {frame} — predicted vs GT absolute EE per dim ({tag})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"switch_frame{frame}_per_dim_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    repo_id = Path(args.dataset_root).name

    policy, preprocessor, policy_config = load_policy_and_processors(args.pretrained_path, device)
    action_stats = extract_action_stats(preprocessor)
    action_norm_mode = policy_config.normalization_mapping["ACTION"]
    from lerobot.policies.factory import make_pre_post_processors as _mpp
    _, postproc = _mpp(policy_cfg=policy_config, pretrained_path=args.pretrained_path,
                       preprocessor_overrides={"device_processor": {"device": str(device)}})
    meta = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(policy_config, meta)
    dataset = LeRobotDataset(repo_id=repo_id, root=args.dataset_root, delta_timestamps=delta_ts,
                             return_uint8=True, episodes=[args.episode])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lerobot_collate_fn)
    ckpt = Path(args.pretrained_path).parent.name
    fig, axes = make_composite_fig(policy_config.type, ckpt, repo_id)
    targets = set(args.frames)

    # Projection: hand-eye extrinsics + intrinsics K for drawing trajectories on the image.
    tip_kin = K = None
    if args.project:
        tip_kin = vp.load_tip_kin(args.extrinsics_config)
        cam_info = args.camera_info_path or vp.find_camera_info(args.dataset_root)
        if cam_info is None:
            print("--project: no camera_info_color.json found under dataset meta/; skipping projection")
        else:
            K = vp.load_K(cam_info)
            print(f"projection on: K fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # Dead-reckon anchor: the pose the mock robot actually drifted to (from a --log).
    anchor_pose = None
    if args.anchor_from_log:
        alog = dict(np.load(f"{args.anchor_from_log}.npz", allow_pickle=False))
        av = alog["ik_ok"].astype(bool)
        aagg = alog["action_agg"][av]  # executed absolute commands (dead-reckoned trajectory)
        anchor_pose = torch.from_numpy(aagg[args.anchor_tick]).unsqueeze(0).float().to(device)
        print(f"anchor_from_log: tick {args.anchor_tick} dead-reckoned pose="
              f"{np.round(aagg[args.anchor_tick], 4).tolist()}")

    for batch in loader:
        if batch is None:
            continue
        is_pad = batch.get("action_is_pad")
        pad_mask = is_pad[0].cpu().numpy() if is_pad is not None else np.zeros(99, dtype=bool)
        # Don't skip padded frames — the prediction doesn't use GT actions (ACT VAE prior).
        # Mask padded GT entries so they don't plot (NaN).
        frame_idx = int(batch["frame_index"][0].item())
        if frame_idx not in targets:
            continue
        if CAMERA_KEY in batch and batch[CAMERA_KEY].dtype == torch.uint8:
            batch[CAMERA_KEY] = batch[CAMERA_KEY].to(torch.float32) / 255.0
        if not batch.get("task"):
            batch["task"] = [args.task]

        gt_abs = batch["action"][0].cpu().numpy()
        gt_ref = gt_abs[1]
        gt_chunk = gt_abs[1:].copy()
        # Mask padded GT entries (frames beyond episode end) → NaN, so they don't plot.
        pm = pad_mask[1:len(gt_chunk) + 1] if len(pad_mask) > 1 else np.zeros(len(gt_chunk), dtype=bool)
        gt_chunk[pm] = np.nan
        gt_traj = gt_abs_to_rel_traj(gt_chunk, gt_ref)
        gt_ang = gt_rel_angles(gt_chunk, gt_ref)

        preprocessor.reset()
        if anchor_pose is not None:
            # Dead-reckon: condition the policy on the drifted pose (from the log),
            # not the dataset GT. The image stays GT (the camera still sees the scene).
            from lerobot.utils.constants import OBS_STATE
            batch[OBS_STATE] = anchor_pose
        with torch.no_grad():
            processed = preprocessor(batch)
            if policy_config.type == "act":
                processed.pop("action", None)
            pred_norm = policy.predict_action_chunk(processed)
        pred_rel = unnormalize_actions(pred_norm, action_stats, action_norm_mode)[0].cpu().numpy()
        pred_traj = rel_actions_to_traj(pred_rel)

        # absolute predicted chunk (postprocessor), same path as the sync deploy/mock
        pred_abs = postproc(pred_norm)
        if isinstance(pred_abs, dict) and "action" in pred_abs:
            pred_abs = pred_abs["action"]
        pred_abs = pred_abs[0].cpu().numpy()

        info = {"ep": args.episode, "frame": frame_idx, "task": args.task, "xyz_err": 0.0,
                "rot_err": 0.0, "grip_pred": float(pred_rel[-1, 9]), "grip_gt": float(gt_chunk[-1, 6])}
        img_rgb = image_to_rgb_uint8(batch[CAMERA_KEY][0])
        # Draw projected trajectories on the camera image (pred green→red, GT cyan).
        if tip_kin is not None and K is not None:
            pred_poses = np.stack([np.eye(4)] + [vp.rot6d_to_matrix(a) for a in pred_rel])
            px, py = vp.project_future(pred_poses, 0, K, tip_kin)
            img_rgb = vp.draw_traj_on_image(img_rgb, np.column_stack([px, py]), "pred")
            if gt_traj is not None:
                T_ref_inv = np.linalg.inv(vp.aa_pose_to_matrix(gt_ref))
                gt_poses = np.stack([T_ref_inv @ vp.aa_pose_to_matrix(a) for a in gt_chunk])
                gpx, gpy = vp.project_future(gt_poses, 0, K, tip_kin)
                img_rgb = vp.draw_traj_on_image(img_rgb, np.column_stack([gpx, gpy]), "gt")
        composite = render_frame(img_rgb, pred_traj, gt_traj, pred_rel, gt_ang, info, fig, axes)
        import cv2
        cv2.imwrite(str(out / f"switch_frame{frame_idx}_overlay.png"),
                    cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        per_dim_figure(pred_abs, gt_chunk, frame_idx, out, "absolute")
        print(f"rendered switch frame {frame_idx} → overlay + per-dim")
        targets.discard(frame_idx)
        if not targets:
            break
    print("outputs in", out)


if __name__ == "__main__":
    main()

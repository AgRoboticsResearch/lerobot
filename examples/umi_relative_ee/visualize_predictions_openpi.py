#!/usr/bin/env python3
"""Prediction visualization for OFFICIAL-openpi (JAX) SROI LoRA checkpoints.

Counterpart of ``visualize_predictions.py`` for checkpoints trained with the
official openpi repo (orbax ``params/`` dirs). Runs in the **openpi venv**
(``~/codes/openpi/.venv/bin/python``) — it loads the policy through
``openpi.policies.policy_config.create_trained_policy`` and the dataset through
openpi's pinned LeRobot (v2.1 per-episode layout, prepared by
``prep_openpi_validation_v21.py`` / ``reshard_openpi_datasets_v21.py``).

Per dataset episode, produces an MP4 where each frame overlays the **predicted**
and **GT** gripper-tip trajectory on the camera image (pred = green->red
gradient, GT = cyan), plus a 3D trajectory inset — the same visual language as
the unified visualizer, so videos are directly comparable across stacks.

Key representation note: the strawberry action column is **start-anchored**
relative-EE (frame-0 ~= identity). The openpi arms were trained to predict in
that same frame (state = current start-anchored pose, no re-anchoring), so both
predicted and GT chunk entries are poses in the episode frame and the SROI
projection applies directly:  p_tip = (T_opt_cam @ inv(pose_t) @ pose_k @
T_cam_ee)[:3, 3].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- args ---- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config_name", required=True,
                   help="openpi TrainConfig name, e.g. pi05_lora_sroi_rot6d")
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint dir containing params/ and assets/")
    p.add_argument("--dataset_root", default="/mnt/data1/sroi/lerobot/sroiv2_strawberry_validation_rotvec",
                   help="v2.1 per-episode LeRobot dataset root (validation)")
    p.add_argument("--repo_id", default="sroiv2_strawberry_validation_rotvec")
    p.add_argument("--episode_indices", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--stride", type=int, default=1,
                   help="predict every Nth frame (1 = every frame)")
    p.add_argument("--horizon", type=int, default=10,
                   help="action chunk length the model predicts")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output_dir", default="outputs/debug/viz_openpi")
    p.add_argument("--extrinsics_config",
                   default=str(Path(__file__).resolve().parent / "camera_gripper_extrinsics_sroi_v2_d405.json"))
    p.add_argument("--camera_info_path", default=None,
                   help="camera_info_color.json for intrinsics; auto-discovered if omitted")
    p.add_argument("--no_gt", action="store_true")
    return p.parse_args()


# ------------------------------------------------------ geometry helpers ---- #

def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Rodrigues axis-angle -> 3x3 (used for the start-anchored 7D actions)."""
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    axis = rv / theta
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotvec_to_matrix(np.asarray(pose7)[3:6])
    T[:3, 3] = np.asarray(pose7)[:3]
    return T


def rot6d_chunk_to_rotvec(chunk: np.ndarray) -> np.ndarray:
    """(T,10) [xyz, rot6d(rows 0-1), grip] -> (T,7) [xyz, rotvec, grip]."""
    from scipy.spatial.transform import Rotation as R
    r6 = chunk[:, 3:9].reshape(-1, 2, 3)
    a, b = r6[:, 0], r6[:, 1]
    r1 = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    r2 = b - (r1 * b).sum(-1, keepdims=True) * r1
    r2 /= (np.linalg.norm(r2, axis=-1, keepdims=True) + 1e-8)
    r3 = np.cross(r1, r2)
    mat = np.stack([r1, r2, r3], axis=1)
    rv = R.from_matrix(mat).as_rotvec()
    return np.concatenate([chunk[:, :3], rv, chunk[:, 9:10]], axis=1)


def rotvec_state_to_rot6d(state: np.ndarray) -> np.ndarray:
    """(7,) -> (10,) input-state conversion for the rot6d arm."""
    from scipy.spatial.transform import Rotation as R
    M = R.from_rotvec(state[3:6]).as_matrix()[:2, :].reshape(6)
    return np.concatenate([state[:3], M, state[6:7]]).astype(np.float32)


# ---------------------------------------------- SROI projection (from viz) -- #

def _load_rigid_transform(config: dict, key: str, config_path: Path) -> np.ndarray:
    T = np.asarray(config[key], dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{config_path}: {key} must be a finite 4x4 matrix")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
        raise ValueError(f"{config_path}: {key} rotation not orthonormal")
    return T


def load_tip_kin(config_path: Path | str):
    cfg = json.load(open(Path(config_path).resolve()))
    if cfg.get("schema_version") != 1:
        raise ValueError("extrinsics schema_version must be 1")
    return (_load_rigid_transform(cfg, "T_optical_camera", Path(config_path)),
            _load_rigid_transform(cfg, "T_camera_gripper_tip", Path(config_path)))


def load_K(cam_info_path: Path):
    return np.array(json.load(open(cam_info_path))["K"], dtype=float).reshape(3, 3)


def find_camera_info(dataset_root: str):
    for p in sorted(Path(dataset_root).glob("meta/camera_info/**/camera_info_color.json")):
        return p
    return None


def project_poses(poses_chunk: np.ndarray, pose_t: np.ndarray, K: np.ndarray, tip_kin) -> np.ndarray:
    """Project start-anchored chunk poses into frame t's camera image.

    poses_chunk: (N,4,4) poses in the episode (start-anchored) frame.
    pose_t: (4,4) current pose in the same frame.
    Returns (N,2) pixel coords, NaN where behind the camera.
    """
    T_opt_cam, T_cam_ee = tip_kin
    inv_t = np.linalg.inv(pose_t)
    n = len(poses_chunk)
    pts = np.empty((n, 3))
    for i in range(n):
        T_rel = inv_t @ poses_chunk[i]
        pts[i] = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    oz = pts[:, 2]
    out = np.stack([fx * pts[:, 0] / oz + cx, fy * pts[:, 1] / oz + cy], axis=1)
    out[oz <= 1e-3] = np.nan
    return out


def green_red_gradient(n: int):
    if n <= 1:
        return [(0.0, 1.0, 0.0)]
    return [((i / (n - 1)), (1.0 - i / (n - 1)), 0.0) for i in range(n)]


# ------------------------------------------------------------------ main ---- #

def main() -> None:
    args = parse_args()
    is_rot6d = "rot6d" in args.config_name

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    from lerobot.common.datasets import lerobot_dataset as ld
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    out_root = Path(args.output_dir) / args.repo_id
    out_root.mkdir(parents=True, exist_ok=True)

    meta = ld.LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    ds = ld.LeRobotDataset(args.repo_id, root=args.dataset_root,
                           delta_timestamps={"action": [t / meta.fps for t in range(args.horizon)]})

    print(f"loading policy {args.config_name} from {args.checkpoint}")
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint,
                                                  default_prompt="pick the strawberry")

    cam_info = Path(args.camera_info_path) if args.camera_info_path else find_camera_info(args.dataset_root)
    assert cam_info is not None, "no camera_info_color.json found under dataset meta/"
    K = load_K(cam_info)
    tip_kin = load_tip_kin(args.extrinsics_config)

    ep_from = [0]
    for _ in range(meta.total_episodes):
        ep_from.append(ep_from[-1] + meta.episodes[len(ep_from) - 1]["length"])

    for ep in args.episode_indices:
        length = meta.episodes[ep]["length"]
        records = []  # per predicted frame: (image, pred_px, gt_px, pred_xyz, gt_xyz, ep_err_mm)
        for fi in range(0, max(1, length - args.horizon), args.stride):
            didx = ep_from[ep] + fi
            item = ds[didx]
            img = item["observation.images.camera"]
            frame = (np.clip(np.transpose(np.asarray(img), (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
            state = np.asarray(item["observation.state"], dtype=np.float32)
            if is_rot6d:
                state = rotvec_state_to_rot6d(state)
            obs = {"image": img, "state": state}
            pred = np.asarray(policy.infer(obs)["actions"])
            if is_rot6d:
                pred = rot6d_chunk_to_rotvec(pred)
            gt = np.asarray(item["action"], dtype=np.float64)  # (T,7) start-anchored
            pose_t = pose7_to_matrix(gt[0])  # state == action[t] by construction
            pred_poses = np.stack([pose7_to_matrix(p) for p in pred])
            gt_poses = np.stack([pose7_to_matrix(g) for g in gt])
            pred_px = project_poses(pred_poses, pose_t, K, tip_kin)
            gt_px = project_poses(gt_poses, pose_t, K, tip_kin)
            records.append((frame, pred_px, gt_px, pred_poses[:, :3, 3], gt_poses[:, :3, 3],
                            float(np.linalg.norm(pred[-1, :3] - gt[-1, :3]) * 1000)))
        n = len(records)
        vid_path = out_root / f"pred_episode_{ep}.mp4"
        writer = imageio.get_writer(vid_path, fps=args.fps, macro_block_size=1)
        for i, (frame, pp, gp, p3, g3, _) in enumerate(records):
            fig = plt.figure(figsize=(10.2, 4.3))
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(frame)
            for s, c in enumerate(green_red_gradient(max(len(pp) - 1, 1))):
                a, b = pp[s], pp[s + 1]
                if np.isfinite(a).all() and np.isfinite(b).all():
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=3.5, solid_capstyle="round")
            ppf = pp[np.isfinite(pp[:, 0])]
            if len(ppf):
                ax.plot(ppf[0, 0], ppf[0, 1], "o", ms=8, mfc="lime", mec="k")
                ax.plot(ppf[-1, 0], ppf[-1, 1], "o", ms=8, mfc="blue", mec="k")
            if not args.no_gt:
                gpf = gp[np.isfinite(gp[:, 0])]
                if len(gpf) > 1:
                    ax.plot(gpf[:, 0], gpf[:, 1], color="cyan", lw=2.5, alpha=0.9)
                if len(gpf):
                    ax.plot(gpf[0, 0], gpf[0, 1], "o", ms=6, mfc="magenta", mec="k")
            ax.set_title(f"ep {ep}  frame {i * args.stride}  (pred green->red, GT cyan)")
            ax.axis("off")

            ax3 = fig.add_subplot(1, 2, 2, projection="3d")
            ax3.plot(g3[:, 0], g3[:, 1], g3[:, 2], color="cyan", lw=2, label="GT")
            for s, c in enumerate(green_red_gradient(max(len(p3) - 1, 1))):
                a, b = p3[s], p3[s + 1]
                ax3.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=c, lw=2.5)
            ax3.scatter(*g3[0], color="magenta", s=25)
            ax3.set_title("tip path (episode frame, m)")
            ax3.legend(loc="upper right", fontsize=7)
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            # libx264/yuv420p requires even dimensions
            buf = buf[: buf.shape[0] // 2 * 2, : buf.shape[1] // 2 * 2]
            writer.append_data(buf.copy())
            plt.close(fig)
        writer.close()

        errs = [r[5] for r in records]
        metrics = {
            "config_name": args.config_name,
            "checkpoint": args.checkpoint,
            "episode": ep,
            "frames": n,
            "endpoint_err_mean_mm": float(np.mean(errs)),
        }
        (out_root / f"pred_episode_{ep}_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  ep{ep}: {n} frames -> {vid_path.name}  mean endpoint {metrics['endpoint_err_mean_mm']:.1f} mm")


if __name__ == "__main__":
    main()

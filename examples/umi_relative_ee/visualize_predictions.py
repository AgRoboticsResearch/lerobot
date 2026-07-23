#!/usr/bin/env python
r"""
Visualize ACT, SmolVLA, or π0.5 UMI relative-EE predictions on a recorded dataset.

The policy is loaded through its saved unified or legacy-compatible UMI
processor. LoRA adapters are detected automatically. The state is derived from
``action[t-1:t+1]``; a task string is supplied for language-conditioned models.

Mode: **open-loop dataset inference**. For each frame of a validation episode we
feed the model the GT observation (image + the two-pose state derived from the
action delta + the task string), predict the 30-step action chunk, unnormalize
to relative rot6d actions, and render a per-frame **panel-composite** video:

  - left:  the camera image with a HUD (episode / frame / task / error);
  - right: predicted-vs-GT 3D trajectory (chunk-start frame) and per-dim
           action curves (xyz / rotation-magnitude) over the chunk.

The panel-composite needs no camera calibration. Pass ``--project`` to also draw
the predicted and ground-truth gripper-tip trajectories on the camera image
using the supplied hand-eye calibration and dataset camera intrinsics.

Usage:
    python examples/umi_relative_ee/visualize_predictions.py \
        --pretrained_path outputs/train/my_umi_policy/checkpoints/last/pretrained_model \
        --dataset_root /mnt/data1/sroi/lerobot/sroiv2_strawberry_picking_lab_validation \
        --episode_indices 0 1 2 \
        --output_dir outputs/debug/viz_umi
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import imageio.v3 as iio  # noqa: E402
except ImportError:
    iio = None

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TASK = "pick the strawberry"
CAMERA_KEY = "observation.images.camera"


# ---------------------------------------------------------------------------
# Geometry / stats helpers (ported from the sibling visualize_predictions.py)
# ---------------------------------------------------------------------------


def aa_pose_to_matrix(pose_7d: np.ndarray) -> np.ndarray:
    """Absolute 7D [xyz, axis-angle(3), gripper] -> 4x4 (gripper ignored)."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(pose_7d[3:6])).as_matrix()
    T[:3, 3] = pose_7d[:3]
    return T


def rot6d_to_matrix(action_10d: np.ndarray) -> np.ndarray:
    """UMI row-based rot6d [dx,dy,dz, r0..r5, gripper] -> 4x4 (gripper ignored)."""
    T = np.eye(4)
    T[:3, 3] = action_10d[:3]
    a1, a2 = np.asarray(action_10d[3:6]), np.asarray(action_10d[6:9])
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    T[:3, :3] = np.stack([b1, b2, b3])  # rows = stored rot6d vectors
    return T


def rot_angle_from_matrix(R: np.ndarray) -> float:
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.arccos(cos))


def unnormalize_actions(tensor: torch.Tensor, stats: dict, mode) -> torch.Tensor:
    """Invert normalization without applying the absolute-EE postprocessor."""
    mode_name = getattr(mode, "value", str(mode)).lower()

    def stat(name: str) -> torch.Tensor:
        return torch.as_tensor(stats[name], device=tensor.device, dtype=tensor.dtype)

    if "quantile" in mode_name:
        low, high = stat("q01"), stat("q99")
        return (tensor + 1.0) * (high - low) / 2.0 + low
    if "min_max" in mode_name:
        low, high = stat("min"), stat("max")
        return (tensor + 1.0) * (high - low) / 2.0 + low
    if "mean_std" in mode_name:
        return tensor * stat("std") + stat("mean")
    if "identity" in mode_name:
        return tensor
    raise ValueError(f"Unsupported action normalization mode: {mode!r}")


def extract_action_stats(preprocessor) -> dict:
    for step in preprocessor.steps:
        if hasattr(step, "stats") and step.stats and "action" in step.stats:
            return step.stats["action"]
    raise ValueError("No 'action' stats in preprocessor (need a trained UMI checkpoint).")


def load_policy_and_processors(model_path: str, device: torch.device):
    """Load policy (LoRA-aware) + saved processors from a checkpoint dir."""
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    policy_class = get_policy_class(policy_config.type)

    if (Path(model_path) / "adapter_config.json").exists():
        # LoRA: load the base policy named in the adapter, then apply the adapter.
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_path)
        base_name = peft_config.base_model_name_or_path
        if not base_name:
            raise ValueError("LoRA adapter_config.json has no base_model_name_or_path.")
        logger.info("LoRA: base='%s' + adapter from %s", base_name, model_path)
        # config=policy_config so the base uses the FINE-TUNED input/output features
        # (e.g. observation.images.camera), not the base pi05 defaults (base_0_rgb ...).
        policy = policy_class.from_pretrained(base_name, config=policy_config)
        policy = PeftModel.from_pretrained(policy, model_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(model_path, local_files_only=True)

    policy.to(device)
    policy.eval()
    policy_config.device = str(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, policy_config


# ---------------------------------------------------------------------------
# Trajectories (relative, chunk-start frame; NO hand-eye transforms needed)
# ---------------------------------------------------------------------------


def rel_actions_to_traj(rel_10d: np.ndarray) -> np.ndarray:
    """Relative 10D chunk -> 3D points in the chunk-start frame (origin prepended)."""
    pts = [np.zeros(3)]
    for a in rel_10d:
        pts.append(rot6d_to_matrix(a)[:3, 3])
    return np.asarray(pts)


def gt_abs_to_rel_traj(gt_abs_chunk: np.ndarray, gt_ref: np.ndarray) -> np.ndarray:
    T_ref_inv = np.linalg.inv(aa_pose_to_matrix(gt_ref))
    pts = [np.zeros(3)]
    for a in gt_abs_chunk:
        pts.append((T_ref_inv @ aa_pose_to_matrix(a))[:3, 3])
    return np.asarray(pts)


def gt_rel_angles(gt_chunk: np.ndarray, gt_ref: np.ndarray) -> np.ndarray:
    T_ref_inv = np.linalg.inv(aa_pose_to_matrix(gt_ref))
    return np.asarray([rot_angle_from_matrix((T_ref_inv @ aa_pose_to_matrix(a))[:3, :3]) for a in gt_chunk])


def pred_rel_angles(rel_10d: np.ndarray) -> np.ndarray:
    return np.asarray([rot_angle_from_matrix(rot6d_to_matrix(a)[:3, :3]) for a in rel_10d])


def image_to_rgb_uint8(obs) -> np.ndarray:
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    if obs.ndim == 4:
        obs = obs[0]
    if obs.ndim == 3 and obs.shape[0] in (1, 3):
        obs = obs.transpose(1, 2, 0)
    if obs.dtype != np.uint8:
        obs = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
    if obs.shape[-1] == 1:
        obs = np.repeat(obs, 3, axis=-1)
    return obs


# ---------------------------------------------------------------------------
# Pixel projection onto the camera image. The projection math, extrinsics
# loader, intrinsics loader, and green->red gradient are COPIED VERBATIM from
# sroi_rosbag_utilities/visualization/visualize_traj_video.py (so this script
# has no runtime dependency on that folder). The dataset `action` IS the camera
# trajectory (CameraTrajectoryTransformed), so a predicted relative chunk
# composes like T_rel = inv(poses[t]) @ poses[k]; project_future does the
# (T_opt_cam @ T_rel @ T_cam_ee) + pinhole math + behind-camera NaN masking.
# ---------------------------------------------------------------------------

DEFAULT_EXTRINSICS = Path(__file__).resolve().parent / "camera_gripper_extrinsics_sroi_v2_d405.json"


def _load_rigid_transform(config: dict, key: str, config_path: Path) -> np.ndarray:  # noqa: ANN001
    """Copied from sroi visualize_traj_video.py."""
    try:
        transform = np.asarray(config[key], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{config_path}: {key} must be a numeric 4x4 matrix") from error
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{config_path}: {key} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{config_path}: {key} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{config_path}: {key} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{config_path}: {key} rotation determinant must be +1")
    return transform


def load_tip_kin(config_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load optical<-camera and camera<-gripper-tip transforms from JSON. (sroi)"""
    config_path = Path(config_path).resolve()
    try:
        with config_path.open() as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load extrinsics config {config_path}: {error}") from error
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"{config_path}: schema_version must be 1")
    return (
        _load_rigid_transform(config, "T_optical_camera", config_path),
        _load_rigid_transform(config, "T_camera_gripper_tip", config_path),
    )


def load_K(cam_info_path: Path | str) -> np.ndarray | None:
    """Load the color camera intrinsics (3x3) from camera_info_color.json, or None. (sroi)"""
    cam_info_path = Path(cam_info_path)
    if not cam_info_path.exists():
        return None
    try:
        K = np.array(json.load(cam_info_path.open())["K"], dtype=float).reshape(3, 3)
    except Exception:
        return None
    return K


def project_future(poses: np.ndarray, t: int, K: np.ndarray, tip_kin) -> tuple[np.ndarray, np.ndarray]:
    """Project the gripper-tip path from frame t through the end into frame t's image.
    Returns (px, py), NaN where behind the camera. (Copied from sroi.)"""
    T_t_inv = np.linalg.inv(poses[t])
    T_opt_cam, T_cam_ee = tip_kin
    pts = np.empty((len(poses) - t, 3))
    for i, k in enumerate(range(t, len(poses))):
        T_rel = T_t_inv @ poses[k]
        pts[i] = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]
    ox, oy, oz = pts[:, 0], pts[:, 1], pts[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    px = fx * ox / oz + cx
    py = fy * oy / oz + cy
    behind = oz <= 1e-3
    px = np.where(behind, np.nan, px)
    py = np.where(behind, np.nan, py)
    return px, py


def _green_red_gradient(n_seg: int) -> list[tuple[float, float, float]]:
    """vispred-style per-segment colors: green (start) -> red (end). (sroi)"""
    if n_seg <= 1:
        return [(0.0, 1.0, 0.0)]
    return [((i / (n_seg - 1)), (1.0 - i / (n_seg - 1)), 0.0) for i in range(n_seg)]


def find_camera_info(dataset_root: str):
    """Locate camera_info_color.json (intrinsics K) under the dataset meta tree."""
    for p in sorted(Path(dataset_root).glob("meta/camera_info/**/camera_info_color.json")):
        return p
    return None


def draw_traj_on_image(img_rgb, pts2d, mode="pred"):
    """Draw a projected trajectory on an RGB image with cv2.

    pred: sroi green->red gradient (_green_red_gradient) + green start / blue stop dots.
    gt:   cyan line + magenta start dot.
    """
    img = img_rgb.copy()
    pts2d = np.asarray(pts2d, dtype=float)
    n = len(pts2d)
    if n < 2:
        return img
    segs = np.stack([pts2d[:-1], pts2d[1:]], axis=1)
    colors = _green_red_gradient(len(segs)) if mode == "pred" else [(0.0, 1.0, 1.0)] * len(segs)
    thick = 2 if mode == "pred" else 3
    for (p1, p2), c in zip(segs, colors):
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        col = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, thick)
    if mode == "pred":
        for pt, col in [(pts2d[0], (0, 255, 0)), (pts2d[-1], (0, 0, 255))]:
            if np.isfinite(pt).all():
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, col, -1)
    elif np.isfinite(pts2d[0]).all():
        cv2.circle(img, (int(pts2d[0][0]), int(pts2d[0][1])), 4, (255, 0, 255), -1)
    return img


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_frame(img_rgb, pred_traj, gt_traj, pred_rel, gt_ang, info, fig, axes):
    ax_img, ax_3d, ax_curves = axes
    ax_img.clear()
    ax_3d.clear()
    ax_curves.clear()

    ax_img.imshow(img_rgb)
    ax_img.set_title(
        f"ep {info['ep']}  frame {info['frame']}  task='{info['task']}'\n"
        f"end xyz err {info['xyz_err'] * 1000:.1f}mm  end rot err {np.degrees(info['rot_err']):.1f}deg  "
        f"grip pred={info['grip_pred']:.2f} gt={info['grip_gt']:.2f}",
        fontsize=8,
    )
    ax_img.axis("off")

    if gt_traj is not None and len(gt_traj) > 1:
        ax_3d.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], "r--", lw=2, label="GT")
        ax_3d.scatter([gt_traj[-1, 0]], [gt_traj[-1, 1]], [gt_traj[-1, 2]], c="r", s=30)
    if len(pred_traj) > 1:
        ax_3d.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], "b-", lw=2, label="Pred")
        ax_3d.scatter([pred_traj[-1, 0]], [pred_traj[-1, 1]], [pred_traj[-1, 2]], c="b", s=30)
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    lim = 0.15
    ax_3d.set_xlim(-lim, lim)
    ax_3d.set_ylim(-lim, lim)
    ax_3d.set_zlim(-lim, lim)
    try:
        ax_3d.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.legend(fontsize=7, loc="upper right")
    ax_3d.set_title("Relative EE trajectory (chunk)", fontsize=8)

    xs = np.arange(len(pred_rel))
    ax_curves.plot(xs, pred_rel[:, 0], "b-", label="px")
    ax_curves.plot(xs, pred_rel[:, 1], "g-", label="py")
    ax_curves.plot(xs, pred_rel[:, 2], "c-", label="pz")
    ax_curves.plot(xs, pred_rel_angles(pred_rel), "m-", label="pred rot|.|")
    if len(gt_ang):
        ax_curves.plot(np.arange(len(gt_ang)), gt_ang, "r--", label="gt rot|.|")
    ax_curves.set_ylim(-0.12, 0.12)
    ax_curves.legend(fontsize=6, ncol=3, loc="upper right")
    ax_curves.set_title("Per-target relative action", fontsize=8)
    ax_curves.set_xlabel("chunk step", fontsize=7)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--episode_indices", type=int, nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/debug/viz_umi")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--no_gt", action="store_true")
    parser.add_argument("--first_frame_debug", action="store_true")
    parser.add_argument(
        "--max_frames_per_episode",
        type=int,
        default=0,
        help="Stop rendering each episode after this many valid frames (0 means all).",
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help="Draw predicted/GT gripper-tip trajectories on the camera image (needs hand-eye + K).",
    )
    parser.add_argument(
        "--extrinsics_config",
        default=DEFAULT_EXTRINSICS,
        help=f"camera_gripper_extrinsics JSON (default: {DEFAULT_EXTRINSICS})",
    )
    parser.add_argument(
        "--camera_info_path",
        default=None,
        help="camera_info_color.json for intrinsics K (auto-found under dataset meta/ if omitted)",
    )
    args = parser.parse_args()

    if iio is None:
        parser.error("imageio is required (pip install imageio)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    repo_id = args.repo_id or Path(args.dataset_root).name

    logger.info("Loading policy + processors from %s", args.pretrained_path)
    policy, preprocessor, policy_config = load_policy_and_processors(args.pretrained_path, device)
    action_stats = extract_action_stats(preprocessor)
    action_norm_mode = policy_config.normalization_mapping["ACTION"]
    chunk = policy_config.chunk_size
    logger.info(
        "policy=%s chunk=%d use_umi_relative_ee=%s",
        policy_config.type,
        chunk,
        getattr(policy_config, "use_umi_relative_ee", False),
    )
    logger.info("action normalization=%s", action_norm_mode)

    meta = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(policy_config, meta)
    logger.info("delta_timestamps keys=%s", {k: (len(v) if v else v) for k, v in (delta_ts or {}).items()})
    episodes = [e for e in args.episode_indices if e < meta.total_episodes]
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_ts,
        return_uint8=True,
        episodes=episodes,
    )
    fps = args.fps or meta.fps
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lerobot_collate_fn)

    # pixel-projection setup (SROI v2 D405 hand-eye + color intrinsics)
    tip_kin = K = None
    if args.project:
        tip_kin = load_tip_kin(args.extrinsics_config)
        cam_info = args.camera_info_path or find_camera_info(args.dataset_root)
        if cam_info is None:
            parser.error("--project needs --camera_info_path (none found under dataset meta/camera_info).")
        K = load_K(cam_info)
        if K is None:
            parser.error(f"could not load intrinsics K from {cam_info}")
        logger.info(
            "projection on: extrinsics=%s K fx=%.1f cx=%.1f cy=%.1f",
            args.extrinsics_config,
            K[0, 0],
            K[0, 2],
            K[1, 2],
        )

    out_dir = Path(args.output_dir) / repo_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 5))
    axes = (fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2, projection="3d"), fig.add_subplot(1, 3, 3))

    cur_ep = None
    frames = []
    t0 = time.perf_counter()
    n_done = 0
    error_rows = []

    def flush(ep):
        nonlocal frames
        if frames:
            path = out_dir / f"pred_episode_{ep}.mp4"
            iio.imwrite(path, np.stack(frames), fps=fps, macro_block_size=1, quality=8)
            logger.info("Saved %s (%d frames)", path, len(frames))
        frames = []

    for batch in loader:
        if batch is None:
            continue
        ep_idx = int(batch["episode_index"][0].item())
        if cur_ep is None:
            cur_ep = ep_idx
        if ep_idx != cur_ep:
            flush(cur_ep)
            cur_ep = ep_idx
            t0 = time.perf_counter()
            n_done = 0
        if args.max_frames_per_episode and n_done >= args.max_frames_per_episode:
            continue

        # skip frames whose [t-1, t, ..., t+chunk-1] window is partially padded
        is_pad = batch.get("action_is_pad")
        if is_pad is not None and bool(is_pad[0].any()):
            continue

        frame_idx = int(batch["frame_index"][0].item())

        # uint8 camera -> float/255 (matches _normalize_uint8_cameras)
        if CAMERA_KEY in batch and batch[CAMERA_KEY].dtype == torch.uint8:
            batch[CAMERA_KEY] = batch[CAMERA_KEY].to(torch.float32) / 255.0
        # task: inject if collate dropped it
        if not batch.get("task"):
            batch["task"] = [args.task]

        # GT absolute 7D delta: [t-1, t, t+1, ...]; chunk-start = action[t] = index 1
        gt_abs = batch["action"][0].cpu().numpy()
        gt_ref = gt_abs[1]
        gt_chunk = gt_abs[1:]
        gt_traj = None if args.no_gt else gt_abs_to_rel_traj(gt_chunk, gt_ref)
        gt_ang = gt_rel_angles(gt_chunk, gt_ref)

        preprocessor.reset()
        with torch.no_grad():
            processed = preprocessor(batch)
            if policy_config.type == "act":
                processed.pop("action", None)
            if args.first_frame_debug and n_done == 0:
                logger.info(
                    "processed shapes: %s",
                    {k: tuple(v.shape) for k, v in processed.items() if torch.is_tensor(v)},
                )
            pred = policy.predict_action_chunk(processed)

        pred_rel = unnormalize_actions(pred, action_stats, action_norm_mode)[0].cpu().numpy()
        pred_traj = rel_actions_to_traj(pred_rel)

        pred_end = pred_traj[-1]
        if gt_traj is not None:
            pred_end_pose = rot6d_to_matrix(pred_rel[-1])
            gt_end_pose = np.linalg.inv(aa_pose_to_matrix(gt_ref)) @ aa_pose_to_matrix(gt_chunk[-1])
            xyz_err = float(np.linalg.norm(pred_end_pose[:3, 3] - gt_end_pose[:3, 3]))
            rot_err = rot_angle_from_matrix(pred_end_pose[:3, :3].T @ gt_end_pose[:3, :3])
        else:
            xyz_err = rot_err = 0.0

        img_rgb = image_to_rgb_uint8(batch[CAMERA_KEY][0])

        # draw projected gripper-tip trajectories on the camera image (sroi project_future)
        if args.project and tip_kin is not None and K is not None:
            pred_poses = np.stack([np.eye(4)] + [rot6d_to_matrix(a) for a in pred_rel])
            px, py = project_future(pred_poses, 0, K, tip_kin)
            img_rgb = draw_traj_on_image(img_rgb, np.column_stack([px, py]), "pred")
            if gt_traj is not None:
                T_ref_inv = np.linalg.inv(aa_pose_to_matrix(gt_ref))
                gt_poses = np.stack([T_ref_inv @ aa_pose_to_matrix(a) for a in gt_chunk])
                gpx, gpy = project_future(gt_poses, 0, K, tip_kin)
                img_rgb = draw_traj_on_image(img_rgb, np.column_stack([gpx, gpy]), "gt")

        info = {
            "ep": ep_idx,
            "frame": frame_idx,
            "task": args.task,
            "xyz_err": xyz_err,
            "rot_err": rot_err,
            "grip_pred": float(pred_rel[-1, 9]),
            "grip_gt": float(gt_chunk[-1, 6]),
        }
        error_rows.append(info.copy())
        frames.append(render_frame(img_rgb, pred_traj, gt_traj, pred_rel, gt_ang, info, fig, axes))

        n_done += 1
        if n_done % 25 == 0:
            logger.info(
                "  ep %d: %d frames (%.0fms/frame)",
                ep_idx,
                n_done,
                1000 * (time.perf_counter() - t0) / n_done,
            )

    flush(cur_ep)
    if error_rows:
        xyz_errors = np.asarray([row["xyz_err"] for row in error_rows])
        rot_errors = np.asarray([row["rot_err"] for row in error_rows])
        grip_errors = np.asarray([abs(row["grip_pred"] - row["grip_gt"]) for row in error_rows])
        summary = {
            "num_frames": len(error_rows),
            "xyz_end_error_mean_m": float(xyz_errors.mean()),
            "xyz_end_error_median_m": float(np.median(xyz_errors)),
            "rotation_end_error_mean_rad": float(rot_errors.mean()),
            "gripper_end_error_mean": float(grip_errors.mean()),
            "frames": error_rows,
        }
        metrics_path = out_dir / "prediction_metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2))
        logger.info(
            "Metrics: %s", json.dumps({key: value for key, value in summary.items() if key != "frames"})
        )
        logger.info("Saved %s", metrics_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()

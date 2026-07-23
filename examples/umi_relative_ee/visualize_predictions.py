#!/usr/bin/env python

r"""
Visualize UMI-style processor pipeline predictions on camera images.

Supports two modes:

1. **Camera mode** (with --cameras): Connects to a physical camera (no robot needed),
   runs policy inference using identity EE state, and overlays predicted trajectories
   on the camera feed.

2. **Dataset mode** (with --dataset_root): Loads a LeRobotDataset and projects
   trajectories onto observation images, saving as MP4 or displaying live.
   - Without --inference: projects GT action trajectories
   - With --inference: runs policy inference per frame

Usage (camera mode — handheld camera, no robot):
    python visualize_predictions.py \
        --pretrained_path outputs/.../pretrained_model \
        --cameras "{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \

Usage (dataset mode — GT):
    python visualize_predictions.py \
        --dataset_root /path/to/dataset \
        --episode_indices 0

Usage (dataset mode — inference):
    python visualize_predictions.py \
        --dataset_root /path/to/dataset \
        --episode_indices 0 \
        --inference \
        --pretrained_path outputs/.../pretrained_model \
        --task "pick the strawberry"
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE, ACTION

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logger = logging.getLogger(__name__)

FPS = 30

DEFAULT_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "sroi-piper",
    "src",
    "utils",
    "piper_urdf",
    "piper_sroiv2.urdf",
)
DEFAULT_URDF_PATH = os.path.normpath(os.path.abspath(DEFAULT_URDF_PATH))


@contextmanager
def _timed_phase(name: str):
    """Time a startup phase and log the elapsed time in ms."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[startup] {name}: {(time.perf_counter() - t0) * 1000:.0f} ms")


# ---------------------------------------------------------------------------
# Camera config parser (matches deploy script)
# ---------------------------------------------------------------------------


def auto_detect_realsense_serial() -> str | None:
    try:
        import pyrealsense2 as rs2

        ctx = rs2.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            return devices[0].get_info(rs2.camera_info.serial_number)
    except Exception:
        pass
    return None


def parse_cameras_config(cameras_str: str | None) -> dict[str, CameraConfig]:
    if not cameras_str or cameras_str.strip() == "":
        return {}
    cameras_dict = yaml.safe_load(cameras_str)
    if not isinstance(cameras_dict, dict):
        raise ValueError(f"Expected dict, got {type(cameras_dict)}")
    cameras: dict[str, CameraConfig] = {}
    for name, config in cameras_dict.items():
        camera_type = config.pop("type", None)
        if camera_type is None:
            raise ValueError(f"Camera '{name}' missing 'type' field")
        if camera_type == "intelrealsense" and "serial_number_or_name" not in config:
            serial = auto_detect_realsense_serial()
            if serial:
                config["serial_number_or_name"] = serial
                logger.info(f"Auto-detected RealSense serial: {serial}")
            else:
                raise ValueError("No RealSense device found and no serial_number_or_name provided")
        if camera_type == "opencv" and "index_or_path" in config and isinstance(config["index_or_path"], int):
            config["index_or_path"] = str(config["index_or_path"])
        camera_config_class = CameraConfig.get_choice_class(camera_type)
        cameras[name] = camera_config_class(**config)
    return cameras


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------


def load_camera_matrix_from_file(path: str) -> np.ndarray:
    with open(path) as f:
        info = json.load(f)
    K = np.array(info["K"], dtype=np.float64).reshape(3, 3)
    logger.info(f"Loaded camera matrix: fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    return K


def load_camera_matrix_from_dataset(dataset_root: str) -> np.ndarray:
    root = Path(dataset_root)
    legacy_path = root / "meta" / "camera_info.json"
    candidates = [legacy_path, *sorted(root.glob("meta/camera_info/**/camera_info_color.json"))]
    for path in candidates:
        if path.exists():
            logger.info("Using camera intrinsics from %s", path)
            return load_camera_matrix_from_file(path)
    raise FileNotFoundError(
        f"No camera intrinsics found under {root / 'meta'}; expected camera_info.json or "
        "camera_info/**/camera_info_color.json"
    )


def auto_detect_camera_intrinsics(cameras: dict) -> np.ndarray | None:
    if rs is None:
        return None
    for cam in cameras.values():
        if hasattr(cam, "rs_pipeline") and cam.rs_pipeline is not None:
            try:
                profile = cam.rs_pipeline.get_active_profile()
                intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                K = np.array(
                    [
                        [intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1],
                    ]
                )
                logger.info(f"Auto-detected RealSense intrinsics: fx={intrinsics.fx:.1f}")
                return K
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Kinematics helpers (Piper URDF, ee_link)
# ---------------------------------------------------------------------------


def get_kinematic_transforms(urdf_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return T_opt_cam (optical→cam) and T_cam_ee (cam→ee) at the neutral pose.

    These are static transforms between fixed links, so we use placo's RobotWrapper
    directly for FK — no KinematicsSolver (that's IK-only machinery). The flags skip
    collision-pair setup and visual-mesh parsing, which avoids the slow self-collision
    scan and the stderr warning spam from the previous 3x RobotKinematics approach.
    """
    import placo

    flags = placo.Flags.ignore_collisions | placo.Flags.collision_as_visual
    robot = placo.RobotWrapper(str(urdf_path), flags)
    robot.update_kinematics()  # required before get_T_a_b reads frame placements
    T_opt_cam = np.asarray(robot.get_T_a_b("camera_optical_link", "camera_link"))
    T_cam_ee = np.asarray(robot.get_T_a_b("camera_link", "ee_link"))
    return T_opt_cam, T_cam_ee


# Hand-eye from the SROI v2 D405 rig config — MATCHES sroi's visualize_traj_video.py
# (so the projected start point agrees). Copied from that script's load_tip_kin /
# _load_rigid_transform so there is no runtime dependency on the sroi folder.
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
    with config_path.open() as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"{config_path}: schema_version must be 1")
    return (
        _load_rigid_transform(config, "T_optical_camera", config_path),
        _load_rigid_transform(config, "T_camera_gripper_tip", config_path),
    )


# ---------------------------------------------------------------------------
# 3-D projection — everything is relative in UMI
# ---------------------------------------------------------------------------


def aa_pose_to_matrix(pose_7d: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(pose_7d[3:6]).as_matrix()
    T[:3, 3] = pose_7d[:3]
    return T


def rot6d_to_matrix(action_10d: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = action_10d[:3]
    a1, a2 = action_10d[3:6], action_10d[6:9]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    T[:3, :3] = np.stack([b1, b2, b3])
    return T


def relative_actions_to_3d_points(
    T_rel_list: list[np.ndarray],
    T_opt_cam: np.ndarray,
    T_cam_ee: np.ndarray,
) -> np.ndarray:
    positions = [(T_opt_cam @ T_cam_ee)[:3, 3].copy()]
    for T_rel in T_rel_list:
        pos = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]
        positions.append(pos.copy())
    return np.array(positions)


def project_points_to_image(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points_3d[:, 2]
    pts = (K @ points_3d.T).T
    points_2d = pts[:, :2] / z[:, None]
    points_2d[z <= 1e-3] = np.nan
    return points_2d


def draw_trajectory_on_image(
    img: np.ndarray,
    points_2d: np.ndarray,
    cmap: str = "pred",
    gripper: np.ndarray | None = None,
) -> np.ndarray:
    img_draw = img.copy()
    n = len(points_2d)
    if n == 0:
        return img_draw

    if cmap == "gt":
        colors = [(255, 255, 0)] * max(1, n - 1)  # cyan in BGR
    else:
        colors = [
            (0, int(255 * (1 - i / max(1, n - 2))), int(255 * i / max(1, n - 2)))
            for i in range(max(1, n - 1))
        ]

    h, w = img.shape[:2]
    for i in range(len(points_2d) - 1):
        if not (np.isfinite(points_2d[i]).all() and np.isfinite(points_2d[i + 1]).all()):
            continue
        pt1 = tuple(points_2d[i].astype(int))
        pt2 = tuple(points_2d[i + 1].astype(int))
        if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
            cv2.line(img_draw, pt1, pt2, colors[i % len(colors)], 2)
            if gripper is not None and i < len(gripper) and gripper[i] < 0.1:
                color = (0, 0, 255) if cmap == "gt" else (255, 0, 0)
                cv2.drawMarker(img_draw, pt1, color, cv2.MARKER_CROSS, markerSize=6, thickness=2)

    for pt, col in [(points_2d[0], (0, 255, 0)), (points_2d[-1], (0, 0, 255))]:
        if not np.isfinite(pt).all():
            continue
        p = tuple(pt.astype(int))
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img_draw, p, 5, col, -1)
    return img_draw


def unnormalize_actions(tensor: torch.Tensor, stats: dict) -> torch.Tensor:
    min_val = torch.as_tensor(stats["min"], device=tensor.device, dtype=tensor.dtype)
    max_val = torch.as_tensor(stats["max"], device=tensor.device, dtype=tensor.dtype)
    return tensor * (max_val - min_val) / 2 + (max_val + min_val) / 2


def extract_action_stats(preprocessor) -> dict:
    for step in preprocessor.steps:
        if hasattr(step, "stats") and step.stats and "action" in step.stats:
            return step.stats["action"]
    raise ValueError(
        "No 'action' statistics found in the saved preprocessor. "
        "Use a trained UMI relative-EE checkpoint, not the unmodified base policy."
    )


def load_policy_and_processors(model_path: str, device: torch.device):
    """Load a checkpoint-selected policy and its saved relative-EE processors."""
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(model_path, local_files_only=True)
    policy.to(device)
    policy.eval()
    policy.config.device = str(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def add_policy_task(batch: dict, policy, task: str | None) -> None:
    """Add the language task required by SmolVLA; leave other policies unchanged."""
    if policy.config.type != "smolvla":
        return
    if not task:
        raise ValueError("SmolVLA visualization requires --task (for this dataset: 'pick the strawberry').")
    batch["task"] = task


# ---------------------------------------------------------------------------
# Camera mode
# ---------------------------------------------------------------------------


def run_camera_mode(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    startup_t0 = time.perf_counter()

    logger.info(f"Loading policy from {args.pretrained_path}")
    with _timed_phase("load policy and pre/post processors"):
        policy, preprocessor, postprocessor = load_policy_and_processors(args.pretrained_path, device)
    logger.info("Policy and processors loaded")

    with _timed_phase("extract action stats"):
        action_stats = extract_action_stats(preprocessor)

    with _timed_phase("load hand-eye transforms (D405 JSON)"):
        T_opt_cam, T_cam_ee = load_tip_kin(args.extrinsics_config)

    camera_matrix = None
    if args.camera_info_path:
        with _timed_phase("load camera matrix from file"):
            camera_matrix = load_camera_matrix_from_file(args.camera_info_path)

    with _timed_phase("parse cameras config"):
        cameras_config = parse_cameras_config(args.cameras)
        if not cameras_config:
            raise ValueError("No cameras configured")

    with _timed_phase("make cameras"):
        cameras = make_cameras_from_configs(cameras_config)

    with _timed_phase("connect cameras"):
        for cam_name, camera in cameras.items():
            camera.connect()
            logger.info(f"Connected camera: {cam_name}")

    if camera_matrix is None:
        with _timed_phase("auto-detect camera intrinsics"):
            camera_matrix = auto_detect_camera_intrinsics(cameras)

    if camera_matrix is None:
        logger.warning("No camera intrinsics — trajectory overlay disabled")

    if args.initial_state:
        current_state = np.array(args.initial_state, dtype=np.float32)
    else:
        current_state = np.array([0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32)

    with _timed_phase("reset policy/processors"):
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    logger.info(f"[startup] total: {(time.perf_counter() - startup_t0) * 1000:.0f} ms")

    step_count = 0
    logger.info("Starting inference loop (Ctrl+C to stop)")

    try:
        while args.num_steps == 0 or step_count < args.num_steps:
            t0 = time.perf_counter()

            state_tensor = torch.from_numpy(current_state).unsqueeze(0).to(device)
            batch = {OBS_STATE: state_tensor}
            add_policy_task(batch, policy, args.task)
            cam_images = {}

            for cam_name, camera in cameras.items():
                img = camera.read()
                cam_images[cam_name] = img
                img_float = img.astype(np.float32) / 255.0
                img_chw = np.transpose(img_float, (2, 0, 1))
                batch[f"observation.images.{cam_name}"] = torch.from_numpy(img_chw).unsqueeze(0).to(device)

            with torch.no_grad():
                processed = preprocessor(batch)
                pred_10d = policy.predict_action_chunk(processed)

            actions_rel = unnormalize_actions(pred_10d, action_stats).cpu().numpy()
            actions_rel = actions_rel[0] if actions_rel.ndim == 3 else actions_rel

            if args.update_state:
                from scipy.spatial.transform import Rotation

                T_ref = aa_pose_to_matrix(current_state)
                T_rel_last = rot6d_to_matrix(actions_rel[-1])
                T_abs_last = T_ref @ T_rel_last
                current_state = np.concatenate(
                    [
                        T_abs_last[:3, 3],
                        Rotation.from_matrix(T_abs_last[:3, :3]).as_rotvec(),
                        [actions_rel[-1, 9]],
                    ]
                ).astype(np.float32)

            T_rel_list = [rot6d_to_matrix(a) for a in actions_rel]
            traj_3d = relative_actions_to_3d_points(T_rel_list, T_opt_cam, T_cam_ee)

            for cam_name in cameras:
                img = cam_images[cam_name]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if camera_matrix is not None:
                    pts_2d = project_points_to_image(traj_3d, camera_matrix)
                    gripper = actions_rel[:, 9] if args.gripper else None
                    img_bgr = draw_trajectory_on_image(img_bgr, pts_2d, gripper=gripper)

                cv2.imshow(f"UMI Prediction: {cam_name}", img_bgr)

            elapsed = time.perf_counter() - t0
            fps_actual = 1.0 / max(elapsed, 1e-6)
            if step_count % 10 == 0:
                logger.info(f"Step {step_count}: {elapsed * 1000:.0f}ms ({fps_actual:.0f} fps)")
            step_count += 1

            key = cv2.waitKey(max(1, int(1000 / args.fps - elapsed)))
            if key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        for camera in cameras.values():
            camera.disconnect()
        logger.info(f"Done after {step_count} steps")


# ---------------------------------------------------------------------------
# Dataset mode
# ---------------------------------------------------------------------------


def get_image_from_sample(sample, camera_name="camera"):
    obs = sample[f"observation.images.{camera_name}"]
    if obs.ndim == 4:
        obs = obs[-1]
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
    if obs.ndim == 3 and obs.shape[0] in [1, 3]:
        obs = obs.transpose(1, 2, 0)
    if obs.dtype != np.uint8:
        obs = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
    if obs.shape[-1] == 1:
        obs = np.repeat(obs, 3, axis=-1)
    return obs


def run_dataset_mode(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = args.dataset_root
    repo_id = Path(dataset_root).name
    camera_name = args.camera_name
    save_mp4 = args.mp4
    startup_t0 = time.perf_counter()

    with _timed_phase("load hand-eye transforms (D405 JSON)"):
        T_opt_cam, T_cam_ee = load_tip_kin(args.extrinsics_config)

    if args.camera_info_path:
        with _timed_phase("load camera matrix from file"):
            camera_matrix = load_camera_matrix_from_file(args.camera_info_path)
    else:
        with _timed_phase("load camera matrix from dataset"):
            try:
                camera_matrix = load_camera_matrix_from_dataset(dataset_root)
            except FileNotFoundError:
                camera_matrix = None
                logger.warning(
                    "No camera intrinsics found. The 3D trajectory will still be rendered, "
                    "but the camera-image overlay is disabled. Provide --camera_info_path to enable it."
                )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    policy = None
    preprocessor = postprocessor = None
    action_stats = None
    if args.inference:
        logger.info(f"Loading policy from {args.pretrained_path}")
        with _timed_phase("load policy and pre/post processors"):
            policy, preprocessor, postprocessor = load_policy_and_processors(args.pretrained_path, device)
        with _timed_phase("extract action stats"):
            action_stats = extract_action_stats(preprocessor)

    fps = getattr(policy.config, "fps", 30) if policy else 30
    chunk_size = policy.config.chunk_size if policy else 30

    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    with _timed_phase("load LeRobotDataset"):
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_root,
            delta_timestamps=delta_timestamps,
        )
    logger.info(f"Dataset: {len(dataset)} frames, {len(dataset.meta.episodes)} episodes")

    logger.info(f"[startup] total: {(time.perf_counter() - startup_t0) * 1000:.0f} ms")

    mode_label = "inference" if args.inference else "gt"

    if save_mp4:
        output_dir = Path(args.output_dir) / repo_id
        output_dir.mkdir(parents=True, exist_ok=True)

    fig_3d = ax_3d = None
    if plt is not None:
        if save_mp4:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure as MplFigure

            fig_3d = MplFigure(figsize=(10, 8))
            FigureCanvasAgg(fig_3d)
        else:
            fig_3d = plt.figure(figsize=(10, 8))
            plt.ion()
            fig_3d.show()
        ax_3d = fig_3d.add_subplot(111, projection="3d")

    def update_3d(traj, frame_idx, ep_idx, label, traj_gt=None):
        if ax_3d is None:
            return
        ax_3d.clear()
        if traj_gt is not None and len(traj_gt) > 1:
            ax_3d.plot(traj_gt[:, 0], traj_gt[:, 1], traj_gt[:, 2], "r--", lw=2, label="GT")
        if len(traj) > 1:
            ax_3d.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                "b-",
                lw=2,
                label="Pred" if traj_gt is not None else "Traj",
            )
            ax_3d.plot([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], "go", ms=6)
            ax_3d.plot([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], "ro", ms=6)
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title(f"Ep {ep_idx} F{frame_idx} {label}")
        ax_3d.legend()
        ax_3d.set_xlim(-0.25, 0.25)
        ax_3d.set_ylim(-0.25, 0.25)
        ax_3d.set_zlim(-0.25, 0.25)
        ax_3d.view_init(elev=20, azim=45)

    def render_fig():
        if fig_3d is None:
            return None
        fig_3d.canvas.draw()
        buf = np.asarray(fig_3d.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    metric_rows = []
    try:
        for ep_idx in args.episode_indices:
            logger.info(f"Processing episode {ep_idx} ({mode_label})")

            ep_info = dataset.meta.episodes[ep_idx]
            ep_length = ep_info["length"]
            start_idx = sum(dataset.meta.episodes[i]["length"] for i in range(ep_idx))

            if args.inference:
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()

            proj_frames = []
            traj3d_frames = []

            num_frames = min(ep_length, args.max_frames) if args.max_frames > 0 else ep_length
            for frame_offset in range(num_frames):
                idx = start_idx + frame_offset
                sample = dataset[idx]
                action_is_pad = sample.get("action_is_pad")
                if action_is_pad is not None and bool(torch.as_tensor(action_is_pad).any()):
                    continue
                img = get_image_from_sample(sample, camera_name)

                if args.inference:
                    actions_all = sample["action"]

                    current_state = actions_all[0] if actions_all.ndim > 1 else actions_all
                    if isinstance(current_state, torch.Tensor):
                        state_t = current_state.unsqueeze(0).to(device)
                    else:
                        state_t = torch.from_numpy(np.asarray(current_state)).unsqueeze(0).to(device)

                    obs_img = sample[f"observation.images.{camera_name}"]
                    if isinstance(obs_img, torch.Tensor):
                        img_t = obs_img.unsqueeze(0).to(device)
                    else:
                        img_t = torch.from_numpy(obs_img).unsqueeze(0).to(device)

                    batch = {
                        OBS_STATE: state_t,
                        f"observation.images.{camera_name}": img_t,
                    }
                    task = args.task or sample.get("task")
                    add_policy_task(batch, policy, task)

                    with torch.no_grad():
                        processed = preprocessor(batch)
                        # ACT VAE: use the prior when no GT action is present (inference); otherwise the
                        # None ACTION left by the preprocessor trips the model's posterior branch.
                        if processed.get(ACTION) is None:
                            processed.pop(ACTION, None)
                        pred_10d = policy.predict_action_chunk(processed)

                    actions_rel = unnormalize_actions(pred_10d, action_stats)[0].cpu().numpy()
                    T_rel_list = [rot6d_to_matrix(a) for a in actions_rel]
                    traj_3d = relative_actions_to_3d_points(T_rel_list, T_opt_cam, T_cam_ee)
                    pts_2d = (
                        project_points_to_image(traj_3d, camera_matrix) if camera_matrix is not None else None
                    )
                    gripper_np = actions_rel[:, 9]

                    traj_3d_gt = None
                    if args.gt:
                        gt_actions = sample["action"]
                        if isinstance(gt_actions, torch.Tensor):
                            gt_actions = gt_actions.cpu().numpy()
                        gt_ref = gt_actions[0]
                        T_ref_inv = np.linalg.inv(aa_pose_to_matrix(gt_ref))
                        gt_rel_list = [T_ref_inv @ aa_pose_to_matrix(a) for a in gt_actions]
                        pred_end = T_rel_list[-1]
                        gt_end = gt_rel_list[-1]
                        rot_delta = pred_end[:3, :3].T @ gt_end[:3, :3]
                        rot_error = float(
                            np.arccos(np.clip((np.trace(rot_delta) - 1.0) / 2.0, -1.0, 1.0))
                        )
                        metric_rows.append(
                            {
                                "episode": int(ep_idx),
                                "frame": int(frame_offset),
                                "xyz_end_error_m": float(
                                    np.linalg.norm(pred_end[:3, 3] - gt_end[:3, 3])
                                ),
                                "rotation_end_error_rad": rot_error,
                                "gripper_end_error": float(abs(actions_rel[-1, 9] - gt_actions[-1, 6])),
                            }
                        )
                        traj_3d_gt = relative_actions_to_3d_points(gt_rel_list, T_opt_cam, T_cam_ee)
                        pts_2d_gt = (
                            project_points_to_image(traj_3d_gt, camera_matrix)
                            if camera_matrix is not None
                            else None
                        )
                else:
                    actions_np = sample["action"]
                    if isinstance(actions_np, torch.Tensor):
                        actions_np = actions_np.cpu().numpy()
                    gt_ref = actions_np[0] if actions_np.ndim > 1 else actions_np
                    T_ref_inv = np.linalg.inv(aa_pose_to_matrix(gt_ref))
                    if actions_np.ndim == 1:
                        actions_np = actions_np[np.newaxis, :]
                    rel_list = [T_ref_inv @ aa_pose_to_matrix(a) for a in actions_np]
                    traj_3d = relative_actions_to_3d_points(rel_list, T_opt_cam, T_cam_ee)
                    pts_2d = (
                        project_points_to_image(traj_3d, camera_matrix) if camera_matrix is not None else None
                    )
                    gripper_np = actions_np[:, 6]
                    pts_2d_gt = traj_3d_gt = None

                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if args.inference:
                    if pts_2d is not None:
                        img_bgr = draw_trajectory_on_image(
                            img_bgr, pts_2d, "pred", gripper_np if args.gripper else None
                        )
                    if args.gt and pts_2d_gt is not None:
                        gt_grip = gt_actions[:, 6] if args.gripper else None
                        img_bgr = draw_trajectory_on_image(img_bgr, pts_2d_gt, "gt", gt_grip)
                elif pts_2d is not None:
                    img_bgr = draw_trajectory_on_image(
                        img_bgr, pts_2d, "gt", gripper_np if args.gripper else None
                    )

                update_3d(traj_3d, frame_offset, ep_idx, mode_label, traj_3d_gt)

                if save_mp4:
                    proj_frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    rendered = render_fig()
                    if rendered is not None:
                        traj3d_frames.append(rendered)
                else:
                    cv2.imshow(f"Projection ({mode_label})", img_bgr)
                    if fig_3d is not None:
                        fig_3d.canvas.draw_idle()
                        fig_3d.canvas.flush_events()
                    key = cv2.waitKey(max(1, int(1000 / fps)))
                    if key == 27:
                        return

                if (frame_offset + 1) % 100 == 0:
                    logger.info(f"  {frame_offset + 1}/{ep_length} frames")

            if save_mp4 and iio is not None:
                proj_path = output_dir / f"proj_{mode_label}_episode_{ep_idx}.mp4"
                iio.imwrite(proj_path, np.stack(proj_frames), fps=fps)
                logger.info(f"Saved {proj_path}")
                if traj3d_frames:
                    traj_path = output_dir / f"traj3d_{mode_label}_episode_{ep_idx}.mp4"
                    iio.imwrite(traj_path, np.stack(traj3d_frames), fps=fps)
                    logger.info(f"Saved {traj_path}")

    finally:
        if plt is not None and fig_3d is not None:
            plt.ioff()
            plt.close(fig_3d)
        cv2.destroyAllWindows()

    if metric_rows and save_mp4:
        summary = {
            "num_frames": len(metric_rows),
            "xyz_end_error_mean_m": float(
                np.mean([row["xyz_end_error_m"] for row in metric_rows])
            ),
            "xyz_end_error_median_m": float(
                np.median([row["xyz_end_error_m"] for row in metric_rows])
            ),
            "rotation_end_error_mean_rad": float(
                np.mean([row["rotation_end_error_rad"] for row in metric_rows])
            ),
            "gripper_end_error_mean": float(
                np.mean([row["gripper_end_error"] for row in metric_rows])
            ),
            "frames": metric_rows,
        }
        metrics_path = output_dir / "prediction_metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2))
        logger.info("Saved %s", metrics_path)

    logger.info("Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Visualize UMI-style processor pipeline predictions")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--urdf_path", type=str, default=DEFAULT_URDF_PATH)
    parser.add_argument(
        "--extrinsics_config",
        type=str,
        default=str(DEFAULT_EXTRINSICS),
        help=f"camera_gripper_extrinsics JSON (default: {DEFAULT_EXTRINSICS}). "
        "Used for projection instead of the URDF so it matches sroi visualize_traj_video.py.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Language task required by SmolVLA, for example 'pick the strawberry'",
    )
    parser.add_argument("--num_steps", type=int, default=0, help="0 = infinite")
    parser.add_argument("--gripper", action="store_true", help="Show gripper state on trajectory")

    # Camera mode
    parser.add_argument(
        "--cameras", type=str, default=None, help="YAML camera config. serial auto-detected if omitted."
    )
    parser.add_argument(
        "--camera_info_path",
        type=str,
        default=None,
        help="camera_info.json for intrinsics (auto-detected if omitted)",
    )

    # State
    parser.add_argument(
        "--initial_state",
        type=float,
        nargs=7,
        default=None,
        help="Initial 7D aa state [x,y,z,wx,wy,wz,gripper]",
    )
    parser.add_argument(
        "--update_state", action="store_true", help="Auto-chain: use last prediction as next state"
    )

    # Dataset mode
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--episode_indices", type=int, nargs="+", default=None)
    parser.add_argument("--max_frames", type=int, default=0, help="Maximum frames per episode; 0 = all")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--gt", action="store_true", help="Overlay GT (with --inference)")
    parser.add_argument("--output_dir", type=str, default="outputs/debug/visualization_umi")
    parser.add_argument("--camera_name", type=str, default="camera")
    parser.add_argument("--mp4", action="store_true", help="Save MP4 instead of display")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.cameras:
        if not args.pretrained_path:
            parser.error("--pretrained_path required for camera mode")
        run_camera_mode(args)
    elif args.dataset_root:
        if not args.episode_indices:
            parser.error("--episode_indices required for dataset mode")
        if args.inference and not args.pretrained_path:
            parser.error("--pretrained_path required with --inference")
        run_dataset_mode(args)
    else:
        parser.error("Provide --cameras (camera mode) or --dataset_root (dataset mode)")


if __name__ == "__main__":
    main()

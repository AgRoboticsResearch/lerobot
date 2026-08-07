#!/usr/bin/env python

r"""    
Deploy ACT policy trained with UMI-style processor pipeline on Piper 6-DOF arm.

Uses synchronous control (no subprocess), same pattern as deploy_relative_ee_processor_so101.py.
The preprocessor/postprocessor handle all rot6d ↔ aa conversions automatically.

Hardware:
  - Piper arm via CAN bus (piper_sdk)
  - Gripper: built-in (via PiperInterface) or external DM4310 (via serial)
  - Camera via LeRobot camera abstraction (intelrealsense or opencv)

Usage:
    python deploy_umi_relative_ee_piper.py \
        --pretrained_path outputs/.../pretrained_model \
        --can_port can0 \
        --gripper_port /dev/ttyACM0 \
        --gripper_type external \
        --cameras "{camera: {type: intelrealsense, serial_number_or_name: '230322274337', fps: 30, width: 640, height: 480}}" \
        --warm_start
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from enum import Enum, auto

import cv2
import numpy as np
import torch
import yaml

from lerobot.cameras import CameraConfig

# Ensure RealSenseCameraConfig is importable for parse_cameras_config
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import OBS_STATE

try:
    from examples.umi_relative_ee.control_logger import ControlLogger, make_control_logger
except ModuleNotFoundError:
    # Supports direct execution from inside examples/umi_relative_ee.
    from control_logger import ControlLogger, make_control_logger  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

HOME_POSE_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.00])
SAFE_POSE_DEG = np.array([0.0, -2.08, 1.8, 1.35, 17.16, 0.0])
START_POSE_DEG = np.array([0.0, 79.2, -31.3, 0.0,   -45.85, 0.0])

# External DM4310 gripper (MIT mode, radians)
GRIPPER_OPEN_RAD = 0
GRIPPER_CLOSED_RAD = -0.91

# Built-in Piper gripper (mm) — typical range
GRIPPER_OPEN_MM = 0.0
GRIPPER_CLOSED_MM = 55.0

DEFAULT_URDF_PATH = os.path.normpath(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "sroi-piper",
    "src", "utils", "piper_urdf", "piper_sroiv2.urdf",
)))
DEFAULT_PIPER_SRC_PATH = os.path.normpath(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "sroi-piper", "src",
)))
DEFAULT_DEPLOY_FRAME = "camera_link"


@contextmanager
def _timed_phase(name: str):
    """Time a startup phase and log the elapsed time in ms."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[startup] {name}: {(time.perf_counter() - t0) * 1000:.0f} ms")


# ---------------------------------------------------------------------------
# Non-blocking keyboard input (pynput.Listener + queue)
# ---------------------------------------------------------------------------

# Recognized key names — normalized to lowercase single chars or special tokens.
_KEY_MAP = {
    "q": "q", "r": "r", "s": "s", ".": "dot", "h": "h",
    "space": "space", "esc": "esc",
}


_KEYMAP_HELP = """
╔══════════════════════════════════════════════════════════════╗
║                    KEYBOARD CONTROLS                         ║
║       (policy inference always runs; these keys              ║
║        only control whether the arm executes it)             ║
╠══════════════════════════════════════════════════════════════╣
║  s          engage control — arm starts following policy     ║
║  SPACE      pause control — arm holds pose, overlay live     ║
║  q          move to START pose, then pause control           ║
║  r          move to SAFE pose, then pause control            ║
║  . (period) execute one chunk (~30 actions) then re-pause    ║
║  h          reprint this keymap                              ║
║  ESC        graceful shutdown (cleanup + safe pose)          ║
║  Ctrl+C     force quit (same cleanup)                        ║
╚══════════════════════════════════════════════════════════════╝
""".strip()


class KeyboardCommandHandler:
    """Background-thread non-blocking keyboard reader.

    pynput.Listener pushes recognized keys onto a queue.Queue; the main loop
    polls it via poll() each tick (microsecond cost). Unrecognized keys are
    dropped silently. Safe to use as a context manager.

    Why a thread over select(stdin): works whether or not an OpenCV window
    has focus, and is independent of terminal focus. pynput is already a
    LeRobot dependency (used by teleoperators/keyboard).
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._listener = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        from pynput import keyboard as pk
        self._listener = pk.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def __enter__(self) -> "KeyboardCommandHandler":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def _on_press(self, key) -> None:
        name = self._normalize(key)
        if name is not None:
            self._queue.put(name)

    @staticmethod
    def _normalize(key) -> str | None:
        # pynput passes either a KeyCode (printable) or a Key (special).
        if hasattr(key, "char"):
            ch = (key.char or "").lower()
            return _KEY_MAP.get(ch)
        if hasattr(key, "name"):
            return _KEY_MAP.get(key.name.lower())
        return None

    def poll(self) -> str | None:
        """Return the next queued key name, or None if empty. Non-blocking."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> list[str]:
        """Pop all pending keys (useful after a long blocking move)."""
        out = []
        while True:
            k = self.poll()
            if k is None:
                break
            out.append(k)
        return out


class LoopState(Enum):
    PAUSED = auto()
    INFERENCE = auto()
    MOVING = auto()
    SHUTDOWN = auto()


def gripper_norm_to_external(gripper_norm: float, kp: float, kd: float) -> tuple:
    pos = gripper_norm * (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD) + GRIPPER_OPEN_RAD
    return kp, kd, pos


def gripper_norm_to_builtin(gripper_norm: float) -> float:
    return gripper_norm * (GRIPPER_CLOSED_MM - GRIPPER_OPEN_MM) + GRIPPER_OPEN_MM


def gripper_external_to_norm(pos_rad: float) -> float:
    return (pos_rad - GRIPPER_OPEN_RAD) / (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD)


def gripper_builtin_to_norm(pos_mm: float) -> float:
    return (pos_mm - GRIPPER_OPEN_MM) / (GRIPPER_CLOSED_MM - GRIPPER_OPEN_MM)


def ee_pose_aa_from_fk(kinematics, joints_deg: np.ndarray, gripper_norm: float) -> np.ndarray:
    ee_T = kinematics.forward_kinematics(joints_deg)
    pos = ee_T[:3, 3]
    from scipy.spatial.transform import Rotation
    aa = Rotation.from_matrix(ee_T[:3, :3]).as_rotvec()
    return np.concatenate([pos, aa, [gripper_norm]]).astype(np.float32)


def move_to_pose(piper, gripper, target_deg, *, label, gripper_kp=0.0, gripper_kd=0.0,
                 duration=3.0, open_gripper=True):
    """Move arm to an absolute joint pose (degrees) and (optionally) open the gripper.

    Sends the target to the Piper and lets its internal controller smooth the
    trajectory; sleeps `duration` s to let it arrive. Blocks the caller — by
    design, the inference loop is paused during this.
    """
    logger.info(f"Moving to {label} pose: {target_deg}")
    piper.write_joints(target_deg)
    if gripper is not None:
        if open_gripper:
            gripper.send_command(kp=gripper_kp, kd=gripper_kd, position=0.0)
    else:
        if open_gripper:
            piper.write_gripper(GRIPPER_OPEN_MM)
    time.sleep(duration)


def move_to_safe(piper, gripper, gripper_kp, gripper_kd, duration=3.0):
    """Move arm to safe pose and open gripper before disable."""
    move_to_pose(
        piper, gripper, SAFE_POSE_DEG,
        label="safe", gripper_kp=gripper_kp, gripper_kd=gripper_kd,
        duration=duration, open_gripper=True,
    )


def resync_ik_safety(ik_pipeline, kinematics, piper):
    """Re-anchor the IK safety check to the arm's real pose after a direct move.

    q/r move the arm via piper.write_joints(), which bypasses ik_pipeline — so
    EEBoundsAndSafety._last_pos stays at the last *pre-move* target. Without
    resyncing, the first action after re-engaging (s) is measured against that
    stale pose and rejected as an "EE jump > 0.05m". Set _last_pos to the current
    EE so the next action is compared to where the arm actually is.
    """
    current_joints = piper.read_joints()
    ee_T = kinematics.forward_kinematics(current_joints)
    ee_pos = ee_T[:3, 3].copy()
    for step in ik_pipeline.steps:
        if hasattr(step, "_last_pos"):
            step._last_pos = ee_pos
    logger.info(f"Resynced IK safety _last_pos to current EE pos: {np.round(ee_pos, 3)}")


# ---------------------------------------------------------------------------
# Visualization helpers (from visualize_predictions.py)
# ---------------------------------------------------------------------------

def get_kinematic_transforms(urdf_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return T_opt_cam (optical→cam) and T_cam_ee (cam→ee) at the neutral pose.

    These are static transforms between fixed links, so we use placo's RobotWrapper
    directly for FK — no KinematicsSolver (that's IK-only machinery). The flags skip
    collision-pair setup and visual-mesh parsing, avoiding the slow self-collision
    scan (~15 s on the Piper URDF) and stderr warning spam from the previous
    3x RobotKinematics approach.
    """
    import placo
    flags = placo.Flags.ignore_collisions | placo.Flags.collision_as_visual
    robot = placo.RobotWrapper(str(urdf_path), flags)
    robot.update_kinematics()
    T_opt_cam = np.asarray(robot.get_T_a_b("camera_optical_link", "camera_link"))
    T_cam_ee = np.asarray(robot.get_T_a_b("camera_link", "ee_link"))
    return T_opt_cam, T_cam_ee


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
    T_rel_list: list[np.ndarray], T_opt_cam: np.ndarray, T_cam_ee: np.ndarray,
) -> np.ndarray:
    positions = [(T_opt_cam @ T_cam_ee)[:3, 3].copy()]
    for T_rel in T_rel_list:
        pos = (T_opt_cam @ T_rel @ T_cam_ee)[:3, 3]
        positions.append(pos.copy())
    return np.array(positions)


def project_points_to_image(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points_3d[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    pts = (K @ points_3d.T).T
    return pts[:, :2] / z


def draw_trajectory_on_image(
    img: np.ndarray, points_2d: np.ndarray, gripper: np.ndarray | None = None,
) -> np.ndarray:
    img_draw = img.copy()
    n = len(points_2d)
    if n == 0:
        return img_draw
    colors = [
        (int(255 * i / max(1, n - 1)), int(255 * (1 - i / max(1, n - 1))), 0)
        for i in range(max(1, n - 1))
    ]
    h, w = img.shape[:2]
    for i in range(len(points_2d) - 1):
        pt1 = tuple(points_2d[i].astype(int))
        pt2 = tuple(points_2d[i + 1].astype(int))
        if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
            cv2.line(img_draw, pt1, pt2, colors[i % len(colors)], 2)
            if gripper is not None and i < len(gripper) and gripper[i] < 0.1:
                cv2.drawMarker(img_draw, pt1, (255, 0, 0), cv2.MARKER_CROSS, markerSize=6, thickness=2)
    for pt, col in [(points_2d[0], (0, 255, 0)), (points_2d[-1], (0, 0, 255))]:
        p = tuple(pt.astype(int))
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img_draw, p, 5, col, -1)
    return img_draw


def extract_action_stats(preprocessor) -> dict:
    for step in preprocessor.steps:
        if hasattr(step, "stats") and step.stats and "action" in step.stats:
            return step.stats["action"]
    raise ValueError("No action stats found in preprocessor")


def unnormalize_actions(tensor: torch.Tensor, stats: dict) -> torch.Tensor:
    min_val = torch.as_tensor(stats["min"], device=tensor.device, dtype=tensor.dtype)
    max_val = torch.as_tensor(stats["max"], device=tensor.device, dtype=tensor.dtype)
    return tensor * (max_val - min_val) / 2 + (max_val + min_val) / 2


def auto_detect_realsense_serial() -> str | None:
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            return devices[0].get_info(rs.camera_info.serial_number)
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
        camera_config_class = CameraConfig.get_choice_class(camera_type)
        cameras[name] = camera_config_class(**config)
    return cameras


def load_policy_and_processors(model_path: str, device: torch.device):
    """Load any supported policy and its checkpointed relative-EE processors."""
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(model_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def add_policy_task(batch: dict, policy, task: str | None) -> None:
    if policy.name not in ("smolvla", "pi05"):
        return
    if not task:
        raise ValueError(
            f"{policy.name} deployment requires --task (for this dataset: 'pick the strawberry')."
        )
    # The pi05/SmolVLA prompt tokenizers read `task` from complementary data as
    # a sequence of strings; a bare string would be iterated char-by-char.
    batch["task"] = [task]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy UMI-style relative EE policy on Piper arm"
    )
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--task", type=str, default=None, help="Language task required by SmolVLA")
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--gripper_port", type=str, default="/dev/ttyACM0")
    parser.add_argument(
        "--gripper_type",
        type=str,
        default="external",
        choices=["external", "builtin"],
        help="external (DM4310 serial) or builtin (PiperInterface)",
    )
    parser.add_argument("--cameras", type=str, default=None,
                        help='YAML camera config. serial_number_or_name auto-detected if omitted. e.g. "{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}"')
    parser.add_argument("--urdf_path", type=str, default=DEFAULT_URDF_PATH)
    parser.add_argument("--deploy_frame", type=str, default=DEFAULT_DEPLOY_FRAME)
    parser.add_argument("--n_action_steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable camera + trajectory visualization")
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument("--ee_bounds_min", type=float, nargs=3, default=[-0.5, -0.5, -0.1])
    parser.add_argument("--ee_bounds_max", type=float, nargs=3, default=[0.5, 0.5, 0.6])
    parser.add_argument("--max_ee_step_m", type=float, default=0.05)
    parser.add_argument("--gripper_kp", type=float, default=5.0)
    parser.add_argument("--gripper_kd", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true",
                        help="No arm/gripper — camera + inference + trajectory overlay only")
    parser.add_argument("--camera_info_path", type=str, default=None,
                        help="camera_info.json for intrinsics (dry_run trajectory projection)")
    parser.add_argument("--initial_state", type=float, nargs=7, default=None,
                        help="Initial 7D aa state [x,y,z,wx,wy,wz,gripper] for dry_run")
    parser.add_argument("--update_state", action="store_true",
                        help="Chain predictions: last predicted action becomes next state (dry_run)")
    parser.add_argument(
        "--log", nargs="?", default=None, const="",
        help="Log every control tick for offline analysis. Off by default. "
             "Bare '--log' writes to logs/sync_<timestamp>.csv (+ .npz/.json); "
             "'--log PATH' uses PATH as the stem (suffixes added automatically).",
    )
    return parser.parse_args()


def run_dry_run(args):
    """Camera-only mode: inference + trajectory overlay, no arm/gripper."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load policy + processors
    logger.info(f"Loading policy from: {args.pretrained_path}")
    policy, preprocessor, postprocessor = load_policy_and_processors(args.pretrained_path, device)
    logger.info("Policy and processors loaded")

    action_stats = extract_action_stats(preprocessor)

    # Kinematic transforms for projection
    T_opt_cam, T_cam_ee = get_kinematic_transforms(args.urdf_path)

    # Camera intrinsics
    camera_matrix = None
    if args.camera_info_path:
        import json
        with open(args.camera_info_path) as f:
            info = json.load(f)
        K_flat = info["K"]
        camera_matrix = np.array(K_flat, dtype=np.float64).reshape(3, 3)
        logger.info(f"Loaded camera matrix: fx={camera_matrix[0,0]:.1f}")

    # Connect camera
    cameras_config = parse_cameras_config(args.cameras)
    if not cameras_config:
        raise ValueError("No cameras configured. Use --cameras to specify.")
    cameras = make_cameras_from_configs(cameras_config)
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info(f"Camera connected: {cam_name}")

    # Auto-detect RealSense intrinsics if not provided
    if camera_matrix is None:
        try:
            import pyrealsense2 as rs
            for cam in cameras.values():
                if hasattr(cam, "rs_pipeline") and cam.rs_pipeline is not None:
                    profile = cam.rs_pipeline.get_active_profile()
                    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                    camera_matrix = np.array([
                        [intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1],
                    ])
                    logger.info(f"Auto-detected RealSense intrinsics: fx={intrinsics.fx:.1f}")
                    break
        except Exception:
            pass

    if camera_matrix is None and not args.no_vis:
        logger.warning("No camera intrinsics — trajectory overlay disabled")

    # Identity state (no robot)
    current_state = np.array(
        args.initial_state if args.initial_state else [0, 0, 0, 0, 0, 0, 0.5],
        dtype=np.float32,
    )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    step_count = 0
    logger.info("Starting dry-run loop (Ctrl+C to stop)")

    try:
        while args.num_steps == 0 or step_count < args.num_steps:
            t0 = time.perf_counter()

            state_tensor = torch.from_numpy(current_state).unsqueeze(0).to(device)
            batch = {OBS_STATE: state_tensor}
            add_policy_task(batch, policy, args.task)

            for cam_name, camera in cameras.items():
                img = camera.read()
                img_float = img.astype(np.float32) / 255.0
                img_chw = np.transpose(img_float, (2, 0, 1))
                batch[f"observation.images.{cam_name}"] = (
                    torch.from_numpy(img_chw).unsqueeze(0).to(device)
                )

            with torch.no_grad():
                processed = preprocessor(batch)
                pred_10d = policy.predict_action_chunk(processed)

            # Unnormalize to get relative rot6d actions
            actions_rel = unnormalize_actions(pred_10d, action_stats).cpu().numpy()
            actions_rel = actions_rel[0] if actions_rel.ndim == 3 else actions_rel

            # Chain state if requested
            if args.update_state:
                from scipy.spatial.transform import Rotation
                T_ref = np.eye(4)
                T_ref[:3, :3] = Rotation.from_rotvec(current_state[3:6]).as_matrix()
                T_ref[:3, 3] = current_state[:3]
                T_rel_last = rot6d_to_matrix(actions_rel[-1])
                T_abs_last = T_ref @ T_rel_last
                current_state = np.concatenate([
                    T_abs_last[:3, 3],
                    Rotation.from_matrix(T_abs_last[:3, :3]).as_rotvec(),
                    [actions_rel[-1, 9]],
                ]).astype(np.float32)

            # Project trajectory
            if not args.no_vis:
                T_rel_list = [rot6d_to_matrix(a) for a in actions_rel]
                traj_3d = relative_actions_to_3d_points(T_rel_list, T_opt_cam, T_cam_ee)

                for cam_name in cameras:
                    cam_img = batch[f"observation.images.{cam_name}"].cpu().numpy()
                    cam_img = np.transpose(cam_img[0], (1, 2, 0))
                    cam_img = (cam_img * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)

                    if camera_matrix is not None:
                        pts_2d = project_points_to_image(traj_3d, camera_matrix)
                        img_bgr = draw_trajectory_on_image(img_bgr, pts_2d)

                    cv2.imshow(f"DRY RUN: {cam_name}", img_bgr)

            elapsed = time.perf_counter() - t0
            fps_actual = 1.0 / max(elapsed, 1e-6)
            if step_count % 10 == 0:
                logger.info(f"Step {step_count}: {elapsed*1000:.0f}ms ({fps_actual:.0f} fps)")
            step_count += 1

            key = cv2.waitKey(1) if not args.no_vis else 0
            if key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        for cam_name, camera in cameras.items():
            try:
                camera.disconnect()
            except Exception:
                pass
        logger.info(f"Dry-run done after {step_count} steps")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    if args.dry_run:
        run_dry_run(args)
        return

    # Import Piper hardware interfaces
    if DEFAULT_PIPER_SRC_PATH not in sys.path:
        sys.path.insert(0, DEFAULT_PIPER_SRC_PATH)

    from modules.piper_interface import PiperInterface

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 1. Load policy + processors ────────────────────────────────────
    model_path = args.pretrained_path
    logger.info(f"Loading policy from: {model_path}")

    with _timed_phase("load policy + processors"):
        policy, preprocessor, postprocessor = load_policy_and_processors(model_path, device)
    logger.info("Policy and processors loaded")

    # Visualization setup (shared by both modes)
    with _timed_phase("extract action stats"):
        action_stats = extract_action_stats(preprocessor) if not args.no_vis else None
    viz_T_opt_cam = viz_T_cam_ee = camera_matrix = None
    if not args.no_vis:
        with _timed_phase("load kinematic transforms (URDF)"):
            viz_T_opt_cam, viz_T_cam_ee = get_kinematic_transforms(args.urdf_path)
        if args.camera_info_path:
            import json
            with open(args.camera_info_path) as f:
                info = json.load(f)
            camera_matrix = np.array(info["K"], dtype=np.float64).reshape(3, 3)
        # Try auto-detect from RealSense later after camera connect

    # ── 2. Initialize kinematics ───────────────────────────────────────
    with _timed_phase("init IK kinematics"):
        kinematics = RobotKinematics(
            urdf_path=args.urdf_path,
            target_frame_name=args.deploy_frame,
            joint_names=ARM_JOINTS,
        )
    logger.info(f"URDF: {args.urdf_path}, frame: {args.deploy_frame}")

    # ── 3. Build IK pipeline (7D aa absolute → 6 joint positions) ──────
    ik_pipeline = RobotProcessorPipeline(
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": args.ee_bounds_min, "max": args.ee_bounds_max},
                max_ee_step_m=args.max_ee_step_m,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=ARM_JOINTS,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # ── 4. Connect Piper arm ───────────────────────────────────────────
    logger.info(f"Connecting Piper arm on {args.can_port}")
    piper = PiperInterface(can_port=args.can_port)
    piper.connect()
    logger.info("Piper arm connected")

    # ── 5. Connect gripper (before any movement) ──────────────────────
    gripper = None
    if args.gripper_type == "external":
        from modules.gripper import Gripper

        logger.info(f"Connecting DM4310 gripper on {args.gripper_port}")
        gripper = Gripper(port=args.gripper_port)
        try:
            gripper.connect()
            gripper.send_command(kp=args.gripper_kp, kd=args.gripper_kd, position=0.0)
            logger.info("External gripper connected (position=0, open)")
        except Exception as e:
            logger.error(f"Gripper connection failed: {e}")
            piper.disconnect()
            logger.info("Arm disabled (gripper required but failed)")
            return
    else:
        logger.info("Using built-in Piper gripper")

    # Move to safe pose then start pose
    move_to_safe(piper, gripper, args.gripper_kp, args.gripper_kd, duration=2.0)

    logger.info(f"Moving to start pose: {START_POSE_DEG}")
    piper.write_joints(START_POSE_DEG)
    if gripper is not None:
        gripper.send_command(kp=args.gripper_kp, kd=args.gripper_kd, position=0.0)
    time.sleep(2.0)

    # ── 6. Connect cameras ─────────────────────────────────────────────
    cameras_config = parse_cameras_config(args.cameras)
    if not cameras_config:
        raise ValueError("No cameras configured. Use --cameras to specify.")
    cameras = make_cameras_from_configs(cameras_config)
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info(f"Camera connected: {cam_name}")

    # Auto-detect RealSense intrinsics if needed
    if not args.no_vis and camera_matrix is None:
        try:
            import pyrealsense2 as rs
            for cam in cameras.values():
                if hasattr(cam, "rs_pipeline") and cam.rs_pipeline is not None:
                    profile = cam.rs_pipeline.get_active_profile()
                    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                    camera_matrix = np.array([
                        [intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1],
                    ])
                    logger.info(f"Auto-detected RealSense intrinsics: fx={intrinsics.fx:.1f}")
                    break
        except Exception:
            pass
        if camera_matrix is None:
            logger.warning("No camera intrinsics — trajectory overlay disabled")

    # ── 7. Warm start ──────────────────────────────────────────────────
    if args.warm_start:
        logger.info(f"Moving to home pose: {HOME_POSE_DEG}")
        piper.write_joints(HOME_POSE_DEG)
        if gripper is not None:
            gripper.send_command(kp=args.gripper_kp, kd=args.gripper_kd, position=0.0)
        else:
            piper.write_gripper(GRIPPER_OPEN_MM)
        time.sleep(3.0)
        # Flush camera frames
        for cam_name, camera in cameras.items():
            camera.read()

    # Reset processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # ── 8. Main control loop ───────────────────────────────────────────
    step_count = 0
    ctrl_log: ControlLogger | None = None
    action_queue = []
    chunk_traj_3d = None
    # UMI relative-EE chunk provenance for the --log (anchor + raw relative per chunk).
    chunk_id = 0
    chunk_ref_ee = None       # 7D EE pose that anchored the current chunk (T_anchor)
    chunk_rel = None          # raw per-step relative model output [T, D] for the current chunk
    chunk_idx = 0             # position within the current chunk (advances per pop)

    logger.info(f"Starting control loop at {args.fps} Hz")
    print(_KEYMAP_HELP)

    ctrl_log = make_control_logger(args, prefix="sync")
    if ctrl_log is not None:
        ctrl_log.start()
    tick = 0

    def log_tick() -> None:
        """Record one control tick to the --log (no-op when logging is off).

        Reads the per-tick `diag` dict + loop locals; called at the end of BOTH
        the PAUSED and INFERENCE branches so every tick is logged.
        """
        if ctrl_log is None:
            return
        if state == LoopState.PAUSED:
            # Sync pops every tick (so the overlay advances during pause) but
            # never runs IK while paused → label distinctly from real skips.
            diag["skip_reason"] = "paused"
        elif not diag["popped"]:
            diag["skip_reason"] = "underrun"
        ctrl_log.log(
            step=step_count,
            tick=tick,
            tick_dt_ms=(time.perf_counter() - t0) * 1000.0,
            state=state.name,
            queue=len(action_queue),
            e2e_ms=None, wire_ms=None, server_ms=None,
            **diag,
        )

    try:
        with KeyboardCommandHandler() as kb:
            state = LoopState.PAUSED
            single_chunk = False
            logger.info("Loop state: PAUSED (press s to engage control)")

            while args.num_steps == 0 or step_count < args.num_steps:
                t0 = time.perf_counter()
                tick += 1
                diag = {
                    "popped": False, "ik_ok": False, "skip_reason": "no_action",
                    "action_timestep": None, "action_ee": None, "ik_joints_rad": None,
                    "current_ee": None, "current_joints_rad": None,
                    "ee_delta_m": None, "joint_delta_max_rad": None, "gripper": None,
                    "work_ms": None,
                    "chunk_id": None, "chunk_ref_ee": None,
                    "action_abs": None, "action_agg": None, "action_rel": None,
                }

                # ── 8a. Drain keyboard queue ────────────────────────────
                key = kb.poll()
                while key is not None:
                    if key == "esc":
                        state = LoopState.SHUTDOWN
                        logger.info("ESC — graceful shutdown")
                        break
                    elif key == "q":
                        logger.info("q — moving to START pose (control paused)")
                        move_to_pose(
                            piper, gripper, START_POSE_DEG,
                            label="start",
                            gripper_kp=args.gripper_kp, gripper_kd=args.gripper_kd,
                            duration=2.0, open_gripper=True,
                        )
                        resync_ik_safety(ik_pipeline, kinematics, piper)
                        action_queue.clear()
                        chunk_traj_3d = None
                        state = LoopState.PAUSED
                        kb.drain()
                    elif key == "r":
                        logger.info("r — moving to SAFE pose (control paused)")
                        move_to_pose(
                            piper, gripper, SAFE_POSE_DEG,
                            label="safe",
                            gripper_kp=args.gripper_kp, gripper_kd=args.gripper_kd,
                            duration=3.0, open_gripper=True,
                        )
                        resync_ik_safety(ik_pipeline, kinematics, piper)
                        action_queue.clear()
                        chunk_traj_3d = None
                        state = LoopState.PAUSED
                        kb.drain()
                    elif key == "s" and state == LoopState.PAUSED:
                        action_queue.clear()
                        chunk_traj_3d = None
                        state = LoopState.INFERENCE
                        logger.info("s — control engaged (arm follows policy)")
                    elif key == "space" and state == LoopState.INFERENCE:
                        state = LoopState.PAUSED
                        logger.info("space — control paused (arm holds pose, inference still runs)")
                    elif key == "dot" and state == LoopState.PAUSED:
                        action_queue.clear()
                        single_chunk = True
                        state = LoopState.INFERENCE
                        logger.info(". — execute one chunk then re-pause")
                    elif key == "h":
                        print(_KEYMAP_HELP)
                    key = kb.poll()

                if state == LoopState.SHUTDOWN:
                    break

                # ── 8b. Read sensors (always — feedback during pause too) ─
                current_joints = piper.read_joints()

                if gripper is not None:
                    grip_pos_rad = gripper.position
                    gripper_norm = gripper_external_to_norm(grip_pos_rad)
                else:
                    grip_pos_mm = piper.read_gripper()
                    gripper_norm = gripper_builtin_to_norm(grip_pos_mm)
                gripper_norm = np.clip(gripper_norm, 0.0, 1.0)

                ee_aa = ee_pose_aa_from_fk(kinematics, current_joints, gripper_norm)
                diag["current_ee"] = ee_aa
                diag["current_joints_rad"] = current_joints
                state_tensor = torch.from_numpy(ee_aa).unsqueeze(0).to(device)
                batch = {OBS_STATE: state_tensor}
                add_policy_task(batch, policy, args.task)

                # ── 8c. Read cameras into batch (viz deferred until 8e) ──
                for cam_name, camera in cameras.items():
                    img = camera.read()  # (H, W, 3) RGB uint8
                    img_float = img.astype(np.float32) / 255.0
                    img_chw = np.transpose(img_float, (2, 0, 1))
                    batch[f"observation.images.{cam_name}"] = (
                        torch.from_numpy(img_chw).unsqueeze(0).to(device)
                    )

                # Update the two-frame relative state every control tick. Generate
                # and convert a complete chunk only when the absolute-action queue
                # is empty, so every action retains the same chunk-start base pose.
                with torch.no_grad():
                    processed = preprocessor(batch)

                if len(action_queue) == 0:
                    with torch.no_grad():
                        pred_norm = policy.predict_action_chunk(processed)

                        viz_pred_rel = None
                        if not args.no_vis and camera_matrix is not None and action_stats is not None:
                            viz_pred_rel = unnormalize_actions(
                                pred_norm.clone(), action_stats
                            ).cpu().numpy()
                            if viz_pred_rel.ndim == 3:
                                viz_pred_rel = viz_pred_rel[0]

                        pred = postprocessor(pred_norm)

                    if isinstance(pred, dict) and "action" in pred:
                        pred = pred["action"]

                    actions_aa = pred[0].cpu().numpy()
                    if step_count < 5:
                        logger.info(f"  Postprocessor output shape: {pred.shape}, dtype: {pred.dtype}")
                        logger.info(f"  First action_aa: {actions_aa[0]}")
                        logger.info(f"  Last action_aa:  {actions_aa[-1]}")
                    action_queue = [
                        actions_aa[i]
                        for i in range(min(args.n_action_steps, len(actions_aa)))
                    ]
                    # UMI chunk provenance for the --log: this chunk is anchored at the
                    # current EE pose (ee_aa), and chunk_rel is its raw relative model
                    # output [T, D] (pred_norm, before the postprocessor made it absolute).
                    chunk_id += 1
                    chunk_ref_ee = ee_aa.copy()
                    chunk_rel = pred_norm[0].detach().cpu().numpy()
                    chunk_idx = 0
                    logger.info(f"Predicted chunk of {len(action_queue)} actions")

                    chunk_traj_3d = None
                    if viz_pred_rel is not None:
                        T_rel_list = [rot6d_to_matrix(a) for a in viz_pred_rel]
                        chunk_traj_3d = relative_actions_to_3d_points(
                            T_rel_list, viz_T_opt_cam, viz_T_cam_ee
                        )

                # Pop one action (queue advances in both states so overlay cadence
                # matches inference cadence — typically one new chunk per n_action_steps).
                action_aa = action_queue.pop(0)
                diag["popped"] = True
                diag["action_ee"] = action_aa
                # Per-tick chunk provenance: anchor + raw relative + the absolute target
                # (SYNC replays one chunk verbatim, so pre-ensemble == executed).
                diag["chunk_id"] = chunk_id
                diag["chunk_ref_ee"] = chunk_ref_ee
                diag["action_abs"] = action_aa
                diag["action_agg"] = action_aa
                if chunk_rel is not None and chunk_idx < len(chunk_rel):
                    diag["action_rel"] = chunk_rel[chunk_idx]
                chunk_idx += 1
                # In single-chunk mode: detect when we just popped the last action
                # of the chunk so we can return to PAUSED after this tick's write.
                last_of_chunk = single_chunk and len(action_queue) == 0

                # ── 8e. Render viz with the freshly-computed overlay ─────
                if not args.no_vis:
                    for cam_name in cameras:
                        cam_img = batch[f"observation.images.{cam_name}"].cpu().numpy()
                        cam_img = np.transpose(cam_img[0], (1, 2, 0))
                        cam_img = (cam_img * 255).astype(np.uint8)
                        img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                        if camera_matrix is not None and chunk_traj_3d is not None:
                            pts_2d = project_points_to_image(chunk_traj_3d, camera_matrix)
                            img_bgr = draw_trajectory_on_image(img_bgr, pts_2d)
                        cv2.imshow(f"Piper: {cam_name}", img_bgr)
                    cv2.waitKey(1)

                # ── 8f. PAUSED: arm holds pose, loop continues ───────────
                if state == LoopState.PAUSED:
                    elapsed = time.perf_counter() - t0
                    diag["work_ms"] = elapsed * 1000.0
                    if elapsed < 1.0 / args.fps:
                        time.sleep(1.0 / args.fps - elapsed)
                    log_tick()
                    continue

                # ── 8g. INFERENCE: IK + write to motors ──────────────────
                action_dict = {
                    "ee.x": float(action_aa[0]),
                    "ee.y": float(action_aa[1]),
                    "ee.z": float(action_aa[2]),
                    "ee.wx": float(action_aa[3]),
                    "ee.wy": float(action_aa[4]),
                    "ee.wz": float(action_aa[5]),
                    "ee.gripper_pos": float(action_aa[6]),
                }
                observation_dict = {
                    f"{name}.pos": float(current_joints[i])
                    for i, name in enumerate(ARM_JOINTS)
                }

                if step_count < 5 or step_count % 100 == 0:
                    logger.info(
                        f"  action_aa: [{action_aa[0]:.4f}, {action_aa[1]:.4f}, {action_aa[2]:.4f}, "
                        f"{action_aa[3]:.4f}, {action_aa[4]:.4f}, {action_aa[5]:.4f}, grip={action_aa[6]:.3f}]"
                    )
                    logger.info(f"  current_joints: {current_joints}")

                ik_failed = False
                try:
                    result = ik_pipeline((action_dict, observation_dict))
                except Exception as e:
                    logger.warning(f"IK failed: {e}, skipping action")
                    ik_failed = True
                    diag["skip_reason"] = "ik_failed"

                if not ik_failed:
                    joint_values = np.array(
                        [result.get(f"{name}.pos", 0.0) for name in ARM_JOINTS]
                    )
                    if step_count < 5 or step_count % 100 == 0:
                        logger.info(f"  IK result joints: {joint_values}")
                        logger.info(f"  joint delta: {joint_values - current_joints}")
                    piper.write_joints(joint_values)

                    gripper_value = float(action_aa[6])
                    if gripper is not None:
                        kp, kd, pos_rad = gripper_norm_to_external(
                            gripper_value, args.gripper_kp, args.gripper_kd
                        )
                        gripper.send_command(kp=kp, kd=kd, position=pos_rad)
                    else:
                        pos_mm = gripper_norm_to_builtin(gripper_value)
                        piper.write_gripper(pos_mm)

                    diag.update(
                        ik_ok=True, skip_reason=None, ik_joints_rad=joint_values,
                        gripper=gripper_value,
                        ee_delta_m=float(np.linalg.norm(action_aa[:3] - ee_aa[:3])),
                        joint_delta_max_rad=float(np.max(np.abs(joint_values - current_joints))),
                    )

                step_count += 1

                # Single-chunk mode: return to PAUSED after executing the whole chunk
                if last_of_chunk:
                    single_chunk = False
                    state = LoopState.PAUSED
                    action_queue.clear()
                    logger.info("Chunk executed — back to PAUSED")

                elapsed = time.perf_counter() - t0
                diag["work_ms"] = elapsed * 1000.0
                if elapsed < 1.0 / args.fps:
                    time.sleep(1.0 / args.fps - elapsed)
                log_tick()

                if step_count % 100 == 0:
                    logger.info(
                        f"Step {step_count}: EE pos [{ee_aa[0]:.3f}, {ee_aa[1]:.3f}, {ee_aa[2]:.3f}]"
                    )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during control loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        for cam_name, camera in cameras.items():
            try:
                camera.disconnect()
            except Exception:
                pass
        if ctrl_log is not None:
            try:
                ctrl_log.close()
            except Exception:
                logger.exception("Failed to write control log")
        # Always return to safe pose before disconnecting
        try:
            move_to_safe(piper, gripper, args.gripper_kp, args.gripper_kd)
        except Exception as e2:
            logger.warning(f"Failed to move to safe pose: {e2}")
            input("Robot is NOT at safe pose. Press Enter to disable motors anyway...")
        if gripper is not None:
            gripper.disconnect()
        piper.disconnect()
        logger.info(f"Disconnected after {step_count} steps")


if __name__ == "__main__":
    main()

"""Shared inference-time dataset recorder for the UMI relative-EE Piper deploy scripts.

Both the sync deploy (``deploy_umi_relative_ee_piper.py``) and the async client
(``async_umi_relative_ee_piper_client.py``) import :class:`DeployDatasetRecorder`
/ :func:`make_deploy_dataset_recorder` from here so their ``--save_dataset``
output shares ONE LeRobotDataset schema — runs from either script can be diffed
directly, and the column notation matches the ``--log`` ControlLogger fields
exactly. Full design contract: ``doc/inference_time_saved_dataset.md``.

Recording semantics:
  * one frame per EXECUTED control tick (an action was popped and sent to IK
    while the loop was in the INFERENCE state — IK-failed attempts are recorded
    too, with ``action.joints`` = NaN and ``action.executed_ok`` = False);
  * one episode per engagement: the episode opens lazily on the first executed
    tick (``s`` / ``.``) and closes on the first non-INFERENCE tick afterwards
    (``SPACE`` / ``q`` / ``r`` / single-chunk end) or at shutdown;
  * without ``--save_dataset`` the factory returns None and every hook is a no-op
    (the deploy loop is untouched).

Notation (= control_logger.py per-tick fields, units fixed for the dataset):
    observation.state      current arm joints in RADIANS (converted here —
                           piper.read_joints() and the diag ``*_rad`` fields are
                           actually degrees)
    observation.ee_pose    current FK EE pose [x,y,z,wx,wy,wz,gripper] (m, rotvec, 0..1)
    observation.images.*   raw camera frames (encoded to mp4 per episode)
    action                 action_agg — executed absolute EE target (post-ensemble)
    action.pre_ensemble    action_abs — the popped chunk's pre-blend absolute target
                           (identical to ``action`` in sync mode)
    action.relative        action_rel — raw NORMALIZED 10D rot6d relative model output;
                           the physical ΔT is recoverable offline as
                           inv(action.reference_ee) ∘ action, or by unnormalizing with
                           the checkpoint's action stats
    action.reference_ee    chunk_ref_ee — the 7D EE pose that anchored the chunk
    action.joints          ik_joints_rad — IK joint command in RADIANS (NaN if IK failed)
    chunk_id               diag chunk_id (float32 for NaN-friendliness)
    action.executed_ok     diag ik_ok (False on invalid / IK-failed ticks)
    e2e_ms/wire_ms/server_ms  async latency columns (NaN in sync; e2e only on the tick
                           a fresh chunk's first action executes)
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Typed for static checkers only; the real import stays lazy so importing this
    # module costs nothing when --save_dataset is off.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

# Camera names become both feature keys and video path components — keep them
# path-safe (this also rejects "/", which the dataset itself forbids in keys).
_CAMERA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_EE_NAMES = ["x", "y", "z", "wx", "wy", "wz", "gripper"]
_REL_NAMES = ["x", "y", "z", "r0x", "r0y", "r0z", "r1x", "r1y", "r1z", "gripper"]
_LATENCY_KEYS = ("e2e_ms", "wire_ms", "server_ms")


def build_deploy_features(arm_joint_names: list[str], camera_shapes: dict[str, tuple[int, int]]) -> dict:
    """Single source of truth for the ``--save_dataset`` feature schema."""
    n_joints = len(arm_joint_names)
    features = {
        "observation.state": {"dtype": "float32", "shape": (n_joints,), "names": list(arm_joint_names)},
        "observation.ee_pose": {"dtype": "float32", "shape": (7,), "names": list(_EE_NAMES)},
        "observation.gripper_position": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": list(_EE_NAMES)},
        "action.pre_ensemble": {"dtype": "float32", "shape": (7,), "names": list(_EE_NAMES)},
        "action.relative": {"dtype": "float32", "shape": (10,), "names": list(_REL_NAMES)},
        "action.reference_ee": {"dtype": "float32", "shape": (7,), "names": list(_EE_NAMES)},
        "action.joints": {"dtype": "float32", "shape": (n_joints,), "names": list(arm_joint_names)},
        "chunk_id": {"dtype": "float32", "shape": (1,), "names": None},
        "action.executed_ok": {"dtype": "bool", "shape": (1,), "names": None},
        **{key: {"dtype": "float32", "shape": (1,), "names": None} for key in _LATENCY_KEYS},
    }
    for cam, (height, width) in camera_shapes.items():
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": None,
        }
    return features


class DeployDatasetRecorder:
    """Records the inference-time stream of one deploy run to a LeRobotDataset.

    The dataset is created lazily on the first recorded frame, so a run where the
    policy is never engaged leaves nothing on disk. Any internal failure disables
    recording (one loud log) instead of propagating into the control loop.
    """

    def __init__(
        self,
        root: Path,
        *,
        repo_id: str,
        fps: int,
        task: str,
        camera_shapes: dict[str, tuple[int, int]],
        arm_joint_names: list[str],
        image_writer_threads: int = 4,
    ) -> None:
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = int(fps)
        self.task = task
        self.camera_shapes = dict(camera_shapes)
        self.arm_joint_names = list(arm_joint_names)
        self._image_writer_threads = image_writer_threads
        self._features = build_deploy_features(arm_joint_names, camera_shapes)
        self._dataset: LeRobotDataset | None = None
        self._episode_open = False
        self._frames_in_episode = 0
        self._episodes_saved = 0
        self._total_frames = 0
        self._closed = False
        self._disabled = False

    # ------------------------------------------------------------------ helpers

    def _ensure_dataset(self) -> LeRobotDataset:
        if self._dataset is None:
            # Lazy import: importing the module stays cheap when --save_dataset is off.
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self._dataset = LeRobotDataset.create(
                self.repo_id,
                fps=self.fps,
                features=self._features,
                root=self.root,
                robot_type="piper",
                use_videos=True,
                image_writer_threads=self._image_writer_threads,
            )
            logger.info(
                "Recording inference-time dataset → %s (repo_id=%s, fps=%s, cameras=%s)",
                self.root,
                self.repo_id,
                self.fps,
                sorted(self.camera_shapes),
            )
        return self._dataset

    @staticmethod
    def _vec(value, width: int) -> np.ndarray:
        if value is None:
            return np.full(width, np.nan, dtype=np.float32)
        return np.asarray(value, dtype=np.float32).reshape(width)

    @staticmethod
    def _joints_vec(value, width: int) -> np.ndarray:
        # Piper joints flow through the deploy loop in DEGREES (piper.read_joints()
        # and the diag *_rad fields are misnomers); LeRobot datasets use radians.
        if value is None:
            return np.full(width, np.nan, dtype=np.float32)
        return np.deg2rad(np.asarray(value, dtype=np.float64)).astype(np.float32).reshape(width)

    @staticmethod
    def _scalar(value) -> np.ndarray:
        return np.array([np.nan if value is None else float(value)], dtype=np.float32)

    def _build_frame(self, images, current_joints, current_ee, diag, gripper_raw, latencies) -> dict:
        latencies = latencies or {}
        frame = {
            "observation.state": self._joints_vec(current_joints, len(self.arm_joint_names)),
            "observation.ee_pose": self._vec(current_ee, 7),
            "observation.gripper_position": self._scalar(gripper_raw),
            "action": self._vec(diag.get("action_agg"), 7),
            "action.pre_ensemble": self._vec(diag.get("action_abs"), 7),
            "action.relative": self._vec(diag.get("action_rel"), 10),
            "action.reference_ee": self._vec(diag.get("chunk_ref_ee"), 7),
            "action.joints": self._joints_vec(diag.get("ik_joints_rad"), len(self.arm_joint_names)),
            "chunk_id": self._scalar(diag.get("chunk_id")),
            "action.executed_ok": np.array([bool(diag.get("ik_ok", False))], dtype=np.bool_),
            **{key: self._scalar(latencies.get(key)) for key in _LATENCY_KEYS},
            "task": self.task,
        }
        for cam in self.camera_shapes:
            frame[f"observation.images.{cam}"] = images[cam]
        return frame

    def _end_episode(self) -> None:
        frames = self._frames_in_episode
        self._episode_open = False
        self._frames_in_episode = 0
        if frames == 0 or self._dataset is None:
            return
        t0 = time.perf_counter()
        self._dataset.save_episode()
        self._episodes_saved += 1
        # The mp4 encode blocks here — by design: the arm is paused/holding and the
        # stall is visible in the companion --log as one large tick_dt_ms.
        logger.info(
            "Saved dataset episode %d (%d frames, video encode %.2fs)",
            self._episodes_saved,
            frames,
            time.perf_counter() - t0,
        )

    # -------------------------------------------------------------------- api

    def note_loop_state(self, state_name: str) -> None:
        """Feed the loop state every tick; closes the open episode on disengage.

        Episode opening is NOT done here — it happens lazily on the first
        ``record_tick``, so states without executed actions never create files.
        """
        if self._closed or self._disabled or not self._episode_open:
            return
        if state_name == "INFERENCE":
            return
        try:
            self._end_episode()
        except Exception:
            logger.exception("Deploy dataset recorder failed while saving an episode — disabling recording")
            self._disabled = True

    def record_tick(
        self,
        *,
        images: dict,
        current_joints,
        current_ee,
        diag: dict,
        gripper_raw=None,
        latencies: dict | None = None,
    ) -> None:
        """Append one frame for an executed tick (opens the episode lazily).

        ``gripper_raw`` is the RAW physical gripper read-back of this tick
        (radians for the external DM4310, mm for the builtin Piper gripper).
        """
        if self._closed or self._disabled:
            return
        try:
            dataset = self._ensure_dataset()
            if not self._episode_open:
                self._episode_open = True
                self._frames_in_episode = 0
            dataset.add_frame(
                self._build_frame(images, current_joints, current_ee, diag, gripper_raw, latencies)
            )
            self._frames_in_episode += 1
            self._total_frames += 1
        except Exception:
            logger.exception("Deploy dataset recorder failed — disabling recording (control continues)")
            self._disabled = True

    def close(self) -> None:
        """Save the trailing episode + finalize. Idempotent; never raises."""
        if self._closed:
            return
        self._closed = True
        if self._dataset is None:
            return  # never engaged → nothing was created on disk
        try:
            if self._episode_open:
                self._end_episode()
            self._dataset.finalize()
            logger.info(
                "Inference-time dataset saved: %s (%d episodes, %d frames)",
                self.root,
                self._episodes_saved,
                self._total_frames,
            )
        except Exception:
            logger.exception("Failed to finalize inference-time dataset %s — it may be incomplete", self.root)


def make_deploy_dataset_recorder(
    args,
    *,
    prefix: str,
    cameras: dict,
    fps: int,
    arm_joint_names: list[str],
    task: str | None = None,
) -> DeployDatasetRecorder | None:
    """Build a :class:`DeployDatasetRecorder` from ``--save_dataset``, or None when off.

    ``args.save_dataset`` is ``None`` (off), ``""`` (bare flag →
    ``outputs/deploy_datasets/<prefix>_<timestamp>``), or a dataset root path that
    must not exist yet (each run creates a fresh dataset; there is no resume).
    Camera shapes are probed with one read per camera — call this AFTER the
    cameras are connected.
    """
    save = getattr(args, "save_dataset", None)
    if save is None:
        return None
    if save:
        root = Path(save)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path("outputs/deploy_datasets") / f"{prefix}_{stamp}"
    if root.exists():
        raise ValueError(f"--save_dataset path already exists: {root} (each run creates a fresh dataset)")

    camera_shapes: dict[str, tuple[int, int]] = {}
    for name, camera in cameras.items():
        if not _CAMERA_NAME_RE.match(name or ""):
            raise ValueError(
                f"Camera name {name!r} is not --save_dataset-safe (need [A-Za-z0-9_-]+): "
                "it is used both as a feature key and as a video path component"
            )
        frame = np.asarray(camera.read())
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Camera {name!r} probe returned shape {frame.shape}, expected (H, W, 3)")
        camera_shapes[name] = (frame.shape[0], frame.shape[1])

    repo_id = re.sub(r"[^A-Za-z0-9._-]", "_", root.name) or f"{prefix}_deploy"
    return DeployDatasetRecorder(
        root,
        repo_id=repo_id,
        fps=fps,
        task=task or f"umi relative-ee deploy ({prefix})",
        camera_shapes=camera_shapes,
        arm_joint_names=arm_joint_names,
    )

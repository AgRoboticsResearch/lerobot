#!/usr/bin/env python

"""Piper robot client for asynchronous UMI relative-EE inference.

The robot, cameras, FK, IK, safety checks, and keyboard controls stay on this
machine. Only policy preprocessing/inference/postprocessing runs on the remote
policy server.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle  # nosec
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import grpc
import numpy as np
import torch

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    visualize_action_queue_size,
)
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STATE

try:
    from examples.umi_relative_ee.deploy_umi_relative_ee_piper import (
        ARM_JOINTS,
        DEFAULT_DEPLOY_FRAME,
        DEFAULT_PIPER_SRC_PATH,
        DEFAULT_URDF_PATH,
        HOME_POSE_DEG,
        SAFE_POSE_DEG,
        START_POSE_DEG,
        KeyboardCommandHandler,
        LoopState,
        ee_pose_aa_from_fk,
        gripper_builtin_to_norm,
        gripper_external_to_norm,
        gripper_norm_to_builtin,
        gripper_norm_to_external,
        move_to_pose,
        move_to_safe,
        parse_cameras_config,
        resync_ik_safety,
    )
except ModuleNotFoundError:
    # Supports direct execution from inside examples/umi_relative_ee.
    from deploy_umi_relative_ee_piper import (  # type: ignore[no-redef]
        ARM_JOINTS,
        DEFAULT_DEPLOY_FRAME,
        DEFAULT_PIPER_SRC_PATH,
        DEFAULT_URDF_PATH,
        HOME_POSE_DEG,
        SAFE_POSE_DEG,
        START_POSE_DEG,
        KeyboardCommandHandler,
        LoopState,
        ee_pose_aa_from_fk,
        gripper_builtin_to_norm,
        gripper_external_to_norm,
        gripper_norm_to_builtin,
        gripper_norm_to_external,
        move_to_pose,
        move_to_safe,
        parse_cameras_config,
        resync_ik_safety,
    )

try:
    from examples.umi_relative_ee.control_logger import ControlLogger, make_control_logger
except ModuleNotFoundError:
    # Supports direct execution from inside examples/umi_relative_ee.
    from control_logger import ControlLogger, make_control_logger  # type: ignore[no-redef]

# Trajectory-projection helpers reused verbatim from visualize_predictions.py so the
# overlay is pixel-identical to its camera-mode projection. Optional: if that module
# cannot be imported, _PROJECTION_AVAILABLE flips False and the client runs HUD-only.
_PROJECTION_AVAILABLE = True
try:
    from examples.umi_relative_ee.visualize_predictions import (
        DEFAULT_EXTRINSICS,
        aa_pose_to_matrix,
        project_future,
        draw_traj_on_image,
        load_tip_kin,
    )
except ModuleNotFoundError:
    try:
        from visualize_predictions import (  # type: ignore[no-redef]
            DEFAULT_EXTRINSICS,
            aa_pose_to_matrix,
            project_future,
            draw_traj_on_image,
            load_tip_kin,
        )
    except Exception:  # optional projection dependency unavailable -> HUD-only

        def _projection_unavailable(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("visualize_predictions import failed; trajectory overlay disabled")

        _PROJECTION_AVAILABLE = False
        aa_pose_to_matrix = project_future = draw_traj_on_image = load_tip_kin = _projection_unavailable  # type: ignore[assignment]
        DEFAULT_EXTRINSICS = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def auto_detect_realsense_intrinsics(cameras: dict) -> np.ndarray | None:
    """Best-effort intrinsics K from a live RealSense pipeline."""
    try:
        import pyrealsense2 as rs

        for cam in cameras.values():
            pipeline = getattr(cam, "rs_pipeline", None)
            if pipeline is None:
                continue
            profile = pipeline.get_active_profile()
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array(
                [
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            logger.info("Auto-detected RealSense intrinsics: fx=%.1f fy=%.1f", intrinsics.fx, intrinsics.fy)
            return K
    except Exception:
        pass
    return None

_ASYNC_KEYMAP_HELP = """
╔══════════════════════════════════════════════════════════════╗
║              ASYNC UMI PIPER CONTROLS                       ║
╠══════════════════════════════════════════════════════════════╣
║  s          engage async policy control                     ║
║  SPACE      pause and discard queued actions                ║
║  q          move to START pose, then remain paused          ║
║  r          move to SAFE pose, then remain paused           ║
║  .          request and execute one fresh chunk             ║
║  h          reprint this keymap                             ║
║  ESC        graceful shutdown and safe pose                 ║
╚══════════════════════════════════════════════════════════════╝
""".strip()


def _loop_telemetry(loop_dt: deque[float]) -> str:
    """HUD/log summary of recent control-loop timing.

    Returns mean tick dt, the effective loop rate (samples / elapsed), and the
    worst spike over the rolling window. A stable loop targeting ``--fps`` shows
    ``loop~33ms (30Hz)`` with a spike close to the mean; a starved loop shows a
    higher mean and a spike well above ``1/fps``.
    """
    if not loop_dt:
        return "loop~--"
    total = sum(loop_dt)
    mean_ms = total / len(loop_dt) * 1000.0
    hz = len(loop_dt) / total if total > 0 else 0.0
    spike_ms = max(loop_dt) * 1000.0
    return f"loop~{mean_ms:.0f}ms ({hz:.0f}Hz) spike~{spike_ms:.0f}ms"


def _draw_hud_text(
    frame: np.ndarray, txt: str, org: tuple[int, int], *, font_scale: float = 0.5
) -> None:
    """Draw a HUD line over a filled black backdrop, then a single yellow pass.

    Replaces the old thick-black-outline + thin-yellow double-draw with a clean
    filled rectangle behind the text: same legibility over any background,
    without the chunky black halo.
    """
    (width, height), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x, y = org
    pad = 4
    cv2.rectangle(
        frame,
        (x - pad, y - height - pad),
        (x + width + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1, cv2.LINE_AA
    )


def _to_np(t) -> np.ndarray | None:
    """None-safe torch tensor / array → 1D float numpy, for control-log fields."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    return np.asarray(t, dtype=float).ravel()


@dataclass
class ActionBuffer:
    """Thread-safe timeline of overlapping absolute EE action chunks."""

    aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    # Optional temporal-ensemble audit callback, fired per overlapping timestep:
    #   on_merge(timestep, existing_abs, incoming_abs, aggregated_abs, ref_ee, chunk_id)
    # Run under self._lock from the receiver thread → must not block or do I/O.
    on_merge: Callable | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._actions: dict[int, TimedAction] = {}
        self.latest_action = -1
        self.action_chunk_size = -1
        self.minimum_chunk_timestamp = float("-inf")
        self.queue_size_history: list[int] = []
        # Monotonic client-side chunk id (unique across the whole run, survives clear()).
        self._chunk_counter = 0

    def clear(self, *, reject_chunks_before: float | None = None) -> None:
        with self._lock:
            self._actions.clear()
            self.action_chunk_size = -1
            if reject_chunks_before is not None:
                self.minimum_chunk_timestamp = reject_chunks_before

    def merge(self, incoming: list[TimedAction]) -> bool:
        if not incoming:
            return False
        incoming = sorted(incoming, key=lambda item: item.get_timestep())
        with self._lock:
            # A response requested before a pause/direct move must never be
            # executed after the robot has changed pose.
            if incoming[0].get_timestamp() < self.minimum_chunk_timestamp:
                return False

            current = {step: item.get_action() for step, item in self._actions.items()}
            future: dict[int, TimedAction] = {}
            self._chunk_counter += 1
            chunk_id = self._chunk_counter  # client-side id for this incoming chunk
            for item in incoming:
                timestep = item.get_timestep()
                if timestep <= self.latest_action:
                    continue
                inc = item.get_action().detach().cpu()  # incoming ABSOLUTE target (pre-blend)
                if timestep in current:
                    existing = current[timestep]  # already-queued absolute (prior chunk)
                    action = self.aggregate_fn(existing, inc)  # blended ABSOLUTE
                    if self.on_merge is not None:
                        try:  # audit: confirm ensemble blends absolute targets, not ΔT's
                            self.on_merge(timestep, existing, inc, action,
                                          getattr(item, "reference_ee", None), chunk_id)
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    action = inc
                ta = TimedAction(timestamp=item.get_timestamp(), timestep=timestep, action=action)
                # Carry the incoming chunk's provenance for pop-time logging. ``action``
                # is the post-blend absolute target; ``last_incoming_abs`` is the latest
                # chunk's pre-blend absolute for this timestep.
                ta.reference_ee = getattr(item, "reference_ee", None)
                ta.relative_action = getattr(item, "relative_action", None)
                ta.chunk_id = chunk_id
                ta.last_incoming_abs = inc
                future[timestep] = ta

            self._actions = future
            self.action_chunk_size = max(self.action_chunk_size, len(incoming))
            return bool(future)

    def pop_next(self) -> TimedAction | None:
        with self._lock:
            self.queue_size_history.append(len(self._actions))
            if not self._actions:
                return None
            timestep = min(self._actions)
            item = self._actions.pop(timestep)
            self.latest_action = timestep
            return item

    def record_size(self) -> None:
        with self._lock:
            self.queue_size_history.append(len(self._actions))

    def size(self) -> int:
        with self._lock:
            return len(self._actions)

    def snapshot(self) -> list[TimedAction]:
        """Return a timestep-sorted copy of the currently queued actions.

        Non-consuming and thread-safe. Used by the camera-only client to project
        the full remaining plan onto the camera image without advancing the queue.
        """
        with self._lock:
            return [self._actions[ts] for ts in sorted(self._actions)]

    def ready_for_observation(self, threshold: float) -> bool:
        with self._lock:
            if self.action_chunk_size <= 0:
                return not self._actions
            return len(self._actions) / self.action_chunk_size <= threshold


class UmiAsyncPolicyClient:
    """Small gRPC client independent of LeRobot's registered Robot classes."""

    def __init__(self, args: argparse.Namespace, action_buffer: ActionBuffer):
        self.args = args
        self.action_buffer = action_buffer
        self.channel = grpc.insecure_channel(
            args.server_address,
            grpc_channel_options(initial_backoff=f"{1 / args.fps:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.shutdown_event = threading.Event()
        self.request_pending = threading.Event()
        self._request_lock = threading.Lock()
        self._request_started_at = 0.0
        # Identity of the currently-pending observation request (its timestep), so a
        # STALE response (from a request invalidated by pause/resume) cannot clear a
        # newer request's pending flag in _receive_actions. None = nothing pending.
        self._pending_timestep: int | None = None
        self._receiver_thread: threading.Thread | None = None
        # Per-chunk timing for the HUD (updated by the receiver thread):
        # wire = send -> response arrives (transport + server compute);
        # server = compute duration reported by the UMI server via TimedAction.
        self.last_wire_ms: float | None = None
        self.last_server_ms: float | None = None
        # Number of accepted chunk merges; clients use it to detect the first pop after a
        # merge for an honest send->execute-first-action (e2e) latency.
        self.merge_count: int = 0
        # Single-worker pool that uploads observations OFF the control thread, so a slow
        # network transfer can't delay motor ticks. Only one upload is in flight at a time
        # (request_pending stays set until the response clears it), so max_workers=1 both
        # serializes uploads and bounds memory.
        self._send_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="umi-obs-upload")

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        try:
            grpc.channel_ready_future(self.channel).result(timeout=self.args.connect_timeout)
            self.stub.Ready(services_pb2.Empty(), timeout=self.args.connect_timeout)
            policy_setup = RemotePolicyConfig(
                policy_type=self.args.policy_type,
                pretrained_name_or_path=self.args.pretrained_path,
                lerobot_features={},
                actions_per_chunk=self.args.actions_per_chunk,
                device=self.args.policy_device,
            )
            self.stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=pickle.dumps(policy_setup)),
                timeout=self.args.policy_load_timeout,
            )
        except (grpc.RpcError, grpc.FutureTimeoutError) as exc:
            logger.error("Could not initialize policy server at %s: %s", self.args.server_address, exc)
            self.channel.close()
            return False

        self.shutdown_event.clear()
        self._receiver_thread = threading.Thread(
            target=self._receive_actions,
            name="umi-action-receiver",
            daemon=True,
        )
        self._receiver_thread.start()
        logger.info(
            "Connected to %s; server loaded %s from %s",
            self.args.server_address,
            self.args.policy_type,
            self.args.pretrained_path,
        )
        return True

    def stop(self) -> None:
        self.shutdown_event.set()
        self._send_executor.shutdown(wait=False, cancel_futures=True)
        self.channel.close()
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=3)

    def invalidate_pending_request(self) -> None:
        """Drop interest in the in-flight request (pause/resume/q/r/s/dot).

        Clears the pending flag AND the pending-request identity, so a stale response
        that arrives later cannot clear a *newer* request's flag (pause/resume race).
        The in-flight RPC itself is not cancellable (gRPC unary), but its response is
        ignored via the identity check in :meth:`_receive_actions`, and the action
        buffer rejects the stale chunk via ``minimum_chunk_timestamp``.
        """
        with self._request_lock:
            self.request_pending.clear()
            self._pending_timestep = None

    def _pending_request_expired(self) -> bool:
        with self._request_lock:
            age = time.monotonic() - self._request_started_at
        if self.request_pending.is_set() and age > self.args.request_timeout:
            logger.warning("Inference request timed out after %.1fs; retrying", age)
            with self._request_lock:
                self.request_pending.clear()
                self._pending_timestep = None
        return not self.request_pending.is_set()

    def can_send(self, *, force: bool = False) -> bool:
        return force or self._pending_request_expired()

    def send_observation(
        self,
        state_pair: np.ndarray,
        images: dict[str, np.ndarray],
        *,
        task: str | None,
        timestep: int,
        must_go: bool,
        force: bool = False,
    ) -> bool:
        if not self.can_send(force=force):
            return False

        raw_observation: dict[str, object] = {
            OBS_STATE: state_pair.astype(np.float32, copy=False),
            **{f"observation.images.{name}": image for name, image in images.items()},
        }
        if task:
            raw_observation["task"] = task
        observation = TimedObservation(
            timestamp=time.time(),
            timestep=timestep,
            observation=raw_observation,
            must_go=must_go,
        )

        self.request_pending.set()
        with self._request_lock:
            self._request_started_at = time.monotonic()
            self._pending_timestep = timestep
        # Upload OFF the control thread: serialize + stream the observation on a worker so
        # a slow network transfer never delays a motor tick. request_pending stays set
        # until the receiver clears it on the response, so only one upload is in flight.
        payload = pickle.dumps(observation)
        self._send_executor.submit(self._upload_observation, payload, timestep, must_go)
        return True

    def _upload_observation(self, payload: bytes, timestep: int, must_go: bool) -> None:
        """Worker: serialize-and-upload an observation without blocking the control loop."""
        try:
            request_iterator = send_bytes_in_chunks(
                payload,
                services_pb2.Observation,
                log_prefix="[UMI CLIENT] Observation",
                silent=True,
            )
            self.stub.SendObservations(request_iterator)
            logger.debug("Sent observation #%s (must_go=%s)", timestep, must_go)
        except grpc.RpcError as exc:
            with self._request_lock:
                # Clear only if this is still our pending request (a newer one may exist).
                if self._pending_timestep is None or self._pending_timestep == timestep:
                    self.request_pending.clear()
                    self._pending_timestep = None
            logger.error("Failed to send observation #%s: %s", timestep, exc)

    def _receive_actions(self) -> None:
        while self.running:
            try:
                response = self.stub.GetActions(services_pb2.Empty())
                payload = getattr(response, "data", b"")
                if not payload:
                    # Empty = the server returned no actions: the obs was rejected
                    # (duplicate timestep), never enqueued (upload slower than
                    # obs_queue_timeout), or inference threw. The pending request will
                    # not produce actions — clear it so the control thread can resend
                    # now, instead of stalling until request_timeout (~10 s).
                    with self._request_lock:
                        self.request_pending.clear()
                        self._pending_timestep = None
                    continue
                actions = pickle.loads(payload)  # nosec
                if not isinstance(actions, list):
                    raise TypeError(f"Expected list[TimedAction], got {type(actions)}")
                accepted = self.action_buffer.merge(actions)
                if accepted:
                    self.merge_count += 1
                # Race fix: only clear the pending flag if this response matches the
                # currently-pending request. A stale response (from a request dropped
                # by pause/resume) must NOT clear a newer request's flag, or two
                # observations could end up in flight.
                resp_ts = actions[0].get_timestep() if actions else None
                with self._request_lock:
                    if self._pending_timestep is None or resp_ts == self._pending_timestep:
                        self.request_pending.clear()
                        self._pending_timestep = None
                if actions:
                    # Wire time: the server copies our send timestamp (time.time() at
                    # send) into every action, so time-of-receipt - first-action
                    # timestamp = transport (both ways) + server compute.
                    wire_ms = (time.time() - actions[0].get_timestamp()) * 1000.0
                    self.last_wire_ms = wire_ms
                    self.last_server_ms = getattr(actions[0], "server_elapsed_ms", None)
                    logger.info(
                        "Received chunk %s:%s (%s actions, wire=%.0fms, server=%s, accepted=%s, queue=%s)",
                        actions[0].get_timestep(),
                        actions[-1].get_timestep(),
                        len(actions),
                        wire_ms,
                        f"{self.last_server_ms:.0f}ms" if self.last_server_ms is not None else "n/a",
                        accepted,
                        self.action_buffer.size(),
                    )
            except grpc.RpcError as exc:
                if self.running:
                    logger.error("Action receiver RPC failed: %s", exc)
                    time.sleep(0.1)
            except Exception:
                logger.exception("Invalid action chunk received from policy server")
                with self._request_lock:
                    self.request_pending.clear()
                    self._pending_timestep = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async UMI relative-EE Piper robot client")
    parser.add_argument("--server_address", default="127.0.0.1:8080")
    parser.add_argument("--pretrained_path", required=True, help="Checkpoint path visible to the server")
    parser.add_argument("--policy_type", default="act")
    parser.add_argument("--policy_device", default="cuda")
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--actions_per_chunk", "--n_action_steps", dest="actions_per_chunk", type=int, default=20
    )
    parser.add_argument("--chunk_size_threshold", type=float, default=0.5)
    parser.add_argument(
        "--aggregate_fn_name",
        choices=["weighted_average", "latest_only", "average", "conservative",
                 "weighted_average_so3", "conservative_so3", "average_so3"],
        default="latest_only",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--connect_timeout", type=float, default=10.0)
    parser.add_argument("--policy_load_timeout", type=float, default=300.0)
    parser.add_argument("--request_timeout", type=float, default=10.0)

    parser.add_argument("--can_port", default="can0")
    parser.add_argument("--gripper_port", default="/dev/ttyACM0")
    parser.add_argument("--gripper_type", choices=["external", "builtin"], default="external")
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--urdf_path", default=DEFAULT_URDF_PATH)
    parser.add_argument("--deploy_frame", default=DEFAULT_DEPLOY_FRAME)
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument("--ee_bounds_min", type=float, nargs=3, default=[-0.5, -0.5, -0.1])
    parser.add_argument("--ee_bounds_max", type=float, nargs=3, default=[0.5, 0.5, 0.6])
    parser.add_argument("--max_ee_step_m", type=float, default=0.05)
    parser.add_argument("--gripper_kp", type=float, default=5.0)
    parser.add_argument("--gripper_kd", type=float, default=0.5)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument(
        "--dryrun", action="store_true",
        help="No arm/gripper/IK: connect only the camera + server, feed a FIXED identity EE state, "
             "and visualize the predicted trajectory (like async_client_with_only_camera.py). "
             "Skips all real robot control; does not affect the normal deploy path.",
    )
    parser.add_argument(
        "--initial_state", type=float, nargs=7, default=None,
        help="--dryrun only: fixed 7D EE state [x,y,z,wx,wy,wz,gripper] fed every tick "
             "(default: identity + 0.5 gripper).",
    )
    parser.add_argument("--extrinsics_config",
                        default=str(DEFAULT_EXTRINSICS) if DEFAULT_EXTRINSICS else None,
                        help="camera_gripper_extrinsics JSON for the projected trajectory overlay "
                             "(default: sibling camera_gripper_extrinsics_sroi_v2_d405.json)")
    parser.add_argument("--camera_info_path", default=None,
                        help="camera_info_color.json intrinsics K (default: auto-detect from RealSense)")
    parser.add_argument("--output_dir", default="outputs/debug/async_piper_deploy",
                        help="Headless: save the annotated camera feed to <output_dir>/camera_live.mp4")
    parser.add_argument("--debug_visualize_queue_size", action="store_true")
    parser.add_argument(
        "--log", nargs="?", default=None, const="",
        help="Log every control tick for offline analysis. Off by default. "
             "Bare '--log' writes to logs/async_<timestamp>.csv (+ .npz/.json); "
             "'--log PATH' uses PATH as the stem (the .csv/.npz/.json suffixes are "
             "added automatically).",
    )
    args = parser.parse_args()
    if not 0 <= args.chunk_size_threshold <= 1:
        parser.error("--chunk_size_threshold must be in [0, 1]")
    if args.actions_per_chunk <= 0:
        parser.error("--actions_per_chunk must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def make_ik_pipeline(args: argparse.Namespace, kinematics: RobotKinematics) -> RobotProcessorPipeline:
    return RobotProcessorPipeline(
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


def execute_ee_action(
    action: torch.Tensor,
    *,
    piper,
    gripper,
    current_joints: np.ndarray,
    ik_pipeline: RobotProcessorPipeline,
    args: argparse.Namespace,
) -> dict:
    """Run IK + write joints/gripper.

    Returns a result dict for logging:
      ok            bool  — did we actually command the arm this tick?
      skip_reason   str|None — "invalid_action" | "ik_failed" | None
      joints_rad    np.ndarray(6,) | None — the IK joint command written
      ee_action     np.ndarray(7,) | None — the popped absolute-EE target
      gripper       float | None — normalized gripper command (0..1)
    """
    action_aa = action.numpy()
    gripper_value = float(np.clip(action_aa[6], 0.0, 1.0)) if action_aa.shape == (7,) else None
    if action_aa.shape != (7,) or not np.isfinite(action_aa).all():
        logger.warning(
            "Rejected invalid EE action: shape=%s finite=%s", action_aa.shape, np.isfinite(action_aa).all()
        )
        return {"ok": False, "skip_reason": "invalid_action", "joints_rad": None,
                "ee_action": action_aa.copy() if action_aa.shape == (7,) else None,
                "gripper": gripper_value}

    action_dict = {
        "ee.x": float(action_aa[0]),
        "ee.y": float(action_aa[1]),
        "ee.z": float(action_aa[2]),
        "ee.wx": float(action_aa[3]),
        "ee.wy": float(action_aa[4]),
        "ee.wz": float(action_aa[5]),
        "ee.gripper_pos": float(action_aa[6]),
    }
    observation_dict = {f"{name}.pos": float(current_joints[index]) for index, name in enumerate(ARM_JOINTS)}
    try:
        result = ik_pipeline((action_dict, observation_dict))
    except Exception as exc:
        logger.warning("IK rejected action: %s", exc)
        return {"ok": False, "skip_reason": "ik_failed", "joints_rad": None,
                "ee_action": action_aa.copy(), "gripper": gripper_value}

    joint_values = np.array([result.get(f"{name}.pos", 0.0) for name in ARM_JOINTS])
    piper.write_joints(joint_values)
    if gripper is not None:
        kp, kd, position = gripper_norm_to_external(
            gripper_value,
            args.gripper_kp,
            args.gripper_kd,
        )
        gripper.send_command(kp=kp, kd=kd, position=position)
    else:
        piper.write_gripper(gripper_norm_to_builtin(gripper_value))
    return {"ok": True, "skip_reason": None, "joints_rad": joint_values,
            "ee_action": action_aa.copy(), "gripper": gripper_value}


def _run_dryrun(args: argparse.Namespace, action_buffer: ActionBuffer, policy_client: "UmiAsyncPolicyClient") -> None:
    """Camera + server only: feed a FIXED identity EE state, never touch the robot.

    Mirrors async_client_with_only_camera.py: same gRPC send/receive, same trajectory
    projection + HUD/timing, but no arm/gripper/IK import, connect, or write. The state
    pair sent every tick is the constant identity pose (or --initial_state).
    """
    from lerobot.cameras.utils import make_cameras_from_configs

    cameras_config = parse_cameras_config(args.cameras)
    if not cameras_config:
        raise ValueError("No cameras parsed from --cameras")
    cameras = make_cameras_from_configs(cameras_config)
    for name, camera in cameras.items():
        camera.connect()
        logger.info("Camera connected: %s", name)

    K = auto_detect_realsense_intrinsics(cameras)
    if args.camera_info_path:
        import json

        try:
            K = np.array(json.loads(Path(args.camera_info_path).read_text())["K"], dtype=float).reshape(3, 3)
            logger.info("Loaded intrinsics K from %s (fx=%.1f)", args.camera_info_path, K[0, 0])
        except Exception as exc:
            logger.warning("Could not load --camera_info_path %s: %s", args.camera_info_path, exc)
    tip_kin = None
    if _PROJECTION_AVAILABLE and args.extrinsics_config and not args.no_vis:
        try:
            tip_kin = load_tip_kin(args.extrinsics_config)
            logger.info("Loaded hand-eye extrinsics for trajectory overlay: %s", args.extrinsics_config)
        except Exception as exc:
            logger.warning("Could not load extrinsics %s: %s — trajectory overlay disabled",
                           args.extrinsics_config, exc)
    if tip_kin is not None and K is not None:
        logger.info("trajectory projection on: fx=%.1f cx=%.1f cy=%.1f", K[0, 0], K[0, 2], K[1, 2])

    # The core of dry-run: a FIXED state. prev == current == identity, every tick.
    current_ee = np.array(
        args.initial_state if args.initial_state else [0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32
    )
    previous_ee = current_ee.copy()
    logger.info("DRY-RUN: fixed EE state=%s (no robot control)", np.round(current_ee, 3).tolist())

    save_mp4 = (not args.no_vis) and args.output_dir and not os.environ.get("DISPLAY")
    live_display = (not args.no_vis) and bool(os.environ.get("DISPLAY"))
    if args.no_vis:
        logger.info("Headless --no_vis: camera + inference only (no window, no file)")
    elif save_mp4:
        logger.info("Headless: saving frames to %s/camera_live.mp4", args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    step = 0
    last_sent_timestep: int | None = None
    first_chunk_seen = False
    last_action: np.ndarray | None = None
    last_e2e_ms: float | None = None
    last_seen_merges = 0
    send_at = 0.0
    loop_dt: deque[float] = deque(maxlen=60)
    loop_t0 = time.perf_counter()
    logger.info("Starting dry-run loop (Ctrl-C to stop)")

    try:
        while args.num_steps == 0 or step < args.num_steps:
            t0 = time.perf_counter()
            images = {name: camera.read() for name, camera in cameras.items()}

            queue_empty = action_buffer.size() == 0
            ready = action_buffer.ready_for_observation(args.chunk_size_threshold)
            should_send = (step == 0) or (
                policy_client.can_send() and ready
                and (queue_empty or last_sent_timestep != action_buffer.latest_action + 1)
            )
            if should_send:
                observation_timestep = max(action_buffer.latest_action + 1, 0)
                send_at = time.perf_counter()
                sent = policy_client.send_observation(
                    np.stack([previous_ee, current_ee]),  # fixed identity pair
                    images,
                    task=args.task,
                    timestep=observation_timestep,
                    must_go=queue_empty or step == 0,
                    force=(step == 0),
                )
                if sent:
                    last_sent_timestep = observation_timestep

            # Pop the next action but DO NOT execute it (no robot). Track timing + last action.
            timed_action = action_buffer.pop_next()
            if timed_action is not None:
                action = timed_action.get_action().detach().cpu().numpy()
                if action.shape != (7,) or not np.isfinite(action).all():
                    logger.warning("Invalid action from server: shape=%s finite=%s",
                                   action.shape, np.isfinite(action).all())
                else:
                    if policy_client.merge_count != last_seen_merges:
                        last_seen_merges = policy_client.merge_count
                        last_e2e_ms = (time.perf_counter() - send_at) * 1000.0 if send_at else None
                    last_action = action
                    if not first_chunk_seen:
                        first_chunk_seen = True
                        logger.info(
                            "✓ SERVER RETURNED A VALID CHUNK — first action=%s  (e2e %.0f ms)",
                            np.round(action, 3), last_e2e_ms or -1.0,
                        )

            if live_display or save_mp4:
                for name, image in images.items():
                    img_rgb = image
                    if tip_kin is not None and K is not None:
                        queued = action_buffer.snapshot()
                        if queued:
                            abs_actions = [qa.get_action().detach().cpu().numpy() for qa in queued]
                            poses = np.stack(
                                [aa_pose_to_matrix(current_ee)]
                                + [aa_pose_to_matrix(a) for a in abs_actions]
                            )
                            px, py = project_future(poses, 0, K, tip_kin)
                            img_rgb = draw_traj_on_image(img_rgb, np.column_stack([px, py]), "pred")
                    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    elapsed_s = time.perf_counter() - loop_t0
                    hud = [
                        f"DRY-RUN  {args.policy_type.upper()}  step {step}  "
                        f"queue={action_buffer.size()}  {elapsed_s:.1f}s",
                        _loop_telemetry(loop_dt),
                        f"state={np.round(current_ee, 3).tolist()}",
                    ]
                    if last_action is not None:
                        hud.append(f"last_action={np.round(last_action, 3).tolist()}")
                    if last_e2e_ms is not None:
                        hud.append(f"e2e~{last_e2e_ms:.0f}ms")
                    if policy_client.last_wire_ms is not None and policy_client.last_server_ms is not None:
                        net_ms = policy_client.last_wire_ms - policy_client.last_server_ms
                        hud.append(
                            f"wire~{policy_client.last_wire_ms:.0f}ms  "
                            f"server~{policy_client.last_server_ms:.0f}ms  "
                            f"net~{max(net_ms, 0.0):.0f}ms"
                        )
                    for y, txt in enumerate(hud, start=1):
                        _draw_hud_text(frame, txt, (10, y * 25))
                    if save_mp4:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    else:
                        cv2.imshow(f"Async Piper DRY-RUN: {name}", frame)
                cv2.waitKey(1)

            if step % 30 == 0:
                logger.info(
                    "step %d: queue=%d first_chunk=%s e2e=%s last_action=%s",
                    step, action_buffer.size(), first_chunk_seen,
                    None if last_e2e_ms is None else f"{last_e2e_ms:.0f}ms",
                    None if last_action is None else np.round(last_action, 3).tolist(),
                )
            step += 1
            work = time.perf_counter() - t0
            time.sleep(max(0.0, 1.0 / args.fps - work))
            # Full tick period (work + sleep) — honest Hz, not work-only time.
            loop_dt.append(time.perf_counter() - t0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if live_display:
            cv2.destroyAllWindows()
        for camera in cameras.values():
            try:
                camera.disconnect()
            except Exception:
                logger.exception("Camera disconnect failed")
        if save_mp4 and frames:
            try:
                import imageio.v3 as iio

                out_path = Path(args.output_dir) / "camera_live.mp4"
                iio.imwrite(out_path, np.stack(frames), fps=args.fps, macro_block_size=1, quality=8)
                logger.info("Saved %s (%d frames)", out_path, len(frames))
            except Exception:
                logger.exception("Failed to write mp4")
        policy_client.stop()
        logger.info("Dry-run done after %d ticks; first_chunk_seen=%s", step, first_chunk_seen)


def run(args: argparse.Namespace) -> None:
    action_buffer = ActionBuffer(get_aggregate_function(args.aggregate_fn_name))
    policy_client = UmiAsyncPolicyClient(args, action_buffer)
    if not policy_client.start():
        return

    # Dry-run: camera + server only, fixed identity state, no robot control. Branching here
    # (before any hardware import/connect) keeps the real deploy path below completely intact.
    if args.dryrun:
        _run_dryrun(args, action_buffer, policy_client)
        return

    piper = None
    gripper = None
    cameras = {}
    piper_connected = False
    gripper_connected = False
    step_count = 0
    ctrl_log: ControlLogger | None = None
    frames: list[np.ndarray] = []
    first_chunk_seen = False
    last_action: np.ndarray | None = None
    last_e2e_ms: float | None = None  # send -> first action of the response chunk executes
    last_seen_merges = 0
    send_at = 0.0
    loop_dt: deque[float] = deque(maxlen=60)
    loop_t0 = time.perf_counter()
    save_mp4 = (not args.no_vis) and args.output_dir and not os.environ.get("DISPLAY")
    live_display = (not args.no_vis) and bool(os.environ.get("DISPLAY"))
    if args.no_vis:
        logger.info("Headless --no_vis: camera + control only (no window, no file)")
    elif save_mp4:
        logger.info("Headless: saving frames to %s/camera_live.mp4", args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        if DEFAULT_PIPER_SRC_PATH not in sys.path:
            sys.path.insert(0, DEFAULT_PIPER_SRC_PATH)
        from modules.piper_interface import PiperInterface

        kinematics = RobotKinematics(
            urdf_path=args.urdf_path,
            target_frame_name=args.deploy_frame,
            joint_names=ARM_JOINTS,
        )
        ik_pipeline = make_ik_pipeline(args, kinematics)

        piper = PiperInterface(can_port=args.can_port)
        piper.connect()
        piper_connected = True
        logger.info("Piper arm connected on %s", args.can_port)

        if args.gripper_type == "external":
            from modules.gripper import Gripper

            gripper = Gripper(port=args.gripper_port)
            gripper.connect()
            gripper_connected = True
            gripper.send_command(kp=args.gripper_kp, kd=args.gripper_kd, position=0.0)
            logger.info("External gripper connected on %s", args.gripper_port)
        else:
            logger.info("Using built-in Piper gripper")

        move_to_safe(piper, gripper, args.gripper_kp, args.gripper_kd, duration=2.0)
        move_to_pose(
            piper,
            gripper,
            START_POSE_DEG,
            label="start",
            gripper_kp=args.gripper_kp,
            gripper_kd=args.gripper_kd,
            duration=2.0,
            open_gripper=True,
        )

        camera_configs = parse_cameras_config(args.cameras)
        if not camera_configs:
            raise ValueError("At least one policy camera is required")
        from lerobot.cameras.utils import make_cameras_from_configs

        cameras = make_cameras_from_configs(camera_configs)
        for name, camera in cameras.items():
            camera.connect()
            logger.info("Camera connected: %s", name)

        # Intrinsics K (auto-detect or --camera_info_path) + hand-eye extrinsics for the
        # projected trajectory overlay (matches visualize_predictions.py camera mode).
        K = auto_detect_realsense_intrinsics(cameras)
        if args.camera_info_path:
            import json

            try:
                K = np.array(
                    json.loads(Path(args.camera_info_path).read_text())["K"], dtype=float
                ).reshape(3, 3)
                logger.info("Loaded intrinsics K from %s (fx=%.1f)", args.camera_info_path, K[0, 0])
            except Exception as exc:
                logger.warning("Could not load --camera_info_path %s: %s", args.camera_info_path, exc)
        tip_kin = None
        if _PROJECTION_AVAILABLE and args.extrinsics_config and not args.no_vis:
            try:
                tip_kin = load_tip_kin(args.extrinsics_config)
                logger.info("Loaded hand-eye extrinsics for trajectory overlay: %s", args.extrinsics_config)
            except Exception as exc:
                logger.warning("Could not load extrinsics %s: %s — trajectory overlay disabled",
                               args.extrinsics_config, exc)
        if tip_kin is not None and K is not None:
            logger.info("trajectory projection on: fx=%.1f cx=%.1f cy=%.1f", K[0, 0], K[0, 2], K[1, 2])
        elif K is not None and not args.no_vis:
            logger.info("No hand-eye extrinsics — HUD only, no trajectory overlay")

        if args.warm_start:
            move_to_pose(
                piper,
                gripper,
                HOME_POSE_DEG,
                label="home",
                gripper_kp=args.gripper_kp,
                gripper_kd=args.gripper_kd,
                duration=3.0,
                open_gripper=True,
            )
            for camera in cameras.values():
                camera.read()

        print(_ASYNC_KEYMAP_HELP)
        logger.info("Loop state: PAUSED (press s to engage)")
        previous_ee: np.ndarray | None = None
        state = LoopState.PAUSED
        force_request = False
        single_chunk = False
        single_actions_executed = 0
        last_sent_timestep: int | None = None

        ctrl_log = make_control_logger(args, prefix="async")
        if ctrl_log is not None:
            ctrl_log.start()

            # Audit the temporal ensemble: on every overlapping-timestep blend, record
            # the existing absolute target (prior chunk), the incoming absolute + its
            # anchor, and the aggregated result, so offline analysis can confirm we blend
            # ABSOLUTE targets (same base frame), not relative ΔT's. Runs in the receiver
            # thread under ActionBuffer's lock → only appends to a list, never blocks.
            def _on_merge(timestep, existing, incoming, aggregated, ref_ee, chunk_id):
                ex, ic, ag, rf = _to_np(existing), _to_np(incoming), _to_np(aggregated), _to_np(ref_ee)
                # Infer the convex blend weight with ag ≈ w*ex + (1-w)*ic (mean over moving dims).
                w = float("nan")
                if ex is not None and ic is not None and ag is not None:
                    denom = ex - ic
                    m = np.abs(denom) > 1e-9
                    if m.any():
                        w = float(np.mean((ag[m] - ic[m]) / denom[m]))
                ctrl_log.log_merge(timestep=int(timestep), chunk_id=int(chunk_id), weight=w,
                                   existing_abs=ex, incoming_abs=ic, aggregated_abs=ag, ref_ee=rf)

            action_buffer.on_merge = _on_merge
        tick = 0

        with KeyboardCommandHandler() as keyboard:
            while args.num_steps == 0 or step_count < args.num_steps:
                loop_started = time.perf_counter()
                tick += 1
                diag = {
                    "popped": False, "ik_ok": False, "skip_reason": "no_action",
                    "action_timestep": None, "action_ee": None, "ik_joints_rad": None,
                    "ee_delta_m": None, "joint_delta_max_rad": None, "gripper": None,
                    "chunk_id": None, "chunk_ref_ee": None,
                    "action_abs": None, "action_agg": None, "action_rel": None,
                }
                # True only on the tick a fresh chunk's first action is popped (when
                # last_e2e_ms is (re)measured). e2e_ms is logged only then, so the JSON
                # e2e stats are response-weighted, not row-weighted (stale value repeated).
                e2e_this_tick = False

                key = keyboard.poll()
                while key is not None:
                    if key == "esc":
                        state = LoopState.SHUTDOWN
                        break
                    if key in {"space", "q", "r"}:
                        state = LoopState.PAUSED
                        single_chunk = False
                        action_buffer.clear(reject_chunks_before=time.time())
                        policy_client.invalidate_pending_request()
                    if key == "q":
                        move_to_pose(
                            piper,
                            gripper,
                            START_POSE_DEG,
                            label="start",
                            gripper_kp=args.gripper_kp,
                            gripper_kd=args.gripper_kd,
                            duration=2.0,
                            open_gripper=True,
                        )
                        resync_ik_safety(ik_pipeline, kinematics, piper)
                        previous_ee = None
                        keyboard.drain()
                    elif key == "r":
                        move_to_pose(
                            piper,
                            gripper,
                            SAFE_POSE_DEG,
                            label="safe",
                            gripper_kp=args.gripper_kp,
                            gripper_kd=args.gripper_kd,
                            duration=3.0,
                            open_gripper=True,
                        )
                        resync_ik_safety(ik_pipeline, kinematics, piper)
                        previous_ee = None
                        keyboard.drain()
                    elif key == "s" and state == LoopState.PAUSED:
                        action_buffer.clear(reject_chunks_before=time.time())
                        policy_client.invalidate_pending_request()
                        state = LoopState.INFERENCE
                        force_request = True
                        single_chunk = False
                        logger.info("Async control engaged")
                    elif key == "space":
                        logger.info("Control paused; queued actions discarded")
                    elif key == "dot" and state == LoopState.PAUSED:
                        action_buffer.clear(reject_chunks_before=time.time())
                        policy_client.invalidate_pending_request()
                        state = LoopState.INFERENCE
                        force_request = True
                        single_chunk = True
                        single_actions_executed = 0
                        logger.info("Requesting one fresh action chunk")
                    elif key == "h":
                        print(_ASYNC_KEYMAP_HELP)
                    key = keyboard.poll()

                if state == LoopState.SHUTDOWN:
                    break

                current_joints = np.asarray(piper.read_joints())
                if gripper is not None:
                    gripper_norm = gripper_external_to_norm(gripper.position)
                else:
                    gripper_norm = gripper_builtin_to_norm(piper.read_gripper())
                gripper_norm = float(np.clip(gripper_norm, 0.0, 1.0))
                current_ee = ee_pose_aa_from_fk(
                    kinematics,
                    current_joints,
                    gripper_norm,
                )
                if previous_ee is None:
                    previous_ee = current_ee.copy()

                images = {name: camera.read() for name, camera in cameras.items()}

                if state == LoopState.INFERENCE:
                    queue_empty = action_buffer.size() == 0
                    ready = action_buffer.ready_for_observation(args.chunk_size_threshold)
                    should_send = force_request or (
                        not single_chunk
                        and ready
                        and policy_client.can_send()
                        and (queue_empty or last_sent_timestep != action_buffer.latest_action + 1)
                    )
                    if should_send:
                        observation_timestep = max(action_buffer.latest_action + 1, 0)
                        send_at = time.perf_counter()
                        sent = policy_client.send_observation(
                            np.stack([previous_ee, current_ee]),
                            images,
                            task=args.task,
                            timestep=observation_timestep,
                            must_go=queue_empty or force_request,
                            force=force_request,
                        )
                        if sent:
                            last_sent_timestep = observation_timestep
                            force_request = False

                    timed_action = action_buffer.pop_next()
                    if timed_action is not None:
                        # e2e = send -> the first action of the response chunk executes. Updated
                        # only on the FIRST pop after a chunk merge (merge_count transition), so
                        # queue dwell never pollutes it.
                        if policy_client.merge_count != last_seen_merges:
                            last_seen_merges = policy_client.merge_count
                            last_e2e_ms = (time.perf_counter() - send_at) * 1000.0 if send_at else None
                            e2e_this_tick = True
                        last_action = timed_action.get_action().detach().cpu().numpy()
                        if not first_chunk_seen:
                            first_chunk_seen = True
                            logger.info(
                                "✓ SERVER RETURNED A VALID CHUNK — first action=%s  (e2e %.0f ms)",
                                np.round(last_action, 3), last_e2e_ms or -1.0,
                            )
                        res = execute_ee_action(
                            timed_action.get_action(),
                            piper=piper,
                            gripper=gripper,
                            current_joints=current_joints,
                            ik_pipeline=ik_pipeline,
                            args=args,
                        )
                        diag.update(
                            popped=True,
                            action_timestep=timed_action.get_timestep(),
                            action_ee=res["ee_action"],
                            ik_joints_rad=res["joints_rad"],
                            ik_ok=res["ok"],
                            skip_reason=res["skip_reason"],
                            gripper=res["gripper"],
                            # UMI chunk provenance: anchor + raw relative + latest chunk's
                            # pre-blend absolute, vs the post-ensemble absolute executed.
                            chunk_id=getattr(timed_action, "chunk_id", None),
                            chunk_ref_ee=_to_np(getattr(timed_action, "reference_ee", None)),
                            action_abs=_to_np(getattr(timed_action, "last_incoming_abs", None)),
                            action_agg=res["ee_action"],
                            action_rel=_to_np(getattr(timed_action, "relative_action", None)),
                        )
                        if res["ee_action"] is not None:
                            diag["ee_delta_m"] = float(
                                np.linalg.norm(res["ee_action"][:3] - current_ee[:3])
                            )
                        if res["joints_rad"] is not None:
                            diag["joint_delta_max_rad"] = float(
                                np.max(np.abs(res["joints_rad"] - current_joints))
                            )
                        step_count += 1
                        if single_chunk:
                            single_actions_executed += 1
                            if action_buffer.size() == 0 and single_actions_executed > 0:
                                state = LoopState.PAUSED
                                single_chunk = False
                                logger.info("Fresh chunk executed; control paused")
                else:
                    action_buffer.record_size()

                previous_ee = current_ee.copy()

                if live_display or save_mp4:
                    for name, image in images.items():
                        img_rgb = image  # RGB uint8
                        # Project the queued absolute EE plan onto the camera image exactly as
                        # visualize_predictions.py projects its predicted chunk: origin = the
                        # current FK EE pose, then each queued target, composed as
                        # (T_opt_cam @ inv(T_origin) @ T_target @ T_cam_ee) and pinholed with K.
                        if tip_kin is not None and K is not None:
                            queued = action_buffer.snapshot()
                            if queued:
                                abs_actions = [qa.get_action().detach().cpu().numpy() for qa in queued]
                                poses = np.stack(
                                    [aa_pose_to_matrix(current_ee)]
                                    + [aa_pose_to_matrix(a) for a in abs_actions]
                                )
                                px, py = project_future(poses, 0, K, tip_kin)
                                img_rgb = draw_traj_on_image(img_rgb, np.column_stack([px, py]), "pred")
                        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        elapsed_s = time.perf_counter() - loop_t0
                        hud = [
                            f"{state.name}  {args.policy_type.upper()}  step {step_count}  "
                            f"queue={action_buffer.size()}  {elapsed_s:.1f}s",
                            _loop_telemetry(loop_dt),
                        ]
                        if last_action is not None:
                            hud.append(f"state={np.round(current_ee, 3).tolist()}")
                            hud.append(f"last_action={np.round(last_action, 3).tolist()}")
                        if last_e2e_ms is not None:
                            hud.append(f"e2e~{last_e2e_ms:.0f}ms")
                        if policy_client.last_wire_ms is not None and policy_client.last_server_ms is not None:
                            net_ms = policy_client.last_wire_ms - policy_client.last_server_ms
                            hud.append(
                                f"wire~{policy_client.last_wire_ms:.0f}ms  "
                                f"server~{policy_client.last_server_ms:.0f}ms  "
                                f"net~{max(net_ms, 0.0):.0f}ms"
                            )
                        for y, txt in enumerate(hud, start=1):
                            _draw_hud_text(frame, txt, (10, y * 25))
                        if save_mp4:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        else:
                            cv2.imshow(f"Async Piper: {name}", frame)
                    cv2.waitKey(1)

                if step_count % 30 == 0:
                    logger.info(
                        "step %d: queue=%d first_chunk=%s e2e=%s last_action=%s %s",
                        step_count, action_buffer.size(), first_chunk_seen,
                        None if last_e2e_ms is None else f"{last_e2e_ms:.0f}ms",
                        None if last_action is None else np.round(last_action, 3).tolist(),
                        _loop_telemetry(loop_dt),
                    )

                work = time.perf_counter() - loop_started
                time.sleep(max(0.0, 1 / args.fps - work))
                # Full tick period (work + pacing sleep) — NOT work-only time, so the
                # reported Hz is honest (the loop is physically always <= --fps).
                tick_dt = time.perf_counter() - loop_started
                loop_dt.append(tick_dt)

                if ctrl_log is not None:
                    if not diag["popped"]:
                        diag["skip_reason"] = (
                            "paused" if state == LoopState.PAUSED else "underrun"
                        )
                    ctrl_log.log(
                        step=step_count,
                        tick=tick,
                        tick_dt_ms=tick_dt * 1000.0,
                        work_ms=work * 1000.0,
                        state=state.name,
                        queue=action_buffer.size(),
                        popped=diag["popped"],
                        ik_ok=diag["ik_ok"],
                        skip_reason=diag["skip_reason"],
                        action_timestep=diag["action_timestep"],
                        chunk_id=diag["chunk_id"],
                        chunk_ref_ee=diag["chunk_ref_ee"],
                        action_abs=diag["action_abs"],
                        action_agg=diag["action_agg"],
                        action_rel=diag["action_rel"],
                        ee_delta_m=diag["ee_delta_m"],
                        joint_delta_max_rad=diag["joint_delta_max_rad"],
                        gripper=diag["gripper"],
                        action_ee=diag["action_ee"],
                        current_ee=current_ee,
                        ik_joints_rad=diag["ik_joints_rad"],
                        current_joints_rad=current_joints,
                        e2e_ms=(last_e2e_ms if e2e_this_tick else None),
                        wire_ms=policy_client.last_wire_ms,
                        server_ms=policy_client.last_server_ms,
                    )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Async Piper client failed")
    finally:
        cv2.destroyAllWindows()
        for camera in cameras.values():
            try:
                camera.disconnect()
            except Exception:
                logger.exception("Camera disconnect failed")
        if save_mp4 and frames:
            try:
                import imageio.v3 as iio

                out_path = Path(args.output_dir) / "camera_live.mp4"
                iio.imwrite(out_path, np.stack(frames), fps=args.fps, macro_block_size=1, quality=8)
                logger.info("Saved %s (%d frames)", out_path, len(frames))
            except Exception:
                logger.exception("Failed to write mp4")
        if piper_connected:
            try:
                move_to_safe(piper, gripper, args.gripper_kp, args.gripper_kd)
            except Exception as exc:
                logger.warning("Could not return to safe pose: %s", exc)
                input("Robot is NOT at safe pose. Press Enter to disable motors anyway...")
        if gripper_connected:
            gripper.disconnect()
        if piper_connected:
            piper.disconnect()
        policy_client.stop()
        if ctrl_log is not None:
            try:
                ctrl_log.close()
            except Exception:
                logger.exception("Failed to write control log")
        if args.debug_visualize_queue_size and action_buffer.queue_size_history:
            visualize_action_queue_size(action_buffer.queue_size_history)
        logger.info("Disconnected after %s executed action steps", step_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(parse_args())

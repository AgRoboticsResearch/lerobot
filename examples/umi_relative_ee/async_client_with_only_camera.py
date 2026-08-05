#!/usr/bin/env python

r"""Camera-only client to test the async UMI relative-EE policy server — no robot.

This talks the *real* gRPC handshake + observation/action protocol to
``async_umi_relative_ee_policy_server.py`` (so it is a genuine end-to-end test
of the server, not a mock), but replaces the Piper arm with a static or
auto-chained identity EE pose. Use it to confirm the server loads a checkpoint
and returns valid ``[N, 7]`` absolute EE chunks from live camera input.

It reuses ``UmiAsyncPolicyClient`` / ``ActionBuffer`` from the deploy client
unchanged — only the robot-side control loop (FK/IK/motors) is removed.

Two terminals, same machine:

    # Terminal 1: start the (empty, generic) server
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python examples/umi_relative_ee/async_umi_relative_ee_policy_server.py \
        --host=127.0.0.1 --port=8080 --fps=30

    # Terminal 2: this camera-only test client
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python examples/umi_relative_ee/async_client_with_only_camera.py \
        --server_address=127.0.0.1:8080 \
        --pretrained_path=outputs/train/act_umi_identity_rot6d_1302/checkpoints/0500000/pretrained_model \
        --policy_type=act \
        --cameras="{camera: {type: intelrealsense, fps: 30, width: 640, height: 480}}" \
        --num_steps=60

For SmolVLA / pi05 add ``--task="pick the strawberry"``.

State semantics (no robot): the EE pose defaults to the identity
``[0, 0, 0, 0, 0, 0, 0.5]``; held still, the UMI 20D relative state collapses to
~identity and the image alone drives the chunk — exactly like the local
``visualize_predictions.py`` camera mode. Pass ``--update_state`` to auto-chain
the last received absolute action as the next pose, synthesizing a motion cue.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.cameras.utils import make_cameras_from_configs

try:
    from examples.umi_relative_ee.async_umi_relative_ee_piper_client import (
        ActionBuffer,
        UmiAsyncPolicyClient,
    )
    from examples.umi_relative_ee.deploy_umi_relative_ee_piper import (
        parse_cameras_config,
    )
except ModuleNotFoundError:
    # Allow direct execution from inside examples/umi_relative_ee.
    from async_umi_relative_ee_piper_client import (  # type: ignore[no-redef]
        ActionBuffer,
        UmiAsyncPolicyClient,
    )
    from deploy_umi_relative_ee_piper import parse_cameras_config  # type: ignore[no-redef]

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── server / protocol ──
    p.add_argument("--server_address", default="127.0.0.1:8080")
    p.add_argument("--pretrained_path", required=True, help="Checkpoint path visible to the SERVER")
    p.add_argument("--policy_type", default="act", help="act | smolvla | pi05")
    p.add_argument("--policy_device", default="cuda")
    p.add_argument("--task", default=None, help="Language task (required for smolvla / pi05)")
    p.add_argument("--actions_per_chunk", "--n_action_steps", dest="actions_per_chunk", type=int, default=20)
    p.add_argument("--chunk_size_threshold", type=float, default=0.5)
    p.add_argument("--aggregate_fn_name", default="latest_only")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--connect_timeout", type=float, default=10.0)
    p.add_argument("--policy_load_timeout", type=float, default=300.0)
    p.add_argument("--request_timeout", type=float, default=10.0)

    # ── camera / state ──
    p.add_argument("--cameras", required=True)
    p.add_argument("--extrinsics_config",
                   default=str(DEFAULT_EXTRINSICS) if DEFAULT_EXTRINSICS else None,
                   help="camera_gripper_extrinsics JSON for the projected trajectory overlay "
                        "(default: sibling camera_gripper_extrinsics_sroi_v2_d405.json)")
    p.add_argument("--camera_info_path", default=None,
                   help="camera_info_color.json intrinsics K (default: auto-detect from RealSense)")
    p.add_argument("--initial_state", type=float, nargs=7, default=None,
                   help="Initial 7D EE state [x,y,z,wx,wy,wz,gripper] (default: identity + 0.5 gripper)")
    p.add_argument("--update_state", action="store_true",
                   help="Chain the last received absolute action as the next pose (synthesize motion)")

    # ── run bounds / output ──
    p.add_argument("--num_steps", type=int, default=0, help="Loop ticks; 0 = run forever (Ctrl-C)")
    p.add_argument("--no_vis", action="store_true", help="Camera + inference only; no window, no file")
    p.add_argument("--output_dir", default="outputs/debug/async_server_test")
    args = p.parse_args()
    if not 0 <= args.chunk_size_threshold <= 1:
        p.error("--chunk_size_threshold must be in [0, 1]")
    if args.actions_per_chunk <= 0:
        p.error("--actions_per_chunk must be positive")
    if args.fps <= 0:
        p.error("--fps must be positive")
    if args.policy_type in ("smolvla", "pi05") and not args.task:
        p.error(f"--task is required for {args.policy_type}")
    return args


def run(args: argparse.Namespace) -> None:
    action_buffer = ActionBuffer(get_aggregate_function(args.aggregate_fn_name))
    policy_client = UmiAsyncPolicyClient(args, action_buffer)
    if not policy_client.start():
        return  # handshake / model-load failure already logged

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
            K = np.array(
                json.loads(Path(args.camera_info_path).read_text())["K"], dtype=float
            ).reshape(3, 3)
            logger.info("Loaded intrinsics K from %s (fx=%.1f)", args.camera_info_path, K[0, 0])
        except Exception as exc:
            logger.warning("Could not load --camera_info_path %s: %s", args.camera_info_path, exc)
    if K is None:
        logger.warning("No RealSense intrinsics detected (non-RealSense camera?); HUD only, no projection")

    # Hand-eye extrinsics for the projected trajectory overlay (matches visualize_predictions.py).
    tip_kin = None
    if _PROJECTION_AVAILABLE and args.extrinsics_config:
        try:
            tip_kin = load_tip_kin(args.extrinsics_config)
            logger.info("Loaded hand-eye extrinsics for trajectory overlay: %s", args.extrinsics_config)
        except Exception as exc:
            logger.warning("Could not load extrinsics %s: %s — trajectory overlay disabled",
                           args.extrinsics_config, exc)
    if tip_kin is not None and K is not None:
        logger.info("trajectory projection on: fx=%.1f cx=%.1f cy=%.1f", K[0, 0], K[0, 2], K[1, 2])
    elif K is not None:
        logger.info("No hand-eye extrinsics — HUD only, no trajectory overlay")

    current_ee = np.array(
        args.initial_state if args.initial_state else [0, 0, 0, 0, 0, 0, 0.5],
        dtype=np.float32,
    )
    previous_ee: np.ndarray | None = None

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
    last_e2e_ms: float | None = None  # send -> first action of the response chunk executes
    last_seen_merges = 0
    send_at = 0.0
    loop_t0 = time.perf_counter()
    logger.info("Starting camera-only test loop (Ctrl-C to stop)")

    try:
        while args.num_steps == 0 or step < args.num_steps:
            t0 = time.perf_counter()

            images = {name: camera.read() for name, camera in cameras.items()}
            if previous_ee is None:
                previous_ee = current_ee.copy()

            # ── send observation when the queue is ready and no request is pending ──
            queue_empty = action_buffer.size() == 0
            ready = action_buffer.ready_for_observation(args.chunk_size_threshold)
            should_send = (step == 0) or (
                policy_client.can_send() and ready and
                (queue_empty or last_sent_timestep != action_buffer.latest_action + 1)
            )
            if should_send:
                observation_timestep = max(action_buffer.latest_action + 1, 0)
                send_at = time.perf_counter()
                sent = policy_client.send_observation(
                    np.stack([previous_ee, current_ee]),
                    images,
                    task=args.task,
                    timestep=observation_timestep,
                    must_go=queue_empty or step == 0,
                    force=(step == 0),
                )
                if sent:
                    last_sent_timestep = observation_timestep

            # ── pop the next queued absolute EE action ──
            timed_action = action_buffer.pop_next()
            if timed_action is not None:
                action = timed_action.get_action().detach().cpu().numpy()
                if action.shape != (7,) or not np.isfinite(action).all():
                    logger.warning("Invalid action from server: shape=%s finite=%s",
                                   action.shape, np.isfinite(action).all())
                else:
                    # e2e = send -> the first action of the response chunk executes. Updated
                    # only on the FIRST pop after a chunk merge (merge_count transition), so
                    # queue dwell never pollutes it. A timestep==last_sent check would race:
                    # on the send tick the boundary action may still be from the old chunk.
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
                    if args.update_state:
                        current_ee = action.astype(np.float32)  # chain: pretend the robot moved here

            previous_ee = current_ee.copy()

            # ── HUD / capture (window, mp4, or nothing) ──
            if live_display or save_mp4:
                name = next(iter(images))
                img_rgb = images[name]  # RGB uint8
                # Project the queued absolute EE plan onto the camera image exactly as
                # visualize_predictions.py projects its predicted chunk: origin = the
                # current EE pose, then each queued target, composed as
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
                    f"{args.policy_type.upper()}  step {step}  queue={action_buffer.size()}  {elapsed_s:.1f}s",
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
                    ypix = y * 25
                    cv2.putText(frame, txt, (10, ypix), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, txt, (10, ypix), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if save_mp4:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    cv2.imshow("async server test (no robot)", frame)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break

            if step % 30 == 0:
                logger.info(
                    "step %d: queue=%d first_chunk=%s e2e=%s last_action=%s",
                    step, action_buffer.size(), first_chunk_seen,
                    None if last_e2e_ms is None else f"{last_e2e_ms:.0f}ms",
                    None if last_action is None else np.round(last_action, 3).tolist(),
                )
            step += 1

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, 1.0 / args.fps - elapsed))
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
        policy_client.stop()
        if save_mp4 and frames:
            try:
                import imageio.v3 as iio

                out_path = Path(args.output_dir) / "camera_live.mp4"
                iio.imwrite(out_path, np.stack(frames), fps=args.fps, macro_block_size=1, quality=8)
                logger.info("Saved %s (%d frames)", out_path, len(frames))
            except Exception:
                logger.exception("Failed to write mp4")
        logger.info("Done after %d ticks; first_chunk_seen=%s", step, first_chunk_seen)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(parse_args())

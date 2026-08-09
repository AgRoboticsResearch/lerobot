#!/usr/bin/env python
"""Offline open-loop replay of the async UMI relative-EE deploy loop.

Drives the REAL temporal-ensemble deploy loop (gRPC policy server + ``ActionBuffer``
+ ``aggregate_fn`` + the per-tick ``--log``) from a validation **dataset** instead of
the real robot/camera — so the ensemble / chunk-stitching / re-plan cadence can be
studied deterministically, with identical input across configs.

Open-loop by default: every tick the policy is fed the dataset's GROUND-TRUTH EE
pose + image (``current_ee`` = GT[t], ``previous_ee`` = GT[t-1]); it does NOT see
where its own commands drove a robot. With ``--dead_reckon``, the image remains
dataset replay but the mock robot state is initialized from GT[0] and then updated
from each executed aggregated absolute target; it is not resynchronized to GT.
The commanded absolute EE target is logged alongside the dataset GT, so the same
analysis figures apply (EE pose, execution strategy, ensemble audit) plus a
command-vs-GT error. No arm, no camera, no IK — the ensemble lives entirely on the
EE action stream. Reuses ``ActionBuffer`` / ``UmiAsyncPolicyClient`` /
``ControlLogger`` unchanged, so the code under test is the real deploy path.

Usage (server must be running, e.g. async_umi_relative_ee_policy_server.py):
  uv run python examples/umi_relative_ee/mock_async_client.py \\
      --server_address 127.0.0.1:8080 --pretrained_path outputs/.../pretrained_model \\
      --dataset <repo_or_path> --episode_indices 0 \\
      --actions_per_chunk 20 --chunk_size_threshold 0.5 --aggregate_fn_name weighted_average \\
      --fps 30 --log
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from examples.umi_relative_ee.async_umi_relative_ee_piper_client import (
        ActionBuffer,
        UmiAsyncPolicyClient,
        _to_np,
    )
    from examples.umi_relative_ee.control_logger import make_control_logger
except ModuleNotFoundError:
    from async_umi_relative_ee_piper_client import (  # type: ignore[no-redef]
        ActionBuffer,
        UmiAsyncPolicyClient,
        _to_np,
    )
    from control_logger import make_control_logger  # type: ignore[no-redef]

logger = logging.getLogger(__name__)
# The UMI relative-EE datasets store the absolute EE trajectory in `action` (the
# processor's UmiDeriveStateFromActionStep builds observation.state from action[:2]),
# so for open-loop replay current_ee = action[t]. Some datasets use observation.ee.
CAMERA_KEY = "observation.images.camera"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline open-loop async deploy replay from a dataset")
    p.add_argument("--server_address", default="127.0.0.1:8080")
    p.add_argument("--pretrained_path", required=True, help="Checkpoint path visible to the server")
    p.add_argument("--policy_type", default="act")
    p.add_argument("--policy_device", default="cuda")
    p.add_argument("--task", default=None)
    p.add_argument("--dataset", required=True, help="LeRobotDataset repo_id or local path")
    p.add_argument("--episode_indices", type=str, default="0",
                   help="Comma list of episodes to replay (default '0')")
    p.add_argument("--root", default=None, help="Local root if --dataset is a repo_id subset")
    p.add_argument("--actions_per_chunk", type=int, default=20)
    p.add_argument("--chunk_size_threshold", type=float, default=0.5)
    p.add_argument("--aggregate_fn_name", default="weighted_average",
                   choices=["weighted_average", "latest_only", "average", "conservative",
                            "weighted_average_so3", "conservative_so3", "average_so3"])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--connect_timeout", type=float, default=10.0)
    p.add_argument("--policy_load_timeout", type=float, default=300.0)
    p.add_argument("--request_timeout", type=float, default=10.0)
    p.add_argument("--max_frames", type=int, default=0, help="0 = replay the whole episode(s)")
    p.add_argument("--ee_key", default="action",
                   help="Dataset column holding the 7D absolute EE pose per frame "
                        "(UMI relative-EE stores it in 'action'; some datasets use 'observation.ee').")
    p.add_argument("--dead_reckon", action="store_true",
                   help="Do not resync mock current_ee to dataset GT; after each tick, "
                        "make the executed aggregate the next mock robot pose.")
    p.add_argument("--log", nargs="?", default=None, const="",
                   help="Off by default. Bare --log → logs/mock_<timestamp>; --log PATH → stem.")
    return p.parse_args()


def _state_to_ee7(state) -> np.ndarray:
    """Coerce a dataset frame's state to the 7D absolute EE pose [x,y,z,rx,ry,rz,gripper]."""
    s = np.asarray(getattr(state, "cpu", lambda: state)() if hasattr(state, "cpu") else state, dtype=np.float32)
    s = np.ravel(s)
    if s.shape == (7,):
        return s
    if s.shape == (14,):
        return s[-7:]                       # [prev, curr] → curr
    if s.shape == (2, 7):
        return s[-1]
    raise ValueError(f"Unexpected observation.state shape {s.shape}; expected 7, 14, or (2,7)")


def _image_to_hwc_uint8(img) -> np.ndarray:
    """Dataset frame image → HWC uint8 RGB (what the server's _prepare_umi_observation expects)."""
    a = np.asarray(getattr(img, "cpu", lambda: img)() if hasattr(img, "cpu") else img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):  # CHW → HWC
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a


def replay_episode(args, episode: int, action_buffer: ActionBuffer,
                   policy_client: UmiAsyncPolicyClient, ctrl_log) -> int:
    """Replay one episode in open loop; return #ticks logged."""
    ds = LeRobotDataset(args.dataset, episodes=[episode], root=args.root) if args.root \
        else LeRobotDataset(args.dataset, episodes=[episode])
    n = len(ds)
    if n < 2:
        logger.warning("Episode %d has %d frames — need ≥2; skipping", episode, n)
        return 0
    logger.info("Episode %d: %d frames at %d Hz (%.1f s)", episode, n, args.fps, n / args.fps)

    robot_ee = _state_to_ee7(ds[0][args.ee_key])
    previous_ee = robot_ee.copy()
    tick = step_count = 0
    last_sent_timestep: int | None = None
    send_at = 0.0
    last_e2e_ms = None
    last_seen_merges = 0

    # ensemble audit callback (same as the real client): record each blend
    if ctrl_log is not None:
        def _on_merge(timestep, existing, incoming, aggregated, ref_ee, chunk_id):
            ex, ic, ag, rf = _to_np(existing), _to_np(incoming), _to_np(aggregated), _to_np(ref_ee)
            w = float("nan")
            if ex is not None and ic is not None and ag is not None:
                den = ex - ic
                m = np.abs(den) > 1e-9
                if m.any():
                    w = float(np.mean((ag[m] - ic[m]) / den[m]))
            ctrl_log.log_merge(timestep=int(timestep), chunk_id=int(chunk_id), weight=w,
                               existing_abs=ex, incoming_abs=ic, aggregated_abs=ag, ref_ee=rf)
        action_buffer.on_merge = _on_merge

    period = 1.0 / args.fps
    for t in range(1, n):
        if args.max_frames and step_count >= args.max_frames:
            break
        t0 = time.perf_counter()
        tick += 1
        diag = {"popped": False, "ik_ok": False, "skip_reason": "no_action",
                "action_timestep": None, "action_ee": None, "ik_joints_rad": None,
                "ee_delta_m": None, "joint_delta_max_rad": None, "gripper": None,
                "chunk_id": None, "chunk_ref_ee": None,
                "action_abs": None, "action_agg": None, "action_rel": None,
                "current_joints_rad": None}
        e2e_this_tick = False  # gate e2e_ms to the response tick (response-weighted)

        dataset_ee = _state_to_ee7(ds[t][args.ee_key])
        current_ee = robot_ee.copy() if args.dead_reckon else dataset_ee
        image = _image_to_hwc_uint8(ds[t][CAMERA_KEY])

        # send an observation whenever the ensemble is ready for a fresh chunk
        queue_empty = action_buffer.size() == 0
        ready = action_buffer.ready_for_observation(args.chunk_size_threshold)
        should_send = (t == 1) or (
            policy_client.can_send() and ready
            and (queue_empty or last_sent_timestep != action_buffer.latest_action + 1)
        )
        if should_send:
            obs_ts = max(action_buffer.latest_action + 1, 0)
            send_at = time.perf_counter()
            sent = policy_client.send_observation(
                np.stack([previous_ee, current_ee]), {CAMERA_KEY.split(".")[-1]: image},
                task=args.task, timestep=obs_ts, must_go=queue_empty or t == 1, force=(t == 1),
            )
            if sent:
                last_sent_timestep = obs_ts

        timed_action = action_buffer.pop_next()
        executed_ee = None
        if timed_action is not None:
            if policy_client.merge_count != last_seen_merges:
                last_seen_merges = policy_client.merge_count
                last_e2e_ms = (time.perf_counter() - send_at) * 1000.0 if send_at else None
                e2e_this_tick = True
            action = timed_action.get_action().detach().cpu().numpy()
            diag.update(popped=True, ik_ok=True, skip_reason=None,
                        action_timestep=timed_action.get_timestep(), action_ee=action,
                        action_agg=action,
                        chunk_id=getattr(timed_action, "chunk_id", None),
                        chunk_ref_ee=_to_np(getattr(timed_action, "reference_ee", None)),
                        action_abs=_to_np(getattr(timed_action, "last_incoming_abs", None)),
                        action_rel=_to_np(getattr(timed_action, "relative_action", None)),
                        gripper=float(action[6]) if action.shape == (7,) else None,
                        ee_delta_m=float(np.linalg.norm(action[:3] - current_ee[:3]))
                        if action.shape == (7,) else None)
            step_count += 1
            executed_ee = action.copy()

        if args.dead_reckon:
            # Match the real loop's observation semantics: the current pose was
            # read before this tick's command, and the executed target becomes
            # the next tick's mock robot pose. With no action, hold position.
            previous_ee = current_ee.copy()
            if executed_ee is not None:
                robot_ee = executed_ee
        else:
            previous_ee = current_ee

        if ctrl_log is not None:
            if not diag["popped"]:
                diag["skip_reason"] = "underrun"
            ctrl_log.log(step=step_count, tick=tick, tick_dt_ms=0.0, work_ms=0.0,
                         state="INFERENCE", queue=action_buffer.size(),
                         current_ee=current_ee, e2e_ms=(last_e2e_ms if e2e_this_tick else None),
                         wire_ms=policy_client.last_wire_ms, server_ms=policy_client.last_server_ms,
                         **diag)
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    return step_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    episodes = [int(x) for x in args.episode_indices.split(",") if x.strip() != ""]

    action_buffer = ActionBuffer(get_aggregate_function(args.aggregate_fn_name))
    policy_client = UmiAsyncPolicyClient(args, action_buffer)
    if not policy_client.start():
        return

    ctrl_log = make_control_logger(args, prefix="mock")
    if ctrl_log is not None:
        ctrl_log.start()
    try:
        total = 0
        for ep in episodes:
            total += replay_episode(args, ep, action_buffer, policy_client, ctrl_log)
            action_buffer.clear(reject_chunks_before=time.time())
            policy_client.invalidate_pending_request()
        logger.info("Replayed %d executed steps across %d episode(s)", total, len(episodes))
    finally:
        if ctrl_log is not None:
            try:
                ctrl_log.close()
            except Exception:
                logger.exception("Failed to write control log")
        policy_client.stop()


if __name__ == "__main__":
    main()

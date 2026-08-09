#!/usr/bin/env python
"""Offline open-loop replay of the SYNC UMI relative-EE deploy loop.

Sister of ``mock_async_client.py`` but for the synchronous deploy
(``deploy_umi_relative_ee_piper.py``): the policy runs IN-PROCESS (no gRPC server),
a full action chunk is predicted whenever the queue drains and replayed one action
per tick (open-loop chunk replay), and every tick is logged with the same
``--log`` schema — so ``make_episode_debug_figs.py`` works on the output unchanged.

Open-loop: each tick feeds the policy the dataset's GROUND-TRUTH EE pose + image
(``current_ee = action[t]``); the policy does not see where its own commands drove a
robot. No arm, no camera, no IK (EE-only log). Reuses ``load_policy_and_processors``
and ``ControlLogger`` so the code under test is the real sync deploy path.

Usage:
  uv run python examples/umi_relative_ee/mock_sync_client.py \\
      --pretrained_path outputs/.../pretrained_model \\
      --dataset <validation-dataset> --episode_indices 0 --ee_key action \\
      --n_action_steps 30 --fps 30 --log
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE

try:
    from examples.umi_relative_ee.control_logger import make_control_logger
except ModuleNotFoundError:
    from control_logger import make_control_logger  # type: ignore[no-redef]

logger = logging.getLogger(__name__)
CAMERA_KEY = "observation.images.camera"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline open-loop SYNC deploy replay from a dataset")
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--dataset", required=True, help="LeRobotDataset repo_id or local path")
    p.add_argument("--episode_indices", type=str, default="0")
    p.add_argument("--root", default=None)
    p.add_argument("--ee_key", default="action",
                   help="Dataset column with the 7D absolute EE pose per frame "
                        "(UMI relative-EE stores it in 'action').")
    p.add_argument("--n_action_steps", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_frames", type=int, default=0, help="0 = whole episode(s)")
    p.add_argument("--dead_reckon", action="store_true",
                   help="Closed-loop exec: seed the pose from GT[0] then integrate commands "
                        "(next pose = last command) instead of resyncing to GT each tick. "
                        "Reveals real chunk-switch behavior vs the GT-feed artifact.")
    p.add_argument("--log", nargs="?", default=None, const="",
                   help="Off by default. Bare --log → logs/mock_sync_<timestamp>; --log PATH → stem.")
    return p.parse_args()


def load_policy_and_processors(model_path: str, device: torch.device):
    """Load a UMI relative-EE policy + checkpointed processors (in-process)."""
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    if not getattr(policy_config, "use_umi_relative_ee", False):
        raise ValueError(f"{model_path} is not a UMI relative-EE checkpoint")
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


def _ee7(state) -> np.ndarray:
    s = np.asarray(state.detach().cpu() if hasattr(state, "detach") else state, dtype=np.float32).ravel()
    if s.shape == (7,):
        return s
    if s.shape == (14,):
        return s[-7:]
    if s.shape == (2, 7):
        return s[-1]
    raise ValueError(f"Unexpected EE state shape {s.shape}; expected 7, 14, or (2,7)")


def add_policy_task(batch: dict, policy, task: str | None) -> None:
    if policy.name not in ("smolvla", "pi05"):
        return
    if not task:
        raise ValueError(f"{policy.name} requires --task")
    batch["task"] = [task]


def replay_episode(args, episode: int, policy, preprocessor, postprocessor, device, ctrl_log) -> int:
    ds = (LeRobotDataset(args.dataset, episodes=[episode], root=args.root) if args.root
          else LeRobotDataset(args.dataset, episodes=[episode]))
    n = len(ds)
    if n < 2:
        logger.warning("Episode %d has %d frames — need ≥2; skipping", episode, n)
        return 0
    logger.info("Episode %d: %d frames at %d Hz (%.1f s)", episode, n, args.fps, n / args.fps)

    policy.reset(); preprocessor.reset(); postprocessor.reset()
    action_queue: list[np.ndarray] = []
    chunk_id = 0
    chunk_ref_ee = None
    chunk_rel = None
    chunk_idx = 0
    tick = step_count = 0
    period = 1.0 / args.fps
    dead_ee = _ee7(ds[0][args.ee_key]) if args.dead_reckon else None  # seed from GT[0]

    for t in range(n):
        if args.max_frames and step_count >= args.max_frames:
            break
        t0 = time.perf_counter()
        tick += 1
        diag = {"popped": False, "ik_ok": False, "skip_reason": "no_action",
                "action_timestep": None, "action_ee": None, "ik_joints_rad": None,
                "current_ee": None, "current_joints_rad": None,
                "ee_delta_m": None, "joint_delta_max_rad": None, "gripper": None,
                "work_ms": None,
                "chunk_id": None, "chunk_ref_ee": None,
                "action_abs": None, "action_agg": None, "action_rel": None}
        e2e_this_tick = False
        infer_ms = None

        current_ee = _ee7(ds[t][args.ee_key])           # GT this tick (logged + drives the image)
        # Dead-reckon mode: the EXEC integrates commands instead of resyncing to GT. The
        # policy is fed the dead-reckoned pose (seed=GT[0], then ← last command every tick),
        # so at a chunk switch the new chunk is anchored at where the commands actually drove
        # the arm — no GT teleport. GT is still logged as current_ee (for the drift trace).
        policy_ee = dead_ee if args.dead_reckon else current_ee
        diag["current_ee"] = current_ee
        image = ds[t][CAMERA_KEY].unsqueeze(0).to(device)  # (1,3,H,W) float [0,1]
        state_tensor = torch.from_numpy(policy_ee).unsqueeze(0).to(device)
        batch = {OBS_STATE: state_tensor}
        add_policy_task(batch, policy, args.task)
        batch[CAMERA_KEY] = image

        with torch.no_grad():
            processed = preprocessor(batch)  # refreshes the 2-frame relative state every tick
            if len(action_queue) == 0:
                t_inf = time.perf_counter()
                pred_norm = policy.predict_action_chunk(processed)
                infer_ms = (time.perf_counter() - t_inf) * 1000.0
                pred = postprocessor(pred_norm)
                if isinstance(pred, dict) and "action" in pred:
                    pred = pred["action"]
                actions_aa = pred[0].cpu().numpy()
                action_queue = [actions_aa[i] for i in range(min(args.n_action_steps, len(actions_aa)))]
                chunk_id += 1
                chunk_ref_ee = policy_ee.copy()           # anchor the policy actually used
                chunk_rel = pred_norm[0].detach().cpu().numpy()
                chunk_idx = 0
                e2e_this_tick = True

        action_aa = action_queue.pop(0)
        if args.dead_reckon:
            dead_ee = action_aa                           # dead-reckon: next pose = this command
        diag.update(popped=True, ik_ok=True, skip_reason=None,
                    action_timestep=float(t),  # dataset frame → enables timestep-aligned debug
                    action_ee=action_aa,
                    chunk_id=chunk_id, chunk_ref_ee=chunk_ref_ee,
                    action_abs=action_aa, action_agg=action_aa,  # sync replays one chunk verbatim
                    gripper=float(action_aa[6]),
                    ee_delta_m=float(np.linalg.norm(action_aa[:3] - current_ee[:3])))
        if chunk_rel is not None and chunk_idx < len(chunk_rel):
            diag["action_rel"] = chunk_rel[chunk_idx]
        chunk_idx += 1
        step_count += 1

        diag["work_ms"] = (time.perf_counter() - t0) * 1000.0
        if ctrl_log is not None:
            ctrl_log.log(step=step_count, tick=tick,
                         tick_dt_ms=diag["work_ms"],  # full tick period proxy (sync has no pacing sleep log)
                         state="INFERENCE", queue=len(action_queue),
                         e2e_ms=(infer_ms if e2e_this_tick else None),  # sync: time to produce a chunk
                         wire_ms=None, server_ms=None, **diag)
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    return step_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Loading policy from %s", args.pretrained_path)
    policy, preprocessor, postprocessor = load_policy_and_processors(args.pretrained_path, device)

    ctrl_log = make_control_logger(args, prefix="mock_sync_dr" if args.dead_reckon else "mock_sync")
    if ctrl_log is not None:
        ctrl_log.start()
    try:
        episodes = [int(x) for x in args.episode_indices.split(",") if x.strip() != ""]
        total = 0
        for ep in episodes:
            total += replay_episode(args, ep, policy, preprocessor, postprocessor, device, ctrl_log)
            preprocessor.reset(); postprocessor.reset(); policy.reset()
        logger.info("Replayed %d executed steps across %d episode(s)", total, len(episodes))
    finally:
        if ctrl_log is not None:
            try:
                ctrl_log.close()
            except Exception:
                logger.exception("Failed to write control log")


if __name__ == "__main__":
    main()

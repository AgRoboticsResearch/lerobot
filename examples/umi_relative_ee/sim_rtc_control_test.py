#!/usr/bin/env python

"""Closed-loop RTC vs no-RTC control test for UMI relative-EE policies on piper-sim.

Runs the *deploy* control path — FK state -> UMI preprocess -> policy chunk ->
postprocess (one anchor per chunk) -> IK safety pipeline -> 30 Hz joint writes
— against the piper-sim MuJoCo gRPC simulator, comparing two execution
strategies with the same checkpoint, seed, and replayed vision:

  no_rtc: deploy-sync semantics. A chunk is replayed until exhausted, then the
          queue is cleared and a fresh unguided chunk replaces it. During
          inference nothing is queued, so the arm starves and the chunk switch
          can jump (the documented SYNC behavior).
  rtc:    RTC semantics. Leftover absolute actions stay in the queue while
          inference runs (no starvation), and the new chunk is guided toward
          the re-anchored leftover tail (reanchor_umi_rtc_prefix) exactly like
          RTCInferenceEngine. The merge skips actions consumed during
          inference (real_delay).

Vision is replayed frame-by-frame from a validation dataset episode
(static-scene approximation — this tests execution dynamics, not perception);
the two-pose EE state window is closed-loop from the simulator's FK, matching
the async-server contract. Per-tick logs (commanded/achieved EE, joints,
underruns, chunk events) are dumped to JSON for plotting.

Requires a running piper-sim server:  piper-sim serve --headless
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.rtc import ActionQueue, RTCConfig, reanchor_umi_rtc_prefix
from lerobot.processor import NormalizerProcessorStep, RobotProcessorPipeline, UmiRelativeActionsStep
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import OBS_STATE

from eval_rtc_dataset import fixed_prefix_length, load_policy_and_processors, set_seed

logger = logging.getLogger(__name__)

# Deploy-script constants (examples/umi_relative_ee/deploy_umi_relative_ee_piper.py)
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME_POSE_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.00])
START_POSE_DEG = np.array([0.0, 79.2, -31.3, 0.0, -45.85, 0.0])
GRIPPER_OPEN_MM = 0.0
GRIPPER_CLOSED_MM = 55.0
CAMERA_KEY = "observation.images.camera"


def gripper_norm_to_builtin(gripper_norm: float) -> float:
    return gripper_norm * (GRIPPER_CLOSED_MM - GRIPPER_OPEN_MM) + GRIPPER_OPEN_MM


def gripper_builtin_to_norm(pos_mm: float) -> float:
    return (pos_mm - GRIPPER_OPEN_MM) / (GRIPPER_CLOSED_MM - GRIPPER_OPEN_MM)


def ee_pose_aa_from_fk(kinematics, joints_deg: np.ndarray, gripper_norm: float) -> np.ndarray:
    ee_T = kinematics.forward_kinematics(joints_deg)
    pos = ee_T[:3, 3]
    aa = Rotation.from_matrix(ee_T[:3, :3]).as_rotvec()
    return np.concatenate([pos, aa, [gripper_norm]]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--episode_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--task", default="pick the strawberry")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n_chunks", type=int, default=8, help="chunks (replans) per run")
    parser.add_argument("--execution_horizon", type=int, default=10)
    parser.add_argument("--inference_delay", type=int, default=4)
    parser.add_argument("--max_guidance_weight", type=float, default=10.0)
    parser.add_argument("--replan_threshold", type=int, default=16,
                        help="rtc arm replans once the queue drops to this many actions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sim_addr", default="127.0.0.1:50052")
    parser.add_argument("--urdf_path", default=None,
                        help="default: the URDF packaged with piper-sim (IK == sim model)")
    parser.add_argument("--deploy_frame", default="camera_link")
    parser.add_argument("--ee_bounds_min", type=float, nargs=3, default=[-0.5, -0.5, -0.1])
    parser.add_argument("--ee_bounds_max", type=float, nargs=3, default=[0.5, 0.5, 0.6])
    parser.add_argument("--max_ee_step_m", type=float, default=0.05)
    parser.add_argument("--output", default="outputs/debug/sim_rtc_control/sim_rtc_control.json")
    return parser.parse_args()


def wait_for_pose(piper, target_deg: np.ndarray, tol_deg: float = 0.5, timeout: float = 6.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if np.max(np.abs(piper.read_joints() - target_deg)) < tol_deg:
            return True
        time.sleep(0.05)
    return False


def build_ik_pipeline(args, kinematics: RobotKinematics) -> RobotProcessorPipeline:
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


def resync_ik_safety(ik_pipeline, kinematics, piper) -> np.ndarray:
    """Re-anchor EEBoundsAndSafety._last_pos to the arm's real pose after a direct move."""
    current_joints = piper.read_joints()
    ee_pos = kinematics.forward_kinematics(current_joints)[:3, 3].copy()
    for step in ik_pipeline.steps:
        if hasattr(step, "_last_pos"):
            step._last_pos = ee_pos
    return ee_pos


def load_episode_frames(dataset: LeRobotDataset, episode: int) -> list[torch.Tensor]:
    """Sequential uint8 camera frames [C,H,W] for one episode."""
    frames: list[torch.Tensor] = []
    for i in range(len(dataset)):
        item = dataset[i]
        if int(item["episode_index"]) != episode:
            continue
        frames.append(item[CAMERA_KEY])
    if not frames:
        raise RuntimeError(f"No frames found for episode {episode}")
    return frames


def to_policy_image(frame: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = frame
    if image.dtype == torch.uint8:
        image = image.to(torch.float32) / 255.0
    return image.unsqueeze(0).to(device)  # [1,C,H,W]


class RunHarness:
    """One (arm, episode) run: closed-loop control through piper-sim with logging."""

    def __init__(self, args, policy_pack, ik_pipeline, kinematics, piper, umi_step, normalizer_step):
        self.args = args
        self.policy, self.core_policy, self.preprocessor, self.postprocessor, self.policy_config = policy_pack
        self.ik_pipeline = ik_pipeline
        self.kinematics = kinematics
        self.piper = piper
        self.umi_step = umi_step
        self.normalizer_step = normalizer_step
        self.device = torch.device(args.device)
        self.rtc_config = RTCConfig(
            enabled=False,
            execution_horizon=args.execution_horizon,
            max_guidance_weight=args.max_guidance_weight,
        )

    def run(self, arm: str, episode: int, frames: list[torch.Tensor]) -> dict[str, Any]:
        args = self.args
        period = 1.0 / args.fps
        max_ticks = args.n_chunks * (self.policy_config.chunk_size + args.execution_horizon) + 90

        # Reset everything to a common, deterministic start.
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        set_seed(args.seed)
        self.piper.write_joints(HOME_POSE_DEG)
        self.piper.write_gripper(GRIPPER_OPEN_MM)
        wait_for_pose(self.piper, HOME_POSE_DEG)
        self.piper.write_joints(START_POSE_DEG)
        wait_for_pose(self.piper, START_POSE_DEG)
        resync_ik_safety(self.ik_pipeline, self.kinematics, self.piper)

        queue = ActionQueue(RTCConfig(enabled=(arm == "rtc")))
        self.core_policy.config.rtc_config = self.rtc_config
        self.core_policy.init_rtc_processor()

        ticks: list[dict[str, Any]] = []
        chunk_events: list[dict[str, Any]] = []
        state = {
            "busy": False,
            "merges": 0,
            "last_infer_s": None,
            "pending_new_chunk": False,
            "error": None,
        }

        def inference_worker(obs_state: torch.Tensor, image: torch.Tensor, qsize_before: int) -> None:
            try:
                idx_before, _, prev_abs = queue.get_left_over_snapshot()
                t0 = time.perf_counter()
                batch = {OBS_STATE: obs_state, CAMERA_KEY: image, "task": [args.task]}
                # no_grad (not inference_mode): the RTC guidance differentiates
                # the denoiser via autograd.grad, which needs a grad-enabled
                # context (it re-enables grad internally under no_grad only).
                with torch.no_grad():
                    processed = self.preprocessor(batch)
                    kwargs: dict[str, Any] = {}
                    guided = False
                    if arm == "rtc" and prev_abs is not None and len(prev_abs) > 0:
                        cached = self.umi_step.get_cached_state()
                        if cached is not None:
                            prefix = reanchor_umi_rtc_prefix(
                                prev_actions_absolute=prev_abs,
                                current_state=cached,
                                normalizer_step=self.normalizer_step,
                                policy_device=args.device,
                            )
                            prefix = fixed_prefix_length(prefix, args.execution_horizon)
                            guided = True
                            kwargs = dict(
                                prev_chunk_left_over=prefix,
                                inference_delay=args.inference_delay,
                                execution_horizon=args.execution_horizon,
                            )
                    self.rtc_config.enabled = guided
                    model_chunk = self.policy.predict_action_chunk(processed, **kwargs)
                    self.rtc_config.enabled = False
                    absolute = self.postprocessor(model_chunk)
                if isinstance(absolute, dict):
                    absolute = absolute["action"]
                model_t = model_chunk.squeeze(0).detach().cpu()
                absolute_t = absolute.squeeze(0).detach().cpu()
                infer_s = time.perf_counter() - t0
                state["last_infer_s"] = infer_s

                if arm == "no_rtc":
                    queue.clear()
                    real_delay = 0
                else:
                    # Actions consumed while this inference ran (skipped on merge).
                    real_delay = 0 if state["merges"] == 0 else min(
                        math.ceil(infer_s * args.fps), len(absolute_t) - 1
                    )
                queue.merge(model_t, absolute_t, real_delay=real_delay, action_index_before_inference=idx_before)
                state["merges"] += 1
                state["pending_new_chunk"] = True
                chunk_events.append(
                    {
                        "chunk_id": state["merges"],
                        "arm": arm,
                        "episode": episode,
                        "infer_s": infer_s,
                        "guided": guided,
                        "real_delay": real_delay,
                        "qsize_before": qsize_before,
                        "prefix_len": int(kwargs["prev_chunk_left_over"].shape[0]) if guided else 0,
                        "first_cmd": absolute_t[0].tolist(),
                    }
                )
                logger.info(
                    "[%s ep%d] chunk %d: infer %.0fms guided=%s qsize %d->%d delay=%d",
                    arm, episode, state["merges"], infer_s * 1000, guided,
                    qsize_before, queue.qsize(), real_delay,
                )
            except Exception as exc:  # noqa: BLE001
                state["error"] = repr(exc)
                logger.exception("Inference worker failed")
            finally:
                state["busy"] = False

        curr_ee = ee_pose_aa_from_fk(
            self.kinematics, self.piper.read_joints(),
            float(np.clip(gripper_builtin_to_norm(self.piper.read_gripper()), 0.0, 1.0)),
        )
        prev_ee = curr_ee.copy()
        last_cmd: np.ndarray | None = None
        last_cmd_chunk = 0
        tick = 0
        t_start = time.perf_counter()

        while state["merges"] < args.n_chunks and tick < max_ticks:
            if state["error"] is not None:
                raise RuntimeError(f"Inference worker failed: {state['error']}")
            t0 = time.perf_counter()

            joints = self.piper.read_joints()
            grip_norm = float(np.clip(gripper_builtin_to_norm(self.piper.read_gripper()), 0.0, 1.0))
            ee_aa = ee_pose_aa_from_fk(self.kinematics, joints, grip_norm)
            prev_ee, curr_ee = curr_ee, ee_aa

            threshold = 0 if arm == "no_rtc" else args.replan_threshold
            if (
                not state["busy"]
                and queue.qsize() <= threshold
                and state["merges"] < args.n_chunks
            ):
                obs_state = torch.from_numpy(np.stack([prev_ee, curr_ee])).unsqueeze(0).to(self.device)
                image = to_policy_image(frames[min(tick, len(frames) - 1)], self.device)
                state["busy"] = True
                threading.Thread(
                    target=inference_worker,
                    args=(obs_state, image, queue.qsize()),
                    daemon=True,
                ).start()

            action = queue.get()
            boundary = False
            if action is not None:
                if state["pending_new_chunk"]:
                    boundary = True
                    state["pending_new_chunk"] = False
                action_aa = action.numpy()
                last_cmd = action_aa
                last_cmd_chunk = state["merges"]

                action_dict = {
                    "ee.x": float(action_aa[0]), "ee.y": float(action_aa[1]), "ee.z": float(action_aa[2]),
                    "ee.wx": float(action_aa[3]), "ee.wy": float(action_aa[4]), "ee.wz": float(action_aa[5]),
                    "ee.gripper_pos": float(action_aa[6]),
                }
                observation_dict = {f"{name}.pos": float(joints[i]) for i, name in enumerate(ARM_JOINTS)}
                ik_ok = True
                joint_values = None
                try:
                    result = self.ik_pipeline((action_dict, observation_dict))
                    joint_values = np.array([result.get(f"{name}.pos", 0.0) for name in ARM_JOINTS])
                    self.piper.write_joints(joint_values)
                    self.piper.write_gripper(gripper_norm_to_builtin(float(action_aa[6])))
                except Exception as exc:  # noqa: BLE001
                    ik_ok = False
                    logger.warning("IK failed at tick %d: %s", tick, exc)

                ticks.append(
                    {
                        "tick": tick, "t": time.perf_counter() - t_start,
                        "arm": arm, "episode": episode,
                        "chunk_id": state["merges"], "boundary": boundary, "underrun": False,
                        "ik_ok": ik_ok, "qsize": queue.qsize(),
                        "cmd_ee": action_aa.tolist(), "ee": ee_aa.tolist(),
                        "cmd_joints": None if joint_values is None else joint_values.tolist(),
                        "joints": joints.tolist(),
                    }
                )
            else:
                # Underrun: arm holds the last commanded target (cmd stays last_cmd).
                ticks.append(
                    {
                        "tick": tick, "t": time.perf_counter() - t_start,
                        "arm": arm, "episode": episode,
                        "chunk_id": last_cmd_chunk, "boundary": False, "underrun": True,
                        "ik_ok": False, "qsize": queue.qsize(),
                        "cmd_ee": (last_cmd if last_cmd is not None else ee_aa).tolist(),
                        "ee": ee_aa.tolist(), "cmd_joints": None, "joints": joints.tolist(),
                    }
                )

            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
            tick += 1

        return {
            "arm": arm,
            "episode": episode,
            "n_frames": len(frames),
            "ticks": ticks,
            "chunks": chunk_events,
            "fps": args.fps,
            "n_chunks_requested": args.n_chunks,
            "error": state["error"],
        }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.urdf_path is None:
        from piper_sim.model import packaged_urdf_path

        args.urdf_path = str(packaged_urdf_path())
    device = torch.device(args.device)
    dataset_root = Path(args.dataset_root).resolve()
    repo_id = args.repo_id or f"local/{dataset_root.name}"

    logger.info("Loading policy from %s", args.pretrained_path)
    policy_pack = load_policy_and_processors(args.pretrained_path, device, num_steps=None)
    preprocessor = policy_pack[2]
    umi_step = next(
        (s for s in preprocessor.steps if isinstance(s, UmiRelativeActionsStep) and s.enabled), None
    )
    normalizer_step = next(
        (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)), None
    )
    if umi_step is None:
        raise RuntimeError("Checkpoint preprocessor has no enabled UmiRelativeActionsStep")

    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.deploy_frame,
        joint_names=ARM_JOINTS,
    )
    ik_pipeline = build_ik_pipeline(args, kinematics)

    from piper_sim.client import PiperInterface

    piper = PiperInterface(address=args.sim_addr)
    piper.connect()
    logger.info("Connected to piper-sim at %s (IK urdf: %s)", args.sim_addr, args.urdf_path)

    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=args.episode_indices, return_uint8=True)
    harness = RunHarness(args, policy_pack, ik_pipeline, kinematics, piper, umi_step, normalizer_step)

    runs = []
    for episode in args.episode_indices:
        frames = load_episode_frames(dataset, episode)
        logger.info("Episode %d: %d replay frames", episode, len(frames))
        for arm in ("no_rtc", "rtc"):
            logger.info("=== arm=%s episode=%d ===", arm, episode)
            runs.append(harness.run(arm, episode, frames))

    piper.disconnect()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"args": vars(args), "runs": runs}, f)
    logger.info("Wrote %s (%d runs)", output, len(runs))


if __name__ == "__main__":
    main()

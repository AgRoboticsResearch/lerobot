#!/usr/bin/env python

"""Guidance-effect test for RTC support in the UMI async policy server.

Drives the REAL gRPC server (examples/umi_relative_ee/async_umi_relative_ee_policy_server.py)
over sequential validation-dataset transitions, mirroring eval_rtc_dataset.py:

  1. Request an unguided chunk A from frame i (the "previous chunk").
  2. Simulate executing `stride` actions; the un-executed tail of A is the leftover.
  3. Request chunk B from frame i+stride WITH rtc_prev_actions_absolute = leftover.
  4. Request chunk C from the SAME frame WITHOUT the RTC key.
  5. Compare how well B (guided) vs C (unguided) track the leftover tail.

Correctness gates:
  - every B action carries rtc_guided=True; no C action does;
  - guided overlap error (vs the executing tail) must be clearly lower than
    unguided — the offline evaluator saw ~40-50% improvement.

Also exercises the unguided path against the same server session, so a pass
covers "RTC works" and "non-RTC requests still work" together.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle  # nosec
import time
from pathlib import Path

import grpc
import numpy as np
import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
CAMERA_KEY = "observation.images.camera"


class RawPolicySession:
    """Minimal sequential request/response client for the UMI policy server."""

    def __init__(self, address: str):
        self.channel = grpc.insecure_channel(address, grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

    def setup(self, policy_type: str, pretrained_path: str, device: str, actions_per_chunk: int) -> None:
        grpc.channel_ready_future(self.channel).result(timeout=10.0)
        self.stub.Ready(services_pb2.Empty(), timeout=10.0)
        specs = RemotePolicyConfig(
            policy_type=policy_type,
            pretrained_name_or_path=pretrained_path,
            lerobot_features={},
            actions_per_chunk=actions_per_chunk,
            device=device,
        )
        self.stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(specs)), timeout=600.0
        )
        logger.info("Server loaded %s from %s", policy_type, pretrained_path)

    def request(
        self,
        state_pair: np.ndarray,
        image_hwc: np.ndarray,
        task: str,
        timestep: int,
        rtc_extras: dict | None = None,
        timeout: float = 90.0,
    ) -> list[TimedAction]:
        observation: dict[str, object] = {
            OBS_STATE: state_pair.astype(np.float32),
            "observation.images.camera": image_hwc,
        }
        if task:
            observation["task"] = task
        if rtc_extras:
            observation.update(rtc_extras)
        timed = TimedObservation(
            timestamp=time.time(), timestep=timestep, observation=observation, must_go=True
        )
        payload = pickle.dumps(timed)
        self.stub.SendObservations(
            send_bytes_in_chunks(payload, services_pb2.Observation, silent=True)
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            response = self.stub.GetActions(services_pb2.Empty())
            data = getattr(response, "data", b"")
            if not data:
                time.sleep(0.02)
                continue
            actions = pickle.loads(data)  # nosec
            if actions and actions[0].get_timestep() == timestep:
                return actions
            logger.warning("Dropping stale response at timestep %s", actions[0].get_timestep() if actions else None)
        raise TimeoutError(f"No response for timestep {timestep} within {timeout:.0f}s")

    def close(self) -> None:
        self.channel.close()


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra = torch.from_numpy(a[3:6])
    rb = torch.from_numpy(b[3:6])
    from lerobot.processor.umi_relative_ee_processor import axis_angle_to_matrix

    relative = axis_angle_to_matrix(ra).T @ axis_angle_to_matrix(rb)
    cosine = ((torch.diagonal(relative).sum() - 1) / 2).clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(cosine)))


def absolute_actions(chunk: list[TimedAction]) -> np.ndarray:
    return np.stack([item.get_action().detach().cpu().numpy() for item in chunk])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server_address", default="127.0.0.1:8080")
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--policy_type", default="pi05")
    parser.add_argument("--policy_device", default="cuda")
    parser.add_argument("--actions_per_chunk", type=int, default=30)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--transitions_per_episode", type=int, default=3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--execution_horizon", type=int, default=10)
    parser.add_argument("--max_guidance_weight", type=float, default=10.0)
    parser.add_argument("--inference_delay", type=int, default=4)
    parser.add_argument("--task", default="pick the strawberry")
    parser.add_argument("--output", default="outputs/debug/async_rtc_test/guidance_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    repo_id = args.repo_id or f"local/{dataset_root.name}"
    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=args.episodes, return_uint8=True)

    episodes: dict[int, list[dict]] = {ep: [] for ep in args.episodes}
    for i in range(len(dataset)):
        item = dataset[i]
        ep = int(item["episode_index"])
        if ep in episodes:
            episodes[ep].append(
                {
                    "image": item[CAMERA_KEY].permute(1, 2, 0).numpy(),  # HWC uint8
                    "action": item["action"].numpy(),  # absolute 7D EE target at this frame
                }
            )

    session = RawPolicySession(args.server_address)
    session.setup(args.policy_type, args.pretrained_path, args.policy_device, args.actions_per_chunk)

    timestep = 0
    results = []
    try:
        for ep in args.episodes:
            frames = episodes[ep]
            starts = np.linspace(1, max(1, len(frames) - args.stride - 1), args.transitions_per_episode)
            for start in map(int, starts):
                # 1. previous chunk (unguided) from frame `start`
                f0 = frames[start]
                state_pair = np.stack([frames[start - 1]["action"], f0["action"]])
                timestep += 1000
                chunk_a = session.request(state_pair, f0["image"], args.task, timestep)
                absolute_a = absolute_actions(chunk_a)

                # 2. execute `stride` actions; the rest is the leftover tail
                leftover = absolute_a[args.stride :]

                # 3. guided chunk from frame start+stride
                f1 = frames[start + args.stride]
                state_pair = np.stack([frames[start + args.stride - 1]["action"], f1["action"]])
                rtc_extras = {
                    "rtc_prev_actions_absolute": leftover.astype(np.float32),
                    "rtc_execution_horizon": args.execution_horizon,
                    "rtc_max_guidance_weight": args.max_guidance_weight,
                    "rtc_inference_delay": args.inference_delay,
                }
                timestep += 1000
                chunk_b = session.request(state_pair, f1["image"], args.task, timestep, rtc_extras)
                absolute_b = absolute_actions(chunk_b)

                # 4. unguided chunk from the same frame
                timestep += 1000
                chunk_c = session.request(state_pair, f1["image"], args.task, timestep)
                absolute_c = absolute_actions(chunk_c)

                # 5. overlap vs the leftover tail
                n = min(args.stride, len(leftover), len(absolute_b), len(absolute_c))
                guided_xyz = np.linalg.norm(absolute_b[:n, :3] - leftover[:n, :3], axis=1) * 1000
                unguided_xyz = np.linalg.norm(absolute_c[:n, :3] - leftover[:n, :3], axis=1) * 1000
                guided_rot = [rotation_error_deg(absolute_b[k], leftover[k]) for k in range(n)]
                unguided_rot = [rotation_error_deg(absolute_c[k], leftover[k]) for k in range(n)]

                flags_ok = all(getattr(item, "rtc_guided", False) for item in chunk_b) and not any(
                    getattr(item, "rtc_guided", False) for item in chunk_c
                )
                improvement = (
                    100.0 * (unguided_xyz.mean() - guided_xyz.mean()) / unguided_xyz.mean()
                    if unguided_xyz.mean() > 0
                    else 0.0
                )
                results.append(
                    {
                        "episode": ep,
                        "frame": start,
                        "overlap_steps": n,
                        "flags_ok": flags_ok,
                        "guided_xyz_mean_mm": float(guided_xyz.mean()),
                        "unguided_xyz_mean_mm": float(unguided_xyz.mean()),
                        "guided_rot_mean_deg": float(np.mean(guided_rot)),
                        "unguided_rot_mean_deg": float(np.mean(unguided_rot)),
                        "improvement_pct": float(improvement),
                    }
                )
                logger.info(
                    "ep%d frame %d: overlap xyz guided %.2fmm vs unguided %.2fmm (%+.1f%%) flags=%s",
                    ep, start, guided_xyz.mean(), unguided_xyz.mean(), improvement, flags_ok,
                )
    finally:
        session.close()

    g = np.array([r["guided_xyz_mean_mm"] for r in results])
    u = np.array([r["unguided_xyz_mean_mm"] for r in results])
    summary = {
        "n_transitions": len(results),
        "all_flags_ok": all(r["flags_ok"] for r in results),
        "guided_xyz_mean_mm": float(g.mean()),
        "unguided_xyz_mean_mm": float(u.mean()),
        "improvement_pct": float(100 * (u.mean() - g.mean()) / u.mean()),
        "per_transition": results,
        "args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\n{n_trans} transitions: guided {g.mean():.2f}mm vs unguided {u.mean():.2f}mm "
        f"({summary['improvement_pct']:+.1f}%), flags_ok={summary['all_flags_ok']}"
        if (n_trans := len(results))
        else "\nNo transitions evaluated"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

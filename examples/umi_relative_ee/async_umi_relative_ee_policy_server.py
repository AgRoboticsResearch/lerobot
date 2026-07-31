#!/usr/bin/env python

"""Policy server for asynchronous UMI relative-EE deployment.

The stock :mod:`lerobot.async_inference.policy_server` prepares observations
from a registered LeRobot ``Robot``.  The Piper deployment is intentionally
different: its state is a 7D absolute end-effector pose computed with FK, and
the checkpointed UMI processor needs two adjacent control-loop poses.

This server accepts that two-pose state directly, runs the checkpointed
preprocessor and policy, postprocesses the *whole* action chunk with one
chunk-start reference, and returns timestamped absolute 7D EE targets.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent import futures
from typing import Any

import grpc
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import (
    Observation,
    TimedAction,
    TimedObservation,
    prepare_image,
    resize_robot_observation_image,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc
from lerobot.utils.constants import ACTION, OBS_STATE


class UmiRelativeEEPolicyServer(PolicyServer):
    """Async policy server preserving the UMI relative-EE deployment contract."""

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        response = super().SendPolicyInstructions(request, context)
        if self.policy is not None:
            self.policy.config.device = str(self.device)
            self.policy.eval()
            self.policy.reset()
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()
        return response

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Reject only duplicate timesteps.

        The generic server also drops observations whose state changes by less
        than one unit.  EE poses are measured in metres/radians, so that
        joint-space heuristic would suppress nearly every Piper update.
        """
        del previous_obs
        with self._predicted_timesteps_lock:
            already_predicted = obs.get_timestep() in self._predicted_timesteps
        if already_predicted:
            self.logger.debug("Skipping already-predicted observation #%s", obs.get_timestep())
        return not already_predicted

    def _prepare_umi_observation(self, observation_t: TimedObservation) -> Observation:
        raw = observation_t.get_observation()
        if OBS_STATE not in raw:
            raise KeyError(f"Client observation is missing {OBS_STATE!r}")

        state = torch.as_tensor(raw[OBS_STATE], dtype=torch.float32)
        if state.ndim == 1:
            state = torch.stack([state, state], dim=0)
        if state.shape != (2, 7):
            raise ValueError(
                f"{OBS_STATE} must contain [previous, current] absolute 7D EE poses; "
                f"got shape {tuple(state.shape)}"
            )

        observation: dict[str, Any] = {OBS_STATE: state.unsqueeze(0)}
        for image_key, feature in self.policy_image_features.items():
            image = raw.get(image_key)
            if image is None:
                short_key = image_key.removeprefix("observation.images.")
                image = raw.get(short_key)
            if image is None:
                raise KeyError(f"Client observation is missing policy camera {image_key!r}")

            image_tensor = torch.as_tensor(image)
            if image_tensor.ndim != 3 or image_tensor.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"Camera {image_key!r} must be an HWC image; got shape {tuple(image_tensor.shape)}"
                )
            image_tensor = resize_robot_observation_image(image_tensor, feature.shape)
            observation[image_key] = prepare_image(image_tensor).unsqueeze(0)

        if "task" in raw:
            observation["task"] = raw["task"]
        return observation

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Run UMI inference and postprocess the complete chunk in one call."""
        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("Policy instructions must be sent before inference")

        started = time.perf_counter()
        observation = self._prepare_umi_observation(observation_t)

        with torch.inference_mode():
            processed = self.preprocessor(observation)
            predicted = self.policy.predict_action_chunk(processed)
            if predicted.ndim == 2:
                predicted = predicted.unsqueeze(0)
            predicted = predicted[:, : self.actions_per_chunk, :]

            # This must remain a single call. UmiAbsoluteActionsStep uses the
            # preprocessor's cached current pose as the base for every target.
            actions = self.postprocessor(predicted)

        if isinstance(actions, dict):
            if ACTION not in actions:
                raise KeyError(f"Postprocessor returned a dict without {ACTION!r}")
            actions = actions[ACTION]
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got action shape {tuple(actions.shape)}")
            actions = actions.squeeze(0)
        if actions.ndim != 2 or actions.shape[-1] != 7:
            raise ValueError(f"Expected an [N, 7] absolute EE action chunk, got {tuple(actions.shape)}")

        self.last_processed_obs = observation_t
        actions = actions.detach().cpu()
        result = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(actions),
            observation_t.get_timestep(),
        )
        self.logger.info(
            "Observation %s -> %s absolute EE actions in %.1fms",
            observation_t.get_timestep(),
            len(result),
            (time.perf_counter() - started) * 1000,
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async UMI relative-EE policy server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--inference_latency", type=float, default=0.0)
    parser.add_argument("--obs_queue_timeout", type=float, default=2.0)
    return parser.parse_args()


def serve(args: argparse.Namespace) -> None:
    config = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    policy_server = UmiRelativeEEPolicyServer(config)
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    address = f"{config.host}:{config.port}"
    grpc_server.add_insecure_port(address)
    policy_server.logger.info("UMI relative-EE PolicyServer started on %s", address)
    grpc_server.start()
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.stop()
        grpc_server.stop(grace=1).wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    serve(parse_args())

#!/usr/bin/env python

"""Policy server for asynchronous UMI relative-EE deployment.

The stock :mod:`lerobot.async_inference.policy_server` prepares observations
from a registered LeRobot ``Robot``.  The Piper deployment is intentionally
different: its state is a 7D absolute end-effector pose computed with FK, and
the checkpointed UMI processor needs two adjacent control-loop poses.

This server accepts that two-pose state directly, runs the checkpointed
preprocessor and policy, postprocesses the *whole* action chunk with one
chunk-start reference, and returns timestamped absolute 7D EE targets.

RTC (Real-Time Chunking): a client may attach ``rtc_prev_actions_absolute``
(the un-executed ABSOLUTE 7D EE targets of the still-running chunk) plus
``rtc_execution_horizon`` / ``rtc_max_guidance_weight`` / ``rtc_inference_delay``
to any observation. The server re-anchors that tail into the current EE frame
(``reanchor_umi_rtc_prefix``), normalizes it with the checkpoint statistics,
and passes it as the RTC guidance prefix to Pi0.5/SmolVLA denoising — the same
contract as ``RTCInferenceEngine``. Observations without the key take the
original unguided path unchanged (including ``torch.inference_mode``).
"""

from __future__ import annotations

import argparse
import gc
import logging
import pickle  # nosec
import time
from concurrent import futures
from typing import Any

import grpc
import numpy as np
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
from lerobot.configs import RTCAttentionSchedule
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.rtc import RTCConfig, reanchor_umi_rtc_prefix
from lerobot.processor import NormalizerProcessorStep, UmiRelativeActionsStep
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.utils.constants import ACTION, OBS_STATE

# Observation-dict key carrying the client's un-executed absolute EE targets.
RTC_PREV_ACTIONS_KEY = "rtc_prev_actions_absolute"
RTC_POLICIES = ("pi05", "pi0", "smolvla")


def _fit_prefix_length(prefix: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Trim or zero-pad the re-anchored RTC prefix to exactly ``target_steps``."""
    if prefix.shape[0] >= target_steps:
        return prefix[:target_steps]
    pad = torch.zeros(
        (target_steps - prefix.shape[0], prefix.shape[-1]), dtype=prefix.dtype, device=prefix.device
    )
    return torch.cat([prefix, pad], dim=0)


class UmiRelativeEEPolicyServer(PolicyServer):
    """Async policy server preserving the UMI relative-EE deployment contract."""

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        # Prefer the checkpoint's actual type over the client's --policy_type. A mismatch
        # would otherwise crash the wrong policy class on the checkpoint's config (e.g.
        # ACT + SmolVLAConfig -> no 'use_vae'). Reading config.json is cheap (no weights);
        # the base loads from request.data, so hand it the corrected spec.
        policy_specs = pickle.loads(request.data)  # nosec
        actual_type = PreTrainedConfig.from_pretrained(policy_specs.pretrained_name_or_path).type
        if actual_type != policy_specs.policy_type:
            self.logger.warning(
                "Client requested policy type '%s' but the checkpoint is '%s'; using the "
                "checkpoint's actual type. Set --policy_type=%s on the client to silence "
                "this warning.",
                policy_specs.policy_type,
                actual_type,
                actual_type,
            )
            policy_specs.policy_type = actual_type
            request.data = pickle.dumps(policy_specs)  # nosec

        # Identity of the requested policy, so we can reuse an already-loaded
        # checkpoint instead of reloading it. The base server reloads on every
        # SendPolicyInstructions: ``self.policy = from_pretrained(...)`` builds the
        # new ~7 GB pi05 on the GPU *before* dropping the old ``self.policy``, so two
        # models briefly coexist and OOM a 16 GB card. Reuse the resident model when
        # the checkpoint matches; otherwise free it first (see ``_free_policy``).
        requested_key = (policy_specs.policy_type, policy_specs.pretrained_name_or_path)
        loaded_key = getattr(self, "_loaded_policy_key", None)
        if self.policy is not None and loaded_key == requested_key:
            # Same checkpoint already on the GPU: refresh the per-request runtime
            # knobs the base would have set and reset processor state. No reload,
            # no extra GPU memory.
            self.device = policy_specs.device
            self.policy_type = policy_specs.policy_type
            self.lerobot_features = policy_specs.lerobot_features
            self.actions_per_chunk = policy_specs.actions_per_chunk
            try:
                self.policy.to(self.device)
            except Exception:
                self.logger.debug("Could not move reused policy to %s", self.device, exc_info=True)
            self.policy.config.device = str(self.device)
            self.policy.eval()
            self.policy.reset()
            if self.preprocessor is not None:
                self.preprocessor.reset()
            if self.postprocessor is not None:
                self.postprocessor.reset()
            self.logger.info(
                "Reusing already-loaded %s policy from %s (reload skipped).",
                policy_specs.policy_type,
                policy_specs.pretrained_name_or_path,
            )
            return services_pb2.Empty()

        # Different checkpoint (or first load): unload the previous policy and
        # processors so the GPU is empty before from_pretrained allocates again.
        self._free_policy()

        response = super().SendPolicyInstructions(request, context)
        if self.policy is not None:
            self.policy.config.device = str(self.device)
            self.policy.eval()
            self.policy.reset()
            self._loaded_policy_key = requested_key
        else:
            self._loaded_policy_key = None
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()
        return response

    def _free_policy(self) -> None:
        """Drop references to the loaded policy/processors and reclaim GPU memory.

        Called before loading a *different* checkpoint so two models never coexist
        on the GPU.
        """
        if self.policy is None and self.preprocessor is None and self.postprocessor is None:
            return
        self.logger.info("Unloading previous policy/processors to free GPU memory...")
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self._loaded_policy_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _rtc_preprocessor_steps(self) -> tuple[UmiRelativeActionsStep | None, NormalizerProcessorStep | None]:
        """(umi_step, normalizer_step) of the loaded preprocessor, cached per preprocessor identity.

        The cache key is the preprocessor object itself, so it stays valid across
        checkpoint-reuse requests and refreshes automatically after a reload.
        """
        cache = getattr(self, "_rtc_steps_cache", None)
        if cache is not None and cache[0] is self.preprocessor:
            return cache[1], cache[2]
        umi = next(
            (s for s in self.preprocessor.steps if isinstance(s, UmiRelativeActionsStep) and s.enabled),
            None,
        )
        normalizer = next(
            (s for s in self.preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        self._rtc_steps_cache = (self.preprocessor, umi, normalizer)
        return umi, normalizer

    def _ensure_rtc_config(self, core_policy, execution_horizon: int, max_guidance_weight: float) -> None:
        """Install/refresh the RTC config + processor on the core policy (token-cached)."""
        token = (id(core_policy), execution_horizon, max_guidance_weight)
        if (
            getattr(self, "_rtc_init_token", None) == token
            and getattr(core_policy, "rtc_processor", None) is not None
        ):
            return
        core_policy.config.rtc_config = RTCConfig(
            enabled=False,
            execution_horizon=execution_horizon,
            max_guidance_weight=max_guidance_weight,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
        core_policy.init_rtc_processor()
        self._rtc_init_token = token

    def _parse_rtc_request(self, observation_t: TimedObservation) -> dict[str, Any] | None:
        """Extract RTC guidance extras from the observation payload, if any.

        Returns None (and the caller takes the ordinary unguided path) when the
        key is absent, the policy does not support RTC, or the payload is
        malformed. Never raises.
        """
        raw = observation_t.get_observation()
        prev_abs = raw.get(RTC_PREV_ACTIONS_KEY) if isinstance(raw, dict) else None
        if prev_abs is None:
            return None
        if self.policy_type not in RTC_POLICIES:
            self.logger.warning(
                "Observation carried %s but policy %s has no RTC hook; predicting unguided",
                RTC_PREV_ACTIONS_KEY,
                self.policy_type,
            )
            return None
        try:
            prev = torch.as_tensor(np.asarray(prev_abs, dtype=np.float32))
            if prev.ndim == 3:
                prev = prev[0]
            if prev.ndim != 2 or prev.shape[-1] != 7 or prev.shape[0] == 0:
                raise ValueError(f"expected [T,7] or [B,T,7] absolute EE targets, got {tuple(prev.shape)}")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Invalid %s payload (%s); predicting unguided", RTC_PREV_ACTIONS_KEY, exc)
            return None
        return {
            "prev_abs": prev,
            "execution_horizon": max(1, int(raw.get("rtc_execution_horizon", 10))),
            "max_guidance_weight": float(raw.get("rtc_max_guidance_weight", 10.0)),
            "inference_delay": max(0, int(raw.get("rtc_inference_delay", 0))),
        }

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
            task = raw["task"]
            # Pi05PrepareStateTokenizerProcessorStep iterates the task sequence
            # directly (no str -> [str] normalization like the SmolVLA
            # pipeline's NewLineTask/Tokenizer steps), so a bare string would be
            # walked character-by-character.
            if self.policy_type == "pi05" and isinstance(task, str):
                task = [task]
            observation["task"] = task
        return observation

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Run UMI inference and postprocess the complete chunk in one call."""
        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("Policy instructions must be sent before inference")

        started = time.perf_counter()
        rtc_request = self._parse_rtc_request(observation_t)
        observation = self._prepare_umi_observation(observation_t)
        t_prepared = time.perf_counter()

        # The RTC guidance differentiates the denoiser via autograd.grad, which
        # cannot run under inference_mode (inference tensors carry no grad_fn);
        # torch.no_grad() works because the RTC processor re-enables grad
        # internally. The unguided path keeps the original inference_mode.
        grad_context = torch.no_grad() if rtc_request is not None else torch.inference_mode()
        with grad_context:
            processed = self.preprocessor(observation)
            t_preprocessed = time.perf_counter()

            predict_kwargs: dict[str, Any] = {}
            core_policy = None
            if rtc_request is not None:
                core_policy = self.policy.get_base_model() if hasattr(self.policy, "get_base_model") else self.policy
                umi_step, normalizer_step = self._rtc_preprocessor_steps()
                cached_state = umi_step.get_cached_state() if umi_step is not None else None
                if umi_step is None or normalizer_step is None or cached_state is None:
                    self.logger.warning(
                        "RTC requested but the preprocessor lacks a cached UMI EE state; predicting unguided"
                    )
                else:
                    self._ensure_rtc_config(
                        core_policy,
                        rtc_request["execution_horizon"],
                        rtc_request["max_guidance_weight"],
                    )
                    prefix = reanchor_umi_rtc_prefix(
                        prev_actions_absolute=rtc_request["prev_abs"],
                        current_state=cached_state,
                        normalizer_step=normalizer_step,
                        policy_device=str(self.device),
                    )
                    prefix = _fit_prefix_length(prefix, rtc_request["execution_horizon"])
                    predict_kwargs = {
                        "prev_chunk_left_over": prefix,
                        "inference_delay": rtc_request["inference_delay"],
                        "execution_horizon": rtc_request["execution_horizon"],
                    }
                    core_policy.config.rtc_config.enabled = True
            try:
                if predict_kwargs:
                    predicted = self.policy.predict_action_chunk(processed, **predict_kwargs)
                else:
                    predicted = self.policy.predict_action_chunk(processed)
            finally:
                if predict_kwargs and core_policy is not None:
                    core_policy.config.rtc_config.enabled = False
            t_inferred = time.perf_counter()
            if predicted.ndim == 2:
                predicted = predicted.unsqueeze(0)
            predicted = predicted[:, : self.actions_per_chunk, :]

            # This must remain a single call. UmiAbsoluteActionsStep uses the
            # preprocessor's cached current pose as the base for every target.
            actions = self.postprocessor(predicted)
        t_postprocessed = time.perf_counter()

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
        # Ship the server-side compute duration to the client so it can split the
        # observed wire time into transport vs server compute.
        server_elapsed_ms = (time.perf_counter() - started) * 1000
        # Chunk-provenance for the temporal-ensemble audit: ``reference_ee`` is the
        # absolute 7D EE pose that anchored this chunk (the observation's current
        # pose, = T_anchor) and ``relative_action`` is the raw per-step model output
        # ΔT. ``action`` (above) is the ABSOLUTE target = T_anchor ∘ ΔT. Logging both
        # lets a client confirm the ensemble blends absolute targets (same base
        # frame), not relative deltas averaged directly. Guarded: never break inference.
        try:
            raw_state = torch.as_tensor(
                observation_t.get_observation().get(OBS_STATE), dtype=torch.float32
            )
            ref_ee = (raw_state[-1] if raw_state.ndim >= 1 else raw_state).detach().cpu().clone()
            rel_per_step = predicted[0].detach().cpu()  # [T, D] raw model output
        except Exception:  # noqa: BLE001
            ref_ee, rel_per_step = None, None
        for i, timed in enumerate(result):
            timed.server_elapsed_ms = server_elapsed_ms
            if ref_ee is not None:
                timed.reference_ee = ref_ee
            if rel_per_step is not None and i < rel_per_step.shape[0]:
                timed.relative_action = rel_per_step[i]
            if predict_kwargs:
                # Tag RTC chunks so the client can apply replace semantics.
                timed.rtc_guided = True
        self.logger.info(
            "Observation %s -> %s absolute EE actions in %.1fms "
            "(prepare %.1fms, preprocess %.1fms, infer %.1fms, postprocess %.1fms)%s",
            observation_t.get_timestep(),
            len(result),
            server_elapsed_ms,
            (t_prepared - started) * 1000,
            (t_preprocessed - t_prepared) * 1000,
            (t_inferred - t_preprocessed) * 1000,
            (t_postprocessed - t_inferred) * 1000,
            " [RTC guided]" if predict_kwargs else "",
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

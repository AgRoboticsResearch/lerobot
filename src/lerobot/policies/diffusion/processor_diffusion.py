#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import torch
from copy import deepcopy
from dataclasses import dataclass
from copy import deepcopy

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UmiAbsoluteActionsStep,
    UmiDeriveStateFromActionStep,
    UmiRelativeActionsStep,
    UmiRelativeStateStep,
    UnnormalizerProcessorStep,
    make_umi_cache_key,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.configs import PipelineFeatureType
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_diffusion import DiffusionConfig


@ProcessorStepRegistry.register("umi_state_time_dim")
@dataclass
class UmiStateTimeDimStep(ProcessorStep):
    """恢复 state 的时间维: DP 的 compute_loss 要求 [B, n_obs_steps, D]。

    UMI 链(派生→相对化→拼装)输出拍平的 [B, 20];n_obs_steps=1 时在此
    unsqueeze 回 [B, 1, 20]。"""
    enabled: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None or state.ndim != 2:
            return transition
        result = deepcopy(transition)
        result_observation = dict(observation)
        result_observation[OBS_STATE] = state.unsqueeze(1)
        result[TransitionKey.OBSERVATION] = result_observation
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        observation_features = result.get(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observation_features:
            feature = observation_features[OBS_STATE]
            observation_features[OBS_STATE] = PolicyFeature(type=feature.type, shape=(1, *feature.shape))
        return result


def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Normalizing the input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving the data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the diffusion policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # SROI UMI 相对 EE 链(与 act/processor_act.py 同款;框架口径=相对动作空间,无开关):
    #   state 从 action 派生 → 动作转相对位姿 → 拼装 20D 相对 state(含速度)。
    cache_key = make_umi_cache_key()
    relative_step = UmiRelativeActionsStep(cache_key=cache_key)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        UmiDeriveStateFromActionStep(),
        relative_step,
        UmiRelativeStateStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        UmiStateTimeDimStep(),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        # 推理端: 相对动作 → 绝对动作(参考帧由 relative_step 的缓存维护)。
        UmiAbsoluteActionsStep(
            cache_key=cache_key,
            relative_step=relative_step,
            single_action_reference_steps=config.n_action_steps,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

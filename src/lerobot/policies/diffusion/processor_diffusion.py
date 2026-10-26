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
    UnnormalizerProcessorStep,
    make_umi_cache_key,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.configs import PipelineFeatureType
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.umi_relative_ee_processor import (
    to_umi_relative_actions,
    to_umi_relative_state,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_diffusion import DiffusionConfig


EE_KEY = "observation.ee"


@ProcessorStepRegistry.register("umi_ee_state")
@dataclass
class UmiEEStateStep(ProcessorStep):
    """ee 对 [t-1, t] → 20D UMI 相对状态,整形为 [B, 2, 10]。

    前半 = 上一帧位姿在当前帧坐标系下的表达(速度),后半 = 恒等+当前夹爪。
    头样本(t=0)时 [t-1] 越界由 dataloader 用帧 0 自复制填充 = 零速,符合
    "episode 从静止开始"的物理事实。"""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        ee = observation.get(EE_KEY)
        if ee is None or ee.ndim != 3 or ee.shape[-2:] != (2, 7):
            return transition
        result = deepcopy(transition)
        result_observation = dict(observation)
        result_observation[OBS_STATE] = to_umi_relative_state(ee).reshape(
            ee.shape[0], 2, 10
        )
        result[TransitionKey.OBSERVATION] = result_observation
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        observation_features = result.get(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observation_features:
            observation_features[OBS_STATE] = PolicyFeature(
                type=observation_features[OBS_STATE].type, shape=(2, 10)
            )
        return result


@ProcessorStepRegistry.register("umi_ee_relative_chunk")
@dataclass
class UmiEERelativeChunkStep(ProcessorStep):
    """动作目标(绝对) → 以 ee[t](观测帧自身位姿)为基准的相对动作。

    chunk[i] = ee[t]⁻¹ · ee[t+1+i]:第一个元素就是到下一帧的真实运动,
    不再是恒等;部署端以实时观测位姿为基准,与训练完全同源。"""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        ee = observation.get(EE_KEY)
        action = transition.get(TransitionKey.ACTION)
        if ee is None or action is None or ee.ndim != 3:
            return transition
        base = ee[:, -1, :]  # delta 0 帧 = 当前观测位姿
        result = deepcopy(transition)
        result[TransitionKey.ACTION] = to_umi_relative_actions(action, base)
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = deepcopy(features)
        action_features = result.get(PipelineFeatureType.ACTION, {})
        if ACTION in action_features:
            action_features[ACTION] = PolicyFeature(
                type=action_features[ACTION].type, shape=(10,)
            )
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

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        UmiEEStateStep(),
        UmiEERelativeChunkStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        # 相对→绝对的还原在部署循环内以实时 ee 观测为基准完成,
        # 此处后处理只做 unnormalize + 回 CPU。
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

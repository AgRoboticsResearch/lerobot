"""Pre/post-processing for LingBot-VA, including the UMI action bridge."""

from typing import Any

import torch

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    UmiAbsoluteAxisAngleActionsStep,
    UmiDeriveStateFromActionStep,
    UmiRelativeAxisAngleActionsStep,
    UmiRelativeStateStep,
    UnnormalizerProcessorStep,
    make_umi_cache_key,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_lingbot_va import LingBotVAConfig


def make_lingbot_va_pre_post_processors(
    config: LingBotVAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build LingBot pipelines with a reversible UMI 7D axis-angle adapter.

    The upstream LingBot latent always has 30 channels, but only
    ``used_action_channel_ids`` are exposed to a dataset.  For our single-arm UMI
    data those seven values are ``relative xyz + relative axis-angle + gripper``.
    They are quantile-normalized to the model's ``[-1, 1]`` action space before
    being scattered into channels 0..6 by the policy.
    """

    cache_key = make_umi_cache_key()
    relative_step = UmiRelativeAxisAngleActionsStep(cache_key=cache_key)
    action_norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: NormalizationMode.IDENTITY,
        FeatureType.ACTION: (
            NormalizationMode.QUANTILES
            if config.use_umi_relative_ee
            else NormalizationMode.IDENTITY
        ),
    }

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        *(
            [UmiDeriveStateFromActionStep(), relative_step, UmiRelativeStateStep()]
            if config.use_umi_relative_ee
            else []
        ),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=action_norm_map,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map={FeatureType.ACTION: action_norm_map[FeatureType.ACTION]},
            stats=dataset_stats,
        ),
        *(
            [
                UmiAbsoluteAxisAngleActionsStep(
                    cache_key=cache_key,
                    relative_step=relative_step,
                    single_action_reference_steps=config.n_action_steps,
                    initial_single_action_reference_steps=(
                        (config.frame_chunk_size - 1) * config.action_per_frame
                    ),
                )
            ]
            if config.use_umi_relative_ee
            else []
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

"""Diffusion processor factory for UMI-style relative end-effector actions."""

from __future__ import annotations

from typing import Any

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.relative_action_processor import (
    AbsoluteRot6dActionsProcessorStep,
    DeriveStateFromActionStep,
    DiffusionTrainingStateDimensionProcessorStep,
    RelativeRot6dActionsProcessorStep,
    RelativeRot6dStateProcessorStep,
    make_relative_ee_cache_key,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_diffusion_relative_ee_pre_post_processors(
    config, dataset_stats: dict | None = None
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    if not isinstance(config, DiffusionConfig):
        raise TypeError(f"Expected DiffusionConfig, got {type(config)}")

    cache_key = make_relative_ee_cache_key()
    relative_step = RelativeRot6dActionsProcessorStep(
        enabled=True,
        exclude_joints=config.relative_exclude_joints,
        action_names=getattr(config, "_action_names", None),
        cache_key=cache_key,
    )

    input_steps: list[Any] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]
    if config.derive_state_from_action:
        input_steps.append(DeriveStateFromActionStep(enabled=True))
    input_steps.extend(
        [
            relative_step,
            RelativeRot6dStateProcessorStep(
                enabled=True,
                exclude_joints=config.relative_exclude_joints,
                state_names=getattr(config, "_state_names", None),
            ),
            DiffusionTrainingStateDimensionProcessorStep(enabled=True),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
        ]
    )

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteRot6dActionsProcessorStep(
            enabled=True, relative_step=relative_step, cache_key=cache_key
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline(steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

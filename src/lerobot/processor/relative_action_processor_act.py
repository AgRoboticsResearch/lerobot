"""ACT processor factory with UMI-style relative rot6d action pipeline.

Creates preprocessor and postprocessor for ACT policies using:
  DeriveStateFromAction → RelativeRot6dActions → RelativeRot6dState → Normalize

This is a standalone module that does NOT modify the existing processor_act.py.
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.configs.types import NormalizationMode
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
    RelativeRot6dActionsProcessorStep,
    RelativeRot6dStateProcessorStep,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

logger = logging.getLogger(__name__)


def make_act_relative_ee_pre_post_processors(
    config,
    dataset_stats: dict | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Create ACT processors with UMI-style relative rot6d action pipeline.

    Pipeline:
      Preprocessor:
        1. RenameObservations
        2. AddBatchDimension
        3. DeviceStep
        4. DeriveStateFromAction (if enabled)
        5. RelativeRot6dActions (7D aa → 10D rot6d relative)
        6. RelativeRot6dState (2×7D aa → 20D rot6d relative)
        7. Normalizer

      Postprocessor:
        1. Unnormalizer
        2. AbsoluteRot6dActions (10D rot6d → 7D aa absolute)
        3. DeviceStep (CPU)

    Args:
        config: ACTRelativeEEConfig or any config with relative action fields.
        dataset_stats: Normalization statistics from dataset.

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines.
    """
    from lerobot.policies.act.configuration_act import ACTConfig

    assert isinstance(config, ACTConfig), (
        f"Expected ACTConfig, got {type(config)}"
    )

    # Build relative action processor step
    action_names = None
    if dataset_stats and "action" in dataset_stats:
        # Try to get action names from dataset metadata
        pass  # action_names set via config

    relative_step = RelativeRot6dActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=getattr(config, "_action_names", None),
    )

    # Build derive state step
    derive_step = DeriveStateFromActionStep(
        enabled=config.derive_state_from_action,
    )

    # Build relative state step
    relative_state_step = RelativeRot6dStateProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        state_names=getattr(config, "_state_names", None),
    )

    # Build input steps
    input_steps: list[Any] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]

    if config.derive_state_from_action:
        input_steps.append(derive_step)

    if config.use_relative_actions:
        input_steps.append(relative_step)
        input_steps.append(relative_state_step)

    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )

    # Build output steps
    output_steps: list[Any] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    if config.use_relative_actions:
        output_steps.append(
            AbsoluteRot6dActionsProcessorStep(
                enabled=True,
                relative_step=relative_step,
            )
        )

    output_steps.append(DeviceProcessorStep(device="cpu"))

    preprocessor = PolicyProcessorPipeline(
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    logger.info("Created ACT relative EE rot6d processor pipeline")
    logger.info(f"  derive_state_from_action: {config.derive_state_from_action}")
    logger.info(f"  use_relative_actions: {config.use_relative_actions}")
    logger.info(f"  pose_dim: {config.pose_dim}")
    logger.info(f"  use_rot6d: {config.use_rot6d}")
    logger.info(f"  Preprocessor steps: {[s.__class__.__name__ for s in preprocessor.steps]}")
    logger.info(f"  Postprocessor steps: {[s.__class__.__name__ for s in postprocessor.steps]}")

    return preprocessor, postprocessor

from __future__ import annotations

import json

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.processor_lingbot_va import make_lingbot_va_pre_post_processors
from lerobot.processor import (
    UmiAbsoluteAxisAngleActionsStep,
    UmiDeriveStateFromActionStep,
    UmiRelativeAxisAngleActionsStep,
    UmiRelativeStateStep,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _config() -> LingBotVAConfig:
    cfg = LingBotVAConfig(
        device="cpu",
        use_umi_relative_ee=True,
        obs_cam_keys=[f"{OBS_IMAGES}.image"],
    )
    cfg.input_features = {
        f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    return cfg


def _stats() -> dict:
    return {
        ACTION: {"q01": torch.full((7,), -1.0), "q99": torch.full((7,), 1.0)},
        OBS_STATE: {"min": torch.full((20,), -1.0), "max": torch.full((20,), 1.0)},
    }


def test_umi_processor_is_seven_dimensional_axis_angle_bridge() -> None:
    cfg = _config()
    preprocessor, postprocessor = make_lingbot_va_pre_post_processors(cfg, _stats())

    assert any(isinstance(step, UmiDeriveStateFromActionStep) for step in preprocessor.steps)
    assert any(isinstance(step, UmiRelativeStateStep) for step in preprocessor.steps)
    relative = next(
        step for step in preprocessor.steps if isinstance(step, UmiRelativeAxisAngleActionsStep)
    )
    absolute = next(
        step for step in postprocessor.steps if isinstance(step, UmiAbsoluteAxisAngleActionsStep)
    )
    assert absolute.relative_step is relative
    assert absolute.single_action_reference_steps == 16
    assert absolute.initial_single_action_reference_steps == 12


def test_axis_angle_reference_horizons_survive_serialization(tmp_path) -> None:
    cfg = _config()
    preprocessor, postprocessor = make_lingbot_va_pre_post_processors(cfg, _stats())
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    post_config = json.loads((tmp_path / "policy_postprocessor.json").read_text())
    absolute = next(
        step
        for step in post_config["steps"]
        if step.get("registry_name") == "umi_absolute_axis_angle_actions"
    )
    assert absolute["config"]["single_action_reference_steps"] == 16
    assert absolute["config"]["initial_single_action_reference_steps"] == 12


def test_non_umi_action_postprocessor_keeps_identity_normalization() -> None:
    cfg = _config()
    cfg.use_umi_relative_ee = False
    _, postprocessor = make_lingbot_va_pre_post_processors(cfg)

    unnormalizer = next(
        step for step in postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)
    )
    assert unnormalizer.norm_map[FeatureType.ACTION] is NormalizationMode.IDENTITY

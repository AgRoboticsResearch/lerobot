from __future__ import annotations

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES


def make_config(**overrides) -> LingBotVAConfig:
    kwargs = {"device": "cpu"}
    kwargs.update(overrides)
    return LingBotVAConfig(**kwargs)


def test_registered_and_factory_constructible() -> None:
    assert "lingbot_va" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("lingbot_va") is LingBotVAConfig
    assert isinstance(make_policy_config("lingbot_va", device="cpu"), LingBotVAConfig)


def test_umi_temporal_queries_include_one_reference_and_sixteen_targets() -> None:
    cfg = make_config(use_umi_relative_ee=True, frame_chunk_size=4, action_per_frame=4)
    assert cfg.chunk_size == cfg.n_action_steps == 16
    assert cfg.action_delta_indices == [-1] + list(range(16))
    assert cfg.observation_delta_indices == list(range(16))


def test_umi_requires_single_arm_axis_angle_channels_zero_through_six() -> None:
    with pytest.raises(ValueError, match="channels 0..6"):
        make_config(use_umi_relative_ee=True, used_action_channel_ids=[7, 8, 9, 10, 11, 12, 13])


def test_validate_features_sets_dataset_facing_seven_dimensional_action() -> None:
    cfg = make_config(use_umi_relative_ee=True, obs_cam_keys=[f"{OBS_IMAGES}.image"])
    cfg.input_features = {
        f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))
    }
    cfg.output_features = {}
    cfg.validate_features()
    assert cfg.output_features[ACTION].shape == (7,)


def test_optimizer_uses_upstream_warmup_then_constant_schedule() -> None:
    cfg = make_config(scheduler_warmup_steps=17)
    assert cfg.get_optimizer_preset().lr == cfg.optimizer_lr
    scheduler = cfg.get_scheduler_preset()
    assert scheduler is not None
    assert scheduler.type == "constant_with_warmup"
    assert scheduler.num_warmup_steps == 17

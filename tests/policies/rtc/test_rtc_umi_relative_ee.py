"""Tests for RTC prefix re-anchoring with UMI relative end-effector actions."""

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.rtc.relative import reanchor_umi_rtc_prefix
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import (
    NormalizerProcessorStep,
    TransitionKey,
    UmiRelativeActionsStep,
    UnnormalizerProcessorStep,
    create_transition,
    to_umi_absolute_actions,
    to_umi_relative_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _dataset_stats() -> dict[str, dict[str, torch.Tensor]]:
    action_offset = torch.linspace(-0.4, 0.5, 10)
    state_offset = torch.linspace(-0.7, 0.8, 20)
    return {
        ACTION: {
            "mean": action_offset,
            "std": torch.linspace(0.5, 1.4, 10),
            "min": action_offset - 2.0,
            "max": action_offset + 3.0,
            "q01": action_offset - 1.5,
            "q99": action_offset + 2.5,
        },
        OBS_STATE: {
            "mean": state_offset,
            "std": torch.linspace(0.5, 1.4, 20),
            "min": state_offset - 2.0,
            "max": state_offset + 3.0,
            "q01": state_offset - 1.5,
            "q99": state_offset + 2.5,
        },
    }


def _make_umi_processors(policy_type: str):
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,))}
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))}
    common = {
        "use_umi_relative_ee": True,
        "chunk_size": 4,
        "n_action_steps": 4,
        "device": "cpu",
        "input_features": input_features,
        "output_features": output_features,
    }

    if policy_type == "pi05":
        config = PI05Config(**common)
        return make_pi05_pre_post_processors(config, _dataset_stats())
    if policy_type == "smolvla":
        config = SmolVLAConfig(**common)
        return make_smolvla_pre_post_processors(config, _dataset_stats())
    raise ValueError(f"Unsupported test policy: {policy_type}")


def _get_step(pipeline, step_type):
    return next(step for step in pipeline.steps if isinstance(step, step_type))


@pytest.mark.parametrize("policy_type", ["pi05", "smolvla"])
def test_umi_rtc_prefix_reanchoring_preserves_absolute_targets(policy_type, monkeypatch):
    """Both policy pipelines rebuild the same absolute leftover targets in the new frame."""
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )
    preprocessor, postprocessor = _make_umi_processors(policy_type)
    relative_step = _get_step(preprocessor, UmiRelativeActionsStep)
    normalizer = _get_step(preprocessor, NormalizerProcessorStep)
    unnormalizer = _get_step(postprocessor, UnnormalizerProcessorStep)

    current_state = torch.tensor([[0.45, -0.2, 0.35, 0.2, -0.15, 0.1, 0.3]])
    prev_actions_absolute = torch.tensor(
        [
            [0.50, -0.18, 0.37, 0.24, -0.10, 0.08, 0.2],
            [0.54, -0.12, 0.40, 0.28, -0.05, 0.04, 0.4],
            [0.58, -0.08, 0.44, 0.32, 0.02, -0.01, 0.7],
        ]
    )

    relative_step(create_transition(observation={OBS_STATE: current_state}))
    model_prefix = reanchor_umi_rtc_prefix(
        prev_actions_absolute=prev_actions_absolute,
        current_state=relative_step.get_cached_state(),
        normalizer_step=normalizer,
        policy_device="cpu",
    )

    expected_relative = to_umi_relative_actions(prev_actions_absolute, current_state[0])
    expected_model_prefix = normalizer(create_transition(action=expected_relative))[TransitionKey.ACTION]
    torch.testing.assert_close(model_prefix, expected_model_prefix, atol=1e-6, rtol=1e-6)

    restored_relative = unnormalizer(create_transition(action=model_prefix))[TransitionKey.ACTION]
    restored_absolute = to_umi_absolute_actions(restored_relative, current_state[0])
    torch.testing.assert_close(restored_absolute, prev_actions_absolute, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("policy_type", ["pi05", "smolvla"])
def test_umi_rtc_prefix_uses_latest_cached_chunk_base(policy_type, monkeypatch):
    """Moving the current EE pose changes model space without changing absolute targets."""
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )
    preprocessor, _ = _make_umi_processors(policy_type)
    relative_step = _get_step(preprocessor, UmiRelativeActionsStep)
    normalizer = _get_step(preprocessor, NormalizerProcessorStep)
    actions_absolute = torch.tensor(
        [[0.6, 0.1, 0.4, 0.3, -0.1, 0.05, 0.5], [0.7, 0.2, 0.5, 0.4, 0.0, 0.1, 0.6]]
    )

    old_state = torch.tensor([[0.2, -0.1, 0.3, 0.1, -0.2, 0.05, 0.4]])
    relative_step(create_transition(observation={OBS_STATE: old_state}))
    old_prefix = reanchor_umi_rtc_prefix(
        actions_absolute, relative_step.get_cached_state(), normalizer, "cpu"
    )

    new_state = torch.tensor([[0.35, 0.0, 0.32, 0.15, -0.1, 0.02, 0.4]])
    relative_step(create_transition(observation={OBS_STATE: new_state}))
    new_prefix = reanchor_umi_rtc_prefix(
        actions_absolute, relative_step.get_cached_state(), normalizer, "cpu"
    )

    assert not torch.allclose(old_prefix, new_prefix)
    expected_new = normalizer(
        create_transition(action=to_umi_relative_actions(actions_absolute, new_state[0]))
    )[TransitionKey.ACTION]
    torch.testing.assert_close(new_prefix, expected_new, atol=1e-6, rtol=1e-6)


def test_umi_rtc_prefix_rejects_non_absolute_queue_actions():
    with pytest.raises(ValueError, match="absolute 7D leftovers"):
        reanchor_umi_rtc_prefix(torch.zeros(3, 10), torch.zeros(1, 7), None, "cpu")

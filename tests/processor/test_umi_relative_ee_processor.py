import json

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act import ACTConfig, make_act_pre_post_processors
from lerobot.policies.pi05 import PI05Config, make_pi05_pre_post_processors
from lerobot.policies.smolvla import SmolVLAConfig, make_smolvla_pre_post_processors
from lerobot.processor import (
    ProcessorStepRegistry,
    UmiAbsoluteActionsStep,
    UmiDeriveStateFromActionStep,
    UmiRelativeActionsStep,
    UmiRelativeStateStep,
    to_umi_absolute_actions,
    to_umi_relative_actions,
    to_umi_relative_state,
)
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def _poses(batch: int = 2, steps: int = 4) -> torch.Tensor:
    poses = torch.zeros(batch, steps, 7)
    poses[..., 0] = torch.linspace(0.0, 0.03, steps)
    poses[..., 3:6] = torch.tensor([0.1, -0.2, 0.05])
    poses[..., 6] = 0.4
    return poses


def test_umi_action_round_trip_uses_one_reference_for_full_chunk():
    poses = _poses()
    reference = poses[:, 0]
    relative = to_umi_relative_actions(poses, reference)
    recovered = to_umi_absolute_actions(relative, reference)

    assert relative.shape == (2, 4, 10)
    torch.testing.assert_close(recovered, poses, atol=1e-5, rtol=1e-5)


def test_umi_derive_and_state_shapes():
    actions = _poses(2, 5)
    transition = create_transition(action=actions)
    derived = UmiDeriveStateFromActionStep()(transition)

    state = derived[TransitionKey.OBSERVATION][OBS_STATE]
    assert state.shape == (2, 2, 7)
    assert derived[TransitionKey.ACTION].shape == (2, 4, 7)
    assert to_umi_relative_state(state).shape == (2, 20)


def test_pi05_umi_config_and_serialized_processor_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )
    config = PI05Config(
        use_umi_relative_ee=True,
        chunk_size=4,
        n_action_steps=4,
        device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
    )
    assert config.use_relative_actions
    assert config.action_delta_indices == [-1, 0, 1, 2, 3]

    stats = {
        OBS_STATE: {"q01": torch.full((20,), -1.0), "q99": torch.full((20,), 1.0)},
        ACTION: {"q01": torch.full((10,), -1.0), "q99": torch.full((10,), 1.0)},
    }
    preprocessor, postprocessor = make_pi05_pre_post_processors(config, stats)
    assert any(isinstance(step, UmiDeriveStateFromActionStep) for step in preprocessor.steps)
    assert any(isinstance(step, UmiRelativeActionsStep) for step in preprocessor.steps)
    assert any(isinstance(step, UmiRelativeStateStep) for step in preprocessor.steps)
    assert any(isinstance(step, UmiAbsoluteActionsStep) for step in postprocessor.steps)

    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    pre_config = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    post_config = json.loads((tmp_path / "policy_postprocessor.json").read_text())
    pre_names = [step.get("registry_name") for step in pre_config["steps"]]
    post_names = [step.get("registry_name") for step in post_config["steps"]]
    assert "umi_relative_actions" in pre_names
    assert "umi_absolute_actions" in post_names


def test_act_and_smolvla_share_canonical_umi_steps(monkeypatch):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )
    features_in = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,))}
    features_out = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))}
    stats = {
        OBS_STATE: {"min": torch.full((20,), -1.0), "max": torch.full((20,), 1.0)},
        ACTION: {"min": torch.full((10,), -1.0), "max": torch.full((10,), 1.0)},
    }

    act = ACTConfig(
        use_umi_relative_ee=True,
        chunk_size=4,
        n_action_steps=4,
        device="cpu",
        input_features=features_in,
        output_features=features_out,
    )
    smolvla = SmolVLAConfig(
        use_umi_relative_ee=True,
        chunk_size=4,
        n_action_steps=4,
        device="cpu",
        input_features=features_in,
        output_features=features_out,
    )

    assert act.action_delta_indices == [-1, 0, 1, 2, 3]
    assert smolvla.action_delta_indices == [-1, 0, 1, 2, 3]
    assert act.normalization_mapping["ACTION"] is NormalizationMode.MIN_MAX
    assert smolvla.normalization_mapping["STATE"] is NormalizationMode.MIN_MAX

    for preprocessor, postprocessor in (
        make_act_pre_post_processors(act, stats),
        make_smolvla_pre_post_processors(smolvla, stats),
    ):
        assert any(isinstance(step, UmiDeriveStateFromActionStep) for step in preprocessor.steps)
        assert any(isinstance(step, UmiRelativeActionsStep) for step in preprocessor.steps)
        assert any(isinstance(step, UmiRelativeStateStep) for step in preprocessor.steps)
        assert any(isinstance(step, UmiAbsoluteActionsStep) for step in postprocessor.steps)


def test_legacy_config_aliases_and_processor_names_remain_loadable():
    config = ACTConfig(
        derive_state_from_action=True,
        use_relative_actions=True,
        pose_dim=6,
        use_rot6d=True,
    )
    assert config.use_umi_relative_ee

    for registry_name in (
        "derive_state_from_action_rot6d",
        "relative_rot6d_actions_processor",
        "relative_rot6d_state_processor",
        "absolute_rot6d_actions_processor",
    ):
        assert ProcessorStepRegistry.get(registry_name) is not None

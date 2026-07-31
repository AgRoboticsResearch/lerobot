import json

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act import ACTConfig, make_act_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
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


def _state_at_x(x: float) -> torch.Tensor:
    state = torch.zeros(1, 7)
    state[:, 0] = x
    return state


def _identity_relative_action(*, steps: int | None = None) -> torch.Tensor:
    shape = (1, 10) if steps is None else (1, steps, 10)
    action = torch.zeros(shape)
    action[..., 3] = 1.0
    action[..., 7] = 1.0
    return action


def test_umi_action_round_trip_uses_one_reference_for_full_chunk():
    poses = _poses()
    reference = poses[:, 0]
    relative = to_umi_relative_actions(poses, reference)
    recovered = to_umi_absolute_actions(relative, reference)

    assert relative.shape == (2, 4, 10)
    torch.testing.assert_close(recovered, poses, atol=1e-5, rtol=1e-5)


def test_umi_queued_single_actions_keep_the_chunk_start_reference():
    relative_step = UmiRelativeActionsStep(cache_key="queued_reference_test")
    absolute_step = UmiAbsoluteActionsStep(
        relative_step=relative_step,
        cache_key=relative_step.cache_key,
        single_action_reference_steps=3,
    )
    action = _identity_relative_action()

    decoded_x = []
    for current_x in (1.0, 2.0, 3.0, 4.0):
        relative_step(create_transition(observation={OBS_STATE: _state_at_x(current_x)}))
        decoded = absolute_step(create_transition(action=action))[TransitionKey.ACTION]
        decoded_x.append(decoded[0, 0])

    torch.testing.assert_close(torch.stack(decoded_x), torch.tensor([1.0, 1.0, 1.0, 4.0]))


def test_umi_full_chunks_use_current_reference_and_clear_a_partial_single_action_pin():
    relative_step = UmiRelativeActionsStep(cache_key="full_chunk_reference_test")
    absolute_step = UmiAbsoluteActionsStep(
        relative_step=relative_step,
        cache_key=relative_step.cache_key,
        single_action_reference_steps=3,
    )

    relative_step(create_transition(observation={OBS_STATE: _state_at_x(1.0)}))
    absolute_step(create_transition(action=_identity_relative_action()))

    relative_step(create_transition(observation={OBS_STATE: _state_at_x(2.0)}))
    chunk = absolute_step(create_transition(action=_identity_relative_action(steps=2)))[
        TransitionKey.ACTION
    ]
    torch.testing.assert_close(chunk[..., 0], torch.full((1, 2), 2.0))

    relative_step(create_transition(observation={OBS_STATE: _state_at_x(3.0)}))
    single = absolute_step(create_transition(action=_identity_relative_action()))[TransitionKey.ACTION]
    torch.testing.assert_close(single[..., 0], torch.tensor([3.0]))


def test_umi_postprocessor_reset_discards_a_partial_single_action_pin():
    relative_step = UmiRelativeActionsStep(cache_key="reset_reference_test")
    absolute_step = UmiAbsoluteActionsStep(
        relative_step=relative_step,
        cache_key=relative_step.cache_key,
        single_action_reference_steps=3,
    )

    relative_step(create_transition(observation={OBS_STATE: _state_at_x(1.0)}))
    absolute_step(create_transition(action=_identity_relative_action()))
    relative_step(create_transition(observation={OBS_STATE: _state_at_x(2.0)}))
    absolute_step.reset()

    decoded = absolute_step(create_transition(action=_identity_relative_action()))[TransitionKey.ACTION]
    torch.testing.assert_close(decoded[..., 0], torch.tensor([2.0]))


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

    post_step_config = next(
        step for step in post_config["steps"] if step.get("registry_name") == "umi_absolute_actions"
    )
    assert post_step_config["config"]["single_action_reference_steps"] == config.n_action_steps

    # Simulate a checkpoint saved before the reference horizon was serialized.
    del post_step_config["config"]["single_action_reference_steps"]
    (tmp_path / "policy_postprocessor.json").write_text(json.dumps(post_config))
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(config, str(tmp_path))
    loaded_relative = next(
        step for step in loaded_preprocessor.steps if isinstance(step, UmiRelativeActionsStep)
    )
    loaded_absolute = next(
        step for step in loaded_postprocessor.steps if isinstance(step, UmiAbsoluteActionsStep)
    )
    assert loaded_absolute.relative_step is loaded_relative
    assert loaded_absolute.single_action_reference_steps == config.n_action_steps


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
        absolute_step = next(
            step for step in postprocessor.steps if isinstance(step, UmiAbsoluteActionsStep)
        )
        assert absolute_step.single_action_reference_steps == 4


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

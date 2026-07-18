# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import importlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.parser import wrap
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_train import _make_offline_dataloader, _run_validation, validate_policy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


class _ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, values: list[float]):
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"value": torch.tensor(self.values[index], dtype=torch.float32)}


class _IdentityPreprocessor:
    def __init__(self):
        self.reset_count = 0

    def __call__(self, batch):
        return batch

    def reset(self):
        self.reset_count += 1


class _LossPolicy(torch.nn.Module):
    """Minimal policy returning a ``(loss, loss_dict)`` tuple like real policies.

    With ``loss_dict=True`` it reports ACT-style ``l1_loss`` / ``kld_loss``
    components derived from the batch mean, so tests can verify they are
    sample-weighted and surfaced separately from the total loss.
    """

    def __init__(self, loss_dict: bool = False):
        super().__init__()
        self.forward_states: list[tuple[bool, bool]] = []
        self._report_components = loss_dict

    def forward(self, batch):
        self.forward_states.append((self.training, torch.is_grad_enabled()))
        value = batch["value"].mean()
        if self._report_components:
            return value, {"l1_loss": float(value), "kld_loss": float(value) / 10.0}
        return value, {}


class _SingleProcessAccelerator:
    def __init__(self):
        self.wait_count = 0

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def gather_for_metrics(values):
        return values

    def wait_for_everyone(self):
        self.wait_count += 1


class _FakeWandb:
    def __init__(self):
        self.logged = []

    def log(self, data, step=None):
        self.logged.append((data, step))


class _FakeValidationLogger:
    def __init__(self):
        self.logged = []

    def log_dict(self, data, step, mode):
        self.logged.append((data, step, mode))


def test_validation_config_defaults():
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="train/repo"))

    assert cfg.validation_dataset is None
    assert cfg.val_freq == 1_000


def test_nested_validation_dataset_cli(monkeypatch):
    @wrap()
    def parse_config(cfg: TrainPipelineConfig):
        return cfg

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--dataset.repo_id=train/repo",
            "--validation_dataset.repo_id=validation/repo",
            "--validation_dataset.root=/tmp/validation",
            "--validation_dataset.episodes=[1,2]",
            "--val_freq=250",
        ],
    )

    cfg = parse_config()

    assert cfg.validation_dataset is not None
    assert cfg.validation_dataset.repo_id == "validation/repo"
    assert cfg.validation_dataset.root == "/tmp/validation"
    assert cfg.validation_dataset.episodes == [1, 2]
    assert cfg.val_freq == 250


def test_validation_loss_is_sample_weighted_and_restores_state():
    dataset = _ValidationDataset([1.0, 2.0, 3.0, 4.0, 10.0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    policy = _LossPolicy()
    policy.train()
    preprocessor = _IdentityPreprocessor()

    metrics = validate_policy(policy, dataloader, preprocessor, _SingleProcessAccelerator())

    # The default fake policy reports no loss components, so only the total is returned.
    assert metrics == {"loss": 4.0}
    assert policy.training
    assert policy.forward_states == [(False, False), (False, False), (False, False)]
    assert preprocessor.reset_count == 2


def test_validation_forwards_loss_components_sample_weighted():
    """Per-component losses (ACT's l1 / kld) must be sample-weighted and returned
    separately from the total, so a collapsing KL can't hide behind the total loss."""
    dataset = _ValidationDataset([1.0, 2.0, 3.0, 4.0, 10.0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    policy = _LossPolicy(loss_dict=True)
    policy.train()
    preprocessor = _IdentityPreprocessor()

    metrics = validate_policy(policy, dataloader, preprocessor, _SingleProcessAccelerator())

    # total and l1 both equal the sample-weighted batch mean = (1+2+3+4+10)/5 = 4.0;
    # kld is l1 / 10, so its sample-weighted mean is 0.4.
    assert set(metrics) == {"loss", "l1_loss", "kld_loss"}
    assert metrics["loss"] == 4.0
    assert metrics["l1_loss"] == 4.0
    assert metrics["kld_loss"] == pytest.approx(0.4)
    assert policy.training


def test_run_validation_logs_total_to_val_namespace():
    """`_run_validation` logs the total loss under the val/ namespace (the logger
    applies the prefix) and returns the scalar total for callers. Validation runs
    only periodically (every val_freq steps), not at step 0."""
    dataset = _ValidationDataset([1.0, 3.0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    policy = _LossPolicy()
    preprocessor = _IdentityPreprocessor()
    accelerator = _SingleProcessAccelerator()
    logger = _FakeValidationLogger()

    loss = _run_validation(
        policy,
        dataloader,
        preprocessor,
        accelerator,
        logger,
        step=10_000,
        is_main_process=True,
    )

    assert loss == 2.0
    assert logger.logged == [({"loss": 2.0}, 10_000, "val")]
    assert accelerator.wait_count == 1
    assert policy.forward_states == [(False, False)]


def test_run_validation_logs_loss_components():
    """Components are forwarded to the logger (logged as val/l1_loss, val/kld_loss
    once WandBLogger applies its mode prefix), alongside the total."""
    dataset = _ValidationDataset([1.0, 3.0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    policy = _LossPolicy(loss_dict=True)
    preprocessor = _IdentityPreprocessor()
    accelerator = _SingleProcessAccelerator()
    logger = _FakeValidationLogger()

    loss = _run_validation(
        policy,
        dataloader,
        preprocessor,
        accelerator,
        logger,
        step=0,
        is_main_process=True,
    )

    # the total is still returned for any caller that needs a scalar
    assert loss == 2.0
    assert accelerator.wait_count == 1
    assert len(logger.logged) == 1
    data, logged_step, mode = logger.logged[0]
    assert (logged_step, mode) == (0, "val")
    assert data["loss"] == 2.0
    assert data["l1_loss"] == 2.0
    assert data["kld_loss"] == pytest.approx(0.2)


def test_validation_dataloader_is_deterministic():
    dataset = _ValidationDataset([3.0, 1.0, 2.0])
    dataloader = _make_offline_dataloader(
        dataset,
        SimpleNamespace(streaming=False),
        SimpleNamespace(),
        num_workers=0,
        batch_size=2,
        device=torch.device("cpu"),
        shuffle=False,
    )

    values = torch.cat([batch["value"] for batch in dataloader])

    torch.testing.assert_close(values, torch.tensor([3.0, 1.0, 2.0]))


def test_wandb_logger_supports_val_namespace():
    fake_wandb = _FakeWandb()
    logger = object.__new__(WandBLogger)
    logger._wandb = fake_wandb
    logger._wandb_custom_step_key = None

    logger.log_dict({"loss": 2.5}, step=1_000, mode="val")

    assert fake_wandb.logged == [({"val/loss": 2.5}, 1_000)]


def test_umi_validation_does_not_recompute_relative_stats(monkeypatch):
    import lerobot.datasets.factory as dataset_factory
    import lerobot.scripts.lerobot_train as train_module

    original_factory = dataset_factory.make_dataset
    original_train_factory = train_module.make_dataset
    module = importlib.import_module("examples.umi_relative_ee.train_relative_ee_processor")

    created_datasets = []
    recomputed_datasets = []

    def make_metadata(*args, **kwargs):
        return SimpleNamespace(fps=30, features={"action": {}})

    def make_dataset(*args, **kwargs):
        dataset = SimpleNamespace(
            num_episodes=1,
            num_frames=4,
            meta=SimpleNamespace(
                info={"features": {"action": {"shape": [7]}}},
                camera_keys=[],
                stats={"action": {"mean": torch.zeros(7)}},
            ),
        )
        created_datasets.append(dataset)
        return dataset

    monkeypatch.setattr(module, "LeRobotDatasetMetadata", make_metadata)
    monkeypatch.setattr(module, "LeRobotDataset", make_dataset)
    monkeypatch.setattr(
        module,
        "recompute_stats",
        lambda dataset, **kwargs: recomputed_datasets.append(dataset),
    )

    policy = SimpleNamespace(
        type="act",
        chunk_size=2,
        action_delta_indices=[0],
        observation_delta_indices=None,
        reward_delta_indices=None,
        use_relative_actions=True,
        derive_state_from_action=False,
        relative_exclude_joints=["gripper"],
    )
    cfg = SimpleNamespace(
        dataset=DatasetConfig(repo_id="train/repo"),
        policy=policy,
        tolerance_s=1e-4,
    )
    validation_config = DatasetConfig(repo_id="validation/repo")

    try:
        module._make_dataset_wrapper(cfg)
        module._make_dataset_wrapper(cfg, dataset_config=validation_config)
    finally:
        dataset_factory.make_dataset = original_factory
        train_module.make_dataset = original_train_factory

    assert recomputed_datasets == [created_datasets[0]]
    assert created_datasets[0].meta.info["features"]["action"]["shape"] == [10]
    assert created_datasets[1].meta.info["features"]["action"]["shape"] == [7]


def test_act_vae_forward_supports_eval_validation():
    config = ACTConfig(
        chunk_size=3,
        n_action_steps=3,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        pretrained_backbone_weights=None,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    policy = ACTPolicy(config)
    policy.eval()
    batch = {
        OBS_STATE: torch.randn(2, 5),
        OBS_ENV_STATE: torch.randn(2, 3),
        ACTION: torch.randn(2, 3, 4),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }

    with torch.no_grad():
        loss, loss_dict = policy.forward(batch)

    assert torch.isfinite(loss)
    assert loss_dict.keys() == {"l1_loss", "kld_loss"}
    assert not policy.training

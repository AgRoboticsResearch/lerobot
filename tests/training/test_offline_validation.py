import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import DatasetConfig
from lerobot.configs.parser import wrap
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import _make_offline_dataloader, _run_validation, validate_policy


class _ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, values: list[float]):
        self.values = values
        self.meta = SimpleNamespace(has_language_columns=False)

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
    def __init__(self, report_components: bool = False):
        super().__init__()
        self.report_components = report_components
        self.forward_states: list[tuple[bool, bool]] = []

    def forward(self, batch):
        self.forward_states.append((self.training, torch.is_grad_enabled()))
        loss = batch["value"].mean()
        components = {"component": float(loss) / 10} if self.report_components else {}
        return loss, components


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
    dataloader = torch.utils.data.DataLoader(
        _ValidationDataset([1.0, 2.0, 3.0, 4.0, 10.0]), batch_size=2, shuffle=False
    )
    policy = _LossPolicy(report_components=True)
    policy.train()
    preprocessor = _IdentityPreprocessor()

    metrics = validate_policy(policy, dataloader, preprocessor, _SingleProcessAccelerator())

    assert metrics["loss"] == 4.0
    assert metrics["component"] == pytest.approx(0.4)
    assert policy.training
    assert policy.forward_states == [(False, False), (False, False), (False, False)]
    assert preprocessor.reset_count == 2


def test_run_validation_logs_val_namespace():
    dataloader = torch.utils.data.DataLoader(
        _ValidationDataset([1.0, 3.0]), batch_size=2, shuffle=False
    )
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
        step=1_000,
        is_main_process=True,
    )

    assert loss == 2.0
    assert logger.logged == [({"loss": 2.0}, 1_000, "val")]
    assert accelerator.wait_count == 1


def test_validation_dataloader_is_deterministic():
    dataloader = _make_offline_dataloader(
        _ValidationDataset([3.0, 1.0, 2.0]),
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

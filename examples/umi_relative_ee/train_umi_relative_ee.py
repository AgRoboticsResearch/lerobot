#!/usr/bin/env python
"""Train ACT, SmolVLA, or π0.5 on raw absolute-7D UMI relative-EE data.

The standard dataset factory performs the shared 7D -> 10D action and derived
20D state setup whenever ``--policy.use_umi_relative_ee=true``. This entrypoint
is intentionally a thin alias around the normal LeRobot trainer.
"""

import lerobot.scripts.lerobot_train as train_module
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


@parser.wrap()
def train_umi_relative_ee(cfg: TrainPipelineConfig) -> None:
    train_module.train(cfg)


if __name__ == "__main__":
    train_umi_relative_ee()

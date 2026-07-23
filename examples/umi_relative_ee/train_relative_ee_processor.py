#!/usr/bin/env python
"""Compatibility launcher for ACT and SmolVLA UMI relative-EE training.

Use ``--policy.use_umi_relative_ee=true``. The shared v5 dataset factory and
policy processor factories now provide the transformation, statistics, and
validation behavior; no runtime monkey-patching is required.
"""

import lerobot.scripts.lerobot_train as train_module
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


@parser.wrap()
def train_with_relative_ee_processor(cfg: TrainPipelineConfig) -> None:
    train_module.train(cfg)


if __name__ == "__main__":
    train_with_relative_ee_processor()

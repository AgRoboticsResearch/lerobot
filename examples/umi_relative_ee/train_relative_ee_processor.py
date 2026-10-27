#!/usr/bin/env python
"""Compatibility launcher for ACT and SmolVLA UMI relative-EE training.

Use ``--policy.use_umi_relative_ee=true``. The example adapter loads generated
Reach AnyPoint chunks; ordinary datasets use the standard dataset factory.
Both paths reuse the existing UMI policy processors.
"""

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

if __package__:
    from .reach_anypoint_dataset import train_with_reach_anypoint
else:
    from reach_anypoint_dataset import train_with_reach_anypoint


@parser.wrap()
def train_with_relative_ee_processor(cfg: TrainPipelineConfig) -> None:
    train_with_reach_anypoint(cfg)


if __name__ == "__main__":
    train_with_relative_ee_processor()

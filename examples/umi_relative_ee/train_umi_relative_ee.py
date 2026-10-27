#!/usr/bin/env python
"""Train ACT, SmolVLA, or π0.5 on raw absolute-7D UMI relative-EE data.

The example adapter loads Reach AnyPoint planning labels when present. Ordinary
UMI datasets use the standard dataset factory and its 7D -> 10D action / 20D
state setup whenever ``--policy.use_umi_relative_ee=true``.
"""

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

if __package__:
    from .reach_anypoint_dataset import train_with_reach_anypoint
else:
    from reach_anypoint_dataset import train_with_reach_anypoint


@parser.wrap()
def train_umi_relative_ee(cfg: TrainPipelineConfig) -> None:
    train_with_reach_anypoint(cfg)


if __name__ == "__main__":
    train_umi_relative_ee()

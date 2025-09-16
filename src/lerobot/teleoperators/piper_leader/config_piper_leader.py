#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PIPERLeaderConfig(TeleoperatorConfig):
    # Piper uses CAN; expose interface name and ignore serial port
    port: str | None = None
    can_name: str = "can_leader"



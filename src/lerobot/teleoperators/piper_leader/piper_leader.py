#!/usr/bin/env python

import logging
import time

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_piper_leader import PIPERLeaderConfig
from ...robots.piper_follower.piper_bus import PIPERMotorsBus, PIPERMotorsBusConfig

logger = logging.getLogger(__name__)


class PIPERLeader(Teleoperator):
    """Piper leader teleoperator using Piper SDK to stream joint angles."""

    config_class = PIPERLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PIPERLeaderConfig):
        super().__init__(config)
        self.config = config

        bus_cfg = PIPERMotorsBusConfig(
            can_name=config.can_name,
            motors={
                "joint_1": (1, "agilex_piper"),
                "joint_2": (2, "agilex_piper"),
                "joint_3": (3, "agilex_piper"),
                "joint_4": (4, "agilex_piper"),
                "joint_5": (5, "agilex_piper"),
                "joint_6": (6, "agilex_piper"),
                "gripper": (7, "agilex_piper"),
            },
        )
        self.bus = PIPERMotorsBus(config=bus_cfg)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.bus.connect()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def setup_motors(self) -> None:
        return

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action_raw = self.bus.read()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        # action_raw already returns radians for joints and meters for gripper
        return {f"{motor}.pos": val for motor, val in action_raw.items()}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")



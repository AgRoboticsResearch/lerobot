#!/usr/bin/env python3

import argparse

from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive calibration for SO100 follower arm")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port, e.g., /dev/ttyACM0 or /dev/ttyUSB0")
    parser.add_argument("--robot-id", default="ik_so101", help="Calibration/profile id (filename stem)")
    parser.add_argument("--no-degrees", action="store_true", help="If set, do not use degrees mode on motors")
    args = parser.parse_args()

    cfg = SO100FollowerConfig(
        id=args.robot_id,
        port=args.port,
        cameras={},
        use_degrees=(not args.no_degrees),
    )
    robot = SO100Follower(cfg)

    # This opens the interactive, on-screen guided calibration
    robot.connect(calibrate=True)
    robot.disconnect()

    print(f"Calibration file (if saved): {robot.calibration_fpath}")


if __name__ == "__main__":
    main()



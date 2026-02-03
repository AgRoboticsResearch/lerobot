#!/usr/bin/env python

"""
Simple script to read and display SO101 robot joint positions.

Usage:
    python read_joints.py --port /dev/ttyACM0
"""

import argparse
import time

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# Motor names for SO101
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main():
    parser = argparse.ArgumentParser(description="Read SO101 robot joint positions")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO101 robot connection",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Read frequency (Hz)",
    )
    args = parser.parse_args()

    # Create robot config
    robot_config = SO101FollowerConfig(
        port=args.port,
        use_degrees=True,
        cameras={},
    )

    # Connect to robot
    robot = SO101Follower(robot_config)
    robot.connect(calibrate=False)
    print(f"Robot connected on {args.port}")

    # Disable torque for all motors (free move mode)
    print("Disabling torque for all motors (robot can be moved freely)...")
    robot.bus.disable_torque()
    print("Torque disabled - you can now manually move the robot\n")

    try:
        print("Reading joint positions (Ctrl+C to stop):")

        while True:
            obs = robot.get_observation()
            joints = [obs[f"{name}.pos"] for name in MOTOR_NAMES]

            # Print in simple key=value format
            print(", ".join([f"{name}={val:.2f}" for name, val in zip(MOTOR_NAMES, joints)]))

            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        robot.disconnect()
        print("Robot disconnected")


if __name__ == "__main__":
    main()

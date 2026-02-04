#!/usr/bin/env python

"""
Simple script to read and display SO101 robot joint positions.

Usage:
    python read_joints.py --robot_id oscar_so101_follower --robot_port /dev/ttyACM0
    python read_joints.py --robot_id oscar_so101_follower --robot_port /dev/ttyACM0 --display
"""

import argparse
import time

import rerun as rr

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

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
        "--robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO101 robot connection",
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default="so101",
        help="Robot ID for loading/saving calibration files",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Read frequency (Hz)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Enable rerun visualization",
    )
    args = parser.parse_args()

    # Initialize rerun if display is enabled
    if args.display:
        init_rerun(session_name="read_joints")

    # Create robot config
    robot_config = SO101FollowerConfig(
        id=args.robot_id,
        port=args.robot_port,
        use_degrees=True,
        cameras={},
    )

    # Connect to robot
    robot = SO101Follower(robot_config)
    robot.connect(calibrate=False)
    print(f"Robot connected on {args.robot_port}")

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

            # Log to rerun if display is enabled
            if args.display:
                log_rerun_data(observation=obs, action={})

            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        if args.display:
            rr.rerun_shutdown()
        robot.disconnect()
        print("Robot disconnected")


if __name__ == "__main__":
    main()

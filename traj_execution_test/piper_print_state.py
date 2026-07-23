#!/usr/bin/env python
"""Read and print Piper robot state without enabling motors."""

import argparse
import time

import numpy as np
from piper_sdk import C_PiperInterface_V2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--rate", type=float, default=0.5, help="Print interval in seconds (0=once)")
    args = parser.parse_args()

    piper = C_PiperInterface_V2(args.can_name)
    piper.ConnectPort()
    print(f"Connected to Piper on {args.can_name}")

    def print_state():
        # Joints
        jm = piper.GetArmJointMsgs().joint_state
        joints = [jm.joint_1, jm.joint_2, jm.joint_3, jm.joint_4, jm.joint_5, jm.joint_6]
        joints_deg = [j / 1000.0 for j in joints]

        # EE pose
        ep = piper.GetArmEndPoseMsgs().end_pose
        pos_mm = [ep.X_axis / 1000.0, ep.Y_axis / 1000.0, ep.Z_axis / 1000.0]
        euler_deg = [ep.RX_axis / 1000.0, ep.RY_axis / 1000.0, ep.RZ_axis / 1000.0]

        # Gripper
        gm = piper.GetArmGripperMsgs().gripper_state
        grip_mm = gm.grippers_angle / 1000.0

        print(f"\n--- Joints (deg): {np.round(joints_deg, 2)}")
        print(f"--- EE pos (mm):  {np.round(pos_mm, 2)}")
        print(f"--- EE rot (deg): {np.round(euler_deg, 2)}")
        print(f"--- Gripper (mm): {grip_mm:.3f}")

    if args.rate <= 0:
        print_state()
    else:
        print(f"Printing state every {args.rate}s. Ctrl+C to stop.")
        try:
            while True:
                print_state()
                time.sleep(args.rate)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()

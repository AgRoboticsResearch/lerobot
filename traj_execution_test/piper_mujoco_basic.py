#!/usr/bin/env python
"""Interactive Piper MuJoCo joint control with GUI sliders.

Uses MuJoCo's passive viewer which provides actuator sliders.
Prints joint positions to terminal when sliders change.

Usage:
    python piper_mujoco_basic.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MJCF_PATH = Path(__file__).parent / "piper_mujoco" / "piper_description.xml"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def main():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print("Piper MuJoCo Joint Control")
    print("Use the sliders in the viewer to move each joint.\n")

    header = "  ".join(f"{j:>8s}" for j in JOINT_NAMES)
    print(f"  {header}")
    print(f"  " + "  ".join(f"{'(deg)':>8s}" for _ in JOINT_NAMES))
    print("-" * 70)

    prev_ctrl = data.ctrl[:6].copy()
    step = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            if not np.allclose(data.ctrl[:6], prev_ctrl, atol=1e-4):
                qpos_deg = np.degrees(data.qpos[:6])
                line = "  ".join(f"{v:>8.2f}" for v in qpos_deg)
                print(f"  {step:4d}  {line}")
                prev_ctrl = data.ctrl[:6].copy()
                step += 1

            viewer.sync()


if __name__ == "__main__":
    main()

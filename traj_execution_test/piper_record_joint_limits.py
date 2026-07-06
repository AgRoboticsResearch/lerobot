#!/usr/bin/env python
"""Continuously track the real joint range of a Piper arm while you move it.

Shows every joint's live angle plus the running min/max seen so far. Move each
joint through its full mechanical range; the running extremes update in real
time. Press Enter ONCE when done to record and print the final ranges.

Motors are never enabled (same as piper_print_state.py), so the arm stays free
to move by hand. Make sure it is in drag/free mode first.

Usage:
    python piper_record_joint_limits.py
    python piper_record_joint_limits.py --gripper --out joint_limits.json
"""

import argparse
import contextlib
import json
import select
import sys
import termios
import time
import tty

import numpy as np
from piper_sdk import C_PiperInterface_V2


def read_joints_deg(piper) -> np.ndarray:
    """Return [joint1..joint6] in degrees (raw millideg / 1000)."""
    jm = piper.GetArmJointMsgs().joint_state
    raw = [jm.joint_1, jm.joint_2, jm.joint_3, jm.joint_4, jm.joint_5, jm.joint_6]
    return np.array(raw, dtype=float) / 1000.0


def read_gripper_mm(piper) -> float:
    return piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0


@contextlib.contextmanager
def cbreak_stdin():
    """Put stdin in cbreak mode (per-char, no echo) so a key can be polled."""
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def poll_key():
    """Return a single pressed key, or None if nothing was pressed (non-blocking)."""
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        return sys.stdin.read(1)
    return None


def draw(frame, prev_n):
    """Overwrite the previous frame in place; return how many lines were drawn."""
    if prev_n:
        sys.stdout.write(f"\033[{prev_n}A")  # move cursor up to first line
    for line in frame:
        sys.stdout.write("\r\033[K" + line + "\n")  # CR + clear-to-EOL + line
    sys.stdout.flush()
    return len(frame)


def render_frame(cur, mins, maxs, with_gripper, g_cur, g_min, g_max):
    lines = [
        "[ Piper joint calibration — move every joint through its full range ]",
        f"{'joint':<8}{'current':>10}{'min':>10}{'max':>10}{'span(°)':>11}{'span(rad)':>12}",
    ]
    for i in range(6):
        span = maxs[i] - mins[i]
        lines.append(
            f"{'joint' + str(i + 1):<8}{cur[i]:>10.2f}{mins[i]:>10.2f}{maxs[i]:>10.2f}"
            f"{span:>11.2f}{np.deg2rad(span):>12.4f}"
        )
    if with_gripper:
        gspan = g_max - g_min
        lines.append(f"{'gripper':<8}{g_cur:>9.2f}mm{g_min:>8.2f}{g_max:>8.2f}{gspan:>9.2f}mm")
    lines.append("─" * 64)
    lines.append("Enter = finish & record   ·   r = reset min/max   ·   Ctrl+C = abort")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Track real Piper joint range by manual movement")
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--gripper", action="store_true", help="Also track the gripper open/close range")
    parser.add_argument("--rate", type=float, default=0.05, help="Display / sampling interval (s)")
    parser.add_argument("--out", type=str, default=None, help="Write final ranges to this JSON file")
    args = parser.parse_args()

    piper = C_PiperInterface_V2(args.can_name)
    piper.ConnectPort()
    print(f"Connected to Piper on {args.can_name}")
    print("Motors NOT enabled — move each joint by hand to its hard stops.")
    print("Sampling... move the arm now.\n")

    # Seed running min/max with the first reading so the table is never empty.
    cur = read_joints_deg(piper)
    mins = cur.copy()
    maxs = cur.copy()
    g_cur = g_min = g_max = read_gripper_mm(piper) if args.gripper else None

    prev_n = 0
    try:
        with cbreak_stdin():
            while True:
                cur = read_joints_deg(piper)
                mins = np.minimum(mins, cur)
                maxs = np.maximum(maxs, cur)
                if args.gripper:
                    g_cur = read_gripper_mm(piper)
                    g_min = min(g_min, g_cur)
                    g_max = max(g_max, g_cur)

                frame = render_frame(cur, mins, maxs, args.gripper, g_cur, g_min, g_max)
                prev_n = draw(frame, prev_n)

                key = poll_key()
                if key in ("\n", "\r"):
                    break
                if key == "r":  # reset running extremes from current pose
                    mins = cur.copy()
                    maxs = cur.copy()
                    if args.gripper:
                        g_min = g_max = g_cur

                time.sleep(args.rate)
    except KeyboardInterrupt:
        print("\nAborted — nothing recorded.")
        return

    # Final summary
    print("\n" + "=" * 64)
    print("RECORDED REAL JOINT RANGES")
    print("=" * 64)
    print(f"{'joint':<8}{'min (deg)':>12}{'max (deg)':>12}{'min (rad)':>12}{'max (rad)':>12}")
    joints = {}
    for i in range(6):
        lo, hi = float(mins[i]), float(maxs[i])
        joints[f"joint{i + 1}"] = {
            "min_deg": lo, "max_deg": hi,
            "min_rad": float(np.deg2rad(lo)), "max_rad": float(np.deg2rad(hi)),
        }
        print(f"joint{i + 1:<2} {lo:>11.2f} {hi:>11.2f} {np.deg2rad(lo):>11.4f} {np.deg2rad(hi):>11.4f}")
    result = {"joints": joints}
    if args.gripper:
        result["gripper"] = {"open_mm": float(min(g_min, g_max)), "closed_mm": float(max(g_min, g_max))}
        print(f"\ngripper: open={result['gripper']['open_mm']:.2f} mm, "
              f"closed={result['gripper']['closed_mm']:.2f} mm")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.out}")
    print("\nDone.")


if __name__ == "__main__":
    main()

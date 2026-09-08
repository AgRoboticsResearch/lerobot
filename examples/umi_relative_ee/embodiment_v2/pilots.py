"""Run bounded pilot comparisons with durable status and a GPU lease per training run."""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

from .report import report
from .train import PilotConfig, prepare
from .watch import atomic_json, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data1/projects/lerobot-embodiment-v2"))
    parser.add_argument("--robot", choices=("piper", "so101"), default="piper")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-freq", type=int, default=2000)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "pilots.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {
            "state": "running",
            "started_utc": utc(),
            "robot": args.robot,
            "steps": args.steps,
            "completed": [],
        }
        status_path = args.root / "pilot-status.json"
        atomic_json(status_path, status)
        try:
            for representation in ("q_delta", "command_delta"):
                status.update(active=representation, phase="prepare")
                atomic_json(status_path, status)
                prepare(args.root, PilotConfig(robot=args.robot, representation=representation))
                status["phase"] = "train_and_development"
                atomic_json(status_path, status)
                command = [
                    sys.executable,
                    "-m",
                    "examples.umi_relative_ee.embodiment_v2.watch",
                    "exclusive",
                    "--root",
                    str(args.root),
                    "--",
                    sys.executable,
                    "-m",
                    "examples.umi_relative_ee.embodiment_v2.train",
                    "--root",
                    str(args.root),
                    "--robot",
                    args.robot,
                    "--representation",
                    representation,
                    "--steps",
                    str(args.steps),
                    "--eval-freq",
                    str(args.eval_freq),
                ]
                subprocess.run(command, check=True)
                status["completed"].append(representation)
                atomic_json(status_path, status)
                report(args.root)
            status.update(state="complete", phase="complete", active=None)
        except BaseException as exc:
            status.update(state="failed", error=repr(exc))
            raise
        finally:
            status["finished_utc"] = utc()
            atomic_json(status_path, status)


if __name__ == "__main__":
    main()

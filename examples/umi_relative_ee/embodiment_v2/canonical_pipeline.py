"""Collect and train the bounded-command ablation while preserving all earlier runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from examples.umi_relative_ee.task_independent_embodiment.core import digest, manifest

from . import train as training
from .canonical import collect
from .report import report
from .watch import atomic_json, utc


def overlay(root, robot):
    """Explicit dataset view: new train executions, original held-out development and baselines."""
    original = training.ROOTS[robot]
    view = root / "bounded_commands" / "dataset_views" / robot
    evidence = {
        "original_root": str(original),
        "original_preflight": digest(original / "preflight.json"),
        "bounded_collection": digest(root / f"canonical/{robot}/native/train_all/manifest.json"),
        "code": digest(Path(__file__)),
    }
    manifest(view / "preflight.json", evidence, resume=True)
    links = {
        "config.json": original / "config.json",
        "train": original / "train",  # read-only original checkpoints for matched baselines
        "collections/native/D0/dev": original / "collections/native/D0/dev",
        "collections/native/D0/train": root / f"canonical/{robot}/native/train_all",
    }
    for name, target in links.items():
        link = view / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"Dataset view target changed: {link}")
        elif link.exists():
            raise ValueError(f"Expected dataset-view symlink: {link}")
        else:
            link.symlink_to(target)
    return view


def prepare(root, robot):
    collect(root, robot, "native", "train", 0)
    view = overlay(root, robot)
    core, learning, _ = training.modules(robot)
    cfg = core.Config.read(view / "config.json")
    print(f"Preparing hindsight windows for {robot} bounded commands", flush=True)
    learning.prepare(
        view / "prepared/native/C0/hindsight", [view / "collections/native/D0/train"], "hindsight", cfg
    )
    training.ROOTS[robot] = view
    training.prepare(root / "bounded_commands", training.PilotConfig(robot=robot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "train"))
    parser.add_argument("--root", type=Path, default=Path("/mnt/data1/projects/lerobot-embodiment-v2"))
    parser.add_argument("--robot", choices=training.ROOTS, default="piper")
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()
    if args.stage == "train":
        training.ROOTS[args.robot] = overlay(args.root, args.robot)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        training.train(
            args.root / "bounded_commands",
            training.PilotConfig(robot=args.robot, steps=args.steps, eval_freq=2000),
            "cuda",
        )
        report(args.root / "bounded_commands")
        return
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "bounded-pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {
            "state": "running",
            "phase": "replay_and_prepare",
            "robot": args.robot,
            "steps": args.steps,
            "pid": os.getpid(),
            "started_utc": utc(),
        }
        path = args.root / "bounded-status.json"
        atomic_json(path, status)
        try:
            prepare(args.root, args.robot)
            status["phase"] = "train_and_development"
            atomic_json(path, status)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "examples.umi_relative_ee.embodiment_v2.watch",
                    "exclusive",
                    "--root",
                    str(args.root),
                    "--",
                    sys.executable,
                    "-m",
                    "examples.umi_relative_ee.embodiment_v2.canonical_pipeline",
                    "train",
                    "--root",
                    str(args.root),
                    "--robot",
                    args.robot,
                    "--steps",
                    str(args.steps),
                ],
                check=True,
            )
            status.update(state="complete", phase="complete")
        except BaseException as exc:
            status.update(state="failed", error=repr(exc))
            raise
        finally:
            status["finished_utc"] = utc()
            atomic_json(path, status)
            print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()

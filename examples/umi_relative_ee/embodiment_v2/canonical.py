"""Replay bounded bootstrap commands and record newly executed inverse-model labels.

This is an offline collection ablation, not a deployable open-loop policy.
A rate-limited servo can receive many different goals that produce the same motion.
Replaying a minimally bounded goal makes the issued command less ambiguous while
retaining the original UMI-shaped trajectory and all physical observations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.umi_relative_ee.task_independent_embodiment.core import digest, manifest, save_npz, write_json

from .train import ROOTS, modules


def bounded_commands(commands, initial, max_step):
    previous = initial.copy()
    result = []
    for command in commands:
        previous = previous + np.clip(command - previous, -max_step, max_step)
        result.append(previous.copy())
    return np.asarray(result)


def replay(source, sim):
    """Issue every new command to physics; never silently replace recorded labels."""
    commands = bounded_commands(
        source["issued"], source["start"], np.deg2rad(sim.cfg.max_joint_vel_deg_s) / sim.cfg.control_hz
    )
    sim.reset(source["start"])
    states = [sim.state()]
    gripper_key = "gripper_actual_rad" if "gripper_actual_rad" in source else "gripper_actual_norm"
    gripper_measurement = (
        sim.core.gripper_pos_rad if gripper_key.endswith("rad") else sim.core.gripper_pos_norm
    )
    issued, delivered, clamped, invalid, grips, latency = ([] for _ in range(6))
    for index, command in enumerate(commands):
        grip = np.interp(source["time"][index], source["query_time"], source["gripper_query"])
        began = time.perf_counter()
        u, d, c, bad = sim.advance(command, grip)
        latency.append(time.perf_counter() - began)
        issued.append(u)
        delivered.append(d)
        clamped.append(c)
        invalid.append(bad)
        grips.append(gripper_measurement())
        states.append(sim.state())
    actual = np.asarray([s[3] for s in states])
    q = np.asarray([s[1] for s in states])
    qd = np.asarray([s[2] for s in states])
    delta = {
        "max_q_difference_rad": float(np.max(np.abs(q - source["q"]))),
        "max_qd_difference_rad_s": float(np.max(np.abs(qd - source["qd"]))),
        "max_pose_matrix_difference": float(np.max(np.abs(actual - source["actual"]))),
        "max_issued_change_rad": float(np.max(np.abs(commands - source["issued"]))),
    }
    # Equivalence is checked, not assumed. Stop the ablation if dynamics differ.
    if delta["max_q_difference_rad"] > 1e-5 or delta["max_pose_matrix_difference"] > 1e-5:
        raise ValueError(f"Bounded replay changed the physical trajectory: {delta}")
    result = dict(source)
    result.update(
        time=np.asarray([s[0] for s in states]),
        q=q,
        qd=qd,
        actual=actual,
        issued=np.asarray(issued),
        delivered=np.asarray(delivered),
        clamped=np.asarray(clamped),
        invalid=np.asarray(invalid),
        errors=np.full(len(commands), ""),
        latency_s=np.asarray(latency),
        encoder_fk=np.asarray([sim.ik.fk(s[1]) for s in states]),
    )
    result[gripper_key] = np.asarray(grips)
    return result, delta


def compatible_completed_piper(directory, evidence):
    """Read completed pre-SO101 Piper data without changing its generator provenance.

    Only this known generator and identical inputs are supported. Incomplete
    collections still require exact source identity, preventing mixed generators.
    """
    path = directory / "manifest.json"
    if evidence["robot"] != "piper" or not path.exists() or not (directory / "complete.json").exists():
        return False
    old = json.loads(path.read_text())
    return old.get("code") == "78fa59984bf7acfb23f058547b57740ffebd59b132f01c07b3a028cb22365ffe" and {
        k: v for k, v in old.items() if k != "code"
    } == {k: v for k, v in evidence.items() if k != "code"}


def collect(root, robot, condition, split, limit):
    core, _, simulation = modules(robot)
    cfg = core.Config.read(ROOTS[robot] / "config.json")
    source = ROOTS[robot] / f"collections/{condition}/D0/{split}"
    if not (source / "complete.json").exists():
        raise ValueError(f"Source collection is incomplete: {source}")
    files = sorted((source / "rollouts").glob("*.npz"))
    files = files[:limit] if limit else files
    directory = root / f"canonical/{robot}/{condition}/{split}_{limit or 'all'}"
    evidence = {
        "robot": robot,
        "condition": condition,
        "split": split,
        "source_manifest": digest(source / "manifest.json"),
        "sources": {p.name: digest(p) for p in files},
        "code": digest(Path(__file__)),
        "legacy_preflight": digest(ROOTS[robot] / "preflight.json"),
    }
    if not compatible_completed_piper(directory, evidence):
        manifest(directory / "manifest.json", evidence, resume=True)
    if (directory / "complete.json").exists():
        return json.loads((directory / "complete.json").read_text())
    (directory / "rollouts").mkdir(exist_ok=True)
    sim = simulation.Simulation(cfg, condition)
    rows = []
    for index, file in enumerate(files):
        out = directory / "rollouts" / file.name
        metadata = out.with_suffix(".json")
        if out.exists() and metadata.exists():
            row = json.loads(metadata.read_text())
            if row["sha256"] != digest(out):
                raise ValueError(f"Corrupt replay: {out}")
        else:
            with np.load(file) as loaded:
                original = dict(loaded)
            result, delta = replay(original, sim)
            save_npz(out, **result)
            row = {
                **json.loads(file.with_suffix(".json").read_text()),
                **delta,
                "source_sha256": digest(file),
                "sha256": digest(out),
                "replayed_bounded_commands": True,
            }
            write_json(metadata, row)
        rows.append(row)
        if index % 100 == 0:
            print(f"canonical {robot}/{condition}/{split}: {index + 1}/{len(files)}", flush=True)
    summary = {
        "rollouts": len(rows),
        "max_q_difference_rad": max(r["max_q_difference_rad"] for r in rows),
        "max_pose_matrix_difference": max(r["max_pose_matrix_difference"] for r in rows),
        "mean_max_issued_change_deg": float(np.rad2deg(np.mean([r["max_issued_change_rad"] for r in rows]))),
    }
    write_json(directory / "complete.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data1/projects/lerobot-embodiment-v2"))
    parser.add_argument("--robot", choices=ROOTS, default="piper")
    parser.add_argument("--condition", choices=("native", "delay40"), default="native")
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--limit", type=int, default=64, help="0 replays all rollouts")
    args = parser.parse_args()
    collect(args.root, args.robot, args.condition, args.split, args.limit)


if __name__ == "__main__":
    main()

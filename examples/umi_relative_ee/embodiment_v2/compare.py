"""Matched closed-loop development comparison with per-rollout feasibility and paired uncertainty."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from examples.umi_relative_ee.task_independent_embodiment.core import digest, manifest, write_json

from .train import (
    ROOTS,
    Controller,
    DirectControllerModel,
    HoldController,
    PilotConfig,
    development_files,
    evaluate,
    modules,
)


class RuntimeIK:
    method = "ik"

    def __call__(self, trajectory, valid, state, past, first_pose, ik):
        return ik.command(first_pose, state[-1, : past.shape[-1]])[0]


def paired_interval(candidate, baseline, episode_ids):
    """Descriptive paired cluster bootstrap; development selection makes this exploratory."""
    difference = np.asarray(candidate) - np.asarray(baseline)
    ids = np.asarray(episode_ids)
    unique = np.unique(ids)
    grouped = [difference[ids == episode] for episode in unique]
    rng = np.random.default_rng(5317)
    samples = []
    for _ in range(2000):
        chosen = rng.integers(len(grouped), size=len(grouped))
        samples.append(float(np.concatenate([grouped[i] for i in chosen]).mean()))
    return {
        "mean_difference_m": float(difference.mean()),
        "cluster_bootstrap_95_m": np.quantile(samples, [0.025, 0.975]).tolist(),
        "source_episode_clusters": len(unique),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data1/projects/lerobot-embodiment-v2"))
    parser.add_argument("--robot", choices=ROOTS, default="piper")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    core, learning, simulation = modules(args.robot)
    legacy = core.Config.read(ROOTS[args.robot] / "config.json")
    files = development_files(PilotConfig(robot=args.robot))
    name = f"command_delta_hindsight_seed1000_{args.steps}steps"
    checkpoints = {
        "v1_hindsight": ROOTS[args.robot] / "train/native/C0/hindsight_seed1000/best.pt",
        "v1_residual": ROOTS[args.robot] / "train/native/C0/residual_seed1000/best.pt",
        "raw_commands": args.root / f"train/{args.robot}/native/{name}/best.pt",
        "bounded_commands": args.root / f"bounded_commands/train/{args.robot}/native/{name}/best.pt",
    }
    out = args.root / f"comparison/{args.robot}_{args.steps}steps"
    manifest(
        out / "manifest.json",
        {
            "checkpoints": {k: digest(v) for k, v in checkpoints.items()},
            "development": {p.name: digest(p) for p in files},
            "code": digest(Path(__file__)),
        },
        resume=True,
    )
    controllers = {"hold_command": HoldController(), "runtime_ik": RuntimeIK()}
    for label, path in checkpoints.items():
        if label.startswith("v1"):
            controller = learning.LearnedController(path, args.device)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["provenance"]["code"] != digest(Path(__file__).with_name("train.py")):
                raise ValueError(f"Training code changed since checkpoint creation: {path}")
            cfg = PilotConfig(**payload["config"])
            model = DirectControllerModel(cfg, payload["joints"]).to(args.device).eval()
            model.load_state_dict(payload["model"])
            controller = Controller(model, cfg.representation, args.device)
        controllers[label] = controller
    sim = simulation.Simulation(legacy, "native")
    reports = {}
    for name, controller in controllers.items():
        controller.robot = args.robot
        path = out / f"{name}.json"
        if not path.exists():
            evaluate(controller, files, sim, legacy, path)
        reports[name] = json.loads(path.read_text())
    episode_ids = []
    for file in files:
        with np.load(file) as roll:
            episode_ids.append(int(roll["source_episode"]))
    paired = {
        name: paired_interval(
            [r["position_rmse_m"] for r in reports["bounded_commands"]["trials"]],
            [r["position_rmse_m"] for r in reports[name]["trials"]],
            episode_ids,
        )
        for name in ("raw_commands", "hold_command", "v1_residual")
    }
    lines = [
        "# Matched development comparison",
        "",
        "64 fixed embodiment-development rollouts; exploratory checkpoint selection, no task validation.",
        "",
        "| Controller | Position (mm) | Rotation (deg) | Failures | Feasible position (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in reports.items():
        metrics = result["means"]
        feasible = [r["position_rmse_m"] for r in result["trials"] if r["ik_feasible"]]
        feasible_mean = f"{np.mean(feasible) * 1000:.2f}" if feasible else "n/a"
        lines.append(
            f"| {name} | {metrics['position_rmse_m'] * 1000:.2f} | "
            f"{metrics['rotation_mean_deg']:.2f} | {metrics['tracking_failed']:.1%} | {feasible_mean} |"
        )
    lines += [
        "",
        "Paired position differences: bounded commands minus comparator. "
        "Negative favors bounded commands. Intervals resample source-episode clusters; "
        "they are descriptive and do not correct for development checkpoint selection.",
        "",
    ]
    for name, values in paired.items():
        low, high = np.asarray(values["cluster_bootstrap_95_m"]) * 1000
        lines.append(
            f"- {name}: {values['mean_difference_m'] * 1000:.2f} mm; 95% interval [{low:.2f}, {high:.2f}]."
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(out / "complete.json", {"paired": paired, "report": str(out / "REPORT.md")})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

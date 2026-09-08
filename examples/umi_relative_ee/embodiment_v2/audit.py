"""Reproduce inverse-label and rate-limit diagnostics using original training/development data."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from examples.umi_relative_ee.task_independent_embodiment.core import digest, write_json


def inverse_audit():
    output = {}
    for robot, namespace, directory in [
        ("piper", "task_independent_embodiment", "lerobot-embodiment-exp"),
        ("so101", "so101_task_independent_embodiment", "lerobot-embodiment-exp-so101"),
    ]:
        core = importlib.import_module(f"examples.umi_relative_ee.{namespace}.core")
        learning = importlib.import_module(f"examples.umi_relative_ee.{namespace}.learning")
        root = Path("/mnt/data1/projects") / directory
        cfg = core.Config.read(root / "config.json")
        prepared = root / "prepared/native/C0/hindsight"
        model = learning.LearnedController(root / "train/native/C0/hindsight_seed1000/best.pt", "cpu").model
        n = 6 if robot == "piper" else 5
        rng = np.random.default_rng(1000)
        arrays = {p.stem: np.load(p, mmap_mode="r") for p in prepared.glob("*.npy")}
        ids = rng.integers(len(arrays["target"]), size=512)
        datasets = {"train": {k: np.asarray(v[ids]) for k, v in arrays.items()}}
        windows = []
        for path in sorted((root / "collections/native/D0/dev/rollouts").glob("*.npz"))[:32]:
            with np.load(path) as raw:
                roll = dict(raw)
            for idx in np.linspace(cfg.history - 1, len(roll["issued"]) - 1, 16, dtype=int):
                windows.append(core.learning_window(roll, idx, cfg, True))
        datasets["dev"] = {k: np.stack([w[k] for w in windows]) for k in windows[0]}
        output[robot] = {}
        for split, data in datasets.items():
            predictions = []
            for begin in range(0, len(data["target"]), 64):
                batch = {k: torch.from_numpy(v[begin : begin + 64]) for k, v in data.items()}
                with torch.inference_mode():
                    predictions.append(model.predict(batch).numpy())
            pred = np.concatenate(predictions)
            truth = data["target"][:, 0]
            q = data["state"][:, -1, :n]
            past = data["past_commands"][:, -1]
            output[robot][split] = {
                "network_mae_deg": float(np.rad2deg(np.abs(pred[:, 0] - truth)).mean()),
                "hold_current_q_mae_deg": float(np.rad2deg(np.abs(q - truth)).mean()),
                "repeat_last_command_mae_deg": float(np.rad2deg(np.abs(past - truth)).mean()),
                "network_mae_per_joint_deg": np.rad2deg(np.abs(pred[:, 0] - truth)).mean(0).tolist(),
                "true_delta_q_mae_per_joint_deg": np.rad2deg(np.abs(truth - q)).mean(0).tolist(),
                "predicted_delta_q_mae_per_joint_deg": np.rad2deg(np.abs(pred[:, 0] - q)).mean(0).tolist(),
                "first_pose_valid_fraction": float(data["trajectory_valid"][:, 0].mean()),
            }
        print(robot, json.dumps(output[robot]), flush=True)
    return output


def saturation_audit():
    roots = {
        "piper": Path("/mnt/data1/projects/lerobot-embodiment-exp"),
        "so101": Path("/mnt/data1/projects/lerobot-embodiment-exp-so101"),
    }
    output = {}
    for robot, root in roots.items():
        cfg = json.loads((root / "config.json").read_text())
        vmax = np.deg2rad(cfg["max_joint_vel_deg_s"])
        files = sorted((root / "collections/native/D0/dev/rollouts").glob("*.npz"))
        entries = []
        for f in files:
            meta = json.loads(f.with_suffix(".json").read_text())
            with np.load(f) as r:
                internal = r["start"].copy()
                limited = []
                beyond_horizon = []
                margins = []
                for index, command in enumerate(r["delivered"]):
                    gap = np.abs(command - internal)
                    remaining = r["time"][-1] - r["time"][index]
                    if index >= 10:
                        limited.append(bool(np.any(gap > vmax / 50 + 1e-6)))
                        beyond_horizon.append(bool(np.any(gap > vmax * min(29 / 30, remaining) + 1e-6)))
                        margins.append(np.rad2deg(gap).max())
                    internal += np.clip(command - internal, -vmax / 50, vmax / 50)
                entries.append(
                    {
                        "file": f.name,
                        "ik_feasible": meta["ik_feasible"],
                        "rate_limited_command_fraction": float(np.mean(limited)),
                        "goal_beyond_remaining_horizon_fraction": float(np.mean(beyond_horizon)),
                        "mean_max_internal_goal_gap_deg": float(np.mean(margins)),
                        "joint_clamp_fraction": float(r["clamped"][10:].mean()),
                    }
                )
        means = {k: float(np.mean([row[k] for row in entries])) for k in entries[0] if k != "file"}
        output[robot] = {"rollouts": len(entries), "means": means, "trials": entries}
        print(robot, json.dumps(means), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data1/projects/lerobot-embodiment-v2"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1000)
    write_json(args.root / "inverse-teacher-forcing-audit.json", inverse_audit())
    write_json(args.root / "saturation-audit.json", saturation_audit())
    evidence = {"audit_code": digest(Path(__file__))}
    for name, directory in [("piper", "lerobot-embodiment-exp"), ("so101", "lerobot-embodiment-exp-so101")]:
        source = Path("/mnt/data1/projects") / directory
        evidence[name] = {
            "preflight": digest(source / "preflight.json"),
            "checkpoint": digest(source / "train/native/C0/hindsight_seed1000/best.pt"),
            "development_manifest": digest(source / "collections/native/D0/dev/manifest.json"),
        }
    write_json(args.root / "audit-provenance.json", evidence)


if __name__ == "__main__":
    main()

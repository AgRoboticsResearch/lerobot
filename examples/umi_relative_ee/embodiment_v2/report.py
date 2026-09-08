"""Summarize development pilots without loading checkpoints or touching validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def report(root: Path):
    lines = [
        "# Embodiment controller development pilots",
        "",
        "Exploratory model selection on embodiment development motions only. "
        "These are not ACT validation results or manipulation success rates.",
        "",
        "All rows within a robot/condition use the same deterministic development subset. "
        "V1 checkpoints have 30,000 updates; pilot update budgets appear in the names. "
        "Best pilot checkpoints are selected by development position RMSE.",
        "",
    ]
    rows = []
    for baseline in sorted(root.glob("development_baselines/*/*/*.json")):
        robot, condition = baseline.parts[-3:-1]
        data = json.loads(baseline.read_text())
        count = len(data["files"])
        group = []
        for method, metrics in data["means"].items():
            group.append(
                {
                    "robot": robot,
                    "condition": condition,
                    "method": method,
                    "rollouts": count,
                    "metrics": metrics,
                }
            )
        for run in sorted((root / "train" / robot / condition).glob("*")):
            manifest_path = run / "manifest.json"
            if not manifest_path.exists():
                continue
            evidence = json.loads(manifest_path.read_text())
            if sorted(evidence["development"]) != sorted(data["files"]):
                continue
            scores = []
            for path in sorted((run / "development").glob("*.json")):
                evaluation = json.loads(path.read_text())
                scores.append((evaluation["means"]["position_rmse_m"], path, evaluation))
            if not scores:
                continue
            _, path, selected = min(scores, key=lambda x: x[0])
            metrics = selected["means"]
            trials = selected["trials"]
            feasible = [r for r in trials if r["ik_feasible"]]
            group.append(
                {
                    "robot": robot,
                    "condition": condition,
                    "method": run.name,
                    "rollouts": count,
                    "metrics": metrics,
                    "step": int(path.stem),
                    "complete": (run / "complete.json").exists(),
                    "beats_hold": metrics["position_rmse_m"]
                    < data["means"]["hold_command"]["position_rmse_m"],
                    "feasible_count": len(feasible),
                    "feasible_position_rmse_m": sum(r["position_rmse_m"] for r in feasible) / len(feasible)
                    if feasible
                    else None,
                }
            )
        lines += [
            f"## {robot} / {condition} / {count} development rollouts",
            "",
            "| Method | Best step | Position RMSE (mm) | Rotation (deg) | Tracking failures | Clamp rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in group:
            m = row["metrics"]
            label = row["method"] + (" (running)" if row.get("complete") is False else "")
            lines.append(
                f"| {label} | {row.get('step', '—')} | {m['position_rmse_m'] * 1000:.2f} | "
                f"{m['rotation_mean_deg']:.2f} | {m['tracking_failed']:.1%} | {m['joint_limit_rate']:.1%} |"
            )
        lines += [
            "",
            "Beating the hold-command baseline is necessary but insufficient for promotion. "
            "Compare orientation, feasibility strata and the IK/residual baseline before scaling.",
            "",
        ]
        rows.extend(group)
    (root / "PILOTS.md").write_text("\n".join(lines) + "\n")
    (root / "pilot-results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data1/projects/lerobot-embodiment-v2"))
    args = parser.parse_args()
    report(args.root)


if __name__ == "__main__":
    main()

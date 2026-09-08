"""Separate feasible and infeasible results without modifying running experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .watch import SWEEPS, atomic_json, utc

METRICS = ("position_rmse_m", "rotation_mean_deg", "tracking_failed", "joint_limit_rate", "execution_error")


def summarize(rows):
    return {
        "trials": len(rows),
        **{key: float(np.mean([r[key] for r in rows])) if rows else None for key in METRICS},
    }


def partition(rows):
    return {
        name: summarize([r for r in rows if bool(r["ik_feasible"]) == feasible])
        for name, feasible in [("feasible", True), ("infeasible", False)]
    }


def reasons(row, speed_limit):
    result = []
    for name, failed in [
        ("workspace", not row["workspace_valid"]),
        ("joint_limits", not row["joint_limits_valid"]),
        ("IK_error", row["ik_error"] is not None),
        ("position_residual", row["max_position_residual_m"] > 0.005),
        ("orientation_residual", row["max_rotation_residual_deg"] > 3),
        ("nominal_velocity", row["max_nominal_velocity_deg_s"] > speed_limit),
    ]:
        if failed:
            result.append(name)
    if bool(result) == bool(row["ik_feasible"]):
        raise ValueError("Stored feasibility differs from the documented screen")
    return result


def metric_row(label, name, values):
    if not values["trials"]:
        return f"| {label} | {name} | 0 | n/a | n/a | n/a | n/a |"
    return (
        f"| {label} | {name} | {values['trials']} | {values['position_rmse_m'] * 1000:.2f} | "
        f"{values['rotation_mean_deg']:.2f} | {values['tracking_failed']:.1%} | {values['joint_limit_rate']:.1%} |"
    )


def infeasible_category(row, speed_limit):
    why = set(reasons(row, speed_limit))
    if not why:
        return None
    if why == {"nominal_velocity"}:
        return "velocity_only"
    if "workspace" in why:
        return "outside_workspace"
    return "other_kinematic"


def table_header():
    return [
        "| Method / seed | Subset | Trials | Position RMSE (mm) | Rotation (deg) | Failures | Clamp rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]


def report(root):
    output = root / "stratified"
    output.mkdir(parents=True, exist_ok=True)
    summaries, failure_reasons, breakdowns, provenance = [], [], [], {}
    lines = [
        "# Feasible and infeasible trajectory analysis",
        "",
        f"Updated: {utc()}",
        "",
        "Feasible means passing the existing nominal sequential-IK screen: workspace and joint limits, "
        "no IK exception, position residual ≤5 mm, orientation residual ≤3°, and nominal joint speed "
        "within the configured limit. Infeasible means failing this screen; it is not a proof that "
        "no posture or controller could execute the trajectory.",
        "",
        "Each subset has its own denominator. Metrics are means over trials within that subset; "
        "failures are the existing >50 mm / >15° or execution-error criterion. "
        "Only completed evaluations are included. Training seeds are shown separately. "
        "ACT and recorded ground-truth targets are never pooled.",
        "",
    ]
    for robot, source in SWEEPS.items():
        cfg = json.loads((source / "config.json").read_text())
        grouped = defaultdict(list)
        screening = defaultdict(dict)
        for path in sorted(source.glob("eval/*/*/*/results.json")):
            if not (path.parent / "complete.json").exists():
                continue
            provenance[str(path)] = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for row in json.loads(path.read_text()):
                key = (row["condition"], row["target"], row["generation"], row["method"], row["seed"])
                grouped[key].append(row)
                screen_key = (row["condition"], row["target"])
                previous = screening[screen_key].get(row["trial"])
                if previous is not None and previous["ik_feasible"] != row["ik_feasible"]:
                    raise ValueError(f"Method-dependent feasibility: {robot}/{row['trial']}")
                screening[screen_key][row["trial"]] = row
        for condition, target in sorted(screening):
            population = list(screening[(condition, target)].values())
            failed = [r for r in population if not r["ik_feasible"]]
            lines += [
                f"## {robot} / {condition} / {target}",
                "",
                f"Fixed queries: **{len(population) - len(failed)} feasible**, "
                f"**{len(failed)} infeasible**, {len(population)} total.",
                "",
            ]
            lines += table_header()
            for key, rows in sorted(grouped.items()):
                if key[:2] != (condition, target):
                    continue
                _, _, generation, method, seed = key
                label = f"{generation}/{method} / {seed}"
                for name, values in partition(rows).items():
                    summaries.append(
                        {
                            "robot": robot,
                            "condition": condition,
                            "target": target,
                            "generation": generation,
                            "method": method,
                            "seed": seed,
                            "subset": name,
                            **values,
                        }
                    )
                    lines.append(metric_row(label, name, values))
                for category in ("velocity_only", "outside_workspace", "other_kinematic"):
                    included = [
                        r for r in rows if infeasible_category(r, cfg["max_joint_vel_deg_s"]) == category
                    ]
                    breakdowns.append(
                        {
                            "robot": robot,
                            "condition": condition,
                            "target": target,
                            "generation": generation,
                            "method": method,
                            "seed": seed,
                            "category": category,
                            **summarize(included),
                        }
                    )
            counts, combinations = Counter(), Counter()
            for row in failed:
                why = reasons(row, cfg["max_joint_vel_deg_s"])
                counts.update(why)
                combinations[" + ".join(why)] += 1
            failure_reasons.append(
                {
                    "robot": robot,
                    "condition": condition,
                    "target": target,
                    "infeasible_trials": len(failed),
                    "overlapping_reasons": dict(counts),
                    "exclusive_combinations": dict(combinations),
                }
            )
            lines += [
                "",
                "Screen failures below count each query once across methods/seeds. "
                "Individual reasons overlap; combinations are mutually exclusive.",
                "",
                "| Screen failure | Queries | Fraction of infeasible |",
                "|---|---:|---:|",
            ]
            for reason, count in counts.most_common():
                lines.append(f"| {reason} | {count} | {count / len(failed):.1%} |")
            lines += ["", "Most common combinations:", ""]
            for combination, count in combinations.most_common(6):
                lines.append(f"- {combination}: {count}/{len(failed)}")
            lines.append("")
    for path in sorted(root.glob("comparison/*/complete.json")):
        directory = path.parent
        robot = directory.name.split("_")[0]
        speed_limit = json.loads((SWEEPS[robot] / "config.json").read_text())["max_joint_vel_deg_s"]
        pilot_rows = {}
        lines += [
            f"## Development pilot / {directory.name}",
            "",
            "Exploratory development-selected checkpoints; separate from task validation above.",
            "",
        ]
        lines += table_header()
        for file in sorted(directory.glob("*.json")):
            if file.name in ("manifest.json", "complete.json"):
                continue
            data = json.loads(file.read_text())
            pilot_rows[file.stem] = data["trials"]
            for subset, values in partition(data["trials"]).items():
                summaries.append(
                    {
                        "robot": directory.name.split("_")[0],
                        "condition": "native",
                        "target": "embodiment_development",
                        "generation": "pilot",
                        "method": file.stem,
                        "seed": 1000,
                        "subset": subset,
                        **values,
                    }
                )
                lines.append(metric_row(file.stem, subset, values))
        lines += [
            "",
            "Infeasible-only breakdown (mutually exclusive): velocity-only screen failures; "
            "outside-workspace queries (including any other violations); remaining kinematic/IK failures. "
            "A nominal-IK velocity failure can still be tracked by another controller.",
            "",
        ]
        lines += table_header()
        for method, rows in pilot_rows.items():
            for category in ("velocity_only", "outside_workspace", "other_kinematic"):
                values = summarize([r for r in rows if infeasible_category(r, speed_limit) == category])
                breakdowns.append(
                    {
                        "robot": robot,
                        "condition": "native",
                        "target": "embodiment_development",
                        "generation": "pilot",
                        "method": method,
                        "seed": 1000,
                        "category": category,
                        **values,
                    }
                )
                lines.append(metric_row(method, category, values))
        lines.append("")
    atomic_json(
        output / "summary.json",
        {
            "updated_utc": utc(),
            "groups": summaries,
            "infeasibility_reasons": failure_reasons,
            "infeasible_breakdown": breakdowns,
            "inputs": provenance,
        },
    )
    if summaries:
        with (output / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data1/projects/lerobot-embodiment-v2"))
    args = parser.parse_args()
    report(args.root)


if __name__ == "__main__":
    main()

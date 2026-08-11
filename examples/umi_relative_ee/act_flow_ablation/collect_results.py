#!/usr/bin/env python
"""Collect compact stage-one metadata and metrics from external run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/media/zfei/Glowat512/projects/lerobot-arch-exp")
RUN_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)_(?P<steps>\d+)steps$")
PARAM_RE = re.compile(r"num_total_params=(?P<params>\d+)")
UPDATE_TIME_RE = re.compile(r"updt_s:(?P<seconds>[0-9.]+)")
VALIDATION_RE = re.compile(r"Validation at step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
WRAPPER_TIME_RE = re.compile(
    r"\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (?P<event>starting|completed)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def parse_run_name(name: str) -> dict[str, Any]:
    match = RUN_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unrecognized run name: {name}")
    values = match.groupdict()
    return {
        "run_name": name,
        "variant": values["variant"],
        "training_seed": int(values["seed"]),
        "steps": int(values["steps"]),
    }


def parse_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"status": "missing_log", "parameters": None, "wall_seconds": None, "validation": []}
    text = log_path.read_text(errors="replace")
    parameter_match = PARAM_RE.search(text)
    update_seconds = [float(match["seconds"]) for match in UPDATE_TIME_RE.finditer(text)]
    times: dict[str, datetime] = {}
    for match in WRAPPER_TIME_RE.finditer(text):
        times[match["event"]] = datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S")
    validation = []
    for match in VALIDATION_RE.finditer(text):
        metrics = {}
        for item in match["metrics"].split(", "):
            name, value = item.split("=", maxsplit=1)
            metrics[name] = float(value)
        validation.append({"step": int(match["step"]), **metrics})
    wall_seconds = None
    if "starting" in times and "completed" in times:
        wall_seconds = (times["completed"] - times["starting"]).total_seconds()
    return {
        "status": "complete" if "completed" in times else "running_or_failed",
        "parameters": int(parameter_match["params"]) if parameter_match else None,
        "wall_seconds": wall_seconds,
        "median_update_seconds": statistics.median(update_seconds) if update_seconds else None,
        "median_updates_per_second": 1 / statistics.median(update_seconds) if update_seconds else None,
        "validation": validation,
    }


def flatten_evaluation(report_path: Path, run: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    summary = report["summary"]
    row = {
        **run,
        "inference_seed": report["seed"],
        "policy_type": report["policy_type"],
        "num_episodes": summary["num_episodes"],
        "num_samples": summary["num_samples"],
        "video_backend": report.get("video_backend"),
        "cuda_peak_memory_bytes": report.get("cuda_peak_memory_bytes"),
    }
    for name, value in report.get("inference_latency_seconds", {}).items():
        row[f"inference_{name}"] = value
    for name, value in summary["episode_balanced"].items():
        row[name] = value
        interval = summary.get("episode_balanced_95ci", {}).get(name)
        if interval:
            row[f"{name}_ci_low"] = interval["low"]
            row[f"{name}_ci_high"] = interval["high"]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    train_rows = []
    validation_rows = []
    evaluation_rows = []
    for run_dir in sorted((args.artifact_root / "train").glob("*")):
        if not run_dir.is_dir() or RUN_RE.fullmatch(run_dir.name) is None:
            continue
        run = parse_run_name(run_dir.name)
        log = parse_log(args.artifact_root / "logs" / f"{run_dir.name}.log")
        train_rows.append({**run, **{key: value for key, value in log.items() if key != "validation"}})
        validation_rows.extend({**run, **metrics} for metrics in log["validation"])
        for report_path in sorted((args.artifact_root / "eval" / run_dir.name).glob("seed*/*.json")):
            evaluation_rows.append(flatten_evaluation(report_path, run))

    result_dir = args.artifact_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result_dir / "stage1_runs.csv", train_rows)
    write_csv(result_dir / "stage1_validation.csv", validation_rows)
    write_csv(result_dir / "stage1_evaluations.csv", evaluation_rows)
    (result_dir / "stage1_results.json").write_text(
        json.dumps(
            {"runs": train_rows, "validation": validation_rows, "evaluations": evaluation_rows},
            indent=2,
        )
        + "\n"
    )
    print(
        f"Collected {len(train_rows)} runs, {len(validation_rows)} validation points, "
        f"and {len(evaluation_rows)} evaluations under {result_dir}"
    )


if __name__ == "__main__":
    main()
